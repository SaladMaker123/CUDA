#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cuda/std/utility>
#include <mma.h>

#include "utils.h"

namespace wmma = nvcuda::wmma;
namespace cde = cuda::device::experimental;

using Barrier = cuda::barrier<cuda::thread_scope_block>;

// =====================================
// Block Tile
// =====================================
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

// =====================================
// WMMA Tile
// =====================================
constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;

// =====================================
// Thread Configuration
// =====================================
constexpr int WARP_SIZE = 32;
constexpr int THREADS = 256;
constexpr int NUM_WARPS = THREADS / WARP_SIZE;

// =====================================
// Group Tile
// =====================================
constexpr int GROUP_M = 2;
constexpr int GROUP_N = 2;

constexpr int GROUP_BM = BM / GROUP_M;
constexpr int GROUP_BN = BN / GROUP_N;

constexpr int WARPS_PER_GROUP = 2;

// =====================================
// Warp Tile
// =====================================
constexpr int WARP_BM = 32;
constexpr int WARP_BN = 64;

constexpr int WARP_TILES_M = WARP_BM / WM;
constexpr int WARP_TILES_N = WARP_BN / WN;

// =====================================
// Block Swizzle
// =====================================
constexpr int SWIZZLE_ROWS = 4;
constexpr int SWIZZLE_COLS = 2;

// =====================================
// Pipeline Configuration
// =====================================
constexpr int NUM_STAGE = 2;

static_assert(NUM_WARPS ==
              GROUP_M * GROUP_N * WARPS_PER_GROUP);

static_assert(GROUP_BM ==
              WARP_BM * WARPS_PER_GROUP);

static_assert(GROUP_BN == WARP_BN);
static_assert(BK % WK == 0);
static_assert(NUM_STAGE == 2);

// =====================================
// HGEMM V3 Kernel
// =====================================
__launch_bounds__(THREADS)
__global__ void hgemm_v3(
    const __grid_constant__ CUtensorMap descA,
    const __grid_constant__ CUtensorMap descB,
    float* C,
    int M,
    int N,
    int K
)
{
    const int tid = threadIdx.x;

    // =====================================
    // Block Swizzle
    // =====================================
    const int tilesM = M / BM;
    const int tilesN = N / BN;

    const int groupsPerRow =
        (tilesN + SWIZZLE_COLS - 1) / SWIZZLE_COLS;

    const int groupId = blockIdx.y;
    const int localId = blockIdx.x;

    const int groupRow = groupId / groupsPerRow;
    const int groupCol = groupId % groupsPerRow;

    const int blockRow =
        groupRow * SWIZZLE_ROWS +
        localId / SWIZZLE_COLS;

    const int blockCol =
        groupCol * SWIZZLE_COLS +
        localId % SWIZZLE_COLS;

    if (blockRow >= tilesM || blockCol >= tilesN)
    {
        return;
    }

    const int blockRowBase = blockRow * BM;
    const int blockColBase = blockCol * BN;

    // =====================================
    // Warp Group Mapping
    // =====================================
    const int warpId = tid / WARP_SIZE;

    const int warpGroupId =
        warpId / WARPS_PER_GROUP;

    const int warpGroupRow =
        warpGroupId / GROUP_N;

    const int warpGroupCol =
        warpGroupId % GROUP_N;

    const int warpInGroup =
        warpId % WARPS_PER_GROUP;

    const int warpRowBase =
        warpGroupRow * GROUP_BM +
        warpInGroup * WARP_BM;

    const int warpColBase =
        warpGroupCol * GROUP_BN;

    // =====================================
    // Shared Memory
    // =====================================
    __shared__ __align__(128)
        __half sA[NUM_STAGE][GROUP_M][GROUP_BM][BK];

    __shared__ __align__(128)
        __half sB[NUM_STAGE][GROUP_N][BK][GROUP_BN];

    // =====================================
    // TMA Barrier
    // =====================================
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ Barrier bar[NUM_STAGE];

    if (tid == 0)
    {
        #pragma unroll
        for (int stage = 0; stage < NUM_STAGE; ++stage)
        {
            init(&bar[stage], THREADS);
        }

        cde::fence_proxy_async_shared_cta();
    }

    __syncthreads();

    // =====================================
    // FP32 Accumulator Fragments
    // =====================================
    wmma::fragment<
        wmma::accumulator,
        WM, WN, WK,
        float
    > cFrag[WARP_TILES_M][WARP_TILES_N];

    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; ++i)
    {
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; ++j)
        {
            wmma::fill_fragment(cFrag[i][j], 0.0f);
        }
    }

    // =====================================
    // Pipeline Parameters
    // =====================================
    const int PHASES = K / BK;

    constexpr int TRANS_BYTES =
        sizeof(sA[0]) + sizeof(sB[0]);

    // V3: Manually track barrier parity
    bool barPhase[NUM_STAGE] = {0, 0};

    // =====================================
    // Preload: Phase 0 -> Buffer 0
    // =====================================
    if (tid == 0)
    {
        // Load A0 and A1
        #pragma unroll
        for (int i = 0; i < GROUP_M; ++i)
        {
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &sA[0][i][0][0],
                &descA,
                0,
                blockRowBase + i * GROUP_BM,
                bar[0]
            );
        }

        // Load B0 and B1
        #pragma unroll
        for (int j = 0; j < GROUP_N; ++j)
        {
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &sB[0][j][0][0],
                &descB,
                blockColBase + j * GROUP_BN,
                0,
                bar[0]
            );
        }

        // V3: Discard arrival token
        (void)cuda::device::barrier_arrive_tx(
            bar[0],
            1,
            TRANS_BYTES
        );
    }
    else
    {
        (void)bar[0].arrive();
    }

    // =====================================
    // Double Buffer Pipeline
    // =====================================
    for (int phase = 0; phase < PHASES; ++phase)
    {
        // =================================
        // Current Compute Stage
        // =================================
        const int calcStage = phase % NUM_STAGE;

        // V3: Wait using fixed barrier index
        switch (calcStage)
        {
            case 0:
                bar[0].wait_parity(barPhase[0]);
                barPhase[0] ^= 1;
                break;

            case 1:
                bar[1].wait_parity(barPhase[1]);
                barPhase[1] ^= 1;
                break;
        }

        // =================================
        // Next Load Stage
        // =================================
        const int loadStage =
            (phase + 1) % NUM_STAGE;

        // =================================
        // Prefetch Next Phase
        // =================================
        if (phase + 1 < PHASES)
        {
            const int kBase = (phase + 1) * BK;

            // V3: Unroll buffer selection
            #pragma unroll
            for (int stage = 0; stage < NUM_STAGE; ++stage)
            {
                if (loadStage == stage)
                {
                    if (tid == 0)
                    {
                        // Load A0 and A1
                        #pragma unroll
                        for (int i = 0; i < GROUP_M; ++i)
                        {
                            cde::cp_async_bulk_tensor_2d_global_to_shared(
                                &sA[stage][i][0][0],
                                &descA,
                                kBase,
                                blockRowBase + i * GROUP_BM,
                                bar[stage]
                            );
                        }

                        // Load B0 and B1
                        #pragma unroll
                        for (int j = 0; j < GROUP_N; ++j)
                        {
                            cde::cp_async_bulk_tensor_2d_global_to_shared(
                                &sB[stage][j][0][0],
                                &descB,
                                blockColBase + j * GROUP_BN,
                                kBase,
                                bar[stage]
                            );
                        }

                        (void)cuda::device::barrier_arrive_tx(
                            bar[stage],
                            1,
                            TRANS_BYTES
                        );
                    }
                    else
                    {
                        (void)bar[stage].arrive();
                    }
                }
            }
        }

        // =================================
        // Compute Current K Tile
        // =================================
        #pragma unroll
        for (int kTile = 0; kTile < BK; kTile += WK)
        {
            // =============================
            // Input Fragments
            // =============================
            wmma::fragment<
                wmma::matrix_a,
                WM, WN, WK,
                __half,
                wmma::row_major
            > aFrag[WARP_TILES_M];

            wmma::fragment<
                wmma::matrix_b,
                WM, WN, WK,
                __half,
                wmma::row_major
            > bFrag[WARP_TILES_N];

            // =============================
            // Load A Fragments
            // =============================
            #pragma unroll
            for (int i = 0; i < WARP_TILES_M; ++i)
            {
                const int aRow =
                    warpInGroup * WARP_BM +
                    i * WM;

                wmma::load_matrix_sync(
                    aFrag[i],
                    &sA[calcStage][warpGroupRow][aRow][kTile],
                    BK
                );
            }

            // =============================
            // Load B Fragments
            // =============================
            #pragma unroll
            for (int j = 0; j < WARP_TILES_N; ++j)
            {
                const int bCol = j * WN;

                wmma::load_matrix_sync(
                    bFrag[j],
                    &sB[calcStage][warpGroupCol][kTile][bCol],
                    GROUP_BN
                );
            }

            // =============================
            // Tensor Core MMA
            // =============================
            #pragma unroll
            for (int i = 0; i < WARP_TILES_M; ++i)
            {
                #pragma unroll
                for (int j = 0; j < WARP_TILES_N; ++j)
                {
                    wmma::mma_sync(
                        cFrag[i][j],
                        aFrag[i],
                        bFrag[j],
                        cFrag[i][j]
                    );
                }
            }
        }

        // Finish reading shared memory
        // before the next TMA transfer
        __syncthreads();
    }

    // =====================================
    // Store Fragments -> Global Memory
    // =====================================
    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; ++i)
    {
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; ++j)
        {
            const int cRow =
                blockRowBase + warpRowBase + i * WM;

            const int cCol =
                blockColBase + warpColBase + j * WN;

            wmma::store_matrix_sync(
                &C[cRow * N + cCol],
                cFrag[i][j],
                N,
                wmma::mem_row_major
            );
        }
    }
}

// =====================================
// Main
// =====================================
int main()
{
    return benchmark_hgemm<
        BM, BN, BK, THREADS,
        SWIZZLE_ROWS, SWIZZLE_COLS
    >(
        hgemm_v3,
        4096, 4096, 4096,
        true,
        10, 20
    );
}