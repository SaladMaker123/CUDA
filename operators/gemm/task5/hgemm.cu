#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "utils.h"
namespace wmma = nvcuda::wmma;

// =====================================
// Block Tile
// =====================================
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 64;
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
constexpr int THREADS = 128;
constexpr int NUM_WARPS = THREADS / WARP_SIZE;
// Scheme B: four 32x32 regions
constexpr int WARP_BM = BM / 2;
constexpr int WARP_BN = BN / 2;
static_assert(NUM_WARPS == 4);
static_assert(BM % WM == 0);
static_assert(BN % WN == 0);
static_assert(BK % WK == 0);
// =====================================
// HGEMM Kernel
// C[M][N] = A[M][K] * B[K][N]
// =====================================
__launch_bounds__(THREADS)
__global__ void hgemm_kernel(
    const __half* A,
    const __half* B,
    float* C,
    int M,
    int N,
    int K
)
{
    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    // =====================================
    // Shared Memory
    // =====================================
    __shared__ __align__(32) __half sA[BM][BK];
    __shared__ __align__(32) __half sB[BK][BN];
    
    // =====================================
    // Warp Mapping: Scheme B
    // =====================================
    const int warpRow = warpId / 2;
    const int warpCol = warpId % 2;
    const int warpRowBase = warpRow * WARP_BM;
    const int warpColBase = warpCol * WARP_BN;
    // =====================================
    // FP32 Accumulator Fragments
    // =====================================
    wmma::fragment<wmma::accumulator,WM, WN, WK,float> cFrag00, cFrag01, cFrag10, cFrag11;
    wmma::fill_fragment(cFrag00, 0.0f);
    wmma::fill_fragment(cFrag01, 0.0f);
    wmma::fill_fragment(cFrag10, 0.0f);
    wmma::fill_fragment(cFrag11, 0.0f);
    // =====================================
    // K Tile Loop
    // =====================================
    for (int kBase = 0; kBase < K; kBase += BK)
    {
        // =================================
        // Global Memory -> Shared Memory: A
        // =================================
        for (int idx = tid; idx < BM * BK; idx += THREADS)
        {
            const int aRow = idx / BK;
            const int aCol = idx % BK;
            const int globalRow = blockRow + aRow;
            const int globalCol = kBase + aCol;
             if (globalRow < M && globalCol < K)
            {
                sA[aRow][aCol] = A[globalRow * K + globalCol];
            }
            else
            {
                sA[aRow][aCol] = __float2half(0.0f);
            }
        }
        // =================================
        // Global Memory -> Shared Memory: B
        // =================================
        for (int idx = tid; idx < BK * BN; idx += THREADS)
        {
            const int bRow = idx / BN;
            const int bCol = idx % BN;
            const int globalRow = kBase + bRow;
            const int globalCol = blockCol + bCol;
            if (globalRow < K && globalCol < N)
            {
                sB[bRow][bCol] = B[globalRow * N + globalCol];
            }
            else
            {
                sB[bRow][bCol] = __float2half(0.0f);
            }
        }
        // All threads finish loading before WMMA
        __syncthreads();
        // =================================
        // Compute Current K Tile
        // =================================
        for (int kTile = 0; kTile < BK; kTile += WK)
        {
            // =============================
            // Input Fragments
            // =============================
            wmma::fragment<wmma::matrix_a,WM, WN, WK,__half,wmma::row_major> aFrag0, aFrag1;
            wmma::fragment<wmma::matrix_b,WM, WN, WK,__half,wmma::row_major> bFrag0, bFrag1;
            // =============================
            // Load A Fragments
            // =============================
            wmma::load_matrix_sync(aFrag0,&sA[warpRowBase][kTile],BK);
            wmma::load_matrix_sync(aFrag1,&sA[warpRowBase + WM][kTile],BK);
            // =============================
            // Load B Fragments
            // =============================
            wmma::load_matrix_sync(bFrag0,&sB[kTile][warpColBase],BN);
            wmma::load_matrix_sync(bFrag1,&sB[kTile][warpColBase + WN],BN);
            // =============================
            // Tensor Core MMA
            // =============================
            wmma::mma_sync(cFrag00,aFrag0,bFrag0,cFrag00);
            wmma::mma_sync(cFrag01,aFrag0,bFrag1,cFrag01);
            wmma::mma_sync(cFrag10,aFrag1,bFrag0,cFrag10);
            wmma::mma_sync(cFrag11,aFrag1,bFrag1,cFrag11);
        }
        // All Warps finish using sA/sB
        // before the next K Tile overwrites them
        __syncthreads();
    }
    // =====================================
    // Store Accumulator Fragments
    // =====================================
    const int cRow = blockRow + warpRowBase;
    const int cCol = blockCol + warpColBase;
     wmma::store_matrix_sync(
        &C[cRow * N + cCol],
        cFrag00,
        N,
        wmma::mem_row_major
    );
    wmma::store_matrix_sync(
        &C[cRow * N + cCol + WN],
        cFrag01,
        N,
        wmma::mem_row_major
    );
    wmma::store_matrix_sync(
        &C[(cRow + WM) * N + cCol],
        cFrag10,
        N,
        wmma::mem_row_major
    );
    wmma::store_matrix_sync(
        &C[(cRow + WM) * N + cCol + WN],
        cFrag11,
        N,
        wmma::mem_row_major
    );
}
int main()
{
    // Correctness Check
    if (benchmark_hgemm<BM, BN, BK, THREADS>(
            hgemm_kernel,
            128, 128, 128,
            true,
            1, 1) != 0)
    {
        return EXIT_FAILURE;
    }

    // Performance Benchmark
    if (benchmark_hgemm<BM, BN, BK, THREADS>(
            hgemm_kernel,
            4096, 4096, 4096,
            false,
            2, 10) != 0)
    {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
