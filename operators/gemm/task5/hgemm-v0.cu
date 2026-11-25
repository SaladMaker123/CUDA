#include "utils.h"
#include <mma.h>

namespace wmma = nvcuda::wmma;
// WMMA Tile
constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;
// Block Tile
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
// Warp Tile
constexpr int WARP_TILE_M = 32;
constexpr int WARP_TILE_N = 32;
// Warp Mapping
constexpr int WARP_NUM_M = BM / WARP_TILE_M;
constexpr int WARP_NUM_N = BN / WARP_TILE_N;
constexpr int WARPS = WARP_NUM_M * WARP_NUM_N;
constexpr int THREADS = WARPS * 32;
// Fragments per Warp
constexpr int FM = WARP_TILE_M / WM;
constexpr int FN = WARP_TILE_N / WN;
// Vector Width
constexpr int VEC = sizeof(float4) / sizeof(__half);
// =====================================
// V0: WMMA GEMM
// =====================================
__launch_bounds__(THREADS)
__global__ void hgemmV0(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
)
{
    __shared__ __align__(32) __half sA[BM][BK];
    __shared__ __align__(32) __half sB[BK][BN];
    const int tid = threadIdx.x;
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    // Warp Mapping
    const int warpId = tid / 32;
    const int warpRow = warpId / WARP_NUM_N;
    const int warpCol = warpId % WARP_NUM_N;
    // A, B and Accumulator Fragments
    wmma::fragment<wmma::matrix_a,WM, WN, WK,__half,wmma::row_major> aFrag[FM];
    wmma::fragment<wmma::matrix_b,WM, WN, WK,__half,wmma::row_major> bFrag[FN];
    wmma::fragment<wmma::accumulator,WM,WN,WK,float> acc[FM][FN]; 
    #pragma unroll
    for (int i = 0; i < FM; ++i)
    {
        #pragma unroll
        for (int j = 0; j < FN; ++j)
        {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }
    // Global -> Shared Mapping
    const int aRow = tid / (BK / VEC);
    const int aCol = (tid % (BK / VEC)) * VEC;

    const int bRow = tid / (BN / VEC);
    const int bCol = (tid % (BN / VEC)) * VEC;

    int PHASES = (K + BK - 1) / BK;
    // =====================================
    // Main Loop
    // =====================================
    for (int phase = 0; phase < PHASES; ++phase)
    {
        // Global -> Shared: A
        *reinterpret_cast<float4*>(&sA[aRow][aCol]) = 
            *reinterpret_cast<const float4*>(&A[(blockRow + aRow) * K + phase * BK + aCol]);
        // Global -> Shared: B
        *reinterpret_cast<float4*>(&sB[bRow][bCol]) = 
            *reinterpret_cast<const float4*>(&B[(phase * BK + bRow) * N+ blockCol + bCol]);
        __syncthreads();
        // =====================================
        // Shared -> Fragment: A
        // =====================================
        #pragma unroll
        for (int i = 0; i < FM; ++i)
        {
            const int aSharedRow = warpRow * WARP_TILE_M + i * WM;
            wmma::load_matrix_sync(aFrag[i],&sA[aSharedRow][0],BK);
        }
        // =====================================
        // Shared -> Fragment: B
        // =====================================
        #pragma unroll
        for (int j = 0; j < FN; ++j)
        {
            const int bSharedCol = warpCol * WARP_TILE_N + j * WN;
            wmma::load_matrix_sync(bFrag[j],&sB[0][bSharedCol],BN);
        }
        // =====================================
        // WMMA Compute
        // =====================================
        #pragma unroll
        for (int i = 0; i < FM; ++i)
        {
            #pragma unroll
            for (int j = 0; j < FN; ++j)
            {
                wmma::mma_sync(acc[i][j],aFrag[i],bFrag[j],acc[i][j]);
            }
        }
        __syncthreads();
    }
    // =====================================
    // Fragment -> Global: C
    // =====================================
    #pragma unroll
    for (int i = 0; i < FM; ++i)
    {
        #pragma unroll
        for (int j = 0; j < FN; ++j)
        {
            const int cRow = blockRow + warpRow * WARP_TILE_M + i * WM;
            const int cCol = blockCol + warpCol * WARP_TILE_N + j * WN;
            wmma::store_matrix_sync(&C[cRow * N + cCol],acc[i][j],N,wmma::mem_row_major);
        }
    }
}
// =====================================
// Kernel Launcher
// =====================================
void launchV0(
    const __half* A,
    const __half* B,
    float* C,
    int M,
    int N,
    int K
)
{
    if (M <= 0 || N <= 0 || K <= 0 ||
        M % BM != 0 ||
        N % BN != 0 ||
        K % BK != 0)
    {
        std::cerr << "Invalid matrix dimensions: "
                  << "M % BM, N % BN and K % BK must be zero.\n";

        std::exit(EXIT_FAILURE);
    }
    const dim3 block(THREADS);
    const dim3 grid(
        N / BN,
        M / BM
    );

    hgemmV0<<<grid, block>>>(
        A, B, C, M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
}

// =====================================
// Main
// ====================================
int main(int argc, char** argv)
{
    return runBenchmark(
        "HGEMM V0 - WMMA",
        launchV0,
        argc,
        argv
    );
}

