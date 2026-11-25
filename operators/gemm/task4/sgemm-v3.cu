#include "utils.h"
#include <cuda_pipeline.h>

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int THREADS = (BM / TM) * (BN / TN);
// =====================================
// V3: cp.async Double Buffering
// =====================================
__launch_bounds__(THREADS)
__global__ void sgemmV3(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
)
{
    __shared__ __align__(16) float sA[2][BM * BK];
    __shared__ __align__(16) float sB[2][BK * BN];

    const int tid = threadIdx.x;
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    // Thread Tile Mapping
    const int threadRow = (tid / (BN / TN)) * TM;
    const int threadCol = (tid % (BN / TN)) * TN;
    // Global -> Shared Mapping
    const int aRow = tid / (BK / 4);
    const int aCol = (tid % (BK / 4)) * 4;

    const int bRow = tid / (BN / 4);
    const int bCol = (tid % (BN / 4)) * 4;
    // Registers
    float acc[TM][TN] = {};
    float regA[TM];
    float regB[TN];
    int PHASES = (K + BK - 1) / BK;
    // =====================================
    // Prefetch: Phase 0
    // =====================================
    __pipeline_memcpy_async(
        &sA[0][aRow * BK + aCol],&A[(blockRow + aRow) * K + aCol],sizeof(float4));
    __pipeline_memcpy_async(
        &sB[0][bRow * BN + bCol],&B[bRow * N + blockCol + bCol],sizeof(float4));
    __pipeline_commit();
    // =====================================
    // Main Loop
    // =====================================
    for (int phase = 0; phase < PHASES; ++phase)
    {
        const int current = phase % 2;
        const int next = (phase + 1) % 2;
        const bool hasNext = (phase + 1) < PHASES;
        // =====================================
        // Prefetch Next Phase
        // =====================================
        if (hasNext)
        {
            __pipeline_memcpy_async(
                &sA[next][aRow * BK + aCol],&A[(blockRow + aRow) * K+ (phase + 1) * BK + aCol],sizeof(float4));
            __pipeline_memcpy_async(&sB[next][bRow * BN + bCol],&B[((phase + 1) * BK + bRow) * N+ blockCol + bCol],
            sizeof(float4));
            __pipeline_commit();
        }
        // =====================================
        // Wait for Current Phase
        // =====================================
        __pipeline_wait_prior(hasNext ? 1 : 0);
        __syncthreads();
        // =====================================
        // Shared -> Register -> FMA
        // =====================================
        #pragma unroll
        for (int k = 0; k < BK; ++k)
        {
            // Load A Register Tile
            #pragma unroll
            for (int i = 0; i < TM; ++i)
            {
                regA[i] = sA[current][(threadRow + i) * BK + k];
            }
            // Load B Register Tile

            #pragma unroll
            for (int j = 0; j < TN; ++j)
            {
                regB[j] = sB[current][k * BN + threadCol + j];
            }
            // TM x TN Outer Product
            #pragma unroll
            for (int i = 0; i < TM; ++i)
            {
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                {
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }
    // =====================================
    // Register -> Global: C
    // =====================================
    #pragma unroll
    for(int i = 0;i<TM;++i)
    {
        const int cRow = blockRow + threadRow + i;
        #pragma unroll
        for(int j = 0;j<TN;++j)
        {
            const int cCol = blockCol + threadCol + j;
            C[cRow * N + cCol] = acc[i][j];
        }
    }
}
// =====================================
// Kernel Launcher
// =====================================
void launchV3(
    const float* A,
    const float* B,
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
    sgemmV3<<<grid, block>>>(
        A, B, C, M, N, K
    );
    CUDA_CHECK(cudaGetLastError());
}
int main(int argc, char** argv)
{
    return runBenchmark(
        "SGEMM V3 - cp.async Double Buffering",
        launchV3,
        argc,
        argv
    );
}
