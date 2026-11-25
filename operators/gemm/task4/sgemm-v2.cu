#include "utils.h"
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;

constexpr int TM = 8;
constexpr int TN = 8;
constexpr int THREADS = (BM / TM) * (BN / TN);
// =====================================
// V2: Register Blocking
// =====================================
__global__ __launch_bounds__(THREADS)
void sgemmV2(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
)
{
    __shared__ float sA[BK][BM];
    __shared__ __align__(16) float sB[BK][BN];
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
    for (int phase = 0; phase < PHASES; ++phase)
    {
        // =====================================
        // Global -> Shared: A
        // =====================================
        const float4 aVec =
            *reinterpret_cast<const float4*>(&A[(blockRow + aRow) * K + phase * BK + aCol]);
        sA[aCol + 0][aRow] = aVec.x;
        sA[aCol + 1][aRow] = aVec.y;
        sA[aCol + 2][aRow] = aVec.z;
        sA[aCol + 3][aRow] = aVec.w;
        // =====================================
        // Global -> Shared: B
        // =====================================
        const float4 bVec =
            *reinterpret_cast<const float4*>(&B[(phase * BK + bRow) * N + blockCol + bCol]);
        *reinterpret_cast<float4*>(&sB[bRow][bCol]) = bVec;
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
                regA[i] = sA[k][threadRow + i];
            }
            // Load B Register Tile
            #pragma unroll
            for (int j = 0; j < TN; ++j)
            {
                regB[j] = sB[k][threadCol + j];
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
    for (int i = 0; i < TM; ++i)
    {
        const int cRow = blockRow + threadRow + i;
        #pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            const int cCol = blockCol + threadCol + j;
            C[cRow * N + cCol] = acc[i][j];
        }
    }
}
// =====================================
// Kernel Launcher
// =====================================

void launchV2(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
)
{
    if (M % BM != 0 || N % BN != 0 || K % BK != 0)
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

    sgemmV2<<<grid, block>>>(
        A, B, C, M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
}

// =====================================
// Main
// =====================================

int main(int argc, char** argv)
{
    return runBenchmark(
        "SGEMM V2 - Register Blocking",
        launchV2,
        argc,
        argv
    );
}