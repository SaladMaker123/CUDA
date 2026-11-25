#include "utils.h"
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 32;
__global__ void sgemmV1(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
)
{
    __shared__ float sA[BM][BK + 1];
    __shared__ float sB[BK][BN + 1];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int cRow = blockIdx.y * BM + ty;
    const int cCol = blockIdx.x * BN + tx;

    float acc = 0.0f;
    const int PHASES = (K + BK - 1) / BK;
    for (int phase = 0; phase < PHASES; ++phase)
    {
        const int aCol = phase * BK + tx;
        const int bRow = phase * BK + ty;

        sA[ty][tx] =(cRow < M && aCol < K) ? A[cRow * K + aCol] : 0.0f;
        sB[ty][tx] =(bRow < K && cCol < N) ? B[bRow * N + cCol] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k)
        {
            acc += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }
     if (cRow < M && cCol < N)
    {
        C[(cRow) * N + cCol] = acc;
    }

}
void launchV1(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
)
{
    const dim3 block(BN, BM);

    const dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    sgemmV1<<<grid, block>>>(
        A, B, C, M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char** argv)
{
    return runBenchmark(
        "SGEMM V1 - Shared Memory Tiling",
        launchV1,
        argc,
        argv
    );
}
    
