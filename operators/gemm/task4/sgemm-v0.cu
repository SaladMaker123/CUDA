
#include "utils.h"

// Adapted from CUDA-Kernel-Library.
// Copyright (c) 2026 Olajide Badejo
// Distributed under the MIT License.
// See LICENSE for the full license text.

constexpr int BM = 16;
constexpr int BN = 16;

__global__ void sgemmV0(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
)
{
    const int cRow = blockIdx.y * BM + threadIdx.y;
    const int cCol = blockIdx.x * BN + threadIdx.x;

    if (cRow >= M || cCol >= N)
    {
        return;
    }

    float acc = 0.0f;

    for (int k = 0; k < K; ++k)
    {
        acc += A[cRow * K + k] * B[k* N + cCol];
    }

    C[static_cast<long long>(cRow) * N + cCol] = acc;
}

void launchV0(
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

    sgemmV0<<<grid, block>>>(
        A, B, C, M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char** argv)
{
    return runBenchmark(
        "SGEMM V0 - Naive GEMM",
        launchV0,
        argc,
        argv
    );
}
