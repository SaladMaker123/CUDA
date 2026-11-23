#pragma once
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// =====================================
// CUDA Error Check
// =====================================
#define CUDA_CHECK(call)                            \
do {                                                \
    cudaError_t err = (call);                       \
    if (err != cudaSuccess) {                       \
        fprintf(stderr,                            \
            "CUDA error: %s, %s\n",                 \
            #call, cudaGetErrorString(err));        \
        return EXIT_FAILURE;                        \
    }                                               \
} while (0)


// =====================================
// CPU Reference SGEMM
// =====================================
static void sgemmRef(
    int M,
    int N,
    int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C
)
{
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k)
            {
                sum += A[row * K + k] * B[k * N + col];
            }

            C[row * N + col] =
                alpha * sum + beta * C[row * N + col];
        }
    }
}


// =====================================
// Benchmark
// =====================================
template <typename KernelFunc>
int benchmark(
    KernelFunc kernel,
    int n,
    bool checkResult,
    int warmup = 4,
    int repeat = 4
)
{
    const int M = n;
    const int N = n;
    const int K = n;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    assert(BM % TM == 0);
    assert(BN % TN == 0);

    assert(M % BM == 0);
    assert(N % BN == 0);
    assert(K % BK == 0);

    // =====================================
    // Grid and Block
    // =====================================
    const int threads = BM * BN / (TM * TN);

    dim3 block(threads);
    dim3 grid(N / BN, M / BM);

    // =====================================
    // Memory Size
    // =====================================
    const size_t bytesA =
        static_cast<size_t>(M) * K * sizeof(float);

    const size_t bytesB =
        static_cast<size_t>(K) * N * sizeof(float);

    const size_t bytesC =
        static_cast<size_t>(M) * N * sizeof(float);

    // =====================================
    // Host Memory
    // =====================================
    float* hA = static_cast<float*>(malloc(bytesA));
    float* hB = static_cast<float*>(malloc(bytesB));
    float* hC = static_cast<float*>(malloc(bytesC));

    assert(hA != nullptr);
    assert(hB != nullptr);
    assert(hC != nullptr);

    for (int i = 0; i < M * K; ++i)
    {
        hA[i] = rand() / static_cast<float>(RAND_MAX);
    }

    for (int i = 0; i < K * N; ++i)
    {
        hB[i] = rand() / static_cast<float>(RAND_MAX);
    }

    // =====================================
    // Device Memory
    // =====================================
    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    CUDA_CHECK(cudaMemcpy(
        dA, hA, bytesA, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dB, hB, bytesB, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemset(dC, 0, bytesC));

    // =====================================
    // Warmup
    // =====================================
    for (int i = 0; i < warmup; ++i)
    {
        kernel<<<grid, block>>>(
            dA, dB, dC,
            M, N, K,
            alpha, beta
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // =====================================
    // CUDA Event Benchmark
    // =====================================
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i)
    {
        kernel<<<grid, block>>>(
            dA, dB, dC,
            M, N, K,
            alpha, beta
        );

        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &totalMs, start, stop
    ));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const double avgMs =
        static_cast<double>(totalMs) / repeat;

    const double tflops =
        2.0 * M * N * K / (avgMs * 1e-3) / 1e12;

    printf(
        "N = %4d, "
        "BM = %3d, BN = %3d, "
        "BK = %3d, TM = %3d, TN = %3d, "
        "Time: %7.3f ms/iter, "
        "%7.3f TFLOPS",
        n,
        BM, BN, BK, TM, TN,
        avgMs, tflops
    );

    // =====================================
    // CPU Correctness Check
    // =====================================
    if (checkResult)
    {
        float* hRef = static_cast<float*>(
            calloc(
                static_cast<size_t>(M) * N,
                sizeof(float)
            )
        );

        assert(hRef != nullptr);

        CUDA_CHECK(cudaMemcpy(
            hC, dC, bytesC, cudaMemcpyDeviceToHost
        ));

        sgemmRef(
            M, N, K,
            alpha,
            hA, hB,
            beta,
            hRef
        );

        float maxError = 0.0f;

        for (int i = 0; i < M * N; ++i)
        {
            const float err = fabsf(hRef[i] - hC[i]);

            if (err > maxError)
            {
                maxError = err;
            }
        }

        printf(", Max Error: %.6e\n", maxError);

        free(hRef);
    }
    else
    {
        printf("\n");
    }

    // =====================================
    // Free Memory
    // =====================================
    free(hA);
    free(hB);
    free(hC);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return EXIT_SUCCESS;
}
