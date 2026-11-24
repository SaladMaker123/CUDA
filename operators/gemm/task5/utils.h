#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
// =====================================
// CUDA Error Check
// =====================================
#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error: %s, %s\n",                \
                #call, cudaGetErrorString(err));               \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)

// =====================================
// cuBLAS Error Check
// =====================================
#define CUBLAS_CHECK(call)                                    \
do {                                                          \
    cublasStatus_t status = (call);                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                    \
        fprintf(stderr, "cuBLAS error: %s, code=%d\n",         \
                #call, static_cast<int>(status));              \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)

// =====================================
// Initialize FP16 Input
// =====================================
static inline void fill_matrix(
    std::vector<__half>& A,
    int multiplier
)
{
    for(size_t i = 0;i<A.size();++i)
    {
        const int value = static_cast<int>((i * multiplier + 7) % 257) - 128;
        A[i] = __half(static_cast<float>(value) / 128.0f);
    }
}
// =====================================
// Unified HGEMM Benchmark
// =====================================
template<int BM,int BN,int BK,int THREADS>
int benchmark_hgemm(
    void (*kernel)(const __half*,const __half*,float*,int,int,int),
    int M,int N,int K,bool check_result,
    int warmup = 4,
    int repeat = 4
)
{
    // =====================================
    // Configuration Check
    // =====================================
    if (M <= 0 || N <= 0 || K <= 0 ||
        M % BM != 0 || N % BN != 0 ||
        warmup < 0 || repeat <= 0)
    {
        fprintf(stderr, "Invalid benchmark configuration\n");
        return EXIT_FAILURE;
    }
    // =====================================
    // Host Memory
    // =====================================
    const size_t sizeA = static_cast<size_t>(M) * K;
    const size_t sizeB = static_cast<size_t>(K) * N;
    const size_t sizeC = static_cast<size_t>(M) * N;
    
    const size_t bytesA = sizeA * sizeof(__half);
    const size_t bytesB = sizeB * sizeof(__half);
    const size_t bytesC = sizeC * sizeof(float);
    std::vector<__half> hA(sizeA);
    std::vector<__half> hB(sizeB);
    fill_matrix(hA, 17);
    fill_matrix(hB, 29);
    // =====================================
    // Device Memory
    // =====================================
    __half* dA = nullptr;
    __half* dB = nullptr;
    float* dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

     CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(dC, 0, bytesC));
    // =====================================
    // Launch Configuration
    // =====================================
    const dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );
    const dim3 block(THREADS);
    
    printf("\nHGEMM: M=%d, N=%d, K=%d\n", M, N, K);
    printf("BM=%d, BN=%d, BK=%d, Threads=%d\n",BM, BN, BK, THREADS);
    // =====================================
    // cuBLAS Correctness Check
    // =====================================
    if (check_result)
    {
        // Run Custom Kernel
        kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> hC(sizeC);
        std::vector<float> hRef(sizeC);
        CUDA_CHECK(cudaMemcpy(
            hC.data(),
            dC,
            bytesC,
            cudaMemcpyDeviceToHost
        ));
        // =================================
        // cuBLAS Reference
        // =================================
        float* dRef = nullptr;
        CUDA_CHECK(cudaMalloc(&dRef, bytesC));
        
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        const float alpha = 1.0f;
        const float beta = 0.0f;
        // Row-major: C = A * B
        // Column-major equivalent: C^T = B^T * A^T
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            dB,
            CUDA_R_16F,
            N,
            dA,
            CUDA_R_16F,
            K,
            &beta,
            dRef,
            CUDA_R_32F,
            N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        ));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            hRef.data(),
            dRef,
            bytesC,
            cudaMemcpyDeviceToHost
        ));
        // =================================
        // Compare Results
        // =================================
        float maxError = 0.0f;
        constexpr float atol = 1e-3f;
        constexpr float rtol = 1e-3f;
        bool passed = true;
        for (size_t i = 0; i < sizeC; ++i)
        {
            const float error =
                std::fabs(hC[i] - hRef[i]);

            if (!std::isfinite(error))
            {
                passed = false;
                continue;
            }

            maxError = std::max(maxError, error);

            const float tolerance =
                atol + rtol * std::fabs(hRef[i]);

            if (!(error <= tolerance))
            {
                passed = false;
            }
        }
        printf("Max Error (cuBLAS): %.6e\n", maxError);

        printf("Validation: %s\n",
               passed ? "PASS" : "FAIL");

        CUBLAS_CHECK(cublasDestroy(handle));
        CUDA_CHECK(cudaFree(dRef));
        if (!passed)
        {
            CUDA_CHECK(cudaFree(dA));
            CUDA_CHECK(cudaFree(dB));
            CUDA_CHECK(cudaFree(dC));

            return EXIT_FAILURE;
        }
    }
    // =====================================
    // Warmup
    // =====================================
    for (int i = 0; i < warmup; ++i)
    {
        kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // =====================================
    // CUDA Event Benchmark
    // =====================================
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i)
    {
        kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float totalMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));
    const double avgMs = static_cast<double>(totalMs) / repeat;
    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double tflops = flops / (avgMs * 1e-3) / 1e12;
    
    printf("Time: %.6f ms\n", avgMs);
    printf("Performance: %.6f TFLOPS\n", tflops);
    
    // =====================================
    // Cleanup
    // =====================================
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return EXIT_SUCCESS;

}