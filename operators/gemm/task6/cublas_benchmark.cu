#include "utils.h"

int main()
{
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;

    constexpr int WARMUP = 10;
    constexpr int REPEAT = 20;

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

    CUDA_CHECK(cudaMemcpy(
        dA, hA.data(),
        bytesA, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dB, hB.data(),
        bytesB, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemset(dC, 0, bytesC));

    // =====================================
    // cuBLAS Configuration
    // =====================================
    cublasHandle_t handle;

    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // =====================================
    // cuBLAS GEMM
    // =====================================
    auto gemm = [&]()
    {
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
            dC,
            CUDA_R_32F,
            N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        ));
    };

    // =====================================
    // Warmup
    // =====================================
    for (int i = 0; i < WARMUP; ++i)
    {
        gemm();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // =====================================
    // CUDA Event Benchmark
    // =====================================
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; ++i)
    {
        gemm();
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &totalMs, start, stop
    ));

    // =====================================
    // Performance
    // =====================================
    const double avgMs =
        static_cast<double>(totalMs) / REPEAT;

    const double flops =
        2.0 * static_cast<double>(M) * N * K;

    const double tflops =
        flops / (avgMs * 1e-3) / 1e12;

    printf("\ncuBLAS HGEMM Benchmark\n");
    printf("M=%d, N=%d, K=%d\n", M, N, K);
    printf("Input: FP16\n");
    printf("Accumulator: FP32\n");
    printf("Output: FP32\n");
    printf("Warmup: %d\n", WARMUP);
    printf("Repeat: %d\n", REPEAT);
    printf("Time: %.6f ms\n", avgMs);
    printf("Performance: %.6f TFLOPS\n", tflops);

    // =====================================
    // Cleanup
    // =====================================
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return EXIT_SUCCESS;
}