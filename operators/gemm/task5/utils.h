#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <climits>

#include "../../../common/cuda_check.h"
#include "../../../common/cublas_check.h"

constexpr int WARMUP = 10;
constexpr int REPEAT = 20;
constexpr int ROUNDS = 3;

constexpr float ATOL = 1e-3f;
constexpr float RTOL = 1e-4f;
constexpr double FROB_TOL = 1e-5;

using LaunchFunc = void (*)(
    const __half*, const __half*, float*,
    int, int, int
);

// =====================================
// Random Initialization
// =====================================

inline void initMatrix(
    std::vector<__half>& data,
    unsigned int seed
)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (__half& x : data)
    {
        x = __half(dist(gen));
    }
}

// =====================================
// cuBLAS HGEMM
// =====================================

inline void launchCublas(
    cublasHandle_t handle,
    const __half* A,
    const __half* B,
    float* C,
    int M,
    int N,
    int K
)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        B,
        CUDA_R_16F,
        N,
        A,
        CUDA_R_16F,
        K,
        &beta,
        C,
        CUDA_R_32F,
        N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    ));
}

// =====================================
// Correctness Check
// =====================================

inline bool checkResult(
    const std::vector<float>& result,
    const std::vector<float>& reference
)
{
    float maxError = 0.0f;
    size_t errorCount = 0;

    double errorSum = 0.0;
    double refSum = 0.0;

    for (size_t i = 0; i < result.size(); ++i)
    {
        if (!std::isfinite(result[i]) ||
            !std::isfinite(reference[i]))
        {
            ++errorCount;
            continue;
        }

        const float error =
            std::fabs(result[i] - reference[i]);

        const float tolerance =
            ATOL + RTOL * std::fabs(reference[i]);

        maxError = std::max(maxError, error);

        const double diff =
            double(result[i]) - double(reference[i]);

        errorSum += diff * diff;

        refSum += double(reference[i])
                * double(reference[i]);

        if (error > tolerance)
        {
            ++errorCount;
        }
    }

    const double frobError =
        std::sqrt(errorSum / std::max(refSum, 1e-30));

    const bool passed =
        errorCount == 0 && frobError <= FROB_TOL;

    std::cout << std::scientific
              << "Max Absolute Error: " << maxError << '\n'
              << "Relative Frobenius:  " << frobError << '\n'
              << "Invalid Elements:   " << errorCount << '\n'
              << "Correctness:        "
              << (passed ? "PASS" : "FAIL")
              << "\n\n";

    return passed;
}

// =====================================
// Median
// =====================================

inline float median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());

    const size_t n = values.size();

    if (n % 2 == 0)
    {
        return (values[n / 2 - 1] +
                values[n / 2]) * 0.5f;
    }

    return values[n / 2];
}

// =====================================
// CUDA Event Timing
// =====================================

template <typename Func>
float benchmark(Func launch)
{
    for (int i = 0; i < WARMUP; ++i)
    {
        launch();
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(REPEAT);

    for (int i = 0; i < REPEAT; ++i)
    {
        CUDA_CHECK(cudaEventRecord(start));

        launch();

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(
            &times[i], start, stop
        ));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return median(times);
}

// =====================================
// Performance Calculation
// =====================================

inline double calculateTFLOPS(
    int M, int N, int K, float ms
)
{
    return 2.0 * M * N * K / (ms * 1e9);
}

// =====================================
// Unified Test Framework
// =====================================

inline int runBenchmark(
    const char* name,
    LaunchFunc launchKernel,
    int argc,
    char** argv
)
{
    int M = 4096;
    int N = 4096;
    int K = 4096;

    if (argc == 2)
    {
        M = N = K = std::atoi(argv[1]);
    }
    else if (argc == 4)
    {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    else if (argc != 1)
    {
        std::cerr << "Usage: " << argv[0]
                  << " [size | M N K]\n";

        return EXIT_FAILURE;
    }

    if (M <= 0 || N <= 0 || K <= 0)
    {
        std::cerr << "Invalid matrix dimensions\n";
        return EXIT_FAILURE;
    }

    const size_t sizeA = size_t(M) * K;
    const size_t sizeB = size_t(K) * N;
    const size_t sizeC = size_t(M) * N;

    if (sizeA > INT_MAX ||
        sizeB > INT_MAX ||
        sizeC > INT_MAX)
    {
        std::cerr << "Matrix exceeds int indexing range\n";
        return EXIT_FAILURE;
    }

    const size_t bytesA = sizeA * sizeof(__half);
    const size_t bytesB = sizeB * sizeof(__half);
    const size_t bytesC = sizeC * sizeof(float);

    // Host Memory

    std::vector<__half> hA(sizeA);
    std::vector<__half> hB(sizeB);

    initMatrix(hA, 11);
    initMatrix(hB, 22);

    // Device Memory

    __half* dA = nullptr;
    __half* dB = nullptr;

    float* dC = nullptr;
    float* dRef = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dRef, bytesC));

    CUDA_CHECK(cudaMemcpy(
        dA, hA.data(), bytesA, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dB, hB.data(), bytesB, cudaMemcpyHostToDevice
    ));

    // cuBLAS Initialization

    cublasHandle_t handle;

    CUBLAS_CHECK(cublasCreate(&handle));

    CUBLAS_CHECK(cublasSetMathMode(
        handle, CUBLAS_DEFAULT_MATH
    ));

    auto cublas = [&]()
    {
        launchCublas(handle, dA, dB, dRef, M, N, K);
    };

    auto kernel = [&]()
    {
        launchKernel(dA, dB, dC, M, N, K);
    };

    std::cout << "\n" << name << '\n'
              << "M = " << M
              << ", N = " << N
              << ", K = " << K << '\n'
              << "Input: FP16\n"
              << "Accumulator: FP32\n"
              << "Output: FP32\n"
              << "Warmup: " << WARMUP << '\n'
              << "Repeat: " << REPEAT << '\n'
              << "Rounds: " << ROUNDS << "\n\n";

    bool passed = true;

    // Correctness Check

    if (launchKernel != nullptr)
    {
        CUDA_CHECK(cudaMalloc(&dC, bytesC));

        CUDA_CHECK(cudaMemset(dC, 0xff, bytesC));
        CUDA_CHECK(cudaMemset(dRef, 0xff, bytesC));

        cublas();
        kernel();

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> hC(sizeC);
        std::vector<float> hRef(sizeC);

        CUDA_CHECK(cudaMemcpy(
            hC.data(), dC, bytesC, cudaMemcpyDeviceToHost
        ));

        CUDA_CHECK(cudaMemcpy(
            hRef.data(), dRef, bytesC, cudaMemcpyDeviceToHost
        ));

        passed = checkResult(hC, hRef);
    }

    // Performance Test

    if (passed)
    {
        std::vector<float> kernelTimes;
        std::vector<float> cublasTimes;

        for (int round = 0; round < ROUNDS; ++round)
        {
            float kernelMs = 0.0f;
            float cublasMs = 0.0f;

            if (launchKernel != nullptr)
            {
                if (round % 2 == 0)
                {
                    kernelMs = benchmark(kernel);
                    cublasMs = benchmark(cublas);
                }
                else
                {
                    cublasMs = benchmark(cublas);
                    kernelMs = benchmark(kernel);
                }

                kernelTimes.push_back(kernelMs);
            }
            else
            {
                cublasMs = benchmark(cublas);
            }

            cublasTimes.push_back(cublasMs);

            std::cout << std::fixed
                      << std::setprecision(3)
                      << "Round " << round + 1;

            if (launchKernel != nullptr)
            {
                std::cout << " | Kernel: "
                          << kernelMs << " ms";
            }

            std::cout << " | cuBLAS: "
                      << cublasMs << " ms\n";
        }

        const float cublasMs = median(cublasTimes);

        const double cublasTFLOPS =
            calculateTFLOPS(M, N, K, cublasMs);

        std::cout << "\n========== Performance ==========\n";

        if (launchKernel != nullptr)
        {
            const float kernelMs = median(kernelTimes);

            const double kernelTFLOPS =
                calculateTFLOPS(M, N, K, kernelMs);

            std::cout << "Kernel Time:    "
                      << kernelMs << " ms\n"
                      << "Kernel TFLOPS:  "
                      << kernelTFLOPS << '\n'
                      << "cuBLAS Time:    "
                      << cublasMs << " ms\n"
                      << "cuBLAS TFLOPS:  "
                      << cublasTFLOPS << '\n'
                      << "Relative Perf:  "
                      << kernelTFLOPS / cublasTFLOPS * 100.0
                      << "%\n";
        }
        else
        {
            std::cout << "cuBLAS Time:    "
                      << cublasMs << " ms\n"
                      << "cuBLAS TFLOPS:  "
                      << cublasTFLOPS << '\n';
        }

        std::cout << "=================================\n";
    }

    // Release Resources

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dRef));

    if (dC != nullptr)
    {
        CUDA_CHECK(cudaFree(dC));
    }

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}