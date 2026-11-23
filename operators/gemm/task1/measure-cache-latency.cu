
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>

#include <cuda_runtime.h>

#include "../../../common/cuda_check.h"
#include "utils.h"

// ==========================================
// Task 1: Cache Latency Kernel
// ==========================================
__launch_bounds__(1)
__global__ void cacheLatencyKernel(
    const unsigned int* buf,
    unsigned int loops,
    unsigned int* sink
)
{
    unsigned int idx = 0;

    for(unsigned int i = 0; i < loops; ++i)
    {
        #pragma unroll
        for(unsigned int j = 0; j < CHASE_UNROLL; ++j)
        {
            idx = buf[idx];
        }
    }

    sink[0] = idx;
}

// ==========================================
// Main
// ==========================================
int main()
{
    constexpr int REPEATS = 5;
    constexpr size_t MIN_LOADS = 2000000;
    constexpr size_t MAX_BYTES = 96ULL * 1024 * 1024;

    const std::vector<size_t> worksetKB = {
        1, 4, 16, 32, 48, 64, 72, 80,
        96, 112, 128, 192, 256, 512,
        1024, 4096, 8192, 16384,
        32768, 49152, 57344,
        65536, 81920, 98304
    };

    CUDA_CHECK(cudaSetDevice(0));

    int coreClock = 0;

    CUDA_CHECK(cudaDeviceGetAttribute(
        &coreClock,
        cudaDevAttrClockRate,
        0
    ));

    // ======================================
    // Device Memory
    // ======================================
    unsigned int* dBuf = nullptr;
    unsigned int* dSink = nullptr;

    CUDA_CHECK(cudaMalloc(&dBuf, MAX_BYTES));
    CUDA_CHECK(cudaMalloc(&dSink, sizeof(unsigned int)));

    // ======================================
    // CUDA Events
    // ======================================
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "========== Cache Latency ==========\n";

    std::cout << std::setw(12) << "Size(KB)"
              << std::setw(16) << "Latency(ns)"
              << std::setw(16) << "Cycles"
              << "\n";

    // ======================================
    // Test Different Working Set Sizes
    // ======================================
    for(size_t sizeKB : worksetKB)
    {
        const size_t bytes = sizeKB * 1024;
        const size_t numElems = bytes / sizeof(unsigned int);

        std::vector<unsigned int> hBuf(numElems);

        createRandomChain(hBuf);

        CUDA_CHECK(cudaMemcpy(
            dBuf,
            hBuf.data(),
            bytes,
            cudaMemcpyHostToDevice
        ));

        // Cover the complete pointer-chasing cycle
        const size_t requiredLoads =
            std::max(MIN_LOADS, numElems);

        const unsigned int loops =
            static_cast<unsigned int>(
                (requiredLoads + CHASE_UNROLL - 1)
                / CHASE_UNROLL
            );

        const double totalLoads =
            static_cast<double>(loops) * CHASE_UNROLL;

        // ==================================
        // Warmup
        // ==================================
        cacheLatencyKernel<<<1, 1>>>(
            dBuf, loops, dSink
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ==================================
        // Benchmark
        // ==================================
        std::vector<float> times(REPEATS);

        for(int r = 0; r < REPEATS; ++r)
        {
            CUDA_CHECK(cudaEventRecord(start));

            cacheLatencyKernel<<<1, 1>>>(
                dBuf, loops, dSink
            );

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms = 0.0f;

            CUDA_CHECK(cudaEventElapsedTime(
                &ms, start, stop
            ));

            times[r] = ms;
        }

        // ==================================
        // Latency Calculation
        // ==================================
        const float medianMs = getMedian(times);

        const double latencyNs =
            medianMs * 1000000.0 / totalLoads;

        const double latencyCycles =
            latencyNs * coreClock / 1000000.0;

        std::cout << std::setw(12) << sizeKB
                  << std::setw(16) << latencyNs
                  << std::setw(16) << latencyCycles
                  << "\n";
    }

    // ======================================
    // Cleanup
    // ======================================
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dBuf));
    CUDA_CHECK(cudaFree(dSink));

    return 0;
}
