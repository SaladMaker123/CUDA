
#include <iostream>
#include <vector>
#include <iomanip>

#include <cuda_runtime.h>

#include "../../../common/cuda_check.h"
#include "utils.h"

// ================================
// Task 1: Shared Memory Latency
// ================================
constexpr int THREADS = 128;
constexpr int REPEATS = 5;
constexpr unsigned int ITERATIONS = 2000000;

constexpr size_t MAX_BYTES = 48ULL * 1024;

// ================================
// Shared Memory Latency Kernel
// ================================
__global__ void sharedLatencyKernel(
    const unsigned int* input,
    size_t numElems,
    unsigned long long* cycles,
    unsigned int* sink
)
{
    extern __shared__ unsigned int sBuf[];

    for (size_t i = threadIdx.x;
         i < numElems;
         i += blockDim.x)
    {
        sBuf[i] = input[i];
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        volatile unsigned int* ptr = sBuf;
        unsigned int idx = 0;

        // Warmup
        for (size_t i = 0; i < numElems; ++i)
        {
            idx = ptr[idx];
        }

        const unsigned long long start = clock64();

        #pragma unroll 1
        for (unsigned int i = 0; i < ITERATIONS; ++i)
        {
            idx = ptr[idx];
        }

        const unsigned long long stop = clock64();

        cycles[0] = stop - start;
        sink[0] = idx;
    }
}

// ================================
// Main
// ================================
int main()
{
    CUDA_CHECK(cudaSetDevice(0));

    int coreClock = 0;

    CUDA_CHECK(cudaDeviceGetAttribute(
        &coreClock,
        cudaDevAttrClockRate,
        0
    ));

    unsigned int* dBuf = nullptr;
    unsigned int* dSink = nullptr;
    unsigned long long* dCycles = nullptr;

    CUDA_CHECK(cudaMalloc(&dBuf, MAX_BYTES));
    CUDA_CHECK(cudaMalloc(&dSink, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dCycles, sizeof(unsigned long long)));

    const std::vector<int> worksetKB = {
        1, 4, 8, 16, 32, 48
    };

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "========== Shared Memory Latency ==========\n";

    std::cout << std::setw(12) << "Size(KiB)"
              << std::setw(18) << "Cycles/Load"
              << std::setw(18) << "Latency(ns)"
              << "\n";

    for (int sizeKB : worksetKB)
    {
        const size_t bytes =
            static_cast<size_t>(sizeKB) * 1024;

        const size_t numElems =
            bytes / sizeof(unsigned int);

        std::vector<unsigned int> hBuf(numElems);

        createRandomChain(hBuf);

        CUDA_CHECK(cudaMemcpy(
            dBuf,
            hBuf.data(),
            bytes,
            cudaMemcpyHostToDevice
        ));

        std::vector<float> times(REPEATS);
        unsigned int result = 0;

        for (int r = 0; r < REPEATS; ++r)
        {
            sharedLatencyKernel<<<1, THREADS, bytes>>>(
                dBuf, numElems, dCycles, dSink
            );

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            unsigned long long elapsedCycles = 0;

            CUDA_CHECK(cudaMemcpy(
                &elapsedCycles,
                dCycles,
                sizeof(unsigned long long),
                cudaMemcpyDeviceToHost
            ));

            times[r] =
                static_cast<float>(
                    static_cast<double>(elapsedCycles) /
                    ITERATIONS
                );
        }

        CUDA_CHECK(cudaMemcpy(
            &result,
            dSink,
            sizeof(unsigned int),
            cudaMemcpyDeviceToHost
        ));

        unsigned int expected = 0;

        const size_t remaining =
            ITERATIONS % numElems;

        for (size_t i = 0; i < remaining; ++i)
        {
            expected = hBuf[expected];
        }

        if (result != expected)
        {
            std::cerr << "Validation failed at "
                      << sizeKB << " KiB\n";
            return 1;
        }

        const double latencyCycles = getMedian(times);

        const double latencyNs =
            latencyCycles * 1000000.0 / coreClock;

        std::cout << std::setw(12) << sizeKB
                  << std::setw(18) << latencyCycles
                  << std::setw(18) << latencyNs
                  << "\n";
    }

    CUDA_CHECK(cudaFree(dBuf));
    CUDA_CHECK(cudaFree(dSink));
    CUDA_CHECK(cudaFree(dCycles));

    std::cout << "\nTask 1 Shared Memory Latency Completed.\n";

    return 0;
}
