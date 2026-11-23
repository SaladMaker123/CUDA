
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <cuda_runtime.h>

#include "../../../common/cuda_check.h"
#include "utils.h"

// ================================
// Task 1: GDDR Bandwidth
// ================================
constexpr int NUM_THREADS_PER_BLOCK = 256;
constexpr int NUM_BLOCKS_PER_GRID = 512;

constexpr int NUM_WARMUP = 4;
constexpr int NUM_REPEAT = 5;

constexpr size_t MAX_BYTES =
    4ULL * 1024 * 1024 * 1024;

// ================================
// GDDR Bandwidth Kernel
// ================================
__launch_bounds__(NUM_THREADS_PER_BLOCK)
__global__ void gddrBandwidthKernel(
    const float4* __restrict__ buf,
    int M,
    float* sink
)
{
    const int tid = threadIdx.x;
    const int stride = NUM_THREADS_PER_BLOCK;

    const size_t offset =
        static_cast<size_t>(blockIdx.x) * M;

    float acc = 0.0f;

    for (int j = tid; j < M; j += stride)
    {
        float4 v = buf[offset + j];

        acc += v.x + v.y + v.z + v.w;
    }

    // Preserve all threads' results
    sink[blockIdx.x * stride + tid] = acc;
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

    float4* dBuf = nullptr;
    float* dSink = nullptr;

    const size_t sinkElems =
        static_cast<size_t>(NUM_BLOCKS_PER_GRID) *
        NUM_THREADS_PER_BLOCK;

    CUDA_CHECK(cudaMalloc(&dBuf, MAX_BYTES));

    CUDA_CHECK(cudaMalloc(
        &dSink,
        sinkElems * sizeof(float)
    ));

    CUDA_CHECK(cudaMemset(
        dBuf, 0xAB, MAX_BYTES
    ));

    // Original initialization pattern
    const uint32_t bits = 0xABABABABu;
    float initializedValue = 0.0f;

    std::memcpy(
        &initializedValue,
        &bits,
        sizeof(float)
    );

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const std::vector<int> worksetKB = {
        16384, 24576, 32768, 40960, 49152,
        57344, 61440, 65536, 73728, 81920,
        98304, 131072, 262144, 524288,
        1048576, 2097152, 4194304
    };

    std::vector<float> times(NUM_REPEAT);
    std::vector<float> hSink(sinkElems);

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "========== GDDR Bandwidth ==========\n";

    std::cout << std::setw(14) << "Size(MiB)"
              << std::setw(18) << "BW(TB/s)"
              << std::setw(20) << "BW(byte/cycle)"
              << "\n";

    for (int sizeKB : worksetKB)
    {
        const size_t bytes =
            static_cast<size_t>(sizeKB) * 1024;

        const int M = static_cast<int>(
            bytes / NUM_BLOCKS_PER_GRID /
            sizeof(float4)
        );

        const size_t actualBytes =
            static_cast<size_t>(M) *
            NUM_BLOCKS_PER_GRID * sizeof(float4);

        // ================================
        // Warmup
        // ================================
        for (int i = 0; i < NUM_WARMUP; ++i)
        {
            gddrBandwidthKernel<<<
                NUM_BLOCKS_PER_GRID,
                NUM_THREADS_PER_BLOCK
            >>>(dBuf, M, dSink);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ================================
        // Benchmark
        // ================================
        for (int r = 0; r < NUM_REPEAT; ++r)
        {
            CUDA_CHECK(cudaEventRecord(start));

            gddrBandwidthKernel<<<
                NUM_BLOCKS_PER_GRID,
                NUM_THREADS_PER_BLOCK
            >>>(dBuf, M, dSink);

            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            CUDA_CHECK(cudaEventElapsedTime(
                &times[r], start, stop
            ));
        }

        // ================================
        // Correctness
        // ================================
        CUDA_CHECK(cudaMemcpy(
            hSink.data(),
            dSink,
            sinkElems * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        const double expected =
            4.0 * initializedValue *
            (M / NUM_THREADS_PER_BLOCK);

        const double tolerance =
            std::max(std::abs(expected) * 1e-3, 1e-18);

        for (size_t i = 0; i < sinkElems; ++i)
        {
            if (!std::isfinite(hSink[i]) ||
                std::abs(hSink[i] - expected) > tolerance)
            {
                std::cerr << "Validation failed at "
                          << sizeKB << " KiB\n";
                return 1;
            }
        }

        // ================================
        // Bandwidth Calculation
        // ================================
        const double medianMs = getMedian(times);

        const double bandwidthTBs =
            actualBytes / (medianMs * 1e-3) / 1e12;

        const double bytesPerCycle =
            bandwidthTBs * 1e9 / coreClock;

        std::cout << std::setw(14) << sizeKB / 1024
                  << std::setw(18) << bandwidthTBs
                  << std::setw(20) << bytesPerCycle
                  << "\n";
    }

    // ================================
    // Cleanup
    // ================================
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dBuf));
    CUDA_CHECK(cudaFree(dSink));

    std::cout << "\nTask 1 GDDR Bandwidth Completed.\n";

    return 0;
}
