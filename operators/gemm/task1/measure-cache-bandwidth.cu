
#include <iostream>
#include <vector>
#include <iomanip>
#include <stdexcept>

#include <cuda_runtime.h>

#include "../../../common/cuda_check.h"
#include "utils.h"

// ================================
// Task 1: Cache Bandwidth
// ================================
constexpr int THREADS = 512;
constexpr int REPEATS = 5;
constexpr int WARMUP = 3;

constexpr size_t TARGET_BYTES =
    512ULL * 1024 * 1024;

constexpr size_t MAX_BYTES =
    64ULL * 1024 * 1024;

// ================================
// Cache Bandwidth Kernel
// ================================
template <bool USE_L1>
__global__ void cacheBandwidthKernel(
    const float* data,
    float* sink,
    size_t numElems,
    int passes
)
{
    const size_t tid =
        blockIdx.x * blockDim.x + threadIdx.x;

    const size_t stride =
        gridDim.x * blockDim.x;

    float sum = 0.0f;

    for (int r = 0; r < passes; ++r)
    {
        for (size_t i = tid; i < numElems; i += stride)
        {
            if constexpr (USE_L1)
            {
                sum += __ldca(data + i);
            }
            else
            {
                sum += __ldcg(data + i);
            }
        }
    }

    sink[tid] = sum;
}

// ================================
// Benchmark
// ================================
template <bool USE_L1>
double benchmark(
    const float* dData,
    float* dSink,
    size_t bytes,
    int blocks
)
{
    const size_t numElems = bytes / sizeof(float);
    const int passes = static_cast<int>(
        (TARGET_BYTES + bytes - 1) / bytes
    );

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < WARMUP; ++i)
    {
        cacheBandwidthKernel<USE_L1><<<blocks, THREADS>>>(
            dData, dSink, numElems, passes
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times(REPEATS);

    for (int r = 0; r < REPEATS; ++r)
    {
        CUDA_CHECK(cudaEventRecord(start));

        cacheBandwidthKernel<USE_L1><<<blocks, THREADS>>>(
            dData, dSink, numElems, passes
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(
            &times[r], start, stop
        ));
    }

    // Validate thread 0
    float result = 0.0f;

    CUDA_CHECK(cudaMemcpy(
        &result, dSink, sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    const size_t stride =
        static_cast<size_t>(blocks) * THREADS;

    const size_t loadsPerPass =
        (numElems + stride - 1) / stride;

    const float expected =
        static_cast<float>(loadsPerPass * passes);

    if (result != expected)
    {
        throw std::runtime_error("Bandwidth validation failed");
    }

    const float medianMs = getMedian(times);

    const double totalBytes =
        static_cast<double>(bytes) * passes;

    const double bandwidthGBs =
        totalBytes / (medianMs * 1000000.0);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return bandwidthGBs;
}

// ================================
// Main
// ================================
int main()
{
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const int smCount = prop.multiProcessorCount;

    std::vector<float> hData(
        MAX_BYTES / sizeof(float), 1.0f
    );

    float* dData = nullptr;
    float* dSink = nullptr;

    CUDA_CHECK(cudaMalloc(&dData, MAX_BYTES));

    const int maxBlocks = smCount * 2;

    CUDA_CHECK(cudaMalloc(
        &dSink,
        static_cast<size_t>(maxBlocks) *
            THREADS * sizeof(float)
    ));

    CUDA_CHECK(cudaMemcpy(
        dData, hData.data(), MAX_BYTES,
        cudaMemcpyHostToDevice
    ));

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "========== L1 Cache Bandwidth ==========\n";

    std::cout << std::setw(14) << "KB/Block"
              << std::setw(18) << "L1+L2(GB/s)"
              << std::setw(18) << "L2-only(GB/s)"
              << "\n";

    for (int sizeKB : {16, 32, 64, 128})
    {
        const size_t bytes =
            static_cast<size_t>(sizeKB) * 1024 * smCount;

        const double l1 = benchmark<true>(
            dData, dSink, bytes, smCount
        );

        const double l2 = benchmark<false>(
            dData, dSink, bytes, smCount
        );

        std::cout << std::setw(14) << sizeKB
                  << std::setw(18) << l1
                  << std::setw(18) << l2
                  << "\n";
    }

    std::cout << "\n========== L2 Cache Bandwidth ==========\n";

    std::cout << std::setw(14) << "Size(MiB)"
              << std::setw(18) << "L2-only(GB/s)"
              << "\n";

    for (int sizeMB : {4, 16, 32, 64})
    {
        const size_t bytes =
            static_cast<size_t>(sizeMB) * 1024 * 1024;

        const double bw = benchmark<false>(
            dData, dSink, bytes, maxBlocks
        );

        std::cout << std::setw(14) << sizeMB
                  << std::setw(18) << bw
                  << "\n";
    }

    CUDA_CHECK(cudaFree(dData));
    CUDA_CHECK(cudaFree(dSink));

    std::cout << "\nTask 1 Cache Bandwidth Completed.\n";

    return 0;
}
