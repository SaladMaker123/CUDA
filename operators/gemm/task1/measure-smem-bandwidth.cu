
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

#include <cuda_runtime.h>

#include "../../../common/cuda_check.h"

// ================================
// Task 1: Shared Memory Bandwidth
// ================================
constexpr int THREADS = 1024;
constexpr int BLOCKS_PER_SM = 32;

constexpr unsigned long long TARGET_BYTES =
    512ULL * 1024 * 1024;

// ================================
// Shared Memory Bandwidth Kernel
// ================================
template <unsigned int SMEM_N4>
__global__ void smemBandwidthKernel(
    int iterations,
    unsigned int* sink
)
{
    extern __shared__ uint4 smem4[];

    const unsigned int tid = threadIdx.x;
    const unsigned int stride = THREADS;

    // Initialize Shared Memory
    for (unsigned int i = tid; i < SMEM_N4; i += stride)
    {
        smem4[i] = make_uint4(i, i, i, i);
    }

    __syncthreads();

   const volatile uint4* ptr = smem4;

    uint4 acc = make_uint4(0, 0, 0, 0);

    for (int i = 0; i < iterations; ++i)
    {
        #pragma unroll 16
        for (unsigned int j = tid; j < SMEM_N4; j += stride)
        {
            acc.x += ptr[j].x;
            acc.y += ptr[j].y;
            acc.z += ptr[j].z;
            acc.w += ptr[j].w;
        }
    }

    sink[blockIdx.x * THREADS + tid] =
        acc.x + acc.y + acc.z + acc.w;
}

// ================================
// Kernel Dispatch
// ================================
template <unsigned int N4>
void launch(
    int iterations,
    int blocks,
    size_t smemBytes,
    unsigned int* sink
)
{
    smemBandwidthKernel<N4><<<
        blocks, THREADS, smemBytes
    >>>(iterations, sink);

    CUDA_CHECK(cudaGetLastError());
}

bool dispatch(
    unsigned int n4,
    int iterations,
    int blocks,
    size_t smemBytes,
    unsigned int* sink
)
{
    switch (n4)
    {
    case 1024:
        launch<1024>(iterations, blocks, smemBytes, sink);
        break;
    case 1536:
        launch<1536>(iterations, blocks, smemBytes, sink);
        break;
    case 2048:
        launch<2048>(iterations, blocks, smemBytes, sink);
        break;
    case 3072:
        launch<3072>(iterations, blocks, smemBytes, sink);
        break;
    case 4096:
        launch<4096>(iterations, blocks, smemBytes, sink);
        break;
    default:
        return false;
    }

    return true;
}

// ================================
// Main
// ================================
int main()
{
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int coreClock = 0;
    int maxSmemOptin = 0;

    CUDA_CHECK(cudaDeviceGetAttribute(
        &coreClock, cudaDevAttrClockRate, 0
    ));

    CUDA_CHECK(cudaDeviceGetAttribute(
        &maxSmemOptin,
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        0
    ));

    // Enable 64 KiB dynamic Shared Memory
    if (maxSmemOptin >= 64 * 1024)
    {
        CUDA_CHECK(cudaFuncSetAttribute(
            smemBandwidthKernel<4096>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            64 * 1024
        ));
    }

    const int blocks =
        prop.multiProcessorCount * BLOCKS_PER_SM;

    unsigned int* dSink = nullptr;

    CUDA_CHECK(cudaMalloc(
        &dSink,
        static_cast<size_t>(blocks) *
            THREADS * sizeof(unsigned int)
    ));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const std::vector<int> sizesKB = {
        16, 24, 32, 48, 64
    };

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "========== Shared Memory Bandwidth ==========\n";

    std::cout << std::setw(12) << "Size(KiB)"
              << std::setw(16) << "BW(TB/s)"
              << std::setw(20) << "BW(byte/cycle)"
              << "\n";

    for (int sizeKB : sizesKB)
    {
        const size_t smemBytes =
            static_cast<size_t>(sizeKB) * 1024;

        if (smemBytes > static_cast<size_t>(maxSmemOptin))
            continue;

        const unsigned int n4 =
            static_cast<unsigned int>(
                smemBytes / sizeof(uint4)
            );

        const size_t bytesPerIteration =
            static_cast<size_t>(blocks) * smemBytes;

        const unsigned long long rawIterations =
            TARGET_BYTES / bytesPerIteration;

        const int iterations = static_cast<int>(
            std::clamp(rawIterations, 100ULL, 200000ULL)
        );

        const double totalBytes =
            static_cast<double>(bytesPerIteration) *
            iterations;

        // Warmup
        dispatch(n4, 10, blocks, smemBytes, dSink);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        CUDA_CHECK(cudaEventRecord(start));

        dispatch(n4, iterations, blocks, smemBytes, dSink);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(
            &ms, start, stop
        ));

        // Validate the first block
        std::vector<unsigned int> hSink(THREADS);

        CUDA_CHECK(cudaMemcpy(
            hSink.data(),
            dSink,
            THREADS * sizeof(unsigned int),
            cudaMemcpyDeviceToHost
        ));

        for (unsigned int tid = 0; tid < THREADS; ++tid)
        {
            unsigned long long expected = 0;

            for (unsigned int j = tid; j < n4; j += THREADS)
            {
                expected += 4ULL * j * iterations;
            }

            if (hSink[tid] != expected)
            {
                std::cerr << "Validation failed at "
                          << sizeKB << " KiB\n";
                return 1;
            }
        }

        const double bandwidthTBs =
            totalBytes / (ms * 1e-3) / 1e12;

        const double bytesPerCycle =
            bandwidthTBs * 1e9 / coreClock;

        std::cout << std::setw(12) << sizeKB
                  << std::setw(16) << bandwidthTBs
                  << std::setw(20) << bytesPerCycle
                  << "\n";
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dSink));

    std::cout << "\nTask 1 Shared Memory Bandwidth Completed.\n";

    return 0;
}
