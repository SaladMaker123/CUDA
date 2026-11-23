#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include "../../../common/cuda_check.h"

// ==========================================
// Task 0: GPU Device Information
// ==========================================
int main()
{
    constexpr int DEVICE_ID = 0;

    CUDA_CHECK(cudaSetDevice(DEVICE_ID));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, DEVICE_ID));

    // ======================================
    // Memory Information
    // ======================================
    size_t freeMem = 0;
    size_t totalMem = 0;

    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));

    // ======================================
    // Clock Information
    // ======================================
    int coreClock = 0;
    int memoryClock = 0;

    CUDA_CHECK(cudaDeviceGetAttribute(
        &coreClock,
        cudaDevAttrClockRate,
        DEVICE_ID
    ));

    CUDA_CHECK(cudaDeviceGetAttribute(
        &memoryClock,
        cudaDevAttrMemoryClockRate,
        DEVICE_ID
    ));

    // ======================================
    // GPU Information
    // ======================================
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "========== GPU Information ==========\n";

    std::cout << "GPU Name: "
              << prop.name << "\n";

    std::cout << "Compute Capability: "
              << prop.major << "." << prop.minor << "\n";

    std::cout << "SM Count: "
              << prop.multiProcessorCount << "\n";

    std::cout << "Core Clock: "
              << coreClock / 1000.0 << " MHz\n";

    std::cout << "Memory Clock: "
              << memoryClock / 1000.0 << " MHz\n";

    // ======================================
    // Memory Hierarchy
    // ======================================
    std::cout << "\n========== Memory Information ==========\n";

    std::cout << "Total Global Memory: "
              << totalMem / (1024.0 * 1024.0 * 1024.0)
              << " GiB\n";

    std::cout << "Free Global Memory: "
              << freeMem / (1024.0 * 1024.0 * 1024.0)
              << " GiB\n";

    std::cout << "Memory Bus Width: "
              << prop.memoryBusWidth << " bits\n";

    std::cout << "L2 Cache Size: "
              << prop.l2CacheSize / 1024.0 << " KiB\n";

    std::cout << "Shared Memory Per SM: "
              << prop.sharedMemPerMultiprocessor / 1024.0
              << " KiB\n";

    std::cout << "Shared Memory Per Block: "
              << prop.sharedMemPerBlock / 1024.0
              << " KiB\n";

    std::cout << "Opt-in Shared Memory Per Block: "
              << prop.sharedMemPerBlockOptin / 1024.0
              << " KiB\n";

    // ======================================
    // Thread and Register Resources
    // ======================================
    std::cout << "\n========== Thread Resources ==========\n";

    std::cout << "Warp Size: "
              << prop.warpSize << "\n";

    std::cout << "Max Threads Per SM: "
              << prop.maxThreadsPerMultiProcessor << "\n";

    std::cout << "Max Blocks Per SM: "
              << prop.maxBlocksPerMultiProcessor << "\n";

    std::cout << "Max Threads Per Block: "
              << prop.maxThreadsPerBlock << "\n";

    std::cout << "Registers Per SM: "
              << prop.regsPerMultiprocessor << "\n";

    std::cout << "Registers Per Block: "
              << prop.regsPerBlock << "\n";

    std::cout << "Max Threads Per Block Dimension: "
              << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", "
              << prop.maxThreadsDim[2] << "\n";

    std::cout << "Max Grid Dimensions: "
              << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", "
              << prop.maxGridSize[2] << "\n";

    std::cout << "\nTask 0 Device Query Completed.\n";

    return 0;
}
