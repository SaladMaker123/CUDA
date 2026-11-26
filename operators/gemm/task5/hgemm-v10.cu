#include "utils.h"
#include <cstdint>
#include <cuda_pipeline.h>

// MMA Tile
constexpr int WM = 16;
constexpr int WN = 8;
constexpr int WK = 16;

// Warp Tile
constexpr int WARP_TILE_M = 64;
constexpr int WARP_TILE_N = 32;

// Vector Width
constexpr int VEC = sizeof(float4) / sizeof(__half);

// Swizzle Size
constexpr int SWIZZLE_SIZE = 8;


// Tile Configuration
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct TileConfig
{
    static constexpr int BM = BLOCK_M;
    static constexpr int BN = BLOCK_N;
    static constexpr int BK = BLOCK_K;

    // Warp Mapping
    static constexpr int WARP_NUM_M = BM / WARP_TILE_M;
    static constexpr int WARP_NUM_N = BN / WARP_TILE_N;

    static constexpr int WARPS = WARP_NUM_M * WARP_NUM_N;
    static constexpr int THREADS = WARPS * 32;

    // Tiles per Warp
    static constexpr int FM = WARP_TILE_M / WM;
    static constexpr int FN = WARP_TILE_N / WN;

    static constexpr int B_PAIRS = FN / 2;
    static constexpr int K_TILES = BK / WK;

    // Vectors per Thread
    static constexpr int LOAD_A = BM * BK / (THREADS * VEC);
    static constexpr int LOAD_B = BK * BN / (THREADS * VEC);

    // Physical Layout of A
    static constexpr int A_SHARED_COLS = 64;
    static constexpr int A_SHARED_ROWS = BM * BK / A_SHARED_COLS;

    // Shared Memory
    static constexpr int SMEM_BYTES =
        2 * (BM * BK + BK * BN) * sizeof(__half);

    // Configuration Constraints
    static_assert(BM % WARP_TILE_M == 0);
    static_assert(BN % WARP_TILE_N == 0);

    static_assert(WARP_TILE_M % WM == 0);
    static_assert(WARP_TILE_N % WN == 0);

    static_assert(BK % WK == 0);
    static_assert(FN % 2 == 0);

    static_assert(BK == 32 || BK == 64);
    static_assert(BN % 64 == 0);

    static_assert(BM * BK % A_SHARED_COLS == 0);

    static_assert(BM * BK % (THREADS * VEC) == 0);
    static_assert(BK * BN % (THREADS * VEC) == 0);

    static_assert(THREADS <= 1024);
    static_assert(SMEM_BYTES <= 99 * 1024);
};


// Configuration List
using Config0 = TileConfig<128, 128, 32>;
using Config1 = TileConfig<128, 256, 32>;
using Config2 = TileConfig<256, 128, 32>;
using Config3 = TileConfig<128, 64, 64>;
using Config4 = TileConfig<64, 128, 64>;
using Config5 = TileConfig<128, 128, 64>;


// A Swizzle
template<typename Config>
__device__ __forceinline__
void swizzleA(
    int row,
    int col,
    int& newRow,
    int& newCol
)
{
    constexpr int BK = Config::BK;

    int mergedCol;

    if constexpr (BK == 32)
    {
        newRow = row / 2;

        mergedCol = (row % 2) * BK + col;
    }
    else
    {
        newRow = row;

        mergedCol = col;
    }

    const int chunk = mergedCol / SWIZZLE_SIZE;
    const int offset = mergedCol % SWIZZLE_SIZE;

    const int mask = newRow % 8;

    const int newChunk = chunk ^ mask;

    newCol = newChunk * SWIZZLE_SIZE + offset;
}


// B Swizzle
__device__ __forceinline__
int swizzleB(int row, int col)
{
    const int chunk = col / SWIZZLE_SIZE;
    const int offset = col % SWIZZLE_SIZE;

    const int mask = row % 8;

    const int newChunk = chunk ^ mask;

    const int newCol = newChunk * SWIZZLE_SIZE + offset;

    return newCol;
}


// Shared Memory Address
__device__ __forceinline__
uint32_t sharedAddr(const void* ptr)
{
    return static_cast<uint32_t>(
        __cvta_generic_to_shared(ptr)
    );
}


// Load A Registers
__device__ __forceinline__
void ldmatrixX4(
    uint32_t& a0,
    uint32_t& a1,
    uint32_t& a2,
    uint32_t& a3,
    const void* ptr
)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"

        : "=r"(a0), "=r"(a1),
          "=r"(a2), "=r"(a3)

        : "r"(sharedAddr(ptr))
    );
}


// Load B Registers
__device__ __forceinline__
void ldmatrixX4Trans(
    uint32_t& b0,
    uint32_t& b1,
    uint32_t& b2,
    uint32_t& b3,
    const void* ptr
)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"

        : "=r"(b0), "=r"(b1),
          "=r"(b2), "=r"(b3)

        : "r"(sharedAddr(ptr))
    );
}


// MMA Instruction
__device__ __forceinline__
void mma(
    float (&acc)[4],
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1
)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"

        : "+f"(acc[0]), "+f"(acc[1]),
          "+f"(acc[2]), "+f"(acc[3])

        : "r"(a0), "r"(a1),
          "r"(a2), "r"(a3),
          "r"(b0), "r"(b1)
    );
}


// GEMM Kernel
template<typename Config>
__launch_bounds__(Config::THREADS)
__global__ void hgemmV10(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
)
{
    constexpr int BM = Config::BM;
    constexpr int BN = Config::BN;
    constexpr int BK = Config::BK;

    constexpr int WARP_NUM_N = Config::WARP_NUM_N;
    constexpr int THREADS = Config::THREADS;

    constexpr int FM = Config::FM;
    constexpr int FN = Config::FN;

    constexpr int B_PAIRS = Config::B_PAIRS;
    constexpr int K_TILES = Config::K_TILES;

    constexpr int LOAD_A = Config::LOAD_A;
    constexpr int LOAD_B = Config::LOAD_B;

    constexpr int A_SHARED_ROWS = Config::A_SHARED_ROWS;
    constexpr int A_SHARED_COLS = Config::A_SHARED_COLS;


    // Dynamic Shared Memory
    extern __shared__ __align__(32) __half smem[];

    auto sA = reinterpret_cast<
        __half (*)[A_SHARED_ROWS][A_SHARED_COLS]
    >(smem);

    auto sB = reinterpret_cast<
        __half (*)[BK][BN]
    >(smem + 2 * BM * BK);


    const int tid = threadIdx.x;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    // Warp Mapping
    const int warpId = tid / 32;
    const int laneId = tid % 32;

    const int warpRow = warpId / WARP_NUM_N;
    const int warpCol = warpId % WARP_NUM_N;

    // Accumulator Registers
    float acc[FM][FN][4] = {};

    int PHASES = (K + BK - 1) / BK;


    // Prefetch Phase 0

    #pragma unroll
    for (int i = 0; i < LOAD_A; ++i)
    {
        const int vecId = tid + i * THREADS;

        const int aRow = vecId / (BK / VEC);
        const int aCol = (vecId % (BK / VEC)) * VEC;

        int aSwizzleRow;
        int aSwizzleCol;

        swizzleA<Config>(
            aRow, aCol,
            aSwizzleRow, aSwizzleCol
        );

        __pipeline_memcpy_async(
            &sA[0][aSwizzleRow][aSwizzleCol],
            &A[(blockRow + aRow) * K + aCol],
            sizeof(float4)
        );
    }

    #pragma unroll
    for (int i = 0; i < LOAD_B; ++i)
    {
        const int vecId = tid + i * THREADS;

        const int bRow = vecId / (BN / VEC);
        const int bCol = (vecId % (BN / VEC)) * VEC;

        const int bSwizzleCol = swizzleB(bRow, bCol);

        __pipeline_memcpy_async(
            &sB[0][bRow][bSwizzleCol],
            &B[bRow * N + blockCol + bCol],
            sizeof(float4)
        );
    }

    __pipeline_commit();


    // Main Loop
    for (int phase = 0; phase < PHASES; ++phase)
    {
        const int current = phase % 2;
        const int next = (phase + 1) % 2;

        // Wait for Current Buffer
        __pipeline_wait_prior(0);
        __syncthreads();


        // Prefetch Next Phase
        if (phase + 1 < PHASES)
        {
            #pragma unroll
            for (int i = 0; i < LOAD_A; ++i)
            {
                const int vecId = tid + i * THREADS;

                const int aRow = vecId / (BK / VEC);
                const int aCol = (vecId % (BK / VEC)) * VEC;

                int aSwizzleRow;
                int aSwizzleCol;

                swizzleA<Config>(
                    aRow, aCol,
                    aSwizzleRow, aSwizzleCol
                );

                __pipeline_memcpy_async(
                    &sA[next][aSwizzleRow][aSwizzleCol],
                    &A[(blockRow + aRow) * K
                       + (phase + 1) * BK + aCol],
                    sizeof(float4)
                );
            }

            #pragma unroll
            for (int i = 0; i < LOAD_B; ++i)
            {
                const int vecId = tid + i * THREADS;

                const int bRow = vecId / (BN / VEC);
                const int bCol = (vecId % (BN / VEC)) * VEC;

                const int bSwizzleCol = swizzleB(bRow, bCol);

                __pipeline_memcpy_async(
                    &sB[next][bRow][bSwizzleCol],
                    &B[((phase + 1) * BK + bRow) * N
                       + blockCol + bCol],
                    sizeof(float4)
                );
            }

            __pipeline_commit();
        }


        // B Register Double Buffering
        uint32_t bReg[2][B_PAIRS][4];


        // Preload B of kTile 0
        #pragma unroll
        for (int j = 0; j < B_PAIRS; ++j)
        {
            const int bSharedCol =
                warpCol * WARP_TILE_N + j * WN * 2;

            const int bRow = laneId % 16;
            const int bCol = bSharedCol + (laneId / 16) * WN;

            const int bSwizzleCol = swizzleB(bRow, bCol);

            ldmatrixX4Trans(
                bReg[0][j][0],
                bReg[0][j][1],
                bReg[0][j][2],
                bReg[0][j][3],
                &sB[current][bRow][bSwizzleCol]
            );
        }


        // Compute Current Phase
        #pragma unroll
        for (int kTile = 0; kTile < K_TILES; ++kTile)
        {
            const int kOffset = kTile * WK;

            const int currentReg = kTile % 2;
            const int nextReg = (kTile + 1) % 2;


            // Shared -> A Registers
            uint32_t aReg[FM][4];

            #pragma unroll
            for (int i = 0; i < FM; ++i)
            {
                const int aSharedRow =
                    warpRow * WARP_TILE_M + i * WM;

                const int aRow = aSharedRow + laneId % 16;
                const int aCol = kOffset + (laneId / 16) * 8;

                int aSwizzleRow;
                int aSwizzleCol;

                swizzleA<Config>(
                    aRow, aCol,
                    aSwizzleRow, aSwizzleCol
                );

                ldmatrixX4(
                    aReg[i][0],
                    aReg[i][1],
                    aReg[i][2],
                    aReg[i][3],
                    &sA[current][aSwizzleRow][aSwizzleCol]
                );
            }


            // Prefetch B of Next kTile
            if (kTile + 1 < K_TILES)
            {
                const int nextKOffset = (kTile + 1) * WK;

                #pragma unroll
                for (int j = 0; j < B_PAIRS; ++j)
                {
                    const int bSharedCol =
                        warpCol * WARP_TILE_N + j * WN * 2;

                    const int bRow = nextKOffset + laneId % 16;
                    const int bCol =
                        bSharedCol + (laneId / 16) * WN;

                    const int bSwizzleCol = swizzleB(bRow, bCol);

                    ldmatrixX4Trans(
                        bReg[nextReg][j][0],
                        bReg[nextReg][j][1],
                        bReg[nextReg][j][2],
                        bReg[nextReg][j][3],
                        &sB[current][bRow][bSwizzleCol]
                    );
                }
            }


            // MMA
            #pragma unroll
            for (int j = 0; j < B_PAIRS; ++j)
            {
                #pragma unroll
                for (int i = 0; i < FM; ++i)
                {
                    // First B Matrix
                    mma(
                        acc[i][2 * j],
                        aReg[i][0],
                        aReg[i][1],
                        aReg[i][2],
                        aReg[i][3],
                        bReg[currentReg][j][0],
                        bReg[currentReg][j][1]
                    );

                    // Second B Matrix
                    mma(
                        acc[i][2 * j + 1],
                        aReg[i][0],
                        aReg[i][1],
                        aReg[i][2],
                        aReg[i][3],
                        bReg[currentReg][j][2],
                        bReg[currentReg][j][3]
                    );
                }
            }
        }
    }


    // Lane Mapping
    const int groupId = laneId / 4;
    const int threadId = laneId % 4;


    // Registers -> Global: Vectorized Store
    #pragma unroll
    for (int i = 0; i < FM; ++i)
    {
        const int cRow =
            blockRow + warpRow * WARP_TILE_M + i * WM;

        #pragma unroll
        for (int j = 0; j < FN; ++j)
        {
            const int cCol =
                blockCol + warpCol * WARP_TILE_N + j * WN;

            const int r0 = cRow + groupId;
            const int r1 = cRow + groupId + 8;

            const int c0 = cCol + threadId * 2;

            *reinterpret_cast<float2*>(&C[r0 * N + c0]) =
                make_float2(acc[i][j][0], acc[i][j][1]);

            *reinterpret_cast<float2*>(&C[r1 * N + c0]) =
                make_float2(acc[i][j][2], acc[i][j][3]);
        }
    }
}


// Kernel Launcher
template<typename Config>
void launchV10(
    const __half* A,
    const __half* B,
    float* C,
    int M, int N, int K
)
{
    constexpr int BM = Config::BM;
    constexpr int BN = Config::BN;
    constexpr int BK = Config::BK;

    constexpr int THREADS = Config::THREADS;
    constexpr int SMEM_BYTES = Config::SMEM_BYTES;

    if (M <= 0 || N <= 0 || K <= 0 ||
        M % BM != 0 ||
        N % BN != 0 ||
        K % BK != 0)
    {
        std::cerr << "Invalid matrix dimensions.\n"
                  << "M must be divisible by " << BM << '\n'
                  << "N must be divisible by " << BN << '\n'
                  << "K must be divisible by " << BK << '\n';

        std::exit(EXIT_FAILURE);
    }

    if (reinterpret_cast<std::uintptr_t>(A) % 16 != 0 ||
        reinterpret_cast<std::uintptr_t>(B) % 16 != 0)
    {
        std::cerr << "A and B must be 16-byte aligned.\n";

        std::exit(EXIT_FAILURE);
    }

    if (reinterpret_cast<std::uintptr_t>(C) % alignof(float2) != 0)
    {
        std::cerr << "C must be 8-byte aligned.\n";

        std::exit(EXIT_FAILURE);
    }

    const dim3 block(THREADS);
    const dim3 grid(N / BN, M / BM);

    hgemmV10<Config><<<grid, block, SMEM_BYTES>>>(
        A, B, C, M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
}


// Configuration Benchmark
template<typename Config>
int runConfig(
    const char* name,
    int argc,
    char** argv
)
{
    int device;

    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop{};

    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));


    // Dynamic Shared Memory Limit
    int maxSharedMemory;

    CUDA_CHECK(cudaDeviceGetAttribute(
        &maxSharedMemory,
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        device
    ));

    if (Config::SMEM_BYTES > maxSharedMemory)
    {
        std::cerr << "Shared memory requirement exceeds "
                  << "device limit.\n";

        return EXIT_FAILURE;
    }


    // Configure Dynamic Shared Memory
    CUDA_CHECK(cudaFuncSetAttribute(
        hgemmV10<Config>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        Config::SMEM_BYTES
    ));


    // Kernel Resource Usage
    cudaFuncAttributes attr{};

    CUDA_CHECK(cudaFuncGetAttributes(
        &attr,
        hgemmV10<Config>
    ));


    // Theoretical Occupancy
    int activeBlocks;

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocks,
        hgemmV10<Config>,
        Config::THREADS,
        Config::SMEM_BYTES
    ));

    if (activeBlocks <= 0)
    {
        std::cerr << "Invalid kernel resource configuration.\n";

        return EXIT_FAILURE;
    }

    const int activeWarps =
        activeBlocks * Config::WARPS;

    const int maxWarps =
        prop.maxThreadsPerMultiProcessor / prop.warpSize;

    const double occupancy =
        100.0 * activeWarps / maxWarps;


    // Print Configuration
    std::cout << "\n========== Configuration ==========\n";

    std::cout << "GPU: " << prop.name << '\n';

    std::cout << "SM Count: "
              << prop.multiProcessorCount << '\n';

    std::cout << "Block Tile: "
              << Config::BM << " x "
              << Config::BN << " x "
              << Config::BK << '\n';

    std::cout << "Warp Tile: "
              << WARP_TILE_M << " x "
              << WARP_TILE_N << '\n';

    std::cout << "Warp Mapping: "
              << Config::WARP_NUM_M << " x "
              << Config::WARP_NUM_N << '\n';

    std::cout << "Warps per Block: "
              << Config::WARPS << '\n';

    std::cout << "Threads per Block: "
              << Config::THREADS << '\n';

    std::cout << "K Tiles: "
              << Config::K_TILES << '\n';

    std::cout << "LOAD_A: "
              << Config::LOAD_A << '\n';

    std::cout << "LOAD_B: "
              << Config::LOAD_B << '\n';

    std::cout << "A Physical Layout: "
              << Config::A_SHARED_ROWS << " x "
              << Config::A_SHARED_COLS << '\n';

    std::cout << "Dynamic Shared Memory: "
              << Config::SMEM_BYTES << " B\n";

    std::cout << "Static Shared Memory: "
              << attr.sharedSizeBytes << " B\n";

    std::cout << "Registers per Thread: "
              << attr.numRegs << '\n';

    std::cout << "Local Memory per Thread: "
              << attr.localSizeBytes << " B\n";

    std::cout << "Active Blocks per SM: "
              << activeBlocks << '\n';

    std::cout << "Active Warps per SM: "
              << activeWarps << '\n';

    std::cout << "Theoretical Occupancy: "
              << std::fixed << std::setprecision(2)
              << occupancy << "%\n";

    std::cout << "===================================\n";


    return runBenchmark(
        name,
        launchV10<Config>,
        argc,
        argv
    );
}


// Main
int main(int argc, char** argv)
{
    if (argc != 2 && argc != 3 && argc != 5)
    {
        std::cerr << "Usage:\n"
                  << argv[0] << " config\n"
                  << argv[0] << " config size\n"
                  << argv[0] << " config M N K\n"
                  << "config: 0 - 5\n";

        return EXIT_FAILURE;
    }

    if (argv[1][0] < '0' ||
        argv[1][0] > '5' ||
        argv[1][1] != '\0')
    {
        std::cerr << "Invalid configuration ID.\n";

        return EXIT_FAILURE;
    }

    const int configId = argv[1][0] - '0';

    switch (configId)
    {
        case 0:
            return runConfig<Config0>(
                "HGEMM V10 - Config 0",
                argc - 1, argv + 1
            );

        case 1:
            return runConfig<Config1>(
                "HGEMM V10 - Config 1",
                argc - 1, argv + 1
            );

        case 2:
            return runConfig<Config2>(
                "HGEMM V10 - Config 2",
                argc - 1, argv + 1
            );

        case 3:
            return runConfig<Config3>(
                "HGEMM V10 - Config 3",
                argc - 1, argv + 1
            );

        case 4:
            return runConfig<Config4>(
                "HGEMM V10 - Config 4",
                argc - 1, argv + 1
            );

        case 5:
            return runConfig<Config5>(
                "HGEMM V10 - Config 5",
                argc - 1, argv + 1
            );

        default:
            return EXIT_FAILURE;
    }
}