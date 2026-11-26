#include "utils.h"
#include <cstdint>
#include <cuda_pipeline.h>

// MMA Tile
constexpr int WM = 16;
constexpr int WN = 8;
constexpr int WK = 16;

// Block Tile
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

static_assert(BK % WK == 0);

// Warp Tile
constexpr int WARP_TILE_M = 64;
constexpr int WARP_TILE_N = 32;

// Warp Mapping
constexpr int WARP_NUM_M = BM / WARP_TILE_M;
constexpr int WARP_NUM_N = BN / WARP_TILE_N;
constexpr int WARPS = WARP_NUM_M * WARP_NUM_N;
constexpr int THREADS = WARPS * 32;

// Fragments per Warp
constexpr int FM = WARP_TILE_M / WM;
constexpr int FN = WARP_TILE_N / WN;

// B Fragment Pairs
constexpr int B_PAIRS = FN / 2;

static_assert(FN % 2 == 0);

// K Tiles
constexpr int K_TILES = BK / WK;

// Vector Width
constexpr int VEC = sizeof(float4) / sizeof(__half);

// Swizzle Size
constexpr int SWIZZLE_SIZE = 8;

// Vectors per Thread
constexpr int LOAD_A = BM * BK / (THREADS * VEC);
constexpr int LOAD_B = BK * BN / (THREADS * VEC);

static_assert(BM * BK % (THREADS * VEC) == 0);
static_assert(BK * BN % (THREADS * VEC) == 0);
static_assert(BM % 2 == 0);
static_assert(BK * 2 / SWIZZLE_SIZE == 8);


// A Swizzle
__device__ __forceinline__
void swizzleA(
    int row,
    int col,
    int& newRow,
    int& newCol
)
{
    newRow = row / 2;

    const int mergedCol = (row % 2) * BK + col;

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


// Load A Fragment
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


// Load Two B Fragments
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
__launch_bounds__(THREADS)
__global__ void hgemmV8A(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
)
{
    __shared__ __align__(32) __half sA[2][BM / 2][BK * 2];
    __shared__ __align__(32) __half sB[2][BK][BN];

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

        swizzleA(
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

        // Protect Current Reads and Previous Buffer Reuse
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

                swizzleA(
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


        // Compute Current Phase
        #pragma unroll
        for (int kTile = 0; kTile < K_TILES; ++kTile)
        {
            const int kOffset = kTile * WK;

            // Shared -> A Registers
            uint32_t a0[FM];
            uint32_t a1[FM];
            uint32_t a2[FM];
            uint32_t a3[FM];

            #pragma unroll
            for (int i = 0; i < FM; ++i)
            {
                const int aSharedRow =
                    warpRow * WARP_TILE_M + i * WM;

                const int aRow = aSharedRow + laneId % 16;
                const int aCol = kOffset + (laneId / 16) * 8;

                int aSwizzleRow;
                int aSwizzleCol;

                swizzleA(
                    aRow, aCol,
                    aSwizzleRow, aSwizzleCol
                );

                ldmatrixX4(
                    a0[i], a1[i], a2[i], a3[i],
                    &sA[current][aSwizzleRow][aSwizzleCol]
                );
            }


            // Shared -> B Registers
            #pragma unroll
            for (int j = 0; j < B_PAIRS; ++j)
            {
                const int bSharedCol =
                    warpCol * WARP_TILE_N + j * WN * 2;

                const int bRow = kOffset + laneId % 16;
                const int bCol = bSharedCol + (laneId / 16) * WN;

                const int bSwizzleCol = swizzleB(bRow, bCol);

                uint32_t b0, b1, b2, b3;

                ldmatrixX4Trans(
                    b0, b1, b2, b3,
                    &sB[current][bRow][bSwizzleCol]
                );


                // MMA
                #pragma unroll
                for (int i = 0; i < FM; ++i)
                {
                    // First B Fragment
                    mma(
                        acc[i][2 * j],
                        a0[i], a1[i], a2[i], a3[i],
                        b0, b1
                    );

                    // Second B Fragment
                    mma(
                        acc[i][2 * j + 1],
                        a0[i], a1[i], a2[i], a3[i],
                        b2, b3
                    );
                }
            }
        }
    }


    // Lane Mapping
    const int groupId = laneId / 4;
    const int threadId = laneId % 4;


    // Registers -> Global
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
            const int c1 = cCol + threadId * 2 + 1;

            C[r0 * N + c0] = acc[i][j][0];
            C[r0 * N + c1] = acc[i][j][1];
            C[r1 * N + c0] = acc[i][j][2];
            C[r1 * N + c1] = acc[i][j][3];
        }
    }
}


// Kernel Launcher
void launchV8A(
    const __half* A,
    const __half* B,
    float* C,
    int M, int N, int K
)
{
    if (M <= 0 || N <= 0 || K <= 0 ||
        M % BM != 0 ||
        N % BN != 0 ||
        K % BK != 0)
    {
        std::cerr << "Invalid matrix dimensions: "
                  << "M % BM, N % BN and K % BK must be zero.\n";

        std::exit(EXIT_FAILURE);
    }

    const dim3 block(THREADS);
    const dim3 grid(N / BN, M / BM);

    hgemmV8A<<<grid, block>>>(A, B, C, M, N, K);

    CUDA_CHECK(cudaGetLastError());
}


// Main
int main(int argc, char** argv)
{
    return runBenchmark(
        "HGEMM V8-A - B Fragment Pair Loading",
        launchV8A,
        argc,
        argv
    );
}