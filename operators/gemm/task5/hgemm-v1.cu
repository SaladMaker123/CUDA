#include "utils.h"
#include <cstdint>
// MMA Tile
constexpr int WM = 16;
constexpr int WN = 8;
constexpr int WK = 16;
// Block Tile
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
static_assert(BK == WK);
// Warp Tile
constexpr int WARP_TILE_M = 32;
constexpr int WARP_TILE_N = 32;
// Warp Mapping
constexpr int WARP_NUM_M = BM / WARP_TILE_M;
constexpr int WARP_NUM_N = BN / WARP_TILE_N;
constexpr int WARPS = WARP_NUM_M * WARP_NUM_N;
constexpr int THREADS = WARPS * 32;
// Fragments per Warp
constexpr int FM = WARP_TILE_M / WM;
constexpr int FN = WARP_TILE_N / WN;
// Vector Width
constexpr int VEC = sizeof(float4) / sizeof(__half);
// =====================================
// Pack FP16 x 2
// =====================================
__device__ __forceinline__
uint32_t packHalf2(__half lo,__half hi)
{
    __half2 value = __halves2half2(lo,hi);
    return *reinterpret_cast<uint32_t*>(&value);
}
// =====================================
// MMA Instruction
// =====================================
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
// =====================================
// V1: MMA GEMM
// =====================================
__launch_bounds__(THREADS)
__global__ void hgemmV1(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
)
{
    __shared__ __align__(32) __half sA[BM][BK];
    __shared__ __align__(32) __half sB[BK][BN];
    const int tid = threadIdx.x;
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    // Warp Mapping
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int warpRow = warpId / WARP_NUM_N;
    const int warpCol = warpId % WARP_NUM_N;
    // Lane Mapping
    const int groupId = laneId / 4;
    const int threadId = laneId % 4;
    // Accumulator Registers
    float acc[FM][FN][4] = {};
    // Global -> Shared Mapping
    const int aRow = tid / (BK / VEC);
    const int aCol = (tid % (BK / VEC)) * VEC;

    const int bRow = tid / (BN / VEC);
    const int bCol = (tid % (BN / VEC)) * VEC;
    int PHASES = (K + BK - 1) / BK;
    // =====================================
    // Main Loop
    // =====================================
    for (int phase = 0; phase < PHASES; ++phase)
    {
        // Global -> Shared: A
        *reinterpret_cast<float4*>(&sA[aRow][aCol]) = 
        *reinterpret_cast<const float4*>(&A[(blockRow + aRow) * K + phase * BK + aCol]);
        // Global -> Shared: B
        *reinterpret_cast<float4*>(&sB[bRow][bCol]) = 
        *reinterpret_cast<const float4*>(&B[(phase * BK + bRow) * N + blockCol + bCol]);
        __syncthreads();
        // =====================================
        // Shared -> Registers -> MMA
        // =====================================
        #pragma unroll
        for(int i = 0;i < FM;++i)
        {
            const int aSharedRow = warpRow * WARP_TILE_M + i * WM;
            //Load A Registers
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sA[aSharedRow + groupId][threadId * 2]);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sA[aSharedRow + groupId + 8][threadId * 2]);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sA[aSharedRow + groupId][threadId * 2 + 8]);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sA[aSharedRow + groupId + 8][threadId * 2 + 8]);
            #pragma unroll
            for (int j = 0; j < FN; ++j)
            {
                const int bSharedCol = warpCol * WARP_TILE_N + j * WN;
                // Load B Registers
                const uint32_t b0 = packHalf2(sB[threadId * 2][bSharedCol + groupId],
                    sB[threadId * 2 + 1][bSharedCol + groupId]);
                const uint32_t b1 = packHalf2(sB[threadId * 2 + 8][bSharedCol + groupId],
                    sB[threadId * 2 + 9][bSharedCol + groupId]);
                // MMA Compute
                mma(
                    acc[i][j],
                    a0, a1, a2, a3,
                    b0, b1
                );
            }
        }
        __syncthreads();
    }
    // =====================================
    // Registers -> Global: C
    // =====================================
    #pragma unroll
    for (int i = 0; i < FM; ++i)
    {
        const int cRow = blockRow + warpRow * WARP_TILE_M + i * WM;
        #pragma unroll
        for (int j = 0; j < FN; ++j)
        {
            const int cCol = blockCol + warpCol * WARP_TILE_N + j * WN;
            const int r0 = cRow + groupId;
            const int c0 = cCol + threadId * 2;

            const int r1 = cRow + groupId + 8;
            const int c1 = cCol + threadId * 2 + 1;
            C[r0 * N + c0] = acc[i][j][0];
            C[r0 * N + c1] = acc[i][j][1];
            C[r1 * N + c0] = acc[i][j][2];
            C[r1 * N + c1] = acc[i][j][3];
        }
    }
}
// =====================================
// Kernel Launcher
// =====================================
void launchV1(
    const __half* A,
    const __half* B,
    float* C,
    int M,
    int N,
    int K
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

    const dim3 grid(
        N / BN,
        M / BM
    );
    hgemmV1<<<grid, block>>>(
        A, B, C, M, N, K
    );
    CUDA_CHECK(cudaGetLastError());
}
// =====================================
// Main
// =====================================

int main(int argc, char** argv)
{
    return runBenchmark(
        "HGEMM V1 - MMA PTX",
        launchV1,
        argc,
        argv
    );
}

