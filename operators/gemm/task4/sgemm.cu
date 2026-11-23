#ifndef BM
#define BM 128
#endif

#ifndef BN
#define BN 128
#endif

#ifndef BK
#define BK 8
#endif

#ifndef TM
#define TM 8
#endif

#ifndef TN
#define TN 8
#endif
#include "utils.h"

// Vectorized load
constexpr int VEC = 4;
// Thread mapping for C
constexpr int THREADS_X = BN / TN;//x方向线程数量
constexpr int THREADS_Y = BM / TM;//y方向线程数量
constexpr int THREADS = THREADS_X * THREADS_Y;
// Thread mapping for A
constexpr int A_VECS_PER_ROW = BK / VEC;
constexpr int LOAD_A_STRIDE = THREADS / A_VECS_PER_ROW;
// Thread mapping for B
constexpr int B_VECS_PER_ROW = BN / VEC;
constexpr int LOAD_B_STRIDE = THREADS / B_VECS_PER_ROW;
#define LOAD_FLOAT4(dst, src) \
    (*reinterpret_cast<float4 *>(dst) = \
     *reinterpret_cast<const float4 *>(src))
static_assert(BM % TM == 0);
static_assert(BN % TN == 0);

static_assert(BK % VEC == 0);
static_assert(BN % VEC == 0);
static_assert(TM % VEC == 0);
static_assert(TN % VEC == 0);

static_assert(THREADS >= 32);
static_assert(THREADS % 32 == 0);
static_assert(THREADS <= 1024);

static_assert(THREADS % A_VECS_PER_ROW == 0);
static_assert(THREADS % B_VECS_PER_ROW == 0);
__launch_bounds__(THREADS)
__global__ void sgemm(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
)
{
    const int tid = threadIdx.x;
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    // A: Global Memory load mapping
    const int aRow = tid / A_VECS_PER_ROW;
    const int aCol = (tid % A_VECS_PER_ROW) * VEC;
    // B: Global Memory load mapping
    const int bRow = tid / B_VECS_PER_ROW;
    const int bCol = (tid % B_VECS_PER_ROW) * VEC;
    // C: Thread tile mapping
    const int threadRow = tid / THREADS_X;
    const int threadCol = tid % THREADS_X;

    const int cTileRow = threadRow * TM;
    const int cTileCol = threadCol * TN;

    const int cRow = blockRow * BM + cTileRow;
    const int cCol = blockCol * BN + cTileCol;
    // Shared Memory
    __shared__ __align__(16) float sA[BK * BM];
    __shared__ __align__(16) float sB[BK * BN];

    // Register tiles
    float acc[TM * TN] = {0.0f};
    alignas(16) float rA[TM];
    alignas(16) float rB[TN];
    // K tiling
    for (int kBase = 0; kBase < K; kBase += BK)
    {
        // =====================================
        // Global Memory -> Shared Memory: A
        // =====================================
        #pragma unroll
        for(int load = 0;load < BM; load += LOAD_A_STRIDE)
        {
            if(aRow + load < BM)
            {
                const int aIdx = (blockRow * BM + aRow + load) * K+ kBase + aCol;
                float4 aVec;
                LOAD_FLOAT4(&aVec, &A[aIdx]);
                // Transposed layout: sA[BK][BM]
                const float* a = reinterpret_cast<const float*>(&aVec);
                #pragma unroll
                for (int x = 0; x < VEC; ++x)
                {
                    const int sIdx = (aCol + x) * BM + aRow + load;
                    sA[sIdx] = a[x];
                }

            }
        }
        // =====================================
        // Global Memory -> Shared Memory: B
        // =====================================
        #pragma unroll
        for (int load = 0;load < BK;load += LOAD_B_STRIDE)
        {
            if (bRow + load < BK)
            {
                const int bIdx = (kBase + bRow + load) * N + blockCol * BN + bCol;
                const int sIdx = (bRow + load) * BN + bCol;
                LOAD_FLOAT4(&sB[sIdx], &B[bIdx]);
            }
        }
        __syncthreads();
        // =====================================
        // Shared Memory -> Register -> FMA
        // =====================================
        #pragma unroll
        for (int k = 0; k < BK; ++k)
        {
            // Load A register tile
            #pragma unroll
            for (int tm = 0; tm < TM; tm += VEC)
            {
                const int sIdx = k * BM + cTileRow + tm;
                LOAD_FLOAT4(&rA[tm], &sA[sIdx]);
            }
            // Load B register tile
            #pragma unroll
            for (int tn = 0; tn < TN; tn += VEC)
            {
                const int sIdx = k * BN + cTileCol + tn;
                LOAD_FLOAT4(&rB[tn], &sB[sIdx]);
            }
            // TM x TN outer product
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) 
            {
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn) 
                {
                    acc[tm * TN + tn] += rA[tm] * rB[tn];
                }
            }
        }
        __syncthreads();
    }
    // =====================================
    // Register -> Global Memory: C
    // =====================================
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm)
    {
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn)
        {
            const int cIdx = (cRow + tm) * N + cCol + tn;
            C[cIdx] = alpha * acc[tm * TN + tn] + beta * C[cIdx];
        }
    }
}
// =========================================
// Main
// =========================================
int main()
{
    printf("\n");
    for (int n = 256; n <= 1024; n *= 2)
    {
        if(benchmark(sgemm, n, true) != 0)
        {
            return EXIT_FAILURE;
        }
    }
    if (benchmark(sgemm, 4096, false) != 0) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}