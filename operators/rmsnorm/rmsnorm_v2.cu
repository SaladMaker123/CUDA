#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <limits>

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
constexpr unsigned int FULL_MASK = 0xffffffffu;

// Warp-level reduction
static __device__ __forceinline__ float warpReduceSum(float sum)
{
    for(int offset = WARP_SIZE / 2; offset > 0;offset /= 2)
    {
        sum += __shfl_down_sync(FULL_MASK,sum,offset);
    }
    return sum;
}
// RMSNorm V2: Warp Shuffle Reduction
template<typename T>
__global__ void rmsnormV2Kernel(
    const T* X,
    const T* W,
    T* Y,
    int N,
    float eps
)
{
    int tid = threadIdx.x;
    int row = blockIdx.x;
    int lane = tid % WARP_SIZE;
    int warp = tid / WARP_SIZE;

    int64_t rowOffset = static_cast<int64_t>(row) * N;
    __shared__ float sSum[NUM_WARPS];
    // STEP 1: Thread-local sum of squares
    float sum = 0.0f;
    for(int col = tid;col < N;col += BLOCK_SIZE)
    {
        float value = static_cast<float>(X[rowOffset + col]);
        sum += value * value;
    }
    // STEP 2: Warp-level reduction
    sum = warpReduceSum(sum);
    // STEP 3: Store warp partial sums
    if(lane == 0)
    {
        sSum[warp] = sum;
    }
    __syncthreads();
    // STEP 4: Block-level reduction
    if(warp == 0)
    {
        float blockSum = lane < NUM_WARPS ? sSum[lane] : 0.0f;
        blockSum = warpReduceSum(blockSum);
        if(lane == 0)
        {
            sSum[0] = rsqrtf(blockSum / static_cast<float>(N) + eps);
        }
    }
    __syncthreads();
    float invRms = sSum[0];
    // STEP 5: Normalize and apply weight
    for(int col = tid; col < N; col += BLOCK_SIZE)
    {
        float value = static_cast<float>(X[rowOffset + col]);

        // Match Qwen2 RMSNorm rounding order
        T normalized = static_cast<T>(value * invRms);

        float weight = static_cast<float>(W[col]);

        Y[rowOffset + col] = static_cast<T>(
            static_cast<float>(normalized) * weight
        );
    }
}
// CUDA Launcher
at::Tensor rmsnormV2(
    at::Tensor X,
    at::Tensor W,
    double eps
)
{
    // STEP 1: Check input tensors
    TORCH_CHECK(X.is_cuda(),
                "X must be a CUDA tensor");

    TORCH_CHECK(W.is_cuda(),
                "W must be a CUDA tensor");

    TORCH_CHECK(X.device() == W.device(),
                "X and W must be on the same device");

    TORCH_CHECK(X.scalar_type() == at::kHalf ||
                X.scalar_type() == at::kFloat,
                "X must be FP16 or FP32");

    TORCH_CHECK(X.scalar_type() == W.scalar_type(),
                "X and W must have the same dtype");

    TORCH_CHECK(X.dim() >= 1,
                "X must have at least one dimension");

    TORCH_CHECK(W.dim() == 1,
                "W must be a 1D tensor");

    TORCH_CHECK(X.is_contiguous(),
                "X must be contiguous");

    TORCH_CHECK(W.is_contiguous(),
                "W must be contiguous");

    TORCH_CHECK(std::isfinite(eps) && eps > 0.0,
                "Epsilon must be finite and positive");

    // STEP 2: Get matrix dimensions
    int64_t hiddenSize = X.size(-1);

    TORCH_CHECK(hiddenSize > 0 &&
                hiddenSize <=
                std::numeric_limits<int>::max(),
                "Invalid hidden size");

    TORCH_CHECK(W.numel() == hiddenSize,
                "Weight size must match hidden size");

    int N = static_cast<int>(hiddenSize);

    int64_t M = X.numel() / N;

    TORCH_CHECK(M <= std::numeric_limits<int>::max(),
                "Grid size is too large");

    // STEP 3: Set CUDA device
    const c10::cuda::CUDAGuard deviceGuard(
        X.device()
    );

    // STEP 4: Allocate output tensor
    auto Y = at::empty_like(X);

    if(M == 0)
    {
        return Y;
    }

    // STEP 5: Get current CUDA stream
    cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream().stream();

    // STEP 6: Launch kernel
    if(X.scalar_type() == at::kHalf)
    {
        rmsnormV2Kernel<c10::Half>
            <<<M, BLOCK_SIZE, 0, stream>>>(
                X.data_ptr<c10::Half>(),
                W.data_ptr<c10::Half>(),
                Y.data_ptr<c10::Half>(),
                N,
                static_cast<float>(eps)
            );
    }
    else
    {
        rmsnormV2Kernel<float>
            <<<M, BLOCK_SIZE, 0, stream>>>(
                X.data_ptr<float>(),
                W.data_ptr<float>(),
                Y.data_ptr<float>(),
                N,
                static_cast<float>(eps)
            );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return Y;
}
