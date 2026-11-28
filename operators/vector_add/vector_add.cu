#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
//CUDA Kernel
__global__ void vectorAddKernel(
    const float* A,
    const float* B,
    float* C,
    int64_t N
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}
// CUDA Launcher
at::Tensor vectorAdd(
    at::Tensor A,
    at::Tensor B
)
{
    // STEP 1: Check input tensors
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    TORCH_CHECK(A.device() == B.device(),
                "A and B must be on the same device");

    TORCH_CHECK(A.scalar_type() == at::kFloat,
                "A must be FP32");
    TORCH_CHECK(B.scalar_type() == at::kFloat,
                "B must be FP32");

    TORCH_CHECK(A.sizes() == B.sizes(),
                "A and B must have the same shape");

    TORCH_CHECK(A.is_contiguous(),
                "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(),
                "B must be contiguous");

    // STEP 2: Set CUDA device
    const c10::cuda::CUDAGuard deviceGuard(A.device());

    // STEP 3: Allocate output tensor
    auto C = at::empty_like(A);

    int64_t N = A.numel();

    if (N == 0)
    {
        return C;
    }

    // STEP 4: Get device pointers
    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // STEP 5: Configure kernel launch
    constexpr int blockSize = 256;

    int64_t gridSize = (N - 1) / blockSize + 1;

    TORCH_CHECK(
        gridSize <= std::numeric_limits<int>::max(),
        "Grid size is too large"
    );

    // STEP 6: Get current CUDA stream
    cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream().stream();

    // STEP 7: Launch kernel
    vectorAddKernel<<<gridSize, blockSize, 0, stream>>>(
        d_A,
        d_B,
        d_C,
        N
    );

    // STEP 8: Check launch errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}