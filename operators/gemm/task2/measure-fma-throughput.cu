
#include <cstdio>
#include <cmath>

#include <cuda_runtime.h>

#include "../../../common/cuda_check.h"

// ================================
// Task 2: FMA Throughput
// ================================
constexpr int BLOCK_SIZE = 256;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int ILP = 4;
constexpr int N_ITER = 40000;

// ================================
// FMA Throughput Kernel
// ================================
__global__ void peak_flops_kernel(
    float* __restrict__ out,
    int n,
    float a,
    float b
)
{
    const int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    float c[TM * TN];

    // Per-thread accumulator initialization
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm)
    {
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn)
        {
            c[tm * TN + tn] =
                static_cast<float>(tm * TN + tn) +
                static_cast<float>(threadIdx.x) * 0.5f;
        }
    }

    // FMA computation
    for (int i = 0; i < n; ++i)
    {
        #pragma unroll
        for (int ilp = 0; ilp < ILP; ++ilp)
        {
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm)
            {
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn)
                {
                    c[tm * TN + tn] = __fmaf_rn(
                        a,
                        b,
                        c[tm * TN + tn]
                    );
                }
            }
        }
    }

    float sum = 0.0f;

    #pragma unroll
    for (int tm = 0; tm < TM; ++tm)
    {
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn)
        {
            sum += c[tm * TN + tn];
        }
    }

    out[idx] = sum;
}

// ================================
// Main
// ================================
int main()
{
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const int sm_count = prop.multiProcessorCount;
    const int n_blocks = sm_count * 28;
    const int total_threads = n_blocks * BLOCK_SIZE;

    constexpr float A = 3.0f;
    constexpr float B = 7.0f;

    float* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_out,
        static_cast<size_t>(total_threads) * sizeof(float)
    ));

    cudaEvent_t ev_start, ev_stop;

    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Warmup
    peak_flops_kernel<<<n_blocks, BLOCK_SIZE>>>(
        d_out, N_ITER, A, B
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    CUDA_CHECK(cudaEventRecord(ev_start));

    peak_flops_kernel<<<n_blocks, BLOCK_SIZE>>>(
        d_out, N_ITER, A, B
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &ms, ev_start, ev_stop
    ));

    // Correctness Check
    float result = 0.0f;

    CUDA_CHECK(cudaMemcpy(
        &result,
        d_out,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    const double expected =
        static_cast<double>(TM * TN) *
        N_ITER * ILP * A * B +
        (TM * TN) * (TM * TN - 1) / 2.0;

    const double relative_error =
        std::fabs(static_cast<double>(result) - expected)
        / expected;

    if (!std::isfinite(result) ||
        relative_error > 1e-3)
    {
        std::printf(
            "Validation: FAIL (result=%.2f, expected=%.2f)\n",
            result, expected
        );

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
        CUDA_CHECK(cudaFree(d_out));

        return 1;
    }

    // TFLOPS Calculation
    const double total_flops =
        static_cast<double>(total_threads) *
        N_ITER * TM * TN * ILP * 2.0;

    const double tflops =
        total_flops / (static_cast<double>(ms) * 1e-3)
        / 1e12;

    std::printf("=== FMA Throughput ===\n\n");

    std::printf("GPU:       %s\n", prop.name);
    std::printf("SM Count:  %d\n", sm_count);
    std::printf("Blocks:    %d\n", n_blocks);
    std::printf("Threads:   %d\n", total_threads);
    std::printf("Time:      %.3f ms\n", ms);
    std::printf("Peak FP32: %.2f TFLOPS\n", tflops);
    std::printf("Validation: PASS\n");

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
