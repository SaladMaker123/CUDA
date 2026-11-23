
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>

#include "../../../common/cuda_check.h"

// ================================
// Task 2: FMA Latency
// ================================
constexpr int N_BLOCKS = 1;
constexpr int N_THREADS = 1;
constexpr int N_ITER = 200000;
constexpr int MAX_ILP = 8;

// ================================
// Warmup Kernel
// ================================
template <int D, int I>
__global__ void kernel_warmup(
    const float* __restrict__ c_init,
    float* __restrict__ c_state,
    int n_iter,
    float a,
    float b
)
{
    float c[I];

    #pragma unroll
    for (int i = 0; i < I; ++i)
    {
        c[i] = c_init[i] +
            static_cast<float>(threadIdx.x) * 0.5f;
    }

    for (int iter = 0; iter < n_iter; ++iter)
    {
        #pragma unroll
        for (int d = 0; d < D; ++d)
        {
            #pragma unroll
            for (int i = 0; i < I; ++i)
            {
                c[i] = __fmaf_rn(a, b, c[i]);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < I; ++i)
    {
        c_state[i] = c[i];
    }
}

// ================================
// Timed Kernel
// ================================
template <int D, int I>
__global__ void kernel_timed(
    const float* __restrict__ c_state,
    float* __restrict__ sink,
    int n_iter,
    float a,
    float b
)
{
    float c[I];

    #pragma unroll
    for (int i = 0; i < I; ++i)
    {
        c[i] = c_state[i] +
            static_cast<float>(threadIdx.x) * 0.5f;
    }

    for (int iter = 0; iter < n_iter; ++iter)
    {
        #pragma unroll
        for (int d = 0; d < D; ++d)
        {
            #pragma unroll
            for (int i = 0; i < I; ++i)
            {
                c[i] = __fmaf_rn(a, b, c[i]);
            }
        }
    }

    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < I; ++i)
    {
        sum += c[i];
    }

    sink[threadIdx.x] = sum;
}

// ================================
// Shared Benchmark Function
// ================================
template <int D, int I>
double measure_case(
    float* h_init,
    float* d_init,
    float* d_state,
    float* d_sink,
    int n_iter,
    int clockRateKHz,
    cudaEvent_t start,
    cudaEvent_t stop
)
{
    for (int i = 0; i < I; ++i)
    {
        h_init[i] = static_cast<float>(i + 1);
    }

    CUDA_CHECK(cudaMemcpy(
        d_init,
        h_init,
        I * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    constexpr float A = 3.0f;
    constexpr float B = 7.0f;

    // Warmup
    kernel_warmup<D, I><<<N_BLOCKS, N_THREADS>>>(
        d_init, d_state, n_iter, A, B
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));

    kernel_timed<D, I><<<N_BLOCKS, N_THREADS>>>(
        d_state, d_sink, n_iter, A, B
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &ms, start, stop
    ));

    float result = 0.0f;

    CUDA_CHECK(cudaMemcpy(
        &result,
        d_sink,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    if (!std::isfinite(result))
    {
        std::fprintf(stderr, "Validation failed\n");
        std::exit(EXIT_FAILURE);
    }

    // Clock rate is in kHz
    const double cyclesPerIter =
        static_cast<double>(ms) *
        clockRateKHz / n_iter;

    return cyclesPerIter;
}

// ================================
// Experiment 1: D x ILP Sweep
// ================================
template <int D, int I>
void run_case(
    float* h_init,
    float* d_init,
    float* d_state,
    float* d_sink,
    int clockRateKHz,
    cudaEvent_t start,
    cudaEvent_t stop
)
{
    const double cycles = measure_case<D, I>(
        h_init, d_init, d_state, d_sink,
        N_ITER, clockRateKHz, start, stop
    );

    std::printf("%6d %6d %14.1f\n", D, I, cycles);
}

template <int D>
void run_ilp_sweep(
    float* h_init,
    float* d_init,
    float* d_state,
    float* d_sink,
    int clockRateKHz,
    cudaEvent_t start,
    cudaEvent_t stop
)
{
    run_case<D, 1>(h_init, d_init, d_state, d_sink,
                   clockRateKHz, start, stop);
    run_case<D, 2>(h_init, d_init, d_state, d_sink,
                   clockRateKHz, start, stop);
    run_case<D, 3>(h_init, d_init, d_state, d_sink,
                   clockRateKHz, start, stop);
    run_case<D, 4>(h_init, d_init, d_state, d_sink,
                   clockRateKHz, start, stop);
    run_case<D, 5>(h_init, d_init, d_state, d_sink,
                   clockRateKHz, start, stop);
    run_case<D, 6>(h_init, d_init, d_state, d_sink,
                   clockRateKHz, start, stop);
    run_case<D, 7>(h_init, d_init, d_state, d_sink,
                   clockRateKHz, start, stop);
    run_case<D, 8>(h_init, d_init, d_state, d_sink,
                   clockRateKHz, start, stop);
}

// ================================
// Main
// ================================
int main()
{
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int clockRateKHz = 0;

    CUDA_CHECK(cudaDeviceGetAttribute(
        &clockRateKHz,
        cudaDevAttrClockRate,
        0
    ));

    float h_init[MAX_ILP];

    float* d_init = nullptr;
    float* d_state = nullptr;
    float* d_sink = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_init, MAX_ILP * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_state, MAX_ILP * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_sink, sizeof(float)
    ));

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ================================
    // Experiment 1
    // ================================
    std::printf(
        "=== Experiment 1: full D x ILP sweep ===\n\n"
    );

    std::printf("%6s %6s %14s\n",
                "D", "ILP", "Cycles/iter");

    std::printf("%6s %6s %14s\n",
                "-", "---", "-----------");

    // Original project order
    run_ilp_sweep<2>(h_init, d_init, d_state, d_sink,
                     clockRateKHz, start, stop);
    run_ilp_sweep<4>(h_init, d_init, d_state, d_sink,
                     clockRateKHz, start, stop);
    run_ilp_sweep<6>(h_init, d_init, d_state, d_sink,
                     clockRateKHz, start, stop);
    run_ilp_sweep<8>(h_init, d_init, d_state, d_sink,
                     clockRateKHz, start, stop);
    run_ilp_sweep<10>(h_init, d_init, d_state, d_sink,
                      clockRateKHz, start, stop);
    run_ilp_sweep<1>(h_init, d_init, d_state, d_sink,
                     clockRateKHz, start, stop);
    run_ilp_sweep<12>(h_init, d_init, d_state, d_sink,
                      clockRateKHz, start, stop);
    run_ilp_sweep<16>(h_init, d_init, d_state, d_sink,
                      clockRateKHz, start, stop);
    run_ilp_sweep<20>(h_init, d_init, d_state, d_sink,
                      clockRateKHz, start, stop);

    // ================================
    // Experiment 2: Empty Baseline
    // ================================
    std::printf(
        "\n=== Experiment 2: empty loop baseline "
        "(D=0, ILP=1) ===\n\n"
    );

    std::printf("%6s %6s %14s\n",
                "D", "ILP", "Cycles/iter");

    const double baseline = measure_case<0, 1>(
        h_init, d_init, d_state, d_sink,
        N_ITER, clockRateKHz, start, stop
    );

    std::printf("%6d %6d %14.1f\n",
                0, 1, baseline);

    // ================================
    // Experiment 3: Iteration Sensitivity
    // ================================
    std::printf(
        "\n=== Experiment 3: n_iter sensitivity "
        "(D=4, ILP=1) ===\n\n"
    );

    std::printf("%10s %14s\n",
                "n_iter", "Cycles/iter");

    const int iters[] = {
        1000, 5000, 20000, 50000,
        100000, 200000, 500000
    };

    for (int ni : iters)
    {
        const double cycles = measure_case<4, 1>(
            h_init, d_init, d_state, d_sink,
            ni, clockRateKHz, start, stop
        );

        std::printf("%10d %14.1f\n", ni, cycles);
    }

    // ================================
    // Cleanup
    // ================================
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_init));
    CUDA_CHECK(cudaFree(d_state));
    CUDA_CHECK(cudaFree(d_sink));

    std::printf(
        "\nTask 2 FMA Latency Completed.\n"
    );

    return 0;
}
