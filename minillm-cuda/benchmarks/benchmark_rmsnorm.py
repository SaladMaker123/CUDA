import statistics

import torch
import flashinfer

from flashinfer.norm import rmsnorm as flashinfer_rmsnorm

from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from minillm.extension import get_extension, rmsnorm_v0, rmsnorm_v1


# Registered CUDA implementations
IMPLEMENTATIONS = {
    "v0": rmsnorm_v0,
    "v1": rmsnorm_v1,
}


# Benchmark configurations
WARMUP = 20
REPEAT = 100
EPS = 1e-6


SHAPES = [
    # Qwen2.5
    (1, 896),
    (37, 896),
    (128, 896),
    (512, 896),
    (2048, 896),
    (4096, 896),

    # Large hidden sizes
    (1, 4096),
    (128, 4096),
    (2048, 4096),

    (1, 8192),
    (128, 8192),
    (2048, 8192),
]


DTYPES = [
    torch.float16,
    torch.float32,
]


# CUDA Event benchmark
def benchmark(operation):

    # Warmup
    for _ in range(WARMUP):
        Y = operation()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []

    # Measurement
    for _ in range(REPEAT):
        start.record()

        Y = operation()

        end.record()
        end.synchronize()

        elapsed_us = start.elapsed_time(end) * 1000.0

        times.append(elapsed_us)

    median_us = statistics.median(times)
    mean_us = statistics.mean(times)

    return median_us, mean_us


# Benchmark one shape and dtype
def benchmark_shape(M, N, dtype):

    X = torch.randn(
        (M, N),
        device="cuda",
        dtype=dtype,
    )

    W = 0.5 + torch.rand(
        N,
        device="cuda",
        dtype=dtype,
    )

    # Original Qwen2 RMSNorm
    reference = Qwen2RMSNorm(
        N,
        eps=EPS,
    ).to(
        device=X.device,
        dtype=dtype,
    )

    reference.weight.copy_(W)
    reference.eval()

    operations = {
        "PyTorch": lambda: reference(X),
    }

    # Custom CUDA implementations
    for name, op in IMPLEMENTATIONS.items():
        operations[name] = (
            lambda op=op: op(X, W, EPS)
        )

    # FlashInfer FP16 baseline
    if dtype == torch.float16:
        operations["FlashInfer"] = (
            lambda: flashinfer_rmsnorm(X, W, eps=EPS)
        )

    # Initialize all implementations before timing
    for operation in operations.values():
        Y = operation()

    torch.cuda.synchronize()

    # Benchmark all implementations
    results = {}

    for name, operation in operations.items():
        results[name] = benchmark(operation)

    # Baseline latency
    ref_median = results["PyTorch"][0]
    v0_median = results["v0"][0]

    fi_median = (
        results["FlashInfer"][0]
        if "FlashInfer" in results
        else None
    )

    # Print results
    for name, (median_us, mean_us) in results.items():

        speedup_ref = ref_median / median_us
        speedup_v0 = v0_median / median_us

        if fi_median is not None:
            fi_ratio = (
                f"{fi_median / median_us * 100:.2f}%"
            )
        else:
            fi_ratio = "N/A"

        print(
            f"{M:6d} "
            f"{N:6d} "
            f"{name:12s} "
            f"{median_us:11.3f} "
            f"{mean_us:11.3f} "
            f"{speedup_ref:8.3f}x "
            f"{speedup_v0:8.3f}x "
            f"{fi_ratio:>10s}"
        )

    print("-" * 85)


@torch.inference_mode()
def main():

    torch.manual_seed(42)

    get_extension()

    print("\n========== RMSNorm Benchmark ==========")

    print("GPU:", torch.cuda.get_device_name())
    print("PyTorch:", torch.__version__)
    print("FlashInfer:", flashinfer.__version__)
    print("CUDA:", torch.version.cuda)
    print("Warmup:", WARMUP)
    print("Repeat:", REPEAT)
    print("Epsilon:", EPS)

    for dtype in DTYPES:

        print(f"\n========== {dtype} ==========")

        print(
            f"{'M':>6s} "
            f"{'N':>6s} "
            f"{'Version':12s} "
            f"{'Median(us)':>11s} "
            f"{'Mean(us)':>11s} "
            f"{'vs Torch':>9s} "
            f"{'vs V0':>9s} "
            f"{'FI Ratio':>10s}"
        )

        print("-" * 85)

        for M, N in SHAPES:
            benchmark_shape(M, N, dtype)

    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()