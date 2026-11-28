import argparse

import torch

from flashinfer.norm import rmsnorm as flashinfer_rmsnorm

from minillm.extension import get_extension, rmsnorm_v0


IMPLEMENTATIONS = {
    "v0": rmsnorm_v0,
    "flashinfer": flashinfer_rmsnorm,
}


@torch.inference_mode()
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        choices=IMPLEMENTATIONS.keys(),
        default="v0",
    )

    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--n", type=int, default=896)

    args = parser.parse_args()

    torch.manual_seed(42)

    get_extension()

    X = torch.randn(
        (args.m, args.n),
        device="cuda",
        dtype=torch.float16,
    )

    W = 0.5 + torch.rand(
        args.n,
        device="cuda",
        dtype=torch.float16,
    )

    EPS = 1e-6

    op = IMPLEMENTATIONS[args.version]

    # Warmup
    for _ in range(20):
        Y = op(X, W, EPS)

    torch.cuda.synchronize()

    print("Version:", args.version)
    print("Shape:", X.shape)
    print("Dtype:", X.dtype)

    # Start profiling
    torch.cuda.profiler.start()

    Y = op(X, W, EPS)

    torch.cuda.synchronize()

    # Stop profiling
    torch.cuda.profiler.stop()

    print("Profiling completed.")


if __name__ == "__main__":
    main()