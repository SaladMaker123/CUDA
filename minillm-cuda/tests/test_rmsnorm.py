import argparse

import torch

from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from minillm.extension import get_extension, rmsnorm_v0, rmsnorm_v1, rmsnorm_v2


# Registered CUDA implementations
IMPLEMENTATIONS = {
    "v0": rmsnorm_v0,
    "v1": rmsnorm_v1,
    "v2": rmsnorm_v2,
}


# Test configurations
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

    # Boundary cases
    (2, 17),
    (3, 1000),
    (1, 1),

    # Different tensor dimensions
    (896,),
    (1, 37, 896),
    (2, 37, 896),

    # Empty tensor
    (0, 896),
]


DTYPES = [
    torch.float16,
    torch.float32,
]


# Qwen2 reference
def reference(X, W, eps):
    module = Qwen2RMSNorm(
        X.size(-1),
        eps=eps,
    ).to(
        device=X.device,
        dtype=X.dtype,
    )

    module.weight.copy_(W)

    return module(X)


# Compare CUDA output with reference
def check_result(Y, Y_ref):
    assert Y.shape == Y_ref.shape
    assert Y.dtype == Y_ref.dtype
    assert Y.device == Y_ref.device

    if Y.dtype == torch.float16:
        atol = 2e-3
        rtol = 2e-3
    else:
        atol = 1e-5
        rtol = 1e-5

    torch.testing.assert_close(
        Y,
        Y_ref,
        atol=atol,
        rtol=rtol,
    )

    if Y.numel() == 0:
        return 0.0

    return (
        Y.float() - Y_ref.float()
    ).abs().max().item()


# Generate test inputs
def make_inputs(shape, dtype):
    X = torch.randn(
        shape,
        device="cuda",
        dtype=dtype,
    )

    W = 0.5 + torch.rand(
        shape[-1],
        device="cuda",
        dtype=dtype,
    )

    return X, W


# Test different shapes
def test_shapes(op, dtype):
    max_error = 0.0

    for shape in SHAPES:
        X, W = make_inputs(shape, dtype)

        Y = op(X, W, 1e-6)

        Y_ref = reference(X, W, 1e-6)

        error = check_result(Y, Y_ref)

        max_error = max(max_error, error)

        print(
            f"Shape {str(shape):20s} "
            f"Max error: {error:.6e}  PASS"
        )

    return len(SHAPES), max_error


# Test numerical stability
def test_numerics(op, dtype):
    X, W = make_inputs((128, 896), dtype)

    cases = [
        ("Zero input", torch.zeros_like(X), 1e-6),
        ("Small input", X * 1e-4, 1e-6),
        ("Large input", X * 100.0, 1e-6),
        ("Different epsilon", X, 1e-5),
    ]

    max_error = 0.0

    for name, A, eps in cases:
        Y = op(A, W, eps)

        Y_ref = reference(A, W, eps)

        error = check_result(Y, Y_ref)

        max_error = max(max_error, error)

        print(
            f"{name:24s} "
            f"Max error: {error:.6e}  PASS"
        )

    return len(cases), max_error


# Test misaligned but contiguous tensors
def test_misaligned(op, dtype):
    M = 3
    N = 896

    X = torch.randn(
        M * N + 1,
        device="cuda",
        dtype=dtype,
    )[1:].view(M, N)

    W = torch.randn(
        N + 1,
        device="cuda",
        dtype=dtype,
    )[1:]

    assert X.is_contiguous()
    assert W.is_contiguous()

    Y = op(X, W, 1e-6)

    Y_ref = reference(X, W, 1e-6)

    error = check_result(Y, Y_ref)

    print(
        f"Misaligned tensors       "
        f"Max error: {error:.6e}  PASS"
    )

    return 1, error


# Test invalid inputs
def test_invalid_inputs(op):
    X, W = make_inputs(
        (37, 896),
        torch.float16,
    )

    cases = [
        (
            "CPU input",
            X.cpu(),
            W,
            1e-6,
            "X must be a CUDA tensor",
        ),
        (
            "CPU weight",
            X,
            W.cpu(),
            1e-6,
            "W must be a CUDA tensor",
        ),
        (
            "Unsupported dtype",
            X.double(),
            W.double(),
            1e-6,
            "X must be FP16 or FP32",
        ),
        (
            "Different dtype",
            X,
            W.float(),
            1e-6,
            "X and W must have the same dtype",
        ),
        (
            "Invalid weight dimension",
            X,
            W.unsqueeze(0),
            1e-6,
            "W must be a 1D tensor",
        ),
        (
            "Invalid weight size",
            X,
            W[:-1],
            1e-6,
            "Weight size must match hidden size",
        ),
        (
            "Non-contiguous input",
            X.transpose(0, 1),
            W,
            1e-6,
            "X must be contiguous",
        ),
        (
            "Non-contiguous weight",
            X,
            torch.randn(
                1792,
                device="cuda",
                dtype=X.dtype,
            )[::2],
            1e-6,
            "W must be contiguous",
        ),
        (
            "Invalid hidden size",
            X[:, :0],
            W[:0],
            1e-6,
            "Invalid hidden size",
        ),
        (
            "Negative epsilon",
            X,
            W,
            -1e-6,
            "Epsilon must be finite and positive",
        ),
        (
            "NaN epsilon",
            X,
            W,
            float("nan"),
            "Epsilon must be finite and positive",
        ),
    ]

    for name, A, B, eps, expected in cases:
        try:
            op(A, B, eps)

        except RuntimeError as error:
            assert expected in str(error), str(error)

            print(f"{name:24s} PASS")

        else:
            raise AssertionError(
                f"{name}: expected RuntimeError"
            )

    return len(cases)


# Test non-default CUDA stream
def test_stream(op, dtype):
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        X, W = make_inputs(
            (128, 896),
            dtype,
        )

        Y = op(X, W, 1e-6)

        result = Y * 2

        Y_ref = reference(X, W, 1e-6)

        ref_result = Y_ref * 2

    stream.synchronize()

    error = check_result(result, ref_result)

    print(
        f"Non-default CUDA stream  "
        f"Max error: {error:.6e}  PASS"
    )

    return 1, error


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        choices=IMPLEMENTATIONS.keys(),
        default="v0",
    )

    args = parser.parse_args()

    torch.manual_seed(42)

    get_extension()

    op = IMPLEMENTATIONS[args.version]

    total_tests = 0

    print("\n========== RMSNorm Correctness ==========")

    print("Version:", args.version)
    print("GPU:", torch.cuda.get_device_name())

    for dtype in DTYPES:
        print(f"\n========== {dtype} ==========")

        max_error = 0.0

        for test in [
            test_shapes,
            test_numerics,
            test_misaligned,
            test_stream,
        ]:
            count, error = test(op, dtype)

            total_tests += count

            max_error = max(max_error, error)

        print(f"\nMaximum absolute error: {max_error:.6e}")

    print("\n========== Invalid Inputs ==========")

    total_tests += test_invalid_inputs(op)

    print("\n========== Test Summary ==========")

    print(f"Version: {args.version}")

    print(f"Passed: {total_tests}/{total_tests}")

    print("All tests passed.")


if __name__ == "__main__":
    main()