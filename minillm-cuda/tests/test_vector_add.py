import torch

from minillm.extension import get_extension, vector_add


def check_result(A, B):

    C = vector_add(A, B)
    reference = A + B

    assert C.shape == reference.shape
    assert C.dtype == reference.dtype
    assert C.device == reference.device

    torch.testing.assert_close(
        C,
        reference,
        rtol=0,
        atol=0
    )

    if C.numel() > 0:
        max_error = (C - reference).abs().max().item()
    else:
        max_error = 0.0

    return max_error


def test_shapes():

    shapes = [
        (0,),
        (1,),
        (255,),
        (256,),
        (257,),
        (896,),
        (4864,),
        (1048576,),
        (37, 896),
        (1, 128, 896),
        (1, 128, 4864),
    ]

    print("\n========== Shape Tests ==========")

    for shape in shapes:

        A = torch.randn(
            shape,
            device="cuda",
            dtype=torch.float32
        )

        B = torch.randn_like(A)

        max_error = check_result(A, B)

        print(
            f"Shape: {str(shape):<20} "
            f"Max error: {max_error:.6e}  PASS"
        )


def expect_error(name, A, B, message):

    try:
        vector_add(A, B)

    except RuntimeError as error:

        if message not in str(error):
            raise

        print(f"{name:<30} PASS")
        return

    raise AssertionError(
        f"{name}: expected an error"
    )


def test_invalid_inputs():

    print("\n========== Invalid Input Tests ==========")

    A = torch.randn(
        (4, 8),
        device="cuda",
        dtype=torch.float32
    )

    B = torch.randn_like(A)

    expect_error(
        "CPU tensor",
        A.cpu(),
        B,
        "A must be a CUDA tensor"
    )

    expect_error(
        "FP16 tensor",
        A.half(),
        B.half(),
        "A must be FP32"
    )

    expect_error(
        "Shape mismatch",
        A,
        B.reshape(8, 4),
        "A and B must have the same shape"
    )

    expect_error(
        "Non-contiguous tensor",
        A.t(),
        B.t(),
        "A must be contiguous"
    )


def test_non_default_stream():

    print("\n========== CUDA Stream Test ==========")

    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):

        A = torch.randn(
            (1048576,),
            device="cuda",
            dtype=torch.float32
        )

        B = torch.randn_like(A)

        C = vector_add(A, B)

        result = C * 2.0

        reference = (A + B) * 2.0

    stream.synchronize()

    torch.testing.assert_close(
        result,
        reference,
        rtol=0,
        atol=0
    )

    max_error = (
        result - reference
    ).abs().max().item()

    print(f"Max error: {max_error:.6e}")
    print("Non-default CUDA stream: PASS")


def main():

    torch.manual_seed(42)

    get_extension()

    with torch.inference_mode():

        test_shapes()

        test_invalid_inputs()

        test_non_default_stream()

    print("\n========== Test Summary ==========")
    print("All Vector Add tests passed.")
    print("==================================")


if __name__ == "__main__":
    main()