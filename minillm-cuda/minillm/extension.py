from pathlib import Path
from functools import lru_cache

from torch.utils.cpp_extension import load


ROOT_DIR = Path(__file__).resolve().parents[2]

SOURCES = [
    ROOT_DIR / "minillm-cuda/csrc/bindings.cpp",
    ROOT_DIR / "operators/vector_add/vector_add.cu",
]


@lru_cache(maxsize=1)
def get_extension():

    return load(
        name="minillm_cuda",
        sources=[str(path) for path in SOURCES],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=True
    )


def vector_add(A, B):

    return get_extension().vector_add(A, B)


if __name__ == "__main__":

    module = get_extension()

    print("\nCUDA extension loaded successfully.")
    print("Module path:", module.__file__)
    print("Vector Add available:", hasattr(module, "vector_add"))