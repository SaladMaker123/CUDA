#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CUDA Project Environment
# GEMM + MiniLLM-CUDA
# ============================================================

CUDA_PKG="cuda-toolkit-13-0"
CUDA_VERSION="13.0.3-1"

NCU_PKG="nsight-compute-2025.2.0"
NCU_VERSION="2025.2.0.11-1"

TORCH_VERSION="2.9.1+cu130"
TRANSFORMERS_VERSION="4.57.6"
ACCELERATE_VERSION="1.11.0"
NUMPY_VERSION="1.26.4"

CUDA_HOME="/usr/local/cuda-13.0"
NCU_DIR="/opt/nvidia/nsight-compute/2025.2.0"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINILLM_DIR="${ROOT_DIR}/minillm-cuda"
VENV_DIR="${MINILLM_DIR}/.venv"

INSTALL_NCU=0

# ============================================================
# 0. Common functions and arguments
# ============================================================

section()
{
    printf '\n============================================================\n'
    printf '%s\n' "$1"
    printf '============================================================\n'
}

fail()
{
    echo "ERROR: $*" >&2
    exit 1
}

for arg in "$@"; do
    case "${arg}" in
        --with-ncu)
            INSTALL_NCU=1
            ;;
        *)
            fail "Unknown argument: ${arg}"
            ;;
    esac
done

trap 'echo "ERROR: Setup failed at line ${LINENO}." >&2' ERR

if [[ "${EUID}" -ne 0 ]]; then
    fail "Run this script as root."
fi

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT

# Prevent accidental NVIDIA driver replacement.
safe_install()
{
    local plan

    plan="$(apt-get -s install "$@")"

    if grep -Eq \
        '^(Inst|Remv) (cuda-drivers|cuda-compat-|nvidia-driver|nvidia-dkms|nvidia-open|nvidia-kernel|nvidia-utils|nvidia-compute-utils|nvidia-firmware|libnvidia-|linux-modules-nvidia|xserver-xorg-video-nvidia)' \
        <<< "${plan}"; then

        echo "${plan}"
        fail "APT would modify NVIDIA driver packages."
    fi

    apt-get install -y "$@"
}

package_installed()
{
    [[ "$(dpkg-query -W -f='${Status}' "$1" 2>/dev/null || true)" \
        == "install ok installed" ]]
}

check_package_version()
{
    local pkg="$1"
    local expected="$2"
    local installed

    if package_installed "${pkg}"; then
        installed="$(dpkg-query -W -f='${Version}' "${pkg}")"

        if [[ "${installed}" != "${expected}" ]]; then
            fail "${pkg}: installed ${installed}, expected ${expected}."
        fi
    fi
}

package_available()
{
    apt-cache madison "$1" | awk -v version="$2" '
        $3 == version { found = 1 }
        END { exit !found }
    '
}

# ============================================================
# 1. Check system and NVIDIA driver
# ============================================================

section "[1/10] Checking system and NVIDIA driver"

source /etc/os-release

if [[ "${ID}" != "ubuntu" ||
      "${VERSION_ID}" != "22.04" && "${VERSION_ID}" != "24.04" ]]; then
    fail "Ubuntu 22.04 or 24.04 is required."
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
    fail "x86_64 architecture is required."
fi

command -v nvidia-smi >/dev/null 2>&1 ||
    fail "nvidia-smi is unavailable."

if [[ -r /proc/driver/nvidia/version ]]; then

    LOADED_DRIVER="$(awk 'NR == 1 {
        for (i = 1; i <= NF; ++i) {
            if ($i ~ /^[0-9]+\.[0-9]+\.[0-9]+$/) {
                print $i
                exit
            }
        }
    }' /proc/driver/nvidia/version)"

    INSTALLED_DRIVER="$(modinfo -F version nvidia 2>/dev/null || true)"

    echo "Loaded NVIDIA driver: ${LOADED_DRIVER}"
    echo "Installed NVIDIA driver: ${INSTALLED_DRIVER}"

    if [[ -n "${LOADED_DRIVER}" &&
          -n "${INSTALLED_DRIVER}" &&
          "${LOADED_DRIVER}" != "${INSTALLED_DRIVER}" ]]; then

        fail "NVIDIA driver mismatch. Reboot the instance first."
    fi
fi

nvidia-smi -L ||
    fail "NVIDIA GPU initialization failed."

GPU_NAME="$(nvidia-smi --query-gpu=name \
    --format=csv,noheader | sed -n '1p')"

DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version \
    --format=csv,noheader | sed -n '1p')"

echo "GPU: ${GPU_NAME}"
echo "Driver: ${DRIVER_VERSION}"

if [[ "${GPU_NAME}" != *"RTX 5050"* &&
      "${GPU_NAME}" != *"RTX 5060"* &&
      "${GPU_NAME}" != *"RTX 5070"* &&
      "${GPU_NAME}" != *"RTX 5080"* &&
      "${GPU_NAME}" != *"RTX 5090"* ]]; then

    fail "This environment requires GeForce RTX 50 series."
fi

if [[ "${DRIVER_VERSION%%.*}" -lt 580 ]]; then
    fail "CUDA 13.x requires NVIDIA driver 580 or newer."
fi

check_package_version "${CUDA_PKG}" "${CUDA_VERSION}"

if [[ "${INSTALL_NCU}" -eq 1 ]]; then
    check_package_version "${NCU_PKG}" "${NCU_VERSION}"
fi

# ============================================================
# 2. Install basic development tools
# ============================================================

section "[2/10] Installing development tools"

export DEBIAN_FRONTEND=noninteractive

apt-get update

safe_install \
    build-essential \
    make \
    cmake \
    ninja-build \
    git \
    wget \
    ca-certificates \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv

gcc --version
g++ --version
python3 --version
ninja --version

# ============================================================
# 3. Install CUDA Toolkit and optional Nsight Compute
# ============================================================

section "[3/10] Installing CUDA Toolkit"

NEED_REPO=0

if ! package_installed "${CUDA_PKG}"; then
    if ! package_available "${CUDA_PKG}" "${CUDA_VERSION}"; then
        NEED_REPO=1
    fi
fi

if [[ "${INSTALL_NCU}" -eq 1 ]]; then
    if ! package_installed "${NCU_PKG}"; then
        if ! package_available "${NCU_PKG}" "${NCU_VERSION}"; then
            NEED_REPO=1
        fi
    fi
fi

if [[ "${NEED_REPO}" -eq 1 ]]; then

    DISTRO="ubuntu${VERSION_ID//./}"
    KEYRING="${TEMP_DIR}/cuda-keyring.deb"

    wget -q -O "${KEYRING}" \
        "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb"

    dpkg -i "${KEYRING}"

    apt-get update
fi

if ! package_installed "${CUDA_PKG}"; then

    package_available "${CUDA_PKG}" "${CUDA_VERSION}" ||
        fail "Pinned CUDA Toolkit package is unavailable."

    safe_install "${CUDA_PKG}=${CUDA_VERSION}"

else
    echo "CUDA Toolkit is already installed."
fi

check_package_version "${CUDA_PKG}" "${CUDA_VERSION}"

if [[ "${INSTALL_NCU}" -eq 1 ]]; then

    if ! package_installed "${NCU_PKG}"; then

        package_available "${NCU_PKG}" "${NCU_VERSION}" ||
            fail "Pinned Nsight Compute package is unavailable."

        safe_install "${NCU_PKG}=${NCU_VERSION}"

    else
        echo "Nsight Compute is already installed."
    fi

    check_package_version "${NCU_PKG}" "${NCU_VERSION}"

else
    echo "Dedicated Nsight Compute installation skipped."
fi

# ============================================================
# 4. Configure CUDA environment
# ============================================================

section "[4/10] Configuring CUDA environment"

ENV_FILE="/etc/profile.d/cuda-gemm-env.sh"
BASHRC_LINE="source ${ENV_FILE}"

cat > "${ENV_FILE}" <<'ENV_EOF'
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="${CUDA_HOME}/bin:${PATH}"

if [[ -x /opt/nvidia/nsight-compute/2025.2.0/ncu ]]; then
    export PATH="/opt/nvidia/nsight-compute/2025.2.0:${PATH}"
fi
ENV_EOF

chmod 644 "${ENV_FILE}"

touch /root/.bashrc

if ! grep -Fxq "${BASHRC_LINE}" /root/.bashrc; then
    printf '\n%s\n' "${BASHRC_LINE}" >> /root/.bashrc
fi

source "${ENV_FILE}"
hash -r

NVCC_PATH="$(command -v nvcc || true)"

if [[ "${NVCC_PATH}" != "${CUDA_HOME}/bin/nvcc" ]]; then
    fail "nvcc resolved to an unexpected path."
fi

NVCC_VERSION="$(nvcc --version)"

echo "${NVCC_VERSION}"

if [[ "${NVCC_VERSION}" != *"release 13.0"* ]]; then
    fail "CUDA Toolkit version mismatch."
fi

echo "CUDA_HOME: ${CUDA_HOME}"
echo "nvcc path: ${NVCC_PATH}"

# ============================================================
# 5. Create Python virtual environment
# ============================================================

section "[5/10] Creating Python virtual environment"

mkdir -p "${MINILLM_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    python3 -m venv "${VENV_DIR}"
else
    echo "Python virtual environment already exists."
fi

PYTHON="${VENV_DIR}/bin/python"

"${PYTHON}" --version

"${PYTHON}" -m pip install \
    "pip==25.3" \
    "setuptools==80.9.0" \
    "wheel==0.45.1"

# ============================================================
# 6. Install PyTorch and Python dependencies
# ============================================================

section "[6/10] Installing PyTorch and Python dependencies"

"${PYTHON}" -m pip install \
    "torch==${TORCH_VERSION}" \
    --index-url https://download.pytorch.org/whl/cu130

"${PYTHON}" -m pip install \
    "transformers==${TRANSFORMERS_VERSION}" \
    "accelerate==${ACCELERATE_VERSION}" \
    "numpy==${NUMPY_VERSION}"

"${PYTHON}" -m pip check

# ============================================================
# 7. Test native CUDA compilation and execution
# ============================================================

section "[7/10] Testing native CUDA"

cat > "${TEMP_DIR}/cuda_test.cu" <<'CUDA_EOF'
#include <cuda_runtime.h>
#include <cstdio>

__global__ void test(float* data)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = tid * 2.0f;
}

int main()
{
    float* data = nullptr;
    float result[1024];

    if (cudaMalloc(&data, 1024 * sizeof(float)) != cudaSuccess)
        return 1;

    test<<<4, 256>>>(data);

    if (cudaGetLastError() != cudaSuccess)
        return 2;

    if (cudaDeviceSynchronize() != cudaSuccess)
        return 3;

    if (cudaMemcpy(
        result,
        data,
        1024 * sizeof(float),
        cudaMemcpyDeviceToHost) != cudaSuccess)
        return 4;

    if (cudaFree(data) != cudaSuccess)
        return 5;

    for (int i = 0; i < 1024; ++i) {
        if (result[i] != i * 2.0f)
            return 6;
    }

    printf("PASS: Native CUDA computation.\n");

    return 0;
}
CUDA_EOF

nvcc -std=c++17 -arch=sm_120 \
    "${TEMP_DIR}/cuda_test.cu" \
    -o "${TEMP_DIR}/cuda_test"

"${TEMP_DIR}/cuda_test"

# ============================================================
# 8. Test PyTorch CUDA computation
# ============================================================

section "[8/10] Testing PyTorch CUDA"

"${PYTHON}" - <<'PYTHON_EOF'
import torch
import transformers
import accelerate
import numpy

from transformers import AutoModelForCausalLM, AutoTokenizer

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("Transformers:", transformers.__version__)
print("Accelerate:", accelerate.__version__)
print("NumPy:", numpy.__version__)

assert torch.__version__ == "2.9.1+cu130"
assert torch.version.cuda == "13.0"
assert transformers.__version__ == "4.57.6"
assert accelerate.__version__ == "1.11.0"
assert numpy.__version__ == "1.26.4"

assert torch.cuda.is_available(), "CUDA is unavailable."

print("GPU:", torch.cuda.get_device_name(0))
print("Compute Capability:", torch.cuda.get_device_capability(0))
print("Supported architectures:", torch.cuda.get_arch_list())

assert torch.cuda.get_device_capability(0) == (12, 0)
assert "sm_120" in torch.cuda.get_arch_list()

a = torch.ones((128, 128), device="cuda", dtype=torch.float16)
b = torch.ones((128, 128), device="cuda", dtype=torch.float16)

c = a @ b

torch.cuda.synchronize()

assert torch.all(c == 128).item()

print("PASS: PyTorch CUDA FP16 matrix multiplication.")
PYTHON_EOF

# ============================================================
# 9. Test PyTorch CUDA Extension
# ============================================================

section "[9/10] Testing PyTorch CUDA Extension"

cat > "${TEMP_DIR}/bindings.cpp" <<'CPP_EOF'
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

cudaError_t launch_test(
    const float* input,
    float* output,
    int n,
    cudaStream_t stream
);

torch::Tensor test(torch::Tensor input)
{
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor.");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous.");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "Input must be float32.");
    TORCH_CHECK(input.numel() > 0, "Input must not be empty.");

    auto output = torch::empty_like(input);

    auto stream = c10::cuda::getCurrentCUDAStream();

    cudaError_t status = launch_test(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(input.numel()),
        stream.stream()
    );

    TORCH_CHECK(status == cudaSuccess,
                cudaGetErrorString(status));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("test", &test);
}
CPP_EOF

cat > "${TEMP_DIR}/kernel.cu" <<'CUDA_EOF'
#include <cuda_runtime.h>

__global__ void test_kernel(
    const float* input,
    float* output,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
        output[tid] = input[tid] * 2.0f;
}

cudaError_t launch_test(
    const float* input,
    float* output,
    int n,
    cudaStream_t stream)
{
    constexpr int THREADS = 256;
    int blocks = (n + THREADS - 1) / THREADS;

    test_kernel<<<blocks, THREADS, 0, stream>>>(
        input, output, n
    );

    return cudaGetLastError();
}
CUDA_EOF

export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=2
export SETUP_TEMP_DIR="${TEMP_DIR}"

"${PYTHON}" - <<'PYTHON_EOF'
import os
import torch

from torch.utils.cpp_extension import load

temp_dir = os.environ["SETUP_TEMP_DIR"]

extension = load(
    name="cuda_environment_test",
    sources=[
        os.path.join(temp_dir, "bindings.cpp"),
        os.path.join(temp_dir, "kernel.cu"),
    ],
    build_directory=temp_dir,
    extra_cflags=["-O2", "-std=c++17"],
    extra_cuda_cflags=["-O2", "-std=c++17"],
    verbose=True,
)

stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    x = torch.arange(1024, device="cuda", dtype=torch.float32)
    y = extension.test(x)
    expected = x * 2

stream.synchronize()

assert torch.equal(y, expected)

print("PASS: PyTorch CUDA Extension.")
PYTHON_EOF

# ============================================================
# 10. Optional Nsight Compute profiling
# ============================================================

section "[10/10] Optional performance profiling"

if [[ "${INSTALL_NCU}" -eq 1 ]]; then

    if [[ ! -x "${NCU_DIR}/ncu" ]]; then
        fail "Pinned Nsight Compute executable was not found."
    fi

    "${NCU_DIR}/ncu" --version

    echo "Testing Nsight Compute performance-counter permission..."

    if "${NCU_DIR}/ncu" \
        --set basic \
        --launch-count 1 \
        "${TEMP_DIR}/cuda_test" \
        > "${TEMP_DIR}/ncu.log" 2>&1; then

        if grep -q 'Section:' "${TEMP_DIR}/ncu.log"; then

            echo "PASS: Nsight Compute profiling."

        else

            cat "${TEMP_DIR}/ncu.log"
            echo "WARNING: No performance metrics were collected."

        fi

    else

        cat "${TEMP_DIR}/ncu.log"

        if grep -q 'ERR_NVGPUCTRPERM' "${TEMP_DIR}/ncu.log"; then
            echo "WARNING: GPU performance-counter permission denied."
        else
            echo "WARNING: Nsight Compute profiling failed."
        fi

        echo "CUDA development can continue."
    fi

else
    echo "Nsight Compute profiling skipped."
fi

# Check whether the existing driver was preserved.
CURRENT_DRIVER="$(nvidia-smi --query-gpu=driver_version \
    --format=csv,noheader | sed -n '1p')"

if [[ "${CURRENT_DRIVER}" != "${DRIVER_VERSION}" ]]; then
    fail "NVIDIA driver version changed unexpectedly."
fi

# ============================================================
# Environment summary
# ============================================================

section "Environment setup completed"

echo "GPU:             ${GPU_NAME}"
echo "NVIDIA Driver:   ${CURRENT_DRIVER}"
echo "CUDA Toolkit:    ${CUDA_VERSION}"
echo "PyTorch:         ${TORCH_VERSION}"
echo "Transformers:    ${TRANSFORMERS_VERSION}"
echo "Accelerate:      ${ACCELERATE_VERSION}"
echo "NumPy:           ${NUMPY_VERSION}"
echo "CUDA_HOME:       ${CUDA_HOME}"
echo "Project root:    ${ROOT_DIR}"
echo "Python venv:     ${VENV_DIR}"

echo
echo "PASS: Native CUDA computation"
echo "PASS: PyTorch CUDA computation"
echo "PASS: PyTorch CUDA Extension"
echo
echo "Activate the Python environment with:"
echo "source \"${VENV_DIR}/bin/activate\""
echo
echo "Environment ready."