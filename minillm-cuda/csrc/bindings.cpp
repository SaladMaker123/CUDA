#include <torch/extension.h>

torch::Tensor vectorAdd(
    torch::Tensor A,
    torch::Tensor B
);

torch::Tensor rmsnormV0(
    torch::Tensor X,
    torch::Tensor W,
    double eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("vector_add", &vectorAdd, "CUDA Vector Add");

    m.def("rmsnorm_v0", &rmsnormV0, "CUDA RMSNorm V0");
}