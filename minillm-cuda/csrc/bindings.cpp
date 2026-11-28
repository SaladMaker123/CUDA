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

torch::Tensor rmsnormV1(
    torch::Tensor X,
    torch::Tensor W,
    double eps
);

torch::Tensor rmsnormV2(
    torch::Tensor X,
    torch::Tensor W,
    double eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("vector_add", &vectorAdd, "CUDA Vector Add");

    m.def("rmsnorm_v0", &rmsnormV0, "CUDA RMSNorm V0");

    m.def("rmsnorm_v1", &rmsnormV1, "CUDA RMSNorm V1");

    m.def("rmsnorm_v2", &rmsnormV2, "CUDA RMSNorm V2");
}