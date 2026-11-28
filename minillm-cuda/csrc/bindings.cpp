#include <torch/extension.h>

torch::Tensor vectorAdd(
    torch::Tensor A,
    torch::Tensor B
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("vector_add", &vectorAdd, "CUDA Vector Add");
}