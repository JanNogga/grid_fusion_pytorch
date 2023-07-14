#include <torch/extension.h>


torch::Tensor vector_add_free_function(const torch::Tensor A, const torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add_free_function", &vector_add_free_function);
}
