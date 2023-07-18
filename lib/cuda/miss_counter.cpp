#include <torch/extension.h>

torch::Tensor increment_misses_free_function(const torch::Tensor Grids, const torch::Tensor Origs, const torch::Tensor Dirs, const torch::Tensor Dists);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("increment_misses_free_function", &increment_misses_free_function);
}
