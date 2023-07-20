#include <torch/extension.h>

torch::Tensor increment_misses_free_function(const torch::Tensor Grids, const torch::Tensor Origs, const torch::Tensor Dirs, const torch::Tensor Dists, const torch::Tensor RangeMin, const torch::Tensor RangeMax, const int n_steps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("increment_misses_free_function", &increment_misses_free_function);
}
