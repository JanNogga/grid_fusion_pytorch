#include <torch/extension.h>
using namespace std;

torch::Tensor counting_model_free_function(const torch::Tensor grid_counter, const torch::Tensor ray_origins, const torch::Tensor ray_directions, const torch::Tensor ray_distances, const torch::Tensor min_range, const torch::Tensor max_range, const int n_steps);
std::vector<torch::Tensor> counting_model_bayes_free_function(const torch::Tensor grid_counter, const torch::Tensor grid_semantic, const torch::Tensor ray_origins, const torch::Tensor ray_directions, const torch::Tensor ray_distances, const torch::Tensor ray_semseg, const torch::Tensor min_range, const torch::Tensor max_range, const int n_steps);
torch::Tensor hit_counter_free_function(const torch::Tensor grid_counter, const torch::Tensor ray_origins, const torch::Tensor ray_directions, const torch::Tensor ray_distances, const torch::Tensor min_range, const torch::Tensor max_range);
std::vector<torch::Tensor> hit_counter_bayes_free_function(const torch::Tensor grid_counter, const torch::Tensor grid_semantic, const torch::Tensor ray_origins, const torch::Tensor ray_directions, const torch::Tensor ray_distances, const torch::Tensor ray_semseg, const torch::Tensor min_range, const torch::Tensor max_range);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("counting_model_free_function", &counting_model_free_function);
    m.def("counting_model_bayes_free_function", &counting_model_bayes_free_function);
    m.def("hit_counter_free_function", &hit_counter_free_function);
    m.def("hit_counter_bayes_free_function", &hit_counter_bayes_free_function);
}
