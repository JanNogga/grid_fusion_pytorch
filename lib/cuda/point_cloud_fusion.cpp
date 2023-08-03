#include <torch/extension.h>
using namespace std;


torch::Tensor point_cloud_hit_counter_free_function(const torch::Tensor grid_counter, const torch::Tensor point_cloud_locations, const torch::Tensor min_range, const torch::Tensor max_range);
std::vector<torch::Tensor> point_cloud_hit_counter_bayes_free_function(const torch::Tensor grid_counter, const torch::Tensor grid_semantic, const torch::Tensor point_cloud_locations, const torch::Tensor point_cloud_logprobs, const torch::Tensor min_range, const torch::Tensor max_range);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("point_cloud_hit_counter_free_function", &point_cloud_hit_counter_free_function);
    m.def("point_cloud_hit_counter_bayes_free_function", &point_cloud_hit_counter_bayes_free_function);
}