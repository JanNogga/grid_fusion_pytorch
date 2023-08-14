#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

template <typename T>
T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

__global__ void point_cloud_hit_counter_free_function_gpu(
    const int num_points,
    const int numel,
    const float H,
    const float W,
    const float D,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> point_cloud_locations,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> min_range,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> max_range,
    //output
    torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> counter_out
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=numel){ //don't go out of bounds
        return;
    }

    int scene_idx = idx / num_points;
    int point_idx = idx % num_points;

    float cur_pos_x = point_cloud_locations[scene_idx][point_idx][0];

    // if the point cloud is padded with nan at this location, dont do anything
    if(cur_pos_x != cur_pos_x){
        return;
    }

    float cur_pos_y = point_cloud_locations[scene_idx][point_idx][1];
    float cur_pos_z = point_cloud_locations[scene_idx][point_idx][2];

    float rel_pos_x = 0;
    float rel_pos_y = 0;
    float rel_pos_z = 0;
    int h = 0;
    int w = 0;
    int d = 0;

    rel_pos_x = (cur_pos_x - min_range[scene_idx][0]) / (max_range[scene_idx][0] - min_range[scene_idx][0]);
    rel_pos_y = (cur_pos_y - min_range[scene_idx][1]) / (max_range[scene_idx][1] - min_range[scene_idx][1]);
    rel_pos_z = (cur_pos_z - min_range[scene_idx][2]) / (max_range[scene_idx][2] - min_range[scene_idx][2]);
    h = (int)(rel_pos_x * (H-1));
    w = (int)(rel_pos_y * (W-1));
    d = (int)(rel_pos_z * (D-1));
    
    if(h >= 0 && h < H && w >= 0 && w < W && d >= 0 && d < D){
        // // increment hit at point location
        atomicAdd(&counter_out[scene_idx][0][h][w][d], 1);
    }
    
    return;
}

torch::Tensor point_cloud_hit_counter_free_function(const torch::Tensor grid_counter, const torch::Tensor point_cloud_locations, const torch::Tensor min_range, const torch::Tensor max_range){
    CHECK(grid_counter.is_cuda()) << "grid_counter should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(point_cloud_locations.is_cuda()) << "point_cloud_locations should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(min_range.is_cuda()) << "min_range should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(max_range.is_cuda()) << "max_range should be in GPU memory! Please call .cuda() on the tensor.";
    torch::Tensor counter_out = grid_counter.detach().clone();
    CHECK(counter_out.is_cuda()) << "counter_out should be in GPU memory! Please call .cuda() on the tensor.";
    int num_scenes = point_cloud_locations.size(0);
    int num_points = point_cloud_locations.size(1);
    int numel = num_scenes*num_points;
    
    float H = (float) grid_counter.size(2);
    float W = (float) grid_counter.size(3);
    float D = (float) grid_counter.size(4);

    const dim3 blocks = {(unsigned int)div_round_up(numel, BLOCK_SIZE), 1, 1};

    point_cloud_hit_counter_free_function_gpu<<<blocks, BLOCK_SIZE>>>(
        num_points,
        numel,
        H,
        W,
        D,
        point_cloud_locations.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        min_range.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        max_range.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        counter_out.packed_accessor32<float,5,torch::RestrictPtrTraits>()
    );

    return counter_out;
}


__global__ void point_cloud_hit_counter_bayes_free_function_gpu(
    const int num_points,
    const int numel,
    const float H,
    const float W,
    const float D,
    const int C,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> point_cloud_locations,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> point_cloud_logprobs,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> min_range,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> max_range,
    //output
    torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> counter_out,
    torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> semantic_map_out
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=numel){ //don't go out of bounds
        return;
    }

    int scene_idx = idx / num_points;
    int point_idx = idx % num_points;

    float cur_pos_x = point_cloud_locations[scene_idx][point_idx][0];

    // if the point cloud is padded with nan at this location, dont do anything
    if(cur_pos_x != cur_pos_x){
        return;
    }

    float cur_pos_y = point_cloud_locations[scene_idx][point_idx][1];
    float cur_pos_z = point_cloud_locations[scene_idx][point_idx][2];

    float rel_pos_x = 0;
    float rel_pos_y = 0;
    float rel_pos_z = 0;
    int h = 0;
    int w = 0;
    int d = 0;

    rel_pos_x = (cur_pos_x - min_range[scene_idx][0]) / (max_range[scene_idx][0] - min_range[scene_idx][0]);
    rel_pos_y = (cur_pos_y - min_range[scene_idx][1]) / (max_range[scene_idx][1] - min_range[scene_idx][1]);
    rel_pos_z = (cur_pos_z - min_range[scene_idx][2]) / (max_range[scene_idx][2] - min_range[scene_idx][2]);
    h = (int)(rel_pos_x * (H-1));
    w = (int)(rel_pos_y * (W-1));
    d = (int)(rel_pos_z * (D-1));
    
    if(h >= 0 && h < H && w >= 0 && w < W && d >= 0 && d < D){
        // // increment hit at point location
        atomicAdd(&counter_out[scene_idx][0][h][w][d], 1);
        // finally apply filtering
        // we don't normalize here yet, but maybe we should
        for (int c = 0; c < C; c++) {
            atomicAdd(&semantic_map_out[scene_idx][c][h][w][d], point_cloud_logprobs[scene_idx][point_idx][c]);
        }
    }
    
    return;
}

std::vector<torch::Tensor> point_cloud_hit_counter_bayes_free_function(const torch::Tensor grid_counter, const torch::Tensor grid_semantic, const torch::Tensor point_cloud_locations, const torch::Tensor point_cloud_logprobs, const torch::Tensor min_range, const torch::Tensor max_range){
    CHECK(grid_counter.is_cuda()) << "grid_counter should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(grid_semantic.is_cuda()) << "grid_semantic should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(point_cloud_locations.is_cuda()) << "point_cloud_locations should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(point_cloud_logprobs.is_cuda()) << "point_cloud_logprobs should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(min_range.is_cuda()) << "min_range should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(max_range.is_cuda()) << "max_range should be in GPU memory! Please call .cuda() on the tensor.";
    
    torch::Tensor counter_out = grid_counter.detach().clone();
    torch::Tensor semantic_map_out = grid_semantic.detach().clone();
    CHECK(counter_out.is_cuda()) << "counter_out should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(semantic_map_out.is_cuda()) << "semantic_map_out should be in GPU memory! Please call .cuda() on the tensor.";
    int num_scenes = point_cloud_locations.size(0);
    int num_points = point_cloud_locations.size(1);
    int numel = num_scenes*num_points;

    float H = (float) grid_counter.size(2);
    float W = (float) grid_counter.size(3);
    float D = (float) grid_counter.size(4);
    float C = grid_semantic.size(1);

    const dim3 blocks = {(unsigned int)div_round_up(numel, BLOCK_SIZE), 1, 1};

    point_cloud_hit_counter_bayes_free_function_gpu<<<blocks, BLOCK_SIZE>>>(
        num_points,
        numel,
        H,
        W,
        D,
        C,
        point_cloud_locations.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        point_cloud_logprobs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        min_range.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        max_range.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        counter_out.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
        semantic_map_out.packed_accessor32<float,5,torch::RestrictPtrTraits>()
    );
    std::vector<torch::Tensor> out;
    out.push_back(counter_out);
    out.push_back(semantic_map_out);
    return out;
}