#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

template <typename T>
T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

__global__ void counting_model_free_function_gpu(
    const int num_scenes,
    const int num_cameras,
    const int num_rays,
    const int numel,
    const float H,
    const float W,
    const float D,
    const int n_steps,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> ray_origins, // 3 is dimension of tensor
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> ray_directions,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> ray_distances,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> min_range,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> max_range,
    //output
    torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> counter_out
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=numel){ //don't go out of bounds
        return;
    }

    int scene_idx = idx / (num_cameras*num_rays);
    int camera_idx = (idx % (num_cameras*num_rays)) / num_rays;
    int ray_idx = idx % num_rays;

    float dist = ray_distances[scene_idx][camera_idx][ray_idx];
    // if the distance is invalid, don't do anything
    if(dist != dist){
        return;
    }

    float step_size = dist / (float) n_steps;
    float rel_pos_x = 0;
    float rel_pos_y = 0;
    float rel_pos_z = 0;
    float cur_pos_x = 0;
    float cur_pos_y = 0;
    float cur_pos_z = 0;

    cur_pos_x += ray_origins[scene_idx][camera_idx][0];
    cur_pos_y += ray_origins[scene_idx][camera_idx][1];
    cur_pos_z += ray_origins[scene_idx][camera_idx][2];

    int h = 0;
    int w = 0;
    int d = 0;
    int h_prev = 0;
    int w_prev = 0;
    int d_prev = 0;
    // all steps until the last one count as a miss
    for (int i = 0; i < n_steps-1; i++)
    {
        cur_pos_x += ray_directions[scene_idx][camera_idx][ray_idx][0] * step_size;
        cur_pos_y += ray_directions[scene_idx][camera_idx][ray_idx][1] * step_size;
        cur_pos_z += ray_directions[scene_idx][camera_idx][ray_idx][2] * step_size;

        rel_pos_x = (cur_pos_x - min_range[scene_idx][0]) / (max_range[scene_idx][0] - min_range[scene_idx][0]);
        rel_pos_y = (cur_pos_y - min_range[scene_idx][1]) / (max_range[scene_idx][1] - min_range[scene_idx][1]);
        rel_pos_z = (cur_pos_z - min_range[scene_idx][2]) / (max_range[scene_idx][2] - min_range[scene_idx][2]);
        h = (int)(rel_pos_x * (H-1));
        w = (int)(rel_pos_y * (W-1));
        d = (int)(rel_pos_z * (D-1));
        if(h < 0 || h >= H || w < 0 || w >= W || d < 0 || d >= D){
            h_prev = h;
            w_prev = w;
            d_prev = d;
            continue;
        }
        if(h != h_prev || w != w_prev || d != d_prev){
            atomicAdd(&counter_out[scene_idx][1][h][w][d], 1);
            h_prev = h;
            w_prev = w;
            d_prev = d;
        }

    }
    // final step, doesnt count as a miss
    cur_pos_x = ray_origins[scene_idx][camera_idx][0] + ray_directions[scene_idx][camera_idx][ray_idx][0] * dist;
    cur_pos_y = ray_origins[scene_idx][camera_idx][1] + ray_directions[scene_idx][camera_idx][ray_idx][1] * dist;
    cur_pos_z = ray_origins[scene_idx][camera_idx][2] + ray_directions[scene_idx][camera_idx][ray_idx][2] * dist;

    rel_pos_x = (cur_pos_x - min_range[scene_idx][0]) / (max_range[scene_idx][0] - min_range[scene_idx][0]);
    rel_pos_y = (cur_pos_y - min_range[scene_idx][1]) / (max_range[scene_idx][1] - min_range[scene_idx][1]);
    rel_pos_z = (cur_pos_z - min_range[scene_idx][2]) / (max_range[scene_idx][2] - min_range[scene_idx][2]);
    h = (int)(rel_pos_x * (H-1));
    w = (int)(rel_pos_y * (W-1));
    d = (int)(rel_pos_z * (D-1));
    
    if(h >= 0 && h < H && w >= 0 && w < W && d >= 0 && d < D){
        // if we changed this voxels miss counter, undo it
        if(h == h_prev && w == w_prev && d == d_prev){
           atomicAdd(&counter_out[scene_idx][1][h][w][d], -1);
        }
        // now increment the hit counter
        atomicAdd(&counter_out[scene_idx][0][h][w][d], 1);
    }
    
    return;
}


using torch::Tensor;

torch::Tensor counting_model_free_function(const torch::Tensor grid_counter, const torch::Tensor ray_origins, const torch::Tensor ray_directions, const torch::Tensor ray_distances, const torch::Tensor min_range, const torch::Tensor max_range, const int n_steps){
    CHECK(grid_counter.is_cuda()) << "grid_counter should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(ray_origins.is_cuda()) << "ray_origins should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(ray_directions.is_cuda()) << "ray_directions should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(ray_distances.is_cuda()) << "ray_distances should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(min_range.is_cuda()) << "min_range should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(max_range.is_cuda()) << "max_range should be in GPU memory! Please call .cuda() on the tensor.";
    
    torch::Tensor counter_out = grid_counter.detach().clone();
    CHECK(counter_out.is_cuda()) << "counter_out should be in GPU memory! Please call .cuda() on the tensor.";
    int num_scenes = ray_distances.size(0);
    int num_cameras = ray_distances.size(1);
    int num_rays = ray_distances.size(2);
    int numel = num_scenes*num_cameras*num_rays;

    float H = (float) grid_counter.size(2);
    float W = (float) grid_counter.size(3);
    float D = (float) grid_counter.size(4);

    const dim3 blocks = {(unsigned int)div_round_up(numel, BLOCK_SIZE), 1, 1};

    counting_model_free_function_gpu<<<blocks, BLOCK_SIZE>>>(
        num_scenes,
        num_cameras,
        num_rays,
        numel,
        H,
        W,
        D,
        n_steps,
        ray_origins.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        ray_directions.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        ray_distances.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        min_range.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        max_range.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        counter_out.packed_accessor32<float,5,torch::RestrictPtrTraits>()
    );
    return counter_out;
}

__global__ void counting_model_bayes_free_function_gpu(
    const int num_scenes,
    const int num_cameras,
    const int num_rays,
    const int numel,
    const float H,
    const float W,
    const float D,
    const int C,
    const int n_steps,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> ray_origins,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> ray_directions,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> ray_distances,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> ray_semseg,
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

    int scene_idx = idx / (num_cameras*num_rays);
    int camera_idx = (idx % (num_cameras*num_rays)) / num_rays;
    int ray_idx = idx % num_rays;

    float dist = ray_distances[scene_idx][camera_idx][ray_idx];
    // if the distance is invalid, don't do anything
    if(dist != dist){
        return;
    }

    float step_size = dist / (float) n_steps;
    float rel_pos_x = 0;
    float rel_pos_y = 0;
    float rel_pos_z = 0;
    float cur_pos_x = 0;
    float cur_pos_y = 0;
    float cur_pos_z = 0;

    cur_pos_x += ray_origins[scene_idx][camera_idx][0];
    cur_pos_y += ray_origins[scene_idx][camera_idx][1];
    cur_pos_z += ray_origins[scene_idx][camera_idx][2];

    int h = 0;
    int w = 0;
    int d = 0;
    int h_prev = 0;
    int w_prev = 0;
    int d_prev = 0;
    // all steps until the last one count as a miss
    for (int i = 0; i < n_steps-1; i++)
    {
        cur_pos_x += ray_directions[scene_idx][camera_idx][ray_idx][0] * step_size;
        cur_pos_y += ray_directions[scene_idx][camera_idx][ray_idx][1] * step_size;
        cur_pos_z += ray_directions[scene_idx][camera_idx][ray_idx][2] * step_size;

        rel_pos_x = (cur_pos_x - min_range[scene_idx][0]) / (max_range[scene_idx][0] - min_range[scene_idx][0]);
        rel_pos_y = (cur_pos_y - min_range[scene_idx][1]) / (max_range[scene_idx][1] - min_range[scene_idx][1]);
        rel_pos_z = (cur_pos_z - min_range[scene_idx][2]) / (max_range[scene_idx][2] - min_range[scene_idx][2]);
        h = (int)(rel_pos_x * (H-1));
        w = (int)(rel_pos_y * (W-1));
        d = (int)(rel_pos_z * (D-1));
        if(h < 0 || h >= H || w < 0 || w >= W || d < 0 || d >= D){
            h_prev = h;
            w_prev = w;
            d_prev = d;
            continue;
        }
        if(h != h_prev || w != w_prev || d != d_prev){
            atomicAdd(&counter_out[scene_idx][1][h][w][d], 1);
            h_prev = h;
            w_prev = w;
            d_prev = d;
        }

    }
    // final step, doesnt count as a miss
    cur_pos_x = ray_origins[scene_idx][camera_idx][0] + ray_directions[scene_idx][camera_idx][ray_idx][0] * dist;
    cur_pos_y = ray_origins[scene_idx][camera_idx][1] + ray_directions[scene_idx][camera_idx][ray_idx][1] * dist;
    cur_pos_z = ray_origins[scene_idx][camera_idx][2] + ray_directions[scene_idx][camera_idx][ray_idx][2] * dist;

    rel_pos_x = (cur_pos_x - min_range[scene_idx][0]) / (max_range[scene_idx][0] - min_range[scene_idx][0]);
    rel_pos_y = (cur_pos_y - min_range[scene_idx][1]) / (max_range[scene_idx][1] - min_range[scene_idx][1]);
    rel_pos_z = (cur_pos_z - min_range[scene_idx][2]) / (max_range[scene_idx][2] - min_range[scene_idx][2]);
    h = (int)(rel_pos_x * (H-1));
    w = (int)(rel_pos_y * (W-1));
    d = (int)(rel_pos_z * (D-1));
    
    if(h >= 0 && h < H && w >= 0 && w < W && d >= 0 && d < D){
        // make sure the voxel class is not invalid
        for (int c = 0; c < C; c++) {
            if(ray_semseg[scene_idx][camera_idx][ray_idx][c] != ray_semseg[scene_idx][camera_idx][ray_idx][c]){
                return;
            }
        }
        // if we changed this voxels miss counter, undo it
        if(h == h_prev && w == w_prev && d == d_prev){
           atomicAdd(&counter_out[scene_idx][1][h][w][d], -1);
        }
        // now increment the hit counter
        atomicAdd(&counter_out[scene_idx][0][h][w][d], 1);
        // finally apply filtering
        // we don't normalize here yet, but maybe we should
        for (int c = 0; c < C; c++) {
            atomicAdd(&semantic_map_out[scene_idx][c][h][w][d], ray_semseg[scene_idx][camera_idx][ray_idx][c]);
        }
    }
    
    return;
}

std::vector<torch::Tensor> counting_model_bayes_free_function(const torch::Tensor grid_counter, const torch::Tensor grid_semantic, const torch::Tensor ray_origins, const torch::Tensor ray_directions, const torch::Tensor ray_distances, const torch::Tensor ray_semseg, const torch::Tensor min_range, const torch::Tensor max_range, const int n_steps){
    CHECK(grid_counter.is_cuda()) << "grid_counter should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(grid_semantic.is_cuda()) << "grid_semantic should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(ray_origins.is_cuda()) << "ray_origins should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(ray_directions.is_cuda()) << "ray_directions should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(ray_distances.is_cuda()) << "ray_distances should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(ray_semseg.is_cuda()) << "ray_semseg should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(min_range.is_cuda()) << "min_range should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(max_range.is_cuda()) << "max_range should be in GPU memory! Please call .cuda() on the tensor.";
    
    torch::Tensor counter_out = grid_counter.detach().clone();
    torch::Tensor semantic_map_out = grid_semantic.detach().clone();
    CHECK(counter_out.is_cuda()) << "counter_out should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(semantic_map_out.is_cuda()) << "semantic_map_out should be in GPU memory! Please call .cuda() on the tensor.";
    int num_scenes = ray_distances.size(0);
    int num_cameras = ray_distances.size(1);
    int num_rays = ray_distances.size(2);
    int numel = num_scenes*num_cameras*num_rays;

    float H = (float) grid_counter.size(2);
    float W = (float) grid_counter.size(3);
    float D = (float) grid_counter.size(4);
    float C = grid_semantic.size(1);

    const dim3 blocks = {(unsigned int)div_round_up(numel, BLOCK_SIZE), 1, 1};

    counting_model_bayes_free_function_gpu<<<blocks, BLOCK_SIZE>>>(
        num_scenes,
        num_cameras,
        num_rays,
        numel,
        H,
        W,
        D,
        C,
        n_steps,
        ray_origins.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        ray_directions.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        ray_distances.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        ray_semseg.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
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

__global__ void hit_counter_free_function_gpu(
    const int num_scenes,
    const int num_cameras,
    const int num_rays,
    const int numel,
    const float H,
    const float W,
    const float D,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> ray_origins, // 3 is dimension of tensor
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> ray_directions,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> ray_distances,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> min_range,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> max_range,
    //output
    torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> counter_out
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=numel){ //don't go out of bounds
        return;
    }

    int scene_idx = idx / (num_cameras*num_rays);
    int camera_idx = (idx % (num_cameras*num_rays)) / num_rays;
    int ray_idx = idx % num_rays;

    float dist = ray_distances[scene_idx][camera_idx][ray_idx];
    // if the distance is invalid, don't do anything
    if(dist != dist){
        return;
    }

    float rel_pos_x = 0;
    float rel_pos_y = 0;
    float rel_pos_z = 0;
    float cur_pos_x = 0;
    float cur_pos_y = 0;
    float cur_pos_z = 0;

    int h = 0;
    int w = 0;
    int d = 0;

    // increment hit at final step
    cur_pos_x = ray_origins[scene_idx][camera_idx][0] + ray_directions[scene_idx][camera_idx][ray_idx][0] * dist;
    cur_pos_y = ray_origins[scene_idx][camera_idx][1] + ray_directions[scene_idx][camera_idx][ray_idx][1] * dist;
    cur_pos_z = ray_origins[scene_idx][camera_idx][2] + ray_directions[scene_idx][camera_idx][ray_idx][2] * dist;

    rel_pos_x = (cur_pos_x - min_range[scene_idx][0]) / (max_range[scene_idx][0] - min_range[scene_idx][0]);
    rel_pos_y = (cur_pos_y - min_range[scene_idx][1]) / (max_range[scene_idx][1] - min_range[scene_idx][1]);
    rel_pos_z = (cur_pos_z - min_range[scene_idx][2]) / (max_range[scene_idx][2] - min_range[scene_idx][2]);
    h = (int)(rel_pos_x * (H-1));
    w = (int)(rel_pos_y * (W-1));
    d = (int)(rel_pos_z * (D-1));
    
    if(h >= 0 && h < H && w >= 0 && w < W && d >= 0 && d < D){
        // now increment the hit counter
        atomicAdd(&counter_out[scene_idx][0][h][w][d], 1);
    }
    
    return;
}


using torch::Tensor;

torch::Tensor hit_counter_free_function(const torch::Tensor grid_counter, const torch::Tensor ray_origins, const torch::Tensor ray_directions, const torch::Tensor ray_distances, const torch::Tensor min_range, const torch::Tensor max_range){
    CHECK(grid_counter.is_cuda()) << "grid_counter should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(ray_origins.is_cuda()) << "ray_origins should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(ray_directions.is_cuda()) << "ray_directions should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(ray_distances.is_cuda()) << "ray_distances should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(min_range.is_cuda()) << "min_range should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(max_range.is_cuda()) << "max_range should be in GPU memory! Please call .cuda() on the tensor.";
    
    torch::Tensor counter_out = grid_counter.detach().clone();
    CHECK(counter_out.is_cuda()) << "counter_out should be in GPU memory! Please call .cuda() on the tensor.";
    int num_scenes = ray_distances.size(0);
    int num_cameras = ray_distances.size(1);
    int num_rays = ray_distances.size(2);
    int numel = num_scenes*num_cameras*num_rays;

    float H = (float) grid_counter.size(2);
    float W = (float) grid_counter.size(3);
    float D = (float) grid_counter.size(4);

    const dim3 blocks = {(unsigned int)div_round_up(numel, BLOCK_SIZE), 1, 1};

    hit_counter_free_function_gpu<<<blocks, BLOCK_SIZE>>>(
        num_scenes,
        num_cameras,
        num_rays,
        numel,
        H,
        W,
        D,
        ray_origins.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        ray_directions.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        ray_distances.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        min_range.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        max_range.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        counter_out.packed_accessor32<float,5,torch::RestrictPtrTraits>()
    );
    return counter_out;
}

__global__ void hit_counter_bayes_free_function_gpu(
    const int num_scenes,
    const int num_cameras,
    const int num_rays,
    const int numel,
    const float H,
    const float W,
    const float D,
    const int C,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> ray_origins,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> ray_directions,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> ray_distances,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> ray_semseg,
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

    int scene_idx = idx / (num_cameras*num_rays);
    int camera_idx = (idx % (num_cameras*num_rays)) / num_rays;
    int ray_idx = idx % num_rays;

    float dist = ray_distances[scene_idx][camera_idx][ray_idx];
    // if the distance is invalid, don't do anything
    if(dist != dist){
        return;
    }

    float rel_pos_x = 0;
    float rel_pos_y = 0;
    float rel_pos_z = 0;
    float cur_pos_x = 0;
    float cur_pos_y = 0;
    float cur_pos_z = 0;

    int h = 0;
    int w = 0;
    int d = 0;

    cur_pos_x = ray_origins[scene_idx][camera_idx][0] + ray_directions[scene_idx][camera_idx][ray_idx][0] * dist;
    cur_pos_y = ray_origins[scene_idx][camera_idx][1] + ray_directions[scene_idx][camera_idx][ray_idx][1] * dist;
    cur_pos_z = ray_origins[scene_idx][camera_idx][2] + ray_directions[scene_idx][camera_idx][ray_idx][2] * dist;

    rel_pos_x = (cur_pos_x - min_range[scene_idx][0]) / (max_range[scene_idx][0] - min_range[scene_idx][0]);
    rel_pos_y = (cur_pos_y - min_range[scene_idx][1]) / (max_range[scene_idx][1] - min_range[scene_idx][1]);
    rel_pos_z = (cur_pos_z - min_range[scene_idx][2]) / (max_range[scene_idx][2] - min_range[scene_idx][2]);
    h = (int)(rel_pos_x * (H-1));
    w = (int)(rel_pos_y * (W-1));
    d = (int)(rel_pos_z * (D-1));
    
    if(h >= 0 && h < H && w >= 0 && w < W && d >= 0 && d < D){
        // make sure the voxel class is not invalid
        for (int c = 0; c < C; c++) {
            if(ray_semseg[scene_idx][camera_idx][ray_idx][c] != ray_semseg[scene_idx][camera_idx][ray_idx][c]){
                return;
            }
        }
        // // increment hit at final step
        atomicAdd(&counter_out[scene_idx][0][h][w][d], 1);
        // finally apply filtering
        // we don't normalize here yet, but maybe we should
        for (int c = 0; c < C; c++) {
            atomicAdd(&semantic_map_out[scene_idx][c][h][w][d], ray_semseg[scene_idx][camera_idx][ray_idx][c]);
        }
    }
    
    return;
}

std::vector<torch::Tensor> hit_counter_bayes_free_function(const torch::Tensor grid_counter, const torch::Tensor grid_semantic, const torch::Tensor ray_origins, const torch::Tensor ray_directions, const torch::Tensor ray_distances, const torch::Tensor ray_semseg, const torch::Tensor min_range, const torch::Tensor max_range){
    CHECK(grid_counter.is_cuda()) << "grid_counter should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(grid_semantic.is_cuda()) << "grid_semantic should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(ray_origins.is_cuda()) << "ray_origins should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(ray_directions.is_cuda()) << "ray_directions should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(ray_distances.is_cuda()) << "ray_distances should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(ray_semseg.is_cuda()) << "ray_semseg should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(min_range.is_cuda()) << "min_range should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(max_range.is_cuda()) << "max_range should be in GPU memory! Please call .cuda() on the tensor.";
    
    torch::Tensor counter_out = grid_counter.detach().clone();
    torch::Tensor semantic_map_out = grid_semantic.detach().clone();
    CHECK(counter_out.is_cuda()) << "counter_out should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(semantic_map_out.is_cuda()) << "semantic_map_out should be in GPU memory! Please call .cuda() on the tensor.";
    int num_scenes = ray_distances.size(0);
    int num_cameras = ray_distances.size(1);
    int num_rays = ray_distances.size(2);
    int numel = num_scenes*num_cameras*num_rays;

    float H = (float) grid_counter.size(2);
    float W = (float) grid_counter.size(3);
    float D = (float) grid_counter.size(4);
    float C = grid_semantic.size(1);

    const dim3 blocks = {(unsigned int)div_round_up(numel, BLOCK_SIZE), 1, 1};

    hit_counter_bayes_free_function_gpu<<<blocks, BLOCK_SIZE>>>(
        num_scenes,
        num_cameras,
        num_rays,
        numel,
        H,
        W,
        D,
        C,
        ray_origins.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        ray_directions.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        ray_distances.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        ray_semseg.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
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