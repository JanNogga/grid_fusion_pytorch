#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void 
counting_model_free_function_gpu(
    const int num_scenes,
    const int num_cameras,
    const int num_rays,
    const int numel,
    const float H,
    const float W,
    const float D,
    const int n_steps,
    const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> Grids, // 5 is dimension of tensor
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Origs,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Dirs,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Dists,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> RangeMin,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> RangeMax,
    //output
    torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> out
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=numel){ //don't go out of bounds
        return;
    }

    int scene_idx = idx / (num_cameras*num_rays);
    int camera_idx = (idx % (num_cameras*num_rays)) / num_rays;
    int ray_idx = idx % num_rays;

    float dist = Dists[scene_idx][camera_idx][ray_idx];

    float step_size = dist / (float) n_steps;
    float rel_pos_x = 0;
    float rel_pos_y = 0;
    float rel_pos_z = 0;
    float cur_pos_x = 0;
    float cur_pos_y = 0;
    float cur_pos_z = 0;

    cur_pos_x += Origs[scene_idx][camera_idx][0];
    cur_pos_y += Origs[scene_idx][camera_idx][1];
    cur_pos_z += Origs[scene_idx][camera_idx][2];

    int h = 0;
    int w = 0;
    int d = 0;
    int h_prev = 0;
    int w_prev = 0;
    int d_prev = 0;
    // all steps until the last one count as a miss
    for (int i = 0; i < n_steps-1; i++)
    {
        cur_pos_x += Dirs[scene_idx][camera_idx][ray_idx][0] * step_size;
        cur_pos_y += Dirs[scene_idx][camera_idx][ray_idx][1] * step_size;
        cur_pos_z += Dirs[scene_idx][camera_idx][ray_idx][2] * step_size;

        rel_pos_x = (cur_pos_x - RangeMin[0]) / (RangeMax[0] - RangeMin[0]);
        rel_pos_y = (cur_pos_y - RangeMin[1]) / (RangeMax[1] - RangeMin[1]);
        rel_pos_z = (cur_pos_z - RangeMin[2]) / (RangeMax[2] - RangeMin[2]);
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
            atomicAdd(&out[scene_idx][0][h][w][d], 1);
            h_prev = h;
            w_prev = w;
            d_prev = d;
        }

    }
    // final step, doesnt count as a miss
    cur_pos_x = Origs[scene_idx][camera_idx][0] + Dirs[scene_idx][camera_idx][ray_idx][0] * dist;
    cur_pos_y = Origs[scene_idx][camera_idx][1] + Dirs[scene_idx][camera_idx][ray_idx][1] * dist;
    cur_pos_z = Origs[scene_idx][camera_idx][2] + Dirs[scene_idx][camera_idx][ray_idx][2] * dist;
    // below works too but small errors could accumulate for the hit counter (not a big deal)
    //cur_pos_x += Dirs[scene_idx][camera_idx][ray_idx][0] * step_size;
    //cur_pos_y += Dirs[scene_idx][camera_idx][ray_idx][1] * step_size;
    //cur_pos_z += Dirs[scene_idx][camera_idx][ray_idx][2] * step_size;

    rel_pos_x = (cur_pos_x - RangeMin[0]) / (RangeMax[0] - RangeMin[0]);
    rel_pos_y = (cur_pos_y - RangeMin[1]) / (RangeMax[1] - RangeMin[1]);
    rel_pos_z = (cur_pos_z - RangeMin[2]) / (RangeMax[2] - RangeMin[2]);
    h = (int)(rel_pos_x * (H-1));
    w = (int)(rel_pos_y * (W-1));
    d = (int)(rel_pos_z * (D-1));
    
    if(h >= 0 && h < H && w >= 0 && w < W && d >= 0 && d < D){
        // if we changed this voxels miss counter, undo it
        if(h == h_prev && w == w_prev && d == d_prev){
           atomicAdd(&out[scene_idx][0][h][w][d], -1);
        }
        // now just increment the hit counter
        atomicAdd(&out[scene_idx][1][h][w][d], 1);
    }
    
    return;
}


using torch::Tensor;


template <typename T>
T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

torch::Tensor counting_model_free_function(const torch::Tensor Grids, const torch::Tensor Origs, const torch::Tensor Dirs, const torch::Tensor Dists, const torch::Tensor RangeMin, const torch::Tensor RangeMax, const int n_steps){
    CHECK(Grids.is_cuda()) << "Grids should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(Origs.is_cuda()) << "Origs should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(Dirs.is_cuda()) << "Dirs should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(Dists.is_cuda()) << "Dists should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(RangeMin.is_cuda()) << "RangeMin should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(RangeMax.is_cuda()) << "RangeMax should be in GPU memory! Please call .cuda() on the tensor.";
    
    torch::Tensor out = Grids.detach().clone();
    CHECK(out.is_cuda()) << "out should be in GPU memory! Please call .cuda() on the tensor.";
    int num_scenes = Dists.size(0);
    int num_cameras = Dists.size(1);
    int num_rays = Dists.size(2);
    int numel = num_scenes*num_cameras*num_rays;

    float H = (float) Grids.size(2);
    float W = (float) Grids.size(3);
    float D = (float) Grids.size(4);

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
        Grids.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
        Origs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        Dirs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        Dists.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        RangeMin.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        RangeMax.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        out.packed_accessor32<float,5,torch::RestrictPtrTraits>()
    );
    return out;
}