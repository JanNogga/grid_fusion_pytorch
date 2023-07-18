#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void 
increment_misses_free_function_gpu(
    const int numel,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Grids, // 4 is dimension of tensor
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Origs,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Dirs,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Dists,
    //output
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> out
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=numel){ //don't go out of bounds
        return;
    }
    // out[idx] = Grids[idx];
    return;
}


using torch::Tensor;


template <typename T>
T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

torch::Tensor increment_misses_free_function(const torch::Tensor Grids, const torch::Tensor Origs, const torch::Tensor Dirs, const torch::Tensor Dists){
    CHECK(Grids.is_cuda()) << "Grids should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(Origs.is_cuda()) << "Origs should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(Dirs.is_cuda()) << "Dirs should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(Dists.is_cuda()) << "Dists should be in GPU memory! Please call .cuda() on the tensor.";
    
    torch::Tensor out = torch::zeros_like(Grids);
    CHECK(out.is_cuda()) << "out should be in GPU memory! Please call .cuda() on the tensor.";
    int numel = Grids.size(0);

    const dim3 blocks = {(unsigned int)div_round_up(numel, BLOCK_SIZE), 1, 1};

    increment_misses_free_function_gpu<<<blocks, BLOCK_SIZE>>>(
        numel,
        Grids.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        Origs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        Dirs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        Dists.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        out.packed_accessor32<float,4,torch::RestrictPtrTraits>()
    );
    return out;
}