#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void 
vector_add_free_function_gpu(
    const int numel,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> A, // 1 is dimension of tensor
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> B,
    //output
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> out
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=numel){ //don't go out of bounds
        return;
    }
    out[idx] = A[idx] +  B[idx];
}


using torch::Tensor;


template <typename T>
T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

torch::Tensor vector_add_free_function(const torch::Tensor A, const torch::Tensor B){
    CHECK(A.is_cuda()) << "A should be in GPU memory! Please call .cuda() on the tensor.";
    CHECK(B.is_cuda()) << "B should be in GPU memory! Please call .cuda() on the tensor.";
    
    torch::Tensor out = torch::empty_like(A);
    CHECK(out.is_cuda()) << "out should be in GPU memory! Please call .cuda() on the tensor.";
    int numel = A.size(0);

    const dim3 blocks = {(unsigned int)div_round_up(numel, BLOCK_SIZE), 1, 1};

    vector_add_free_function_gpu<<<blocks, BLOCK_SIZE>>>(
        numel,
        A.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        B.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        out.packed_accessor32<float,1,torch::RestrictPtrTraits>()
    );
    return out;
}
