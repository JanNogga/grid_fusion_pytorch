# grid_fusion_pytorch
Efficient operations for fusing depth maps or point clouds with or without semantic annotation in a 3D voxel grid in pytorch. Corresponding backward passes are WIP.
Uses [TORCH.UTILS.CPP_EXTENSION](https://pytorch.org/docs/stable/cpp_extension.html#torch-utils-cpp-extension) following the structure of [DirectVoxGO](https://github.com/sunset1995/DirectVoxGO).

## Setup
```console
pip install grid-fusion-pytorch
```

## Requirements
[PyTorch](https://pytorch.org/get-started/locally/) must be installed with CUDA support. 

## Usage
Check out the [colab example](https://pytorch.org/docs/stable/cpp_extension.html#torch-utils-cpp-extension).
