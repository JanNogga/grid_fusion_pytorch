# grid_fusion_pytorch
Efficient operations for fusing semantically annotated RGB-D measurements in a 3D voxel grid in pytorch. 
Uses [TORCH.UTILS.CPP_EXTENSION](https://pytorch.org/docs/stable/cpp_extension.html#torch-utils-cpp-extension) following the structure of [DirectVoxGO](https://github.com/sunset1995/DirectVoxGO).

## Setup

1. Clone this repository.
```console
git clone https://github.com/JanNogga/grid_fusion_pytorch.git
```
2. Build the docker image.
```console
cd grid_fusion_pytorch/docker && chmod +x build.sh && chmod +x run.sh && ./build.sh
```
3. Run a container.
```console
./run.sh
```
4. In the container, switch to this repository.
```console
cd grid_fusion_pytorch/
```
5. Finally use the custom cuda kernels. The cuda kernel defined in *lib/cuda* is compiled just-in-time.
```console
python run.py
```
