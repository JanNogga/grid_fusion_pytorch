import os

from torch.utils.cpp_extension import load

parent_dir = os.path.dirname(os.path.abspath(__file__))

def bayes_filter_placeholder():
    return None

counting_model_util_cuda = load(
        name='counting_model_util_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/counting_model.cpp', 'cuda/counting_model.cu']],
        verbose=True)

def apply_counting_model(Grids, Origs, Dirs, Dists, RangeMin, RangeMax, Semseg=None, n_steps=16):
    # compute number of class channels
    n_classes = Grids.shape[1] - 2
    # if class channels are present, check if we have a semantic segmentation
    if n_classes > 0:
        # if there is no semantic segmentation, warn the user and apply counting model without it
        if Semseg is None:
            print('Class channels detected in voxel grid, but no semantic segmentation provided! Applying counting model without Bayes filter.')
            return counting_model_util_cuda.counting_model_free_function(Grids, Origs, Dirs, Dists, RangeMin, RangeMax, n_steps)
        else:
            # if there is a semantic segmentation, make sure that the number of classes matches
            assert Semseg.shape[-1] == n_classes
            return bayes_filter_placeholder()
    # if there are no class channels, apply counting model without Bayes filter
    else:
        # if there is a semantic segmentation, warn the user and apply counting model without using it
        if Semseg is not None:
            print('Semantic segmentation provided, but no class channels present in voxel grid! Applying counting model without Bayes filter.')
        return counting_model_util_cuda.counting_model_free_function(Grids, Origs, Dirs, Dists, RangeMin, RangeMax, n_steps)
