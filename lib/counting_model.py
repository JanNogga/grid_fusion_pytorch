import os

import torch
from torch.utils.cpp_extension import load

parent_dir = os.path.dirname(os.path.abspath(__file__))

counting_model_util_cuda = load(
        name='counting_model_util_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/counting_model.cpp', 'cuda/counting_model.cu']],
        verbose=True)

def apply_counting_model(Grids, Origs, Dirs, Dists, RangeMin, RangeMax, Semseg=None, n_steps=16, background_range = 5., verbose=False, invalidate_background=False):
    # compute number of class channels
    n_classes = Grids.shape[1] - 2
    # if class channels are present, check if we have a semantic segmentation
    if n_classes > 0:
        # if there is no semantic segmentation, warn the user and apply counting model without it
        if Semseg is None:
            if verbose:
                print('Class channels detected in voxel grid, but no semantic segmentation provided! Applying counting model without Bayes filter.')
            return counting_model_util_cuda.counting_model_free_function(Grids, Origs, Dirs, Dists, RangeMin, RangeMax, n_steps)
        else:
            # if there is a semantic segmentation, make sure that the number of classes matches
            assert Semseg.shape[-1] == n_classes
            # modify distances for background class rays
            dists_background = Dists.clone()
            dists_background[Semseg.sum(-1) > 0] = background_range if not invalidate_background else -1.
            grid_update = counting_model_util_cuda.counting_model_bayes_free_function(Grids, Origs, Dirs, Dists, RangeMin, RangeMax, Semseg, n_steps)
            grid_update[:,2:] -= torch.logsumexp(grid_update[:,2:], dim=1, keepdim=True)
            return grid_update
    # if there are no class channels, apply counting model without Bayes filter
    else:
        # if there is a semantic segmentation, warn the user and apply counting model without using it
        if Semseg is not None:
            dists_background = Dists.clone()
            dists_background[Semseg.sum(-1) > 0] = background_range if not invalidate_background else -1.
            if verbose:
                print('Semantic segmentation provided, but no class channels present in voxel grid! Applying counting model without Bayes filter.')
        else:
            dists_background = Dists
        return counting_model_util_cuda.counting_model_free_function(Grids, Origs, Dirs, dists_background, RangeMin, RangeMax, n_steps)
