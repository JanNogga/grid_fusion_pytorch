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

def apply_counting_model(grid_counter, ray_origins, ray_directions, ray_distances, min_range, max_range, grid_semantic=None, ray_semseg=None, n_steps=2048, background_range = 5., verbose=False, invalidate_background=False):
    assert grid_counter.is_cuda
    assert ray_origins.is_cuda
    assert ray_directions.is_cuda
    assert ray_distances.is_cuda
    assert grid_semantic is None or grid_semantic.is_cuda
    assert ray_semseg is None or ray_semseg.is_cuda
    # TODO: assert validity of inputs and document function
    # make sure the shape of grid_counter is suitable
    if len(grid_counter.shape) == 4:
        if verbose:
            print('Warning: grid_counter should be 5D, but is 4D. Adding dummy batch dimension.')
        grid_counter = grid_counter.unsqueeze(0)
    elif len(grid_counter.shape) == 3:
        if verbose:
            print('Warning: grid_counter should be 5D, but is 3D. Adding dummy batch and channel dimensions.')
        grid_counter = grid_counter.unsqueeze(0).unsqueeze(1)
    # check whether there is a channel for a miss counter - if not assume that only hits are counted
    n_counters = grid_counter.shape[1]
    # make sure the shapes of min_range and max_range are suitable
    if len(min_range.shape) == 1:
        if verbose:
            print('Warning: min_range should be 2D, but is not. Adding dummy batch dimension.')
        min_range = min_range.unsqueeze(0)
    if min_range.shape[0] != grid_counter.shape[0]:
        if verbose:
            print('Warning: min_range should have the same batch dimension as grid_counter, but does not. Assuming equal ranges for all scenes in batch.')
        min_range = min_range.expand(grid_counter.shape[0], -1)
    if len(max_range.shape) == 1:
        if verbose:
            print('Warning: max_range should be 2D, but is not. Adding dummy batch dimension.')
        max_range = max_range.unsqueeze(0)
    if max_range.shape[0] != grid_counter.shape[0]:
        if verbose:
            print('Warning: max_range should have the same batch dimension as grid_counter, but does not. Assuming equal ranges for all scenes in batch.')
        max_range = max_range.expand(grid_counter.shape[0], -1)
    # if provided, make sure the shape of the semantic voxel grid is suitable
    if grid_semantic is not None:
        if len(grid_semantic.shape) == 4:
            if verbose:
                print('Warning: grid_semantic should be 5D, but is 4D. Adding dummy batch dimension.')
            grid_semantic = grid_semantic.unsqueeze(0)
        # compute number of class channels
        n_classes = grid_semantic.shape[1]
    else:
        n_classes = 0
    # if only hits need to be counted, refer to the corresponding functions
    if n_counters == 1:
        # check whether a semantic map was provided
        if n_classes > 0:
            # if there is no semantic segmentation for rays, warn the user and apply counting model without it
            if ray_semseg is None:
                if verbose:
                    print('Warning: semantic voxel grid detected, but no semantic segmentation provided for rays! Counting hits without Bayes filter.')
                return counting_model_util_cuda.hit_counter_free_function(grid_counter, ray_origins, ray_directions, ray_distances, min_range, max_range, n_steps)
            else:
            # if there is a semantic segmentation, make sure that the number of classes matches
                assert ray_semseg.shape[-1] == n_classes
                # modify distances for background class rays
                ray_distances_background = ray_distances.clone()
                ray_distances_background[torch.isnan(ray_semseg).any(-1)] = float('nan') if invalidate_background else background_range
                # obtain aggregated hit counter and fused semantic map
                grid_counter_out, grid_semantic_out = counting_model_util_cuda.hit_counter_bayes_free_function(grid_counter, grid_semantic, ray_origins, ray_directions, ray_distances_background, ray_semseg, min_range, max_range, n_steps)
                # normalize semantic map
                grid_semantic_out -= torch.logsumexp(grid_semantic_out, dim=1, keepdim=True)
                return grid_counter_out, grid_semantic_out
        # if there is no semantic map, aggregate hit counter without Bayes filter
        else:
            # if there is a semantic segmentation, warn the user, mask background and count hits without Bayes filter
            if ray_semseg is not None:
                ray_distances_background = ray_distances.clone()
                # we never want to count background rays as hits
                #ray_distances_background[ray_semseg.sum(-1) > 0] = -1.
                ray_distances_background[torch.isnan(ray_semseg).any(-1)] = float('nan')
                if verbose:
                    print('Warning: semantic segmentation along rays detected, but no semantic map given! Incrementing hit counter for foreground pixels without Bayes filter.')
            else:
                ray_distances_background = ray_distances
            return counting_model_util_cuda.hit_counter_free_function(grid_counter, ray_origins, ray_directions, ray_distances_background, min_range, max_range, n_steps)
    # on the other hand, if misses are counted too, refer to the corresponding functions
    else:
        if n_counters > 2:
            if verbose:
                print('Warning: grid_counter should have 1 or 2 channels, but has more. Only using first two channels.')
        # check whether a semantic map was provided
        if n_classes > 0:
            # if there is no semantic segmentation, warn the user and apply counting model without it
            if ray_semseg is None:
                if verbose:
                    print('Warning: semantic voxel grid detected, but no semantic segmentation provided for rays! Applying counting model without Bayes filter.')
                return counting_model_util_cuda.counting_model_free_function(grid_counter, ray_origins, ray_directions, ray_distances, min_range, max_range, n_steps), None
            else:
                # if there is a semantic segmentation, make sure that the number of classes matches
                assert ray_semseg.shape[-1] == n_classes
                # modify distances for background class rays
                ray_distances_background = ray_distances.clone()
                ray_distances_background[torch.isnan(ray_semseg).any(-1)] = float('nan') if invalidate_background else background_range
                # apply counting model and obtain fused semantic map
                grid_counter_out, grid_semantic_out = counting_model_util_cuda.counting_model_bayes_free_function(grid_counter, grid_semantic, ray_origins, ray_directions, ray_distances_background, ray_semseg, min_range, max_range, n_steps)
                # normalize semantic map
                grid_semantic_out -= torch.logsumexp(grid_semantic_out, dim=1, keepdim=True)
                return grid_counter_out, grid_semantic_out
        # if there are no class channels, apply counting model without Bayes filter
        else:
            # if there is a semantic segmentation, warn the user and apply counting model without using it
            if ray_semseg is not None:
                ray_distances_background = ray_distances.clone()
                ray_distances_background[torch.isnan(ray_semseg).any(-1)] = float('nan') if invalidate_background else background_range
                if verbose:
                    print('Warning: semantic segmentation along rays detected, but no semantic map given! Applying counting model without Bayes filter.')
            else:
                ray_distances_background = ray_distances
            return counting_model_util_cuda.counting_model_free_function(grid_counter, ray_origins, ray_directions, ray_distances_background, min_range, max_range, n_steps), None
