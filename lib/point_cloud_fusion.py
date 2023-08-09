import os
import torch
from torch.utils.cpp_extension import load

parent_dir = os.path.dirname(os.path.abspath(__file__))

point_cloud_fusion_util_cuda = load(
        name='point_cloud_fusion_util_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/point_cloud_fusion.cpp', 'cuda/point_cloud_fusion.cu']],
        verbose=True)

def apply_point_cloud_fusion(grid_counter, point_cloud_locations, min_range, max_range, grid_semantic=None, point_cloud_logprobs=None, verbose=False):
    assert grid_counter.is_cuda
    assert point_cloud_locations.is_cuda
    assert min_range.is_cuda
    assert max_range.is_cuda
    assert grid_semantic is None or grid_semantic.is_cuda
    assert point_cloud_logprobs is None or point_cloud_logprobs.is_cuda
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
    # check how many channels the counter grid has
    n_counters = grid_counter.shape[1]
    if n_counters > 1:
        if verbose:
            print('Warning: grid_counter should have 1 channel, but has more. Only using first channel.')
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
    # check whether a semantic map was provided
    if n_classes > 0:
        # if there is no semantic segmentation for the point cloud, warn the user and apply counting model without it
        if point_cloud_logprobs is None:
            if verbose:
                print('Warning: semantic voxel grid detected, but no semantic segmentation provided for point_clouds! Counting hits without Bayes filter.')
            return point_cloud_fusion_util_cuda.point_cloud_hit_counter_free_function(grid_counter, point_cloud_locations, min_range, max_range)
        else:
        # if there is a semantic segmentation, make sure that the number of classes matches
            assert point_cloud_logprobs.shape[-1] == n_classes
            # modify distances for background class rays
            grid_counter_out, grid_semantic_out = point_cloud_fusion_util_cuda.point_cloud_hit_counter_bayes_free_function(grid_counter, grid_semantic, point_cloud_locations, point_cloud_logprobs, min_range, max_range)
            # normalize semantic map
            grid_semantic_out -= torch.logsumexp(grid_semantic_out, dim=1, keepdim=True)
            return grid_counter_out, grid_semantic_out
    # if there is no semantic map, aggregate hit counter without Bayes filter
    else:
        # if there is a semantic segmentation, warn the user and count hits without Bayes filter
        if point_cloud_logprobs is not None:
            if verbose:
                print('Warning: semantic segmentation for point cloud detected, but no semantic map given! Incrementing hit counter without Bayes filter.')
        return point_cloud_fusion_util_cuda.point_cloud_hit_counter_free_function(grid_counter, point_cloud_locations, min_range, max_range)