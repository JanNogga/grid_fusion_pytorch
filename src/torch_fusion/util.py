import torch
import torch.nn.functional as F

import os
import zipfile
from urllib import request

# adapted from https://github.com/vsitzmann/light-field-networks/blob/master/geometry.py
# label_masks, world_cams, cam_ks, n_rays=-1, depth=None
def sample_rays(world_cam, cam_k, n_rays=-1, semseg=None, depth=None, H=None, W=None, normalize=True):
    assert semseg is not None or depth is not None or (H is not None and W is not None)
    device = world_cam.device
    # get ray origins from world_cam
    ray_origs = world_cam[..., :3, 3]
    # get relevant data sizes
    batch_size, n_cams, *_ = world_cam.shape
    if semseg is not None:
        H, W = semseg.shape[-2:]
    elif depth is not None:
        H, W = depth.shape[-2:]
    else:
        H, W = H, W
    # get pixel indices
    yx = torch.cartesian_prod(torch.arange(H), torch.arange(W)).to(device)
    # parse intrinsics matrices
    fx = cam_k[..., 0, :1]
    fy = cam_k[..., 1, 1:2]
    cx = cam_k[..., 0, 2:3]
    cy = cam_k[..., 1, 2:3]
    # if desired sample random rays per camera
    if n_rays == -1:
        y_cam = yx[..., 0]
        x_cam = yx[..., 1]
    else:
        # TODO - IMPROVE UPON randperm!!!
        rand_inds = torch.randperm(H*W, device=device)[:n_rays]
        y_cam = yx[rand_inds, 0]
        x_cam = yx[rand_inds, 1]
    # obtain corresponding pixel labels if necessary
    gt_labels = semseg[...,y_cam, x_cam].permute(0,1,3,2) if semseg is not None else None
    # if necessary obtain depth
    gt_depth = depth[...,y_cam, x_cam].permute(0,1,3,2) if depth is not None else None
    # get homogeneous pixel coordinates
    x_lift = (x_cam + 0.5 - cx) / fx
    y_lift = (y_cam + 0.5 - cy) / fy
    cam_coords_hom = torch.stack([x_lift, y_lift, torch.ones_like(x_lift), torch.ones_like(x_lift)], dim=-1)
    # convert to world coordinates
    # Sitzmann et al. use this
    # -> world_coords = torch.einsum('b...ij,b...kj->b...ki', cam_pose, cam_coords_hom)[..., :3]
    # more readable version (swap if this is bad for performance)
    world_coords = (world_cam.unsqueeze(-3) @ cam_coords_hom.unsqueeze(-1))[...,:3, 0]
    # get normalized ray directions
    if normalize:
        ray_dirs = F.normalize(world_coords - ray_origs.unsqueeze(-2), dim=-1)
    else:
        ray_dirs = world_coords - ray_origs.unsqueeze(-2)
    return ray_origs, ray_dirs, gt_labels, gt_depth

def download_example_data(out_dir, verbose=True):
    data_url = 'https://uni-bonn.sciebo.de/s/EM9iewOyFuwu2sk/download'
    local_file = 'grid_fusion_pytorch_example_data.zip'
    if os.path.isdir(out_dir):
        if verbose:
            print('Output directory already exists. Check if data is already downloaded, else delete ' + out_dir + ' and try again.')
    else:
        if verbose:
            print('Downloading data.')
        request.urlretrieve(data_url, local_file)
        os.mkdir(out_dir)
        if verbose:
                print('Unpacking data archive.')
        with zipfile.ZipFile(local_file, 'r') as zip_ref:
            zip_ref.extractall(out_dir)
        print('Removing data archive after unpacking.')
        os.remove(local_file)
        
# semseg: batch_size x n_cams x h * w x 1
# out: batch_size x n_cams x h * w x C
def soften_semseg(semseg, eps=1e-4, num_classes=None):
    if num_classes is None:
        num_classes = semseg.max().item() + 1
    semseg_delta = torch.eye(num_classes).to(semseg.device)[semseg]
    # finally soften delta peaks
    ret = torch.where(semseg_delta == 0, torch.ones_like(semseg_delta)*eps, torch.ones_like(semseg_delta)-(num_classes-1)*eps)
    return ret

def calc_reflect_prob(hits, misses):
    return torch.where(hits > 0, hits/(hits + misses), 0)