import torch
import torch.nn.functional as F

import os
import zipfile
from urllib import request

CMAP_39 = torch.tensor([[188,143,143],    # master_chef_can -  0
                        [161,203,242],    # cracker_box -  1
                        [227,88,34],      # sugar_box -  2
                        [240,128,128],    # tomato_soup_can -  3
                        [247,167,0],      # mustard_bottle -  4
                        [0,191,255],      # tuna_fish_can -  5
                        [102,205,170],    # pudding_box -  6
                        [255,69,0],       # gelatin_box -  7
                        [100,68,34],      # potted_meat_can -  8
                        [243,195,0],      # banana -  9
                        [252,90,141],     # strawberry - 10
                        [154,191,89],     # apple - 11
                        [252,227,190],    # peach - 12
                        [81,59,75],       # plum - 13
                        [100,149,237],    # pitcher_base - 14
                        [137,44,22],      # bleach_cleanser - 15
                        [0,100,0],        # bowl - 16
                        [219,210,0],      # mug - 17
                        [179,68,108],     # sponge - 18
                        [41,183,0],       # spatula - 19
                        [96,78,151],      # power_drill - 20
                        [249,147,120],    # wood_block - 21
                        [205,133,63],     # scissors - 22
                        [0,255,0],        # large_marker - 23
                        [255,0,0],        # adjustable_wrench - 24
                        [255,165,0],      # flat_screwdriver - 25
                        [199,21,133],     # hammer - 26
                        [0,250,154],      # medium_clamp - 27
                        [0,103,166],      # extra_large_clamp - 28
                        [230,143,172],    # softball - 29
                        [223,255,79],     # tennis_ball - 30
                        [139,0,139],      # racquetball - 31
                        [0,136,85],       # golf_ball - 32
                        [0,0,255],        # foam_brick - 33
                        [132,132,130],    # dice - 34
                        [240,128,128],    # colored_wood_blocks - 35
                        [195,179,129],    # toy_airplane - 36
                        [0,255,255],      # rubiks_cube - 37
                        [191,0,50]])/255. # red_box - 38


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


def visualize_voxel_grid(grid_counter, grid_semantic=None, color_map=None, confidence_threshold=0.3):
    assert len(grid_counter.shape) == 4 or len(grid_counter.shape) == 5
    assert (grid_semantic is None) == (color_map is None)
    assert grid_semantic is None or (grid_semantic.shape[0] == color_map.shape[0]) or (grid_semantic.shape[1] == color_map.shape[0])
    assert grid_semantic is None or len(grid_semantic.shape) == 4 or len(grid_semantic.shape) == 5
    assert color_map is None or len(color_map.shape) == 2
    counts = grid_counter if len(grid_counter.shape) == 4 else grid_counter[0]
    misses_available = (counts.shape[0] != 1)
    semantics_available = not (grid_semantic is None)
    hits = counts[0]
    if misses_available:
        misses = counts[1]
        # probabilities as hits / (hits + misses)
        reflect_prob = calc_reflect_prob(hits, misses)
    if semantics_available:
        semantic_channels = torch.exp(grid_semantic) if len(grid_semantic.shape) == 4 else torch.exp(grid_semantic[0])
        # compute class label per voxel
        class_labels = semantic_channels.argmax(0)
        # convert class labels to colors
        vis_grid = color_map[class_labels]
        # mask out color where the prediction confidence is too low
        not_confident_mask = semantic_channels.max(0)[0] < confidence_threshold
        vis_grid[not_confident_mask,:] = torch.zeros(3).to(grid_counter.device)
    # trivial probability 1 when hit and 0 if not - ignores misses
    hit_flag = torch.minimum(hits, torch.ones(1).to(grid_counter.device))
    foreground_color = torch.zeros(*hits.shape,3).to(grid_counter.device) if not semantics_available else vis_grid
    # make a background color for compositing based on hits (and misses)
    background_color = torch.ones(*hits.shape,3).to(grid_counter.device)
    # composite using only hits
    hit_vis = hit_flag[...,None] * foreground_color + (1 - hit_flag[...,None]) * background_color
    # composite using the reflectance
    if misses_available:
        reflect_vis = reflect_prob[...,None] * foreground_color + (1 - reflect_prob[...,None]) * background_color
    # dictionary for results
    vis_result = {'reflectance': {}, 'hits': {}}
    # now permute, flip for vis along specific axis for hits and potentially reflectance slices along axis
    if misses_available:
        reflect_vis_x = (255*reflect_vis.permute(0,2,1,3).flip(1)).int().cpu().numpy()
        reflect_vis_y = (255*reflect_vis.permute(1,2,0,3).flip(1)).int().cpu().numpy()
        reflect_vis_z = (255*reflect_vis.permute(2,1,0,3).flip(0)).int().cpu().numpy()
        vis_result['reflectance']['x_axis'] = reflect_vis_x
        vis_result['reflectance']['y_axis'] = reflect_vis_y
        vis_result['reflectance']['z_axis'] = reflect_vis_z
    hit_vis_x = (255*hit_vis.permute(0,2,1,3).flip(1)).int().cpu().numpy()
    hit_vis_y = (255*hit_vis.permute(1,2,0,3).flip(1)).int().cpu().numpy()
    hit_vis_z = (255*hit_vis.permute(2,1,0,3).flip(0)).int().cpu().numpy()
    vis_result['hits']['x_axis'] = hit_vis_x
    vis_result['hits']['y_axis'] = hit_vis_y
    vis_result['hits']['z_axis'] = hit_vis_z
    return vis_result
