import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch_scatter
from typing import Optional, Tuple, List, Union, Callable

# convert Nx3 grid coords to 1D index given world size
# example: inds tensor of shape [N, 3], shape tensor([64, 64, 64])
# ...then output tensor is shape [N, 1]
def batched_ravel_index(inds, shape):
    assert inds.shape[-1] == len(shape)
    tmp = torch.cumprod(shape.flip(0), 0).flip(0)
    tmp[:-1] = tmp[1:].clone()
    tmp[-1] = 1
    return inds @ tmp[:,None]

# convert Nx1 1D index to Nx3 grid coords given world size
def batched_unravel_index(inds, shape):
    tmp = torch.cumprod(shape.flip(0), 0)
    tmp[1:] = tmp[:-1].clone()
    tmp[0] = 1
    inds_tmp = torch.div(inds, tmp, rounding_mode='trunc')
    return (inds_tmp % shape.flip(0)).flip(-1)

def get_grid(batch_size, channels, world_size, device, dtype=torch.float32):
    voxel_grid = torch.log(torch.full([batch_size, 1 + channels + 1, *world_size], 
                                      1./channels, dtype=dtype, device=device))
    voxel_grid[:,0] = 0
    voxel_grid[:,-1:] = 0
    return voxel_grid

# point cloud channels: x + y + z + class channels + features
# grid channels: density + class channels + features + hit counter + refine counter
def batch_fuse_to_grid(pcd, grid=None, world_size=None, range_min=None,
                        range_max=None, channels=None, density_step=None, dtype=torch.float32):
    assert grid is not None or (world_size is not None and channels is not None)
    batch_size = pcd.shape[0]
    device = pcd.device
    dtype = pcd.dtype
    if grid is None:
        voxel_grid = get_grid(batch_size, channels, world_size, device, dtype=dtype)
    else:
        voxel_grid = grid
        grid_size = voxel_grid.shape[-3:]
    grid_size = world_size.clone().float().to(device)
    batched_grid_size = torch.tensor([*world_size, batch_size]).float().to(device)
    # pcd is a batch of xyz + C semantic channels point cloud padded to N points per batch, identify original vals
    dim = len(grid_size)
    pad_mask = (pcd[...,dim:] >= 0.).all(-1)
    # create batch-dim fingerprint of point cloud so we don't forget what is from which batch
    fingerprint = torch.arange(batch_size).unsqueeze(-1).expand_as(pcd[...,0]).unsqueeze(-1).to(device)
    # mask the point cloud to use only valid points
    pcd_masked = pcd[pad_mask]
    # split up the rest into coordinates and distributions
    coords = pcd_masked[..., :dim]
    dists = pcd_masked[..., dim:]
    # measurements distributions are processed in logspace
    dists = torch.log(dists)
    #print('Coords and Dists shape:', coords.shape, dists.shape)
    # convert 3D coordinates to 3D voxel grid indices
    rel_inds = torch.div((coords - range_min[:3].to(device)) / (range_max[:3].to(device) - range_min[:3].to(device)), (1/grid_size), rounding_mode='trunc')
    # apply fingerprint so we don't forget which batch each valid index belongs to
    rel_inds = torch.cat([rel_inds, fingerprint[pad_mask]], -1)
    #print('ALLCLOSE?', torch.allclose(rel_inds_bak, rel_inds % grid_size.view((1,3))))
    # convert (3+1)D grid indices to corresponding 1D indices for uniqueness check
    inds = batched_ravel_index(rel_inds.double(), batched_grid_size.double()).long()
    # get N' unique voxel ids
    unique, inverse = torch.unique(inds, sorted=False, return_inverse=True)
    #print('Unique and Inverse shapes:', unique.shape, inverse.shape)
    # fuse new measurements accordingly
    #print('Scatter: dist and inv shapes:', dists.shape, inverse.shape)
    vals = torch_scatter.scatter(dists, inverse, dim=0, reduce='sum')
    # compute how many times each voxel was updated
    val_counter = torch_scatter.scatter(torch.ones_like(dists[:,:1]), inverse, dim=0, reduce='sum')
    #print('Update vals shape:', vals.shape)
    #unravel unique 1D indices to get (3+1)D voxel ids
    inds = batched_unravel_index(unique.unsqueeze(-1).double(), batched_grid_size.double()).long()
    #print('voxel ids shape:', inds.shape)
    #print(voxel_grid[inds[...,-1], :,  inds[...,0], inds[...,1], inds[...,2]].shape)
    # update density
    voxel_grid[inds[...,-1], 0,  inds[...,0], inds[...,1], inds[...,2]] += density_step * val_counter.squeeze(-1)
    # update hit counter
    voxel_grid[inds[...,-1], -1,  inds[...,0], inds[...,1], inds[...,2]] += val_counter.squeeze(-1)
    # fuse combined values to grid -> data shaped N' x C is updated here
    voxel_grid[inds[...,-1], 1:-1,  inds[...,0], inds[...,1], inds[...,2]] += vals
    # numerically stable as possible log-sum-exp normalization
    voxel_grid[inds[...,-1], 1:-1,  inds[...,0], inds[...,1], inds[...,2]] -= torch.logsumexp(voxel_grid[inds[...,-1], 1:-1,  inds[...,0], inds[...,1], inds[...,2]] , dim=1, keepdim=True)
    #print('Batched voxel_grid')
    return voxel_grid

# out: if class_channels: batch_size x N_max x (C+1)
#                   else: batch_size x N_max x 1
def batched_grid_forward(xyz, grid, range_min, range_max, class_channels=True):    
    shape = xyz.shape[:-1]
    batch_size = grid.shape[0]
    dtype = grid.dtype
    xyz = xyz.reshape(batch_size,1,1,-1,3)
    ind_norm = ((xyz - range_min[:3].to(xyz.device)) / (range_max[:3].to(xyz.device) - range_min[:3].to(xyz.device))).flip((-1,)) * 2 - 1
    if class_channels:
        out = F.grid_sample(grid[:,1:-1], ind_norm.to(dtype), mode='bilinear', align_corners=True).squeeze(-2).squeeze(-2)
    else:
        out = F.grid_sample(grid[:,:1], ind_norm.to(dtype), mode='bilinear', align_corners=True).squeeze(-2).squeeze(-2)
    out = out.permute(0,2,1)
    return out

# adapted from https://github.com/vsitzmann/light-field-networks/blob/master/geometry.py
# label_masks, world_cams, cam_ks, n_rays=-1, depth=None
def sample_rays(world_cam, cam_k, n_rays=-1, semseg=None, depth=None, H=None, W=None):
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
    ray_dirs = F.normalize(world_coords - ray_origs.unsqueeze(-2), dim=-1)
    return ray_origs, ray_dirs, gt_labels, gt_depth

### get uniformly spaced points from rays
# partially adapted from https://colab.research.google.com/drive/1TppdSsLz8uKoNwqJqDGg8se8BHQcvg_K
## IN:
# ray_origs:                         batch_size x n_cams x 3
# ray_dirs:                          batch_size x n_cams x n_rays x 3
# n_points - samples per ray:        default 500
# t_near - nearest sample distance:  default 0.1
# t_near - furthest sample distance: default 2.
## OUT:
# sample_points: batch_size x n_cams x n_rays x n_points x 3
# z: batch_size x n_cams x n_rays x n_points x 1
def sample_points_uniform(ray_origs, ray_dirs, n_points=500, t_near=0.1, t_far=2., perturb=True):
    batch_size, n_cams, n_rays = ray_dirs.shape[:-1]
    z = torch.linspace(t_near, t_far, n_points, device=ray_origs.device)
    if perturb:
        mids = .5 * (z[1:] + z[:-1])
        high = torch.cat([mids, z[-1:]], dim=-1)
        low = torch.concat([z[:1], mids], dim=-1)
        t_rand = torch.rand([batch_size, n_cams, n_rays, n_points], device=z.device)
        z = low.view(1,1,1,-1) + (high - low).view(1,1,1,-1) * t_rand
        z = z.unsqueeze(-1)
        samples = ray_origs.unsqueeze(-2).unsqueeze(-2) + ray_dirs.unsqueeze(-2) * z
    else:
        samples = (ray_origs.unsqueeze(-2) + (ray_dirs.permute(0,1,3,2).unsqueeze(-1) * z.view(1,1,-1)).flatten(-2).permute(0,1,3,2)).reshape((batch_size, n_cams, n_rays, n_points, -1))
        z = z.view(1,1,1,-1).expand_as(samples[...,0]).unsqueeze(-1)
    return samples, z

def process_density(density, z, verbose=False):
    density_mask = density > 1e-6
    if verbose:
        print('density_mask:', density_mask.shape)
    # obtain distance between samples delta
    delta = z[..., 1:,:] - z[..., :-1,:]
    delta = torch.cat([delta, 1e25 * torch.ones_like(delta[..., :1,:])], dim=-2)
    if verbose:
        print('delta:', delta.shape)
    # accumulate transmittance - TODO: SCATTER OR SOMETHING TO IMPROVE EFFICIENCY
    transmittance = torch.zeros_like(density)
    transmittance[density_mask] = delta[density_mask]*density[density_mask]
    transmittance = torch.exp(-1*torch.cumsum(transmittance, dim=-2))
    transmittance = torch.roll(transmittance, 1, -2)
    transmittance[:,:,:,0,:] = 0.
    if verbose:
        print('transmittance:', transmittance.shape)
    # get alpha
    masked_alpha = 1 - torch.exp(-1*delta[density_mask]*density[density_mask])
    if verbose:
        print('masked_alpha:', masked_alpha.shape)
    # calculate weights
    masked_weights = transmittance[density_mask] * masked_alpha
    if verbose:
        print('masked_weights:', masked_weights.shape)
    return density_mask, transmittance, masked_weights

### get ray origins, directions, and corresponding GT semseg
## IN:
# grids       - batch of voxel grids:             batch_size x 1 + channels + 2 x VX x VY x WZ
# grid_config - dictionary containing grid info:  dict
# world_cam   - transforms cam -> world frame:    batch_size x n_cams x 4 x 4
# cam_k       - camera intrinsics:                batch_size x n_cams x 3 x 3
# n_rays      - number of random rays per cam:    int or default -1
# n_points    - number of samples per ray:        int or default 500
# semseg      - optional segmentation masks:      batch_size x n_cams x 1 x H x W or default None
# depth       - optional depth image:             batch_size x n_cams x 1 x H x W or default None
# H           - optional image height:            int or None
# W           - optional image width:             int or None
## OUT:
# render:        batch_size x n_cams x n_rays x channels
# render_depth:  batch_size x n_cams x n_rays
# gt_pixels:     batch_size x n_cams x n_rays x 1 or None
# gt_depth:      batch_size x n_cams x n_rays x 1 or None
def render_grids(grids, grid_config, world_cam, cam_k, n_rays=-1, n_points=500, n_points_resample=None, semseg=None, depth=None, H=None, W=None, verbose=False, hierarchical=True, downsample_density=True, max_chunksize=500000000):
    if verbose:
        print('render_grids()')
        print('grids, world_cam, cam_k:')
        print(grids.shape, world_cam.shape, cam_k.shape)
    assert semseg is not None or depth is not None or (H is not None and W is not None)
    # determine relevant sizes to the problem
    if semseg is not None:
        H, W = semseg.shape[-2:]
    elif depth is not None:
        H, W = depth.shape[-2:]
    else:
        H, W = H, W
    batch_size, n_cams = world_cam.shape[:2]
    ray_count = n_rays if n_rays != -1 else H*W
    device = world_cam.device
    dtype=grids.dtype
    # get ray directions and origins
    ray_origs, ray_dirs, gt_labels, gt_depth = sample_rays(world_cam, cam_k, n_rays=n_rays, semseg=semseg, depth=depth, H=H, W=W)
    if verbose:
        print('ray_origs, ray_dirs:')
        print(ray_origs.shape, ray_dirs.shape)
        if gt_labels is not None:
            print('gt labels:', gt_labels.shape)
        if gt_depth is not None:
            print('gt depth:', gt_depth.shape)
    # get 3D point samples along rays along with camera z
    sample_points, z = sample_points_uniform(ray_origs, ray_dirs, n_points=n_points, t_near=grid_config['t_near'], t_far=grid_config['t_far'])
    z = z.to(dtype)
    if verbose:
        print('sample_points, z:')
        print(sample_points.shape, z.shape)
    # partition the samples into chunks to address memory requirements
    sample_points_chunks = get_chunks(sample_points.flatten(1,-2), dim=1, chunksize=max_chunksize)
    if verbose:
        print('sample point chunks:', [sample_chunk.shape for sample_chunk in sample_points_chunks])
    # query the density
    density = []
    for sample_chunk in sample_points_chunks:
        if hierarchical and downsample_density:
            ## downsample density grid
            scaled_grids, scaled_grid_config = scale_grids(grids, grid_config, torch.div(grid_config['world_size'],2, rounding_mode='trunc'))
            density.append(batched_grid_forward(sample_chunk.to(dtype), scaled_grids, 
                                           range_min=scaled_grid_config['range_min'],
                                           range_max=scaled_grid_config['range_max'],
                                           class_channels=False))
        else:
            density.append(batched_grid_forward(sample_chunk.to(dtype), grids, 
                                           range_min=grid_config['range_min'],
                                           range_max=grid_config['range_max'],
                                           class_channels=False))
    density = torch.cat(density, dim=1).view(batch_size,n_cams,ray_count,n_points,-1)
    if verbose:
        print('density:', density.shape)
    # process density
    density_mask, transmittance, masked_weights = process_density(density, z, verbose=verbose)
    ## only for hierarchical sampling:
    if hierarchical:
        weights = torch.zeros_like(density)
        weights[density_mask] = masked_weights
        if verbose:
            print('weights:', weights.shape)
        z_mid = .5 * (z[..., 1:,0] + z[..., :-1,0])
        if n_points_resample is None:
            n_points_resample = n_points // 4  
        n_points += n_points_resample
        z_new = sample_pdf(z_mid, weights[..., 1:-1,0], n_points_resample,
                              perturb=True).unsqueeze(-1).detach()
        if verbose:
            print('z new', z_new.shape)
        # combine sample points
        z, _ = torch.sort(torch.cat([z, z_new], dim=-2), dim=-2)
        z = z.to(dtype)
        sample_points = ray_origs.unsqueeze(-2).unsqueeze(-2) + ray_dirs.unsqueeze(-2) * z
        if verbose:
            print('samples new', sample_points.shape)
        # query the density again
        sample_points_chunks = get_chunks(sample_points.flatten(1,-2), dim=1, chunksize=max_chunksize)
        density = []
        for sample_chunk in sample_points_chunks:
            density.append(batched_grid_forward(sample_chunk.to(dtype), grids, 
                                       range_min=grid_config['range_min'],
                                       range_max=grid_config['range_max'],
                                       class_channels=False))
        density = torch.cat(density, dim=1).view(batch_size,n_cams,ray_count,n_points,-1)
        density_mask, transmittance, masked_weights = process_density(density, z, verbose=verbose)
    # only query semantics for points with nonzero density
    density_mask_flat = density_mask.flatten(1,-1)
    non_trivial_per_batch = density_mask_flat.sum(-1)
    masked_samples = [sample_points.flatten(1,-2)[i][density_mask_flat[i]] for i in range(batch_size)]
    masked_samples = pad_sequence(masked_samples, batch_first=True)

    if verbose:
        print('masked_samples:', masked_samples.shape)
    # query for semantics
    masked_samples_chunks = get_chunks(masked_samples, dim=1, chunksize=max_chunksize)
    padded_class_logits = []
    for masked_sample_chunk in masked_samples_chunks:
        padded_class_logits.append(batched_grid_forward(masked_samples.to(dtype), grids, 
                                       range_min=grid_config['range_min'],
                                       range_max=grid_config['range_max'], 
                                       class_channels=True))
    padded_class_logits = torch.cat(padded_class_logits, dim=1)
    # recover only logits which were not part of padding
    masked_class_logits =  torch.cat([padded_class_logits[i,:non_trivial_per_batch[i]] for i in range(batch_size)])
    if verbose:
        print('masked_class_logits:', masked_class_logits.shape)
    # USING TORCH SCATTER - Not more efficient, seems to have wonky grads
    ## obtain 3d point full ray ids:
    #ray_ids = torch.arange(sample_points[...,0,0].numel(), device=device).view(*sample_points.shape[:3],1)
    #ray_ids = ray_ids.expand_as(sample_points[...,0])
    #if verbose:
    #    print('ray_ids:', ray_ids.shape)
    # mask these to relevant rays
    #masked_ray_ids = ray_ids[density_mask.squeeze(-1)]
    #if verbose:
    #    print('masked_ray_ids:', masked_ray_ids.shape)
    #_, inverse_ray_ids = torch.unique(masked_ray_ids, sorted=False, return_inverse=True)
    #if verbose:
    #    print('inverse_ray_ids:', inverse_ray_ids.shape)
    # calculate rendering integral along nontrivial rays
    #masked_render = torch_scatter.scatter(masked_weights.unsqueeze(-1) * masked_class_logits, inverse_ray_ids, dim=0, reduce='sum')
    #if verbose:
    #    print('masked_render:', masked_render.shape)
    ### USING TORCH SCATTER
    # get mask of nontrivial rays
    #ray_mask = torch.any(density_mask.squeeze(-1), dim=-1)
    #if verbose:
    #    print('ray_mask:', ray_mask.shape)
    # fill in rendering result correctly
    render = torch.zeros((batch_size, n_cams, ray_count, n_points, masked_class_logits.shape[-1]), device=device, dtype=dtype)
    render[density_mask.squeeze(-1)] = masked_weights.unsqueeze(-1) * masked_class_logits
    render = torch.sum(render, dim=-2)
    ### USING TORCH SCATTER
    #render = torch.zeros((batch_size, n_cams, ray_count, masked_class_logits.shape[-1]), device=device)
    #render[ray_mask] = masked_render
    if verbose:
        print('render:', render.shape)
    # repeat for depth
    render_depth = torch.zeros((batch_size, n_cams, ray_count, n_points), device=device, dtype=dtype)
    render_depth[density_mask.squeeze(-1)] = masked_weights * z.expand_as(sample_points[...,:1])[density_mask]
    render_depth = torch.sum(render_depth, dim=-1)
    # USING TORCH SCATTER
    #masked_depth_render = torch_scatter.scatter(masked_weights * z.expand_as(sample_points[...,:1])[density_mask], inverse_ray_ids, dim=0, reduce='sum')
    #render_depth = torch.zeros((batch_size, n_cams, ray_count), device=semseg.device)
    #render_depth[ray_mask] = masked_depth_render
    if verbose:
        print('render_depth', render_depth.shape)
    return render, render_depth, transmittance[...,-1,0], gt_labels, gt_depth


def scale_grids(grids, grid_config, new_size):
    scaled_grids = F.interpolate(grids, size=tuple(new_size.tolist()), mode='area')
    scaled_grid_config = grid_config.copy()
    scaled_grid_config['world_size'] = new_size.long()
    axis_range = scaled_grid_config['range_max'][:3] - scaled_grid_config['range_min'][:3]
    axis_range /= torch.min(axis_range)
    scaled_grid_config['voxel_size'] = (axis_range/scaled_grid_config['world_size'].float()).mean()
    return scaled_grids, scaled_grid_config

# functions from https://colab.research.google.com/drive/1TppdSsLz8uKoNwqJqDGg8se8BHQcvg_K

def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    perturb: bool = False
) -> torch.Tensor:
    """
    Apply inverse transform sampling to a weighted set of points.
    """

    # Normalize weights to get PDF.
    pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True) # [n_rays, weights.shape[-1]]

    # Convert PDF to CDF.
    cdf = torch.cumsum(pdf, dim=-1) # [n_rays, weights.shape[-1]]
    cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1) # [n_rays, weights.shape[-1] + 1]

    # Take sample positions to grab from CDF. Linear when perturb == 0.
    if not perturb:
        u = torch.linspace(0., 1., n_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples]) # [n_rays, n_samples]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device) # [n_rays, n_samples]

    # Find indices along CDF where values in u would be placed.
    u = u.contiguous() # Returns contiguous tensor with same values.
    inds = torch.searchsorted(cdf, u, right=True) # [n_rays, n_samples]

    # Clamp indices that are out of bounds.
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1) # [n_rays, n_samples, 2]

    # Sample from cdf and the corresponding bin centers.
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1,
                       index=inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1,
                        index=inds_g)

    # Convert samples to ray length.
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples # [n_rays, n_samples]

def get_chunks(
    inputs: torch.Tensor,
    dim = 0,
    chunksize: int = 10
) -> List[torch.Tensor]:
    """
    Divide an input into chunks at specified dim.
    """
    slc = [slice(None)] * len(inputs.shape)
    if inputs.shape[dim] <= chunksize:
        return [inputs]
    else:
        ret = []
        for i in range(0, inputs.shape[dim], chunksize):
            slc = [slice(None)] * len(inputs.shape)
            slc[dim] = slice(i, i + chunksize)
            ret.append(inputs[tuple(slc)])
        return ret


def get_data_chunks(
    data: tuple,
    dim = 0,
    chunksize: int = 10
    ) -> List[tuple]:
    _, semseg, cam_pose, depth, cam_k, gt = data
    semseg_chunks = get_chunks(semseg, dim, chunksize)
    cam_pose_chunks = get_chunks(cam_pose, dim, chunksize)
    depth_chunks = get_chunks(depth, dim, chunksize)
    cam_k_chunks = get_chunks(cam_k, dim, chunksize)
    gt_chunks = get_chunks(gt, dim, chunksize)
    ret = [(None,semseg_chunks[i],cam_pose_chunks[i],depth_chunks[i],cam_k_chunks[i],gt_chunks[i]) for i in range(len(semseg_chunks))]
    return semseg.shape[dim], [(chunk.shape[dim], ret[i]) for i, chunk in enumerate(semseg_chunks)]
