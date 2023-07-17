import os
import torch
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import itertools
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class PointCloudDataset(Dataset):
    """Dataset for auto-generated cluttered bin scenes.

    Args:
        root (string): Directory where the dataset is located.
        split (string, optional): The data split to use, ``full``, ``train`` or ``val`` 
        num_steps (int, optional): The maximum number of fusion steps, -1 to use all
        num_cams (int, optional): Number of cameras to use for refinement, -1 to use all
        random_pcd (bool, optional): Whether to randomly select num_steps point clouds
        cam_world (bool, optional): Whether to return T as world -> cam
    """
    def __init__(
        self,
        root: str,
        split: str = 'full',
        num_steps: int = -1,
        num_cams: int = -1,
        random_pcd: bool = True,
        cam_world: bool = True,
        is_testset: bool = False
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.num_steps = num_steps
        self.num_cams = num_cams
        self.random_pcd = random_pcd
        self.cam_world = cam_world
        
        # load all metadata/filepaths:
        self.fusion_fp = np.load(self.root / 'metadata' / 'fusion_fp.npy', allow_pickle=True)
        self.refine_fp = np.load(self.root / 'metadata' / 'refine_fp.npy', allow_pickle=True)
        self.gt_fp = np.load(self.root / 'metadata' / 'gt_fp.npy', allow_pickle=True)
        self.class_labels = np.load(self.root / 'metadata' / 'class_labels.npy', allow_pickle=True)
        self.cam_k = torch.load(self.root / 'metadata' / 'cam_k.pt')
        self.range_min = torch.load(self.root / 'metadata' / 'range_min.pt')
        self.range_max = torch.load(self.root / 'metadata' / 'range_max.pt')
        self.is_testset = is_testset
        self.fusion_in_fp = np.load(self.root / 'metadata' / 'fusion_in_fp.npy', allow_pickle=True) if self.is_testset else None
        self.squished_classes = torch.load(self.root / 'metadata' / 'squished_classes.pt') if self.is_testset else None
        self.num_classes = self.class_labels.shape[0]
        
        # simply split by scenes if desired
        if self.split != 'full':
            n_scenes = self.fusion_fp.shape[0]
            indices = np.arange(n_scenes)
            split_frac = int(np.ceil(0.8 * n_scenes))
            train_idx, valid_idx = indices[:split_frac], indices[split_frac:]
            if self.split == 'train':
                idx = train_idx
            elif self.split == 'val':
                idx = valid_idx
            else:
                raise NotImplementedError
            # apply split
            self.fusion_fp = self.fusion_fp[idx]
            self.refine_fp = self.refine_fp[idx]
            self.gt_fp = self.gt_fp[idx]
            
        self.max_num_steps = self.fusion_fp.shape[1]
        if self.num_steps == -1:
            self.num_steps = self.max_num_steps
        else:
            self.num_steps = min(self.num_steps, self.max_num_steps)
        
        self.max_num_cams = self.refine_fp.shape[1]
        if self.num_cams == -1:
            self.num_cams = self.max_num_cams
        else:
            self.num_cams = min(self.num_cams, self.max_num_cams)
        
    def transform_pcd(self, pcd):
        ret = pcd.float()
        ret /= 65535.
        ret += 0.5
        ret *= (self.range_max - self.range_min)
        return ret + self.range_min
            
    def update_steps(self, steps: int):
        if steps == -1:
            self.num_steps = self.max_num_steps
        else:
            self.num_steps = min(steps, self.max_num_steps)
        
    def __len__(self) -> int:
        return self.fusion_fp.shape[0]
        
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        # load point clouds
        pcd_fp = self.fusion_fp[idx]
        if self.random_pcd:
            pcd_fp = np.random.choice(pcd_fp, size=self.num_steps, replace=False)
        else:
            pcd_fp = pcd_fp[:self.num_steps]
        pcd_list = [self.transform_pcd(torch.load(str(self.root) + fp)) for fp in pcd_fp]
        
        cams_fp = self.refine_fp[idx]
        cams_fp = cams_fp[:self.num_cams]
        semseg_list, cam_pose_list, depth_list = [], [], []
        for seg_fp, pose_fp, depth_fp in cams_fp:
            semseg_list.append(torch.load(str(self.root) + seg_fp).long())
            depth_list.append(torch.load(str(self.root) + depth_fp).float()/10000.)
            # return camera pose as T: world -> cam
            if self.cam_world:
                world_cam = torch.load(str(self.root) + pose_fp)
                cam_world = torch.zeros_like(world_cam)
                cam_world[:3,-1] = -1 * world_cam[:3,:3].T @ world_cam[:3,-1]
                cam_world[:3,:3] = world_cam[:3,:3].T
                cam_world[-1,-1] = 1.
                cam_pose_list.append(cam_world)
            # else return camera pose as T: cam -> world
            else:
                cam_pose_list.append(torch.load(str(self.root) + pose_fp))
        
        semseg_batch = torch.stack(semseg_list)
        depth_batch = torch.stack(depth_list)
        cam_pose_batch = torch.stack(cam_pose_list)
        # batch the semantic point clouds, padding to same length with a nonsensical value
        #pcd = pad_sequence(pcd_list, batch_first=True, padding_value=-1.0)
        # finally load the 'ground truth' point cloud
        gt_pcd = self.transform_pcd(torch.load(str(self.root) + self.gt_fp[idx]))
        
        if not self.is_testset:
            return pcd_list, semseg_batch, cam_pose_batch, depth_batch, self.cam_k.expand_as(cam_pose_batch[:, :3, :3]), gt_pcd
        else:
            fusion_in_fp = self.fusion_in_fp[idx]
            semseg_in_list, semseg_aug_in_list, depth_in_list, depth_aug_in_list, depth_mask_in_list, cam_pose_in_list = [], [], [], [], [], []
            for i in range(fusion_in_fp.shape[0]):
                semseg_in_list.append(torch.load(str(self.root) + fusion_in_fp[i,0]).squeeze(0))
                semseg_aug_in_list.append(torch.load(str(self.root) + fusion_in_fp[i,1]).squeeze(0))
                depth_in_list.append(torch.load(str(self.root) + fusion_in_fp[i,2]).squeeze(0))
                depth_aug_in_list.append(torch.load(str(self.root) + fusion_in_fp[i,3]).squeeze(0))
                try:
                    depth_mask_in_list.append(torch.load(str(self.root) + fusion_in_fp[i,4]).squeeze(0))
                except:
                    pass
                cam_pose_in_list.append(torch.load(str(self.root) + fusion_in_fp[i,5]).squeeze(0))
            depth_mask_in_batch = None if len(depth_mask_in_list) == 0 else torch.stack(depth_mask_in_list)
            testset_data = (torch.stack(semseg_in_list), torch.stack(semseg_aug_in_list), torch.stack(depth_in_list), torch.stack(depth_aug_in_list), depth_mask_in_batch, torch.stack(cam_pose_in_list))
            return pcd_list, semseg_batch, cam_pose_batch, depth_batch, self.cam_k.expand_as(cam_pose_batch[:, :3, :3]), gt_pcd, testset_data

# input: list of tuples, each like output of __getitem__
class CustomCollate():
    def __init__(self, min_num_steps, max_num_steps):
        assert min_num_steps <= max_num_steps
        self.min_num_steps = min_num_steps
        self.max_num_steps = max_num_steps
        
    def set_steps(self, min_num_steps, max_num_steps):
        assert min_num_steps <= max_num_steps
        self.min_num_steps = min_num_steps
        self.max_num_steps = max_num_steps
        
    def __call__(self, batch):
        num_steps = np.random.randint(self.min_num_steps, self.max_num_steps + 1)
        pcds, semsegs, cam_poses, depths, cam_ks, gts = [[] for i in range(num_steps)], [], [], [], [], []
        for pcd_list, semseg, cam_pose, depth, cam_k, gt in batch:
            assert len(pcd_list) >= self.max_num_steps
            for i in range(num_steps):
                pcds[i].append(pcd_list[i])
            semsegs.append(semseg)
            cam_poses.append(cam_pose)
            depths.append(depth)
            cam_ks.append(cam_k)
            gts.append(gt)
        pcds = [pad_sequence(pcd_list, batch_first=True, padding_value=-1.0) for pcd_list in pcds]
        gts = pad_sequence(gts, batch_first=True, padding_value=-1.0)
        return pcds, torch.stack(semsegs), torch.stack(cam_poses), torch.stack(depths), torch.stack(cam_ks), gts

def get_datasets(config):
    train_dataset = PointCloudDataset(root=config['dataset']['root'], split='train',
                                      num_steps=config['dataset']['num_steps'], num_cams=config['dataset']['num_cams'],
                                      random_pcd=config['dataset']['random_pcd'], cam_world=False)
    val_dataset = PointCloudDataset(root=config['dataset']['root'], split='val',
                                    num_steps=config['dataset']['num_steps'], num_cams=config['dataset']['num_cams'],
                                    random_pcd=False, cam_world=False)
    return train_dataset, val_dataset
