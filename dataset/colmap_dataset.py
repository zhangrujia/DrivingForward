"""
COLMAP-backed dataset for feed-forward evaluation of self-collected
surround-view data (Route A: minimal change, reuse the nuScenes pipeline).

It produces, per camera, exactly the same sample dict that
NuScenesdataset.__getitem__ yields *after* packnet's get_transforms, so the
downstream stack_sample + align_dataset + model.estimate path is unchanged.

Key field mapping (see dataset/colmap_util.py and dataset/data_util.py):
  rgb / rgb_original          : current-frame image  [C,H,W] float tensor
  rgb_context / *_original    : list over temporal contexts, each [C,H,W]
  intrinsics                  : 3x3 K, scaled to the resized (H,W)
  extrinsics                  : 4x4 cam->vehicle  (== nuScenes 'extrinsics')
  ego_pose                    : list of 4x4 cam_T_cam (current -> context)
  token / sensor_name / ...   : bookkeeping

Notes
-----
* SF eval only needs a forward neighbour. We mirror NuScenesdataset: the
  backward context (frame id -1) reuses the current frame (identity relative
  pose), the forward context (frame id +1) uses the real next frame.
* Ground-truth depth is NOT produced (eval reconstruction metrics compare
  rendered vs GT images only), so with_depth is ignored here.
"""
import os

import numpy as np
import PIL.Image as pil
import torch
import torchvision.transforms as transforms

from .colmap_util import (read_cameras_json, read_images_poses,
                          list_frames_for_camera)
from .nuscenes_dataset import stack_sample
from .data_util import align_dataset


# 4 pinhole wide cameras compatible with the nuScenes-trained weights.
_DEFAULT_COLMAP_CAMERAS = [
    'front_left_bottom_wide',
    'front_right_bottom_wide',
    'rear_left_bottom_wide',
    'rear_right_bottom_wide',
]


class ColmapDataset(torch.utils.data.Dataset):
    """Loader for a COLMAP reconstruction of custom surround-view captures."""

    def __init__(self, path, split,
                 cameras=None,
                 back_context=0,
                 forward_context=0,
                 data_transform=None,          # accepted for API parity; unused
                 depth_type=None,              # accepted for API parity; unused
                 scale_range=2,
                 with_pose=None,
                 with_ego_pose=None,
                 with_mask=None,
                 image_shape=(352, 640),       # (H, W) the networks expect
                 **kwargs):
        super().__init__()
        self.path = path
        self.split = split
        self.dataset_idx = 0

        # camera subset: config may override via cfg['data']['cameras'];
        # fall back to the 4 compatible wide cameras.
        self.cameras = cameras if cameras else list(_DEFAULT_COLMAP_CAMERAS)
        self.num_cameras = len(self.cameras)
        self.scales = np.arange(scale_range + 2)

        self.bwd = back_context
        self.fwd = forward_context
        self.has_context = back_context + forward_context > 0

        self.with_pose = with_pose
        self.with_ego_pose = with_ego_pose
        self.with_mask = with_mask

        self.height, self.width = int(image_shape[0]), int(image_shape[1])

        # --- load COLMAP metadata -------------------------------------
        self.cam_meta = read_cameras_json(os.path.join(path, 'cameras.json'))
        self.poses = read_images_poses(
            os.path.join(path, 'sparse', 'txt', 'images.txt'))

        # frames common to every selected camera, intersected and sorted
        per_cam_frames = [set(list_frames_for_camera(self.poses, c))
                          for c in self.cameras]
        common = set.intersection(*per_cam_frames) if per_cam_frames else set()
        self.frames = sorted(common)

        # keep only frames whose required neighbours exist
        self.samples = []
        for i, fr in enumerate(self.frames):
            has_next = (i + 1) < len(self.frames)
            has_prev = (i - 1) >= 0
            if self.fwd and not has_next:
                continue
            if self.bwd and split != 'eval_SF' and not has_prev:
                continue
            self.samples.append(i)

        self._to_tensor = transforms.ToTensor()
        self._resize = transforms.Resize((self.height, self.width),
                                         interpolation=pil.BILINEAR)

    def __len__(self):
        return len(self.samples)

    # -- helpers -------------------------------------------------------
    def _frame_path(self, cam, frame_name):
        return os.path.join(self.path, 'images', cam, frame_name)

    def _load_rgb(self, cam, frame_name):
        with open(self._frame_path(cam, frame_name), 'rb') as f:
            with pil.open(f) as img:
                return img.convert('RGB')

    def _scaled_K(self, cam):
        """Intrinsics scaled from native (W0,H0) to (self.width,self.height)."""
        meta = self.cam_meta[cam]
        K = meta['K'].copy().astype(np.float32)
        sx = self.width / float(meta['width'])
        sy = self.height / float(meta['height'])
        K[0, :] *= sx   # fx, cx
        K[1, :] *= sy   # fy, cy
        return K

    def _rel_pose(self, cam, cur_name, other_name):
        """cam_T_cam: current-frame camera -> other-frame camera (4x4)."""
        w_cur = self.poses[cam + '/' + cur_name]      # cam->world (current)
        w_oth = self.poses[cam + '/' + other_name]    # cam->world (other)
        rel = np.linalg.inv(w_oth) @ w_cur            # cur_cam -> oth_cam
        return rel.astype(np.float32)

    # -- main ----------------------------------------------------------
    def __getitem__(self, idx):
        fidx = self.samples[idx]
        cur_name = self.frames[fidx]
        token = os.path.splitext(cur_name)[0]  # e.g. '0217_53662391844'

        contexts = []
        if self.bwd:
            contexts.append(-1)
        if self.fwd:
            contexts.append(1)

        sample = []
        for cam in self.cameras:
            img = self._resize(self._load_rgb(cam, cur_name))
            img_t = self._to_tensor(img)  # [C,H,W] in [0,1]

            K = self._scaled_K(cam)
            extr = self.cam_meta[cam]['cam_to_vehicle'].astype(np.float32)

            data = {
                'idx': idx,
                'token': token,
                'sensor_name': cam,
                'contexts': contexts,
                'filename': os.path.join(cam, cur_name),
                'rgb': img_t,
                'rgb_original': img_t,
                'intrinsics': K,
            }
            if self.with_pose:
                data['extrinsics'] = extr

            # Self-occlusion mask: custom captures have none, so use an
            # all-ones mask (every pixel valid). The volumetric fusion net
            # always reads inputs['mask']; this keeps all pixels active.
            data['mask'] = torch.ones(1, self.height, self.width,
                                      dtype=torch.float32)

            # temporal contexts
            if self.has_context:
                ctx_imgs, ctx_poses = [], []
                for fid in contexts:
                    if fid == -1:
                        if self.split == 'eval_SF':
                            # SF: backward frame == current frame
                            nbr_name = cur_name
                        else:
                            nbr_name = self.frames[fidx - 1]
                    else:  # fid == 1, forward
                        nbr_name = self.frames[fidx + 1]

                    nbr_img = self._to_tensor(self._resize(
                        self._load_rgb(cam, nbr_name)))
                    ctx_imgs.append(nbr_img)

                    if fid == -1 and self.split == 'eval_SF':
                        rel = np.eye(4, dtype=np.float32)
                    else:
                        rel = self._rel_pose(cam, cur_name, nbr_name)
                    ctx_poses.append(rel)

                data['rgb_context'] = ctx_imgs
                data['rgb_context_original'] = ctx_imgs
                if self.with_ego_pose:
                    data['ego_pose'] = ctx_poses

            sample.append(data)

        sample = stack_sample(sample)
        sample = align_dataset(sample, self.scales, contexts)
        return sample
