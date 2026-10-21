"""
Global multi-frame 3D-Gaussian reconstruction for DrivingForward (SF mode)
on self-collected COLMAP surround-view data.

Pipeline
--------
For a continuous range of frames (with an optional stride to keep the output
small):

  1. Feed each frame's N camera images through the pretrained depth + gaussian
     networks (SF mode -> one Gaussian set per frame, all cameras aggregated
     in that frame's EGO / vehicle coordinate system).
  2. Look up the absolute vehicle pose in the world frame,
     world_from_vehicle(t), from the COLMAP reconstruction:
         world_from_vehicle = world_from_cam(ref)  @  cam_from_vehicle(ref)
     where world_from_cam(ref) comes from sparse/txt/images.txt and
     cam_from_vehicle(ref) is the inverse of sensor_to_vehicle from
     cameras.json. ref = the rig's reference camera (rig_config.json
     "ref_camera_id"); falls back to 'mid_center_top_wide'.
  3. Rigidly transform every Gaussian ego->world:
         xyz_world = R @ xyz_ego + t
         rotation(quat) left-multiplied by R
         scale       unchanged (rigid)
         SH          rotated by R (Wigner-D, via models.gaussian.rotate_sh)
  4. Concatenate all frames, optionally voxel-downsample, and write ONE global
     .ply compatible with 3DGS viewers (SuperSplat / SIBR / gsplat).

This script does NOT modify eval.py / the training pipeline. It reuses
ColmapDataset, the model, and the existing ply exporter.

Example
-------
python -W ignore reconstruct_global_ply.py \
    --weight_path /media/.../pretrained-models/DrivingForward/weights_SF \
    --data_path   /media/.../3DGS/Datasets/colmap_test_full \
    --start 0 --end 60 --stride 5 \
    --voxel 0.1 \
    --out ./results/global.ply
"""
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

import utils
from models import DrivingForwardModel
from models.gaussian import rotate_sh
from models.gaussian.ply_export import export_gaussians_to_ply
from models.gaussian.GaussianRender import _collect_valid_gaussians
from dataset.colmap_util import read_cameras_json, qvec2rotmat
from dataset import construct_dataset


# ---------------------------------------------------------------------------
# pose helpers
# ---------------------------------------------------------------------------
def _ref_camera_name(data_path):
    """Resolve the rig reference camera name from rig_config.json.

    rig_config.json stores ref_camera_id + a list of cameras with camera_id /
    image_prefix. We map ref_camera_id -> image_prefix (stripped of '/').
    Falls back to 'mid_center_top_wide' if anything is missing.
    """
    fallback = 'mid_center_top_wide'
    cfg_path = os.path.join(data_path, 'rig_config.json')
    if not os.path.isfile(cfg_path):
        return fallback
    try:
        with open(cfg_path, 'r') as f:
            rig = json.load(f)
        rig = rig[0] if isinstance(rig, list) else rig
        ref_id = rig.get('ref_camera_id')
        for cam in rig.get('cameras', []):
            if cam.get('camera_id') == ref_id:
                return cam['image_prefix'].rstrip('/')
    except Exception as e:
        print(f'[warn] failed to parse rig_config.json ({e}); '
              f'using ref camera {fallback}')
    return fallback


def build_world_from_vehicle(data_path, cam_meta, poses, ref_cam):
    """Return dict {frame_name: 4x4 world_from_vehicle (float64)}.

    world_from_vehicle(t) = world_from_cam_ref(t) @ cam_from_vehicle_ref

    - world_from_cam_ref(t): from images.txt (already cam->world), keyed by
      '<ref_cam>/<frame_name>'.
    - cam_from_vehicle_ref: inverse of sensor_to_vehicle (cam->vehicle) from
      cameras.json for the reference camera.
    """
    if ref_cam not in cam_meta:
        raise KeyError(f'ref camera {ref_cam!r} not in cameras.json '
                       f'({list(cam_meta.keys())})')

    cam_to_vehicle = cam_meta[ref_cam]['cam_to_vehicle'].astype(np.float64)  # 4x4
    vehicle_from_cam = np.linalg.inv(cam_to_vehicle)   # cam_from_vehicle

    out = {}
    prefix = ref_cam + '/'
    for key, world_from_cam in poses.items():
        if not key.startswith(prefix):
            continue
        frame_name = key[len(prefix):]
        out[frame_name] = world_from_cam @ vehicle_from_cam
    return out


def quat_mul(q1, q2):
    """Hamilton product of two (w,x,y,z) quaternion tensors [N,4]."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=1)


def rotmat_to_quat(R):
    """3x3 rotation matrix -> (w,x,y,z) quaternion (numpy, float64)."""
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z], dtype=np.float64)


# ---------------------------------------------------------------------------
# ego -> world transform of one frame's Gaussians
# ---------------------------------------------------------------------------
def transform_gaussians_to_world(xyz, rot, scale, opacity, sh, world_from_vehicle):
    """Apply a rigid ego->world transform to one frame's Gaussians.

    xyz      [N,3]   rot [N,4] (w,x,y,z)   scale [N,3]
    opacity  [N,1]   sh  [N,d_sh,3]
    world_from_vehicle: 4x4 numpy
    """
    device = xyz.device
    R = torch.tensor(world_from_vehicle[:3, :3], dtype=torch.float32, device=device)
    t = torch.tensor(world_from_vehicle[:3, 3],  dtype=torch.float32, device=device)

    # positions
    xyz_w = xyz @ R.T + t

    # rotations: compose world-rotation with each gaussian's local quat
    q_R = torch.tensor(rotmat_to_quat(world_from_vehicle[:3, :3]),
                       dtype=torch.float32, device=device).unsqueeze(0)  # [1,4]
    q_R = q_R.expand(rot.shape[0], -1)
    rot_w = quat_mul(q_R, rot)
    rot_w = torch.nn.functional.normalize(rot_w, dim=1)

    # SH rotation (Wigner-D). sh: [N, d_sh, 3] -> rotate_sh wants [..., n] with
    # the SH coeffs on the last axis, per channel. Transpose to [N,3,d_sh].
    sh_t = sh.permute(0, 2, 1).contiguous()                 # [N,3,d_sh]
    R_batch = R.view(1, 1, 3, 3).expand(sh_t.shape[0], sh_t.shape[1], 3, 3)
    sh_rot = rotate_sh(sh_t, R_batch)                       # [N,3,d_sh]
    sh_w = sh_rot.permute(0, 2, 1).contiguous()             # [N,d_sh,3]

    return xyz_w, rot_w, scale, opacity, sh_w


# ---------------------------------------------------------------------------
# voxel downsample
# ---------------------------------------------------------------------------
def voxel_downsample(xyz, voxel_size):
    """Return indices keeping ONE gaussian per occupied voxel (first seen).

    Simple hash on floored integer voxel coords. Keeps memory low and is
    deterministic. xyz: [N,3] tensor.
    """
    if voxel_size is None or voxel_size <= 0:
        return torch.arange(xyz.shape[0], device=xyz.device)
    coords = torch.floor(xyz / voxel_size).to(torch.int64)      # [N,3]
    # map 3D voxel coords to a single key; offset to keep positive-ish
    keys = coords[:, 0] * 73856093 ^ coords[:, 1] * 19349663 ^ coords[:, 2] * 83492791
    _, first_idx = np.unique(keys.cpu().numpy(), return_index=True)
    first_idx.sort()
    return torch.tensor(first_idx, device=xyz.device, dtype=torch.long)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Global multi-frame Gaussian '
                                            'reconstruction -> single .ply')
    p.add_argument('--config_file', default='./configs/nuscenes/colmap.yaml',
                   type=str, help='config yaml (default colmap SF config)')
    p.add_argument('--weight_path', required=True, type=str,
                   help='directory containing depth_net.pth / gs_net.pth / '
                        'pose_net.pth')
    p.add_argument('--data_path', default=None, type=str,
                   help='override cfg data_path (the COLMAP dir with '
                        'cameras.json + sparse/txt/images.txt)')
    p.add_argument('--start', default=0, type=int, help='first frame index')
    p.add_argument('--end', default=-1, type=int,
                   help='last frame index (exclusive); -1 = all frames')
    p.add_argument('--stride', default=5, type=int,
                   help='keep every k-th frame to limit output size')
    p.add_argument('--voxel', default=0.1, type=float,
                   help='voxel size (meters) for downsampling; <=0 disables')
    p.add_argument('--max_points', default=0, type=int,
                   help='optional hard cap on total gaussians (0 = no cap); '
                        'random-subsamples if exceeded')
    p.add_argument('--out', default='./results/global.ply', type=str,
                   help='output .ply path')
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    # SF mode is required for per-frame independent reconstruction.
    cfg = utils.get_config(args.config_file, mode='eval',
                           weight_path=args.weight_path, novel_view_mode='SF')
    if args.data_path is not None:
        cfg['data']['data_path'] = args.data_path

    data_path = cfg['data']['data_path']
    print(f'[info] data_path    = {data_path}')
    print(f'[info] weight_path  = {args.weight_path}')
    print(f'[info] frames       = [{args.start}, '
          f'{"all" if args.end < 0 else args.end}) stride {args.stride}')

    # --- model ---
    model = DrivingForwardModel(cfg, 0)
    model.load_weights()
    model.set_eval()

    # --- absolute vehicle poses in world ---
    cam_meta = read_cameras_json(os.path.join(data_path, 'cameras.json'))
    from dataset.colmap_util import read_images_poses
    poses = read_images_poses(os.path.join(data_path, 'sparse', 'txt',
                                           'images.txt'))
    ref_cam = _ref_camera_name(data_path)
    print(f'[info] ref camera   = {ref_cam}')
    world_from_vehicle = build_world_from_vehicle(data_path, cam_meta, poses,
                                                  ref_cam)
    print(f'[info] frames with vehicle pose: {len(world_from_vehicle)}')

    # --- dataset (SF eval split) ---
    _aug = {'image_shape': (int(cfg['training']['height']),
                            int(cfg['training']['width'])),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (), 'crop_eval_borders': ()}
    dataset = construct_dataset(cfg, 'eval', **_aug)
    n = len(dataset)
    end = n if args.end < 0 else min(args.end, n)
    sel = list(range(args.start, end, args.stride))
    print(f'[info] dataset samples = {n}; selected = {len(sel)}')

    all_xyz, all_rot, all_scale, all_opacity, all_sh = [], [], [], [], []
    num_cams = cfg['data']['num_cams']
    rank = 0
    skipped = 0

    for idx in tqdm(sel, desc='reconstructing'):
        inputs = dataset[idx]
        # collate a single sample into a batch of size 1
        inputs = _collate_one(inputs, rank)
        token = inputs['token'][0]
        frame_name = token + '.jpg'   # ColmapDataset token = name without ext
        if frame_name not in world_from_vehicle:
            # try .png
            frame_name_png = token + '.png'
            if frame_name_png in world_from_vehicle:
                frame_name = frame_name_png
            else:
                skipped += 1
                continue
        w_from_v = world_from_vehicle[frame_name]

        outputs = model.estimate(inputs)
        if cfg['training']['gaussian']:
            model.gs_net = model.models['gs_net']
            for cam in range(num_cams):
                model.get_gaussian_data(inputs, outputs, cam)

        # SF: aggregate all cameras at frame_id 0 (current frame, ego coords)
        fx, fr, fsc, fop, fsh = [], [], [], [], []
        for cam in range(num_cams):
            xyz_v, rot_v, scale_v, opacity_v, sh_v = _collect_valid_gaussians(
                outputs, cam, 0, 0)
            fx.append(xyz_v); fr.append(rot_v); fsc.append(scale_v)
            fop.append(opacity_v); fsh.append(sh_v)
        xyz = torch.cat(fx, 0); rot = torch.cat(fr, 0); scale = torch.cat(fsc, 0)
        opacity = torch.cat(fop, 0); sh = torch.cat(fsh, 0)

        xyz_w, rot_w, scale_w, op_w, sh_w = transform_gaussians_to_world(
            xyz, rot, scale, opacity, sh, w_from_v)

        # per-frame voxel downsample to keep memory bounded while merging
        keep = voxel_downsample(xyz_w, args.voxel)
        all_xyz.append(xyz_w[keep].cpu())
        all_rot.append(rot_w[keep].cpu())
        all_scale.append(scale_w[keep].cpu())
        all_opacity.append(op_w[keep].cpu())
        all_sh.append(sh_w[keep].cpu())

    if skipped:
        print(f'[warn] {skipped} selected frame(s) had no vehicle pose and '
              f'were skipped')
    if not all_xyz:
        raise RuntimeError('No frames reconstructed. Check --start/--end/'
                           '--stride and that frame names match images.txt.')

    xyz = torch.cat(all_xyz, 0)
    rot = torch.cat(all_rot, 0)
    scale = torch.cat(all_scale, 0)
    opacity = torch.cat(all_opacity, 0)
    sh = torch.cat(all_sh, 0)
    print(f'[info] merged gaussians (pre-global-voxel): {xyz.shape[0]:,}')

    # global voxel downsample across all frames (dedup overlap region)
    keep = voxel_downsample(xyz, args.voxel)
    xyz, rot, scale, opacity, sh = (xyz[keep], rot[keep], scale[keep],
                                    opacity[keep], sh[keep])
    print(f'[info] after global voxel ({args.voxel} m): {xyz.shape[0]:,}')

    if args.max_points and xyz.shape[0] > args.max_points:
        perm = torch.randperm(xyz.shape[0])[:args.max_points]
        xyz, rot, scale, opacity, sh = (xyz[perm], rot[perm], scale[perm],
                                        opacity[perm], sh[perm])
        print(f'[info] capped to max_points: {xyz.shape[0]:,}')

    n_written, path = export_gaussians_to_ply(
        xyz=xyz, rotation=rot, scale=scale, opacity=opacity, sh=sh,
        path=args.out, opacity_is_activated=True, scale_is_activated=True)
    print(f'[done] wrote {n_written:,} gaussians to {path}')


def _collate_one(sample, rank):
    """Turn a single ColmapDataset sample dict into a batch-of-1 on device,
    mirroring PyTorch default_collate (which the eval DataLoader would do)
    for batch_size=1: numpy/tensor -> tensor with a leading batch dim on the
    target device; bookkeeping fields wrapped in a length-1 list.

    After stack_sample+align_dataset each value is already [n_cam, ...]
    (tensor) or [n_cam, ...] (numpy); we add the batch dim -> [1, n_cam, ...].
    """
    _NO_DEVICE = ['idx', 'dataset_idx', 'sensor_name', 'filename', 'token']
    out = {}
    for k, v in sample.items():
        if k in _NO_DEVICE:
            out[k] = [v]            # batchify bookkeeping fields
            continue
        if isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0).float().to(rank)
        elif isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v).unsqueeze(0).float().to(rank)
        elif isinstance(v, list):
            # temporal context lists: list over contexts of tensors/ndarrays
            batched = []
            for x in v:
                if isinstance(x, torch.Tensor):
                    batched.append(x.unsqueeze(0).float().to(rank))
                elif isinstance(x, np.ndarray):
                    batched.append(torch.from_numpy(x).unsqueeze(0).float().to(rank))
                else:
                    batched.append(x)
            out[k] = batched
        else:
            out[k] = v
    # estimate() computes extrinsics_inv itself, but it expects a tensor
    # 'extrinsics'; nothing else to add here.
    return out


if __name__ == '__main__':
    main()
