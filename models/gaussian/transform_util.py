"""
Rigid ego->world transform + voxel downsample helpers for merging per-frame
3D Gaussians into a single global point cloud.

These are pure tensor/numpy utilities with no dependency on the model or the
datasets, so they can be imported from both the trainer (eval-time scene merge)
and the standalone reconstruct_*_ply.py scripts without circular imports.
"""
import numpy as np
import torch

from .utils import rotate_sh


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


def transform_gaussians_to_world(xyz, rot, scale, opacity, sh, world_from_vehicle,
                                 rotate_sh_flag=False):
    """Apply a rigid ego->world transform to one frame's Gaussians.

    xyz      [N,3]   rot [N,4] (w,x,y,z)   scale [N,3]
    opacity  [N,1]   sh  [N,d_sh,3]
    world_from_vehicle: 4x4 numpy

    rotate_sh_flag: if True, rotate high-order SH by R. Off by default: the DC
        term (SH degree 0) is rotation-invariant and is what most 3DGS viewers
        render, so we skip the expensive per-point Wigner-D (which can also OOM
        on millions of points). Enable for best fidelity on smaller clouds.
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

    # SH rotation (Wigner-D). sh: [N, d_sh, 3].
    if rotate_sh_flag:
        sh_t = sh.permute(0, 2, 1).contiguous()             # [N,3,d_sh]
        R_batch = R.view(1, 1, 3, 3).expand(sh_t.shape[0], sh_t.shape[1], 3, 3)
        R_batch = R_batch.contiguous()
        sh_rot = rotate_sh(sh_t, R_batch)                   # [N,3,d_sh]
        sh_w = sh_rot.permute(0, 2, 1).contiguous()         # [N,d_sh,3]
    else:
        sh_w = sh

    return xyz_w, rot_w, scale, opacity, sh_w


def voxel_downsample(xyz, voxel_size):
    """Return indices keeping ONE gaussian per occupied voxel (first seen).

    Simple hash on floored integer voxel coords. Keeps memory low and is
    deterministic. xyz: [N,3] tensor.
    """
    if voxel_size is None or voxel_size <= 0:
        return torch.arange(xyz.shape[0], device=xyz.device)
    coords = torch.floor(xyz / voxel_size).to(torch.int64)      # [N,3]
    keys = coords[:, 0] * 73856093 ^ coords[:, 1] * 19349663 ^ coords[:, 2] * 83492791
    _, first_idx = np.unique(keys.cpu().numpy(), return_index=True)
    first_idx.sort()
    return torch.tensor(first_idx, device=xyz.device, dtype=torch.long)
