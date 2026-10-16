"""
Export the feed-forward 3D Gaussian scene to a .ply file compatible with the
official 3D Gaussian Splatting viewers (SuperSplat, the original SIBR viewer,
gsplat-based tools, etc.).

The model predicts, per valid pixel, a set of Gaussians with:
    xyz      [N, 3]      position in world/ego coordinates
    rotation [N, 4]      quaternion (r, x, y, z)
    scale    [N, 3]      per-axis scale (already in linear space as used by the
                         rasterizer; stored as log() to match the official format)
    opacity  [N, 1]      opacity in [0, 1] (stored as inverse-sigmoid/logit)
    sh       [N, d_sh, 3] spherical-harmonics coefficients, d_sh = (deg+1)^2

Official .ply layout expects:
    x, y, z, nx, ny, nz,
    f_dc_0..2            (SH degree 0, one value per color channel)
    f_rest_0..(3*(d_sh-1)-1)  (remaining SH, CHANNEL-major: all of R, then G, then B)
    opacity,
    scale_0..2,
    rot_0..3
"""
import numpy as np
import torch
from plyfile import PlyData, PlyElement


def _build_attribute_list(d_sh):
    attrs = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # DC term: 3 channels
    for i in range(3):
        attrs.append(f'f_dc_{i}')
    # rest: 3 * (d_sh - 1), channel-major
    for i in range(3 * (d_sh - 1)):
        attrs.append(f'f_rest_{i}')
    attrs.append('opacity')
    for i in range(3):
        attrs.append(f'scale_{i}')
    for i in range(4):
        attrs.append(f'rot_{i}')
    return attrs


def export_gaussians_to_ply(xyz, rotation, scale, opacity, sh, path,
                            opacity_is_activated=True, scale_is_activated=True):
    """
    Save one scene's Gaussians to a .ply file.

    Args
        xyz:      tensor/array [N, 3]
        rotation: tensor/array [N, 4]
        scale:    tensor/array [N, 3]
        opacity:  tensor/array [N, 1] or [N]
        sh:       tensor/array [N, d_sh, 3]  (d_sh-major, channel last)
        path:     output .ply path
        opacity_is_activated: if True, opacity is in [0,1] and will be converted
                              back to logit space (official format stores logits).
        scale_is_activated:   if True, scale is linear and will be stored as log().
    """
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    xyz = to_np(xyz)                       # [N, 3]
    rotation = to_np(rotation)             # [N, 4]
    scale = to_np(scale)                   # [N, 3]
    opacity = to_np(opacity).reshape(-1, 1)  # [N, 1]
    sh = to_np(sh)                         # [N, d_sh, 3]

    n = xyz.shape[0]
    d_sh = sh.shape[1]

    # --- SH split into DC (deg 0) and the rest, in official channel-major order ---
    # sh: [N, d_sh, 3] -> DC [N, 3] (the deg-0 coeff per channel)
    f_dc = sh[:, 0, :]                     # [N, 3]
    # rest: [N, d_sh-1, 3] -> transpose to [N, 3, d_sh-1] -> flatten channel-major
    f_rest = sh[:, 1:, :]                  # [N, d_sh-1, 3]
    f_rest = np.transpose(f_rest, (0, 2, 1))          # [N, 3, d_sh-1]
    f_rest = f_rest.reshape(n, -1)                     # [N, 3*(d_sh-1)], channel-major

    # --- opacity: store as logit if currently activated (in [0,1]) ---
    if opacity_is_activated:
        eps = 1e-6
        o = np.clip(opacity, eps, 1 - eps)
        opacity_store = np.log(o / (1 - o))
    else:
        opacity_store = opacity

    # --- scale: store as log if currently linear ---
    if scale_is_activated:
        scale_store = np.log(np.clip(scale, 1e-8, None))
    else:
        scale_store = scale

    normals = np.zeros((n, 3), dtype=np.float32)

    attrs = _build_attribute_list(d_sh)
    dtype = [(a, 'f4') for a in attrs]
    elements = np.empty(n, dtype=dtype)

    data = np.concatenate(
        [xyz, normals, f_dc, f_rest, opacity_store, scale_store, rotation],
        axis=1,
    ).astype(np.float32)

    for idx, a in enumerate(attrs):
        elements[a] = data[:, idx]

    el = PlyElement.describe(elements, 'vertex')
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    PlyData([el]).write(path)
    return n, path
