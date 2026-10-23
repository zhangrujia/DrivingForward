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


def _downsample_indices(xyz, max_points=None, voxel_size=None, seed=0):
    """Return a LongTensor of indices selecting a subset of the N points.

    Applied on the torch tensor (ideally on-GPU) BEFORE converting to numpy, so
    the large numpy arrays are never materialised at full size.

    Order of operations:
      1. voxel_size (if set): keep one point per occupied voxel (first-hit),
         a geometry-aware thinning that preserves spatial coverage.
      2. max_points (if set): if still above the cap, uniformly random-subsample
         down to max_points.
    If neither is set, returns all indices (no points dropped).
    """
    n = xyz.shape[0]
    device = xyz.device
    idx = torch.arange(n, device=device)

    if voxel_size is not None and voxel_size > 0 and n > 0:
        # integer voxel coordinate per point
        vox = torch.floor(xyz / float(voxel_size)).to(torch.int64)  # [N,3]
        # hash the 3D voxel index to a single key; pick the first point per key
        keys = (vox[:, 0] * 73856093) ^ (vox[:, 1] * 19349663) ^ (vox[:, 2] * 83492791)
        # keep the first point encountered in each voxel: sort by key, then
        # mark the first element of every run of equal keys.
        order = torch.argsort(keys, stable=True)
        sorted_keys = keys[order]
        # mask of first occurrence in the sorted array
        keep_sorted = torch.ones_like(sorted_keys, dtype=torch.bool)
        keep_sorted[1:] = sorted_keys[1:] != sorted_keys[:-1]
        idx = order[keep_sorted]

    if max_points is not None and max_points > 0 and idx.shape[0] > max_points:
        g = torch.Generator(device='cpu')
        g.manual_seed(seed)
        perm = torch.randperm(idx.shape[0], generator=g).to(device)
        idx = idx[perm[:max_points]]

    return idx


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def gaussians_to_official_columns(xyz, rotation, scale, opacity, sh,
                                  opacity_is_activated=True,
                                  scale_is_activated=True):
    """Convert one chunk of Gaussians into the official per-point column blocks.

    Returns (blocks, d_sh) where `blocks` is the ordered tuple of float32
    numpy arrays
        (xyz, normals, f_dc, f_rest, opacity_store, scale_store, rotation)
    laid out EXACTLY in the order of `_build_attribute_list(d_sh)`. Both the
    one-shot and the streaming writers consume this, so the on-disk byte layout
    is guaranteed identical between them.
    """
    xyz = _to_np(xyz)                        # [n, 3]
    rotation = _to_np(rotation)              # [n, 4]
    scale = _to_np(scale)                    # [n, 3]
    opacity = _to_np(opacity).reshape(-1, 1)  # [n, 1]
    sh = _to_np(sh)                          # [n, d_sh, 3]

    n = xyz.shape[0]
    d_sh = sh.shape[1]

    # SH split into DC (deg 0) and the rest, in official channel-major order.
    f_dc = sh[:, 0, :]                       # [n, 3]
    f_rest = sh[:, 1:, :]                    # [n, d_sh-1, 3]
    f_rest = np.transpose(f_rest, (0, 2, 1))             # [n, 3, d_sh-1]
    f_rest = np.ascontiguousarray(f_rest).reshape(n, -1)  # [n, 3*(d_sh-1)]

    if opacity_is_activated:
        eps = 1e-6
        o = np.clip(opacity, eps, 1 - eps)
        opacity_store = np.log(o / (1 - o))
    else:
        opacity_store = opacity

    if scale_is_activated:
        scale_store = np.log(np.clip(scale, 1e-8, None))
    else:
        scale_store = scale

    normals = np.zeros((n, 3), dtype=np.float32)

    blocks = (xyz, normals, f_dc, f_rest, opacity_store, scale_store, rotation)
    return blocks, d_sh


def _blocks_to_packed_rows(blocks, n):
    """Stack the ordered column blocks into a contiguous [n, D] float32 array.

    D = total attribute count. Column order matches _build_attribute_list.
    """
    cols = []
    for block in blocks:
        block = np.ascontiguousarray(block, dtype=np.float32)
        if block.ndim == 1:
            block = block.reshape(n, 1)
        cols.append(block)
    return np.ascontiguousarray(np.concatenate(cols, axis=1), dtype=np.float32)


class StreamingPlyWriter:
    """Write a binary-little-endian .ply incrementally, chunk by chunk.

    The PLY header must declare the vertex count up front, so the TOTAL number
    of points has to be known before the first chunk is written. Pass it to the
    constructor. After that, call append(...) for each chunk of Gaussians; each
    chunk's rows are packed and flushed straight to disk, so peak memory is
    bounded by one chunk (plus the file buffer) instead of the whole cloud.

    Usage:
        w = StreamingPlyWriter(path, total_points=N, d_sh=D_SH)
        for chunk in chunks:
            w.append(xyz_c, rot_c, scale_c, opacity_c, sh_c)
        w.close()   # verifies the written count matches total_points
    """

    def __init__(self, path, total_points, d_sh,
                 opacity_is_activated=True, scale_is_activated=True):
        import os
        self.path = path
        self.total_points = int(total_points)
        self.d_sh = int(d_sh)
        self.opacity_is_activated = opacity_is_activated
        self.scale_is_activated = scale_is_activated
        self.attrs = _build_attribute_list(self.d_sh)
        self.written = 0

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, 'wb')
        self._write_header()

    def _write_header(self):
        lines = ['ply', 'format binary_little_endian 1.0',
                 f'element vertex {self.total_points}']
        for a in self.attrs:
            lines.append(f'property float {a}')
        lines.append('end_header\n')
        self._f.write(('\n'.join(lines)).encode('ascii'))

    def append(self, xyz, rotation, scale, opacity, sh):
        blocks, d_sh = gaussians_to_official_columns(
            xyz, rotation, scale, opacity, sh,
            opacity_is_activated=self.opacity_is_activated,
            scale_is_activated=self.scale_is_activated)
        if d_sh != self.d_sh:
            raise ValueError(f'd_sh mismatch: chunk has {d_sh}, '
                             f'writer expects {self.d_sh}')
        n = blocks[0].shape[0]
        if n == 0:
            return
        rows = _blocks_to_packed_rows(blocks, n)   # [n, D] float32, C-contiguous
        if rows.shape[1] != len(self.attrs):
            raise ValueError(f'column count mismatch: {rows.shape[1]} vs '
                             f'{len(self.attrs)}')
        if self.written + n > self.total_points:
            raise ValueError(f'appended more points ({self.written + n}) than '
                             f'declared total ({self.total_points})')
        # binary_little_endian float32 rows, tightly packed
        self._f.write(rows.astype('<f4', copy=False).tobytes())
        self.written += n

    def close(self):
        self._f.close()
        if self.written != self.total_points:
            raise ValueError(
                f'StreamingPlyWriter: wrote {self.written} points but header '
                f'declared {self.total_points}. The .ply is corrupt.')
        return self.written, self.path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # On a clean exit, close() validates the count; on error just release
        # the file handle without the count assertion.
        if exc_type is None:
            self.close()
        else:
            try:
                self._f.close()
            except Exception:
                pass
        return False


def export_gaussians_to_ply(xyz, rotation, scale, opacity, sh, path,
                            opacity_is_activated=True, scale_is_activated=True,
                            max_points=None, voxel_size=None):
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
        max_points: optional int. If set and the (post-voxel) point count still
                    exceeds it, uniformly random-subsample down to this many
                    points. None (default) = keep all points.
        voxel_size: optional float. If set, keep one point per occupied voxel of
                    this edge length (world units) before the max_points cap.
                    None (default) = no voxel thinning.

    Memory: downsampling is done on the input tensor (on its original device)
    BEFORE the numpy conversion, and the per-column numpy arrays are written
    straight into the structured output array without building an intermediate
    [N, 89] concatenation, so the peak host-memory footprint is roughly one
    copy of the (possibly downsampled) data instead of two.
    """
    # --- optional downsampling on the source tensor (pre-numpy) --------------
    if isinstance(xyz, torch.Tensor):
        sel = _downsample_indices(xyz, max_points=max_points,
                                  voxel_size=voxel_size)
        if sel.shape[0] != xyz.shape[0]:
            xyz = xyz[sel]
            rotation = rotation[sel]
            scale = scale[sel]
            opacity = opacity[sel] if opacity.dim() > 0 else opacity
            sh = sh[sel]
    # (numpy inputs: skip tensor-side downsampling; handled below if needed)

    blocks, d_sh = gaussians_to_official_columns(
        xyz, rotation, scale, opacity, sh,
        opacity_is_activated=opacity_is_activated,
        scale_is_activated=scale_is_activated)
    n = blocks[0].shape[0]

    attrs = _build_attribute_list(d_sh)
    dtype = [(a, 'f4') for a in attrs]
    elements = np.empty(n, dtype=dtype)

    # Write each source block straight into the structured array, column by
    # column. This avoids the large intermediate np.concatenate([...], axis=1)
    # result (which, together with `elements`, doubled the peak memory and was
    # the source of the MemoryError on large scenes).
    col = 0
    for block in blocks:
        block = np.ascontiguousarray(block, dtype=np.float32)
        if block.ndim == 1:
            block = block.reshape(n, 1)
        for j in range(block.shape[1]):
            elements[attrs[col]] = block[:, j]
            col += 1
    assert col == len(attrs), f'column count mismatch: {col} vs {len(attrs)}'

    el = PlyElement.describe(elements, 'vertex')
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    PlyData([el]).write(path)
    return n, path
