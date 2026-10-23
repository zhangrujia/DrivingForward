"""
Diagnose why a 3DGS .ply renders as a black screen in web viewers.

Reads a .ply (streaming the vertex data in chunks so even a huge file is safe)
and reports the statistics that most often explain a black / empty viewport:

  1. POSITION range + centroid. nuScenes ego_pose lives in a global map frame,
     so merged clouds can sit hundreds/thousands of metres from the origin.
     Many web viewers frame the camera at the origin -> the cloud is off-screen
     (looks black). Huge coordinate magnitudes are the #1 cause here.
  2. OPACITY (stored as logit). After sigmoid, if almost everything is ~0 the
     scene is invisible.
  3. SCALE (stored as log). After exp(), NaN/inf or extreme values blow up or
     collapse the splats.
  4. COLOR from DC SH (0.5 + C0*f_dc_*). If it collapses to ~0 everything is
     black even when geometry is fine.
  5. NaN / inf counts across all of the above.

Usage:
    python ply_diagnose.py INPUT.ply [--chunk N]
"""
import argparse
import os

import numpy as np
from plyfile import PlyData

SH_C0 = 0.28209479177387814


def _names(v):
    return [p.name for p in v.properties]


def _fmt(a):
    return (f'min={np.nanmin(a):.4g} max={np.nanmax(a):.4g} '
            f'mean={np.nanmean(a):.4g}')


def diagnose(path, chunk=1_000_000):
    ply = PlyData.read(path)
    v = ply['vertex']
    names = set(_names(v))
    total = v.count
    print(f'[diagnose] file: {path}')
    print(f'[diagnose] points: {total:,}')
    print(f'[diagnose] properties ({len(names)}): {sorted(names)}')

    # running accumulators
    pos_min = np.full(3, np.inf); pos_max = np.full(3, -np.inf)
    pos_sum = np.zeros(3, dtype=np.float64)
    op_min, op_max, op_sum = np.inf, -np.inf, 0.0
    alpha_lt01 = 0  # count of points with sigmoid(opacity) < 0.1
    sc_min, sc_max = np.inf, -np.inf           # linear scale after exp
    col_sum = np.zeros(3, dtype=np.float64)
    col_min = np.full(3, np.inf); col_max = np.full(3, -np.inf)
    nan_pos = nan_op = nan_sc = nan_col = 0
    n_seen = 0

    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        b = v[start:end]
        n = end - start
        n_seen += n

        pos = np.stack([b['x'], b['y'], b['z']], 1).astype(np.float64)
        pos_min = np.minimum(pos_min, np.nanmin(pos, 0))
        pos_max = np.maximum(pos_max, np.nanmax(pos, 0))
        pos_sum += np.nansum(pos, 0)
        nan_pos += int(np.isnan(pos).any(1).sum() + np.isinf(pos).any(1).sum())

        op = b['opacity'].astype(np.float64)
        op_min = min(op_min, np.nanmin(op)); op_max = max(op_max, np.nanmax(op))
        op_sum += np.nansum(op)
        alpha = 1.0 / (1.0 + np.exp(-op))
        alpha_lt01 += int((alpha < 0.1).sum())
        nan_op += int(np.isnan(op).sum() + np.isinf(op).sum())

        sc = np.stack([b['scale_0'], b['scale_1'], b['scale_2']], 1
                      ).astype(np.float64)
        sc_lin = np.exp(sc)
        sc_min = min(sc_min, np.nanmin(sc_lin))
        sc_max = max(sc_max, np.nanmax(sc_lin))
        nan_sc += int(np.isnan(sc_lin).any(1).sum() + np.isinf(sc_lin).any(1).sum())

        fdc = np.stack([b['f_dc_0'], b['f_dc_1'], b['f_dc_2']], 1
                       ).astype(np.float64)
        rgb = 0.5 + SH_C0 * fdc
        col_sum += np.nansum(rgb, 0)
        col_min = np.minimum(col_min, np.nanmin(rgb, 0))
        col_max = np.maximum(col_max, np.nanmax(rgb, 0))
        nan_col += int(np.isnan(rgb).any(1).sum() + np.isinf(rgb).any(1).sum())

    centroid = pos_sum / max(n_seen, 1)
    extent = pos_max - pos_min
    print('\n=== POSITION (world coords) ===')
    print(f'  x: [{pos_min[0]:.2f}, {pos_max[0]:.2f}]  '
          f'y: [{pos_min[1]:.2f}, {pos_max[1]:.2f}]  '
          f'z: [{pos_min[2]:.2f}, {pos_max[2]:.2f}]')
    print(f'  centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, '
          f'{centroid[2]:.2f})')
    print(f'  extent (bbox size): ({extent[0]:.2f}, {extent[1]:.2f}, '
          f'{extent[2]:.2f})')
    dist = float(np.linalg.norm(centroid))
    if dist > 100:
        print(f'  >>> WARNING: centroid is {dist:.1f} m from origin. Many web '
              f'viewers frame the camera at (0,0,0); the cloud is likely '
              f'OFF-SCREEN (black screen). Recenter to the origin.')

    print('\n=== OPACITY ===')
    print(f'  logit: min={op_min:.4g} max={op_max:.4g} '
          f'mean={op_sum/max(n_seen,1):.4g}')
    frac = alpha_lt01 / max(n_seen, 1)
    print(f'  sigmoid(opacity) < 0.1: {alpha_lt01:,} points ({frac*100:.1f}%)')
    if frac > 0.95:
        print('  >>> WARNING: almost all points are near-transparent; scene '
              'will look invisible.')

    print('\n=== SCALE (linear, after exp) ===')
    print(f'  min={sc_min:.4g} max={sc_max:.4g}')
    if sc_max > 100 or not np.isfinite(sc_max):
        print('  >>> WARNING: extreme/non-finite scale; splats may blow up.')

    print('\n=== COLOR (DC SH -> RGB, 0..1) ===')
    cmean = col_sum / max(n_seen, 1)
    print(f'  R: [{col_min[0]:.3f}, {col_max[0]:.3f}] mean={cmean[0]:.3f}')
    print(f'  G: [{col_min[1]:.3f}, {col_max[1]:.3f}] mean={cmean[1]:.3f}')
    print(f'  B: [{col_min[2]:.3f}, {col_max[2]:.3f}] mean={cmean[2]:.3f}')
    if np.all(cmean < 0.05):
        print('  >>> WARNING: colors collapse to black.')

    print('\n=== NaN / inf counts ===')
    print(f'  position: {nan_pos:,}  opacity: {nan_op:,}  '
          f'scale: {nan_sc:,}  color: {nan_col:,}')
    if nan_pos or nan_op or nan_sc or nan_col:
        print('  >>> WARNING: non-finite values present; many viewers drop or '
              'mis-render such points.')

    print('\n[diagnose] done.')


def main():
    ap = argparse.ArgumentParser(description='Diagnose black-screen causes in '
                                             'an exported 3DGS .ply.')
    ap.add_argument('input', help='input .ply path')
    ap.add_argument('--chunk', type=int, default=1_000_000)
    args = ap.parse_args()
    diagnose(args.input, chunk=args.chunk)


if __name__ == '__main__':
    main()
