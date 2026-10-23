"""
Stream-convert a (possibly huge) 3D Gaussian Splatting .ply into the compact
antimatter15 .splat format.

Why: the exported scene_merged.ply stores full spherical harmonics (SH) -- up
to 45 floats/point of f_rest_* -- which dominate the file size (a 17 GB ply is
mostly SH). The .splat format keeps only what a real-time web viewer renders:
position, scale, a single DC color, opacity, and rotation -- 32 bytes/point.
Dropping the higher-order SH typically shrinks the file by 60-70% with little
perceptual change, and the result loads in https://antimatter15.com/splat/ ,
SuperSplat, and most web viewers.

Memory: the .ply is read in fixed-size CHUNKS (default 1,000,000 points) and
each chunk is written straight to the output, so peak RAM is bounded by one
chunk regardless of how large the input is. A 17 GB input never lands in RAM
all at once.

.splat layout, 32 bytes per point (little-endian):
    position  3 x float32   (12 B)
    scale     3 x float32   (12 B)   -- LINEAR scale = exp(scale_* from ply)
    color     4 x uint8     ( 4 B)   -- R,G,B from DC SH, A from sigmoid(opacity)
    rotation  4 x uint8     ( 4 B)   -- normalized quat (w,x,y,z) mapped to 0..255

Usage:
    python ply_to_splat.py INPUT.ply [-o OUTPUT.splat] [--chunk N]
                                      [--max-points M] [--sort]
"""
import argparse
import os
import struct
import sys

import numpy as np
from plyfile import PlyData

# SH DC -> RGB constant (deg-0 band): C0 = 0.28209479177387814
SH_C0 = 0.28209479177387814


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _get_prop_names(vertex_el):
    return [p.name for p in vertex_el.properties]


def convert(input_path, output_path, chunk=1_000_000, max_points=None,
            sort_by_size=False):
    # Open the ply with mmap so the giant data array is NOT read into RAM; we
    # slice it chunk by chunk. plyfile memory-maps the binary payload when
    # given a filename, so vertex[...] slices are lazy reads from disk.
    ply = PlyData.read(input_path)
    vertex = ply['vertex']
    names = set(_get_prop_names(vertex))
    total = vertex.count

    required = {'x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                'scale_0', 'scale_1', 'scale_2',
                'rot_0', 'rot_1', 'rot_2', 'rot_3'}
    missing = required - names
    if missing:
        raise ValueError(f'input .ply is missing required properties: '
                         f'{sorted(missing)}')

    if max_points is not None and max_points > 0 and total > max_points:
        # uniform stride so coverage is preserved without loading everything
        stride = int(np.ceil(total / max_points))
        print(f'[convert] {total:,} points > max {max_points:,}: keeping every '
              f'{stride}th point')
    else:
        stride = 1

    # optional global size sort requires scanning scales once; skip by default
    # (it would need all scales in RAM). We stream in file order instead.
    if sort_by_size:
        print('[convert] --sort ignored in streaming mode (would need all '
              'points in RAM); writing in file order.')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    written = 0
    with open(output_path, 'wb') as fout:
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            block = vertex[start:end]  # lazy slice -> structured array in RAM

            # apply stride-based subsampling within this block
            if stride > 1:
                # global indices start..end; keep those divisible by stride
                gidx = np.arange(start, end)
                keep = (gidx % stride) == 0
                if not keep.any():
                    continue
                block = block[keep]

            n = block.shape[0]
            if n == 0:
                continue

            pos = np.stack([block['x'], block['y'], block['z']], axis=1
                           ).astype(np.float32)

            # scale: ply stores log(scale); viewer wants linear scale
            scale = np.stack([block['scale_0'], block['scale_1'],
                              block['scale_2']], axis=1).astype(np.float32)
            scale = np.exp(scale)

            # color: DC SH -> [0,1] RGB, then 0..255
            f_dc = np.stack([block['f_dc_0'], block['f_dc_1'],
                             block['f_dc_2']], axis=1).astype(np.float32)
            rgb = 0.5 + SH_C0 * f_dc
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

            # alpha: ply stores logit(opacity); viewer wants 0..255
            alpha = sigmoid(block['opacity'].astype(np.float32))
            alpha = np.clip(alpha * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1)

            color = np.concatenate([rgb, alpha], axis=1)  # [n,4] uint8

            # rotation: normalize quat, map [-1,1]->[0,255] as (w,x,y,z)
            quat = np.stack([block['rot_0'], block['rot_1'], block['rot_2'],
                             block['rot_3']], axis=1).astype(np.float32)
            norm = np.linalg.norm(quat, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            quat = quat / norm
            quat_u8 = np.clip(quat * 128.0 + 128.0, 0, 255).astype(np.uint8)

            # pack: [pos(12) scale(12) color(4) rot(4)] = 32 bytes/point
            out = np.zeros((n, 32), dtype=np.uint8)
            out[:, 0:12] = pos.view(np.uint8).reshape(n, 12)
            out[:, 12:24] = scale.view(np.uint8).reshape(n, 12)
            out[:, 24:28] = color
            out[:, 28:32] = quat_u8
            fout.write(out.tobytes())

            written += n
            print(f'\r[convert] {written:,} points written...', end='',
                  flush=True)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'\n[convert] done: {written:,} points -> {output_path} '
          f'({size_mb:.1f} MiB)')
    return written, output_path


def main():
    ap = argparse.ArgumentParser(
        description='Stream-convert a 3DGS .ply to the compact .splat format '
                    '(drops higher-order SH).')
    ap.add_argument('input', help='input .ply path (e.g. scene_merged.ply)')
    ap.add_argument('-o', '--output', default=None,
                    help='output .splat path (default: input with .splat ext)')
    ap.add_argument('--chunk', type=int, default=1_000_000,
                    help='points processed per chunk (default 1,000,000). '
                         'Lower it if RAM is tight.')
    ap.add_argument('--max-points', type=int, default=0,
                    help='if >0 and the cloud is larger, uniformly keep at most '
                         'this many points (stride subsample).')
    ap.add_argument('--sort', action='store_true',
                    help='(ignored in streaming mode) sort by gaussian size.')
    args = ap.parse_args()

    out = args.output
    if out is None:
        base = os.path.splitext(args.input)[0]
        out = base + '.splat'

    convert(args.input, out, chunk=args.chunk,
            max_points=(args.max_points or None), sort_by_size=args.sort)


if __name__ == '__main__':
    main()
