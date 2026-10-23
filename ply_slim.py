"""
Stream-slim a (possibly huge) 3D Gaussian Splatting .ply by dropping the
higher-order spherical-harmonics (f_rest_*) while keeping the output a
standard, viewer-compatible .ply.

Why: the exported scene_merged.ply stores full SH -- up to 45 floats/point of
f_rest_* -- which dominate the file size. For just viewing the scene, only the
DC color (f_dc_*) matters; the higher bands add view-dependent shading that
most quick inspections don't need. Dropping f_rest_* turns the cloud into an
SH degree-0 (DC-only) .ply that SuperSplat, the SIBR viewer, and other standard
3DGS tools load directly -- at 60-70% smaller size.

Kept properties (standard DC-only 3DGS layout):
    x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2,
    opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3
Dropped:
    f_rest_* (all higher-order SH)

Output is binary_little_endian .ply. Normals (nx,ny,nz) are preserved as-is
(they are zeros in our export) for maximum viewer compatibility.

Memory: the input is read in fixed-size CHUNKS (default 1,000,000 points) and
each slimmed chunk is streamed straight to the output file, so peak RAM is
bounded by one chunk regardless of input size. A 17 GB input never lands in
RAM all at once. The output header's vertex count is known up front from the
input, so no second pass is needed.

Usage:
    python ply_slim.py INPUT.ply [-o OUTPUT.ply] [--chunk N] [--max-points M]
"""
import argparse
import os

import numpy as np
from plyfile import PlyData


# properties kept in the slimmed output, in canonical order
KEEP_PROPS = ['x', 'y', 'z', 'nx', 'ny', 'nz',
              'f_dc_0', 'f_dc_1', 'f_dc_2',
              'opacity', 'scale_0', 'scale_1', 'scale_2',
              'rot_0', 'rot_1', 'rot_2', 'rot_3']
# the minimum set that must exist in the input (normals are optional)
REQUIRED = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2',
            'opacity', 'scale_0', 'scale_1', 'scale_2',
            'rot_0', 'rot_1', 'rot_2', 'rot_3']


def _prop_names(vertex_el):
    return [p.name for p in vertex_el.properties]


def _write_header(fout, out_props, count):
    lines = ['ply', 'format binary_little_endian 1.0',
             f'element vertex {count}']
    for p in out_props:
        lines.append(f'property float {p}')
    lines.append('end_header\n')
    fout.write(('\n'.join(lines)).encode('ascii'))


def slim(input_path, output_path, chunk=1_000_000, max_points=None):
    ply = PlyData.read(input_path)          # mmaps the binary payload
    vertex = ply['vertex']
    names = set(_prop_names(vertex))
    total = vertex.count

    missing = [p for p in REQUIRED if p not in names]
    if missing:
        raise ValueError(f'input .ply is missing required properties: {missing}')

    # keep only the canonical props that actually exist in the input
    out_props = [p for p in KEEP_PROPS if p in names]
    have_normals = all(p in names for p in ('nx', 'ny', 'nz'))

    # stride subsample if a point cap is requested
    if max_points is not None and max_points > 0 and total > max_points:
        stride = int(np.ceil(total / max_points))
        kept_total = len(range(0, total, stride))
        print(f'[slim] {total:,} points > max {max_points:,}: keeping every '
              f'{stride}th point -> {kept_total:,}')
    else:
        stride = 1
        kept_total = total

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    written = 0
    with open(output_path, 'wb') as fout:
        _write_header(fout, out_props, kept_total)

        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            block = vertex[start:end]       # lazy slice -> structured array

            if stride > 1:
                gidx = np.arange(start, end)
                keep = (gidx % stride) == 0
                if not keep.any():
                    continue
                block = block[keep]

            n = block.shape[0]
            if n == 0:
                continue

            # pack kept columns into a contiguous [n, D] float32 array, in the
            # exact order of out_props
            cols = []
            for p in out_props:
                if p in ('nx', 'ny', 'nz') and not have_normals:
                    cols.append(np.zeros(n, dtype=np.float32))
                else:
                    cols.append(block[p].astype(np.float32))
            rows = np.ascontiguousarray(np.stack(cols, axis=1), dtype='<f4')
            fout.write(rows.tobytes())

            written += n
            print(f'\r[slim] {written:,} points written...', end='', flush=True)

    if written != kept_total:
        # fix the header if the actual count drifted (shouldn't normally happen)
        print(f'\n[slim] WARNING: wrote {written:,} but header said '
              f'{kept_total:,}; rewriting header.')
        _rewrite_header_count(output_path, out_props, written)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'\n[slim] done: {written:,} points, {len(out_props)} props/point '
          f'-> {output_path} ({size_mb:.1f} MiB)')
    return written, output_path


def _rewrite_header_count(path, out_props, count):
    """Rewrite only the vertex-count line in place (same byte length kept by
    padding is not needed because we rewrite the whole header, which has a
    fixed structure independent of count digits only if lengths match; to be
    safe we just regenerate the file prefix). For simplicity and safety we do
    a full header-prefix replacement by streaming the body after end_header."""
    with open(path, 'rb') as f:
        data = f.read(4096)
    idx = data.find(b'end_header\n')
    if idx < 0:
        print('[slim] could not locate header to fix count; leaving as-is.')
        return
    header_len = idx + len(b'end_header\n')
    # read body, rewrite with corrected header
    with open(path, 'rb') as f:
        f.seek(header_len)
        body = f.read()
    with open(path, 'wb') as f:
        _write_header(f, out_props, count)
        f.write(body)


def main():
    ap = argparse.ArgumentParser(
        description='Stream-slim a 3DGS .ply by dropping higher-order SH '
                    '(f_rest_*), keeping a standard viewer-compatible .ply.')
    ap.add_argument('input', help='input .ply (e.g. scene_merged.ply)')
    ap.add_argument('-o', '--output', default=None,
                    help='output .ply (default: input name + "_slim.ply")')
    ap.add_argument('--chunk', type=int, default=1_000_000,
                    help='points per chunk (default 1,000,000). Lower if RAM '
                         'is tight.')
    ap.add_argument('--max-points', type=int, default=0,
                    help='if >0 and larger, uniformly keep at most this many '
                         'points (stride subsample).')
    args = ap.parse_args()

    out = args.output
    if out is None:
        base = os.path.splitext(args.input)[0]
        out = base + '_slim.ply'

    slim(args.input, out, chunk=args.chunk,
         max_points=(args.max_points or None))


if __name__ == '__main__':
    main()
