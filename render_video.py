"""
Scheme A: render a time range of a nuScenes scene into per-camera videos.

Given a start/end CAM_FRONT image name (or timestamps) of a single scene, this
script collects the continuous key-frames in between, runs the existing
DrivingForward feed-forward reconstruction (SF or MF) on every frame using each
frame's *original* camera viewpoint, and encodes the rendered frames into 6
separate videos (one per camera: CAM_FRONT.mp4, CAM_FRONT_LEFT.mp4, ...).

It does NOT modify the model or the existing datasets. It builds a small
temporary split file listing the in-range tokens in temporal order, then reuses
NuScenesdataset + DrivingForwardModel exactly as eval.py would.

MF note: MF dereferences each camera's prev AND next frame, so a scene's first
frame (no prev) / last frame (no next) cannot be reconstructed in MF and are
skipped automatically (with a printed notice). SF only needs a next frame.

Usage (defaults reproduce the requested 1532402927612460 -> 1532402946762460
range of scene n015-2018-07-24-11-22-45+0800):

    python -W ignore render_video.py \
        --weight_path ./weights_MF --novel_view_mode MF \
        --dataroot /media/zhangrujia/Call17812007125/3DGS/Datasets/v1.0-mini \
        --start_ts 1532402927612460 --end_ts 1532402946762460 \
        --fps 6 --out_dir ./results/video_MF

You can also pass image names instead of timestamps:
    --start_image n015-...__CAM_FRONT__1532402927612460.jpg
    --end_image   n015-...__CAM_FRONT__1532402946762460.jpg
"""
import argparse
import os
import re

CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
           'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']

# split file name NuScenesdataset will read from dataset/nuscenes/<split>.txt
_TMP_SPLIT = 'video_tmp'
_SPLIT_DIR = 'dataset/nuscenes'


def _ts_from_image_name(name):
    """Extract the trailing timestamp integer from a CAM_FRONT image name."""
    m = re.search(r'__(\d+)\.(?:jpg|png)$', name)
    if not m:
        raise ValueError(f'Cannot parse timestamp from image name: {name}')
    return int(m.group(1))


def collect_range_tokens(nusc, start_ts, end_ts, novel_view_mode):
    """Return in-range sample tokens (sorted by CAM_FRONT timestamp).

    Only key-frames of the scene whose CAM_FRONT timestamp falls in
    [start_ts, end_ts] are returned. For MF, frames missing prev/next are
    dropped; for SF, frames missing next are dropped. The scene is identified
    by whichever sample owns the start_ts CAM_FRONT key-frame.
    """
    # map CAM_FRONT key-frame timestamp -> sample token, and remember scene
    rows = []
    start_scene = None
    for s in nusc.sample:
        cf = nusc.get('sample_data', s['data']['CAM_FRONT'])
        ts = cf['timestamp']
        rows.append((ts, s['token'], s['scene_token'], s['prev'], s['next']))
        if ts == start_ts:
            start_scene = s['scene_token']

    if start_scene is None:
        # fall back: pick the scene of the sample closest to start_ts
        closest = min(rows, key=lambda r: abs(r[0] - start_ts))
        start_scene = closest[2]
        print(f'[warn] exact start_ts not found; using scene of closest '
              f'frame ts={closest[0]}')

    in_range = [r for r in rows
                if r[2] == start_scene and start_ts <= r[0] <= end_ts]
    in_range.sort(key=lambda r: r[0])

    kept, skipped = [], []
    for ts, tok, _scene, prev, nxt in in_range:
        if novel_view_mode == 'MF' and (not prev or not nxt):
            skipped.append((ts, tok, 'MF needs prev+next'))
            continue
        if novel_view_mode == 'SF' and not nxt:
            skipped.append((ts, tok, 'SF needs next'))
            continue
        kept.append(tok)

    print(f'In-range key-frames: {len(in_range)}, '
          f'usable ({novel_view_mode}): {len(kept)}, skipped: {len(skipped)}')
    for ts, tok, why in skipped:
        print(f'  [skip] ts={ts} token={tok} ({why})')
    return kept


def collect_range_tokens_json(dataroot, version, start_ts, end_ts,
                              novel_view_mode):
    """Same selection as collect_range_tokens, but parses nuScenes json tables
    directly (no nuscenes package required). Used only for --dry_run."""
    import json
    meta = os.path.join(dataroot, version)
    sd = json.load(open(os.path.join(meta, 'sample_data.json')))
    samples = json.load(open(os.path.join(meta, 'sample.json')))
    cs = json.load(open(os.path.join(meta, 'calibrated_sensor.json')))
    sensor = json.load(open(os.path.join(meta, 'sensor.json')))
    cs2sensor = {c['token']: c['sensor_token'] for c in cs}
    sensor2chan = {s['token']: s['channel'] for s in sensor}

    front_ts = {}
    for d in sd:
        if d.get('is_key_frame') and \
           sensor2chan.get(cs2sensor.get(d['calibrated_sensor_token'])) == 'CAM_FRONT':
            front_ts[d['sample_token']] = d['timestamp']

    rows, start_scene = [], None
    for s in samples:
        ts = front_ts.get(s['token'])
        if ts is None:
            continue
        rows.append((ts, s['token'], s['scene_token'], s['prev'], s['next']))
        if ts == start_ts:
            start_scene = s['scene_token']
    if start_scene is None:
        closest = min(rows, key=lambda r: abs(r[0] - start_ts))
        start_scene = closest[2]
        print(f'[warn] exact start_ts not found; using scene of closest '
              f'frame ts={closest[0]}')

    in_range = [r for r in rows
                if r[2] == start_scene and start_ts <= r[0] <= end_ts]
    in_range.sort(key=lambda r: r[0])

    kept, skipped = [], []
    for ts, tok, _scene, prev, nxt in in_range:
        if novel_view_mode == 'MF' and (not prev or not nxt):
            skipped.append((ts, tok, 'MF needs prev+next'))
            continue
        if novel_view_mode == 'SF' and not nxt:
            skipped.append((ts, tok, 'SF needs next'))
            continue
        kept.append(tok)

    print(f'In-range key-frames: {len(in_range)}, '
          f'usable ({novel_view_mode}): {len(kept)}, skipped: {len(skipped)}')
    for ts, tok, why in skipped:
        print(f'  [skip] ts={ts} token={tok} ({why})')
    return kept


def write_tmp_split(tokens):
    os.makedirs(_SPLIT_DIR, exist_ok=True)
    path = os.path.join(_SPLIT_DIR, f'{_TMP_SPLIT}.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(tokens) + '\n')
    return path


def build_eval_dataset(cfg, split_name):
    """Construct a NuScenesdataset bound to our temporary split, mirroring
    DrivingForwardModel.set_eval_dataloader's augmentation settings."""
    from external.dataset import get_transforms
    from dataset.nuscenes_dataset import NuScenesdataset

    _aug = {
        'image_shape': (int(cfg['training']['height']),
                        int(cfg['training']['width'])),
        'jittering': (0.0, 0.0, 0.0, 0.0),
        'crop_train_borders': (),
        'crop_eval_borders': (),
    }
    req = cfg['data']['val_requirements']
    dataset = NuScenesdataset(
        cfg['data']['data_path'], split_name,
        cameras=cfg['data']['cameras'],
        back_context=cfg['data']['back_context'],
        forward_context=cfg['data']['forward_context'],
        data_transform=get_transforms('train', **_aug),
        depth_type=cfg['data']['depth_type'] if 'gt_depth' in req else None,
        with_pose='gt_pose' in req,
        with_ego_pose='gt_ego_pose' in req,
        with_mask='mask' in req,
        version=cfg['data'].get('version', 'v1.0-mini'),
    )
    return dataset


def to_uint8_hwc(img_chw):
    """[C,H,W] float in [0,1] (torch) -> [H,W,C] uint8 numpy."""
    import numpy as np
    arr = img_chw.detach().clip(0, 1).mul(255).round().byte().cpu().numpy()
    return np.transpose(arr, (1, 2, 0))


def main():
    parser = argparse.ArgumentParser(description='render a scene time range '
                                      'into per-camera videos (Scheme A)')
    parser.add_argument('--config_file', default='./configs/nuscenes/main.yaml')
    parser.add_argument('--weight_path', default='./weights_MF')
    parser.add_argument('--novel_view_mode', default='MF', choices=['MF', 'SF'])
    parser.add_argument('--dataroot', default=None,
                        help='override cfg data_path (nuScenes dataroot)')
    parser.add_argument('--version', default=None,
                        help='override cfg nuScenes version (e.g. v1.0-mini)')
    parser.add_argument('--start_ts', type=int, default=1532402927612460)
    parser.add_argument('--end_ts', type=int, default=1532402946762460)
    parser.add_argument('--start_image', default=None,
                        help='CAM_FRONT image name; overrides --start_ts')
    parser.add_argument('--end_image', default=None,
                        help='CAM_FRONT image name; overrides --end_ts')
    parser.add_argument('--fps', type=int, default=6)
    parser.add_argument('--out_dir', default=None,
                        help='output dir (default ./results/video_<mode>)')
    parser.add_argument('--dry_run', action='store_true',
                        help='only enumerate in-range tokens and write the '
                             'temporary split; do NOT load the model, run '
                             'inference, or encode videos. Useful to sanity '
                             'check the frame selection without a GPU env.')
    args = parser.parse_args()

    if args.start_image:
        args.start_ts = _ts_from_image_name(args.start_image)
    if args.end_image:
        args.end_ts = _ts_from_image_name(args.end_image)
    assert args.start_ts <= args.end_ts, 'start_ts must be <= end_ts'

    # ---- resolve dataroot / version ----------------------------------------
    # For a real run we build the full cfg via utils.get_config (which imports
    # torch). For --dry_run we only need dataroot/version/cameras, so we read
    # the yaml directly and avoid pulling in torch / the model stack.
    if args.dry_run:
        import yaml
        with open(args.config_file, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg['model']['novel_view_mode'] = args.novel_view_mode
    else:
        import utils
        weight_path = os.path.join('./pretrained',
                                   os.path.normpath(args.weight_path))
        cfg = utils.get_config(args.config_file, mode='eval',
                               weight_path=weight_path,
                               novel_view_mode=args.novel_view_mode)
    if args.dataroot:
        cfg['data']['data_path'] = args.dataroot
    if args.version:
        cfg['data']['version'] = args.version

    out_dir = args.out_dir or f'./results/video_{args.novel_view_mode}'
    os.makedirs(out_dir, exist_ok=True)

    # ---- enumerate in-range tokens (temporal order) ------------------------
    # Prefer the nuScenes API (matches the rest of the project). If the
    # nuscenes package is unavailable (e.g. a lightweight dry-run shell), fall
    # back to parsing the raw json tables directly.
    try:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version=cfg['data'].get('version', 'v1.0-mini'),
                        dataroot=cfg['data']['data_path'], verbose=False)
        tokens = collect_range_tokens(nusc, args.start_ts, args.end_ts,
                                      args.novel_view_mode)
    except ImportError:
        if not args.dry_run:
            raise
        print('[dry-run] nuscenes package not found; parsing json tables.')
        tokens = collect_range_tokens_json(
            cfg['data']['data_path'], cfg['data'].get('version', 'v1.0-mini'),
            args.start_ts, args.end_ts, args.novel_view_mode)
    if not tokens:
        raise SystemExit('No usable frames in the requested range.')
    split_path = write_tmp_split(tokens)
    print(f'temporary split -> {split_path} ({len(tokens)} frames)')

    if args.dry_run:
        print('\n[dry-run] frame selection OK. Temporal token order:')
        for i, tok in enumerate(tokens):
            print(f'  #{i:02d} {tok}')
        print(f'\n[dry-run] would render {len(tokens)} frames x '
              f'{len(CAMERAS)} cameras -> {len(CAMERAS)} videos '
              f'@ {args.fps}fps into {out_dir}')
        print('[dry-run] skipping model load / inference / encoding.')
        return

    # ---- build model and dataset ------------------------------------------
    import torch
    from tqdm import tqdm
    from models import DrivingForwardModel
    model = DrivingForwardModel(cfg, 0)
    model.load_weights()
    model.set_eval()

    dataset = build_eval_dataset(cfg, _TMP_SPLIT)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=cfg['eval'].get('eval_num_workers', 0), pin_memory=True)

    # MF renders the current-frame viewpoint (gaussian_color,0,0);
    # SF renders the next-frame novel view (gaussian_color,1,0).
    render_fid = 0 if args.novel_view_mode == 'MF' else 1

    # per-camera frame buffers, in temporal order
    frames = {cam: [] for cam in CAMERAS}

    with torch.no_grad():
        for inputs in tqdm(loader, desc=f'rendering {args.novel_view_mode}'):
            outputs, _ = model.process_batch(inputs, 0)
            for ci, cam in enumerate(CAMERAS):
                img = outputs[('cam', ci)][('gaussian_color', render_fid, 0)][0]
                frames[cam].append(to_uint8_hwc(img))

    # ---- encode one video per camera --------------------------------------
    import imageio
    try:
        import imageio_ffmpeg  # noqa: F401  (ensures the mp4 writer backend)
    except ImportError:
        raise SystemExit(
            'imageio-ffmpeg is required to write mp4 files.\n'
            '  pip install imageio-ffmpeg\n'
            'The rendered frames were produced successfully; only the video '
            'encoding step needs this package.')
    for cam in CAMERAS:
        seq = frames[cam]
        if not seq:
            print(f'[warn] no frames for {cam}, skipping')
            continue
        out_path = os.path.join(out_dir, f'{cam}.mp4')
        imageio.mimwrite(out_path, seq, fps=args.fps,
                         quality=8, macro_block_size=1)
        print(f'wrote {len(seq)} frames @ {args.fps}fps -> {out_path}')

    print('\nDone. Per-camera videos are in:', out_dir)


if __name__ == '__main__':
    main()
