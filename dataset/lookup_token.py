"""
Look up the original input data for a given nuScenes sample token.

Prints: which scene / frame position the sample belongs to, and the 6
surround-view image file paths (the exact images fed to the model), plus the
prev/next sample tokens so you can locate the temporal neighbours.

Usage:
    python dataset/lookup_token.py 356d81f38dd9473ba590f39e266f54e5
    python dataset/lookup_token.py <token> --version v1.0-mini --dataroot ./input_data/nuscenes/
"""
import argparse
import os

from nuscenes.nuscenes import NuScenes

CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
           'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('token', type=str, help='sample token to look up')
    parser.add_argument('--dataroot', default='./input_data/nuscenes/')
    parser.add_argument('--version', default='v1.0-mini')
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    sample = nusc.get('sample', args.token)
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])

    # frame index within the scene
    frame_idx = 0
    cur = scene['first_sample_token']
    while cur and cur != args.token:
        cur = nusc.get('sample', cur)['next']
        frame_idx += 1

    print('=' * 70)
    print(f'token         : {args.token}')
    print(f'scene         : {scene["name"]}  (token={sample["scene_token"]})')
    print(f'scene desc    : {scene["description"]}')
    print(f'location      : {log["location"]}   date: {log["date_captured"]}')
    print(f'frame in scene: #{frame_idx} / {scene["nbr_samples"]-1} (0-based)')
    print(f'timestamp     : {sample["timestamp"]}')
    print(f'prev sample   : {sample["prev"] or "(none - first frame)"}')
    print(f'next sample   : {sample["next"] or "(none - last frame)"}')
    print('-' * 70)
    print('6 surround-view input images (relative to dataroot):')
    for cam in CAMERAS:
        sd = nusc.get('sample_data', sample['data'][cam])
        full = os.path.join(args.dataroot, sd['filename'])
        exists = 'OK' if os.path.exists(full) else 'MISSING'
        print(f'  {cam:16s} [{exists}] {sd["filename"]}')
    print('=' * 70)


if __name__ == '__main__':
    main()
