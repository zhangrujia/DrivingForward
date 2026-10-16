"""
Generate split token lists for the nuScenes v1.0-mini dataset.

The original eval_SF.txt / eval_MF.txt / train.txt contain sample tokens from
v1.0-trainval, which do NOT exist in the mini database. This script rebuilds
those split files using only mini samples.

SF / MF modes require a forward (next) frame for every camera
(see NuScenesdataset.get_context / get_cam_T_cam, which call
dataset.get('sample_data', cam_sample['next'])). Samples that are the last
keyframe of a scene have an empty 'next' and would crash, so they are excluded.

Usage:
    python dataset/gen_mini_split.py \
        --dataroot ./input_data/nuscenes/ \
        --version v1.0-mini \
        --splits eval_SF eval_MF train
"""
import argparse
import os

from nuscenes.nuscenes import NuScenes

CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
           'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']


def has_valid_next(nusc, sample):
    """Return True if every camera in this sample has a non-empty 'next'."""
    for cam in CAMERAS:
        sd_token = sample['data'].get(cam)
        if not sd_token:
            return False
        sd = nusc.get('sample_data', sd_token)
        if not sd['next']:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='./input_data/nuscenes/')
    parser.add_argument('--version', default='v1.0-mini')
    parser.add_argument('--splits', nargs='+',
                        default=['eval_SF', 'eval_MF', 'train'])
    parser.add_argument('--out_dir', default='dataset/nuscenes')
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # keep only samples that have a valid forward frame for every camera
    valid_tokens = [s['token'] for s in nusc.sample if has_valid_next(nusc, s)]
    print(f'total mini samples: {len(nusc.sample)}, '
          f'usable (with next): {len(valid_tokens)}')

    os.makedirs(args.out_dir, exist_ok=True)
    for split in args.splits:
        out_path = os.path.join(args.out_dir, f'{split}.txt')
        # back up the original trainval split once
        if os.path.exists(out_path) and not os.path.exists(out_path + '.trainval'):
            os.rename(out_path, out_path + '.trainval')
            print(f'backed up original -> {out_path}.trainval')
        with open(out_path, 'w') as f:
            f.write('\n'.join(valid_tokens) + '\n')
        print(f'wrote {len(valid_tokens)} tokens -> {out_path}')


if __name__ == '__main__':
    main()
