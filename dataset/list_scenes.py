"""
List the available capture videos (nuScenes scenes) for the configured dataset.

Used by `eval.py --list_scenes` so the user can discover which scene index /
name to pass to `eval.py --scene ...`.

Besides the raw scene list, it also reports, per scene, how many frames are
actually usable in the SF / MF eval splits (the split files exclude frames
that are missing the required prev/next neighbours).
"""
import os

import yaml
from nuscenes.nuscenes import NuScenes


def _load_split_tokens(split):
    path = os.path.join('dataset', 'nuscenes', '{}.txt'.format(split))
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return set(l.strip() for l in f if l.strip())


def print_scenes(config_file='./configs/nuscenes/main.yaml'):
    with open(config_file, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)

    data_path = cfg['data']['data_path']
    version = cfg['data'].get('version', 'v1.0-mini')

    nusc = NuScenes(version=version, dataroot=data_path, verbose=False)

    sf_tokens = _load_split_tokens('eval_SF')
    mf_tokens = _load_split_tokens('eval_MF')

    print('=' * 78)
    print('Available capture videos (scenes) in {} @ {}'.format(version, data_path))
    print('=' * 78)
    header = '{:>5}  {:<14} {:>6}  {:>8}  {:>8}  {}'.format(
        'index', 'name', 'frames', 'SF-usable', 'MF-usable', 'description')
    print(header)
    print('-' * 78)
    for idx, sc in enumerate(nusc.scene):
        # count usable frames per split by walking the scene's sample list
        sf_n = mf_n = 0
        cur = sc['first_sample_token']
        while cur:
            if sf_tokens is not None and cur in sf_tokens:
                sf_n += 1
            if mf_tokens is not None and cur in mf_tokens:
                mf_n += 1
            cur = nusc.get('sample', cur)['next']
        sf_disp = sf_n if sf_tokens is not None else '-'
        mf_disp = mf_n if mf_tokens is not None else '-'
        desc = sc['description'][:30]
        print('{:>5}  {:<14} {:>6}  {:>8}  {:>8}  {}'.format(
            idx, sc['name'], sc['nbr_samples'], sf_disp, mf_disp, desc))
    print('=' * 78)
    print('Run a single scene with:  '
          'python -W ignore eval.py --weight_path ./weights_MF '
          '--novel_view_mode MF --scene <index|name>')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='./configs/nuscenes/main.yaml')
    args = parser.parse_args()
    print_scenes(args.config_file)
