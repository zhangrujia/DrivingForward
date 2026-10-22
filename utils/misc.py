import os
import yaml 
from collections import defaultdict
 
import torch

_NUSC_CAM_LIST = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
_REL_CAM_DICT = {0: [1,2], 1: [0,3], 2: [0,4], 3: [1,5], 4: [2,5], 5: [3,4]}


def camera2ind(cameras):
    """
    This function transforms camera name list to indices 
    """    
    indices = []
    for cam in cameras:
        if cam in _NUSC_CAM_LIST:
            ind = _NUSC_CAM_LIST.index(cam)
        else:
            ind = None
        indices.append(ind)
    return indices


def get_relcam(cameras):
    """
    This function returns relative camera indices from given camera list
    """
    relcam_dict = defaultdict(list)
    indices = camera2ind(cameras)
    for ind in indices:
        relcam_dict[ind] = []
        # Unknown cameras (not in the nuScenes layout, e.g. custom COLMAP
        # captures) map to None and have no predefined spatial neighbours.
        if ind is None or ind not in _REL_CAM_DICT:
            continue
        relcam_cand = _REL_CAM_DICT[ind]
        for cand in relcam_cand:
            if cand in indices:
                relcam_dict[ind].append(cand)
    return relcam_dict        
    

def parse_export_ply_cams(spec, cameras):
    """Parse the --export_ply_cams selector into a sorted list of camera indices.

    Args:
        spec: 'all' (or '' / None) for every camera, otherwise a comma-separated
              list of camera indices (0-based) and/or camera names
              (e.g. '0,5' or 'CAM_FRONT,CAM_BACK').
        cameras: the ordered camera-name list from cfg['data']['cameras'].

    Returns:
        None if all cameras are selected (default), otherwise a sorted list of
        unique integer indices into `cameras`.
    """
    if spec is None:
        return None
    spec = str(spec).strip()
    if spec == '' or spec.lower() == 'all':
        return None

    indices = []
    for tok in spec.split(','):
        tok = tok.strip()
        if tok == '':
            continue
        if tok.lstrip('-').isdigit():
            idx = int(tok)
            if idx < 0 or idx >= len(cameras):
                raise ValueError(
                    'export_ply_cams index {} out of range (valid 0..{} for '
                    'cameras {})'.format(idx, len(cameras) - 1, cameras))
            indices.append(idx)
        elif tok in cameras:
            indices.append(cameras.index(tok))
        else:
            raise ValueError(
                'export_ply_cams: "{}" is neither a valid index (0..{}) nor a '
                'known camera name {}'.format(tok, len(cameras) - 1, cameras))
    if not indices:
        return None
    return sorted(set(indices))


def get_config(config, mode='train', weight_path='./', novel_view_mode='MF',
               debug=False, debug_samples=3, scene=None,
               make_video=False, video_fps=6, export_ply_cams='all',
               merge_scene_ply=False, merge_voxel=0.1):
    """
    This function reads the configuration file and return as dictionary
    """
    with open(config, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)

        cfg_name = os.path.splitext(os.path.basename(config))[0]
        print('Experiment: ', cfg_name)

        _log_path = os.path.join(cfg['data']['log_dir'], cfg_name)
        cfg['data']['log_path'] = _log_path
        cfg['data']['save_weights_root'] = os.path.join(_log_path, 'models')
        cfg['data']['num_cams'] = len(cfg['data']['cameras'])
        cfg['data']['rel_cam_list'] = get_relcam(cfg['data']['cameras'])

        cfg['model']['mode'] = mode
        cfg['model']['novel_view_mode'] = novel_view_mode

        cfg['load']['load_weights_dir'] = weight_path

        # which capture video (scene) to run; None = all scenes (default)
        cfg['data']['scene'] = scene
            
        if mode == 'eval':
            cfg['ddp']['world_size'] = 1
            cfg['ddp']['gpus'] = [0]
            cfg['training']['batch_size'] = cfg['eval']['eval_batch_size']
            cfg['training']['depth_flip'] = False
            cfg['eval']['debug'] = debug
            cfg['eval']['debug_samples'] = debug_samples
            cfg['eval']['make_video'] = make_video
            cfg['eval']['video_fps'] = video_fps

            # which cameras to include in the exported .ply.
            # None = all 6 cameras (default full surround view).
            ply_cam_indices = parse_export_ply_cams(export_ply_cams,
                                                     cfg['data']['cameras'])
            cfg['eval']['export_ply_cams'] = ply_cam_indices
            cfg['eval']['merge_scene_ply'] = merge_scene_ply
            cfg['eval']['merge_voxel'] = merge_voxel

            # separate output dirs per novel-view mode (SF/MF), scene, and debug
            # run, so results from different runs don't mix / overwrite each other.
            suffix = '_' + novel_view_mode
            if scene is not None:
                # sanitize the scene selector so it is filesystem-safe
                safe_scene = str(scene).replace('/', '_').replace(' ', '_')
                suffix += '_scene' + safe_scene
            if debug:
                suffix += '_debug'
            if 'save_path' in cfg['eval']:
                cfg['eval']['save_path'] = cfg['eval']['save_path'] + suffix
            if 'export_ply_path' in cfg['eval']:
                ply_suffix = suffix
                # when exporting a subset of cameras, tag the ply dir so partial
                # exports don't overwrite the full-surround ones.
                if ply_cam_indices is not None:
                    ply_suffix += '_cam' + '-'.join(str(c) for c in ply_cam_indices)
                cfg['eval']['export_ply_path'] = cfg['eval']['export_ply_path'] + ply_suffix
    return cfg

    
def pretty_ts(ts):
    """
    This function prints amount of time taken in user friendly way.
    """
    second = int(ts)
    minute = second // 60
    hour = minute // 60
    return f'{hour:02d}h{(minute%60):02d}m{(second%60):02d}s'


def cal_depth_error(pred, target):
    """
    This function calculates depth error using various metrics.
    """
    abs_rel = torch.mean(torch.abs(pred-target) / target)
    sq_rel = torch.mean((pred-target).pow(2) / target)
    rmse = torch.sqrt(torch.mean((pred-target).pow(2)))
    rmse_log = torch.sqrt(torch.mean((torch.log(target) - torch.log(pred)).pow(2)))

    thresh = torch.max((target/pred), (pred/ target))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25**2).float().mean()
    a3 = (thresh < 1.25**3).float().mean()
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3