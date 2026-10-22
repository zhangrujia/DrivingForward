import argparse 
import os
 
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
    
import utils
from models import DrivingForwardModel
from trainer import DrivingForwardTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='evaluation script')
    parser.add_argument('--config_file', default='./configs/nuscenes/main.yaml', type=str, help='config yaml file path')
    parser.add_argument('--weight_path', default='./weights', type=str, help='weight path')
    parser.add_argument('--novel_view_mode', default='MF', type=str, help='MF of SF')
    parser.add_argument('--debug', action='store_true',
                        help='debug mode: only run a few samples to check the pipeline')
    parser.add_argument('--debug_samples', default=3, type=int,
                        help='number of samples to run in debug mode (default: 3)')
    parser.add_argument('--scene', default=None, type=str,
                        help='select a single capture video (nuScenes scene) to run inference on. '
                             'Accepts a scene index (0-based, e.g. 0..9 for v1.0-mini) or a scene '
                             'name (e.g. scene-0061). Default: None = run all scenes.')
    parser.add_argument('--list_scenes', action='store_true',
                        help='print the available capture videos (scenes) with their index / name / '
                             'frame count, then exit without running inference.')
    parser.add_argument('--make_video', action='store_true',
                        help='after inference, encode one side-by-side mp4 per camera from the saved '
                             'images (left = ground-truth, right = prediction). Requires save_images '
                             '(enabled by default in the eval config).')
    parser.add_argument('--video_fps', default=6, type=int,
                        help='frames per second for the generated per-camera videos (default: 6).')
    parser.add_argument('--export_ply_cams', default='all', type=str,
                        help='which cameras to include in the exported .ply. "all" (default) exports '
                             'the full 6-camera surround view. Otherwise pass a comma-separated list '
                             'of camera indices (0-based) or camera names, e.g. "0", "0,5", '
                             '"CAM_FRONT" or "CAM_FRONT,CAM_BACK".')
    parser.add_argument('--merge_scene_ply', action='store_true',
                        help='after inference, merge every evaluated frame into a SINGLE global .ply '
                             'for the whole scene: each frame\'s Gaussians are transformed from its ego '
                             'frame into a shared world frame via the nuScenes ego_pose, then '
                             'concatenated. Use together with --scene (merging across different scenes '
                             'is not geometrically meaningful). Respects --export_ply_cams.')
    parser.add_argument('--merge_voxel', default=0.1, type=float,
                        help='voxel size (meters) for downsampling the merged global .ply; <=0 disables '
                             '(default: 0.1).')
    args = parser.parse_args() 
    return args

def test(cfg):
    print("Evaluating reconstruction")
    model = DrivingForwardModel(cfg, 0)
    trainer = DrivingForwardTrainer(cfg, 0, use_tb = False)
    trainer.evaluate(model)

if __name__ == '__main__':
    args = parse_args()
    weight_path = os.path.join('./pretrained', os.path.normpath(args.weight_path))

    # just list the available capture videos (scenes) and exit
    if args.list_scenes:
        from dataset.list_scenes import print_scenes
        print_scenes(args.config_file)
        raise SystemExit(0)

    cfg = utils.get_config(args.config_file, mode='eval', weight_path=weight_path, novel_view_mode=args.novel_view_mode,
                           debug=args.debug, debug_samples=args.debug_samples, scene=args.scene,
                           make_video=args.make_video, video_fps=args.video_fps,
                           export_ply_cams=args.export_ply_cams,
                           merge_scene_ply=args.merge_scene_ply, merge_voxel=args.merge_voxel)
        
    test(cfg)
