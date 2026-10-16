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
    cfg = utils.get_config(args.config_file, mode='eval', weight_path=weight_path, novel_view_mode=args.novel_view_mode,
                           debug=args.debug, debug_samples=args.debug_samples)
        
    test(cfg)
