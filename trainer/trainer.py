import time
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch import Tensor

from utils import Logger

from lpips import LPIPS
from jaxtyping import Float, UInt8
from skimage.metrics import structural_similarity
from einops import reduce

from PIL import Image
from pathlib import Path
from einops import rearrange, repeat
from typing import Union
import numpy as np

from tqdm import tqdm

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]

class DrivingForwardTrainer:
    """
    Trainer class for training and evaluation
    """
    def __init__(self, cfg, rank, use_tb=True):
        self.read_config(cfg)
        self.rank = rank        
        if rank == 0:
            self.logger = Logger(cfg, use_tb)
            self.depth_metric_names = self.logger.get_metric_names()

        self.lpips = LPIPS(net="vgg").cuda(rank)

    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def learn(self, model):
        """
        This function sets training process.
        """        
        train_dataloader = model.train_dataloader()
        if self.rank == 0:
            val_dataloader = model.val_dataloader()
            self.val_iter = iter(val_dataloader)
        
        self.step = 0
        start_time = time.time()
        for self.epoch in range(self.num_epochs):
                
            self.train(model, train_dataloader, start_time)
            
            # save model after each epoch using rank 0 gpu 
            if self.rank == 0:
                model.save_model(self.epoch)
                print('-'*110) 
                
            if self.ddp_enable:
                dist.barrier()
                
        if self.rank == 0:
            self.logger.close_tb()
        
    def train(self, model, data_loader, start_time):
        """
        This function trains models.
        """
        model.set_train()
        pbar = tqdm(total=len(data_loader), desc='training on epoch {}'.format(self.epoch), mininterval=100)
        for batch_idx, inputs in enumerate(data_loader):         
            before_op_time = time.time()
            model.optimizer.zero_grad(set_to_none=True)
            outputs, losses = model.process_batch(inputs, self.rank)
            losses['total_loss'].backward()
            model.optimizer.step()

            if self.rank == 0: 
                self.logger.update(
                    'train', 
                    self.epoch, 
                    self.world_size,
                    batch_idx, 
                    self.step,
                    start_time,
                    before_op_time, 
                    inputs,
                    outputs,
                    losses
                )

                if self.logger.is_checkpoint(self.step):
                    self.validate(model)

            self.step += 1
            pbar.update(1)

        pbar.close()
        model.lr_scheduler.step()
        
    @torch.no_grad()
    def validate(self, model, vis_results=False):
        """
        This function validates models on the validation dataset to monitor training process.
        """
        val_dataloader = model.val_dataloader()
        val_iter = iter(val_dataloader)
        
        # Ensure the model is in validation mode
        model.set_val()

        avg_reconstruction_metric = defaultdict(float)

        inputs = next(val_iter)
        outputs, _ = model.process_batch(inputs, self.rank)
            
        psnr, ssim, lpips= self.compute_reconstruction_metrics(inputs, outputs)

        avg_reconstruction_metric['psnr'] += psnr   
        avg_reconstruction_metric['ssim'] += ssim
        avg_reconstruction_metric['lpips'] += lpips

        print('Validation reconstruction result...\n')
        print(f"\n{inputs['token'][0]}")
        self.logger.print_perf(avg_reconstruction_metric, 'reconstruction')

        # Set the model back to training mode
        model.set_train()

    @torch.no_grad()
    def evaluate(self, model):
        """
        This function evaluates models on validation dataset of samples with context.
        """
        eval_dataloader = model.eval_dataloader()

        # load model
        model.load_weights()
        model.set_eval()

        avg_reconstruction_metric = defaultdict(float)

        count = 0
        eval_tokens = []  # processed sample tokens, in evaluation (temporal) order

        # optional: accumulate every frame's Gaussians (transformed into a
        # shared world frame) to write ONE merged global .ply for the scene.
        merge_ply = getattr(self, 'merge_scene_ply', False)
        merge_buf = {'xyz': [], 'rot': [], 'scale': [], 'opacity': [], 'sh': []}
        merge_nusc = None
        if merge_ply:
            try:
                merge_nusc = eval_dataloader.dataset.dataset  # NuScenes handle
            except AttributeError:
                print('[merge_ply] dataset has no nuScenes handle; '
                      'disabling merge.')
                merge_ply = False
            if getattr(self, 'scene', None) is None:
                print('[merge_ply] WARNING: --scene not set. Merging frames '
                      'across different scenes shares one world frame only '
                      'within a scene; cross-scene clouds will not align.')

        process = tqdm(eval_dataloader)
        debug = getattr(self, 'debug', False)
        debug_samples = getattr(self, 'debug_samples', 3)
        if debug:
            print(f'[DEBUG] debug mode on: running only {debug_samples} sample(s)')
        for batch_idx, inputs in enumerate(process):
            outputs, _ = model.process_batch(inputs, self.rank)
            
            psnr, ssim, lpips= self.compute_reconstruction_metrics(inputs, outputs)

            avg_reconstruction_metric['psnr'] += psnr   
            avg_reconstruction_metric['ssim'] += ssim
            avg_reconstruction_metric['lpips'] += lpips
            count += 1
            eval_tokens.append(inputs['token'][0])

            if merge_ply:
                self._accumulate_scene_ply(inputs, outputs, merge_nusc, merge_buf)

            process.set_description(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, LPIPS: {lpips:.4f}")

            print(f"\n{inputs['token'][0]}")
            print(f"avg PSNR: {avg_reconstruction_metric['psnr']/count:.4f}, avg SSIM: {avg_reconstruction_metric['ssim']/count:.4f}, avg LPIPS: {avg_reconstruction_metric['lpips']/count:.4f}")

            if debug and count >= debug_samples:
                print(f'[DEBUG] stopped after {count} sample(s)')
                break

        # average over the number of samples actually processed
        divisor = count if debug else len(eval_dataloader)
        avg_reconstruction_metric['psnr'] /= divisor
        avg_reconstruction_metric['ssim'] /= divisor
        avg_reconstruction_metric['lpips'] /= divisor

        print('Evaluation reconstruction result...\n')
        self.logger.print_perf(avg_reconstruction_metric, 'reconstruction')

        # optionally encode per-camera side-by-side (GT | prediction) videos
        # from the images just saved during evaluation.
        if getattr(self, 'make_video', False):
            self._make_eval_videos(eval_tokens)

        # optionally write ONE merged global .ply for the whole scene.
        if merge_ply:
            self._write_merged_scene_ply(merge_buf)

    def _accumulate_scene_ply(self, inputs, outputs, nusc, buf):
        """Collect this frame's selected-camera Gaussians, transform them from
        the ego frame into the shared world frame (via the nuScenes ego_pose),
        and append to the merge buffer (kept on CPU to bound GPU memory)."""
        import torch
        import numpy as np
        from pyquaternion import Quaternion
        from models.gaussian.GaussianRender import _collect_valid_gaussians
        from models.gaussian.transform_util import (transform_gaussians_to_world,
                                                    voxel_downsample)

        num_cams = self.num_cams
        sel_cams = getattr(self, 'export_ply_cams', None)
        if sel_cams is None:
            sel_cams = list(range(num_cams))

        # reference camera's ego_pose -> world_from_ego(t) for this frame
        token = inputs['token'][0]
        ref_cam = self.cameras[0] if getattr(self, 'cameras', None) else 'CAM_FRONT'
        sample_nusc = nusc.get('sample', token)
        ref_sd = nusc.get('sample_data', sample_nusc['data'][ref_cam])
        ego = nusc.get('ego_pose', ref_sd['ego_pose_token'])
        w_from_ego = np.eye(4, dtype=np.float64)
        w_from_ego[:3, :3] = Quaternion(ego['rotation']).rotation_matrix
        w_from_ego[:3, 3] = np.array(ego['translation'], dtype=np.float64)

        # aggregate selected cameras at the current frame (frame_id 0, ego coords)
        fx, fr, fsc, fop, fsh = [], [], [], [], []
        for cam in sel_cams:
            xyz_v, rot_v, scale_v, opacity_v, sh_v = _collect_valid_gaussians(
                outputs, cam, 0, 0)
            fx.append(xyz_v); fr.append(rot_v); fsc.append(scale_v)
            fop.append(opacity_v); fsh.append(sh_v)
        xyz = torch.cat(fx, 0); rot = torch.cat(fr, 0); scale = torch.cat(fsc, 0)
        opacity = torch.cat(fop, 0); sh = torch.cat(fsh, 0)

        xyz_w, rot_w, scale_w, op_w, sh_w = transform_gaussians_to_world(
            xyz, rot, scale, opacity, sh, w_from_ego, rotate_sh_flag=False)

        # per-frame voxel downsample to bound memory while merging
        voxel = getattr(self, 'merge_voxel', 0.1)
        keep = voxel_downsample(xyz_w, voxel)
        buf['xyz'].append(xyz_w[keep].cpu())
        buf['rot'].append(rot_w[keep].cpu())
        buf['scale'].append(scale_w[keep].cpu())
        buf['opacity'].append(op_w[keep].cpu())
        buf['sh'].append(sh_w[keep].cpu())

    def _write_merged_scene_ply(self, buf):
        """Concatenate all accumulated frames, global voxel-downsample, and
        write a single global .ply for the scene."""
        import os
        import torch
        from models.gaussian.ply_export import export_gaussians_to_ply
        from models.gaussian.transform_util import voxel_downsample

        if not buf['xyz']:
            print('[merge_ply] no frames accumulated; nothing to write.')
            return

        xyz = torch.cat(buf['xyz'], 0)
        rot = torch.cat(buf['rot'], 0)
        scale = torch.cat(buf['scale'], 0)
        opacity = torch.cat(buf['opacity'], 0)
        sh = torch.cat(buf['sh'], 0)
        print(f'\n[merge_ply] merged gaussians (pre-global-voxel): '
              f'{xyz.shape[0]:,}')

        voxel = getattr(self, 'merge_voxel', 0.1)
        keep = voxel_downsample(xyz, voxel)
        xyz, rot, scale, opacity, sh = (xyz[keep], rot[keep], scale[keep],
                                        opacity[keep], sh[keep])
        print(f'[merge_ply] after global voxel ({voxel} m): {xyz.shape[0]:,}')

        # output under the ply export dir, named scene_merged.ply
        out_dir = getattr(self, 'export_ply_path', './results/ply')
        out_path = os.path.join(out_dir, 'scene_merged.ply')
        n_written, path = export_gaussians_to_ply(
            xyz=xyz, rotation=rot, scale=scale, opacity=opacity, sh=sh,
            path=out_path, opacity_is_activated=True, scale_is_activated=True)
        print(f'[merge_ply] wrote {n_written:,} gaussians -> {path}')

    def _make_eval_videos(self, eval_tokens):
        """Encode one side-by-side (left=GT, right=prediction) mp4 per camera
        from the images saved under self.save_path during evaluation."""
        if not getattr(self, 'save_images', False):
            print('[make_video] save_images is disabled, no images to encode. '
                  'Enable save_images in the eval config to use --make_video.')
            return
        from utils.make_video import make_comparison_videos

        cameras = getattr(self, 'cameras', None)
        if not cameras:
            cameras = [str(i) for i in range(self.num_cams)]
        fps = getattr(self, 'video_fps', 6)
        print(f'\n[make_video] encoding {self.num_cams} per-camera comparison '
              f'video(s) from {len(eval_tokens)} frame(s) under {self.save_path}')
        make_comparison_videos(
            save_path=self.save_path,
            tokens=eval_tokens,
            cameras=cameras,
            fps=fps,
            cam_indices=list(range(self.num_cams)),
        )

    def save_image(
        self,
        image: FloatImage,
        path: Union[Path, str],
    ) -> None:
        """Save an image. Assumed to be in range 0-1."""

        # Create the parent directory if it doesn't already exist.
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)

        # Save the image.
        Image.fromarray(self.prep_image(image)).save(path)


    def prep_image(self, image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
        # Handle batched images.
        if image.ndim == 4:
            image = rearrange(image, "b c h w -> c h (b w)")

        # Handle single-channel images.
        if image.ndim == 2:
            image = rearrange(image, "h w -> () h w")

        # Ensure that there are 3 or 4 channels.
        channel, _, _ = image.shape
        if channel == 1:
            image = repeat(image, "() h w -> c h w", c=3)
        assert image.shape[0] in (3, 4)

        image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
        return rearrange(image, "c h w -> h w c").cpu().numpy()

    @torch.no_grad()
    def compute_reconstruction_metrics(self, inputs, outputs):
        """
        This function computes reconstruction metrics.
        """
        psnr = 0.0
        ssim = 0.0
        lpips = 0.0
        if self.novel_view_mode == 'SF':
            frame_id = 1
        elif self.novel_view_mode == 'MF':
            frame_id = 0
        else:
            raise ValueError(f"Invalid novel view mode: {self.novel_view_mode}")
        for cam in range(self.num_cams):
            rgb_gt = inputs[('color', frame_id, 0)][:, cam, ...]
            image = outputs[('cam', cam)][('gaussian_color', frame_id, 0)]
            psnr += self.compute_psnr(rgb_gt, image).mean()
            ssim += self.compute_ssim(rgb_gt, image).mean()
            lpips += self.compute_lpips(rgb_gt, image).mean()
            if self.save_images:
                assert self.eval_batch_size == 1
                if self.novel_view_mode == 'SF':
                    self.save_image(image, Path(self.save_path) / inputs['token'][0] / f"{cam}.png")
                    self.save_image(rgb_gt, Path(self.save_path) / inputs['token'][0] / f"{cam}_gt.png")
                    self.save_image(inputs[('color', 0, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_0_gt.png")
                elif self.novel_view_mode == 'MF':
                    self.save_image(image, Path(self.save_path) / inputs['token'][0] / f"{cam}.png")
                    self.save_image(rgb_gt, Path(self.save_path) / inputs['token'][0] / f"{cam}_gt.png")
                    self.save_image(inputs[('color', -1, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_prev_gt.png")
                    self.save_image(inputs[('color', 1, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_next_gt.png")
        psnr /= self.num_cams
        ssim /= self.num_cams
        lpips /= self.num_cams
        return psnr, ssim, lpips
    
    @torch.no_grad()
    def compute_psnr(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        ground_truth = ground_truth.clip(min=0, max=1)
        predicted = predicted.clip(min=0, max=1)
        mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
        return -10 * mse.log10()
    
    @torch.no_grad()
    def compute_lpips(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        value = self.lpips.forward(ground_truth, predicted, normalize=True)
        return value[:, 0, 0, 0]
    
    @torch.no_grad()
    def compute_ssim(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        ssim = [
            structural_similarity(
                gt.detach().cpu().numpy(),
                hat.detach().cpu().numpy(),
                win_size=11,
                gaussian_weights=True,
                channel_axis=0,
                data_range=1.0,
            )
            for gt, hat in zip(ground_truth, predicted)
        ]
        return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)
