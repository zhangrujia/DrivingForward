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
        #
        # To keep peak host memory bounded regardless of how many frames the
        # scene has, each frame's (voxel-downsampled) Gaussians are flushed to a
        # per-frame shard file on disk immediately, instead of being held in a
        # growing in-RAM list. At the end we stream the shards back and write
        # the final .ply chunk-by-chunk (see _write_merged_scene_ply).
        merge_ply = getattr(self, 'merge_scene_ply', False)
        merge_ctx = None
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
        if merge_ply:
            import os
            import tempfile
            out_dir = getattr(self, 'export_ply_path', './results/ply')
            os.makedirs(out_dir, exist_ok=True)
            shard_dir = tempfile.mkdtemp(prefix='merge_ply_shards_', dir=out_dir)
            merge_ctx = {'shard_dir': shard_dir, 'shards': [], 'total': 0}
            print(f'[merge_ply] streaming per-frame shards to {shard_dir}')

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
                self._accumulate_scene_ply(inputs, outputs, merge_nusc, merge_ctx)

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
            self._write_merged_scene_ply(merge_ctx)

    def _accumulate_scene_ply(self, inputs, outputs, nusc, ctx):
        """Collect this frame's selected-camera Gaussians, transform them from
        the ego frame into the shared world frame (via the nuScenes ego_pose),
        voxel-downsample, and FLUSH them to a per-frame shard file on disk.

        Flushing each frame immediately (instead of holding it in RAM) keeps
        the peak host-memory footprint bounded by a single frame regardless of
        how many frames the scene contains. The final .ply is assembled later
        by _write_merged_scene_ply, which streams these shards back."""
        import os
        import torch
        import numpy as np
        from pyquaternion import Quaternion
        from models.gaussian.GaussianRender import _collect_valid_gaussians
        from models.gaussian.transform_util import (transform_gaussians_to_world,
                                                    voxel_downsample)

        # In debug mode, merge ONLY the first 2 frames so a quick test produces
        # a tiny .ply (useful for diagnosing the black-screen / viewer issue).
        if getattr(self, 'debug', False):
            DEBUG_MERGE_FRAMES = 2
            if len(ctx['shards']) >= DEBUG_MERGE_FRAMES:
                return
            print(f'[merge_ply][DEBUG] accumulating frame '
                  f'{len(ctx["shards"]) + 1}/{DEBUG_MERGE_FRAMES}')

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

        # aggregate selected cameras' Gaussians for this sample.
        #
        # IMPORTANT: which temporal frame holds the predicted Gaussians depends
        # on the novel-view mode, and MUST match pts2render:
        #   - MF: the model predicts Gaussians at the adjacent frames [-1, +1];
        #         there are NO valid Gaussians at frame_id 0.
        #   - SF: the model predicts Gaussians at the current frame [0].
        # Using the wrong frame_id (e.g. always 0 in MF) collects empty/near-
        # empty points, producing a merged .ply far smaller than a single-frame
        # export. All these points are already warped into the CURRENT sample's
        # ego frame (via e2c_extr = cam_T_cam(0, f) @ extrinsics_inv), so the
        # current-sample ego_pose above is the correct ego->world transform.
        mode = getattr(self, 'novel_view_mode', 'MF')
        exp_frames = [-1, 1] if mode == 'MF' else [0]

        fx, fr, fsc, fop, fsh = [], [], [], [], []
        for cam in sel_cams:
            for frame_id in exp_frames:
                xyz_v, rot_v, scale_v, opacity_v, sh_v = _collect_valid_gaussians(
                    outputs, cam, frame_id, 0)
                fx.append(xyz_v); fr.append(rot_v); fsc.append(scale_v)
                fop.append(opacity_v); fsh.append(sh_v)
        xyz = torch.cat(fx, 0); rot = torch.cat(fr, 0); scale = torch.cat(fsc, 0)
        opacity = torch.cat(fop, 0); sh = torch.cat(fsh, 0)

        xyz_w, rot_w, scale_w, op_w, sh_w = transform_gaussians_to_world(
            xyz, rot, scale, opacity, sh, w_from_ego, rotate_sh_flag=False)

        # per-frame voxel downsample to bound memory while merging
        voxel = getattr(self, 'merge_voxel', 0.1)
        keep = voxel_downsample(xyz_w, voxel)

        # flush this frame's surviving Gaussians to a compressed shard on disk
        shard_idx = len(ctx['shards'])
        shard_path = os.path.join(ctx['shard_dir'], f'frame_{shard_idx:05d}.npz')
        np.savez(
            shard_path,
            xyz=xyz_w[keep].cpu().numpy().astype(np.float32),
            rot=rot_w[keep].cpu().numpy().astype(np.float32),
            scale=scale_w[keep].cpu().numpy().astype(np.float32),
            opacity=op_w[keep].cpu().numpy().astype(np.float32),
            sh=sh_w[keep].cpu().numpy().astype(np.float32),
        )
        n_shard = int(keep.shape[0])
        ctx['shards'].append({'path': shard_path, 'n': n_shard})
        ctx['total'] += n_shard

    def _write_merged_scene_ply(self, ctx):
        """Assemble the final global .ply by STREAMING the per-frame shards
        from disk, so the whole cloud is never materialised in RAM at once.

        Two passes:
          1. Load only each shard's xyz (one column, cheap), concatenate, and
             run the global voxel downsample to decide which points survive.
          2. Stream the shards again; for each, select its surviving points and
             append them to a StreamingPlyWriter that flushes straight to disk.
        """
        import os
        import shutil
        import numpy as np
        import torch
        from models.gaussian.ply_export import StreamingPlyWriter
        from models.gaussian.transform_util import voxel_downsample

        shards = ctx['shards'] if ctx else []
        if not shards:
            print('[merge_ply] no frames accumulated; nothing to write.')
            if ctx and os.path.isdir(ctx['shard_dir']):
                shutil.rmtree(ctx['shard_dir'], ignore_errors=True)
            return

        voxel = getattr(self, 'merge_voxel', 0.1)
        out_dir = getattr(self, 'export_ply_path', './results/ply')
        # include the voxel size in the filename so runs with different
        # --merge_voxel values don't overwrite each other. Replace the decimal
        # point with 'p' so the filename has no extra dots (0 -> "0",
        # 0.1 -> "0p1", 0.5 -> "0p5", 0.25 -> "0p25").
        voxel_tag = f'{voxel:g}'.replace('.', 'p')
        out_path = os.path.join(out_dir, f'scene_merged_voxel{voxel_tag}.ply')

        try:
            # --- pass 1: global voxel downsample using only xyz ---------------
            print(f'\n[merge_ply] merged gaussians (pre-global-voxel): '
                  f'{ctx["total"]:,}')
            xyz_all = np.concatenate(
                [np.load(s['path'])['xyz'] for s in shards], axis=0)
            keep_global = voxel_downsample(torch.from_numpy(xyz_all), voxel)
            keep_global = keep_global.cpu().numpy()

            # a boolean mask over the global (shard-concatenated) index space
            total = ctx['total']
            keep_mask = np.zeros(total, dtype=bool)
            keep_mask[keep_global] = True
            n_final = int(keep_mask.sum())

            # Recenter: nuScenes ego_pose lives in a global map frame, so the
            # merged cloud sits hundreds of metres from the origin. Web viewers
            # frame the camera at (0,0,0), putting the cloud off-screen (black
            # screen). Subtract the centroid of the KEPT points so the cloud is
            # centered at the origin while preserving inter-frame alignment.
            recenter = getattr(self, 'merge_recenter', True)
            if recenter:
                center = xyz_all[keep_global].mean(axis=0).astype(np.float32)
                print(f'[merge_ply] recentering cloud by centroid '
                      f'({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})')
            else:
                center = np.zeros(3, dtype=np.float32)
            del keep_global, xyz_all
            print(f'[merge_ply] after global voxel ({voxel} m): {n_final:,}')

            # peek d_sh from the first shard
            with np.load(shards[0]['path']) as z0:
                d_sh = int(z0['sh'].shape[1])

            # --- pass 2: stream each shard, select survivors, append ----------
            writer = StreamingPlyWriter(
                out_path, total_points=n_final, d_sh=d_sh,
                opacity_is_activated=True, scale_is_activated=True)
            offset = 0
            for s in shards:
                n = s['n']
                local_mask = keep_mask[offset:offset + n]
                offset += n
                if not local_mask.any():
                    continue
                with np.load(s['path']) as z:
                    sel = np.nonzero(local_mask)[0]
                    xyz_sel = z['xyz'][sel].astype(np.float32) - center
                    writer.append(
                        xyz=xyz_sel,
                        rotation=z['rot'][sel],
                        scale=z['scale'][sel],
                        opacity=z['opacity'][sel],
                        sh=z['sh'][sel],
                    )
            n_written, path = writer.close()
            print(f'[merge_ply] wrote {n_written:,} gaussians -> {path}')
        finally:
            # clean up the temporary shard directory
            if os.path.isdir(ctx['shard_dir']):
                shutil.rmtree(ctx['shard_dir'], ignore_errors=True)

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
