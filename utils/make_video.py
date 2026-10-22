"""
Encode per-camera side-by-side comparison videos from the images saved during
evaluation.

For each camera, every evaluated frame is laid out as:  [ground-truth | prediction]
(left = GT, right = rendered prediction), and the frames are concatenated in the
order they were evaluated (temporal order for a single scene). One mp4 per camera
is produced, so 6 cameras -> 6 videos.

The images are the ones written by DrivingForwardTrainer.compute_reconstruction_metrics:
    {save_path}/{token}/{cam}.png      -> prediction (rendered)
    {save_path}/{token}/{cam}_gt.png   -> ground truth
where `cam` is the 0-based camera index.

This module has no torch dependency; it only needs numpy + imageio (+ Pillow to
read the pngs). It is safe to import on a CPU-only machine.
"""
import os


def _load_png(path):
    """Load a png as an [H, W, 3] uint8 numpy array (drops alpha if present)."""
    import numpy as np
    from PIL import Image
    img = Image.open(path).convert('RGB')
    return np.asarray(img, dtype='uint8')


def _hstack_gt_pred(gt, pred):
    """Horizontally stack GT (left) and prediction (right).

    If the two images differ in height (shouldn't normally happen), pad the
    shorter one with black so hstack is valid.
    """
    import numpy as np
    if gt.shape[0] != pred.shape[0]:
        h = max(gt.shape[0], pred.shape[0])

        def pad(a):
            if a.shape[0] == h:
                return a
            padded = np.zeros((h, a.shape[1], a.shape[2]), dtype=a.dtype)
            padded[:a.shape[0]] = a
            return padded
        gt, pred = pad(gt), pad(pred)
    return np.concatenate([gt, pred], axis=1)


def make_comparison_videos(save_path, tokens, cameras, fps=6,
                           out_dir=None, cam_indices=None):
    """Encode one side-by-side (GT | pred) mp4 per camera.

    Args:
        save_path: directory that contains one sub-folder per evaluated token,
            each holding {cam}.png (prediction) and {cam}_gt.png (ground truth).
        tokens: list of sample tokens in the order they should appear in the
            video (temporal order). Duplicates/missing folders are skipped.
        cameras: list of camera names (used only for the output file names).
        fps: frames per second for the encoded videos.
        out_dir: where to write the mp4s (default: <save_path>/videos).
        cam_indices: optional list of integer camera indices matching
            `cameras`; defaults to range(len(cameras)).

    Returns:
        list of written video file paths.
    """
    import imageio

    if out_dir is None:
        out_dir = os.path.join(save_path, 'videos')
    os.makedirs(out_dir, exist_ok=True)

    if cam_indices is None:
        cam_indices = list(range(len(cameras)))

    # de-duplicate tokens while preserving order
    seen = set()
    ordered_tokens = []
    for t in tokens:
        if t and t not in seen:
            seen.add(t)
            ordered_tokens.append(t)

    written = []
    for cam_name, ci in zip(cameras, cam_indices):
        frames = []
        missing = 0
        for tok in ordered_tokens:
            pred_path = os.path.join(save_path, tok, f'{ci}.png')
            gt_path = os.path.join(save_path, tok, f'{ci}_gt.png')
            if not (os.path.exists(pred_path) and os.path.exists(gt_path)):
                missing += 1
                continue
            gt = _load_png(gt_path)
            pred = _load_png(pred_path)
            frames.append(_hstack_gt_pred(gt, pred))

        if not frames:
            print(f'[make_video] camera {cam_name} (idx {ci}): no frames found, '
                  f'skipping')
            continue

        out_path = os.path.join(out_dir, f'{cam_name}.mp4')
        imageio.mimwrite(out_path, frames, fps=fps, quality=8,
                         macro_block_size=1)
        note = f' ({missing} frames missing images)' if missing else ''
        print(f'[make_video] {cam_name}: {len(frames)} frames @ {fps}fps '
              f'-> {out_path}{note}')
        written.append(out_path)

    if written:
        print(f'[make_video] done. {len(written)} video(s) in {out_dir}')
    else:
        print(f'[make_video] no videos written (no usable frames under '
              f'{save_path}).')
    return written
