
import torch
from .gaussian_renderer import render
from einops import rearrange

left_cam_dict = {2:0, 0:1, 4:2, 1:3, 5:4, 3:5}
right_cam_dict = {0:2, 1:0, 2:4, 3:1, 4:5, 5:3}

def get_adj_cams(cam):
    adj_cams = [cam]
    adj_cams.append(left_cam_dict[cam])
    adj_cams.append(right_cam_dict[cam])
    return adj_cams

def _collect_valid_gaussians(outputs, cam, frame_id, i):
    """Extract the valid (masked) Gaussians of one camera/frame for sample i.

    Returns (xyz, rot, scale, opacity, sh) tensors, each already filtered by
    the per-point validity mask. Used both for rendering aggregation and for
    the optional full-scene .ply export.
    """
    valid_i = outputs[('cam', cam)][('pts_valid', frame_id, 0)][i, :]
    xyz_i = outputs[('cam', cam)][('xyz', frame_id, 0)][i, :, :]
    rot_i = outputs[('cam', cam)][('rot_maps', frame_id, 0)][i, :, :, :].permute(1, 2, 0).view(-1, 4)
    scale_i = outputs[('cam', cam)][('scale_maps', frame_id, 0)][i, :, :, :].permute(1, 2, 0).view(-1, 3)
    opacity_i = outputs[('cam', cam)][('opacity_maps', frame_id, 0)][i, :, :, :].permute(1, 2, 0).view(-1, 1)
    sh_i = rearrange(outputs[('cam', cam)][('sh_maps', frame_id, 0)][i, :, :, :], "p srf r xyz d_sh -> (p srf r) d_sh xyz").contiguous()
    return (
        xyz_i[valid_i].view(-1, 3),
        rot_i[valid_i].view(-1, 4),
        scale_i[valid_i].view(-1, 3),
        opacity_i[valid_i].view(-1, 1),
        sh_i[valid_i],
    )

def pts2render(inputs, outputs, cam_num, novel_cam, novel_frame_id, bg_color, mode='MF',
               export_ply_path=None):
    bs, _, height, width = inputs[('color', 0, 0)][:, novel_cam, ...].shape
    render_novel_list = []
    for i in range(bs):
        xyz_i_valid = []
        # rgb_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []
        sh_i_valid = []
        if mode == 'SF':
            frame_id = 0
            for cam in range(cam_num):
                xyz_v, rot_v, scale_v, opacity_v, sh_v = _collect_valid_gaussians(outputs, cam, frame_id, i)
                xyz_i_valid.append(xyz_v)
                rot_i_valid.append(rot_v)
                scale_i_valid.append(scale_v)
                opacity_i_valid.append(opacity_v)
                sh_i_valid.append(sh_v)

        elif mode == 'MF':
            for frame_id in [-1, 1]:
                cam = novel_cam
                xyz_v, rot_v, scale_v, opacity_v, sh_v = _collect_valid_gaussians(outputs, cam, frame_id, i)
                xyz_i_valid.append(xyz_v)
                rot_i_valid.append(rot_v)
                scale_i_valid.append(scale_v)
                opacity_i_valid.append(opacity_v)
                sh_i_valid.append(sh_v)

        pts_xyz_i = torch.concat(xyz_i_valid, dim=0)
        # pts_rgb_i = torch.concat(rgb_i_valid, dim=0)
        # pts_rgb_i = pts_rgb_i * 0.5 + 0.5
        rot_i = torch.concat(rot_i_valid, dim=0)
        scale_i = torch.concat(scale_i_valid, dim=0)
        opacity_i = torch.concat(opacity_i_valid, dim=0)
        sh_i = torch.concat(sh_i_valid, dim=0)

        # optionally export the full 3D Gaussian scene for this sample to .ply
        if export_ply_path is not None:
            from .ply_export import export_gaussians_to_ply

            if mode == 'MF':
                # Rendering only uses novel_cam's prev/next points, but the
                # exported scene should be the full surround view: aggregate
                # every camera across both temporal frames. All points are in
                # the shared world/ego frame (via e2c/c2e extrinsics), so they
                # align geometrically.
                exp_xyz, exp_rot, exp_scale, exp_opacity, exp_sh = [], [], [], [], []
                for cam in range(cam_num):
                    for frame_id in [-1, 1]:
                        xyz_v, rot_v, scale_v, opacity_v, sh_v = _collect_valid_gaussians(outputs, cam, frame_id, i)
                        exp_xyz.append(xyz_v)
                        exp_rot.append(rot_v)
                        exp_scale.append(scale_v)
                        exp_opacity.append(opacity_v)
                        exp_sh.append(sh_v)
                export_xyz = torch.concat(exp_xyz, dim=0)
                export_rot = torch.concat(exp_rot, dim=0)
                export_scale = torch.concat(exp_scale, dim=0)
                export_opacity = torch.concat(exp_opacity, dim=0)
                export_sh = torch.concat(exp_sh, dim=0)
            else:
                # SF already aggregates all cameras into pts_xyz_i etc.
                export_xyz = pts_xyz_i
                export_rot = rot_i
                export_scale = scale_i
                export_opacity = opacity_i
                export_sh = sh_i

            # sh here is [N, d_sh, 3] (channel last), matching the exporter.
            export_gaussians_to_ply(
                xyz=export_xyz,
                rotation=export_rot,
                scale=export_scale,
                opacity=export_opacity,
                sh=export_sh,
                path=export_ply_path,
                opacity_is_activated=True,
                scale_is_activated=True,
            )

        novel_FovX_i = outputs[('cam', novel_cam)][('FovX', novel_frame_id, 0)][i]
        novel_FovY_i = outputs[('cam', novel_cam)][('FovY', novel_frame_id, 0)][i]
        novel_world_view_transform_i = outputs[('cam', novel_cam)][('world_view_transform', novel_frame_id, 0)][i]
        novel_function_proj_transform_i = outputs[('cam', novel_cam)][('full_proj_transform', novel_frame_id, 0)][i]
        novel_camera_center_i = outputs[('cam', novel_cam)][('camera_center', novel_frame_id, 0)][i]

        render_novel_i = render(novel_FovX=novel_FovX_i,
                                novel_FovY=novel_FovY_i,
                                novel_height=height,
                                novel_width=width,
                                novel_world_view_transform=novel_world_view_transform_i,
                                novel_full_proj_transform=novel_function_proj_transform_i,
                                novel_camera_center=novel_camera_center_i,
                                pts_xyz=pts_xyz_i, 
                                pts_rgb=None, 
                                rotations=rot_i, 
                                scales=scale_i, 
                                opacity=opacity_i, 
                                shs=sh_i, 
                                bg_color=bg_color)
        render_novel_list.append(render_novel_i.unsqueeze(0))

    novel = torch.concat(render_novel_list, dim=0)

    return novel
