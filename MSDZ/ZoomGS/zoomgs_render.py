#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import matplotlib.pyplot as plt
import torch
import os
from tqdm import tqdm
import numpy as np
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.compat import resolve_device, set_runtime_device
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel

import cv2
import time
from tqdm import tqdm

from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, generate_spiral_path
from utils.general_utils import vis_depth

from utils.image_utils import psnr
from utils.loss_utils import ssim
import numpy as np
from lpipsPyTorch import lpips

import utils.Spline as Spline
import numpy as np
import torch
from scene.cameras import Camera


# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
import torch
import math
import torch.nn as nn

from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import utils.Spline as Spline
import numpy as np
import torch
from scene.cameras import Camera
from utils.loss_utils import l1_loss, ssim


def create_video_from_frames(frame_paths, output_path, fps):
    if fps <= 0:
        raise ValueError("Video FPS must be greater than 0.")
    if not frame_paths:
        raise ValueError("No rendered frames were provided for video creation.")

    first_frame = cv2.imread(frame_paths[0], cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError("Failed to read rendered frame: {}".format(frame_paths[0]))

    height, width = first_frame.shape[:2]
    frame_size = (width, height)
    makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(
            "Failed to create MP4 with the OpenCV mp4v codec: {}".format(output_path)
        )

    try:
        for frame_path in tqdm(frame_paths, desc="Encoding video", unit="frame"):
            frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("Failed to read rendered frame: {}".format(frame_path))
            if (frame.shape[1], frame.shape[0]) != frame_size:
                raise ValueError(
                    "Rendered frame {} has resolution {}, expected {}.".format(
                        frame_path,
                        (frame.shape[1], frame.shape[0]),
                        frame_size,
                    )
                )
            writer.write(frame)
    finally:
        writer.release()

    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("OpenCV created an empty MP4 file: {}".format(output_path))
    print("[ZoomGS] Video saved to {}".format(os.path.abspath(output_path)))


class WideCamera(nn.Module):
    def __init__(self, camrea):
        super(WideCamera, self).__init__()

        self.uid = camrea.uid
        self.colmap_id = camrea.colmap_id
        self.R = camrea.R
        self.T = camrea.T
        self.FoVx = camrea.FoVx
        self.FoVy = camrea.FoVy

        self.image_name = camrea.image_name
        self.image_path = getattr(camrea, "image_path", None)

        try:
            self.data_device = resolve_device(camrea.data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {camrea.data_device} failed, fallback to default cuda device" )
            self.data_device = resolve_device("cuda")

        self.original_image = camrea.original_image
        self.image_width = camrea.image_width
        self.image_height = camrea.image_height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = camrea.trans
        self.scale = camrea.scale

        self.world_view_transform = torch.as_tensor(
            getWorld2View2(self.R, self.T, self.trans, self.scale), dtype=torch.float32, device=self.data_device
        ).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def update(self, ):
        self.world_view_transform = torch.as_tensor(
            getWorld2View2(self.R, self.T, self.trans, self.scale), dtype=torch.float32, device=self.data_device
        ).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def apply_digital_zoom(camera, source_zoom, target_zoom):
    zoom_ratio = target_zoom / source_zoom
    if math.isclose(zoom_ratio, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        return camera

    camera.FoVx = 2.0 * math.atan(math.tan(float(camera.FoVx) / 2.0) / zoom_ratio)
    camera.FoVy = 2.0 * math.atan(math.tan(float(camera.FoVy) / 2.0) / zoom_ratio)
    camera.update()
    return camera


def generate_linear_camera(start_camera, end_camera, N_C, N_I):
    linear_cameras = []

    start_pose = np.concatenate([np.array(start_camera.R), np.expand_dims(np.array(start_camera.T),-1)], axis=-1) 
    start_pose = torch.Tensor(start_pose).unsqueeze(dim=0)

    end_pose = np.concatenate([np.array(end_camera.R), np.expand_dims(np.array(end_camera.T),-1)], axis=-1) 
    end_pose = torch.Tensor(end_pose).unsqueeze(dim=0)

    poses_start_se3 = Spline.SE3_to_se3_N(start_pose[:, :3, :4])  
    poses_end_se3 = Spline.SE3_to_se3_N(end_pose[:, :3, :4])     

    pose_nums = torch.cat([torch.zeros((1, N_C)), torch.arange(N_I).reshape(1, -1)], -1).repeat(poses_start_se3.shape[0], 1) 

    seg_pos_x = torch.arange(poses_start_se3.shape[0]).reshape([poses_start_se3.shape[0], 1]).repeat(1, N_C+N_I)
    se3_start = poses_start_se3[seg_pos_x, :]   
    se3_end = poses_end_se3[seg_pos_x, :]       

    spline_poses = Spline.SplineN_linear(se3_start, se3_end, pose_nums, N_I).cpu().numpy()

    nums = N_C + N_I
    pose_nums = torch.arange(N_C + N_I).reshape(1, -1)
    pose_time = pose_nums[0] / (nums - 1)
    start_FovX = np.array(start_camera.FoVx)
    start_FovY = np.array(start_camera.FoVy)

    end_FovX = np.array(end_camera.FoVx)
    end_FovY = np.array(end_camera.FoVy)

    l_FovX =  (1 - pose_time) * start_FovX + pose_time * end_FovX    
    l_FovY =  (1 - pose_time) * start_FovY + pose_time * end_FovY 

    l_FovX = l_FovX.cpu().numpy()
    l_FovY = l_FovY.cpu().numpy()

    linear_cameras.append(start_camera)
    for kk in range(1, nums-1):
        cam = WideCamera(Camera(colmap_id=start_camera.colmap_id, R=spline_poses[kk, :3, :3], T=spline_poses[kk, :3, 3], 
                            FoVx=l_FovX[kk], FoVy=l_FovY[kk], image=start_camera.original_image,
                            gt_alpha_mask=None, image_name=start_camera.image_name, uid=start_camera.uid,
                            trans=start_camera.trans, scale=start_camera.scale,
                            data_device=start_camera.data_device, image_path=start_camera.image_path)
        )
        linear_cameras.append(cam)
    linear_cameras.append(end_camera)
    return linear_cameras

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args, refine=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    log_dir = os.path.join(model_path, name, "ours_{}".format(iteration), "metrics.txt")
    f = open(log_dir, 'a')

    uw_psnr_list = []
    uw_ssim_list = []
    uw_lpips_list = []

    wide_psnr_list = []
    wide_ssim_list = []
    wide_lpips_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if view.image_name.split('_')[0] == "uw":
            rendering = render(view, gaussians, pipeline, background, info=None)
            gt = view.original_image[0:3, :, :]

            render_image = rendering["render"].clamp(0., 1.).unsqueeze(0)
            gt = gt.clamp(0., 1.).unsqueeze(0)
            psnr_metric = psnr(render_image, gt).mean().double().item()
            ssim_metric = ssim(render_image, gt).mean().double().item()
            lpips_metric = lpips(render_image, gt, net_type='vgg').mean().double().item()

            uw_psnr_list.append(psnr_metric)
            uw_ssim_list.append(ssim_metric)
            uw_lpips_list.append(lpips_metric)

        if view.image_name.split('_')[0] == "w":
            rendering = render(view, gaussians, pipeline, background, info={"c":1.0, "target":args.target})
            gt = view.original_image[0:3, :, :]

            render_image = rendering["render"].clamp(0., 1.).unsqueeze(0)
            gt = gt.clamp(0., 1.).unsqueeze(0)
            psnr_metric = psnr(render_image, gt).mean().double().item()
            ssim_metric = ssim(render_image, gt).mean().double().item()
            lpips_metric = lpips(render_image, gt, net_type='vgg').mean().double().item()

            wide_psnr_list.append(psnr_metric)
            wide_ssim_list.append(ssim_metric)
            wide_lpips_list.append(lpips_metric)

        torchvision.utils.save_image(rendering["render"], os.path.join(render_path, view.image_name + '.png'))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

    print('UW: PSNR:', np.mean(np.array(uw_psnr_list)), 'SSIM:', np.mean(np.array(uw_ssim_list)), 'LPIPS:', np.mean(np.array(uw_lpips_list)))
    print('Wide: PSNR:', np.mean(np.array(wide_psnr_list)), 'SSIM:', np.mean(np.array(wide_ssim_list)), 'LPIPS:', np.mean(np.array(wide_lpips_list)))
    f.write('UW : PSNR:' + str(np.mean(np.array(uw_psnr_list))) + str(np.mean(np.array(uw_ssim_list))) + str(np.mean(np.array(uw_lpips_list))) +'\n')
    f.write('Wide : PSNR:' + str(np.mean(np.array(wide_psnr_list))) + str(np.mean(np.array(wide_ssim_list))) + str(np.mean(np.array(wide_lpips_list))) +'\n')
    f.flush()
    f.close()


def find_view_by_name(views, image_selector, camera_type):
    selector_name = image_selector.replace("\\", "/").rsplit("/", 1)[-1]
    selector_stem = os.path.splitext(selector_name)[0].casefold()
    matches = [view for view in views if view.image_name.casefold() == selector_stem]

    if not matches:
        available = ", ".join(view.image_name for view in views)
        raise ValueError(
            "Could not find {camera_type} image '{selector}'. Available images: {available}".format(
                camera_type=camera_type,
                selector=image_selector,
                available=available,
            )
        )
    return matches[0]


def select_render_pair(uw_views, wide_views, args):
    has_uw_name = bool(args.uw_image.strip())
    has_wide_name = bool(args.wide_image.strip())
    if has_uw_name != has_wide_name:
        raise ValueError("--uw_image and --wide_image must be specified together.")

    if has_uw_name:
        uw_view = find_view_by_name(uw_views, args.uw_image, "UW")
        wide_view = find_view_by_name(wide_views, args.wide_image, "W")
        pair_name = "{}_to_{}".format(uw_view.image_name, wide_view.image_name)
        return uw_view, wide_view, pair_name

    if args.pair_index < 0 or args.pair_index >= min(len(uw_views), len(wide_views)):
        raise ValueError(
            "--pair_index {} is out of range; this scene has {} index-paired images.".format(
                args.pair_index, min(len(uw_views), len(wide_views))
            )
        )

    return uw_views[args.pair_index], wide_views[args.pair_index], str(args.pair_index)


def render_set_linear(model_path, name, iteration, views, gaussians, pipeline, background, args, refine=None):
    render_path = os.path.join(model_path, name, "zoom_sequences")

    makedirs(render_path, exist_ok=True)

    uw_views = []
    wide_views = []

    for view in views:
        if view.image_name.split('_')[0] == "uw":
            uw_views.append(WideCamera(view))
        elif view.image_name.split('_')[0] == "w":
            wide_views.append(WideCamera(view))

    uw_views.sort(key=lambda camera: camera.image_name)
    wide_views.sort(key=lambda camera: camera.image_name)
    uw_view, wide_view, pair_name = select_render_pair(uw_views, wide_views, args)
    apply_digital_zoom(uw_view, args.uw_source_zoom, args.uw_target_zoom)

    pair_output_path = os.path.abspath(os.path.join(render_path, pair_name))
    uw_path = os.path.abspath(uw_view.image_path) if uw_view.image_path else uw_view.image_name
    wide_path = os.path.abspath(wide_view.image_path) if wide_view.image_path else wide_view.image_name
    tqdm.write(
        "\n[ZoomGS] Rendering image pair\n"
        "  UW image : {uw_path}\n"
        "  W image  : {wide_path}\n"
        "  UW zoom  : {source_zoom:.3f}x -> {target_zoom:.3f}x\n"
        "  Output   : {output_path}".format(
            uw_path=uw_path,
            wide_path=wide_path,
            source_zoom=args.uw_source_zoom,
            target_zoom=args.uw_target_zoom,
            output_path=pair_output_path,
        )
    )

    N_C = 160
    N_I = 32
    # To release FI burden, x0.6 ~ 0.85 can be implemented by SR, 0.85 ~ 1.0 can beimplemented by continuous camera transition
    # Thus N_C camera encodings are set to 0., and N_I camera encodings are set to (0., 1.)
    c_views = np.concatenate((np.zeros(N_C), np.linspace(0., 1., N_I)), 0)
    linear_views = generate_linear_camera(uw_view, wide_view, N_C, N_I)

    makedirs(pair_output_path, exist_ok=True)
    frame_paths = []
    for ii in tqdm(range(len(linear_views)), desc="Rendering {}".format(pair_name), unit="frame"):
        view = linear_views[ii]
        c = c_views[ii]

        rendering = render(view, gaussians, pipeline, background, info={"c":c, "target":args.target})
        render_image = rendering["render"].clamp(0., 1.).unsqueeze(0)
        frame_path = os.path.join(pair_output_path, '%04d.png'%ii)
        torchvision.utils.save_image(render_image, frame_path)
        frame_paths.append(frame_path)

    if not args.skip_video:
        video_path = os.path.join(render_path, "videos", "{}.mp4".format(pair_name))
        create_video_from_frames(frame_paths, video_path, args.video_fps)



def render_sets(dataset : ModelParams, pipeline : PipelineParams, args):

    with torch.no_grad():
        runtime_device = set_runtime_device(args.data_device)
        gaussians = GaussianModel(args.sh_degree, device=runtime_device)
        scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False, get_all_cam="all")
        refine = None

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=runtime_device)

        all_views = scene.getTrainCameras() + scene.getTestCameras()
        render_set_linear(dataset.model_path, "train", scene.loaded_iter, all_views, gaussians, pipeline, background, args, refine)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_depth", action="store_true")

    parser.add_argument("--target", default="", type=str)
    # Set both defaults to fixed image names, for example "uw_001.jpg" and "w_001.jpg".
    parser.add_argument("--uw_image", default="", type=str)
    parser.add_argument("--wide_image", default="", type=str)
    parser.add_argument("--pair_index", default=3, type=int)
    parser.add_argument("--uw_source_zoom", default=0.5, type=float)
    parser.add_argument("--uw_target_zoom", default=0.6, type=float)
    parser.add_argument("--video_fps", default=30.0, type=float)
    parser.add_argument("--skip_video", action="store_true")
    args = get_combined_args(parser)
    if args.uw_source_zoom <= 0:
        parser.error("--uw_source_zoom must be greater than 0")
    if args.uw_target_zoom <= 0:
        parser.error("--uw_target_zoom must be greater than 0")
    if args.video_fps <= 0:
        parser.error("--video_fps must be greater than 0")
    print("Rendering " + args.model_path)


    # Initialize system state (RNG)
    safe_state(args.quiet, args.data_device)

    render_sets(model.extract(args), pipeline.extract(args), args)
