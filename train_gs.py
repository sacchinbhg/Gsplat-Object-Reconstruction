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

import os
import sys
import uuid
import json
from argparse import ArgumentParser, Namespace
from random import randint
from typing import Optional

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.functional.regression import pearson_corrcoef
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, monodisp
from utils.pose_utils import update_pose, get_loss_tracking
from torch.utils.tensorboard.writer import SummaryWriter
TENSORBOARD_FOUND = True

def training(args, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if args.use_dust3r:
        print('Using pose refinement from dust3r')
        from gaussian_renderer import render_w_pose as render
    else:
        from gaussian_renderer import render

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, extra_opts=args)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()

        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Setup pose optimizer (for dust3r or bundle_adjust)
        pose_optimizer = None
        if (args.use_dust3r or args.bundle_adjust) and hasattr(viewpoint_cam, "cam_rot_delta"):
            pose_opt_params = [
                {"params": [viewpoint_cam.cam_rot_delta], "lr": opt.pose_lr_rot},
                {"params": [viewpoint_cam.cam_trans_delta], "lr": opt.pose_lr_trans}
            ]
            pose_optimizer = torch.optim.Adam(pose_opt_params)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # Loss and backprop
        loss, Ll1 = cal_loss(opt, args, image, render_pkg, viewpoint_cam, bg, tb_writer=tb_writer, iteration=iteration, mono_loss_type=args.mono_loss_type)
        loss.backward()

        iter_end.record()

        # Optimizer step
        with torch.no_grad():
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                if pose_optimizer is not None and iteration < opt.pose_iterations:
                    pose_optimizer.step()
                    pose_optimizer.zero_grad(set_to_none=True)
                    _ = update_pose(viewpoint_cam)

            # Logging and diagnostics
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            num_gauss = len(gaussians._xyz)
            if iteration % 10 == 0:
                progress_bar.set_postfix({'Loss': f"{ema_loss_for_log:.7f}", 'n': f"{num_gauss}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and num_gauss < opt.max_num_splats:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

                if iteration % opt.remove_outliers_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.remove_outliers(opt, iteration, linear=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/ckpt{iteration}.pth")

        # Pose update monitoring
        if hasattr(viewpoint_cam, "cam_rot_delta"):
            print(f"[ITER {iteration}] Pose delta norm: {torch.norm(viewpoint_cam.cam_rot_delta).item():.4f}")

    from scipy.spatial.transform import Rotation as R
    est_R = viewpoint_cam.R
    true_R = viewpoint_cam.R_gt
    R_diff = R.from_matrix(est_R @ true_R.T)
    rot_angle_deg = np.linalg.norm(R_diff.as_rotvec()) * (180.0 / np.pi)

    # --- Translation recovery error ---
    est_T = viewpoint_cam.T
    true_T = viewpoint_cam.T_gt
    trans_error = np.linalg.norm(est_T - true_T)

    print(f"Recovered pose error for cam {viewpoint_cam.uid}:")
    print(f"  Rotation error: {rot_angle_deg:.2f} deg (applied: {viewpoint_cam.rot_noise_deg:.2f} deg)")
    print(f"  Translation error: {trans_error:.3f} m (applied: {np.linalg.norm(viewpoint_cam.trans_noise_vec):.3f} m)")



def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
            args.model_path = os.path.join("./output/", unique_str)
        else:
            unique_str = str(uuid.uuid4())
            args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        torch.cuda.empty_cache()

def cal_loss(opt, args, image, render_pkg, viewpoint_cam, bg, silhouette_loss_type="bce", mono_loss_type="mid", tb_writer: Optional[SummaryWriter]=None, iteration=0):
    """
    Calculate the loss of the image, contains l1 loss and ssim loss.
    l1 loss: Ll1 = l1_loss(image, gt_image)
    ssim loss: Lssim = 1 - ssim(image, gt_image)
    Optional: [silhouette loss, monodepth loss]
    """
    gt_image = viewpoint_cam.original_image.to(image.dtype).cuda()
    if opt.random_background:
        gt_image = gt_image * viewpoint_cam.mask + bg[:, None, None] * (1 - viewpoint_cam.mask).squeeze()
    Ll1 = l1_loss(image, gt_image)
    Lssim = (1.0 - ssim(image, gt_image))
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim
    if tb_writer is not None:
        tb_writer.add_scalar('loss/l1_loss', Ll1, iteration)
        tb_writer.add_scalar('loss/ssim_loss', Lssim, iteration)

    if hasattr(args, "use_mask") and args.use_mask:
        if silhouette_loss_type == "bce":
            silhouette_loss = F.binary_cross_entropy(render_pkg["rendered_alpha"], viewpoint_cam.mask)
        elif silhouette_loss_type == "mse":
            silhouette_loss = F.mse_loss(render_pkg["rendered_alpha"], viewpoint_cam.mask)
        else:
            raise NotImplementedError
        loss = loss + opt.lambda_silhouette * silhouette_loss
        if tb_writer is not None:
            tb_writer.add_scalar('loss/silhouette_loss', silhouette_loss, iteration)

    if hasattr(viewpoint_cam, "mono_depth") and viewpoint_cam.mono_depth is not None:
        if mono_loss_type == "mid":
            # we apply masked monocular loss
            gt_mask = torch.where(viewpoint_cam.mask > 0.5, True, False)
            render_mask = torch.where(render_pkg["rendered_alpha"] > 0.5, True, False)
            mask = torch.logical_and(gt_mask, render_mask)
            if mask.sum() < 10:
                depth_loss = 0.0
            else:
                disp_mono = 1 / viewpoint_cam.mono_depth[mask].clamp(1e-6) # shape: [N]
                disp_render = 1 / render_pkg["rendered_depth"][mask].clamp(1e-6) # shape: [N]
                depth_loss = monodisp(disp_mono, disp_render, 'l1')[-1]
        elif mono_loss_type == "pearson":
            disp_mono = 1 / viewpoint_cam.mono_depth[viewpoint_cam.mask > 0.5].clamp(1e-6) # shape: [N]
            disp_render = 1 / render_pkg["rendered_depth"][viewpoint_cam.mask > 0.5].clamp(1e-6) # shape: [N]
            depth_loss = (1 - pearson_corrcoef(disp_render, -disp_mono)).mean()
        elif mono_loss_type == "dust3r":
            gt_mask = torch.where(viewpoint_cam.mask > 0.5, True, False)
            render_mask = torch.where(render_pkg["rendered_alpha"] > 0.5, True, False)
            mask = torch.logical_and(gt_mask, render_mask)
            if mask.sum() < 10:
                depth_loss = 0.0
            else:
                disp_mono = 1 / viewpoint_cam.mono_depth[mask].clamp(1e-6) # shape: [N]
                disp_render = 1 / render_pkg["rendered_depth"][mask].clamp(1e-6) # shape: [N]
                depth_loss = torch.abs((disp_render - disp_mono)).mean()
            depth_loss *= (opt.iterations - iteration) / opt.iterations # linear scheduler
        else:
            raise NotImplementedError

        loss = loss + args.mono_depth_weight * depth_loss
        if tb_writer is not None:
            tb_writer.add_scalar('loss/depth_loss', depth_loss, iteration)

    if args.use_dust3r:
        image_ab = (torch.exp(viewpoint_cam.exposure_a)) * image + viewpoint_cam.exposure_b
        tracking_loss = get_loss_tracking(image_ab, render_pkg["rendered_alpha"], viewpoint_cam) + args.lambda_t_norm * torch.abs(viewpoint_cam.cam_trans_delta).mean()
        loss = loss + tracking_loss
        if tb_writer is not None:
            tb_writer.add_scalar('loss/tracking_loss', tracking_loss, iteration)

    return loss, Ll1

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    ### some exp args
    parser.add_argument("--sparse_view_num", type=int, default=-1, 
                        help="Use sparse view or dense view, if sparse_view_num > 0, use sparse view, \
                        else use dense view. In sparse setting, sparse views will be used as training data, \
                        others will be used as testing data.")
    parser.add_argument("--use_mask", default=True, help="Use masked image, by default True")
    parser.add_argument('--use_dust3r', action='store_true', default=False,
                        help='use dust3r estimated poses')
    parser.add_argument('--dust3r_json', type=str, default=None)
    parser.add_argument("--init_pcd_name", default='origin', type=str, 
                        help="the init pcd name. 'random' for random, 'origin' for pcd from the whole scene")
    parser.add_argument("--transform_the_world", action="store_true", help="Transform the world to the origin")
    parser.add_argument('--mono_depth_weight', type=float, default=0.0005, help="The rate of monodepth loss")
    parser.add_argument('--lambda_t_norm', type=float, default=0.0005)
    parser.add_argument('--mono_loss_type', type=str, default="mid")

    parser.add_argument('--bundle_adjust', action='store_true',
                    help='Enable BARF-style camera pose optimization')

    parser.add_argument('--pose_noise', action='store_true',
                        help='Inject synthetic pose noise for testing optimization')

    parser.add_argument('--pose_noise_deg', type=float, default=10.0,
                        help='Degrees of rotation noise applied to initial camera pose')

    parser.add_argument('--pose_noise_trans', type=float, default=0.1,
                        help='Magnitude of translation noise applied to initial camera pose (in meters)')


    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, 
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
