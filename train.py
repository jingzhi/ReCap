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
import time
from random import randint

import torch
import torchvision
from tqdm import tqdm

from configs import get_cfg
from gaussian import GaussianModel
from light import LightModel
from renderer import PBR_renderer
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import draw_depth, draw_normal, psnr
from utils.loss_utils import (
    delta_normal_loss,
    l1_loss,
    predicted_normal_loss,
    ssim,
    zero_one_loss,
)


def neutral_specular_loss(s, lambda_neutral=0.5):
    # Calculate the mean across the color channels (dim=1)
    s_mean = s.mean(dim=1, keepdim=True)  # Shape: [N, 1]

    # Repeat the mean to match s's shape for RGB comparison
    s_neutral = s_mean.expand(-1, 3)  # Shape: [N, 3]

    # Calculate the L2 loss between s and its neutral (gray) version
    loss_neutral = torch.norm(s - s_neutral, p=float("inf"), dim=1)
    loss_neutral = lambda_neutral * loss_neutral.mean()

    return loss_neutral


def energy_conservation_loss(s, b, lambda_con=0.2):
    # Compute the L2 norm of s and b
    s_norm = torch.norm(s, p=float("inf"), dim=1)
    b_norm = torch.norm(b, p=float("inf"), dim=1)

    # Calculate the energy conservation loss
    total_norm = s_norm + b_norm
    loss = torch.clamp(total_norm - 1, min=0) * lambda_con

    # Return the mean loss for batch processing
    return loss.mean()


def training(cfg):
    scene = Scene(cfg.scene)

    gaussians = GaussianModel(
        cfg.model.sh_degree,
    )
    gaussians.create_from_pcd(scene.get_point_cloud, scene.get_cameras_extent)
    gaussians.training_setup(cfg.optim)

    light = LightModel(cfg.light.cubemap_res, cfg.scene.train_envs)
    light.training_setup(cfg.optim)
    light.train()

    renderer = PBR_renderer(cfg.render, cfg.scene.white_background)

    # load checkpoint
    first_iter = 1
    if cfg.checkpoint_path:
        checkpoint = torch.load(cfg.checkpoint_path)
        model_params = checkpoint["gaussians"]
        light_params = checkpoint["light"]
        first_iter = checkpoint["iteration"]
        gaussians.restore(model_params, cfg.optim)
        light.load_state_dicts(light_params)
        print(f"Load checkpoint from {cfg.checkpoint_path}")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(cfg.optim.iterations), desc="Training progress")
    for iteration in range(first_iter, cfg.optim.iterations + 1):

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #    gaussians.albedo_sh.oneup_sh_order()

        if cfg.debug_visual and iteration in cfg.debug_visual_iterations:
            render_visual = True
        else:
            render_visual = False
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            light.train()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        light.build_mips()
        # Render
        if iteration < cfg.optim.init_until_iter:
            render_pkg = renderer.render(
                viewpoint_camera=viewpoint_cam,
                gaussian=gaussians,
                light=light,
                render_cfg=cfg.render,
                diffuse_only=True,
                render_visual=render_visual,
            )
        else:
            render_pkg = renderer.render(
                viewpoint_camera=viewpoint_cam,
                gaussian=gaussians,
                light=light,
                render_cfg=cfg.render,
                render_visual=render_visual,
            )
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        # Depth normal
        depth_image, normal_image, normal_from_depth, alpha = (
            renderer.render_depth_normal(
                gaussian=gaussians, viewpoint_camera=viewpoint_cam
            )
        )

        # Compute Loss
        gt_image = viewpoint_cam.original_image.cuda()

        ## Reconstruction loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - cfg.optim.lambda_dssim) * Ll1 + cfg.optim.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )

        ## Normal loss
        normal_loss = predicted_normal_loss(normal_image, normal_from_depth, alpha)
        loss += cfg.optim.lambda_predicted_normal * normal_loss

        ## Neutral loss
        reflectance = gaussians.get_reflectance
        vis_reflectance = reflectance[visibility_filter]
        reflectance_loss = neutral_specular_loss(
            vis_reflectance, lambda_neutral=cfg.optim.lambda_neutral
        )
        loss += reflectance_loss

        ## Energy Conservation loss
        albedo = gaussians.get_albedo
        vis_albedo = albedo[visibility_filter]
        energy_loss = energy_conservation_loss(
            vis_reflectance, vis_albedo, lambda_con=cfg.optim.lambda_con
        )
        loss += energy_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                    }
                )
                progress_bar.update(10)
            if iteration == cfg.optim.iterations:
                progress_bar.close()

            # Densification
            if iteration < cfg.optim.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if (
                    iteration > cfg.optim.densify_from_iter
                    and iteration % cfg.optim.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > cfg.optim.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        cfg.optim.densify_grad_threshold,
                        0.005,
                        scene.get_cameras_extent,
                        size_threshold,
                    )

                if iteration % cfg.optim.opacity_reset_interval == 0 or (
                    cfg.scene.white_background
                    and iteration == cfg.optim.densify_from_iter
                ):
                    gaussians.reset_opacity()

        # Optimizer step
        if iteration < cfg.optim.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            gaussians.update_learning_rate(iteration)
            light.step()

        # debugging visualizations
        if render_visual:
            with torch.no_grad():
                os.makedirs(os.path.join(cfg.model_dir, "debug"), exist_ok=True)
                albedo_img, roughness_img, reflectance_img = renderer.render_material(
                    gaussian=gaussians, viewpoint_camera=viewpoint_cam
                )
                draw_depth(
                    depth_image,
                    os.path.join(cfg.model_dir, "debug", f"depth_img_{iteration}.png"),
                )
                draw_normal(
                    normal_image,
                    os.path.join(cfg.model_dir, "debug", f"normal_img_{iteration}.png"),
                )
                draw_normal(
                    normal_from_depth,
                    os.path.join(
                        cfg.model_dir, "debug", f"normal_from_depth_img_{iteration}.png"
                    ),
                )
                torchvision.utils.save_image(
                    albedo_img,
                    os.path.join(cfg.model_dir, "debug", f"albedo_img_{iteration}.png"),
                )
                torchvision.utils.save_image(
                    roughness_img,
                    os.path.join(
                        cfg.model_dir, "debug", f"roughnss_img_{iteration}.png"
                    ),
                )
                torchvision.utils.save_image(
                    reflectance_img,
                    os.path.join(
                        cfg.model_dir, "debug", f"reflectance_img_{iteration}.png"
                    ),
                )
                torchvision.utils.save_image(
                    image,
                    os.path.join(cfg.model_dir, "debug", f"render_img_{iteration}.png"),
                )
                torchvision.utils.save_image(
                    render_pkg["diffuse_img"],
                    os.path.join(
                        cfg.model_dir, "debug", f"diffuse_img_{iteration}.png"
                    ),
                )
                torchvision.utils.save_image(
                    render_pkg["specular_img"],
                    os.path.join(
                        cfg.model_dir,
                        "debug",
                        f"specular_img_{iteration}.png",
                    ),
                )
                light.save(
                    save_dir=os.path.join(cfg.model_dir, "learned_envs"),
                    suffix=iteration,
                )

    print(f"\n[ITER {iteration}] Saving Checkpoint")
    torch.save(
        {
            "gaussians": gaussians.capture(),
            "light": light.get_state_dicts(),
            "iteration": iteration,
        },
        cfg.model_dir + "/chkpt" + str(iteration) + ".pth",
    )
    light.save(save_dir=os.path.join(cfg.model_dir, "learned_envs"))


if __name__ == "__main__":
    # Parse argument and get config
    cfg = get_cfg()

    # Initialize system state (RNG)
    safe_state(cfg.quiet)
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)

    os.makedirs(cfg.model_dir, exist_ok=True)
    training(cfg)

    # All done
    print("\nTraining complete.")
