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
from os import makedirs

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from configs import get_cfg
from gaussian import GaussianModel
from light import LightModel
from renderer import PBR_renderer
from scene import Scene
from utils.general_utils import safe_state


def render_set(gaussians, light, views, renderer, save_path, env_name):

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()

        render_path = os.path.join(save_path, env_name, "renders")
        makedirs(render_path, exist_ok=True)
        render_pkg = renderer.render(
            viewpoint_camera=view,
            gaussian=gaussians,
            light=light,
            render_cfg=cfg.render,
            debug=False,
            relight=True,
            light_name=cfg.render.light_name,
        )

        torch.cuda.synchronize()

        torchvision.utils.save_image(
            render_pkg["render"],
            os.path.join(render_path, "{0:05d}".format(idx) + ".png"),
        )


def render_sets(cfg):
    with torch.no_grad():
        gaussians = GaussianModel(
            cfg.model.sh_degree,
        )
        scene = Scene(cfg.scene)
        light = LightModel(cfg.light.cubemap_res, [cfg.render.light_name])
        renderer = PBR_renderer(cfg.render, cfg.scene.white_background)

        ##Loading
        checkpoint = torch.load(cfg.checkpoint_path)
        model_params = checkpoint["gaussians"]
        light_params = checkpoint["light"]
        loaded_iter = checkpoint["iteration"]
        gaussians.restore(model_params, cfg.optim)
        print(f"Loaded checkpoint from {cfg.checkpoint_path}")
        if os.path.exists(cfg.render.light_hdr_path):
            light.load_env_map(cfg.render.light_name, cfg.render.light_hdr_path)
            print(f"Loaded light from {cfg.render.light_hdr_path}")
        else:
            light_params = checkpoint["light"]
            light.load_state_dicts(light_params)
            print(f"Using learned light")

        light.save(save_dir=os.path.join(cfg.model_dir, "relight_envs"))
        render_set(
            gaussians=gaussians,
            light=light,
            views=scene.getTestCameras(),
            renderer=renderer,
            save_path=os.path.join(cfg.model_dir, "test"),
            env_name=cfg.render.save_name,
        )


if __name__ == "__main__":
    # Set up command line argument parser
    cfg = get_cfg()
    cfg.defrost()
    cfg.freeze()
    # Initialize system state (RNG)
    safe_state(cfg.quiet)

    render_sets(cfg)
