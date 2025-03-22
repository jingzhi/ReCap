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

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p

from .sh_model import SphericalHarmonics


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.material_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._albedo = torch.empty(0)
        self._roughness = torch.empty(0)
        self._reflectance = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.num_points = 0
        self.setup_functions()

    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self._albedo,
            self._roughness,
            self._reflectance,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self._albedo,
            self._roughness,
            self._reflectance,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.num_points = self._xyz.shape[0]

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_normal(self, cam_center) -> torch.Tensor:
        normal = self.get_min_axis(cam_center)
        return normal

    @property
    def get_albedo(self) -> torch.Tensor:
        return self.material_activation(self._albedo)

    @property
    def get_roughness(self) -> torch.Tensor:
        return self.material_activation(self._roughness)

    @property
    def get_reflectance(self) -> torch.Tensor:
        return self.material_activation(self._reflectance)

    @property
    def get_refl_strength_to_total(self):
        s = self.get_reflectance
        s_norm = torch.norm(s, p=float("inf"), dim=1)
        return (s_norm > 0.1).sum() / s_norm.shape[0]

    def get_depth(self, world_view_transform):
        pts = self.get_xyz  # N*3
        hom_pts = torch.cat([pts, torch.ones(pts.shape[0], 1).cuda()], dim=-1)  # N*4
        tpts = torch.matmul(hom_pts, world_view_transform)
        z = tpts[:, 2:3]
        z = z.repeat(1, 3)
        return z

    def get_min_axis(self, cam_o):
        pts = self.get_xyz
        p2o = cam_o[None] - pts
        scales = self.get_scaling
        min_axis_id = torch.argmin(scales, dim=-1, keepdim=True)
        min_axis = torch.zeros_like(scales).scatter(1, min_axis_id, 1)

        rot_matrix = build_rotation(self.get_rotation)
        ndir = torch.bmm(rot_matrix, min_axis.unsqueeze(-1)).squeeze(-1)

        neg_msk = torch.sum(p2o * ndir, dim=-1) < 0
        ndir[neg_msk] = -ndir[neg_msk]  # make sure normal orient to camera
        return ndir

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        # position
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.num_points = fused_point_cloud.shape[0]
        print(f"Number of points at initialisation : {self.num_points}")

        # scale and rotation
        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((self.num_points, 4), device="cuda")
        rots[:, 0] = 1

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        # opacity
        opacity = inverse_sigmoid(
            0.1 * torch.ones((self.num_points, 1), dtype=torch.float, device="cuda")
        )
        self._opacity = nn.Parameter(opacity.requires_grad_(True))
        # normal

        normal = torch.zeros((self.num_points, 3), device="cuda")
        normal[:, 2] = 1
        self._normal = nn.Parameter(normal.requires_grad_(True))

        # albedo spherical harmonics
        # fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # self.albedo_sh = SphericalHarmonics(
        #    num_points=self.num_points, max_order=self.max_sh_degree
        # ).cuda()
        # self.albedo_sh.set_dc(fused_color)

        albedo = torch.zeros((self.num_points, 3), device="cuda")
        self._albedo = nn.Parameter(albedo.requires_grad_(True))
        # material
        roughness = torch.zeros((self.num_points, 1), device="cuda")
        reflectance = torch.zeros((self.num_points, 3), device="cuda")
        self._roughness = nn.Parameter(roughness.requires_grad_(True))
        self._reflectance = nn.Parameter(reflectance.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
            {
                "params": [self._albedo],
                "lr": training_args.material_lr_init,
                "name": "albedo",
            },
            {
                "params": [self._roughness],
                "lr": training_args.material_lr_init,
                "name": "roughness",
            },
            {
                "params": [self._reflectance],
                "lr": training_args.material_lr_init,
                "name": "reflectance",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self.material_scheduler_args = get_expon_lr_func(
            lr_init=training_args.material_lr_init,
            lr_final=training_args.material_lr_final,
            lr_delay_mult=training_args.material_lr_delay_mult,
            max_steps=training_args.material_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr
            elif param_group["name"] in [
                "albedo",
                "roughness",
                "reflectance",
            ]:
                lr = self.material_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def reset_opacity(self):
        opacity_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacity_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._albedo = optimizable_tensors["albedo"]
        self._roughness = optimizable_tensors["roughness"]
        self._reflectance = optimizable_tensors["reflectance"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_albedo,
        new_roughness,
        new_reflectance,
        new_opacity,
        new_scaling,
        new_rotation,
    ):
        d = {
            "xyz": new_xyz,
            "opacity": new_opacity,
            "albedo": new_albedo,
            "roughness": new_roughness,
            "reflectance": new_reflectance,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._albedo = optimizable_tensors["albedo"]
        self._roughness = optimizable_tensors["roughness"]
        self._reflectance = optimizable_tensors["reflectance"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_albedo = self._albedo[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)
        new_reflectance = self._reflectance[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz=new_xyz,
            new_albedo=new_albedo,
            new_roughness=new_roughness,
            new_reflectance=new_reflectance,
            new_opacity=new_opacity,
            new_scaling=new_scaling,
            new_rotation=new_rotation,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_albedo = self._albedo[selected_pts_mask]
        new_roughness = self._roughness[selected_pts_mask]
        new_reflectance = self._reflectance[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz=new_xyz,
            new_albedo=new_albedo,
            new_roughness=new_roughness,
            new_reflectance=new_reflectance,
            new_opacity=new_opacity,
            new_scaling=new_scaling,
            new_rotation=new_rotation,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)
        self.num_points = self._xyz.shape[0]

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
