# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math

import numpy as np
import nvdiffrast.torch as dr
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from gaussian import GaussianModel
from light import LightModel
from light.NVDIFFREC.util import dot, reflect, safe_normalize
from scene.cameras import MiniCam
from utils.graphics_utils import (
    aces_film,
    blender_to_opengl,
    get_model_view_matrix_for_normal,
    inverse_aces_film,
    lin2rgb,
    normal_from_depth_image,
    reinhard_tm,
)
from utils.sh_utils import eval_sh


def prepare_normal(normal, old_cam):
    old_world_view = old_cam.world_view_transform.T
    new_world_view = get_model_view_matrix_for_normal(old_world_view, homo=True).cuda()
    hom_normal = torch.cat(
        [normal, torch.ones(normal.shape[0], 1).cuda()], dim=-1
    )  # N*4
    normal = torch.matmul(hom_normal, new_world_view.T)
    normal = hom_normal[:, 0:3]
    normal = safe_normalize(normal)
    return normal


def rendered_world2cam(viewpoint_cam, normal, alpha, bg_color):
    # normal: (3, H, W), alpha: (H, W), bg_color: (3)
    # normal_cam: (3, H, W)
    _, H, W = normal.shape
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()
    normal_world = normal.permute(1, 2, 0).reshape(-1, 3)  # (HxW, 3)
    normal_cam = (
        torch.cat([normal_world, torch.ones_like(normal_world[..., 0:1])], axis=-1)
        @ torch.inverse(torch.inverse(extrinsic_matrix).transpose(0, 1))[..., :3]
    )
    normal_cam = normal_cam.reshape(H, W, 3).permute(2, 0, 1)  # (H, W, 3)

    background = bg_color[..., None, None]
    normal_cam = normal_cam * alpha[None, ...] + background * (1.0 - alpha[None, ...])

    return normal_cam


# render 360 lighting for a single gaussian
def render_lighting(
    gaussian: GaussianModel, resolution=(512, 1024), sampled_index=None
):
    if gaussian.brdf_mode == "envmap":
        lighting = extract_env_map(gaussian.brdf_mlp, resolution)  # (H, W, 3)
        lighting = lighting.permute(2, 0, 1)  # (3, H, W)
    else:
        raise NotImplementedError

    return lighting


# render 360 lighting for a single gaussian
def render_lighting_2(
    gaussian: GaussianModel, resolution=(512, 1024), sampled_index=None
):
    if gaussian.brdf_mode == "envmap":
        lighting = extract_env_map(gaussian.brdf_mlp_2, resolution)  # (H, W, 3)
        lighting = lighting.permute(2, 0, 1)  # (3, H, W)
    else:
        raise NotImplementedError

    return lighting


def normalize_normal_inplace(normal, alpha):
    # normal: (3, H, W), alpha: (H, W)
    fg_mask = (alpha[None, ...] > 0.0).repeat(3, 1, 1)
    normal = torch.where(
        fg_mask, torch.nn.functional.normalize(normal, p=2, dim=0), normal
    )


class PBR_renderer:
    def __init__(self, render_cfg, white_background):
        self.brdf_lut = self.get_brdf_lut(render_cfg.brdf_lut_path).cuda()
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        self.white_background = True
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    def get_brdf_lut(self, brdf_lut_path) -> torch.Tensor:
        brdf_lut = torch.from_numpy(
            np.fromfile(brdf_lut_path, dtype=np.float32).reshape(1, 256, 256, 2)
        )
        return brdf_lut

    def get_rasterizer(
        self,
        viewpoint_camera,
        active_sh_degree=0,
        scaling_modifier=1.0,
        white_background=True,
    ):
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        if not white_background == self.white_background:
            bg_color = [1, 1, 1] if white_background else [0, 0, 0]
            bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        else:
            bg = self.background
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        return rasterizer

    def render(
        self,
        viewpoint_camera,
        gaussian: GaussianModel,
        light: LightModel,
        render_cfg,
        override_color=None,
        debug=False,
        speed=False,
        diffuse_only=False,
        relight=False,
        light_name=None,
        render_visual=False,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                gaussian.get_xyz,
                dtype=gaussian.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        means3D = gaussian.get_xyz
        means2D = screenspace_points
        opacity = gaussian.get_opacity
        normal = gaussian.get_normal(viewpoint_camera.camera_center)
        albedo = gaussian.get_albedo
        roughness = gaussian.get_roughness
        reflectance = gaussian.get_reflectance

        # view direction
        view_dir = safe_normalize(
            viewpoint_camera.camera_center.repeat(gaussian.num_points, 1)
            - gaussian.get_xyz
        )
        view_dir_opengl = blender_to_opengl(view_dir)
        normal_opengl = blender_to_opengl(normal)

        # Diffuse lookup
        if relight:
            env_name = light_name
        else:
            env_name = viewpoint_camera.env_name
        diffuse_envmap = light.get_diffuse(env_name)  # [6,16,16,3]

        diffuse_light = dr.texture(
            diffuse_envmap.unsqueeze(0),
            normal_opengl.unsqueeze(0).unsqueeze(0).contiguous(),
            filter_mode="linear",
            boundary_mode="cube",
        )  # [1, H, W, 3]

        # specular
        NoV = torch.clamp(dot(normal, view_dir), min=1e-6)
        # precalculated brdf intergation map in split sum approximation
        brdf_integration_uv = torch.cat((NoV, roughness), dim=-1)  # [1, N, 2]
        brdf_integration_lookup = dr.texture(
            self.brdf_lut,  # [1, 256, 256, 2]
            brdf_integration_uv.unsqueeze(0).unsqueeze(0).contiguous(),  # [1, 1, N, 2]
            filter_mode="linear",
            boundary_mode="clamp",
        )  # [1, 1, N, 2]

        # Roughness adjusted specular env lookup
        ref_dir_opengl = reflect(view_dir_opengl, normal_opengl)
        miplevel = light.get_mip(env_name, roughness)  # [1, H, W, 1]
        spec_env_maps = light.get_specular(env_name)

        spec_light = dr.texture(
            spec_env_maps[0].unsqueeze(0),  # [1, 6, env_res, env_res, 3]
            ref_dir_opengl.unsqueeze(0).unsqueeze(0).contiguous(),  # [1, 1, N, 3]
            mip=list(m.unsqueeze(0) for m in spec_env_maps[1:]),
            mip_level_bias=miplevel.transpose(0, 1).unsqueeze(0),  # [1, 1, W]
            filter_mode="linear-mipmap-linear",
            boundary_mode="cube",
        )

        # Compute aggregate lighting
        ##########################
        diffuse_rgb = albedo * diffuse_light
        F0 = reflectance
        spec_reflectance = (
            F0 * brdf_integration_lookup[..., 0:1] + brdf_integration_lookup[..., 1:2]
        )  # [1, 1, N, 3]
        specular_rgb = spec_light * spec_reflectance  # [1, H, W, 3]
        if diffuse_only:
            render_rgb = diffuse_rgb
        else:
            render_rgb = diffuse_rgb + specular_rgb  # [1, H, W, 3]

        render_rgb = render_rgb.clip(0, 1)
        render_rgb = lin2rgb(render_rgb)  # .clip(0.0, 1.0)

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rasterizer = self.get_rasterizer(
            viewpoint_camera
        )  # , gaussian.albedo_sh.active_order)
        shs = None
        scales = None
        rotations = None
        cov3D_precomp = None
        if render_cfg.compute_cov3D_python:
            cov3D_precomp = gaussian.get_covariance(scaling_modifier)
        else:
            scales = gaussian.get_scaling
            rotations = gaussian.get_rotation

        colors_precomp = render_rgb.squeeze()  # (N, 3)
        # render image
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        if render_visual:
            with torch.no_grad():
                # render diffuse image
                colors_precomp = diffuse_rgb
                diffuse_img, _ = rasterizer(
                    means3D=means3D,
                    means2D=means2D,
                    shs=shs,
                    colors_precomp=colors_precomp,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=cov3D_precomp,
                )
                # render specular image
                colors_precomp = specular_rgb
                specular_img, _ = rasterizer(
                    means3D=means3D,
                    means2D=means2D,
                    shs=shs,
                    colors_precomp=colors_precomp,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=cov3D_precomp,
                )
        else:
            diffuse_img = None
            specular_img = None

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        out = {
            "render": rendered_image,  # .clip(0.0, 1.0),
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "diffuse_img": diffuse_img,
            "specular_img": specular_img,
        }
        return out

    def render_depth_normal(self, gaussian, viewpoint_camera):
        # Rasterization
        screenspace_points = (
            torch.zeros_like(
                gaussian.get_xyz,
                dtype=gaussian.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        means3D = gaussian.get_xyz
        means2D = screenspace_points

        rasterizer = self.get_rasterizer(viewpoint_camera)
        shs = None
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = gaussian.get_scaling
        rotations = gaussian.get_rotation
        opacity = gaussian.get_opacity

        # get depth
        colors_precomp = gaussian.get_depth(
            viewpoint_camera.world_view_transform
        ).cuda()
        depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )[0]
        # get normal
        colors_precomp = gaussian.get_normal(
            viewpoint_camera.camera_center
        ).cuda()  # (N, 3)
        colors_precomp = prepare_normal(colors_precomp, viewpoint_camera)

        normal_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )[0]
        # get alpha
        rasterizer = self.get_rasterizer(viewpoint_camera, white_background=False)
        colors_precomp = torch.ones_like(means3D).cuda()
        alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )[0]
        # get normal_ref from depth
        intrinsic_matrix, extrinsic_matrix = viewpoint_camera.get_calib_matrix_nerf()
        normal_ref = normal_from_depth_image(
            depth_image[0, ...],
            intrinsic_matrix.to(depth_image.device),
            extrinsic_matrix.to(depth_image.device),
        )
        background = self.background[..., None, None]
        normal_ref = normal_ref.permute(2, 0, 1)
        normal_ref = normal_ref * alpha + background * (1.0 - alpha)

        return depth_image, normal_image, normal_ref, alpha

    def render_material(self, gaussian, viewpoint_camera):
        # Rasterization
        screenspace_points = (
            torch.zeros_like(
                gaussian.get_xyz,
                dtype=gaussian.get_xyz.dtype,
                requires_grad=False,
                device="cuda",
            )
            + 0
        )

        means3D = gaussian.get_xyz
        means2D = screenspace_points

        rasterizer = self.get_rasterizer(viewpoint_camera)
        shs = None
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = gaussian.get_scaling
        rotations = gaussian.get_rotation
        opacity = gaussian.get_opacity

        # get albedo
        colors_precomp = gaussian.get_albedo.cuda()
        albedo_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )[0]
        # get roughness
        colors_precomp = gaussian.get_roughness.repeat(1, 3).cuda()
        roughness_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )[0]
        # get reflectance
        if gaussian.get_reflectance.shape[1] == 1:
            colors_precomp = gaussian.get_reflectance.repeat(1, 3).cuda()
        else:
            colors_precomp = gaussian.get_reflectance.cuda()
        reflectance_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )[0]

        return albedo_image, roughness_image, reflectance_image

    def render_reflectance(self, gaussian, viewpoint_camera):
        # Rasterization
        screenspace_points = (
            torch.zeros_like(
                gaussian.get_xyz,
                dtype=gaussian.get_xyz.dtype,
                requires_grad=False,
                device="cuda",
            )
            + 0
        )

        means3D = gaussian.get_xyz
        means2D = screenspace_points

        rasterizer = self.get_rasterizer(viewpoint_camera)
        shs = None
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = gaussian.get_scaling
        rotations = gaussian.get_rotation
        opacity = gaussian.get_opacity

        # get reflectance
        colors_precomp = gaussian.get_reflectance.cuda()
        reflectance_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )[0]

        return reflectance_image
