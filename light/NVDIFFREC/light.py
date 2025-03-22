# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os

import numpy as np
import nvdiffrast.torch as dr
import torch

from . import renderutils as ru
from . import util

######################################################################################
# Utility functions
######################################################################################


class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return util.avg_pool_nhwc(cubemap, (2, 2))

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(
            6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda"
        )
        for s in range(6):
            gy, gx = torch.meshgrid(
                torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
            )
            # indexing='ij')
            v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(
                dout[None, ...] * 0.25,
                v[None, ...].contiguous(),
                filter_mode="linear",
                boundary_mode="cube",
            )
        return out


######################################################################################
# Split-sum environment map light source with automatic mipmap generation
######################################################################################


class EnvironmentLight(torch.nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base):
        super(EnvironmentLight, self).__init__()
        self.mtx = None
        self.base = torch.nn.Parameter(base.clone().detach(), requires_grad=True)
        self.register_parameter("env_base", self.base)

    def xfm(self, mtx):
        self.mtx = mtx

    def clone(self):
        return EnvironmentLight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def get_mip(self, roughness):
        return torch.where(
            roughness < self.MAX_ROUGHNESS,
            (
                torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS)
                - self.MIN_ROUGHNESS
            )
            / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS)
            * (len(self.specular) - 2),
            (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS)
            / (1.0 - self.MAX_ROUGHNESS)
            + len(self.specular)
            - 2,
        )

    def build_mips(self, cutoff=0.99, tonemap_fn=None):
        if tonemap_fn:
            self.specular = [tonemap_fn(self.base)]
        else:
            self.specular = [self.base]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])
        # self.diffuse = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)
        self.roughness_list = []
        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (
                self.MAX_ROUGHNESS - self.MIN_ROUGHNESS
            ) + self.MIN_ROUGHNESS
            self.roughness_list.append(roughness)
            self.specular[idx] = ru.specular_cubemap(
                self.specular[idx], roughness, cutoff
            )
        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.base - white))


######################################################################################
# Load and store
######################################################################################


# Load from latlong .HDR file
def _load_env_hdr_sdr(fn, scale=1.0):
    latlong_img = (
        torch.tensor(util.load_image(fn), dtype=torch.float32, device="cuda") * scale
    )
    cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])
    l = EnvironmentLight(cubemap)
    l.build_mips()

    return l


def load_env(fn, scale=1.0):
    if os.path.splitext(fn)[1].lower() == ".hdr":
        return _load_env_hdr_sdr(fn, scale)
    elif os.path.splitext(fn)[1].lower() == ".png":
        return _load_env_hdr_sdr(fn, scale)
    else:
        assert False, "Unknown envlight extension %s" % os.path.splitext(fn)[1]


def save_env_map(fn, light, diffuse=False, save_mip=False):
    assert isinstance(
        light, EnvironmentLight
    ), "Can only save EnvironmentLight currently"
    if isinstance(light, EnvironmentLight):
        if diffuse:
            color = util.cubemap_to_latlong(light.diffuse, [512, 1024])
            util.save_image_raw(fn, color.detach().cpu().numpy())
        elif save_mip:
            print("saving mip")
            for idx in range(len(light.specular) - 1):
                print(f"saving mip {idx}")
                roughness = (idx / (len(light.specular) - 2)) * (
                    light.MAX_ROUGHNESS - light.MIN_ROUGHNESS
                ) + light.MIN_ROUGHNESS
                color = util.cubemap_to_latlong(light.specular[idx], [512, 1024])
                save_path = fn.replace(".hdr", f"level{idx}_rough{roughness}.hdr")
                print(f"saving mip {idx} to {save_path}")
                util.save_image_raw(
                    save_path,
                    color.detach().cpu().numpy(),
                )
            color = util.cubemap_to_latlong(light.specular[-1], [512, 1024])
            save_path = fn.replace(".hdr", f"level5_diffuse.hdr")
            print(f"saving mip 5 to {save_path}")
            util.save_image_raw(
                save_path,
                color.detach().cpu().numpy(),
            )
        else:
            color = util.cubemap_to_latlong(light.base, [512, 1024])
            util.save_image_raw(fn, color.detach().cpu().numpy())


######################################################################################
# Create trainable env map with random initialization
######################################################################################


def create_trainable_env_rnd(base_res, scale=0.5, bias=0.25):
    base = (
        torch.rand(6, base_res, base_res, 3, dtype=torch.float32, device="cuda") * scale
        + bias
    )
    return EnvironmentLight(base)


def extract_env_map(light, resolution=[512, 1024]):
    assert isinstance(
        light, EnvironmentLight
    ), "Can only save EnvironmentLight currently"
    color = util.cubemap_to_latlong(light.base, resolution)
    return color
