import os

import torch
import torch.optim as optim
from utils.graphics_utils import inverse_aces_film
from utils.schedulers import LogLerpScheduler
from utils.system_utils import mkdir_p

from .NVDIFFREC.light import (
    EnvironmentLight,
    create_trainable_env_rnd,
    extract_env_map,
    load_env,
    save_env_map,
)
from .NVDIFFREC.util import cubemap_to_latlong


class LightModel:
    def __init__(self, cubemap_res, env_names):
        self.cubemap_res = cubemap_res
        if isinstance(env_names, str):
            # for single environment as string
            self.env_names = [env_names]
        elif isinstance(env_names, list):
            # for any number of environments as list
            self.env_names = env_names
        else:
            raise TypeError("env_names must be a list or a string")
        envmap_list = [
            create_trainable_env_rnd(self.cubemap_res, scale=0.0, bias=0.5)
            for env in env_names
        ]
        self.envmaps = dict(zip(self.env_names, envmap_list))
        self.optimizer = None
        self.scheduler = None

    def training_setup(self, optim_cfg):
        param_group = []
        for key, val in self.envmaps.items():
            param_group.append(
                {
                    "params": list(val.parameters()),
                    "lr": optim_cfg.light_lr_init,
                    "name": f"envmap_{key}",
                }
            )
        self.optimizer = optim.Adam(param_group)
        self.scheduler = LogLerpScheduler(
            self.optimizer,
            lr_init=optim_cfg.light_lr_init,
            lr_final=optim_cfg.light_lr_final,
            max_steps=optim_cfg.iterations,
        )

    def train(self):
        for key, val in self.envmaps.items():
            val.train()

    def build_mips(self):
        for key, val in self.envmaps.items():
            val.build_mips()

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.clamp()

    def clamp(self, min=1e-8, max=None):
        with torch.no_grad():
            for key, val in self.envmaps.items():
                val.clamp_(min=min, max=max)

    def save(self, save_dir, suffix="", save_mip=False):
        os.makedirs(save_dir, exist_ok=True)
        for key, val in self.envmaps.items():
            save_path = f"{save_dir}/{key}_learned_base{suffix}.hdr"
            save_env_map(save_path, self.envmaps[key])
            save_path = f"{save_dir}/{key}_learned_diffuse{suffix}.hdr"
            save_env_map(save_path, self.envmaps[key], diffuse=True)
            if save_mip:
                save_path = f"{save_dir}/{key}_learned_mip{suffix}.hdr"
                save_env_map(save_path, self.envmaps[key], save_mip=True)

    def get_diffuse(self, env_name):
        return self.envmaps[env_name].diffuse

    def get_mip(self, env_name, roughness):
        return self.envmaps[env_name].get_mip(roughness)

    def get_mip_images(self, env_name):
        mipmaps = []
        light = self.envmaps[env_name]
        for idx in range(len(light.specular)):
            color = cubemap_to_latlong(light.specular[idx], [512, 1024])
            mipmaps.append(color)
        return mipmaps

    def get_specular(self, env_name):
        return self.envmaps[env_name].specular

    def load_env_map(self, env_name, hdr_env_path):
        self.envmaps[env_name] = load_env(hdr_env_path)

    def load_state_dicts(self, state_dicts):
        for key, val in self.envmaps.items():
            val.load_state_dict(state_dicts[key])
        self.build_mips()
        if self.optimizer:
            self.optimizer.load_state_dict(state_dicts["optimizer"])
            self.scheduler.load_state_dict(state_dicts["scheduler"])

    def get_state_dicts(self):
        state_dicts = {}
        for key, val in self.envmaps.items():
            state_dicts[key] = val.state_dict()
        state_dicts["optimizer"] = self.optimizer.state_dict()
        state_dicts["scheduler"] = self.scheduler.state_dict()
        return state_dicts

    def compute_mip_similarity(self, env_name):
        self.env_maps[env_name]
