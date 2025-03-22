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

import json
import os

import torch

from dataset import get_sceneinfo
from utils.camera_utils import cameraList_from_camInfos


class Scene:
    def __init__(self, scene_cfg):
        self.loaded_iter = None
        self.train_cameras = {}
        self.test_cameras = {}
        self.resolution_scales = scene_cfg.resolution_scales
        self.scene_info = get_sceneinfo(scene_cfg)

        for resolution_scale in self.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                self.scene_info.train_cameras, resolution_scale, scene_cfg
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                self.scene_info.test_cameras, resolution_scale, scene_cfg
            )

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    @property
    def get_point_cloud(self):
        return self.scene_info.point_cloud

    @property
    def get_cameras_extent(self):
        return self.scene_info.nerf_normalization["radius"]
