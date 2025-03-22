# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import cv2
import numpy as np
import torch
import torchvision
from matplotlib import cm
from PIL import Image


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[0, ...]]


def draw_depth(depth, save_path, cmap="turbo", min=None, max=None):
    near_plane = float(torch.min(depth)) if min is None else min
    far_plane = float(torch.max(depth)) if max is None else max

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)

    colored_image = apply_colormap(depth, cmap=cmap)
    colored_image = (colored_image * 255).byte()
    img = Image.fromarray(colored_image.cpu().numpy())
    img.save(save_path)


def erode(img_in, erode_size=4):
    img_out = np.copy(img_in)
    kernel = np.ones((erode_size, erode_size), np.uint8)
    img_out = cv2.erode(img_out, kernel, iterations=1)

    return img_out


def draw_normal(normals, save_path):
    """
    Converts a normal tensor of shape (3, H, W) in the range [-1, 1] to a normal map in the range [0, 1].
    Then saves the map as an image.

    Args:
    normals: (3, H, W) tensor containing the normal vectors in tangent space.
             Each channel should have values in the range [-1, 1].
    """
    # Ensure normals are in the expected range [-1, 1]

    # Map normals from the range [-1, 1] to [0, 1]
    normal_map = 0.5 * normals + 0.5

    # Convert to 8-bit by scaling to [0, 255]
    normal_map = (normal_map * 255).clamp(0, 255).byte()

    # Permute the tensor from (3, H, W) to (H, W, 3) to fit image format
    normal_map = normal_map.permute(1, 2, 0)

    # Convert to a PIL image and return
    img = Image.fromarray(normal_map.cpu().numpy())
    img.save(save_path)


def compute_luminance(img):
    # Assuming img is a 3D numpy array with shape (height, width, 3)
    # Use standard RGB to luminance conversion
    luminance = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    return luminance
