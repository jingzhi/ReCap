import csv
import os
from argparse import ArgumentParser

import imageio.v2 as imageio
import numpy as np
import torch
import torchvision
from lpips import LPIPS
from PIL import Image
from tqdm import tqdm, trange

from utils.image_utils import psnr as get_psnr
from utils.loss_utils import ssim as get_ssim

lpips_fn = LPIPS(net="vgg").cuda()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The path to the output directory that stores the relighting results.",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        help="The path to the output directory that stores the relighting ground truth.",
    )
    parser.add_argument("--light_name", type=str, default="", help="save env name")
    parser.add_argument("--gt_name", type=str, default="", help="gt env name")
    parser.add_argument("--notes", type=str, default="", help="notes for exp")
    parser.add_argument(
        "--white_background",
        action="store_true",
        help="background for gt evaluation",
    )
    args = parser.parse_args()
    light_name = args.light_name
    gt_name = args.gt_name
    print(f"evaluation {light_name}")
    num_test = len(os.listdir(args.output_dir))
    psnr_avg = 0.0
    ssim_avg = 0.0
    lpips_avg = 0.0
    for idx in trange(num_test):
        with torch.no_grad():
            prediction = np.array(
                Image.open(os.path.join(args.output_dir, f"{idx:05}.png"))
            )
            prediction = (
                torch.from_numpy(prediction).cuda().permute(2, 0, 1) / 255.0
            )  # [3, H, W]
            gt_img = np.array(
                Image.open(
                    os.path.join(args.gt_dir, f"rgba_{idx:04}_{gt_name}.png")
                ).convert("RGBA")
            )
            alpha_mask = torch.from_numpy(gt_img[..., 3]).cuda() / 255.0
            gt_img = gt_img[..., :3]
            gt_img = (
                torch.from_numpy(gt_img).cuda().permute(2, 0, 1) / 255.0
            )  # [3, H, W]
            if args.white_background:
                bg = torch.ones_like(gt_img)
            else:
                bg = torch.zeros_like(gt_img)
            gt_img = gt_img * alpha_mask + bg * (1 - alpha_mask)
            psnr_avg += get_psnr(gt_img, prediction).mean().double()
            ssim_avg += get_ssim(gt_img, prediction).mean().double()
            lpips_avg += lpips_fn(gt_img, prediction).mean().double()

    print(f"{light_name} psnr_avg: {psnr_avg / num_test}")
    print(f"{light_name} ssim_avg: {ssim_avg / num_test}")
    print(f"{light_name} lpips_avg: {lpips_avg / num_test}")

    csv_file_path = os.path.join(
        os.path.dirname(args.output_dir), "../test_results.csv"
    )
    test_results = [
        {
            "env_name": f"{light_name}",
            "psnr": f"{psnr_avg/num_test}",
            "ssim": f"{ssim_avg/num_test}",
            "lpips": f"{lpips_avg/num_test}",
            "notes": args.notes,
        }
    ]
    file_exists = os.path.isfile(csv_file_path)
    print(csv_file_path)
    with open(csv_file_path, mode="a", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["env_name", "psnr", "ssim", "lpips", "notes"]
        )

        # Write the header only if the file does not exist
        if not file_exists:
            writer.writeheader()

        # Append the data rows
        writer.writerows(test_results)
