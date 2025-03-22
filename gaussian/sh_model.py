import numpy as np
import torch

from utils.sh_utils import RGB2SH


class SphericalHarmonics(torch.nn.Module):
    def __init__(self, num_points, max_order=3):
        super(SphericalHarmonics, self).__init__()
        self.max_order = max_order
        self.active_order = 0
        num_coeffs = (self.max_order + 1) ** 2
        self.sh_coeffs_dc = torch.nn.Parameter(
            torch.zeros(num_points, 3, 1).contiguous()
        )
        self.sh_coeffs_rest = torch.nn.Parameter(
            torch.zeros(num_points, 3, num_coeffs - 1).contiguous()
        )

    def set_dc(self, color):
        with torch.no_grad():
            self.sh_coeffs_dc[:, :3, 0] = color

    def oneup_sh_order(self):
        if self.active_order < self.max_order:
            self.active_order += 1

    def forward(self, directions):
        """
        Compute the radiance given a set of input directions using SH coefficients.
        Args:
            directions: A tensor of shape (N, 2) where N is the number of directions,
                        and each direction is given as (theta, phi).
        Returns:
            Radiance values for each input direction.
        """
        output = torch.zeros(directions.size(0), device=directions.device)
        idx = 0
        for l in range(self.order + 1):
            for m in range(-l, l + 1):
                sh_basis = torch.tensor(
                    [spherical_harmonic(l, m, theta, phi) for theta, phi in directions],
                    device=directions.device,
                )
                output += self.sh_coeffs[idx] * sh_basis
                idx += 1
        return output
