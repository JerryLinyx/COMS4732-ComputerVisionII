import torch
import torch.nn as nn
import numpy as np
from src.dataset_3d import pixels_to_rays

def batched_T_i(sigmas: torch.Tensor, delta: torch.Tensor, device: str = "cuda"):
    start = torch.ones((sigmas.shape[0], 1, 1)).to(device)

    exp_factors = torch.exp(-1 * sigmas * delta).to(device)

    return torch.cumprod(torch.cat([start, exp_factors[:, :-1]], dim=1), dim=1)


def sample_along_rays(
    r_os: torch.Tensor,
    r_ds: torch.Tensor,
    near: float,
    far: float,
    num_samples_along_ray: int,
    perturb: bool = True,
    device: str = "cuda",
):
    """Sample points along rays.

    Args:
        r_os: torch.Tensor of shape (num_pixels, 3) representing the ray origins
        r_ds: torch.Tensor of shape (num_pixels, 3) representing the ray directions
        near: float representing the near plane distance
        far: float representing the far plane distance
        num_samples_along_ray: int representing the number of samples to take along each ray
        perturb: bool representing whether to perturb the samples (True for training, False for testing)
        device: str representing the device to run on

    Returns:
        samples: torch.Tensor of shape (num_pixels, num_samples_along_ray, 3) representing
        the 3D positions of the samples along each ray
    """
    # TODO implement yourself
    return None


def volrend(
    sigmas: torch.Tensor,
    rgbs: torch.Tensor,
    near: float,
    far: float,
    num_samples_along_ray: int,
    device: str = "cuda",
):
    """Volume rendering along rays using the discrete approximation.

    Args:
        sigmas: torch.Tensor of shape (num_pixels, num_samples, 1) representing the density at each sample
        rgbs: torch.Tensor of shape (num_pixels, num_samples, 3) representing the color at each sample
        near: float representing the near plane distance
        far: float representing the far plane distance
        num_samples_along_ray: int representing the number of samples along each ray
        device: str representing the device to run on

    Returns:
        rendered_colors: torch.Tensor of shape (num_pixels, 3) representing the accumulated color for each ray
    """
    step_size = (far - near) / num_samples_along_ray
    delta = torch.tensor([step_size]).to(device)
    T = batched_T_i(sigmas, delta)

    weights = T * (1 - torch.exp(-1 * sigmas * delta)).to(device)

    return torch.sum(weights * rgbs, dim=1)


def predict_rgbs(
    model: nn.Module,
    xyzs: torch.Tensor,
    r_ds: torch.Tensor,
    near: float,
    far: float,
    num_samples_along_ray: int,
):
    """Predict colors from a model.

    Args:
        model: nn.Module representing the NeRF model
        xyzs: torch.Tensor of shape (num_pixels, num_samples, 3) representing sample positions along rays
        r_ds: torch.Tensor of shape (num_pixels, 3) representing the ray directions
        near: float representing the near plane distance
        far: float representing the far plane distance
        num_samples_along_ray: int representing the number of samples along each ray

    Returns:
        predicted_rgbs: torch.Tensor of shape (num_pixels, 3) representing the predicted colors
    """
    rgbs, sigmas = model(xyzs, r_ds)
    return volrend(sigmas, rgbs, near=near, far=far, num_samples_along_ray=num_samples_along_ray)
