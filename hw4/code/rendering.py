import torch
import torch.nn as nn
from dataset_3d import pixels_to_rays

def batched_T_i(sigmas: torch.Tensor, delta: torch.Tensor, device: str | None = None):
    if device is None:
        device = str(sigmas.device)
    start = torch.ones((sigmas.shape[0], 1, 1), device=device)

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
    r_os = r_os.to(device=device, dtype=torch.float32)
    r_ds = r_ds.to(device=device, dtype=torch.float32)

    num_rays = r_os.shape[0]
    step_size = (far - near) / num_samples_along_ray
    base_t = near + torch.arange(
        num_samples_along_ray, device=device, dtype=torch.float32
    ) * step_size
    ts = base_t.unsqueeze(0).expand(num_rays, -1)

    if perturb:
        ts = ts + torch.rand_like(ts) * step_size

    return r_os.unsqueeze(1) + ts.unsqueeze(-1) * r_ds.unsqueeze(1)


def volrend(
    sigmas: torch.Tensor,
    rgbs: torch.Tensor,
    near: float,
    far: float,
    num_samples_along_ray: int,
    device: str | None = None,
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
    if device is None:
        device = str(sigmas.device)

    step_size = (far - near) / num_samples_along_ray
    delta = torch.tensor([step_size], device=device)
    T = batched_T_i(sigmas, delta, device=device)

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


@torch.no_grad()
def render_rays(
    model: nn.Module,
    r_os: torch.Tensor,
    r_ds: torch.Tensor,
    near: float,
    far: float,
    num_samples_along_ray: int,
    chunk_size: int = 8192,
    device: str = "cuda",
):
    """Render a batch of rays in chunks to avoid OOM."""
    outputs = []
    for start in range(0, r_os.shape[0], chunk_size):
        end = start + chunk_size
        chunk_r_os = r_os[start:end].to(device=device, dtype=torch.float32)
        chunk_r_ds = r_ds[start:end].to(device=device, dtype=torch.float32)
        xyzs = sample_along_rays(
            r_os=chunk_r_os,
            r_ds=chunk_r_ds,
            near=near,
            far=far,
            num_samples_along_ray=num_samples_along_ray,
            perturb=False,
            device=device,
        )
        chunk_rgbs = predict_rgbs(
            model=model,
            xyzs=xyzs,
            r_ds=chunk_r_ds,
            near=near,
            far=far,
            num_samples_along_ray=num_samples_along_ray,
        )
        outputs.append(chunk_rgbs.detach().cpu())

    return torch.cat(outputs, dim=0)


@torch.no_grad()
def render_image(
    model: nn.Module,
    K: torch.Tensor,
    c2w: torch.Tensor,
    image_height: int,
    image_width: int,
    near: float,
    far: float,
    num_samples_along_ray: int,
    chunk_size: int = 8192,
    device: str = "cuda",
):
    """Render an image from a camera pose."""
    xs = torch.arange(image_width, dtype=torch.float32) + 0.5
    ys = torch.arange(image_height, dtype=torch.float32) + 0.5
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    uvs = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)

    r_os, r_ds = pixels_to_rays(K=K, c2w=c2w, uvs=uvs, device=device)
    rgbs = render_rays(
        model=model,
        r_os=r_os,
        r_ds=r_ds,
        near=near,
        far=far,
        num_samples_along_ray=num_samples_along_ray,
        chunk_size=chunk_size,
        device=device,
    )
    return rgbs.reshape(image_height, image_width, 3)
