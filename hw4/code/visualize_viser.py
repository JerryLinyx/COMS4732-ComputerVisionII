import viser
import time
import numpy as np
import torch
import tyro
from dataclasses import dataclass

from src.dataset_3d import load_data, RaysData
from src.rendering import sample_along_rays


@dataclass
class Config:
    data_path: str = "data/lego_200x200.npz"
    near: float = None
    far: float = None
    num_samples_along_ray: int = None
    num_rays: int = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    port: int = 8080


def main(cfg: Config):
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, K = load_data(data_path=cfg.data_path)

    H, W = images_train.shape[1], images_train.shape[2]

    dataset = RaysData(images_train.to(cfg.device), K.to(cfg.device), c2ws_train.to(cfg.device))

    # Verify uvs aren't flipped
    uvs_start = 0
    uvs_end = 40_000
    sample_uvs = dataset.uvs[uvs_start:uvs_end].cpu()
    assert torch.all(
        images_train[0, sample_uvs[:, 1], sample_uvs[:, 0]]
        == dataset.gt_rgbs[uvs_start:uvs_end].cpu()
    )
    print("UVs assertion passed!")

    # Sample random rays from the first image
    num_pixels_per_image = H * W
    indices = np.random.randint(low=0, high=num_pixels_per_image, size=cfg.num_rays)

    rays_o = dataset.rays_o[indices]
    rays_d = dataset.rays_d[indices]

    points = sample_along_rays(
        r_os=rays_o,
        r_ds=rays_d,
        near=cfg.near,
        far=cfg.far,
        num_samples_along_ray=cfg.num_samples_along_ray,
        perturb=True,
        device=cfg.device,
    )  # (num_rays, num_samples, 3)

    # Convert to numpy for viser
    rays_o_np = rays_o.cpu().detach().numpy()
    rays_d_np = rays_d.cpu().detach().numpy()
    points_np = points.cpu().detach().numpy()
    images_np = images_train.cpu().detach().numpy()
    c2ws_np = c2ws_train.cpu().detach().numpy()
    K_np = K.cpu().detach().numpy()

    server = viser.ViserServer(port=cfg.port, share=True)

    fov = float(2 * np.arctan2(H / 2, K_np[0, 0]))
    aspect = float(W / H)

    for i, (image, c2w) in enumerate(zip(images_np, c2ws_np)):
        server.scene.add_camera_frustum(
            f"/cameras/{i}",
            fov=fov,
            aspect=aspect,
            scale=0.15,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=image,
        )

    for i, (o, d) in enumerate(zip(rays_o_np, rays_d_np)):
        positions = np.stack((o, o + d * cfg.far))
        server.scene.add_spline_catmull_rom(
            f"/rays/{i}",
            positions=positions,
        )

    server.scene.add_point_cloud(
        "/samples",
        colors=np.zeros_like(points_np).reshape(-1, 3),
        points=points_np.reshape(-1, 3),
        point_size=0.03,
    )

    print(f"Viser server running on port {cfg.port}. Press Ctrl+C to stop.")
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
