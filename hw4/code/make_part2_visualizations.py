from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from dataset_3d import load_data, RaysData
from rendering import sample_along_rays


@dataclass
class Config:
    data_path: str = str(Path(__file__).resolve().parents[1] / "lego_200x200.npz")
    output_dir: str = str(Path(__file__).resolve().parents[1] / "web" / "assets")
    num_rays: int = 100
    num_samples_along_ray: int = 32
    near: float = 2.0
    far: float = 6.0
    source_camera_index: int = 0
    seed: int = 42
    preview_stride: int = 10
    preview_scale: float = 0.30
    frustum_depth: float = 0.32
    detail_preview_stride: int = 8
    detail_preview_scale: float = 1.85
    detail_frustum_depth: float = 0.55


def transform_points_to_local(points: np.ndarray, ref_c2w: np.ndarray) -> np.ndarray:
    rot = ref_c2w[:3, :3]
    trans = ref_c2w[:3, 3]
    return (rot.T @ (points - trans).T).T


def transform_dirs_to_local(dirs: np.ndarray, ref_c2w: np.ndarray) -> np.ndarray:
    rot = ref_c2w[:3, :3]
    return (rot.T @ dirs.T).T


def camera_geometry_local(
    image: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    ref_c2w: np.ndarray,
    *,
    frustum_depth: float,
    preview_stride: int,
    preview_scale: float,
):
    h, w = image.shape[:2]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    origin_local = transform_points_to_local(c2w[:3, 3][None, :], ref_c2w)[0]
    rot_local = ref_c2w[:3, :3].T @ c2w[:3, :3]

    corners_uv = np.array(
        [
            [0.0, 0.0],
            [w - 1.0, 0.0],
            [w - 1.0, h - 1.0],
            [0.0, h - 1.0],
        ],
        dtype=np.float32,
    )
    corners_cam = np.stack(
        [
            (corners_uv[:, 0] - cx) / fx * frustum_depth,
            (corners_uv[:, 1] - cy) / fy * frustum_depth,
            np.full(4, frustum_depth, dtype=np.float32),
        ],
        axis=1,
    )
    corners_local = (rot_local @ corners_cam.T).T + origin_local

    preview_img = image[::preview_stride, ::preview_stride]
    ph, pw = preview_img.shape[:2]
    xs = np.linspace(0, w - 1, pw, dtype=np.float32)
    ys = np.linspace(0, h - 1, ph, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    preview_cam = np.stack(
        [
            (grid_x - cx) / fx * frustum_depth * preview_scale,
            (grid_y - cy) / fy * frustum_depth * preview_scale,
            np.full_like(grid_x, frustum_depth),
        ],
        axis=-1,
    )
    preview_local = preview_cam.reshape(-1, 3) @ rot_local.T + origin_local
    preview_local = preview_local.reshape(ph, pw, 3)

    return {
        "origin": origin_local,
        "corners": corners_local,
        "preview_points": preview_local,
        "preview_image": preview_img,
    }


def draw_camera(ax, geom: dict, *, outline_lw: float = 1.5, alpha: float = 1.0):
    origin = geom["origin"]
    corners = geom["corners"]
    preview_pts = geom["preview_points"]
    preview_img = geom["preview_image"]

    plane = Poly3DCollection([corners], facecolor=(0, 0, 0, 0.03), edgecolor="none")
    ax.add_collection3d(plane)

    loop = np.vstack([corners, corners[:1]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="black", linewidth=outline_lw, alpha=alpha)
    for corner in corners:
        seg = np.vstack([origin, corner])
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="black", linewidth=outline_lw, alpha=alpha)
    ax.scatter(*origin, color="black", s=18, alpha=alpha)

    ax.plot_surface(
        preview_pts[:, :, 0],
        preview_pts[:, :, 1],
        preview_pts[:, :, 2],
        facecolors=np.clip(preview_img, 0.0, 1.0),
        shade=False,
        linewidth=0,
        antialiased=False,
        alpha=alpha,
    )


def style_axes(ax):
    ax.set_axis_off()
    ax.grid(False)
    ax.set_proj_type("persp", focal_length=0.8)


def fit_axes(ax, points: np.ndarray, *, pad: float):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    radius = max(radius + pad, 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def main(cfg: Config):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_train, c2ws_train, _, _, _, K = load_data(cfg.data_path)
    ref_c2w = c2ws_train[cfg.source_camera_index]

    dataset = RaysData(
        torch.as_tensor(images_train, dtype=torch.float32),
        K.to(torch.float32),
        torch.as_tensor(c2ws_train, dtype=torch.float32),
        device="cpu",
    )

    h, w = images_train.shape[1:3]
    base = cfg.source_camera_index * h * w
    indices = base + np.random.choice(h * w, size=cfg.num_rays, replace=False)
    rays_o = dataset.rays_o[indices].numpy()
    rays_d = dataset.rays_d[indices].numpy()
    sample_points = sample_along_rays(
        torch.from_numpy(rays_o),
        torch.from_numpy(rays_d),
        near=cfg.near,
        far=cfg.far,
        num_samples_along_ray=cfg.num_samples_along_ray,
        perturb=False,
        device="cpu",
    ).numpy()

    rays_o_local = transform_points_to_local(rays_o, ref_c2w)
    rays_d_local = transform_dirs_to_local(rays_d, ref_c2w)
    sample_points_local = transform_points_to_local(sample_points.reshape(-1, 3), ref_c2w).reshape(
        cfg.num_rays, cfg.num_samples_along_ray, 3
    )

    camera_positions_world = c2ws_train[:, :3, 3]
    source_pos = camera_positions_world[cfg.source_camera_index]
    dists = np.linalg.norm(camera_positions_world - source_pos[None, :], axis=1)
    sorted_idxs = np.argsort(dists)

    front_camera_idxs = sorted_idxs[:12]
    side_camera_idxs = sorted_idxs[:20]

    overview_geometries = [
        camera_geometry_local(
            image=images_train[i],
            c2w=c2ws_train[i],
            K=K.numpy(),
            ref_c2w=ref_c2w,
            frustum_depth=cfg.frustum_depth,
            preview_stride=cfg.preview_stride,
            preview_scale=cfg.preview_scale,
        )
        for i in range(len(images_train))
    ]
    detail_geometries = [
        camera_geometry_local(
            image=images_train[i],
            c2w=c2ws_train[i],
            K=K.numpy(),
            ref_c2w=ref_c2w,
            frustum_depth=cfg.detail_frustum_depth,
            preview_stride=cfg.detail_preview_stride,
            preview_scale=cfg.detail_preview_scale,
        )
        for i in range(len(images_train))
    ]

    # Overview
    fig = plt.figure(figsize=(7.2, 7.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    for i, geom in enumerate(overview_geometries):
        alpha = 1.0 if i in side_camera_idxs else 0.5
        draw_camera(ax, geom, outline_lw=1.0, alpha=alpha)
    for origin, direction in zip(rays_o_local, rays_d_local):
        line = np.stack([origin, origin + direction * cfg.far], axis=0)
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color="black", linewidth=1.2, alpha=0.75)
    ax.scatter(
        sample_points_local[:, :, 0].reshape(-1),
        sample_points_local[:, :, 1].reshape(-1),
        sample_points_local[:, :, 2].reshape(-1),
        s=10,
        c="#D4A017",
        depthshade=False,
    )
    overview_points = np.concatenate(
        [transform_points_to_local(camera_positions_world, ref_c2w), sample_points_local.reshape(-1, 3)], axis=0
    )
    fit_axes(ax, overview_points, pad=0.35)
    ax.view_init(elev=24, azim=-54)
    style_axes(ax)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(output_dir / "part2_vis_overview.png", dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Front
    fig = plt.figure(figsize=(7.2, 7.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    for i in front_camera_idxs:
        draw_camera(ax, detail_geometries[i], outline_lw=1.65, alpha=1.0)
    for origin, direction in zip(rays_o_local, rays_d_local):
        line = np.stack([origin, origin + direction * cfg.far], axis=0)
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color="black", linewidth=1.6, alpha=0.8)
    ax.scatter(
        sample_points_local[:, :, 0].reshape(-1),
        sample_points_local[:, :, 1].reshape(-1),
        sample_points_local[:, :, 2].reshape(-1),
        s=12,
        c="#D4A017",
        depthshade=False,
    )
    front_points = np.concatenate(
        [
            np.stack([detail_geometries[i]["origin"] for i in front_camera_idxs], axis=0),
            np.concatenate([detail_geometries[i]["corners"] for i in front_camera_idxs], axis=0),
            sample_points_local.reshape(-1, 3),
        ],
        axis=0,
    )
    fit_axes(ax, front_points, pad=0.10)
    ax.view_init(elev=10, azim=-78)
    style_axes(ax)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(output_dir / "part2_vis_front.png", dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Side / oblique with more cameras and wider scale.
    fig = plt.figure(figsize=(7.2, 7.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    for i in side_camera_idxs:
        draw_camera(ax, detail_geometries[i], outline_lw=1.55, alpha=1.0)
    for origin, direction in zip(rays_o_local, rays_d_local):
        line = np.stack([origin, origin + direction * cfg.far], axis=0)
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color="black", linewidth=1.45, alpha=0.78)
    ax.scatter(
        sample_points_local[:, :, 0].reshape(-1),
        sample_points_local[:, :, 1].reshape(-1),
        sample_points_local[:, :, 2].reshape(-1),
        s=12,
        c="#D4A017",
        depthshade=False,
    )
    side_points = np.concatenate(
        [
            np.stack([detail_geometries[i]["origin"] for i in side_camera_idxs], axis=0),
            np.concatenate([detail_geometries[i]["corners"] for i in side_camera_idxs], axis=0),
            sample_points_local.reshape(-1, 3),
        ],
        axis=0,
    )
    fit_axes(ax, side_points, pad=0.28)
    ax.view_init(elev=14, azim=-16)
    style_axes(ax)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(output_dir / "part2_vis_oblique.png", dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main(tyro.cli(Config))
