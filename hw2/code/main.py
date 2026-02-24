"""
Author: Yuxuan Lin (yl6061@columbia.edu), assisted by GPT-5.2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from scipy.ndimage import maximum_filter

from visual import (
    add_text,
    draw_points,
    make_match_visualization,
    make_nndr_histogram,
    make_top5_descriptor_grid,
    resize_to_height,
    save_image,
    side_by_side,
    write_index_html,
)


@dataclass(frozen=True)
class HarrisParams:
    block_size: int = 2
    ksize: int = 3
    k: float = 0.04


@dataclass(frozen=True)
class DescriptorParams:
    patch_size: int = 40
    desc_size: int = 8
    blur_sigma: float = 1.0


@dataclass(frozen=True)
class NmsParams:
    step1_window: int = 3
    step1_threshold_rel: float = 0.02
    nms_divisor: float = 20.0
    nms_min_window: int = 15
    nms_max_window: int = 201
    nms_threshold_rel: float = 0.0005


@dataclass(frozen=True)
class MatchParams:
    nndr_threshold: float = 0.93
    mutual_nn: bool = True
    l2_normalize: bool = True


@dataclass(frozen=True)
class StitchParams:
    enabled: bool = True
    panorama_height: int = 800
    nndr_threshold: float = 0.97
    mutual_nn: bool = True
    ransac_reproj_threshold_px: float = 3.0
    max_iters: int = 4000
    nms_divisor: float = 33.0
    nms_threshold_rel: float = 0.0005
    max_corners: int = 800
    descriptor_blur_sigma: float = 2.0


@dataclass(frozen=True)
class Scene:
    name: str
    img1: str
    img2: str


def _ensure_odd(n: int) -> int:
    return n if (n % 2 == 1) else (n + 1)


def _nms_window_size(shape_hw: tuple[int, int], params: NmsParams) -> int:
    h, w = shape_hw
    raw = int(round(min(h, w) / params.nms_divisor))
    raw = max(params.nms_min_window, min(params.nms_max_window, raw))
    return _ensure_odd(raw)


def _nms_window_size_custom(shape_hw: tuple[int, int], divisor: float, min_window: int, max_window: int) -> int:
    h, w = shape_hw
    raw = int(round(min(h, w) / float(divisor)))
    raw = max(int(min_window), min(int(max_window), raw))
    return _ensure_odd(raw)


def load_rgb(path: Path) -> np.ndarray:
    im = iio.imread(path)
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=-1)
    if im.shape[-1] == 4:
        im = im[:, :, :3]
    if im.dtype != np.uint8:
        im = np.clip(im, 0, 255).astype(np.uint8)
    return im


def rgb_to_gray_f32(im_rgb: np.ndarray) -> np.ndarray:
    gray_u8 = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)
    return gray_u8.astype(np.float32) / 255.0


def harris_response(gray_f32: np.ndarray, params: HarrisParams) -> np.ndarray:
    return cv2.cornerHarris(gray_f32, params.block_size, params.ksize, params.k)


def find_local_maxima(
    response: np.ndarray,
    window_size: int,
    threshold_rel: float,
    edge_discard: int,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if window_size % 2 != 1:
        raise ValueError("window_size must be odd")
    if threshold_rel < 0:
        raise ValueError("threshold_rel must be >= 0")

    r = response.astype(np.float32, copy=True)
    r[:edge_discard, :] = -np.inf
    r[-edge_discard:, :] = -np.inf
    r[:, :edge_discard] = -np.inf
    r[:, -edge_discard:] = -np.inf

    finite = r[np.isfinite(r)]
    if finite.size == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    threshold = float(threshold_rel) * float(np.max(finite))
    max_f = maximum_filter(r, size=window_size, mode="constant", cval=-np.inf)
    peaks = (r == max_f) & (r > threshold)

    ys, xs = np.nonzero(peaks)
    values = r[ys, xs]
    order = np.argsort(-values)
    if max_points is not None:
        order = order[:max_points]

    coords_yx = np.stack([ys[order], xs[order]], axis=1).astype(np.int32, copy=False)
    values = values[order].astype(np.float32, copy=False)
    return coords_yx, values


def _gaussian_blur(patch: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return patch
    # Let OpenCV pick the kernel size.
    return cv2.GaussianBlur(patch, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)


def extract_descriptors(
    gray_f32: np.ndarray,
    corners_yx: np.ndarray,
    params: DescriptorParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      descs: (N, 64) float32, bias/gain normalized
      coords_yx: (N, 2) int32, corresponding corner locations
      raw_patches: (N, 8, 8) float32 in [0,1], for visualization
    """
    if corners_yx.size == 0:
        return (
            np.zeros((0, params.desc_size * params.desc_size), dtype=np.float32),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, params.desc_size, params.desc_size), dtype=np.float32),
        )

    half = params.patch_size // 2
    h, w = gray_f32.shape

    descs: list[np.ndarray] = []
    coords: list[tuple[int, int]] = []
    raw_patches: list[np.ndarray] = []

    for y, x in corners_yx:
        y_i = int(y)
        x_i = int(x)
        if y_i - half < 0 or y_i + half > h or x_i - half < 0 or x_i + half > w:
            continue

        patch = gray_f32[y_i - half : y_i + half, x_i - half : x_i + half]
        if patch.shape != (params.patch_size, params.patch_size):
            continue

        blurred = _gaussian_blur(patch, sigma=params.blur_sigma)
        raw_small = cv2.resize(blurred, (params.desc_size, params.desc_size), interpolation=cv2.INTER_AREA).astype(
            np.float32, copy=False
        )

        v = raw_small.copy()
        v -= float(v.mean())
        std = float(v.std())
        if std < 1e-6:
            continue
        v /= std

        descs.append(v.reshape(-1))
        coords.append((y_i, x_i))
        raw_patches.append(raw_small)

    if not descs:
        return (
            np.zeros((0, params.desc_size * params.desc_size), dtype=np.float32),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, params.desc_size, params.desc_size), dtype=np.float32),
        )

    return (
        np.stack(descs, axis=0).astype(np.float32, copy=False),
        np.array(coords, dtype=np.int32),
        np.stack(raw_patches, axis=0).astype(np.float32, copy=False),
    )


def maybe_l2_normalize(desc: np.ndarray) -> np.ndarray:
    if desc.size == 0:
        return desc
    norms = np.linalg.norm(desc, axis=1, keepdims=True).astype(np.float32, copy=False)
    norms = np.maximum(norms, 1e-12)
    return (desc / norms).astype(np.float32, copy=False)


def dist_ssd(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    if x.ndim != 2 or c.ndim != 2:
        raise ValueError("x and c must be 2D arrays")
    if x.shape[1] != c.shape[1]:
        raise ValueError("Descriptor dims must match")
    d = (np.sum(x * x, axis=1, keepdims=True) + np.sum(c * c, axis=1)[None, :] - 2.0 * (x @ c.T)).astype(
        np.float32, copy=False
    )
    return np.maximum(d, 0.0)


def match_features(
    desc1: np.ndarray,
    desc2: np.ndarray,
    params: MatchParams,
) -> tuple[list[dict], np.ndarray]:
    """
    Returns:
      matches: list of dict with keys i1, i2, i2_2, ratio, d1, d2
      ratios_all: (N1,) NNDR ratios for all desc1 features
    """
    if desc1.shape[0] == 0 or desc2.shape[0] < 2:
        return [], np.zeros((desc1.shape[0],), dtype=np.float32)

    if params.l2_normalize:
        desc1 = maybe_l2_normalize(desc1)
        desc2 = maybe_l2_normalize(desc2)

    dmat = dist_ssd(desc1, desc2)  # (n1, n2)
    idx = np.argsort(dmat, axis=1)[:, :2]
    nn1 = idx[:, 0]
    nn2 = idx[:, 1]

    row = np.arange(dmat.shape[0])
    d1 = np.sqrt(dmat[row, nn1])
    d2 = np.sqrt(dmat[row, nn2])
    ratios = (d1 / (d2 + 1e-12)).astype(np.float32)

    keep = ratios < float(params.nndr_threshold)
    if params.mutual_nn:
        best_i_for_j = np.argmin(dmat, axis=0)
        keep &= (row == best_i_for_j[nn1])

    matches: list[dict] = []
    for i in np.where(keep)[0]:
        matches.append(
            {
                "i1": int(i),
                "i2": int(nn1[i]),
                "i2_2": int(nn2[i]),
                "ratio": float(ratios[i]),
                "d1": float(d1[i]),
                "d2": float(d2[i]),
            }
        )
    matches.sort(key=lambda m: m["ratio"])
    return matches, ratios


def estimate_homography_ransac(
    pts1_xy: np.ndarray,
    pts2_xy: np.ndarray,
    reproj_threshold_px: float,
    max_iters: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if pts1_xy.shape[0] < 4 or pts2_xy.shape[0] < 4:
        return None, None
    h, mask = cv2.findHomography(
        pts2_xy.astype(np.float32),
        pts1_xy.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(reproj_threshold_px),
        maxIters=int(max_iters),
        confidence=0.995,
    )
    if h is None or mask is None:
        return None, None
    mask = mask.reshape(-1).astype(bool)
    return h.astype(np.float64), mask


def warp_and_blend_panorama(im1_rgb: np.ndarray, im2_rgb: np.ndarray, h21: np.ndarray) -> np.ndarray:
    """
    Warp im2 into im1 coordinates using H21 (maps points from im2 -> im1) and blend.
    """
    h1, w1 = im1_rgb.shape[:2]
    h2, w2 = im2_rgb.shape[:2]
    corners2 = np.array([[0, 0], [w2 - 1, 0], [w2 - 1, h2 - 1], [0, h2 - 1]], dtype=np.float64)
    corners1 = np.array([[0, 0], [w1 - 1, 0], [w1 - 1, h1 - 1], [0, h1 - 1]], dtype=np.float64)

    def _warp_pts(pts_xy: np.ndarray, hmat: np.ndarray) -> np.ndarray:
        pts_h = np.concatenate([pts_xy, np.ones((pts_xy.shape[0], 1))], axis=1)
        out = (hmat @ pts_h.T).T
        out = out[:, :2] / out[:, 2:3]
        return out

    warped2 = _warp_pts(corners2, h21)
    all_pts = np.vstack([corners1, warped2])
    min_xy = np.floor(all_pts.min(axis=0)).astype(int)
    max_xy = np.ceil(all_pts.max(axis=0)).astype(int)
    tx = -min_xy[0]
    ty = -min_xy[1]
    out_w = int(max_xy[0] - min_xy[0] + 1)
    out_h = int(max_xy[1] - min_xy[1] + 1)

    translate = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
    h21_t = translate @ h21

    pano1 = cv2.warpPerspective(im1_rgb, translate, (out_w, out_h))
    pano2 = cv2.warpPerspective(im2_rgb, h21_t, (out_w, out_h))

    # Build masks by warping all-ones images (robust to valid black pixels).
    m1 = np.ones((h1, w1), dtype=np.uint8)
    m2 = np.ones((h2, w2), dtype=np.uint8)
    mask1 = (cv2.warpPerspective(m1, translate, (out_w, out_h)) > 0).astype(np.uint8)
    mask2 = (cv2.warpPerspective(m2, h21_t, (out_w, out_h)) > 0).astype(np.uint8)

    overlap = (mask1 == 1) & (mask2 == 1)
    if int(overlap.sum()) > 200:
        p1 = pano1[overlap].astype(np.float32)
        p2 = pano2[overlap].astype(np.float32)
        m1 = p1.mean(axis=0)
        m2 = p2.mean(axis=0)
        s1 = p1.std(axis=0)
        s2 = p2.std(axis=0)
        gain = s1 / (s2 + 1e-6)
        bias = m1 - gain * m2
        pano2 = np.clip(pano2.astype(np.float32) * gain[None, None, :] + bias[None, None, :], 0, 255).astype(np.uint8)

    # feather weights via distance transform
    dist1 = cv2.distanceTransform(mask1, distanceType=cv2.DIST_L2, maskSize=3)
    dist2 = cv2.distanceTransform(mask2, distanceType=cv2.DIST_L2, maskSize=3)
    w1_map = dist1 / (dist1 + dist2 + 1e-6)
    w2_map = 1.0 - w1_map

    only1 = (mask1 == 1) & (mask2 == 0)
    only2 = (mask2 == 1) & (mask1 == 0)
    both = (mask1 == 1) & (mask2 == 1)

    out = np.zeros_like(pano1)
    out[only1] = pano1[only1]
    out[only2] = pano2[only2]
    if np.any(both):
        w1b = w1_map[both][:, None]
        w2b = w2_map[both][:, None]
        blended = (w1b * pano1[both].astype(np.float32) + w2b * pano2[both].astype(np.float32)).astype(np.uint8)
        out[both] = blended

    union = (mask1 == 1) | (mask2 == 1)
    ys, xs = np.nonzero(union)
    if ys.size == 0:
        return out
    y0, y1c = int(ys.min()), int(ys.max()) + 1
    x0, x1c = int(xs.min()), int(xs.max()) + 1
    return out[y0:y1c, x0:x1c]


def process_scene(
    scene: Scene,
    images_dir: Path,
    assets_dir: Path,
    display_height: int,
    harris_params: HarrisParams,
    desc_params: DescriptorParams,
    nms_params: NmsParams,
    match_params: MatchParams,
    stitch_params: StitchParams,
) -> dict:
    im1 = load_rgb(images_dir / scene.img1)
    im2 = load_rgb(images_dir / scene.img2)
    gray1 = rgb_to_gray_f32(im1)
    gray2 = rgb_to_gray_f32(im2)

    edge = desc_params.patch_size // 2

    r1 = harris_response(gray1, harris_params)
    r2 = harris_response(gray2, harris_params)

    # Step 1 corners (dense-ish)
    step1_c1, _ = find_local_maxima(
        r1, window_size=_ensure_odd(nms_params.step1_window), threshold_rel=nms_params.step1_threshold_rel, edge_discard=edge
    )
    step1_c2, _ = find_local_maxima(
        r2, window_size=_ensure_odd(nms_params.step1_window), threshold_rel=nms_params.step1_threshold_rel, edge_discard=edge
    )

    # Step 2 NMS corners
    w1 = _nms_window_size(gray1.shape, nms_params)
    w2 = _nms_window_size(gray2.shape, nms_params)
    nms_c1, _ = find_local_maxima(r1, window_size=w1, threshold_rel=nms_params.nms_threshold_rel, edge_discard=edge)
    nms_c2, _ = find_local_maxima(r2, window_size=w2, threshold_rel=nms_params.nms_threshold_rel, edge_discard=edge)

    # Step 3 descriptors
    desc1, coords1, patches1 = extract_descriptors(gray1, nms_c1, desc_params)
    desc2, coords2, patches2 = extract_descriptors(gray2, nms_c2, desc_params)

    # Step 4 matching
    matches, ratios = match_features(desc1, desc2, match_params)

    # Visualizations
    scene_dir = assets_dir / scene.name
    scene_dir.mkdir(parents=True, exist_ok=True)

    # raw images
    im1_small, s1 = resize_to_height(im1, display_height)
    im2_small, s2 = resize_to_height(im2, display_height)
    save_image(scene_dir / "01_original.png", side_by_side(im1_small, im2_small))

    # Step 1 overlay
    c1s = np.round(step1_c1.astype(np.float32) * s1).astype(np.int32)
    c2s = np.round(step1_c2.astype(np.float32) * s2).astype(np.int32)
    step1_vis = side_by_side(draw_points(im1_small, c1s, (255, 60, 60), radius=1), draw_points(im2_small, c2s, (255, 60, 60), radius=1))
    step1_vis = add_text(step1_vis, [f"Step 1: Harris corners", f"img1: {len(step1_c1)} points", f"img2: {len(step1_c2)} points"])
    save_image(scene_dir / "02_step1_harris.png", step1_vis)

    # Step 2 overlay
    c1n = np.round(nms_c1.astype(np.float32) * s1).astype(np.int32)
    c2n = np.round(nms_c2.astype(np.float32) * s2).astype(np.int32)
    step2_vis = side_by_side(draw_points(im1_small, c1n, (255, 60, 60), radius=2), draw_points(im2_small, c2n, (255, 60, 60), radius=2))
    step2_vis = add_text(
        step2_vis,
        [
            f"Step 2: NMS (window ~ min_dim/{nms_params.nms_divisor:g})",
            f"win1={w1}px  img1: {len(nms_c1)} points",
            f"win2={w2}px  img2: {len(nms_c2)} points",
        ],
    )
    save_image(scene_dir / "03_step2_nms.png", step2_vis)

    # NNDR histogram
    make_nndr_histogram(
        ratios,
        threshold=match_params.nndr_threshold,
        out_path=scene_dir / "04_nndr_hist.png",
        title=f"{scene.name}: NNDR distribution (SSD/L2 on normalized 8x8 descriptors)",
    )

    # Top-5 descriptor matches (img1 / img2 1NN / img2 2NN)
    make_top5_descriptor_grid(
        patches1,
        patches2,
        matches,
        out_path=scene_dir / "05_top5_descriptors.png",
        title=f"{scene.name}: Top matches by NNDR",
    )

    # Match visualization (Option 2)
    make_match_visualization(
        im1,
        im2,
        coords1,
        coords2,
        matches,
        out_path=scene_dir / "06_matches.png",
        display_height=display_height,
        title=None,
    )

    stitch_meta: dict | None = None
    stitch_matches: list[dict] = []
    if stitch_params.enabled:
        w1s = _nms_window_size_custom(gray1.shape, stitch_params.nms_divisor, nms_params.nms_min_window, nms_params.nms_max_window)
        w2s = _nms_window_size_custom(gray2.shape, stitch_params.nms_divisor, nms_params.nms_min_window, nms_params.nms_max_window)
        stitch_c1, _ = find_local_maxima(
            r1,
            window_size=w1s,
            threshold_rel=float(stitch_params.nms_threshold_rel),
            edge_discard=edge,
            max_points=int(stitch_params.max_corners),
        )
        stitch_c2, _ = find_local_maxima(
            r2,
            window_size=w2s,
            threshold_rel=float(stitch_params.nms_threshold_rel),
            edge_discard=edge,
            max_points=int(stitch_params.max_corners),
        )

        stitch_desc_params = DescriptorParams(
            patch_size=desc_params.patch_size,
            desc_size=desc_params.desc_size,
            blur_sigma=float(stitch_params.descriptor_blur_sigma),
        )
        sdesc1, scoords1, _ = extract_descriptors(gray1, stitch_c1, stitch_desc_params)
        sdesc2, scoords2, _ = extract_descriptors(gray2, stitch_c2, stitch_desc_params)

        stitch_match_params = MatchParams(
            nndr_threshold=float(stitch_params.nndr_threshold),
            mutual_nn=bool(stitch_params.mutual_nn),
            l2_normalize=bool(match_params.l2_normalize),
        )
        stitch_matches, _ = match_features(sdesc1, sdesc2, stitch_match_params)

    if stitch_params.enabled and len(stitch_matches) >= 4:
        # Build match points
        pts1 = []
        pts2 = []
        for m in stitch_matches:
            y1, x1 = scoords1[m["i1"]]
            y2, x2 = scoords2[m["i2"]]
            pts1.append((x1, y1))
            pts2.append((x2, y2))
        pts1 = np.array(pts1, dtype=np.float64)
        pts2 = np.array(pts2, dtype=np.float64)

        # Downscale for panorama rendering (and scale points accordingly).
        pano_h = int(stitch_params.panorama_height)
        im1_p, s1p = resize_to_height(im1, pano_h)
        im2_p, s2p = resize_to_height(im2, pano_h)
        pts1_p = (pts1 * s1p).astype(np.float64)
        pts2_p = (pts2 * s2p).astype(np.float64)

        h21, inliers = estimate_homography_ransac(
            pts1_xy=pts1_p,
            pts2_xy=pts2_p,
            reproj_threshold_px=float(stitch_params.ransac_reproj_threshold_px),
            max_iters=int(stitch_params.max_iters),
        )
        if h21 is not None and inliers is not None and int(inliers.sum()) >= 4:
            inlier_matches = [stitch_matches[i] for i in np.where(inliers)[0]]
            inlier_matches.sort(key=lambda m: m["ratio"])

            pano = warp_and_blend_panorama(im1_p, im2_p, h21)
            save_image(scene_dir / "07_panorama.png", pano)

            inlier_pts1 = pts1_p[inliers]
            inlier_pts2 = pts2_p[inliers]
            coords1_inlier_yx = np.round(np.stack([inlier_pts1[:, 1], inlier_pts1[:, 0]], axis=1)).astype(np.int32)
            coords2_inlier_yx = np.round(np.stack([inlier_pts2[:, 1], inlier_pts2[:, 0]], axis=1)).astype(np.int32)
            make_match_visualization(
                im1_p,
                im2_p,
                coords1_inlier_yx,
                coords2_inlier_yx,
                [{"i1": i, "i2": i} for i in range(coords1_inlier_yx.shape[0])],
                out_path=scene_dir / "07_inliers.png",
                display_height=pano_h,
                title=f"RANSAC inliers: {int(inliers.sum())}/{len(stitch_matches)}",
            )

            stitch_meta = {
                "panorama_height": pano_h,
                "nndr_threshold": float(stitch_params.nndr_threshold),
                "ransac_reproj_threshold_px": float(stitch_params.ransac_reproj_threshold_px),
                "inliers": int(inliers.sum()),
                "total_matches": int(len(stitch_matches)),
                "nms_window_img1": int(w1s),
                "nms_window_img2": int(w2s),
                "corners_img1": int(scoords1.shape[0]),
                "corners_img2": int(scoords2.shape[0]),
                "descriptor_blur_sigma": float(stitch_desc_params.blur_sigma),
            }

    meta = {
        "scene": scene.name,
        "images": {"img1": scene.img1, "img2": scene.img2},
        "params": {
            "harris": asdict(harris_params),
            "descriptor": asdict(desc_params),
            "nms": asdict(nms_params) | {"nms_window_img1": w1, "nms_window_img2": w2},
            "matching": asdict(match_params),
            "stitching": asdict(stitch_params)
            | (stitch_meta or {"inliers": 0, "total_matches": int(len(stitch_matches)) if stitch_params.enabled else 0}),
        },
        "counts": {
            "step1_corners": {"img1": int(step1_c1.shape[0]), "img2": int(step1_c2.shape[0])},
            "nms_corners": {"img1": int(nms_c1.shape[0]), "img2": int(nms_c2.shape[0])},
            "descriptors": {"img1": int(desc1.shape[0]), "img2": int(desc2.shape[0])},
            "matches": int(len(matches)),
        },
    }
    (scene_dir / "results.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="COMS4732 HW2: automatic feature matching + visualizations.")
    parser.add_argument("--display-height", type=int, default=600, help="Height (px) of images in the web visualizations.")
    parser.add_argument("--nndr-threshold", type=float, default=0.85, help="NN distance ratio threshold (1NN/2NN).")
    parser.add_argument("--panorama", action="store_true", help="Generate extra-credit panorama (RANSAC homography + stitching).")
    parser.add_argument("--panorama-height", type=int, default=800, help="Height (px) for panorama rendering.")
    parser.add_argument("--scenes", nargs="*", default=["joshua", "lake"], help="Which scenes to run.")
    args = parser.parse_args()

    hw2_dir = Path(__file__).resolve().parents[1]
    images_dir = hw2_dir / "images"
    web_dir = hw2_dir / "web"
    assets_dir = web_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    all_scenes = {
        "north": Scene("north", "north1.jpg", "north2.jpg"),
        "joshua": Scene("joshua", "joshua1.jpg", "joshua2.jpg"),
        "lake": Scene("lake", "lake1.jpg", "lake2.jpg"),
    }

    missing = [s for s in args.scenes if s not in all_scenes]
    if missing:
        raise SystemExit(f"Unknown scene(s): {missing}. Available: {sorted(all_scenes.keys())}")

    harris_params = HarrisParams()
    desc_params = DescriptorParams()
    nms_params = NmsParams()
    match_params = MatchParams(nndr_threshold=float(args.nndr_threshold), mutual_nn=True, l2_normalize=True)
    stitch_params = StitchParams(enabled=bool(args.panorama), panorama_height=int(args.panorama_height))

    metas: list[dict] = []
    for name in args.scenes:
        metas.append(
            process_scene(
                all_scenes[name],
                images_dir=images_dir,
                assets_dir=assets_dir,
                display_height=int(args.display_height),
                harris_params=harris_params,
                desc_params=desc_params,
                nms_params=nms_params,
                match_params=match_params,
                stitch_params=stitch_params,
            )
        )

    write_index_html(web_dir=web_dir, metas=metas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
