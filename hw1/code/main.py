"""
COMS4732 HW1

Authors: Yuxuan Lin (yl6061@columbia.edu), assisted by GPT-5.2

Running this script will:
1) Align all provided images using both single-scale (exhaustive search) and
   multi-resolution pyramid alignment.
2) Save color results into ../web/assets.
3) Print a summary table of displacements for the README/index.

Usage (from repo root):
    python code/main.py

Notes
- Blue channel is the reference; reported shifts are (dy, dx) applied to
  the moving channel to align with Blue.
- Scoring uses NCC (Normalized Cross Correlation) on gradient magnitude
  within the interior (12% border cropped). L2 is not used in this implementation.
- Single-scale: exhaustive search ±15 px (low-res baseline).
- Pyramid: 2x2 downsample, coarse ±25 px at coarsest level, then ±2 px per level.
"""

from __future__ import annotations


import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import imageio.v3 as iio
import numpy as np


# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]  # /HW1/hw1
DATA_DIR = REPO_ROOT / "coms4732_hw1_data"
EXTRA_DIR = DATA_DIR / "additional"
ASSET_DIR = REPO_ROOT / "web" / "assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AlignResult:
    name: str
    method: str
    g_shift: Tuple[int, int]
    r_shift: Tuple[int, int]
    output_path: Path
    runtime: float = 0.0


def crop_for_score(img: np.ndarray, pct: float = 0.12) -> np.ndarray:
    """Crop borders for robust scoring (wrap artifacts, scratches)."""
    h, w = img.shape
    dy, dx = int(h * pct), int(w * pct)
    return img[dy : h - dy, dx : w - dx]


def grad_mag(img: np.ndarray) -> np.ndarray:
    """Simple gradient magnitude using central differences."""
    gy, gx = np.gradient(img)
    return np.hypot(gx, gy)


def ncc_score(a: np.ndarray, b: np.ndarray, use_gradients: bool = True) -> float:
    """Normalized cross correlation between two same-shaped arrays."""
    if use_gradients:
        a = grad_mag(a)
        b = grad_mag(b)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -np.inf
    return float(np.sum(a * b) / denom)


def align_single(
    ref: np.ndarray,
    img: np.ndarray,
    search_radius: int = 15,
    base_shift: Tuple[int, int] = (0, 0),
    use_gradients: bool = True,
    crop_pct: float = 0.12,
) -> Tuple[int, int]:
    """Exhaustive search around base_shift within search_radius using NCC."""

    ref_c = crop_for_score(ref, pct=crop_pct)
    best_score = -np.inf
    best_shift = base_shift

    for dy in range(base_shift[0] - search_radius, base_shift[0] + search_radius + 1):
        for dx in range(base_shift[1] - search_radius, base_shift[1] + search_radius + 1):
            shifted = np.roll(img, shift=(dy, dx), axis=(0, 1))
            score = ncc_score(ref_c, crop_for_score(shifted, pct=crop_pct), use_gradients=use_gradients)
            if score > best_score:
                best_score = score
                best_shift = (dy, dx)

    return best_shift


def align_pyramid(
    ref: np.ndarray,
    img: np.ndarray,
    coarse_radius: int = 25,
    fine_radius: int = 2,
    min_size: int = 400,
    use_gradients: bool = True,
    crop_pct: float = 0.12,
) -> Tuple[int, int]:
    """Recursive image pyramid alignment returning (dy, dx)."""

    if min(ref.shape) <= min_size:
        return align_single(ref, img, search_radius=coarse_radius, use_gradients=use_gradients, crop_pct=crop_pct)

    def downsample_half(arr: np.ndarray) -> np.ndarray:
        # Simple 2x2 average downsample; crop odd dimensions
        h, w = arr.shape
        arr = arr[: h - h % 2, : w - w % 2]
        return arr.reshape(arr.shape[0] // 2, 2, arr.shape[1] // 2, 2).mean(axis=(1, 3))

    ref_small = downsample_half(ref)
    img_small = downsample_half(img)

    shift_small = align_pyramid(
        ref_small, img_small, coarse_radius, fine_radius, min_size, use_gradients=use_gradients, crop_pct=crop_pct
    )
    # Propagate shift to current scale
    propagated = (int(shift_small[0] * 2), int(shift_small[1] * 2))

    return align_single(ref, img, search_radius=fine_radius, base_shift=propagated, use_gradients=use_gradients, crop_pct=crop_pct)


def split_channels(im: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = int(np.floor(im.shape[0] / 3.0))
    b = im[:height]
    g = im[height : 2 * height]
    r = im[2 * height : 3 * height]
    return b, g, r


def load_grayscale(im_path: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
    im = iio.imread(im_path)
    if im.ndim == 3:
        # If imageio returns HxWxC, average to grayscale
        im = im.mean(axis=2)

    im = im.astype(np.float64)
    max_val = np.max(im)
    if max_val > 1.0:
        im = im / max_val
    im = np.clip(im, 0.0, 1.0)

    return im, im.shape


def align_image(im_path: Path, method: str = "pyramid") -> AlignResult:
    im, shape = load_grayscale(im_path)
    b, g, r = split_channels(im)

    if method == "single":
        g_shift = align_single(b, g, search_radius=15)
        r_shift = align_single(b, r, search_radius=15)
    elif method == "pyramid":
        g_shift = align_pyramid(b, g)
        r_shift = align_pyramid(b, r)
    else:
        raise ValueError(f"Unknown method: {method}")

    aligned_g = np.roll(g, shift=g_shift, axis=(0, 1))
    aligned_r = np.roll(r, shift=r_shift, axis=(0, 1))

    color = np.dstack([aligned_r, aligned_g, b])
    color = np.clip(color, 0, 1)

    out_name = f"{im_path.stem}_{method}.jpg"
    out_path = ASSET_DIR / out_name
    iio.imwrite(out_path, (color * 255).astype(np.uint8))

    return AlignResult(
        name=im_path.name,
        method=method,
        g_shift=g_shift,
        r_shift=r_shift,
        output_path=out_path,
    )


def run_batch():
    # Low-res set for single-scale baseline
    single_imgs = ["cathedral.jpg", "monastery.jpg", "tobolsk.jpg"]

    # Full set for pyramid (provided images only; excludes user-added extras)
    pyramid_imgs = [
        "cathedral.jpg",
        "monastery.jpg",
        "tobolsk.jpg",
        "church.tif",
        "emir.tif",
        "harvesters.tif",
        "icon.tif",
        "italil.tif",
        "lastochikino.tif",
        "lugano.tif",
        "melons.tif",
        "self_portrait.tif",
        "siren.tif",
        "three_generations.tif",
    ]

    # User-downloaded extras for the "Additional" section (placed in coms4732_hw1_data/additional/)
    extras_dir = DATA_DIR / "additional"
    extra_imgs = ["sobor.tif", "parovoz.tif", "khan.tif"]

    results: List[AlignResult] = []

    for name in single_imgs:
        print(f"[single] aligning {name}")
        t0 = time.perf_counter()
        r = align_image(DATA_DIR / name, method="single")
        r.runtime = time.perf_counter() - t0
        results.append(r)
        print(f"    done in {r.runtime:.2f}s")

    for name in pyramid_imgs:
        print(f"[pyramid] aligning {name}")
        t0 = time.perf_counter()
        r = align_image(DATA_DIR / name, method="pyramid")
        r.runtime = time.perf_counter() - t0
        results.append(r)
        print(f"    done in {r.runtime:.2f}s")

    extras_results: List[AlignResult] = []
    for name in extra_imgs:
        print(f"[extras] aligning {name}")
        t0 = time.perf_counter()
        path = extras_dir / name if (extras_dir / name).exists() else DATA_DIR / name
        r = align_image(path, method="pyramid")
        r.runtime = time.perf_counter() - t0
        extras_results.append(r)
        print(f"    done in {r.runtime:.2f}s")

    summary = {
        "single": [r.__dict__ for r in results if r.method == "single"],
        "pyramid": [r.__dict__ for r in results if r.method == "pyramid" and r.name not in extra_imgs],
        "extras": [r.__dict__ for r in extras_results],
    }

    results_json = REPO_ROOT / "web" / "assets" / "results.json"
    with open(results_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Also write a JS helper for file:// viewing
    results_js = REPO_ROOT / "web" / "assets" / "results.js"
    with open(results_js, "w") as f:
        f.write("window.RESULTS = ")
        json.dump(summary, f, default=str)
        f.write(";")

    print("\nAlignment complete. Outputs saved to web/assets.")
    for r in results:
        print(
            f"{r.name:22s} {r.method:7s} G{r.g_shift} R{r.r_shift} -> assets/{r.output_path.name}"
        )


if __name__ == "__main__":
    run_batch()
