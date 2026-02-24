"""
Author: Yuxuan Lin (yl6061@columbia.edu), assisted by GPT-5.2
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCENE_DESCRIPTIONS: dict[str, str] = {
    "joshua": "Photos taken by me at Joshua Tree National Park.",
    "lake": "Photos taken by me at Lake Powell at Page.",
    "north": "Provided by staff (debug/example only; not a submission scene).",
}


def resize_to_height(im_rgb: np.ndarray, target_h: int) -> tuple[np.ndarray, float]:
    h, w = im_rgb.shape[:2]
    if h == target_h:
        return im_rgb.copy(), 1.0
    scale = float(target_h) / float(h)
    new_w = max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(im_rgb, (new_w, target_h), interpolation=interp)
    return resized, scale


def draw_points(
    im_rgb: np.ndarray,
    points_yx: np.ndarray,
    color_rgb: tuple[int, int, int],
    radius: int = 2,
) -> np.ndarray:
    if points_yx.size == 0:
        return im_rgb
    out = im_rgb.copy()
    h, w = out.shape[:2]
    ys = points_yx[:, 0].astype(np.int32, copy=False)
    xs = points_yx[:, 1].astype(np.int32, copy=False)
    r = int(radius)
    for dy in range(-r, r + 1):
        y = ys + dy
        valid_y = (y >= 0) & (y < h)
        if not np.any(valid_y):
            continue
        y = y[valid_y]
        x_base = xs[valid_y]
        for dx in range(-r, r + 1):
            x = x_base + dx
            valid_x = (x >= 0) & (x < w)
            if not np.any(valid_x):
                continue
            out[y[valid_x], x[valid_x]] = color_rgb
    return out


def side_by_side(
    im1: np.ndarray,
    im2: np.ndarray,
    gap: int = 20,
    bg_rgb: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    h = max(im1.shape[0], im2.shape[0])
    w = im1.shape[1] + gap + im2.shape[1]
    canvas = np.full((h, w, 3), bg_rgb, dtype=np.uint8)
    canvas[: im1.shape[0], : im1.shape[1]] = im1
    x2 = im1.shape[1] + gap
    canvas[: im2.shape[0], x2 : x2 + im2.shape[1]] = im2
    return canvas


def add_text(im_rgb: np.ndarray, lines: Iterable[str], xy: tuple[int, int] = (10, 10)) -> np.ndarray:
    pil = Image.fromarray(im_rgb)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()
    x, y = xy
    pad = 4
    line_h = font.getbbox("Ag")[3] + 2
    lines = list(lines)
    if not lines:
        return im_rgb

    max_w = max(draw.textlength(line, font=font) for line in lines)
    box = (x, y, x + int(max_w) + 2 * pad, y + int(line_h) * len(lines) + 2 * pad)
    draw.rectangle(box, fill=(0, 0, 0))
    yy = y + pad
    for line in lines:
        draw.text((x + pad, yy), line, fill=(255, 255, 255), font=font)
        yy += line_h
    return np.array(pil)


def save_image(path: Path, im_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, im_rgb)


def make_nndr_histogram(
    ratios: np.ndarray,
    threshold: float,
    out_path: Path,
    title: str,
    bins: int = 50,
    size: tuple[int, int] = (700, 420),
) -> None:
    w, h = size
    pil = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()

    ratios = ratios[np.isfinite(ratios)]
    ratios = ratios[(ratios >= 0.0) & (ratios <= 1.0)]
    hist, _ = np.histogram(ratios, bins=bins, range=(0.0, 1.0))
    max_count = int(hist.max()) if hist.size else 1

    margin_l, margin_r, margin_t, margin_b = 60, 20, 40, 60
    x0, y0 = margin_l, margin_t
    x1, y1 = w - margin_r, h - margin_b

    draw.line([(x0, y1), (x1, y1)], fill=(0, 0, 0), width=2)
    draw.line([(x0, y1), (x0, y0)], fill=(0, 0, 0), width=2)

    plot_w = x1 - x0
    plot_h = y1 - y0

    # Ticks + numeric labels
    tick_len = 6
    for t in np.linspace(0.0, 1.0, 6):
        xx = x0 + float(t) * plot_w
        draw.line([(xx, y1), (xx, y1 + tick_len)], fill=(0, 0, 0), width=1)
        label = f"{t:.1f}"
        tw = draw.textlength(label, font=font)
        draw.text((xx - tw / 2, y1 + tick_len + 2), label, fill=(0, 0, 0), font=font)

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yy = y1 - frac * plot_h
        draw.line([(x0 - tick_len, yy), (x0, yy)], fill=(0, 0, 0), width=1)
        val = int(round(frac * max_count))
        label = str(val)
        tw = draw.textlength(label, font=font)
        draw.text((x0 - tick_len - tw - 4, yy - 6), label, fill=(0, 0, 0), font=font)

    bin_w = plot_w / bins
    for i, count in enumerate(hist):
        if count <= 0:
            continue
        bx0 = x0 + i * bin_w
        bx1 = x0 + (i + 1) * bin_w
        bh = (count / max_count) * plot_h
        by0 = y1 - bh
        draw.rectangle([(bx0, by0), (bx1, y1)], fill=(60, 90, 220))

    thr = float(np.clip(threshold, 0.0, 1.0))
    tx = x0 + thr * plot_w
    draw.line([(tx, y0), (tx, y1)], fill=(220, 50, 50), width=3)

    draw.text((10, 10), title, fill=(0, 0, 0), font=font)
    draw.text((10, h - 25), f"NNDR threshold = {thr:.2f}", fill=(220, 50, 50), font=font)
    draw.text((w // 2 - 70, h - 48), "NNDR (1NN / 2NN)", fill=(0, 0, 0), font=font)
    draw.text((10, margin_t - 14), "count", fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def make_top5_descriptor_grid(
    patches1: np.ndarray,
    patches2: np.ndarray,
    matches: list[dict],
    out_path: Path,
    title: str,
    tile_px: int = 96,
    pad: int = 12,
) -> None:
    k = min(5, len(matches))
    if k == 0:
        pil = Image.new("RGB", (600, 200), (255, 255, 255))
        draw = ImageDraw.Draw(pil)
        font = ImageFont.load_default()
        draw.text((10, 10), title, fill=(0, 0, 0), font=font)
        draw.text((10, 60), "No matches found under threshold.", fill=(0, 0, 0), font=font)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(out_path)
        return

    header_h = 58
    cols = 3
    width = pad + cols * tile_px + (cols - 1) * pad + pad
    height = header_h + pad + k * tile_px + (k - 1) * pad + pad
    pil = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()

    draw.text((10, 10), title, fill=(0, 0, 0), font=font)
    draw.text((10, 28), "Rows: best matches by NNDR. Columns: img1 / img2 1NN / img2 2NN.", fill=(0, 0, 0), font=font)
    col_titles = ["img1 desc", "img2 1NN", "img2 2NN"]
    for c, txt in enumerate(col_titles):
        x = pad + c * (tile_px + pad)
        draw.text((x, header_h - 16), txt, fill=(0, 0, 0), font=font)

    for r in range(k):
        m = matches[r]
        idx1 = m["i1"]
        idx2 = m["i2"]
        idx2_2 = m["i2_2"]
        row_patches = [patches1[idx1], patches2[idx2], patches2[idx2_2]]
        y = header_h + pad + r * (tile_px + pad)
        draw.text((10, y + tile_px // 2 - 6), f"#{r+1}  ratio={m['ratio']:.3f}", fill=(0, 0, 0), font=font)

        for c, p in enumerate(row_patches):
            p_u8 = np.clip(p * 255.0, 0, 255).astype(np.uint8)
            p_big = cv2.resize(p_u8, (tile_px, tile_px), interpolation=cv2.INTER_NEAREST)
            tile = np.stack([p_big, p_big, p_big], axis=-1)
            x = pad + c * (tile_px + pad)
            pil.paste(Image.fromarray(tile), (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def make_match_visualization(
    im1_rgb: np.ndarray,
    im2_rgb: np.ndarray,
    coords1_yx: np.ndarray,
    coords2_yx: np.ndarray,
    matches: list[dict],
    out_path: Path,
    display_height: int,
    title: str | None = None,
    gap: int = 20,
) -> None:
    im1_small, s1 = resize_to_height(im1_rgb, display_height)
    im2_small, s2 = resize_to_height(im2_rgb, display_height)
    canvas = side_by_side(im1_small, im2_small, gap=gap)

    offset_x2 = im1_small.shape[1] + gap
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()

    c1 = np.round(coords1_yx.astype(np.float32) * s1).astype(np.int32)
    c2 = np.round(coords2_yx.astype(np.float32) * s2).astype(np.int32)

    matched1 = set()
    matched2 = set()
    for m in matches:
        matched1.add(m["i1"])
        matched2.add(m["i2"])

    r = 3
    for i, (y, x) in enumerate(c1):
        if i in matched1:
            continue
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(220, 50, 50))
    for j, (y, x) in enumerate(c2):
        if j in matched2:
            continue
        xx = offset_x2 + int(x)
        draw.ellipse((xx - r, y - r, xx + r, y + r), fill=(220, 50, 50))

    for m in matches:
        y1, x1 = c1[m["i1"]]
        y2, x2 = c2[m["i2"]]
        x2o = offset_x2 + int(x2)
        draw.line([(int(x1), int(y1)), (x2o, int(y2))], fill=(40, 200, 40), width=2)
        draw.ellipse((int(x1) - r, int(y1) - r, int(x1) + r, int(y1) + r), fill=(40, 200, 40))
        draw.ellipse((x2o - r, int(y2) - r, x2o + r, int(y2) + r), fill=(40, 200, 40))

    caption = title or f"Matches: {len(matches)} (green)  Unmatched: {len(c1)-len(matched1)}/{len(c2)-len(matched2)} (red)"
    draw.text(
        (10, 10),
        caption,
        fill=(255, 255, 255),
        font=font,
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def write_index_html(web_dir: Path, metas: list[dict]) -> None:
    parts: list[str] = []
    parts.append(
        """<!doctype html>
<!--
Author: Yuxuan Lin (yl6061@columbia.edu), assisted by GPT-5.2
-->
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>COMS4732 HW2 - Automatic Feature Matching</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji"; margin: 24px; color: #111; }
    h1 { margin: 0 0 8px; }
    .scene { margin: 28px 0 42px; padding: 18px 18px 8px; border: 1px solid #e5e7eb; border-radius: 12px; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 14px; }
    img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e5e7eb; }
    .scroll-x { overflow-x: auto; padding-bottom: 6px; }
    img.pano { height: 520px; width: auto; max-width: none; }
    .meta { font-size: 14px; color: #374151; }
    .tech { font-size: 14px; color: #111827; line-height: 1.35; margin: 12px 0 8px; }
    .tech ul { margin: 8px 0 0; padding-left: 18px; }
    code { background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <h1>HW2: Automatic Feature Matching Across Images</h1>
  <div class="meta">
    Author: <code>Yuxuan Lin (yl6061@columbia.edu)</code>, assisted by <code>GPT-5.2</code>
  </div>
  <div class="tech">
    <div><b>Step choices</b></div>
    <ul>
      <li><b>Step 1 (Harris):</b> OpenCV <code>cornerHarris</code> with <code>block_size=2</code>, <code>ksize=3</code>, <code>k=0.04</code>. Visualization corners are 3x3 local maxima above a relative threshold.</li>
      <li><b>Step 2 (NMS):</b> Local-maximum suppression in a square window sized by image scale; window sizes are reported per image.</li>
      <li><b>Step 3 (Descriptor):</b> 40x40 grayscale patch, Gaussian blur (anti-alias), downsample to 8x8, bias/gain normalize (mean 0, std 1).</li>
      <li><b>Step 4 (Matching):</b> SSD (L2) on descriptors with optional L2-normalization, NNDR thresholding, plus mutual nearest-neighbor filtering.</li>
      <li><b>Extra credit:</b> Use a looser matching config for RANSAC homography, then warp + feather blend. Simple overlap-based color gain/bias correction is applied before blending.</li>
    </ul>
  </div>
"""
    )

    for meta in metas:
        scene = meta["scene"]
        img1 = meta["images"]["img1"]
        img2 = meta["images"]["img2"]
        counts = meta["counts"]
        nms_win1 = meta["params"]["nms"]["nms_window_img1"]
        nms_win2 = meta["params"]["nms"]["nms_window_img2"]
        nndr_thr = meta["params"]["matching"]["nndr_threshold"]
        harris = meta["params"]["harris"]
        desc = meta["params"]["descriptor"]
        nms = meta["params"]["nms"]
        matching = meta["params"]["matching"]
        parts.append('<div class="scene">')
        parts.append(f"  <h2>Scene: {scene}</h2>")
        scene_desc = SCENE_DESCRIPTIONS.get(scene)
        if scene_desc:
            parts.append(f'  <div class="meta">{scene_desc}</div>')
        parts.append(
            f'  <div class="meta">Images: <code>{img1}</code> + <code>{img2}</code> · '
            f'NMS window: img1={nms_win1}px, img2={nms_win2}px · NNDR threshold={nndr_thr:.2f} · '
            f"Matches={counts['matches']}</div>"
        )
        parts.append('  <div class="meta">')
        parts.append(
            "    Params: "
            f"Harris(block={harris['block_size']}, ksize={harris['ksize']}, k={harris['k']}) · "
            f"Descriptor(patch={desc['patch_size']}->desc={desc['desc_size']}, blur_sigma={desc['blur_sigma']}) · "
            f"Step1(3x3, thr_rel={nms['step1_threshold_rel']}) · "
            f"NMS(thr_rel={nms['nms_threshold_rel']}, divisor={nms['nms_divisor']}) · "
            f"Matching(metric=SSD, mutual={matching['mutual_nn']}, l2_norm={matching['l2_normalize']})"
        )
        parts.append("  </div>")
        parts.append('  <div class="grid">')
        parts.append(f'    <div><div class="meta">Original pair</div><img src="assets/{scene}/01_original.png" /></div>')
        parts.append(
            f'    <div><div class="meta">Step 1: Harris corners</div><img src="assets/{scene}/02_step1_harris.png" /></div>'
        )
        parts.append(f'    <div><div class="meta">Step 2: NMS corners</div><img src="assets/{scene}/03_step2_nms.png" /></div>')
        parts.append(
            f'    <div><div class="meta">Step 4.2: NNDR histogram</div><img src="assets/{scene}/04_nndr_hist.png" /></div>'
        )
        parts.append(
            f'    <div><div class="meta">Step 4.1: Top-5 matches (img1 / img2 1NN / img2 2NN)</div><img src="assets/{scene}/05_top5_descriptors.png" /></div>'
        )
        parts.append(
            f'    <div><div class="meta">Step 4.3: Match visualization (green lines = matches, red dots = unmatched)</div><img src="assets/{scene}/06_matches.png" /></div>'
        )

        if (web_dir / "assets" / scene / "07_panorama.png").exists():
            parts.append(
                f'    <div><div class="meta">Extra credit: panorama (scroll horizontally if needed)</div><div class="scroll-x"><img class="pano" src="assets/{scene}/07_panorama.png" /></div></div>'
            )
        if (web_dir / "assets" / scene / "07_inliers.png").exists():
            parts.append(
                f'    <div><div class="meta">Extra credit: RANSAC inlier matches</div><img src="assets/{scene}/07_inliers.png" /></div>'
            )

        parts.append("  </div>")
        parts.append("</div>")

    parts.append("</body></html>")
    (web_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")
