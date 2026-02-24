# Author: Yuxuan Lin (yl6061@columbia.edu), assisted by GPT-5.2

# COMS4732 HW2: Automatic Feature Matching Across Images

This folder contains my implementation of HW2 (Harris → NMS → descriptor → matching) and the generated web visualizations.

## How to run

From the repo root:

```bash
python3 hw2/code/main.py --scenes joshua lake --display-height 600 --nndr-threshold 0.85
```

This writes:
- HTML: `hw2/web/index.html`
- Images + per-scene metadata: `hw2/web/assets/<scene>/`

To generate the extra-credit panorama:

```bash
python3 hw2/code/main.py --scenes joshua lake --display-height 600 --nndr-threshold 0.85 --panorama --panorama-height 800
```

## Scenes

Submission scenes:
- `joshua`: `joshua1.jpg` + `joshua2.jpg`
  - Photos taken by me at Joshua Tree National Park.
- `lake`: `lake1.jpg` + `lake2.jpg`
  - Photos taken by me at Lake Powell at Page.

Debug/staff scene:
- `north`: `north1.jpg` + `north2.jpg`
  - Provided by staff (debug/example only; not a submission scene).

## Implementation details / hyperparameters

### Step 1: Harris Corner Detection
- Harris response: OpenCV `cornerHarris(gray, block_size=2, ksize=3, k=0.04)`
- Step-1 corner visualization is produced by taking 3×3 local maxima and applying a relative threshold:
  - local max window: `3×3`
  - threshold: `response > 0.02 * max_response`

### Step 2: Non-Maximal Suppression (NMS)
- Simplified NMS: keep points that are the local maximum in a square window.
- NMS window size is scaled with image size:
  - `window = odd(round(min(H, W) / 20))`, clamped to `[15, 201]`
  - per-image chosen window sizes are recorded in `web/assets/<scene>/results.json`
- NMS threshold: `response > 0.0005 * max_response`

### Step 3: Descriptor Extraction (MOPS-like)
For each NMS corner:
- Extract a `40×40` axis-aligned grayscale patch (discard if it would go out of bounds).
- Apply Gaussian blur (anti-aliasing) with `sigma=1.0`.
- Downsample to `8×8` with area resampling.
- Bias/gain normalize the `8×8` patch:
  - subtract mean
  - divide by stddev

### Step 4: Feature Matching
- Similarity metric: **SSD / L2** between the 64D normalized descriptors.
- NNDR (Nearest-Neighbor Distance Ratio): `ratio = d(1NN) / d(2NN)` (using Euclidean distances).
- Threshold: `NNDR < 0.85` (also drawn on the histogram).
- Mutual nearest-neighbor filter: enabled (keeps matches only if best match is mutual).
- Descriptors are also L2-normalized before computing distances.

## Output / deliverables

Each scene contains:
- Step 1: original pair + Harris corners overlay
- Step 2: NMS corners overlay + reported NMS window sizes
- Step 4.2: NNDR histogram with threshold
- Step 4.1: top-5 matches shown as (img1 patch / img2 1NN patch / img2 2NN patch)
- Step 4.3: match visualization (green lines = matches, red dots = unmatched)

If `--panorama` is enabled and RANSAC succeeds (≥4 inliers):
- Extra credit: panorama image `07_panorama.png` (RANSAC homography + feather blending)
- Extra credit: RANSAC inlier match visualization `07_inliers.png`

Notes on extra credit implementation:
- Uses a slightly different matching configuration for stitching (more corners + more permissive NNDR), then relies on RANSAC to reject outliers.
- After warping, applies simple color gain/bias correction for the warped image (estimated from the overlap region), then feather blends using distance transforms.
