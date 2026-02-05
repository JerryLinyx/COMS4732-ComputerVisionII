# HW1 Reproduction Guide

## Environment
- Python 3.11 (Anaconda base)
- Dependencies: `numpy`, `imageio`, `Pillow`

## Data
Place the released archive at `coms4732_hw1_data/` (already included in this repo).
Channel order per plate: **Blue, Green, Red** from top to bottom.

## How to run
From the repo root:

```bash
python code/main.py
```

What the script does:
- Runs single exhaustive search (±15 px) on low-res JPGs.
- Runs multi-resolution pyramid alignment on the full set (JPG + TIF).
- Scores with NCC (Normalized Cross Correlation) on gradient magnitude (Sobel) after mean-centering; crops 12% border for robustness. L2 is not used.
- Pyramid: downsample by 2; coarse search ±25 px at coarsest level, refine ±2 px per level.
- Saves outputs to `web/assets/*_{single|pyramid}.jpg`.
- Writes `web/assets/results.json` and `web/assets/results.js` summarizing displacements (dy, dx) applied to align G/R to B.
- Extra examples processed: `sobor.tif`, `parovoz.tif`, `khan.tif`.

Expected runtime: ~2 minutes for the full batch on a MacBook Pro M3MAX.

## Viewing results
Open `web/index.html` in a browser (works offline). It auto-loads `assets/results.json` and displays all figures plus shifts and runtime.

## Packaging for Gradescope
```
hw1.zip
├─code/
│ ├─main.py
│ └─README.md
└─web/
   ├─index.html
   ├─assets/   (all generated JPGs + results.json)
   └─submission.pdf
```
Ensure the zip is < 100 MB (use JPG outputs and a compressed PDF).
