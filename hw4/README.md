# COMS4732 HW4: Neural Fields and Neural Radiance Fields

## Yuxuan Lin (yl6061@columbia.edu), assisted by GPT

This folder contains my HW4 implementation, experiment outputs, and the generated report page for:

- Part 1: fitting a 2D neural field to an image
- Part 2: fitting a NeRF to the provided Lego multi-view dataset

The report page is in:

- `hw4/web/index.html`

## Environment

The final training runs were executed with:

- conda env: `cv-mps-conda`
- device: `mps`

Note:

- On this machine, PyTorch MPS only worked correctly outside the Codex sandbox.
- The code also supports CPU fallback via `--device auto`.

## Main Files

- `hw4/code/models.py`
- `hw4/code/dataset_3d.py`
- `hw4/code/rendering.py`
- `hw4/code/train_part1.py`
- `hw4/code/train_nerf.py`
- `hw4/code/visualize_viser.py`

## How to Run

From the repo root:

### Part 1: provided image

```bash
conda activate cv-mps-conda
python hw4/code/train_part1.py \
  --image-path hw4/images/part1.jpg \
  --train-iters 2000 \
  --batch-size 10000 \
  --snapshot-every 250 \
  --output-dir hw4/outputs/part1/part1_ref_mps \
  --device auto
```

### Part 1: own image

```bash
conda activate cv-mps-conda
python hw4/code/train_part1.py \
  --image-path hw4/images/IMG_6873.jpg \
  --max-image-dim 512 \
  --train-iters 2000 \
  --batch-size 10000 \
  --snapshot-every 250 \
  --output-dir hw4/outputs/part1/img6873_mps \
  --device auto
```

### Part 2: final Lego NeRF run

```bash
conda activate cv-mps-conda
python hw4/code/train_nerf.py \
  --train-iters 2500 \
  --num-rays 4096 \
  --num-samples-along-ray 32 \
  --val-every 100 \
  --chunk-size 4096 \
  --render-test-video \
  --output-dir hw4/outputs/part2/mps_2500_r4096_s32 \
  --device auto \
  --seed 42
```

### Rays / samples visualization

```bash
conda activate cv-mps-conda
python hw4/code/visualize_viser.py
```

## Part 1: Final Config and Results

Implementation summary:

- Input: normalized 2D pixel coordinates `(u, v)`
- Positional encoding: sinusoidal PE
- Network: 4-layer MLP with ReLU activations and final Sigmoid
- Loss: MSE
- Metric: PSNR
- Optimizer: Adam

Final default config:

- PE frequencies: `10`
- Hidden width: `256`
- Hidden layers: `4`
- Learning rate: `1e-2`
- Batch size: `10000`
- Training iterations: `2000`

Hyperparameter comparison grid:

- PE frequencies: `{2, 10}`
- Widths: `{64, 256}`

Final results:

- Provided image (`part1.jpg`): `29.247 dB`
- Own image (`IMG_6873.jpg`, max dimension `512`): `31.337 dB`

Primary output folders:

- `hw4/outputs/part1/part1_ref_mps`
- `hw4/outputs/part1/img6873_mps`

## Part 2: Final Config and Results

Implementation summary:

- Part 2.1: convert pixel centers into ray origins and ray directions using `K` and `c2w`
- Part 2.2: uniformly sample points along rays between `near` and `far`, with perturbation during training
- Part 2.3: precompute a flattened ray dataset with `RaysData`
- Part 2.4: use a NeRF MLP with positional encoding for 3D points and view directions
- Part 2.5: composite per-sample colors and densities with discrete volume rendering

Final config:

- Dataset: `hw4/lego_200x200.npz`
- Near / far: `2.0 / 6.0`
- Samples per ray: `32`
- Rays per iteration: `4096`
- Training iterations: `2500`
- Learning rate: `5e-4`
- Hidden width: `256`
- Number of layers: `8`
- Skip layer: `4`
- XYZ PE frequencies: `10`
- Direction PE frequencies: `4`
- Validation images: `6`
- Seed: `42`

Important implementation note:

- I changed the density head activation from a plain `ReLU` to `Softplus` and initialized the density bias to a small positive value.
- This avoided a failure mode where the density branch collapsed to zero during longer runs, causing black renders.

Final result:

- Best validation PSNR: `23.103 dB`

Primary output folder:

- `hw4/outputs/part2/mps_2500_r4096_s32`

## Deliverables Checklist

### Part 1

Generated and packaged in `hw4/web/assets/`:

- Model architecture and hyperparameters described in `hw4/web/index.html`
- Training progression on the provided image
- Training progression on one of my own images
- 2x2 hyperparameter grid for the provided image
- 2x2 hyperparameter grid for my own image
- PSNR curve

Main report assets:

- `part1_ref_progression.png`
- `part1_ref_final.png`
- `part1_ref_hyperparameter_grid.png`
- `part1_ref_psnr_curve.png`
- `part1_own_progression.png`
- `part1_own_final.png`
- `part1_own_hyperparameter_grid.png`
- `part1_own_psnr_curve.png`

### Part 2

Generated and packaged in `hw4/web/assets/`:

- Brief description of how each part is implemented
- Rays and samples visualization with cameras
- Training progression visualization across iterations
- Validation PSNR curve
- Spherical rendering video using `c2ws_test`

Main report assets:

- `part2_vis_overview.png`
- `part2_vis_front.png`
- `part2_vis_oblique.png`
- `part2_progression.png`
- `part2_training_loss.png`
- `part2_validation_psnr.png`
- `part2_lego_test_video.gif`

## Report Page

Submission webpage:

- `hw4/web/index.html`

The page references only stable copied assets from `hw4/web/assets/`, rather than directly pointing into experiment output folders.

## Notes

- Part 2 reached the homework target of `23+ dB` validation PSNR.
- The current README documents the completed Part 1 and Part 2 workflow.
- Part 3 can be added later with its own dataset creation and COLMAP pipeline notes if needed.
