from dataclasses import dataclass
from pathlib import Path

from utils import configure_matplotlib

configure_matplotlib()

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tyro

from dataset_3d import RaysData, load_data
from models import NeRFMLP
from rendering import predict_rgbs, render_image, sample_along_rays
from utils import ensure_dir, hw4_root, mse_to_psnr, save_image, select_device, to_uint8_image


@torch.no_grad()
def evaluate_validation_psnr(
    model: NeRFMLP,
    images_val: torch.Tensor,
    c2ws_val: torch.Tensor,
    K: torch.Tensor,
    near: float,
    far: float,
    num_samples_along_ray: int,
    chunk_size: int,
    device: torch.device,
    num_val_images: int,
):
    psnrs = []
    for image, c2w in zip(images_val[:num_val_images], c2ws_val[:num_val_images]):
        pred = render_image(
            model=model,
            K=K,
            c2w=c2w,
            image_height=image.shape[0],
            image_width=image.shape[1],
            near=near,
            far=far,
            num_samples_along_ray=num_samples_along_ray,
            chunk_size=chunk_size,
            device=str(device),
        )
        mse = F.mse_loss(pred, image.cpu()).item()
        psnrs.append(mse_to_psnr(mse))
    return float(sum(psnrs) / max(len(psnrs), 1))


@torch.no_grad()
def save_validation_render(
    model: NeRFMLP,
    image: torch.Tensor,
    c2w: torch.Tensor,
    K: torch.Tensor,
    near: float,
    far: float,
    num_samples_along_ray: int,
    chunk_size: int,
    device: torch.device,
    output_path: Path,
):
    pred = render_image(
        model=model,
        K=K,
        c2w=c2w,
        image_height=image.shape[0],
        image_width=image.shape[1],
        near=near,
        far=far,
        num_samples_along_ray=num_samples_along_ray,
        chunk_size=chunk_size,
        device=str(device),
    )
    save_image(pred, output_path)


def save_psnr_curve(psnrs: list[float], output_path: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(psnrs)
    plt.xlabel("Validation Checkpoint")
    plt.ylabel("PSNR (dB)")
    plt.title("Validation PSNR")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


@dataclass
class Config:
    data_path: str = str(hw4_root() / "lego_200x200.npz")
    output_dir: str = str(hw4_root() / "outputs" / "part2" / "lego")
    device: str = "auto"
    near: float = 2.0
    far: float = 6.0
    num_samples_along_ray: int = 32
    num_rays: int = 2048
    train_iters: int = 1000
    learning_rate: float = 5e-4
    hidden_dim: int = 256
    xyz_pe_frequencies: int = 10
    dir_pe_frequencies: int = 4
    num_layers: int = 8
    skip_layer: int = 4
    val_every: int = 100
    chunk_size: int = 4096
    num_val_images: int = 6
    render_test_video: bool = True
    test_video_fps: int = 12
    seed: int = 42


def main(cfg: Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = select_device(cfg.device)
    output_dir = ensure_dir(cfg.output_dir)
    renders_dir = ensure_dir(output_dir / "validation_renders")

    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, K = load_data(cfg.data_path)
    images_train = torch.from_numpy(images_train).to(torch.float32)
    c2ws_train = torch.from_numpy(c2ws_train).to(torch.float32)
    images_val = torch.from_numpy(images_val).to(torch.float32)
    c2ws_val = torch.from_numpy(c2ws_val).to(torch.float32)
    c2ws_test = torch.from_numpy(c2ws_test).to(torch.float32)
    K = K.to(torch.float32)

    train_dataset = RaysData(
        images=images_train.to(device),
        K=K.to(device),
        c2ws=c2ws_train.to(device),
        split="train",
        device=str(device),
    )

    model = NeRFMLP(
        xyz_pe_frequencies=cfg.xyz_pe_frequencies,
        dir_pe_frequencies=cfg.dir_pe_frequencies,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        skip_layer=cfg.skip_layer,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    val_psnrs: list[float] = []
    train_losses: list[float] = []

    for step in range(1, cfg.train_iters + 1):
        rays_o, rays_d, gt_rgbs = train_dataset.sample_rays(cfg.num_rays)
        xyzs = sample_along_rays(
            r_os=rays_o,
            r_ds=rays_d,
            near=cfg.near,
            far=cfg.far,
            num_samples_along_ray=cfg.num_samples_along_ray,
            perturb=True,
            device=str(device),
        )
        pred_rgbs = predict_rgbs(
            model=model,
            xyzs=xyzs,
            r_ds=rays_d,
            near=cfg.near,
            far=cfg.far,
            num_samples_along_ray=cfg.num_samples_along_ray,
        )
        loss = F.mse_loss(pred_rgbs, gt_rgbs)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        train_losses.append(float(loss.detach().cpu().item()))

        if step == 1 or step % cfg.val_every == 0 or step == cfg.train_iters:
            val_psnr = evaluate_validation_psnr(
                model=model,
                images_val=images_val,
                c2ws_val=c2ws_val,
                K=K,
                near=cfg.near,
                far=cfg.far,
                num_samples_along_ray=cfg.num_samples_along_ray,
                chunk_size=cfg.chunk_size,
                device=device,
                num_val_images=cfg.num_val_images,
            )
            val_psnrs.append(val_psnr)

            save_validation_render(
                model=model,
                image=images_val[0],
                c2w=c2ws_val[0],
                K=K,
                near=cfg.near,
                far=cfg.far,
                num_samples_along_ray=cfg.num_samples_along_ray,
                chunk_size=cfg.chunk_size,
                device=device,
                output_path=renders_dir / f"val0_step_{step:05d}.png",
            )
            print(f"step={step} train_loss={train_losses[-1]:.6f} val_psnr={val_psnr:.3f}")

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=150)
    plt.close()

    save_psnr_curve(val_psnrs, output_dir / "validation_psnr.png")
    torch.save(
        {
            "train_losses": train_losses,
            "val_psnrs": val_psnrs,
            "val_every": cfg.val_every,
            "seed": cfg.seed,
        },
        output_dir / "metrics.pt",
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(cfg),
            "image_height": int(images_train.shape[1]),
            "image_width": int(images_train.shape[2]),
            "K": K,
        },
        output_dir / "checkpoint.pt",
    )

    if cfg.render_test_video:
        frames = []
        for idx, c2w in enumerate(c2ws_test):
            frame = render_image(
                model=model,
                K=K,
                c2w=c2w,
                image_height=int(images_train.shape[1]),
                image_width=int(images_train.shape[2]),
                near=cfg.near,
                far=cfg.far,
                num_samples_along_ray=cfg.num_samples_along_ray,
                chunk_size=cfg.chunk_size,
                device=str(device),
            )
            frame_path = output_dir / f"test_frame_{idx:03d}.png"
            save_image(frame, frame_path)
            frames.append(to_uint8_image(frame))

        imageio.mimsave(
            output_dir / "lego_test_video.gif",
            frames,
            fps=cfg.test_video_fps,
            loop=0,
        )

    print(f"Device: {device}")
    if val_psnrs:
        print(f"Final validation PSNR: {val_psnrs[-1]:.3f} dB")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Config))
