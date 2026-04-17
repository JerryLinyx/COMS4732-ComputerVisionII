from dataclasses import dataclass
from pathlib import Path

from utils import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import tyro

from models import NeuralField2D
from utils import ensure_dir, hw4_root, load_image, mse_to_psnr, save_image, select_device


class PixelSampler:
    def __init__(self, image: torch.Tensor, device: torch.device):
        self.image = image.to(device=device, dtype=torch.float32)
        self.device = device
        self.height, self.width = self.image.shape[:2]

        xs = torch.arange(self.width, device=device, dtype=torch.float32)
        ys = torch.arange(self.height, device=device, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        self.coords = torch.stack(
            (grid_x.reshape(-1) / self.width, grid_y.reshape(-1) / self.height), dim=-1
        )
        self.colors = self.image.reshape(-1, 3)

    def sample(self, batch_size: int):
        indices = torch.randint(0, self.coords.shape[0], (batch_size,), device=self.device)
        return self.coords[indices], self.colors[indices]


@torch.no_grad()
def render_full_image(
    model: NeuralField2D,
    height: int,
    width: int,
    device: torch.device,
    chunk_size: int = 65536,
) -> torch.Tensor:
    xs = torch.arange(width, device=device, dtype=torch.float32)
    ys = torch.arange(height, device=device, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    coords = torch.stack((grid_x.reshape(-1) / width, grid_y.reshape(-1) / height), dim=-1)

    preds = []
    for start in range(0, coords.shape[0], chunk_size):
        preds.append(model(coords[start : start + chunk_size]).detach().cpu())
    return torch.cat(preds, dim=0).reshape(height, width, 3)


def train_single_run(
    image: torch.Tensor,
    output_dir: Path,
    device: torch.device,
    pe_frequencies: int,
    hidden_dim: int,
    hidden_layers: int,
    learning_rate: float,
    train_iters: int,
    batch_size: int,
    snapshot_every: int,
):
    sampler = PixelSampler(image=image, device=device)
    model = NeuralField2D(
        pe_frequencies=pe_frequencies,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: list[float] = []
    snapshot_paths: list[Path] = []

    for step in range(1, train_iters + 1):
        coords, gt_rgbs = sampler.sample(batch_size)
        pred_rgbs = model(coords)
        loss = F.mse_loss(pred_rgbs, gt_rgbs)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        history.append(mse_to_psnr(loss))

        if step == 1 or step % snapshot_every == 0 or step == train_iters:
            rendered = render_full_image(model, sampler.height, sampler.width, device=device)
            snapshot_path = output_dir / f"render_step_{step:05d}.png"
            save_image(rendered, snapshot_path)
            snapshot_paths.append(snapshot_path)

    rendered = render_full_image(model, sampler.height, sampler.width, device=device)
    gt_image = image.cpu()
    final_mse = F.mse_loss(rendered, gt_image).item()
    final_psnr = mse_to_psnr(final_mse)

    plt.figure(figsize=(6, 4))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("PSNR (dB)")
    plt.title("Part 1 Training PSNR")
    plt.tight_layout()
    plt.savefig(output_dir / "psnr_curve.png", dpi=150)
    plt.close()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "pe_frequencies": pe_frequencies,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "height": sampler.height,
            "width": sampler.width,
        },
        output_dir / "checkpoint.pt",
    )

    return {
        "model": model,
        "history": history,
        "final_render": rendered,
        "final_psnr": final_psnr,
        "snapshot_paths": snapshot_paths,
    }


def save_grid(
    image: torch.Tensor,
    grid_output_path: Path,
    device: torch.device,
    pe_levels: list[int],
    widths: list[int],
    hidden_layers: int,
    learning_rate: float,
    train_iters: int,
    batch_size: int,
    snapshot_every: int,
):
    fig, axes = plt.subplots(len(pe_levels), len(widths), figsize=(4 * len(widths), 4 * len(pe_levels)))
    axes = axes.reshape(len(pe_levels), len(widths))

    for row, pe_level in enumerate(pe_levels):
        for col, width in enumerate(widths):
            cell_dir = ensure_dir(grid_output_path.parent / f"grid_pe{pe_level}_w{width}")
            result = train_single_run(
                image=image,
                output_dir=cell_dir,
                device=device,
                pe_frequencies=pe_level,
                hidden_dim=width,
                hidden_layers=hidden_layers,
                learning_rate=learning_rate,
                train_iters=train_iters,
                batch_size=batch_size,
                snapshot_every=snapshot_every,
            )
            axes[row, col].imshow(result["final_render"].numpy())
            axes[row, col].axis("off")
            axes[row, col].set_title(f"PE={pe_level}, W={width}\nPSNR={result['final_psnr']:.2f}")

    fig.tight_layout()
    fig.savefig(grid_output_path, dpi=150)
    plt.close(fig)


@dataclass
class Config:
    image_path: str = str(hw4_root() / "images" / "part1.jpg")
    output_dir: str = str(hw4_root() / "outputs" / "part1" / "default_run")
    device: str = "auto"
    max_image_dim: int = 512
    pe_frequencies: int = 10
    hidden_dim: int = 256
    hidden_layers: int = 4
    learning_rate: float = 1e-2
    train_iters: int = 2000
    batch_size: int = 10_000
    snapshot_every: int = 250
    make_grid: bool = True
    grid_pe_levels: str = "2,10"
    grid_widths: str = "64,256"


def main(cfg: Config):
    device = select_device(cfg.device)
    image = load_image(cfg.image_path, max_image_dim=cfg.max_image_dim)
    output_dir = ensure_dir(cfg.output_dir)

    result = train_single_run(
        image=image,
        output_dir=output_dir,
        device=device,
        pe_frequencies=cfg.pe_frequencies,
        hidden_dim=cfg.hidden_dim,
        hidden_layers=cfg.hidden_layers,
        learning_rate=cfg.learning_rate,
        train_iters=cfg.train_iters,
        batch_size=cfg.batch_size,
        snapshot_every=cfg.snapshot_every,
    )
    save_image(result["final_render"], output_dir / "final_render.png")

    if cfg.make_grid:
        pe_levels = [int(x) for x in cfg.grid_pe_levels.split(",") if x]
        widths = [int(x) for x in cfg.grid_widths.split(",") if x]
        save_grid(
            image=image,
            grid_output_path=output_dir / "hyperparameter_grid.png",
            device=device,
            pe_levels=pe_levels,
            widths=widths,
            hidden_layers=cfg.hidden_layers,
            learning_rate=cfg.learning_rate,
            train_iters=cfg.train_iters,
            batch_size=cfg.batch_size,
            snapshot_every=cfg.snapshot_every,
        )

    print(f"Device: {device}")
    print(f"Final PSNR: {result['final_psnr']:.3f} dB")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Config))
