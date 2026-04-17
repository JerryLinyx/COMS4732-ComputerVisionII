import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def hw4_root() -> Path:
    return Path(__file__).resolve().parents[1]


def select_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def mse_to_psnr(mse: float | torch.Tensor) -> float:
    if isinstance(mse, torch.Tensor):
        mse = float(mse.detach().cpu().item())
    mse = max(float(mse), 1e-10)
    return float(10.0 * np.log10(1.0 / mse))


def load_image(image_path: str | Path, max_image_dim: int | None = None) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    if max_image_dim is not None and max_image_dim > 0:
        width, height = image.size
        longest = max(width, height)
        if longest > max_image_dim:
            scale = max_image_dim / longest
            new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array)


def to_uint8_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).round().astype(np.uint8)


def save_image(image: torch.Tensor | np.ndarray, image_path: str | Path) -> None:
    Image.fromarray(to_uint8_image(image)).save(image_path)


def configure_matplotlib() -> None:
    mpl_dir = ensure_dir(hw4_root() / ".mplconfig")
    cache_dir = ensure_dir(hw4_root() / ".cache")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
