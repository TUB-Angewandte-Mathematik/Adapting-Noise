import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.utils import make_grid, save_image


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def to_uint8_rgb(imgs: torch.Tensor, size: Optional[int]) -> torch.Tensor:
    """Map float tensors in [-1,1] to uint8 RGB and optionally resize."""
    if imgs.dim() != 4:
        raise ValueError("Expected BCHW tensor for image conversion")
    imgs = imgs.clamp_(-1.0, 1.0)
    imgs = (imgs + 1.0) / 2.0
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)
    if size is not None:
        imgs = F.interpolate(imgs, size=(size, size), mode="bilinear", align_corners=False)
    return (imgs * 255.0).round().clamp(0, 255).to(torch.uint8)


class Uint8Dataset(Dataset):
    def __init__(self, tensor_uint8: torch.Tensor) -> None:
        self.tensor = tensor_uint8

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self.tensor.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:  # pragma: no cover - trivial
        return self.tensor[index]


def compute_fid(
    real: torch.Tensor,
    gen: torch.Tensor,
    *,
    device: torch.device,
    image_size: int,
    batch_size: int,
) -> float:
    try:
        import torch_fidelity
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("torch_fidelity is required for FID computation") from exc

    real_uint8 = to_uint8_rgb(real, image_size)
    gen_uint8 = to_uint8_rgb(gen, image_size)

    real_ds = Uint8Dataset(real_uint8.cpu())
    gen_ds = Uint8Dataset(gen_uint8.cpu())

    metrics = torch_fidelity.calculate_metrics(
        input1=real_ds,
        input2=gen_ds,
        fid=True,
        batch_size=batch_size,
        cuda=(device.type == "cuda"),
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"])


def save_image_grid(
    images: torch.Tensor,
    *,
    path: str,
    nrow: int = 8,
) -> np.ndarray:
    """Save a grid of [-1,1]-scaled images and return an array for logging."""
    directory = os.path.dirname(path)
    _ensure_dir(directory)
    images = images.clamp(-1.0, 1.0)
    save_image(((images + 1.0) / 2.0).cpu(), path, nrow=nrow)
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1.0, 1.0))
    return grid.permute(1, 2, 0).cpu().numpy()


def reshape_flat_samples(samples: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if samples.dim() == 2:
        return samples.view(samples.shape[0], *shape)
    if samples.shape[1:] == shape:
        return samples
    raise ValueError("Unexpected sample shape for image reshape")
