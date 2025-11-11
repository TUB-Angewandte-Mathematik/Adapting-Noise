from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import math
import numpy as np
import torch
from torch.special import i0e, i1e

try:
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - torchvision is optional
    datasets = None
    transforms = None
    DataLoader = None

Tensor = torch.Tensor


def _as_device_dtype(
    device: Optional[torch.device | str],
    dtype: Optional[torch.dtype],
) -> Tuple[torch.device, torch.dtype]:
    if device is None:
        resolved_device = torch.device("cpu")
    else:
        resolved_device = torch.device(device)
    if dtype is None:
        resolved_dtype = torch.get_default_dtype()
    else:
        resolved_dtype = dtype
    return resolved_device, resolved_dtype


def _log_normal_1d(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    var = std ** 2
    return -0.5 * ((x - mean) ** 2) / var - torch.log(std) - 0.5 * math.log(2 * math.pi)


def _logsumexp(a: Tensor, dim: int = -1) -> Tensor:
    return torch.logsumexp(a, dim=dim)


class BaseDistribution2D:
    """Interface for 2D distributions."""

    has_log_prob: bool = False

    def sample(
        self,
        n: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        raise NotImplementedError

    def log_prob(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Analytic log-density not available for this distribution.")


@dataclass
class CheckerboardStripes(BaseDistribution2D):
    low: float = -4.0
    high: float = 4.0

    has_log_prob: bool = True

    def _pick_square(self, n: int, device, dtype) -> Tensor:
        low_i = int(math.floor(self.low))
        high_i = int(math.floor(self.high))
        I = torch.arange(low_i, high_i, device=device)
        J = torch.arange(low_i, high_i, device=device)
        ii, jj = torch.meshgrid(I, J, indexing="ij")
        mask = ((ii + jj) % 2 == 0)
        valid = torch.stack([ii[mask], jj[mask]], dim=-1)
        idx = torch.randint(0, valid.shape[0], (n,), device=device)
        return valid[idx].to(dtype)

    def sample(
        self,
        n: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        device, dtype = _as_device_dtype(device, dtype)
        squares = self._pick_square(n, device, dtype)
        offs = torch.rand(n, 2, device=device, dtype=dtype)
        return squares + offs

    def log_prob(self, x: Tensor) -> Tensor:
        area_total = (self.high - self.low) ** 2
        log_const = -math.log(area_total / 2.0)
        i = torch.floor(x[..., 0])
        j = torch.floor(x[..., 1])
        inside = (x[..., 0] >= self.low) & (x[..., 0] <= self.high) & \
                 (x[..., 1] >= self.low) & (x[..., 1] <= self.high) & \
                 (((i + j) % 2) == 0)
        out = x.new_full(x.shape[:-1], float("-inf"))
        out[inside] = log_const
        return out


@dataclass
class GridGMM9(BaseDistribution2D):
    spacing: float = 1.0
    var: float = 0.0025
    weights: Optional[Sequence[float]] = None

    has_log_prob: bool = True

    def __post_init__(self):
        coords = (-float(self.spacing), 0.0, float(self.spacing))
        self._means = tuple((x, y) for x in coords for y in coords)
        if self.weights is None:
            w = [0.01, 0.1, 0.3, 0.2, 0.02, 0.15, 0.02, 0.15, 0.05]
        else:
            if len(self.weights) != len(self._means):
                raise ValueError(f"weights must have length {len(self._means)}")
            w = list(self.weights)
        total = sum(w)
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self.weights = tuple(ww / total for ww in w)
        self._logw = None

    def sample(
        self,
        n: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        device, dtype = _as_device_dtype(device, dtype)
        weights = torch.tensor(self.weights, device=device, dtype=dtype)
        cat = torch.distributions.Categorical(probs=weights)
        idx = cat.sample((n,))
        means = torch.tensor(self._means, device=device, dtype=dtype)
        std = math.sqrt(self.var)
        noise = std * torch.randn(n, 2, device=device, dtype=dtype)
        return means[idx] + noise

    def log_prob(self, x: Tensor) -> Tensor:
        if self._logw is None or self._logw.device != x.device or self._logw.dtype != x.dtype:
            self._logw = torch.log(torch.tensor(self.weights, device=x.device, dtype=x.dtype))
        means = x.new_tensor(self._means)
        diff = x[:, None, :] - means[None, :, :]
        quad = (diff ** 2).sum(dim=-1) / self.var
        log_comp = -0.5 * (quad + 2 * math.log(2 * math.pi * self.var))
        return _logsumexp(self._logw + log_comp, dim=-1)


@dataclass
class NealFunnel2D(BaseDistribution2D):
    sigma1: float = 3.0
    alpha: float = 1.0

    has_log_prob: bool = True

    def sample(
        self,
        n: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        device, dtype = _as_device_dtype(device, dtype)
        x1 = self.sigma1 * torch.randn(n, 1, device=device, dtype=dtype)
        std2 = torch.exp(0.5 * self.alpha * x1)
        x2 = std2 * torch.randn(n, 1, device=device, dtype=dtype)
        return torch.cat([x1, x2], dim=-1)

    def log_prob(self, x: Tensor) -> Tensor:
        x1, x2 = x[..., 0], x[..., 1]
        lp1 = _log_normal_1d(x1, x1.new_tensor(0.0), x1.new_tensor(self.sigma1))
        var2 = torch.exp(self.alpha * x1)
        lp2 = -0.5 * (x2 ** 2) / var2 - 0.5 * (math.log(2 * math.pi) + self.alpha * x1)
        return lp1 + lp2


class ZScoreWrapper(BaseDistribution2D):
    """Wrap a base sampler to operate in z-scored coordinates."""

    def __init__(self, base: BaseDistribution2D, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.base = base
        self.mean = mean
        self.std = std
        self.has_log_prob = getattr(base, "has_log_prob", False)

    def sample(
        self,
        n: int,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        raw = self.base.sample(n, device=device, dtype=dtype)
        mean = self.mean.to(raw.device, raw.dtype)
        std = self.std.to(raw.device, raw.dtype)
        return (raw - mean) / std

    def log_prob(self, x: Tensor) -> Tensor:
        if not hasattr(self.base, "log_prob"):
            raise AttributeError("Wrapped sampler does not implement log_prob")
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype)
        raw = x * std + mean
        log_det = torch.log(std.abs()).sum()
        return self.base.log_prob(raw) - log_det

    def to_raw(self, x: Tensor) -> Tensor:
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype)
        return x * std + mean

    def __getattr__(self, attr):
        return getattr(self.base, attr)


class MNISTSampler:
    """Random batches from MNIST with optional flattening."""

    def __init__(
        self,
        *,
        root: str = "./data",
        train: bool = True,
        download: bool = True,
        transform=None,
        flatten: bool = True,
        preload_batch_size: int = 1024,
    ) -> None:
        if datasets is None or transforms is None or DataLoader is None:
            raise ImportError(
                "torchvision is required for the MNIST sampler but is not available"
            )

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )

        dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )

        loader = DataLoader(dataset, batch_size=preload_batch_size, shuffle=False)
        data_chunks = []
        label_chunks = []
        for images, lbls in loader:
            if flatten:
                images = images.view(images.shape[0], -1)
            data_chunks.append(images)
            label_chunks.append(lbls)

        if not data_chunks:
            raise RuntimeError("MNIST dataset is empty or failed to load.")

        self.data = torch.cat(data_chunks, dim=0).contiguous()
        self.labels = torch.cat(label_chunks, dim=0).contiguous()
        self.flatten = flatten
        self.image_shape = (1, 28, 28)
        self.dim = self.data.shape[1] if flatten else self.data.shape[1:]

    def sample(
        self,
        n: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        device, dtype = _as_device_dtype(device, dtype)
        idx = torch.randint(0, self.data.shape[0], (n,))
        batch = self.data[idx].to(device=device, dtype=dtype)
        return batch

    def sample_with_labels(
        self,
        n: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = _as_device_dtype(device, dtype)
        idx = torch.randint(0, self.data.shape[0], (n,))
        images = self.data[idx].to(device=device, dtype=dtype)
        label_tensor = self.labels[idx].to(device=device)
        return images, label_tensor


class CIFAR10Sampler:
    """Random batches from CIFAR-10 with optional flattening."""

    num_classes: int = 10

    def __init__(
        self,
        *,
        root: str = "./data",
        train: bool = True,
        download: bool = True,
        transform=None,
        flatten: bool = False,
        preload_batch_size: int = 1024,
    ) -> None:
        if datasets is None or transforms is None or DataLoader is None:
            raise ImportError(
                "torchvision is required for the CIFAR-10 sampler but is not available"
            )

        if transform is None:
            transform = transforms.Compose(
                [    
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

        dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )

        loader = DataLoader(dataset, batch_size=preload_batch_size, shuffle=False)

        data_storage = []
        label_storage = []
        for images, labels in loader:
            if flatten:
                images = images.view(images.shape[0], -1)
            data_storage.append(images)
            label_storage.append(labels)

        self.data = torch.cat(data_storage, dim=0).contiguous()
        self.labels = torch.cat(label_storage, dim=0)
        self.flatten = bool(flatten)
        self.image_shape = (3, 32, 32)
        self.dim = self.data.shape[1] if self.flatten else self.data.shape[1:]

        # Cache per-class indices for deterministic subsets.
        class_indices = []
        for cls in range(self.num_classes):
            mask = torch.nonzero(self.labels == cls, as_tuple=False).view(-1)
            class_indices.append(mask)
        self.class_indices = class_indices
        self.class_counts = tuple(int(idx.shape[0]) for idx in class_indices)
        self.num_samples = int(self.data.shape[0])

    def _gather(
        self,
        idx: torch.Tensor,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = _as_device_dtype(device, dtype)
        batch = self.data[idx]
        labels = self.labels[idx]
        if dtype is not None and batch.dtype != dtype:
            batch = batch.to(dtype)
        return batch.to(device), labels.to(device)

    def sample(
        self,
        n: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        idx = torch.randint(0, self.data.shape[0], (n,), dtype=torch.long)
        batch, _ = self._gather(idx, device=device, dtype=dtype)
        return batch

    def sample_with_labels(
        self,
        n: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.data.shape[0], (n,), dtype=torch.long)
        return self._gather(idx, device=device, dtype=dtype)

    def sample_class_subset(
        self,
        cls: int,
        count: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (0 <= cls < self.num_classes):
            raise ValueError(f"Class index out of range: {cls}")
        pool = self.class_indices[cls]
        if count > int(pool.shape[0]):
            raise ValueError(
                f"Requested {count} samples for class {cls}, but only {int(pool.shape[0])} available"
            )
        choice = torch.randperm(pool.shape[0])[:count]
        idx = pool[choice]
        return self._gather(idx, device=device, dtype=dtype)


class TorchKacConstantSampler:
    """
    Mixture sampler for the 1D Kac displacement. Combines the atomic mass at
    ±c t with a continuous component tabulated via inverse CDF.
    """

    def __init__(
        self,
        a: float,
        c: float,
        T: float,
        M: int,
        K: int = 1024,
        *,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if T <= 0.0:
            raise ValueError("TorchKacConstantSampler requires T > 0.")
        if M <= 0 or K <= 0:
            raise ValueError("Lookup grid sizes M and K must be positive.")

        self.a = float(a)
        self.c = float(c)
        self.beta = self.a / self.c
        self.T = float(T)
        self.M = int(M)
        self.K = int(K)
        self.device, self.dtype = _as_device_dtype(device, dtype)

        t_grid = np.linspace(0.0, self.T, self.M + 1, dtype=np.float64)
        U = np.linspace(0.0, 1.0, self.K + 1, dtype=np.float64)
        X_table = np.zeros((self.M + 1, self.K + 1), dtype=np.float64)
        F_table = np.zeros((self.M + 1, self.K + 1), dtype=np.float64)

        table_device = torch.device("cpu")
        for j, t in enumerate(t_grid):
            ct = self.c * t
            if ct < 1e-16:
                X_table[j] = 0.0
                F_table[j] = U
                continue

            norm = -np.expm1(-self.a * t)
            r = np.sqrt(np.maximum(ct * ct - (ct * U) ** 2, 0.0))
            z = self.beta * r
            exp_fac = np.exp(z - self.a * t)

            z_t = torch.from_numpy(z).to(table_device, dtype=torch.float64)
            exp_fac_t = torch.from_numpy(exp_fac).to(table_device, dtype=torch.float64)

            term1 = self.beta * exp_fac_t * i0e(z_t)
            small_mask = z_t <= 1e-6
            series = 0.5 + (z_t ** 2) / 16.0 + (z_t ** 4) / 384.0
            ratio_t = torch.where(
                small_mask,
                series,
                i1e(z_t) / z_t,
            )
            term2 = self.beta * ct * exp_fac_t * ratio_t
            Kz = 0.5 * (term1 + term2)
            f = 2.0 * (Kz.cpu().numpy() / norm) * ct

            dU = U[1:] - U[:-1]
            F = np.empty(self.K + 1, dtype=np.float64)
            F[0] = 0.0
            F[1:] = np.cumsum(0.5 * (f[:-1] + f[1:]) * dU)
            if F[-1] > 0:
                F /= F[-1]
            else:
                F = U

            X_table[j] = ct * U
            F_table[j] = F

        quantiles = np.linspace(0.0, 1.0, self.K + 1, dtype=np.float64)
        invC = np.empty_like(X_table)
        for j in range(self.M + 1):
            invC[j] = np.interp(quantiles, F_table[j], X_table[j])

        self.t_grid = torch.tensor(t_grid, device=self.device, dtype=self.dtype)
        self.invC_table = torch.tensor(invC, device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def sample(self, t: Tensor, dim: int = 1) -> Tensor:
        orig_shape = t.shape
        t_flat = t.reshape(-1).to(self.device).to(self.dtype)
        total = t_flat.numel()
        expanded = t_flat.unsqueeze(1).expand(-1, dim).reshape(-1)

        mix_u = torch.rand(total * dim, device=self.device, dtype=self.dtype)
        cont_u = torch.rand_like(mix_u)
        p0 = torch.exp(-self.a * expanded)
        is_atomic = mix_u < p0

        dt = self.T / self.M
        j = torch.clamp((expanded / dt).floor().long(), 0, self.M - 1)
        alpha = (expanded - self.t_grid[j]) / dt

        ut = torch.clamp(cont_u, max=(self.K - 1) / self.K) * self.K
        k = ut.floor().long()
        frac = ut - k

        x0 = self.invC_table[j, k]
        x1 = self.invC_table[j, k + 1]
        y0 = self.invC_table[j + 1, k]
        y1 = self.invC_table[j + 1, k + 1]

        xj = x0 + frac * (x1 - x0)
        xj1 = y0 + frac * (y1 - y0)
        x_cont = xj + alpha * (xj1 - xj)

        magnitude = torch.where(is_atomic, self.c * expanded, x_cont)
        signs = torch.where(torch.rand_like(magnitude) < 0.5, 1.0, -1.0).to(self.dtype)
        return (signs * magnitude).view(*orig_shape, dim)


class TorchQuantileSampler:
    """Quantile sampler for the MMD baseline flow."""

    def __init__(
        self,
        b: float,
        *,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if b <= 0:
            raise ValueError("Parameter b must be positive for TorchQuantileSampler.")
        self.b = float(b)
        self.device, self.dtype = _as_device_dtype(device, dtype)

    @torch.no_grad()
    def sample(self, t: Tensor, dim: int = 1) -> Tuple[Tensor, Tensor]:
        t = t.to(self.device).to(self.dtype)
        orig_shape = t.shape
        U = torch.rand(*orig_shape, dim, device=self.device, dtype=self.dtype)
        scale = self.b * (1.0 - torch.exp(-t / self.b))
        scale = scale[..., None]
        return (2.0 * U - 1.0) * scale, U


def get_distribution(name: str, **kwargs):
    name = name.lower()
    if name in {"checker", "checkerboard"}:
        return CheckerboardStripes(**kwargs)
    if name in {"gridgmm", "gridgmm9", "gmmgrid"}:
        return GridGMM9(**kwargs)
    if name in {"funnel", "nealfunnel"}:
        base = NealFunnel2D(**kwargs)
        mean = torch.zeros(2)
        std = torch.tensor(
            [
                base.sigma1,
                math.exp(0.25 * (base.alpha ** 2) * (base.sigma1 ** 2)),
            ]
        )
        return ZScoreWrapper(base, mean, std)
    if name in {"mnist"}:
        return MNISTSampler(**kwargs)
    if name in {"cifar", "cifar10", "cifar-10"}:
        return CIFAR10Sampler(**kwargs)
    raise ValueError(f"Unknown distribution name: {name}")
