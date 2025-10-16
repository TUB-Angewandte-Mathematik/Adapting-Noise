"""Velocity helper for the Kac baseline flow."""

from __future__ import annotations

import torch
from torch.special import i0e, i1e


@torch.no_grad()
def _ratio_I1_I0(z: torch.Tensor) -> torch.Tensor:
    """Stable evaluation of I1(z) / I0(z)."""

    small = z < 1e-12
    r_series = 0.5 * z
    r_full = i1e(z) / i0e(z)
    return torch.where(small, r_series, r_full)


@torch.no_grad()
def compute_velocity_kac(
    x: torch.Tensor,
    t: torch.Tensor,
    a: float,
    c: torch.Tensor | float,
    epsilon: float = 1e-6,
    T: float = 1.0,
) -> torch.Tensor:
    """Return the Kac process velocity with boundary clamp."""

    x = x.to(dtype=torch.float64)
    t = t.to(dtype=torch.float64, device=x.device)
    eps = float(epsilon)
    a = float(a)

    if not torch.is_tensor(c):
        c = torch.tensor(c, dtype=torch.float64, device=x.device)
    else:
        c = c.to(dtype=torch.float64, device=x.device)

    r2 = (c * t) ** 2 - x ** 2
    r = torch.sqrt(torch.clamp(r2, min=0.0))

    z = (a / c) * r
    R = _ratio_I1_I0(z)

    tiny = z < 1e-12
    invR = 1.0 / R
    denom_general = t + (r / c) * invR
    denom_tiny = t + 2.0 / a
    denom = torch.where(tiny, denom_tiny, denom_general)

    v_cont = x / denom

    threshold = c * t - eps * (t / T) * c
    boundary = torch.sign(x) * c
    mask = x.abs() >= threshold

    v = torch.where(mask, boundary, v_cont)
    return v.to(dtype=torch.float32)

