"""Velocity helper for the MMD baseline flow."""

from __future__ import annotations

import torch


@torch.no_grad()
def compute_velocity_mmd(
    x: torch.Tensor,
    t: torch.Tensor,
    b: float,
    *,
    disp: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the closed-form MMD velocity field.

    Args:
        x: State samples ``(..., d)``.
        t: Time values broadcastable to ``x`` (forward time variable).
        b: Positive scale parameter of the flow.
        disp: Optional displacement ``2u-1`` from the uniform quantile sampler.

    When ``disp`` is supplied we can use the simplified expression
    ``v(t, x) = disp * exp(-t / b)`` which avoids numerical issues as ``t → 0``.
    """

    b = float(b)
    work_dtype = x.dtype
    t = t.to(device=x.device, dtype=work_dtype)

    if disp is not None:
        disp = disp.to(device=x.device, dtype=work_dtype)
        scale = torch.exp(-t / b)
        while scale.ndim < disp.ndim:
            scale = scale.unsqueeze(-1)
        return (disp * scale).to(dtype=work_dtype)

    denom = torch.expm1(t / b)
    denom = denom.clamp_min(torch.finfo(work_dtype).tiny)
    while denom.ndim < x.ndim:
        denom = denom.unsqueeze(-1)
    v = x / (b * denom)
    return v.to(dtype=work_dtype)

