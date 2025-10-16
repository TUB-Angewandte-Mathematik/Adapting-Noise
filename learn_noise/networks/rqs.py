
# simple_quantile_rqs.py
# -------------------------------------------------------------
# RQS-spline-based quantile ε(u, t | x0) with t-gating.
#
# Optional zero-mean mode:
#   If enforce_zero_mean=True, we center the spline residual by
#   subtracting m_t := E_Z[h_t(Z)] with Z~N(0,1), estimated using
#   fixed MC nodes stored as buffers (differentiable through h_t).
#
# Design:
# - Base term uses Φ^{-1}(u) (standard normal icdf), guaranteeing
#   correct t=1 behavior.
# - A monotone, bijective Rational-Quadratic Spline (RQS) h(z; θ)
#   acts on z = Φ^{-1}(u). We blend between identity and h via a
#   convex weight β(t) ∈ [0,1] to preserve monotonicity in u:
#
#     ε(u, t | x0) = t * [ (1 - β(t)) * z + β(t) * h(z; θ(t, x0)) ]
#                  = t * z + t * β(t) * (h(z; θ) - z)
#
#   with β(t) = σ(β_logit) * 4 t (1-t), so β(0)=β(1)=0.
# - h is an increasing RQS on ℝ with K bins and linear tails
#   (Durkan et al., "Neural Spline Flows"). Parameters θ are
#   conditioned on (t, x0) through a small MLP.
#
# Monotonicity:
# - z = Φ^{-1}(u) is strictly increasing in u.
# - h(z; θ) is strictly increasing in z.
# - A convex combination of increasing functions remains increasing.
#
# Endpoints:
# - t=0 → ε=0
# - t=1 → ε=Φ^{-1}(u) (independent of x0)
#
# The module exposes:
# - class SimpleQuantileRQS(nn.Module)
#   .forward(u, t, return_dqdt=False) -> ε or (ε, dε/dt)
#
# -------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Utilities: RQS implementation
# ------------------------------

@dataclass
class RQSConfig:
    n_bins: int = 16
    bound: float = 3.0
    min_bin_width: float = 1e-3
    min_bin_height: float = 1e-3
    min_derivative: float = 1e-5 #war 3


def _params_to_spline(
    raw_w: torch.Tensor, raw_h: torch.Tensor, raw_s: torch.Tensor, cfg: RQSConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Map raw parameters to valid RQS widths, heights, derivatives and knot positions.
    Shapes:
      raw_w: (..., K)
      raw_h: (..., K)
      raw_s: (..., K+1)
    Returns:
      xk, yk: (..., K+1)  knot coordinates in x/y (from -B to +B)
      w, h:   (..., K)    bin widths/heights (sum to 2B)
      s:      (..., K+1)  positive derivatives at knots
    """
    K = raw_w.shape[-1]
    B = cfg.bound

    # Normalize widths/heights to sum to 1, then scale to 2B, with min constraints
    w_probs = F.softmax(raw_w, dim=-1)
    h_probs = F.softmax(raw_h, dim=-1)

    # Enforce minimums while preserving total
    w = cfg.min_bin_width + (1.0 - cfg.min_bin_width * K) * w_probs
    h = cfg.min_bin_height + (1.0 - cfg.min_bin_height * K) * h_probs

    # Rescale to domain/range lengths 2B
    w = w * (2.0 * B)
    h = h * (2.0 * B)

    # Positive derivatives
    s = F.softplus(raw_s) + cfg.min_derivative

    # Cum-sums to compute knot positions
    cw = torch.cumsum(w, dim=-1)
    ch = torch.cumsum(h, dim=-1)

    # Prepend zeros and add left bound
    zeros = torch.zeros_like(cw[..., :1])
    xk = -B + torch.cat([zeros, cw], dim=-1)  # (..., K+1)
    yk = -B + torch.cat([zeros, ch], dim=-1)  # (..., K+1)

    return xk, yk, w, h, s


def _rqs_forward(
    x: torch.Tensor,
    xk: torch.Tensor,
    yk: torch.Tensor,
    w: torch.Tensor,
    h: torch.Tensor,
    s: torch.Tensor,
    cfg: RQSConfig,
) -> torch.Tensor:
    y, _ = _rqs_forward_with_derivative(x, xk, yk, w, h, s, cfg, return_dydx=False)
    return y


def _rqs_forward_with_derivative(
    x: torch.Tensor,
    xk: torch.Tensor,
    yk: torch.Tensor,
    w: torch.Tensor,
    h: torch.Tensor,
    s: torch.Tensor,
    cfg: RQSConfig,
    return_dydx: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Vectorized RQS forward pass. Optionally returns analytic derivative dy/dx.
    Linear tails use the endpoint slopes. Returns (y, dy/dx) both shaped like x
    when return_dydx=True and dy/dx=None otherwise.
    """
    K = w.shape[-1]
    BND = cfg.bound

    left_mask = x <= -BND
    right_mask = x >= BND
    mid_mask = ~(left_mask | right_mask)

    left_edges = xk[..., :-1]
    idx = torch.sum(x.unsqueeze(-1) >= left_edges, dim=-1) - 1
    idx = idx.clamp(min=0, max=K - 1)
    idx_exp = idx.unsqueeze(-1)

    xk_i = torch.gather(xk[..., :-1], dim=-1, index=idx_exp).squeeze(-1)
    yk_i = torch.gather(yk[..., :-1], dim=-1, index=idx_exp).squeeze(-1)
    w_i = torch.gather(w, dim=-1, index=idx_exp).squeeze(-1)
    h_i = torch.gather(h, dim=-1, index=idx_exp).squeeze(-1)

    s_i = torch.gather(s[..., :-1], dim=-1, index=idx_exp).squeeze(-1)
    idx_r = (idx + 1).clamp(max=K)
    s_ip1 = torch.gather(s, dim=-1, index=idx_r.unsqueeze(-1)).squeeze(-1)

    x_mid = torch.where(mid_mask, x, torch.zeros_like(x))
    xi = (x_mid - xk_i) / w_i.clamp(min=1e-12)
    xi = xi.clamp(0.0, 1.0)

    secant = h_i / w_i.clamp(min=1e-12)
    one_minus_xi = 1.0 - xi
    num = secant * xi * xi + s_i * xi * one_minus_xi
    den = secant + (s_i + s_ip1 - 2.0 * secant) * xi * one_minus_xi
    den_clamped = den.clamp(min=1e-12)
    y_mid = yk_i + h_i * (num / den_clamped)

    s_left = s[..., :1].squeeze(-1)
    s_right = s[..., -1].squeeze(-1)
    y_left = (-BND) + s_left * (x + BND)
    y_right = (BND) + s_right * (x - BND)

    y = torch.where(left_mask, y_left, torch.where(right_mask, y_right, y_mid))
    if return_dydx:
        deriv_num = secant * secant * (
            s_ip1 * xi * xi
            + 2.0 * secant * xi * one_minus_xi
            + s_i * one_minus_xi * one_minus_xi
        )
        deriv_mid = deriv_num / (den_clamped * den_clamped)
        dydx = torch.where(left_mask, s_left, torch.where(right_mask, s_right, deriv_mid))
        dydx = dydx.clamp_min(1e-12)
    else:
        dydx = None

    return y, dydx
