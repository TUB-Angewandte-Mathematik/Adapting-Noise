from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rqs import RQSConfig, _params_to_spline, _rqs_forward_with_derivative


class RQSQuantile(nn.Module):
    """Shared RQS quantile with optional bounded or logit input transforms."""

    def __init__(
        self,
        dim: int,
        *,
        n_bins: int = 64,
        bound: float = 10.0,
        num_layers: int = 1,
        eps: float = 1e-6,
        input_transform: str = "logit",
        condition_dim: Optional[int] = None,
        image_shape: Optional[Tuple[int, ...]] = None,
        normalize_z: bool = False,
        normalize_eps: Optional[float] = None,
        **unused_kwargs,
    ) -> None:
        super().__init__()

        if unused_kwargs:
            unexpected = ", ".join(sorted(unused_kwargs))
            raise TypeError(f"Unexpected keyword arguments for RQSQuantile: {unexpected}")
        if condition_dim not in (None, 0):
            raise NotImplementedError("Label-conditioned quantiles are not supported in RQSQuantile.")
        if normalize_z:
            raise NotImplementedError("normalize_z=True is not supported in RQSQuantile.")
        if normalize_eps is not None:
            raise NotImplementedError("normalize_eps is not supported in RQSQuantile.")

        self.dim = int(dim)
        self.eps = float(eps)
        self.cfg = RQSConfig(n_bins=int(n_bins), bound=float(bound))
        self.num_layers = int(num_layers)
        self.image_shape = tuple(image_shape) if image_shape is not None else None

        transform_key = input_transform.lower().strip()
        transform_aliases = {
            "logit": "logit",
            "sigmoid": "logit",
            "bounded": "bounded",
            "linear": "bounded",
            "affine": "bounded",
        }
        if transform_key not in transform_aliases:
            valid = ", ".join(sorted({alias for alias in transform_aliases.values()}))
            raise ValueError(f"Unsupported input_transform='{input_transform}'. Expected one of {{{valid}}}.")
        self.input_transform = transform_aliases[transform_key]

        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

        self.raw_w = nn.ParameterList()
        self.raw_h = nn.ParameterList()
        self.raw_s = nn.ParameterList()
        for _ in range(self.num_layers):
            self.raw_w.append(nn.Parameter(torch.zeros(1, dim, self.cfg.n_bins)))
            self.raw_h.append(nn.Parameter(torch.zeros(1, dim, self.cfg.n_bins)))
            self.raw_s.append(nn.Parameter(torch.zeros(1, dim, self.cfg.n_bins + 1)))

    def _expand_tau(self, tau: torch.Tensor, batch: int) -> torch.Tensor:
        if tau.dim() == 2 and tau.shape[1] == 1:
            return tau.expand(batch, self.dim)
        if tau.shape == (batch, self.dim):
            return tau
        raise ValueError("tau must be shaped (B,1) or (B,D)")

    def forward(
        self,
        u: torch.Tensor,
        tau: torch.Tensor,
        *,
        return_dqdt: bool = False,
        requires_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if u.dim() != 2 or u.shape[1] != self.dim:
            raise ValueError(f"Expected u to have shape (B, {self.dim}); got {tuple(u.shape)}")
        batch = u.shape[0]
        tau_nd = self._expand_tau(tau, batch)
        need_dq_du = bool(requires_grad and return_dqdt)
        q, _ = self._compute_quantile_and_jac(u, return_dq_du=need_dq_du)
        eps = tau_nd * q
        if return_dqdt:
            dqdt = q
            if not requires_grad:
                return eps.detach(), dqdt.detach()
            return eps, dqdt
        return eps

    def base_quantile(self, u: torch.Tensor) -> torch.Tensor:
        if u.dim() != 2 or u.shape[1] != self.dim:
            raise ValueError(f"Expected u to have shape (B, {self.dim}); got {tuple(u.shape)}")
        q, _ = self._compute_quantile_and_jac(u, return_dq_du=False)
        return q

    def base_quantile_with_dqdu(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if u.dim() != 2 or u.shape[1] != self.dim:
            raise ValueError(f"Expected u to have shape (B, {self.dim}); got {tuple(u.shape)}")
        q, dq_du = self._compute_quantile_and_jac(u, return_dq_du=True)
        if dq_du is None:
            raise RuntimeError("Expected dq/du when return_dq_du=True")
        return q, dq_du

    def _compute_quantile_and_jac(
        self,
        u: torch.Tensor,
        *,
        return_dq_du: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, dim = u.shape
        if dim != self.dim:
            raise ValueError(f"Expected u to have dim={self.dim}, got {dim}")

        u_safe = u.clamp(self.eps, 1.0 - self.eps)
        mask_inner = (u > self.eps) & (u < 1.0 - self.eps)
        mask_inner = mask_inner.to(dtype=u.dtype)

        if self.input_transform == "logit":
            z = torch.log(u_safe) - torch.log1p(-u_safe)
            if return_dq_du:
                min_denom = self.eps if self.eps > 0.0 else 1e-12
                denom = (u_safe * (1.0 - u_safe)).clamp_min(min_denom)
                dz_du = mask_inner / denom
            else:
                dz_du = None
        elif self.input_transform == "bounded":
            scale = 2.0 * self.cfg.bound
            z = (u_safe - 0.5) * scale
            if return_dq_du:
                dz_du = mask_inner * scale
            else:
                dz_du = None
        else:
            raise RuntimeError(f"Unhandled input_transform='{self.input_transform}'")

        if return_dq_du:
            dydz_total = torch.ones_like(u_safe)
        else:
            dydz_total = None

        z_curr = z

        for layer_idx in range(self.num_layers):
            xk, yk, widths, heights, slopes = _params_to_spline(
                self.raw_w[layer_idx],
                self.raw_h[layer_idx],
                self.raw_s[layer_idx],
                self.cfg,
            )

            if xk.shape[0] != batch:
                xk = xk.expand(batch, -1, -1)
                yk = yk.expand(batch, -1, -1)
                widths = widths.expand(batch, -1, -1)
                heights = heights.expand(batch, -1, -1)
                slopes = slopes.expand(batch, -1, -1)

            z_curr, dydz = _rqs_forward_with_derivative(
                z_curr,
                xk,
                yk,
                widths,
                heights,
                slopes,
                self.cfg,
                return_dydx=return_dq_du,
            )

            if return_dq_du and dydz is not None and dydz_total is not None:
                dydz_total = dydz_total * dydz

        scale = F.softplus(self.log_scale) + 1e-4
        q = z_curr * scale.view(1, dim) + self.bias.view(1, dim)

        if return_dq_du and dydz_total is not None and dz_du is not None:
            dq_du = (scale.view(1, dim) * dydz_total) * dz_du
            return q, dq_du

        return q, None

    def diag_du(
        self,
        u: torch.Tensor,
        tau: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> torch.Tensor:
        if u.dim() != 2 or u.shape[1] != self.dim:
            raise ValueError(f"Expected u to have shape (B, {self.dim}); got {tuple(u.shape)}")
        batch = u.shape[0]
        tau_nd = self._expand_tau(tau, batch).detach()
        _, dq_du = self._compute_quantile_and_jac(u, return_dq_du=True)
        if dq_du is None:
            raise RuntimeError("Expected dq/du when return_dq_du=True")
        diag = tau_nd * dq_du
        if not create_graph:
            diag = diag.detach()
        return diag
