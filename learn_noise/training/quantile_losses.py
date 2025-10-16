"""Loss helpers shared across quantile training stages."""

from __future__ import annotations

import torch


def detached_mse(target_velocity: torch.Tensor, velocity_net: torch.Tensor) -> torch.Tensor:
    """
    Expand MSE with detached target norm so quantile gradients only flow via the cross term.
    """
    target_detached = target_velocity.detach()
    norm_v_net = velocity_net.pow(2).mean()
    norm_target = target_detached.pow(2).mean()
    cross_term = -2.0 * (velocity_net * target_velocity).mean()
    loss_velocity = norm_v_net + norm_target + cross_term
    return loss_velocity


def regularization_logdet(quantile, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Jacobian log-determinant penalty used during joint quantile training."""
    reg_logdet = torch.zeros((), device=u.device)

    with torch.set_grad_enabled(True):
        t_1_full = torch.ones_like(t)
        diag = quantile.diag_du(
            u,
            t_1_full,
            create_graph=True,
        )
    logdet = torch.log(diag.clamp_min(1e-12)).sum(dim=1)
    reg_logdet = (-logdet).mean()
    return reg_logdet
