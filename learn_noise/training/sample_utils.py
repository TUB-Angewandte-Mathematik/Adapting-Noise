"""Shared sampling utilities for training/evaluation loops."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torchdiffeq import odeint

from learn_noise.networks.model_wrapper import ODEWrapper, TorchWrapper


def default_t_eval(args, device: torch.device) -> torch.Tensor:
    """Default integration grid for evaluation trajectories."""
    return torch.linspace(1.0, 0.0, args.num_steps_eval, device=device)


def generate_quantile_samples(
    num_samples: int,
    *,
    batch_size: int,
    device: torch.device,
    dim: int,
    u_eps: float,
    quantile: torch.nn.Module,
    ode_func: ODEWrapper,
    t_eval: torch.Tensor,
    wrapper: TorchWrapper,
    eval_model: torch.nn.Module,
    u_source: Optional[torch.Tensor] = None,
    image_shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """Draw samples from the current quantile + ODE pair for evaluation/logging."""
    wrapper.model = eval_model
    prev_mode = eval_model.training
    eval_model.eval()

    outputs: list[torch.Tensor] = []
    produced = 0
    with torch.inference_mode():
        while produced < num_samples:
            remaining = num_samples - produced
            cur_bs = min(batch_size, remaining)
            attempt_bs = cur_bs

            while attempt_bs > 0:
                u_unit = U = ones = None
                eps_init = traj = None
                try:
                    end_idx = produced + attempt_bs
                    if u_source is not None:
                        u_unit = u_source[produced:end_idx].to(device=device)
                        if u_unit.dim() != 2 or u_unit.shape[1] != dim:
                            u_unit = u_unit.reshape(attempt_bs, dim)
                    else:
                        u_unit = torch.rand(attempt_bs, dim, device=device)
                    U = u_eps + (1 - 2 * u_eps) * u_unit
                    ones = torch.ones(attempt_bs, 1, device=device)
                    eps_init = quantile(U, ones)
                    wrapper.set_labels(None)

                    traj = odeint(ode_func, eps_init, t_eval, method="euler")
                    outputs.append(traj[-1].detach().cpu())
                    del traj, eps_init, U, u_unit, ones
                    produced += attempt_bs
                    break
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower() or attempt_bs <= 1:
                        raise
                    del traj, eps_init, U, u_unit, ones
                    torch.cuda.empty_cache()
                    attempt_bs = max(1, attempt_bs // 2)
            else:
                raise RuntimeError("Unable to generate samples even with batch size 1.")

    if prev_mode:
        eval_model.train()
    wrapper.set_labels(None)

    return torch.cat(outputs, dim=0)


def generate_baseline_samples(
    num_samples: int,
    *,
    batch_size: int,
    device: torch.device,
    dim: int,
    t_eval: torch.Tensor,
    ode_func: ODEWrapper,
    wrapper: TorchWrapper,
    eval_model: torch.nn.Module,
    latent_sampler: Callable[[tuple[int, ...]], torch.Tensor],
    latents: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Integrate the learnt baseline flow to produce samples for evaluation."""
    wrapper.model = eval_model
    prev_mode = eval_model.training
    eval_model.eval()

    outputs: list[torch.Tensor] = []
    produced = 0
    while produced < num_samples:
        cur_bs = min(batch_size, num_samples - produced)
        if latents is not None:
            z0 = latents[produced:produced + cur_bs].to(device)
        else:
            z0 = latent_sampler((cur_bs, dim))
        if labels is not None:
            lbl_batch = labels[produced:produced + cur_bs].to(device)
            wrapper.set_labels(lbl_batch)
        else:
            wrapper.set_labels(None)
        traj = odeint(ode_func, z0, t_eval, method="euler")
        outputs.append(traj[-1].detach().cpu())
        produced += cur_bs

    if prev_mode:
        eval_model.train()
    wrapper.set_labels(None)

    return torch.cat(outputs, dim=0) if outputs else torch.empty(0, dim)
