from __future__ import annotations

from typing import Tuple

import torch


def get_optimizer_and_scheduler(args, model: torch.nn.Module, quantile: torch.nn.Module) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Construct the joint optimizer and warmup scheduler for model/quantile."""
    base_lr = float(args.lr)
    freeze_quantile = bool(getattr(args, "freeze_quantile", False))

    param_groups = [{"params": list(model.parameters()), "lr": base_lr}]
    if not freeze_quantile and any(p.requires_grad for p in quantile.parameters()):
        param_groups.append({"params": list(quantile.parameters()), "lr": float(args.q_lr)})

    warmup_steps = max(0, int(getattr(args, "warmup_lr", 0)))

    optimizer = torch.optim.Adam(param_groups)

    def _warmup_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / warmup_steps)

    if len(param_groups) > 1:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[_warmup_lambda, lambda _: 1.0],
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_lambda)

    return optimizer, scheduler


def get_quantile_scheduler(args, optimizer: torch.optim.Optimizer):
    """Provide a piecewise-constant/linear scheduler for the quantile LR."""
    quantile_const_iters = getattr(args, "quantile_const_iters", 20_000)
    quantile_decay_iters = getattr(args, "quantile_decay_iters", 5_000)

    quantile_schedule_total = quantile_const_iters + quantile_decay_iters
    if len(optimizer.param_groups) > 1:
        quantile_base_lr = optimizer.param_groups[1]["lr"]
    else:
        quantile_base_lr = 0.0

    print(
        "[train_fm_quantile] quantile_schedule"
        f" (const={quantile_const_iters}, decay={quantile_decay_iters})"
    )

    def _quantile_lr_schedule(step: int) -> float:
        if step < quantile_const_iters:
            return quantile_base_lr
        if step < quantile_schedule_total:
            decay_progress = (step - quantile_const_iters + 1) / max(1, quantile_decay_iters)
            return max(0.0, quantile_base_lr * (1.0 - decay_progress))
        return 0.0

    return _quantile_lr_schedule, quantile_schedule_total
