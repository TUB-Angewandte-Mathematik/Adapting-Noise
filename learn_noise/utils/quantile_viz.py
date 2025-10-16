import os
import time
import random
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import wandb


def _sym_lims_from_array(arr_np, q=99.5, floor=3.0, ceil=12.0):
    """Robust symmetric limits based on finite values only.

    Computes the q-th percentile of the absolute values in ``arr_np`` and
    returns symmetric x/y limits. Falls back to ``floor`` when the input
    contains NaN/Inf or has no finite values.
    """
    a = np.abs(np.asarray(arr_np))
    # Use only finite entries to avoid propagating NaN/Inf into percentile
    finite_vals = a[np.isfinite(a)]
    if finite_vals.size == 0:
        r = float(floor)
    else:
        r = float(np.percentile(finite_vals, q))
        if not np.isfinite(r) or r <= 0:
            r = float(floor)
    r = float(np.clip(r, floor, ceil))
    return (-r, r), (-r, r)


def _rand_name(prefix):
    return f"{prefix}-{int(time.time())}-{random.randint(0, 999999):06d}"


@torch.no_grad()
def log_noise_slices(quantile, device, step, times=(0.0, 0.03, 0.1, 0.15, 0.25, 0.5, 0.75, 0.9, 0.98, 1.0), N=4000, dim=2, u_eps=5e-5):
    quantile.eval()
    fig, axes = plt.subplots(1, len(times), figsize=(3.0 * len(times), 3.0), constrained_layout=True)
    for ax, tt in zip(axes, times):
        U = torch.rand(N, dim, device=device).clamp(u_eps, 1 - u_eps)
        tau = torch.full((N, 1), float(tt), device=device)
        eps = quantile(U, tau)  # (N,dim)
        e = eps.detach().cpu().numpy()
        if dim >= 2:
            ax.scatter(e[:, 0], e[:, 1], s=3, alpha=0.4)
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.hist(e[:, 0], bins=60, alpha=0.8)
        ax.set_title(f"τ = {tt:.2f}")
        ax.grid(True, alpha=0.2)
    wandb.log({"quantile/noise_slices": wandb.Image(fig)}, step=step)
    plt.close(fig)


@torch.no_grad()
def log_xt_slices(quantile, sampler, device, step, times=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0), N=4000, dim=2, u_eps=5e-5):
    """Plot xτ with fixed axes: frame of reference from data x0 and τ=1 noise."""
    quantile.eval()
    x0 = sampler.sample(N, device=device)
    U = torch.rand(N, dim, device=device).clamp(u_eps, 1 - u_eps)

    # build a consistent frame using both x0 and τ=1
    eps1 = quantile(U, torch.ones(N, 1, device=device))
    ref = torch.cat([x0, eps1], dim=0).detach().cpu().numpy()
    (xmin, xmax), (ymin, ymax) = _sym_lims_from_array(ref)

    fig, axes = plt.subplots(1, len(times), figsize=(3.0 * len(times), 3.0), constrained_layout=True)
    for ax, tt in zip(axes, times):
        tau = torch.full((N, 1), float(tt), device=device)
        eps = quantile(U, tau)
        xt = (1.0 - tt) * x0 + eps
        e = xt.detach().cpu().numpy()
        ax.scatter(e[:, 0], e[:, 1], s=3, alpha=0.4)
        ax.set_title(f"xτ, τ={tt:.2f}")
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.2)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    wandb.log({"quantile/state_slices": wandb.Image(fig)}, step=step)
    plt.close(fig)


@torch.no_grad()
def log_noise_slices_fixed(
    quantile,
    x_fixed: torch.Tensor,        # (K, D) chosen x0's
    device,
    step: int,
    times=(0.0, 0.03, 0.1, 0.15, 0.25, 0.5, 0.75, 0.9, 0.98, 1.0),
    Nu=4000,
    dim=2,
    u_eps=5e-5,
    share_u=True,
    same_axes_across_rows=False,
    title_prefix="quantile/noise_slices_fixedx0",
    include_country=False,
    sampler=None,
    country_zoom=1.5,
):
    """Log noise slices for a few fixed x0 across τ. Optionally add true samples background."""
    quantile.eval()

    K = x_fixed.shape[0]
    cols = len(times) + (1 if include_country else 0)
    fig, axes = plt.subplots(K, cols, figsize=(3.0 * cols, 3.0 * K), constrained_layout=True)
    if K == 1:
        axes = axes[None, :]

    # Optionally precompute reference limits using both ε(τ=1) and x0
    if include_country and sampler is not None:
        country = sampler.sample(20000, device=device).detach().cpu().numpy()
    else:
        country = None

    U_all = None
    if share_u:
        U_all = [torch.rand(Nu, dim, device=device).clamp(u_eps, 1 - u_eps) for _ in range(K)]

    for k in range(K):
        col_offset = 0
        axc = None
        if include_country and sampler is not None:
            axc = axes[k, 0]
            if country is not None:
                axc.scatter(country[:, 0], country[:, 1], s=1, alpha=0.25, c='#e74c3c')
            # plot x0; limits will be set after we compute them to ensure visibility
            axc.scatter(x_fixed[k, 0].item(), x_fixed[k, 1].item(), s=25, c='yellow', edgecolors='black')
            axc.set_title("data")
            axc.set_aspect('equal', 'box')
            axc.grid(True, alpha=0.2)
            col_offset = 1

        # Fix axes using combined support across τ for this x0
        ref_pts = []
        for tt in times:
            U_tmp = (U_all[k] if share_u else torch.rand(Nu, dim, device=device).clamp(u_eps, 1 - u_eps))
            tau = torch.full((Nu, 1), float(tt), device=device)
            eps = quantile(U_tmp, tau)
            ref_pts.append(((1.0 - tt) * x_fixed[k].unsqueeze(0) + eps).detach().cpu().numpy())
        ref_np = np.concatenate(ref_pts, axis=0)
        (xmin, xmax), (ymin, ymax) = _sym_lims_from_array(ref_np)

        # Optionally zoom out a bit for the country panel, and always include x0
        if axc is not None:
            x_center = 0.5 * (xmin + xmax)
            y_center = 0.5 * (ymin + ymax)
            x_half = 0.5 * (xmax - xmin) * float(country_zoom)
            y_half = 0.5 * (ymax - ymin) * float(country_zoom)
            xmin_c = x_center - x_half
            xmax_c = x_center + x_half
            ymin_c = y_center - y_half
            ymax_c = y_center + y_half

            # Ensure the selected x0 is within the limits
            x0x = x_fixed[k, 0].item()
            x0y = x_fixed[k, 1].item()
            xmin_c = min(xmin_c, x0x)
            xmax_c = max(xmax_c, x0x)
            ymin_c = min(ymin_c, x0y)
            ymax_c = max(ymax_c, x0y)

            axc.set_xlim(xmin_c, xmax_c)
            axc.set_ylim(ymin_c, ymax_c)

        for j, tt in enumerate(times):
            ax = axes[k, j + col_offset]
            U = U_all[k] if share_u else torch.rand(Nu, dim, device=device).clamp(u_eps, 1 - u_eps)
            tau = torch.full((Nu, 1), float(tt), device=device)
            eps = quantile(U, tau)
            e = eps.detach().cpu().numpy()
            ax.scatter(e[:, 0], e[:, 1], s=3, alpha=0.4)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            if k == 0:
                ax.set_title(f"τ = {tt:.2f}")
            ax.grid(True, alpha=0.2)

    wandb.log({title_prefix: wandb.Image(fig)}, step=step)
    plt.close(fig)
