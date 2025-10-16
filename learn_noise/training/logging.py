from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle  # (not used now, but fine to keep)
from torchdiffeq import odeint

import learn_noise.utils.evaluation as evaluation
from learn_noise.utils.colors import COL_DENSITY
import learn_noise.utils.plotting_traj as plot_traj
from learn_noise.utils.image_eval import compute_fid, reshape_flat_samples, save_image_grid
from learn_noise.utils.image_latent_viz import (
    build_latent_visualizations,
    make_channel_pixel_histograms,
)
from learn_noise.utils.figure_saving import save_figure
from learn_noise.training.common import (
    count_parameters,
    make_fixed_uniform,
    write_model_size_summary,
)


@dataclass
class FidConfig:
    enabled: bool
    interval: int
    num_gen: int
    image_size: int
    batch_size: int
    gen_batch: int
    real_cache: torch.Tensor | None


def prepare_fid(args, sampler, device, image_shape, dim) -> FidConfig:
    """Build FID bookkeeping and cache real samples when enabled."""
    is_image_task = (image_shape is not None) and (math.prod(image_shape) == dim)
    interval = int(getattr(args, "fid_eval_interval", 0))
    num_gen = int(getattr(args, "fid_num_gen", 0))
    enabled = is_image_task and interval > 0 and num_gen > 0
    if enabled:
        image_size = int(getattr(args, "fid_image_size", image_shape[-1]))
        batch_size = max(1, int(getattr(args, "fid_batch_size", args.batch_size)))
        gen_batch = max(1, int(getattr(args, "fid_gen_batch", args.batch_size)))
        with torch.no_grad():
            real_samples = sampler.sample(num_gen, device=device, dtype=torch.float32)
            real_imgs = reshape_flat_samples(real_samples, torch.Size(image_shape))
        real_cache = real_imgs.detach().cpu()
    else:
        image_size = batch_size = gen_batch = 0
        real_cache = None
    return FidConfig(enabled, interval, num_gen, image_size, batch_size, gen_batch, real_cache)


def prepare_fixed_u(args, is_image_task, image_shape, device):
    """Maintain deterministic uniform seeds for visualization/evaluation."""
    fixed_u_vis = None
    if is_image_task and args.sample_vis_interval > 0 and args.sample_vis_count > 0:
        vis_shape = (args.sample_vis_count, *image_shape)
        buf = getattr(args, "_fixed_double_fm_vis_u", None)
        if buf is None or buf.shape[0] < args.sample_vis_count:
            buf = make_fixed_uniform(vis_shape, seed=args.seed + 73, device=device)
            setattr(args, "_fixed_double_fm_vis_u", buf)
        fixed_u_vis = buf
    fixed_eval_u = None if is_image_task else getattr(args, "_fixed_quantile_eval_u", None)
    return fixed_u_vis, fixed_eval_u


def wandb_global_steps(args):
    """Derive the global step offset, accounting for pretraining + existing runs."""
    global_step_offset = int(getattr(args, "_pretrain_step_offset", 0))
    if global_step_offset > 0 and wandb.run is not None:
        run_step = getattr(wandb.run, "step", None)
        if run_step is not None:
            run_step = int(run_step)
            if run_step >= global_step_offset:
                global_step_offset = run_step + 1

    return global_step_offset


def track_model_parameters(args, model, quantile, global_step_offset):
    """Log parameter counts for UNet/quantile models and write a JSON summary."""
    unet_params = count_parameters(model)
    quantile_params = count_parameters(quantile)
    model_size_stats = {
        "method": "quantile_joint",
        "target_dataset": getattr(args, "target_dataset", None),
        "params_unet": unet_params,
        "params_quantile": quantile_params,
        "params_total": unet_params + quantile_params,
        "freeze_quantile": bool(getattr(args, "freeze_quantile", False)),
        "quantile_rqs_layers": getattr(args, "q_rqs_layers", None),
        "quantile_rqs_bins": getattr(args, "q_rqs_bins", None),
    }
    channel_mult = getattr(args, "unet_channel_mult", None)
    if channel_mult is not None:
        model_size_stats["unet_channel_mult"] = tuple(channel_mult)
    for attr in ("unet_model_channels", "unet_num_res_blocks", "unet_attention_resolutions"):
        value = getattr(args, attr, None)
        if value is not None:
            model_size_stats[attr] = value
    write_model_size_summary(args.runs_dir, model_size_stats)
    wandb.log(
        {
            "params/unet": float(unet_params),
            "params/quantile": float(quantile_params),
            "params/total": float(unet_params + quantile_params),
        },
        step=global_step_offset,
    )

def log_real_rgb_histogram_once(
    *,
    args,
    sampler,
    image_shape,
    device: torch.device,
    step: int = 0,
    samples_key: str = "latent/real_rgb_hist",
) -> None:
    """Log a single set of RGB histograms for real data if not already emitted."""
    if getattr(args, "_logged_real_rgb_hist", False):
        return

    if image_shape is None or len(image_shape) != 3:
        return

    channels = image_shape[0]
    if channels not in {1, 3}:  # only meaningful for grayscale/RGB
        return

    try:
        import wandb  # local import to avoid hard dependency when disabled
    except ImportError:  # pragma: no cover - wandb optional
        return

    sample_count = int(getattr(args, "real_hist_samples", 4096))
    max_available = getattr(sampler, "num_samples", None)
    if max_available is not None:
        sample_count = max(1, min(sample_count, int(max_available)))

    try:
        real_flat = sampler.sample(sample_count, device=device, dtype=torch.float32)
    except TypeError:
        real_flat, _ = sampler.sample_with_labels(sample_count, device=device, dtype=torch.float32)
    except AttributeError:
        return

    real_flat = real_flat.detach().cpu()
    hist_fig = make_channel_pixel_histograms(real_flat, image_shape)
    runs_dir = getattr(args, "runs_dir", None)
    output_dir = os.path.join(runs_dir, "latent_viz") if runs_dir else None
    save_figure(hist_fig, output_dir=output_dir, key=samples_key, step=step)
    wandb.log({samples_key: wandb.Image(hist_fig)}, step=step)
    plt.close(hist_fig)
    args._logged_real_rgb_hist = True

# -------------------- COLORS --------------------
COL_BG_LIGHT = "#F6F7F9"   # kept for consistency; we don't paint it (background is transparent)
COL_BG_DARK  = "#E9EDF2"
COL_PATH     = "#4DD83B"#"#4682B4"
COL_START    = "#2B485F"#"#F1D76F"
COL_END      = "#D0202C"#"#1B1B3A"
COL_DENSITY  = "#F5F5F5"
COL_LATENT   = "#6A6D75"
COL_GENERATED = "#4C9AE8"

# -------------------- UTILS --------------------
def _get_lowdim_limits(args) -> tuple[float, float, float, float]:
    dataset = getattr(args, "target_dataset", "") or ""
    if dataset.lower() == "funnel":
        return -10.0, 10.0, -20.0, 20.0
    return -4.0, 4.0, -4.0, 4.0


def _log_scatter_snapshot(
    points,
    *,
    args,
    step: int,
    key: str,
    filename: str,
    color: str,
    alpha: float,
    marker_size: float = 8.0,
) -> None:
    if points is None:
        return
    pts = torch.as_tensor(points).detach().cpu().numpy()
    if pts.ndim != 2 or pts.shape[1] < 2:
        return
    x_min, x_max, y_min, y_max = _get_lowdim_limits(args)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=140)
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=marker_size,
        c=color,
        alpha=alpha,
        edgecolors="none",
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", "box")

    out_path = os.path.join(args.runs_dir, f"{filename}_step_{step:06d}.png")
    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    wandb.log({key: wandb.Image(out_path)}, step=step)
    plt.close(fig)


def _log_time_slice_grid(
    *,
    args,
    step: int,
    trajectories: np.ndarray,
    t_vals: torch.Tensor,
    key: str,
    filename: str,
    num_slices: int = 6,
    marker_size: float = 6.0,
) -> None:
    if trajectories is None or t_vals is None:
        return
    if trajectories.ndim != 3 or trajectories.shape[-1] < 2:
        return
    if getattr(args, "dim", 0) != 2:
        return
    num_slices = max(2, int(num_slices))

    t_grid = t_vals.detach().cpu().numpy()
    if t_grid.ndim != 1:
        t_grid = t_grid.reshape(-1)
    if t_grid.size == 0:
        return

    total_steps = trajectories.shape[0]
    if total_steps <= 1:
        return

    if t_grid[0] > t_grid[-1]:
        idx_floats = np.linspace(total_steps - 1, 0, num_slices)
        indices = np.round(idx_floats).astype(int)
        indices[0] = total_steps - 1
        indices[-1] = 0
    else:
        idx_floats = np.linspace(0, total_steps - 1, num_slices)
        indices = np.round(idx_floats).astype(int)
        indices[0] = 0
        indices[-1] = total_steps - 1
    indices = np.clip(indices, 0, total_steps - 1)

    x_min, x_max, y_min, y_max = _get_lowdim_limits(args)

    rows = 1
    cols = num_slices
    panel_size = 6
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(panel_size * cols, panel_size * rows),
        dpi=160,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes = axes.reshape(rows, cols)

    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    fig.subplots_adjust(
        left=0.002,
        right=0.998,
        bottom=0.002,
        top=0.998,
        wspace=0.001,
        hspace=0.0,
    )

    def _render_slice(ax, pts):
        ax.set_facecolor("none")
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=marker_size,
            c=COL_GENERATED,
            alpha=0.4,
            edgecolors="none",
            linewidths=0.0,
            rasterized=True,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect("equal", "box")
        ax.margins(0.0, 0.0)

    flat_axes = axes.reshape(-1)
    slice_exports: list[tuple[np.ndarray, float, int]] = []
    for slice_order, (ax, idx) in enumerate(zip(flat_axes, indices)):
        pts = trajectories[idx]
        _render_slice(ax, pts)
        t_val = float(t_grid[idx]) if 0 <= idx < t_grid.shape[0] else float("nan")
        slice_exports.append((pts, t_val, slice_order))

    out_path = os.path.join(args.runs_dir, f"{filename}_step_{step:06d}.png")
    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.0,
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    wandb.log({key: wandb.Image(out_path)}, step=step)
    plt.close(fig)

    for pts, t_val, slice_order in slice_exports:
        slice_fig, slice_ax = plt.subplots(1, 1, figsize=(panel_size, panel_size), dpi=180)
        slice_fig.patch.set_facecolor("none")
        slice_fig.patch.set_alpha(0.0)
        _render_slice(slice_ax, pts)
        slice_fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

        t_str = f"{t_val:.4f}" if math.isfinite(t_val) else "nan"
        t_slug = t_str.replace(".", "p")
        slice_path = os.path.join(
            args.runs_dir,
            f"{filename}_slice{slice_order:02d}_t{t_slug}_step_{step:06d}.png",
        )
        slice_fig.savefig(
            slice_path,
            dpi=220,
            bbox_inches="tight",
            pad_inches=0.0,
            transparent=True,
            facecolor="none",
            edgecolor="none",
        )
        slice_key = f"{key}/t_{t_str}_slice_{slice_order:02d}"
        wandb.log({slice_key: wandb.Image(slice_path)}, step=step)
        plt.close(slice_fig)


def _infer_tile_size(sampler, default=1.0):
    """Guess the sampler's checker tile size."""
    for name in ("tile_size", "checker_size", "period", "grid_step", "cell"):
        if hasattr(sampler, name):
            v = float(getattr(sampler, name))
            if v > 0:
                return v
    return float(default)

@torch.no_grad()
def _draw_sampler_background(
    ax,
    *,
    sampler,
    device: torch.device,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    density_grid: int = 240,
) -> None:
    """
    Transparent background; checker rendered as a single RGBA image (no seams).
    Colored tiles only for cells fully inside [x_min,x_max]×[y_min,y_max] (no skinny border tiles).
    Density overlay appears ONLY on those colored tiles.
    """
    ax.set_facecolor((1, 1, 1, 0))

    # --- grid aligned to sampler tile size ---
    s  = _infer_tile_size(sampler, default=1.0)
    x0 = np.floor(x_min / s) * s
    x1 = np.ceil(x_max / s) * s
    y0 = np.floor(y_min / s) * s
    y1 = np.ceil(y_max / s) * s

    nx = int(round((x1 - x0) / s))
    ny = int(round((y1 - y0) / s))

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    prev_dark = ((ii + jj) % 2 == 1).astype(np.float32)

    # invert parity per your last request: "clear ↔ color"
    colored = 1.0 - prev_dark

    # keep ONLY tiles fully inside the plotting window (kills skinny border tiles)
    tile_x_left   = x0 + ii * s
    tile_x_right  = tile_x_left + s
    tile_y_bottom = y0 + jj * s
    tile_y_top    = tile_y_bottom + s
    fully_inside = (
        (tile_x_left >= x_min) &
        (tile_x_right <= x_max) &
        (tile_y_bottom >= y_min) &
        (tile_y_top   <= y_max)
    ).astype(np.float32)

    color_mask = colored * fully_inside  # 1 on colored full tiles, 0 otherwise

    # --- checker as one RGBA image (no borders, no AA seams) ---
    checker = np.zeros((ny, nx, 4), dtype=np.float32)
    r, g, b, _ = to_rgba(COL_BG_DARK)
    checker[..., 0] = r * color_mask
    checker[..., 1] = g * color_mask
    checker[..., 2] = b * color_mask
    checker[..., 3] = color_mask  # alpha 1 only on fully-inside colored tiles

    ax.imshow(
        checker,
        extent=(x0, x1, y0, y1),
        origin="lower",
        interpolation="nearest",
        resample=False,
        filternorm=False,
        aspect="auto",
        zorder=0.1,
    )

    # --- faint density overlay, masked to the SAME fully-inside colored tiles ---
    if getattr(sampler, 'has_log_prob', False):
        gx = np.linspace(x_min, x_max, density_grid, dtype=np.float32)
        gy = np.linspace(y_min, y_max, density_grid, dtype=np.float32)
        XX, YY = np.meshgrid(gx, gy, indexing='xy')
        coords = np.stack([XX, YY], axis=-1).reshape(-1, 2)

        try:
            grid_t = torch.from_numpy(coords).to(device=device, dtype=torch.float32)
            logp = sampler.log_prob(grid_t)
        except Exception:
            grid_t = torch.from_numpy(coords).to('cpu', dtype=torch.float32)
            logp = sampler.log_prob(grid_t)

        if logp is not None:
            lp = logp.detach().cpu().numpy().reshape(gy.size, gx.size)
            finite = np.isfinite(lp)
            if finite.any():
                lo, hi = np.percentile(lp[finite], [5.0, 95.0])
                if hi - lo < 1e-6:
                    alpha = finite.astype(np.float32)
                else:
                    alpha = np.clip((lp - lo) / (hi - lo), 0.0, 1.0)
                    alpha[~finite] = 0.0

                # pixel-resolution mask for fully-inside colored tiles
                ix = np.floor((XX - x0) / s).astype(int)
                iy = np.floor((YY - y0) / s).astype(int)

                # bounds check (points falling outside x0..x1 may happen numerically)
                valid = (
                    (ix >= 0) & (ix < nx) &
                    (iy >= 0) & (iy < ny)
                )
                pix_mask = np.zeros_like(alpha, dtype=np.float32)
                if valid.any():
                    # map to the same color_mask grid
                    cm = color_mask  # (ny, nx)
                    pix_mask[valid] = cm[iy[valid], ix[valid]]

                rgba = np.zeros((gy.size, gx.size, 4), dtype=np.float32)
                r, g, b, _ = to_rgba(COL_DENSITY)
                rgba[..., 0] = r
                rgba[..., 1] = g
                rgba[..., 2] = b
                rgba[..., 3] = alpha * pix_mask * 0.18  # faint & masked

                ax.imshow(
                    rgba,
                    extent=(x_min, x_max, y_min, y_max),
                    origin='lower',
                    interpolation='nearest',
                    resample=False,
                    filternorm=False,
                    aspect='auto',
                    zorder=0.2,
                )


def _render_trajectory_panel(
    *,
    args,
    step: int,
    sampler,
    device: torch.device,
    trajectories: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    show_legend: bool = True,
    show_title: bool = False,
) -> None:
    """Draw and log a snapshot of low-dimensional trajectories with a clear background."""
    x_min, x_max, y_min, y_max = _get_lowdim_limits(args)

    fig, ax = plt.subplots(1, 1, figsize=(14, 14), dpi=140)

    # Make the entire figure transparent (important for exports).
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    dataset_lower = (getattr(args, "target_dataset", "") or "").lower()
    use_checker = dataset_lower in {"checker", "checkerboard"}
    if use_checker:
        _draw_sampler_background(
            ax,
            sampler=sampler,
            device=device,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

    # --- trajectories ---
    line_kwargs = dict(color=COL_PATH, alpha=0.35, linewidth=1.0, solid_capstyle="round", zorder=1)
    for idx in range(starts.shape[0]):
        ax.plot(trajectories[:, idx, 0], trajectories[:, idx, 1], **line_kwargs)

    # --- start/end points ---
    ax.scatter(
        starts[:, 0], starts[:, 1],
        s=14, c=COL_START, alpha=0.9, edgecolors="none",
        # label='start (τ=1)',
        zorder=2,
    )
    ax.scatter(
        ends[:, 0], ends[:, 1],
        s=18, c=COL_END, alpha=0.7, edgecolors="none",
        # label='end (τ=0)',
        zorder=3,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if show_legend:
        legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
        if legend is not None:
            for text in legend.get_texts():
                text.set_color("#2B2B2B")

    ax.set_aspect("equal", "box")
    ax.set_title("" if not show_title else f"Trajectories @ step {step}", pad=12, color="#2B2B2B")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Transparent export: background is fully clear except our drawn elements.
    traj_path = os.path.join(args.runs_dir, f"trajectories_step_{step:06d}.png")
    fig.savefig(
        traj_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    wandb.log({"fm/trajectories_clean": wandb.Image(traj_path)}, step=step)
    plt.close(fig)

# -------------------- LOGGING HOOKS --------------------
def log_baseline_evaluation(
    *,
    args,
    step: int,
    ema_model,
    wrapper,
    ode_func,
    sampler,
    noise_sampler: Callable[[tuple[int, ...]], torch.Tensor],
    x0_batch: torch.Tensor,
    device: torch.device,
    do_light: bool,
    do_heavy: bool,
) -> None:
    """Run evaluation and logging for the baseline FM trainer."""
    ema_model.eval()
    wrapper.model = ema_model
    with torch.inference_mode():
        if do_light:
            if hasattr(ode_func, 'reset_nfe'):
                ode_func.reset_nfe()
            num_traj = min(20000, args.eval_sample)
            if num_traj > 0:
                eps1 = noise_sampler((num_traj, args.dim)).to(device)

                if args.dim == 2:
                    _log_scatter_snapshot(
                        eps1,
                        args=args,
                        step=step,
                        key="baseline/latent_cloud",
                        filename="baseline_latent_cloud",
                        color=COL_LATENT,
                        alpha=0.4,
                        marker_size=9.0,
                    )

                if args.target_dataset == 'funnel':
                    xlim = (-20.0, 20.0)
                    ylim = (-100.0, 100.0)
                else:
                    xlim = (-4.0, 4.0)
                    ylim = (-4.0, 4.0)

                plot_traj.visualize_and_save(
                    ode_func,
                    noise=eps1,
                    T=1.0,
                    output_dir=args.runs_dir,
                    num_steps=50,
                    num_samples=2000,
                    dim=args.dim,
                    device=device,
                    step=step,
                    wandb_key="fm/trajectory_gif",
                    filename=f"trajectory_step_{step:06d}",
                    xlim=xlim,
                    ylim=ylim,
                )

                t_vals = torch.linspace(1.0, 0.0, args.num_steps_eval, device=device)
                n_paths = min(5000, eps1.shape[0])
                if n_paths > 0:
                    x_traj = odeint(ode_func, eps1[:n_paths], t_vals, method='dopri5')
                    X = x_traj.detach().cpu().numpy()
                    starts = X[0]
                    ends = X[-1]

                    _render_trajectory_panel(
                        args=args,
                        step=step,
                        sampler=sampler,
                        device=device,
                        trajectories=X,
                        starts=starts,
                        ends=ends,
                        show_legend=True,
                        show_title=False,
                    )

                    if args.dim == 2:
                        _log_scatter_snapshot(
                            ends,
                            args=args,
                            step=step,
                            key="baseline/generated_cloud",
                            filename="baseline_generated_cloud",
                            color=COL_GENERATED,
                            alpha=0.45,
                            marker_size=9.0,
                        )
                        _log_time_slice_grid(
                            args=args,
                            step=step,
                            trajectories=X,
                            t_vals=t_vals,
                            key="baseline/trajectory_slices",
                            filename="baseline_trajectory_slices",
                        )

        evaluation.heavy_eval_batched(
            args,
            x0_batch,
            ode_func,
            sampler,
            noise=noise_sampler,
            step=step,
            big_eval=do_heavy,
            device=device,
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ema_model.train()


def log_baseline_image_metrics(
    *,
    args,
    step: int,
    eval_model,
    wrapper,
    device: torch.device,
    image_shape,
    sampler,
    sample_vis_interval: int,
    sample_vis_count: int,
    sample_vis_nrow: int,
    sample_dir: str,
    fid_interval: int,
    fid_num_gen: int,
    fid_batch_size: int,
    fid_image_size: int,
    fid_gen_batch: int,
    fid_real_cache,
    noise_sampler: Callable[[tuple[int, ...]], torch.Tensor],
    generate_samples: Callable[..., torch.Tensor],
    fixed_noise: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Log baseline image metrics: sample grids and FID."""
    log_samples = (
        sample_vis_interval > 0
        and sample_vis_count > 0
        and ((step + 1) % sample_vis_interval == 0)
    )
    log_fid = (
        fid_interval > 0
        and fid_num_gen > 0
        and fid_real_cache is not None
        and ((step + 1) % fid_interval == 0)
    )

    fid_reference = getattr(args, "_conditional_fid_sampler", None)
    real_fid_sampler = fid_reference if fid_reference is not None else sampler

    if not (log_samples or log_fid):
        return fixed_noise

    prev_mode = eval_model.training
    eval_model.eval()
    wrapper.model = eval_model

    with torch.inference_mode():
        if log_samples:
            if sample_dir:
                os.makedirs(sample_dir, exist_ok=True)
            if fixed_noise is None or fixed_noise.shape[0] < sample_vis_count:
                fixed_noise = noise_sampler((sample_vis_count, args.dim))
            latents = fixed_noise[:sample_vis_count]
            generated_samples = generate_samples(sample_vis_count, latents=latents)
            vis_imgs = reshape_flat_samples(generated_samples, torch.Size(image_shape))
            grid_np = save_image_grid(
                vis_imgs,
                path=os.path.join(sample_dir, f'step_{step:06d}.png'),
                nrow=sample_vis_nrow,
            )
            wandb.log({"samples/grid": wandb.Image(grid_np)}, step=step)

        if log_fid:
            try:
                gen_samples = generate_samples(fid_num_gen)
                gen_imgs = reshape_flat_samples(gen_samples, torch.Size(image_shape))
                fid_val = compute_fid(
                    fid_real_cache,
                    gen_imgs,
                    device=device,
                    image_size=fid_image_size,
                    batch_size=fid_batch_size,
                )
                wandb.log({"metrics/fid": float(fid_val)}, step=step)
            except ImportError as exc:
                warned = getattr(args, "_fid_import_warned", False)
                if not warned:
                    print(f"[fid] Skipping FID evaluation: {exc}")
                    args._fid_import_warned = True

    if prev_mode:
        eval_model.train()

    return fixed_noise


def log_quantile_image_metrics(
    *,
    args,
    step: int,
    eval_model,
    wrapper,
    quantile,
    device: torch.device,
    image_shape,
    sample_vis_interval: int,
    sample_vis_count: int,
    sample_vis_nrow: int,
    sample_dir: str,
    fid_interval: int,
    fid_num_gen: int,
    fid_batch_size: int,
    fid_image_size: int,
    fid_gen_batch: int,
    fid_real_cache,
    generate_samples: Callable[..., torch.Tensor],
    fixed_u_vis: Optional[torch.Tensor],
    u_eps: float,
) -> Optional[torch.Tensor]:
    del fid_gen_batch

    log_samples = (
        sample_vis_interval > 0
        and sample_vis_count > 0
        and ((step + 1) % sample_vis_interval == 0)
    )
    log_fid = (
        fid_interval > 0
        and fid_num_gen > 0
        and fid_real_cache is not None
        and ((step + 1) % fid_interval == 0)
    )
    latent_viz_samples = int(args.latent_viz_samples)
    log_latent = log_samples and latent_viz_samples > 0

    if not (log_samples or log_fid or log_latent):
        return fixed_u_vis

    prev_mode = eval_model.training
    eval_model.eval()
    wrapper.model = eval_model
    prev_quant_mode = quantile.training
    quantile.eval()

    try:
        with torch.inference_mode():
            if log_samples:
                if sample_dir:
                    os.makedirs(sample_dir, exist_ok=True)
                if fixed_u_vis is None or fixed_u_vis.shape[0] < sample_vis_count:
                    fixed_u_vis = torch.rand(sample_vis_count, args.dim, device=device).detach().cpu()
                u_source = fixed_u_vis[:sample_vis_count]
                generated_samples = generate_samples(
                    sample_vis_count,
                    u_source=u_source,
                )
                vis_imgs = reshape_flat_samples(generated_samples, torch.Size(image_shape))
                grid_np = save_image_grid(
                    vis_imgs,
                    path=os.path.join(sample_dir, f'step_{step:06d}.png'),
                    nrow=sample_vis_nrow,
                )
                wandb.log({"samples/grid": wandb.Image(grid_np)}, step=step)
            if log_fid:
                try:
                    gen_samples = generate_samples(fid_num_gen)
                    gen_imgs = reshape_flat_samples(gen_samples, torch.Size(image_shape))
                    fid_val = compute_fid(
                        fid_real_cache,
                        gen_imgs,
                        device=device,
                        image_size=fid_image_size,
                        batch_size=fid_batch_size,
                    )
                    wandb.log({"metrics/fid": float(fid_val)}, step=step)
                except ImportError as exc:
                    warned = getattr(args, "_fid_import_warned", False)
                    if not warned:
                        print(f"[fid] Skipping FID evaluation: {exc}")
                        args._fid_import_warned = True
            if log_latent:
                num_latent = min(latent_viz_samples, 1024)
                if num_latent > 0:
                    unit_u = torch.rand(num_latent, args.dim, device=device)
                    U_latent = u_eps + (1 - 2 * u_eps) * unit_u
                    ones_latent = torch.ones(num_latent, 1, device=device)
                    eps_latent = quantile(U_latent, ones_latent)
                    latents_cpu = eps_latent.detach().cpu()
                    viz_payload = build_latent_visualizations(
                        latents_cpu,
                        image_shape=image_shape,
                        atlas_images=None,
                    )
                    wandb_payload = {}
                    if viz_payload.mean_std_fig is not None:
                        wandb_payload["latent/mean_std"] = wandb.Image(viz_payload.mean_std_fig)
                    if viz_payload.mean_fig is not None:
                        wandb_payload["latent/mean"] = wandb.Image(viz_payload.mean_fig)
                    if viz_payload.std_fig is not None:
                        wandb_payload["latent/std"] = wandb.Image(viz_payload.std_fig)
                    if viz_payload.hist_qq_fig is not None:
                        wandb_payload["latent/hist_qq"] = wandb.Image(viz_payload.hist_qq_fig)
                    if viz_payload.pca_fig is not None:
                        wandb_payload["latent/pca"] = wandb.Image(viz_payload.pca_fig)
                    if viz_payload.corr_fig is not None:
                        wandb_payload["latent/correlation"] = wandb.Image(viz_payload.corr_fig)
                    if wandb_payload:
                        wandb.log(wandb_payload, step=step)
                    for fig in [
                        viz_payload.mean_std_fig,
                        viz_payload.mean_fig,
                        viz_payload.std_fig,
                        viz_payload.hist_qq_fig,
                        viz_payload.pca_fig,
                        viz_payload.corr_fig,
                        viz_payload.atlas_fig,
                    ]:
                        if fig is not None:
                            plt.close(fig)
    finally:
        if prev_mode:
            eval_model.train()
        if prev_quant_mode:
            quantile.train()

    return fixed_u_vis
def log_quantile_low_dim_metrics(
    *,
    args,
    step: int,
    eval_model,
    wrapper,
    ode_func,
    sampler,
    quantile,
    x0_batch: torch.Tensor,
    device: torch.device,
    do_light: bool,
    do_heavy: bool,
    u_eps: float,
    fixed_eval_u: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Run trajectory plots and Sinkhorn/MMD evaluation for low-dimensional targets."""
    if not (do_light or do_heavy):
        return fixed_eval_u

    prev_mode = eval_model.training
    eval_model.eval()
    wrapper.model = eval_model
    prev_quant_mode = quantile.training
    quantile.eval()

    with torch.inference_mode():
        if do_light:
            if hasattr(ode_func, 'reset_nfe'):
                ode_func.reset_nfe()
            num_traj = min(10000, args.eval_sample)
            if num_traj > 0:
                if fixed_eval_u is None or fixed_eval_u.shape[0] < num_traj:
                    fixed_eval_u = torch.rand(num_traj, args.dim, device=device)
                Uv = u_eps + (1 - 2 * u_eps) * fixed_eval_u[:num_traj]
                eps1 = quantile(
                    Uv,
                    torch.ones(num_traj, 1, device=device),
                )


                if args.target_dataset == 'funnel':
                    xlim = (-20.0, 20.0)
                    ylim = (-100.0, 100.0)
                else:
                    xlim = (-4.0, 4.0)
                    ylim = (-4.0, 4.0)
                plot_traj.visualize_and_save(
                    ode_func,
                    noise=eps1,
                    T=1.0,
                    output_dir=args.runs_dir,
                    num_steps=50,
                    num_samples=2000,
                    dim=args.dim,
                    device=device,
                    step=step,
                    wandb_key="fm/trajectory_gif",
                    filename=f"trajectory_step_{step:06d}",
                    xlim=xlim,
                    ylim=ylim,
                )

                t_vals = torch.linspace(1.0, 0.0, args.num_steps_eval, device=device)
                n_paths = min(10000, eps1.shape[0])
                if n_paths > 0:
                    x_traj = odeint(ode_func, eps1[:n_paths], t_vals, method='dopri5')
                    X = x_traj.detach().cpu().numpy()
                    starts = X[0]
                    ends = X[-1]

                    _render_trajectory_panel(
                        args=args,
                        step=step,
                        sampler=sampler,
                        device=device,
                        trajectories=X,
                        starts=starts,
                        ends=ends,
                        show_legend=True,
                        show_title=False,
                    )

                    _log_time_slice_grid(
                        args=args,
                        step=step,
                        trajectories=X,
                        t_vals=t_vals,
                        key="quantile/trajectory_slices",
                        filename="trajectory_slices",
                    )

        if do_light and hasattr(ode_func, 'reset_nfe'):
            ode_func.reset_nfe()

        evaluation.heavy_eval_batched(
            args,
            x0_batch,
            ode_func,
            sampler,
            step=step,
            big_eval=do_heavy,
            device=device,
            quantile=quantile,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if prev_mode:
        eval_model.train()
    if prev_quant_mode:
        quantile.train()

    return fixed_eval_u
