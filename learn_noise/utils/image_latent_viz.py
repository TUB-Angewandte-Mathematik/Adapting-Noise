"""Utilities for visualizing high-dimensional image latents."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

PER_CLASS_HIST_MIN = 64


CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

DATASET_CLASS_NAMES = {
    "cifar10": CIFAR10_CLASS_NAMES,
    "cifar": CIFAR10_CLASS_NAMES,
    "cifar_lt": CIFAR10_CLASS_NAMES,
    "cifar10_lt": CIFAR10_CLASS_NAMES,
}


def resolve_class_names(dataset: Optional[str], num_classes: Optional[int]) -> Optional[List[str]]:
    if dataset is None:
        return None
    names = DATASET_CLASS_NAMES.get(dataset.lower())
    if names is None:
        return None
    if num_classes is not None:
        return list(names[:num_classes])
    return list(names)


def _convert_for_imshow(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[0] in {1, 3}:
        # CHW -> HWC
        return np.moveaxis(image, 0, -1)
    if image.shape[-1] in {1, 3}:
        return image
    raise ValueError("Unexpected image layout for imshow")


def _reshape_latents(latents: torch.Tensor, image_shape: Sequence[int]) -> np.ndarray:
    latents_np = latents.detach().cpu().numpy()
    return latents_np.reshape(latents.shape[0], *image_shape)


def _compute_mean_std(latents: torch.Tensor, image_shape: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    reshaped = _reshape_latents(latents, image_shape)
    mean_img = reshaped.mean(axis=0)
    std_img = reshaped.std(axis=0)
    return mean_img, std_img


def make_mean_std_figure(
    latents: torch.Tensor,
    image_shape: Sequence[int],
    *,
    mean_img: Optional[np.ndarray] = None,
    std_img: Optional[np.ndarray] = None,
) -> plt.Figure:
    if mean_img is None or std_img is None:
        mean_img, std_img = _compute_mean_std(latents, image_shape)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
    ax_mean, ax_std = axes

    im_mean = ax_mean.imshow(_convert_for_imshow(mean_img), cmap="viridis")
    ax_mean.set_title("Latent mean")
    ax_mean.axis("off")
    fig.colorbar(im_mean, ax=ax_mean, fraction=0.046, pad=0.04)

    im_std = ax_std.imshow(_convert_for_imshow(std_img), cmap="magma")
    ax_std.set_title("Latent std")
    ax_std.axis("off")
    fig.colorbar(im_std, ax=ax_std, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def make_stat_image_figure(
    stat_image: np.ndarray,
    image_shape: Sequence[int],
    *,
    cmap: Optional[str] = None,
) -> plt.Figure:
    disp = _convert_for_imshow(stat_image)
    height = disp.shape[0]
    width = disp.shape[1]
    target_cell = 1.1
    scale = target_cell / max(height, width) if max(height, width) > 0 else target_cell
    fig_w = max(target_cell, width * scale)
    fig_h = max(target_cell, height * scale)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=140)
    if cmap is not None:
        ax.imshow(disp, cmap=cmap)
    else:
        ax.imshow(disp)
    ax.axis("off")
    fig.tight_layout(pad=0.0)
    return fig


def _standard_normal_quantiles(n: int, device: torch.device) -> torch.Tensor:
    prob = (torch.arange(1, n + 1, device=device, dtype=torch.float32) - 0.5) / float(n)
    return math.sqrt(2.0) * torch.erfinv(2.0 * prob - 1.0)


def make_channel_pixel_histograms(
    latents: torch.Tensor,
    image_shape: Sequence[int],
    pixel_coords: Optional[Iterable[Tuple[int, int]]] = None,
    *,
    num_bins: int = 60,
    title_prefix: str = "",
) -> plt.Figure:
    reshaped = latents.reshape(latents.shape[0], *image_shape)
    cdim, h, w = image_shape
    if pixel_coords is None:
        pixel_coords = [
            (0, 0),
            (0, 16),
            (16, 0),
            (16, 16),
            (24, 24),
            (8, 24),
        ]
    coords = list(pixel_coords)

    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    channel_labels = ["Red", "Green", "Blue"] if cdim >= 3 else ["Channel 0"]

    total_rows = len(coords)
    fig, axes = plt.subplots(
        total_rows,
        cdim * 2,
        figsize=(4 * cdim, 3 * total_rows),
        dpi=140,
        squeeze=False,
    )
    device = latents.device

    for row_offset, (y, x) in enumerate(coords):
        for ch in range(cdim):
            samples = reshaped[:, ch, y, x].to(torch.float32)
            samples_np = samples.cpu().numpy()
            col_hist = ch * 2
            col_qq = col_hist + 1

            ax_hist = axes[row_offset, col_hist]
            color = colors[ch % len(colors)]
            ax_hist.hist(samples_np, bins=num_bins, density=True, color=color, alpha=0.7)
            xs = np.linspace(samples_np.min(), samples_np.max(), 400)
            normal_pdf = (1.0 / math.sqrt(2 * math.pi)) * np.exp(-0.5 * xs ** 2)
            ax_hist.plot(xs, normal_pdf, color="#333333", linewidth=1.0)
            ax_hist.set_title(f"{title_prefix}{channel_labels[ch]} @(y={y}, x={x})")
            ax_hist.set_xlabel("Value")
            ax_hist.set_ylabel("Density")

            sorted_samples, _ = torch.sort(samples)
            theor = _standard_normal_quantiles(len(sorted_samples), device=device)
            ax_qq = axes[row_offset, col_qq]
            ax_qq.scatter(theor.cpu().numpy(), sorted_samples.cpu().numpy(), s=10, alpha=0.6, color=color)
            ax_qq.plot(theor.cpu().numpy(), theor.cpu().numpy(), color="#d62728", linewidth=1.0)
            ax_qq.set_title(f"{title_prefix}QQ vs N(0,1)")
            ax_qq.set_xlabel("Theoretical quantile")
            ax_qq.set_ylabel("Sample quantile")

    fig.tight_layout()
    return fig


def make_latent_pca_scatter(
    latents: torch.Tensor,
    *,
    color_source: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
) -> plt.Figure:
    centered = latents - latents.mean(dim=0, keepdim=True)
    cov = torch.matmul(centered.T, centered) / max(1, centered.shape[0] - 1)
    # eigh for stability on symmetric covariance
    evals, evecs = torch.linalg.eigh(cov)
    top2 = evecs[:, -2:]
    proj = torch.matmul(centered, top2)
    proj_np = proj.cpu().numpy()

    if labels is not None and num_classes is not None and num_classes > 0:
        colors = labels.cpu().numpy()
        cbar_label = "Class label"
        cmap = "tab10"
        norm_args = dict(vmin=0, vmax=num_classes - 1)
    elif color_source is not None:
        colors = color_source.cpu().numpy()
        cbar_label = "Sample mean intensity"
        cmap = "viridis"
        norm_args = {}
    else:
        colors = centered.norm(dim=1).cpu().numpy()
        cbar_label = "Latent norm"
        cmap = "viridis"
        norm_args = {}

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=140)
    sc = ax.scatter(proj_np[:, 0], proj_np[:, 1], c=colors, cmap=cmap, s=12, alpha=0.7, **norm_args)
    ax.set_title("Latent PCA (top-2 components)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(cbar_label)
    if labels is not None and num_classes is not None and num_classes > 0 and class_names is not None:
        ticks = list(range(num_classes))
        cbar.set_ticks(ticks)
        display_names = [class_names[t] if 0 <= t < len(class_names) else str(t) for t in ticks]
        cbar.set_ticklabels(display_names)
    fig.tight_layout()
    return fig


def make_correlation_heatmap(
    latents: torch.Tensor,
    image_shape: Sequence[int],
    *,
    patch_size: int = 10,
) -> plt.Figure:
    reshaped = latents.reshape(latents.shape[0], *image_shape)
    _, h, w = image_shape
    ps = min(patch_size, h, w)
    y0 = (h - ps) // 2
    x0 = (w - ps) // 2
    patch = reshaped[:, :, y0 : y0 + ps, x0 : x0 + ps]
    flat = patch.reshape(patch.shape[0], -1).cpu().numpy()
    corr = np.corrcoef(flat, rowvar=False)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=140)
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title(f"Latent correlation, patch {ps}x{ps}")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Feature index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def make_atlas_figure(
    images: torch.Tensor,
    nrow: int,
    *,
    labels: Optional[torch.Tensor] = None,
    class_names: Optional[Sequence[str]] = None,
) -> plt.Figure:
    grid = make_grid(images, nrow=nrow, normalize=False)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = np.clip((grid_np + 1.0) / 2.0, 0.0, 1.0)

    fig, ax = plt.subplots(1, 1, figsize=(nrow * 1.1, max(1, images.shape[0] // nrow) * 1.1), dpi=140)
    ax.imshow(_convert_for_imshow(grid_np))
    ax.axis("off")

    if labels is not None:
        labels_np = labels.cpu().numpy()
        total_rows = int(math.ceil(images.shape[0] / nrow))
        img_h = grid_np.shape[0]
        cell_h = img_h / max(1, total_rows)
        for row in range(total_rows):
            anchor = row * nrow
            if anchor >= labels_np.shape[0]:
                break
            label_idx = int(labels_np[anchor])
            if class_names is not None and 0 <= label_idx < len(class_names):
                text = class_names[label_idx]
            else:
                text = f"Class {label_idx}"
            ax.text(
                4,
                row * cell_h + cell_h / 2.0,
                text,
                color="white",
                fontsize=10,
                ha="left",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5),
            )

    fig.tight_layout()
    return fig


def make_temporal_latent_grid(
    latents: torch.Tensor,
    *,
    tau_values: Sequence[float],
    row_labels: Optional[Sequence[str]] = None,
) -> plt.Figure:
    """
    Render a panel showing latent snapshots across multiple τ values.

    Args:
        latents: Tensor shaped (N, T, C, H, W) containing latent images per sample/time.
        tau_values: Sequence of τ values aligned with dimension T (displayed as column headers).
        row_labels: Optional per-sample labels rendered along the left edge.
    """
    if latents.dim() != 5:
        raise ValueError("Expected (N, T, C, H, W) tensor for temporal latent grid")
    num_samples, num_steps, _, _, _ = latents.shape
    if num_steps != len(tau_values):
        raise ValueError("tau_values must match the second dimension of latents")

    flat = latents.reshape(num_samples * num_steps, *latents.shape[2:])
    grid = make_grid(
        flat,
        nrow=num_steps,
        normalize=False,
    )
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = np.clip(grid_np, 0.0, 1.0)

    fig_w = max(1.8, num_steps * 1.4)
    fig_h = max(1.8, num_samples * 1.2)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=140)
    ax.imshow(_convert_for_imshow(grid_np))
    ax.axis("off")

    height, width = grid_np.shape[0], grid_np.shape[1]
    if num_steps > 0:
        cell_w = width / num_steps
        for idx, tau in enumerate(tau_values):
            ax.text(
                (idx + 0.5) * cell_w,
                0.025 * height,
                f"τ={tau:.2f}",
                ha="center",
                va="top",
                fontsize=10,
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
            )

    if row_labels is not None and len(row_labels) == num_samples:
        cell_h = height / max(1, num_samples)
        for row_idx, label in enumerate(row_labels):
            ax.text(
                5,
                (row_idx + 0.5) * cell_h,
                label,
                ha="left",
                va="center",
                fontsize=10,
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
            )

    fig.tight_layout(pad=0.05)
    return fig


def make_quantile_location_diagnostic(
    data_samples: torch.Tensor,
    model_samples: torch.Tensor,
    *,
    u_values: np.ndarray,
    locs: Sequence[Tuple[int, int]],
    channel_names: Optional[Sequence[str]] = None,
    dataset: Optional[str] = None,
) -> plt.Figure:
    """
    Compare empirical data statistics with model quantiles for selected pixel locations.
    The panel renders (per location & channel) the empirical CDF, quantile function,
    and density estimate derived from histogram smoothing.
    """
    if data_samples.dim() != 4 or model_samples.dim() != 4:
        raise ValueError("Expected (N, C, H, W) tensors for data/model samples")
    if data_samples.shape[1:] != model_samples.shape[1:]:
        raise ValueError("data_samples and model_samples must share channel/height/width")
    if len(locs) == 0:
        raise ValueError("locs must contain at least one (y, x) entry")

    channels, height, width = data_samples.shape[1:]
    if channel_names is None:
        if channels == 3:
            channel_names = ["Red", "Green", "Blue"]
        elif channels == 1:
            channel_names = ["Intensity"]
        else:
            channel_names = [f"Channel {idx}" for idx in range(channels)]

    u_vals = np.asarray(u_values, dtype=np.float64)
    if u_vals.ndim != 1 or u_vals.size < 2:
        raise ValueError("u_values must be a 1D array with at least two entries")

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    num_blocks = len(locs) * channels
    fig_w = max(3.4 * num_blocks, 6.5)
    fig_h = 6.4
    fig, axs = plt.subplots(
        2,
        num_blocks,
        figsize=(fig_w, fig_h),
        dpi=160,
        squeeze=False,
    )

    col_idx = 0
    for (y, x) in locs:
        if not (0 <= y < height and 0 <= x < width):
            raise ValueError(f"Location $(y={y}, x={x})$ out of bounds for shape (H={height}, W={width})")
        for ch in range(channels):
            data_px = data_samples[:, ch, y, x].detach().cpu().numpy()
            model_px = model_samples[:, ch, y, x].detach().cpu().numpy()
            if data_px.size == 0 or model_px.size == 0:
                continue

            color = palette[ch % len(palette)]
            chan_name = channel_names[ch] if ch < len(channel_names) else f"Channel {ch}"
            title_prefix = f"$(y={y}, x={x})$"

            ax_quant = axs[0, col_idx]
            ax_pdf = axs[1, col_idx]

            q_emp = np.quantile(data_px, u_vals, method="linear")
            q_mod = np.quantile(model_px, u_vals, method="linear")
            ax_quant.plot(u_vals, q_emp, color="#444444", linewidth=1.5, label="Empirical")
            ax_quant.plot(u_vals, q_mod, color=color, linewidth=1.8, label="Model")
            ax_quant.fill_between(u_vals, q_emp, q_mod, color=color, alpha=0.18)
            ax_quant.set_title(f"{title_prefix} — Quantile")
            ax_quant.set_xlabel("u")
            ax_quant.set_ylabel("Value")

            bins = max(15, int(np.sqrt(max(data_px.size, model_px.size))))
            data_hist, data_edges = np.histogram(data_px, bins=bins, density=True)
            model_hist, model_edges = np.histogram(model_px, bins=bins, density=True)
            data_centers = 0.5 * (data_edges[:-1] + data_edges[1:])
            model_centers = 0.5 * (model_edges[:-1] + model_edges[1:])
            ax_pdf.plot(data_centers, data_hist, color="#777777", linewidth=1.2, label="Empirical")
            ax_pdf.plot(model_centers, model_hist, color=color, linewidth=1.8, label="Model")
            ax_pdf.set_title(f"{title_prefix} — Density")
            ax_pdf.set_xlabel("Value")
            ax_pdf.set_ylabel("Density")
            ax_pdf.set_ylim(0.0, 5.0)

            for sub_ax in (ax_quant, ax_pdf):
                sub_ax.legend(frameon=False, fontsize=8)
                sub_ax.grid(alpha=0.25, linewidth=0.6)

            col_idx += 1

    while col_idx < num_blocks:
        for row in range(2):
            axs[row, col_idx].axis("off")
        col_idx += 1

    dataset_name = dataset.upper() if dataset is not None else ""
    #if dataset_name:
    #    fig.suptitle(f"Quantiles — {dataset_name}", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


@dataclass
class LatentVizPayload:
    mean_std_fig: Optional[plt.Figure]
    mean_fig: Optional[plt.Figure]
    std_fig: Optional[plt.Figure]
    hist_qq_fig: Optional[plt.Figure]
    pca_fig: Optional[plt.Figure]
    corr_fig: Optional[plt.Figure]
    atlas_fig: Optional[plt.Figure]
    per_class_hist_figs: Optional[dict[int, plt.Figure]] = None


def build_latent_visualizations(
    latents: torch.Tensor,
    *,
    image_shape: Sequence[int],
    atlas_images: Optional[torch.Tensor] = None,
    atlas_nrow: Optional[int] = None,
    color_source: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    num_classes: Optional[int] = None,
    atlas_labels: Optional[torch.Tensor] = None,
    class_names: Optional[Sequence[str]] = None,
    dataset: Optional[str] = None,
) -> LatentVizPayload:
    if latents.dim() != 2:
        raise ValueError("Expected flattened latent batch for visualization")

    mean_img, std_img = _compute_mean_std(latents, image_shape)

    dataset_name = dataset.lower() if dataset is not None else None
    if dataset_name == "mnist":
        cmap = "gray" if image_shape and image_shape[0] == 1 else None
        mean_std = None
        mean_fig = make_stat_image_figure(mean_img, image_shape, cmap=cmap)
        std_fig = make_stat_image_figure(std_img, image_shape, cmap=cmap)
    else:
        mean_fig = None
        std_fig = None
        mean_std = make_mean_std_figure(latents, image_shape, mean_img=mean_img, std_img=std_img)

    per_class_hist_figs: Optional[dict[int, plt.Figure]] = None
    if labels is not None and num_classes is not None and num_classes > 0:
        hist_qq = None
        per_class_hist_figs = {}
        labels_long = labels.to(torch.long)
        for cls in range(num_classes):
            mask = labels_long == cls
            if mask.sum() >= PER_CLASS_HIST_MIN:
                prefix = (
                    f"{class_names[cls]} " if class_names is not None and 0 <= cls < len(class_names) else f"Class {cls} "
                )
                fig_cls = make_channel_pixel_histograms(
                    latents[mask],
                    image_shape,
                    pixel_coords=None,
                    title_prefix=prefix,
                )
                per_class_hist_figs[cls] = fig_cls
    else:
        hist_qq = make_channel_pixel_histograms(
            latents,
            image_shape,
            pixel_coords=None,
        )
    pca_fig = make_latent_pca_scatter(
        latents,
        color_source=color_source,
        labels=labels,
        num_classes=num_classes,
        class_names=class_names,
    )
    corr_fig = make_correlation_heatmap(latents, image_shape)

    atlas_fig = None
    if atlas_images is not None:
        nrow = atlas_nrow if atlas_nrow is not None else int(math.sqrt(atlas_images.shape[0])) or 1
        atlas_fig = make_atlas_figure(
            atlas_images,
            nrow=nrow,
            labels=atlas_labels,
            class_names=class_names,
        )

    return LatentVizPayload(
        mean_std_fig=mean_std,
        mean_fig=mean_fig,
        std_fig=std_fig,
        hist_qq_fig=hist_qq,
        pca_fig=pca_fig,
        corr_fig=corr_fig,
        atlas_fig=atlas_fig,
        per_class_hist_figs=per_class_hist_figs,
    )
