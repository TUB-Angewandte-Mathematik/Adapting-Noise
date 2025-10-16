import wandb
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter, NullLocator

from learn_noise.utils.colors import COL_DENSITY

def compute_log_joint_grid(
        sampler, 
        x1_range, 
        x2_range, 
        n1=300, 
        n2=300
    ):
    x1_lin = np.linspace(*x1_range, n1)
    x2_lin = np.linspace(*x2_range, n2)
    X1, X2 = np.meshgrid(x1_lin, x2_lin, indexing="ij")
    grid = np.stack([X1.ravel(), X2.ravel()], axis=-1)
    logp = sampler.log_prob(torch.from_numpy(grid)).cpu().numpy().reshape(n1, n2)
    return x1_lin, x2_lin, logp

def _analytic_funnel_x2_pdf(x2_grid: np.ndarray, scale1: float, gh_n: int = 80) -> np.ndarray:
    """
    Compute the marginal p(x2) via Gauss–Hermite quadrature:
      x1 ~ N(0, scale1^2), x2 | x1 ~ N(0, exp(x1))
      p(x2) = E_{x1}[ N(x2; 0, exp(x1)) ]

    Using GH: E_{Z~N(0,1)}[f(Z)] ≈ (1/sqrt(pi)) Σ w_i f(√2 x_i)
      => E_{X1~N(0,scale1^2)}[g(X1)] ≈ (1/sqrt(pi)) Σ w_i g(scale1 * √2 * x_i)

    Args:
      x2_grid: array of x2 values
      scale1:  std of x1 (e.g., 3.0)
      gh_n:    quadrature order
    """
    from numpy.polynomial.hermite import hermgauss

    x2 = x2_grid.astype(np.float64)
    nodes, weights = hermgauss(gh_n)
    # Transform nodes for Normal(0, scale1^2)
    x1_vals = (np.sqrt(2.0) * scale1) * nodes  # shape (N,)
    w_norm = weights / np.sqrt(np.pi)          # weights for E[·]

    # For each x1, the conditional is N(0, var=exp(x1))
    var = np.exp(x1_vals)                      # (N,)
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)

    # Compute mixture pdf across quadrature nodes
    # pdf_i(x2) = 1/sqrt(2π var_i) * exp(-x2^2 / (2 var_i))
    # Vectorized: (N,1) over x1 nodes vs (1,M) over x2 grid
    var_col = var[:, None]
    coef = inv_sqrt_2pi / np.sqrt(var_col)
    expo = np.exp(- (x2[None, :]**2) / (2.0 * var_col))
    pdf_matrix = coef * expo                    # (N, M)
    pdf = (w_norm[:, None] * pdf_matrix).sum(axis=0)

    # Numerical floor to avoid 0 on log-y plots
    return np.maximum(pdf, 1e-300)

def plot_funnel_2d(
        generated, 
        sampler, 
        step, 
        big_eval=False,
        path=None
    ):
    # ----- fixed axis ranges -----
    X2_MIN, X2_MAX = -999.0, 999.0   # horizontal (x-axis): x2
    X1_MIN, X1_MAX =  -20.0,    20.0   # vertical (y-axis):   x1

    n_data = generated.shape[0]
    # Use a larger true sample for a smoother red outline in tails
    n_true = n_data  # max(n_data, min(200_000, 10 * n_data))
    S_data = sampler.sample(n_true,)

    x1_d = generated[:, 0]
    x2_d = generated[:, 1]

    x1_m = S_data[:, 0]
    x2_m = S_data[:, 1]

    # ----- evaluate TRUE log p(x1,x2) on the fixed grid -----
    x1_lin, x2_lin, logp = compute_log_joint_grid(
        sampler, (X1_MIN, X1_MAX), (X2_MIN, X2_MAX), n1=320, n2=360
    )

    # histogram bins aligned with fixed axes
    bins_x2 = np.linspace(X2_MIN, X2_MAX, 50)
    bins_x1 = np.linspace(X1_MIN, X1_MAX, 50)

    # ----- figure -----
    fig = plt.figure(figsize=(8, 8), dpi=160)
    GAP = 0.05 # <— small gap so top “1000” and right “0” don’t collide
    gs = GridSpec(4, 4, figure=fig, hspace=GAP, wspace=GAP)
    ax_main  = fig.add_subplot(gs[1:, :3])
    ax_top   = fig.add_subplot(gs[0, :3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

    teal = "#7fb8c8"
    red  = "#e74c3c"

    # main: true log-joint with floor at -20 mapped to black
    ax_main.set_facecolor("black")
    cmap = plt.cm.magma.copy()
    cmap.set_under("black")
    log_floor = -20.0
    vmax = float(np.max(logp))
    ax_main.imshow(
        logp,
        origin="lower",
        extent=[X2_MIN, X2_MAX, X1_MIN, X1_MAX],
        aspect="auto",
        cmap=cmap,
        vmin=log_floor,
        vmax=vmax,
    )

    # scatter (teal)
    ax_main.scatter(x2_d, x1_d, s=6, alpha=0.5,
                    color=teal, linewidths=0, edgecolors="none")

    ax_main.set_xlabel(r"$x_2$", color="white")
    ax_main.set_ylabel(r"$x_1$", color="white")
    # fixed limits
    ax_main.set_xlim(X2_MIN, X2_MAX)
    ax_main.set_ylim(X1_MIN, X1_MAX)

    # top: x2 marginal (log-y) — data filled teal, model red outline
    ax_top.set_yscale("log")
    ax_top.hist(x2_m, bins=bins_x2, density=True, histtype="step", color=red, linewidth=2.0)
    ax_top.hist(x2_d, bins=bins_x2, density=True, color=teal, alpha=0.35, edgecolor=teal)

    # Analytic overlay for p(x2) using Gauss–Hermite quadrature
    x2_centers = 0.5 * (bins_x2[:-1] + bins_x2[1:])
    scale1 = float(getattr(sampler, 'scale1', torch.tensor(3.0)).item())
    px2 = _analytic_funnel_x2_pdf(x2_centers, scale1=scale1, gh_n=80)
    ax_top.plot(x2_centers, px2, color="#1f77b4", linewidth=2.2, alpha=0.95, label="analytic", zorder=5)
    ax_top.tick_params(labelbottom=False)

    # right: x1 marginal (horizontal) — data filled teal, model red outline
    ax_right.hist(x1_d, bins=bins_x1, density=True, orientation="horizontal",
                  color=teal, alpha=0.35, edgecolor=teal)
    ax_right.hist(x1_m, bins=bins_x1, density=True, orientation="horizontal",
                  histtype="step", color=red, linewidth=2.0)
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel(r"$p(x_1)$")

    out = "funnel_true_all_blackfloor_fixed.pdf"
    plt.savefig(os.path.join(path, f'samples_epoch_{step:03d}.pdf'))
    if big_eval:
      wandb.log({"eval/scatter_plot_big": wandb.Image(plt)}, step=step)
    else:
      wandb.log({"eval/scatter_plot": wandb.Image(plt)}, step=step)

    plt.close()


def _sym_limits_from_arrays(a: np.ndarray, b: np.ndarray, q: float = 99.5,
                            x1_floor: float = 3.0, x1_ceil: float = 20.0,
                            x2_floor: float = 3.0, x2_ceil: float = 1000.0):
    both = np.concatenate([a, b], axis=0)
    ax1 = np.percentile(np.abs(both[:, 0]), q)
    ax2 = np.percentile(np.abs(both[:, 1]), q)
    r1 = float(np.clip(ax1, x1_floor, x1_ceil))
    r2 = float(np.clip(ax2, x2_floor, x2_ceil))
    return (-r1, r1), (-r2, r2)


def plot_generic_2d(
        generated,
        sampler,
        step,
        big_eval=False,
        path=None
    ):
    """Generic 2D plotting for targets without analytic log_prob.

    - No density background
    - Scatter of generated samples overlaid with true samples
    - Top/right marginals as hist overlays (data red outline, model filled teal)
    - Adaptive symmetric limits from percentiles to avoid outliers dominating
    """
    n_data = generated.shape[0]
    S_data = sampler.sample(n_data,)

    x1_d = generated[:, 0].cpu().numpy() if isinstance(generated, torch.Tensor) else generated[:, 0]
    x2_d = generated[:, 1].cpu().numpy() if isinstance(generated, torch.Tensor) else generated[:, 1]

    x1_m = S_data[:, 0].cpu().numpy() if isinstance(S_data, torch.Tensor) else S_data[:, 0]
    x2_m = S_data[:, 1].cpu().numpy() if isinstance(S_data, torch.Tensor) else S_data[:, 1]

    gen_np = np.stack([x1_d, x2_d], axis=-1)
    data_np = np.stack([x1_m, x2_m], axis=-1)
    (x1_min, x1_max), (x2_min, x2_max) = _sym_limits_from_arrays(gen_np, data_np)

    bins_x1 = np.linspace(x1_min, x1_max, 60)
    bins_x2 = np.linspace(x2_min, x2_max, 60)

    fig = plt.figure(figsize=(8, 8), dpi=160)
    GAP = 0.05
    gs = GridSpec(4, 4, figure=fig, hspace=GAP, wspace=GAP)
    ax_main  = fig.add_subplot(gs[1:, :3])
    ax_top   = fig.add_subplot(gs[0, :3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

    teal = "#7fb8c8"
    red  = "#e74c3c"

    ax_main.scatter(x2_m, x1_m, s=4, alpha=0.25, color=red, linewidths=0, edgecolors="none")
    ax_main.scatter(x2_d, x1_d, s=6, alpha=0.6, color=teal, linewidths=0, edgecolors="none")
    ax_main.set_xlabel(r"$x_2$")
    ax_main.set_ylabel(r"$x_1$")
    ax_main.set_xlim(x2_min, x2_max)
    ax_main.set_ylim(x1_min, x1_max)

    ax_top.hist(x2_m, bins=bins_x2, density=True, histtype="step", color=red, linewidth=2.0)
    ax_top.hist(x2_d, bins=bins_x2, density=True, color=teal, alpha=0.35, edgecolor=teal)
    ax_top.tick_params(labelbottom=False)

    ax_right.hist(x1_d, bins=bins_x1, density=True, orientation="horizontal",
                  color=teal, alpha=0.35, edgecolor=teal)
    ax_right.hist(x1_m, bins=bins_x1, density=True, orientation="horizontal",
                  histtype="step", color=red, linewidth=2.0)
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel(r"$p(x_1)$")

    plt.savefig(os.path.join(path, f'samples_epoch_{step:03d}.pdf'))
    if big_eval:
        wandb.log({"eval/scatter_plot_big": wandb.Image(plt)}, step=step)
    else:
        wandb.log({"eval/scatter_plot": wandb.Image(plt)}, step=step)
    plt.close()


@torch.no_grad()
def plot_latent_colored_by_target_norm(
    latent: torch.Tensor,           # eps at t=1, shape (N,2)
    targets: torch.Tensor,          # generated x at t=0, shape (N,2)
    step: int,
    path: str,
    big_eval: bool = False,
    title: str = "Latent colored by ||x||",
):
    """
    Two-panel figure:
    - Left: raw Gaussian samples in latent space.
    - Right: same latent points colored by the norm of their reached target.

    Logged to W&B under eval keys (big vs light eval differentiated by suffix).
    """
    os.makedirs(path or ".", exist_ok=True)

    L = latent.detach().cpu().numpy() if isinstance(latent, torch.Tensor) else latent
    X = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else targets

    import numpy as np
    norms = np.linalg.norm(X, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), dpi=150, constrained_layout=True)

    # Left: plain latent Gaussian
    ax0 = axes[0]
    latent_color = COL_DENSITY if big_eval else "#808080"
    marker_size = 3 if big_eval else 4
    latent_alpha = 0.22 if big_eval else 0.4
    ax0.scatter(L[:, 0], L[:, 1], s=marker_size, alpha=latent_alpha, color=latent_color, linewidths=0)
    ax0.set_title("Latent Gaussian (t=1)")
    ax0.set_aspect('equal', 'box')
    if big_eval:
        ax0.grid(False)
    else:
        ax0.grid(True, alpha=0.2)

    # Right: latent colored by ||x||
    ax1 = axes[1]
    cmap = "afmhot" if big_eval else "viridis"
    color_alpha = 0.6 if big_eval else 0.7
    h = ax1.scatter(L[:, 0], L[:, 1], s=5, c=norms, cmap=cmap, alpha=color_alpha, linewidths=0)
    default_title = "Latent colored by ||x||"
    effective_title = "Latent Colored by ||x||" if title.strip().lower() == default_title.lower() else title
    ax1.set_title(effective_title)
    ax1.set_aspect('equal', 'box')
    if big_eval:
        ax1.grid(False)
    else:
        ax1.grid(True, alpha=0.2)
    cbar = fig.colorbar(h, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("||x||")

    out_name = f'latent_color_epoch_{int(step):03d}.png'
    fig.savefig(os.path.join(path or ".", out_name), bbox_inches='tight', dpi=150)
    wb_key = "eval/latent_color_big" if big_eval else "eval/latent_color"
    wandb.log({wb_key: wandb.Image(fig)}, step=step)
    plt.close(fig)
