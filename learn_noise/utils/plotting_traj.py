import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import animation
from torchdiffeq import odeint
import wandb

# ----------------------------- helpers -----------------------------

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

# -------------------------- trajectory gif -------------------------

@torch.no_grad()
def visualize_and_save(
    ode_func,
    noise,                 # (N,2) tensor at t=T
    T,
    output_dir,
    num_steps=50,
    num_samples=None,      # optional subsample of noise for plotting
    dim=2,
    device="cuda",
    step=None,             # NEW: optional wandb step
    wandb_key="trajectory",# NEW: key used when logging to W&B
    filename="trajectory", # base filename (without extension)
    xlim=None,             # fixed x-axis limits (tuple) if provided
    ylim=None,             # fixed y-axis limits (tuple) if provided
):
    """
    Integrate x'(t)=v and render a GIF with a two-phase time grid (fast then slow).
    Accepts W&B logging args (step, wandb_key). Uses PillowWriter (no ImageMagick).
    """
    assert dim == 2, "visualize_and_save assumes 2D currently."

    os.makedirs(output_dir, exist_ok=True)

    # optionally subsample
    if num_samples is not None and noise.shape[0] > num_samples:
        idx = torch.randperm(noise.shape[0], device=noise.device)[:num_samples]
        noise = noise[idx]

    # make sure everything's on device
    noise = noise.to(device)

    # two-phase grid
    slow_frac = 0.2
    fast_steps = max(1, int(num_steps * (1 - slow_frac)))
    slow_steps = max(1, num_steps - fast_steps)
    t_fast = np.linspace(T, T * slow_frac, fast_steps, endpoint=False)
    t_slow = np.linspace(T * slow_frac, 0.0, slow_steps)
    t_vals = torch.tensor(np.concatenate([t_fast, t_slow]), device=device, dtype=torch.float32)

    # integrate
    x_traj = odeint(ode_func, noise, t_vals, method='dopri5')  # (num_steps, N, 2)

    # figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    sm_xlim = sm_ylim = None
    alpha = 0.09

    def update(frame):
        nonlocal sm_xlim, sm_ylim
        ax.clear()
        x_frame = _to_numpy(x_traj[frame])
        # axis limits: fixed if provided, else smooth dynamic
        if xlim is not None and ylim is not None:
            sm_xlim, sm_ylim = xlim, ylim
        else:
            # smooth axis limits
            min_v = x_frame.min(axis=0)
            max_v = x_frame.max(axis=0)
            pad = 0.2 * (max_v - min_v + 1e-3)
            raw_xlim = (min_v[0] - pad[0], max_v[0] + pad[0])
            raw_ylim = (min_v[1] - pad[1], max_v[1] + pad[1])
            if sm_xlim is None:
                sm_xlim, sm_ylim = raw_xlim, raw_ylim
            else:
                sm_xlim = (
                    alpha * raw_xlim[0] + (1 - alpha) * sm_xlim[0],
                    alpha * raw_xlim[1] + (1 - alpha) * sm_xlim[1],
                )
                sm_ylim = (
                    alpha * raw_ylim[0] + (1 - alpha) * sm_ylim[0],
                    alpha * raw_ylim[1] + (1 - alpha) * sm_ylim[1],
                )
        ax.set_xlim(*sm_xlim)
        ax.set_ylim(*sm_ylim)
        ax.set_title(f"t = {float(t_vals[frame]):.2f}")
        ax.scatter(x_frame[:, 0], x_frame[:, 1], s=2, alpha=0.5)

    pause = 20
    frames = list(range(len(t_vals))) + [len(t_vals) - 1] * pause
    ani = animation.FuncAnimation(fig, update, frames=frames, repeat=False)

    # Save with Pillow (no ImageMagick requirement)
    gif_path = os.path.join(output_dir, f'{filename}.gif')
    try:
        writer = animation.PillowWriter(fps=5)
        ani.save(gif_path, writer=writer)
    finally:
        plt.close(fig)

    # W&B logging
    wandb.log({wandb_key: wandb.Video(gif_path, format="gif")}, step=step)

# -------------------------- density background ---------------------

@torch.no_grad()
def compute_log_joint_grid(sampler, x1_range, x2_range, n1=300, n2=300, device="cpu"):
    x1_lin = np.linspace(*x1_range, n1)
    x2_lin = np.linspace(*x2_range, n2)
    X1, X2 = np.meshgrid(x1_lin, x2_lin, indexing="ij")
    grid = np.stack([X1.ravel(), X2.ravel()], axis=-1)
    grid_t = torch.from_numpy(grid).to(device=device, dtype=torch.float32)
    logp = sampler.log_prob(grid_t).detach().cpu().numpy().reshape(n1, n2)
    return x1_lin, x2_lin, logp

# ------------------------------- plots -----------------------------

@torch.no_grad()
def plot_2d(generated, sampler, step, path=None, device="cpu"):
    """
    Scatter the generated samples over a fixed-range true log-density background
    plus marginals on the top/right. Saves PNG and logs figure to W&B.
    """
    os.makedirs(path or ".", exist_ok=True)

    # fixed ranges (x2 horizontal, x1 vertical)
    X2_MIN, X2_MAX = -1000.0, 1000.0
    X1_MIN, X1_MAX =   -20.0,   20.0

    g = generated
    if isinstance(g, torch.Tensor):
        g = g.detach().cpu().numpy()

    n_data = g.shape[0]
    S_data = sampler.sample((n_data,)).detach().cpu().numpy()

    x1_d, x2_d = g[:, 0], g[:, 1]
    x1_m, x2_m = S_data[:, 0], S_data[:, 1]

    # background log p on fixed grid
    _, _, logp = compute_log_joint_grid(
        sampler, (X1_MIN, X1_MAX), (X2_MIN, X2_MAX), n1=320, n2=360, device=device
    )

    bins_x2 = np.linspace(X2_MIN, X2_MAX, 140)
    bins_x1 = np.linspace(X1_MIN, X1_MAX, 120)

    fig = plt.figure(figsize=(8, 8), dpi=160)
    gs = GridSpec(4, 4, figure=fig, hspace=0.02, wspace=0.02)
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

    # scatter
    ax_main.scatter(x2_d, x1_d, s=6, alpha=0.5, color=teal, linewidths=0)
    ax_main.set_xlabel(r"$x_2$", color="white")
    ax_main.set_ylabel(r"$x_1$", color="white")
    ax_main.set_xlim(X2_MIN, X2_MAX)
    ax_main.set_ylim(X1_MIN, X1_MAX)
    ax_main.tick_params(colors="white")

    # top marginal (x2)
    ax_top.set_yscale("log")
    ax_top.hist(x2_d, bins=bins_x2, density=True, color=teal, alpha=0.35, edgecolor=teal)
    ax_top.hist(x2_m, bins=bins_x2, density=True, histtype="step", color=red, linewidth=2.0)
    ax_top.tick_params(labelbottom=False)

    # right marginal (x1)
    ax_right.hist(x1_d, bins=bins_x1, density=True, orientation="horizontal",
                  color=teal, alpha=0.35, edgecolor=teal)
    ax_right.hist(x1_m, bins=bins_x1, density=True, orientation="horizontal",
                  histtype="step", color=red, linewidth=2.0)
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel(r"$p(x_1)$")

    out_path = os.path.join(path or ".", f'samples_epoch_{int(step):03d}.png')
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    wandb.log({"val_plot": wandb.Image(fig)}, step=step)
    plt.close(fig)

@torch.no_grad()
def plot_latent(samples, step, path):
    """
    Simple latent scatter; saves and logs. Accepts torch or numpy.
    """
    os.makedirs(path or ".", exist_ok=True)

    S = samples
    if isinstance(S, torch.Tensor):
        S_np = S.detach().cpu().numpy()
        mean0 = float(S[:, 0].mean().item())
        var0  = float(S[:, 0].var(unbiased=False).item())
    else:
        S_np = np.asarray(S)
        mean0 = float(S_np[:, 0].mean())
        var0  = float(S_np[:, 0].var())

    fig = plt.figure(figsize=(4, 4), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(S_np[:, 0], S_np[:, 1], s=2, alpha=0.5)
    ax.set_aspect('equal', 'box')
    ax.grid(True, alpha=0.2)

    png_path = os.path.join(path, f'latent_samples_epoch_{int(step):03d}.png')
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    wandb.log({"Latent": wandb.Image(fig),
               "Latent/mean_x": mean0,
               "Latent/var_x": var0}, step=step)
    plt.close(fig)

@torch.no_grad()
def plot_cdf_1d_monotone(net_cdf, x_min=None, x_max=None, n=1200, device="cpu", title="Learned CDF", step=0):
    """
    Dense plot of the learned CDF. Uses the network's configured domain if needed.
    """
    net_cdf.eval()
    x_min = float(net_cdf.xmin.item() if x_min is None else x_min)
    x_max = float(net_cdf.xmax.item() if x_max is None else x_max)

    xs = torch.linspace(x_min, x_max, n, device=device).unsqueeze(-1)
    cdf = net_cdf(xs).squeeze(-1).clamp(0., 1.).detach().cpu().numpy()
    xs  = xs.squeeze(-1).detach().cpu().numpy()

    fig = plt.figure(dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xs, cdf, linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    wandb.log({"CDF": wandb.Image(fig)}, step=step)
    plt.close(fig)