import math
import os
import json
from typing import Callable, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import wandb
from tqdm import tqdm
from geomloss import SamplesLoss

try:
    import ot  # type: ignore
except ImportError:
    ot = None

from learn_noise.networks import RQSQuantile
import learn_noise.utils.sampler as smpl
from learn_noise.training.logging import log_real_rgb_histogram_once
from learn_noise.utils.quantile_viz import log_noise_slices, log_xt_slices
from learn_noise.utils.image_eval import reshape_flat_samples, save_image_grid
from learn_noise.utils.image_latent_viz import build_latent_visualizations


def build_quantile(args, device: torch.device, dim: int):
    """Instantiate the unified RQS quantile for both low-dimensional and image tasks."""
    input_transform = getattr(args, "q_input_transform", "logit")

    common_kwargs = dict(
        dim=dim,
        n_bins=int(args.q_rqs_bins),
        bound=float(args.q_rqs_bound),
        num_layers=int(args.q_rqs_layers),
        eps=float(args.q_u_eps),
        input_transform=input_transform,
    )

    grouping = getattr(args, "q_condition_grouping", None)
    if grouping is not None:
        print("[build_quantile] q_condition_grouping is deprecated and will be ignored.")

    if getattr(args, "image_shape", None) is not None:
        if bool(getattr(args, "q_normalize_z", False)):
            raise ValueError("q_normalize_z is no longer supported by the shared RQSQuantile.")
        if hasattr(args, "q_normalize_eps") and getattr(args, "q_normalize_eps"):
            raise ValueError("q_normalize_eps is no longer supported by the shared RQSQuantile.")

    return RQSQuantile(**common_kwargs).to(device)


def build_transport_objective(
    args,
    device: torch.device,
) -> Tuple[Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Dict[str, float]]], bool, bool]:
    metric_name_source = getattr(args, "q_objective", None)
    if metric_name_source is None:
        metric_name_source = getattr(args, "q_ot_metric","energy")
    metric_name = metric_name_source.lower()
    if metric_name == 'plan':
        metric_name = 'plan_action'

    blur = max(float(args.q_ot_epsilon) if hasattr(args, "q_ot_epsilon") else 1.0, 1e-8)

    if metric_name in {'energy', 'sinkhorn'}:
        samples_kwargs = {}
        scaling = args.q_ot_scaling if hasattr(args, "q_ot_scaling") else None
        if scaling is not None:
            samples_kwargs['scaling'] = float(scaling)
        samples_loss = SamplesLoss(metric_name, blur=blur, **samples_kwargs)

        def objective(x0: torch.Tensor, dqdt: Optional[torch.Tensor] = None):

            value = samples_loss(x0, dqdt)
            metric_key = f"quantile_ot/{metric_name}"
            metrics = {
                "quantile_ot/transport": float(value.item()),
                metric_key: float(value.item()),
            }
            return value, metrics

        return objective, False, False

    if metric_name == 'plan_action':
        if ot is None:
            raise ImportError("POT (Python Optimal Transport) package is required for q_objective='plan_action'.")

        def objective(x0: torch.Tensor, dqdt_selected: Optional[torch.Tensor] = None):
            if dqdt_selected is None:
                raise ValueError("plan_action objective requires dqdt values; set return_dqdt=True.")
            batch = x0.shape[0]
            
            cost_matrix = 0.5 * (x0.unsqueeze(1) - dqdt_selected.unsqueeze(0)).pow(2).sum(dim=-1)
            cost_np = cost_matrix.detach().cpu().numpy()
            weights = np.full(batch, 1.0 / batch, dtype=cost_np.dtype)
            plan_np = ot.emd(weights, weights, cost_np)
            plan = torch.from_numpy(plan_np).to(device=x0.device, dtype=cost_matrix.dtype)

            idx_best = torch.argmax(plan, dim=0)
            dqdt_best = dqdt_selected

            v_selected = -x0[idx_best] + dqdt_best
            kinetic = 0.5 * v_selected.pow(2).sum(dim=1)
            action_loss = kinetic.mean()

            matched_cost = cost_matrix[torch.arange(batch, device=x0.device), idx_best].mean()
            metrics = {
                "quantile_ot/action": float(action_loss.item()),
                "quantile_ot/plan_cost": float(matched_cost.item()),
                "quantile_ot/transport": float(action_loss.item()),
            }
            return action_loss, metrics

        return objective, True, True

    raise ValueError(
        f"Unsupported q_objective '{metric_name}' (expected 'energy', 'sinkhorn', or 'plan_action')."
    )


def _compute_kl_regularizer(
    quantile: nn.Module,
    U: torch.Tensor,
    tau: torch.Tensor,
    lambda_reg: float,
    device: torch.device,
) -> torch.Tensor:
    if lambda_reg <= 0.0:
        return torch.zeros((), device=device)

    jac_diag = quantile.diag_du(U, tau, create_graph=True)
    logdet = torch.log(jac_diag.clamp_min(1e-12)).sum(dim=1)
    extra_logdet = getattr(quantile, "_last_logabsdet_extra", None)
    if extra_logdet is not None:
        logdet = logdet + extra_logdet.to(logdet.dtype)
    return (-logdet).mean()


def _get_quantile_eval_model(
    quantile: nn.Module,
    ema_quantile: Optional[AveragedModel],
) -> nn.Module:
    return ema_quantile if ema_quantile is not None else quantile


def _maybe_log_image_visualizations(
    *,
    args,
    quant_eval,
    step: int,
    global_step: int,
    is_image_task: bool,
    sample_vis_interval: int,
    sample_vis_count: int,
    sample_vis_nrow: int,
    sample_dir: str,
    fixed_u_vis: Optional[torch.Tensor],
    u_eps: float,
    image_shape: Optional[Tuple[int, ...]],
    device: torch.device,
    dim: int,
) -> Optional[torch.Tensor]:
    if not (is_image_task and sample_vis_interval > 0 and sample_vis_count > 0):
        return fixed_u_vis

    log_noise = ((step + 1) % sample_vis_interval == 0)
    latent_viz_samples = int(args.latent_viz_samples)
    log_latent = log_noise and latent_viz_samples > 0

    if not (log_noise or log_latent):
        return fixed_u_vis

    os.makedirs(sample_dir, exist_ok=True)

    prev_mode = quant_eval.training
    quant_eval.eval()

    try:
        with torch.inference_mode():
            if log_noise:
                if fixed_u_vis is None or fixed_u_vis.shape[0] < sample_vis_count:
                    fixed_u_vis = torch.rand(sample_vis_count, dim, device=device).detach().cpu()
                u_noise = fixed_u_vis[:sample_vis_count].to(device)
                U_noise = u_eps + (1 - 2 * u_eps) * u_noise
                tau_noise = torch.ones(sample_vis_count, 1, device=device)
                eps_noise = quant_eval(U_noise, tau_noise)
                noise_imgs = reshape_flat_samples(eps_noise, torch.Size(image_shape))
                grid_np = save_image_grid(
                    noise_imgs,
                    path=os.path.join(sample_dir, f'step_{global_step:06d}_noise.png'),
                    nrow=sample_vis_nrow,
                )
                wandb.log({"quantile_ot/noise_grid": wandb.Image(grid_np)}, step=global_step)
            if log_latent:
                num_latent = min(latent_viz_samples, 1024)
                if num_latent > 0:
                    unit_u = torch.rand(num_latent, dim, device=device)
                    U_latent = u_eps + (1 - 2 * u_eps) * unit_u
                    ones_latent = torch.ones(num_latent, 1, device=device)
                    eps_latent = quant_eval(U_latent, ones_latent)
                    latents_cpu = eps_latent.detach().cpu()
                    viz_payload = build_latent_visualizations(
                        latents_cpu,
                        image_shape=image_shape,
                        atlas_images=None,
                    )
                    wandb_payload = {}
                    if viz_payload.mean_std_fig is not None:
                        wandb_payload["quantile_ot/mean_std"] = wandb.Image(viz_payload.mean_std_fig)
                    if viz_payload.hist_qq_fig is not None:
                        wandb_payload["quantile_ot/hist_qq"] = wandb.Image(viz_payload.hist_qq_fig)
                    if viz_payload.pca_fig is not None:
                        wandb_payload["quantile_ot/pca"] = wandb.Image(viz_payload.pca_fig)
                    if viz_payload.corr_fig is not None:
                        wandb_payload["quantile_ot/correlation"] = wandb.Image(viz_payload.corr_fig)
                    if wandb_payload:
                        wandb.log(wandb_payload, step=global_step)
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
            quant_eval.train()

    return fixed_u_vis
def _maybe_run_low_dim_evals(
    *,
    args,
    quant_eval: nn.Module,
    sampler,
    device: torch.device,
    global_step: int,
    step: int,
    dim: int,
    u_eps: float,
    normal: torch.distributions.Normal,
) -> None:
    if dim > 2:
        return

    interval = int(args.q_val_interval)
    if interval <= 0:
        return

    if (step + 1) % interval != 0 and step != 0:
        return

    prev_mode = quant_eval.training
    quant_eval.eval()

    with torch.no_grad():
        q_val_samples = int(args.q_val_samples)

    with torch.inference_mode():
        log_noise_slices(
            quant_eval,
            device,
            global_step,
            times=(0.0, 0.03, 0.07, 0.15, 0.25, 0.5, 0.75, 0.9, 0.98, 1.0),
            N=4000,
            dim=dim,
            u_eps=u_eps,
        )

        log_xt_slices(
            quant_eval,
            sampler,
            device,
            global_step,
            times=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
            N=4000,
            dim=dim,
            u_eps=u_eps,
        )



    if prev_mode:
        quant_eval.train()


def pretrain_quantile(args, *, log_step_offset: int = 0) -> Tuple[nn.Module, str, int]:
    """Quantile training with configurable OT objectives and structured logging."""
    device = torch.device(args.device)
    dim = args.dim
    sampler = smpl.get_distribution(args.target_dataset)
    image_shape = getattr(args, "image_shape", None)
    if image_shape is not None:
        image_shape = tuple(image_shape)
    is_image = image_shape is not None

    quantile = build_quantile(args, device, dim)
    opt = torch.optim.Adam([{"params": quantile.parameters(), "lr": args.q_lr}])

    use_ema = bool(args.q_use_ema)
    ema_quantile = AveragedModel(quantile, multi_avg_fn=get_ema_multi_avg_fn(args.q_ema)) if use_ema else None

    transport_objective, requires_dqdt, _ = build_transport_objective(args, device)

    u_eps = float(args.q_u_eps)
    batch_size = int(args.q_batch)
    q_loss_weight = float(getattr(args, "q_loss_weight", 1.0))
    lambda_reg = float(getattr(args, "lambda_reg", 0.0))


    normal = torch.distributions.Normal(0.0, 1.0)

    def _fixed_sampler_batch(batch_size: int, seed_offset: int = 0):
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(args.seed + seed_offset)
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + seed_offset)
            return sampler.sample(batch_size, device=device, dtype=torch.float32)

    def _fixed_uniform(shape, seed_offset: int = 0):
        gen = torch.Generator()
        gen.manual_seed(args.seed + seed_offset)
        base = torch.rand(shape, generator=gen, dtype=torch.float32)
        return base.to(device)

    if not hasattr(args, "_fixed_quantile_ot_x0") or args._fixed_quantile_ot_x0.shape[0] != batch_size:
        args._fixed_quantile_ot_x0 = _fixed_sampler_batch(batch_size, seed_offset=5)
    if not hasattr(args, "_fixed_quantile_ot_u01") or args._fixed_quantile_ot_u01.shape[0] != batch_size:
        args._fixed_quantile_ot_u01 = _fixed_uniform((batch_size, dim), seed_offset=7)

    x0_fixed = args._fixed_quantile_ot_x0
    u_unit_fixed = args._fixed_quantile_ot_u01

    image_dim = math.prod(image_shape) if image_shape is not None else None
    is_image_task = image_shape is not None and image_dim == dim

    if is_image_task:
        log_real_rgb_histogram_once(
            args=args,
            sampler=sampler,
            image_shape=image_shape,
            device=device,
            step=0,
        )

    sample_dir = os.path.join(args.runs_dir, "quantile_ot", "samples")
    fixed_u_vis: Optional[torch.Tensor] = None
    if is_image_task and args.sample_vis_interval > 0 and args.sample_vis_count > 0:
        if not hasattr(args, "_fixed_quantile_ot_vis_u") or args._fixed_quantile_ot_vis_u.shape[0] < args.sample_vis_count:
            args._fixed_quantile_ot_vis_u = _fixed_uniform((args.sample_vis_count, dim), seed_offset=19)
        fixed_u_vis = args._fixed_quantile_ot_vis_u

    step_offset = max(int(log_step_offset), 0)
    if wandb.run is not None:
        run_step = getattr(wandb.run, "step", None)
        if run_step is not None and int(run_step) >= step_offset:
            step_offset = int(run_step) + 1

    total_steps = int(args.q_ntrain)
    B= batch_size
    D = args.dim
    for step in tqdm(range(total_steps), desc="Quantile training (ot)"):
        quantile.train()
        opt.zero_grad(set_to_none=True)
        global_step = step_offset + step
        # --- Data ---
        with torch.no_grad():
            x0 = sampler.sample(B, device=device, dtype=torch.float32)

        rand_u = torch.rand(B, D, device=device, dtype=x0.dtype)
        U = u_eps + (1 - 2 * u_eps) * rand_u
        tau_one = torch.ones(B, 1, device=device)

        eps, dqdt = quantile(
            U,
            tau_one,
            return_dqdt=True,
            requires_grad=True,
        )
        transport_loss, transport_metrics = transport_objective(x0, dqdt)
        reg_logdet = (
            _compute_kl_regularizer(
                quantile,
                U,
                tau_one,
                lambda_reg,
                device,
            )
            if lambda_reg > 0.0
            else torch.zeros((), device=device)
        )

        loss = q_loss_weight * transport_loss + lambda_reg * reg_logdet
        loss.backward()
        grad_quant = torch.nn.utils.clip_grad_norm_(quantile.parameters(), args.model_grad_clip)
        opt.step()
        if use_ema and ema_quantile is not None:
            ema_quantile.update_parameters(quantile)

        metrics = {
            "quantile_ot/loss_total": float(loss.item()),
            "quantile_ot/loss_q": float(transport_loss.item()),
            "quantile_ot/loss_reg_logdet": float(reg_logdet.item()),
            "quantile_ot/grad": float(grad_quant.item()),
        }
        metrics.update(transport_metrics)
        wandb.log(metrics, step=global_step)

        quant_eval = _get_quantile_eval_model(quantile, ema_quantile if use_ema else None)
        fixed_u_vis = _maybe_log_image_visualizations(
            args=args,
            quant_eval=quant_eval,
            step=step,
            global_step=global_step,
            is_image_task=is_image_task,
            sample_vis_interval=args.sample_vis_interval,
            sample_vis_count=args.sample_vis_count,
            sample_vis_nrow=max(1, args.sample_vis_nrow),
            sample_dir=sample_dir,
            fixed_u_vis=fixed_u_vis,
            u_eps=u_eps,
            image_shape=image_shape,
            device=device,
            dim=dim,
        )

        _maybe_run_low_dim_evals(
            args=args,
            quant_eval=quant_eval,
            sampler=sampler,
            device=device,
            global_step=global_step,
            step=step,
            dim=dim,
            u_eps=u_eps,
            normal=normal,
        )

    q_dir = os.path.join(args.runs_dir, "quantile_ot")
    os.makedirs(q_dir, exist_ok=True)
    ckpt_path = os.path.join(q_dir, "quantile.pt")
    ckpt_payload = {
        "state_dict": quantile.state_dict(),
        "dim": dim,
        "eps": args.q_u_eps,
        "type": "rqs",
        "n_bins": int(args.q_rqs_bins),
        "bound": float(args.q_rqs_bound),
        "layers": int(args.q_rqs_layers),
    }
    torch.save(ckpt_payload, ckpt_path)

    if use_ema and ema_quantile is not None:
        ckpt_path_ema = os.path.join(q_dir, "quantile_ema.pt")
        ema_payload = {**ckpt_payload, "state_dict": ema_quantile.state_dict(), "ema": args.q_ema}
        torch.save(ema_payload, ckpt_path_ema)

    config = {
        "dim": dim,
        "type": "rqs",
        "mode": "ot",
        "ot_epsilon": float(args.q_ot_epsilon) if hasattr(args, "q_ot_epsilon") else 1.0,
        "objective": getattr(args, "q_objective", getattr(args, "q_ot_metric", getattr(args, "q_ot_objective", "energy"))),
        "rqs": {
            "n_bins": int(args.q_rqs_bins),
            "bound": float(args.q_rqs_bound),
            "layers": int(args.q_rqs_layers),
        },
    }
    with open(os.path.join(q_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    setattr(args, "_pretrain_step_offset", step_offset + total_steps)

    return _get_quantile_eval_model(quantile, ema_quantile if use_ema else None), ckpt_path, total_steps
