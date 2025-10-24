
from __future__ import annotations

import ast
import math, re
import os
import time
from typing import Callable, Optional
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import wandb
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm

import learn_noise.utils.sampler as smpl
from learn_noise.networks.model_wrapper import TorchWrapper, ODEWrapper
from learn_noise.training.common import (
    seed_all,
    make_fixed_sampler,
    minibatch_ot_pairing,
    count_parameters,
    write_model_size_summary,
)
from learn_noise.training.logging import (
    log_baseline_evaluation,
    log_baseline_image_metrics,
    log_real_rgb_histogram_once,
)
from learn_noise.utils.image_eval import reshape_flat_samples
from learn_noise.utils.velocity_kac import compute_velocity_kac
from learn_noise.utils.velocity_mmd import compute_velocity_mmd
from learn_noise.training.sample_utils import generate_baseline_samples

def _make_latent_sampler(name: str, *, device: torch.device, args: Optional[object] = None) -> Callable:
    lname = name.lower()

    if lname in {"gauss", "gaussian", "normal"}:
        def _sample(shape): return torch.randn(*shape, device=device)
        return _sample

    if lname in {"uniform", "uni"}:
        def _sample(shape): return torch.rand(*shape, device=device) * 4.0 - 2.0
        return _sample

    if lname in {"student_t", "student-t", "studentt"}:
        default_dtype = torch.get_default_dtype()

        def _coerce_param(value, fallback):
            if value is None:
                return fallback
            if isinstance(value, str):
                text = value.strip()
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed = float(text)
                return parsed
            return value

        df_param = _coerce_param(getattr(args, "student_t_df", None) if args is not None else None, 4.0)
        scale_param = _coerce_param(getattr(args, "student_t_scale", None) if args is not None else None, 1.0)

        df_tensor = torch.as_tensor(df_param, dtype=default_dtype, device=device)
        scale_tensor = torch.as_tensor(scale_param, dtype=default_dtype, device=device)
        loc_tensor = torch.zeros_like(df_tensor, dtype=default_dtype, device=device)

        dist = torch.distributions.StudentT(df=df_tensor, loc=loc_tensor, scale=scale_tensor)
        batch_shape = dist.batch_shape  # usually () or (dim,)

        def _sample(shape):
            if not shape:
                sample_shape = torch.Size()
            elif len(batch_shape) == 0:
                sample_shape = torch.Size(shape)
            else:
                if len(shape) < len(batch_shape):
                    raise ValueError(
                        "Requested Student-t sample shape is too small for batch parameters: "
                        f"shape={shape}, batch_shape={tuple(batch_shape)}"
                    )
                expected = tuple(batch_shape)
                actual = tuple(shape[-len(batch_shape):])
                if actual != expected:
                    raise ValueError(
                        "Student-t latent requires the trailing dimensions to match the parameter shape: "
                        f"expected {expected}, got {shape}"
                    )
                sample_shape = torch.Size(shape[:-len(batch_shape)])

            samples = dist.sample(sample_shape)
            if isinstance(samples, torch.Tensor):
                return samples
            return torch.as_tensor(samples, dtype=default_dtype, device=device)

        return _sample

  
    raise ValueError(f"Unknown baseline latent '{name}'")



def train_fm_baseline(
    args: SimpleNamespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Train the baseline flow-matching model."""
    device = torch.device(args.device)
    seed_all(args.seed)

    sampler = smpl.get_distribution(args.target_dataset)

    warmup_steps = max(0, int(getattr(args, "warmup_lr", 0)))

    def _warmup_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / warmup_steps)

    
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_lambda)

    flow_type = getattr(args, "baseline_flow", "linear").lower()
    flow_T = float(getattr(args, "baseline_flow_T", 1.0))
    if flow_T <= 0.0:
        raise ValueError("baseline_flow_T must be positive.")

    use_minibatch_ot = bool(getattr(args, "use_minibatch_ot", False))

    latent_sampler_train: Optional[Callable[[tuple[int, ...]], torch.Tensor]] = None
    latent_sampler_eval: Callable[[tuple[int, ...]], torch.Tensor]
    mmd_params: Optional[dict[str, object]] = None
    kac_params: Optional[dict[str, object]] = None

    if flow_type == "linear":
        latent_sampler_train = _make_latent_sampler(args.baseline_latent, device=device, args=args)

        def _latent_eval(shape: tuple[int, ...]) -> torch.Tensor:
            assert latent_sampler_train is not None
            return latent_sampler_train(shape)

        latent_sampler_eval = _latent_eval

    elif flow_type == "mmd":
        mmd_b = float(getattr(args, "baseline_mmd_b", 1.0))
        if mmd_b <= 0.0:
            raise ValueError("baseline_mmd_b must be positive.")
        mmd_sampler = smpl.TorchQuantileSampler(b=mmd_b, device=device, dtype=torch.float32)

        def _latent_eval(shape: tuple[int, ...]) -> torch.Tensor:
            batch = int(shape[0]) if len(shape) > 0 else 1
            t_final = torch.full((batch,), flow_T, device=device)
            latents, _ = mmd_sampler.sample(t_final, dim=args.dim)
            return latents.view(batch, -1)

        latent_sampler_eval = _latent_eval
        mmd_params = {"b": mmd_b, "sampler": mmd_sampler}

    elif flow_type == "kac":
        if use_minibatch_ot:
            print("[baseline] Disabling minibatch OT pairing for baseline_flow='kac'.")
            use_minibatch_ot = False
        kac_a = float(getattr(args, "baseline_kac_a", 9.0))
        kac_c = float(getattr(args, "baseline_kac_c", 3.0))
        kac_eps = float(getattr(args, "baseline_kac_epsilon", 1e-6))
        kac_M = int(getattr(args, "baseline_kac_lookup_M", 5000))
        kac_K = int(getattr(args, "baseline_kac_lookup_K", 1024))
        kac_sampler = smpl.TorchKacConstantSampler(
            a=kac_a,
            c=kac_c,
            T=flow_T,
            M=kac_M,
            K=kac_K,
            device=device,
            dtype=torch.float32,
        )

        def _latent_eval(shape: tuple[int, ...]) -> torch.Tensor:
            batch = int(shape[0]) if len(shape) > 0 else 1
            t_final = torch.full((batch,), flow_T, device=device)
            latents = kac_sampler.sample(t_final, dim=args.dim)
            return latents.view(batch, -1)

        latent_sampler_eval = _latent_eval
        kac_params = {
            "a": kac_a,
            "c": kac_c,
            "epsilon": kac_eps,
            "sampler": kac_sampler,
        }

    else:
        raise ValueError(f"Unsupported baseline_flow '{flow_type}'")

    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema))
    wrapper = TorchWrapper(ema)
    ode_func = ODEWrapper(wrapper).to(device)

    unet_params = count_parameters(model)
    model_size_stats = {
        "method": "baseline_fm",
        "target_dataset": getattr(args, "target_dataset", None),
        "params_unet": unet_params,
        "params_quantile": 0,
        "params_total": unet_params,
    }
    channel_mult = getattr(args, "unet_channel_mult", None)
    if channel_mult is not None:
        model_size_stats["unet_channel_mult"] = tuple(channel_mult)
    for attr in ("unet_model_channels", "unet_num_res_blocks", "unet_attention_resolutions"):
        value = getattr(args, attr, None)
        if value is not None:
            model_size_stats[attr] = value
    write_model_size_summary(args.runs_dir, model_size_stats)
    wandb.log({
        "params/unet": float(unet_params),
        "params/quantile": 0.0,
        "params/total": float(unet_params),
    }, step=0)

    fixed_sampler = make_fixed_sampler(sampler, seed=args.seed, device=device)
    if not hasattr(args, "_fixed_baseline_x0"):
        args._fixed_baseline_x0 = fixed_sampler(args.batch_size, seed_offset=0)
    x0_fixed = args._fixed_baseline_x0

    image_shape = getattr(args, "image_shape", None)
    image_dim = math.prod(image_shape) if image_shape is not None else None
    is_image_task = image_shape is not None and image_dim == args.dim


    if is_image_task:
        log_real_rgb_histogram_once(
            args=args,
            sampler=sampler,
            image_shape=image_shape,
            device=device,
            step=0,
        )

    checkpoint_dir = os.path.join(args.runs_dir, "baseline_fm")
    os.makedirs(checkpoint_dir, exist_ok=True)

    fid_interval = int(args.fid_eval_interval) if hasattr(args, "fid_eval_interval") else 0
    fid_num_gen = int(args.fid_num_gen) if hasattr(args, "fid_num_gen") else 0
    fid_batch_size = max(1, int(getattr(args, "fid_batch_size", args.batch_size))) if fid_interval > 0 else 0
    fid_gen_batch = max(1, int(getattr(args, "fid_gen_batch", args.batch_size))) if fid_interval > 0 else 0
    fid_image_size = (
        int(getattr(args, "fid_image_size", 0)) if (fid_interval > 0 and image_shape is not None) else 0
    )
    fid_real_cache = None
    if is_image_task and fid_interval > 0 and fid_num_gen > 0:
        with torch.no_grad():
            real_samples = sampler.sample(fid_num_gen, device=device, dtype=torch.float32)
            fid_real_cache = reshape_flat_samples(real_samples, torch.Size(image_shape)).detach().cpu()

    sample_vis_interval = int(getattr(args, "sample_vis_interval", 0))
    sample_vis_count = int(getattr(args, "sample_vis_count", 0))
    sample_vis_nrow = int(getattr(args, "sample_vis_nrow", 8))

    sample_dir = os.path.join(checkpoint_dir, "samples") if is_image_task else ""
    t_eval = torch.linspace(1.0, 0.0, args.num_steps_eval, device=device)

    fixed_vis_noise = getattr(args, "_fixed_baseline_vis_noise", None) if is_image_task else None

    train_time_accumulator = 0.0

    for step in tqdm(range(args.epochs), desc="Flow-matching baseline"):
        iter_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        pairing_cost = None
        x_0 = sampler.sample(args.batch_size, device=device, dtype=torch.float32)
        if x_0.dim() > 2:
            x_0 = x_0.view(x_0.shape[0], -1)
        t = torch.rand(args.batch_size, 1, device=device)
        t_actual = t.squeeze(1) * flow_T

        if flow_type == "linear":
            assert latent_sampler_train is not None
            z = latent_sampler_train((args.batch_size, args.dim))
            if use_minibatch_ot:
                idx_best, transport_plan = minibatch_ot_pairing(x_0, z)
                x_0 = x_0[idx_best]
                z = z[idx_best]
                pairing_cost = transport_plan.max(dim=0).values.mean()
            x_t = (1.0 - t) * x_0 + t * z
            velocity_target = -x_0 + z

        elif flow_type == "mmd":
            assert mmd_params is not None
            tau_raw, U = mmd_params["sampler"].sample(t_actual, dim=args.dim)
            tau = tau_raw.view(args.batch_size, -1).to(device=device)
            U = U.view(args.batch_size, -1).to(device=device)
            disp = 2.0 * U - 1.0
            f = 1.0 - t
            x_t = f * x_0 + tau
            velocity_noise = compute_velocity_mmd(
                x=disp,
                t=t_actual.unsqueeze(1),
                b=mmd_params["b"],
                disp=disp,
            )
            velocity_noise = velocity_noise * flow_T
            velocity_target = -x_0 + velocity_noise

        elif flow_type == "kac":
            assert kac_params is not None
            tau_raw = kac_params["sampler"].sample(t_actual, dim=args.dim)
            tau = tau_raw.view(args.batch_size, -1).to(device=device)
            f = 1.0 - t
            x_t = f * x_0 + tau
            velocity_noise = compute_velocity_kac(
                x_t - f * x_0,
                t_actual.unsqueeze(1),
                a=kac_params["a"],
                c=kac_params["c"],
                epsilon=kac_params["epsilon"],
                T=flow_T,
            )
            velocity_target = -x_0 + velocity_noise

        else:
            raise RuntimeError(f"Unexpected baseline_flow '{flow_type}' at training time")
        velocity_pred = model(t, x_t)

        loss = F.mse_loss(velocity_pred, velocity_target)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.model_grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update_parameters(model)

        train_time_accumulator += time.perf_counter() - iter_start

        log_payload = {
            'loss/velocity': float(loss.item()),
            'grad/model_velocity': float(grad_norm.item()),
        }
        if pairing_cost is not None:
            log_payload['metrics/minibatch_ot_cost'] = float(pairing_cost.item())
        wandb.log(log_payload, step=step)

        do_light = (args.eval_sample > 0) and (((step + 1) % args.eval_step) == 0)
        do_heavy = (args.big_eval_samples > 0) and (((step + 1) % args.big_eval_step) == 0)
        
        ###################### EVAL 2D ######################
        if not is_image_task and (do_light or do_heavy):
            log_baseline_evaluation(
                args=args,
                step=step,
                ema_model=ema,
                wrapper=wrapper,
                ode_func=ode_func,
                sampler=sampler,
                noise_sampler=latent_sampler_eval,
                x0_batch=x_0,
                device=device,
                do_light=do_light,
                do_heavy=do_heavy,

            )

        ###################### EVAL IMAGES ######################
        if is_image_task:
            run_samples = (
                sample_vis_interval > 0
                and sample_vis_count > 0
                and ((step + 1) % sample_vis_interval == 0)
            )
            run_fid = (
                fid_interval > 0
                and fid_num_gen > 0
                and fid_real_cache is not None
                and ((step + 1) % fid_interval == 0)
            )
            if run_samples or run_fid:
                if fid_gen_batch > 0:
                    batch_size_for_logging = fid_gen_batch
                else:
                    fallback_bs = sample_vis_count if sample_vis_count > 0 else args.batch_size
                    batch_size_for_logging = max(1, fallback_bs)

                def generate_for_logging(
                    count: int,
                    *,
                    latents: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
                    return generate_baseline_samples(
                        count,
                        batch_size=batch_size_for_logging,
                        device=device,
                        dim=args.dim,
                        t_eval=t_eval,
                        ode_func=ode_func,
                        wrapper=wrapper,
                        eval_model=ema,
                        latent_sampler=latent_sampler_eval,
                        latents=latents,
                    )

                fixed_vis_noise = log_baseline_image_metrics(
                    args=args,
                    step=step,
                    eval_model=ema,
                    wrapper=wrapper,
                    device=device,
                    image_shape=image_shape,
                    sampler=sampler,
                    sample_vis_interval=sample_vis_interval,
                    sample_vis_count=sample_vis_count,
                    sample_vis_nrow=max(1, sample_vis_nrow),
                    sample_dir=sample_dir,
                    fid_interval=fid_interval,
                    fid_num_gen=fid_num_gen,
                    fid_batch_size=fid_batch_size,
                    fid_image_size=fid_image_size,
                    fid_gen_batch=fid_gen_batch,
                    fid_real_cache=fid_real_cache,
                    noise_sampler=latent_sampler_eval,
                    generate_samples=generate_for_logging,
                    fixed_noise=fixed_vis_noise,
                )
                if fixed_vis_noise is not None:
                    args._fixed_baseline_vis_noise = fixed_vis_noise
        
        current_step = step + 1
        if current_step % 20_000 == 0:
            ckpt_suffix = f"step_{current_step:06d}.pt"

            model_payload = {
                "step": current_step,
                "state_dict": model.state_dict(),
            }
            torch.save(model_payload, os.path.join(checkpoint_dir, f"model_{ckpt_suffix}"))

            if ema is not None:
                ema_payload = {
                    "step": current_step,
                    "state_dict": ema.state_dict(),
                }
                torch.save(ema_payload, os.path.join(checkpoint_dir, f"ema_{ckpt_suffix}"))



    final_step = args.epochs
    ckpt_suffix = f"step_{final_step:06d}.pt"
    model_payload = {
        "step": final_step,
        "state_dict": model.state_dict(),
    }
    torch.save(model_payload, os.path.join(checkpoint_dir, f"model_{ckpt_suffix}"))

    if ema is not None:
        ema_payload = {
            "step": final_step,
            "state_dict": ema.state_dict(),
        }
        torch.save(ema_payload, os.path.join(checkpoint_dir, f"ema_{ckpt_suffix}"))

    runtime_path = os.path.join(args.runs_dir, "runtime_training_only.txt")
    os.makedirs(args.runs_dir, exist_ok=True)
    with open(runtime_path, "w", encoding="utf-8") as fh:
        fh.write(f"{train_time_accumulator:.6f}\n")
