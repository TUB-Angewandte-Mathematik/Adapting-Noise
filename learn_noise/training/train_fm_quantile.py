import math
import os
from types import SimpleNamespace
from typing import Optional

import torch
import wandb
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm
from geomloss import SamplesLoss

from learn_noise.training.pretrain_quantile import build_quantile, pretrain_quantile
from learn_noise.training.common import (
    seed_all,
    ensure_args_defaults,
    minibatch_ot_pairing,
    prepare_paths,
)
from learn_noise.training.optim import get_optimizer_and_scheduler, get_quantile_scheduler
from learn_noise.training.quantile_losses import detached_mse, regularization_logdet
from learn_noise.networks.model_wrapper import TorchWrapper, ODEWrapper
import learn_noise.utils.sampler as smpl
from learn_noise.training.logging import (
    log_quantile_image_metrics,
    log_quantile_low_dim_metrics,
    log_real_rgb_histogram_once,
    prepare_fid,
    prepare_fixed_u,
    track_model_parameters,
    wandb_global_steps,
)
from learn_noise.training.sample_utils import generate_quantile_samples, default_t_eval


def get_quantile(args, device):
    if args.mode == 'pretrain_quantile':
        quantile, _, _ = pretrain_quantile(args)
    else:    
        quantile = build_quantile(args, device, args.dim)
    
    ckpt_path = args.quantile_checkpoint
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict):
            if 'quantile' in state:
                quantile.load_state_dict(state['quantile'])
            elif 'state_dict' in state:
                quantile.load_state_dict(state['state_dict'])
            else:
                quantile.load_state_dict(state)
        else:
            quantile.load_state_dict(state)
        print('[train_fm_quantile] Loaded quantile checkpoint from', ckpt_path)

    freeze_quantile = bool(args.freeze_quantile)
    if freeze_quantile:
        for param in quantile.parameters():
            param.requires_grad_(False)
    
    return quantile

def train_fm_quantile(
    args: SimpleNamespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Train the joint flow-matching + quantile model."""
    device = torch.device(args.device)
    seed_all(args.seed)

    global_step_offset = wandb_global_steps(args)
   
    quantile = get_quantile(args, device)

    track_model_parameters(args, model, quantile, global_step_offset)

    optimizer, scheduler = get_optimizer_and_scheduler(args, model, quantile)
    _quantile_lr_schedule, quantile_schedule_total = get_quantile_scheduler(args, optimizer)

    sampler = smpl.get_distribution(args.target_dataset)
    
    ema = None
    ema_started = False
    ema_start_step = int(args.ema_start_step)

    ema_avg_fn = get_ema_multi_avg_fn(args.ema)
    wrapper = TorchWrapper(model)
    ode_func = ODEWrapper(wrapper).to(device)

    if args.metric == 'mmd':
        metric = SamplesLoss('energy')
    elif args.metric == 'SD':
        metric = SamplesLoss(blur=args.metric_blur)
    else:
        metric = None

    def _active_eval_model():
        return ema if ema is not None else model

    q_loss_weight = float(getattr(args, "q_loss_weight", 1.0))
    u_eps = float(args.q_u_eps)

    raw_image_shape = getattr(args, "image_shape", None)
    if raw_image_shape is None:
        image_shape = None
    else:
        image_shape = tuple(raw_image_shape)
        if len(image_shape) == 0:
            image_shape = None
    image_dim = math.prod(image_shape) if image_shape is not None else None
    is_image_task = (image_shape is not None) and (image_dim == args.dim)
    
    # defaults + core scalars
    ensure_args_defaults(args, image_shape)
    lambda_reg_weight = float(args.lambda_reg)

    # one-time real histogram
    if is_image_task:
        log_real_rgb_histogram_once(args=args, sampler=sampler, image_shape=image_shape, device=device, step=0)

    # FID setup, dirs, fixed-u, eval grid
    fid = prepare_fid(args, sampler, device, image_shape, args.dim)
    paths = prepare_paths(args)
    fixed_u_vis, fixed_eval_u = prepare_fixed_u(args, is_image_task, image_shape, device)
    t_eval_default = default_t_eval(args, device)

    global_step = global_step_offset

    # ----- Training -----
    for step in tqdm(range(args.epochs), desc="Flow-matching (test trajectories)"):
        model.train()
        if not args.freeze_quantile and step < quantile_schedule_total:
            quantile.train()
        else:
            quantile.eval()
        optimizer.zero_grad(set_to_none=True)
        global_step = global_step_offset + step

        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]["lr"] = _quantile_lr_schedule(global_step)

        if (not ema_started) and global_step >= ema_start_step:
            ema = AveragedModel(model, multi_avg_fn=ema_avg_fn)
            ema.to(device)
            ema.eval()
            wrapper.model = ema
            ema_started = True

        # ---- Sample data ----
        x_0 = sampler.sample(args.batch_size, device=device, dtype=torch.float32)
        if x_0.dim() > 2:
            x_0 = x_0.reshape(x_0.shape[0], -1)

        # U ~ Uniform[u_eps, 1-u_eps]
        u = u_eps + (1 - 2 * u_eps) * torch.rand(x_0.shape[0], args.dim, device=device, dtype=x_0.dtype)

        # ---- Minibatch OT pairing ----
        pairing_cost = None
        if args.use_minibatch_ot:
            with torch.no_grad():
                t_1 = torch.ones(x_0.shape[0], 1, device=device, dtype=x_0.dtype)
                Q_no_grad = quantile(u, t_1).reshape(x_0.shape)
            
            transport_indices, transport_plan = minibatch_ot_pairing(x_0, Q_no_grad)
            pairing_cost = transport_plan.max(dim=0).values.mean()
            idx_best = transport_indices.to(torch.long)
            x_0 = x_0[idx_best]

        t = torch.rand(x_0.shape[0], 1, device=device)

        # ---- Forward ----
        need_grad = step < quantile_schedule_total
        eps, dqdt = quantile(u, t, return_dqdt=True, requires_grad=need_grad)
        
        x_t = (1.0 - t) * x_0 + eps
        target_velocity = -x_0 + dqdt

        velocity_net = model(t, x_t)

        w2_loss = 0.5 * target_velocity.pow(2).sum(dim=1).mean()

        if args.metric == 'ot':
            loss_q = w2_loss
        elif args.metric in {'mmd', 'SD'}:
            if metric is None:
                raise RuntimeError("Metric operator not initialised for metric='" + str(args.metric) + "'")
            loss_q = metric(dqdt, x_0)
        else:
            raise NotImplementedError

        if lambda_reg_weight > 0 and step < quantile_schedule_total and not args.freeze_quantile:
            reg_logdet = regularization_logdet(quantile, u, t)
        else: 
            reg_logdet = 0
       
        loss_velocity = detached_mse(target_velocity, velocity_net)
        loss = loss_velocity + q_loss_weight * loss_q + lambda_reg_weight * reg_logdet

        loss.backward()
        
        grad_model = torch.nn.utils.clip_grad_norm_(model.parameters(), args.model_grad_clip)
        grad_quantile = torch.nn.utils.clip_grad_norm_(quantile.parameters(), args.model_grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        if ema is not None:
            ema.update_parameters(model)

        log_payload = {
            'loss/velocity': float(loss_velocity.item()),
            'loss/q': float(loss_q.item()),
            'loss/w2': float(w2_loss.item()),
            'loss/reg_logdet': float(reg_logdet.item()) if torch.is_tensor(reg_logdet) else float(reg_logdet),
            'loss/total': float(loss.item()),
            'grad/model_velocity': float(grad_model.item()),
            'grad/quantile': float(grad_quantile.item()),
        }
        if pairing_cost is not None:
            log_payload['metrics/minibatch_ot_cost'] = float(pairing_cost.item())
        wandb.log(log_payload, step=global_step)

        do_light = (args.eval_sample > 0) and (((step + 1) % args.eval_step) == 0)
        do_heavy = (args.big_eval_samples > 0) and (((step + 1) % args.big_eval_step) == 0)

        # ---- Plotting ----
        if is_image_task:
            run_samples = (
                args.sample_vis_interval > 0
                and args.sample_vis_count > 0
                and ((step + 1) % args.sample_vis_interval == 0)
            )
            run_latent = False
            run_fid = (
                fid.interval > 0
                and fid.num_gen > 0
                and fid.real_cache is not None
                and ((step + 1) % fid.interval == 0)
            )
            if run_samples or run_latent or run_fid:
                batch_size_for_logging = (
                    fid.gen_batch
                    if fid.gen_batch > 0
                    else max(1, args.sample_vis_count if args.sample_vis_count > 0 else args.batch_size)
                )
                cap_attr = int(getattr(args, "eval_generate_batch_cap", 0))
                if cap_attr <= 0:
                    cap_attr = int(getattr(args, "eval_batch_cap", 0))
                if cap_attr <= 0 and hasattr(args, "batch_size"):
                    cap_attr = int(getattr(args, "batch_size"))
                if cap_attr > 0:
                    max_generate_batch = max(1, cap_attr)
                else:
                    max_generate_batch = None

                eval_model_for_logging = _active_eval_model()
                def generate_for_logging(
                    count: int,
                    *,
                    u_source: Optional[torch.Tensor] = None,
                    max_batch: Optional[int] = None,
                ) -> torch.Tensor:
                    local_batch = batch_size_for_logging
                    if max_generate_batch is not None:
                        local_batch = min(local_batch, max_generate_batch)
                    if max_batch is not None:
                        local_batch = min(local_batch, max_batch)
                    return generate_quantile_samples(
                        count,
                        batch_size=local_batch,
                        device=device,
                        dim=args.dim,
                        u_eps=u_eps,
                        quantile=quantile,
                        ode_func=ode_func,
                        t_eval=t_eval_default,
                        wrapper=wrapper,
                        eval_model=eval_model_for_logging,
                        u_source=u_source,
                        image_shape=image_shape,
                    )
                fixed_u_vis = log_quantile_image_metrics(
                    args=args,
                    step=global_step,
                    eval_model=eval_model_for_logging,
                    wrapper=wrapper,
                    quantile=quantile,
                    device=device,
                    image_shape=image_shape,
                    sample_vis_interval=args.sample_vis_interval,
                    sample_vis_count=args.sample_vis_count,
                    sample_vis_nrow=max(1, args.sample_vis_nrow),
                    sample_dir=paths.sample_dir,          
                    fid_interval=fid.interval,             
                    fid_num_gen=fid.num_gen,              
                    fid_batch_size=fid.batch_size,        
                    fid_image_size=fid.image_size,        
                    fid_gen_batch=fid.gen_batch,          
                    fid_real_cache=fid.real_cache,
                    generate_samples=generate_for_logging,        
                    fixed_u_vis=fixed_u_vis,
                    u_eps=u_eps,
                )
                if fixed_u_vis is not None:
                    args._fixed_double_fm_vis_u = fixed_u_vis
        if (step + 1) % 20_000 == 0:
            ckpt_suffix = f"step_{global_step:06d}.pt"
            quantile_payload = {
                "step": global_step,
                "state_dict": quantile.state_dict(),
                "u_eps": u_eps,
                "dim": args.dim,
            }
            torch.save(quantile_payload, os.path.join(paths.checkpoint_dir, f"quantile_{ckpt_suffix}"))
            if ema is not None:
                ema_payload = {
                    "step": global_step,
                    "state_dict": ema.state_dict(),
                }
                torch.save(ema_payload, os.path.join(paths.checkpoint_dir, f"ema_{ckpt_suffix}"))

        # Low-dimensional trajectory / Sinkhorn metrics retain their own cadence.
        if not is_image_task and (do_light or do_heavy):
            eval_model = _active_eval_model()
            fixed_eval_u = log_quantile_low_dim_metrics(
                args=args,
                step=global_step,
                eval_model=eval_model,
                wrapper=wrapper,
                ode_func=ode_func,
                sampler=sampler,
                quantile=quantile,
                x0_batch=x_0,
                device=device,
                do_light=do_light,
                do_heavy=do_heavy,
                u_eps=u_eps,
                fixed_eval_u=fixed_eval_u,
            )
            if fixed_eval_u is not None:
                args._fixed_quantile_eval_u = fixed_eval_u

    ckpt_suffix = f"step_{global_step:06d}.pt"
    quantile_payload = {
        "step": global_step,
        "state_dict": quantile.state_dict(),
        "u_eps": u_eps,
        "dim": args.dim,
    }
    torch.save(quantile_payload, os.path.join(paths.checkpoint_dir, f"quantile_{ckpt_suffix}"))
    if ema is not None:
        ema_payload = {
            "step": global_step,
            "state_dict": ema.state_dict(),
        }
        torch.save(ema_payload, os.path.join(paths.checkpoint_dir, f"ema_{ckpt_suffix}"))

    runtime_path = os.path.join(args.runs_dir, "runtime_training_only.txt")
    os.makedirs(args.runs_dir, exist_ok=True)
    

    return quantile, ema
