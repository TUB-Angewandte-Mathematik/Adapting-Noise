import torch
import wandb
from tqdm import tqdm
from torchdiffeq import odeint
from typing import Dict, Tuple

import learn_noise.utils.plotting as plot

_MAX_SEED = 2 ** 31 - 1


def _get_seed(base_seed: int, offset: int) -> int:
    seed = (base_seed + offset) % _MAX_SEED
    if seed <= 0:
        seed += 1
    return seed


def _get_target_cache(args) -> Dict[Tuple[str, int], torch.Tensor]:
    if not hasattr(args, "_eval_fixed_targets") or args._eval_fixed_targets is None:
        setattr(args, "_eval_fixed_targets", {})
    return args._eval_fixed_targets


def _get_uniform_cache(args) -> Dict[Tuple[int, int], torch.Tensor]:
    if not hasattr(args, "_eval_fixed_u") or args._eval_fixed_u is None:
        setattr(args, "_eval_fixed_u", {})
    return args._eval_fixed_u


def _to_raw_if_needed(sampler, tensor: torch.Tensor) -> torch.Tensor:
    return sampler.to_raw(tensor) if hasattr(sampler, "to_raw") else tensor


def _base_sampler(sampler):
    return sampler.base if hasattr(sampler, "base") else sampler


def _fixed_ground_truth(args, sampler, total: int, device: torch.device) -> torch.Tensor:
    cache = _get_target_cache(args)
    target_dataset = args.target_dataset if hasattr(args, "target_dataset") else "unknown"
    key = (target_dataset, total)
    if key not in cache:
        base_seed = int(args.seed) if hasattr(args, "seed") else 0
        seed = _get_seed(base_seed, 1009 + 131 * total)
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            samples = sampler.sample(total, device=device, dtype=torch.float32)
            samples = _to_raw_if_needed(sampler, samples).cpu()
        cache[key] = samples
    return cache[key]


def _fixed_uniform(args, total: int, dim: int, offset: int = 0) -> torch.Tensor:
    cache = _get_uniform_cache(args)
    key = (total, dim)
    if key not in cache:
        base_seed = int(args.seed) if hasattr(args, "seed") else 0
        seed = _get_seed(base_seed, 2027 + 137 * total + offset)
        gen = torch.Generator()
        gen.manual_seed(seed)
        cache[key] = torch.rand((total, dim), generator=gen, dtype=torch.float32)
    return cache[key]

@torch.no_grad()
def heavy_eval_batched(
    args,
    x_0, 
    ode_func, 
    sampler,
    step, 
    big_eval=False,
    device='cpu', 
    noise = None,
    quantile=None,
):
    """
    Massive eval to probe tails with VRAM-safe batching.
    - Generates eps at τ=1 in batches (Student-t base)
    - Integrates ODE to t=0, collects running NLL mean
    - Keeps a capped subset for plotting (both latent eps and generated x)
    - Logs GeomLoss Sinkhorn/MMD metrics on cached subsets for non-funnel targets
    """
    dim = args.dim
    output_dir = args.runs_dir

    device = torch.device(device)

    if big_eval:
        total    = int(args.big_eval_samples)
    else: 
        total    = int(args.eval_sample)

    if total <= 0:
        return

    batch_size    = int(args.eval_batch)

    keep  = total#int(args.eval_plot_samples)
    assert batch_size > 0, "big_eval_batch must be > 0"

    t_vals = torch.linspace(1, 0.0, args.num_steps_eval, device=device)

    nll_sum = 0.0
    seen = 0

    kept_x = []
    kept_eps = []
    kept_count = 0

    u_unit_cache = None
    if quantile is not None and total > 0:
        u_unit_cache = _fixed_uniform(args, total, dim)

    target_name = (args.target_dataset if hasattr(args, "target_dataset") else "funnel").lower()
    raw_sampler = _base_sampler(sampler)

    # progress loop
    num_loops = (total + batch_size - 1) // batch_size
    for loop_idx in range(num_loops):
        current_batch_size = min(batch_size, total - seen)
        if current_batch_size <= 0:
            break
        # Initial noise at τ=1: prefer quantile if provided for consistency
        if quantile is not None:
            u_eps = float(args.q_u_eps) if hasattr(args, "q_u_eps") else 5e-5
            u_slice = u_unit_cache[seen: seen + current_batch_size].to(device)
            Uv = u_eps + (1 - 2 * u_eps) * u_slice
            ones_t = torch.ones(current_batch_size, 1, device=device)
            with torch.no_grad():
                eps = quantile(Uv, ones_t)
        elif noise is not None:
            eps = noise((current_batch_size, dim)).to(device)
        else:
            eps = torch.randn(current_batch_size, dim, device=device)

        x_T = eps

        trajectory = odeint(ode_func, x_T, t_vals, method="dopri5")
        x_gen = trajectory[-1]   # (cur_bs, dim)

        # Accumulate NLL sum to compute global mean at the end
        nll_sum += (-sampler.log_prob(x_gen)).sum().item()
        seen += current_batch_size

        # Keep a proportionate random subset from this batch for plotting
        per_batch_keep = max(1, int(round(keep * (current_batch_size / total)))) if keep > 0 else 0
        if per_batch_keep > 0:
            #idx = torch.randperm(current_batch_size, device=device)[:per_batch_keep]
            kept_x.append(x_gen.detach().cpu())
            kept_eps.append(eps.detach().cpu())
            kept_count += per_batch_keep

    x_gen = torch.stack(kept_x, dim=0).reshape(-1, dim)
    x_gen_raw = _to_raw_if_needed(sampler, x_gen)
    eps_kept = torch.stack(kept_eps, dim=0).reshape(-1, dim) if kept_eps else None

    '''# Plot (downsample to exactly 'keep' if we slightly overshot)
    if keep > 0 and kept_x:
        X = torch.cat(kept_x, dim=0)
        E = torch.cat(kept_eps, dim=0)
        if X.shape[0] > keep:
            perm = torch.randperm(X.shape[0])[:keep]
            X = X[perm]
            E = E[perm]'''

    # Choose plotting pipeline based on target
    if target_name in {"funnel", "nealfunnel"}:
        plot.plot_funnel_2d(x_gen_raw, raw_sampler, step, big_eval, output_dir)
    else:
        plot.plot_generic_2d(x_gen, sampler, step, big_eval, output_dir)
    #print(funnel_eval.evaluate_x2_marginal_metrics(x_gen))
    
    # New: latent colored by norm of reached target x
    if eps_kept is not None:
        plot.plot_latent_colored_by_target_norm(eps_kept, x_gen_raw, step, output_dir, big_eval=big_eval)
