#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
import yaml
import wandb
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from learn_noise.cli_images import _build_unet, _make_namespace
from learn_noise.training.pretrain_quantile import build_quantile
from learn_noise.training.train_fm_baseline import (
    _generate_baseline_samples,
    _make_latent_sampler,
)
from learn_noise.training.train_fm_quantile import _generate_samples as _generate_quantile_samples
from learn_noise.networks.model_wrapper import TorchWrapper, ODEWrapper
from learn_noise.utils.image_eval import compute_fid, reshape_flat_samples


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _latest_checkpoint(
    run_dir: Path,
    subdir: str,
    prefixes: Iterable[str],
    step: Optional[int],
) -> Path:
    folder = run_dir / subdir
    for prefix in prefixes:
        pattern = f"{prefix}_*.pt"
        files = sorted(folder.glob(pattern))
        if not files:
            continue
        if step is None:
            return files[-1]
        target = folder / f"{prefix}_{step:06d}.pt"
        if target.exists():
            return target
        # Fallback: pick checkpoint whose step is closest to requested value
        steps = []
        for f in files:
            try:
                suffix = f.stem.split("_")[-1]
                steps.append((abs(int(suffix) - step), int(suffix), f))
            except ValueError:
                continue
        if steps:
            steps.sort()
            return steps[0][2]
    raise FileNotFoundError(
        f"No checkpoint found in {folder} for prefixes {list(prefixes)}"
    )


def _load_real_images(dataset: str, data_root: str, samples: int, batch: int, device: torch.device) -> torch.Tensor:
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset = dataset.lower()
    if dataset not in {"cifar10", "cifar", "cifar-10"}:
        raise ValueError(f"Unsupported dataset '{dataset}' for FID evaluation")
    ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4)
    collected = []
    total = 0
    for imgs, _ in loader:
        collected.append(imgs.to(device))
        total += imgs.shape[0]
        if total >= samples:
            break
    real = torch.cat(collected, dim=0)[:samples]
    return real


def _prepare_velocity(args, device: torch.device, checkpoint: Path) -> torch.nn.Module:
    model = _build_unet(args)
    state = torch.load(checkpoint, map_location=device)
    if "state_dict" in state:
        state_dict = state["state_dict"]
    elif "model" in state:
        state_dict = state["model"]
    else:
        state_dict = state
    cleaned = {}
    for k, v in state_dict.items():
        if k == "n_averaged":
            continue
        if k.startswith("module."):
            cleaned[k[len("module.") :]] = v
        else:
            cleaned[k] = v
    state_dict = cleaned
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def _prepare_quantile(args, device: torch.device, dim: int, checkpoint: Path) -> torch.nn.Module:
    quantile = build_quantile(args, device=device, dim=dim)
    state = torch.load(checkpoint, map_location=device)
    state_dict = state.get("state_dict", state)
    cleaned = {}
    for k, v in state_dict.items():
        if k == "n_averaged":
            continue
        if k.startswith("module."):
            cleaned[k[len("module.") :]] = v
        else:
            cleaned[k] = v
    state_dict = cleaned
    quantile.load_state_dict(state_dict, strict=False)
    if "u_eps" in state:
        args.q_u_eps = float(state["u_eps"])
    quantile.to(device)
    quantile.eval()
    return quantile


def generate_samples(
    *,
    args,
    model,
    quantile: Optional[torch.nn.Module],
    device: torch.device,
    num_samples: int,
    batch_size: int,
    num_steps_eval: int,
) -> torch.Tensor:
    dim = args.dim
    t_eval = torch.linspace(1.0, 0.0, num_steps_eval, device=device)
    wrapper = TorchWrapper(model)
    ode_func = ODEWrapper(wrapper).to(device)

    outputs = []
    generated = 0
    with torch.no_grad():
        if quantile is not None:
            u_eps = float(args.q_u_eps)
            while generated < num_samples:
                cur = min(batch_size, num_samples - generated)
                samples = _generate_quantile_samples(
                    cur,
                    batch_size=batch_size,
                    device=device,
                    dim=dim,
                    u_eps=u_eps,
                    quantile=quantile,
                    ode_func=ode_func,
                    t_eval=t_eval,
                    wrapper=wrapper,
                    eval_model=model,
                )
                outputs.append(samples)
                generated += cur
        else:
            latent_sampler = _make_latent_sampler(args.baseline_latent, device=device)
            while generated < num_samples:
                cur = min(batch_size, num_samples - generated)
                samples = _generate_baseline_samples(
                    cur,
                    batch_size=batch_size,
                    device=device,
                    dim=dim,
                    t_eval=t_eval,
                    ode_func=ode_func,
                    wrapper=wrapper,
                    eval_model=model,
                    latent_sampler=latent_sampler,
                )
                outputs.append(samples)
                generated += cur
    return torch.cat(outputs, dim=0)


def main() -> None:  # pragma: no cover - CLI only
    parser = argparse.ArgumentParser(description="Compute FID for CIFAR runs")
    parser.add_argument("run_dir", type=Path, help="Run directory containing config.yaml")
    parser.add_argument("--step", type=int, help="Checkpoint step (default: latest)")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=Path, help="Optional path to save generated samples")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument(
        "--num-steps",
        type=int,
        nargs="+",
        help="Euler step counts for sampling (default: 100 and 20)",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging (disabled by default)")
    args = parser.parse_args()

    run_dir = args.run_dir
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = _load_yaml(config_path)
    cfg["runs_dir"] = str(run_dir)
    ns = _make_namespace(cfg)
    ns.device = args.device

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    quantile_dir = run_dir / "quantile_fm"
    has_quantile = quantile_dir.exists() and any(quantile_dir.glob("quantile_*.pt"))

    if has_quantile:
        vel_ckpt = _latest_checkpoint(run_dir, "quantile_fm", ("ema", "model"), args.step)
        quant_ckpt = _latest_checkpoint(run_dir, "quantile_fm", ("quantile",), args.step)
    else:
        vel_ckpt = _latest_checkpoint(run_dir, "baseline_fm", ("ema", "model"), args.step)
        quant_ckpt = None

    model = _prepare_velocity(ns, device, vel_ckpt)
    quantile = _prepare_quantile(ns, device, ns.dim, quant_ckpt) if quant_ckpt is not None else None

    num_steps_list: Sequence[int]
    if args.num_steps is None or len(args.num_steps) == 0:
        num_steps_list = (100, 10)
    else:
        num_steps_list = tuple(int(x) for x in args.num_steps)

    real_imgs = _load_real_images(ns.target_dataset, args.data_root, args.samples, args.batch, device)
    image_shape = torch.Size(ns.image_shape) if ns.image_shape is not None else None
    if image_shape is None:
        raise ValueError("FID is only defined for image runs")

    run_name = run_dir.name

    # Initialise W&B (disabled by default)
    if args.wandb and cfg.get("wandb_project"):
        wandb_kwargs = {
            "project": cfg.get("wandb_project"),
            "entity": cfg.get("wandb_entity"),
            "group": cfg.get("wandb_group"),
            "name": f"fid-{run_name}",
            "config": {
                "samples": args.samples,
                "batch": args.batch,
                "step_override": args.step,
                "num_steps_eval": list(num_steps_list),
            },
        }
        wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
        wandb.init(**wandb_kwargs)
    else:
        wandb.init(mode="disabled")

    for steps in num_steps_list:
        samples = generate_samples(
            args=ns,
            model=model,
            quantile=quantile,
            device=device,
            num_samples=args.samples,
            batch_size=args.batch,
            num_steps_eval=steps,
        )
        gen_imgs = reshape_flat_samples(samples.to(device), image_shape)

        fid = compute_fid(
            real=real_imgs,
            gen=gen_imgs,
            device=device,
            image_size=int(getattr(ns, "fid_image_size", 32) or 32),
            batch_size=args.batch,
        )
        print(f"FID ({run_name}, steps={steps}): {fid:.4f}")
        wandb.log({
            "fid/value": fid,
            "fid/num_steps": steps,
        })
        grid_count = min(64, gen_imgs.shape[0])
        samples_clamped = gen_imgs[:grid_count].clamp(-1.0, 1.0)
        samples_uint8 = ((samples_clamped + 1.0) / 2.0).mul(255.0).byte().cpu()
        images = [wandb.Image(img.permute(1, 2, 0).numpy()) for img in samples_uint8]
        wandb.log({f"samples/steps_{steps}": images})

        grid = make_grid(samples_clamped, nrow=8, normalize=True, value_range=(-1.0, 1.0))
        grid_np = grid.permute(1, 2, 0).detach().cpu().numpy()
        wandb.log({f"samples/grid_steps_{steps}": wandb.Image(grid_np)})

        if args.output is not None:
            out_path = args.output
            if len(num_steps_list) > 1:
                out_path = out_path.with_name(f"{out_path.stem}_steps{steps}{out_path.suffix}")
            torch.save(gen_imgs.cpu(), out_path)

    wandb.finish()


if __name__ == "__main__":
    main()
