
import json
import os
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

try:
    import ot  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ot = None

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    linear_sum_assignment = None


def seed_all(seed: int) -> None:
    """Seed python, numpy, and torch (including CUDA when available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_fixed_sampler(sampler, *, seed: int, device: torch.device) -> Callable[[int, int], torch.Tensor]:
    """Return a sampler that always yields the same batch for a given seed offset."""
    def _sample(batch_size: int, seed_offset: int = 0) -> torch.Tensor:
        devices = [device] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + seed_offset)
            if device.type == 'cuda' and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + seed_offset)
            return sampler.sample(batch_size, device=device, dtype=torch.float32)
    return _sample


def make_fixed_uniform(shape: Tuple[int, ...], *, seed: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a fixed tensor of uniform samples in [0, 1) with deterministic seeding."""
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    base = torch.rand(shape, generator=gen, dtype=dtype)
    return base.to(device)


def count_parameters(module: torch.nn.Module) -> int:
    """Return the total number of parameters in the module."""
    return sum(int(p.numel()) for p in module.parameters())


def write_model_size_summary(runs_dir: str, stats: Dict[str, object]) -> Path:
    """Persist model size statistics to `<runs_dir>/model_size.json`."""
    path = Path(runs_dir) / "model_size.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, value in stats.items():
        if isinstance(value, (list, tuple)):
            serializable[key] = list(value)
        else:
            serializable[key] = value
    with path.open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2, sort_keys=True)
    return path


def ensure_args_defaults(args, image_shape):
    """Populate optional training flags with defaults if missing."""
    defaults = {
        "latent_viz_samples": 0,
        "latent_atlas_grid": 1,
        "sample_vis_interval": 0,
        "sample_vis_count": 0,
        "sample_vis_nrow": 8,
        "fid_eval_interval": 0,
        "fid_num_gen": 0,
        "fid_batch_size": getattr(args, "batch_size", 1),
        "fid_gen_batch": getattr(args, "batch_size", 1),
        "fid_image_size": image_shape[-1] if image_shape is not None else 0,
        "q_loss_weight": 1.0,
        "q_u_eps": 0.0,
        "lambda_reg": 0.0,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


@dataclass
class Paths:
    sample_dir: str
    checkpoint_dir: str


def prepare_paths(args) -> Paths:
    """Ensure run directories exist and return their locations."""
    sample_dir = os.path.join(args.runs_dir, "samples")
    checkpoint_dir = os.path.join(args.runs_dir, "quantile_fm")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return Paths(sample_dir, checkpoint_dir)


def minibatch_ot_pairing(x0, x1, *, entropic_eps=None, hard_match=True):
    """
    x0: (B, D) source batch ~ q0
    x1: (B, D) target batch ~ q1
    entropic_eps: ignored (always uses exact EMD as requested)
    hard_match:   ignored (pairing via row-wise argmax of EMD plan)

    Returns:
        idx1: (B,) indices so that pair i is (x0[i], x1[idx1[i]])
        P   : (B,B) EMD transport plan (torch.Tensor)
    """
    if ot is None:
        raise ImportError("POT (Python Optimal Transport) package is required for OT pairing")

    if x0.shape[0] != x1.shape[0]:
        raise ValueError(f"x0 and x1 must have same batch size; got {x0.shape[0]} vs {x1.shape[0]}")

    device = x0.device
    with torch.no_grad():
        C= torch.cdist(x0,x1).cpu().numpy()
        # CPU/double for POT
        x0d = x0.detach().cpu().numpy()
        x1d = x1.detach().cpu().numpy()

        # Squared Euclidean cost
        #C = ot.dist(x0d, x1d, metric='euclidean') ** 2
        a = ot.unif(len(x0d))
        b = ot.unif(len(x1d))

        # Exact EMD plan
        P_np = ot.emd(a, b, C)

        # Back to torch on original device
        P = torch.tensor(P_np, dtype=torch.float32, device=device)

        # Row-wise argmax pairing (ties resolve to first max)
        idx1 = torch.argmax(P, dim=0)

    return idx1, P
