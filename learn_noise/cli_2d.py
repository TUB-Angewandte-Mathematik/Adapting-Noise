from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch
import wandb
from learn_noise.configs import apply_overrides, list_configs, load_config, parse_override_strings
from learn_noise.networks import MLP, MyMLPNoSpaceEmbedding
from learn_noise.training import pretrain_quantile, train_fm_baseline, train_fm_quantile


def _build_model(args: SimpleNamespace) -> torch.nn.Module:
    if args.target_dataset in {"funnel", "nealfunnel"}:
        model = MyMLPNoSpaceEmbedding(
            input_dim=args.dim,
            out_dim=args.dim,
            hidden_size=args.hidden_size,
            hidden_layers=args.hidden_layers,
            time_emb=args.time_embedding,
            concat_t_emb=bool(args.concat_t_emb),
        )
    else:
        model = MLP(
            input_dim=args.dim,
            out_dim=args.dim,
            hidden_size=args.hidden_size,
            hidden_layers=args.hidden_layers,
            time_emb=args.time_embedding,
            input_emb='sinusoidal',
            concat_t_emb=bool(args.concat_t_emb),
        )
    return model.to(torch.device(args.device))


def _make_namespace(cfg: Dict[str, object]) -> SimpleNamespace:
    cfg = dict(cfg)
    image_shape = cfg.get("image_shape")
    if image_shape in (None, "null"):
        cfg["image_shape"] = None
    else:
        cfg["image_shape"] = tuple(image_shape)
    cfg.setdefault("latent_viz_samples", 0)
    cfg.setdefault("latent_atlas_grid", 1)
    cfg.setdefault("sample_vis_interval", 0)
    cfg.setdefault("sample_vis_count", 0)
    cfg.setdefault("sample_vis_nrow", 8)
    cfg.setdefault("fid_eval_interval", 0)
    cfg.setdefault("fid_num_gen", 0)
    cfg.setdefault("fid_batch_size", cfg.get("batch_size", 0))
    cfg.setdefault("fid_gen_batch", cfg.get("batch_size", 0))
    cfg.setdefault("fid_image_size", 0)
    cfg.setdefault("lambda_reg", 0.0)
    return SimpleNamespace(**cfg)


def _init_wandb(args: SimpleNamespace, run_name: str, disabled: bool) -> None:
    if disabled:
        wandb.init(mode="disabled")
        return
    project = getattr(args, "wandb_project", None)
    if not project:
        wandb.init(mode="disabled")
        return
    wandb_kwargs = {
        "project": project,
        "entity": getattr(args, "wandb_entity", None),
        "group": getattr(args, "wandb_group", None),
        "name": run_name,
        "config": vars(args),
    }
    wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
    wandb.init(**wandb_kwargs)


def _save_config(run_dir: Path, cfg: Dict[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    import yaml

    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn-Noise 2D experiments")
    parser.add_argument("--config", type=str, default="default", choices=list_configs("2d"))
    parser.add_argument("--pretrain", action="store_true", help="Run quantile pretraining before joint training")
    parser.add_argument("--baseline", action="store_true", help="Run baseline FM instead of quantile FM")
    parser.add_argument("--dataset", type=str, help="Override target_dataset in the config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--output-root", type=Path, default=Path("results_2d"))
    parser.add_argument("--quantile-checkpoint", type=str)
    parser.add_argument("--freeze-quantile", action="store_true")
    parser.add_argument("--use-minibatch-ot", action="store_true")
    parser.add_argument("--baseline-flow", type=str, choices=["linear", "mmd", "kac"])
    parser.add_argument("--baseline-latent", type=str)
    parser.add_argument("--baseline-flow-T", type=float)
    parser.add_argument("--baseline-mmd-b", type=float)
    parser.add_argument("--baseline-kac-a", type=float)
    parser.add_argument("--baseline-kac-c", type=float)
    parser.add_argument("--baseline-kac-epsilon", type=float)
    parser.add_argument("--baseline-kac-lookup-M", type=int)
    parser.add_argument("--baseline-kac-lookup-K", type=int)
    parser.add_argument("--q-loss-weight", type=float, help="Override q_loss_weight")
    parser.add_argument(
        "--q-objective",
        type=str,
        choices=["energy", "sinkhorn", "plan_action"],
        help="Override quantile OT objective during pretraining",
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--wandb-group", type=str)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config entries (top-level only)",
    )
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()

    base_cfg = load_config("2d", cli_args.config)
    base_cfg.setdefault("student_t_df", 4.0)
    base_cfg.setdefault("student_t_scale", 1.0)
    base_cfg.setdefault("quantile_const_iters", None)
    base_cfg.setdefault("quantile_decay_iters", None)
    overrides = parse_override_strings(cli_args.override)

    if cli_args.dataset is not None:
        overrides["target_dataset"] = cli_args.dataset
    if cli_args.seed is not None:
        overrides["seed"] = cli_args.seed
    if cli_args.device is not None:
        overrides["device"] = cli_args.device
    if cli_args.quantile_checkpoint is not None:
        overrides["quantile_checkpoint"] = cli_args.quantile_checkpoint
    if cli_args.freeze_quantile:
        overrides["freeze_quantile"] = True
    if cli_args.use_minibatch_ot:
        overrides["use_minibatch_ot"] = True
    if cli_args.baseline_flow is not None:
        overrides["baseline_flow"] = cli_args.baseline_flow
    if cli_args.baseline_latent is not None:
        overrides["baseline_latent"] = cli_args.baseline_latent
    if cli_args.baseline_flow_T is not None:
        overrides["baseline_flow_T"] = cli_args.baseline_flow_T
    if cli_args.baseline_mmd_b is not None:
        overrides["baseline_mmd_b"] = cli_args.baseline_mmd_b
    if cli_args.baseline_kac_a is not None:
        overrides["baseline_kac_a"] = cli_args.baseline_kac_a
    if cli_args.baseline_kac_c is not None:
        overrides["baseline_kac_c"] = cli_args.baseline_kac_c
    if cli_args.baseline_kac_epsilon is not None:
        overrides["baseline_kac_epsilon"] = cli_args.baseline_kac_epsilon
    if cli_args.baseline_kac_lookup_M is not None:
        overrides["baseline_kac_lookup_M"] = cli_args.baseline_kac_lookup_M
    if cli_args.baseline_kac_lookup_K is not None:
        overrides["baseline_kac_lookup_K"] = cli_args.baseline_kac_lookup_K
    if cli_args.q_loss_weight is not None:
        overrides["q_loss_weight"] = cli_args.q_loss_weight
    if cli_args.q_objective is not None:
        overrides["q_objective"] = cli_args.q_objective
    if cli_args.wandb_project is not None:
        overrides["wandb_project"] = cli_args.wandb_project
    if cli_args.wandb_entity is not None:
        overrides["wandb_entity"] = cli_args.wandb_entity
    if cli_args.wandb_group is not None:
        overrides["wandb_group"] = cli_args.wandb_group
    if cli_args.name is not None:
        overrides["name"] = cli_args.name

    apply_overrides(base_cfg, overrides)

    run_suffix = base_cfg.get("name") or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    dataset = base_cfg["target_dataset"]
    run_name = f"{dataset}-{run_suffix}"

    run_dir = cli_args.output_root / dataset / run_name
    base_cfg["runs_dir"] = str(run_dir)

    # Namespace for trainers
    args = _make_namespace(base_cfg)
    args.mode = "fm_baseline" if cli_args.baseline else "fm_and_quantile"

    # Set seeds early for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    _save_config(run_dir, base_cfg)

    wandb_disabled = cli_args.no_wandb or not bool(base_cfg.get("log_wandb", True))
    _init_wandb(args, run_name, wandb_disabled)

    model = _build_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if cli_args.pretrain:
        pretrain_cfg = dict(base_cfg)
        pretrain_cfg["runs_dir"] = str(run_dir / "pretrain")
        pretrain_args = _make_namespace(pretrain_cfg)
        pretrain_args.mode = "pretrain_quantile"
        _save_config(Path(pretrain_args.runs_dir), pretrain_cfg)
        _, _, pretrain_steps = pretrain_quantile(pretrain_args)
        args._pretrain_step_offset = int(getattr(pretrain_args, "_pretrain_step_offset", pretrain_steps))
        # Use resulting checkpoint automatically if saved
        quant_ckpt = Path(pretrain_args.runs_dir) / "quantile_ot" / "quantile.pt"
        if quant_ckpt.exists():
            args.quantile_checkpoint = str(quant_ckpt)

    args.runs_dir = str(run_dir)

    if cli_args.baseline:
        train_fm_baseline(args, model, optimizer)
    else:
        train_fm_quantile(args, model, optimizer)

    if not wandb_disabled:
        wandb.finish()


if __name__ == "__main__":
    main()
