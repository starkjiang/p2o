#!/usr/bin/env python3
"""
scripts/train.py — Run preference-optimisation experiments from the command line.

Basic usage (train all five methods with defaults):
    python scripts/train.py

Train a single method:
    python scripts/train.py --methods p2o

Override any Config field:
    python scripts/train.py --methods dpo p2o --n_train_per_ds 200 --n_epochs 3

All results, plots, and a results.json are written to --output_dir (default ./outputs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

# Allow `python scripts/train.py` from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer

from p2o.config import Config
from p2o.data import build_loaders
from p2o.trainer import train_model, VALID_METHODS
from p2o.plot import plot_training_curves, plot_final_bars, print_result_tables


# CLI.

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="P²O vs DPO/IPO/KTO/PKTO — preference optimisation benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--methods", nargs="+", default=list(VALID_METHODS),
        choices=VALID_METHODS,
        help="Which methods to run (space-separated).",
    )

    # Every Config field can be overridden from the CLI
    cfg_defaults = Config()
    for field_name, field in Config.__dataclass_fields__.items():
        default = getattr(cfg_defaults, field_name)
        ftype = type(default) if default is not None else str
        p.add_argument(f"--{field_name}", type=ftype, default=default,
                       help=f"Config.{field_name}")

    p.add_argument("--no_plots", action="store_true",
                   help="Skip matplotlib figures (useful on headless servers).")
    return p


# Serialisation helper.

def _ser(obj):
    if isinstance(obj, dict):
        return {k: _ser(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ser(v) for v in obj]
    try:
        return float(obj)
    except (TypeError, ValueError):
        return obj


# Main.

def main() -> None:
    args = parser = _build_parser().parse_args()

    # Build Config from parsed args (drop the non-Config arguments)
    cfg_kwargs = {
        f: getattr(args, f)
        for f in Config.__dataclass_fields__
    }
    cfg = Config(**cfg_kwargs)

    # Reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f" GPU :{torch.cuda.get_device_name(0)}")
        print(f" VRAM:{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    methods = args.methods
    print(f"\nMethods: {methods}")
    print(f"Config: n_train={cfg.n_train_per_ds}/ds  n_eval={cfg.n_eval_per_ds}/ds  "
          f"epochs={cfg.n_epochs}  bs={cfg.batch_size}")
    print(f"β={cfg.beta}  ε={cfg.eps_clip}  λ={cfg.lam_kl}  K={cfg.K_proximal}")

    # Load tokenizer and frozen reference model.
    print(f"\nLoading tokenizer and frozen reference model ({cfg.model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ref_model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    ref_model.eval().to(device)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in ref_model.parameters())
    print(f"Params: {n_params:,} ({n_params / 1e6:.0f}M)")

    # Load datasets.
    (
        train_loader_hh, train_loader_shp, train_loader_uf,
        eval_loader_hh,  eval_loader_shp,  eval_loader_uf,
    ) = build_loaders(cfg, tokenizer)

    batches_ds = min(
        len(train_loader_hh),
        len(train_loader_shp),
        len(train_loader_uf),
    )
    batches_ep = batches_ds * 3
    print(f"\nBatches: {batches_ds}/ds × 3 datasets × {cfg.n_epochs} epochs "
          f"= {batches_ep * cfg.n_epochs} total per method")

    # Train all methods.
    histories: dict = {}
    for method in methods:
        hist = train_model(
            method, ref_model,
            train_loader_hh, train_loader_shp, train_loader_uf,
            eval_loader_hh,  eval_loader_shp,  eval_loader_uf,
            cfg, device,
        )
        # Map internal keys ("dpo") to display labels ("DPO") for plots
        label = "P²O" if method == "p2o" else method.upper()
        histories[label] = hist

    # Verify x-axis alignment.
    if len(histories) > 1:
        ref_bx  = next(iter(histories.values()))["batch_x"]
        ref_ebx = next(iter(histories.values()))["eval_batch_x"]
        all_ok  = True
        for name, h in histories.items():
            ok = (h["batch_x"] == ref_bx) and (h["eval_batch_x"] == ref_ebx)
            if not ok:
                all_ok = False
            marker = "Checked" if ok else "X"
            print(f"  {marker}  {name}  "
                  f"train={len(h['batch_x'])} eval={len(h['eval_batch_x'])} match={ok}")
        print("Checked! All x-axes identical" if all_ok else "X x-axis mismatch — check dataset loading")

    # Print tables.
    print_result_tables(histories)

    # Save results.json.
    out = dict(
        config=cfg.to_dict(),
        results={
            name: dict(
                hh_rlhf=_ser(h["final_hh"]),
                shp=_ser(h["final_shp"]),
                ultrafeedback=_ser(h["final_uf"]),
                training_time_min=h["training_time"] / 60,
            )
            for name, h in histories.items()
        },
    )
    json_path = os.path.join(cfg.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # Plots.
    if not args.no_plots:
        plot_training_curves(
            histories, cfg, batches_ep,
            save_path=os.path.join(cfg.output_dir, "curves.png"),
        )
        plot_final_bars(
            histories, cfg,
            save_path=os.path.join(cfg.output_dir, "final_bars.png"),
        )

    print(f"\nAll outputs in: {cfg.output_dir}/")


if __name__ == "__main__":
    main()
