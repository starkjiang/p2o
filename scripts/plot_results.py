#!/usr/bin/env python3
"""
scripts/plot_results.py — Regenerate plots from a saved results.json.

Usage:
    python scripts/plot_results.py outputs/results.json
    python scripts/plot_results.py outputs/results.json --output_dir outputs/plots
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p2o.config import Config
from p2o.plot import plot_training_curves, plot_final_bars, print_result_tables


def main() -> None:
    p = argparse.ArgumentParser(description="Re-plot P²O results from a saved JSON file.")
    p.add_argument("results_json", help="Path to results.json produced by scripts/train.py")
    p.add_argument("--output_dir", default=None,
                   help="Directory for saved PNGs (defaults to same folder as JSON).")
    p.add_argument("--no_show", action="store_true",
                   help="Save figures without displaying them (for headless servers).")
    args = p.parse_args()

    with open(args.results_json) as f:
        data = json.load(f)

    cfg        = Config.from_dict(data.get("config", {}))
    results    = data["results"]
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.results_json))
    os.makedirs(output_dir, exist_ok=True)

    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    print_result_tables(results)

    # Training-curve figure requires the full history (batch_x, eval metrics arrays).
    # If the JSON contains those arrays, plot them; otherwise skip gracefully.
    has_curves = any("batch_x" in h for h in results.values())
    if has_curves:
        # batches_ep: infer from the first history that has it
        batches_ep = 0
        for h in results.values():
            if "batch_x" in h and h["batch_x"]:
                # Rough estimate: assume 3 datasets, so epoch boundary ≈ max(batch_x) / n_epochs / 3 * 3
                batches_ep = max(h["batch_x"]) // cfg.n_epochs
                break
        plot_training_curves(
            results, cfg, batches_ep,
            save_path=os.path.join(output_dir, "curves.png"),
        )
    else:
        print("(Training curve arrays not found in JSON — skipping curves plot.)")

    plot_final_bars(
        results, cfg,
        save_path=os.path.join(output_dir, "final_bars.png"),
    )

    print(f"\nPlots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
