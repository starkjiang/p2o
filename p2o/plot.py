"""
plot.py — Publication-quality plots for preference-optimisation results.
 
Public API
----------
plot_training_curves(histories, cfg, batches_ep, save_path=None)
    Five-row 3-column figure: training curves + per-dataset eval curves.
 
plot_final_bars(histories, cfg, save_path=None)
    Combined grouped bar chart for ALL datasets present in histories.
 
print_result_tables(histories)
    Console summary tables, one per dataset.
"""
 
from __future__ import annotations
 
from pathlib import Path
from typing import Dict, List, Optional
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
from p2o.config import Config
 
 
# ── Dataset display labels ────────────────────────────────────────────────────
 
DS_DISPLAY = {
    "hh":   "HH-RLHF",
    "shp":  "SHP",
    "uf":   "UltraFeedback",
    "orca": "Orca DPO",
    "pku":  "PKU-SafeRLHF",
    "ufb":  "UF-Binarized",
}
 
 
# Palette and style constants.
 
COLORS = {
    "DPO": "#c0392b",
    "IPO": "#d4a017",
    "KTO": "#6c3483",
    "P²O": "#117733",
    "PKTO": "#0072B2",
}
MARKERS = {
    "DPO": "o", "IPO": "s", "KTO": "^", "P²O": "D", "PKTO": "P",
}
LINESTYLES = {
    "DPO": "-",
    "IPO": "--",
    "KTO": "-.",
    "P²O": ":",
    "PKTO": (0, (3, 1, 1, 1)),
}
 
_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "grid.color": "#cccccc",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "legend.facecolor": "white",
    "legend.edgecolor": "#aaaaaa",
    "legend.framealpha": 1.0,
    "lines.linewidth": 2.0,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
}
 
# Hatch patterns for distinguishing datasets in bar charts.
_HATCHES = ["//", "xx", "", "\\\\", "..", "oo"]
_ALPHAS  = [0.90, 0.75, 0.60, 0.85, 0.70, 0.55]
 
 
# ── Helpers ────────────────────────────────────────────────────────────────────
 
def _smooth(a, w: int = 5) -> np.ndarray:
    a = np.array(a, dtype=float)
    if np.all(np.isnan(a)) or len(a) < w:
        return a
    return np.convolve(
        np.pad(a, (w // 2, w - w // 2 - 1), mode="edge"),
        np.ones(w) / w,
        mode="valid",
    )
 
 
def _style_ax(ax) -> None:
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color("black")
        sp.set_linewidth(0.8)
    ax.grid(True, color="#cccccc", linestyle="--", alpha=0.6, linewidth=0.7)
    ax.tick_params(direction="in", length=4, width=0.8)
 
 
def _ep_line(ax, batches_ep: int, n_epochs: int) -> None:
    if n_epochs > 1:
        ax.axvline(
            batches_ep,
            color="#888888", lw=1.0, ls=":", alpha=0.8,
            label="Epoch 2",
        )
 
 
def _detect_datasets(histories: Dict) -> List[str]:
    """Detect which datasets are present in histories (from final_* keys)."""
    ref_h = next(iter(histories.values()))
    ds_order = ["hh", "shp", "uf", "orca", "pku", "ufb"]
    return [ds for ds in ds_order if f"final_{ds}" in ref_h]
 
 
# ── Figure 1: Training curves ─────────────────────────────────────────────────
 
def plot_training_curves(
    histories: Dict,
    cfg: Config,
    batches_ep: int,
    save_path: Optional[str] = None,
) -> None:
    """
    Dynamic multi-row × 3-column figure based on datasets present.
 
    Rows:
      - Row 0: Train loss | first two dataset accuracies
      - Row 1: Train acc  | first two dataset margins
      - Row 2: first two dataset KLs | Implicit rewards
      - Rows 3+: remaining datasets (accuracy | margin | KL) — one row per dataset
      - Last row: Grad norm | Clip fraction | Train reward margin
    """
    plt.rcParams.update(_RC)
 
    ref_h = next(iter(histories.values()))
    bx = np.array(ref_h["batch_x"])
    ebx = np.array(ref_h["eval_batch_x"])
    datasets = _detect_datasets(histories)
 
    # Build the figure with dynamic rows
    # Rows 0-2: standard (train + first 2 datasets), then 1 row per extra dataset,
    # then 1 diagnostic row.
    extra_ds = datasets[2:]  # datasets beyond the first two
    n_rows = 3 + len(extra_ds) + 1  # rows 0-2 (standard) + extra ds rows + diagnostics
 
    fig = plt.figure(figsize=(22, 4.2 * n_rows))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.55, wspace=0.35)
 
    ds_str = " · ".join(DS_DISPLAY.get(d, d.upper()) for d in datasets)
    fig.suptitle(
        f"DPO · IPO · KTO · P²O · PKTO — Training curves & {len(datasets)}-dataset eval",
        fontsize=13, fontweight="bold", color="black", y=0.998,
    )
 
    def _ax(row, col):
        a = fig.add_subplot(gs[row, col])
        _style_ax(a)
        return a
 
    def _ep(a):
        _ep_line(a, batches_ep, cfg.n_epochs)
 
    # ── Row 0: Train loss + first two datasets accuracy ───────────────────
    a = _ax(0, 0)
    for name, h in histories.items():
        a.plot(bx, _smooth(h["loss"]), color=COLORS[name],
               lw=2.0, ls=LINESTYLES[name], label=name)
    _ep(a)
    a.set_title("Training Loss (not comparable across methods)", fontsize=11, color="#555555")
    a.set_xlabel("Outer batch"); a.set_ylabel("Loss"); a.legend(fontsize=9)
 
    for i, ds in enumerate(datasets[:2]):
        a = _ax(0, i + 1)
        key = f"eval_{ds}_accuracy"
        for name, h in histories.items():
            if key in h:
                a.plot(ebx, h[key], color=COLORS[name],
                       lw=2.0, ls=LINESTYLES[name], marker=MARKERS[name], ms=4, label=name)
        a.axhline(0.5, color="#888888", ls=":", lw=1.2, label="Random"); _ep(a)
        label = DS_DISPLAY.get(ds, ds.upper())
        a.set_title(f"Eval ({label}) Accuracy ↑", fontsize=11, fontweight="bold")
        a.set_xlabel("Outer batch"); a.set_ylabel("Accuracy")
        a.set_ylim(0.35, 1.0); a.legend(fontsize=9)
 
    # ── Row 1: Train accuracy + first two datasets margin ─────────────────
    a = _ax(1, 0)
    for name, h in histories.items():
        a.plot(bx, _smooth(h["reward_accuracy"]), color=COLORS[name],
               lw=2.0, ls=LINESTYLES[name], label=name)
    a.axhline(0.5, color="#888888", ls=":", lw=1.2); _ep(a)
    a.set_title("Train Accuracy (watch for overfit gap)", fontsize=11, color="#555555")
    a.set_xlabel("Outer batch"); a.set_ylabel("Accuracy")
    a.set_ylim(0.35, 1.0); a.legend(fontsize=9)
 
    for i, ds in enumerate(datasets[:2]):
        a = _ax(1, i + 1)
        key = f"eval_{ds}_margin"
        for name, h in histories.items():
            if key in h:
                a.plot(ebx, h[key], color=COLORS[name],
                       lw=2.0, ls=LINESTYLES[name], marker=MARKERS[name], ms=4, label=name)
        a.axhline(0, color="#888888", ls=":", lw=1.2); _ep(a)
        label = DS_DISPLAY.get(ds, ds.upper())
        a.set_title(f"Eval ({label}) Margin ↑", fontsize=11, fontweight="bold")
        a.set_xlabel("Outer batch"); a.set_ylabel("h⁺ − h⁻"); a.legend(fontsize=9)
 
    # ── Row 2: First two datasets KL + Implicit rewards ───────────────────
    for i, ds in enumerate(datasets[:2]):
        a = _ax(2, i)
        key = f"eval_{ds}_kl"
        for name, h in histories.items():
            if key in h:
                a.plot(ebx, h[key], color=COLORS[name],
                       lw=2.0, ls=LINESTYLES[name], marker=MARKERS[name], ms=4, label=name)
        _ep(a)
        label = DS_DISPLAY.get(ds, ds.upper())
        a.set_title(f"Eval ({label}) KL ↓", fontsize=11, fontweight="bold")
        a.set_xlabel("Outer batch"); a.set_ylabel("Token-mean KL"); a.legend(fontsize=9)
 
    a = _ax(2, 2)
    for name, h in histories.items():
        a.plot(bx, _smooth(h["chosen_reward"]),   color=COLORS[name],
               lw=2.0, ls=LINESTYLES[name], label=f"{name} y⁺")
        a.plot(bx, _smooth(h["rejected_reward"]), color=COLORS[name],
               lw=1.2, ls="--", alpha=0.5, label=f"{name} y⁻")
    a.axhline(0, color="#888888", ls=":", lw=1.2); _ep(a)
    a.set_title("Implicit Rewards (diagnostic)", fontsize=11, color="#555555")
    a.set_xlabel("Outer batch"); a.set_ylabel("β·log(π_θ/π_ref)"); a.legend(fontsize=7)
 
    # ── Rows 3+: remaining datasets (accuracy | margin | KL) ─────────────
    row_idx = 3
    for ds in extra_ds:
        label = DS_DISPLAY.get(ds, ds.upper())
 
        a = _ax(row_idx, 0)
        key = f"eval_{ds}_accuracy"
        for name, h in histories.items():
            if key in h:
                a.plot(ebx, h[key], color=COLORS[name],
                       lw=2.0, ls=LINESTYLES[name], marker=MARKERS[name], ms=4, label=name)
        a.axhline(0.5, color="#888888", ls=":", lw=1.2, label="Random"); _ep(a)
        a.set_title(f"Eval ({label}) Accuracy ↑", fontsize=11, fontweight="bold")
        a.set_xlabel("Outer batch"); a.set_ylabel("Accuracy")
        a.set_ylim(0.35, 1.0); a.legend(fontsize=9)
 
        a = _ax(row_idx, 1)
        key = f"eval_{ds}_margin"
        for name, h in histories.items():
            if key in h:
                a.plot(ebx, h[key], color=COLORS[name],
                       lw=2.0, ls=LINESTYLES[name], marker=MARKERS[name], ms=4, label=name)
        a.axhline(0, color="#888888", ls=":", lw=1.2); _ep(a)
        a.set_title(f"Eval ({label}) Margin ↑", fontsize=11, fontweight="bold")
        a.set_xlabel("Outer batch"); a.set_ylabel("h⁺ − h⁻"); a.legend(fontsize=9)
 
        a = _ax(row_idx, 2)
        key = f"eval_{ds}_kl"
        for name, h in histories.items():
            if key in h:
                a.plot(ebx, h[key], color=COLORS[name],
                       lw=2.0, ls=LINESTYLES[name], marker=MARKERS[name], ms=4, label=name)
        _ep(a)
        a.set_title(f"Eval ({label}) KL ↓", fontsize=11, fontweight="bold")
        a.set_xlabel("Outer batch"); a.set_ylabel("Token-mean KL"); a.legend(fontsize=9)
 
        row_idx += 1
 
    # ── Last row: diagnostics ─────────────────────────────────────────────
    a = _ax(row_idx, 0)
    for name, h in histories.items():
        a.plot(bx, _smooth(h["grad_norm"]), color=COLORS[name],
               lw=2.0, ls=LINESTYLES[name], label=name)
    _ep(a)
    a.set_title("Gradient Norm (diagnostic)", fontsize=11, color="#555555")
    a.set_xlabel("Outer batch"); a.set_ylabel("‖∇θ‖"); a.legend(fontsize=9)
 
    a = _ax(row_idx, 1)
    if "P²O" in histories:
        a.plot(bx, _smooth(histories["P²O"]["clip_frac"]),
               color=COLORS["P²O"], lw=2.0, ls=LINESTYLES["P²O"], label="P²O")
    if "PKTO" in histories:
        a.plot(bx, _smooth(histories["PKTO"]["clip_frac"]),
               color=COLORS["PKTO"], lw=2.0, ls=LINESTYLES["PKTO"], label="PKTO")
    a.axhline(0.3, color="#888888", ls=":", lw=1.2, label="0.3 guideline"); _ep(a)
    a.set_title("P²O / PKTO Clip Fraction (diagnostic)", fontsize=11, color="#555555")
    a.set_xlabel("Outer batch"); a.set_ylabel("Fraction clipped")
    a.set_ylim(0, 1); a.legend(fontsize=9)
 
    a = _ax(row_idx, 2)
    for name, h in histories.items():
        a.plot(bx, _smooth(h["reward_margin"]), color=COLORS[name],
               lw=2.0, ls=LINESTYLES[name], label=name)
    a.axhline(0, color="#888888", ls=":", lw=1.2); _ep(a)
    a.set_title("Train Reward Margin (diagnostic)", fontsize=11, color="#555555")
    a.set_xlabel("Outer batch"); a.set_ylabel("h⁺ − h⁻"); a.legend(fontsize=9)
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    plt.show()
 
 
# ── Figure 2: Combined final comparison bars ──────────────────────────────────
 
def plot_final_bars(
    histories: Dict,
    cfg: Config,
    save_path: Optional[str] = None,
) -> None:
    """
    1×3 grouped bar chart showing final results for ALL datasets present
    in histories. Each dataset gets a separate bar per method for each metric
    (accuracy, margin, KL).
 
    Dynamically detects datasets from final_* keys in histories.
    """
    plt.rcParams.update(_RC)
 
    datasets = _detect_datasets(histories)
    n_ds = len(datasets)
 
    if n_ds == 0:
        print("  (No final eval data found — skipping bar chart.)")
        return
 
    NAMES = list(histories.keys())
    COLS = [COLORS[n] for n in NAMES]
    x = np.arange(len(NAMES))
 
    ds_keys   = [f"final_{ds}" for ds in datasets]
    ds_labels = [DS_DISPLAY.get(ds, ds.upper()) for ds in datasets]
    ds_hatches = [_HATCHES[i % len(_HATCHES)] for i in range(n_ds)]
    ds_alphas  = [_ALPHAS[i % len(_ALPHAS)] for i in range(n_ds)]
 
    METRICS = [
        ("reward_accuracy", "Reward Accuracy ↑", "Accuracy",    0.5),
        ("reward_margin",   "Reward Margin ↑",   "h⁺ − h⁻",     0.0),
        ("kl",              "Token-mean KL ↓",   "KL",          None),
    ]
 
    # Scale figure width based on number of datasets
    fig_w = max(18, 6 * n_ds)
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, 7))
    fig.patch.set_facecolor("white")
 
    ds_title = " · ".join(ds_labels)
    fig.suptitle(
        f"Final Evaluation — {ds_title}",
        fontsize=13, fontweight="bold", color="black",
    )
 
    total_w = min(0.85, 0.15 * n_ds + 0.25)
    bar_w = total_w / n_ds
    offsets = np.linspace(-(total_w - bar_w) / 2, (total_w - bar_w) / 2, n_ds)
 
    for ax, (mkey, title, ylbl, ref) in zip(axes, METRICS):
        _style_ax(ax)
        all_bars = []
        for ds_key, ds_label, hatch, alpha, offset in zip(
            ds_keys, ds_labels, ds_hatches, ds_alphas, offsets
        ):
            vals = []
            for n in NAMES:
                if ds_key in histories[n]:
                    vals.append(histories[n][ds_key][mkey])
                else:
                    vals.append(0.0)
            bars = ax.bar(
                x + offset, vals, bar_w,
                color=COLS, alpha=alpha,
                edgecolor="black", linewidth=0.6,
                hatch=hatch, label=ds_label,
            )
            all_bars.extend(list(bars))
 
        # Value labels on bars
        fontsize = max(5.0, 7.0 - 0.3 * n_ds)
        for bar in all_bars:
            v = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + (0.003 if v >= 0 else -0.015),
                f"{v:.2f}",
                ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold", color="black",
            )
 
        if ref is not None:
            ax.axhline(ref, color="#888888", ls=":", lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(NAMES)
        ax.set_title(title, fontsize=11, fontweight="bold", color="black")
        ax.set_ylabel(ylbl)
        ax.legend(fontsize=8, loc="best")
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    plt.show()
 
 
# ── Console result tables ─────────────────────────────────────────────────────
 
def print_result_tables(histories: Dict) -> None:
    """Print a ranked result table for each dataset present in histories."""
    datasets = _detect_datasets(histories)
 
    for ds in datasets:
        ds_key = f"final_{ds}"
        ds_name = DS_DISPLAY.get(ds, ds.upper())
 
        print(f"\n{'=' * 62}")
        print(f"FINAL RESULTS — {ds_name}")
        print(f"{'=' * 62}")
        print(f"{'Method':<6}  {'Accuracy':>10}  {'Margin':>10}  {'KL':>10}  {'Time(m)':>8}")
        print(f"{'-' * 52}")
        for name, h in histories.items():
            if ds_key not in h:
                continue
            fe = h[ds_key]
            print(
                f"{name:<6}  {fe['reward_accuracy']:>10.4f}  "
                f"{fe['reward_margin']:>10.4f}  "
                f"{fe['kl']:>10.5f}  "
                f"{h['training_time'] / 60:>8.1f}"
            )
 
        present = {n: h for n, h in histories.items() if ds_key in h}
        if present:
            best_acc = max(present, key=lambda n: present[n][ds_key]["reward_accuracy"])
            best_mg  = max(present, key=lambda n: present[n][ds_key]["reward_margin"])
            best_kl  = min(present, key=lambda n: present[n][ds_key]["kl"])
            print(
                f"\n  Best accuracy : {best_acc}  |  "
                f"Best margin : {best_mg}  |  "
                f"Lowest KL : {best_kl}"
            )