"""
plot.py — Publication-quality plots for preference-optimisation results.

Public API
----------
plot_training_curves(histories, cfg, batches_ep, save_path=None)
    Five-row 3-column figure: training curves + per-dataset eval curves.

plot_final_bars(histories, cfg, save_path=None)
    1×3 grouped bar chart: Eval-A / Eval-B / Eval-C side by side.

print_result_tables(histories)
    Console summary tables, one per dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from p2o.config import Config


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


# Figure 1: Training curves.

def plot_training_curves(
    histories: Dict,
    cfg: Config,
    batches_ep: int,
    save_path: Optional[str] = None,
) -> None:
    """
    5-row × 3-column figure:
      Row 0 : Train loss | Eval-A accuracy     | Eval-B accuracy
      Row 1 : Train acc  | Eval-A margin       | Eval-B margin
      Row 2 : Eval-A KL  | Eval-B KL           | Implicit rewards
      Row 3 : Eval-C UF accuracy | UF margin   | UF KL
      Row 4 : Grad norm  | P²O/PKTO clip frac  | Train reward margin
    """
    plt.rcParams.update(_RC)

    ref_h = next(iter(histories.values()))
    bx = np.array(ref_h["batch_x"])
    ebx = np.array(ref_h["eval_batch_x"])

    fig = plt.figure(figsize=(22, 21))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.55, wspace=0.35)
    fig.suptitle(
        "DPO · IPO · KTO · P²O · PKTO — Training curves & 3-dataset eval",
        fontsize=13, fontweight="bold", color="black", y=0.998,
    )

    def _ax(row, col):
        a = fig.add_subplot(gs[row, col])
        _style_ax(a)
        return a

    def _ep(a):
        _ep_line(a, batches_ep, cfg.n_epochs)

    # Row 0.
    a = _ax(0, 0)
    for name, h in histories.items():
        a.plot(bx, _smooth(h["loss"]), color=COLORS[name],
               lw=2.0, ls=LINESTYLES[name], label=name)
    _ep(a)
    a.set_title("Training Loss (not comparable across methods)", fontsize=11, color="#555555")
    a.set_xlabel("Outer batch"); a.set_ylabel("Loss"); a.legend(fontsize=9)

    a = _ax(0, 1)
    for name, h in histories.items():
        a.plot(ebx, h["eval_hh_accuracy"], color=COLORS[name],
               lw=2.0, ls=LINESTYLES[name], marker=MARKERS[name], ms=4, label=name)
    a.axhline(0.5, color="#888888", ls=":", lw=1.2, label="Random"); _ep(a)
    a.set_title("Eval-A (HH) Accuracy ↑  [CRITICAL]", fontsize=11, fontweight="bold")
    a.set_xlabel("Outer batch"); a.set_ylabel("Accuracy")
    a.set_ylim(0.35, 1.0); a.legend(fontsize=9)

    a = _ax(0, 2)
    for name, h in histories.items():
        a.plot(ebx, h["eval_shp_accuracy"], color=COLORS[name],
               lw=2.0, ls=LINESTYLES[name], marker=MARKERS[name], ms=4, label=name)
    a.axhline(0.5, color="#888888", ls=":", lw=1.2, label="Random"); _ep(a)
    a.set_title("Eval-B (SHP) Accuracy ↑  [CRITICAL]", fontsize=11, fontweight="bold")
    a.set_xlabel("Outer batch"); a.set_ylabel("Accuracy")
    a.set_ylim(0.35, 1.0); a.legend(fontsize=9)

    # Row 1.
    a = _ax(1, 0)
    for name, h in histories.items():
        a.plot(bx, _smooth(h["reward_accuracy"]), color=COLORS[name],
               lw=2.0, ls=LINESTYLES[name], label=name)
    a.axhline(0.5, color="#888888", ls=":", lw=1.2); _ep(a)
    a.set_title("Train Accuracy (watch for overfit gap)", fontsize=11, color="#555555")
    a.set_xlabel("Outer batch"); a.set_ylabel("Accuracy")
    a.set_ylim(0.35, 1.0); a.legend(fontsize=9)

    a = _ax(1, 1)
    for name, h in histories.items():
        a.plot(ebx, h["eval_hh_margin"], color=COLORS[name],
               lw=2.0, ls=LINESTYLES[name], marker=MARKERS[name], ms=4, label=name)
    a.axhline(0, color="#888888", ls=":", lw=1.2); _ep(a)
    a.set_title("Eval-A (HH) Margin ↑  [CRITICAL]", fontsize=11, fontweight="bold")
    a.set_xlabel("Outer batch"); a.set_ylabel("h⁺ − h⁻"); a.legend(fontsize=9)

    a = _ax(1, 2)
    for name, h in histories.items():
        a.plot(ebx, h["eval_shp_margin"], color=COLORS[name],
               lw=2.0, ls=LINESTYLES[name], marker=MARKERS[name], ms=4, label=name)
    a.axhline(0, color="#888888", ls=":", lw=1.2); _ep(a)
    a.set_title("Eval-B (SHP) Margin ↑  [CRITICAL]", fontsize=11, fontweight="bold")
    a.set_xlabel("Outer batch"); a.set_ylabel("h⁺ − h⁻"); a.legend(fontsize=9)

    # Row 2.
    a = _ax(2, 0)
    for name, h in histories.items():
        a.plot(ebx, h["eval_hh_kl"], color=COLORS[name],
               lw=2.0, ls=LINESTYLES[name], marker=MARKERS[name], ms=4, label=name)
    _ep(a)
    a.set_title("Eval-A (HH) KL ↓  [CRITICAL]", fontsize=11, fontweight="bold")
    a.set_xlabel("Outer batch"); a.set_ylabel("Token-mean KL"); a.legend(fontsize=9)

    a = _ax(2, 1)
    for name, h in histories.items():
        a.plot(ebx, h["eval_shp_kl"], color=COLORS[name],
               lw=2.0, ls=LINESTYLES[name], marker=MARKERS[name], ms=4, label=name)
    _ep(a)
    a.set_title("Eval-B (SHP) KL ↓  [CRITICAL]", fontsize=11, fontweight="bold")
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

    # Row 3: UltraFeedback eval.
    a = _ax(3, 0)
    for name, h in histories.items():
        a.plot(ebx, h["eval_uf_accuracy"], color=COLORS[name],
               lw=2.0, ls=LINESTYLES[name], marker=MARKERS[name], ms=4, label=name)
    a.axhline(0.5, color="#888888", ls=":", lw=1.2, label="Random"); _ep(a)
    a.set_title("Eval-C (UF) Accuracy ↑  [CRITICAL]", fontsize=11, fontweight="bold")
    a.set_xlabel("Outer batch"); a.set_ylabel("Accuracy")
    a.set_ylim(0.35, 1.0); a.legend(fontsize=9)

    a = _ax(3, 1)
    for name, h in histories.items():
        a.plot(ebx, h["eval_uf_margin"], color=COLORS[name],
               lw=2.0, ls=LINESTYLES[name], marker=MARKERS[name], ms=4, label=name)
    a.axhline(0, color="#888888", ls=":", lw=1.2); _ep(a)
    a.set_title("Eval-C (UF) Margin ↑  [CRITICAL]", fontsize=11, fontweight="bold")
    a.set_xlabel("Outer batch"); a.set_ylabel("h⁺ − h⁻"); a.legend(fontsize=9)

    a = _ax(3, 2)
    for name, h in histories.items():
        a.plot(ebx, h["eval_uf_kl"], color=COLORS[name],
               lw=2.0, ls=LINESTYLES[name], marker=MARKERS[name], ms=4, label=name)
    _ep(a)
    a.set_title("Eval-C (UF) KL ↓  [CRITICAL]", fontsize=11, fontweight="bold")
    a.set_xlabel("Outer batch"); a.set_ylabel("Token-mean KL"); a.legend(fontsize=9)

    # ── Row 4: diagnostics.
    a = _ax(4, 0)
    for name, h in histories.items():
        a.plot(bx, _smooth(h["grad_norm"]), color=COLORS[name],
               lw=2.0, ls=LINESTYLES[name], label=name)
    _ep(a)
    a.set_title("Gradient Norm (diagnostic)", fontsize=11, color="#555555")
    a.set_xlabel("Outer batch"); a.set_ylabel("‖∇θ‖"); a.legend(fontsize=9)

    a = _ax(4, 1)
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

    a = _ax(4, 2)
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


# Figure 2: Final comparison bars.

def plot_final_bars(
    histories:  Dict,
    cfg: Config,
    save_path: Optional[str] = None,
) -> None:
    """
    1×3 grouped bar chart showing final Eval-A / Eval-B / Eval-C results
    for each metric (accuracy, margin, KL).
    """
    plt.rcParams.update(_RC)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Final Evaluation — Eval-A (HH) · Eval-B (SHP) · Eval-C (UF)",
        fontsize=13, fontweight="bold", color="black",
    )

    NAMES = list(histories.keys())
    COLS = [COLORS[n] for n in NAMES]
    x = np.arange(len(NAMES))

    DATASETS = ["final_hh",  "final_shp",  "final_uf"]
    DS_LABELS = ["Eval-A (HH)", "Eval-B (SHP)", "Eval-C (UF)"]
    DS_HATCHES = ["//", "xx", ""]
    DS_ALPHA = [0.90, 0.70, 0.50]

    METRICS = [
        ("reward_accuracy", "Reward Accuracy ↑", "Accuracy",    0.5),
        ("reward_margin",   "Reward Margin ↑",   "h⁺ − h⁻",     0.0),
        ("kl",              "Token-mean KL ↓",   "KL",          None),
    ]

    total_w = 0.7
    n_ds = len(DATASETS)
    bar_w = total_w / n_ds
    offsets = np.linspace(-(total_w - bar_w) / 2, (total_w - bar_w) / 2, n_ds)

    for ax, (mkey, title, ylbl, ref) in zip(axes, METRICS):
        _style_ax(ax)
        all_bars = []
        for ds_key, ds_label, hatch, alpha, offset in zip(
            DATASETS, DS_LABELS, DS_HATCHES, DS_ALPHA, offsets
        ):
            vals = [histories[n][ds_key][mkey] for n in NAMES]
            bars = ax.bar(
                x + offset, vals, bar_w,
                color=COLS, alpha=alpha,
                edgecolor="black", linewidth=0.6,
                hatch=hatch, label=ds_label,
            )
            all_bars.extend(list(bars))

        for bar in all_bars:
            v = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + (0.003 if v >= 0 else -0.015),
                f"{v:.2f}",
                ha="center", va="bottom",
                fontsize=6.5, fontweight="bold", color="black",
            )

        if ref is not None:
            ax.axhline(ref, color="#888888", ls=":", lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(NAMES)
        ax.set_title(title, fontsize=11, fontweight="bold", color="black")
        ax.set_ylabel(ylbl)
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {save_path}")
    plt.show()


# Console result tables.

def print_result_tables(histories: Dict) -> None:
    """Print a ranked result table for each of the three eval datasets."""
    splits = [
        ("final_hh",  "Eval-A (HH-RLHF)"),
        ("final_shp", "Eval-B (SHP)"),
        ("final_uf",  "Eval-C (UltraFeedback)"),
    ]
    for ds_key, ds_name in splits:
        print(f"\n{'=' * 62}")
        print(f"FINAL RESULTS — {ds_name}")
        print(f"{'=' * 62}")
        print(f"{'Method':<6}  {'Accuracy':>10}  {'Margin':>10}  {'KL':>10}  {'Time(m)':>8}")
        print(f"{'-' * 52}")
        for name, h in histories.items():
            fe = h[ds_key]
            print(
                f"{name:<6}  {fe['reward_accuracy']:>10.4f}  "
                f"{fe['reward_margin']:>10.4f}  "
                f"{fe['kl']:>10.5f}  "
                f"{h['training_time'] / 60:>8.1f}"
            )
        best_acc = max(histories, key=lambda n: histories[n][ds_key]["reward_accuracy"])
        best_mg = max(histories, key=lambda n: histories[n][ds_key]["reward_margin"])
        best_kl = min(histories, key=lambda n: histories[n][ds_key]["kl"])
        print(
            f"\n  Best accuracy : {best_acc}  |  "
            f"Best margin : {best_mg}  |  "
            f"Lowest KL : {best_kl}"
        )
