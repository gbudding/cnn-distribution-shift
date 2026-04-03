"""
plot_results.py
---------------
Standalone script to regenerate all publication-quality figures
from saved result files.  Run this after all experiments are complete.

Produces:
  fig1_baseline_curve.pdf         — baseline yield-threshold curve
  fig2_h1_individual.pdf          — single corruptions, all severities
  fig3_h1_combinations.pdf        — combination corruptions at severity 3
  fig4_h2_heatmaps.pdf            — ΔAUC-YT and Wasserstein heatmaps (side-by-side)
  fig5_h3_comparison.pdf          — correction method comparison

Usage
-----
    python src/experiments/plot_results.py \
        --baseline  runs/results/wheat/baseline_scores.npz \
        --h1-dir    runs/results/wheat/h1 \
        --h2-dir    runs/results/wheat/h2 \
        --h3-dir    runs/results/wheat/h3 \
        --out       runs/results/wheat/paper_figures
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from experiments.baseline_curve import yield_threshold_curve, auc_yt

# ---- Style settings -------------------------------------------
plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 10,
    "axes.titlesize"   : 11,
    "axes.labelsize"   : 10,
    "legend.fontsize"  : 8,
    "xtick.labelsize"  : 9,
    "ytick.labelsize"  : 9,
    "figure.dpi"       : 150,
})

SINGLE_TAGS   = ["brightness", "blur", "noise"]
ALL_TAGS      = SINGLE_TAGS + ["brightness+blur", "brightness+noise",
                               "blur+noise", "all"]
SEVERITIES    = [1, 2, 3, 4, 5]
METHOD_LABELS = {
    "no_correction": "No correction",
    "channel_norm" : "Channel norm.",
    "bn_adapt"     : "BN adaptation",
    "tent"         : "TENT",
}
METHOD_COLORS = {
    "no_correction": "#d62728",
    "channel_norm" : "#ff7f0e",
    "bn_adapt"     : "#2ca02c",
    "tent"         : "#9467bd",
}


def load_h1_yields(h1_dir, tag, severity):
    npz = os.path.join(h1_dir, "scores", f"{tag}_sev{severity}.npz")
    if not os.path.exists(npz):
        return None, None
    d = np.load(npz)
    return d["thresholds"], d["yields"]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---- Figure 1: Baseline curve ---------------------------------

def fig1_baseline(baseline_path, out):
    d  = np.load(baseline_path)
    th = d["thresholds"]
    yi = d["yields"]
    auc = auc_yt(th, yi)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(th, yi, color="#1f77b4", linewidth=2.5,
            label=f"Clean baseline  (AUC-YT={auc:.3f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Yield (fraction accepted)")
    ax.set_title("Baseline Yield-Threshold Curve")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out, "fig1_baseline_curve.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---- Figure 2: H1 individual corruptions (3×1 subplots) -------

def fig2_h1_individual(baseline_path, h1_dir, out):
    bl    = np.load(baseline_path)
    bl_th = bl["thresholds"]
    bl_yi = bl["yields"]
    bl_auc = auc_yt(bl_th, bl_yi)

    cmap = plt.cm.YlOrRd
    sev_colors = [cmap(0.2 + 0.15*i) for i in range(5)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for ax, tag in zip(axes, SINGLE_TAGS):
        ax.plot(bl_th, bl_yi, color="black", linewidth=2,
                linestyle="--", label=f"Baseline ({bl_auc:.3f})", zorder=5)

        for sev in SEVERITIES:
            th, yi = load_h1_yields(h1_dir, tag, sev)
            if th is None:
                continue
            auc = auc_yt(th, yi)
            ax.plot(th, yi, color=sev_colors[sev-1], linewidth=1.5,
                    label=f"Sev {sev}  ({auc:.3f})")

        ax.set_title(tag.capitalize())
        ax.set_xlabel("Threshold")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Yield")
    fig.suptitle("H1 — Single Corruptions (All Severities)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out, "fig2_h1_individual.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---- Figure 3: H1 combinations at severity 3 ------------------

def fig3_h1_combinations(baseline_path, h1_dir, out, sev=3):
    bl    = np.load(baseline_path)
    bl_th = bl["thresholds"]
    bl_yi = bl["yields"]
    bl_auc = auc_yt(bl_th, bl_yi)

    combo_tags = ["brightness+blur", "brightness+noise", "blur+noise", "all"]
    all_plot_tags = SINGLE_TAGS + combo_tags
    cmap   = plt.cm.tab10
    colors = [cmap(i) for i in range(len(all_plot_tags))]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(bl_th, bl_yi, color="black", linewidth=2.5,
            linestyle="--", label=f"Baseline", zorder=10)

    for i, tag in enumerate(all_plot_tags):
        th, yi = load_h1_yields(h1_dir, tag, sev)
        if th is None:
            continue
        auc = auc_yt(th, yi)
        lw  = 2.0 if tag in SINGLE_TAGS else 1.5
        ls  = "-"  if tag in SINGLE_TAGS else ":"
        ax.plot(th, yi, color=colors[i], linewidth=lw, linestyle=ls,
                label=f"{tag}  ({auc:.3f})")

    ax.set_xlabel("Threshold"); ax.set_ylabel("Yield")
    ax.set_title(f"H1 — All Corruptions at Severity {sev}", fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out, "fig3_h1_combinations.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---- Figure 4: H2 heatmaps side-by-side -----------------------

def fig4_h2_heatmaps(h2_dir, out):
    rows_full = read_csv(os.path.join(h2_dir, "h2_results.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (field, label, cmap) in zip(axes, [
        ("delta_auc_yt", "|ΔAUC-YT|", "YlOrRd"),
        ("wasserstein",  "Wasserstein Distance", "Blues"),
    ]):
        mat = np.zeros((len(ALL_TAGS), len(SEVERITIES)))
        for row in rows_full:
            if row["tag"] not in ALL_TAGS:
                continue
            i = ALL_TAGS.index(row["tag"])
            j = SEVERITIES.index(int(row["severity"]))
            mat[i, j] = abs(float(row[field]))

        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, label=label, fraction=0.046)
        ax.set_xticks(range(len(SEVERITIES)))
        ax.set_xticklabels([f"Sev {s}" for s in SEVERITIES])
        ax.set_yticks(range(len(ALL_TAGS)))
        ax.set_yticklabels(ALL_TAGS, fontsize=9)
        ax.set_title(f"H2 — {label}", fontsize=11)

        for ii in range(len(ALL_TAGS)):
            for jj in range(len(SEVERITIES)):
                v     = mat[ii, jj]
                color = "white" if v > mat.max() * 0.65 else "black"
                ax.text(jj, ii, f"{v:.3f}", ha="center", va="center",
                        color=color, fontsize=7)

    fig.tight_layout()
    path = os.path.join(out, "fig4_h2_heatmaps.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---- Figure 5: H3 method comparison (bar chart) ---------------

def fig5_h3_comparison(h3_dir, out):
    rows = read_csv(os.path.join(h3_dir, "h3_results.csv"))

    methods_present = list(dict.fromkeys(r["method"] for r in rows))

    # Mean recovery per method per single-corruption type
    fig, ax = plt.subplots(figsize=(8, 5))
    x     = np.arange(len(SINGLE_TAGS))
    width = 0.8 / len(methods_present)

    for mi, method in enumerate(methods_present):
        means = []
        for tag in SINGLE_TAGS:
            tag_rows = [r for r in rows
                        if r["method"] == method and r["tag"] == tag]
            means.append(np.mean([float(r["pct_recovery"]) for r in tag_rows])
                         if tag_rows else 0.0)
        offset = (mi - len(methods_present)/2 + 0.5) * width
        bars = ax.bar(x + offset, means, width,
                      label=METHOD_LABELS.get(method, method),
                      color=METHOD_COLORS.get(method, f"C{mi}"),
                      alpha=0.85, edgecolor="white")

    ax.axhline(y=100, color="black", linewidth=1, linestyle="--",
               label="Full recovery (100%)")
    ax.axhline(y=0, color="grey", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Corruption Type")
    ax.set_ylabel("Mean AUC-YT Recovery (%)")
    ax.set_title("H3 — Correction Method Comparison", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in SINGLE_TAGS])
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(out, "fig5_h3_comparison.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ------------------------------------------------------------------

def main(args):
    os.makedirs(args.out, exist_ok=True)

    print("Generating paper figures ...")
    fig1_baseline(args.baseline, args.out)
    fig2_h1_individual(args.baseline, args.h1_dir, args.out)
    fig3_h1_combinations(args.baseline, args.h1_dir, args.out)

    if args.h2_dir and os.path.exists(os.path.join(args.h2_dir, "h2_results.csv")):
        fig4_h2_heatmaps(args.h2_dir, args.out)

    if args.h3_dir and os.path.exists(os.path.join(args.h3_dir, "h3_results.csv")):
        fig5_h3_comparison(args.h3_dir, args.out)

    print(f"\nAll figures saved to {args.out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate all paper figures from saved results.")
    parser.add_argument("--baseline",  required=True)
    parser.add_argument("--h1-dir",    required=True)
    parser.add_argument("--h2-dir",    default=None)
    parser.add_argument("--h3-dir",    default=None)
    parser.add_argument("--out",       required=True)
    args = parser.parse_args()
    main(args)
