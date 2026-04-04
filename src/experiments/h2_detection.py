"""
h2_detection.py
---------------
H2 experiment: can yield-threshold curve deformation be automatically
detected from the confidence score distribution?

Uses the .npz score files already produced by h1_deviations.py and
computes two detection metrics for each condition:

  1. ΔAUC-YT  — absolute change in area under the yield-threshold curve
               relative to the clean baseline.

  2. Wasserstein-1 distance (Earth Mover's Distance) between the
               clean baseline confidence score distribution and the
               deviated confidence score distribution.

Statistical significance:
  For each corruption type, we have 5 severity levels, giving 5
  (ΔAUC-YT, Wasserstein) pairs.  We test whether the mean ΔAUC-YT
  across severities is significantly different from zero using a
  one-sample t-test against H0: μ = 0.

Outputs (all in --out directory):
  h2_results.csv          — full table: tag, severity, delta_auc_yt, wasserstein
  h2_summary.csv          — per-tag summary: mean, std, t-stat, p-value
  figures/
    h2_heatmap_delta_auc.pdf   — heatmap: |ΔAUC-YT| per tag × severity
    h2_heatmap_wasserstein.pdf — heatmap: Wasserstein per tag × severity
    h2_scatter.pdf             — scatter: Wasserstein vs ΔAUC-YT

Usage
-----
    python src/experiments/h2_detection.py \
        --h1-dir  runs/results/wheat/h1 \
        --baseline runs/results/wheat/baseline_scores.npz \
        --out      runs/results/wheat/h2
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wasserstein_distance

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


SINGLE_TAGS = ["brightness", "blur", "noise"]
COMBO_TAGS  = ["brightness+blur", "brightness+noise", "blur+noise", "all"]
ALL_TAGS    = SINGLE_TAGS + COMBO_TAGS
SEVERITIES  = [1, 2, 3, 4, 5]


def load_scores(npz_path):
    """Load scores array from an npz file."""
    data = np.load(npz_path)
    return data["scores"]


def plot_heatmap(matrix, row_labels, col_labels, title, cbar_label,
                 save_path, cmap="YlOrRd"):
    """Plot a coloured heatmap with value annotations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label=cbar_label)

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels([f"Sev {s}" for s in col_labels], fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title(title, fontsize=13, pad=12)

    # Annotate each cell
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {save_path}")


def plot_scatter(results, save_path):
    """Scatter plot: Wasserstein distance vs |ΔAUC-YT|."""
    tag_markers = {
        "brightness": "o", "blur": "s", "noise": "^",
        "brightness+blur": "D", "brightness+noise": "P",
        "blur+noise": "X", "all": "*",
    }
    tag_colors = {
        "brightness": "#e41a1c", "blur": "#377eb8", "noise": "#4daf4a",
        "brightness+blur": "#984ea3", "brightness+noise": "#ff7f00",
        "blur+noise": "#a65628", "all": "#f781bf",
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    plotted = set()
    for row in results:
        tag     = row["tag"]
        x       = abs(row["delta_auc_yt"])
        y       = row["wasserstein"]
        marker  = tag_markers.get(tag, "o")
        color   = tag_colors.get(tag, "grey")
        label   = tag if tag not in plotted else None
        ax.scatter(x, y, marker=marker, color=color, s=60,
                   alpha=0.8, label=label)
        plotted.add(tag)

    ax.set_xlabel("|ΔAUC-YT|", fontsize=12)
    ax.set_ylabel("Wasserstein Distance", fontsize=12)
    ax.set_title("H2 — Detection Metrics per Condition", fontsize=13)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {save_path}")


def main(args):
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)

    # ---- Load baseline scores -----------------------------------
    baseline_data   = np.load(args.baseline)
    baseline_scores = baseline_data["scores"]
    bl_auc          = float(np.trapezoid(baseline_data["yields"],
                                     baseline_data["thresholds"]))
    print(f"Baseline AUC-YT: {bl_auc:.4f}")
    print(f"Baseline score distribution: "
          f"mean={baseline_scores.mean():.3f}  "
          f"std={baseline_scores.std():.3f}")

    # ---- Compute metrics for every condition --------------------
    results = []
    scores_dir = os.path.join(args.h1_dir, "scores")

    for tag in ALL_TAGS:
        for sev in SEVERITIES:
            npz_path = os.path.join(scores_dir, f"{tag}_sev{sev}.npz")
            if not os.path.exists(npz_path):
                print(f"  WARNING: {npz_path} not found, skipping.")
                continue

            data   = np.load(npz_path)
            scores = data["scores"]
            auc    = float(data["auc_yt"])
            delta  = auc - bl_auc
            w_dist = float(wasserstein_distance(baseline_scores, scores))

            results.append({
                "tag"         : tag,
                "severity"    : sev,
                "auc_yt"      : auc,
                "delta_auc_yt": delta,
                "wasserstein" : w_dist,
            })

            print(f"  {tag:20s}  sev{sev}  "
                  f"ΔAUC-YT={delta:+.4f}  W={w_dist:.4f}")

    # ---- Write full results CSV ---------------------------------
    csv_path = os.path.join(args.out, "h2_results.csv")
    fieldnames = ["tag", "severity", "auc_yt", "delta_auc_yt", "wasserstein"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nFull results → {csv_path}")

    # ---- Per-tag summary with t-test ----------------------------
    summary = []
    summary_path = os.path.join(args.out, "h2_summary.csv")

    print("\nPer-tag statistical summary (one-sample t-test, H0: μ_ΔAUC = 0):")
    for tag in ALL_TAGS:
        tag_rows = [r for r in results if r["tag"] == tag]
        if not tag_rows:
            continue
        deltas = [r["delta_auc_yt"] for r in tag_rows]
        wdists = [r["wasserstein"]  for r in tag_rows]

        mean_d = np.mean(deltas)
        std_d  = np.std(deltas,  ddof=1)
        mean_w = np.mean(wdists)

        t_stat, p_val = stats.ttest_1samp(deltas, popmean=0.0)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01
              else ("*" if p_val < 0.05 else "ns"))

        print(f"  {tag:20s}  "
              f"mean_ΔAUC={mean_d:+.4f} (±{std_d:.4f})  "
              f"mean_W={mean_w:.4f}  "
              f"t={t_stat:+.2f}  p={p_val:.4f}  {sig}")

        summary.append({
            "tag"          : tag,
            "mean_delta_auc": mean_d,
            "std_delta_auc" : std_d,
            "mean_wasserstein": mean_w,
            "t_stat"       : t_stat,
            "p_value"      : p_val,
            "significance" : sig,
        })

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["tag", "mean_delta_auc", "std_delta_auc",
                           "mean_wasserstein", "t_stat", "p_value",
                           "significance"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSummary → {summary_path}")

    # ---- Heatmaps -----------------------------------------------
    # Build matrices [n_tags × n_severities]
    n_tags = len(ALL_TAGS)
    n_sev  = len(SEVERITIES)
    delta_mat = np.zeros((n_tags, n_sev))
    wdist_mat = np.zeros((n_tags, n_sev))

    for row in results:
        i = ALL_TAGS.index(row["tag"])
        j = SEVERITIES.index(row["severity"])
        delta_mat[i, j] = abs(row["delta_auc_yt"])
        wdist_mat[i, j] = row["wasserstein"]

    plot_heatmap(
        delta_mat, ALL_TAGS, SEVERITIES,
        title="H2 — |ΔAUC-YT| per Corruption Type × Severity",
        cbar_label="|ΔAUC-YT|",
        save_path=os.path.join(args.out, "figures", "h2_heatmap_delta_auc.pdf"))

    plot_heatmap(
        wdist_mat, ALL_TAGS, SEVERITIES,
        title="H2 — Wasserstein Distance per Corruption Type × Severity",
        cbar_label="Wasserstein Distance",
        save_path=os.path.join(args.out, "figures",
                               "h2_heatmap_wasserstein.pdf"))

    # ---- Scatter ------------------------------------------------
    plot_scatter(results,
                 save_path=os.path.join(args.out, "figures", "h2_scatter.pdf"))

    print(f"\nAll H2 outputs saved to {args.out}/")
    print("Next: run h3_correction.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="H2: compute detection metrics for deviated conditions.")
    parser.add_argument("--h1-dir",    required=True,
                        help="Directory containing H1 score .npz files")
    parser.add_argument("--baseline",  required=True,
                        help="Path to baseline_scores.npz")
    parser.add_argument("--out",       required=True)
    args = parser.parse_args()
    main(args)
