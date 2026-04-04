"""
h3_correction.py
----------------
H3 experiment: inference-time correction methods.

Two methods evaluated on all 7x5 deviated test conditions:

  Method 1 - Per-channel input normalisation (channel_norm)
    Each test batch is normalised to zero mean / unit std per channel
    AFTER standard preprocessing, replacing the fixed ImageNet stats
    with stats estimated from the test batch itself.
    Corrects for global brightness and contrast shifts.

  Method 2 - BatchNorm statistics adaptation (bn_adapt)
    BN layers switched to train mode during a warm-up pass over the
    full test set, accumulating running statistics from the deviated
    distribution. Then switched back to eval mode for inference.
    Based on Schneider et al. (NeurIPS 2020) / Li et al. (2017).
"""

import argparse
import copy
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader

SRC_DIR = os.path.dirname(os.path.abspath(__file__))  # = src/
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data.dataset     import GrainSetDataset, eval_transforms
from data.corruptions import get_all_corruption_configs
from models.train     import load_model
from experiments.baseline_curve import yield_threshold_curve, auc_yt

SINGLE_TAGS = ["brightness", "blur", "noise"]
COMBO_TAGS  = ["brightness+blur", "brightness+noise", "blur+noise", "all"]
ALL_TAGS    = SINGLE_TAGS + COMBO_TAGS
SEVERITIES  = [1, 2, 3, 4, 5]


# ------------------------------------------------------------------
# Loader helper
# ------------------------------------------------------------------

def make_loader(datalist_path, corruption, batch_size=64):
    """DataLoader with corruption applied before normalisation."""
    transform = T.Compose([
        T.Resize((224, 224)),
        corruption,
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std =(0.229, 0.224, 0.225)),
    ])
    ds = GrainSetDataset(datalist_path, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)


# ------------------------------------------------------------------
# Method 1 - Per-channel input normalisation
# ------------------------------------------------------------------

def run_channel_norm(model, loader, device):
    """
    Replace per-batch ImageNet normalisation with per-batch
    channel statistics estimated from the test batch itself.

    For each batch, compute mean and std across all images and
    spatial locations per channel, then standardise.
    This corrects for systematic brightness/contrast shifts.
    """
    model.eval()
    all_scores, all_labels = [], []

    # ImageNet stats as tensors for re-normalisation
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406],
                                  device=device).view(1, 3, 1, 1)
    imagenet_std  = torch.tensor([0.229, 0.224, 0.225],
                                  device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            # Undo ImageNet normalisation -> back to [~0,1] space
            images_raw = images * imagenet_std + imagenet_mean

            # Compute per-channel mean and std across batch + spatial dims
            mean = images_raw.mean(dim=[0, 2, 3], keepdim=True)
            std  = images_raw.std (dim=[0, 2, 3], keepdim=True).clamp(min=1e-6)

            # Normalise to zero mean / unit std, then re-apply ImageNet scale
            images_norm = (images_raw - mean) / std
            images_norm = images_norm * imagenet_std + imagenet_mean

            logits = model(images_norm)
            probs  = torch.softmax(logits, dim=1)
            all_scores.append(probs[:, 1].cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


# ------------------------------------------------------------------
# Method 2 - BatchNorm statistics adaptation
# ------------------------------------------------------------------

def run_bn_adapt(model_orig, loader, device):
    """
    Adapt BatchNorm running statistics to the test distribution.

    1. Deep-copy the model (don't modify the original)
    2. Switch all BN layers to train mode (keeps affine params frozen)
    3. Warm-up pass: run all test batches through to accumulate stats
       (momentum=None uses cumulative moving average over all batches)
    4. Switch back to eval mode and run inference with adapted stats

    Based on: Schneider et al. (NeurIPS 2020), Li et al. (2017)
    """
    model = copy.deepcopy(model_orig)
    model.to(device)

    # Step 1: set BN layers to train mode, use cumulative average
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.momentum = None          # cumulative moving average
            m.reset_running_stats()    # start fresh for this test set

    # Step 2: warm-up pass to accumulate BN statistics
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            model(images)   # forward pass updates running_mean/var

    # Step 3: freeze BN back to eval mode with adapted stats
    model.eval()

    # Step 4: inference with adapted stats
    all_scores, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            all_scores.append(probs[:, 1].cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load baseline
    bl_data   = np.load(args.baseline)
    bl_scores = bl_data["scores"]
    bl_thresh = bl_data["thresholds"]
    bl_yields = bl_data["yields"]
    bl_auc    = float(auc_yt(bl_thresh, bl_yields))
    print(f"Baseline AUC-YT: {bl_auc:.4f}")

    # Load model
    model_orig = load_model(args.checkpoint, device)

    test_path = os.path.join(args.datalist, "test.txt")
    configs   = get_all_corruption_configs()
    methods   = ["no_correction", "channel_norm", "bn_adapt"]

    all_results = []

    print(f"\nRunning H3 ({len(methods)} methods x {len(configs)} conditions)...\n")

    for cfg in configs:
        tag      = cfg["tag"]
        severity = cfg["severity"]
        corr     = cfg["corruption"]

        # Load H1 scores for no_correction baseline
        h1_npz = os.path.join(args.h1_dir, "scores",
                               f"{tag}_sev{severity}.npz")
        h1_data       = np.load(h1_npz)
        no_corr_scores = h1_data["scores"]
        no_corr_labels = h1_data["labels"]
        no_corr_auc    = float(h1_data["auc_yt"])

        # Build loader for this corruption
        loader = make_loader(test_path, corr, batch_size=64)

        # Run correction methods
        s1, l1 = run_channel_norm(model_orig, loader, device)
        s2, l2 = run_bn_adapt(model_orig, loader, device)

        method_scores = {
            "no_correction": (no_corr_scores, no_corr_labels),
            "channel_norm" : (s1, l1),
            "bn_adapt"     : (s2, l2),
        }

        # Compute metrics
        deficit = bl_auc - no_corr_auc

        for method, (scores, labels) in method_scores.items():
            th, yi = yield_threshold_curve(scores)
            a      = auc_yt(th, yi)
            delta  = a - bl_auc
            w_dist = float(wasserstein_distance(bl_scores, scores))
            recovery = (100 * (a - no_corr_auc) / deficit
                        if abs(deficit) > 1e-6 else 100.0)

            all_results.append({
                "method"      : method,
                "tag"         : tag,
                "severity"    : severity,
                "auc_yt"      : a,
                "delta_auc_yt": delta,
                "wasserstein" : w_dist,
                "pct_recovery": recovery,
            })

        auc_ch = auc_yt(*yield_threshold_curve(s1))
        auc_bn = auc_yt(*yield_threshold_curve(s2))
        print(f"  {tag:20s}  sev{severity}  | "
              f"no_corr={no_corr_auc:.4f} | "
              f"ch_norm={auc_ch:.4f} | "
              f"bn_adapt={auc_bn:.4f}")

    # Write CSV
    csv_path   = os.path.join(args.out, "h3_results.csv")
    fieldnames = ["method", "tag", "severity", "auc_yt",
                  "delta_auc_yt", "wasserstein", "pct_recovery"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults -> {csv_path}")

    # Summary
    print("\nRecovery summary (mean % AUC-YT recovery):")
    for method in methods:
        rows     = [r for r in all_results if r["method"] == method]
        mean_rec = np.mean([r["pct_recovery"] for r in rows])
        mean_w   = np.mean([r["wasserstein"]  for r in rows])
        print(f"  {method:20s}  "
              f"mean_recovery={mean_rec:+6.1f}%  "
              f"mean_wasserstein={mean_w:.4f}")

    # Recovery heatmap
    fig, axes = plt.subplots(1, len(methods), figsize=(5*len(methods), 5),
                             sharey=True)
    for ax, method in zip(axes, methods):
        mat = np.zeros((len(ALL_TAGS), len(SEVERITIES)))
        for row in all_results:
            if row["method"] != method:
                continue
            i = ALL_TAGS.index(row["tag"])
            j = SEVERITIES.index(row["severity"])
            mat[i, j] = row["pct_recovery"]

        im = ax.imshow(mat, cmap="RdYlGn", vmin=-20, vmax=120, aspect="auto")
        ax.set_xticks(range(len(SEVERITIES)))
        ax.set_xticklabels([f"S{s}" for s in SEVERITIES], fontsize=9)
        ax.set_yticks(range(len(ALL_TAGS)))
        ax.set_yticklabels(ALL_TAGS, fontsize=9)
        ax.set_title(method.replace("_", "\n"), fontsize=10)
        for ii in range(len(ALL_TAGS)):
            for jj in range(len(SEVERITIES)):
                v = mat[ii, jj]
                c = "black" if 10 < v < 90 else "white"
                ax.text(jj, ii, f"{v:.0f}%", ha="center",
                        va="center", color=c, fontsize=7)

    fig.colorbar(im, ax=axes[-1], label="% AUC-YT Recovery")
    fig.suptitle("H3 - AUC-YT Recovery by Method", fontsize=13)
    fig.tight_layout()
    fig_path = os.path.join(args.out, "figures", "h3_recovery_heatmap.pdf")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved -> {fig_path}")
    print(f"\nAll H3 outputs saved to {args.out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="H3: evaluate inference-time correction methods.")
    parser.add_argument("--datalist",   required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline",   required=True)
    parser.add_argument("--h1-dir",     required=True)
    parser.add_argument("--out",        required=True)
    args = parser.parse_args()
    main(args)
