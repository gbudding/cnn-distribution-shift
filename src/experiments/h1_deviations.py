"""
h1_deviations.py
----------------
H1 experiment: does imaging deviation deform the yield-threshold curve?

For each of the 7 deviation types (3 individual + 4 combinations) at
5 severity levels, this script:
  1. Applies the corruption to every image in the test set.
  2. Runs the frozen model and collects confidence scores.
  3. Computes the yield-threshold curve and AUC-YT.
  4. Saves scores and metrics.

Outputs (all in --out directory):
  scores/
    <tag>_sev<N>.npz        — scores, labels, thresholds, yields per condition
  h1_results.csv            — flat table: tag, severity, auc_yt, delta_auc_yt
  figures/
    h1_individual_sev3.pdf  — curves for single corruptions at severity 3
    h1_brightness.pdf       — all 5 severities for brightness
    h1_blur.pdf             — all 5 severities for blur
    h1_noise.pdf            — all 5 severities for noise
    h1_combinations_sev3.pdf— combination corruptions at severity 3

Usage
-----
    python src/experiments/h1_deviations.py \
        --datalist   runs/datalist/wheat \
        --checkpoint runs/checkpoints/wheat_resnet50/best.pth \
        --baseline   runs/results/wheat/baseline_scores.npz \
        --out        runs/results/wheat/h1
"""

import argparse
import csv
import os
import sys
from itertools import product

import numpy as np
import torch
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data.dataset       import GrainSetDataset, eval_transforms
from data.corruptions   import get_all_corruption_configs
from models.train       import load_model
from experiments.baseline_curve import (
    get_confidence_scores, yield_threshold_curve, auc_yt, plot_curve
)


SINGLE_TAGS = ["brightness", "blur", "noise"]
COMBO_TAGS  = ["brightness+blur", "brightness+noise", "blur+noise", "all"]
ALL_TAGS    = SINGLE_TAGS + COMBO_TAGS


def make_deviated_loader(datalist_path, corruption, batch_size, num_workers):
    """Build a DataLoader that applies a corruption before normalisation."""
    transform = T.Compose([
        T.Resize((224, 224)),
        corruption,                   # PIL → PIL corruption
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std =(0.229, 0.224, 0.225)),
    ])
    ds = GrainSetDataset(datalist_path, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def plot_single_tag(thresholds, tag, tag_results, baseline_yields,
                    save_path):
    """Plot all 5 severities for one corruption type on a single axis."""
    yields_dict = {"Baseline (clean)": baseline_yields}
    cmap = plt.cm.YlOrRd
    for sev in range(1, 6):
        key = f"sev{sev}"
        if key in tag_results:
            yields_dict[f"Severity {sev}"] = tag_results[key]
    plot_curve(thresholds, yields_dict,
               title=f"Yield-Threshold Curve — {tag.capitalize()} (all severities)",
               save_path=save_path)


def main(args):
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "scores"),  exist_ok=True)
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load baseline -------------------------------------------
    baseline     = np.load(args.baseline)
    bl_thresholds = baseline["thresholds"]
    bl_yields     = baseline["yields"]
    bl_auc        = float(auc_yt(bl_thresholds, bl_yields))
    print(f"Baseline AUC-YT: {bl_auc:.4f}")

    # ---- Load model ----------------------------------------------
    model     = load_model(args.checkpoint, device)
    test_path = os.path.join(args.datalist, "test.txt")

    # ---- Run all corruption conditions ---------------------------
    configs     = get_all_corruption_configs()
    all_results = []   # list of dicts for CSV output

    # Organise for plotting: tag → sev → yields
    per_tag: dict = {tag: {} for tag in ALL_TAGS}

    print(f"\nRunning {len(configs)} deviation conditions ...")

    for i, cfg in enumerate(configs):
        tag      = cfg["tag"]
        severity = cfg["severity"]
        corr     = cfg["corruption"]
        label    = f"{tag}_sev{severity}"

        print(f"  [{i+1:3d}/{len(configs)}]  {label} ...", end=" ", flush=True)

        loader = make_deviated_loader(
            test_path, corr, args.batch_size, args.num_workers)
        scores, labels = get_confidence_scores(model, loader, device)

        thresholds, yields = yield_threshold_curve(scores)
        auc                = auc_yt(thresholds, yields)
        delta_auc          = auc - bl_auc

        print(f"AUC-YT={auc:.4f}  Δ={delta_auc:+.4f}")

        # Save scores
        np.savez(os.path.join(args.out, "scores", f"{label}.npz"),
                 scores=scores, labels=labels,
                 thresholds=thresholds, yields=yields,
                 auc_yt=auc, delta_auc_yt=delta_auc)

        all_results.append({
            "tag"         : tag,
            "severity"    : severity,
            "auc_yt"      : auc,
            "delta_auc_yt": delta_auc,
        })

        per_tag[tag][f"sev{severity}"] = yields

    # ---- Write CSV -----------------------------------------------
    csv_path = os.path.join(args.out, "h1_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["tag", "severity", "auc_yt", "delta_auc_yt"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults table → {csv_path}")

    # ---- Figures -------------------------------------------------
    # 1. Individual corruptions at severity 3 (representative mid-point)
    sev = 3
    yields_dict_sev3 = {"Baseline (clean)": bl_yields}
    for tag in SINGLE_TAGS:
        key = f"sev{sev}"
        if key in per_tag[tag]:
            yields_dict_sev3[f"{tag.capitalize()} sev{sev}"] = per_tag[tag][key]
    plot_curve(
        bl_thresholds, yields_dict_sev3,
        title=f"H1 — Individual Corruptions (Severity {sev})",
        save_path=os.path.join(args.out, "figures",
                               f"h1_individual_sev{sev}.pdf"))

    # 2. One plot per single-corruption type (all 5 severities)
    for tag in SINGLE_TAGS:
        plot_single_tag(
            bl_thresholds, tag, per_tag[tag], bl_yields,
            save_path=os.path.join(args.out, "figures", f"h1_{tag}.pdf"))

    # 3. Combination corruptions at severity 3
    yields_dict_combo = {"Baseline (clean)": bl_yields}
    for tag in COMBO_TAGS:
        key = f"sev{sev}"
        if key in per_tag[tag]:
            yields_dict_combo[tag] = per_tag[tag][key]
    plot_curve(
        bl_thresholds, yields_dict_combo,
        title=f"H1 — Combination Corruptions (Severity {sev})",
        save_path=os.path.join(args.out, "figures",
                               f"h1_combinations_sev{sev}.pdf"))

    print(f"\nAll H1 outputs saved to {args.out}/")
    print("Next: run h2_detection.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="H1: run model over all deviated test conditions.")
    parser.add_argument("--datalist",    required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--baseline",    required=True,
                        help="Path to baseline_scores.npz")
    parser.add_argument("--out",         required=True)
    parser.add_argument("--batch-size",  type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
