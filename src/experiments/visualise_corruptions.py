"""
visualise_corruptions.py
------------------------
Generates a grid image showing sample wheat kernel images under each
corruption type and severity level.

Layout:
  Rows    : corruption types (brightness, blur, noise)
  Columns : severity levels (0=clean, 1, 2, 3, 4, 5)

Output: runs/results/wheat_tiny/figures/corruption_grid.pdf
        runs/results/wheat_tiny/figures/corruption_grid.png

Usage:
    python src/experiments/visualise_corruptions.py \
        --datalist runs/datalist/wheat_tiny \
        --out      runs/results/wheat_tiny/figures
"""

import argparse
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data.corruptions import (
    BrightnessShift, GaussianBlur, GaussianNoise,
    BRIGHTNESS_FACTORS, BLUR_SIGMAS, NOISE_STDS
)
from data.split import read_datalist


def load_random_image(datalist_path, label=1, seed=42):
    """Load a random image of the given label (1=good kernel)."""
    samples = read_datalist(datalist_path)
    rng     = random.Random(seed)
    pool    = [(p, l) for p, l in samples if l == label]
    path, _ = rng.choice(pool)
    return Image.open(path).convert("RGB")


def apply_corruption(img, name, severity):
    """Apply a named corruption at a given severity (1-5)."""
    idx = severity - 1
    if name == "brightness":
        return BrightnessShift(BRIGHTNESS_FACTORS[idx])(img)
    elif name == "blur":
        return GaussianBlur(BLUR_SIGMAS[idx])(img)
    elif name == "noise":
        return GaussianNoise(NOISE_STDS[idx], seed=0)(img)
    else:
        raise ValueError(f"Unknown corruption: {name}")


def make_corruption_grid(datalist_path, out_dir, seed=42):
    os.makedirs(out_dir, exist_ok=True)

    corruptions = [
        ("Brightness shift", "brightness",
         [f"×{f:.2f}" for f in BRIGHTNESS_FACTORS]),
        ("Gaussian blur",    "blur",
         [f"σ={s}" for s in BLUR_SIGMAS]),
        ("Gaussian noise",   "noise",
         [f"std={s}" for s in NOISE_STDS]),
    ]

    severities  = [0, 1, 2, 3, 4, 5]   # 0 = clean
    n_rows      = len(corruptions)
    n_cols      = len(severities)
    thumb_size  = 160
    pad         = 8
    label_h     = 22
    header_h    = 30

    fig_w = n_cols * (thumb_size + pad) + pad + 110  # 110 for row labels
    fig_h = n_rows * (thumb_size + pad + label_h) + pad + header_h

    fig = plt.figure(figsize=(fig_w / 72, fig_h / 72), dpi=150)
    fig.patch.set_facecolor("#f8f8f8")

    col_labels = ["Clean"] + [f"Severity {s}" for s in range(1, 6)]

    # Use a different random seed per corruption type so we see varied kernels
    seeds = [42, 99, 17]

    for row_idx, ((corr_label, corr_name, param_labels), row_seed) in \
            enumerate(zip(corruptions, seeds)):

        img_clean = load_random_image(datalist_path, label=1, seed=row_seed)
        img_clean = img_clean.resize((thumb_size, thumb_size), Image.LANCZOS)

        for col_idx, sev in enumerate(severities):
            # Position
            left = 110 + pad + col_idx * (thumb_size + pad)
            top  = header_h + pad + row_idx * (thumb_size + pad + label_h)

            ax = fig.add_axes([
                left   / fig_w,
                1 - (top + thumb_size) / fig_h,
                thumb_size / fig_w,
                thumb_size / fig_h,
            ])

            if sev == 0:
                img = img_clean
                border_color = "#2ca02c"
            else:
                img = apply_corruption(img_clean, corr_name, sev)
                border_color = "#d62728"

            ax.imshow(np.array(img))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2)

            # Severity parameter label below image
            ax_label = fig.add_axes([
                left   / fig_w,
                1 - (top + thumb_size + label_h) / fig_h,
                thumb_size / fig_w,
                label_h / fig_h,
            ])
            ax_label.set_axis_off()
            param_text = "—" if sev == 0 else param_labels[sev - 1]
            ax_label.text(0.5, 0.5, param_text,
                          ha="center", va="center",
                          fontsize=6.5, color="#444444",
                          transform=ax_label.transAxes)

        # Row label (corruption type name)
        row_center_y = (header_h + pad + row_idx * (thumb_size + pad + label_h)
                        + thumb_size / 2)
        fig.text(
            100 / fig_w,
            1 - row_center_y / fig_h,
            corr_label,
            ha="right", va="center",
            fontsize=8, fontweight="bold",
            rotation=0, color="#222222",
        )

    # Column headers
    for col_idx, col_label in enumerate(col_labels):
        left = 110 + pad + col_idx * (thumb_size + pad) + thumb_size / 2
        fig.text(
            left / fig_w,
            1 - (header_h / 2) / fig_h,
            col_label,
            ha="center", va="center",
            fontsize=8, fontweight="bold",
            color="#222222",
        )

    # Legend
    green_patch = mpatches.Patch(color="#2ca02c", label="Clean baseline")
    red_patch   = mpatches.Patch(color="#d62728", label="Deviated")
    fig.legend(handles=[green_patch, red_patch],
               loc="lower right", fontsize=7,
               bbox_to_anchor=(0.99, 0.01),
               framealpha=0.8)

    fig.suptitle(
        "Imaging Deviations Applied to GrainSet Wheat Kernels",
        fontsize=10, fontweight="bold", y=0.99,
        color="#111111",
    )

    # Save
    pdf_path = os.path.join(out_dir, "corruption_grid.pdf")
    png_path = os.path.join(out_dir, "corruption_grid.png")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor(),
                dpi=150)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate corruption visualisation grid.")
    parser.add_argument("--datalist", required=True,
                        help="Directory containing test.txt")
    parser.add_argument("--out",      required=True,
                        help="Output directory for figures")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    make_corruption_grid(
        datalist_path=os.path.join(args.datalist, "test.txt"),
        out_dir=args.out,
        seed=args.seed,
    )
