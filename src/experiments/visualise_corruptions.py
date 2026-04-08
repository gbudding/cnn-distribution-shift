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
    """Load a random good kernel image, cropped to single view."""
    samples = read_datalist(datalist_path)
    rng     = random.Random(seed)
    pool    = [(p, l) for p, l in samples if l == label]
    path, _ = rng.choice(pool)
    img     = Image.open(path).convert("RGB")
    # GrainSet images contain two views side by side — crop to left half
    w, h = img.size
    return img.crop((0, 0, w // 2, h))


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


def fit_to_square(img, size):
    """
    Resize image to fit within size x size while preserving aspect ratio,
    then pad with black to make it exactly size x size.
    """
    img.thumbnail((size, size), Image.LANCZOS)
    w, h = img.size
    square = Image.new('RGB', (size, size), (0, 0, 0))
    # Centre the image in the square
    offset_x = (size - w) // 2
    offset_y = (size - h) // 2
    square.paste(img, (offset_x, offset_y))
    return square


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

    severities  = [0, 1, 2, 3, 4, 5]
    n_rows      = len(corruptions)
    n_cols      = len(severities)
    thumb_size  = 160
    pad         = 8
    label_h     = 22
    row_label_w = 120   # width reserved for row labels on the left
    header_h    = 50    # increased to avoid title overlap with col headers
    title_h     = 30    # space for the main title above headers

    fig_w = row_label_w + n_cols * (thumb_size + pad) + pad
    fig_h = title_h + header_h + n_rows * (thumb_size + label_h + pad) + pad + 30

    fig = plt.figure(figsize=(fig_w / 72, fig_h / 72), dpi=150)
    fig.patch.set_facecolor("#f8f8f8")

    col_labels = ["Clean"] + [f"Severity {s}" for s in range(1, 6)]
    seeds = [42, 99, 17]

    for row_idx, ((corr_label, corr_name, param_labels), row_seed) in \
            enumerate(zip(corruptions, seeds)):

        img_clean = load_random_image(datalist_path, label=1, seed=row_seed)
        img_clean = fit_to_square(img_clean, thumb_size)

        for col_idx, sev in enumerate(severities):
            left = row_label_w + pad + col_idx * (thumb_size + pad)
            top  = title_h + header_h + pad + row_idx * (thumb_size + pad + label_h)

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
                spine.set_linewidth(2.5)

            # Parameter label below image
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
                          fontsize=7, color="#444444",
                          transform=ax_label.transAxes)

        # Row label on the left
        row_center_y = (title_h + header_h + pad
                        + row_idx * (thumb_size + label_h + pad)
                        + thumb_size / 2)
        fig.text(
            (row_label_w - 8) / fig_w,
            1 - row_center_y / fig_h,
            corr_label,
            ha="right", va="center",
            fontsize=9, fontweight="bold",
            color="#222222",
        )

    # Column headers — positioned in header_h band, below title
    for col_idx, col_label in enumerate(col_labels):
        cx = row_label_w + pad + col_idx * (thumb_size + pad) + thumb_size / 2
        # Place column headers at vertical midpoint of header band
        cy = title_h + header_h / 2
        fig.text(
            cx / fig_w,
            1 - cy / fig_h,
            col_label,
            ha="center", va="center",
            fontsize=8.5, fontweight="bold",
            color="#222222",
        )

    # Main title — sits in title_h band at the very top
    fig.text(
        0.5, 1 - title_h / 2 / fig_h,
        "Imaging Deviations Applied to GrainSet Wheat Kernels",
        ha="center", va="center",
        fontsize=11, fontweight="bold",
        color="#111111",
    )

    # Legend — bottom right
    green_patch = mpatches.Patch(color="#2ca02c", label="Clean baseline")
    red_patch   = mpatches.Patch(color="#d62728", label="Deviated")
    fig.legend(handles=[green_patch, red_patch],
               loc="lower right", fontsize=8,
               bbox_to_anchor=(0.99, 0.01),
               framealpha=0.9)

    # Save
    pdf_path = os.path.join(out_dir, "corruption_grid.pdf")
    png_path = os.path.join(out_dir, "corruption_grid.png")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
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