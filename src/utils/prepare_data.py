"""
prepare_data.py
---------------
Top-level script that runs the full data preparation pipeline:

  1. Parse the GrainSet XML annotation file.
  2. Print a class distribution summary.
  3. Create stratified 70/15/15 train/val/test splits.
  4. Write datalist .txt files to disk.
  5. Run a DataLoader smoke-test to confirm images load correctly.

Run this once before training.  The datalist files it creates are
the only inputs needed by the training and experiment scripts.

Usage
-----
    python prepare_data.py \
        --xml   /data/wheat/wheat.xml \
        --root  /data/wheat/ \
        --out   runs/datalist/wheat \
        --view  UP

Expected output
---------------
    Parsed 200,000 samples
      Good (NOR): 120,000  (60.0%)
      Bad  (DU+IM): 80,000  (40.0%)

    Split summary:
      Train :  140,000  (good 84,000 / bad 56,000)
      Val   :   30,000  (good 18,000 / bad 12,000)
      Test  :   30,000  (good 18,000 / bad 12,000)

    Datalist files written to runs/datalist/wheat/

    DataLoader smoke-test:
      Loaded batch: images torch.Size([8, 3, 224, 224])  labels [1, 0, 1, ...]
      Pixel range after normalisation: [-2.12, 2.64]
    All checks passed ✓
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

# Make sure src/ is on the path when running from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.parse_annotations import parse_xml, class_distribution
from data.split              import create_splits, read_datalist
from data.dataset            import GrainSetDataset, eval_transforms, get_class_weights


def main(args):
    # ------------------------------------------------------------------
    # 1. Parse XML
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Parsing annotations from: {args.xml}")
    print(f"Image root:               {args.root}")
    print(f"Camera view:              {args.view}")
    print("=" * 60)

    samples = parse_xml(args.xml, args.root, view=args.view)

    if len(samples) == 0:
        print("ERROR: No samples found. Check --xml and --root paths.")
        sys.exit(1)

    dist = class_distribution(samples)
    print(f"\nParsed {dist['total']:,} samples")
    print(f"  Good (NOR)   : {dist['good']:,}  ({dist['pct_good']:.1f}%)")
    print(f"  Bad  (DU+IM) : {dist['bad']:,}  ({dist['pct_bad']:.1f}%)")

    # ------------------------------------------------------------------
    # 2. Create splits
    # ------------------------------------------------------------------
    print(f"\nCreating splits → {args.out}")
    os.makedirs(args.out, exist_ok=True)
    create_splits(samples, out_dir=args.out, seed=args.seed)

    # ------------------------------------------------------------------
    # 3. DataLoader smoke-test on the test split
    # ------------------------------------------------------------------
    print("\nRunning DataLoader smoke-test on test split ...")
    test_path = os.path.join(args.out, "test.txt")
    test_ds   = GrainSetDataset(test_path, transform=eval_transforms())

    n_good, n_bad = test_ds.class_counts()
    print(f"  Test dataset: {len(test_ds):,} samples  "
          f"(good={n_good:,}  bad={n_bad:,})")

    weights = get_class_weights(test_ds)
    print(f"  Class weights: bad={weights[0]:.3f}  good={weights[1]:.3f}")

    loader = DataLoader(test_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=False)

    imgs, labels = next(iter(loader))
    assert imgs.shape == (args.batch_size, 3, 224, 224), \
        f"Unexpected image shape: {imgs.shape}"
    assert imgs.dtype == torch.float32, \
        f"Unexpected dtype: {imgs.dtype}"

    print(f"  Loaded batch : {imgs.shape}   dtype={imgs.dtype}")
    print(f"  Labels       : {labels.tolist()}")
    print(f"  Pixel range  : [{imgs.min():.2f}, {imgs.max():.2f}]")

    # ------------------------------------------------------------------
    # 4. Save a quick summary file
    # ------------------------------------------------------------------
    summary_path = os.path.join(args.out, "dataset_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Species : {args.xml}\n")
        f.write(f"View    : {args.view}\n")
        f.write(f"Seed    : {args.seed}\n\n")
        f.write(f"Total   : {dist['total']:,}\n")
        f.write(f"Good    : {dist['good']:,}  ({dist['pct_good']:.1f}%)\n")
        f.write(f"Bad     : {dist['bad']:,}  ({dist['pct_bad']:.1f}%)\n\n")

        for split_name in ["train", "val", "test"]:
            split_samples = read_datalist(
                os.path.join(args.out, f"{split_name}.txt"))
            n_g = sum(1 for _, l in split_samples if l == 1)
            n_b = len(split_samples) - n_g
            f.write(f"{split_name:6s}: {len(split_samples):,}  "
                    f"(good={n_g:,}  bad={n_b:,})\n")

    print(f"\nSummary written to {summary_path}")
    print("\nAll checks passed ✓")
    print("="*60)
    print("Next step: train the model")
    print("  python src/models/train.py \\")
    print(f"    --datalist {args.out} \\")
    print("    --out runs/checkpoints/wheat_resnet50")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare GrainSet data: parse XML, split, and verify.")
    parser.add_argument(
        "--xml",  required=True,
        help="Path to the species XML file, e.g. /data/wheat/wheat.xml")
    parser.add_argument(
        "--root", required=True,
        help="Directory containing the PNG image files")
    parser.add_argument(
        "--out",  required=True,
        help="Output directory for datalist .txt files")
    parser.add_argument(
        "--view", default="UP", choices=["UP", "DOWN"],
        help="Which camera view to use (default: UP)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)")
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for smoke-test (default: 8)")
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers for smoke-test (default: 0)")

    args = parser.parse_args()
    main(args)
