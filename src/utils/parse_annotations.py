"""
parse_annotations.py
--------------------
Parses the GrainSet XML annotation file for a single grain species
(e.g. wheat.xml) and returns a list of (image_path, binary_label) pairs.

GrainSet XML format (from Table 2 of Fan et al. 2023):
  <dataset>
    <element ID="..." DU_grain="NOR" weight="..." species="..." ...>
      ...
    </element>
    ...
  </dataset>

DU_grain categories:
  NOR  – Normal kernel             → label 1  (GOOD)
  F&S  – Fusarium & Shrivelled     → label 0  (BAD)
  SD   – Sprouted                  → label 0  (BAD)
  MY   – Mouldy                    → label 0  (BAD)
  BN   – Broken                    → label 0  (BAD)
  AP   – Attacked by pests         → label 0  (BAD)
  BP   – Black point (wheat only)  → label 0  (BAD)
  HD   – Heated (maize/sorghum)    → label 0  (BAD)
  UN   – Unripe (rice only)        → label 0  (BAD)
  IM   – Impurities                → label 0  (BAD)

The ID attribute is used to locate the image file.  Images live at:
  <data_root>/<ID>_UP.png   (top-camera view)
  <data_root>/<ID>_DOWN.png (bottom-camera view)

We use the UP view only, consistent with the authors' practice.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple


# All categories that map to label 1 (good/normal).
# Everything else maps to 0 (bad).
GOOD_CATEGORIES = {"NOR"}


def parse_xml(xml_path: str, data_root: str,
              view: str = "UP") -> List[Tuple[str, int]]:
    """
    Parse a GrainSet annotation XML and return a flat list of
    (absolute_image_path, binary_label) tuples.

    Parameters
    ----------
    xml_path : str
        Path to the species XML file, e.g. '/data/wheat/wheat.xml'.
    data_root : str
        Directory that contains the PNG image files.
        Typically the same folder as xml_path or a subfolder called 'images'.
    view : str
        Which camera view to use.  'UP' (default) or 'DOWN'.

    Returns
    -------
    List of (path, label) tuples.
        label = 1  → good (NOR)
        label = 0  → bad  (all DU categories + impurities)
    """
    data_root = Path(data_root)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    samples: List[Tuple[str, int]] = []
    missing = 0

    for elem in root.iter("element"):
        image_id  = elem.get("ID")
        du_grain  = elem.get("DU_grain", "").strip()

        if image_id is None or du_grain == "":
            continue

        image_file = data_root / f"{image_id}_{view}.png"

        if not image_file.exists():
            missing += 1
            continue

        label = 1 if du_grain in GOOD_CATEGORIES else 0
        samples.append((str(image_file), label))

    if missing > 0:
        print(f"[parse_xml] Warning: {missing} image files not found "
              f"under {data_root} (skipped).")

    return samples


def class_distribution(samples: List[Tuple[str, int]]) -> dict:
    """Return a dict with counts for each class label."""
    from collections import Counter
    counts = Counter(label for _, label in samples)
    total  = len(samples)
    return {
        "total"  : total,
        "good"   : counts[1],
        "bad"    : counts[0],
        "pct_good": 100 * counts[1] / total if total else 0,
        "pct_bad" : 100 * counts[0] / total if total else 0,
    }


if __name__ == "__main__":
    # Quick smoke-test — update paths to your local data location.
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test the XML parser on a GrainSet species.")
    parser.add_argument("xml",  help="Path to species XML, e.g. wheat.xml")
    parser.add_argument("root", help="Directory containing PNG images")
    parser.add_argument("--view", default="UP", choices=["UP", "DOWN"])
    args = parser.parse_args()

    samples = parse_xml(args.xml, args.root, view=args.view)
    dist    = class_distribution(samples)

    print(f"Parsed {dist['total']:,} samples")
    print(f"  Good (NOR): {dist['good']:,}  ({dist['pct_good']:.1f}%)")
    print(f"  Bad  (DU+IM): {dist['bad']:,}  ({dist['pct_bad']:.1f}%)")
    print(f"First 3 samples:")
    for path, lbl in samples[:3]:
        print(f"  [{lbl}] {path}")
