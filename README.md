# Distribution Shift in CNN-Based Seed Sorting

> **Assignment 2 project — Deep Neural Engineering (DNE)**
>
> Gerben Budding

## Overview

This repository contains all code for the paper:
*Distribution Shift in CNN-Based Seed Sorting: Causes, Detection, and Inference-Time Correction*

The project investigates how imaging deviations (brightness, blur, noise) deform the yield-threshold curve of a CNN-based seed sorting system, whether these deformations can be automatically detected, and whether inference-time interventions can restore the curve without retraining.

---

## Repository Structure

```
.
├── data/                       ← download instructions (data not committed)
│   └── download.md
├── runs/
│   ├── datalist/               ← generated train/val/test split files
│   ├── checkpoints/            ← saved model weights
│   └── results/                ← experiment outputs (metrics, figures)
├── src/
│   ├── data/
│   │   ├── parse_annotations.py   ← XML parser
│   │   ├── split.py               ← stratified train/val/test split
│   │   ├── dataset.py             ← PyTorch Dataset + transforms
│   │   ├── corruptions.py         ← H1 imaging deviation functions
│   │   ├── prepare_data.py        ← top-level data prep script
│   │   └── sanity_check.py        ← visual + statistical checks
│   ├── models/
│   │   └── train.py               ← ResNet-50 training script
│   └── experiments/
│       ├── baseline_curve.py      ← compute clean yield-threshold curve
│       ├── h1_deviations.py       ← H1: curves under imaging deviations
│       ├── h2_detection.py        ← H2: AUC-YT + Wasserstein detection
│       └── h3_correction.py       ← H3: BN re-estimation + TENT
├── notebooks/
│   └── results_visualisation.ipynb
├── requirements.txt
└── README.md                   ← this file
```

---

## Requirements

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.0
torchvision>=0.15
timm>=0.9
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
matplotlib>=3.7
Pillow>=10.0
tqdm>=4.65
pandas>=2.0
```

---

## Reproducing All Results

### Step 1 — Download the data

Download the **GrainSet wheat** dataset from Figshare:

```bash
# Wheat (200K images, ~6 GB)
wget -O data/wheat.zip https://figshare.com/ndownloader/files/40902838

# Optional: Sorghum (102K images, ~3 GB) for generalisation experiments
wget -O data/sorghum.zip https://figshare.com/ndownloader/files/40902923

cd data && unzip wheat.zip && unzip sorghum.zip
```

After unzipping, the structure should be:
```
data/
  wheat/
    wheat.xml
    <ID>_UP.png
    <ID>_DOWN.png
    ...
```

### Step 2 — Prepare the data pipeline

```bash
python src/data/prepare_data.py \
    --xml   data/wheat/wheat.xml \
    --root  data/wheat/ \
    --out   runs/datalist/wheat

# Verify with sanity checks
python src/data/sanity_check.py \
    --datalist runs/datalist/wheat \
    --out      runs/sanity_check/wheat
```

### Step 3 — Train the model

```bash
python src/models/train.py \
    --datalist  runs/datalist/wheat \
    --out       runs/checkpoints/wheat_resnet50 \
    --epochs    50 \
    --batch-size 128
```

### Step 4 — Baseline yield-threshold curve

```bash
python src/experiments/baseline_curve.py \
    --datalist   runs/datalist/wheat \
    --checkpoint runs/checkpoints/wheat_resnet50/best.pth \
    --out        runs/results/wheat
```

### Step 5 — H1: Imaging deviations

```bash
python src/experiments/h1_deviations.py \
    --datalist   runs/datalist/wheat \
    --checkpoint runs/checkpoints/wheat_resnet50/best.pth \
    --out        runs/results/wheat/h1
```

### Step 6 — H2: Detection

```bash
python src/experiments/h2_detection.py \
    --results-dir runs/results/wheat/h1 \
    --out         runs/results/wheat/h2
```

### Step 7 — H3: Correction

```bash
python src/experiments/h3_correction.py \
    --datalist   runs/datalist/wheat \
    --checkpoint runs/checkpoints/wheat_resnet50/best.pth \
    --out        runs/results/wheat/h3
```

---

## Code Structure Notes

- All scripts accept `--help` for full argument documentation.
- Results are saved as `.npz` files (arrays) and `.csv` files (tables).
- Figures are saved as `.pdf` for inclusion in the LaTeX report.
- All random seeds are fixed to 42 for reproducibility.
- The test split is **never used during training or model selection**.

## Development Notes

The source code in this repository was developed with assistance from 
Claude (Anthropic), an AI assistant, used primarily for code generation, 
debugging, and architectural decisions. All experimental results, 
analysis, and written conclusions are the author's own.
