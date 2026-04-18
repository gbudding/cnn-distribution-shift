from scipy.stats import spearmanr
import numpy as np

severities = np.array([1, 2, 3, 4, 5])

# Fill in the |ΔAUC-YT| at severities 1..5 for each deviation type.
# (Absolute values — they're already positive for most, but wrap in abs() to be safe.)
results = {
    "noise":      [0.016, ____, 0.187, ____, 0.423],
    "blur":       [0.001, ____, 0.079, ____, 0.136],
    "brightness": [0.013, ____, 0.002, ____, 0.085],
}

for name, deltas in results.items():
    rho, p = spearmanr(severities, np.abs(deltas))
    print(f"{name}: rho={rho:.3f}, p={p:.4f}")