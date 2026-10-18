"""
Optimal Zone Identification via Shannon Entropy and Coefficient of Variation
============================================================================

This script combines the surrogate GPR models for capacitance density and
leakage current density to identify the optimal compositional region of
the PMMA/PVPh-ZrO2 ternary space (see Fig. 2 "OPTIMAL ZONE" of the paper).

The optimality score is defined as:

        S = H * (1 - CV)

where
    H  : Shannon entropy of the softmax-normalized capacitance / leakage
         maps. A high H indicates both objectives are high simultaneously.
    CV : Coefficient of variation of the combined map, computed with error
         propagation. (1 - CV) penalizes regions of large GPR uncertainty.

The maps for the mean and sigma of each target property are expected to be
present in the working directory as produced by scripts
"01_capacitance_density_gpr.py" and "02_leakage_current_gpr.py".
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
from ternary_diagram import TernaryDiagram

np.set_printoptions(suppress=True)


# =============================================================================
# 1) LOAD GPR MEAN AND SIGMA MAPS
# =============================================================================
mean_leak  = pd.read_csv("leakage_current_gpr_mean.csv").iloc[:, 3]
mean_cap   = pd.read_csv("capacitance_density_gpr_mean.csv").iloc[:, 3]
sigma_leak = pd.read_csv("leakage_current_gpr_sigma.csv").iloc[:, 3]
sigma_cap  = pd.read_csv("capacitance_density_gpr_sigma.csv").iloc[:, 3]
grid       = pd.read_csv("leakage_current_gpr_mean.csv").iloc[:, 0:3]


# =============================================================================
# 2) HELPERS
# =============================================================================
def rescale(array, old_min, old_max, new_min, new_max):
    """Linear rescaling from [old_min, old_max] to [new_min, new_max]."""
    return new_min + (array - old_min) * (new_max - new_min) / (old_max - old_min)


def get_softmax_entropy(s1, s2, temp=1.0):
    """
    Shannon entropy (base 2) of the softmax-normalized score pair (s1, s2).

    H is maximized when both objectives are equally dominant on a given
    composition and tends to zero when one strongly dominates the other.
    """
    scores = np.vstack([s1, s2]).T / temp
    probs = softmax(scores, axis=1)
    # Entropy: -sum(p * log(p)); epsilon avoids log(0).
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1) / np.log(2)
    return entropy


# =============================================================================
# 3) MIN-MAX NORMALIZATION OF THE MEAN MAPS
# =============================================================================
mean_leak_norm = rescale(mean_leak, mean_leak.min(), mean_leak.max(), 0, 1)
mean_cap_norm  = rescale(mean_cap,  mean_cap.min(),  mean_cap.max(),  0, 1)


# =============================================================================
# 4) COEFFICIENT OF VARIATION (CV) OF THE COMBINED MAP
# =============================================================================
# Pearson correlation coefficient between the two surrogate predictions.
corr_matrix = np.corrcoef(mean_leak, mean_cap)
r = corr_matrix[0, 1]

# Error-propagation formula for the sum of two correlated variables.
numerator = sigma_leak ** 2 + sigma_cap ** 2 + 2 * r * sigma_leak * sigma_cap
denominator = mean_leak + mean_cap
cv_total = np.sqrt(numerator) / denominator
cv_term = 1 - cv_total  # Reward low-uncertainty regions.


# =============================================================================
# 5) SHANNON ENTROPY OF THE SOFTMAX-NORMALIZED MAPS
# =============================================================================
entropy = get_softmax_entropy(mean_leak_norm, mean_cap_norm, temp=1)


# =============================================================================
# 6) OPTIMALITY SCORE  S = H * (1 - CV)   (%)
# =============================================================================
optimality_score = 100 * entropy * cv_term


# =============================================================================
# 7) TERNARY CONTOUR OF THE OPTIMAL ZONE
# =============================================================================
component_names = ["PMMA", "ZrO2", "PVPh"]

fig, ax = plt.subplots(facecolor="w")
td = TernaryDiagram(component_names, ax=ax)
mappable = td.contour(
    vector=grid, z=optimality_score, cmap="nipy_spectral",
    flag_cbar=False,
    vmin=optimality_score.min(), vmax=optimality_score.max(),
    extend="neither",
)
td.colorbar(mappable=mappable, label="Optimal Zone [%]")
plt.savefig("optimal_zone.png", dpi=300, bbox_inches="tight")
plt.show()
