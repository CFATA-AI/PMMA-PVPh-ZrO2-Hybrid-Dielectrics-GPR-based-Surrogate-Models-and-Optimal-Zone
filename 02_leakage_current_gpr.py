"""
Leakage Current Density Surrogate Model via Gaussian Process Regression (GPR)
=============================================================================

This script trains a Gaussian Process Regression (GPR) surrogate model to
predict the leakage current density of PMMA/PVPh-ZrO2 hybrid dielectric
films across the ternary compositional space. The target is modeled on the
-log10(J_leak) scale, which better reflects the orders-of-magnitude span
typical of leakage currents.

Pipeline
--------
1. Load experimental dataset (composition, leakage current density).
2. Map 3-part compositions to 2D Euclidean space via the Isometric
   Log-Ratio (ILR) transform (Aitchison geometry).
3. Standardize ILR features and train a GPR with a Matern (nu = 3/2)
   kernel plus a white-noise term. Hyperparameters are optimized through
   Leave-One-Out Cross-Validation (LOOCV).
4. Evaluate predictive performance (RMSE and relative errors).
5. Predict mean and standard deviation on a dense ternary grid.
6. Compare the model-predicted coefficient of variation against the one
   obtained from experimental replicates (KDE comparison).
7. Export maps and validate on a hold-out test set.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
from ternary_diagram import TernaryDiagram
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

np.set_printoptions(suppress=True)


# =============================================================================
# 1) LOAD EXPERIMENTAL DATA
# =============================================================================
file = pd.read_csv("leakage_current_experimental.csv")

component_names = file.columns[:-1]
X = file.iloc[:, :-1].to_numpy()      # Compositions (PMMA, ZrO2, PVPh)
y = file.iloc[:, -1].to_numpy()       # Leakage current density
y_log = -np.log10(y + 1e-12)          # -log10 transform (epsilon avoids log(0))


# =============================================================================
# 2) ILR TRANSFORM (3-PART COMPOSITIONS -> 2D)
# =============================================================================
def ilr_transform(comps):
    """Isometric Log-Ratio transform from the 2-simplex to R^2."""
    A, B, C = comps[:, 0], comps[:, 1], comps[:, 2]
    z1 = np.sqrt(1 / 2) * np.log(A / B)
    z2 = np.sqrt(2 / 3) * np.log(np.sqrt(A * B) / C)
    return np.vstack([z1, z2]).T


def ilr_inverse(z):
    """Inverse ILR transform from R^2 back to the 2-simplex."""
    z1, z2 = z[:, 0], z[:, 1]
    A = np.exp( np.sqrt(1 / 2) * z1 + np.sqrt(1 / 6) * z2)
    B = np.exp(-np.sqrt(1 / 2) * z1 + np.sqrt(1 / 6) * z2)
    C = np.exp(-np.sqrt(2 / 3) * z2)
    S = A + B + C
    return np.vstack([A / S, B / S, C / S]).T


Z = ilr_transform(X)
scaler = StandardScaler()


# =============================================================================
# 3) GPR TRAINING WITH LEAVE-ONE-OUT CROSS-VALIDATION
# =============================================================================
loocv = LeaveOneOut()

best_error = float("inf")
best_error_history = [best_error]
n_iterations = 1  # Repeat the LOOCV procedure to pick the most consistent model

for i in range(n_iterations):

    print(f"\nIteration {i + 1} of {n_iterations}")

    best_error_fold = float("inf")
    y_pred_test_list = []
    y_real_test_list = []

    # Matern (nu = 3/2) + white noise. Bounds control the search space
    # of the hyperparameter optimizer (L-BFGS-B).
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1e1))
    )

    for train_idx, test_idx in loocv.split(Z):
        X_train, X_test = Z[train_idx], Z[test_idx]
        y_train, y_test = y_log[train_idx], y_log[test_idx]

        # Fit scaler on the training fold to avoid data leakage.
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=20,
            optimizer="fmin_l_bfgs_b",
            alpha=0.01,
        )
        gpr.fit(X_train, y_train)

        y_pred_test = gpr.predict(X_test)
        y_pred_test_list.append(y_pred_test[0])
        y_real_test_list.append(y_test[0])

        error_fold = 100 * abs((y_pred_test - y_test) / y_test)[0]

        if error_fold < best_error_fold:
            best_error_fold = error_fold
            best_model_fold = gpr
            best_kernel_fold = best_model_fold.kernel_

    error_rel = round(np.sqrt(mean_squared_error(y_pred_test_list, y_real_test_list)), 7)
    print(f"Iteration {i} RMSE (-log scale) = {error_rel}")

    if error_rel < best_error:
        best_error = error_rel
        best_error_history.append(best_error)
        best_model = best_model_fold
        best_kernel = best_kernel_fold
        print(f" === New best model: RMSE (-log) = {error_rel} ===")

    # Early stopping if improvement is negligible.
    if best_error_history[-1] / best_error_history[-2] >= 0.90:
        print("Early stopping triggered")
        break


# =============================================================================
# 4) LOOCV METRICS REPORT -- Leakage current density (J_leak)
# =============================================================================
y_pred_log_arr = np.asarray(y_pred_test_list)   # -log10(J)
y_real_log_arr = np.asarray(y_real_test_list)   # -log10(J)

rmse_log_J = np.sqrt(mean_squared_error(y_real_log_arr, y_pred_log_arr))
eps_rel_log_J = 100.0 * np.abs(y_pred_log_arr - y_real_log_arr) / np.abs(y_real_log_arr)

# Back-projection to original scale: J = 10 ** (-y_log)
y_pred_orig = 10 ** (-y_pred_log_arr)
y_real_orig = 10 ** (-y_real_log_arr)
eps_rel_orig_J = 100.0 * np.abs(y_pred_orig - y_real_orig) / y_real_orig

print("\n===================================================")
print("   LOOCV REPORT - Leakage current density (J_leak)")
print("===================================================")
print(f"RMSE_log                     = {rmse_log_J:.4f}")
print(f"Mean   eps_rel (log scale)   = {eps_rel_log_J.mean():.3f} %")
print(f"Median eps_rel (log scale)   = {np.median(eps_rel_log_J):.3f} %")
print(f"Mean   eps_rel (orig. scale) = {eps_rel_orig_J.mean():.3f} %")
print(f"Median eps_rel (orig. scale) = {np.median(eps_rel_orig_J):.3f} %")
print(f"Best kernel                  = {best_kernel}")
print("===================================================\n")


# =============================================================================
# 5) BUILD TERNARY GRID AND PREDICT
# =============================================================================
resolution = 100
grid = []
for i in range(resolution + 1):
    for j in range(resolution + 1 - i):
        k = resolution - i - j
        grid.append((i / resolution, j / resolution, k / resolution))

# Clip the grid away from the simplex boundary (ILR is not defined at 0).
eps = 1e-12
grid = np.array(grid)
grid = np.clip(grid, eps, 1)
grid = grid / grid.sum(axis=1, keepdims=True)

Zgrid = ilr_transform(grid)

# Refit the scaler on the full training set and transform the prediction grid.
scaler.fit(Z)
Z_scaled = scaler.transform(Z)
Zgrid_scaled = scaler.transform(Zgrid)

final_model = GaussianProcessRegressor(kernel=best_kernel, normalize_y=True, alpha=0.01)
final_model.fit(Z_scaled, y_log)
y_pred_log, y_sigma_log = final_model.predict(Zgrid_scaled, return_std=True)


# =============================================================================
# 6) TERNARY CONTOUR -- MEAN PREDICTION (-log10 scale)
# =============================================================================
fig, ax = plt.subplots(facecolor="w")
td = TernaryDiagram(component_names, ax=ax)
mappable = td.contour(
    vector=grid, z=y_pred_log, cmap="nipy_spectral",
    flag_cbar=False, vmin=y_pred_log.min(), vmax=y_pred_log.max(),
)
td.scatter(vector=X, marker="o", c="#FFFFFF", s=30, edgecolors="#000000FF")

cb = td.colorbar(mappable=mappable, label=r"$-\log(J_{\mathrm{leak}})$")
cb.formatter = ticker.ScalarFormatter(useMathText=True)
cb.formatter.set_scientific(True)
cb.formatter.set_powerlimits((0, 0))
cb.update_ticks()
plt.savefig("leakage_current.png", dpi=300, bbox_inches="tight")
plt.show()


# =============================================================================
# 7) TERNARY CONTOUR -- STANDARD DEVIATION (-log10 scale)
# =============================================================================
fig, ax = plt.subplots(facecolor="w")
td = TernaryDiagram(component_names, ax=ax)
mappable = td.contour(
    vector=grid, z=y_sigma_log, cmap="nipy_spectral",
    flag_cbar=False, vmin=y_sigma_log.min(), vmax=y_sigma_log.max(),
)
td.scatter(vector=X, marker="o", c="#FFFFFF", s=30, edgecolors="#000000FF")

cb = td.colorbar(mappable=mappable, label=r"$\sigma_{-\log J_{\mathrm{leak}}}$")
cb.formatter = ticker.ScalarFormatter(useMathText=True)
cb.formatter.set_scientific(True)
cb.formatter.set_powerlimits((0, 0))
cb.update_ticks()
plt.show()


# =============================================================================
# 8) TERNARY CONTOUR -- COEFFICIENT OF VARIATION
# =============================================================================
cv_values = 100 * y_sigma_log / np.abs(y_pred_log + 1e-12)

fig, ax = plt.subplots(facecolor="w")
td = TernaryDiagram(component_names, ax=ax)
mappable = td.contour(
    vector=grid, z=cv_values, cmap="nipy_spectral",
    flag_cbar=False, vmin=cv_values.min(), vmax=cv_values.max(),
)
td.scatter(vector=X, marker="o", c="#FFFFFF", s=30, edgecolors="#000000FF")
td.colorbar(mappable=mappable, label="Coefficient of Variation [%]")
plt.show()


# =============================================================================
# 9) PER-DATAPOINT RELATIVE ERROR (BEST MODEL)
# =============================================================================
Z_eval_scaled = scaler.transform(ilr_transform(X))

data_indices = []
data_errors = []
for train_idx, test_idx in loocv.split(Z_eval_scaled):
    X_test = Z_eval_scaled[test_idx]
    y_test_log = y_log[test_idx]

    y_pred_log_i, _ = final_model.predict(X_test, return_std=True)
    y_pred_val = y_pred_log_i[0]
    y_real_val = y_test_log[0]

    error_rel = 100 * abs((y_pred_val - y_real_val) / y_real_val)
    data_indices.append(test_idx[0] + 1)
    data_errors.append(error_rel)

data_indices = np.array(data_indices).flatten()
data_errors = np.array(data_errors).flatten()

df_errors = pd.DataFrame({"data_index": data_indices, "rel_error": data_errors})
summary = df_errors.groupby(["data_index"]).median().reset_index()

plt.figure()
plt.xticks(np.arange(1, 1 + file.shape[0]))
plt.grid()
plt.plot(summary.iloc[:, 0], summary.iloc[:, 1], linewidth=0.7)
plt.scatter(data_indices, data_errors, alpha=0.2,
            edgecolors="darkblue", linewidths=0.5, s=100)
plt.xlabel("ith-data")
plt.ylabel("Relative error [%]")
plt.title("Relative error in best model")
plt.show()

print(f"\nFinal mean relative error of the best model: {round(data_errors.mean(), 3)} %")


# =============================================================================
# 10) RELATIVE-ERROR HISTOGRAM
# =============================================================================
plt.rcParams.update({"font.size": 17})
plt.figure(figsize=(14, 7))
n, bins, patches = plt.hist(data_errors, bins=25, edgecolor="white",
                            linewidth=0.5, alpha=1.0)
n = n.astype("int")

low, high = 0.25, 0.75
for i in range(len(patches)):
    fraction = n[i] / max(n)
    rescaled_color = low + (fraction * (high - low))
    patches[i].set_facecolor(plt.cm.bone_r(rescaled_color))

plt.xlabel("Relative error distribution [%]")
plt.ylabel("Frequency [a.u.]")
plt.show()


# =============================================================================
# 11) EXPORT PREDICTION GRID (MEAN AND SIGMA)
# =============================================================================
y_pred_grid = y_pred_log.reshape(-1, 1)
pd.DataFrame(np.hstack((grid, y_pred_grid))).to_csv(
    "leakage_current_gpr_mean.csv", index=False
)

y_sigma_grid = y_sigma_log.reshape(-1, 1)
pd.DataFrame(np.hstack((grid, y_sigma_grid))).to_csv(
    "leakage_current_gpr_sigma.csv", index=False
)


# =============================================================================
# 12) EXPERIMENTAL CV VS GPR CV (KDE COMPARISON)
# =============================================================================
# Experimental CV from replicates (grouped by composition).
# The target column is read by name so the CSV header must match.
target_col = file.columns[-1]  # e.g. "Leakage_Current_Density"

file_log = file.copy()
file_log[target_col] = -np.log10(file_log[target_col] + 1e-12)

df_stats = file_log.groupby(
    ["PMMA", "ZrO2", "PVPh"], as_index=False
)[target_col].agg(["mean", "std"])
df_stats["CV_exp"] = 100 * (df_stats["std"] / df_stats["mean"])
cv_experimental = df_stats["CV_exp"].dropna()  # Drop points without replicates

# GPR CV at the experimental compositions.
X_unique = df_stats[["PMMA", "ZrO2", "PVPh"]].to_numpy()
Z_unique_scaled = scaler.transform(ilr_transform(X_unique))
y_pred_mu, y_pred_std = final_model.predict(Z_unique_scaled, return_std=True)
cv_gpr_at_points = 100 * y_pred_std / np.abs(y_pred_mu + 1e-12)

# GPR CV on the full prediction grid (global uncertainty).
cv_gpr_grid = 100 * y_sigma_log / np.abs(y_pred_log + 1e-12)

kde_exp = gaussian_kde(cv_experimental)
kde_gpr_local = gaussian_kde(cv_gpr_at_points)
kde_gpr_global = gaussian_kde(cv_gpr_grid)

x_range = np.linspace(0, 50, 1000)

plt.figure(figsize=(10, 6))
plt.plot(x_range, kde_exp(x_range), label="Experimental (replicates)",
         color="blue", lw=2)
plt.plot(x_range, kde_gpr_local(x_range), label="GPR (at experimental points)",
         color="green", linestyle="--", lw=2)
plt.plot(x_range, kde_gpr_global(x_range), label="GPR (whole diagram)",
         color="red", alpha=0.5)
plt.fill_between(x_range, kde_exp(x_range), color="blue", alpha=0.1)
plt.xlabel("Coefficient of Variation [%]")
plt.ylabel("Probability Density")
plt.title("Model-uncertainty validation")
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, 50)
plt.show()


# =============================================================================
# 13) VALIDATION ON HOLD-OUT TEST SET
# =============================================================================
# New compositions and their experimental leakage current. Compositions
# must sum to 1.0 (2-simplex).
data_test = {
    "PMMA":            [0.435, 0.47, 0.1, 0.4],
    "ZrO2":            [0.372, 0.37, 0.3, 0.5],
    "PVPh":            [0.100, 0.15, 0.6, 0.1],
    "Real_Value_Orig": [1.11e-6, 3.12e-7, 4.01e-8, 1.20e-6],  # Experimental J_leak
}

df_new = pd.DataFrame(data_test)
X_new = df_new[["PMMA", "ZrO2", "PVPh"]].to_numpy()

# Transform experimental values to the model's target scale (-log10).
y_new_real_log = -np.log10(df_new["Real_Value_Orig"].to_numpy() + 1e-12)

# ILR transform and scale using the training-set scaler (no refit).
Z_new = ilr_transform(X_new)
Z_new_scaled = scaler.transform(Z_new)

y_new_pred_log, y_new_sigma_log = final_model.predict(Z_new_scaled, return_std=True)

# Relative error is evaluated on the -log10 scale, which is the natural
# scale for a quantity that spans several orders of magnitude.
error_rel_new = 100 * np.abs((y_new_pred_log - y_new_real_log) / y_new_real_log)

results_test = pd.DataFrame({
    "Real (-log10)":      np.round(y_new_real_log, 3),
    "Predicted (-log10)": np.round(y_new_pred_log, 3),
    "Std Dev":            np.round(y_new_sigma_log, 3),
    "Rel Err [%]":        np.round(error_rel_new, 2),
})

print("\n--- HOLD-OUT TEST SET RESULTS ---")
print(results_test)
print(f"\nMean relative error on test set (log scale): {round(error_rel_new.mean(), 3)} %")
