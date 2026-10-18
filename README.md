# PMMA/PVPh-ZrO2 Hybrid Dielectrics: GPR-based Surrogate Models and Optimal Zone

This repository contains the Python codes used in the paper
*"PMMA/PVPh-ZrO2 Hybrid Dielectrics for MIM Capacitors"* to train the
Gaussian Process Regressions urrogate models of capacitance density
and leakage current density across the PMMA/PVPh-ZrO2 ternary space, and
to identify the optimal compositional zone through an entropy- and
uncertainty-weighted score.

The scripts reproduce the workflow illustrated in Figure 2 (Pipeline) of
the manuscript and the zonal validation reported in the Supporting
Information.

## Workflow

```
   SAMPLING          -> DATA PROCESSING (ILR + log scale) -> GPR (Matern 3/2 + white noise)
                                                                  |
                                                                  v
                                             CAPACITANCE DENSITY & LEAKAGE CURRENT MAPS
                                                                  |
                                                                  v
                              Min-Max + SoftMax -> Shannon Entropy (H)
                                                   + Coefficient of Variation (CV)
                                                   -> OPTIMAL ZONE  S = H * (1 - CV)
```

## Repository layout

| File                                      | Purpose |
|-------------------------------------------|---------|
| `01_capacitance_density_gpr.py`           | Trains the GPR surrogate for capacitance density. Exports the mean and sigma ternary maps. |
| `02_leakage_current_gpr.py`               | Trains the GPR surrogate for -log10(leakage current). Exports the mean and sigma maps and compares the predicted CV with the experimental CV computed from replicates. |
| `03_optimal_zone_softmax.py`              | Combines both surrogate maps via softmax-normalized Shannon entropy and the propagated coefficient of variation to produce the optimality score S = H * (1 - CV). |
| `capacitance_density_experimental.csv`    | Experimental capacitance density dataset (PMMA, ZrO2, PVPh, Capacitance_Density). |
| `leakage_current_experimental.csv`        | Experimental leakage current dataset (PMMA, ZrO2, PVPh, Leakage_Current_Density). |

## Execution order

The scripts are numbered according to their execution order. Scripts 3,
4, and 5 depend on the ternary-grid maps produced by scripts 1 and 2.

```bash
python 01_capacitance_density_gpr.py
python 02_leakage_current_gpr.py
python 03_optimal_zone_softmax.py
```

## Requirements

Python 3.9+ with the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Methodology summary

### ILR transform (Aitchison geometry)

Three-part compositions `(A, B, C)` lie on the 2-simplex and cannot be
treated as Euclidean coordinates. The Isometric Log-Ratio (ILR) transform
maps them to `R^2` while preserving distances and angles in the simplex,
which is essential before fitting a kernel-based regressor.

### Gaussian Process Regression

A GPR with a Matern kernel (nu = 3/2) plus a white-noise term is fitted
on the log-transformed target. Hyperparameters are optimized by
Leave-One-Out Cross-Validation (LOOCV); performance is quantified with
the RMSE on the log scale and both the mean and median relative error on
the original scale.

### Optimal zone

The optimality score is defined as:

```
S = H * (1 - CV)
```

where `H` is the base-2 Shannon entropy of the softmax-normalized
capacitance/leakage pair (high when both objectives are simultaneously
large) and `CV` is the propagated coefficient of variation (small when
the GPR is confident).
