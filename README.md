# Recidivism Risk Prediction — Hybrid Neural Network

Research codebase for the master's dissertation:

> **«Development of a Hybrid Neural Network Algorithm for Predicting the Risk of Recidivism in the Criminal Sphere»**
> Viktor Kossinov · Astana IT University · 2025–2027
> Supervisor: Dr. A. K. Zhumadillayeva

[![CI](https://github.com/vkossinov/recidivism-hybrid/actions/workflows/ci.yml/badge.svg)](https://github.com/vkossinov/recidivism-hybrid/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

---

## Overview

This repository implements a hybrid two-branch neural network for post-release recidivism risk prediction. The model:

- Separates **static** offender features (demographics, criminal history) and **dynamic** supervision-period features (employment, drug tests, violations) into dedicated MLP branches.
- Fuses them via a learnable **gated fusion** mechanism.
- Predicts **year-by-year arrest hazards** using a discrete-time survival GRU head — yielding both the 3-year risk and a temporal profile.
- Includes a full **fairness audit** (equalized-odds post-processing) and **explainability** (SHAP, permutation importance).

**Best result on the official NIJ test split:** Ensemble (hybrid + XGBoost) — AUC 0.820, Brier 0.168, ECE 0.016.

---

## Repository Structure

```
src/recidivism/       # Main Python package
  data.py             # NIJ / COMPAS loading and preprocessing
  metrics.py          # AUC, F1, Brier, ECE, fairness gaps
  baselines.py        # Logistic Regression, Random Forest, XGBoost
  hybrid.py           # HybridSurvivalNet + MLPNet + survival loss
  train.py            # Training loops with early stopping
  fairness.py         # Equalized-odds post-processing (fairlearn)
  explain.py          # SHAP (trees) and permutation importance (neural)
scripts/
  run_experiments.py  # Quick baseline run → results/
  run_full_study.py   # Full study: tuning, ablation, ensemble, fairness, figures
notebooks/
  01_eda_nij.ipynb
  02_baselines_nij.ipynb
  03_hybrid_model.ipynb
  04_fairness_compas.ipynb
tests/
  test_metrics.py     # Unit tests for metrics module
  test_hybrid.py      # Unit tests for model architecture
results/              # Metrics CSVs, figures (generated — not committed)
data/raw/             # Raw datasets (not committed — see Data section)
```

---

## Data

| Dataset | Source | Records | Notes |
|---|---|---|---|
| NIJ Recidivism Challenge | [data.ojp.usdoj.gov](https://data.ojp.usdoj.gov/Courts/NIJ-s-Recidivism-Challenge-Full-Dataset/ynf5-u8nk) | 25,835 | Primary — place at `data/raw/nij_recidivism_full.csv` |
| COMPAS (ProPublica) | [GitHub](https://github.com/propublica/compas-analysis) | 6,172 | Fairness validation — place at `data/raw/compas-scores-two-years.csv` |

Raw data files are **not committed** (`.gitignore`) due to size and redistribution constraints.

---

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# Run full study (requires both datasets in data/raw/)
.venv\Scripts\python scripts\run_full_study.py
```

---

## Running Tests

```powershell
pip install pytest pytest-cov flake8
pytest tests/ -v --cov=src/recidivism
```

Tests use only synthetic data — no dataset required.

---

## CI/CD

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs automatically on every push and pull request:

1. **Environment setup** — Python 3.12, pinned dependencies from `requirements.txt`
2. **Flake8 linting** — PEP 8 compliance check on `src/` and `tests/`
3. **pytest** — unit tests with coverage report (≥ 60% required)
4. **Import gate** — verifies all public module imports resolve cleanly

---

## Key Results

| Model | AUC | F1 | Brier | ECE |
|---|---|---|---|---|
| **Ensemble (hybrid + XGBoost)** | **0.820** | **0.795** | **0.168** | **0.016** |
| XGBoost | 0.820 | 0.791 | 0.169 | 0.020 |
| HybridSurvivalNet (ours) | 0.808 | 0.791 | 0.174 | 0.029 |
| Random Forest | 0.805 | 0.791 | 0.178 | 0.052 |
| MLP | 0.803 | 0.781 | 0.177 | 0.021 |
| Logistic Regression | 0.790 | 0.778 | 0.181 | 0.015 |

**Ablation:** MLP 0.803 → +survival head 0.805 → +two branches 0.807 → +gated fusion 0.808

**Fairness (hybrid, equalized-odds gap):** Gender 0.194 → 0.064 | Race 0.076 → 0.029 | Accuracy cost ≈ 1.4 p.p.

---

## Technology Stack

| Component | Technology | Justification |
|---|---|---|
| Language | Python 3.12 | Ecosystem standard for ML/AI |
| Deep learning | PyTorch 2.4 | Dynamic graphs, GRU support, GPU |
| Gradient boosting | XGBoost 2.1 | State-of-the-art tabular baseline |
| Explainability | SHAP 0.46 | TreeSHAP + permutation importance |
| Fairness | fairlearn 0.11 | ThresholdOptimizer (equalized odds) |
| VCS | Git + GitHub | Distributed versioning, CI/CD |
| Testing | pytest + flake8 | Automated quality gates |

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

If you use this code, please cite:

```
Kossinov, V., & Zhumadillayeva, A. K. (2026).
Application of Hybrid Neural Network Architectures for Predicting Recidivism
with Explainable Outcomes. [Manuscript in preparation].
Astana IT University.
```
