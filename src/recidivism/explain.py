"""Explainability utilities (XAI): SHAP for tree models,
permutation importance for arbitrary predictors including the hybrid network."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def shap_tree_values(model, X: pd.DataFrame, max_samples: int = 2000, seed: int = 42):
    """TreeSHAP values for XGBoost / random forest.

    Returns ``(explainer, shap_values, X_sample)``.
    """
    import shap

    rng = np.random.default_rng(seed)
    if len(X) > max_samples:
        X = X.iloc[rng.choice(len(X), max_samples, replace=False)]
    explainer = shap.TreeExplainer(model)
    return explainer, explainer(X), X


def permutation_importance(
    predict_proba_1d,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = 5,
    max_samples: int = 4000,
    seed: int = 42,
) -> pd.DataFrame:
    """Model-agnostic importance: drop in AUC when a feature is shuffled."""
    rng = np.random.default_rng(seed)
    if len(X) > max_samples:
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]

    base_auc = roc_auc_score(y, predict_proba_1d(X))
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[col] = rng.permutation(X_perm[col].to_numpy())
            drops.append(base_auc - roc_auc_score(y, predict_proba_1d(X_perm)))
        rows.append({"feature": col, "auc_drop_mean": np.mean(drops), "auc_drop_std": np.std(drops)})

    return (
        pd.DataFrame(rows)
        .sort_values("auc_drop_mean", ascending=False)
        .reset_index(drop=True)
    )
