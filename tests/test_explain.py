"""Unit tests for src/recidivism/explain.py."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from recidivism.explain import permutation_importance, shap_tree_values


RNG = np.random.default_rng(1)


def _toy_frame(n: int = 60, p: int = 4):
    X = pd.DataFrame(
        RNG.normal(size=(n, p)),
        columns=[f"f{i}" for i in range(p)],
    )
    y = (X["f0"] + RNG.normal(scale=0.3, size=n) > 0).astype(int)
    return X, y


def test_permutation_importance_shape():
    X, y = _toy_frame()

    def predict_proba_1d(x):
        return 1 / (1 + np.exp(-np.asarray(x)[:, 0]))

    result = permutation_importance(
        predict_proba_1d, X, y, n_repeats=2, max_samples=40, seed=0)
    assert list(result.columns) == ["feature", "auc_drop_mean", "auc_drop_std"]
    assert set(result["feature"]) == set(X.columns)
    # Most important feature should involve f0 for this synthetic score
    assert result.iloc[0]["feature"] == "f0"


def test_permutation_importance_subsamples():
    X, y = _toy_frame(n=50)

    def predict_proba_1d(x):
        return np.clip(np.asarray(x)[:, 0] * 0.1 + 0.5, 0.01, 0.99)

    result = permutation_importance(
        predict_proba_1d, X, y, n_repeats=1, max_samples=20, seed=1)
    assert len(result) == X.shape[1]


def test_shap_tree_values():
    X, y = _toy_frame(n=40)
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    explainer, values, X_sample = shap_tree_values(
        model, X, max_samples=15, seed=0)
    assert explainer is not None
    assert len(X_sample) <= 15
    assert values is not None
