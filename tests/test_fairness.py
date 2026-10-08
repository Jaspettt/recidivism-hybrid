"""Unit tests for src/recidivism/fairness.py."""

import numpy as np
import pandas as pd

from recidivism.fairness import (
    PrefitProbaEstimator,
    mitigate_equalized_odds,
    mitigation_report,
)


RNG = np.random.default_rng(0)


def _synthetic_binary(n: int = 80):
    X = RNG.normal(size=(n, 3))
    y = (X[:, 0] + RNG.normal(scale=0.5, size=n) > 0).astype(int)
    sensitive = pd.Series(np.where(X[:, 1] > 0, "A", "B"))

    # Simple score correlated with label
    def predict_proba_1d(x):
        x = np.asarray(x)
        return 1 / (1 + np.exp(-x[:, 0]))

    return X, y, sensitive, predict_proba_1d


def test_prefit_proba_estimator():
    X, y, _, predict_proba_1d = _synthetic_binary()
    est = PrefitProbaEstimator(predict_proba_1d)
    assert est.__sklearn_is_fitted__()
    assert est.fit(X, y) is est
    proba = est.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    pred = est.predict(X)
    assert set(np.unique(pred)).issubset({0, 1})


def test_mitigate_equalized_odds_shape():
    X, y, sensitive, predict_proba_1d = _synthetic_binary()
    # fairlearn needs both classes in each group; ensure that
    mid = len(X) // 2
    X_tr, X_te = X[:mid], X[mid:]
    y_tr = y[:mid]
    s_tr, s_te = sensitive[:mid], sensitive[mid:]
    y_hat = mitigate_equalized_odds(
        predict_proba_1d, X_tr, y_tr, s_tr, X_te, s_te, seed=0)
    assert y_hat.shape == (len(X_te),)
    assert set(np.unique(y_hat)).issubset({0, 1})


def test_mitigation_report_keys():
    X, y, sensitive, predict_proba_1d = _synthetic_binary(n=100)
    mid = 60
    report = mitigation_report(
        predict_proba_1d,
        X[:mid], y[:mid], sensitive[:mid],
        X[mid:], y[mid:], sensitive[mid:],
        seed=0,
    )
    for key in (
        "table_before", "table_after",
        "gaps_before", "gaps_after",
        "accuracy_before", "accuracy_after",
    ):
        assert key in report
    assert 0.0 <= report["accuracy_before"] <= 1.0
    assert 0.0 <= report["accuracy_after"] <= 1.0
