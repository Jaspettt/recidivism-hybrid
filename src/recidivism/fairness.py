"""Bias mitigation: equalized-odds post-processing for arbitrary models.

Wraps any probability function (sklearn pipelines, the hybrid torch model)
into an sklearn-compatible estimator so that fairlearn's ``ThresholdOptimizer``
(Hardt et al., 2016) can be applied uniformly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin

from .metrics import fairness_gaps, group_fairness_table


class PrefitProbaEstimator(BaseEstimator, ClassifierMixin):
    """sklearn-compatible shell around a ``predict_proba_1d``-style callable.

    Also forces float64 output: fairlearn 0.14 fails on float32 probabilities
    (e.g. XGBoost) under pandas 3.x.
    """

    def __init__(self, predict_proba_1d):
        self.predict_proba_1d = predict_proba_1d
        self.classes_ = np.array([0, 1])

    def fit(self, *args, **kwargs):
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = np.asarray(self.predict_proba_1d(X), dtype=np.float64)
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __sklearn_is_fitted__(self) -> bool:
        return True


def mitigate_equalized_odds(
    predict_proba_1d,
    X_train, y_train, sensitive_train,
    X_test, sensitive_test,
    seed: int = 42,
) -> np.ndarray:
    """Fit group-specific thresholds on train, return mitigated 0/1 test predictions."""
    optimizer = ThresholdOptimizer(
        estimator=PrefitProbaEstimator(predict_proba_1d),
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    optimizer.fit(X_train, y_train, sensitive_features=sensitive_train)
    return optimizer.predict(X_test, sensitive_features=sensitive_test, random_state=seed)


def mitigation_report(
    predict_proba_1d,
    X_train, y_train, sensitive_train,
    X_test, y_test, sensitive_test,
    threshold: float = 0.5,
    seed: int = 42,
) -> dict:
    """Fairness tables and gaps before/after equalized-odds post-processing."""
    p_test = np.asarray(predict_proba_1d(X_test), dtype=np.float64)
    before = group_fairness_table(y_test, p_test, sensitive_test, threshold)

    y_fair = mitigate_equalized_odds(
        predict_proba_1d, X_train, y_train, sensitive_train, X_test, sensitive_test, seed)
    after = group_fairness_table(y_test, y_fair.astype(float), sensitive_test)

    acc_before = float(((p_test >= threshold).astype(int) == y_test).mean())
    acc_after = float((y_fair == y_test).mean())
    return {
        "table_before": before,
        "table_after": after,
        "gaps_before": fairness_gaps(before),
        "gaps_after": fairness_gaps(after),
        "accuracy_before": acc_before,
        "accuracy_after": acc_after,
    }
