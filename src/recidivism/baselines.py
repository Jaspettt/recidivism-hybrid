"""Baseline models: logistic regression, random forest, XGBoost.

These reproduce the approaches dominating the literature
(Tollenaar & van der Heijden 2019; Rajendran & Sundararajan 2021)
and serve as the reference point for the hybrid neural network.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def make_baselines(seed: int = 42) -> dict[str, object]:
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, C=1.0, random_state=seed)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=seed,
        ),
        "xgboost": XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=seed,
        ),
    }
