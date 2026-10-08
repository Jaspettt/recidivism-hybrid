"""Unit tests for src/recidivism/train.py with tiny synthetic data."""

import numpy as np
import pandas as pd

from recidivism.train import train_hybrid, train_mlp


RNG = np.random.default_rng(2)


def _mlp_data(n: int = 60, p: int = 6):
    X = pd.DataFrame(RNG.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    y = (X["x0"] + RNG.normal(scale=0.4, size=n) > 0).astype(int)
    # Ensure both classes present for stratify
    y[0], y[1] = 0, 1
    return X, y


def _hybrid_data(n: int = 60):
    static_cols = [f"s{i}" for i in range(5)]
    dynamic_cols = [f"d{i}" for i in range(3)]
    X = pd.DataFrame(
        RNG.normal(size=(n, len(static_cols) + len(dynamic_cols))),
        columns=static_cols + dynamic_cols,
    )
    y_years = np.zeros((n, 3), dtype=float)
    for i in range(n):
        t = int(RNG.integers(0, 4))
        if t < 3:
            y_years[i, t] = 1.0
    # guarantee both event / no-event for stratification
    y_years[0] = [1, 0, 0]
    y_years[1] = [0, 0, 0]
    return X, y_years, static_cols, dynamic_cols


def test_train_mlp_predict():
    X, y = _mlp_data()
    wrapper, info = train_mlp(
        X, y,
        epochs=3, batch_size=16, patience=2,
        val_size=0.25, verbose=False, seed=0,
    )
    proba = wrapper.predict_proba_1d(X)
    assert proba.shape == (len(X),)
    assert np.isfinite(proba).all()
    assert "best_val_auc" in info
    assert len(info["history"]) >= 1


def test_train_hybrid_predict():
    X, y_years, static_cols, dynamic_cols = _hybrid_data()
    wrapper, info = train_hybrid(
        X, y_years, static_cols, dynamic_cols,
        epochs=3, batch_size=16, patience=2,
        static_hidden=(16, 8), dynamic_hidden=(8, 4), fused_dim=8,
        val_size=0.25, verbose=False, seed=0,
    )
    proba = wrapper.predict_proba_1d(X)
    hazards = wrapper.predict_hazards(X)
    assert proba.shape == (len(X),)
    assert hazards.shape == (len(X), 3)
    assert np.isfinite(proba).all()
    assert np.isfinite(hazards).all()
    assert "best_val_auc" in info


def test_train_hybrid_static_only_and_no_gate():
    X, y_years, static_cols, dynamic_cols = _hybrid_data()
    wrapper, _ = train_hybrid(
        X, y_years, static_cols, dynamic_cols=None,
        epochs=2, batch_size=32, patience=1,
        static_hidden=(8,), fused_dim=8, use_gate=False,
        val_size=0.25, verbose=True, seed=1,
    )
    proba = wrapper.predict_proba_1d(X)
    assert proba.shape == (len(X),)
