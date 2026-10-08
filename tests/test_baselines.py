"""Unit tests for src/recidivism/baselines.py."""

from recidivism.baselines import make_baselines


def test_make_baselines_keys():
    models = make_baselines(seed=0)
    assert set(models) == {"logreg", "random_forest", "xgboost"}


def test_make_baselines_fit_predict():
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y = (X[:, 0] > 0).astype(int)
    models = make_baselines(seed=0)
    for name, model in models.items():
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (40, 2), name
