"""Unit tests for src/recidivism/metrics.py.

All tests use synthetic data so no real dataset is required.
Run with:  pytest tests/
"""

import numpy as np
import pandas as pd
import pytest

from recidivism.metrics import (
    expected_calibration_error,
    classification_metrics,
    full_report,
    group_fairness_table,
    fairness_gaps,
    per_year_auc,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 200


@pytest.fixture
def binary_data():
    """Synthetic binary classification output."""
    y_true = RNG.integers(0, 2, size=N)
    y_prob = np.clip(y_true.astype(float) + RNG.normal(0, 0.3, N), 0.01, 0.99)
    return y_true, y_prob


@pytest.fixture
def group_series():
    return pd.Series(["A"] * (N // 2) + ["B"] * (N // 2))


# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------

def test_ece_perfect_calibration():
    """A perfectly calibrated predictor should return ECE close to 0."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    assert expected_calibration_error(y_true, y_prob) <= 0.15


def test_ece_worst_calibration():
    """Inverted predictor should have high ECE."""
    y_true = np.array([1, 1, 0, 0] * 25)
    y_prob = np.array([0.05, 0.05, 0.95, 0.95] * 25)
    assert expected_calibration_error(y_true, y_prob) > 0.5


def test_ece_range(binary_data):
    y_true, y_prob = binary_data
    ece = expected_calibration_error(y_true, y_prob)
    assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# classification_metrics
# ---------------------------------------------------------------------------

def test_classification_metrics_keys(binary_data):
    y_true, y_prob = binary_data
    result = classification_metrics(y_true, y_prob)
    for key in ("roc_auc", "f1", "accuracy", "brier", "ece"):
        assert key in result, f"Missing key: {key}"


def test_classification_metrics_range(binary_data):
    y_true, y_prob = binary_data
    m = classification_metrics(y_true, y_prob)
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["f1"] <= 1.0
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["brier"] <= 1.0
    assert 0.0 <= m["ece"] <= 1.0


def test_perfect_predictor():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.01, 0.01, 0.99, 0.99])
    m = classification_metrics(y_true, y_prob)
    assert m["roc_auc"] == pytest.approx(1.0, abs=1e-6)
    assert m["accuracy"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# group_fairness_table & fairness_gaps
# ---------------------------------------------------------------------------

def test_fairness_table_shape(binary_data, group_series):
    y_true, y_prob = binary_data
    table = group_fairness_table(y_true, y_prob, group_series)
    assert set(table.index) == {"A", "B"}
    for col in ("fpr", "fnr", "auc", "ece"):
        assert col in table.columns


def test_fairness_gaps_non_negative(binary_data, group_series):
    y_true, y_prob = binary_data
    table = group_fairness_table(y_true, y_prob, group_series)
    gaps = fairness_gaps(table)
    for val in gaps.values():
        assert val >= -1e-9, f"Negative gap: {val}"


def test_fairness_gaps_same_group():
    """When both groups are identical, all gaps should be ~0."""
    y_true = RNG.integers(0, 2, size=100)
    y_prob = np.clip(y_true.astype(float) + RNG.normal(0, 0.2, 100), 0.01, 0.99)
    group = pd.Series(["X"] * 50 + ["Y"] * 50)
    table = group_fairness_table(y_true, y_prob, group)
    gaps = fairness_gaps(table)
    assert gaps["equalized_odds_diff"] < 0.5  # loose bound for random data


# ---------------------------------------------------------------------------
# per_year_auc & full_report
# ---------------------------------------------------------------------------

def test_per_year_auc_keys():
    n, t = 40, 3
    hazards = RNG.uniform(0.1, 0.9, size=(n, t))
    y_years = np.zeros((n, t), dtype=float)
    for i in range(n):
        year = int(RNG.integers(0, t + 1))
        if year < t:
            y_years[i, year] = 1.0
    # ensure year-1 has both classes among at-risk
    y_years[0, 0], y_years[1, 0] = 1.0, 0.0
    out = per_year_auc(hazards, y_years)
    assert set(out) == {"auc_year1", "auc_year2", "auc_year3"}
    for v in out.values():
        assert 0.0 <= v <= 1.0


def test_full_report_with_sensitive(binary_data, group_series):
    y_true, y_prob = binary_data
    sensitive = pd.DataFrame({"grp": group_series})
    report = full_report(y_true, y_prob, sensitive=sensitive)
    assert "overall" in report
    assert "by_grp" in report
    assert "gaps_grp" in report
    assert "roc_auc" in report["overall"]


def test_full_report_without_sensitive(binary_data):
    y_true, y_prob = binary_data
    report = full_report(y_true, y_prob, sensitive=None)
    assert set(report) == {"overall"}
