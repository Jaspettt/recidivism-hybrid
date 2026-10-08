"""Evaluation metrics: discrimination, calibration and algorithmic fairness."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE: weighted average of |mean predicted prob - observed rate| per bin."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """The metric set fixed in the individual work plan: AUC, F1, calibration."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
    }


def per_year_auc(hazards: np.ndarray, y_years: np.ndarray) -> dict[str, float]:
    """Discrimination of per-year hazards in the discrete-time survival setting.

    For year *t* the AUC is computed among individuals still at risk
    (no arrest in years < t), comparing hazard_t against the year-t event.
    """
    out = {}
    cum = np.cumsum(y_years, axis=1)
    for t in range(y_years.shape[1]):
        at_risk = (cum[:, t] - y_years[:, t]) == 0
        out[f"auc_year{t + 1}"] = float(
            roc_auc_score(y_years[at_risk, t], hazards[at_risk, t]))
    return out


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------

def _confusion_rates(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Returns (FPR, FNR)."""
    neg, pos = (y_true == 0), (y_true == 1)
    fpr = float(y_pred[neg].mean()) if neg.sum() else np.nan
    fnr = float(1 - y_pred[pos].mean()) if pos.sum() else np.nan
    return fpr, fnr


def group_fairness_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Per-group metrics for a sensitive attribute (race, gender, ...)."""
    group = pd.Series(np.asarray(group), name="group")
    y_pred = (y_prob >= threshold).astype(int)
    rows = []
    for g in sorted(group.dropna().unique()):
        mask = (group == g).to_numpy()
        fpr, fnr = _confusion_rates(y_true[mask], y_pred[mask])
        rows.append({
            "group": g,
            "n": int(mask.sum()),
            "base_rate": float(y_true[mask].mean()),
            "selection_rate": float(y_pred[mask].mean()),
            "auc": float(roc_auc_score(y_true[mask], y_prob[mask]))
            if len(np.unique(y_true[mask])) > 1 else np.nan,
            "fpr": fpr,
            "fnr": fnr,
            "ece": expected_calibration_error(y_true[mask], y_prob[mask]),
        })
    return pd.DataFrame(rows).set_index("group")


def fairness_gaps(table: pd.DataFrame) -> dict[str, float]:
    """Aggregated gaps between groups (max - min across groups).

    - demographic_parity_diff: selection rate gap
    - equalized_odds_diff: max of FPR gap and FNR gap (Hardt et al., 2016)
    - calibration_gap: per-group ECE gap
    """
    sel = table["selection_rate"]
    fpr, fnr = table["fpr"], table["fnr"]
    return {
        "demographic_parity_diff": float(sel.max() - sel.min()),
        "equalized_odds_diff": float(max(fpr.max() - fpr.min(), fnr.max() - fnr.min())),
        "auc_gap": float(table["auc"].max() - table["auc"].min()),
        "calibration_gap": float(table["ece"].max() - table["ece"].min()),
    }


def full_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitive: pd.DataFrame | None = None,
    threshold: float = 0.5,
) -> dict:
    """Combined report: overall quality + fairness for each sensitive attribute."""
    report: dict = {"overall": classification_metrics(y_true, y_prob, threshold)}
    if sensitive is not None:
        for col in sensitive.columns:
            table = group_fairness_table(y_true, y_prob, sensitive[col], threshold)
            report[f"by_{col}"] = table
            report[f"gaps_{col}"] = fairness_gaps(table)
    return report
