"""Loading and preprocessing of the NIJ Recidivism Challenge and COMPAS datasets.

NIJ Recidivism Challenge (Georgia, 2013-2015): 25,835 individuals released
from prison to parole supervision, 54 columns, official train/test split
(``Training_Sample``). Targets: arrest within year 1 / 2 / 3 after release.

COMPAS (ProPublica, Broward County FL): used as a secondary dataset for
validating fairness metrics and mitigation techniques.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# NIJ dataset
# ---------------------------------------------------------------------------

NIJ_TARGET_3Y = "Recidivism_Within_3years"
NIJ_TARGET_YEARS = [
    "Recidivism_Arrest_Year1",
    "Recidivism_Arrest_Year2",
    "Recidivism_Arrest_Year3",
]
NIJ_SENSITIVE = ["Race", "Gender"]

# Ordinal categorical variables with an explicit ordering.
ORDINAL_MAPS: dict[str, dict[str, float]] = {
    "Age_at_Release": {
        "18-22": 20, "23-27": 25, "28-32": 30, "33-37": 35,
        "38-42": 40, "43-47": 45, "48 or older": 50,
    },
    "Education_Level": {
        "Less than HS diploma": 0,
        "High School Diploma": 1,
        "At least some college": 2,
    },
    "Prison_Years": {
        "Less than 1 year": 0.5,
        "1-2 years": 1.5,
        "Greater than 2 to 3 years": 2.5,
        "More than 3 years": 4.0,
    },
}

# Nominal categorical variables -> one-hot encoding.
NIJ_ONEHOT = ["Supervision_Level_First", "Prison_Offense"]

# Features describing the dynamics of the supervision period
# (the "temporal" branch of the hybrid model).
NIJ_DYNAMIC_PREFIXES = (
    "Violations_",
    "Delinquency_Reports",
    "Program_Attendances",
    "Program_UnexcusedAbsences",
    "Residence_Changes",
    "Avg_Days_per_DrugTest",
    "DrugTests_",
    "Percent_Days_Employed",
    "Jobs_Per_Year",
    "Employment_Exempt",
)

_COUNT_RE = re.compile(r"(\d+)")


def _parse_capped_count(value: object) -> float:
    """Parse values like ``'10 or more'`` / ``'3 or more'`` / ``'7'`` to float."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    match = _COUNT_RE.search(str(value))
    return float(match.group(1)) if match else np.nan


def _to_binary(series: pd.Series) -> pd.Series:
    """Map TRUE/FALSE (str or bool) to 1/0, keeping NaN."""
    mapping = {
        True: 1.0, False: 0.0,
        "TRUE": 1.0, "FALSE": 0.0,
        "True": 1.0, "False": 0.0,
        "true": 1.0, "false": 0.0,
    }
    return series.map(mapping)


def _is_boolean_like(series: pd.Series) -> bool:
    vals = set(series.dropna().unique().tolist())
    return len(vals) > 0 and vals <= {True, False, "TRUE", "FALSE", "True", "False", "true", "false"}


@dataclass
class NIJData:
    """Container with the fully prepared NIJ dataset."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray          # binary: recidivism within 3 years
    y_test: np.ndarray
    y_years_train: np.ndarray    # (n, 3) event indicators per year
    y_years_test: np.ndarray
    sensitive_train: pd.DataFrame
    sensitive_test: pd.DataFrame
    static_cols: list[str] = field(default_factory=list)
    dynamic_cols: list[str] = field(default_factory=list)


def load_nij_raw(path: str | Path = "data/raw/nij_recidivism_full.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_nij(df: pd.DataFrame, include_sensitive: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transform the raw NIJ table into a numeric feature matrix.

    Returns ``(X, meta)`` where ``meta`` keeps targets, the official split flag
    and sensitive attributes.
    """
    df = df.copy()

    meta = pd.DataFrame({
        "Training_Sample": df["Training_Sample"].astype(int),
        NIJ_TARGET_3Y: _to_binary(df[NIJ_TARGET_3Y]).astype(int),
        "Race": df["Race"].astype(str),
        "Gender": df["Gender"].astype(str),
    })
    for col in NIJ_TARGET_YEARS:
        meta[col] = _to_binary(df[col]).astype(int)

    drop_cols = ["ID", "Training_Sample", NIJ_TARGET_3Y, *NIJ_TARGET_YEARS]
    if not include_sensitive:
        drop_cols += NIJ_SENSITIVE
    feats = df.drop(columns=drop_cols)

    parts: list[pd.DataFrame] = []
    for col in feats.columns:
        s = feats[col]
        if col in ORDINAL_MAPS:
            parts.append(s.map(ORDINAL_MAPS[col]).astype(float).rename(col).to_frame())
        elif col in NIJ_ONEHOT or (include_sensitive and col in NIJ_SENSITIVE):
            dummies = pd.get_dummies(s, prefix=col, dummy_na=s.isna().any(), dtype=float)
            dummies.columns = [c.replace(" ", "_") for c in dummies.columns]
            parts.append(dummies)
        elif _is_boolean_like(s):
            parts.append(_to_binary(s).rename(col).to_frame())
        elif pd.api.types.is_string_dtype(s) or s.dtype == object:
            parts.append(s.map(_parse_capped_count).rename(col).to_frame())
        else:
            parts.append(s.astype(float).rename(col).to_frame())

    X = pd.concat(parts, axis=1)

    # Missing values: median imputation + missing-indicator columns.
    for col in X.columns[X.isna().any()].tolist():
        X[f"{col}__missing"] = X[col].isna().astype(float)
        X[col] = X[col].fillna(X[col].median())

    return X, meta


def split_feature_groups(columns: list[str]) -> tuple[list[str], list[str]]:
    """Split feature columns into static and supervision-dynamics groups."""
    dynamic = [c for c in columns if c.startswith(NIJ_DYNAMIC_PREFIXES)]
    static = [c for c in columns if c not in dynamic]
    return static, dynamic


def prepare_nij(
    path: str | Path = "data/raw/nij_recidivism_full.csv",
    include_sensitive: bool = False,
) -> NIJData:
    """Full pipeline: load -> preprocess -> official train/test split."""
    raw = load_nij_raw(path)
    X, meta = preprocess_nij(raw, include_sensitive=include_sensitive)

    train_mask = (meta["Training_Sample"] == 1).to_numpy()
    static_cols, dynamic_cols = split_feature_groups(X.columns.tolist())

    y = meta[NIJ_TARGET_3Y].to_numpy()
    y_years = meta[NIJ_TARGET_YEARS].to_numpy()

    return NIJData(
        X_train=X.loc[train_mask].reset_index(drop=True),
        X_test=X.loc[~train_mask].reset_index(drop=True),
        y_train=y[train_mask],
        y_test=y[~train_mask],
        y_years_train=y_years[train_mask],
        y_years_test=y_years[~train_mask],
        sensitive_train=meta.loc[train_mask, NIJ_SENSITIVE].reset_index(drop=True),
        sensitive_test=meta.loc[~train_mask, NIJ_SENSITIVE].reset_index(drop=True),
        static_cols=static_cols,
        dynamic_cols=dynamic_cols,
    )


# ---------------------------------------------------------------------------
# COMPAS dataset
# ---------------------------------------------------------------------------

COMPAS_TARGET = "two_year_recid"
COMPAS_FEATURES = [
    "age", "sex", "juv_fel_count", "juv_misd_count", "juv_other_count",
    "priors_count", "c_charge_degree",
]


def load_compas_raw(path: str | Path = "data/raw/compas-scores-two-years.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_compas(
    path: str | Path = "data/raw/compas-scores-two-years.csv",
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Standard ProPublica filtering; returns ``(X, y, race)``."""
    df = load_compas_raw(path)
    df = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].reset_index(drop=True)

    X = df[COMPAS_FEATURES].copy()
    X["sex"] = (X["sex"] == "Male").astype(float)
    X["c_charge_degree"] = (X["c_charge_degree"] == "F").astype(float)
    X = X.astype(float)

    y = df[COMPAS_TARGET].to_numpy()
    race = df["race"]
    return X, y, race
