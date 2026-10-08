"""Unit tests for src/recidivism/data.py using synthetic mini tables."""

from pathlib import Path

import numpy as np
import pandas as pd

from recidivism.data import (
    NIJ_TARGET_3Y,
    NIJ_TARGET_YEARS,
    _is_boolean_like,
    _parse_capped_count,
    _to_binary,
    prepare_compas,
    prepare_nij,
    preprocess_nij,
    split_feature_groups,
)


def _mini_nij_df() -> pd.DataFrame:
    """Minimal NIJ-like table covering ordinal, one-hot, bool, count, numeric."""
    return pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "Training_Sample": [1, 1, 0, 0],
        NIJ_TARGET_3Y: ["TRUE", "FALSE", True, False],
        "Recidivism_Arrest_Year1": ["TRUE", "FALSE", "FALSE", "FALSE"],
        "Recidivism_Arrest_Year2": ["FALSE", "FALSE", "TRUE", "FALSE"],
        "Recidivism_Arrest_Year3": ["FALSE", "FALSE", "FALSE", "FALSE"],
        "Race": ["BLACK", "WHITE", "BLACK", "WHITE"],
        "Gender": ["M", "F", "M", "F"],
        "Age_at_Release": ["18-22", "28-32", "48 or older", "23-27"],
        "Education_Level": [
            "Less than HS diploma", "High School Diploma",
            "At least some college", "High School Diploma",
        ],
        "Prison_Years": [
            "Less than 1 year", "1-2 years",
            "Greater than 2 to 3 years", "More than 3 years",
        ],
        "Supervision_Level_First": ["High", "Standard", "High", None],
        "Prison_Offense": ["Property", "Violent", "Drug", "Property"],
        "Employment_Exempt": ["TRUE", "FALSE", "TRUE", "FALSE"],
        "Violations_ElectronicMonitoring": ["0", "3 or more", "10 or more", "7"],
        "Percent_Days_Employed": [0.5, 0.8, np.nan, 0.2],
        "Prior_Arrest_Episodes_Felony": [1.0, 2.0, 0.0, 3.0],
    })


def test_parse_capped_count():
    assert _parse_capped_count("10 or more") == 10.0
    assert _parse_capped_count("3 or more") == 3.0
    assert _parse_capped_count("7") == 7.0
    assert _parse_capped_count(5) == 5.0
    assert np.isnan(_parse_capped_count(np.nan))
    assert np.isnan(_parse_capped_count("none"))


def test_to_binary_and_boolean_like():
    s = pd.Series(["TRUE", "FALSE", True, False, np.nan])
    out = _to_binary(s)
    assert list(out.dropna()) == [1.0, 0.0, 1.0, 0.0]
    assert _is_boolean_like(pd.Series(["TRUE", "FALSE"]))
    assert not _is_boolean_like(pd.Series(["a", "b"]))
    assert not _is_boolean_like(pd.Series([np.nan, np.nan]))


def test_preprocess_nij_shapes():
    X, meta = preprocess_nij(_mini_nij_df())
    assert len(X) == 4
    assert NIJ_TARGET_3Y in meta.columns
    for col in NIJ_TARGET_YEARS:
        assert col in meta.columns
    assert "Race" not in X.columns
    assert "Percent_Days_Employed__missing" in X.columns


def test_preprocess_nij_include_sensitive():
    X, _ = preprocess_nij(_mini_nij_df(), include_sensitive=True)
    assert any(c.startswith("Race_") for c in X.columns)
    assert any(c.startswith("Gender_") for c in X.columns)


def test_split_feature_groups():
    cols = [
        "Age_at_Release",
        "Violations_ElectronicMonitoring",
        "Percent_Days_Employed",
        "Prior_Arrest_Episodes_Felony",
        "Jobs_Per_Year",
    ]
    static, dynamic = split_feature_groups(cols)
    assert "Age_at_Release" in static
    assert "Prior_Arrest_Episodes_Felony" in static
    assert "Violations_ElectronicMonitoring" in dynamic
    assert "Percent_Days_Employed" in dynamic
    assert "Jobs_Per_Year" in dynamic


def test_prepare_nij_roundtrip(tmp_path: Path):
    path = tmp_path / "nij.csv"
    _mini_nij_df().to_csv(path, index=False)
    data = prepare_nij(path)
    assert len(data.X_train) == 2
    assert len(data.X_test) == 2
    assert data.y_train.shape == (2,)
    assert data.y_years_train.shape == (2, 3)
    assert set(data.sensitive_train.columns) == {"Race", "Gender"}
    assert data.static_cols
    assert data.dynamic_cols


def test_prepare_compas(tmp_path: Path):
    df = pd.DataFrame({
        "days_b_screening_arrest": [0, 10, 100, -5],
        "is_recid": [0, 1, 1, 0],
        "c_charge_degree": ["F", "M", "O", "F"],
        "score_text": ["Low", "High", "Low", "Medium"],
        "age": [25, 30, 40, 22],
        "sex": ["Male", "Female", "Male", "Male"],
        "juv_fel_count": [0, 1, 0, 0],
        "juv_misd_count": [0, 0, 1, 0],
        "juv_other_count": [0, 0, 0, 1],
        "priors_count": [1, 5, 2, 0],
        "two_year_recid": [0, 1, 1, 0],
        "race": ["African-American", "Caucasian", "Other", "Caucasian"],
    })
    path = tmp_path / "compas.csv"
    df.to_csv(path, index=False)
    X, y, race = prepare_compas(path)
    # row with days=100 (and charge O) filtered out → 3 remain
    assert len(X) == 3
    assert y.shape == (3,)
    assert len(race) == 3
    assert set(X.columns) >= {
        "age", "sex", "priors_count", "c_charge_degree",
    }
