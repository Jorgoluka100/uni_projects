from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from run import (
    GROUP_COLUMN,
    REQUIRED_COLUMNS,
    TARGET,
    clean_data,
    feature_columns,
    regression_metrics,
    subject_holdout,
)


def _sample_data() -> pd.DataFrame:
    rows = []
    numeric = sorted(REQUIRED_COLUMNS - {GROUP_COLUMN})
    for subject in range(10):
        for visit in range(6):
            row = {column: float(subject + visit + 1) for column in numeric}
            row[GROUP_COLUMN] = subject
            row[TARGET] = float(10 + subject + visit)
            row["motor_UPDRS"] = float(5 + subject + visit)
            rows.append(row)
    return pd.DataFrame(rows)


def test_cleaning_converts_infinite_features_to_imputable_missing_values():
    frame = _sample_data()
    frame.loc[0, "PPE"] = np.inf
    frame.loc[1, TARGET] = np.inf
    cleaned, report = clean_data(frame)
    assert cleaned["PPE"].isna().sum() == 1
    assert report["rows_removed_missing_group_or_target"] == 1
    assert np.isfinite(cleaned[TARGET]).all()


def test_subject_holdout_keeps_people_disjoint():
    frame = _sample_data()
    train_idx, test_idx = subject_holdout(frame, random_state=42)
    train_subjects = set(frame.iloc[train_idx][GROUP_COLUMN])
    test_subjects = set(frame.iloc[test_idx][GROUP_COLUMN])
    assert train_subjects.isdisjoint(test_subjects)


def test_default_feature_policy_excludes_motor_updrs():
    frame = _sample_data()
    features = feature_columns(frame, include_motor_updrs=False)
    assert "motor_UPDRS" not in features
    assert GROUP_COLUMN not in features
    assert TARGET not in features


def test_regression_metrics_reject_misaligned_or_non_finite_values():
    with pytest.raises(ValueError, match="aligned"):
        regression_metrics([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="finite"):
        regression_metrics([1.0, np.inf], [1.0, 2.0])
