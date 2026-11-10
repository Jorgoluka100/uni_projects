from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.analysis import (
    calibration_table,
    classify_metric_change,
    decision_from_report,
    worst_severity,
)


def test_calibration_table_rejects_invalid_probability_inputs():
    with pytest.raises(ValueError, match="within"):
        calibration_table(np.array([0, 1]), np.array([0.2, 1.2]))
    with pytest.raises(ValueError, match="binary"):
        calibration_table(np.array([0, 2]), np.array([0.2, 0.8]))


def test_monitoring_policy_rejects_unknown_severity():
    with pytest.raises(ValueError, match="unknown"):
        worst_severity(["green", "critical"])
    with pytest.raises(ValueError, match="unknown"):
        decision_from_report({"batch": "bad", "alert": "critical"})


def test_alert_thresholds_must_be_ordered_and_finite():
    assert classify_metric_change(0.3, 0.1, 0.25) == "red"
    with pytest.raises(ValueError, match="thresholds"):
        classify_metric_change(0.3, 0.4, 0.2)

from run import ece, generate, psi


def test_constant_reference_distribution_does_not_hide_feature_drift():
    assert psi(np.ones(100), np.full(100, 2.0)) > 0.5


def test_model_watch_generation_and_ece_validate_inputs():
    with pytest.raises(ValueError, match="at least 100"):
        generate(20, 42)
    with pytest.raises(ValueError, match="within"):
        ece(np.array([0, 1]), np.array([-0.1, 1.1]))
