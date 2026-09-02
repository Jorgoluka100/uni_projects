from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from run import LEAKAGE_COLUMNS, TARGET, add_features, chronological_split, feature_columns, metrics


def sample_frame(rows: int = 100):
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h")
    frame = pd.DataFrame({
        "instant": np.arange(rows),
        "dteday": timestamps.normalize(),
        "season": 1,
        "yr": 0,
        "mnth": timestamps.month,
        "hr": timestamps.hour,
        "holiday": 0,
        "weekday": timestamps.dayofweek,
        "workingday": 1,
        "weathersit": 1,
        "temp": 0.5,
        "atemp": 0.5,
        "hum": 0.5,
        "windspeed": 0.2,
        "casual": 10,
        "registered": 20,
        "cnt": 30,
        "timestamp": timestamps,
    })
    return add_features(frame)


def test_leakage_columns_are_excluded():
    frame = sample_frame()
    features = feature_columns(frame)
    assert TARGET not in features
    assert not LEAKAGE_COLUMNS.intersection(features)


def test_chronological_split_is_ordered():
    train, validation, test, _ = chronological_split(sample_frame(200))
    assert train.timestamp.max() < validation.timestamp.min()
    assert validation.timestamp.max() < test.timestamp.min()


def test_metrics_are_zero_for_perfect_prediction():
    y = np.array([1.0, 4.0, 9.0])
    result = metrics(y, y)
    assert result.mae == 0.0
    assert result.rmse == 0.0
