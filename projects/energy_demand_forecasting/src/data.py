from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv"
LOOKBACK = 60
HORIZON = 14


def load_daily_data(url: str = DATA_URL) -> pd.DataFrame:
    frame = pd.read_csv(url, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    frame = frame.rename(
        columns={
            "Consumption": "consumption",
            "Wind": "wind",
            "Solar": "solar",
            "Wind+Solar": "wind_solar",
        }
    )
    if len(frame) != 4383:
        raise ValueError(f"expected 4,383 daily records, found {len(frame):,}")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("daily index must be sorted and unique")
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    if len(expected.difference(frame.index)):
        raise ValueError("source contains missing calendar dates")
    if frame["consumption"].isna().any() or (frame["consumption"] <= 0).any():
        raise ValueError("consumption target must be complete and positive")
    return frame


def chronological_windows(frame: pd.DataFrame):
    """Fit scaling on train only, then create 60-day -> 14-day windows."""
    n = len(frame)
    train_end = int(0.70 * n)
    validation_end = int(0.85 * n)
    values = frame[["consumption"]].to_numpy(dtype="float32")
    scaler = StandardScaler().fit(values[:train_end])
    scaled = scaler.transform(values).astype("float32")

    def make_windows(first_target: int, last_target: int):
        X, y, dates = [], [], []
        for index in range(max(first_target, LOOKBACK), last_target - HORIZON + 1):
            X.append(scaled[index - LOOKBACK:index])
            y.append(scaled[index:index + HORIZON, 0])
            dates.append(frame.index[index])
        return np.asarray(X), np.asarray(y), np.asarray(dates)

    train = make_windows(0, train_end)
    validation = make_windows(train_end, validation_end)
    test = make_windows(validation_end, n)
    if not (train[2].max() < validation[2].min() < test[2].min()):
        raise AssertionError("window chronology is invalid")
    return scaler, train, validation, test


def invert_scale(scaler: StandardScaler, values) -> np.ndarray:
    array = np.asarray(values)
    return scaler.inverse_transform(array.reshape(-1, 1)).reshape(array.shape)
