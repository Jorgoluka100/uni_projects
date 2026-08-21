from __future__ import annotations

import pandas as pd

from .data import OPERATIONAL_INPUT_COLUMNS

ENGINEERED_FEATURES = [
    "call_failure_rate",
    "seconds_per_call",
    "sms_per_call",
    "contact_diversity",
    "customer_value_per_month",
    "usage_per_month",
]
MODEL_FEATURES = OPERATIONAL_INPUT_COLUMNS + ENGINEERED_FEATURES


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame[OPERATIONAL_INPUT_COLUMNS].copy().astype(float)
    call_denominator = data["frequency_of_use"].clip(lower=0) + 1.0
    subscription_denominator = data["subscription_length"].clip(lower=1)
    data["call_failure_rate"] = data["call_failure"] / (
        data["call_failure"] + data["frequency_of_use"] + 1.0
    )
    data["seconds_per_call"] = data["seconds_of_use"] / call_denominator
    data["sms_per_call"] = data["frequency_of_sms"] / call_denominator
    data["contact_diversity"] = data["distinct_called_numbers"] / call_denominator
    data["customer_value_per_month"] = data["customer_value"] / subscription_denominator
    data["usage_per_month"] = data["frequency_of_use"] / subscription_denominator
    return data[MODEL_FEATURES]
