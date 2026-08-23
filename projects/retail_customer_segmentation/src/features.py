from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

RFM_COLUMNS = ["recency_days", "frequency_orders", "monetary_value"]


def build_customer_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate clean line-level transactions into one row per customer."""
    required = {
        "CustomerID",
        "InvoiceNo",
        "InvoiceDate",
        "Quantity",
        "line_revenue",
    }
    missing = sorted(required - set(transactions.columns))
    if missing:
        raise ValueError(f"Missing required transaction columns: {missing}")

    snapshot_date = transactions["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    customer = (
        transactions.groupby("CustomerID", as_index=False)
        .agg(
            last_purchase=("InvoiceDate", "max"),
            first_purchase=("InvoiceDate", "min"),
            frequency_orders=("InvoiceNo", "nunique"),
            monetary_value=("line_revenue", "sum"),
            total_items=("Quantity", "sum"),
            transaction_lines=("InvoiceNo", "size"),
        )
    )

    customer["recency_days"] = (snapshot_date - customer["last_purchase"].dt.normalize()).dt.days
    customer["customer_tenure_days"] = (
        customer["last_purchase"].dt.normalize() - customer["first_purchase"].dt.normalize()
    ).dt.days + 1
    customer["average_order_value"] = customer["monetary_value"] / customer["frequency_orders"]

    if customer["CustomerID"].duplicated().any():
        raise AssertionError("Customer feature table is not one row per customer")
    if customer[RFM_COLUMNS].isna().any().any():
        raise AssertionError("RFM features contain missing values")
    if customer["recency_days"].lt(0).any():
        raise AssertionError("Recency cannot be negative")
    if customer["frequency_orders"].le(0).any():
        raise AssertionError("Frequency must be positive")
    if customer["monetary_value"].le(0).any():
        raise AssertionError("Monetary value must be positive")

    return customer.sort_values("CustomerID").reset_index(drop=True)


def cap_extreme_values(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> tuple[pd.DataFrame, dict]:
    """Winsorise extreme RFM values while keeping every customer in the analysis."""
    columns = columns or RFM_COLUMNS
    capped = frame.copy()
    caps: dict[str, dict[str, float]] = {}

    for column in columns:
        lower = float(capped[column].quantile(lower_quantile))
        upper = float(capped[column].quantile(upper_quantile))
        if lower > upper:
            raise ValueError(f"Invalid quantile bounds for {column}")
        capped[column] = capped[column].clip(lower=lower, upper=upper)
        caps[column] = {"lower": lower, "upper": upper}

    return capped, caps


def prepare_clustering_matrix(
    customer_features: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, RobustScaler, dict]:
    """Log-transform skewed RFM features and robust-scale them for distance models."""
    capped, caps = cap_extreme_values(customer_features)

    transformed = pd.DataFrame(index=capped.index)
    transformed["recency_log1p"] = np.log1p(capped["recency_days"].astype(float))
    transformed["frequency_log1p"] = np.log1p(capped["frequency_orders"].astype(float))
    transformed["monetary_log1p"] = np.log1p(capped["monetary_value"].astype(float))

    scaler = RobustScaler()
    matrix = scaler.fit_transform(transformed)

    if not np.isfinite(matrix).all():
        raise AssertionError("Clustering matrix contains non-finite values")

    metadata = {
        "winsorisation_caps": caps,
        "transforms": {
            "recency_days": "log1p after 1st/99th percentile clipping",
            "frequency_orders": "log1p after 1st/99th percentile clipping",
            "monetary_value": "log1p after 1st/99th percentile clipping",
        },
        "scaler": "RobustScaler",
    }
    return matrix, transformed, scaler, metadata
