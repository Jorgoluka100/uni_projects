from __future__ import annotations

import pandas as pd


def summarize_clusters(customer_features: pd.DataFrame, labels) -> pd.DataFrame:
    """Return an interpretable customer-segment table using original business units."""
    segmented = customer_features.copy()
    segmented["cluster"] = labels

    summary = (
        segmented.groupby("cluster", as_index=False)
        .agg(
            customers=("CustomerID", "size"),
            median_recency_days=("recency_days", "median"),
            median_frequency_orders=("frequency_orders", "median"),
            median_monetary_value=("monetary_value", "median"),
            mean_order_value=("average_order_value", "mean"),
            total_revenue=("monetary_value", "sum"),
        )
        .sort_values("median_monetary_value", ascending=False)
        .reset_index(drop=True)
    )
    summary["customer_share"] = summary["customers"] / summary["customers"].sum()
    summary["revenue_share"] = summary["total_revenue"] / summary["total_revenue"].sum()
    return summary


def add_relative_segment_labels(summary: pd.DataFrame) -> pd.DataFrame:
    """Attach descriptive labels based on observed cluster profiles, not hidden assumptions."""
    labelled = summary.copy()
    recency_mid = labelled["median_recency_days"].median()
    frequency_mid = labelled["median_frequency_orders"].median()
    monetary_mid = labelled["median_monetary_value"].median()

    def describe(row) -> str:
        recent = row["median_recency_days"] <= recency_mid
        frequent = row["median_frequency_orders"] >= frequency_mid
        valuable = row["median_monetary_value"] >= monetary_mid
        if recent and frequent and valuable:
            return "high_value_active"
        if recent and not frequent and valuable:
            return "recent_high_spend"
        if recent and not valuable:
            return "recent_lower_value"
        if not recent and valuable:
            return "valuable_at_risk"
        return "inactive_lower_value"

    labelled["relative_profile"] = labelled.apply(describe, axis=1)
    return labelled


def validate_segment_output(customer_features: pd.DataFrame, labels) -> None:
    if len(customer_features) != len(labels):
        raise AssertionError("Every customer must receive exactly one cluster label")
    if pd.Series(labels).isna().any():
        raise AssertionError("Cluster labels contain missing values")
    if pd.Series(labels).nunique() < 2:
        raise AssertionError("Clustering collapsed to fewer than two segments")
