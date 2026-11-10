import numpy as np
import pandas as pd

from src.evaluation import summarize_clusters, validate_segment_output
from src.features import build_customer_features, prepare_clustering_matrix
from src.model import cluster_stability, evaluate_cluster_counts, fit_final_kmeans


def _clean_transactions() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2025-01-01")
    for customer in range(1, 31):
        orders = 1 + (customer % 5)
        for order in range(orders):
            rows.append(
                {
                    "InvoiceNo": f"{customer:03d}-{order:02d}",
                    "InvoiceDate": start + pd.Timedelta(days=customer * 2 + order),
                    "CustomerID": str(customer),
                    "StockCode": f"S{order:02d}",
                    "Description": "fixture",
                    "Quantity": 1 + customer % 4,
                    "UnitPrice": 2.0 + customer,
                    "line_revenue": float((1 + customer % 4) * (2.0 + customer)),
                    "Country": "United Kingdom",
                }
            )
    return pd.DataFrame(rows)


def test_customer_features_are_one_row_per_customer():
    customer = build_customer_features(_clean_transactions())
    assert len(customer) == 30
    assert customer["CustomerID"].is_unique
    assert customer["recency_days"].ge(0).all()
    assert customer["frequency_orders"].gt(0).all()
    assert customer["monetary_value"].gt(0).all()


def test_clustering_pipeline_returns_stable_valid_shapes():
    customer = build_customer_features(_clean_transactions())
    matrix, transformed, _, metadata = prepare_clustering_matrix(customer)

    assert matrix.shape == (30, 3)
    assert transformed.shape == (30, 3)
    assert np.isfinite(matrix).all()
    assert metadata["scaler"] == "RobustScaler"

    selection = evaluate_cluster_counts(matrix, k_values=[2, 3, 4])
    assert selection.selected_k in {2, 3, 4}
    assert set(selection.diagnostics["k"]) == {2, 3, 4}

    _, labels = fit_final_kmeans(matrix, selection.selected_k)
    validate_segment_output(customer, labels)
    summary = summarize_clusters(customer, labels)
    assert summary["customers"].sum() == 30
    assert np.isclose(summary["customer_share"].sum(), 1.0)

    stability = cluster_stability(matrix, selection.selected_k, labels, seeds=(3, 5, 7))
    assert stability["runs"] == 3
    assert 0.0 <= stability["min"] <= 1.0

def test_segmentation_rejects_invalid_feature_values_and_cluster_inputs():
    import pytest

    customer = build_customer_features(_clean_transactions())
    customer.loc[0, "monetary_value"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        prepare_clustering_matrix(customer)
    with pytest.raises(ValueError, match="finite"):
        evaluate_cluster_counts(np.array([[0.0], [1.0], [np.nan]]))
    with pytest.raises(ValueError, match="n_clusters"):
        fit_final_kmeans(np.arange(30, dtype=float).reshape(10, 3), 1)


def test_transaction_aggregation_rejects_missing_customer_and_bad_dates():
    import pytest

    frame = _clean_transactions()
    frame.loc[0, "CustomerID"] = None
    with pytest.raises(ValueError, match="CustomerID"):
        build_customer_features(frame)
