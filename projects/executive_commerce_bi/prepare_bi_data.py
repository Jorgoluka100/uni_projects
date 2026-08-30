"""Prepare governed Power BI / Tableau exports from verified e-commerce Parquet marts."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "headline_kpis": {
        "commercial_orders", "unique_customers", "merchandise_value_brl",
        "average_order_value_brl", "freight_value_brl", "total_payment_value_brl",
    },
    "customer_order_frequency": {
        "customers", "repeat_customers", "repeat_customer_pct",
        "average_orders_per_customer", "average_customer_merchandise_value_brl",
    },
    "seller_concentration_summary": {
        "active_sellers", "largest_seller_share_pct", "top_ten_seller_share_pct", "hhi_index",
    },
    "monthly_performance": {
        "order_month", "orders", "customers", "merchandise_value_brl",
        "average_order_value_brl", "merchandise_value_mom_pct", "merchandise_value_rank",
    },
    "category_performance": {
        "category_name", "orders", "items", "merchandise_value_brl", "freight_value_brl",
        "freight_to_merchandise_pct", "average_review_score", "merchandise_value_rank",
    },
    "delivery_review_summary": {
        "delivery_status", "delivered_orders", "average_review_score",
        "average_days_late", "delivered_order_share_pct",
    },
    "payment_behaviour": {
        "payment_type", "payment_rows", "orders", "payment_value_brl",
        "average_installments", "order_penetration_pct",
    },
    "top_categories_by_customer_state": {
        "customer_state", "category_name", "orders", "merchandise_value_brl", "category_rank",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_verified(source_dir: Path, name: str) -> pd.DataFrame:
    path = source_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing verified upstream table: {path}")
    frame = pd.read_parquet(path)
    missing = REQUIRED_COLUMNS[name] - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")
    return frame


def build_outputs(source_dir: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {name: read_verified(source_dir, name) for name in REQUIRED_COLUMNS}

    headline = tables["headline_kpis"].iloc[0].to_dict()
    repeat = tables["customer_order_frequency"].iloc[0].to_dict()
    concentration = tables["seller_concentration_summary"].iloc[0].to_dict()
    executive = pd.DataFrame([{**headline, **repeat, **concentration}])

    export_frames = {
        "executive_kpis": executive,
        "monthly_performance": tables["monthly_performance"].sort_values("order_month"),
        "category_performance": tables["category_performance"].sort_values("merchandise_value_rank").head(20),
        "delivery_review_summary": tables["delivery_review_summary"],
        "payment_behaviour": tables["payment_behaviour"],
        "state_category_mix": tables["top_categories_by_customer_state"],
    }

    long_rows: list[dict[str, object]] = []
    for label, column in [
        ("Commercial Orders", "commercial_orders"),
        ("Unique Customers", "unique_customers"),
        ("Merchandise Value (BRL)", "merchandise_value_brl"),
        ("Average Order Value (BRL)", "average_order_value_brl"),
        ("Repeat Customer %", "repeat_customer_pct"),
        ("Top 10 Seller Share %", "top_ten_seller_share_pct"),
    ]:
        long_rows.append({"section": "Executive KPI", "dimension": label, "metric": label, "value": executive.iloc[0][column], "secondary_value": None, "sort_order": len(long_rows) + 1})

    for _, row in tables["monthly_performance"].iterrows():
        month = pd.Timestamp(row["order_month"]).strftime("%Y-%m")
        for metric, column in [("Monthly GMV", "merchandise_value_brl"), ("Monthly Orders", "orders"), ("Average Order Value", "average_order_value_brl")]:
            long_rows.append({"section": "Trend", "dimension": month, "metric": metric, "value": row[column], "secondary_value": row.get("merchandise_value_mom_pct"), "sort_order": int(pd.Timestamp(row["order_month"]).strftime("%Y%m"))})

    for _, row in tables["category_performance"].sort_values("merchandise_value_rank").head(15).iterrows():
        for metric, column in [("Category GMV", "merchandise_value_brl"), ("Category Orders", "orders"), ("Average Review", "average_review_score")]:
            long_rows.append({"section": "Category", "dimension": row["category_name"], "metric": metric, "value": row[column], "secondary_value": row["freight_to_merchandise_pct"], "sort_order": int(row["merchandise_value_rank"])})

    for _, row in tables["delivery_review_summary"].iterrows():
        long_rows.append({"section": "Delivery", "dimension": row["delivery_status"], "metric": "Average Review", "value": row["average_review_score"], "secondary_value": row["delivered_order_share_pct"], "sort_order": 1 if row["delivery_status"] == "on_time" else 2})

    long_frame = pd.DataFrame(long_rows)
    export_frames["tableau_dashboard_long"] = long_frame

    manifest: dict[str, object] = {"source": "verified ecommerce_sql_analytics Parquet exports", "files": {}}
    for name, frame in export_frames.items():
        destination = output_dir / f"{name}.csv"
        frame.to_csv(destination, index=False, date_format="%Y-%m-%d")
        manifest["files"][destination.name] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "sha256": sha256(destination),
        }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source"
        output = root / "output"
        source.mkdir()
        fixtures = {
            "headline_kpis": pd.DataFrame([{"commercial_orders": 3, "unique_customers": 2, "merchandise_value_brl": 250.0, "average_order_value_brl": 83.33, "freight_value_brl": 25.0, "total_payment_value_brl": 275.0}]),
            "customer_order_frequency": pd.DataFrame([{"customers": 2, "repeat_customers": 1, "repeat_customer_pct": 50.0, "average_orders_per_customer": 1.5, "average_customer_merchandise_value_brl": 125.0}]),
            "seller_concentration_summary": pd.DataFrame([{"active_sellers": 2, "largest_seller_share_pct": 70.0, "top_ten_seller_share_pct": 100.0, "hhi_index": 5800.0}]),
            "monthly_performance": pd.DataFrame([{"order_month": pd.Timestamp("2017-01-01"), "orders": 3, "customers": 2, "merchandise_value_brl": 250.0, "average_order_value_brl": 83.33, "merchandise_value_mom_pct": None, "merchandise_value_rank": 1}]),
            "category_performance": pd.DataFrame([{"category_name": "health_beauty", "orders": 3, "items": 4, "merchandise_value_brl": 250.0, "freight_value_brl": 25.0, "freight_to_merchandise_pct": 10.0, "average_review_score": 4.2, "merchandise_value_rank": 1}]),
            "delivery_review_summary": pd.DataFrame([{"delivery_status": "on_time", "delivered_orders": 2, "average_review_score": 4.5, "average_days_late": 0.0, "delivered_order_share_pct": 66.67}, {"delivery_status": "late", "delivered_orders": 1, "average_review_score": 2.0, "average_days_late": 5.0, "delivered_order_share_pct": 33.33}]),
            "payment_behaviour": pd.DataFrame([{"payment_type": "credit_card", "payment_rows": 3, "orders": 3, "payment_value_brl": 275.0, "average_installments": 2.0, "order_penetration_pct": 100.0}]),
            "top_categories_by_customer_state": pd.DataFrame([{"customer_state": "SP", "category_name": "health_beauty", "orders": 3, "merchandise_value_brl": 250.0, "category_rank": 1}]),
        }
        for name, frame in fixtures.items():
            frame.to_parquet(source / f"{name}.parquet", index=False)
        manifest = build_outputs(source, output)
        assert (output / "executive_kpis.csv").exists()
        assert (output / "tableau_dashboard_long.csv").exists()
        assert manifest["files"]["executive_kpis.csv"]["rows"] == 1
        print("Executive Commerce BI data-contract self-test passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare governed exports for Power BI and Tableau")
    parser.add_argument("--source-dir", type=Path, default=Path("../ecommerce_sql_analytics/artifacts/tables"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        manifest = build_outputs(args.source_dir, args.output_dir)
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
