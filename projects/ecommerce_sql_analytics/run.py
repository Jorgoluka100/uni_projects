"""Build and verify the Olist DuckDB analytics project."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import duckdb
import pandas as pd

from src.config import ProjectConfig
from src.data import ensure_dataset
from src.validate import assert_integrity, run_integrity_checks
from src.warehouse import build_analytics, connect, export_analytics, load_raw_tables

RETAINED_EVIDENCE = {
    "commercial_orders": 98199,
    "unique_customers": 94983,
    "merchandise_value_brl": 13494400.74,
    "repeat_customer_pct": 3.03,
    "strongest_complete_month": "2017-11-01",
}


def _create_table(connection: duckdb.DuckDBPyConnection, name: str, frame: pd.DataFrame) -> None:
    registration = f"fixture_{name}"
    connection.register(registration, frame)
    connection.execute(f"CREATE OR REPLACE TABLE raw.{name} AS SELECT * FROM {registration}")
    connection.unregister(registration)


def build_synthetic_fixture(connection: duckdb.DuckDBPyConnection) -> None:
    """Create a tiny relational fixture that includes repeat customers and join traps."""
    customers = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3", "c4"],
            "customer_unique_id": ["u1", "u1", "u2", "u3"],
            "customer_zip_code_prefix": [1000, 1000, 2000, 3000],
            "customer_city": ["sao_paulo", "sao_paulo", "rio", "curitiba"],
            "customer_state": ["SP", "SP", "RJ", "PR"],
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4"],
            "customer_id": ["c1", "c2", "c3", "c4"],
            "order_status": ["delivered", "delivered", "delivered", "canceled"],
            "order_purchase_timestamp": [
                "2017-01-05 08:00:00",
                "2017-02-05 09:00:00",
                "2017-02-10 10:00:00",
                "2017-03-01 11:00:00",
            ],
            "order_approved_at": [
                "2017-01-05 09:00:00",
                "2017-02-05 10:00:00",
                "2017-02-10 11:00:00",
                "2017-03-01 12:00:00",
            ],
            "order_delivered_carrier_date": [
                "2017-01-06 12:00:00",
                "2017-02-06 12:00:00",
                "2017-02-11 12:00:00",
                None,
            ],
            "order_delivered_customer_date": [
                "2017-01-10 12:00:00",
                "2017-02-20 12:00:00",
                "2017-02-14 12:00:00",
                None,
            ],
            "order_estimated_delivery_date": [
                "2017-01-12 00:00:00",
                "2017-02-15 00:00:00",
                "2017-02-16 00:00:00",
                "2017-03-20 00:00:00",
            ],
        }
    )
    order_items = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2", "o3", "o4"],
            "order_item_id": [1, 2, 1, 1, 1],
            "product_id": ["p1", "p2", "p1", "p2", "p1"],
            "seller_id": ["s1", "s1", "s1", "s2", "s1"],
            "shipping_limit_date": ["2017-01-07"] * 5,
            "price": [100.0, 50.0, 80.0, 20.0, 999.0],
            "freight_value": [10.0, 5.0, 5.0, 5.0, 20.0],
        }
    )
    payments = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o2", "o3", "o4"],
            "payment_sequential": [1, 2, 1, 1, 1],
            "payment_type": ["credit_card", "voucher", "credit_card", "debit_card", "credit_card"],
            "payment_installments": [2, 1, 1, 1, 10],
            "payment_value": [120.0, 45.0, 85.0, 25.0, 1019.0],
        }
    )
    reviews = pd.DataFrame(
        {
            "review_id": ["r1", "r1b", "r2", "r3"],
            "order_id": ["o1", "o1", "o2", "o3"],
            "review_score": [5, 4, 2, 5],
            "review_comment_title": [None] * 4,
            "review_comment_message": [None] * 4,
            "review_creation_date": ["2017-01-11", "2017-01-12", "2017-02-21", "2017-02-15"],
            "review_answer_timestamp": [
                "2017-01-11 09:00:00",
                "2017-01-12 09:00:00",
                "2017-02-21 09:00:00",
                "2017-02-15 09:00:00",
            ],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": ["p1", "p2"],
            "product_category_name": ["cat_a", "cat_b"],
            "product_name_lenght": [10, 10],
            "product_description_lenght": [20, 20],
            "product_photos_qty": [1, 1],
            "product_weight_g": [100, 200],
            "product_length_cm": [10, 20],
            "product_height_cm": [5, 5],
            "product_width_cm": [5, 10],
        }
    )
    sellers = pd.DataFrame(
        {
            "seller_id": ["s1", "s2"],
            "seller_zip_code_prefix": [1000, 2000],
            "seller_city": ["sao_paulo", "rio"],
            "seller_state": ["SP", "RJ"],
        }
    )
    category_translation = pd.DataFrame(
        {
            "product_category_name": ["cat_a", "cat_b"],
            "product_category_name_english": ["category_a", "category_b"],
        }
    )

    for name, frame in {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "payments": payments,
        "reviews": reviews,
        "products": products,
        "sellers": sellers,
        "category_translation": category_translation,
    }.items():
        _create_table(connection, name, frame)


def self_test() -> None:
    connection = connect(":memory:")
    build_synthetic_fixture(connection)
    build_analytics(connection, Path(__file__).parent / "sql")
    checks = run_integrity_checks(connection)
    assert_integrity(checks)

    headline = connection.execute("SELECT * FROM analytics.headline_kpis").fetchone()
    columns = [item[0] for item in connection.description]
    headline_dict = dict(zip(columns, headline))
    assert headline_dict["commercial_orders"] == 3
    assert headline_dict["unique_customers"] == 2
    assert abs(float(headline_dict["merchandise_value_brl"]) - 250.0) < 0.01

    # o1 has two items and two payment rows. A raw three-way join would create four
    # combinations; the order mart must still contain one o1 row with R$150 GMV.
    o1 = connection.execute(
        "SELECT item_count, merchandise_value_brl, payment_rows FROM analytics.order_mart WHERE order_id='o1'"
    ).fetchone()
    assert o1 == (2, 150.0, 2)

    repeat = connection.execute("SELECT repeat_customer_pct FROM analytics.customer_order_frequency").fetchone()[0]
    assert abs(float(repeat) - 50.0) < 0.01
    print("E-commerce SQL analytics self-test passed.")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def full_run(config: ProjectConfig) -> dict[str, object]:
    source_hashes = ensure_dataset(config)
    connection = connect(config.database_path)
    load_raw_tables(connection, config.data_dir)
    build_analytics(connection, Path(__file__).parent / "sql")

    checks = run_integrity_checks(connection)
    assert_integrity(checks)
    exports = export_analytics(connection, config.output_dir / "tables")

    headline = connection.execute("SELECT * FROM analytics.headline_kpis").df().iloc[0].to_dict()
    repeat = connection.execute("SELECT * FROM analytics.customer_order_frequency").df().iloc[0].to_dict()
    strongest = connection.execute(
        "SELECT order_month, merchandise_value_brl FROM analytics.monthly_performance ORDER BY merchandise_value_brl DESC LIMIT 1"
    ).fetchone()

    observed = {
        "commercial_orders": int(headline["commercial_orders"]),
        "unique_customers": int(headline["unique_customers"]),
        "merchandise_value_brl": round(float(headline["merchandise_value_brl"]), 2),
        "repeat_customer_pct": round(float(repeat["repeat_customer_pct"]), 2),
        "strongest_complete_month": str(strongest[0]),
        "strongest_month_merchandise_value_brl": round(float(strongest[1]), 2),
    }
    retained_match = (
        observed["commercial_orders"] == RETAINED_EVIDENCE["commercial_orders"]
        and observed["unique_customers"] == RETAINED_EVIDENCE["unique_customers"]
        and abs(observed["merchandise_value_brl"] - RETAINED_EVIDENCE["merchandise_value_brl"]) < 0.01
        and abs(observed["repeat_customer_pct"] - RETAINED_EVIDENCE["repeat_customer_pct"]) < 0.01
        and observed["strongest_complete_month"].startswith(RETAINED_EVIDENCE["strongest_complete_month"])
    )

    verification = {
        "project": "E-commerce Sales and Customer Analysis",
        "verification_pass": bool(retained_match and all(check.passed for check in checks)),
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "source_hashes": source_hashes,
        "observed": observed,
        "retained_reference": RETAINED_EVIDENCE,
        "retained_reference_match": retained_match,
        "integrity_checks": [asdict(check) for check in checks],
        "exports": {path.name: _file_sha256(path) for path in exports},
        "limitations": [
            "Historical anonymised marketplace data; results do not describe Olist's current business.",
            "Merchandise value is not profit because product cost and operating expense are unavailable.",
            "Delivery/review relationships are observational and should not be interpreted as causal effects.",
            "Later acquisition cohorts have less time to mature and are therefore right-censored.",
        ],
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "verification.json").write_text(json.dumps(verification, indent=2, default=str), encoding="utf-8")
    if not verification["verification_pass"]:
        raise AssertionError("Full dataset output did not match retained verified evidence")
    return verification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Olist e-commerce DuckDB analytics warehouse")
    parser.add_argument("--self-test", action="store_true", help="Run a fast synthetic relational test")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--database", type=Path, default=Path("artifacts/ecommerce.duckdb"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    config = ProjectConfig(output_dir=args.output_dir, database_path=args.database)
    verification = full_run(config)
    print(json.dumps(verification, indent=2, default=str))


if __name__ == "__main__":
    main()
