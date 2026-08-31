"""Build compact BI-ready CSVs from the verified Olist analytics warehouse.

The source dataset and warehouse logic are owned by projects/ecommerce_sql_analytics.
This script deliberately reuses that pipeline so the Power BI and Tableau work do
not invent a second set of business rules or headline numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
ECOMMERCE = ROOT / "projects" / "ecommerce_sql_analytics"
PROJECT = Path(__file__).resolve().parent
DATA_DIR = PROJECT / "data"
ARTIFACT_DIR = PROJECT / ".artifacts"
DB_PATH = ARTIFACT_DIR / "ecommerce.duckdb"

# Reuse the production SQL project's source, warehouse and validation code rather
# than maintaining a second implementation for the dashboards.
sys.path.insert(0, str(ECOMMERCE))
from src.config import ProjectConfig  # noqa: E402
from src.data import ensure_dataset  # noqa: E402
from src.validate import assert_integrity, run_integrity_checks  # noqa: E402
from src.warehouse import build_analytics, connect, load_raw_tables  # noqa: E402


def run_verified_warehouse() -> None:
    """Rebuild Olist marts and verify them against retained executed evidence."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    config = ProjectConfig(database_path=DB_PATH, output_dir=ARTIFACT_DIR)
    source_hashes = ensure_dataset(config)

    con = connect(DB_PATH)
    load_raw_tables(con, config.data_dir)
    build_analytics(con, ECOMMERCE / "sql")
    checks = run_integrity_checks(con)
    assert_integrity(checks)

    headline = con.execute("SELECT * FROM analytics.headline_kpis").df().iloc[0].to_dict()
    repeat = con.execute("SELECT * FROM analytics.customer_order_frequency").df().iloc[0].to_dict()
    strongest = con.execute(
        "SELECT order_month, merchandise_value_brl FROM analytics.monthly_performance "
        "ORDER BY merchandise_value_brl DESC LIMIT 1"
    ).fetchone()

    observed = {
        "commercial_orders": int(headline["commercial_orders"]),
        "unique_customers": int(headline["unique_customers"]),
        "merchandise_value_brl": round(float(headline["merchandise_value_brl"]), 2),
        "repeat_customer_pct": round(float(repeat["repeat_customer_pct"]), 2),
        "strongest_complete_month": str(strongest[0])[:7],
        "strongest_month_merchandise_value_brl": round(float(strongest[1]), 2),
    }
    retained = json.loads((ECOMMERCE / "results" / "verified_summary.json").read_text(encoding="utf-8"))[
        "headline_metrics"
    ]
    expected = {
        "commercial_orders": int(retained["commercial_orders"]),
        "unique_customers": int(retained["unique_customers"]),
        "merchandise_value_brl": round(float(retained["merchandise_value_brl"]), 2),
        "repeat_customer_pct": round(float(retained["repeat_customer_pct"]), 2),
        "strongest_complete_month": str(retained["strongest_complete_month"]),
        "strongest_month_merchandise_value_brl": round(
            float(retained["strongest_month_merchandise_value_brl"]), 2
        ),
    }
    if observed != expected:
        raise AssertionError(
            "Rebuilt BI warehouse does not match retained executed evidence. "
            f"observed={observed}; expected={expected}"
        )

    audit = {
        "verification_pass": True,
        "source_hashes": source_hashes,
        "integrity_checks": [
            {"name": check.name, "passed": bool(check.passed), "detail": check.detail}
            for check in checks
        ],
        "observed": observed,
    }
    (ARTIFACT_DIR / "bi_warehouse_verification.json").write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8"
    )
    con.close()


def export_query(con: duckdb.DuckDBPyConnection, filename: str, query: str) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    frame = con.execute(query).df()
    frame.to_csv(path, index=False)
    return len(frame)


def build_exports() -> dict[str, int]:
    run_verified_warehouse()
    con = duckdb.connect(str(DB_PATH), read_only=True)

    counts: dict[str, int] = {}
    counts["kpis.csv"] = export_query(
        con,
        "kpis.csv",
        """
        WITH commercial AS (
            SELECT *
            FROM analytics.order_mart
            WHERE commercial_order
        ),
        customer_frequency AS (
            SELECT customer_unique_id, COUNT(*) AS order_count
            FROM commercial
            WHERE customer_unique_id IS NOT NULL
            GROUP BY customer_unique_id
        )
        SELECT
            COUNT(*)::BIGINT AS commercial_orders,
            COUNT(DISTINCT customer_unique_id)::BIGINT AS unique_customers,
            ROUND(SUM(merchandise_value_brl), 2) AS merchandise_value_brl,
            ROUND(AVG(merchandise_value_brl), 2) AS avg_order_value_brl,
            ROUND(100.0 * (SELECT AVG(CASE WHEN order_count >= 2 THEN 1.0 ELSE 0.0 END) FROM customer_frequency), 2) AS repeat_customer_pct,
            ROUND(100.0 * AVG(CASE WHEN delivered_late THEN 1.0 WHEN delivered_late = FALSE THEN 0.0 END), 2) AS late_delivery_pct,
            ROUND(AVG(review_score), 2) AS avg_review_score,
            ROUND(AVG(review_score) FILTER (WHERE delivered_late = FALSE), 2) AS on_time_review_score,
            ROUND(AVG(review_score) FILTER (WHERE delivered_late = TRUE), 2) AS late_review_score
        FROM commercial
        """,
    )

    counts["monthly_performance.csv"] = export_query(
        con,
        "monthly_performance.csv",
        """
        SELECT
            order_month,
            COUNT(*)::BIGINT AS orders,
            COUNT(DISTINCT customer_unique_id)::BIGINT AS unique_customers,
            ROUND(SUM(merchandise_value_brl), 2) AS merchandise_value_brl,
            ROUND(SUM(freight_value_brl), 2) AS freight_value_brl,
            ROUND(AVG(merchandise_value_brl), 2) AS avg_order_value_brl,
            ROUND(100.0 * AVG(CASE WHEN delivered_late THEN 1.0 WHEN delivered_late = FALSE THEN 0.0 END), 2) AS late_delivery_pct,
            ROUND(AVG(review_score), 2) AS avg_review_score
        FROM analytics.order_mart
        WHERE commercial_order
        GROUP BY order_month
        ORDER BY order_month
        """,
    )

    counts["category_performance.csv"] = export_query(
        con,
        "category_performance.csv",
        """
        SELECT
            category_name,
            COUNT(*)::BIGINT AS items,
            COUNT(DISTINCT order_id)::BIGINT AS orders,
            COUNT(DISTINCT customer_unique_id)::BIGINT AS unique_customers,
            ROUND(SUM(item_price_brl), 2) AS merchandise_value_brl,
            ROUND(AVG(item_price_brl), 2) AS avg_item_price_brl,
            ROUND(100.0 * AVG(CASE WHEN delivered_late THEN 1.0 WHEN delivered_late = FALSE THEN 0.0 END), 2) AS late_delivery_pct,
            ROUND(AVG(review_score), 2) AS avg_review_score
        FROM analytics.item_mart
        WHERE commercial_order
        GROUP BY category_name
        ORDER BY merchandise_value_brl DESC
        """,
    )

    counts["state_performance.csv"] = export_query(
        con,
        "state_performance.csv",
        """
        SELECT
            COALESCE(customer_state, 'Unknown') AS customer_state,
            COUNT(*)::BIGINT AS orders,
            COUNT(DISTINCT customer_unique_id)::BIGINT AS unique_customers,
            ROUND(SUM(merchandise_value_brl), 2) AS merchandise_value_brl,
            ROUND(AVG(merchandise_value_brl), 2) AS avg_order_value_brl,
            ROUND(100.0 * AVG(CASE WHEN delivered_late THEN 1.0 WHEN delivered_late = FALSE THEN 0.0 END), 2) AS late_delivery_pct,
            ROUND(AVG(review_score), 2) AS avg_review_score
        FROM analytics.order_mart
        WHERE commercial_order
        GROUP BY customer_state
        ORDER BY merchandise_value_brl DESC
        """,
    )

    counts["payment_mix.csv"] = export_query(
        con,
        "payment_mix.csv",
        """
        SELECT
            p.payment_type,
            COUNT(*)::BIGINT AS payment_rows,
            COUNT(DISTINCT p.order_id)::BIGINT AS orders,
            ROUND(SUM(COALESCE(p.payment_value, 0)), 2) AS payment_value_brl,
            ROUND(AVG(COALESCE(p.payment_installments, 0)), 2) AS avg_installments
        FROM raw.payments p
        JOIN analytics.order_mart o USING (order_id)
        WHERE o.commercial_order
        GROUP BY p.payment_type
        ORDER BY payment_value_brl DESC
        """,
    )

    counts["tableau_marketplace_story.csv"] = export_query(
        con,
        "tableau_marketplace_story.csv",
        """
        WITH order_category AS (
            SELECT
                order_id,
                order_month,
                COALESCE(customer_state, 'Unknown') AS customer_state,
                COALESCE(category_name, 'unknown') AS category_name,
                customer_unique_id,
                SUM(item_price_brl) AS category_merchandise_value_brl,
                MAX(CASE WHEN delivered_late THEN 1 WHEN delivered_late = FALSE THEN 0 END) AS delivered_late,
                MAX(review_score) AS review_score
            FROM analytics.item_mart
            WHERE commercial_order
            GROUP BY order_id, order_month, customer_state, category_name, customer_unique_id
        )
        SELECT
            order_month,
            customer_state,
            category_name,
            COUNT(*)::BIGINT AS order_category_rows,
            COUNT(DISTINCT customer_unique_id)::BIGINT AS unique_customers,
            ROUND(SUM(category_merchandise_value_brl), 2) AS merchandise_value_brl,
            ROUND(AVG(category_merchandise_value_brl), 2) AS avg_order_category_value_brl,
            ROUND(100.0 * AVG(delivered_late), 2) AS late_delivery_pct,
            ROUND(AVG(review_score), 2) AS avg_review_score
        FROM order_category
        GROUP BY order_month, customer_state, category_name
        ORDER BY order_month, merchandise_value_brl DESC
        """,
    )

    con.close()
    (DATA_DIR / "manifest.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return counts


def validate_exports() -> None:
    counts = build_exports()
    expected = {
        "kpis.csv": 1,
        "monthly_performance.csv": 20,
    }
    for name, exact_rows in expected.items():
        if counts[name] != exact_rows:
            raise AssertionError(f"Unexpected {name} row count: {counts[name]} != {exact_rows}")
    if counts["category_performance.csv"] < 60:
        raise AssertionError("Category export unexpectedly small")
    if counts["state_performance.csv"] < 20:
        raise AssertionError("State export unexpectedly small")
    if counts["tableau_marketplace_story.csv"] < 1000:
        raise AssertionError("Tableau story export unexpectedly small")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    validate_exports()
