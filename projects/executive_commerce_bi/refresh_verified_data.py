from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import duckdb


PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]
ECOMMERCE = REPO / "projects" / "ecommerce_sql_analytics"
sys.path.insert(0, str(ECOMMERCE))

from run import ProjectConfig, ensure_dataset  # noqa: E402
from src.data import load_raw_tables  # noqa: E402
from src.validate import assert_integrity, run_integrity_checks  # noqa: E402
from src.warehouse import build_analytics, connect  # noqa: E402

from prepare_bi_data import build_outputs  # noqa: E402


ARTIFACTS = PROJECT / ".artifacts"
DB_PATH = ARTIFACTS / "ecommerce.duckdb"
SOURCE_TABLES = ARTIFACTS / "source_tables"
DATA_DIR = PROJECT / "data"
VERIFIED_SUMMARY = ECOMMERCE / "results" / "verified_summary.json"


def scalar(con: duckdb.DuckDBPyConnection, query: str):
    row = con.execute(query).fetchone()
    return None if row is None else row[0]


def verify_retained_headline(con: duckdb.DuckDBPyConnection) -> dict:
    expected = json.loads(VERIFIED_SUMMARY.read_text(encoding="utf-8"))
    current = {
        "commercial_orders": int(
            scalar(con, "SELECT commercial_orders FROM analytics.headline_kpis")
        ),
        "unique_customers": int(
            scalar(con, "SELECT unique_customers FROM analytics.headline_kpis")
        ),
        "merchandise_value_brl": round(
            float(scalar(con, "SELECT merchandise_value_brl FROM analytics.headline_kpis")), 2
        ),
        "repeat_customer_pct": round(
            float(
                scalar(
                    con,
                    """
                    SELECT 100.0 * SUM(CASE WHEN order_count >= 2 THEN 1 ELSE 0 END) / COUNT(*)
                    FROM analytics.customer_frequency
                    """,
                )
            ),
            2,
        ),
    }
    strongest = con.execute(
        """
        SELECT order_month, merchandise_value_brl
        FROM analytics.monthly_performance
        ORDER BY merchandise_value_brl DESC, order_month
        LIMIT 1
        """
    ).fetchone()
    current["strongest_complete_month"] = str(strongest[0])
    current["strongest_month_gmv_brl"] = round(float(strongest[1]), 2)

    comparable = (
        "commercial_orders",
        "unique_customers",
        "merchandise_value_brl",
        "repeat_customer_pct",
        "strongest_complete_month",
        "strongest_month_gmv_brl",
    )
    mismatches = {
        key: {"expected": expected[key], "observed": current[key]}
        for key in comparable
        if current[key] != expected[key]
    }
    if mismatches:
        raise AssertionError(
            "Pinned warehouse no longer reproduces retained BI evidence: "
            + json.dumps(mismatches, indent=2, default=str)
        )
    return current


def export_verified_source_tables(con: duckdb.DuckDBPyConnection) -> None:
    if SOURCE_TABLES.exists():
        shutil.rmtree(SOURCE_TABLES)
    SOURCE_TABLES.mkdir(parents=True, exist_ok=True)

    exports = {
        "headline_kpis.parquet": "SELECT * FROM analytics.headline_kpis",
        "monthly_performance.parquet": "SELECT * FROM analytics.monthly_performance ORDER BY order_month",
        "category_performance.parquet": "SELECT * FROM analytics.category_performance ORDER BY merchandise_value_brl DESC",
        "delivery_quality.parquet": "SELECT * FROM analytics.delivery_quality ORDER BY delivery_status",
        "customer_frequency.parquet": "SELECT * FROM analytics.customer_frequency ORDER BY customer_unique_id",
        "payment_mix.parquet": "SELECT * FROM analytics.payment_mix ORDER BY payment_value_brl DESC",
        "order_mart.parquet": "SELECT * FROM analytics.order_mart",
        "item_mart.parquet": "SELECT * FROM analytics.item_mart",
    }
    for filename, query in exports.items():
        path = (SOURCE_TABLES / filename).as_posix().replace("'", "''")
        con.execute(f"COPY ({query}) TO '{path}' (FORMAT PARQUET)")


def add_operational_exports(con: duckdb.DuckDBPyConnection, manifest: dict) -> None:
    state_query = """
        SELECT
            customer_state,
            COUNT(*) AS commercial_orders,
            COUNT(DISTINCT customer_unique_id) AS unique_customers,
            ROUND(SUM(merchandise_value_brl), 2) AS merchandise_value_brl,
            ROUND(AVG(merchandise_value_brl), 2) AS avg_order_value_brl,
            ROUND(100.0 * AVG(CASE WHEN delivered_late THEN 1 ELSE 0 END), 2) AS late_delivery_pct,
            ROUND(AVG(review_score), 2) AS avg_review_score
        FROM analytics.order_mart
        WHERE commercial_order
        GROUP BY customer_state
        ORDER BY merchandise_value_brl DESC, customer_state
    """
    state_df = con.execute(state_query).fetchdf()
    state_path = DATA_DIR / "powerbi_state_performance.csv"
    state_df.to_csv(state_path, index=False)
    manifest["files"][state_path.name] = {
        "rows": int(len(state_df)),
        "columns": list(state_df.columns),
    }

    seller_query = """
        WITH seller_orders AS (
            SELECT
                i.seller_id,
                MAX(i.seller_state) AS seller_state,
                COUNT(DISTINCT i.order_id) AS orders,
                ROUND(SUM(i.item_price_brl), 2) AS merchandise_value_brl,
                ROUND(100.0 * AVG(CASE WHEN i.delivered_late THEN 1 ELSE 0 END), 2) AS late_delivery_pct,
                ROUND(AVG(i.review_score), 2) AS avg_review_score
            FROM analytics.item_mart i
            WHERE i.commercial_order
            GROUP BY i.seller_id
        )
        SELECT
            seller_id,
            seller_state,
            orders,
            merchandise_value_brl,
            late_delivery_pct,
            avg_review_score,
            CASE
                WHEN late_delivery_pct >= 20 OR avg_review_score < 3.5 THEN 'Priority review'
                WHEN late_delivery_pct >= 12 OR avg_review_score < 4.0 THEN 'Watch'
                ELSE 'Healthy'
            END AS operational_status
        FROM seller_orders
        ORDER BY merchandise_value_brl DESC, seller_id
    """
    seller_df = con.execute(seller_query).fetchdf()
    seller_path = DATA_DIR / "powerbi_seller_operations.csv"
    seller_df.to_csv(seller_path, index=False)
    manifest["files"][seller_path.name] = {
        "rows": int(len(seller_df)),
        "columns": list(seller_df.columns),
    }


def enrich_tableau_story(con: duckdb.DuckDBPyConnection, manifest: dict) -> None:
    tableau_path = DATA_DIR / "tableau_marketplace_story.csv"
    tableau_df = con.execute(
        """
        SELECT
            order_month,
            commercial_orders,
            unique_customers,
            merchandise_value_brl,
            avg_order_value_brl
        FROM analytics.monthly_performance
        ORDER BY order_month
        """
    ).fetchdf()
    tableau_df.to_csv(tableau_path, index=False)
    manifest["files"][tableau_path.name] = {
        "rows": int(len(tableau_df)),
        "columns": list(tableau_df.columns),
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    config = ProjectConfig(database_path=DB_PATH, output_dir=ARTIFACTS)
    source_hashes = ensure_dataset(config)

    con = connect(DB_PATH)
    load_raw_tables(con, config.data_dir)
    build_analytics(con, ECOMMERCE / "sql")
    checks = run_integrity_checks(con)
    assert_integrity(checks)
    retained_headline = verify_retained_headline(con)

    export_verified_source_tables(con)
    manifest = build_outputs(SOURCE_TABLES, DATA_DIR)
    add_operational_exports(con, manifest)
    enrich_tableau_story(con, manifest)
    manifest["verification"] = {
        "verification_pass": True,
        "dataset_version": config.dataset_version,
        "archive_sha256": config.archive_sha256,
        "source_hashes": source_hashes,
        "headline": retained_headline,
        "integrity_checks": [
            {
                "name": check.name,
                "passed": bool(check.passed),
                "value": check.value,
                "expectation": check.expectation,
            }
            for check in checks
        ],
    }
    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    con.close()
    print(json.dumps(manifest["verification"], indent=2, default=str))


if __name__ == "__main__":
    main()
