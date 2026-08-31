"""Rebuild the pinned Olist warehouse and refresh both BI dashboard data sets.

This is the recruiter-facing reproducibility path for the BI project. It reuses the
SQL project's real download, warehouse and integrity logic, verifies retained
headline evidence, then feeds the existing Power BI/Tableau export contract.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parent
ECOMMERCE = ROOT / "projects" / "ecommerce_sql_analytics"
ARTIFACTS = PROJECT / ".artifacts"
SOURCE_TABLES = ARTIFACTS / "verified_tables"
DATA_DIR = PROJECT / "data"
DB_PATH = ARTIFACTS / "ecommerce.duckdb"

sys.path.insert(0, str(ECOMMERCE))
from src.config import ProjectConfig  # noqa: E402
from src.data import ensure_dataset  # noqa: E402
from src.validate import assert_integrity, run_integrity_checks  # noqa: E402
from src.warehouse import build_analytics, connect, load_raw_tables  # noqa: E402

from prepare_bi_data import REQUIRED_COLUMNS, build_outputs  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_retained_headline(con) -> dict[str, object]:
    """Check the rebuild against the retained executed headline evidence."""
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
    retained = json.loads(
        (ECOMMERCE / "results" / "verified_summary.json").read_text(encoding="utf-8")
    )["headline_metrics"]
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
            "Rebuilt warehouse differs from retained executed evidence. "
            f"observed={observed}; expected={expected}"
        )
    return observed


def export_verified_source_tables(con) -> None:
    SOURCE_TABLES.mkdir(parents=True, exist_ok=True)
    for table_name in REQUIRED_COLUMNS:
        frame = con.execute(f"SELECT * FROM analytics.{table_name}").df()
        frame.to_parquet(SOURCE_TABLES / f"{table_name}.parquet", index=False)


def add_operational_exports(con, manifest: dict[str, object]) -> None:
    """Add BI-specific views that make regional and seller risk explorable."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    extras = {
        "state_performance.csv": con.execute(
            """
            SELECT
                customer_state,
                COUNT(*)::BIGINT AS orders,
                COUNT(DISTINCT customer_unique_id)::BIGINT AS unique_customers,
                ROUND(SUM(merchandise_value_brl), 2) AS merchandise_value_brl,
                ROUND(AVG(merchandise_value_brl), 2) AS average_order_value_brl,
                ROUND(100.0 * AVG(CASE WHEN delivered_late THEN 1.0 WHEN delivered_late = FALSE THEN 0.0 END), 2) AS late_delivery_pct,
                ROUND(AVG(review_score), 2) AS average_review_score
            FROM analytics.order_mart
            WHERE commercial_order AND customer_state IS NOT NULL
            GROUP BY customer_state
            ORDER BY merchandise_value_brl DESC
            """
        ).df(),
        "seller_operational_review.csv": con.execute(
            "SELECT * FROM analytics.seller_operational_review"
        ).df(),
    }
    files = manifest.setdefault("files", {})
    for filename, frame in extras.items():
        destination = DATA_DIR / filename
        frame.to_csv(destination, index=False)
        files[filename] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "sha256": sha256(destination),
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
    manifest["verification"] = {
        "verification_pass": True,
        "dataset_version": config.dataset_version,
        "archive_sha256": config.archive_sha256,
        "source_hashes": source_hashes,
        "headline": retained_headline,
        "integrity_checks": [
            {"name": check.name, "passed": bool(check.passed), "detail": check.detail}
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
