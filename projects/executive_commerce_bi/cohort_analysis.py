"""Generate correctly right-censored cohort-retention evidence.

A sparse retention table can omit both future months and observed months with zero
returning customers. That is useful for storage but unsafe for an aggregate curve:
zero-retention observed cohorts would disappear from the denominator. This module
builds the complete *eligible* cohort-age grid, fills observed zero-return months
with zero, and excludes only months that were not yet observable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data"
DB_PATH = PROJECT / ".artifacts" / "ecommerce.duckdb"
MANIFEST = DATA / "manifest.json"
SUMMARY = DATA / "analysis_summary.json"
VERIFIED_ANALYSIS = PROJECT / "VERIFIED_ANALYSIS.md"
COMPLETE_MONTH_EXCLUSIVE = "2018-09-01"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register_file(manifest: dict, path: Path, frame: pd.DataFrame) -> None:
    manifest.setdefault("files", {})[path.name] = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "sha256": sha256(path),
    }


def build_right_censored_cohorts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        f"""
        WITH customer_months AS (
            SELECT DISTINCT
                customer_unique_id,
                order_month
            FROM analytics.order_mart
            WHERE commercial_order
              AND customer_unique_id IS NOT NULL
              AND order_month < DATE '{COMPLETE_MONTH_EXCLUSIVE}'
        ),
        customer_cohorts AS (
            SELECT
                customer_unique_id,
                MIN(order_month) AS cohort_month
            FROM customer_months
            GROUP BY customer_unique_id
        ),
        cohort_sizes AS (
            SELECT
                cohort_month,
                COUNT(*)::BIGINT AS cohort_customers
            FROM customer_cohorts
            WHERE cohort_month >= DATE '2017-01-01'
              AND cohort_month < DATE '{COMPLETE_MONTH_EXCLUSIVE}'
            GROUP BY cohort_month
        ),
        activity AS (
            SELECT
                c.cohort_month,
                DATE_DIFF('month', c.cohort_month, m.order_month)::INTEGER AS month_number,
                COUNT(DISTINCT m.customer_unique_id)::BIGINT AS active_customers
            FROM customer_months m
            JOIN customer_cohorts c USING (customer_unique_id)
            WHERE c.cohort_month >= DATE '2017-01-01'
              AND c.cohort_month < DATE '{COMPLETE_MONTH_EXCLUSIVE}'
              AND DATE_DIFF('month', c.cohort_month, m.order_month) BETWEEN 0 AND 12
            GROUP BY c.cohort_month, month_number
        ),
        ages AS (
            SELECT range::INTEGER AS month_number
            FROM range(0, 13)
        ),
        eligible_grid AS (
            SELECT
                s.cohort_month,
                a.month_number,
                s.cohort_customers
            FROM cohort_sizes s
            CROSS JOIN ages a
            WHERE s.cohort_month + a.month_number * INTERVAL '1 month'
                  < DATE '{COMPLETE_MONTH_EXCLUSIVE}'
        )
        SELECT
            g.cohort_month,
            g.month_number,
            COALESCE(a.active_customers, 0)::BIGINT AS active_customers,
            g.cohort_customers,
            ROUND(100.0 * COALESCE(a.active_customers, 0) / NULLIF(g.cohort_customers, 0), 2) AS retention_pct
        FROM eligible_grid g
        LEFT JOIN activity a
          ON a.cohort_month = g.cohort_month
         AND a.month_number = g.month_number
        ORDER BY g.cohort_month, g.month_number
        """
    ).df()


def build_weighted_curve(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby("month_number", as_index=False).agg(
        observable_cohorts=("cohort_month", "count"),
        active_customers=("active_customers", "sum"),
        eligible_cohort_customers=("cohort_customers", "sum"),
        average_cohort_retention_pct=("retention_pct", "mean"),
    )
    grouped["weighted_retention_pct"] = (
        100.0 * grouped["active_customers"] / grouped["eligible_cohort_customers"]
    ).round(2)
    grouped["average_cohort_retention_pct"] = grouped["average_cohort_retention_pct"].round(2)
    return grouped[
        [
            "month_number",
            "observable_cohorts",
            "active_customers",
            "eligible_cohort_customers",
            "weighted_retention_pct",
            "average_cohort_retention_pct",
        ]
    ].sort_values("month_number")


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Verified warehouse not found: {DB_PATH}")

    con = duckdb.connect(str(DB_PATH), read_only=True)
    raw = build_right_censored_cohorts(con)
    con.close()
    curve = build_weighted_curve(raw)

    if raw.empty or curve.empty:
        raise AssertionError("Cohort retention analysis produced no rows")
    month_zero = curve.loc[curve["month_number"] == 0].iloc[0]
    if float(month_zero["weighted_retention_pct"]) != 100.0:
        raise AssertionError("Month-zero weighted retention must equal 100%")
    if not curve["observable_cohorts"].is_monotonic_decreasing:
        raise AssertionError("Eligible cohort count should not increase with cohort age")
    if (raw["active_customers"] > raw["cohort_customers"]).any():
        raise AssertionError("Active customers cannot exceed cohort size")
    if not (raw["active_customers"] == 0).any():
        raise AssertionError("Expected observed zero-retention cohort-months in complete grid")

    raw_path = DATA / "analysis_cohort_retention.csv"
    curve_path = DATA / "analysis_retention_curve.csv"
    raw.to_csv(raw_path, index=False, date_format="%Y-%m-%d")
    curve.to_csv(curve_path, index=False)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    register_file(manifest, raw_path, raw)
    register_file(manifest, curve_path, curve)

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    retention_summary: dict[str, float | int] = {}
    for month in [1, 3, 6, 12]:
        match = curve.loc[curve["month_number"] == month]
        if not match.empty:
            row = match.iloc[0]
            retention_summary[f"month_{month}_weighted_retention_pct"] = round(float(row["weighted_retention_pct"]), 2)
            retention_summary[f"month_{month}_observable_cohorts"] = int(row["observable_cohorts"])
    summary["cohort_retention"] = retention_summary
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest["business_analysis"] = summary
    manifest["files"][SUMMARY.name]["sha256"] = sha256(SUMMARY)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    m1 = retention_summary["month_1_weighted_retention_pct"]
    m3 = retention_summary["month_3_weighted_retention_pct"]
    m6 = retention_summary["month_6_weighted_retention_pct"]
    section = (
        "\n## Cohort retention\n\n"
        "The retention curve uses an **eligible cohort-age grid**: observed months with no returning customers are recorded as zero, while genuinely future months are excluded. This avoids both zero-fill censoring bias and sparse-table denominator bias.\n\n"
        f"- Weighted month-1 retention: **{m1:.2f}%**\n"
        f"- Weighted month-3 retention: **{m3:.2f}%**\n"
        f"- Weighted month-6 retention: **{m6:.2f}%**\n\n"
        "The low cohort retention reinforces the repeat-customer finding without making recent cohorts look worse simply because their future follow-up months do not yet exist.\n"
    )
    analysis_text = VERIFIED_ANALYSIS.read_text(encoding="utf-8")
    marker = "\n## Cohort retention\n"
    if marker in analysis_text:
        analysis_text = analysis_text.split(marker, 1)[0].rstrip() + "\n"
    VERIFIED_ANALYSIS.write_text(analysis_text.rstrip() + "\n" + section, encoding="utf-8")

    print(json.dumps(retention_summary, indent=2))


if __name__ == "__main__":
    main()
