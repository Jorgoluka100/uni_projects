"""Generate cohort-retention evidence from the verified Olist warehouse.

The upstream SQL table is right-censored: a cohort/month row exists only when that
month was actually observable. This script keeps that property and adds a weighted
retention curve for dashboard use.
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register_file(manifest: dict, path: Path, frame: pd.DataFrame) -> None:
    manifest.setdefault("files", {})[path.name] = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "sha256": sha256(path),
    }


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Verified warehouse not found: {DB_PATH}")

    con = duckdb.connect(str(DB_PATH), read_only=True)
    raw = con.execute(
        """
        SELECT
            cohort_month,
            month_number,
            active_customers,
            cohort_customers,
            retention_pct
        FROM analytics.cohort_retention
        ORDER BY cohort_month, month_number
        """
    ).df()
    curve = con.execute(
        """
        SELECT
            month_number,
            COUNT(*)::BIGINT AS observable_cohorts,
            SUM(active_customers)::BIGINT AS active_customers,
            SUM(cohort_customers)::BIGINT AS eligible_cohort_customers,
            ROUND(100.0 * SUM(active_customers) / NULLIF(SUM(cohort_customers), 0), 2) AS weighted_retention_pct,
            ROUND(AVG(retention_pct), 2) AS average_cohort_retention_pct
        FROM analytics.cohort_retention
        GROUP BY month_number
        ORDER BY month_number
        """
    ).df()
    con.close()

    if raw.empty or curve.empty:
        raise AssertionError("Cohort retention analysis produced no rows")
    if float(curve.loc[curve["month_number"] == 0, "weighted_retention_pct"].iloc[0]) != 100.0:
        raise AssertionError("Month-zero weighted retention must equal 100%")
    if not curve["observable_cohorts"].is_monotonic_decreasing:
        raise AssertionError("Observable cohort count should not increase with cohort age")

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

    m1 = retention_summary.get("month_1_weighted_retention_pct")
    m3 = retention_summary.get("month_3_weighted_retention_pct")
    m6 = retention_summary.get("month_6_weighted_retention_pct")
    section = (
        "\n## Cohort retention\n\n"
        "The warehouse uses acquisition cohorts and only includes follow-up months that were actually observable, so later cohorts are **right-censored rather than filled with artificial zeroes**.\n\n"
        f"- Weighted month-1 retention: **{m1:.2f}%**\n"
        f"- Weighted month-3 retention: **{m3:.2f}%**\n"
        f"- Weighted month-6 retention: **{m6:.2f}%**\n\n"
        "The low cohort retention reinforces the repeat-customer finding, while the censoring rule avoids making recent cohorts look worse simply because future months do not exist yet.\n"
    )
    analysis_text = VERIFIED_ANALYSIS.read_text(encoding="utf-8")
    marker = "\n## Cohort retention\n"
    if marker in analysis_text:
        analysis_text = analysis_text.split(marker, 1)[0].rstrip() + "\n"
    VERIFIED_ANALYSIS.write_text(analysis_text.rstrip() + "\n" + section, encoding="utf-8")

    print(json.dumps(retention_summary, indent=2))


if __name__ == "__main__":
    main()
