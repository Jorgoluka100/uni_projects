"""Decision-oriented business analysis for the Executive Commerce BI project.

The dashboard should answer business questions, not just display charts. This module
runs on the verified DuckDB warehouse and produces compact, inspectable outputs for
Power BI, Tableau and the GitHub project page.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(frame: pd.DataFrame, output_dir: Path, filename: str, manifest: dict[str, Any]) -> None:
    destination = output_dir / filename
    frame.to_csv(destination, index=False, date_format="%Y-%m-%d")
    manifest.setdefault("files", {})[filename] = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "sha256": sha256(destination),
    }


def build_business_analysis(con, output_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Create the analysis tables and a recruiter-readable insight summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    monthly = con.execute(
        """
        SELECT
            order_month,
            orders,
            customers,
            merchandise_value_brl,
            average_order_value_brl,
            merchandise_value_mom_pct,
            merchandise_value_rank
        FROM analytics.monthly_performance
        ORDER BY order_month
        """
    ).df()

    customer_segments = con.execute(
        """
        SELECT
            customer_segment,
            COUNT(*)::BIGINT AS customers,
            SUM(orders)::BIGINT AS orders,
            ROUND(SUM(merchandise_value_brl), 2) AS merchandise_value_brl,
            ROUND(AVG(orders), 2) AS average_orders_per_customer,
            ROUND(AVG(merchandise_value_brl), 2) AS average_customer_value_brl,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS customer_share_pct,
            ROUND(100.0 * SUM(merchandise_value_brl) / SUM(SUM(merchandise_value_brl)) OVER (), 2) AS merchandise_value_share_pct
        FROM analytics.customer_value_segments
        GROUP BY customer_segment
        ORDER BY merchandise_value_brl DESC
        """
    ).df()

    delivery_buckets = con.execute(
        """
        WITH scoped AS (
            SELECT
                order_id,
                merchandise_value_brl,
                review_score,
                days_late,
                CASE
                    WHEN delivered_late = FALSE THEN 'On time'
                    WHEN days_late BETWEEN 1 AND 3 THEN '1-3 days late'
                    WHEN days_late BETWEEN 4 AND 7 THEN '4-7 days late'
                    WHEN days_late BETWEEN 8 AND 14 THEN '8-14 days late'
                    ELSE '15+ days late'
                END AS delivery_bucket,
                CASE
                    WHEN delivered_late = FALSE THEN 1
                    WHEN days_late BETWEEN 1 AND 3 THEN 2
                    WHEN days_late BETWEEN 4 AND 7 THEN 3
                    WHEN days_late BETWEEN 8 AND 14 THEN 4
                    ELSE 5
                END AS bucket_order
            FROM analytics.order_mart
            WHERE commercial_order
              AND delivered_late IS NOT NULL
        )
        SELECT
            delivery_bucket,
            COUNT(*)::BIGINT AS orders,
            ROUND(SUM(merchandise_value_brl), 2) AS merchandise_value_brl,
            ROUND(AVG(days_late), 2) AS average_days_late,
            ROUND(AVG(review_score), 2) AS average_review_score,
            ROUND(100.0 * AVG(CASE WHEN review_score = 1 THEN 1.0 ELSE 0.0 END), 2) AS one_star_review_pct,
            ROUND(100.0 * AVG(CASE WHEN review_score = 5 THEN 1.0 ELSE 0.0 END), 2) AS five_star_review_pct,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS delivered_order_share_pct
        FROM scoped
        GROUP BY delivery_bucket, bucket_order
        ORDER BY bucket_order
        """
    ).df()

    state_analysis = con.execute(
        """
        SELECT
            customer_state,
            COUNT(*)::BIGINT AS orders,
            COUNT(DISTINCT customer_unique_id)::BIGINT AS unique_customers,
            ROUND(SUM(merchandise_value_brl), 2) AS merchandise_value_brl,
            ROUND(AVG(merchandise_value_brl), 2) AS average_order_value_brl,
            ROUND(100.0 * AVG(CASE WHEN delivered_late THEN 1.0 WHEN delivered_late = FALSE THEN 0.0 END), 2) AS late_delivery_pct,
            ROUND(SUM(CASE WHEN delivered_late THEN merchandise_value_brl ELSE 0 END), 2) AS late_order_merchandise_value_brl,
            ROUND(AVG(review_score), 2) AS average_review_score,
            ROUND(100.0 * AVG(CASE WHEN review_score <= 2 THEN 1.0 ELSE 0.0 END), 2) AS low_review_order_pct
        FROM analytics.order_mart
        WHERE commercial_order
          AND customer_state IS NOT NULL
        GROUP BY customer_state
        ORDER BY merchandise_value_brl DESC
        """
    ).df()

    category_analysis = con.execute(
        """
        WITH order_category AS (
            SELECT
                order_id,
                category_name,
                SUM(item_price_brl) AS merchandise_value_brl,
                SUM(item_freight_brl) AS freight_value_brl,
                MAX(CASE WHEN delivered_late THEN 1 WHEN delivered_late = FALSE THEN 0 END) AS delivered_late,
                MAX(review_score) AS review_score
            FROM analytics.item_mart
            WHERE commercial_order
            GROUP BY order_id, category_name
        )
        SELECT
            category_name,
            COUNT(*)::BIGINT AS orders,
            ROUND(SUM(merchandise_value_brl), 2) AS merchandise_value_brl,
            ROUND(SUM(freight_value_brl), 2) AS freight_value_brl,
            ROUND(100.0 * SUM(freight_value_brl) / NULLIF(SUM(merchandise_value_brl), 0), 2) AS freight_to_merchandise_pct,
            ROUND(100.0 * AVG(delivered_late), 2) AS late_delivery_pct,
            ROUND(SUM(CASE WHEN delivered_late = 1 THEN merchandise_value_brl ELSE 0 END), 2) AS late_order_merchandise_value_brl,
            ROUND(AVG(review_score), 2) AS average_review_score,
            ROUND(100.0 * AVG(CASE WHEN review_score <= 2 THEN 1.0 ELSE 0.0 END), 2) AS low_review_order_pct
        FROM order_category
        GROUP BY category_name
        HAVING COUNT(*) >= 100
        ORDER BY merchandise_value_brl DESC
        """
    ).df()

    seller_risk = con.execute(
        """
        SELECT
            operational_status,
            COUNT(*)::BIGINT AS sellers,
            SUM(orders)::BIGINT AS seller_orders,
            ROUND(SUM(merchandise_value_brl), 2) AS merchandise_value_brl,
            ROUND(AVG(late_delivery_rate_pct), 2) AS average_late_delivery_rate_pct,
            ROUND(AVG(average_review_score), 2) AS average_review_score
        FROM analytics.seller_operational_review
        GROUP BY operational_status
        ORDER BY CASE WHEN operational_status = 'review_priority' THEN 0 ELSE 1 END
        """
    ).df()

    payment = con.execute(
        "SELECT * FROM analytics.payment_behaviour ORDER BY payment_value_brl DESC"
    ).df()

    # Transparent management priority: rank material value currently exposed to late delivery.
    state_priority = state_analysis.loc[state_analysis["orders"] >= 500].copy()
    state_priority["entity_type"] = "state"
    state_priority["entity_name"] = state_priority["customer_state"]
    state_priority = state_priority.sort_values("late_order_merchandise_value_brl", ascending=False)

    category_priority = category_analysis.loc[category_analysis["orders"] >= 500].copy()
    category_priority["entity_type"] = "category"
    category_priority["entity_name"] = category_priority["category_name"]
    category_priority = category_priority.sort_values("late_order_merchandise_value_brl", ascending=False)

    priority_columns = [
        "entity_type",
        "entity_name",
        "orders",
        "merchandise_value_brl",
        "late_delivery_pct",
        "late_order_merchandise_value_brl",
        "average_review_score",
        "low_review_order_pct",
    ]
    operational_priority = pd.concat(
        [state_priority[priority_columns], category_priority[priority_columns]],
        ignore_index=True,
    ).sort_values("late_order_merchandise_value_brl", ascending=False)
    operational_priority.insert(0, "priority_rank", range(1, len(operational_priority) + 1))

    _write_csv(monthly, output_dir, "analysis_monthly_growth.csv", manifest)
    _write_csv(customer_segments, output_dir, "analysis_customer_segments.csv", manifest)
    _write_csv(delivery_buckets, output_dir, "analysis_delivery_impact.csv", manifest)
    _write_csv(state_analysis, output_dir, "analysis_state_performance.csv", manifest)
    _write_csv(category_analysis, output_dir, "analysis_category_performance.csv", manifest)
    _write_csv(seller_risk, output_dir, "analysis_seller_risk.csv", manifest)
    _write_csv(payment, output_dir, "analysis_payment_mix.csv", manifest)
    _write_csv(operational_priority, output_dir, "analysis_operational_priority.csv", manifest)

    headline = con.execute("SELECT * FROM analytics.headline_kpis").df().iloc[0]
    repeat = con.execute("SELECT * FROM analytics.customer_order_frequency").df().iloc[0]
    delivery = con.execute("SELECT * FROM analytics.delivery_review_summary").df()
    on_time = delivery.loc[delivery["delivery_status"] == "on_time"].iloc[0]
    late = delivery.loc[delivery["delivery_status"] == "late"].iloc[0]

    strongest_month = monthly.sort_values("merchandise_value_brl", ascending=False).iloc[0]
    top_state = state_analysis.iloc[0]
    top_category = category_analysis.iloc[0]
    priority = operational_priority.iloc[0] if not operational_priority.empty else None
    repeat_segments = customer_segments[customer_segments["customer_segment"].str.startswith("repeat")]

    summary: dict[str, Any] = {
        "scope": {
            "commercial_orders": int(headline["commercial_orders"]),
            "unique_customers": int(headline["unique_customers"]),
            "merchandise_value_brl": round(float(headline["merchandise_value_brl"]), 2),
        },
        "growth": {
            "strongest_complete_month": pd.Timestamp(strongest_month["order_month"]).strftime("%Y-%m"),
            "strongest_month_merchandise_value_brl": round(float(strongest_month["merchandise_value_brl"]), 2),
            "strongest_month_orders": int(strongest_month["orders"]),
        },
        "customers": {
            "repeat_customer_pct": round(float(repeat["repeat_customer_pct"]), 2),
            "repeat_segment_merchandise_value_share_pct": round(float(repeat_segments["merchandise_value_share_pct"].sum()), 2),
        },
        "delivery": {
            "late_delivery_share_pct": round(float(late["delivered_order_share_pct"]), 2),
            "on_time_average_review": round(float(on_time["average_review_score"]), 2),
            "late_average_review": round(float(late["average_review_score"]), 2),
            "review_score_gap": round(float(on_time["average_review_score"] - late["average_review_score"]), 2),
        },
        "market": {
            "top_state": str(top_state["customer_state"]),
            "top_state_merchandise_value_brl": round(float(top_state["merchandise_value_brl"]), 2),
            "top_category": str(top_category["category_name"]),
            "top_category_merchandise_value_brl": round(float(top_category["merchandise_value_brl"]), 2),
        },
        "management_priority": None if priority is None else {
            "entity_type": str(priority["entity_type"]),
            "entity_name": str(priority["entity_name"]),
            "late_order_merchandise_value_brl": round(float(priority["late_order_merchandise_value_brl"]), 2),
            "late_delivery_pct": round(float(priority["late_delivery_pct"]), 2),
            "average_review_score": round(float(priority["average_review_score"]), 2),
        },
    }

    summary_path = output_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest.setdefault("files", {})[summary_path.name] = {
        "rows": 1,
        "columns": list(summary.keys()),
        "sha256": sha256(summary_path),
    }

    priority_text = "No material late-delivery priority was produced."
    if priority is not None:
        priority_text = (
            f"The largest transparent operational exposure is **{priority['entity_type']} "
            f"{priority['entity_name']}**, with **R${float(priority['late_order_merchandise_value_brl']):,.0f}** "
            f"of merchandise value attached to late orders, a **{float(priority['late_delivery_pct']):.1f}%** "
            f"late-delivery rate and **{float(priority['average_review_score']):.2f}/5** average review score."
        )

    insights = f"""# Verified Business Analysis\n\nThis analysis is generated from the pinned Olist warehouse used by the Power BI and Tableau project. It is designed around management questions rather than chart count.\n\n## Executive readout\n\n- **Commercial scope:** {int(headline['commercial_orders']):,} orders, {int(headline['unique_customers']):,} unique customers and **R${float(headline['merchandise_value_brl']):,.2f}** merchandise value.\n- **Peak complete month:** {pd.Timestamp(strongest_month['order_month']).strftime('%Y-%m')} with **R${float(strongest_month['merchandise_value_brl']):,.2f}** GMV across {int(strongest_month['orders']):,} orders.\n- **Retention challenge:** only **{float(repeat['repeat_customer_pct']):.2f}%** of customers repeat, while repeat-customer segments account for **{float(repeat_segments['merchandise_value_share_pct'].sum()):.2f}%** of merchandise value.\n- **Delivery matters:** late deliveries represent **{float(late['delivered_order_share_pct']):.2f}%** of delivered commercial orders. Average review score falls from **{float(on_time['average_review_score']):.2f}/5** on time to **{float(late['average_review_score']):.2f}/5** when late — a **{float(on_time['average_review_score'] - late['average_review_score']):.2f}-point gap**.\n- **Largest market:** {top_state['customer_state']} contributes **R${float(top_state['merchandise_value_brl']):,.2f}** GMV.\n- **Largest analysed category:** {top_category['category_name']} contributes **R${float(top_category['merchandise_value_brl']):,.2f}** GMV.\n\n## Management priority\n\n{priority_text}\n\nThe priority table deliberately ranks **late-order merchandise value**, not an opaque composite score. A manager can therefore see exactly why an entity is high priority and then use late-delivery rate, review score and low-review share as context.\n\n## Dashboard questions\n\n1. Is commercial value growing, and which months drive the change?\n2. How much value comes from repeat customers versus one-time customers?\n3. How sharply does customer satisfaction deteriorate as delivery delay increases?\n4. Which states and categories combine material value with operational risk?\n5. Which sellers require operational review?\n6. How do customers pay, and where is instalment behaviour concentrated?\n\n## Generated evidence\n\nThe `data/analysis_*.csv` files and `analysis_summary.json` are regenerated by GitHub Actions from the pinned source dataset and checked alongside the Power BI/Tableau source contracts.\n"""
    insights_path = output_dir.parent / "VERIFIED_ANALYSIS.md"
    insights_path.write_text(insights, encoding="utf-8")

    return summary
