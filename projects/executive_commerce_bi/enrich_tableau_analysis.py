"""Add the verified customer/service analysis to Tableau's tidy long-form extract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data"
TABLEAU = DATA / "tableau_dashboard_long.csv"
MANIFEST = DATA / "manifest.json"
DEEP_SECTIONS = {"Customer Segment", "Delay Impact", "Operational Priority"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base = pd.read_csv(TABLEAU)
    base = base.loc[~base["section"].isin(DEEP_SECTIONS)].copy()
    next_sort = int(base["sort_order"].max()) + 1
    rows: list[dict[str, object]] = []

    customer = pd.read_csv(DATA / "analysis_customer_segments.csv")
    for i, row in customer.iterrows():
        rows.append({
            "section": "Customer Segment",
            "dimension": str(row["customer_segment"]).replace("_", " ").title(),
            "metric": "GMV Share",
            "value": row["merchandise_value_share_pct"],
            "secondary_value": row["customer_share_pct"],
            "sort_order": next_sort + int(i),
        })
    next_sort += len(customer)

    delivery = pd.read_csv(DATA / "analysis_delivery_impact.csv")
    for i, row in delivery.iterrows():
        rows.append({
            "section": "Delay Impact",
            "dimension": row["delivery_bucket"],
            "metric": "Average Review",
            "value": row["average_review_score"],
            "secondary_value": row["one_star_review_pct"],
            "sort_order": next_sort + int(i),
        })
    next_sort += len(delivery)

    priority = pd.read_csv(DATA / "analysis_operational_priority.csv").head(15)
    for i, row in priority.iterrows():
        rows.append({
            "section": "Operational Priority",
            "dimension": f"{row['entity_name']} ({row['entity_type']})",
            "metric": "Late Order GMV",
            "value": row["late_order_merchandise_value_brl"],
            "secondary_value": row["late_delivery_pct"],
            "sort_order": next_sort + int(i),
        })

    enriched = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    enriched.to_csv(TABLEAU, index=False)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["files"][TABLEAU.name] = {
        "rows": int(len(enriched)),
        "columns": list(enriched.columns),
        "sha256": sha256(TABLEAU),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Tableau deep-analysis extract: {len(enriched)} rows")


if __name__ == "__main__":
    main()
