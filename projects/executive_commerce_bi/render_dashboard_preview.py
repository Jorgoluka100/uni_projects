"""Render the GitHub dashboard preview directly from verified BI analysis outputs."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data"
OUTPUT = PROJECT / "dashboard_preview.svg"
WIDTH = 1440
HEIGHT = 980


def money_short(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"R${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"R${value / 1_000:.0f}K"
    return f"R${value:,.0f}"


def text(x: float, y: float, value: object, size: int = 18, weight: int = 400, anchor: str = "start", fill: str = "#172033") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Inter,Segoe UI,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{escape(str(value))}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str = "#ffffff", rx: int = 18, stroke: str = "#e7eaf0") -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def line_chart(monthly: pd.DataFrame, x: float, y: float, w: float, h: float) -> list[str]:
    values = monthly["merchandise_value_brl"].astype(float).tolist()
    labels = pd.to_datetime(monthly["order_month"]).dt.strftime("%b %y").tolist()
    vmin, vmax = min(values), max(values)
    usable_h = h - 55
    usable_w = w - 20
    points = []
    for i, value in enumerate(values):
        px = x + 10 + usable_w * i / max(1, len(values) - 1)
        py = y + 15 + usable_h * (1 - (value - vmin) / max(1.0, vmax - vmin))
        points.append((px, py))
    svg = [f'<polyline points="{" ".join(f"{px:.1f},{py:.1f}" for px, py in points)}" fill="none" stroke="#2563eb" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>']
    for px, py in points:
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="#2563eb"/>')
    for idx in [0, 5, 10, 15, len(labels) - 1]:
        if 0 <= idx < len(labels):
            px, _ = points[idx]
            svg.append(text(px, y + h - 8, labels[idx], 12, 400, "middle", "#687386"))
    return svg


def horizontal_bars(frame: pd.DataFrame, label_col: str, value_col: str, x: float, y: float, w: float, row_h: float, formatter, bar_fill: str = "#0f766e") -> list[str]:
    svg: list[str] = []
    max_value = max(float(frame[value_col].max()), 1.0)
    label_w = 120
    for i, row in frame.reset_index(drop=True).iterrows():
        yy = y + i * row_h
        label = str(row[label_col])
        value = float(row[value_col])
        svg.append(text(x, yy + 18, label, 14, 500, "start", "#384152"))
        bx = x + label_w
        bw = (w - label_w - 95) * value / max_value
        svg.append(f'<rect x="{bx}" y="{yy + 5}" width="{max(2, bw):.1f}" height="16" rx="8" fill="{bar_fill}" opacity="0.88"/>')
        svg.append(text(x + w, yy + 18, formatter(value), 13, 600, "end", "#172033"))
    return svg


def main() -> None:
    summary = json.loads((DATA / "analysis_summary.json").read_text(encoding="utf-8"))
    monthly = pd.read_csv(DATA / "analysis_monthly_growth.csv")
    delivery = pd.read_csv(DATA / "analysis_delivery_impact.csv")
    customers = pd.read_csv(DATA / "analysis_customer_segments.csv")
    priority = pd.read_csv(DATA / "analysis_operational_priority.csv").head(6)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#f6f8fb"/>',
        text(48, 54, "Executive Commerce Intelligence", 30, 700),
        text(48, 84, "Verified Olist marketplace analysis · Power BI + Tableau portfolio", 15, 400, "start", "#687386"),
        text(1392, 54, "GitHub-generated from verified data", 13, 600, "end", "#2563eb"),
    ]

    cards = [
        ("Commercial orders", f"{summary['scope']['commercial_orders']:,}", "#2563eb"),
        ("Unique customers", f"{summary['scope']['unique_customers']:,}", "#0f766e"),
        ("Merchandise value", money_short(summary['scope']['merchandise_value_brl']), "#7c3aed"),
        ("Repeat customers", f"{summary['customers']['repeat_customer_pct']:.2f}%", "#b45309"),
    ]
    card_w = 318
    for i, (label, value, accent) in enumerate(cards):
        x = 48 + i * 338
        parts.append(rect(x, 112, card_w, 116))
        parts.append(f'<rect x="{x}" y="112" width="7" height="116" rx="4" fill="{accent}" stroke="none"/>')
        parts.append(text(x + 24, 150, label, 14, 600, "start", "#687386"))
        parts.append(text(x + 24, 198, value, 30, 700))

    # Growth panel
    parts.append(rect(48, 252, 850, 300))
    parts.append(text(72, 286, "Commercial growth", 20, 700))
    parts.append(text(72, 310, "Complete-month GMV · Jan 2017 to Aug 2018", 13, 400, "start", "#687386"))
    parts.extend(line_chart(monthly, 72, 330, 802, 195))
    peak = summary["growth"]
    parts.append(text(850, 286, f"Peak {peak['strongest_complete_month']}", 13, 600, "end", "#2563eb"))
    parts.append(text(850, 308, money_short(peak["strongest_month_merchandise_value_brl"]), 17, 700, "end"))

    # Delivery panel
    parts.append(rect(922, 252, 470, 300))
    parts.append(text(946, 286, "Delivery → customer experience", 20, 700))
    parts.append(text(946, 310, "Average review score by delivery delay", 13, 400, "start", "#687386"))
    parts.extend(horizontal_bars(delivery, "delivery_bucket", "average_review_score", 946, 336, 420, 39, lambda v: f"{v:.2f}/5", "#dc2626"))
    parts.append(text(946, 532, f"Late-order share: {summary['delivery']['late_delivery_share_pct']:.2f}% · review gap: {summary['delivery']['review_score_gap']:.2f} points", 13, 600, "start", "#9f1239"))

    # Customer segment panel
    parts.append(rect(48, 576, 610, 344))
    parts.append(text(72, 610, "Customer value structure", 20, 700))
    parts.append(text(72, 634, "GMV share by customer segment", 13, 400, "start", "#687386"))
    customer_view = customers[["customer_segment", "merchandise_value_share_pct"]].copy()
    customer_view["customer_segment"] = customer_view["customer_segment"].str.replace("_", " ").str.title()
    parts.extend(horizontal_bars(customer_view, "customer_segment", "merchandise_value_share_pct", 72, 662, 560, 46, lambda v: f"{v:.1f}%", "#7c3aed"))
    parts.append(text(72, 876, "Only 3.03% of customers repeat; high-value one-time buyers dominate value.", 13, 600, "start", "#5b21b6"))

    # Priority panel
    parts.append(rect(682, 576, 710, 344))
    parts.append(text(706, 610, "Operational priority", 20, 700))
    parts.append(text(706, 634, "Material value currently attached to late orders", 13, 400, "start", "#687386"))
    priority_view = priority.copy()
    priority_view["entity_name"] = priority_view.apply(lambda r: f"{r['entity_name']} ({r['entity_type']})", axis=1)
    parts.extend(horizontal_bars(priority_view, "entity_name", "late_order_merchandise_value_brl", 706, 660, 650, 37, money_short, "#0f766e"))
    parts.append(text(706, 892, "Priority is ranked by late-order GMV — transparent, auditable, and decision-oriented.", 13, 600, "start", "#0f766e"))

    parts.append(text(48, 956, "Source: Olist Brazilian e-commerce public dataset · pinned dataset version + hash checks · generated by GitHub Actions", 12, 400, "start", "#7b8495"))
    parts.append("</svg>")
    OUTPUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
