# Dashboard Story & UX Contract

## Design principle

Every page should answer one management question in under ten seconds, then allow a reviewer to drill into the evidence.

## Page 1 — Executive Pulse

**Question:** What is happening to marketplace scale and customer economics?

Top row: Commercial Orders · Unique Customers · GMV · AOV · Repeat Customer %.

Main visual: monthly merchandise value with order volume available as tooltip / secondary context.

Side callout: strongest complete month and a concise observation that repeat purchase is low relative to the size of the customer base.

## Page 2 — Category & Customer Mix

**Question:** Where is value coming from?

- ranked horizontal category GMV
- category orders vs average review
- state/category mix
- repeat-customer KPI carried through as context

Use rank rather than a rainbow palette. The visual hierarchy should emphasise the leading few categories and keep long tails visually quiet.

## Page 3 — Delivery & Experience

**Question:** Where might operational friction be affecting customer experience?

Primary comparison: average review score for on-time/early vs late deliveries.

Supporting evidence: delivered-order share and average days late.

Interview talking point: the 4.28 vs 2.55 review gap is large, but still observational rather than causal.

## Page 4 — Marketplace Health

**Question:** Are there dependencies or operational risks in the seller/payment ecosystem?

- top-10 seller share
- largest seller share
- seller value / late-rate / review table
- payment value and penetration

## Interaction contract

- consistent date/state/category filters where the underlying table supports them
- cross-highlighting should preserve analytical context
- tooltips show the metric definition, not only the value
- no 3D charts
- no decorative gauge where a number or bullet comparison is clearer
- red/green only when semantics are explicit, never as the sole encoding
- every percentage is labelled with its denominator in documentation

## Portfolio presentation

The GitHub preview is a static evidence board using retained verified values. The interactive Power BI/Tableau versions should preserve the same KPI hierarchy and metric definitions.
