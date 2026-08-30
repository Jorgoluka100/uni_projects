# KPI Dictionary

The same definitions are used by Power BI and Tableau. Dashboard logic should not redefine these metrics independently.

| KPI | Definition | Grain / caveat |
| --- | --- | --- |
| Commercial Orders | Count of orders that contain at least one item, exclude cancelled/unavailable orders, and fall in the comparable reporting window | 1 row per order |
| Unique Customers | Distinct `customer_unique_id` over commercial orders | Source identity is anonymised |
| Merchandise Value | Sum of item price, excluding freight | Revenue-like marketplace value, **not profit** |
| Average Order Value | Merchandise value / commercial orders | Order grain |
| Repeat Customer % | Customers with 2+ commercial orders / customers | Complete-month reporting window |
| Monthly GMV Growth | Month-over-month change in merchandise value | First month is null |
| Category GMV | Sum of item price by translated category | Item grain |
| Average Review | Mean retained review score | Latest retained review per order in upstream mart |
| Late Delivery | Actual customer delivery timestamp > estimated delivery timestamp | Orders with both timestamps available |
| Delivery Share | Delivered orders in status / all comparable delivered orders | Excludes unknown delivery status |
| Top-10 Seller Share | Merchandise-value share held by ten largest sellers | Concentration indicator, not a risk conclusion by itself |
| Payment Penetration | Orders using a payment type / commercial orders | One order can use multiple payment rows/types |

## Reporting guardrails

1. Never join raw order items directly to raw payments for revenue reporting.
2. Merchandise value and payment value answer different questions and should not be treated as interchangeable.
3. Late-delivery and review results are observational; the dashboard does not claim causality.
4. Historical Olist data does not represent current Olist performance.
5. Later acquisition cohorts have less time to mature.
