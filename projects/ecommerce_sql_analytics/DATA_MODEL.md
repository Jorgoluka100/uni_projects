# Data model and grain contract

The original Olist dataset contains nine related files. The central reporting problem is that several child tables have different grains: an order can contain multiple items, multiple payment rows and more than one review record. Joining those raw tables directly can multiply monetary values.

## Semantic layer

```text
raw.customers ───────┐
                     │
raw.orders ──────────┼────> analytics.order_mart   [1 row / order]
                     │             │
raw.order_items ─────┤             │
raw.payments ────────┤             └────> order/customer/delivery KPIs
raw.reviews ─────────┘

raw.order_items ─────────> analytics.item_mart    [1 row / order item]
raw.products ────────────────┤
raw.category_translation ────┤
raw.sellers ─────────────────┘
                             └────> category/seller/product KPIs
```

## `analytics.order_mart`

Exactly one row per `order_id`. Before joining to orders:

- order items are aggregated to order-level merchandise and freight totals;
- payment rows are aggregated to order-level payment totals and method counts;
- reviews are ranked and reduced to one record per order.

Use this mart for orders, customers, payments, delivery performance and review analysis.

## `analytics.item_mart`

Exactly one row per order item. It enriches items with product category, seller and order-level context. Use this mart for merchandise categories, seller mix and product analysis.

## Commercial-order definition

A record is considered a commercial order when it has at least one order item and its status is not `canceled` or `unavailable`. The definition is explicit in SQL so metrics are reproducible rather than dependent on an analyst's ad-hoc filter.

## Integrity contract

The project fails its release checks when:

- a core source primary key is duplicated;
- a required source foreign key is orphaned;
- `order_mart` is not exactly one row per source order;
- `item_mart` does not reconcile to the source item count;
- merchandise value differs between the order and item grains;
- negative item/freight values or invalid review scores appear.
