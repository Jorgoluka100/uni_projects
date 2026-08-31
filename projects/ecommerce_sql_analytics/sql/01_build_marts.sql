-- Build semantic marts at explicit grains.
-- order_mart: exactly one row per order.
-- item_mart: exactly one row per order item.
-- Child tables are aggregated before joining to orders to prevent many-to-many
-- multiplication of merchandise value, payments or review records.
--
-- Scope contract:
-- commercial_order represents the full eligible commercial population: an order
-- must contain at least one item and must not be cancelled or unavailable.
-- Comparable complete-month reporting windows are applied by downstream analyses
-- (for example monthly trends and repeat-customer analysis), not baked into this
-- reusable population flag.

CREATE OR REPLACE TABLE analytics.order_mart AS
WITH item_agg AS (
    SELECT
        order_id,
        COUNT(*) AS item_count,
        SUM(COALESCE(price, 0)) AS merchandise_value_brl,
        SUM(COALESCE(freight_value, 0)) AS freight_value_brl
    FROM raw.order_items
    GROUP BY order_id
),
payment_agg AS (
    SELECT
        order_id,
        COUNT(*) AS payment_rows,
        COUNT(DISTINCT payment_type) AS payment_method_count,
        SUM(COALESCE(payment_value, 0)) AS total_paid_brl,
        MAX(COALESCE(payment_installments, 0)) AS max_installments
    FROM raw.payments
    GROUP BY order_id
),
review_ranked AS (
    SELECT
        order_id,
        review_score,
        TRY_CAST(review_creation_date AS TIMESTAMP) AS review_creation_ts,
        TRY_CAST(review_answer_timestamp AS TIMESTAMP) AS review_answer_ts,
        ROW_NUMBER() OVER (
            PARTITION BY order_id
            ORDER BY TRY_CAST(review_answer_timestamp AS TIMESTAMP) DESC NULLS LAST,
                     TRY_CAST(review_creation_date AS TIMESTAMP) DESC NULLS LAST,
                     review_id DESC
        ) AS review_rank
    FROM raw.reviews
),
review_one AS (
    SELECT order_id, review_score, review_creation_ts, review_answer_ts
    FROM review_ranked
    WHERE review_rank = 1
)
SELECT
    o.order_id,
    o.customer_id,
    c.customer_unique_id,
    c.customer_city,
    c.customer_state,
    o.order_status,
    TRY_CAST(o.order_purchase_timestamp AS TIMESTAMP) AS purchase_ts,
    DATE_TRUNC('month', TRY_CAST(o.order_purchase_timestamp AS TIMESTAMP))::DATE AS order_month,
    TRY_CAST(o.order_approved_at AS TIMESTAMP) AS approved_ts,
    TRY_CAST(o.order_delivered_carrier_date AS TIMESTAMP) AS carrier_handoff_ts,
    TRY_CAST(o.order_delivered_customer_date AS TIMESTAMP) AS delivered_customer_ts,
    TRY_CAST(o.order_estimated_delivery_date AS TIMESTAMP) AS estimated_delivery_ts,
    COALESCE(i.item_count, 0) AS item_count,
    COALESCE(i.merchandise_value_brl, 0.0) AS merchandise_value_brl,
    COALESCE(i.freight_value_brl, 0.0) AS freight_value_brl,
    COALESCE(p.payment_rows, 0) AS payment_rows,
    COALESCE(p.payment_method_count, 0) AS payment_method_count,
    COALESCE(p.total_paid_brl, 0.0) AS total_paid_brl,
    COALESCE(p.max_installments, 0) AS max_installments,
    r.review_score,
    r.review_creation_ts,
    r.review_answer_ts,
    (
        COALESCE(i.item_count, 0) > 0
        AND LOWER(COALESCE(o.order_status, '')) NOT IN ('canceled', 'unavailable')
    ) AS commercial_order,
    CASE
        WHEN TRY_CAST(o.order_delivered_customer_date AS TIMESTAMP) IS NULL
          OR TRY_CAST(o.order_estimated_delivery_date AS TIMESTAMP) IS NULL THEN NULL
        ELSE TRY_CAST(o.order_delivered_customer_date AS TIMESTAMP)
             > TRY_CAST(o.order_estimated_delivery_date AS TIMESTAMP)
    END AS delivered_late,
    CASE
        WHEN TRY_CAST(o.order_delivered_customer_date AS TIMESTAMP) IS NULL
          OR TRY_CAST(o.order_estimated_delivery_date AS TIMESTAMP) IS NULL THEN NULL
        ELSE GREATEST(
            DATE_DIFF(
                'day',
                TRY_CAST(o.order_estimated_delivery_date AS TIMESTAMP),
                TRY_CAST(o.order_delivered_customer_date AS TIMESTAMP)
            ),
            0
        )
    END AS days_late
FROM raw.orders o
LEFT JOIN raw.customers c USING (customer_id)
LEFT JOIN item_agg i USING (order_id)
LEFT JOIN payment_agg p USING (order_id)
LEFT JOIN review_one r USING (order_id);

CREATE OR REPLACE TABLE analytics.item_mart AS
SELECT
    i.order_id,
    i.order_item_id,
    i.product_id,
    i.seller_id,
    COALESCE(i.price, 0.0) AS item_price_brl,
    COALESCE(i.freight_value, 0.0) AS item_freight_brl,
    p.product_category_name,
    COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS category_name,
    s.seller_city,
    s.seller_state,
    o.customer_unique_id,
    o.customer_state,
    o.order_status,
    o.purchase_ts,
    o.order_month,
    o.commercial_order,
    o.delivered_late,
    o.days_late,
    o.review_score
FROM raw.order_items i
JOIN analytics.order_mart o USING (order_id)
LEFT JOIN raw.products p USING (product_id)
LEFT JOIN raw.category_translation t USING (product_category_name)
LEFT JOIN raw.sellers s USING (seller_id);