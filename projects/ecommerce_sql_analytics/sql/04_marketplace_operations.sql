-- Product, delivery, seller and payment analysis.

CREATE OR REPLACE TABLE analytics.category_performance AS
WITH category_rollup AS (
    SELECT
        category_name,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(*) AS items,
        SUM(item_price_brl) AS merchandise_value_brl,
        SUM(item_freight_brl) AS freight_value_brl,
        AVG(review_score) AS average_review_score
    FROM analytics.item_mart
    WHERE commercial_order
    GROUP BY category_name
)
SELECT
    category_name,
    orders,
    items,
    ROUND(merchandise_value_brl, 2) AS merchandise_value_brl,
    ROUND(freight_value_brl, 2) AS freight_value_brl,
    ROUND(100.0 * freight_value_brl / NULLIF(merchandise_value_brl, 0), 2) AS freight_to_merchandise_pct,
    ROUND(average_review_score, 2) AS average_review_score,
    RANK() OVER (ORDER BY merchandise_value_brl DESC) AS merchandise_value_rank
FROM category_rollup
ORDER BY merchandise_value_rank, category_name;

CREATE OR REPLACE TABLE analytics.delivery_review_summary AS
SELECT
    CASE WHEN delivered_late THEN 'late' ELSE 'on_time' END AS delivery_status,
    COUNT(*) AS delivered_orders,
    ROUND(AVG(review_score), 2) AS average_review_score,
    ROUND(AVG(days_late), 2) AS average_days_late,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS delivered_order_share_pct
FROM analytics.order_mart
WHERE commercial_order
  AND delivered_late IS NOT NULL
GROUP BY delivery_status
ORDER BY delivery_status;

CREATE OR REPLACE TABLE analytics.seller_operational_review AS
WITH order_seller AS (
    SELECT DISTINCT
        i.order_id,
        i.seller_id,
        i.seller_state,
        i.delivered_late,
        i.review_score,
        i.item_price_brl
    FROM analytics.item_mart i
    WHERE i.commercial_order
),
seller_metrics AS (
    SELECT
        seller_id,
        seller_state,
        COUNT(DISTINCT order_id) AS orders,
        SUM(item_price_brl) AS merchandise_value_brl,
        100.0 * AVG(CASE WHEN delivered_late THEN 1.0 ELSE 0.0 END) AS late_delivery_rate_pct,
        AVG(review_score) AS average_review_score
    FROM order_seller
    GROUP BY seller_id, seller_state
    HAVING COUNT(DISTINCT order_id) >= 50
),
benchmark AS (
    SELECT AVG(late_delivery_rate_pct) AS average_seller_late_rate
    FROM seller_metrics
)
SELECT
    seller_id,
    seller_state,
    orders,
    ROUND(merchandise_value_brl, 2) AS merchandise_value_brl,
    ROUND(late_delivery_rate_pct, 2) AS late_delivery_rate_pct,
    ROUND(average_review_score, 2) AS average_review_score,
    CASE
        WHEN late_delivery_rate_pct > average_seller_late_rate + 10
         AND average_review_score < 3.8 THEN 'review_priority'
        ELSE 'monitor'
    END AS operational_status
FROM seller_metrics
CROSS JOIN benchmark
ORDER BY
    CASE WHEN operational_status = 'review_priority' THEN 0 ELSE 1 END,
    late_delivery_rate_pct DESC,
    orders DESC;

CREATE OR REPLACE TABLE analytics.payment_behaviour AS
SELECT
    p.payment_type,
    COUNT(*) AS payment_rows,
    COUNT(DISTINCT p.order_id) AS orders,
    ROUND(SUM(p.payment_value), 2) AS payment_value_brl,
    ROUND(AVG(p.payment_installments), 2) AS average_installments,
    ROUND(100.0 * COUNT(DISTINCT p.order_id) / SUM(COUNT(DISTINCT p.order_id)) OVER (), 2) AS order_share_index_pct
FROM raw.payments p
JOIN analytics.order_mart o USING (order_id)
WHERE o.commercial_order
GROUP BY p.payment_type
ORDER BY payment_value_brl DESC;

CREATE OR REPLACE TABLE analytics.top_categories_by_customer_state AS
WITH state_category AS (
    SELECT
        customer_state,
        category_name,
        COUNT(DISTINCT order_id) AS orders,
        SUM(item_price_brl) AS merchandise_value_brl
    FROM analytics.item_mart
    WHERE commercial_order
      AND customer_state IS NOT NULL
    GROUP BY customer_state, category_name
)
SELECT
    customer_state,
    category_name,
    orders,
    ROUND(merchandise_value_brl, 2) AS merchandise_value_brl,
    RANK() OVER (
        PARTITION BY customer_state
        ORDER BY merchandise_value_brl DESC
    ) AS category_rank
FROM state_category
QUALIFY category_rank <= 3
ORDER BY customer_state, category_rank, category_name;
