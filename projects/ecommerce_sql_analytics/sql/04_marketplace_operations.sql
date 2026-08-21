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
    SELECT
        order_id,
        seller_id,
        MAX(seller_state) AS seller_state,
        SUM(item_price_brl) AS seller_order_value_brl,
        MAX(CASE WHEN delivered_late THEN 1 ELSE 0 END) AS delivered_late,
        MAX(review_score) AS review_score
    FROM analytics.item_mart
    WHERE commercial_order
    GROUP BY order_id, seller_id
),
seller_metrics AS (
    SELECT
        seller_id,
        seller_state,
        COUNT(*) AS orders,
        SUM(seller_order_value_brl) AS merchandise_value_brl,
        100.0 * AVG(delivered_late) AS late_delivery_rate_pct,
        AVG(review_score) AS average_review_score
    FROM order_seller
    GROUP BY seller_id, seller_state
    HAVING COUNT(*) >= 50
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
WITH commercial_orders AS (
    SELECT order_id
    FROM analytics.order_mart
    WHERE commercial_order
),
by_method AS (
    SELECT
        p.payment_type,
        COUNT(*) AS payment_rows,
        COUNT(DISTINCT p.order_id) AS orders,
        SUM(p.payment_value) AS payment_value_brl,
        AVG(p.payment_installments) AS average_installments
    FROM raw.payments p
    JOIN commercial_orders o USING (order_id)
    GROUP BY p.payment_type
),
benchmark AS (
    SELECT COUNT(*) AS commercial_orders
    FROM commercial_orders
)
SELECT
    payment_type,
    payment_rows,
    orders,
    ROUND(payment_value_brl, 2) AS payment_value_brl,
    ROUND(average_installments, 2) AS average_installments,
    ROUND(100.0 * orders / NULLIF(commercial_orders, 0), 2) AS order_penetration_pct
FROM by_method
CROSS JOIN benchmark
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
