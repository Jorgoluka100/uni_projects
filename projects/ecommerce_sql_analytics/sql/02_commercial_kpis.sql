-- Headline commercial KPIs and complete-month trend reporting.

CREATE OR REPLACE TABLE analytics.headline_kpis AS
SELECT
    COUNT(*) FILTER (WHERE commercial_order) AS commercial_orders,
    COUNT(DISTINCT customer_unique_id) FILTER (WHERE commercial_order) AS unique_customers,
    ROUND(SUM(merchandise_value_brl) FILTER (WHERE commercial_order), 2) AS merchandise_value_brl,
    ROUND(AVG(merchandise_value_brl) FILTER (WHERE commercial_order), 2) AS average_order_value_brl,
    ROUND(SUM(freight_value_brl) FILTER (WHERE commercial_order), 2) AS freight_value_brl,
    ROUND(SUM(total_paid_brl) FILTER (WHERE commercial_order), 2) AS total_payment_value_brl
FROM analytics.order_mart;

CREATE OR REPLACE TABLE analytics.monthly_performance AS
WITH monthly AS (
    SELECT
        order_month,
        COUNT(*) AS orders,
        COUNT(DISTINCT customer_unique_id) AS customers,
        SUM(merchandise_value_brl) AS merchandise_value_brl,
        AVG(merchandise_value_brl) AS average_order_value_brl
    FROM analytics.order_mart
    WHERE commercial_order
      AND order_month >= DATE '2017-01-01'
      AND order_month < DATE '2018-09-01'
    GROUP BY order_month
)
SELECT
    order_month,
    orders,
    customers,
    ROUND(merchandise_value_brl, 2) AS merchandise_value_brl,
    ROUND(average_order_value_brl, 2) AS average_order_value_brl,
    ROUND(
        100.0 * (merchandise_value_brl - LAG(merchandise_value_brl) OVER (ORDER BY order_month))
        / NULLIF(LAG(merchandise_value_brl) OVER (ORDER BY order_month), 0),
        2
    ) AS merchandise_value_mom_pct,
    RANK() OVER (ORDER BY merchandise_value_brl DESC) AS merchandise_value_rank
FROM monthly
ORDER BY order_month;

CREATE OR REPLACE TABLE analytics.customer_order_frequency AS
WITH customer_orders AS (
    SELECT
        customer_unique_id,
        COUNT(*) AS order_count,
        SUM(merchandise_value_brl) AS lifetime_merchandise_value_brl
    FROM analytics.order_mart
    WHERE commercial_order
      AND order_month >= DATE '2017-01-01'
      AND order_month < DATE '2018-09-01'
    GROUP BY customer_unique_id
)
SELECT
    COUNT(*) AS customers,
    COUNT(*) FILTER (WHERE order_count >= 2) AS repeat_customers,
    ROUND(100.0 * COUNT(*) FILTER (WHERE order_count >= 2) / COUNT(*), 2) AS repeat_customer_pct,
    ROUND(AVG(order_count), 3) AS average_orders_per_customer,
    ROUND(AVG(lifetime_merchandise_value_brl), 2) AS average_customer_merchandise_value_brl
FROM customer_orders;
