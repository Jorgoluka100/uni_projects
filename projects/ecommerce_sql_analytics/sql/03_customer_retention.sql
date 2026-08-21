-- Acquisition cohorts use the first observed commercial order across full history.
-- Cohorts are displayed only for complete months. Future months are not manufactured
-- as zeroes, so later cohorts are naturally right-censored.

CREATE OR REPLACE TABLE analytics.cohort_retention AS
WITH customer_months AS (
    SELECT DISTINCT
        customer_unique_id,
        order_month
    FROM analytics.order_mart
    WHERE commercial_order
      AND customer_unique_id IS NOT NULL
),
customer_cohorts AS (
    SELECT
        customer_unique_id,
        MIN(order_month) AS cohort_month
    FROM customer_months
    GROUP BY customer_unique_id
),
activity AS (
    SELECT
        c.cohort_month,
        m.order_month,
        DATE_DIFF('month', c.cohort_month, m.order_month) AS month_number,
        COUNT(DISTINCT m.customer_unique_id) AS active_customers
    FROM customer_months m
    JOIN customer_cohorts c USING (customer_unique_id)
    WHERE c.cohort_month >= DATE '2017-01-01'
      AND c.cohort_month < DATE '2018-09-01'
      AND m.order_month < DATE '2018-09-01'
    GROUP BY c.cohort_month, m.order_month, month_number
),
cohort_sizes AS (
    SELECT cohort_month, active_customers AS cohort_customers
    FROM activity
    WHERE month_number = 0
)
SELECT
    a.cohort_month,
    a.month_number,
    a.active_customers,
    s.cohort_customers,
    ROUND(100.0 * a.active_customers / NULLIF(s.cohort_customers, 0), 2) AS retention_pct
FROM activity a
JOIN cohort_sizes s USING (cohort_month)
WHERE a.month_number BETWEEN 0 AND 12
ORDER BY a.cohort_month, a.month_number;

CREATE OR REPLACE TABLE analytics.customer_value_segments AS
WITH customer_value AS (
    SELECT
        customer_unique_id,
        COUNT(*) AS orders,
        SUM(merchandise_value_brl) AS merchandise_value_brl,
        MAX(purchase_ts) AS last_purchase_ts
    FROM analytics.order_mart
    WHERE commercial_order
      AND customer_unique_id IS NOT NULL
    GROUP BY customer_unique_id
),
scored AS (
    SELECT
        *,
        NTILE(5) OVER (ORDER BY merchandise_value_brl) AS value_quintile,
        NTILE(5) OVER (ORDER BY last_purchase_ts) AS recency_quintile
    FROM customer_value
)
SELECT
    customer_unique_id,
    orders,
    ROUND(merchandise_value_brl, 2) AS merchandise_value_brl,
    last_purchase_ts,
    value_quintile,
    recency_quintile,
    CASE
        WHEN orders >= 2 AND value_quintile >= 4 THEN 'repeat_high_value'
        WHEN orders >= 2 THEN 'repeat'
        WHEN value_quintile = 5 THEN 'one_time_high_value'
        ELSE 'one_time'
    END AS customer_segment
FROM scored;
