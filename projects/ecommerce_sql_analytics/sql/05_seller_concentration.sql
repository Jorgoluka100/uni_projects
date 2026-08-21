-- Marketplace concentration measured on merchandise value in complete calendar months.

CREATE OR REPLACE TABLE analytics.seller_concentration_detail AS
WITH seller_value AS (
    SELECT
        seller_id,
        SUM(item_price_brl) AS merchandise_value_brl
    FROM analytics.item_mart
    WHERE commercial_order
      AND order_month >= DATE '2017-01-01'
      AND order_month < DATE '2018-09-01'
    GROUP BY seller_id
),
scored AS (
    SELECT
        seller_id,
        merchandise_value_brl,
        100.0 * merchandise_value_brl / SUM(merchandise_value_brl) OVER () AS merchandise_value_share_pct,
        ROW_NUMBER() OVER (ORDER BY merchandise_value_brl DESC, seller_id) AS seller_rank
    FROM seller_value
)
SELECT
    seller_id,
    seller_rank,
    ROUND(merchandise_value_brl, 2) AS merchandise_value_brl,
    ROUND(merchandise_value_share_pct, 4) AS merchandise_value_share_pct,
    ROUND(SUM(merchandise_value_share_pct) OVER (ORDER BY seller_rank ROWS UNBOUNDED PRECEDING), 4)
        AS cumulative_merchandise_value_share_pct
FROM scored
ORDER BY seller_rank;

CREATE OR REPLACE TABLE analytics.seller_concentration_summary AS
SELECT
    COUNT(*) AS active_sellers,
    ROUND(MAX(merchandise_value_share_pct), 2) AS largest_seller_share_pct,
    ROUND(SUM(merchandise_value_share_pct) FILTER (WHERE seller_rank <= 10), 2) AS top_ten_seller_share_pct,
    ROUND(SUM(POWER(merchandise_value_share_pct, 2)), 2) AS hhi_index
FROM analytics.seller_concentration_detail;
