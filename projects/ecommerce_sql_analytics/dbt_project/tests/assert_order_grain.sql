-- The final mart must contain exactly one row per source order.
select
    order_id,
    count(*) as row_count
from {{ ref('fct_orders') }}
group by order_id
having count(*) != 1
