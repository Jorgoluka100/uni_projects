select
    order_id,
    count(*) as item_count,
    sum(coalesce(price, 0)) as merchandise_value_brl,
    sum(coalesce(freight_value, 0)) as freight_value_brl
from {{ source('raw', 'order_items') }}
group by order_id
