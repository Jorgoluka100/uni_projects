select
    order_id,
    customer_id,
    lower(order_status) as order_status,
    try_cast(order_purchase_timestamp as timestamp) as purchase_ts,
    try_cast(order_delivered_customer_date as timestamp) as delivered_customer_ts,
    try_cast(order_estimated_delivery_date as timestamp) as estimated_delivery_ts
from {{ source('raw', 'orders') }}
