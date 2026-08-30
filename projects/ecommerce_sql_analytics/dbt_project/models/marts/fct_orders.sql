select
    o.order_id,
    o.customer_id,
    c.customer_unique_id,
    c.customer_city,
    c.customer_state,
    o.order_status,
    o.purchase_ts,
    date_trunc('month', o.purchase_ts)::date as order_month,
    coalesce(i.item_count, 0) as item_count,
    coalesce(i.merchandise_value_brl, 0.0) as merchandise_value_brl,
    coalesce(i.freight_value_brl, 0.0) as freight_value_brl,
    coalesce(p.payment_rows, 0) as payment_rows,
    coalesce(p.payment_method_count, 0) as payment_method_count,
    coalesce(p.total_paid_brl, 0.0) as total_paid_brl,
    coalesce(p.max_installments, 0) as max_installments,
    r.review_score,
    (
        coalesce(i.item_count, 0) > 0
        and o.order_status not in ('canceled', 'unavailable')
        and o.purchase_ts >= timestamp '2017-01-01 00:00:00'
        and o.purchase_ts < timestamp '2018-09-01 00:00:00'
    ) as commercial_order,
    case
        when o.delivered_customer_ts is null or o.estimated_delivery_ts is null then null
        else o.delivered_customer_ts > o.estimated_delivery_ts
    end as delivered_late
from {{ ref('stg_orders') }} o
left join {{ ref('stg_customers') }} c using (customer_id)
left join {{ ref('int_order_items') }} i using (order_id)
left join {{ ref('int_payments') }} p using (order_id)
left join {{ ref('int_reviews') }} r using (order_id)
