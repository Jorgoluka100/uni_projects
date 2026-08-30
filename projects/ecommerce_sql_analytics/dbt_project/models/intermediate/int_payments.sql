select
    order_id,
    count(*) as payment_rows,
    count(distinct payment_type) as payment_method_count,
    sum(coalesce(payment_value, 0)) as total_paid_brl,
    max(coalesce(payment_installments, 0)) as max_installments
from {{ source('raw', 'payments') }}
group by order_id
