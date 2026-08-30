select
    customer_id,
    customer_unique_id,
    nullif(trim(customer_city), '') as customer_city,
    upper(customer_state) as customer_state
from {{ source('raw', 'customers') }}
