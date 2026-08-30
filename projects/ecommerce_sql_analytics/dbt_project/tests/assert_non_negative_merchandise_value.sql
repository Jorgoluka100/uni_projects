select *
from {{ ref('fct_orders') }}
where merchandise_value_brl < 0
