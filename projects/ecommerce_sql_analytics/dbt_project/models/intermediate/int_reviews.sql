with ranked as (
    select
        order_id,
        review_score,
        try_cast(review_creation_date as timestamp) as review_creation_ts,
        try_cast(review_answer_timestamp as timestamp) as review_answer_ts,
        row_number() over (
            partition by order_id
            order by try_cast(review_answer_timestamp as timestamp) desc nulls last,
                     try_cast(review_creation_date as timestamp) desc nulls last,
                     review_id desc
        ) as review_rank
    from {{ source('raw', 'reviews') }}
)
select
    order_id,
    review_score,
    review_creation_ts,
    review_answer_ts
from ranked
where review_rank = 1
