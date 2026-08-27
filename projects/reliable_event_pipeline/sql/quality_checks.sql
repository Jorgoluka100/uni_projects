-- Duplicate business keys: must return zero rows.
SELECT event_id, COUNT(*) AS n
FROM events
GROUP BY event_id
HAVING COUNT(*) > 1;

-- Required identifiers: must be zero.
SELECT COUNT(*) AS null_customer_ids
FROM events
WHERE customer_id IS NULL OR TRIM(customer_id) = '';

-- Late arrivals are retained and visible rather than silently discarded.
SELECT batch_name, COUNT(*) AS late_rows
FROM events
WHERE is_late = 1
GROUP BY batch_name;

-- Reconciliation by source for quick pipeline investigation.
SELECT source, COUNT(*) AS events, ROUND(SUM(amount), 2) AS total_amount
FROM events
GROUP BY source
ORDER BY source;
