-- Aviation Strategy PostgreSQL v2
--
-- Purpose: replace the ambiguous "PostgreSQL-compatible DuckDB" claim in the restored
-- notebook with a genuinely reproducible PostgreSQL workload. This script creates a
-- deterministic synthetic aviation fact table, runs an analytical route/fleet query,
-- captures EXPLAIN (ANALYZE, BUFFERS) before and after targeted indexes, and preserves
-- query semantics. Synthetic business figures are methodology evidence only.

BEGIN;

DROP TABLE IF EXISTS flight_operations;
DROP TABLE IF EXISTS aircraft_assets;

CREATE TABLE aircraft_assets (
    aircraft_id text PRIMARY KEY,
    model_name text NOT NULL,
    seat_capacity integer NOT NULL CHECK (seat_capacity > 0),
    fuel_burn_rate_index numeric(8,4) NOT NULL CHECK (fuel_burn_rate_index > 0)
);

CREATE TABLE flight_operations (
    flight_id bigint PRIMARY KEY,
    departure_date date NOT NULL,
    route_name text NOT NULL,
    aircraft_id text NOT NULL REFERENCES aircraft_assets(aircraft_id),
    ticket_revenue numeric(14,2) NOT NULL CHECK (ticket_revenue >= 0),
    ancillary_revenue numeric(14,2) NOT NULL CHECK (ancillary_revenue >= 0),
    load_factor numeric(6,5) NOT NULL CHECK (load_factor BETWEEN 0 AND 1),
    operating_cost numeric(14,2) NOT NULL CHECK (operating_cost >= 0)
);

INSERT INTO aircraft_assets VALUES
('B738', 'Boeing 737-800', 189, 1.0500),
('B73M', 'Boeing 737-MAX8200', 197, 0.8400),
('A320', 'Airbus A320neo', 186, 0.8800);

-- 1,000,000 deterministic rows so query-planning differences are visible.
INSERT INTO flight_operations
SELECT
    g AS flight_id,
    DATE '2025-01-01' + ((g - 1) % 730)::integer AS departure_date,
    (ARRAY['DUB-STN','STN-BCN','DUB-CDG','BER-STN','MAD-DUB','STN-FCO'])[((g - 1) % 6) + 1] AS route_name,
    (ARRAY['B738','B73M','A320'])[((g - 1) % 3) + 1] AS aircraft_id,
    round((7000 + ((g * 37) % 9000))::numeric, 2) AS ticket_revenue,
    round((1800 + ((g * 19) % 4200))::numeric, 2) AS ancillary_revenue,
    round((0.70 + (((g * 13) % 2900) / 10000.0))::numeric, 5) AS load_factor,
    round((5600 + ((g * 29) % 5200))::numeric, 2) AS operating_cost
FROM generate_series(1, 1000000) AS g;

ANALYZE aircraft_assets;
ANALYZE flight_operations;

-- Data-quality and reconciliation checks.
DO $$
BEGIN
    IF (SELECT count(*) FROM flight_operations) <> 1000000 THEN
        RAISE EXCEPTION 'unexpected fact-row count';
    END IF;
    IF EXISTS (
        SELECT 1 FROM flight_operations f
        LEFT JOIN aircraft_assets a USING (aircraft_id)
        WHERE a.aircraft_id IS NULL
    ) THEN
        RAISE EXCEPTION 'orphan aircraft key';
    END IF;
END $$;

-- BEFORE INDEX: retain this plan when running in psql with \o or copy it into an
-- interview artefact. The half-open date predicate is deliberately sargable.
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT
    f.route_name,
    a.model_name,
    count(*) AS total_flights,
    round(sum(f.ticket_revenue + f.ancillary_revenue), 2) AS gross_revenue,
    round(sum(f.operating_cost), 2) AS total_costs,
    round(avg(f.load_factor) * 100, 2) AS avg_occupancy_pct,
    round(sum(f.ticket_revenue + f.ancillary_revenue - f.operating_cost), 2) AS net_profit
FROM flight_operations f
JOIN aircraft_assets a ON a.aircraft_id = f.aircraft_id
WHERE f.departure_date >= DATE '2026-01-01'
  AND f.departure_date <  DATE '2026-04-01'
  AND f.route_name IN ('DUB-STN','STN-BCN')
GROUP BY f.route_name, a.model_name
ORDER BY net_profit DESC;

-- Query-aligned index: equality on route first, then date range, with aircraft key
-- and monetary columns available to reduce heap work where PostgreSQL can exploit it.
CREATE INDEX idx_flight_ops_route_date
    ON flight_operations (route_name, departure_date, aircraft_id)
    INCLUDE (ticket_revenue, ancillary_revenue, operating_cost, load_factor);

ANALYZE flight_operations;

-- AFTER INDEX: compare actual time, shared hit/read blocks, scan type and rows removed.
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT
    f.route_name,
    a.model_name,
    count(*) AS total_flights,
    round(sum(f.ticket_revenue + f.ancillary_revenue), 2) AS gross_revenue,
    round(sum(f.operating_cost), 2) AS total_costs,
    round(avg(f.load_factor) * 100, 2) AS avg_occupancy_pct,
    round(sum(f.ticket_revenue + f.ancillary_revenue - f.operating_cost), 2) AS net_profit
FROM flight_operations f
JOIN aircraft_assets a ON a.aircraft_id = f.aircraft_id
WHERE f.departure_date >= DATE '2026-01-01'
  AND f.departure_date <  DATE '2026-04-01'
  AND f.route_name IN ('DUB-STN','STN-BCN')
GROUP BY f.route_name, a.model_name
ORDER BY net_profit DESC;

-- Semantic reconciliation: total net profit must equal revenue minus cost at the
-- underlying filtered grain.
WITH detailed AS (
    SELECT sum(ticket_revenue + ancillary_revenue - operating_cost) AS net
    FROM flight_operations
    WHERE departure_date >= DATE '2026-01-01'
      AND departure_date < DATE '2026-04-01'
      AND route_name IN ('DUB-STN','STN-BCN')
), grouped AS (
    SELECT sum(net_profit) AS net
    FROM (
        SELECT route_name, aircraft_id,
               sum(ticket_revenue + ancillary_revenue - operating_cost) AS net_profit
        FROM flight_operations
        WHERE departure_date >= DATE '2026-01-01'
          AND departure_date < DATE '2026-04-01'
          AND route_name IN ('DUB-STN','STN-BCN')
        GROUP BY route_name, aircraft_id
    ) x
)
SELECT detailed.net AS detailed_net, grouped.net AS grouped_net,
       detailed.net = grouped.net AS reconciled
FROM detailed CROSS JOIN grouped;

COMMIT;
