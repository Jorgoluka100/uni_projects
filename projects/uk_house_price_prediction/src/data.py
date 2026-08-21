from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import duckdb
import pandas as pd

SOURCE_URLS = {
    2025: "https://price-paid-data.publicdata.landregistry.gov.uk/pp-2025.csv",
    2026: "https://price-paid-data.publicdata.landregistry.gov.uk/pp-2026.csv",
}

MODEL_START = pd.Timestamp("2025-01-01")
VALIDATION_START = pd.Timestamp("2025-10-01")
TEST_START = pd.Timestamp("2026-01-01")
TEST_END = pd.Timestamp("2026-07-01")

CATEGORICAL_FEATURES = [
    "postcode_district",
    "postcode_area",
    "property_type",
    "old_new",
    "duration",
    "town_city",
    "district",
    "county",
]
NUMERIC_FEATURES = ["transfer_year", "transfer_month", "month_sin", "month_cos"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def download_price_paid(cache_dir: Path) -> dict[int, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    for year, url in SOURCE_URLS.items():
        path = cache_dir / f"pp-{year}.csv"
        if not path.exists():
            urlretrieve(url, path)
        paths[year] = path
    return paths


def load_residential(paths: dict[int, Path]) -> pd.DataFrame:
    """Load the exact modelling population used in the verified notebook."""
    con = duckdb.connect(database=":memory:")
    files = ",".join(repr(str(path)) for path in paths.values())
    con.execute(
        f"""
        CREATE VIEW price_paid AS
        SELECT
            column00 AS transaction_id,
            CAST(column01 AS BIGINT) AS price,
            CAST(strptime(column02, '%Y-%m-%d %H:%M') AS DATE) AS transfer_date,
            NULLIF(trim(column03), '') AS postcode,
            column04 AS property_type,
            column05 AS old_new,
            column06 AS duration,
            column11 AS town_city,
            column12 AS district,
            column13 AS county,
            column14 AS ppd_category,
            column15 AS record_status
        FROM read_csv([{files}], header=false, all_varchar=true)
        """
    )
    frame = con.execute(
        """
        SELECT
            transaction_id,
            price,
            transfer_date,
            postcode,
            split_part(postcode, ' ', 1) AS postcode_district,
            regexp_extract(postcode, '^[A-Z]+', 0) AS postcode_area,
            property_type,
            old_new,
            duration,
            town_city,
            district,
            county,
            EXTRACT(year FROM transfer_date)::INTEGER AS transfer_year,
            EXTRACT(month FROM transfer_date)::INTEGER AS transfer_month,
            sin(2*pi()*EXTRACT(month FROM transfer_date)/12) AS month_sin,
            cos(2*pi()*EXTRACT(month FROM transfer_date)/12) AS month_cos
        FROM price_paid
        WHERE property_type IN ('D', 'S', 'T', 'F')
          AND ppd_category = 'A'
          AND price BETWEEN 20000 AND 5000000
          AND postcode IS NOT NULL
          AND transfer_date >= DATE '2025-01-01'
          AND transfer_date < DATE '2026-07-01'
        ORDER BY transfer_date, transaction_id
        """
    ).df()
    con.close()
    frame["transfer_date"] = pd.to_datetime(frame["transfer_date"])
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].fillna("UNKNOWN").astype(str)
    return frame


def temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame[(frame.transfer_date >= MODEL_START) & (frame.transfer_date < VALIDATION_START)].copy()
    validation = frame[(frame.transfer_date >= VALIDATION_START) & (frame.transfer_date < TEST_START)].copy()
    test = frame[(frame.transfer_date >= TEST_START) & (frame.transfer_date < TEST_END)].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("temporal split produced an empty partition")
    if train.transfer_date.max() >= validation.transfer_date.min():
        raise AssertionError("train/validation chronology is invalid")
    if validation.transfer_date.max() >= test.transfer_date.min():
        raise AssertionError("validation/test chronology is invalid")
    return train, validation, test
