"""Download, validate and split official BTS on-time performance data."""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

BTS_BASE = "https://transtats.bts.gov/PREZIP"
RAW_COLUMNS = {
    "FlightDate",
    "Month",
    "DayOfWeek",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "ArrDelayMinutes",
    "Cancelled",
    "Diverted",
}
BASE_COLUMNS = [
    "FlightDate",
    "month",
    "day_of_week",
    "carrier",
    "origin",
    "dest",
    "route",
    "crs_dep_minutes",
    "crs_arr_minutes",
    "crs_elapsed_minutes",
    "distance_miles",
    "delay_minutes",
]


@dataclass(frozen=True)
class DataConfig:
    year: int = 2026
    train_months: tuple[int, ...] = (1, 2, 3)
    validation_months: tuple[int, ...] = (4,)
    test_months: tuple[int, ...] = (5,)
    max_rows_per_train_month: int | None = 120_000
    max_rows_validation: int | None = 120_000
    max_rows_test: int | None = 180_000
    seed: int = 42
    cache_dir: Path = Path("data/bts_cache")
    request_timeout_seconds: int = 120

    def validate(self) -> None:
        month_groups = self.train_months + self.validation_months + self.test_months
        if not month_groups:
            raise ValueError("At least one month is required")
        if any(month not in range(1, 13) for month in month_groups):
            raise ValueError("All months must be between 1 and 12")
        if set(self.train_months) & set(self.validation_months):
            raise ValueError("Train and validation months overlap")
        if set(self.train_months) & set(self.test_months):
            raise ValueError("Train and test months overlap")
        if set(self.validation_months) & set(self.test_months):
            raise ValueError("Validation and test months overlap")


def hhmm_to_minutes(series: pd.Series) -> pd.Series:
    """Convert BTS HHMM schedule fields to minutes after midnight."""
    values = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    hours = (values // 100).clip(0, 23)
    minutes = (values % 100).clip(0, 59)
    return hours * 60 + minutes


def bts_zip_url(year: int, month: int) -> str:
    if month not in range(1, 13):
        raise ValueError(f"month must be 1..12; got {month}")
    return (
        f"{BTS_BASE}/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_"
        f"{year}_{month}.zip"
    )


def download_bts_month(
    year: int,
    month: int,
    cache_dir: Path,
    timeout: int = 120,
) -> pd.DataFrame:
    """Download one BTS month once, then reuse the cached zip on later runs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"bts_on_time_{year}_{month:02d}.zip"

    if not zip_path.exists():
        url = bts_zip_url(year, month)
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        zip_path.write_bytes(response.content)

    with zipfile.ZipFile(zip_path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found inside {zip_path}")
        with archive.open(csv_names[0]) as handle:
            frame = pd.read_csv(
                handle,
                usecols=lambda name: name.strip() in RAW_COLUMNS,
                low_memory=False,
            )

    frame.columns = [name.strip() for name in frame.columns]
    missing = sorted(RAW_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"BTS {year}-{month:02d} is missing expected fields: {missing}")
    return frame


def clean_flights(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean raw records and keep only fields available at schedule time plus the target."""
    missing = sorted(RAW_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    data = frame.copy()
    data["FlightDate"] = pd.to_datetime(data["FlightDate"], errors="coerce")
    numeric_columns = [
        "Month",
        "DayOfWeek",
        "CRSDepTime",
        "CRSArrTime",
        "CRSElapsedTime",
        "Distance",
        "ArrDelayMinutes",
        "Cancelled",
        "Diverted",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.loc[
        data["FlightDate"].notna()
        & data["ArrDelayMinutes"].notna()
        & data["Cancelled"].fillna(0).eq(0)
        & data["Diverted"].fillna(0).eq(0)
    ].copy()

    data["month"] = data["Month"].astype("Int64")
    data["day_of_week"] = data["DayOfWeek"].astype("Int64")
    data["carrier"] = data["Reporting_Airline"].fillna("UNKNOWN").astype(str)
    data["origin"] = data["Origin"].fillna("UNKNOWN").astype(str)
    data["dest"] = data["Dest"].fillna("UNKNOWN").astype(str)
    data["route"] = data["origin"] + "→" + data["dest"]
    data["crs_dep_minutes"] = hhmm_to_minutes(data["CRSDepTime"])
    data["crs_arr_minutes"] = hhmm_to_minutes(data["CRSArrTime"])
    data["crs_elapsed_minutes"] = data["CRSElapsedTime"].clip(lower=1)
    data["distance_miles"] = data["Distance"].clip(lower=0)
    data["delay_minutes"] = data["ArrDelayMinutes"].clip(lower=0)

    data = data.dropna(subset=BASE_COLUMNS).reset_index(drop=True)
    if data.empty:
        raise ValueError("No usable completed flights remain after cleaning")
    if not data["delay_minutes"].ge(0).all():
        raise AssertionError("Delay target must be non-negative")
    if not data["distance_miles"].ge(0).all():
        raise AssertionError("Distance must be non-negative")
    if not data["crs_elapsed_minutes"].gt(0).all():
        raise AssertionError("Scheduled elapsed time must be positive")
    return data[BASE_COLUMNS]


def deterministic_sample(frame: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or len(frame) <= n:
        return frame.copy()
    return frame.sample(n=n, random_state=seed).copy()


def _check_temporal_order(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> None:
    if not train["FlightDate"].max() < valid["FlightDate"].min():
        raise AssertionError("Training dates overlap validation dates")
    if not valid["FlightDate"].max() < test["FlightDate"].min():
        raise AssertionError("Validation dates overlap test dates")


def load_temporal_splits(config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and clean all requested months, then enforce a strict temporal holdout."""
    config.validate()
    months = sorted(set(config.train_months + config.validation_months + config.test_months))
    monthly: dict[int, pd.DataFrame] = {}

    for month in months:
        print(f"Loading BTS {config.year}-{month:02d}...")
        raw = download_bts_month(
            config.year,
            month,
            cache_dir=config.cache_dir,
            timeout=config.request_timeout_seconds,
        )
        monthly[month] = clean_flights(raw)
        print(f"  usable rows: {len(monthly[month]):,}")

    train = pd.concat(
        [
            deterministic_sample(monthly[month], config.max_rows_per_train_month, config.seed + month)
            for month in config.train_months
        ],
        ignore_index=True,
    )
    valid = pd.concat([monthly[month] for month in config.validation_months], ignore_index=True)
    test = pd.concat([monthly[month] for month in config.test_months], ignore_index=True)
    valid = deterministic_sample(valid, config.max_rows_validation, config.seed + 100)
    test = deterministic_sample(test, config.max_rows_test, config.seed + 200)

    _check_temporal_order(train, valid, test)
    return train, valid, test
