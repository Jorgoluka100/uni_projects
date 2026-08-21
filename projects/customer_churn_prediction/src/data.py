from __future__ import annotations

import hashlib
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATASET_URL = "https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip"
ARCHIVE_SHA256 = "696c3a1812267980751f30d19193a1b430ac4ad76172bb089e664c96813ead66"
CSV_SHA256 = "90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1"
CSV_NAME = "Customer Churn.csv"
SEED = 42

OPERATIONAL_INPUT_COLUMNS = [
    "call_failure",
    "complains",
    "subscription_length",
    "charge_amount",
    "seconds_of_use",
    "frequency_of_use",
    "frequency_of_sms",
    "distinct_called_numbers",
    "tariff_plan",
    "customer_value",
]
EXCLUDED_FIELDS = {
    "status": "semantic proxy-risk feature; sensitivity analysis only",
    "age": "subgroup monitoring only",
    "age_group": "subgroup monitoring only",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snake_case(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def download_dataset(cache_dir: Path) -> Path:
    """Download the pinned UCI source and refuse silent source changes."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "iranian_churn_dataset.zip"
    csv_path = cache_dir / CSV_NAME

    if not archive_path.exists():
        request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response, archive_path.open("wb") as target:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                target.write(chunk)

    if sha256_file(archive_path) != ARCHIVE_SHA256:
        raise ValueError("UCI archive SHA-256 does not match the pinned project source")

    with zipfile.ZipFile(archive_path) as archive:
        if CSV_NAME not in archive.namelist():
            raise ValueError(f"{CSV_NAME!r} is missing from the UCI archive")
        archive.extract(CSV_NAME, cache_dir)

    if sha256_file(csv_path) != CSV_SHA256:
        raise ValueError("extracted churn CSV SHA-256 does not match the pinned project source")
    return csv_path


def load_dataset(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame.columns = [_snake_case(column) for column in frame.columns]
    required = set(OPERATIONAL_INPUT_COLUMNS) | {"churn", "status", "age", "age_group"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"source schema is missing expected columns: {missing}")
    if len(frame) != 3150:
        raise ValueError(f"expected 3,150 source rows, found {len(frame):,}")
    if frame[list(required)].isna().any().any():
        raise ValueError("unexpected missing values in the pinned modelling fields")
    return frame


def operational_profile_groups(frame: pd.DataFrame) -> pd.Series:
    """Stable group IDs so identical predictor profiles cannot cross a split."""
    return pd.util.hash_pandas_object(frame[OPERATIONAL_INPUT_COLUMNS], index=False).astype(str)


def protected_holdout_split(frame: pd.DataFrame):
    """Replicate the notebook's 5-fold grouped, stratified protected holdout."""
    X = frame[OPERATIONAL_INPUT_COLUMNS].copy()
    y = frame["churn"].astype(int).copy()
    groups = operational_profile_groups(frame)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_index, test_index = next(splitter.split(X, y, groups=groups))
    train = frame.iloc[train_index].reset_index(drop=True)
    holdout = frame.iloc[test_index].reset_index(drop=True)

    train_profiles = set(operational_profile_groups(train))
    holdout_profiles = set(operational_profile_groups(holdout))
    if train_profiles & holdout_profiles:
        raise AssertionError("identical predictor profiles crossed the holdout boundary")
    return train, holdout
