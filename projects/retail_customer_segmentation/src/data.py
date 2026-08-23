from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
EXPECTED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}


def download_dataset(cache_dir: Path) -> Path:
    """Download the UCI Online Retail workbook and return the local xlsx path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "online_retail.zip"

    if not archive_path.exists():
        request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response, archive_path.open("wb") as target:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                target.write(chunk)

    with zipfile.ZipFile(archive_path) as archive:
        workbook_names = [name for name in archive.namelist() if name.lower().endswith(".xlsx")]
        if len(workbook_names) != 1:
            raise ValueError(f"Expected one xlsx workbook, found {workbook_names}")
        workbook_name = workbook_names[0]
        workbook_path = cache_dir / Path(workbook_name).name
        if not workbook_path.exists():
            archive.extract(workbook_name, cache_dir)
            extracted = cache_dir / workbook_name
            if extracted != workbook_path:
                extracted.replace(workbook_path)
    return workbook_path


def load_raw_transactions(workbook_path: Path) -> pd.DataFrame:
    frame = pd.read_excel(workbook_path, engine="openpyxl")
    missing = sorted(EXPECTED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Source schema is missing expected columns: {missing}")
    return frame


def audit_raw_data(frame: pd.DataFrame) -> dict:
    """Return a machine-readable profile before cleaning."""
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "missing_by_column": {key: int(value) for key, value in frame.isna().sum().items()},
        "non_positive_quantity_rows": int((pd.to_numeric(frame["Quantity"], errors="coerce") <= 0).sum()),
        "non_positive_unit_price_rows": int((pd.to_numeric(frame["UnitPrice"], errors="coerce") <= 0).sum()),
        "cancelled_invoice_rows": int(frame["InvoiceNo"].astype("string").str.upper().str.startswith("C", na=False).sum()),
        "unique_invoices": int(frame["InvoiceNo"].nunique(dropna=True)),
        "unique_customers": int(frame["CustomerID"].nunique(dropna=True)),
        "date_min": str(pd.to_datetime(frame["InvoiceDate"], errors="coerce").min()),
        "date_max": str(pd.to_datetime(frame["InvoiceDate"], errors="coerce").max()),
    }


def clean_transactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Create a customer-level modelling base from messy transactional data.

    Rules are explicit and auditable rather than silently dropping rows:
    - remove exact duplicate lines;
    - require customer, invoice, stock code, timestamp and country;
    - remove cancellations and non-positive quantity/price rows;
    - normalize text and dtypes;
    - create line revenue and validate the final table.
    """
    work = frame.copy()
    initial_rows = len(work)

    duplicate_mask = work.duplicated(keep="first")
    duplicate_rows = int(duplicate_mask.sum())
    work = work.loc[~duplicate_mask].copy()

    for column in ["InvoiceNo", "StockCode", "Description", "Country"]:
        work[column] = work[column].astype("string").str.strip()

    work["InvoiceDate"] = pd.to_datetime(work["InvoiceDate"], errors="coerce")
    work["Quantity"] = pd.to_numeric(work["Quantity"], errors="coerce")
    work["UnitPrice"] = pd.to_numeric(work["UnitPrice"], errors="coerce")
    work["CustomerID"] = pd.to_numeric(work["CustomerID"], errors="coerce")

    work["Country"] = work["Country"].str.replace(r"\s+", " ", regex=True)
    work["Description"] = work["Description"].str.replace(r"\s+", " ", regex=True)
    work["is_cancelled"] = work["InvoiceNo"].str.upper().str.startswith("C", na=False)

    reason_masks = {
        "missing_customer_id": work["CustomerID"].isna(),
        "missing_invoice_no": work["InvoiceNo"].isna() | work["InvoiceNo"].eq(""),
        "missing_stock_code": work["StockCode"].isna() | work["StockCode"].eq(""),
        "missing_invoice_date": work["InvoiceDate"].isna(),
        "missing_country": work["Country"].isna() | work["Country"].eq(""),
        "cancelled_invoice": work["is_cancelled"],
        "non_positive_quantity": work["Quantity"].isna() | work["Quantity"].le(0),
        "non_positive_unit_price": work["UnitPrice"].isna() | work["UnitPrice"].le(0),
    }

    invalid_mask = np.zeros(len(work), dtype=bool)
    removed_by_rule: dict[str, int] = {}
    for name, mask in reason_masks.items():
        removed_by_rule[name] = int(mask.sum())
        invalid_mask |= mask.to_numpy()

    clean = work.loc[~invalid_mask].copy()
    clean["CustomerID"] = clean["CustomerID"].round().astype("int64").astype("string")
    clean["Quantity"] = clean["Quantity"].astype("int64")
    clean["line_revenue"] = clean["Quantity"] * clean["UnitPrice"]

    clean = clean[
        [
            "InvoiceNo",
            "InvoiceDate",
            "CustomerID",
            "StockCode",
            "Description",
            "Quantity",
            "UnitPrice",
            "line_revenue",
            "Country",
        ]
    ].sort_values(["InvoiceDate", "InvoiceNo", "StockCode"], kind="stable")
    clean = clean.reset_index(drop=True)

    if clean.empty:
        raise ValueError("Cleaning removed every row; inspect source or cleaning rules")
    if clean.duplicated().any():
        raise AssertionError("Exact duplicates remain after cleaning")
    if clean[["InvoiceNo", "InvoiceDate", "CustomerID", "StockCode", "Quantity", "UnitPrice", "Country"]].isna().any().any():
        raise AssertionError("Required final fields contain missing values")
    if not clean["Quantity"].gt(0).all():
        raise AssertionError("Final dataset contains non-positive quantities")
    if not clean["UnitPrice"].gt(0).all():
        raise AssertionError("Final dataset contains non-positive prices")
    if not clean["line_revenue"].gt(0).all():
        raise AssertionError("Final dataset contains non-positive line revenue")

    report = {
        "source_rows": int(initial_rows),
        "exact_duplicates_removed": duplicate_rows,
        "rule_counts_before_combining": removed_by_rule,
        "rows_removed_after_deduplication": int(len(work) - len(clean)),
        "final_rows": int(len(clean)),
        "retained_share": float(len(clean) / initial_rows),
        "final_unique_customers": int(clean["CustomerID"].nunique()),
        "final_unique_invoices": int(clean["InvoiceNo"].nunique()),
        "final_revenue": float(clean["line_revenue"].sum()),
        "final_date_min": clean["InvoiceDate"].min().isoformat(),
        "final_date_max": clean["InvoiceDate"].max().isoformat(),
    }
    return clean, report


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
