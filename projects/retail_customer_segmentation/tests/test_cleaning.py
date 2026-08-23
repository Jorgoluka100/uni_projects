import pandas as pd

from src.data import audit_raw_data, clean_transactions


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "InvoiceNo": ["10001", "10001", "C10002", "10003", "10004", "10005", "10006"],
            "StockCode": ["A1", "A1", "A2", "A3", "A4", "A5", "A6"],
            "Description": [" Mug ", " Mug ", "Return", "Tea", "Plate", "Bowl", "Glass"],
            "Quantity": [2, 2, -1, 1, 0, 1, 3],
            "InvoiceDate": pd.to_datetime(
                [
                    "2025-01-01 09:00",
                    "2025-01-01 09:00",
                    "2025-01-02 10:00",
                    "2025-01-03 11:00",
                    "2025-01-04 12:00",
                    "2025-01-05 13:00",
                    "2025-01-06 14:00",
                ]
            ),
            "UnitPrice": [5.0, 5.0, 4.0, 3.0, 2.0, 0.0, 7.0],
            "CustomerID": [1, 1, 2, None, 4, 5, 6],
            "Country": [" United Kingdom "] * 7,
        }
    )


def test_raw_audit_counts_quality_problems():
    audit = audit_raw_data(_fixture())
    assert audit["rows"] == 7
    assert audit["exact_duplicate_rows"] == 1
    assert audit["cancelled_invoice_rows"] == 1
    assert audit["non_positive_quantity_rows"] == 2
    assert audit["non_positive_unit_price_rows"] == 1


def test_cleaning_is_explicit_and_produces_valid_rows():
    clean, report = clean_transactions(_fixture())

    assert report["exact_duplicates_removed"] == 1
    assert len(clean) == 2
    assert set(clean["CustomerID"]) == {"1", "6"}
    assert clean["Quantity"].gt(0).all()
    assert clean["UnitPrice"].gt(0).all()
    assert clean["line_revenue"].gt(0).all()
    assert not clean.duplicated().any()
    assert clean.loc[clean["CustomerID"].eq("1"), "Country"].iloc[0] == "United Kingdom"
