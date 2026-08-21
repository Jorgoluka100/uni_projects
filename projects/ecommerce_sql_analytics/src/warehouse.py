"""DuckDB warehouse construction and SQL execution."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .config import FILE_TABLES


def connect(database_path: Path | str = ":memory:") -> duckdb.DuckDBPyConnection:
    if str(database_path) != ":memory:":
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database_path))
    connection.execute("CREATE SCHEMA IF NOT EXISTS raw")
    connection.execute("CREATE SCHEMA IF NOT EXISTS analytics")
    return connection


def load_raw_tables(connection: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    """Load each verified CSV into a raw DuckDB table."""
    for filename, table_name in FILE_TABLES.items():
        path = (data_dir / filename).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        safe_path = str(path).replace("'", "''")
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE raw.{table_name} AS
            SELECT *
            FROM read_csv_auto('{safe_path}', header = TRUE, sample_size = -1, all_varchar = FALSE)
            """
        )


def execute_sql_file(connection: duckdb.DuckDBPyConnection, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    connection.execute(sql)


def build_analytics(connection: duckdb.DuckDBPyConnection, sql_dir: Path) -> None:
    """Execute numbered SQL modules in deterministic order."""
    sql_files = sorted(sql_dir.glob("*.sql"))
    if not sql_files:
        raise FileNotFoundError(f"No SQL files found under {sql_dir}")
    for path in sql_files:
        print(f"Executing {path.name}")
        execute_sql_file(connection, path)


def dataframe(connection: duckdb.DuckDBPyConnection, query: str) -> pd.DataFrame:
    return connection.execute(query).df()


def export_analytics(connection: duckdb.DuckDBPyConnection, output_dir: Path) -> list[Path]:
    """Export compact recruiter-readable result tables as Parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = [
        "headline_kpis",
        "monthly_performance",
        "customer_order_frequency",
        "cohort_retention",
        "category_performance",
        "delivery_review_summary",
        "seller_operational_review",
        "seller_concentration_summary",
        "payment_behaviour",
        "top_categories_by_customer_state",
    ]
    outputs: list[Path] = []
    for table in tables:
        destination = (output_dir / f"{table}.parquet").resolve()
        safe_destination = str(destination).replace("'", "''")
        connection.execute(
            f"COPY analytics.{table} TO '{safe_destination}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        outputs.append(destination)
    return outputs
