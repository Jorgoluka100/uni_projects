from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

REQUIRED_COLUMNS = (
    "event_id",
    "customer_id",
    "event_time",
    "source",
    "event_type",
    "amount",
)


@dataclass(frozen=True)
class BatchMetrics:
    batch_name: str
    input_rows: int
    valid_rows: int
    rejected_rows: int
    duplicate_rows_in_batch: int
    existing_rows_skipped: int
    late_rows: int
    inserted_rows: int
    warehouse_rows_after: int


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialise(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            event_time TEXT NOT NULL,
            source TEXT NOT NULL,
            event_type TEXT NOT NULL,
            amount REAL NOT NULL,
            is_late INTEGER NOT NULL CHECK (is_late IN (0, 1)),
            ingested_at TEXT NOT NULL,
            batch_name TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS rejected_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_name TEXT NOT NULL,
            row_number INTEGER NOT NULL,
            reason TEXT NOT NULL,
            payload TEXT NOT NULL,
            rejected_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS batch_audit (
            batch_name TEXT PRIMARY KEY,
            input_rows INTEGER NOT NULL,
            valid_rows INTEGER NOT NULL,
            rejected_rows INTEGER NOT NULL,
            duplicate_rows_in_batch INTEGER NOT NULL,
            existing_rows_skipped INTEGER NOT NULL,
            late_rows INTEGER NOT NULL,
            inserted_rows INTEGER NOT NULL,
            warehouse_rows_after INTEGER NOT NULL,
            completed_at TEXT NOT NULL
        );
        """
    )
    conn.commit()


def _parse_utc(value: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _current_max_event_time(conn: sqlite3.Connection) -> datetime | None:
    row = conn.execute("SELECT MAX(event_time) AS max_event_time FROM events").fetchone()
    if not row or row["max_event_time"] is None:
        return None
    return _parse_utc(row["max_event_time"])


def _normalise(row: dict[str, str]) -> dict[str, object]:
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in row]
    if missing_columns:
        raise ValueError(f"missing columns: {', '.join(missing_columns)}")

    event_id = row["event_id"].strip()
    customer_id = row["customer_id"].strip()
    source = row["source"].strip().lower()
    event_type = row["event_type"].strip().lower()

    if not event_id:
        raise ValueError("missing event_id")
    if not customer_id:
        raise ValueError("missing customer_id")
    if not source:
        raise ValueError("missing source")
    if not event_type:
        raise ValueError("missing event_type")

    event_time = _parse_utc(row["event_time"])
    amount = float(row["amount"])
    if amount < 0:
        raise ValueError("negative amount")

    return {
        "event_id": event_id,
        "customer_id": customer_id,
        "event_time": event_time,
        "source": source,
        "event_type": event_type,
        "amount": amount,
    }


def ingest_csv(
    conn: sqlite3.Connection,
    csv_path: str | Path,
    *,
    batch_name: str,
    allowed_lateness_hours: int = 48,
) -> BatchMetrics:
    """Validate one CSV batch and load it idempotently into the warehouse.

    Late arrivals are accepted but flagged. Rows older than the current warehouse
    maximum event time minus ``allowed_lateness_hours`` are considered late.
    """
    initialise(conn)
    if conn.execute("SELECT 1 FROM batch_audit WHERE batch_name = ?", (batch_name,)).fetchone():
        raise ValueError(f"batch_name already processed: {batch_name}")

    previous_max = _current_max_event_time(conn)
    watermark = previous_max - timedelta(hours=allowed_lateness_hours) if previous_max else None
    now = datetime.now(timezone.utc).isoformat()

    input_rows = valid_rows = rejected_rows = 0
    duplicate_rows_in_batch = existing_rows_skipped = late_rows = inserted_rows = 0
    seen_ids: set[str] = set()

    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_headers = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing_headers:
            raise ValueError(f"missing required headers: {', '.join(missing_headers)}")

        for row_number, raw in enumerate(reader, start=2):
            input_rows += 1
            try:
                clean = _normalise(raw)
            except Exception as exc:
                rejected_rows += 1
                conn.execute(
                    "INSERT INTO rejected_events(batch_name,row_number,reason,payload,rejected_at) VALUES(?,?,?,?,?)",
                    (batch_name, row_number, str(exc), json.dumps(raw, sort_keys=True), now),
                )
                continue

            event_id = str(clean["event_id"])
            if event_id in seen_ids:
                duplicate_rows_in_batch += 1
                rejected_rows += 1
                conn.execute(
                    "INSERT INTO rejected_events(batch_name,row_number,reason,payload,rejected_at) VALUES(?,?,?,?,?)",
                    (batch_name, row_number, "duplicate event_id within batch", json.dumps(raw, sort_keys=True), now),
                )
                continue
            seen_ids.add(event_id)
            valid_rows += 1

            event_dt = clean["event_time"]
            assert isinstance(event_dt, datetime)
            is_late = int(watermark is not None and event_dt < watermark)
            late_rows += is_late

            before = conn.total_changes
            conn.execute(
                """
                INSERT OR IGNORE INTO events(
                    event_id, customer_id, event_time, source, event_type,
                    amount, is_late, ingested_at, batch_name
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    clean["event_id"],
                    clean["customer_id"],
                    event_dt.isoformat(),
                    clean["source"],
                    clean["event_type"],
                    clean["amount"],
                    is_late,
                    now,
                    batch_name,
                ),
            )
            if conn.total_changes > before:
                inserted_rows += 1
            else:
                existing_rows_skipped += 1

    warehouse_rows_after = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    metrics = BatchMetrics(
        batch_name=batch_name,
        input_rows=input_rows,
        valid_rows=valid_rows,
        rejected_rows=rejected_rows,
        duplicate_rows_in_batch=duplicate_rows_in_batch,
        existing_rows_skipped=existing_rows_skipped,
        late_rows=late_rows,
        inserted_rows=inserted_rows,
        warehouse_rows_after=warehouse_rows_after,
    )
    conn.execute(
        """
        INSERT INTO batch_audit(
            batch_name,input_rows,valid_rows,rejected_rows,duplicate_rows_in_batch,
            existing_rows_skipped,late_rows,inserted_rows,warehouse_rows_after,completed_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?)
        """,
        (*asdict(metrics).values(), now),
    )
    conn.commit()
    return metrics


def quality_summary(conn: sqlite3.Connection) -> dict[str, object]:
    initialise(conn)
    total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    duplicate_ids = conn.execute(
        "SELECT COUNT(*) FROM (SELECT event_id FROM events GROUP BY event_id HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    null_business_keys = conn.execute(
        "SELECT COUNT(*) FROM events WHERE customer_id IS NULL OR TRIM(customer_id) = ''"
    ).fetchone()[0]
    late_rows = conn.execute("SELECT COUNT(*) FROM events WHERE is_late = 1").fetchone()[0]
    rejected = conn.execute("SELECT COUNT(*) FROM rejected_events").fetchone()[0]
    revenue = conn.execute("SELECT ROUND(SUM(amount), 2) FROM events").fetchone()[0] or 0.0
    return {
        "warehouse_rows": total,
        "duplicate_event_ids": duplicate_ids,
        "null_customer_ids": null_business_keys,
        "late_rows": late_rows,
        "rejected_rows": rejected,
        "total_amount": revenue,
        "verification_pass": duplicate_ids == 0 and null_business_keys == 0,
    }
