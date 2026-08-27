from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_FIELDS = {
    "order_id",
    "customer_id",
    "order_ts",
    "updated_at",
    "amount_gbp",
    "status",
}
ALLOWED_STATUS = {"paid", "refunded", "cancelled"}


@dataclass(frozen=True)
class PipelineResult:
    source_file: str
    source_sha256: str
    skipped: bool
    raw_rows: int
    accepted_rows: int
    quarantined_rows: int
    gold_rows: int

    def as_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_fixture(path: Path) -> None:
    """Create deterministic JSONL with a duplicate update and one invalid row."""
    rows = [
        {
            "order_id": "o-1001",
            "customer_id": "c-1",
            "order_ts": "2026-08-01T10:15:00Z",
            "updated_at": "2026-08-01T10:20:00Z",
            "amount_gbp": 24.50,
            "status": "paid",
        },
        {
            "order_id": "o-1002",
            "customer_id": "c-2",
            "order_ts": "2026-08-01T11:00:00Z",
            "updated_at": "2026-08-01T11:05:00Z",
            "amount_gbp": 15.00,
            "status": "paid",
        },
        {
            "order_id": "o-1002",
            "customer_id": "c-2",
            "order_ts": "2026-08-01T11:00:00Z",
            "updated_at": "2026-08-01T12:00:00Z",
            "amount_gbp": 15.00,
            "status": "refunded",
        },
        {
            "order_id": "o-1003",
            "customer_id": "c-3",
            "order_ts": "2026-08-02T09:30:00Z",
            "updated_at": "2026-08-02T09:31:00Z",
            "amount_gbp": -3.00,
            "status": "paid",
        },
        {
            "order_id": "o-1004",
            "customer_id": "c-1",
            "order_ts": "2026-08-02T13:10:00Z",
            "updated_at": "2026-08-02T13:11:00Z",
            "amount_gbp": 52.00,
            "status": "paid",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def parse_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append(
                    {
                        "_line_number": line_number,
                        "_parse_error": f"invalid_json:{exc.msg}",
                        "_raw_line": line.rstrip("\n"),
                    }
                )
                continue
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def row_error(row: dict) -> str | None:
    if "_parse_error" in row:
        return row["_parse_error"]

    missing = sorted(field for field in REQUIRED_FIELDS if field not in row)
    if missing:
        return "missing_fields:" + ",".join(missing)

    if not str(row["order_id"]).strip() or not str(row["customer_id"]).strip():
        return "blank_identifier"

    try:
        amount = float(row["amount_gbp"])
    except (TypeError, ValueError):
        return "invalid_amount"

    if amount < 0:
        return "negative_amount"

    if row["status"] not in ALLOWED_STATUS:
        return "invalid_status"

    for field in ("order_ts", "updated_at"):
        try:
            datetime.fromisoformat(str(row[field]).replace("Z", "+00:00"))
        except ValueError:
            return f"invalid_{field}"

    return None


def initialise_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS ingestion_log (
            source_sha256 TEXT PRIMARY KEY,
            source_file TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS bronze_orders (
            source_sha256 TEXT NOT NULL,
            line_number INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS quarantine_orders (
            source_sha256 TEXT NOT NULL,
            line_number INTEGER NOT NULL,
            error_reason TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            quarantined_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS silver_orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            order_ts TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            amount_gbp REAL NOT NULL CHECK (amount_gbp >= 0),
            status TEXT NOT NULL CHECK (status IN ('paid', 'refunded', 'cancelled')),
            source_sha256 TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS gold_daily_order_metrics (
            order_date TEXT PRIMARY KEY,
            gross_paid_gbp REAL NOT NULL,
            paid_orders INTEGER NOT NULL,
            refunded_orders INTEGER NOT NULL,
            cancelled_orders INTEGER NOT NULL,
            distinct_customers INTEGER NOT NULL
        );
        """
    )


def refresh_gold(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM gold_daily_order_metrics")
    conn.execute(
        """
        INSERT INTO gold_daily_order_metrics (
            order_date,
            gross_paid_gbp,
            paid_orders,
            refunded_orders,
            cancelled_orders,
            distinct_customers
        )
        SELECT
            substr(order_ts, 1, 10) AS order_date,
            ROUND(SUM(CASE WHEN status = 'paid' THEN amount_gbp ELSE 0 END), 2),
            SUM(CASE WHEN status = 'paid' THEN 1 ELSE 0 END),
            SUM(CASE WHEN status = 'refunded' THEN 1 ELSE 0 END),
            SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END),
            COUNT(DISTINCT customer_id)
        FROM silver_orders
        GROUP BY substr(order_ts, 1, 10)
        ORDER BY order_date
        """
    )


def quality_checks(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        "duplicate_order_ids": conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT order_id
                FROM silver_orders
                GROUP BY order_id
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0],
        "negative_amounts": conn.execute(
            "SELECT COUNT(*) FROM silver_orders WHERE amount_gbp < 0"
        ).fetchone()[0],
        "invalid_statuses": conn.execute(
            """
            SELECT COUNT(*) FROM silver_orders
            WHERE status NOT IN ('paid', 'refunded', 'cancelled')
            """
        ).fetchone()[0],
        "null_required": conn.execute(
            """
            SELECT COUNT(*) FROM silver_orders
            WHERE order_id IS NULL
               OR customer_id IS NULL
               OR order_ts IS NULL
               OR updated_at IS NULL
               OR amount_gbp IS NULL
               OR status IS NULL
            """
        ).fetchone()[0],
    }


def run_pipeline(source: Path, db_path: Path) -> PipelineResult:
    source = source.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    now = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(db_path)
    initialise_db(conn)

    already_loaded = conn.execute(
        "SELECT 1 FROM ingestion_log WHERE source_sha256 = ?",
        (source_hash,),
    ).fetchone()

    if already_loaded:
        accepted_rows = conn.execute("SELECT COUNT(*) FROM silver_orders").fetchone()[0]
        gold_rows = conn.execute(
            "SELECT COUNT(*) FROM gold_daily_order_metrics"
        ).fetchone()[0]
        conn.close()
        return PipelineResult(
            source_file=str(source),
            source_sha256=source_hash,
            skipped=True,
            raw_rows=0,
            accepted_rows=accepted_rows,
            quarantined_rows=0,
            gold_rows=gold_rows,
        )

    rows = parse_jsonl(source)
    raw_rows = len(rows)
    quarantined = 0

    try:
        with conn:
            latest_by_order: dict[str, dict] = {}

            for row in rows:
                line_number = int(row["_line_number"])
                payload = {
                    key: value
                    for key, value in row.items()
                    if not key.startswith("_")
                }
                payload_json = json.dumps(payload, sort_keys=True)

                conn.execute(
                    """
                    INSERT INTO bronze_orders
                    VALUES (?, ?, ?, ?)
                    """,
                    (source_hash, line_number, payload_json, now),
                )

                error = row_error(row)
                if error:
                    quarantined += 1
                    conn.execute(
                        """
                        INSERT INTO quarantine_orders
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (source_hash, line_number, error, payload_json, now),
                    )
                    continue

                order_id = str(row["order_id"])
                previous = latest_by_order.get(order_id)
                if previous is None or str(row["updated_at"]) > str(previous["updated_at"]):
                    latest_by_order[order_id] = row

            for row in latest_by_order.values():
                existing = conn.execute(
                    "SELECT updated_at FROM silver_orders WHERE order_id = ?",
                    (str(row["order_id"]),),
                ).fetchone()

                if existing and str(existing[0]) > str(row["updated_at"]):
                    continue

                conn.execute(
                    """
                    INSERT INTO silver_orders (
                        order_id,
                        customer_id,
                        order_ts,
                        updated_at,
                        amount_gbp,
                        status,
                        source_sha256
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(order_id) DO UPDATE SET
                        customer_id = excluded.customer_id,
                        order_ts = excluded.order_ts,
                        updated_at = excluded.updated_at,
                        amount_gbp = excluded.amount_gbp,
                        status = excluded.status,
                        source_sha256 = excluded.source_sha256
                    """,
                    (
                        str(row["order_id"]),
                        str(row["customer_id"]),
                        str(row["order_ts"]),
                        str(row["updated_at"]),
                        float(row["amount_gbp"]),
                        str(row["status"]),
                        source_hash,
                    ),
                )

            refresh_gold(conn)
            checks = quality_checks(conn)
            if any(checks.values()):
                raise RuntimeError(f"data quality checks failed: {checks}")

            conn.execute(
                "INSERT INTO ingestion_log VALUES (?, ?, ?)",
                (source_hash, str(source), now),
            )
    except Exception:
        conn.close()
        raise

    accepted_rows = conn.execute("SELECT COUNT(*) FROM silver_orders").fetchone()[0]
    gold_rows = conn.execute(
        "SELECT COUNT(*) FROM gold_daily_order_metrics"
    ).fetchone()[0]
    conn.close()

    return PipelineResult(
        source_file=str(source),
        source_sha256=source_hash,
        skipped=False,
        raw_rows=raw_rows,
        accepted_rows=accepted_rows,
        quarantined_rows=quarantined,
        gold_rows=gold_rows,
    )


def write_manifest(result: PipelineResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an idempotent bronze/silver/gold order pipeline."
    )
    parser.add_argument("--source", type=Path, default=Path("data/orders.jsonl"))
    parser.add_argument("--db", type=Path, default=Path("artifacts/orders.sqlite"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/run_manifest.json"),
    )
    parser.add_argument(
        "--seed-fixture",
        action="store_true",
        help="Create the deterministic demo input before running.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed_fixture:
        seed_fixture(args.source)
    if not args.source.exists():
        raise FileNotFoundError(
            f"{args.source} does not exist. Use --seed-fixture or provide JSONL input."
        )
    result = run_pipeline(args.source, args.db)
    write_manifest(result, args.manifest)
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
