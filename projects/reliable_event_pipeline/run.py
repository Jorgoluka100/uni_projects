from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from src.pipeline import connect, ingest_csv, quality_summary

ROOT = Path(__file__).resolve().parent


def run_demo(db_path: Path) -> dict[str, object]:
    conn = connect(db_path)
    first = ingest_csv(conn, ROOT / "fixtures" / "batch_1.csv", batch_name="batch_1")
    second = ingest_csv(conn, ROOT / "fixtures" / "batch_2.csv", batch_name="batch_2")
    summary = quality_summary(conn)
    conn.close()
    return {
        "batches": [first.__dict__, second.__dict__],
        "quality": summary,
        "verification_pass": bool(summary["verification_pass"]),
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        result = run_demo(Path(tmp) / "events.db")
    assert result["verification_pass"] is True
    assert result["quality"]["warehouse_rows"] == 7
    assert result["quality"]["duplicate_event_ids"] == 0
    assert result["quality"]["null_customer_ids"] == 0
    assert result["quality"]["late_rows"] == 1
    assert result["quality"]["rejected_rows"] == 2
    print("self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=ROOT / "artifacts" / "events.db")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "verified_run.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    args.db.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.db.exists():
        args.db.unlink()
    result = run_demo(args.db)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["verification_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
