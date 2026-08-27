from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.pipeline import connect, ingest_csv, quality_summary

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_two_batches_are_clean_and_idempotent_by_event_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "events.db")
            first = ingest_csv(conn, ROOT / "fixtures" / "batch_1.csv", batch_name="batch_1")
            second = ingest_csv(conn, ROOT / "fixtures" / "batch_2.csv", batch_name="batch_2")
            summary = quality_summary(conn)

        self.assertEqual(first.inserted_rows, 4)
        self.assertEqual(first.rejected_rows, 2)
        self.assertEqual(first.duplicate_rows_in_batch, 1)
        self.assertEqual(second.inserted_rows, 3)
        self.assertEqual(second.existing_rows_skipped, 1)
        self.assertEqual(second.late_rows, 1)
        self.assertEqual(summary["warehouse_rows"], 7)
        self.assertEqual(summary["duplicate_event_ids"], 0)
        self.assertEqual(summary["null_customer_ids"], 0)
        self.assertTrue(summary["verification_pass"])

    def test_reusing_a_batch_name_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "events.db")
            ingest_csv(conn, ROOT / "fixtures" / "batch_1.csv", batch_name="batch_1")
            with self.assertRaises(ValueError):
                ingest_csv(conn, ROOT / "fixtures" / "batch_1.csv", batch_name="batch_1")


if __name__ == "__main__":
    unittest.main()
