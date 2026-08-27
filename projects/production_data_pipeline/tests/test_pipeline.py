from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import sys
import unittest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pipeline import run_pipeline, seed_fixture


class PipelineTests(unittest.TestCase):
    def test_pipeline_quarantines_bad_rows_and_is_idempotent(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "orders.jsonl"
            db_path = root / "orders.sqlite"

            seed_fixture(source)
            first = run_pipeline(source, db_path)

            self.assertFalse(first.skipped)
            self.assertEqual(first.raw_rows, 5)
            self.assertEqual(first.quarantined_rows, 1)
            self.assertEqual(first.accepted_rows, 3)
            self.assertEqual(first.gold_rows, 2)

            second = run_pipeline(source, db_path)
            self.assertTrue(second.skipped)
            self.assertEqual(second.accepted_rows, 3)

            conn = sqlite3.connect(db_path)
            statuses = dict(
                conn.execute(
                    "SELECT order_id, status FROM silver_orders ORDER BY order_id"
                ).fetchall()
            )
            self.assertEqual(statuses["o-1002"], "refunded")
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM quarantine_orders"
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM bronze_orders"
                ).fetchone()[0],
                5,
            )
            conn.close()


if __name__ == "__main__":
    unittest.main()
