from __future__ import annotations

import unittest
from pathlib import Path

from run import build_synthetic_fixture
from src.validate import assert_integrity, run_integrity_checks
from src.warehouse import build_analytics, connect


class EcommerceSqlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = connect(":memory:")
        build_synthetic_fixture(self.connection)
        build_analytics(self.connection, Path(__file__).parents[1] / "sql")

    def tearDown(self) -> None:
        self.connection.close()

    def test_integrity_checks_pass(self) -> None:
        checks = run_integrity_checks(self.connection)
        assert_integrity(checks)
        self.assertTrue(all(check.passed for check in checks))

    def test_semantic_mart_prevents_many_to_many_revenue_inflation(self) -> None:
        naive_value = self.connection.execute(
            """
            SELECT SUM(i.price)
            FROM raw.order_items i
            JOIN raw.payments p USING (order_id)
            WHERE i.order_id = 'o1'
            """
        ).fetchone()[0]
        mart_value = self.connection.execute(
            "SELECT merchandise_value_brl FROM analytics.order_mart WHERE order_id = 'o1'"
        ).fetchone()[0]
        self.assertEqual(float(naive_value), 300.0)
        self.assertEqual(float(mart_value), 150.0)

    def test_latest_review_is_selected_once_per_order(self) -> None:
        score = self.connection.execute(
            "SELECT review_score FROM analytics.order_mart WHERE order_id = 'o1'"
        ).fetchone()[0]
        self.assertEqual(int(score), 4)

    def test_canceled_order_is_not_commercial(self) -> None:
        commercial = self.connection.execute(
            "SELECT commercial_order FROM analytics.order_mart WHERE order_id = 'o4'"
        ).fetchone()[0]
        self.assertFalse(bool(commercial))

    def test_repeat_customer_cohort_is_preserved(self) -> None:
        month_one = self.connection.execute(
            """
            SELECT active_customers, cohort_customers, retention_pct
            FROM analytics.cohort_retention
            WHERE cohort_month = DATE '2017-01-01' AND month_number = 1
            """
        ).fetchone()
        self.assertEqual(tuple(month_one), (1, 1, 100.0))

    def test_qualify_returns_at_most_three_categories_per_state(self) -> None:
        max_categories = self.connection.execute(
            """
            SELECT MAX(category_count)
            FROM (
                SELECT customer_state, COUNT(*) AS category_count
                FROM analytics.top_categories_by_customer_state
                GROUP BY customer_state
            )
            """
        ).fetchone()[0]
        self.assertLessEqual(int(max_categories), 3)


if __name__ == "__main__":
    unittest.main()
