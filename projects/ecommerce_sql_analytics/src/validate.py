"""Warehouse integrity, grain and financial reconciliation checks."""
from __future__ import annotations

from dataclasses import dataclass

import duckdb


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    value: float | int
    expectation: str


def _scalar(connection: duckdb.DuckDBPyConnection, query: str) -> float | int:
    value = connection.execute(query).fetchone()[0]
    return 0 if value is None else value


def run_integrity_checks(connection: duckdb.DuckDBPyConnection) -> list[CheckResult]:
    """Check keys, foreign keys, semantic grain and headline financial reconciliation."""
    checks: list[CheckResult] = []

    uniqueness_checks = {
        "customers_customer_id_unique": ("raw.customers", "customer_id"),
        "orders_order_id_unique": ("raw.orders", "order_id"),
        "products_product_id_unique": ("raw.products", "product_id"),
        "sellers_seller_id_unique": ("raw.sellers", "seller_id"),
    }
    for name, (table, key) in uniqueness_checks.items():
        duplicates = int(
            _scalar(
                connection,
                f"SELECT COUNT(*) FROM (SELECT {key} FROM {table} GROUP BY {key} HAVING COUNT(*) > 1)",
            )
        )
        checks.append(CheckResult(name, duplicates == 0, duplicates, "0 duplicate keys"))

    orphan_queries = {
        "orders_customer_fk": """
            SELECT COUNT(*) FROM raw.orders o
            LEFT JOIN raw.customers c USING (customer_id)
            WHERE c.customer_id IS NULL
        """,
        "items_order_fk": """
            SELECT COUNT(*) FROM raw.order_items i
            LEFT JOIN raw.orders o USING (order_id)
            WHERE o.order_id IS NULL
        """,
        "items_product_fk": """
            SELECT COUNT(*) FROM raw.order_items i
            LEFT JOIN raw.products p USING (product_id)
            WHERE p.product_id IS NULL
        """,
        "items_seller_fk": """
            SELECT COUNT(*) FROM raw.order_items i
            LEFT JOIN raw.sellers s USING (seller_id)
            WHERE s.seller_id IS NULL
        """,
        "payments_order_fk": """
            SELECT COUNT(*) FROM raw.payments p
            LEFT JOIN raw.orders o USING (order_id)
            WHERE o.order_id IS NULL
        """,
        "reviews_order_fk": """
            SELECT COUNT(*) FROM raw.reviews r
            LEFT JOIN raw.orders o USING (order_id)
            WHERE o.order_id IS NULL
        """,
    }
    for name, query in orphan_queries.items():
        orphans = int(_scalar(connection, query))
        checks.append(CheckResult(name, orphans == 0, orphans, "0 orphan rows"))

    raw_orders = int(_scalar(connection, "SELECT COUNT(*) FROM raw.orders"))
    mart_orders = int(_scalar(connection, "SELECT COUNT(*) FROM analytics.order_mart"))
    distinct_mart_orders = int(_scalar(connection, "SELECT COUNT(DISTINCT order_id) FROM analytics.order_mart"))
    checks.extend(
        [
            CheckResult("order_mart_row_count", mart_orders == raw_orders, mart_orders, f"{raw_orders} rows"),
            CheckResult(
                "order_mart_one_row_per_order",
                distinct_mart_orders == mart_orders,
                distinct_mart_orders,
                f"{mart_orders} distinct order ids",
            ),
        ]
    )

    raw_items = int(_scalar(connection, "SELECT COUNT(*) FROM raw.order_items"))
    mart_items = int(_scalar(connection, "SELECT COUNT(*) FROM analytics.item_mart"))
    checks.append(CheckResult("item_mart_row_count", mart_items == raw_items, mart_items, f"{raw_items} rows"))

    order_value = float(
        _scalar(
            connection,
            "SELECT COALESCE(SUM(merchandise_value_brl), 0) FROM analytics.order_mart WHERE commercial_order",
        )
    )
    item_value = float(
        _scalar(
            connection,
            "SELECT COALESCE(SUM(item_price_brl), 0) FROM analytics.item_mart WHERE commercial_order",
        )
    )
    delta = abs(order_value - item_value)
    checks.append(CheckResult("merchandise_value_reconciliation", delta < 0.01, round(delta, 6), "< R$0.01 difference"))

    negative_prices = int(
        _scalar(connection, "SELECT COUNT(*) FROM raw.order_items WHERE price < 0 OR freight_value < 0")
    )
    checks.append(CheckResult("non_negative_item_values", negative_prices == 0, negative_prices, "0 negative values"))

    invalid_reviews = int(
        _scalar(connection, "SELECT COUNT(*) FROM raw.reviews WHERE review_score NOT BETWEEN 1 AND 5")
    )
    checks.append(CheckResult("review_score_range", invalid_reviews == 0, invalid_reviews, "scores between 1 and 5"))

    return checks


def assert_integrity(checks: list[CheckResult]) -> None:
    failed = [check for check in checks if not check.passed]
    if failed:
        summary = "; ".join(f"{check.name}={check.value} expected {check.expectation}" for check in failed)
        raise AssertionError(f"Warehouse integrity checks failed: {summary}")
