"""Focused SQL analytics fundamentals using an in-memory SQLite database."""

import sqlite3
import pandas as pd


def main() -> None:
    con = sqlite3.connect(":memory:")

    customers = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "region": ["London", "London", "Leeds", "Bristol"],
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": [101, 102, 103, 104, 105, 106],
            "customer_id": [1, 1, 2, 3, 4, 4],
            "amount": [120.0, 80.0, 220.0, 95.0, 140.0, 60.0],
            "order_date": [
                "2026-01-01",
                "2026-01-12",
                "2026-02-03",
                "2026-02-10",
                "2026-03-05",
                "2026-03-20",
            ],
        }
    )

    customers.to_sql("customers", con, index=False)
    orders.to_sql("orders", con, index=False)

    query = """
    WITH customer_value AS (
      SELECT customer_id,
             COUNT(*) AS orders,
             SUM(amount) AS revenue
      FROM orders
      GROUP BY customer_id
    )
    SELECT c.region,
           COUNT(*) AS customers,
           SUM(v.orders) AS orders,
           ROUND(SUM(v.revenue), 2) AS revenue,
           RANK() OVER (ORDER BY SUM(v.revenue) DESC) AS revenue_rank
    FROM customer_value v
    JOIN customers c USING (customer_id)
    GROUP BY c.region
    ORDER BY revenue DESC
    """

    print(pd.read_sql_query(query, con).to_string(index=False))


if __name__ == "__main__":
    main()
