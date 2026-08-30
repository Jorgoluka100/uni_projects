"""Create a deterministic DuckDB source fixture for the dbt smoke test."""
from pathlib import Path

import duckdb

DB_PATH = Path(__file__).parent / "artifacts" / "dbt_smoke.duckdb"


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE SCHEMA raw")

    con.execute("""
        CREATE TABLE raw.customers (
            customer_id VARCHAR,
            customer_unique_id VARCHAR,
            customer_city VARCHAR,
            customer_state VARCHAR
        )
    """)
    con.execute("""
        INSERT INTO raw.customers VALUES
        ('c1','u1','sao_paulo','SP'),
        ('c2','u1','sao_paulo','SP'),
        ('c3','u2','rio','RJ')
    """)

    con.execute("""
        CREATE TABLE raw.orders (
            order_id VARCHAR,
            customer_id VARCHAR,
            order_status VARCHAR,
            order_purchase_timestamp VARCHAR,
            order_delivered_customer_date VARCHAR,
            order_estimated_delivery_date VARCHAR
        )
    """)
    con.execute("""
        INSERT INTO raw.orders VALUES
        ('o1','c1','delivered','2017-01-05 08:00:00','2017-01-10 12:00:00','2017-01-12 00:00:00'),
        ('o2','c2','delivered','2017-02-05 09:00:00','2017-02-20 12:00:00','2017-02-15 00:00:00'),
        ('o3','c3','delivered','2017-02-10 10:00:00','2017-02-14 12:00:00','2017-02-16 00:00:00')
    """)

    con.execute("""
        CREATE TABLE raw.order_items (
            order_id VARCHAR,
            order_item_id INTEGER,
            product_id VARCHAR,
            seller_id VARCHAR,
            price DOUBLE,
            freight_value DOUBLE
        )
    """)
    con.execute("""
        INSERT INTO raw.order_items VALUES
        ('o1',1,'p1','s1',100.0,10.0),
        ('o1',2,'p2','s1',50.0,5.0),
        ('o2',1,'p1','s1',80.0,5.0),
        ('o3',1,'p2','s2',20.0,5.0)
    """)

    con.execute("""
        CREATE TABLE raw.payments (
            order_id VARCHAR,
            payment_sequential INTEGER,
            payment_type VARCHAR,
            payment_installments INTEGER,
            payment_value DOUBLE
        )
    """)
    con.execute("""
        INSERT INTO raw.payments VALUES
        ('o1',1,'credit_card',2,120.0),
        ('o1',2,'voucher',1,45.0),
        ('o2',1,'credit_card',1,85.0),
        ('o3',1,'debit_card',1,25.0)
    """)

    con.execute("""
        CREATE TABLE raw.reviews (
            review_id VARCHAR,
            order_id VARCHAR,
            review_score INTEGER,
            review_creation_date VARCHAR,
            review_answer_timestamp VARCHAR
        )
    """)
    con.execute("""
        INSERT INTO raw.reviews VALUES
        ('r1','o1',5,'2017-01-11','2017-01-11 09:00:00'),
        ('r1b','o1',4,'2017-01-12','2017-01-12 09:00:00'),
        ('r2','o2',2,'2017-02-21','2017-02-21 09:00:00'),
        ('r3','o3',5,'2017-02-15','2017-02-15 09:00:00')
    """)

    con.close()
    print(f"Prepared dbt fixture: {DB_PATH}")


if __name__ == "__main__":
    main()
