from __future__ import annotations


def build_sessions(events):
    """Aggregate event-level clickstream data to one row per real session."""
    from pyspark.sql import functions as F

    return (
        events.groupBy("session_id")
        .agg(
            F.count("*").alias("clicks"),
            F.countDistinct("product_id").alias("unique_products"),
            F.countDistinct("main_category").alias("categories_viewed"),
            F.avg("price").alias("avg_viewed_price"),
            F.max("page").alias("max_page"),
            F.min("event_date").alias("session_date"),
            F.first("country").alias("country"),
        )
        .withColumn(
            "depth_segment",
            F.when(F.col("clicks") >= 10, "10+ clicks")
            .when(F.col("clicks") >= 5, "5-9 clicks")
            .when(F.col("clicks") >= 2, "2-4 clicks")
            .otherwise("1 click"),
        )
    )


def engagement_funnel(sessions) -> dict[str, int]:
    """Return engagement depth; deliberately not labelled as a purchase funnel."""
    from pyspark.sql import functions as F

    row = sessions.agg(
        F.count("*").alias("all_sessions"),
        F.sum((F.col("clicks") >= 2).cast("int")).alias("two_plus"),
        F.sum((F.col("clicks") >= 5).cast("int")).alias("five_plus"),
        F.sum((F.col("clicks") >= 10).cast("int")).alias("ten_plus"),
    ).first()
    return {
        "all_sessions": int(row.all_sessions),
        "2+ clicks": int(row.two_plus),
        "5+ clicks": int(row.five_plus),
        "10+ clicks": int(row.ten_plus),
    }


def replicate_for_load_test(events, target_rows: int = 1_000_000):
    """Create clearly labelled replicated rows for Spark load testing only."""
    from pyspark.sql import functions as F

    source_rows = events.count()
    repeats = max(1, (target_rows + source_rows - 1) // source_rows)
    replicated = (
        events.crossJoin(events.sparkSession.range(repeats).withColumnRenamed("id", "replica_id"))
        .limit(target_rows)
        .withColumn("load_test_replica", F.lit(True))
    )
    return replicated
