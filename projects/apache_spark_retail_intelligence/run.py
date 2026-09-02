from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType, StructField, StructType, TimestampType

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
RESULTS.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)
SEED = 42


def build_spark(app_name: str = "RetailIntelligence") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def expected_schema() -> StructType:
    return StructType(
        [
            StructField("event_id", LongType(), False),
            StructField("customer_id", LongType(), False),
            StructField("product_id", LongType(), False),
            StructField("channel", StringType(), False),
            StructField("event_ts", TimestampType(), False),
            StructField("quantity", IntegerType(), False),
            StructField("unit_price", DoubleType(), False),
            StructField("discount", DoubleType(), False),
            StructField("is_return", IntegerType(), False),
            StructField("repeat_purchase", IntegerType(), False),
        ]
    )


def generate_events(spark: SparkSession, rows: int) -> DataFrame:
    if rows < 1000:
        raise ValueError("Use at least 1,000 rows for a meaningful Spark workload")

    base = spark.range(0, rows).withColumnRenamed("id", "event_id")
    epoch = F.to_timestamp(F.lit("2025-01-01 00:00:00"))
    df = (
        base
        .withColumn("customer_id", (F.pmod(F.hash("event_id", F.lit(SEED)), F.lit(50000)) + 1).cast("long"))
        .withColumn("product_id", (F.pmod(F.hash("event_id", F.lit(7)), F.lit(4000)) + 1).cast("long"))
        .withColumn(
            "channel",
            F.when(F.pmod(F.col("event_id"), F.lit(10)) < 5, F.lit("web"))
            .when(F.pmod(F.col("event_id"), F.lit(10)) < 8, F.lit("mobile"))
            .otherwise(F.lit("marketplace")),
        )
        .withColumn(
            "event_ts",
            F.expr("timestampadd(MINUTE, cast(pmod(event_id * 37, 525600) as int), timestamp'2025-01-01 00:00:00')"),
        )
        .withColumn("quantity", (F.pmod(F.hash("event_id", F.lit(11)), F.lit(5)) + 1).cast("int"))
        .withColumn("unit_price", (F.lit(5.0) + F.pmod(F.hash("event_id", F.lit(13)), F.lit(49500)) / F.lit(100.0)).cast("double"))
        .withColumn("discount", (F.pmod(F.hash("event_id", F.lit(17)), F.lit(3000)) / F.lit(10000.0)).cast("double"))
        .withColumn("is_return", (F.pmod(F.hash("event_id", F.lit(19)), F.lit(20)) == 0).cast("int"))
        .withColumn(
            "repeat_purchase",
            (
                (F.pmod(F.col("customer_id"), F.lit(7)) < 4)
                | ((F.col("channel") == "mobile") & (F.col("discount") > 0.12))
            ).cast("int"),
        )
    )
    return df.select([field.name for field in expected_schema().fields])


def validate_raw(df: DataFrame) -> dict[str, int]:
    required = [field.name for field in expected_schema().fields]
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    row_count = df.count()
    duplicate_event_ids = df.groupBy("event_id").count().where(F.col("count") > 1).count()
    null_conditions = [F.col(c).isNull().cast("int") for c in required]
    null_cells = df.select(sum(null_conditions).alias("nulls")).agg(F.sum("nulls")).first()[0] or 0
    invalid_quantity = df.where(F.col("quantity") <= 0).count()
    invalid_price = df.where(F.col("unit_price") <= 0).count()
    invalid_discount = df.where((F.col("discount") < 0) | (F.col("discount") > 1)).count()

    report = {
        "rows": int(row_count),
        "duplicate_event_ids": int(duplicate_event_ids),
        "null_cells": int(null_cells),
        "invalid_quantity": int(invalid_quantity),
        "invalid_price": int(invalid_price),
        "invalid_discount": int(invalid_discount),
    }
    if any(report[key] for key in report if key != "rows"):
        raise ValueError(f"Raw data contract failed: {report}")
    return report


def enrich_events(df: DataFrame) -> DataFrame:
    return (
        df
        .withColumn("gross_revenue", F.col("quantity") * F.col("unit_price"))
        .withColumn("discount_value", F.col("gross_revenue") * F.col("discount"))
        .withColumn("net_revenue_before_returns", F.col("gross_revenue") - F.col("discount_value"))
        .withColumn(
            "net_revenue",
            F.when(F.col("is_return") == 1, -F.col("net_revenue_before_returns"))
            .otherwise(F.col("net_revenue_before_returns")),
        )
        .withColumn("event_date", F.to_date("event_ts"))
        .withColumn("event_month", F.date_trunc("month", "event_ts"))
        .withColumn("hour", F.hour("event_ts"))
        .withColumn("day_of_week", F.dayofweek("event_ts"))
    )


def build_customer_features(events: DataFrame) -> DataFrame:
    customer_window = Window.partitionBy("customer_id").orderBy(F.col("event_ts").asc())
    ordered = (
        events
        .withColumn("previous_event_ts", F.lag("event_ts").over(customer_window))
        .withColumn(
            "days_since_previous",
            F.datediff(F.to_date("event_ts"), F.to_date("previous_event_ts")),
        )
    )

    max_ts = ordered.agg(F.max("event_ts").alias("max_ts")).first()["max_ts"]
    features = (
        ordered
        .groupBy("customer_id")
        .agg(
            F.count("event_id").alias("transactions"),
            F.sum("quantity").alias("units"),
            F.sum("net_revenue").alias("net_revenue"),
            F.avg("unit_price").alias("avg_unit_price"),
            F.avg("discount").alias("avg_discount"),
            F.avg("is_return").alias("return_rate"),
            F.countDistinct("product_id").alias("unique_products"),
            F.max("event_ts").alias("last_event_ts"),
            F.avg("days_since_previous").alias("avg_days_between_orders"),
            F.max("repeat_purchase").alias("repeat_purchase"),
            F.sum(F.when(F.col("channel") == "web", 1).otherwise(0)).alias("web_orders"),
            F.sum(F.when(F.col("channel") == "mobile", 1).otherwise(0)).alias("mobile_orders"),
            F.sum(F.when(F.col("channel") == "marketplace", 1).otherwise(0)).alias("marketplace_orders"),
        )
        .withColumn("recency_days", F.datediff(F.lit(max_ts.date().isoformat()), F.to_date("last_event_ts")))
        .fillna({"avg_days_between_orders": 365.0})
        .withColumn("revenue_per_transaction", F.col("net_revenue") / F.greatest(F.col("transactions"), F.lit(1)))
        .withColumn("mobile_share", F.col("mobile_orders") / F.greatest(F.col("transactions"), F.lit(1)))
    )
    return features


def business_kpis(events: DataFrame) -> dict[str, object]:
    totals = events.agg(
        F.count("event_id").alias("events"),
        F.countDistinct("customer_id").alias("customers"),
        F.sum("net_revenue").alias("net_revenue"),
        F.avg("is_return").alias("return_rate"),
        F.avg("discount").alias("avg_discount"),
    ).first().asDict()

    channel_rows = (
        events.groupBy("channel")
        .agg(
            F.count("event_id").alias("events"),
            F.sum("net_revenue").alias("net_revenue"),
            F.avg("is_return").alias("return_rate"),
        )
        .orderBy(F.desc("net_revenue"))
        .collect()
    )
    channels = [row.asDict() for row in channel_rows]
    for row in channels:
        for key, value in list(row.items()):
            if isinstance(value, float):
                row[key] = float(value)
    return {
        "totals": {
            "events": int(totals["events"]),
            "customers": int(totals["customers"]),
            "net_revenue": float(totals["net_revenue"]),
            "return_rate": float(totals["return_rate"]),
            "avg_discount": float(totals["avg_discount"]),
        },
        "channels": channels,
    }


def fit_repeat_purchase_model(features: DataFrame) -> tuple[Pipeline, object, dict[str, float]]:
    feature_columns = [
        "transactions",
        "units",
        "net_revenue",
        "avg_unit_price",
        "avg_discount",
        "return_rate",
        "unique_products",
        "avg_days_between_orders",
        "recency_days",
        "revenue_per_transaction",
        "mobile_share",
    ]
    train, test = features.randomSplit([0.8, 0.2], seed=SEED)
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features", handleInvalid="keep")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
    classifier = LogisticRegression(
        featuresCol="features",
        labelCol="repeat_purchase",
        maxIter=100,
        regParam=0.05,
        elasticNetParam=0.1,
    )
    pipeline = Pipeline(stages=[assembler, scaler, classifier])
    model = pipeline.fit(train)
    predictions = model.transform(test).cache()

    auc = BinaryClassificationEvaluator(
        labelCol="repeat_purchase",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    ).evaluate(predictions)
    accuracy = MulticlassClassificationEvaluator(
        labelCol="repeat_purchase",
        predictionCol="prediction",
        metricName="accuracy",
    ).evaluate(predictions)
    f1 = MulticlassClassificationEvaluator(
        labelCol="repeat_purchase",
        predictionCol="prediction",
        metricName="f1",
    ).evaluate(predictions)
    test_rows = predictions.count()
    metrics = {
        "test_rows": int(test_rows),
        "auc": float(auc),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }
    predictions.unpersist()
    return pipeline, model, metrics


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000000)
    parser.add_argument("--write-parquet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    try:
        raw = generate_events(spark, args.rows).cache()
        audit = validate_raw(raw)
        events = enrich_events(raw).cache()
        customers = build_customer_features(events).cache()
        kpis = business_kpis(events)
        _, model, model_metrics = fit_repeat_purchase_model(customers)

        if args.write_parquet:
            output = str((ARTIFACTS / "customer_features_parquet").resolve())
            customers.repartition(8, "customer_id").write.mode("overwrite").partitionBy("repeat_purchase").parquet(output)

        model_path = str((ARTIFACTS / "spark_repeat_purchase_pipeline").resolve())
        model.write().overwrite().save(model_path)

        feature_summary = customers.agg(
            F.count("customer_id").alias("customers"),
            F.avg("transactions").alias("avg_transactions"),
            F.avg("net_revenue").alias("avg_customer_revenue"),
            F.avg("return_rate").alias("avg_customer_return_rate"),
            F.avg("repeat_purchase").alias("repeat_purchase_rate"),
        ).first().asDict()
        feature_summary = {
            key: (int(value) if key == "customers" else float(value))
            for key, value in feature_summary.items()
        }

        payload = {
            "raw_audit": audit,
            "business_kpis": kpis,
            "customer_feature_summary": feature_summary,
            "model_metrics": model_metrics,
            "spark": {
                "version": spark.version,
                "shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
                "adaptive_execution": spark.conf.get("spark.sql.adaptive.enabled"),
            },
            "limitations": [
                "Synthetic workload: engineering benchmark, not real retailer behaviour.",
                "Local Spark mode does not benchmark cloud cluster cost or skew under production concurrency.",
                "Repeat-purchase label is generated and therefore model metrics must not be sold as external performance evidence.",
            ],
        }
        save_json(RESULTS / "metrics.json", payload)
        print(json.dumps(payload, indent=2, default=str))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
