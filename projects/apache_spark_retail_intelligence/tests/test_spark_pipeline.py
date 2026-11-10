import pytest

pyspark = pytest.importorskip("pyspark")

from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from run import build_customer_features, build_spark, enrich_events, expected_schema, generate_events, validate_raw


@pytest.fixture(scope="module")
def spark():
    session = build_spark("RetailIntelligenceTests")
    yield session
    session.stop()


def test_generated_event_contract(spark):
    df = generate_events(spark, 2000)
    report = validate_raw(df)
    assert report["rows"] == 2000
    assert report["duplicate_event_ids"] == 0
    assert report["null_cells"] == 0


def test_customer_features_have_one_row_per_customer(spark):
    events = enrich_events(generate_events(spark, 3000))
    features = build_customer_features(events)
    assert features.count() == features.select("customer_id").distinct().count()
    assert features.where("transactions <= 0").count() == 0

def test_raw_contract_rejects_empty_data_and_invalid_binary_flags(spark):
    from pyspark.sql import functions as F

    empty = spark.createDataFrame([], expected_schema())
    with pytest.raises(ValueError, match="must not be empty"):
        validate_raw(empty)
    invalid = generate_events(spark, 1000).withColumn(
        "repeat_purchase",
        F.when(F.col("event_id") == 0, F.lit(2)).otherwise(F.col("repeat_purchase")),
    )
    with pytest.raises(ValueError, match="Raw data contract failed"):
        validate_raw(invalid)


def test_generation_and_customer_features_reject_invalid_sizes(spark):
    with pytest.raises(ValueError, match="at least 1,000"):
        generate_events(spark, 999)
    empty = spark.createDataFrame([], expected_schema())
    with pytest.raises(ValueError, match="non-empty"):
        build_customer_features(empty)
