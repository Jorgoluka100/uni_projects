from __future__ import annotations

import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

from src.conversion import deterministic_split, threshold_metrics


@pytest.fixture(scope="module")
def spark():
    session = SparkSession.builder.master("local[1]").appName("ClickstreamConversionTests").getOrCreate()
    yield session
    session.stop()


def _scored(spark, rows):
    schema = StructType([
        StructField("score", DoubleType(), False),
        StructField("label", IntegerType(), False),
    ])
    return spark.createDataFrame(rows, schema)


def test_threshold_metrics_reject_invalid_threshold_and_probabilities(spark):
    with pytest.raises(ValueError, match="threshold"):
        threshold_metrics(_scored(spark, [(0.2, 0), (0.8, 1)]), 1.5)
    with pytest.raises(ValueError, match="score must be finite"):
        threshold_metrics(_scored(spark, [(1.2, 1)]), 0.5)


def test_threshold_metrics_returns_confusion_derived_rates(spark):
    result = threshold_metrics(_scored(spark, [(0.9, 1), (0.8, 0), (0.2, 0), (0.1, 1)]), 0.5)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["alert_rate"] == 0.5


def test_deterministic_split_rejects_empty_and_reserved_hash_frames(spark):
    schema = StructType([StructField("event_id", IntegerType(), False)])
    empty = spark.createDataFrame([], schema)
    with pytest.raises(ValueError, match="empty"):
        deterministic_split(empty)
    reserved = spark.createDataFrame([(1, 7)], ["event_id", "row_hash"])
    with pytest.raises(ValueError, match="reserved"):
        deterministic_split(reserved)
