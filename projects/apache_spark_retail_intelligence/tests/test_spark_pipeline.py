import pytest

pyspark = pytest.importorskip("pyspark")

from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from run import build_customer_features, build_spark, enrich_events, generate_events, validate_raw


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
