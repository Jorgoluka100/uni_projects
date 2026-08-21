from __future__ import annotations

SEED = 42
CATEGORICAL = ["Month", "VisitorType"]
NUMERIC = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "WeekendInt",
]
LEAKAGE_EXCLUDED = ["PageValues"]


def deterministic_split(frame):
    """Use row-content hashing so the 70/15/15 split is reproducible and disjoint."""
    from pyspark.sql import functions as F

    hashed = frame.withColumn(
        "row_hash",
        F.pmod(F.xxhash64(*[F.col(column) for column in frame.columns]), F.lit(100)),
    )
    return (
        hashed.filter("row_hash < 70"),
        hashed.filter("row_hash >= 70 AND row_hash < 85"),
        hashed.filter("row_hash >= 85"),
    )


def build_pipeline():
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import GBTClassifier
    from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

    indexers = [
        StringIndexer(inputCol=column, outputCol=f"{column}_idx", handleInvalid="keep")
        for column in CATEGORICAL
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{column}_idx" for column in CATEGORICAL],
        outputCols=[f"{column}_ohe" for column in CATEGORICAL],
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=NUMERIC + [f"{column}_ohe" for column in CATEGORICAL],
        outputCol="features",
        handleInvalid="keep",
    )
    model = GBTClassifier(
        labelCol="label",
        featuresCol="features",
        maxIter=50,
        maxDepth=5,
        stepSize=0.05,
        subsamplingRate=0.8,
        seed=SEED,
    )
    return Pipeline(stages=indexers + [encoder, assembler, model])


def threshold_metrics(scored, threshold: float) -> dict[str, float]:
    from pyspark.sql import functions as F

    row = (
        scored.withColumn("pred", (F.col("score") >= float(threshold)).cast("int"))
        .agg(
            F.sum(((F.col("pred") == 1) & (F.col("label") == 1)).cast("int")).alias("tp"),
            F.sum(((F.col("pred") == 1) & (F.col("label") == 0)).cast("int")).alias("fp"),
            F.sum(((F.col("pred") == 0) & (F.col("label") == 1)).cast("int")).alias("fn"),
            F.sum(((F.col("pred") == 0) & (F.col("label") == 0)).cast("int")).alias("tn"),
        )
        .first()
    )
    precision = row.tp / max(row.tp + row.fp, 1)
    recall = row.tp / max(row.tp + row.fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    total = row.tp + row.fp + row.fn + row.tn
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "alert_rate": float((row.tp + row.fp) / total),
    }
