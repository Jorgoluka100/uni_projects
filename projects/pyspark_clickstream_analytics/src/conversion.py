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

    if not frame.columns:
        raise ValueError("frame must contain columns")
    if "row_hash" in frame.columns:
        raise ValueError("input frame already contains the reserved row_hash column")
    if frame.limit(1).count() == 0:
        raise ValueError("cannot split an empty clickstream frame")
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
    import math

    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and within [0, 1]")
    required = {"score", "label"}
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"scored frame is missing columns: {missing}")
    invalid = scored.filter(
        F.col("score").isNull()
        | F.isnan(F.col("score"))
        | (F.col("score") < 0)
        | (F.col("score") > 1)
        | F.col("label").isNull()
        | (~F.col("label").isin(0, 1))
    ).limit(1).count()
    if invalid:
        raise ValueError("score must be finite in [0, 1] and label must be binary 0/1")
    row = (
        scored.withColumn("pred", (F.col("score") >= float(threshold)).cast("int"))
        .agg(
            F.sum(((F.col("pred") == 1) & (F.col("label") == 1)).cast("int")).alias("tp"),
            F.sum(((F.col("pred") == 1) & (F.col("label") == 0)).cast("int")).alias("fp"),
            F.sum(((F.col("pred") == 0) & (F.col("label") == 1)).cast("int")).alias("fn"),
            F.sum(((F.col("pred") == 0) & (F.col("label") == 0)).cast("int")).alias("tn"),
            F.count("*").alias("total"),
        )
        .first()
    )
    if not row or row.total == 0:
        raise ValueError("scored frame must not be empty")
    precision = row.tp / max(row.tp + row.fp, 1)
    recall = row.tp / max(row.tp + row.fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    total = row.total
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "alert_rate": float((row.tp + row.fp) / total),
    }
