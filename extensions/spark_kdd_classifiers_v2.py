"""Production-style Spark ML benchmark for the restored Logistic Regression and Naive Bayes notebooks.

KDD Cup 1999 is retained only as a historical distributed-ML benchmark. This script
uses sklearn's maintained loader, removes exact duplicate rows, builds Spark feature
pipelines on training data only, compares Logistic Regression and Multinomial Naive
Bayes, reports PR-AUC/ROC-AUC plus confusion metrics on an untouched test split, and
verifies that the selected serialized Spark PipelineModel reproduces predictions.

Usage:
    pip install pyspark scikit-learn pandas numpy
    python extensions/spark_kdd_classifiers_v2.py --output-dir spark_kdd_artifacts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]


def decode(value: object) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, (bytes, bytearray)) else str(value)


def main() -> int:
    import numpy as np
    import pandas as pd
    from sklearn.datasets import fetch_kddcup99
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.ml.classification import LogisticRegression, NaiveBayes
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import MinMaxScaler, OneHotEncoder, StringIndexer, VectorAssembler
    from pyspark.sql import SparkSession, functions as F

    p=argparse.ArgumentParser(); p.add_argument("--output-dir",type=Path,default=Path("spark_kdd_artifacts")); p.add_argument("--seed",type=int,default=42); p.add_argument("--sample",type=int,default=150_000); args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    bunch=fetch_kddcup99(percent10=True,as_frame=True,shuffle=False)
    pdf=bunch.frame.copy(); target_name=bunch.target.name if getattr(bunch.target,"name",None) else pdf.columns[-1]
    if target_name not in pdf.columns: target_name=pdf.columns[-1]

    # sklearn's KDD loader may expose many numeric features with object dtype because
    # the raw file also contains byte-valued categoricals. Decode only the three
    # documented categorical predictors and the label; explicitly coerce every other
    # predictor to numeric so Spark receives a real schema rather than strings.
    missing_cats=sorted(set(CATEGORICAL_COLUMNS)-set(pdf.columns))
    if missing_cats: raise KeyError(f"missing KDD categorical columns: {missing_cats}")
    for col in CATEGORICAL_COLUMNS:
        pdf[col]=pdf[col].map(decode)
    raw_target=pdf[target_name].map(decode)
    numeric_columns=[c for c in pdf.columns if c not in {*CATEGORICAL_COLUMNS,target_name}]
    for col in numeric_columns:
        pdf[col]=pd.to_numeric(pdf[col],errors="raise")

    pdf["label"]=(raw_target.astype(str)!="normal.").astype(int)
    pdf=pdf.drop(columns=[target_name]).drop_duplicates().reset_index(drop=True)
    if args.sample and len(pdf)>args.sample:
        # Fixed-seed sample only to keep portfolio reruns practical; split happens afterwards.
        pdf=pdf.sample(args.sample,random_state=args.seed).reset_index(drop=True)

    cat_cols=CATEGORICAL_COLUMNS
    num_cols=[c for c in pdf.columns if c not in {*cat_cols,"label"}]
    if not num_cols: raise AssertionError("numeric feature contract is empty")
    if not all(pd.api.types.is_numeric_dtype(pdf[c]) for c in num_cols):
        bad=[c for c in num_cols if not pd.api.types.is_numeric_dtype(pdf[c])]
        raise AssertionError(f"non-numeric columns leaked into numeric feature set: {bad}")

    spark=SparkSession.builder.master("local[*]").appName("PortfolioSparkKDDV2").config("spark.sql.execution.arrow.pyspark.enabled","false").getOrCreate(); spark.sparkContext.setLogLevel("WARN")
    sdf=spark.createDataFrame(pdf)
    # Stable hash split, preventing preprocessing from seeing validation/test rows.
    all_cols=[c for c in sdf.columns if c!="label"]; sdf=sdf.withColumn("_split",F.pmod(F.xxhash64(*[F.col(c) for c in all_cols]),F.lit(100)))
    train=sdf.filter(F.col("_split")<70).drop("_split").cache(); valid=sdf.filter((F.col("_split")>=70)&(F.col("_split")<85)).drop("_split").cache(); test=sdf.filter(F.col("_split")>=85).drop("_split").cache()
    counts={name:df.count() for name,df in [("train",train),("validation",valid),("test",test)]}
    if min(counts.values())==0: raise AssertionError(f"empty split: {counts}")
    indexers=[StringIndexer(inputCol=c,outputCol=f"{c}__idx",handleInvalid="keep") for c in cat_cols]
    encoder=OneHotEncoder(inputCols=[f"{c}__idx" for c in cat_cols],outputCols=[f"{c}__ohe" for c in cat_cols],handleInvalid="keep")
    assembler=VectorAssembler(inputCols=num_cols+[f"{c}__ohe" for c in cat_cols],outputCol="raw_features",handleInvalid="keep")
    scaler=MinMaxScaler(inputCol="raw_features",outputCol="features")
    classifiers={"logistic_regression":LogisticRegression(labelCol="label",featuresCol="features",maxIter=80,regParam=.01,elasticNetParam=.0),"naive_bayes":NaiveBayes(labelCol="label",featuresCol="features",modelType="multinomial",smoothing=1.0)}
    evaluators={"pr_auc":BinaryClassificationEvaluator(labelCol="label",rawPredictionCol="rawPrediction",metricName="areaUnderPR"),"roc_auc":BinaryClassificationEvaluator(labelCol="label",rawPredictionCol="rawPrediction",metricName="areaUnderROC")}
    def metrics(pred):
        cm={(int(r["label"]),int(r["prediction"])):int(r["count"]) for r in pred.groupBy("label","prediction").count().collect()}; tp=cm.get((1,1),0); tn=cm.get((0,0),0); fp=cm.get((0,1),0); fn=cm.get((1,0),0)
        return {"pr_auc":evaluators["pr_auc"].evaluate(pred),"roc_auc":evaluators["roc_auc"].evaluate(pred),"precision":tp/max(tp+fp,1),"recall":tp/max(tp+fn,1),"specificity":tn/max(tn+fp,1),"confusion":{"tn":tn,"fp":fp,"fn":fn,"tp":tp}}
    fitted={}; validation={}
    for name,clf in classifiers.items():
        model=Pipeline(stages=[*indexers,encoder,assembler,scaler,clf]).fit(train); fitted[name]=model; validation[name]=metrics(model.transform(valid))
    winner=max(validation,key=lambda name:validation[name]["pr_auc"]); test_metrics={name:metrics(model.transform(test)) for name,model in fitted.items()}
    model_path=args.output_dir/f"{winner}_pipeline"; fitted[winner].write().overwrite().save(str(model_path)); reloaded=PipelineModel.load(str(model_path))
    sample=test.limit(1000).cache(); before=[(r.label,r.prediction) for r in fitted[winner].transform(sample).select("label","prediction").collect()]; after=[(r.label,r.prediction) for r in reloaded.transform(sample).select("label","prediction").collect()]; reload_match=before==after
    prevalence=float(train.agg(F.avg("label")).first()[0]); report={"dataset":"KDD Cup 1999 10% via sklearn maintained loader","freshness_warning":"Historical benchmark only; not evidence of modern intrusion-detection performance.","deduplicated_rows":len(pdf),"splits":counts,"schema":{"categorical":cat_cols,"numeric_feature_count":len(num_cols)},"attack_prevalence_train":prevalence,"validation":validation,"selected_on_validation_pr_auc":winner,"untouched_test":test_metrics,"serialized_pipeline":str(model_path),"reload_prediction_match":reload_match,"methodology":["documented categorical columns decoded explicitly; all remaining predictors coerced to numeric","exact duplicates removed before split","stable row-hash train/validation/test split","categorical index/OHE and min-max scaling fitted inside Spark Pipeline on training data only","model choice made on validation PR-AUC; test opened only for final reporting"]}
    (args.output_dir/"evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2)); spark.stop(); return 0 if reload_match else 1

if __name__=="__main__": raise SystemExit(main())
