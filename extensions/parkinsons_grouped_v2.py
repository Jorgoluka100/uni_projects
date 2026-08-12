"""Patient-grouped evaluation for the restored Parkinson's progression notebook.

The original notebook randomly split repeated measurements, allowing the same subject
to appear in train and test. This extension holds out complete subjects, removes the
closely related motor_UPDRS outcome from predictors of total_UPDRS, compares a simple
training-median baseline against a Random Forest, and bootstraps uncertainty at the
subject level.

Educational/non-clinical use only. The UCI Parkinsons Telemonitoring dataset is a
historical research dataset and this script does not establish clinical validity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"


def main() -> int:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    p=argparse.ArgumentParser(); p.add_argument("--output-dir",type=Path,default=Path("parkinsons_grouped_artifacts")); p.add_argument("--seed",type=int,default=42); p.add_argument("--bootstrap-rounds",type=int,default=2000); args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA_URL); required={"subject#","total_UPDRS","motor_UPDRS"}; missing=required-set(df.columns)
    if missing: raise KeyError(missing)
    target="total_UPDRS"; group="subject#"; excluded={target,"motor_UPDRS",group}; features=[c for c in df.columns if c not in excluded]
    X=df[features]; y=df[target].to_numpy(float); groups=df[group].to_numpy()
    outer=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=args.seed); trainval,test=next(outer.split(X,y,groups)); inner=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=args.seed+1); tr_rel,val_rel=next(inner.split(X.iloc[trainval],y[trainval],groups[trainval])); train=trainval[tr_rel]; valid=trainval[val_rel]
    if set(groups[train])&set(groups[test]) or set(groups[valid])&set(groups[test]): raise AssertionError("subject leakage")
    model=Pipeline([("impute",SimpleImputer(strategy="median")),("rf",RandomForestRegressor(n_estimators=500,min_samples_leaf=3,max_features=.8,n_jobs=-1,random_state=args.seed))]); model.fit(X.iloc[train],y[train]); baseline=float(np.median(y[train]))
    def evaluate(idx):
        pred=model.predict(X.iloc[idx]); truth=y[idx]; return {"rows":len(idx),"subjects":len(set(groups[idx])),"mae":mean_absolute_error(truth,pred),"rmse":mean_squared_error(truth,pred)**.5,"r2":r2_score(truth,pred),"baseline_median_mae":mean_absolute_error(truth,np.full(len(idx),baseline)),"mae_improvement_vs_baseline":1-mean_absolute_error(truth,pred)/mean_absolute_error(truth,np.full(len(idx),baseline))},pred
    val_metrics,_=evaluate(valid); test_metrics,test_pred=evaluate(test)
    # Cluster bootstrap: resample subjects, not rows, preserving within-subject dependence.
    rng=np.random.default_rng(args.seed); test_subjects=np.unique(groups[test]); test_groups=groups[test]; truth=y[test]; values=[]
    for _ in range(args.bootstrap_rounds):
        sampled=rng.choice(test_subjects,size=len(test_subjects),replace=True); indices=np.concatenate([np.where(test_groups==subject)[0] for subject in sampled]); values.append(mean_absolute_error(truth[indices],test_pred[indices]))
    ci=np.quantile(values,[.025,.975]).tolist(); test_metrics["subject_bootstrap_mae_ci95"]={"low":ci[0],"high":ci[1],"rounds":args.bootstrap_rounds}
    report={"dataset":"UCI Parkinsons Telemonitoring","source":DATA_URL,"rows":len(df),"subjects":int(df[group].nunique()),"target":target,"excluded_predictors":[group,"motor_UPDRS"],"split":"complete-subject train/validation/test holdout","split_subjects":{"train":len(set(groups[train])),"validation":len(set(groups[valid])),"test":len(set(groups[test]))},"validation":val_metrics,"untouched_test":test_metrics,"guardrails":["educational/non-clinical model only","historical research cohort; external validity is not established","motor_UPDRS excluded to avoid using a closely related clinical outcome as a shortcut","uncertainty is bootstrapped by subject rather than treating repeated rows as independent"]}
    (args.output_dir/"evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); pd.DataFrame({"subject":groups[test],"truth":truth,"prediction":test_pred}).to_csv(args.output_dir/"test_predictions.csv",index=False); print(json.dumps(report,indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
