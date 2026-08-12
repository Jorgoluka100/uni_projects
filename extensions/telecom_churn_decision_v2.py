"""Decision-ready completion layer for Strategic Telecom Churn + Predictive SQL.

Keeps the original synthetic-business-case premise but fixes the modelling/evaluation
weaknesses: one-hot preprocessing is fitted on training data only, model selection and
review-capacity thresholding use validation data only, the test split is untouched,
and SQL aggregate outputs are reconciled against pandas at the same customer grain.

Synthetic methodology demo: reported metrics do not describe a real telecom market.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> int:
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    p=argparse.ArgumentParser(); p.add_argument("--output-dir",type=Path,default=Path("telecom_churn_artifacts")); p.add_argument("--seed",type=int,default=42); p.add_argument("--review-capacity",type=float,default=.15); args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed); n=12000
    df=pd.DataFrame({"customer_id":[f"TEL-{i:06d}" for i in range(n)],"contract":rng.choice(["Month-to-month","One year","Two year"],n,p=[.5,.2,.3]),"tenure":rng.integers(1,73,n),"monthly_charges":rng.uniform(20,120,n),"region":rng.choice(["North","South","East","West"],n),"payment_method":rng.choice(["Electronic check","Mailed check","Bank transfer","Credit card"],n)})
    df["total_charges"]=df["monthly_charges"]*df["tenure"]
    logit=-2.4+1.3*(df["contract"]=="Month-to-month")+.012*(df["monthly_charges"]-70)-.018*(df["tenure"]-24)+.35*(df["payment_method"]=="Electronic check")
    probability=1/(1+np.exp(-logit)); df["has_churned"]=(rng.random(n)<probability).astype(int)
    if df["customer_id"].duplicated().any(): raise AssertionError("duplicate customer IDs")
    trainval,test=train_test_split(np.arange(n),test_size=.20,stratify=df["has_churned"],random_state=args.seed); train,valid=train_test_split(trainval,test_size=.25,stratify=df.iloc[trainval]["has_churned"],random_state=args.seed+1)
    features=["contract","tenure","monthly_charges","region","payment_method"] # total_charges intentionally omitted: deterministic derivative of current charge * tenure.
    cats=["contract","region","payment_method"]; nums=["tenure","monthly_charges"]
    pre=ColumnTransformer([("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),cats),("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums)])
    candidates={"logistic":LogisticRegression(max_iter=1000,class_weight="balanced",random_state=args.seed),"random_forest":RandomForestClassifier(n_estimators=350,min_samples_leaf=8,class_weight="balanced_subsample",n_jobs=-1,random_state=args.seed)}
    fitted={}; val_scores={}
    for name,est in candidates.items():
        model=Pipeline([("pre",pre),("model",est)]); model.fit(df.iloc[train][features],df.iloc[train]["has_churned"]); prob=model.predict_proba(df.iloc[valid][features])[:,1]; val_scores[name]={"pr_auc":average_precision_score(df.iloc[valid]["has_churned"],prob),"roc_auc":roc_auc_score(df.iloc[valid]["has_churned"],prob)}; fitted[name]=model
    winner=max(val_scores,key=lambda name:val_scores[name]["pr_auc"]); model=fitted[winner]; val_prob=model.predict_proba(df.iloc[valid][features])[:,1]
    # Review exactly the highest-risk validation fraction; threshold is never tuned on test.
    threshold=float(np.quantile(val_prob,1-args.review_capacity,method="higher")); test_prob=model.predict_proba(df.iloc[test][features])[:,1]; pred=(test_prob>=threshold).astype(int); truth=df.iloc[test]["has_churned"].to_numpy()
    test_metrics={"pr_auc":average_precision_score(truth,test_prob),"roc_auc":roc_auc_score(truth,test_prob),"threshold_from_validation":threshold,"review_rate":float(pred.mean()),"precision_at_capacity":precision_score(truth,pred,zero_division=0),"recall_at_capacity":recall_score(truth,pred,zero_division=0),"positive_prevalence":float(truth.mean())}
    # Reconcile SQL and pandas aggregations at identical region/contract grain.
    conn=sqlite3.connect(":memory:"); df.to_sql("customers",conn,index=False)
    sql=pd.read_sql_query("SELECT region, contract, COUNT(*) AS total_subs, SUM(has_churned) AS churn_count, AVG(monthly_charges) AS avg_mrr FROM customers GROUP BY region, contract ORDER BY region, contract",conn)
    pan=df.groupby(["region","contract"],as_index=False).agg(total_subs=("customer_id","size"),churn_count=("has_churned","sum"),avg_mrr=("monthly_charges","mean")).sort_values(["region","contract"]).reset_index(drop=True); sql=sql.sort_values(["region","contract"]).reset_index(drop=True)
    reconciled=bool((sql[["total_subs","churn_count"]].to_numpy()==pan[["total_subs","churn_count"]].to_numpy()).all() and np.allclose(sql["avg_mrr"],pan["avg_mrr"]))
    report={"data":"deterministic synthetic telecom methodology demo","rows":n,"split":{"train":len(train),"validation":len(valid),"test":len(test)},"features":features,"validation_model_selection":val_scores,"selected_model":winner,"review_capacity_target":args.review_capacity,"untouched_test":test_metrics,"sql_pandas_reconciled":reconciled,"limitations":["synthetic labels encode assumed business relationships","metrics do not transfer to a real telecom population","capacity threshold must be reset against real retention-team capacity and intervention value"]}
    (args.output_dir/"evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); sql.to_csv(args.output_dir/"sql_segment_metrics.csv",index=False); print(json.dumps(report,indent=2)); return 0 if reconciled else 1

if __name__=="__main__": raise SystemExit(main())
