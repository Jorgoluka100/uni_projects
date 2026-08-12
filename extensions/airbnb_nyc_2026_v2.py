"""NYC Airbnb market analysis refresh using Inside Airbnb's 14 June 2026 snapshot.

This extension separates descriptive market analysis from predictive evaluation and
uses neighbourhood-group holdouts so test rows come from neighbourhoods unseen during
training. It caches the official public snapshot locally and documents the source
limitations instead of treating availability as confirmed bookings.

Source: Inside Airbnb, New York City, 14 June 2026, CC BY 4.0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

DATA_URL = "https://data.insideairbnb.com/united-states/ny/new-york-city/2026-06-14/data/listings.csv.gz"
SNAPSHOT_DATE = "2026-06-14"


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()


def main() -> int:
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    p=argparse.ArgumentParser(); p.add_argument("--data-dir",type=Path,default=Path("data_cache")); p.add_argument("--output-dir",type=Path,default=Path("airbnb_nyc_2026_artifacts")); p.add_argument("--seed",type=int,default=42); args=p.parse_args()
    args.data_dir.mkdir(parents=True,exist_ok=True); args.output_dir.mkdir(parents=True,exist_ok=True)
    cached=args.data_dir/"nyc_listings_2026-06-14.csv.gz"
    if not cached.exists(): urllib.request.urlretrieve(DATA_URL,cached)
    frame=pd.read_csv(cached,low_memory=False)
    required={"id","price","room_type","neighbourhood_cleansed","neighbourhood_group_cleansed","latitude","longitude","accommodates","bedrooms","beds","minimum_nights","number_of_reviews","reviews_per_month","availability_365","host_listings_count"}
    missing=sorted(required-set(frame.columns))
    if missing: raise KeyError(f"missing required columns: {missing}")
    if frame["id"].duplicated().any(): raise ValueError("duplicate listing IDs")
    raw_rows=len(frame)
    frame["price_usd"]=pd.to_numeric(frame["price"].astype(str).str.replace(r"[$,]","",regex=True),errors="coerce")
    # Price modelling excludes impossible/missing values and the extreme top 0.5% tail; exclusion counts are retained.
    valid=frame[frame["price_usd"].gt(0)].copy()
    cap=float(valid["price_usd"].quantile(.995)); model_df=valid[valid["price_usd"].le(cap)].copy()
    descriptive=(valid.groupby(["neighbourhood_group_cleansed","room_type"],dropna=False).agg(listings=("id","count"),median_price_usd=("price_usd","median"),median_minimum_nights=("minimum_nights","median"),median_availability_365=("availability_365","median"),median_reviews_per_month=("reviews_per_month","median")).reset_index())
    descriptive.to_csv(args.output_dir/"market_summary.csv",index=False)

    numeric=["latitude","longitude","accommodates","bedrooms","beds","minimum_nights","number_of_reviews","reviews_per_month","availability_365","host_listings_count"]
    categorical=["room_type","neighbourhood_group_cleansed"]
    X=model_df[numeric+categorical]; y=np.log1p(model_df["price_usd"].to_numpy()); groups=model_df["neighbourhood_cleansed"].astype(str).to_numpy()
    outer=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=args.seed); trainval_idx,test_idx=next(outer.split(X,y,groups))
    inner=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=args.seed+1); tr_rel,val_rel=next(inner.split(X.iloc[trainval_idx],y[trainval_idx],groups[trainval_idx])); train_idx=trainval_idx[tr_rel]; val_idx=trainval_idx[val_rel]
    if set(groups[train_idx]) & set(groups[test_idx]): raise AssertionError("neighbourhood leakage into test")
    if set(groups[train_idx]) & set(groups[val_idx]): raise AssertionError("neighbourhood leakage into validation")
    pre=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median"))]),numeric),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)])
    model=Pipeline([("pre",pre),("model",RandomForestRegressor(n_estimators=250,min_samples_leaf=3,n_jobs=-1,random_state=args.seed))])
    model.fit(X.iloc[train_idx],y[train_idx]); baseline=float(np.median(y[train_idx]))
    def evaluate(indices):
        truth=np.expm1(y[indices]); pred=np.expm1(model.predict(X.iloc[indices])); base=np.full(len(indices),np.expm1(baseline))
        mae=mean_absolute_error(truth,pred); base_mae=mean_absolute_error(truth,base)
        return {"rows":len(indices),"mae_usd":mae,"median_ae_usd":median_absolute_error(truth,pred),"r2":r2_score(truth,pred),"baseline_mae_usd":base_mae,"mae_improvement_vs_median":1-mae/base_mae}
    metrics={"validation":evaluate(val_idx),"untouched_test":evaluate(test_idx)}
    report={"source":{"provider":"Inside Airbnb","snapshot_date":SNAPSHOT_DATE,"url":DATA_URL,"license":"CC BY 4.0","cached_sha256":sha256(cached)},"target":"snapshot listing price in USD; not realised revenue","rows":{"raw":raw_rows,"positive_price":len(valid),"model_after_99_5pct_cap":len(model_df)},"price_cap_usd_99_5pct":cap,"split":{"strategy":"GroupShuffleSplit by neighbourhood_cleansed","train_rows":len(train_idx),"validation_rows":len(val_idx),"test_rows":len(test_idx),"test_neighbourhoods":len(set(groups[test_idx]))},"metrics":metrics,"interpretation_guardrails":["descriptive listing snapshot is not a transaction dataset","availability_365 does not distinguish booked nights from host-blocked nights","listing coordinates are anonymised/offset by Airbnb","predictive test measures price estimation transfer to unseen neighbourhoods, not future price changes or revenue"]}
    (args.output_dir/"evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
