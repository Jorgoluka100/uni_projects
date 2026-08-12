"""Stability-first clustering evaluation for the restored Clustering notebook.

This extension treats clustering as a decision tool rather than a single silhouette
score. It compares candidate k values using silhouette, Davies-Bouldin and
Calinski-Harabasz indices, then measures resampling stability with adjusted Rand
index and exports interpretable cluster profiles.

By default it uses sklearn's Wine dataset as a reproducible methodology demo. Pass
--csv and --features to evaluate a user-supplied numeric dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_wine
    from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    parser=argparse.ArgumentParser(); parser.add_argument("--csv",type=Path); parser.add_argument("--features",nargs="*"); parser.add_argument("--output-dir",type=Path,default=Path("clustering_artifacts")); parser.add_argument("--seed",type=int,default=42); parser.add_argument("--bootstraps",type=int,default=40)
    args=parser.parse_args(); rng=np.random.default_rng(args.seed)
    if args.csv:
        frame=pd.read_csv(args.csv); features=args.features or frame.select_dtypes(include=np.number).columns.tolist()
        if len(features)<2: raise ValueError("at least two numeric features are required")
        source=f"user CSV: {args.csv}"
    else:
        bunch=load_wine(as_frame=True); frame=bunch.frame.drop(columns=["target"]); features=frame.columns.tolist(); source="sklearn Wine dataset (methodology demo)"
    X=frame[features].replace([np.inf,-np.inf],np.nan).dropna()
    if len(X)<50: raise ValueError("need at least 50 complete rows")
    scaled=StandardScaler().fit_transform(X)
    rows=[]; labels_by_k={}
    for k in range(2,min(9,len(X)//10+1)):
        model=KMeans(n_clusters=k,n_init=20,random_state=args.seed).fit(scaled); labels=model.labels_; labels_by_k[k]=labels
        stability=[]
        for b in range(args.bootstraps):
            idx=np.sort(rng.choice(len(X),size=max(30,int(.8*len(X))),replace=False))
            boot=KMeans(n_clusters=k,n_init=10,random_state=args.seed+b+1).fit_predict(scaled[idx])
            stability.append(adjusted_rand_score(labels[idx],boot))
        rows.append({"k":k,"silhouette":silhouette_score(scaled,labels),"davies_bouldin":davies_bouldin_score(scaled,labels),"calinski_harabasz":calinski_harabasz_score(scaled,labels),"stability_ari_mean":float(np.mean(stability)),"stability_ari_p10":float(np.quantile(stability,.10))})
    scores=pd.DataFrame(rows)
    # Transparent composite rank; no metric is allowed to dominate alone.
    scores["rank_sum"]=(scores["silhouette"].rank(ascending=False)+scores["davies_bouldin"].rank()+scores["calinski_harabasz"].rank(ascending=False)+scores["stability_ari_mean"].rank(ascending=False))
    chosen=int(scores.sort_values(["rank_sum","stability_ari_mean"],ascending=[True,False]).iloc[0]["k"])
    prof=X.copy(); prof["cluster"]=labels_by_k[chosen]
    profiles=prof.groupby("cluster")[features].agg(["mean","median"])
    sizes=prof["cluster"].value_counts().sort_index().rename("rows")
    args.output_dir.mkdir(parents=True,exist_ok=True); scores.to_csv(args.output_dir/"candidate_scores.csv",index=False); profiles.to_csv(args.output_dir/"cluster_profiles.csv"); sizes.to_csv(args.output_dir/"cluster_sizes.csv")
    summary={"source":source,"rows":len(X),"features":features,"chosen_k":chosen,"selection_rule":"lowest aggregate rank across silhouette, Davies-Bouldin, Calinski-Harabasz and resampling ARI","limitations":["clusters are descriptive, not ground-truth classes","stability does not prove business usefulness","profiles require domain review before operational use"]}
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
