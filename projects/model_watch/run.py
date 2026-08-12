from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,brier_score_loss,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES=["age","income","tenure_months","usage"]


def generate(n:int,seed:int,shift:float=0.0,concept:float=0.0)->pd.DataFrame:
    rng=np.random.default_rng(seed); segment=rng.choice(["A","B"],n,p=[0.7-max(0,shift)*0.08,0.3+max(0,shift)*0.08]); age=rng.normal(39+3*shift,11,n).clip(18,80); income=rng.lognormal(np.log(42000*(1+0.10*shift)),0.45,n); tenure=rng.gamma(2.4+0.15*shift,12,n).clip(1,120); usage=rng.normal(52-5*shift,14,n).clip(0,100)
    logit=-2.2+0.018*(age-40)+0.000012*(income-42000)-0.018*(tenure-24)+0.025*(usage-50)+0.55*(segment=="B")+concept*(0.04*(age-40)-0.03*(usage-50)); p=1/(1+np.exp(-logit)); y=(rng.random(n)<p).astype(int)
    return pd.DataFrame({"age":age,"income":income,"tenure_months":tenure,"usage":usage,"segment":segment,"target":y})


def psi(reference:np.ndarray,current:np.ndarray,bins:int=10)->float:
    edges=np.unique(np.quantile(reference,np.linspace(0,1,bins+1)))
    if len(edges)<3:return 0.0
    edges[0],edges[-1]=-np.inf,np.inf; r=np.histogram(reference,bins=edges)[0].astype(float); c=np.histogram(current,bins=edges)[0].astype(float); r=np.clip(r/r.sum(),1e-6,None); c=np.clip(c/c.sum(),1e-6,None); return float(np.sum((c-r)*np.log(c/r)))


def ece(y:np.ndarray,p:np.ndarray,bins:int=10)->float:
    edges=np.linspace(0,1,bins+1); score=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(p>=lo)&(p<(hi if hi<1 else hi+1e-12))
        if mask.any():score+=mask.mean()*abs(float(y[mask].mean()-p[mask].mean()))
    return float(score)


def subgroup_metrics(df:pd.DataFrame,p:np.ndarray)->dict:
    out={}
    for seg in sorted(df.segment.unique()):
        m=df.segment.to_numpy()==seg;y=df.target.to_numpy()[m];ps=p[m];out[seg]={"rows":int(m.sum()),"prevalence":float(y.mean()),"brier":float(brier_score_loss(y,ps)),"mean_score":float(ps.mean())}
    return out


def evaluate_batch(model,ref:pd.DataFrame,batch:pd.DataFrame,ref_auc:float,name:str)->dict:
    p=model.predict_proba(batch[FEATURES])[:,1];y=batch.target.to_numpy();drift={}
    for col in FEATURES:
        stat,pv=ks_2samp(ref[col],batch[col]);drift[col]={"psi":psi(ref[col].to_numpy(),batch[col].to_numpy()),"ks_stat":float(stat),"ks_pvalue":float(pv)}
    auc=float(roc_auc_score(y,p));pr=float(average_precision_score(y,p));brier=float(brier_score_loss(y,p));calib=ece(y,p);max_psi=max(v["psi"] for v in drift.values());auc_drop=ref_auc-auc;level="green";reasons=[]
    if max_psi>=0.25 or auc_drop>=0.08 or calib>=0.08:level="red"
    elif max_psi>=0.10 or auc_drop>=0.04 or calib>=0.05:level="amber"
    if max_psi>=0.10:reasons.append(f"feature drift PSI={max_psi:.3f}")
    if auc_drop>=0.04:reasons.append(f"ROC-AUC drop={auc_drop:.3f}")
    if calib>=0.05:reasons.append(f"ECE={calib:.3f}")
    return {"batch":name,"rows":len(batch),"drift":drift,"max_psi":max_psi,"performance":{"roc_auc":auc,"pr_auc":pr,"brier":brier,"ece":calib,"auc_drop_vs_reference":auc_drop},"subgroups":subgroup_metrics(batch,p),"alert":level,"reasons":reasons}


def run(output_dir:Path,seed:int=42)->dict:
    output_dir.mkdir(parents=True,exist_ok=True);train=generate(18000,seed);reference=generate(7000,seed+1);model=Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=1000,random_state=seed))]);model.fit(train[FEATURES],train.target);ref_p=model.predict_proba(reference[FEATURES])[:,1];ref_auc=float(roc_auc_score(reference.target,ref_p))
    batches=[("stable",generate(7000,seed+2,0.0,0.0)),("mild_shift",generate(7000,seed+3,0.7,0.0)),("feature_shift",generate(7000,seed+4,1.5,0.2)),("concept_shift",generate(7000,seed+5,2.0,0.8))];reports=[evaluate_batch(model,reference,b,ref_auc,n) for n,b in batches]
    model_path=output_dir/"model.joblib";joblib.dump(model,model_path);reloaded=joblib.load(model_path);parity=bool(np.allclose(model.predict_proba(reference[FEATURES].head(100)),reloaded.predict_proba(reference[FEATURES].head(100))))
    payload={"project":"ModelWatch","verification_pass":bool(parity and reports[0]["alert"]=="green" and reports[-1]["alert"]=="red"),"scope":"deterministic production-monitoring simulation; not a live production system","reference":{"rows":len(reference),"roc_auc":ref_auc,"pr_auc":float(average_precision_score(reference.target,ref_p)),"brier":float(brier_score_loss(reference.target,ref_p)),"ece":ece(reference.target.to_numpy(),ref_p)},"batches":reports,"model_reload_parity":parity,"policy":{"amber":"PSI>=0.10 or AUC drop>=0.04 or ECE>=0.05","red":"PSI>=0.25 or AUC drop>=0.08 or ECE>=0.08","action":"red recommends investigation/retraining; this demo never auto-deploys a replacement model"}}
    (output_dir/"verification.json").write_text(json.dumps(payload,indent=2),encoding="utf-8");pd.DataFrame([{**{"batch":r["batch"],"alert":r["alert"],"max_psi":r["max_psi"]},**r["performance"]} for r in reports]).to_csv(output_dir/"batch_summary.csv",index=False);print(json.dumps(payload,indent=2));return payload


def self_test()->None:
    out=run(Path("/tmp/modelwatch_selftest"),123);assert out["verification_pass"];assert out["batches"][0]["alert"]=="green";assert out["batches"][-1]["alert"]=="red";print("ModelWatch self-test passed.")


def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=Path("modelwatch_artifacts"));p.add_argument("--seed",type=int,default=42);p.add_argument("--self-test",action="store_true");a=p.parse_args()
    if a.self_test:self_test();return 0
    r=run(a.output_dir,a.seed);return 0 if r["verification_pass"] else 1

if __name__=="__main__":raise SystemExit(main())
