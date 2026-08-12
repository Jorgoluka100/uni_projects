from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def simulate(n: int, seed: int, effect: float) -> pd.DataFrame:
    rng=np.random.default_rng(seed); pre=rng.normal(100,20,n); segment=rng.choice(["new","returning"],n,p=[0.35,0.65]); treatment=rng.integers(0,2,n); noise=rng.normal(0,12,n)
    outcome=25+0.62*pre+4.0*(segment=="returning")+effect*treatment+noise; guardrail=rng.normal(5.0,1.1,n)+0.02*treatment
    return pd.DataFrame({"pre_metric":pre,"segment":segment,"treatment":treatment,"outcome":outcome,"guardrail":guardrail})


def mean_effect(y: np.ndarray,t: np.ndarray)->tuple[float,float,float]:
    yt,yc=y[t==1],y[t==0]; effect=float(yt.mean()-yc.mean()); se=math.sqrt(float(yt.var(ddof=1)/len(yt)+yc.var(ddof=1)/len(yc))); return effect,effect-1.96*se,effect+1.96*se


def cuped_adjust(y:np.ndarray,x:np.ndarray)->tuple[np.ndarray,float]:
    theta=float(np.cov(y,x,ddof=1)[0,1]/np.var(x,ddof=1)); return y-theta*(x-x.mean()),theta


def stratified_bootstrap(y:np.ndarray,t:np.ndarray,rounds:int,seed:int)->tuple[float,float]:
    rng=np.random.default_rng(seed); ti,ci=np.where(t==1)[0],np.where(t==0)[0]; values=[]
    for _ in range(rounds):
        ts=rng.choice(ti,len(ti),replace=True); cs=rng.choice(ci,len(ci),replace=True); values.append(float(y[ts].mean()-y[cs].mean()))
    return tuple(float(x) for x in np.quantile(values,[0.025,0.975]))


def power_and_mde(sd:float,n:int,alpha:float=0.05,power:float=0.80)->dict[str,float]:
    per_arm=n/2; return {"alpha":alpha,"target_power":power,"mde_outcome_units":float((norm.ppf(1-alpha/2)+norm.ppf(power))*sd*math.sqrt(2/per_arm))}


def run(n:int,seed:int,effect:float,output_dir:Path)->dict:
    output_dir.mkdir(parents=True,exist_ok=True); df=simulate(n,seed,effect); y=df.outcome.to_numpy(); t=df.treatment.to_numpy(); pre=df.pre_metric.to_numpy()
    raw=mean_effect(y,t); adjusted,theta=cuped_adjust(y,pre); cuped=mean_effect(adjusted,t); boot=stratified_bootstrap(adjusted,t,1500,seed+99)
    variance_reduction=1-float(np.var(adjusted,ddof=1))/float(np.var(y,ddof=1)); guard=mean_effect(df.guardrail.to_numpy(),t); decision="ship" if cuped[1]>0 and guard[1]>-0.20 else "hold"
    payload={"project":"ExperimentLab","verification_pass":bool(abs(cuped[0]-effect)<1.0 and variance_reduction>0.30),"scope":"deterministic synthetic randomized-experiment methodology demo","rows":n,"known_simulated_effect":effect,"raw_effect":{"estimate":raw[0],"ci95":[raw[1],raw[2]]},"cuped_effect":{"estimate":cuped[0],"ci95":[cuped[1],cuped[2]],"bootstrap_ci95":list(boot),"theta":theta},"variance_reduction":variance_reduction,"guardrail_effect":{"estimate":guard[0],"ci95":[guard[1],guard[2]],"non_inferiority_margin":-0.20},"power":power_and_mde(float(np.std(adjusted,ddof=1)),n),"decision":decision,"rules":["random assignment","fixed-horizon inference","CUPED uses a pre-treatment covariate","bootstrap resamples within treatment arms","guardrail must not cross the pre-declared harm margin"]}
    df.to_csv(output_dir/"experiment_data.csv",index=False); (output_dir/"verification.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2)); return payload


def self_test()->None:
    out=run(12000,42,2.5,Path("/tmp/experimentlab_selftest")); assert out["verification_pass"]; assert out["variance_reduction"]>0.30; assert out["cuped_effect"]["ci95"][0]<2.5<out["cuped_effect"]["ci95"][1]; print("ExperimentLab self-test passed.")


def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--rows",type=int,default=20000); p.add_argument("--seed",type=int,default=42); p.add_argument("--effect",type=float,default=2.5); p.add_argument("--output-dir",type=Path,default=Path("experimentlab_artifacts")); p.add_argument("--self-test",action="store_true"); a=p.parse_args()
    if a.self_test: self_test(); return 0
    r=run(a.rows,a.seed,a.effect,a.output_dir); return 0 if r["verification_pass"] else 1

if __name__=="__main__": raise SystemExit(main())
