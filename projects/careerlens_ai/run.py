from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

SKILLS = {
    "python": ["python"], "sql": ["sql", "postgresql", "duckdb"], "pandas": ["pandas"],
    "numpy": ["numpy"], "scikit-learn": ["scikit-learn", "sklearn"], "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "keras"], "pyspark": ["pyspark", "spark"],
    "mlops": ["mlops", "model monitoring", "drift", "ci/cd"],
    "nlp": ["nlp", "natural language processing", "text classification", "retrieval"],
    "computer vision": ["computer vision", "cnn", "image classification"],
    "experimentation": ["a/b testing", "experimentation", "causal inference"],
    "statistics": ["statistics", "statistical"], "fastapi": ["fastapi", "api"],
    "docker": ["docker", "containers"], "cloud": ["aws", "gcp", "azure", "cloud"],
}

DEMO_JOBS = [
    {"job_id":"DS1","title":"Graduate Data Scientist","description":"Python SQL pandas scikit-learn statistics experimentation. Build predictive models, validate on held-out data and explain results to product teams."},
    {"job_id":"DS2","title":"Junior Decision Scientist","description":"Python SQL causal inference A/B testing experimentation statistics product analytics and stakeholder communication."},
    {"job_id":"ML1","title":"Junior Machine Learning Engineer","description":"Python PyTorch scikit-learn FastAPI Docker MLOps model monitoring CI/CD and production inference services."},
    {"job_id":"ML2","title":"Computer Vision Engineer","description":"Python PyTorch computer vision CNN image classification model export inference optimisation and monitoring."},
    {"job_id":"AI1","title":"Applied AI Engineer","description":"Python NLP retrieval embeddings FastAPI evaluation MLOps and product-focused AI systems."},
    {"job_id":"AI2","title":"NLP Engineer","description":"Python natural language processing retrieval text classification transformers evaluation and APIs."},
    {"job_id":"DA1","title":"Data Analyst","description":"SQL Python pandas dashboards statistics commercial analysis and clear stakeholder reporting."},
    {"job_id":"DE1","title":"Junior Data Engineer","description":"SQL Python PySpark Spark data pipelines cloud orchestration and data quality."},
    {"job_id":"TF1","title":"Deep Learning Engineer","description":"Python TensorFlow Keras neural networks model serving experiment tracking and monitoring."},
    {"job_id":"BA1","title":"Business Intelligence Analyst","description":"SQL reporting dashboards KPI design stakeholder management and commercial insight."},
]

BENCHMARK = [
    {"name":"data_scientist","profile":"Python SQL pandas scikit-learn statistics machine learning experimentation", "relevant":{"DS1","DS2","DA1"}},
    {"name":"ml_engineer","profile":"Python PyTorch FastAPI Docker MLOps model monitoring computer vision", "relevant":{"ML1","ML2","AI1"}},
    {"name":"ai_nlp","profile":"Python NLP retrieval embeddings text classification FastAPI evaluation", "relevant":{"AI1","AI2","ML1"}},
    {"name":"data_platform","profile":"Python SQL PySpark Spark data pipelines data quality cloud", "relevant":{"DE1","DA1","DS1"}},
]


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def extract_skills(text: str) -> set[str]:
    corpus = normalise(text); found = set()
    for canonical, aliases in SKILLS.items():
        if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", corpus) for alias in aliases): found.add(canonical)
    return found


@dataclass
class RankedJob:
    job_id: str; title: str; semantic_score: float; skill_score: float; final_score: float; matched_skills: list[str]; missing_skills: list[str]


class CareerLens:
    def __init__(self, jobs: pd.DataFrame):
        required = {"job_id", "title", "description"}
        if not required.issubset(jobs.columns): raise ValueError(f"jobs must contain {sorted(required)}")
        self.jobs = jobs.reset_index(drop=True).copy()
        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)),
        ])
        self.job_matrix = self.vectorizer.fit_transform(self.jobs["description"].map(normalise))
        self.job_skills = [extract_skills(x) for x in self.jobs["description"]]

    def rank(self, profile: str, top_k: int | None = None) -> list[RankedJob]:
        semantic = cosine_similarity(self.vectorizer.transform([normalise(profile)]), self.job_matrix)[0]
        profile_skills = extract_skills(profile); ranked = []
        for idx, row in self.jobs.iterrows():
            js = self.job_skills[idx]; overlap = len(profile_skills & js); union = len(profile_skills | js)
            skill_score = overlap / union if union else 0.0; final = 0.72 * float(semantic[idx]) + 0.28 * skill_score
            ranked.append(RankedJob(str(row.job_id), str(row.title), float(semantic[idx]), float(skill_score), float(final), sorted(profile_skills & js), sorted(js - profile_skills)))
        ranked.sort(key=lambda x: (-x.final_score, x.job_id)); return ranked[:top_k] if top_k else ranked


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    for i, job_id in enumerate(ranked_ids, 1):
        if job_id in relevant: return 1.0 / i
    return 0.0


def evaluate(engine: CareerLens, k: int = 5) -> dict[str, float]:
    mrr, recall, ndcg = [], [], []
    for case in BENCHMARK:
        ids = [x.job_id for x in engine.rank(case["profile"])]; relevant = case["relevant"]; top = ids[:k]
        mrr.append(reciprocal_rank(ids, relevant)); recall.append(len(set(top) & relevant) / len(relevant))
        y_true = np.array([[1.0 if x in relevant else 0.0 for x in ids]]); y_score = np.array([[len(ids)-i for i in range(len(ids))]], dtype=float)
        ndcg.append(float(ndcg_score(y_true, y_score, k=k)))
    return {"mrr":float(np.mean(mrr)), f"recall@{k}":float(np.mean(recall)), f"ndcg@{k}":float(np.mean(ndcg)), "queries":len(BENCHMARK)}


def run(profile: str, jobs_path: Path | None, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True); jobs = pd.read_csv(jobs_path) if jobs_path else pd.DataFrame(DEMO_JOBS); engine = CareerLens(jobs)
    metrics = evaluate(engine); ranking = engine.rank(profile, top_k=10); pd.DataFrame([asdict(x) for x in ranking]).to_csv(output_dir/"ranking.csv", index=False)
    payload = {"project":"CareerLens AI","verification_pass":bool(metrics["mrr"] >= 0.75 and metrics["ndcg@5"] >= 0.70),"scope":"deterministic information-retrieval methodology demo; demo jobs are synthetic unless --jobs-csv is supplied","benchmark":metrics,"top_recommendation":asdict(ranking[0]),"design":["hybrid word/character TF-IDF similarity plus explicit skill overlap","transparent inspectable skill aliases","ranking evaluated with MRR, Recall@5 and NDCG@5","missing-skill output is descriptive; it is not a hiring guarantee"]}
    (output_dir/"verification.json").write_text(json.dumps(payload, indent=2), encoding="utf-8"); print(json.dumps(payload, indent=2)); return payload


def self_test() -> None:
    engine = CareerLens(pd.DataFrame(DEMO_JOBS)); assert engine.rank("Python PyTorch FastAPI Docker MLOps model monitoring", top_k=1)[0].job_id == "ML1"
    metrics = evaluate(engine); assert all(math.isfinite(v) for k,v in metrics.items() if k != "queries"); assert metrics["mrr"] >= 0.75 and metrics["ndcg@5"] >= 0.70
    print("CareerLens self-test passed.")


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--profile",default="MSc AI and Data Science. Python SQL PyTorch TensorFlow scikit-learn PySpark pandas MLOps NLP computer vision FastAPI"); p.add_argument("--jobs-csv",type=Path); p.add_argument("--output-dir",type=Path,default=Path("careerlens_artifacts")); p.add_argument("--self-test",action="store_true"); a=p.parse_args()
    if a.self_test: self_test(); return 0
    r=run(a.profile,a.jobs_csv,a.output_dir); return 0 if r["verification_pass"] else 1

if __name__ == "__main__": raise SystemExit(main())
