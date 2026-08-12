# CareerLens AI — Job Fit & Skill-Gap Retrieval

**Decision:** given a candidate profile and a set of job descriptions, which roles deserve attention first and which explicit skills are missing?

CareerLens is an information-retrieval project rather than an ATS predictor. It combines word + character TF-IDF similarity with transparent skill-overlap scoring, then evaluates ranking quality using **MRR, Recall@5 and NDCG@5**.

```bash
pip install numpy pandas scikit-learn
python projects/careerlens_ai/run.py --self-test
python projects/careerlens_ai/run.py --output-dir careerlens_artifacts
```

Supply a real job corpus with `--jobs-csv jobs.csv`; required columns are `job_id,title,description`.

Outputs: `ranking.csv` and `verification.json`.

**Evidence boundary:** the built-in job corpus is deterministic and synthetic, so benchmark metrics demonstrate the retrieval/evaluation mechanics, not real hiring-market accuracy. Missing skills are descriptive and never a guarantee of employability or selection.
