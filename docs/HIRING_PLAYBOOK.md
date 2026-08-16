# Project Notes for Applications and Interviews

I keep this page as a reminder of which projects are most useful for different roles and, more importantly, what I would actually say about them in an interview.

## Which projects I would lead with

| Role | Projects I would start with |
|---|---|
| Graduate / Junior Data Scientist | AeroFlow, ExperimentLab, UK House Prices |
| Graduate / Junior ML Engineer | VisionForge, GroundedRAG, ModelWatch |
| AI / Applied AI Internship | GroundedRAG, VisionForge, CareerLens AI |
| Graduate / Junior Data Analyst | SQL Sales, ExperimentLab, NYC Airbnb, Bootcamp projects |
| Data / AI Graduate Scheme | AeroFlow, SQL Sales, GroundedRAG, VisionForge |

## Five stories I can explain without a script

### GroundedRAG: I wanted to test more than whether a prompt looked good

I built a small local RAG system that retrieves policy and incident documents, returns sources and can refuse a question when the retrieved evidence is weak. I then added a read-only ticket analytics tool and test cases for prompt injection.

The frozen fixture passes all of its retrieval, routing and safety checks. The important point is the scope: it is a small deterministic test set, so I use those scores to check the software contract, not to claim production-level AI performance.

### Recommender: I found leakage and reran it

When I reviewed the movie recommender, I found that later user interactions were appearing in histories used during evaluation. I changed the split so each user's history stops before the held-out item and removed 1,756 future interactions before rerunning the ranking metrics.

That is a better interview story than simply quoting the final Recall@10 because it shows why the validation setup matters.

### AeroFlow: the first version failed its baseline

I first treated flight delay as a regression problem. The model passed the code checks but did not beat a simple baseline on MAE. Rather than keep polishing a weak formulation, I changed the question to: which flights are most likely to arrive at least 15 minutes late?

On untouched May 2026 data the classifier reached PR-AUC 0.291 against a 0.215 delay rate, and the highest-risk 10% of flights had 1.58x the normal delay rate.

### ExperimentLab: not every data problem is prediction

I built a simulated 20,000-row randomized experiment so I could practise treatment-effect estimation, CUPED, confidence intervals, guardrails and power calculations. CUPED reduced outcome variance by 50.7% in the simulation.

The useful part is being able to explain how I would decide whether an experiment result is strong enough to act on, rather than just train another model.

### ModelWatch: what happens after a model is trained?

ModelWatch compares incoming batches with a reference dataset and checks drift, ROC/PR performance and calibration. I deliberately created stable, feature-shift and concept-shift batches to see whether the alert rules behaved sensibly. The stable batch stays green and the larger shifts trigger investigation.

## Numbers I can use when relevant

- GroundedRAG: all retrieval, routing and injection tests pass on the small frozen synthetic fixture; I always state the fixture scope.
- VisionForge: 85.9% test accuracy, 85.8% macro-F1; 90.4% accuracy on accepted predictions at 89.1% coverage.
- AeroFlow: PR-AUC 0.291 vs 0.215 delay rate; 1.58x lift in the top-risk decile on untouched May 2026 data.
- UK House Prices: MAE £81,805 and R² 0.604 on 216,564 untouched 2026 sales.
- NYC Airbnb 2026: MAE $68.97 vs $121.90 median baseline on unseen neighbourhoods.
- Customer Churn: PR-AUC 0.955 and ROC-AUC 0.990.
- Recommender: Recall@10 0.501 vs 0.463 popularity baseline after the leakage fix.
- TensorFlow Energy: MAE 43.51 vs 53.18 seasonal baseline.
- PySpark Clickstream: PR-AUC 0.351 vs 0.155 positive rate.
- ExperimentLab: 50.7% CUPED variance reduction on simulated data.

## Results I would not oversell

Some projects use synthetic or historical data because they were built to practise a method rather than make a real-world claim. GroundedRAG, CareerLens, ExperimentLab, ModelWatch, ConsultAI, the fraud/AML demo and the telecom demo fall into that category. The KDD work uses an old cybersecurity dataset. The Parkinson's result is deliberately kept even though the grouped model lost to its baseline. The medical notebooks are learning projects, not clinical systems.

The rule I use is simple: if I cannot explain where a number came from, how the split worked and what the limitation is, it should not go on an application.
