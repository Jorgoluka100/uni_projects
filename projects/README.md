# Smaller Python Projects

These are the projects I built as normal Python applications rather than large notebooks. I use them to practise the engineering side of data and AI work: running from the command line, saving results, writing tests and, where it makes sense, exposing an API.

## GroundedRAG v2

A local RAG demo for a small set of policy and incident documents. It combines BM25/TF-IDF with an LSA-based dense representation, returns the source documents used for an answer, refuses low-confidence questions and can route a read-only ticket analytics tool. I also added prompt-injection test cases so the tool is not called on the attack examples.

The perfect scores in its frozen test fixture are there to check that the code behaves as expected. They are not a claim that the system would score 100% on real company data or with a production LLM.

## CareerLens AI

A small information-retrieval project that ranks jobs against a candidate profile and shows the main skill matches and gaps. The fixture is synthetic and is mainly used to demonstrate ranking and evaluation code.

## ExperimentLab

An A/B-testing project covering a treatment-effect estimate, CUPED, bootstrap confidence intervals, a guardrail check and a power/MDE calculation. The dataset is simulated so I can test the analysis against a known effect.

## ModelWatch

A model-monitoring demo that compares new batches with a reference dataset and checks drift, discrimination and calibration. The data shifts are deliberately created so I can test whether the monitoring rules respond in the expected direction.

## Running the projects

Each folder has its own README with the command needed to run it. The GitHub Actions workflow in `.github/workflows/new-projects-ci.yml` reruns the project self-tests and result checks.
