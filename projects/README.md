# Python Projects

These are the projects in this repository that I built as normal Python applications rather than as large notebooks. I use them to practise the parts of data and AI work that sit around the model itself: command-line runs, tests, saved outputs, APIs and basic packaging.

## Retrieval-augmented support assistant

A local retrieval application over a small frozen set of policy and incident documents. It combines keyword and latent-semantic retrieval, returns the documents used for an answer, abstains on weak matches and exposes a read-only ticket analytics route through FastAPI. I also included prompt-injection cases to check that the tool is not executed on the attack examples.

The perfect scores in the frozen fixture are there to verify the implementation. They are not presented as real-world LLM performance.

[Open project](grounded_rag/)

## Experiment analysis

A small A/B-testing project covering a treatment-effect estimate, CUPED, bootstrap confidence intervals, a guardrail check and a power/MDE calculation. The data is simulated so I can compare the analysis against a known effect.

[Open project](experiment_lab/)

## Model monitoring

A monitoring demo that compares new batches with a reference dataset and checks drift, discrimination and calibration. The shifts are deliberately introduced so I can test whether the monitoring rules react in the expected direction.

[Open project](model_watch/)

## Job matching / retrieval experiment

A small information-retrieval project that ranks jobs against a candidate profile and reports the main skill matches and gaps. The fixture is synthetic and the project is mainly an exercise in ranking, evaluation and explaining why an item was retrieved.

[Open project](careerlens_ai/)

## Running the projects

Each folder has its own README with the commands needed to run it. The workflow in `.github/workflows/new-projects-ci.yml` reruns the project self-tests and validates the retained result files.
