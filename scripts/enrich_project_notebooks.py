"""Build the final recruiter-facing notebook layer without deleting original work.

Portfolio rule:

    one project -> one notebook -> one complete application story

The notebook must lead with the actual analysis, not with a wall of helper functions.
Original cells are preserved, then canonical Python is retained as engineering evidence.
The analysis-first layer is deliberately written as direct notebook code: load/inspect,
EDA, visualisation, evidence review and decision framing are visible immediately.

A major portfolio notebook should aim for roughly 1,000 meaningful code lines when
that depth is genuinely useful. The target is a guide, not permission to pad code.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = ROOT / "projects"
TARGET_LOW = 600
TARGET_IDEAL = 1000
TARGET_HIGH = 1400
ENRICHMENT_TAG = "portfolio-enrichment"

EXCLUDED_PARTS = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
}

PROJECT_CONTEXT = {
    "flight_delay_risk": {
        "objective": "Predict departure-delay risk early enough for operations teams to prioritise buffers, passenger communication and recovery actions.",
        "decision": "Convert calibrated delay risk into an operational watchlist, then inspect which routes, carriers, airports and time windows create avoidable disruption.",
    },
    "customer_churn_prediction": {
        "objective": "Identify customers at meaningful churn risk while avoiding a retention campaign that spends money on everyone.",
        "decision": "Use calibrated probabilities and retention economics to decide who should receive an intervention and where manual review is justified.",
    },
    "uk_house_price_prediction": {
        "objective": "Estimate residential transaction values with time-aware validation rather than a random split that leaks future market conditions.",
        "decision": "Use the model as a valuation-screening tool, expose uncertainty and error slices, and avoid pretending the estimate is a formal survey or valuation.",
    },
    "retail_customer_segmentation": {
        "objective": "Turn raw transaction history into defensible customer segments that marketing teams can actually act on.",
        "decision": "Prioritise segments using recency, frequency, monetary value, stability and commercial opportunity instead of relying on cluster labels alone.",
    },
    "knn_product_quality": {
        "objective": "Classify product/wine quality groups with an interpretable distance-based model and show why feature scaling matters for KNN.",
        "decision": "Route confident cases automatically and send ambiguous nearest-neighbour cases to review with the neighbour evidence visible.",
    },
    "xgboost_bike_demand": {
        "objective": "Forecast demand with chronological validation so operations can prepare capacity without leaking future information.",
        "decision": "Translate forecast errors into staffing/rebalancing risk and identify the hours or weather conditions where extra capacity is most valuable.",
    },
    "statistical_marketing_mix": {
        "objective": "Estimate channel relationships and uncertainty rather than reporting a single opaque prediction score.",
        "decision": "Use robust inference, diagnostics and scenarios to support budget discussion while clearly separating association from causal claims.",
    },
    "experiment_lab": {
        "objective": "Evaluate product experiments with power, variance reduction, guardrails and uncertainty rather than declaring winners from a p-value alone.",
        "decision": "Ship, iterate or stop based on effect size, confidence interval, guardrails and practical business value.",
    },
    "parkinsons_progression": {
        "objective": "Model Parkinson's telemonitoring outcomes while preventing subject leakage between train and validation groups.",
        "decision": "Treat results as educational modelling evidence only; use grouped error analysis and uncertainty rather than any clinical recommendation.",
    },
    "grounded_rag": {
        "objective": "Build a retrieval application that can cite evidence, abstain when support is weak and resist simple prompt-injection attempts.",
        "decision": "Answer only when retrieval and grounding checks pass; otherwise abstain or route to a safe tool/human path.",
    },
    "deep_learning_marketing_response": {
        "objective": "Train a real neural network on bank-marketing data and compare it against a simpler baseline instead of assuming deep learning wins.",
        "decision": "Choose the model and threshold from measured validation performance, calibration and campaign economics, not architecture complexity.",
    },
    "nlp_document_intelligence": {
        "objective": "Classify documents while exposing confidence, errors and the text features that drive the decision.",
        "decision": "Auto-route high-confidence documents and send uncertain or unusual text to human review.",
    },
    "image_classification_confidence": {
        "objective": "Classify bean-leaf disease images while measuring confidence, calibration and failure modes rather than only top-line accuracy.",
        "decision": "Use selective prediction so uncertain images are escalated instead of forcing an unsafe confident label.",
    },
    "energy_demand_forecasting": {
        "objective": "Forecast energy demand against strong seasonal baselines using time-ordered validation.",
        "decision": "Use forecast intervals and seasonal error slices to support capacity planning while identifying periods requiring extra reserve.",
    },
    "model_watch": {
        "objective": "Detect data quality, distribution, discrimination and calibration changes after a model is deployed.",
        "decision": "Escalate, investigate or retrain only when monitored signals cross documented policy thresholds.",
    },
    "reliable_event_pipeline": {
        "objective": "Ingest events reliably despite duplicates, invalid records, late arrivals and replayed batches.",
        "decision": "Accept valid events once, quarantine bad records, reconcile counts and make replay behaviour observable.",
    },
    "apache_spark_retail_intelligence": {
        "objective": "Demonstrate production-style Spark transformations at scale using explicit schemas, windows, customer features and partition-aware output.",
        "decision": "Build a Customer 360 table that can support segmentation and downstream analytics without hiding data-quality failures.",
    },
    "pyspark_clickstream_analytics": {
        "objective": "Turn high-volume clickstream events into behavioural, funnel and conversion features using distributed Spark operations.",
        "decision": "Identify where users drop out of the journey and which behavioural signals deserve product or marketing attention.",
    },
    "ecommerce_sql_analytics": {
        "objective": "Build a trustworthy analytical model from raw e-commerce tables with explicit grain, tests and reproducible SQL/dbt transformations.",
        "decision": "Use governed revenue, order, customer and cohort metrics so business decisions are made from consistent definitions.",
    },
    "executive_commerce_bi": {
        "objective": "Convert governed commerce data into an executive BI layer with auditable KPI definitions and reusable semantic-model assets.",
        "decision": "Surface the few KPIs and segments that require executive action, then retain drill-down evidence for investigation.",
    },
    "linear_regression_energy_efficiency": {
        "objective": "Use ordinary linear regression as a transparent baseline for estimating building heating load from physical design variables.",
        "decision": "Screen candidate building designs, quantify residual risk, compare regularised/non-linear alternatives and flag high-load configurations for redesign.",
    },
}

PROJECT_LIMITATION_NOTES = {
    "parkinsons_progression": (
        "This project is educational and non-clinical. The UCI telemonitoring data is historical, "
        "the cohort is limited, and model performance does not establish safety or clinical utility. "
        "A real clinical workflow would require external validation across sites and populations, "
        "prospective evaluation, governance, clinician oversight and regulatory review before any use in care."
    ),
}


def _tags(cell: dict) -> set[str]:
    return set(cell.get("metadata", {}).get("tags", []))


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def _markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"tags": [ENRICHMENT_TAG]},
        "source": text.splitlines(keepends=True),
    }


def _direct_code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [ENRICHMENT_TAG, "analysis-first"]},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def _source_code(text: str, rel_path: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "tags": [ENRICHMENT_TAG, "source-mirror", "skip-execution"],
            "source_file": rel_path,
        },
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def _meaningful_code_lines(cells: list[dict]) -> int:
    total = 0
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        for line in _source(cell).splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                total += 1
    return total


def _discover_python(project_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in project_dir.rglob("*.py"):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        files.append(path)
    return sorted(files, key=lambda p: (len(p.relative_to(project_dir).parts), str(p)))


def _already_mirrored(existing_code: str, source: str) -> bool:
    normalized = source.strip()
    if not normalized:
        return True
    return normalized in existing_code


def _has_limitations_story(cells: list[dict]) -> bool:
    text = "\n".join(_source(cell).lower() for cell in cells)
    terms = ("limitation", "caveat", "risk", "future work", "next step")
    return any(term in text for term in terms)


def _limitations_cell(project_dir: Path) -> dict:
    note = PROJECT_LIMITATION_NOTES.get(
        project_dir.name,
        (
            "This portfolio project is bounded by the documented dataset, validation design and retained "
            "evidence. Important limitations include dataset representativeness, possible distribution shift, "
            "measurement quality and the gap between offline evaluation and production use. Next steps should "
            "include stronger external or time-separated validation, monitoring, operational cost calibration "
            "and human review where the application can affect consequential decisions."
        ),
    )
    return _markdown(f"## Limitations and next steps\n\n{note}\n")


def _analysis_first_cells(project_dir: Path) -> list[dict]:
    context = PROJECT_CONTEXT.get(
        project_dir.name,
        {
            "objective": "Turn the documented dataset into a reproducible analysis or application that answers a real decision question.",
            "decision": "Use measured evidence, error analysis and documented limitations to support the final decision rather than reporting a metric in isolation.",
        },
    )

    cells: list[dict] = []
    cells.append(_markdown(
        "# Notebook-first application walkthrough\n\n"
        f"**Problem / objective:** {context['objective']}\n\n"
        f"**Decision / solution:** {context['decision']}\n\n"
        "This front section is intentionally analysis-first. It uses direct notebook code for inspection, "
        "EDA, visualisation and evidence review. The original notebook work is preserved below, followed by "
        "modular production code where that adds engineering evidence.\n"
    ))

    cells.append(_direct_code(
        "from pathlib import Path\n"
        "import json\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "import warnings\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "warnings.filterwarnings('ignore')\n"
        f"PROJECT_SLUG = {project_dir.name!r}\n"
        "ROOT = Path.cwd()\n"
        "if not (ROOT / 'projects').exists():\n"
        "    candidate = ROOT.parent.parent if ROOT.name == PROJECT_SLUG else ROOT\n"
        "    if (candidate / 'projects').exists():\n"
        "        ROOT = candidate\n"
        "PROJECT = ROOT / 'projects' / PROJECT_SLUG\n"
        "if not PROJECT.exists() and Path.cwd().name == PROJECT_SLUG:\n"
        "    PROJECT = Path.cwd()\n"
        "    ROOT = PROJECT.parent.parent\n"
        "assert PROJECT.exists(), f'Project directory not found: {PROJECT}'\n"
        "print('Repository root:', ROOT.resolve())\n"
        "print('Project:', PROJECT.resolve())\n"
    ))

    cells.append(_markdown(
        "## 1. Find the real data and retained evidence\n\n"
        "Instead of hiding the dataset behind a helper function, start by seeing what the project actually ships: "
        "raw/small data, fixtures, outputs, results and verified evidence. External large datasets remain reproducibly downloadable from the documented source.\n"
    ))
    cells.append(_direct_code(
        "candidate_files = []\n"
        "for pattern in ('*.csv', '*.parquet', '*.json', '*.tsv', '*.txt'):\n"
        "    candidate_files.extend(PROJECT.rglob(pattern))\n"
        "verified_dir = ROOT / 'verified' / PROJECT_SLUG\n"
        "if verified_dir.exists():\n"
        "    for pattern in ('*.csv', '*.parquet', '*.json', '*.tsv', '*.txt'):\n"
        "        candidate_files.extend(verified_dir.rglob(pattern))\n"
        "candidate_files = sorted({p.resolve() for p in candidate_files if p.is_file()})\n"
        "file_inventory = pd.DataFrame({\n"
        "    'file': [str(p.relative_to(ROOT)) if ROOT in p.parents else str(p) for p in candidate_files],\n"
        "    'suffix': [p.suffix.lower() for p in candidate_files],\n"
        "    'size_kb': [round(p.stat().st_size / 1024, 1) for p in candidate_files],\n"
        "})\n"
        "display(file_inventory.head(40))\n"
        "print(f'Inspectable local data/evidence files: {len(file_inventory):,}')\n"
    ))

    cells.append(_markdown(
        "## 2. Direct tabular data audit\n\n"
        "The code below deliberately avoids a project-specific wrapper. It opens the first sensible local tabular asset, "
        "shows its schema and quality profile, and makes the data issues visible before modelling. If the full raw dataset is "
        "external, run the project's documented download cell/entry point first and rerun this section.\n"
    ))
    cells.append(_direct_code(
        "tabular_candidates = [p for p in candidate_files if p.suffix.lower() in {'.csv', '.tsv', '.parquet'}]\n"
        "preferred = [p for p in tabular_candidates if not any(token in p.name.lower() for token in ('metric', 'summary', 'verification'))]\n"
        "tabular_path = (preferred or tabular_candidates or [None])[0]\n"
        "df = None\n"
        "if tabular_path is not None:\n"
        "    if tabular_path.suffix.lower() == '.parquet':\n"
        "        df = pd.read_parquet(tabular_path)\n"
        "    else:\n"
        "        sep = '\\t' if tabular_path.suffix.lower() == '.tsv' else ','\n"
        "        df = pd.read_csv(tabular_path, sep=sep, nrows=200_000)\n"
        "    print('Loaded:', tabular_path)\n"
        "    print('Shape:', df.shape)\n"
        "    display(df.head())\n"
        "    audit = pd.DataFrame({\n"
        "        'dtype': df.dtypes.astype(str),\n"
        "        'missing': df.isna().sum(),\n"
        "        'missing_pct': (100 * df.isna().mean()).round(2),\n"
        "        'unique': df.nunique(dropna=False),\n"
        "    }).sort_values(['missing_pct', 'unique'], ascending=[False, False])\n"
        "    display(audit.head(30))\n"
        "    print('Duplicate rows:', int(df.duplicated().sum()))\n"
        "else:\n"
        "    print('No local CSV/TSV/Parquet found yet. Use the project README/run path to download or build the documented dataset, then rerun this audit.')\n"
    ))

    cells.append(_markdown(
        "## 3. Exploratory data analysis and visualisation\n\n"
        "These plots are intentionally created in the notebook rather than described in prose. They expose distribution, missingness, "
        "scale, category balance and numeric relationships before any final model decision.\n"
    ))
    cells.append(_direct_code(
        "if df is not None and len(df):\n"
        "    missing_pct = (100 * df.isna().mean()).sort_values(ascending=False).head(20)\n"
        "    missing_pct = missing_pct[missing_pct > 0]\n"
        "    if len(missing_pct):\n"
        "        plt.figure(figsize=(10, 4))\n"
        "        missing_pct.plot(kind='bar')\n"
        "        plt.title('Missing values by feature (%)')\n"
        "        plt.ylabel('Missing %')\n"
        "        plt.xticks(rotation=60, ha='right')\n"
        "        plt.tight_layout()\n"
        "        plt.show()\n"
        "\n"
        "    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[:8]\n"
        "    for col in numeric_cols:\n"
        "        series = pd.to_numeric(df[col], errors='coerce').dropna()\n"
        "        if len(series):\n"
        "            plt.figure(figsize=(8, 4))\n"
        "            plt.hist(series, bins=30, alpha=0.8)\n"
        "            plt.axvline(series.median(), linestyle='--', label=f'median={series.median():.2f}')\n"
        "            plt.title(f'Distribution: {col}')\n"
        "            plt.xlabel(col)\n"
        "            plt.ylabel('Count')\n"
        "            plt.legend()\n"
        "            plt.tight_layout()\n"
        "            plt.show()\n"
        "\n"
        "    categorical_cols = [c for c in df.columns if c not in numeric_cols and df[c].nunique(dropna=False) <= 30][:4]\n"
        "    for col in categorical_cols:\n"
        "        counts = df[col].fillna('<missing>').astype(str).value_counts().head(15)\n"
        "        plt.figure(figsize=(9, 4))\n"
        "        counts.sort_values().plot(kind='barh')\n"
        "        plt.title(f'Top categories: {col}')\n"
        "        plt.xlabel('Rows')\n"
        "        plt.tight_layout()\n"
        "        plt.show()\n"
        "\n"
        "    if len(numeric_cols) >= 2:\n"
        "        corr = df[numeric_cols].corr(numeric_only=True)\n"
        "        plt.figure(figsize=(8, 6))\n"
        "        image = plt.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')\n"
        "        plt.colorbar(image, label='Correlation')\n"
        "        plt.xticks(range(len(corr.columns)), corr.columns, rotation=60, ha='right')\n"
        "        plt.yticks(range(len(corr.index)), corr.index)\n"
        "        plt.title('Numeric correlation matrix')\n"
        "        plt.tight_layout()\n"
        "        plt.show()\n"
        "\n"
        "    if len(numeric_cols) >= 2:\n"
        "        x_col, y_col = numeric_cols[0], numeric_cols[-1]\n"
        "        sample = df[[x_col, y_col]].dropna().sample(min(3000, len(df.dropna(subset=[x_col, y_col]))), random_state=42)\n"
        "        if len(sample):\n"
        "            plt.figure(figsize=(7, 5))\n"
        "            plt.scatter(sample[x_col], sample[y_col], alpha=0.35, s=18)\n"
        "            plt.xlabel(x_col)\n"
        "            plt.ylabel(y_col)\n"
        "            plt.title(f'{y_col} versus {x_col}')\n"
        "            plt.tight_layout()\n"
        "            plt.show()\n"
        "else:\n"
        "    print('Run the documented data-build/download path, then rerun this section to render raw-data EDA.')\n"
    ))

    cells.append(_markdown(
        "## 4. Inspect the measured results, not just the code\n\n"
        "A portfolio project is stronger when it retains evidence. This section reads machine-readable JSON/CSV outputs and "
        "turns scalar metrics into a quick visual comparison.\n"
    ))
    cells.append(_direct_code(
        "json_files = [p for p in candidate_files if p.suffix.lower() == '.json']\n"
        "metric_rows = []\n"
        "for path in json_files[:30]:\n"
        "    try:\n"
        "        payload = json.loads(path.read_text(encoding='utf-8'))\n"
        "    except Exception:\n"
        "        continue\n"
        "    stack = [('', payload)]\n"
        "    while stack:\n"
        "        prefix, value = stack.pop()\n"
        "        if isinstance(value, dict):\n"
        "            for key, child in value.items():\n"
        "                stack.append((f'{prefix}.{key}' if prefix else str(key), child))\n"
        "        elif isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):\n"
        "            metric_rows.append({\n"
        "                'file': str(path.relative_to(ROOT)) if ROOT in path.parents else str(path),\n"
        "                'metric': prefix,\n"
        "                'value': float(value),\n"
        "            })\n"
        "metrics_df = pd.DataFrame(metric_rows)\n"
        "if len(metrics_df):\n"
        "    display(metrics_df.head(40))\n"
        "    plot_df = metrics_df[np.isfinite(metrics_df['value'])].copy()\n"
        "    plot_df = plot_df[plot_df['value'].abs() < 1_000_000].head(20)\n"
        "    if len(plot_df):\n"
        "        labels = (plot_df['file'].str.split('/').str[-1] + ' :: ' + plot_df['metric']).tolist()\n"
        "        plt.figure(figsize=(10, max(4, 0.35 * len(plot_df))))\n"
        "        plt.barh(range(len(plot_df)), plot_df['value'])\n"
        "        plt.yticks(range(len(plot_df)), labels)\n"
        "        plt.title('Retained project metrics / evidence')\n"
        "        plt.tight_layout()\n"
        "        plt.show()\n"
        "else:\n"
        "    print('No scalar JSON evidence found. Run the project and retain metrics/results before treating it as complete.')\n"
    ))

    cells.append(_markdown(
        "## 5. Reproduce the application\n\n"
        "The notebook should be understandable without running anything, but a reviewer can reproduce the canonical application below. "
        "The switch is off by default so opening the notebook never triggers a long training job unexpectedly.\n"
    ))
    cells.append(_direct_code(
        "RUN_PROJECT = False\n"
        "entrypoint = PROJECT / 'run.py'\n"
        "if RUN_PROJECT and entrypoint.exists():\n"
        "    subprocess.run([sys.executable, str(entrypoint)], cwd=PROJECT, check=True)\n"
        "elif entrypoint.exists():\n"
        "    print(f'Reproduce with: cd {PROJECT} && {sys.executable} run.py')\n"
        "else:\n"
        "    print('This project uses a different documented entry point; see README.md in the project folder.')\n"
    ))

    cells.append(_markdown(
        "## 6. Decision / solution\n\n"
        f"{context['decision']}\n\n"
        "The final recommendation should be tied to the measured validation evidence and error analysis below. "
        "A model is not the solution by itself; the solution is the decision process built around it.\n"
    ))
    return cells


def enrich_notebook(project_dir: Path, notebook_path: Path) -> dict[str, object]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])

    # Remove only content from previous enrichment runs. Original university/project
    # work and builder cells are preserved byte-for-byte at cell level.
    preserved = [cell for cell in cells if ENRICHMENT_TAG not in _tags(cell)]
    existing_code = "\n".join(_source(cell) for cell in preserved if cell.get("cell_type") == "code")

    analysis_layer = _analysis_first_cells(project_dir)
    engineering_layer: list[dict] = []
    added_files: list[str] = []

    for path in _discover_python(project_dir):
        rel = str(path.relative_to(project_dir))
        source = path.read_text(encoding="utf-8")
        if _already_mirrored(existing_code, source):
            continue
        if not engineering_layer:
            engineering_layer.append(_markdown(
                "# Engineering appendix — canonical application source\n\n"
                "The analysis and visual evidence come first. The cells below preserve additional canonical Python from this project "
                "for reviewers who want to inspect pipelines, APIs, tests, feature code, monitoring and reusable implementation details.\n"
            ))
        engineering_layer.append(_markdown(f"## Canonical source: `{rel}`\n"))
        engineering_layer.append(_source_code(source, rel))
        added_files.append(rel)
        existing_code += "\n" + source

    final_cells = analysis_layer + preserved + engineering_layer

    if not _has_limitations_story(final_cells):
        final_cells.append(_limitations_cell(project_dir))

    code_lines = _meaningful_code_lines(final_cells)
    if TARGET_LOW <= code_lines <= TARGET_HIGH:
        depth = "within the major-project guide"
    elif code_lines > TARGET_HIGH:
        depth = "larger than the usual guide; keep the extra code only when it is genuinely project-specific"
    else:
        depth = "below the major-project guide and should grow only through substantive analysis/application depth"

    final_cells.append(_markdown(
        "# Portfolio depth check\n\n"
        f"**Meaningful code lines visible in this notebook:** {code_lines:,}. "
        f"For a major recruiter-facing application the working target is roughly **{TARGET_IDEAL:,} meaningful lines**, "
        f"with a practical guide of about {TARGET_LOW:,}–{TARGET_HIGH:,} depending on the problem. "
        f"This notebook is {depth}.\n\n"
        "Line count is not a quality metric by itself. Add code only when it improves the real project: data acquisition, validation, "
        "cleaning, EDA, visualisation, feature engineering, baselines, model comparison, tuning, leakage control, error analysis, "
        "explainability, uncertainty, inference, tests, monitoring, deployment or decision logic.\n"
    ))

    notebook["cells"] = final_cells
    notebook.setdefault("metadata", {})["portfolio_depth"] = {
        "meaningful_code_lines": code_lines,
        "target_ideal": TARGET_IDEAL,
        "target_band": [TARGET_LOW, TARGET_HIGH],
        "analysis_first": True,
        "original_cells_preserved": len(preserved),
        "canonical_python_files_added": added_files,
    }
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "project": project_dir.name,
        "code_lines": code_lines,
        "added_files": added_files,
        "original_cells_preserved": len(preserved),
    }


def main() -> None:
    reports: list[dict[str, object]] = []
    for project_dir in sorted(p for p in PROJECTS_ROOT.iterdir() if p.is_dir()):
        notebook_path = project_dir / "project_notebook.ipynb"
        if not notebook_path.exists():
            continue
        reports.append(enrich_notebook(project_dir, notebook_path))

    print(json.dumps({
        "rule": "analysis first + preserve original project + retain full application code",
        "ideal_meaningful_code_lines": TARGET_IDEAL,
        "projects": reports,
    }, indent=2))


if __name__ == "__main__":
    main()
