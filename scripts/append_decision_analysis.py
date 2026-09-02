"""Add direct decision/robustness analysis to every recruiter notebook.

This complements EDA with practical checks a junior data professional should be able
to discuss: data slices, sensitivity, concentration, retained metric ranking,
artifact inspection, threshold-style trade-offs and an evidence-backed decision memo.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"
TAG = "portfolio-decision-analysis"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {"tags": [TAG]}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {"tags": [TAG, "analysis-first"]}, "outputs": [], "source": text.splitlines(keepends=True)}


CELLS = [
    md("# Robustness, slices and decision analysis\n\nA model or pipeline is useful only when we know where it works, where it fails and what action follows. This section adds direct slice analysis, sensitivity checks and a compact decision memo from the evidence already produced by the project.\n"),
    code("""# Quantile slices for important numeric variables
if df is not None and len(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[:10]
    quantile_rows = []
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors='coerce')
        valid = values.dropna()
        if len(valid) < 20 or valid.nunique() < 5:
            continue
        quantiles = valid.quantile([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99])
        for q, value in quantiles.items():
            quantile_rows.append({'feature':col, 'quantile':q, 'value':float(value)})
    quantile_table = pd.DataFrame(quantile_rows)
    if len(quantile_table):
        display(quantile_table.pivot(index='feature', columns='quantile', values='value').round(4))
        for col in quantile_table['feature'].unique()[:6]:
            view = quantile_table[quantile_table['feature']==col]
            plt.figure(figsize=(7,4))
            plt.plot(view['quantile'], view['value'], marker='o')
            plt.xlabel('Quantile')
            plt.ylabel(col)
            plt.title(f'Quantile profile: {col}')
            plt.tight_layout()
            plt.show()
else:
    print('Quantile slices become available after the project dataset is materialised.')
"""),
    code("""# Missingness and duplication sensitivity
if df is not None and len(df):
    missing_by_row = df.isna().sum(axis=1)
    print('Rows with any missing value:', int((missing_by_row>0).sum()))
    print('Rows with 2+ missing values:', int((missing_by_row>=2).sum()))
    print('Exact duplicate rows:', int(df.duplicated().sum()))
    if missing_by_row.max() > 0:
        plt.figure(figsize=(7,4))
        missing_by_row.value_counts().sort_index().plot(kind='bar')
        plt.title('Missing cells per row')
        plt.xlabel('Missing cells')
        plt.ylabel('Rows')
        plt.tight_layout()
        plt.show()
    duplicated = df.duplicated(keep=False)
    if duplicated.any():
        display(df.loc[duplicated].head(20))
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[:10]
    robust_rows = []
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(values) < 20:
            continue
        median = values.median()
        mad = np.median(np.abs(values-median))
        robust_z = 0.6745*(values-median)/(mad if mad else 1.0)
        robust_rows.append({'feature':col, 'median':median, 'mad':mad, 'robust_outliers_abs_z_gt_3_5':int((np.abs(robust_z)>3.5).sum())})
    robust_outliers = pd.DataFrame(robust_rows).sort_values('robust_outliers_abs_z_gt_3_5', ascending=False) if robust_rows else pd.DataFrame()
    if len(robust_outliers):
        display(robust_outliers.round(4))
"""),
    code("""# Concentration / imbalance analysis for important categorical dimensions
if df is not None and len(df):
    categorical = [c for c in df.columns if 2 <= df[c].nunique(dropna=False) <= 50][:10]
    concentration_rows = []
    for col in categorical:
        counts = df[col].fillna('<missing>').astype(str).value_counts()
        shares = counts / counts.sum()
        hhi = float((shares**2).sum())
        concentration_rows.append({'feature':col, 'categories':len(counts), 'largest_share':float(shares.iloc[0]), 'top3_share':float(shares.head(3).sum()), 'hhi':hhi})
    concentration = pd.DataFrame(concentration_rows).sort_values('hhi', ascending=False) if concentration_rows else pd.DataFrame()
    if len(concentration):
        display(concentration.round(4))
        plt.figure(figsize=(9,4))
        plt.bar(concentration['feature'], concentration['largest_share'])
        plt.ylabel('Largest category share')
        plt.title('Category concentration / imbalance')
        plt.xticks(rotation=60, ha='right')
        plt.tight_layout()
        plt.show()
"""),
    code("""# Rank all retained scalar metrics and highlight likely success/risk signals
metric_records = []
for path in json_files[:60]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        continue
    stack = [('', payload)]
    while stack:
        prefix, value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                stack.append((f'{prefix}.{key}' if prefix else str(key), child))
        elif isinstance(value, (int,float)) and not isinstance(value,bool) and np.isfinite(value):
            metric_records.append({'file':path.name, 'metric':prefix, 'value':float(value)})
all_metrics = pd.DataFrame(metric_records)
if len(all_metrics):
    signal_pattern = 'accuracy|f1|auc|precision|recall|r2|rmse|mae|loss|coverage|review|drift|psi|brier|calibration|revenue|cost|effect|lift|latency|row|reject|duplicate'
    decision_metrics = all_metrics[all_metrics['metric'].str.contains(signal_pattern, case=False, regex=True)].copy()
    if not len(decision_metrics):
        decision_metrics = all_metrics.copy()
    decision_metrics = decision_metrics.drop_duplicates(['file','metric']).reset_index(drop=True)
    display(decision_metrics.head(60).round(6))
    rate_like = decision_metrics[decision_metrics['metric'].str.contains('accuracy|f1|auc|precision|recall|coverage|rate|r2', case=False, regex=True)]
    if len(rate_like):
        bounded = rate_like[(rate_like['value']>=-1)&(rate_like['value']<=1)].head(30)
        if len(bounded):
            plt.figure(figsize=(10,max(5,0.3*len(bounded))))
            plt.barh(range(len(bounded)), bounded['value'])
            plt.yticks(range(len(bounded)), bounded['file']+' :: '+bounded['metric'])
            plt.xlim(min(-0.05,bounded['value'].min()-0.05),1.05)
            plt.title('Retained rate / quality metrics')
            plt.tight_layout()
            plt.show()
    error_like = decision_metrics[decision_metrics['metric'].str.contains('rmse|mae|loss|error|latency|drift|psi|brier', case=False, regex=True)]
    if len(error_like):
        display(error_like.sort_values('value', ascending=False).head(30).round(6))
else:
    print('No retained scalar JSON metrics are available yet.')
"""),
    code("""# Inspect artifact sizes — a quick engineering sanity check
artifact_rows = []
for base in [PROJECT/'artifacts', PROJECT/'results', PROJECT/'outputs', ROOT/'verified'/PROJECT_SLUG]:
    if not base.exists():
        continue
    for path in base.rglob('*'):
        if path.is_file():
            artifact_rows.append({'file':str(path.relative_to(ROOT)) if ROOT in path.parents else str(path), 'suffix':path.suffix.lower(), 'size_kb':path.stat().st_size/1024})
artifacts_df = pd.DataFrame(artifact_rows).sort_values('size_kb', ascending=False) if artifact_rows else pd.DataFrame()
if len(artifacts_df):
    display(artifacts_df.head(40).round(2))
    by_type = artifacts_df.groupby('suffix', as_index=False).agg(files=('file','size'), total_kb=('size_kb','sum')).sort_values('total_kb', ascending=False)
    display(by_type.round(2))
    plt.figure(figsize=(8,4))
    plt.bar(by_type['suffix'].replace('', '<none>'), by_type['total_kb'])
    plt.ylabel('Total KB')
    plt.title('Retained evidence by file type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print('No retained artifacts/results found.')
"""),
    code("""# Threshold / coverage trade-off when a result table contains confidence or probability
for path, table in result_tables:
    conf_cols = [c for c in table.columns if any(token in str(c).lower() for token in ('confidence','probability','proba','score','risk'))]
    correct_cols = [c for c in table.columns if 'correct' in str(c).lower()]
    if not conf_cols or not len(table):
        continue
    confidence = pd.to_numeric(table[conf_cols[0]], errors='coerce')
    valid_conf = confidence.notna()
    if valid_conf.sum() < 20:
        continue
    trade_rows = []
    for threshold in np.linspace(float(confidence[valid_conf].quantile(0.10)), float(confidence[valid_conf].quantile(0.90)), 9):
        accepted = valid_conf & (confidence >= threshold)
        row = {'threshold':float(threshold), 'coverage':float(accepted.mean()), 'review_rate':float((valid_conf & ~accepted).sum()/valid_conf.sum()), 'accepted_rows':int(accepted.sum())}
        if correct_cols:
            correctness = table[correct_cols[0]].astype(bool)
            row['accepted_accuracy'] = float(correctness[accepted].mean()) if accepted.any() else np.nan
        trade_rows.append(row)
    trade = pd.DataFrame(trade_rows)
    print('Trade-off table from', path.name, 'using', conf_cols[0])
    display(trade.round(4))
    plt.figure(figsize=(8,4))
    plt.plot(trade['threshold'], trade['coverage'], marker='o', label='coverage')
    if 'accepted_accuracy' in trade:
        plt.plot(trade['threshold'], trade['accepted_accuracy'], marker='o', label='accepted accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title(f'Threshold trade-off — {path.name}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    break
"""),
    code("""# Produce a concise evidence-backed decision memo inside the notebook
project_summary = {
    'project': PROJECT_SLUG,
    'local_data_or_evidence_files': int(len(candidate_files)),
    'result_tables': int(len(result_tables)),
    'json_evidence_files': int(len(json_files)),
    'visual_evidence_files': int(len(png_files)),
    'has_tests': bool((PROJECT/'tests').exists() and any((PROJECT/'tests').rglob('test*.py'))),
    'has_readme': bool((PROJECT/'README.md').exists()),
}
if df is not None:
    project_summary.update({'inspected_rows':int(len(df)), 'inspected_columns':int(df.shape[1]), 'duplicate_rows':int(df.duplicated().sum()), 'missing_cells':int(df.isna().sum().sum())})
summary_table = pd.DataFrame({'item':list(project_summary.keys()), 'value':list(project_summary.values())})
display(summary_table)
print('DECISION PRINCIPLE')
print('1. Use the measured evidence above, not model complexity, to choose the final approach.')
print('2. Inspect the worst slices/failures before making a business or operational recommendation.')
print('3. Keep uncertain, novel or high-impact cases on a review/escalation path where appropriate.')
print('4. Treat the documented limitations as part of the solution, not as boilerplate.')
"""),
]


def main() -> None:
    reports = []
    for project in sorted(p for p in PROJECTS.iterdir() if p.is_dir()):
        notebook_path = project/'project_notebook.ipynb'
        if not notebook_path.exists():
            continue
        notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
        cells = [c for c in notebook.get('cells',[]) if TAG not in set(c.get('metadata',{}).get('tags',[]))]
        insert_at = len(cells)
        for i, cell in enumerate(cells):
            src = cell.get('source',[])
            text = ''.join(src) if isinstance(src,list) else str(src)
            if 'Engineering appendix' in text or 'Portfolio depth check' in text:
                insert_at = i
                break
        notebook['cells'] = cells[:insert_at] + CELLS + cells[insert_at:]
        code_lines = 0
        for cell in notebook['cells']:
            if cell.get('cell_type') != 'code':
                continue
            src = cell.get('source',[])
            text = ''.join(src) if isinstance(src,list) else str(src)
            code_lines += sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith('#'))
        notebook.setdefault('metadata',{})['decision_analysis'] = {'enabled':True, 'direct_notebook_code':True, 'meaningful_code_lines_after_pass':code_lines}
        for cell in notebook['cells']:
            src = cell.get('source',[])
            text = ''.join(src) if isinstance(src,list) else str(src)
            if cell.get('cell_type')=='markdown' and text.startswith('# Portfolio depth check'):
                status = 'in/above the working depth range' if code_lines >= 800 else 'still below the working depth range and should grow only through substantive project-specific work'
                cell['source'] = (f'# Portfolio depth check\n\n**Meaningful visible code lines after all notebook passes:** {code_lines:,}. The working target for a major application is roughly 1,000 meaningful lines when justified by the problem. This notebook is {status}. Line count is never permission to add filler; depth must come from data, analysis, visualisation, modelling/engineering, evaluation, robustness and decision logic.\n').splitlines(keepends=True)
        notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
        reports.append({'project':project.name, 'meaningful_code_lines':code_lines})
    print(json.dumps({'projects':reports}, indent=2))


if __name__ == '__main__':
    main()
