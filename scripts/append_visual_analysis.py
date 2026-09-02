"""Append deeper direct EDA and evidence analysis to every recruiter notebook.

Idempotent: only cells tagged ``portfolio-visual-analysis`` are rebuilt.
Original notebook cells are never removed.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"
TAG = "portfolio-visual-analysis"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {"tags": [TAG]}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {"tags": [TAG, "analysis-first"]}, "outputs": [], "source": text.splitlines(keepends=True)}


CELLS = [
    md("# Deeper exploratory analysis and retained evidence\n\nThese direct notebook cells extend the initial EDA with data-quality, scale, relationship, output and error diagnostics. They are intentionally visible here rather than hidden behind project helper functions.\n"),
    code("""# Extended data-quality scorecard
if df is not None and len(df):
    quality_rows = []
    for col in df.columns:
        series = df[col]
        row = {
            'feature': col,
            'dtype': str(series.dtype),
            'rows': len(series),
            'missing': int(series.isna().sum()),
            'missing_pct': float(100 * series.isna().mean()),
            'unique': int(series.nunique(dropna=False)),
            'unique_pct': float(100 * series.nunique(dropna=False) / max(len(series), 1)),
        }
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors='coerce').dropna()
            if len(values):
                q1, q3 = values.quantile([0.25, 0.75])
                iqr = q3 - q1
                row.update({
                    'mean': float(values.mean()),
                    'median': float(values.median()),
                    'std': float(values.std()),
                    'p05': float(values.quantile(0.05)),
                    'p95': float(values.quantile(0.95)),
                    'skew': float(values.skew()),
                    'iqr_outliers': int(((values < q1 - 1.5*iqr) | (values > q3 + 1.5*iqr)).sum()),
                })
        quality_rows.append(row)
    deep_quality = pd.DataFrame(quality_rows)
    display(deep_quality.sort_values(['missing_pct','unique'], ascending=[False,False]).head(40))
    if 'iqr_outliers' in deep_quality:
        outlier_view = deep_quality.dropna(subset=['iqr_outliers']).sort_values('iqr_outliers', ascending=False).head(15)
        if len(outlier_view):
            plt.figure(figsize=(10,4))
            plt.bar(outlier_view['feature'], outlier_view['iqr_outliers'])
            plt.title('Potential IQR outliers by feature')
            plt.ylabel('Rows')
            plt.xticks(rotation=60, ha='right')
            plt.tight_layout()
            plt.show()
    card = deep_quality.sort_values('unique', ascending=False).head(20)
    plt.figure(figsize=(10,4))
    plt.bar(card['feature'], card['unique'])
    plt.title('Feature cardinality')
    plt.ylabel('Unique values')
    plt.xticks(rotation=60, ha='right')
    plt.tight_layout()
    plt.show()
    print('Constant columns:', deep_quality.loc[deep_quality['unique'] <= 1, 'feature'].tolist())
    print('High-missing columns:', deep_quality.loc[deep_quality['missing_pct'] >= 30, 'feature'].tolist())
    print('Possible identifier columns:', deep_quality.loc[deep_quality['unique_pct'] >= 95, 'feature'].tolist()[:20])
else:
    print('Materialise the documented dataset to run the extended data-quality scorecard.')
"""),
    code("""# Numeric distributions, spread and strongest pairwise relationships
if df is not None and len(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[:12]
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(values) < 5:
            continue
        clipped = values.clip(values.quantile(0.01), values.quantile(0.99))
        plt.figure(figsize=(8,4))
        plt.hist(clipped, bins=35, alpha=0.82)
        plt.axvline(values.median(), linestyle='--', label=f'median={values.median():.3g}')
        plt.axvline(values.mean(), linestyle=':', label=f'mean={values.mean():.3g}')
        plt.title(f'Distribution: {col} (1st–99th percentile)')
        plt.xlabel(col)
        plt.ylabel('Rows')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(8,3))
        plt.boxplot(values, vert=False, showfliers=True)
        plt.title(f'Spread / outliers: {col}')
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        pairs = []
        for i, left in enumerate(corr.columns):
            for right in corr.columns[i+1:]:
                value = corr.loc[left, right]
                if pd.notna(value):
                    pairs.append({'feature_a': left, 'feature_b': right, 'correlation': float(value), 'abs_correlation': float(abs(value))})
        corr_pairs = pd.DataFrame(pairs).sort_values('abs_correlation', ascending=False) if pairs else pd.DataFrame()
        if len(corr_pairs):
            display(corr_pairs.head(20).round(4))
            for _, pair in corr_pairs.head(4).iterrows():
                sample = df[[pair['feature_a'], pair['feature_b']]].dropna()
                if len(sample) > 3000:
                    sample = sample.sample(3000, random_state=42)
                plt.figure(figsize=(7,5))
                plt.scatter(sample[pair['feature_a']], sample[pair['feature_b']], alpha=0.30, s=16)
                plt.xlabel(pair['feature_a'])
                plt.ylabel(pair['feature_b'])
                plt.title(f\"{pair['feature_a']} vs {pair['feature_b']} (r={pair['correlation']:.2f})\")
                plt.tight_layout()
                plt.show()
    categorical = [c for c in df.columns if 2 <= df[c].nunique(dropna=False) <= 20][:8]
    for col in categorical:
        counts = df[col].fillna('<missing>').astype(str).value_counts().head(20)
        shares = 100 * counts / counts.sum()
        display(pd.DataFrame({'rows': counts, 'share_pct': shares.round(2)}))
        plt.figure(figsize=(8,4))
        counts.sort_values().plot(kind='barh')
        plt.title(f'Category balance: {col}')
        plt.xlabel('Rows')
        plt.tight_layout()
        plt.show()
else:
    print('Materialise the documented dataset to run distribution diagnostics.')
"""),
    code("""# Temporal coverage where date/time fields exist
if df is not None and len(df):
    time_cols = [c for c in df.columns if any(token in str(c).lower() for token in ('date','time','timestamp','datetime'))]
    print('Date/time candidates:', time_cols[:10])
    for col in time_cols[:4]:
        converted = pd.to_datetime(df[col], errors='coerce')
        valid = converted.dropna()
        if len(valid) >= max(10, int(0.25*len(df))):
            print(col, 'range:', valid.min(), '→', valid.max())
            monthly = valid.dt.to_period('M').value_counts().sort_index()
            if len(monthly) > 1:
                plt.figure(figsize=(10,4))
                plt.plot(monthly.index.astype(str), monthly.values, marker='o')
                plt.title(f'Rows over time: {col}')
                plt.ylabel('Rows')
                plt.xticks(rotation=70, ha='right')
                plt.tight_layout()
                plt.show()
"""),
    md("## Retained outputs and error analysis\n\nA strong portfolio keeps inspectable evidence. The cells below profile compact result tables and automatically detect prediction-like columns for residual or misclassification analysis.\n"),
    code("""# Load compact result/evidence tables
result_tables = []
for base in [PROJECT/'results', PROJECT/'outputs', PROJECT/'artifacts', ROOT/'verified'/PROJECT_SLUG]:
    if not base.exists():
        continue
    for path in sorted(base.rglob('*')):
        if path.is_file() and path.suffix.lower() in {'.csv','.tsv','.parquet'} and path.stat().st_size < 25_000_000:
            try:
                if path.suffix.lower() == '.parquet':
                    table = pd.read_parquet(path)
                else:
                    table = pd.read_csv(path, sep='\t' if path.suffix.lower() == '.tsv' else ',')
            except Exception as exc:
                print('Could not read', path.name, '-', exc)
                continue
            result_tables.append((path, table))
            print('\nRESULT TABLE:', path.relative_to(ROOT) if ROOT in path.parents else path)
            print('shape=', table.shape)
            display(table.head(15))
            numeric = table.select_dtypes(include=np.number).columns.tolist()[:12]
            if numeric:
                display(table[numeric].describe().T.round(4))
print('Inspectable result tables:', len(result_tables))
"""),
    code("""# Automatic regression/classification-style error diagnostics
actual_tokens = ('actual','target','truth','y_true','observed','label')
pred_tokens = ('prediction','predicted','forecast','y_pred')
confidence_tokens = ('confidence','probability','proba','risk','uncertainty')
for path, table in result_tables:
    actual_cols = [c for c in table.columns if any(token in str(c).lower() for token in actual_tokens)]
    pred_cols = [c for c in table.columns if any(token in str(c).lower() for token in pred_tokens)]
    conf_cols = [c for c in table.columns if any(token in str(c).lower() for token in confidence_tokens)]
    if actual_cols and pred_cols and len(table):
        actual_col = actual_cols[0]
        pred_col = next((c for c in pred_cols if c != actual_col), pred_cols[0])
        actual_num = pd.to_numeric(table[actual_col], errors='coerce')
        pred_num = pd.to_numeric(table[pred_col], errors='coerce')
        numeric_mask = actual_num.notna() & pred_num.notna()
        if numeric_mask.sum() >= 10:
            residual = actual_num[numeric_mask] - pred_num[numeric_mask]
            abs_error = residual.abs()
            print('\n', path.name, '| MAE=', round(float(abs_error.mean()),5), '| RMSE=', round(float(np.sqrt(np.mean(residual**2))),5), '| bias=', round(float(residual.mean()),5))
            plt.figure(figsize=(7,5))
            plt.scatter(actual_num[numeric_mask], pred_num[numeric_mask], alpha=0.35, s=18)
            lo = min(actual_num[numeric_mask].min(), pred_num[numeric_mask].min())
            hi = max(actual_num[numeric_mask].max(), pred_num[numeric_mask].max())
            plt.plot([lo,hi],[lo,hi], linestyle='--')
            plt.xlabel(str(actual_col))
            plt.ylabel(str(pred_col))
            plt.title(f'Actual vs predicted — {path.name}')
            plt.tight_layout()
            plt.show()
            plt.figure(figsize=(7,4))
            plt.hist(residual, bins=30, alpha=0.82)
            plt.axvline(0, linestyle='--')
            plt.title(f'Residual distribution — {path.name}')
            plt.tight_layout()
            plt.show()
            worst_idx = abs_error.nlargest(min(15,len(abs_error))).index
            cols = list(dict.fromkeys([actual_col,pred_col]+conf_cols[:2]))
            worst = table.loc[worst_idx, cols].copy()
            worst['absolute_error'] = abs_error.loc[worst_idx].values
            display(worst.sort_values('absolute_error', ascending=False))
        else:
            agreement = table[actual_col].astype(str) == table[pred_col].astype(str)
            print('\n', path.name, '| classification agreement=', round(float(agreement.mean()),4))
            if (~agreement).any():
                display(table.loc[~agreement, [actual_col,pred_col]+conf_cols[:2]].head(20))
    elif conf_cols:
        for col in conf_cols[:2]:
            values = pd.to_numeric(table[col], errors='coerce').dropna()
            if len(values) >= 10:
                plt.figure(figsize=(7,4))
                plt.hist(values, bins=30, alpha=0.82)
                plt.title(f'{col} distribution — {path.name}')
                plt.tight_layout()
                plt.show()
"""),
    code("""# Display retained visual evidence from actual project runs
png_files = []
for base in [PROJECT/'results', PROJECT/'outputs', PROJECT/'artifacts', ROOT/'verified'/PROJECT_SLUG]:
    if base.exists():
        png_files.extend(sorted(base.rglob('*.png')))
print('Retained PNG figures:', len(png_files))
for path in png_files[:12]:
    try:
        image = plt.imread(path)
        plt.figure(figsize=(10,6))
        plt.imshow(image)
        plt.axis('off')
        plt.title(str(path.relative_to(ROOT)) if ROOT in path.parents else path.name)
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print('Could not display', path.name, '-', exc)
"""),
    code("""# Reproducibility and evidence checklist
checks = [
    {'check':'README present', 'status':(PROJECT/'README.md').exists()},
    {'check':'Recruiter notebook present', 'status':(PROJECT/'project_notebook.ipynb').exists()},
    {'check':'Python implementation present', 'status':any(PROJECT.rglob('*.py'))},
    {'check':'Tests present', 'status':(PROJECT/'tests').exists() and any((PROJECT/'tests').rglob('test*.py'))},
    {'check':'Result/evidence files present', 'status':bool(candidate_files)},
    {'check':'Machine-readable JSON evidence', 'status':bool(json_files)},
    {'check':'Retained visual evidence', 'status':bool(png_files)},
]
checklist = pd.DataFrame(checks)
display(checklist)
print('Evidence checklist pass rate:', f\"{100*checklist['status'].mean():.1f}%\")
print('A failed item is a prompt to strengthen the project, not something to hide.')
"""),
]


def main() -> None:
    report = []
    for project in sorted(p for p in PROJECTS.iterdir() if p.is_dir()):
        notebook_path = project / 'project_notebook.ipynb'
        if not notebook_path.exists():
            continue
        notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
        existing = notebook.get('cells', [])
        kept = [cell for cell in existing if TAG not in set(cell.get('metadata', {}).get('tags', []))]
        insert_at = len(kept)
        for i, cell in enumerate(kept):
            source = cell.get('source', [])
            text = ''.join(source) if isinstance(source, list) else str(source)
            if 'Engineering appendix' in text or 'Portfolio depth check' in text:
                insert_at = i
                break
        notebook['cells'] = kept[:insert_at] + CELLS + kept[insert_at:]
        notebook.setdefault('metadata', {})['visual_analysis'] = {'enabled': True, 'direct_notebook_code': True, 'added_cells': len(CELLS)}
        notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
        code_lines = 0
        for cell in notebook['cells']:
            if cell.get('cell_type') != 'code':
                continue
            src = cell.get('source', [])
            text = ''.join(src) if isinstance(src, list) else str(src)
            code_lines += sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith('#'))
        report.append({'project': project.name, 'meaningful_code_lines': code_lines})
    print(json.dumps({'projects': report}, indent=2))


if __name__ == '__main__':
    main()
