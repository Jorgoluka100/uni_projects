"""Reproducible synthetic health-data cleaning and leakage-safe preprocessing."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = {'age': (18, 100), 'sleep_hours': (0, 24), 'steps': (0, 100000), 'resting_hr': (30, 220)}
REQUIRED = ['record_id', *NUMERIC, 'activity']


def sample_data(n=200, seed=42):
    """Artificial observations only; injected errors are intentional."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({'record_id': [f'S{i:04d}' for i in range(n)],
        'age': rng.integers(18, 85, n).astype(str),
        'sleep_hours': rng.uniform(4, 10, n).round(1).astype(str),
        'steps': rng.integers(500, 18000, n).astype(str),
        'resting_hr': rng.integers(45, 100, n).astype(str),
        'activity': rng.choice(['low', 'moderate', 'high', ' HIGH ', 'unknown'], n)})
    frame.loc[0, 'age'] = 'two'
    frame.loc[1, 'sleep_hours'] = '29'
    frame.loc[2, 'steps'] = '-30'
    frame.loc[3, 'resting_hr'] = ''
    frame.loc[4, 'age'] = '999'
    return pd.concat([frame, frame.iloc[:5]], ignore_index=True)


def clean_data(raw):
    missing = set(REQUIRED) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    frame = raw[REQUIRED].copy()
    frame['record_id'] = frame['record_id'].astype('string').str.strip()
    if frame['record_id'].isna().any() or frame['record_id'].eq('').any():
        raise ValueError('Every observation needs a nonempty record_id')
    original_count = len(frame)
    frame = frame.drop_duplicates()
    duplicate_count = original_count - len(frame)
    if frame['record_id'].duplicated().any():
        raise ValueError('Conflicting observations share a record_id; resolve upstream')
    report = {'input_rows': original_count, 'exact_duplicates_removed': duplicate_count,
              'output_rows': len(frame), 'numeric_issues': {}}
    for column, (lower, upper) in NUMERIC.items():
        text = frame[column].astype('string').str.strip()
        absent = text.isna() | text.isin(['', 'NA', 'N/A', 'unknown'])
        values = pd.to_numeric(text.mask(absent), errors='coerce')
        invalid = values.notna() & ~values.between(lower, upper)
        report['numeric_issues'][column] = {
            'missing': int(absent.sum()),
            'unparseable': int((values.isna() & ~absent).sum()),
            'out_of_range': int(invalid.sum())}
        frame[column] = values.mask(invalid).astype(float)
    activity = frame['activity'].astype('string').str.strip().str.lower()
    valid = activity.isin(['low', 'moderate', 'high'])
    report['unknown_activity_values'] = int((~valid).sum())
    frame['activity'] = activity.where(valid).astype(object).where(valid, np.nan)
    report['missing_after_cleaning'] = {c: int(frame[c].isna().sum()) for c in REQUIRED}
    return frame.reset_index(drop=True), report


def make_preprocessor():
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True,
                                                      keep_empty_features=True)),
                              ('scale', StandardScaler())]), list(NUMERIC)),
        ('category', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='unknown')),
                               ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
         ['activity'])], remainder='drop')


def run(input_path=None, output_dir='outputs'):
    raw = pd.read_csv(input_path, dtype=str) if input_path else sample_data()
    cleaned, report = clean_data(raw)
    if len(cleaned) < 5:
        raise ValueError('At least five unique observations are required')
    train, test = train_test_split(cleaned, test_size=0.2, random_state=42)
    processor = make_preprocessor()
    train_values = processor.fit_transform(train)
    test_values = processor.transform(test)
    names = processor.get_feature_names_out()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / 'raw.csv', index=False)
    cleaned.to_csv(output / 'cleaned.csv', index=False)
    for name, values, records in [('train', train_values, train), ('test', test_values, test)]:
        table = pd.DataFrame(values, columns=names)
        table.insert(0, 'record_id', records.record_id.to_numpy())
        table.to_csv(output / f'{name}_features.csv', index=False)
    report.update(train_rows=len(train), test_rows=len(test), feature_count=len(names),
                  source='user CSV' if input_path else 'synthetic demo, seed 42')
    (output / 'quality_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', help='CSV with the documented schema; otherwise use synthetic data')
    parser.add_argument('--output-dir', default='outputs')
    args = parser.parse_args()
    run(args.input, args.output_dir)
