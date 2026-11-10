from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from run import add_model_features, design_columns, fit_ols, generate_weekly_data, split_time


def test_generator_is_reproducible():
    first = generate_weekly_data(104, seed=123)
    second = generate_weekly_data(104, seed=123)
    assert np.allclose(first["sales"], second["sales"])
    assert first["week"].equals(second["week"])


def test_time_holdout_is_strictly_future():
    data = add_model_features(generate_weekly_data(120))
    train, test = split_time(data, holdout_weeks=20)
    assert train["week"].max() < test["week"].min()


def test_ols_contains_expected_terms():
    data = add_model_features(generate_weekly_data(140))
    train, _ = split_time(data, holdout_weeks=20)
    model = fit_ols(train)
    for term in design_columns():
        assert term in model.params.index

def test_time_split_rejects_unsorted_or_duplicate_weeks():
    import pytest

    data = add_model_features(generate_weekly_data(120))
    with pytest.raises(ValueError, match="chronologically ordered"):
        split_time(data.iloc[::-1].reset_index(drop=True), holdout_weeks=20)
    duplicate = data.copy()
    duplicate.loc[1, "week"] = duplicate.loc[0, "week"]
    with pytest.raises(ValueError, match="unique"):
        split_time(duplicate, holdout_weeks=20)


def test_ols_rejects_non_finite_model_inputs():
    import pytest

    data = add_model_features(generate_weekly_data(120))
    train, _ = split_time(data, holdout_weeks=20)
    train.loc[0, "sales"] = np.inf
    with pytest.raises(ValueError, match="finite numeric"):
        fit_ols(train)
