from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from run import MarketingMLP, choose_threshold, prepare_frame, score


def test_network_output_shape():
    model = MarketingMLP(input_dim=20)
    batch = torch.randn(8, 20)
    assert model(batch).shape == (8,)


def test_duration_is_removed_for_pre_contact_use():
    frame = pd.DataFrame({
        "duration": [100, 200],
        "pdays": [999, 3],
        "poutcome": ["nonexistent", "success"],
        "campaign": [1, 2],
        "euribor3m": [1.2, 1.3],
        "emp_var_rate": [0.1, 0.2],
        "y": [0, 1],
    })
    prepared = prepare_frame(frame)
    assert "duration" not in prepared.columns
    assert "was_previously_contacted" in prepared.columns


def test_threshold_respects_contact_constraint():
    truth = np.array([0, 0, 0, 1, 1, 1, 0, 0], dtype=float)
    probabilities = np.array([0.01, 0.03, 0.05, 0.90, 0.80, 0.60, 0.20, 0.10])
    threshold = choose_threshold(truth, probabilities, max_contact_rate=0.50)
    result = score(truth, probabilities, threshold)
    assert 0.0 <= threshold <= 1.0
    assert result.positive_rate <= 0.50 + 1e-9
