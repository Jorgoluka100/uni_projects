from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from run import confidence_policy, normalise_text


def test_text_normalisation_redacts_common_contact_patterns():
    text = "Email me at person@example.com and see https://example.com/page"
    cleaned = normalise_text(text)
    assert "person@example.com" not in cleaned
    assert "https://example.com/page" not in cleaned
    assert "EMAIL" in cleaned
    assert "URL" in cleaned


def test_confidence_policy_routes_uncertain_rows_to_review():
    probabilities = np.array([[0.9, 0.1], [0.52, 0.48], [0.2, 0.8]])
    accepted = confidence_policy(probabilities, threshold=0.60)
    assert accepted.tolist() == [True, False, True]
