"""Numerical parity checks for exported inference formats."""
from __future__ import annotations

import numpy as np


def parity_report(reference: np.ndarray, candidate: np.ndarray, atol: float, rtol: float) -> dict[str, float | bool]:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {candidate.shape}")
    difference = np.abs(reference - candidate)
    return {
        "pass": bool(np.allclose(reference, candidate, atol=atol, rtol=rtol)),
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()) if difference.size else 0.0,
    }
