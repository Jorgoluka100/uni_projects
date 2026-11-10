from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from run import GroundedRAG


@pytest.fixture()
def system() -> GroundedRAG:
    return GroundedRAG()


def test_retrieval_rejects_empty_query_and_invalid_top_k(system: GroundedRAG) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        system.retrieve("   ")
    with pytest.raises(ValueError, match="positive integer"):
        system.retrieve("access control", top_k=0)


def test_answer_rejects_invalid_abstention_configuration(system: GroundedRAG) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        system.answer("")
    with pytest.raises(ValueError, match="positive integer"):
        system.answer("access control", top_k=-1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        system.answer("access control", abstain_threshold=float("nan"))


def test_valid_query_still_returns_grounded_citation(system: GroundedRAG) -> None:
    result = system.answer("How often is privileged access reviewed?")
    assert result["citations"] == ["SEC-001"]
    assert not result["abstained"]
