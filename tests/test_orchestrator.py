"""Tests for the coordinator's decision-making strategies."""

import pytest

from heart_diagnosis.orchestrator import CoordinatorAgent
from heart_diagnosis.schemas import AgentDiagnosis, CriticReview


def dx(name, risk, conf, decision):
    return AgentDiagnosis(
        agent_name=name, risk_score=risk, confidence=conf, decision=decision,
    )


def panel():
    return [
        dx("a", 0.8, 0.9, "Positive"),
        dx("b", 0.7, 0.8, "Positive"),
        dx("c", 0.2, 0.6, "Negative"),
    ]


def test_weighted_positive():
    coord = CoordinatorAgent(strategy="weighted")
    out = coord.aggregate(panel())
    assert out.final_decision == "Positive"
    assert 0 <= out.final_score <= 1


def test_majority_counts_votes():
    coord = CoordinatorAgent(strategy="majority")
    out = coord.aggregate(panel())
    # 2 of 3 positive -> Positive
    assert out.final_decision == "Positive"


def test_bayesian_runs():
    coord = CoordinatorAgent(strategy="bayesian")
    out = coord.aggregate(panel())
    assert out.final_decision in {"Positive", "Negative"}


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        CoordinatorAgent(strategy="nonsense")


def test_insufficient_data_short_circuit():
    coord = CoordinatorAgent(strategy="weighted")
    abstainers = [dx("a", 0.5, 0.05, "Negative"), dx("b", 0.5, 0.05, "Negative")]
    out = coord.aggregate(abstainers)
    assert out.final_decision == "Negative"
    assert "Insufficient" in out.explanation


def test_critic_adjustment_applied():
    coord = CoordinatorAgent(strategy="weighted")
    base = coord.aggregate(panel()).final_score
    crit = CriticReview(adjustment=-0.3, concern="overcalled")
    adjusted = coord.aggregate(panel(), critic=crit).final_score
    assert adjusted < base


def test_tie_defaults_to_negative():
    coord = CoordinatorAgent(strategy="weighted")
    tie = [dx("a", 0.5, 0.8, "Negative"), dx("b", 0.5, 0.8, "Negative")]
    out = coord.aggregate(tie)
    # confidence high enough to not short-circuit; score 0.5 -> Negative
    assert out.final_decision == "Negative"
