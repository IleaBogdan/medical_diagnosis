"""Tests for the structured-output schemas and their normalization."""

import pytest

from heart_diagnosis.schemas import (
    AgentDiagnosis,
    CoordinatorDecision,
    CriticReview,
    RiskAssessment,
)


def test_agent_decision_normalization():
    dx = AgentDiagnosis(
        agent_name="a", risk_score=0.8, confidence=0.9,
        decision="heart_disease",
    )
    assert dx.decision == "Positive"

    dx2 = AgentDiagnosis(
        agent_name="a", risk_score=0.1, confidence=0.9, decision="no",
    )
    assert dx2.decision == "Negative"


def test_agent_score_clamping():
    dx = AgentDiagnosis(
        agent_name="a", risk_score=5, confidence=-2, decision="Positive",
    )
    assert dx.risk_score == 1.0
    assert dx.confidence == 0.0


def test_critic_adjustment_clamped():
    c = CriticReview(adjustment=0.99)
    assert c.adjustment == 0.3
    c2 = CriticReview(adjustment=-5)
    assert c2.adjustment == -0.3


def test_risk_level_normalization():
    r = RiskAssessment(risk_if_missed="very HIGH risk")
    assert r.risk_if_missed == "High"


def test_coordinator_decision_bounds():
    d = CoordinatorDecision(
        final_score=2.0, final_decision="Positive", agreement_level=-1,
    )
    assert d.final_score == 1.0
    assert d.agreement_level == 0.0
