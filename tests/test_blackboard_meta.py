"""Tests for the blackboard and the meta-agents (critic + risk)."""

from heart_diagnosis.blackboard import Blackboard
from heart_diagnosis.meta_agents import CriticAgent, RiskStratificationAgent
from heart_diagnosis.schemas import AgentDiagnosis


def dx(name, risk, conf, decision):
    return AgentDiagnosis(
        agent_name=name, risk_score=risk, confidence=conf, decision=decision,
    )


def test_blackboard_latest_and_mean():
    board = Blackboard(patient={})
    board.post(0, dx("a", 0.2, 0.5, "Negative"))
    board.post(0, dx("b", 0.8, 0.5, "Positive"))
    board.post(1, dx("a", 0.6, 0.5, "Positive"))  # a revised
    latest = {d.agent_name: d for d in board.latest_opinions()}
    assert latest["a"].risk_score == 0.6
    assert abs(board.panel_mean_risk() - 0.7) < 1e-9


def test_blackboard_peer_summary_excludes_self():
    board = Blackboard(patient={})
    board.post(0, dx("a", 0.2, 0.5, "Negative"))
    board.post(0, dx("b", 0.8, 0.5, "Positive"))
    summary = board.peer_summary(exclude="a")
    assert "b:" in summary
    assert "a:" not in summary


def test_critic_flags_premature_closure():
    # one alarmed specialist while panel leans benign
    opinions = [
        dx("a", 0.85, 0.9, "Positive"),
        dx("b", 0.2, 0.8, "Negative"),
        dx("c", 0.2, 0.8, "Negative"),
    ]
    review = CriticAgent()._review_heuristic(opinions)
    assert review.adjustment > 0
    assert review.concern


def test_risk_high_when_panel_positive():
    opinions = [dx("a", 0.8, 0.9, "Positive"), dx("b", 0.7, 0.8, "Positive")]
    assessment = RiskStratificationAgent()._assess_heuristic(opinions)
    assert assessment.risk_if_missed == "High"
    assert assessment.recommended_actions


def test_risk_low_when_panel_benign():
    opinions = [dx("a", 0.1, 0.9, "Negative"), dx("b", 0.2, 0.8, "Negative")]
    assessment = RiskStratificationAgent()._assess_heuristic(opinions)
    assert assessment.risk_if_missed == "Low"
