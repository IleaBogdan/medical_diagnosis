"""Tests for specialist agents (heuristic engine) and feature risk model."""

from heart_diagnosis.agents import (
    SpecialistAgent,
    load_specialist_agents,
)


def make_agent():
    return SpecialistAgent(
        name="cardiology_agent",
        role="Cardiologist",
        goal="g",
        backstory="b",
        features=["cp", "exang", "oldpeak", "slope", "thalach"],
        reliability=1.0,
    )


def test_load_specialist_agents_excludes_coordinator():
    agents = load_specialist_agents()
    names = {a.name for a in agents}
    assert "coordinator_agent" not in names
    assert "cardiology_agent" in names
    # every specialist must declare features
    assert all(a.features for a in agents)


def test_no_data_agent_abstains():
    agent = make_agent()
    patient = {f: None for f in agent.features}
    dx = agent.diagnose_heuristic(patient)
    assert dx.decision == "Negative"
    assert dx.confidence <= 0.1


def test_high_risk_patient_is_positive():
    agent = make_agent()
    patient = {"cp": 4, "exang": 1, "oldpeak": 3.5, "slope": 3, "thalach": 100}
    dx = agent.diagnose_heuristic(patient)
    assert dx.decision == "Positive"
    assert dx.risk_score > 0.5


def test_low_risk_patient_is_negative():
    agent = make_agent()
    patient = {"cp": 2, "exang": 0, "oldpeak": 0.0, "slope": 1, "thalach": 190}
    dx = agent.diagnose_heuristic(patient)
    assert dx.decision == "Negative"
    assert dx.risk_score < 0.5


def test_revise_moves_toward_peers():
    agent = make_agent()
    patient = {"cp": 4, "exang": 1, "oldpeak": 3.5, "slope": 3, "thalach": 100}
    prior = agent.diagnose_heuristic(patient)
    revised = agent.revise_heuristic(prior, peer_mean=0.0)
    # peers strongly disagree -> revised risk should drop below prior
    assert revised.risk_score < prior.risk_score
