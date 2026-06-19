"""Structured output schemas for agents and the coordinator.

These mirror the JSON contracts described in the assignment so evaluation is
trivial and the LLM outputs are machine-checkable.
"""

from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field, field_validator

Decision = Literal["Positive", "Negative"]


class AgentDiagnosis(BaseModel):
    """Structured output produced by a single specialist agent."""

    agent_name: str
    features_used: List[str] = Field(default_factory=list)
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    decision: Decision
    evidence: Dict[str, str] = Field(default_factory=dict)
    explanation: str = ""

    @field_validator("risk_score", "confidence", mode="before")
    @classmethod
    def _clamp(cls, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, v))

    @field_validator("decision", mode="before")
    @classmethod
    def _normalize_decision(cls, v):
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"positive", "pos", "yes", "heart_disease", "disease", "1", "true"}:
                return "Positive"
            if s in {"negative", "neg", "no", "no_heart_disease", "healthy", "0", "false"}:
                return "Negative"
        return v


RiskLevel = Literal["Low", "Moderate", "High"]


class CriticReview(BaseModel):
    """Output of the devil's-advocate Critic agent.

    The critic reads the whole panel and guards against groupthink / premature
    diagnostic closure. ``adjustment`` is a signed delta applied to the final
    score by the coordinator (bounded to a small range).
    """

    agrees_with_panel: bool = True
    concern: str = ""
    adjustment: float = Field(default=0.0, ge=-0.3, le=0.3)
    explanation: str = ""

    @field_validator("adjustment", mode="before")
    @classmethod
    def _clamp_adj(cls, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(-0.3, min(0.3, v))


class RiskAssessment(BaseModel):
    """Output of the Risk-Stratification agent (triage / safety)."""

    leading_hypothesis: str = "coronary artery disease"
    risk_if_missed: RiskLevel = "Moderate"
    recommended_actions: List[str] = Field(default_factory=list)
    explanation: str = ""

    @field_validator("risk_if_missed", mode="before")
    @classmethod
    def _normalize_risk(cls, v):
        if isinstance(v, str):
            s = v.strip().lower()
            for level in ("high", "moderate", "low"):
                if level in s:
                    return level.capitalize()
        return v


class CoordinatorDecision(BaseModel):
    """Final aggregated decision produced by the coordinator agent."""

    final_score: float = Field(ge=0.0, le=1.0)
    final_decision: Decision
    agreement_level: float = Field(ge=0.0, le=1.0)
    explanation: str = ""

    @field_validator("final_score", "agreement_level", mode="before")
    @classmethod
    def _clamp(cls, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, v))


# JSON schema text injected into the LLM prompts (kept close to the assignment).
AGENT_SCHEMA_TEXT = """
Return your answer STRICTLY as a single JSON object with these fields:
{
  "agent_name": "string",
  "features_used": ["feature1", "feature2"],
  "risk_score": 0.0,            // probability of heart disease, float in [0,1]
  "confidence": 0.0,            // how confident you are, float in [0,1]
  "decision": "Positive" or "Negative",
  "evidence": { "feature_name": "brief interpretation" },
  "explanation": "short clinical reasoning summary"
}
No text outside the JSON object.
""".strip()

COORDINATOR_SCHEMA_TEXT = """
Return your answer STRICTLY as a single JSON object with these fields:
{
  "final_score": 0.0,                 // aggregated probability, float in [0,1]
  "final_decision": "Positive" or "Negative",
  "agreement_level": 0.0,             // how much the specialists agreed, float in [0,1]
  "explanation": "short explanation"
}
No text outside the JSON object.
""".strip()

CRITIC_SCHEMA_TEXT = """
You are a devil's advocate. Challenge the panel and guard against premature
closure. Return STRICTLY a single JSON object:
{
  "agrees_with_panel": true,          // false if you think the panel is wrong
  "concern": "main concern / what could be missed",
  "adjustment": 0.0,                  // signed change to the final risk, in [-0.3, 0.3]
  "explanation": "short reasoning"
}
No text outside the JSON object.
""".strip()

RISK_SCHEMA_TEXT = """
You triage how dangerous it would be to MISS this diagnosis. Return STRICTLY a
single JSON object:
{
  "leading_hypothesis": "string",
  "risk_if_missed": "Low" or "Moderate" or "High",
  "recommended_actions": ["string"],
  "explanation": "short reasoning"
}
No text outside the JSON object.
""".strip()
