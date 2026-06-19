"""Meta-agents that reason over *other agents'* outputs.

These agents demonstrate inter-agent communication and higher-order
coordination: they consume the specialist panel from the blackboard rather than
raw features.

* CriticAgent              - devil's advocate; guards against premature closure.
* RiskStratificationAgent  - triage; how dangerous is it to miss this?
"""

from __future__ import annotations

import json
from typing import List, Optional

from .blackboard import Blackboard
from .llm import OpenAIJSON
from .schemas import (
    CRITIC_SCHEMA_TEXT,
    RISK_SCHEMA_TEXT,
    AgentDiagnosis,
    CriticReview,
    RiskAssessment,
)


def _panel_json(opinions: List[AgentDiagnosis]) -> str:
    return json.dumps([o.model_dump() for o in opinions], indent=2)


class CriticAgent:
    """Devil's advocate. Challenges the panel to avoid groupthink."""

    name = "critic_agent"
    role = "Devil's-Advocate Reviewer"

    def review(
        self,
        opinions: List[AgentDiagnosis],
        engine: str = "heuristic",
        llm: Optional[OpenAIJSON] = None,
    ) -> CriticReview:
        if engine == "openai" and llm is not None:
            try:
                return self._review_llm(opinions, llm)
            except Exception:
                pass
        return self._review_heuristic(opinions)

    def _review_llm(
        self, opinions: List[AgentDiagnosis], llm: OpenAIJSON
    ) -> CriticReview:
        system = (
            "You are a devil's-advocate physician on a diagnostic panel. Your job "
            "is to challenge consensus, surface what could be missed, and prevent "
            "premature diagnostic closure. Heart disease is dangerous to miss."
        )
        user = (
            "The specialist panel produced these opinions:\n"
            f"{_panel_json(opinions)}\n\n"
            f"{CRITIC_SCHEMA_TEXT}"
        )
        return CriticReview(**llm.complete_json(system, user))

    def _review_heuristic(self, opinions: List[AgentDiagnosis]) -> CriticReview:
        if not opinions:
            return CriticReview(explanation="No panel to review.")
        risks = [o.risk_score for o in opinions]
        mean = sum(risks) / len(risks)
        mx, mn = max(risks), min(risks)
        spread = mx - mn
        mean_conf = sum(o.confidence for o in opinions) / len(opinions)

        adjustment = 0.0
        concerns: List[str] = []

        # Safety: one alarmed specialist while the panel leans benign -> don't
        # miss disease (anti premature-closure, raise risk).
        if mx >= 0.7 and mean < 0.5:
            adjustment += min(0.2, mx - mean)
            concerns.append(
                "at least one specialist flags high risk while the panel leans "
                "benign - avoid prematurely ruling out disease"
            )
        # Strong disagreement -> flag uncertainty.
        if spread >= 0.3:
            concerns.append("substantial disagreement among specialists")
        # Thin evidence -> caution.
        if mean_conf < 0.4:
            concerns.append("panel confidence is low (sparse data)")
            adjustment += 0.0  # uncertainty noted, no directional push

        adjustment = max(-0.3, min(0.3, round(adjustment, 3)))
        agrees = abs(adjustment) < 0.05 and spread < 0.3
        explanation = (
            f"Reviewed {len(opinions)} opinions: mean risk {mean:.2f}, "
            f"spread {spread:.2f}, mean confidence {mean_conf:.2f}. "
            + ("; ".join(concerns) if concerns else "No major red flags.")
        )
        return CriticReview(
            agrees_with_panel=agrees,
            concern="; ".join(concerns),
            adjustment=adjustment,
            explanation=explanation,
        )


class RiskStratificationAgent:
    """Triage agent: how dangerous would it be to miss this diagnosis?"""

    name = "risk_agent"
    role = "Risk-Stratification / Triage"

    ACTIONS = {
        "High": ["12-lead ECG", "serial troponins", "urgent cardiology consult"],
        "Moderate": ["resting ECG", "exercise stress test", "lipid panel"],
        "Low": ["risk-factor modification", "routine outpatient follow-up"],
    }

    def assess(
        self,
        opinions: List[AgentDiagnosis],
        engine: str = "heuristic",
        llm: Optional[OpenAIJSON] = None,
    ) -> RiskAssessment:
        if engine == "openai" and llm is not None:
            try:
                return self._assess_llm(opinions, llm)
            except Exception:
                pass
        return self._assess_heuristic(opinions)

    def _assess_llm(
        self, opinions: List[AgentDiagnosis], llm: OpenAIJSON
    ) -> RiskAssessment:
        system = (
            "You are the triage physician. Given the panel's opinions, judge how "
            "dangerous it would be to MISS coronary disease and recommend actions."
        )
        user = (
            "Panel opinions:\n"
            f"{_panel_json(opinions)}\n\n"
            f"{RISK_SCHEMA_TEXT}"
        )
        return RiskAssessment(**llm.complete_json(system, user))

    def _assess_heuristic(self, opinions: List[AgentDiagnosis]) -> RiskAssessment:
        if not opinions:
            level = "Moderate"
            mean = 0.5
        else:
            mean = sum(o.risk_score for o in opinions) / len(opinions)
            # Missing cardiac disease is costly, so thresholds are cautious.
            if mean >= 0.5:
                level = "High"
            elif mean >= 0.35:
                level = "Moderate"
            else:
                level = "Low"
        return RiskAssessment(
            leading_hypothesis="coronary artery disease",
            risk_if_missed=level,
            recommended_actions=self.ACTIONS[level],
            explanation=(
                f"Panel mean risk {mean:.2f} -> {level} danger if missed. "
                "Cardiac diagnoses are weighted toward caution."
            ),
        )
