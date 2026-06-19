"""Coordinator agent: aggregate specialist diagnoses into a final decision.

Supports several pluggable decision-making strategies and can fold in the
Critic agent's adjustment and the Risk-Stratification agent's danger level.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

from .llm import OpenAIJSON
from .schemas import (
    COORDINATOR_SCHEMA_TEXT,
    AgentDiagnosis,
    CoordinatorDecision,
    CriticReview,
    RiskAssessment,
)

STRATEGIES = ("weighted", "majority", "bayesian")


def _logit(p: float) -> float:
    p = min(0.999, max(0.001, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CoordinatorAgent:
    """Aggregates agent outputs using a selectable decision strategy."""

    def __init__(
        self,
        agent_reliability: Optional[Dict[str, float]] = None,
        strategy: str = "weighted",
    ):
        self.agent_reliability = agent_reliability or {}
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Use one of {STRATEGIES}.")
        self.strategy = strategy

    def _weight(self, dx: AgentDiagnosis) -> float:
        reliability = self.agent_reliability.get(dx.agent_name, 1.0)
        return max(1e-6, reliability * dx.confidence)

    # -- strategies ------------------------------------------------------------
    def _score_weighted(self, diagnoses, weights) -> float:
        total = sum(weights)
        return sum(w * d.risk_score for w, d in zip(weights, diagnoses)) / total

    def _score_majority(self, diagnoses, weights) -> float:
        # weighted proportion voting Positive
        total = sum(weights)
        pos = sum(w for w, d in zip(weights, diagnoses) if d.decision == "Positive")
        return pos / total

    def _score_bayesian(self, diagnoses, weights) -> float:
        total = sum(weights)
        pooled = sum(w * _logit(d.risk_score) for w, d in zip(weights, diagnoses)) / total
        return _sigmoid(pooled)

    def _compute_score(self, diagnoses, weights) -> float:
        if self.strategy == "majority":
            return self._score_majority(diagnoses, weights)
        if self.strategy == "bayesian":
            return self._score_bayesian(diagnoses, weights)
        return self._score_weighted(diagnoses, weights)

    # -- aggregation -----------------------------------------------------------
    def aggregate(
        self,
        diagnoses: List[AgentDiagnosis],
        critic: Optional[CriticReview] = None,
        risk: Optional[RiskAssessment] = None,
    ) -> CoordinatorDecision:
        if not diagnoses:
            return CoordinatorDecision(
                final_score=0.5, final_decision="Negative", agreement_level=0.0,
                explanation="No agent outputs were available.",
            )

        if all(d.confidence <= 0.1 for d in diagnoses):
            return CoordinatorDecision(
                final_score=0.5, final_decision="Negative", agreement_level=0.0,
                explanation=(
                    "Insufficient information to assess heart-disease risk. The "
                    "input did not map to cardiac risk factors. Please share "
                    "details such as chest pain, exercise tolerance, age, sex, "
                    "blood pressure, cholesterol, or ECG/imaging results."
                ),
            )

        weights = [self._weight(d) for d in diagnoses]
        total_w = sum(weights)
        base_score = self._compute_score(diagnoses, weights)

        final_score = base_score
        parts = [f"strategy={self.strategy}, base score {base_score:.2f}"]
        if critic is not None and abs(critic.adjustment) > 0:
            final_score = max(0.0, min(1.0, final_score + critic.adjustment))
            parts.append(f"critic adj {critic.adjustment:+.2f} ({critic.concern or 'ok'})")

        final_decision = "Positive" if final_score > 0.5 else "Negative"
        agree_w = sum(
            w for w, d in zip(weights, diagnoses) if d.decision == final_decision
        )
        agreement_level = agree_w / total_w

        pos = [d.agent_name for d in diagnoses if d.decision == "Positive"]
        neg = [d.agent_name for d in diagnoses if d.decision == "Negative"]
        if risk is not None:
            parts.append(f"risk-if-missed: {risk.risk_if_missed}")
        explanation = (
            f"{', '.join(parts)} -> {final_score:.2f} -> {final_decision}. "
            f"For disease: {pos or 'none'}; against: {neg or 'none'}. "
            f"Weighted agreement {agreement_level:.0%}."
        )
        return CoordinatorDecision(
            final_score=round(final_score, 3),
            final_decision=final_decision,
            agreement_level=round(agreement_level, 3),
            explanation=explanation,
        )

    def aggregate_llm(
        self, diagnoses: List[AgentDiagnosis], llm: OpenAIJSON,
        critic: Optional[CriticReview] = None,
        risk: Optional[RiskAssessment] = None,
    ) -> CoordinatorDecision:
        """Let an LLM attending physician synthesize the final decision."""
        panel = [d.model_dump() for d in diagnoses]
        reliab = {d.agent_name: self.agent_reliability.get(d.agent_name, 1.0) for d in diagnoses}
        extra = ""
        if critic is not None:
            extra += f"\nCritic review:\n{json.dumps(critic.model_dump(), indent=2)}"
        if risk is not None:
            extra += f"\nRisk stratification:\n{json.dumps(risk.model_dump(), indent=2)}"
        system = (
            "You are the attending physician coordinating a panel of specialist "
            "agents for heart-disease diagnosis. Weigh each specialist by its "
            "reliability and stated confidence, take the critic and risk notes "
            "into account, resolve disagreements, and give one final decision."
        )
        user = (
            f"Specialist reliability weights:\n{json.dumps(reliab, indent=2)}\n\n"
            f"Specialist structured opinions:\n{json.dumps(panel, indent=2)}\n"
            f"{extra}\n\n{COORDINATOR_SCHEMA_TEXT}"
        )
        try:
            return CoordinatorDecision(**llm.complete_json(system, user))
        except Exception:
            return self.aggregate(diagnoses, critic=critic, risk=risk)
