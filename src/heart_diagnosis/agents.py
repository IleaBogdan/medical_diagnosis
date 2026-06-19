"""Specialist agents.

Each agent reasons over a subset of features and returns an ``AgentDiagnosis``.
Two engines are supported:

* ``openai``    -> the agent is an LLM persona that returns structured JSON.
* ``heuristic`` -> a deterministic clinical scoring function (no API key needed).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

from .data_loader import FEATURES
from .llm import OpenAIJSON
from .schemas import AGENT_SCHEMA_TEXT, AgentDiagnosis


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


# --- Deterministic per-feature risk model (used by the heuristic engine) ------
# Each function maps a raw feature value to a [0,1] "points toward disease".

def _risk_age(v):
    return _clamp((v - 40.0) / 40.0)


def _risk_sex(v):
    return 0.62 if int(v) == 1 else 0.38


def _risk_cp(v):
    return {1: 0.55, 2: 0.30, 3: 0.25, 4: 0.80}.get(int(v), 0.5)


def _risk_exang(v):
    return 0.75 if int(v) == 1 else 0.32


def _risk_oldpeak(v):
    return _clamp(v / 4.0)


def _risk_slope(v):
    return {1: 0.25, 2: 0.65, 3: 0.80}.get(int(v), 0.5)


def _risk_thalach(v):
    return _clamp((170.0 - v) / 90.0)


def _risk_thal(v):
    return {3: 0.20, 6: 0.72, 7: 0.85}.get(int(v), 0.5)


def _risk_ca(v):
    return {0: 0.20, 1: 0.60, 2: 0.80, 3: 0.90}.get(int(v), _clamp(v / 3.0))


def _risk_restecg(v):
    return {0: 0.32, 1: 0.60, 2: 0.55}.get(int(v), 0.5)


def _risk_chol(v):
    return _clamp((v - 180.0) / 160.0)


def _risk_fbs(v):
    return 0.58 if int(v) == 1 else 0.45


def _risk_trestbps(v):
    return _clamp((v - 110.0) / 80.0)


FEATURE_RISK = {
    "age": _risk_age,
    "sex": _risk_sex,
    "cp": _risk_cp,
    "exang": _risk_exang,
    "oldpeak": _risk_oldpeak,
    "slope": _risk_slope,
    "thalach": _risk_thalach,
    "thal": _risk_thal,
    "ca": _risk_ca,
    "restecg": _risk_restecg,
    "chol": _risk_chol,
    "fbs": _risk_fbs,
    "trestbps": _risk_trestbps,
}


@dataclass
class SpecialistAgent:
    """A single specialist agent loaded from ``agents.yaml``."""

    name: str
    role: str
    goal: str
    backstory: str
    features: List[str]
    reliability: float = 1.0

    # -- prompt construction ---------------------------------------------------
    def _render_patient_block(
        self, patient: Dict[str, Optional[float]], complaint: Optional[str]
    ) -> str:
        lines = []
        for f in self.features:
            info = FEATURES[f]
            lines.append("- " + info.render(patient.get(f)))
        block = "\n".join(lines)
        if complaint:
            block = f"Patient's own description of the problem:\n\"{complaint}\"\n\n" + block
        return block

    def system_prompt(self) -> str:
        return (
            f"You are a {self.role.strip()}.\n"
            f"Goal: {self.goal.strip()}\n"
            f"Background: {self.backstory.strip()}\n\n"
            "You only see the subset of clinical data relevant to your "
            "specialty. Give an honest, well-calibrated opinion about whether "
            "this patient has heart disease, and lower your confidence when key "
            "data is missing."
        )

    def user_prompt(
        self, patient: Dict[str, Optional[float]], complaint: Optional[str]
    ) -> str:
        return (
            f"Patient data available to you ({', '.join(self.features)}):\n"
            f"{self._render_patient_block(patient, complaint)}\n\n"
            f"Your agent_name is \"{self.name}\".\n"
            f"{AGENT_SCHEMA_TEXT}"
        )

    # -- engines ---------------------------------------------------------------
    def diagnose_llm(
        self,
        patient: Dict[str, Optional[float]],
        llm: OpenAIJSON,
        complaint: Optional[str] = None,
    ) -> AgentDiagnosis:
        data = llm.complete_json(self.system_prompt(), self.user_prompt(patient, complaint))
        data.setdefault("agent_name", self.name)
        data.setdefault("features_used", self.features)
        return AgentDiagnosis(**data)

    def diagnose_heuristic(
        self,
        patient: Dict[str, Optional[float]],
        complaint: Optional[str] = None,
    ) -> AgentDiagnosis:
        scores = []
        evidence: Dict[str, str] = {}
        known = 0
        for f in self.features:
            val = patient.get(f)
            info = FEATURES[f]
            if val is None:
                evidence[f] = "unknown (not measured)"
                continue
            known += 1
            r = FEATURE_RISK[f](val)
            scores.append(r)
            tilt = "raises" if r >= 0.55 else ("lowers" if r <= 0.45 else "neutral for")
            evidence[f] = f"{info.render(val)} -> {tilt} cardiac risk (r={r:.2f})"

        if known == 0:
            # No relevant data: cannot assert disease. Stay Negative, near-zero
            # confidence, and say so explicitly.
            return AgentDiagnosis(
                agent_name=self.name,
                features_used=self.features,
                risk_score=0.5,
                confidence=0.05,
                decision="Negative",
                evidence=evidence,
                explanation=(
                    f"{self.role.strip()}: no relevant data provided for my "
                    f"features ({', '.join(self.features)}); cannot assess "
                    "cardiac risk."
                ),
            )

        risk = sum(scores) / len(scores)
        # Diagnose disease only when evidence tips past 0.5 (a coin flip is not
        # a diagnosis), so ties default to Negative.
        decision = "Positive" if risk > 0.5 else "Negative"
        # confidence: strong when far from 0.5 and most features are known.
        coverage = known / max(1, len(self.features))
        confidence = _clamp(0.5 + abs(risk - 0.5)) * (0.5 + 0.5 * coverage)
        explanation = (
            f"{self.role.strip()} assessment over {known}/{len(self.features)} "
            f"available features -> mean risk {risk:.2f}, "
            f"{'suggestive of' if decision == 'Positive' else 'against'} heart disease."
        )
        return AgentDiagnosis(
            agent_name=self.name,
            features_used=self.features,
            risk_score=round(risk, 3),
            confidence=round(confidence, 3),
            decision=decision,
            evidence=evidence,
            explanation=explanation,
        )

    def diagnose(
        self,
        patient: Dict[str, Optional[float]],
        engine: str = "heuristic",
        llm: Optional[OpenAIJSON] = None,
        complaint: Optional[str] = None,
    ) -> AgentDiagnosis:
        if engine == "openai":
            assert llm is not None, "OpenAI engine requires an llm client"
            try:
                return self.diagnose_llm(patient, llm, complaint)
            except Exception as exc:  # graceful degradation per-agent
                fallback = self.diagnose_heuristic(patient, complaint)
                fallback.explanation += f" [LLM error, used heuristic: {exc}]"
                return fallback
        return self.diagnose_heuristic(patient, complaint)

    # -- debate: revise after seeing peers -------------------------------------
    def revise_heuristic(
        self, prior: AgentDiagnosis, peer_mean: Optional[float]
    ) -> AgentDiagnosis:
        # Abstaining agents (no data) stay out of the vote.
        if prior.confidence <= 0.1 or peer_mean is None:
            return prior
        alpha = 0.25  # how much weight to give peers
        new_risk = _clamp((1 - alpha) * prior.risk_score + alpha * peer_mean)
        decision = "Positive" if new_risk > 0.5 else "Negative"
        agreement = 1.0 - abs(prior.risk_score - peer_mean)
        new_conf = _clamp(prior.confidence * (0.85 + 0.3 * agreement))
        return prior.model_copy(
            update={
                "risk_score": round(new_risk, 3),
                "confidence": round(new_conf, 3),
                "decision": decision,
                "explanation": (
                    prior.explanation
                    + f" [debate: peers mean {peer_mean:.2f}; "
                    f"risk {prior.risk_score:.2f}->{new_risk:.2f}]"
                ),
            }
        )

    def revise_llm(
        self,
        patient: Dict[str, Optional[float]],
        prior: AgentDiagnosis,
        peer_summary: str,
        llm: OpenAIJSON,
        complaint: Optional[str] = None,
    ) -> AgentDiagnosis:
        import json as _json

        user = (
            self.user_prompt(patient, complaint)
            + "\n\nYour earlier opinion was:\n"
            + _json.dumps(prior.model_dump(), indent=2)
            + "\n\nOther specialists said:\n"
            + peer_summary
            + "\n\nReconsider in light of your peers. You may keep or change your "
            "view. Return the SAME JSON schema as before."
        )
        data = llm.complete_json(self.system_prompt(), user)
        data.setdefault("agent_name", self.name)
        data.setdefault("features_used", self.features)
        return AgentDiagnosis(**data)

    def revise(
        self,
        patient: Dict[str, Optional[float]],
        prior: AgentDiagnosis,
        peer_summary: str,
        peer_mean: Optional[float],
        engine: str = "heuristic",
        llm: Optional[OpenAIJSON] = None,
        complaint: Optional[str] = None,
    ) -> AgentDiagnosis:
        if engine == "openai" and llm is not None:
            try:
                return self.revise_llm(patient, prior, peer_summary, llm, complaint)
            except Exception:
                return self.revise_heuristic(prior, peer_mean)
        return self.revise_heuristic(prior, peer_mean)


def _config_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")


def load_specialist_agents(agents_yaml: Optional[str] = None) -> List[SpecialistAgent]:
    """Load every feature-specialist agent (excludes the coordinator)."""
    path = agents_yaml or os.path.join(_config_dir(), "agents.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    agents = []
    for name, spec in cfg.items():
        if "features" not in spec:  # skip coordinator / non-specialists
            continue
        agents.append(
            SpecialistAgent(
                name=name,
                role=spec.get("role", name),
                goal=spec.get("goal", ""),
                backstory=spec.get("backstory", ""),
                features=list(spec["features"]),
                reliability=float(spec.get("reliability", 1.0)),
            )
        )
    return agents
