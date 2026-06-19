"""Blackboard: the shared medium for inter-agent communication.

Agents do not call each other directly. Instead they read the patient case and
publish their structured opinions to a shared blackboard; later agents (and
later debate rounds) read what their peers posted. This is a classic
blackboard architecture for multi-agent coordination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .schemas import AgentDiagnosis


@dataclass
class Blackboard:
    """Shared working memory for one patient case."""

    patient: Dict[str, Optional[float]]
    complaint: Optional[str] = None
    # round -> list of opinions posted in that round
    rounds: Dict[int, List[AgentDiagnosis]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def post(self, round_idx: int, diagnosis: AgentDiagnosis) -> None:
        self.rounds.setdefault(round_idx, []).append(diagnosis)

    def latest_opinions(self) -> List[AgentDiagnosis]:
        """The most recent opinion of each agent across all rounds."""
        latest: Dict[str, AgentDiagnosis] = {}
        for r in sorted(self.rounds):
            for dx in self.rounds[r]:
                latest[dx.agent_name] = dx
        return list(latest.values())

    def peer_summary(self, exclude: Optional[str] = None) -> str:
        """A short text summary of peers' latest opinions (for communication)."""
        lines = []
        for dx in self.latest_opinions():
            if exclude is not None and dx.agent_name == exclude:
                continue
            lines.append(
                f"- {dx.agent_name}: {dx.decision} "
                f"(risk={dx.risk_score:.2f}, confidence={dx.confidence:.2f}) "
                f"- {dx.explanation}"
            )
        return "\n".join(lines) if lines else "(no peer opinions yet)"

    def panel_mean_risk(self, exclude: Optional[str] = None) -> Optional[float]:
        opinions = [
            d for d in self.latest_opinions()
            if exclude is None or d.agent_name != exclude
        ]
        if not opinions:
            return None
        return sum(d.risk_score for d in opinions) / len(opinions)
