"""The diagnosis crew.

Assembles the specialist agents, the meta-agents (critic, risk) and the
coordinator, runs the chosen interaction protocol and aggregation strategy, and
returns the full structured result. This is the cardiovascular adaptation of the
CrewAI `medical_diagnosis` template's ``crew.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .agents import SpecialistAgent, load_specialist_agents
from .blackboard import Blackboard
from .llm import OpenAIJSON, resolve_engine
from .meta_agents import CriticAgent, RiskStratificationAgent
from .orchestrator import CoordinatorAgent
from .protocols import run_protocol
from .schemas import AgentDiagnosis, CoordinatorDecision, CriticReview, RiskAssessment


@dataclass
class DiagnosisResult:
    agent_outputs: List[AgentDiagnosis]
    final: CoordinatorDecision
    engine: str
    protocol: str = "single"
    strategy: str = "weighted"
    critic: Optional[CriticReview] = None
    risk: Optional[RiskAssessment] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "protocol": self.protocol,
            "strategy": self.strategy,
            "agent_outputs": [a.model_dump() for a in self.agent_outputs],
            "critic": self.critic.model_dump() if self.critic else None,
            "risk": self.risk.model_dump() if self.risk else None,
            "coordinator": self.final.model_dump(),
            "notes": self.notes,
        }


class HeartDiagnosisCrew:
    def __init__(
        self,
        engine: str = "auto",
        model: Optional[str] = None,
        llm_coordinator: bool = False,
        protocol: str = "single",
        strategy: str = "weighted",
        use_critic: bool = True,
        use_risk: bool = True,
    ):
        self.engine = resolve_engine(engine)
        self.protocol = protocol
        self.strategy = strategy
        self.use_critic = use_critic
        self.use_risk = use_risk

        self.agents: List[SpecialistAgent] = load_specialist_agents()
        self.coordinator = CoordinatorAgent(
            agent_reliability={a.name: a.reliability for a in self.agents},
            strategy=strategy,
        )
        self.critic_agent = CriticAgent()
        self.risk_agent = RiskStratificationAgent()
        self.llm_coordinator = llm_coordinator
        self.llm: Optional[OpenAIJSON] = (
            OpenAIJSON(model=model) if self.engine == "openai" else None
        )

    def diagnose(
        self,
        patient: Dict[str, Optional[float]],
        complaint: Optional[str] = None,
    ) -> DiagnosisResult:
        board = Blackboard(patient=patient, complaint=complaint)
        outputs = run_protocol(
            self.protocol, self.agents, board, engine=self.engine, llm=self.llm
        )

        critic = None
        if self.use_critic:
            critic = self.critic_agent.review(outputs, engine=self.engine, llm=self.llm)
        risk = None
        if self.use_risk:
            risk = self.risk_agent.assess(outputs, engine=self.engine, llm=self.llm)

        if self.engine == "openai" and self.llm_coordinator and self.llm is not None:
            final = self.coordinator.aggregate_llm(outputs, self.llm, critic=critic, risk=risk)
        else:
            final = self.coordinator.aggregate(outputs, critic=critic, risk=risk)

        return DiagnosisResult(
            agent_outputs=outputs,
            final=final,
            engine=self.engine,
            protocol=self.protocol,
            strategy=self.strategy,
            critic=critic,
            risk=risk,
            notes=board.notes,
        )
