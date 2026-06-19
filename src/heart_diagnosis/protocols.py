"""Interaction protocols: how the specialist agents communicate.

* single  - each specialist gives one independent opinion (no communication).
* debate  - two rounds: independent opinions, then every specialist revises
            after reading its peers' opinions from the blackboard.
"""

from __future__ import annotations

from typing import List, Optional

from .agents import SpecialistAgent
from .blackboard import Blackboard
from .llm import OpenAIJSON
from .schemas import AgentDiagnosis

PROTOCOLS = ("single", "debate")


def run_single(
    agents: List[SpecialistAgent],
    board: Blackboard,
    engine: str = "heuristic",
    llm: Optional[OpenAIJSON] = None,
) -> List[AgentDiagnosis]:
    for agent in agents:
        dx = agent.diagnose(board.patient, engine=engine, llm=llm, complaint=board.complaint)
        board.post(0, dx)
    return board.latest_opinions()


def run_debate(
    agents: List[SpecialistAgent],
    board: Blackboard,
    engine: str = "heuristic",
    llm: Optional[OpenAIJSON] = None,
    rounds: int = 2,
) -> List[AgentDiagnosis]:
    # Round 0: independent opinions.
    run_single(agents, board, engine=engine, llm=llm)
    board.notes.append("Round 0: independent specialist opinions posted.")

    # Subsequent rounds: revise after reading peers.
    for r in range(1, rounds):
        priors = {d.agent_name: d for d in board.latest_opinions()}
        for agent in agents:
            prior = priors.get(agent.name)
            if prior is None:
                continue
            peer_mean = board.panel_mean_risk(exclude=agent.name)
            peer_summary = board.peer_summary(exclude=agent.name)
            revised = agent.revise(
                board.patient,
                prior,
                peer_summary,
                peer_mean,
                engine=engine,
                llm=llm,
                complaint=board.complaint,
            )
            board.post(r, revised)
        board.notes.append(f"Round {r}: specialists revised after seeing peers.")
    return board.latest_opinions()


def run_protocol(
    protocol: str,
    agents: List[SpecialistAgent],
    board: Blackboard,
    engine: str = "heuristic",
    llm: Optional[OpenAIJSON] = None,
) -> List[AgentDiagnosis]:
    if protocol == "debate":
        return run_debate(agents, board, engine=engine, llm=llm)
    return run_single(agents, board, engine=engine, llm=llm)
