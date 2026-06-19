# Architecture

This document describes the extended multi-agent architecture and how it maps to
the four project emphases: **agent autonomy & specialization**, **inter-agent
communication**, **coordination & aggregation of results**, and **software
engineering best practices**.

## High-level flow

```
                         ┌──────────────────────────┐
   free text  ──intake──▶│   Blackboard (shared)    │
   or features           │  patient case + opinions │
                         └────────────┬─────────────┘
                                      │ read patient
        ┌───────────────┬────────────┼─────────────┬───────────────┐
        ▼               ▼            ▼             ▼               ▼
  cardiology       imaging      metabolic     demographics     (Round 0:
   agent            agent         agent          agent         independent)
        └───────────────┴──────┬─────┴─────────────┴───────────────┘
                               │ post opinions
                 (debate) Round 1: each specialist READS peers
                               │ and revises its opinion
                               ▼
                   ┌───────────────────────┐
                   │   Critic agent        │  reads whole panel,
                   │  (devil's advocate)   │  outputs score adjustment
                   └───────────┬───────────┘
                   ┌───────────▼───────────┐
                   │  Risk-stratification  │  reads panel,
                   │  agent (triage)       │  outputs risk_if_missed + actions
                   └───────────┬───────────┘
                               ▼
                   ┌───────────────────────┐
                   │   Coordinator         │  strategy ∈ {weighted,
                   │   (attending)         │  majority, bayesian}
                   └───────────┬───────────┘
                               ▼
                   final_score / final_decision /
                   agreement_level / explanation
```

## 1. Agent autonomy & specialization

Two kinds of agents:

**Feature specialists** (in `agents.yaml`, loaded by `agents.py`) each own a
clinically meaningful *subset* of the dataset features and decide independently:

| Agent | Features | Role |
|-------|----------|------|
| `cardiology_agent`   | cp, exang, oldpeak, slope, thalach | ischemia/angina |
| `imaging_agent`      | thal, ca, restecg | perfusion imaging |
| `metabolic_agent`    | chol, fbs, trestbps | metabolic risk |
| `demographics_agent` | age, sex | demographic risk |

**Meta-agents** (`meta_agents.py`) reason over *other agents' outputs*:

- `CriticAgent` — a devil's advocate that challenges consensus and guards
  against premature diagnostic closure. It emits a bounded `adjustment` to the
  final score (e.g. raises risk when one specialist is alarmed but the panel
  leans benign).
- `RiskStratificationAgent` — a triage agent that judges how dangerous it would
  be to *miss* the diagnosis (`risk_if_missed`) and recommends actions.

Every agent has two interchangeable engines: an **LLM persona** (OpenAI/OpenRouter
JSON mode) and a **deterministic heuristic** (so the system runs offline and is
unit-testable).

## 2. Inter-agent communication

Communication is mediated by a **`Blackboard`** (`blackboard.py`) — agents never
call each other directly. They post structured opinions per round; later agents
and later rounds read peers via `peer_summary()` / `panel_mean_risk()`.

The **debate protocol** (`protocols.py`) implements real communication:

- **Round 0:** specialists post independent opinions.
- **Round 1:** each specialist reads its peers and `revise()`s — moving toward
  consensus while keeping its own evidence (heuristic), or re-reasoning given
  peers (LLM). Abstaining (no-data) agents stay out of the vote.

## 3. Coordination & aggregation

The **`CoordinatorAgent`** (`orchestrator.py`) supports pluggable strategies:

- `weighted` — reliability × confidence weighted mean of risk scores.
- `majority` — weighted vote share of `Positive` decisions.
- `bayesian` — confidence-weighted log-odds (logit) pooling.

It then folds in the critic's `adjustment` and reports the risk level, computing
`final_score`, `final_decision` (Positive iff score > 0.5; ties → Negative),
`agreement_level`, and an `explanation`. An optional LLM coordinator
(`--llm-coordinator`) can synthesize the final decision instead.

A safety rule short-circuits to a clear *"insufficient information"* result when
no specialist had usable data.

## 4. Software engineering best practices

- **Version control / forking:** this project is a fork-style adaptation of the
  CrewAI [`medical_diagnosis`](https://github.com/RoxanaSz/medical_diagnosis)
  template. Recommended workflow: fork → feature branch → PR, with `CHANGELOG.md`
  capturing notable changes.
- **Tests:** `tests/` holds a pytest suite (schemas, agents, blackboard,
  meta-agents, strategies, protocols, intake, end-to-end). Run `pytest`.
- **Separation of concerns:** data, schemas, agents, communication, protocols,
  aggregation, evaluation and CLI each live in their own module.
- **Typed contracts:** Pydantic models validate every agent/coordinator output.
- **Reproducibility:** dataset is cached locally; evaluation is deterministic in
  heuristic mode and records the full config in `evaluation_report.json`.

## Module map

| Module | Responsibility |
|--------|----------------|
| `data_loader.py` | download/cache + parse UCI dataset, feature metadata |
| `schemas.py` | Pydantic contracts (agent, coordinator, critic, risk) |
| `agents.py` | feature specialists (diagnose + debate revise) |
| `meta_agents.py` | critic + risk-stratification agents |
| `blackboard.py` | shared communication memory |
| `protocols.py` | single / debate interaction protocols |
| `orchestrator.py` | coordinator + aggregation strategies |
| `intake.py` | natural-language complaint → features |
| `crew.py` | wires everything into one pipeline |
| `evaluate.py` | dataset metrics + report |
| `report.py` | console / markdown / history rendering |
| `main.py` | CLI (evaluate / diagnose / chat) |
