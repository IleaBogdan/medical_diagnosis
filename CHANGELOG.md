# Changelog

All notable changes to this project are documented here.

## [0.2.0] - Architecture & functionality extension

### Added
- **New agents**
  - `CriticAgent` — devil's advocate that challenges the panel and guards
    against premature diagnostic closure (emits a bounded score adjustment).
  - `RiskStratificationAgent` — triage agent that assigns `risk_if_missed`
    (Low/Moderate/High) and recommends actions.
- **Inter-agent communication**
  - `Blackboard` shared memory; agents publish/read structured opinions.
  - **Debate protocol** (`--protocol debate`): a second round where each
    specialist revises after reading its peers.
- **Decision-making mechanisms**
  - Pluggable aggregation strategies (`--strategy weighted|majority|bayesian`).
- **Software engineering**
  - pytest test suite under `tests/` (27 tests).
  - `ARCHITECTURE.md` and this `CHANGELOG.md`.
  - CLI flags `--protocol`, `--strategy`, `--no-critic`, `--no-risk` across all
    commands; evaluation report now records the full configuration.

### Changed
- Coordinator now folds in critic adjustment and risk level; ties default to
  `Negative` (a coin flip is not a diagnosis).

## [0.1.0] - Initial system

### Added
- Multi-agent heart-disease diagnosis on the UCI Heart Disease dataset.
- Four feature-specialist agents mapped to clinical feature subsets.
- Coordinator with weighted-confidence voting and structured JSON outputs.
- OpenAI engine + offline heuristic fallback; OpenRouter auto-detection.
- Natural-language intake (`chat`), single-patient `diagnose`, dataset
  `evaluate`, persistent Q&A history, and reports.
