# Multi-Agent Heart Disease Diagnosis

A multi-agent medical-diagnosis system mapped onto the
[UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).
It adapts the CrewAI [`medical_diagnosis`](https://github.com/RoxanaSz/medical_diagnosis)
template to the cardiovascular domain.

Each **specialist agent** is mapped to a clinically meaningful **subset of the
dataset features** (a real-world clinical role) and returns a structured JSON
diagnosis. A **coordinator agent** then aggregates all agent outputs using
weighted-confidence voting and produces the final diagnosis.

## How it maps to the assignment

**Domain:** cardiovascular disease diagnosis.
**Target:** presence (`Positive`) or absence (`Negative`) of heart disease
(`num > 0` in the dataset).

### Agents → feature subsets (clinical roles)

| Agent | Clinical role | Features used |
|-------|---------------|---------------|
| `cardiology_agent`   | Interventional Cardiologist (ischemia/angina) | `cp`, `exang`, `oldpeak`, `slope`, `thalach` |
| `imaging_agent`      | Nuclear Cardiology / Imaging Specialist       | `thal`, `ca`, `restecg` |
| `metabolic_agent`    | Preventive Cardiology / Metabolic Risk        | `chol`, `fbs`, `trestbps` |
| `demographics_agent` | Epidemiology / Demographic Risk               | `age`, `sex` |
| `coordinator_agent`  | Attending Physician (aggregation)             | aggregates all of the above |

### Structured agent output

```json
{
  "agent_name": "string",
  "features_used": ["feature1", "feature2"],
  "risk_score": 0.0,
  "confidence": 0.0,
  "decision": "Positive",
  "evidence": { "feature_name": "brief interpretation" },
  "explanation": "short clinical reasoning summary"
}
```

### Coordinator output

```json
{
  "final_score": 0.0,
  "final_decision": "Positive",
  "agreement_level": 0.0,
  "explanation": "short explanation"
}
```

The coordinator weights each specialist by `reliability * confidence`, computes
a risk score with the chosen strategy, decides `Positive` when
`final_score > 0.5` (ties → `Negative`), and reports the weighted agreement.

## Extended architecture (v0.2)

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full picture. Highlights:

- **Extra agents** beyond the feature specialists:
  - `CriticAgent` – devil's advocate; challenges the panel and adjusts the score
    to avoid premature diagnostic closure.
  - `RiskStratificationAgent` – triage; assigns `risk_if_missed`
    (Low/Moderate/High) and recommends actions.
- **Inter-agent communication** via a shared `Blackboard` and a two-round
  **debate protocol** (`--protocol debate`) where specialists revise after
  reading their peers.
- **Decision-making strategies** (`--strategy weighted|majority|bayesian`).
- **Tests** under `tests/` (run `pytest`), plus `ARCHITECTURE.md` and
  `CHANGELOG.md`.

These are controllable per run:

```bash
PYTHONPATH=src python -m heart_diagnosis.main diagnose --row 2 \
    --protocol debate --strategy bayesian
PYTHONPATH=src python -m heart_diagnosis.main evaluate --engine heuristic \
    --strategy majority --no-critic
```

## Two engines

- **`openai`** – every agent is an LLM persona that reads only its slice of the
  patient data and returns the JSON schema above (OpenAI JSON mode).
- **`heuristic`** – a deterministic clinical scoring model (no API key needed) so
  the project always runs offline. It is used automatically when no key is set,
  and as a per-agent fallback if an LLM call fails.

`--engine auto` (default) picks `openai` when `OPENAI_API_KEY` is present,
otherwise `heuristic`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional, for the LLM doctors:
cp .env.example .env   # then put your real OPENAI_API_KEY inside
```

Run everything with `PYTHONPATH=src` (or `pip install -e .` to get the
`heart-diagnosis` command).

## Quick start with `./run.sh`

The easiest way to run everything (handles the venv + `PYTHONPATH` for you):

```bash
./run.sh                              # interactive chat (describe your symptoms)
./run.sh chat --text "58yo man, chest pain on stairs, cholesterol 290"
./run.sh diagnose --row 1             # diagnose a dataset patient
./run.sh diagnose --row 2 --protocol debate --strategy bayesian
./run.sh evaluate --engine heuristic  # evaluate on the UCI dataset
./run.sh test                         # run the pytest suite
```

Any arguments after `./run.sh` are forwarded to the CLI.

## Usage

### 1. Evaluate on the real dataset

```bash
PYTHONPATH=src python -m heart_diagnosis.main evaluate --engine heuristic
PYTHONPATH=src python -m heart_diagnosis.main evaluate --engine openai --sample 30
```

Writes `evaluation_report.json` with accuracy, precision, recall, F1, ROC AUC,
confusion matrix and per-patient results. The offline heuristic engine scores
~0.77 accuracy / 0.90 ROC AUC on all 303 patients; the LLM engine typically
does better.

### 2. Diagnose one patient (structured input)

```bash
# from a dataset row (prints the true label too)
PYTHONPATH=src python -m heart_diagnosis.main diagnose --row 1

# from explicit values
PYTHONPATH=src python -m heart_diagnosis.main diagnose \
    --age 67 --sex 1 --cp 4 --trestbps 160 --chol 286 --ca 3 --thal 3

# from a JSON file of features
PYTHONPATH=src python -m heart_diagnosis.main diagnose --patient-json my_patient.json
```

### 3. Describe your problem in plain language (LLM doctors)

This is the "talk to the doctors" mode. You type what you feel; an LLM intake
step turns it into structured features, the specialist agents interpret them,
and the coordinator gives a verdict.

```bash
# interactive
PYTHONPATH=src python -m heart_diagnosis.main chat --engine openai

# one-shot
PYTHONPATH=src python -m heart_diagnosis.main chat --engine openai \
    --text "I'm a 58 year old man with crushing chest pain when I climb stairs, cholesterol 290, diabetic."
```

Without an API key, `chat` still runs on the heuristic engine with a simple
keyword/number extractor (limited but functional).

> **Disclaimer:** this is an educational project, not a medical device. Do not
> use it for real clinical decisions.

## Project layout

```
src/heart_diagnosis/
├── config/
│   ├── agents.yaml      # agent roles + feature subsets + reliability
│   └── tasks.yaml       # per-agent task instructions
├── data_loader.py       # download/cache + parse UCI dataset, feature metadata
├── schemas.py           # Pydantic models for the JSON contracts
├── llm.py               # OpenAI/OpenRouter JSON-mode wrapper + engine resolution
├── agents.py            # specialist agents (diagnose + debate revise)
├── meta_agents.py       # critic + risk-stratification agents
├── blackboard.py        # shared inter-agent communication memory
├── protocols.py         # single / debate interaction protocols
├── orchestrator.py      # coordinator + aggregation strategies
├── intake.py            # natural-language complaint -> structured features
├── crew.py              # assembles all agents into a pipeline
├── evaluate.py          # dataset evaluation + metrics
├── report.py            # console / markdown / history rendering
└── main.py              # CLI (evaluate | diagnose | chat)
tests/                   # pytest suite
ARCHITECTURE.md          # extended design
CHANGELOG.md             # version history
```

## Tests

```bash
pytest
```

## Outputs

- `evaluation_report.json` – dataset metrics + per-patient predictions
- `diagnosis_output.json` – full structured output for the **latest** diagnosis (overwritten each run)
- `heart_disease_report.md` – human-readable **latest** single-diagnosis report (overwritten each run)

Every `diagnose` and `chat` run also **appends** the question + answer to a
persistent history (never overwritten):

- `diagnosis_history.jsonl` – one JSON record per run (machine-readable log)
- `diagnosis_history.md` – the same history in readable form
