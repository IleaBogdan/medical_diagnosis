#!/usr/bin/env python
"""Command-line entry point for the multi-agent heart disease diagnosis system.

Subcommands
-----------
evaluate   Run the system over the UCI Heart Disease dataset and report metrics.
diagnose   Diagnose one patient from explicit feature flags or a JSON file.
chat       Describe symptoms in plain language; the LLM doctors interpret them.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .crew import HeartDiagnosisCrew
from .data_loader import PREDICTOR_COLUMNS, load_dataset, patient_to_dict
from .intake import empty_features, parse_complaint
from .llm import OpenAIJSON, resolve_engine
from .report import append_history, format_console, save_reports


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def cmd_evaluate(args: argparse.Namespace) -> None:
    from .evaluate import evaluate

    evaluate(
        engine=args.engine,
        limit=args.limit,
        sample=args.sample,
        seed=args.seed,
        model=args.model,
        llm_coordinator=args.llm_coordinator,
        protocol=args.protocol,
        strategy=args.strategy,
        use_critic=not args.no_critic,
        use_risk=not args.no_risk,
        output_path=args.output,
    )


# --------------------------------------------------------------------------- #
# diagnose (structured input)
# --------------------------------------------------------------------------- #
def cmd_diagnose(args: argparse.Namespace) -> None:
    features = empty_features()

    if args.patient_json:
        with open(args.patient_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        for k in PREDICTOR_COLUMNS:
            if k in loaded and loaded[k] is not None:
                features[k] = float(loaded[k])
    elif args.row is not None:
        df = load_dataset()
        features = patient_to_dict(df.iloc[args.row])
        print(f"Loaded patient row {args.row} (true label: "
              f"{'Positive' if df.iloc[args.row]['target'] == 1 else 'Negative'})\n")

    for col in PREDICTOR_COLUMNS:
        val = getattr(args, col, None)
        if val is not None:
            features[col] = float(val)

    crew = HeartDiagnosisCrew(
        engine=args.engine, model=args.model, llm_coordinator=args.llm_coordinator,
        protocol=args.protocol, strategy=args.strategy,
        use_critic=not args.no_critic, use_risk=not args.no_risk,
    )
    result = crew.diagnose(features)
    print(format_console(result, features=features))
    save_reports(result, features=features)
    append_history(result, features=features)
    print("\nSaved: diagnosis_output.json, heart_disease_report.md")
    print("Appended to history: diagnosis_history.jsonl, diagnosis_history.md")


# --------------------------------------------------------------------------- #
# chat (natural-language symptoms)
# --------------------------------------------------------------------------- #
def _run_complaint(crew: HeartDiagnosisCrew, llm: Optional[OpenAIJSON], text: str):
    features = parse_complaint(text, engine=crew.engine, llm=llm)
    result = crew.diagnose(features, complaint=text)
    print(format_console(result, features=features, complaint=text))
    save_reports(result, features=features, complaint=text)
    append_history(result, features=features, complaint=text)
    print("(saved to diagnosis_history.jsonl / diagnosis_history.md)")


def cmd_chat(args: argparse.Namespace) -> None:
    engine = resolve_engine(args.engine)
    crew = HeartDiagnosisCrew(
        engine=args.engine, model=args.model, llm_coordinator=args.llm_coordinator,
        protocol=args.protocol, strategy=args.strategy,
        use_critic=not args.no_critic, use_risk=not args.no_risk,
    )
    llm = crew.llm

    if engine == "heuristic":
        print("(!) Running without an LLM (heuristic engine). Natural-language\n"
              "    parsing is limited. Set OPENAI_API_KEY for full LLM doctors.\n")

    if args.text:
        _run_complaint(crew, llm, args.text)
        return

    print("Describe what you are experiencing (symptoms, age, history...).")
    print("Type 'quit' or Ctrl-D to exit.\n")
    while True:
        try:
            text = input("You> ").strip()
        except EOFError:
            print()
            break
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            continue
        _run_complaint(crew, llm, text)
        print()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="heart-diagnosis",
        description="Multi-agent heart disease diagnosis on the UCI dataset.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--engine", choices=["auto", "openai", "heuristic"], default="auto",
        help="LLM engine. 'auto' uses OpenAI if OPENAI_API_KEY is set.",
    )
    common.add_argument("--model", default=None, help="OpenAI model name.")
    common.add_argument(
        "--llm-coordinator", action="store_true",
        help="Use an LLM (not weighted voting) for the coordinator step.",
    )
    common.add_argument(
        "--protocol", choices=["single", "debate"], default="single",
        help="Interaction protocol. 'debate' = specialists revise after seeing peers.",
    )
    common.add_argument(
        "--strategy", choices=["weighted", "majority", "bayesian"], default="weighted",
        help="Aggregation strategy used by the coordinator.",
    )
    common.add_argument(
        "--no-critic", action="store_true",
        help="Disable the devil's-advocate Critic agent.",
    )
    common.add_argument(
        "--no-risk", action="store_true",
        help="Disable the Risk-Stratification (triage) agent.",
    )

    pe = sub.add_parser("evaluate", parents=[common], help="Evaluate on the dataset.")
    pe.add_argument("--limit", type=int, default=None, help="Use first N patients.")
    pe.add_argument("--sample", type=int, default=None, help="Random sample of N patients.")
    pe.add_argument("--seed", type=int, default=42)
    pe.add_argument("--output", default="evaluation_report.json")
    pe.set_defaults(func=cmd_evaluate)

    pd_ = sub.add_parser("diagnose", parents=[common], help="Diagnose one patient.")
    pd_.add_argument("--patient-json", help="Path to a JSON file of features.")
    pd_.add_argument("--row", type=int, help="Use a row index from the dataset.")
    for col in PREDICTOR_COLUMNS:
        pd_.add_argument(f"--{col}", type=float, default=None, help=f"{col} value.")
    pd_.set_defaults(func=cmd_diagnose)

    pc = sub.add_parser("chat", parents=[common], help="Free-text symptom intake.")
    pc.add_argument("--text", help="One-shot complaint instead of interactive prompt.")
    pc.set_defaults(func=cmd_chat)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
