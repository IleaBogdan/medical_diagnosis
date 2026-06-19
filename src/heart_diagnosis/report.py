"""Human-readable rendering of a diagnosis result."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from .crew import DiagnosisResult
from .intake import describe_features


def format_console(
    result: DiagnosisResult,
    features: Optional[dict] = None,
    complaint: Optional[str] = None,
) -> str:
    lines = []
    lines.append("=" * 64)
    lines.append("  MULTI-AGENT HEART DISEASE DIAGNOSIS")
    lines.append("=" * 64)
    if complaint:
        lines.append(f'Patient says: "{complaint}"')
    if features is not None:
        lines.append("\nInterpreted clinical features:")
        lines.append(describe_features(features))
    lines.append("\n----- Specialist panel -----")
    for a in result.agent_outputs:
        lines.append(f"\n  {a.agent_name}  ({', '.join(a.features_used)})")
        lines.append(f"    decision   : {a.decision}")
        lines.append(f"    risk_score : {a.risk_score:.2f}   confidence: {a.confidence:.2f}")
        lines.append(f"    reasoning  : {a.explanation}")
        for feat, interp in a.evidence.items():
            lines.append(f"      - {feat}: {interp}")
    if result.critic is not None:
        c = result.critic
        lines.append("\n----- Critic (devil's advocate) -----")
        lines.append(f"  agrees with panel : {c.agrees_with_panel}")
        lines.append(f"  score adjustment  : {c.adjustment:+.2f}")
        lines.append(f"  concern           : {c.concern or 'none'}")
        lines.append(f"  reasoning         : {c.explanation}")
    if result.risk is not None:
        r = result.risk
        lines.append("\n----- Risk stratification (triage) -----")
        lines.append(f"  risk if missed    : {r.risk_if_missed}")
        lines.append(f"  recommended       : {', '.join(r.recommended_actions)}")
        lines.append(f"  reasoning         : {r.explanation}")
    f = result.final
    lines.append("\n----- Coordinator (attending physician) -----")
    lines.append(f"  protocol/strategy : {result.protocol} / {result.strategy}")
    lines.append(f"  FINAL DECISION : {f.final_decision}")
    lines.append(f"  final_score    : {f.final_score:.2f}")
    lines.append(f"  agreement      : {f.agreement_level:.2f}")
    lines.append(f"  explanation    : {f.explanation}")
    lines.append("=" * 64)
    return "\n".join(lines)


def format_markdown(
    result: DiagnosisResult,
    features: Optional[dict] = None,
    complaint: Optional[str] = None,
) -> str:
    md = ["# Heart Disease Diagnosis Report", ""]
    md.append(f"**Engine:** {result.engine}")
    if complaint:
        md.append(f"\n**Patient description:** {complaint}")
    if features is not None:
        md.append("\n## Interpreted features\n")
        md.append("```\n" + describe_features(features) + "\n```")
    md.append("\n## Specialist agent outputs\n")
    for a in result.agent_outputs:
        md.append(f"### {a.agent_name}\n")
        md.append("```json")
        md.append(json.dumps(a.model_dump(), indent=2))
        md.append("```\n")
    if result.critic is not None:
        md.append("## Critic (devil's advocate)\n")
        md.append("```json")
        md.append(json.dumps(result.critic.model_dump(), indent=2))
        md.append("```\n")
    if result.risk is not None:
        md.append("## Risk stratification (triage)\n")
        md.append("```json")
        md.append(json.dumps(result.risk.model_dump(), indent=2))
        md.append("```\n")
    md.append(f"## Coordinator final decision (protocol={result.protocol}, "
              f"strategy={result.strategy})\n")
    md.append("```json")
    md.append(json.dumps(result.final.model_dump(), indent=2))
    md.append("```")
    return "\n".join(md)


def save_reports(
    result: DiagnosisResult,
    features: Optional[dict] = None,
    complaint: Optional[str] = None,
    json_path: str = "diagnosis_output.json",
    md_path: str = "heart_disease_report.md",
) -> None:
    payload = result.to_dict()
    if complaint:
        payload["complaint"] = complaint
    if features is not None:
        payload["interpreted_features"] = features
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(format_markdown(result, features, complaint))


def _question_text(features: Optional[dict], complaint: Optional[str]) -> str:
    if complaint:
        return complaint
    if features is not None:
        known = {k: v for k, v in features.items() if v is not None}
        if known:
            return ", ".join(f"{k}={v:g}" for k, v in known.items())
    return "(no input provided)"


def append_history(
    result: DiagnosisResult,
    features: Optional[dict] = None,
    complaint: Optional[str] = None,
    jsonl_path: str = "diagnosis_history.jsonl",
    md_path: str = "diagnosis_history.md",
) -> None:
    """Append this question + answer to a persistent history (never overwrites)."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    question = _question_text(features, complaint)
    final = result.final

    record = {
        "timestamp": timestamp,
        "engine": result.engine,
        "question": question,
        "interpreted_features": features,
        "agent_outputs": [a.model_dump() for a in result.agent_outputs],
        "final": final.model_dump(),
    }
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    with open(md_path, "a", encoding="utf-8") as f:
        f.write(f"## {timestamp}  (engine: {result.engine})\n\n")
        f.write(f"**Q:** {question}\n\n")
        f.write(
            f"**A:** {final.final_decision} "
            f"(final_score={final.final_score:.2f}, "
            f"agreement={final.agreement_level:.2f})\n\n"
        )
        f.write(f"{final.explanation}\n\n")
        f.write("<details><summary>Specialist panel</summary>\n\n")
        for a in result.agent_outputs:
            f.write(
                f"- **{a.agent_name}**: {a.decision} "
                f"(risk={a.risk_score:.2f}, conf={a.confidence:.2f}) - "
                f"{a.explanation}\n"
            )
        f.write("\n</details>\n\n---\n\n")
