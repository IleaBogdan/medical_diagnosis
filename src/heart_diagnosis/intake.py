"""Natural-language intake.

Turns a free-text patient complaint ("I'm a 58 year old man with crushing chest
pain when I climb stairs...") into the structured feature encoding used by the
dataset, so the specialist agents can reason over it. Unknown features become
``None`` and the specialists lower their confidence accordingly.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from .data_loader import FEATURES, PREDICTOR_COLUMNS
from .llm import OpenAIJSON

_FEATURE_ENCODING_HELP = """
Feature encodings (use exactly these numeric codes, or null if not stated):
- age: number, years
- sex: 0 = female, 1 = male
- cp (chest pain type): 1 = typical angina, 2 = atypical angina,
    3 = non-anginal pain, 4 = asymptomatic
- trestbps: resting blood pressure in mm Hg
- chol: serum cholesterol in mg/dl
- fbs: fasting blood sugar > 120 mg/dl, 0 = no, 1 = yes
- restecg: 0 = normal, 1 = ST-T abnormality, 2 = left ventricular hypertrophy
- thalach: maximum heart rate achieved (bpm)
- exang: exercise-induced angina, 0 = no, 1 = yes
- oldpeak: ST depression induced by exercise (number)
- slope: 1 = upsloping, 2 = flat, 3 = downsloping
- ca: number of major vessels (0-3)
- thal: 3 = normal, 6 = fixed defect, 7 = reversible defect
""".strip()


def empty_features() -> Dict[str, Optional[float]]:
    return {c: None for c in PREDICTOR_COLUMNS}


def parse_complaint_llm(text: str, llm: OpenAIJSON) -> Dict[str, Optional[float]]:
    system = (
        "You are a clinical intake assistant. Extract structured cardiovascular "
        "features from the patient's free-text description. Only fill a field if "
        "the text clearly supports it; otherwise use null. Never invent values."
    )
    user = (
        f"Patient description:\n\"{text}\"\n\n"
        f"{_FEATURE_ENCODING_HELP}\n\n"
        "Return ONLY a JSON object with keys: "
        f"{', '.join(PREDICTOR_COLUMNS)}. Use null for anything not stated."
    )
    data = llm.complete_json(system, user)
    out = empty_features()
    for k in PREDICTOR_COLUMNS:
        v = data.get(k)
        if v is None:
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = None
    return out


def parse_complaint_heuristic(text: str) -> Dict[str, Optional[float]]:
    """Very small keyword/number extractor used when no LLM is available."""
    out = empty_features()
    t = text.lower()

    m = re.search(r"(\d{1,3})\s*(?:years|year|yo|y/o|yrs)", t)
    if m:
        out["age"] = float(m.group(1))

    if re.search(r"\b(male|man|m,|gentleman|he|his)\b", t):
        out["sex"] = 1.0
    if re.search(r"\b(female|woman|f,|lady|she|her)\b", t):
        out["sex"] = 0.0

    if "chest pain" in t or "angina" in t or "chest tightness" in t or "chest pressure" in t:
        if "exertion" in t or "exercise" in t or "stairs" in t or "walking" in t:
            out["cp"] = 1.0  # typical angina
        else:
            out["cp"] = 2.0  # atypical angina
    if "no chest pain" in t or "asymptomatic" in t:
        out["cp"] = 4.0

    if "angina" in t and ("exercise" in t or "exertion" in t or "stairs" in t):
        out["exang"] = 1.0

    m = re.search(r"(?:bp|blood pressure)[^\d]*(\d{2,3})", t)
    if m:
        out["trestbps"] = float(m.group(1))
    m = re.search(r"cholesterol[^\d]*(\d{2,3})", t)
    if m:
        out["chol"] = float(m.group(1))
    if "diabet" in t or "high blood sugar" in t:
        out["fbs"] = 1.0

    return out


def parse_complaint(
    text: str, engine: str = "heuristic", llm: Optional[OpenAIJSON] = None
) -> Dict[str, Optional[float]]:
    if engine == "openai" and llm is not None:
        try:
            return parse_complaint_llm(text, llm)
        except Exception:
            return parse_complaint_heuristic(text)
    return parse_complaint_heuristic(text)


def describe_features(features: Dict[str, Optional[float]]) -> str:
    """Pretty one-per-line rendering of the (possibly partial) feature set."""
    lines = []
    for f in PREDICTOR_COLUMNS:
        lines.append("  - " + FEATURES[f].render(features.get(f)))
    return "\n".join(lines)
