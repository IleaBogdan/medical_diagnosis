"""LLM access layer.

Provides a thin OpenAI JSON-mode wrapper plus helpers to decide which engine to
use. If no API key is available the rest of the system falls back to a
deterministic clinical-heuristic engine (see ``agents.py``), so the project is
always runnable.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"))


def get_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")


def get_base_url() -> Optional[str]:
    """Resolve the API base URL.

    Honors OPENAI_BASE_URL / OPENAI_API_BASE. If the key looks like an
    OpenRouter key (``sk-or-...``) or OPENROUTER_API_KEY is set, default to the
    OpenRouter endpoint automatically.
    """
    base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    if base:
        return base
    key = get_api_key() or ""
    if os.getenv("OPENROUTER_API_KEY") or key.startswith("sk-or-"):
        return OPENROUTER_BASE
    return None


def default_model() -> str:
    m = os.getenv("HEART_DIAGNOSIS_MODEL")
    if m:
        return m
    # OpenRouter requires provider-prefixed model ids.
    if (get_base_url() or "").startswith("https://openrouter.ai"):
        return "openai/gpt-4o-mini"
    return "gpt-4o-mini"


DEFAULT_MODEL = default_model()


def resolve_engine(preferred: str = "auto") -> str:
    """Resolve the requested engine to a concrete one.

    ``auto`` -> ``openai`` when a key is present, otherwise ``heuristic``.
    """
    preferred = (preferred or "auto").lower()
    if preferred == "auto":
        return "openai" if has_openai_key() else "heuristic"
    if preferred == "openai" and not has_openai_key():
        raise RuntimeError(
            "Engine 'openai' requested but no API key is set. "
            "Set OPENAI_API_KEY (or OPENROUTER_API_KEY) in a .env file, "
            "or use --engine heuristic."
        )
    return preferred


def extract_json(raw: str) -> dict:
    """Best-effort extraction of a single JSON object from model text."""
    if not raw or not raw.strip():
        raise ValueError("Empty model output")
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in: {raw[:200]!r}")
    return json.loads(cleaned[start : end + 1])


class OpenAIJSON:
    """Minimal OpenAI chat client that returns parsed JSON objects."""

    def __init__(self, model: Optional[str] = None, temperature: float = 0.2):
        from openai import OpenAI

        base_url = get_base_url()
        api_key = get_api_key()
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model or default_model()
        self.temperature = temperature

    def complete_json(self, system: str, user: str) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return extract_json(resp.choices[0].message.content)

    def complete_text(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
