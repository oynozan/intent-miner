"""The two schema'd LLM calls the pipeline makes.

Both go through ``llm.providers``, which runs OpenAI (gpt-5-nano) as primary and falls
back to Anthropic. The schemas here are strict: every property is required and every
object sets ``additionalProperties: false``. That is not Anthropic-specific -- OpenAI's
strict structured-output mode requires exactly the same, so one schema serves both.

Per-provider model/effort/token choices live in ``core.config`` and are threaded
through ``params`` below. Note the two tasks are asymmetric in what they need: expand
runs once and its tree is the only thing every later stage can filter, so it gets the
higher effort and token budget; score runs ~30 times per run on a cheaper setting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.config import settings
from llm import providers

log = logging.getLogger(__name__)

PROMPTS = Path(__file__).parent / "prompts"


def prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8")


# --- expand_tree ------------------------------------------------------------------

LEAF_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "description": {
            "type": "string",
            "description": "One sentence as the sufferer would describe it. This gets embedded.",
        },
        "pain_phrases": {"type": "array", "items": {"type": "string"}},
        "negative_terms": {"type": "array", "items": {"type": "string"}},
        "icp_hint": {"type": "string"},
        "queries": {
            "type": "object",
            "properties": {
                "reddit": {"type": "array", "items": {"type": "string"}},
                "quora": {"type": "array", "items": {"type": "string"}},
                "linkedin": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["reddit", "quora", "linkedin"],
            "additionalProperties": False,
        },
    },
    "required": ["label", "description", "pain_phrases", "negative_terms", "icp_hint", "queries"],
    "additionalProperties": False,
}

TREE_SCHEMA = {
    "type": "object",
    "properties": {
        "root": {
            "type": "object",
            "properties": {"label": {"type": "string"}, "description": {"type": "string"}},
            "required": ["label", "description"],
            "additionalProperties": False,
        },
        "branches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                    "leaves": {"type": "array", "items": LEAF_SCHEMA},
                },
                "required": ["label", "description", "leaves"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["root", "branches"],
    "additionalProperties": False,
}


def expand_tree(input_text: str, icp: str | None = None) -> dict[str, Any]:
    """One LLM call: product description -> tree of pains in the sufferer's words."""
    s = settings()
    user = f"<solution>\n{input_text.strip()}\n</solution>"
    if icp:
        user += f"\n\n<icp>\n{icp.strip()}\n</icp>"

    return providers.complete_json(
        system=prompt("expand"),
        user=user,
        schema=TREE_SCHEMA,
        schema_name="pain_tree",
        params={
            "openai": {
                "model": s.openai_expand_model,
                "effort": s.openai_expand_effort,
                "max_tokens": s.openai_expand_max_tokens,
            },
            "anthropic": {"model": s.expand_model, "effort": "high", "max_tokens": 16_000},
        },
    )


# --- score_batch ------------------------------------------------------------------

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pair_id": {"type": "string"},
                    "is_seeking": {"type": "boolean"},
                    "pain_match": {"type": "integer"},
                    "icp_match": {"type": "integer"},
                    "actionable": {"type": "boolean"},
                    "reason": {"type": "string"},
                    "reply_angle": {"type": "string"},
                },
                "required": [
                    "pair_id", "is_seeking", "pain_match", "icp_match",
                    "actionable", "reason", "reply_angle",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def score_batch(pairs_payload: str) -> list[dict[str, Any]]:
    """Score up to ~10 candidate/leaf pairs in one call."""
    s = settings()
    result = providers.complete_json(
        system=prompt("score"),
        user=pairs_payload,
        schema=SCORE_SCHEMA,
        schema_name="pair_scores",
        params={
            "openai": {
                "model": s.openai_score_model,
                "effort": s.openai_score_effort,
                "max_tokens": s.openai_score_max_tokens,
            },
            "anthropic": {"model": s.score_model, "effort": "medium", "max_tokens": 8_000},
        },
    )
    return result["results"]
