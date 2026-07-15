"""Anthropic client and the two schema'd calls the pipeline makes.

Model choice is deliberate and asymmetric:

* ``expand_tree`` uses Opus. It runs once per run and its output is the only thing
  every later stage can filter -- a bad pain translation cannot be recovered by any
  amount of downstream cleverness. This is the one place a frontier model pays for
  itself on a per-run basis (~$0.26).
* ``score_batch`` uses Haiku across ~30-45 calls. Cheap, but the plan requires baking
  it off against Sonnet on labelled data before trusting it: pain-vs-mention and
  genuine-asker-vs-vendor are exactly the judgement calls a small model fumbles.

Both use ``output_config.format`` (structured outputs) rather than prompt-begging for
JSON. Assistant prefill -- the old way to force a JSON shape -- returns 400 on Opus 4.8
and Haiku 4.5, so it is not an option even as a fallback.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import anthropic

from core.config import settings

log = logging.getLogger(__name__)

PROMPTS = Path(__file__).parent / "prompts"

_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings().anthropic_api_key or None)
    return _client


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
    """One Opus call: product description -> tree of pains in the sufferer's words."""
    user = f"<solution>\n{input_text.strip()}\n</solution>"
    if icp:
        user += f"\n\n<icp>\n{icp.strip()}\n</icp>"

    response = client().messages.create(
        model=settings().expand_model,
        max_tokens=16_000,
        system=prompt("expand"),
        thinking={"type": "adaptive"},
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": TREE_SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"expand_tree refused: {response.stop_details}")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


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
    """Score up to ~10 candidate/leaf pairs in one call.

    No prompt caching here on purpose: the rubric is ~800 tokens, well under the
    4,096-token minimum cacheable prefix for Haiku 4.5. Adding cache_control would
    silently no-op (cache_creation_input_tokens: 0, no error) and mislead anyone
    reading the code into thinking caching was handled.
    """
    response = client().messages.create(
        model=settings().score_model,
        max_tokens=8_000,
        system=prompt("score"),
        output_config={"format": {"type": "json_schema", "schema": SCORE_SCHEMA}},
        messages=[{"role": "user", "content": pairs_payload}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"score_batch refused: {response.stop_details}")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)["results"]
