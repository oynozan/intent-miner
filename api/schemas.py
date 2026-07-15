from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CreateRun(BaseModel):
    input_text: str = Field(min_length=20, description="Describe the solution. Prose, not keywords.")
    icp: str | None = Field(default=None, description="Who has this problem, if you know.")


class RunCreated(BaseModel):
    run_id: str


class RunStatus(BaseModel):
    run_id: str
    status: str
    stats: dict[str, Any]
    created_at: datetime
    finished_at: datetime | None


class Node(BaseModel):
    id: str
    parent_id: str | None
    depth: int
    kind: str
    label: str
    description: str
    pain_phrases: list[str] = []
    negative_terms: list[str] = []
    icp_hint: str | None = None


class Result(BaseModel):
    url: str
    platform: str
    title: str | None
    posted_at: datetime | None
    engagement: int
    final: float
    is_seeking: bool
    pain_match: int
    icp_match: int
    actionable: bool
    reason: str | None
    reply_angle: str | None
    node_id: str
    node_label: str


class LeafResults(BaseModel):
    """Results grouped by leaf.

    Grouping is the point, not a convenience. Which branch of the tree converts tells
    you where the demand actually is, which is worth more than any individual link.
    """
    node_id: str
    node_label: str
    node_description: str
    count: int
    results: list[Result]
