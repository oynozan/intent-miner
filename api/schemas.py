from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# Documented value sets. These are plain strings in the database (a run can fail into
# an unexpected state, and platforms are added over time), so they are not enforced as
# enums on the response models -- but they are the values a client should expect, so
# they are documented on the fields and surfaced here for the OpenAPI description.
RUN_STATUSES = "pending, expanding, discovering, fetching, prefiltering, scoring, done, failed"
PLATFORMS = "quora, linkedin"  # reddit is deferred -- see the project README
NODE_KINDS = "root, branch, leaf"


class CreateRun(BaseModel):
    input_text: str = Field(
        min_length=20,
        description=(
            "Describe the solution in prose, not keywords. The pipeline translates this "
            "product description into the pain language its customers actually use, so a "
            "full sentence works far better than a feature list."
        ),
        examples=[
            "An API that removes the background from a video automatically, without a "
            "green screen, using AI segmentation. Handles hair and motion blur."
        ],
    )
    icp: str | None = Field(
        default=None,
        description=(
            "The ideal customer profile: who has this problem. Optional, but if you know "
            "it, it biases every query toward how that person talks about the pain."
        ),
        examples=["solo video editor doing client work"],
    )


class RunCreated(BaseModel):
    run_id: str = Field(
        description="Poll this id at GET /runs/{run_id} to follow the run.",
        examples=["ff95e261-9667-4451-a7bf-f01a97b2c544"],
    )


class RunStatus(BaseModel):
    run_id: str = Field(examples=["ff95e261-9667-4451-a7bf-f01a97b2c544"])
    status: str = Field(
        description=(
            f"Current stage. One of: {RUN_STATUSES}. The run advances pending -> "
            "expanding -> discovering -> fetching -> prefiltering -> scoring -> done. "
            "`failed` is terminal; check `stats` and the worker logs for the cause."
        ),
        examples=["done"],
    )
    stats: dict[str, Any] = Field(
        description=(
            "Per-stage metrics, accumulated as the run progresses. Keys appear as their "
            "stage completes: `leaves`, `queries_compiled`, `queries_after_dedupe`, "
            "`candidates_discovered`, `candidates_fetched`, `dropped_by_negative_terms`, "
            "`pairs_surviving`, `kill_rate`, `scored`, `leads`, `rejected_not_seeking`, "
            "`rejected_not_actionable`. `kill_rate` is the prefilter's drop fraction; "
            "`leads` is how many scored posts survived the is_seeking/actionable gates."
        ),
        examples=[
            {
                "leaves": 16,
                "queries_compiled": 160,
                "queries_after_dedupe": 148,
                "candidates_discovered": 2013,
                "candidates_fetched": 1442,
                "dropped_by_negative_terms": 88,
                "pairs_surviving": 312,
                "kill_rate": 0.85,
                "scored": 312,
                "leads": 41,
                "rejected_not_seeking": 214,
                "rejected_not_actionable": 57,
            }
        ],
    )
    created_at: datetime
    finished_at: datetime | None = Field(
        default=None, description="Set when the run reaches `done` or `failed`; null while running."
    )


class Node(BaseModel):
    """One node of the pain tree: the root, a branch (a job-to-be-done), or a leaf (a
    specific pain, with the phrasings and queries derived from it)."""

    id: str = Field(examples=["3f2c1b90-1a2b-4c3d-9e8f-aabbccddeeff"])
    parent_id: str | None = Field(default=None, description="Null for the root node.")
    depth: int = Field(description="0 root, 1 branch, 2 leaf.", examples=[2])
    kind: str = Field(description=f"One of: {NODE_KINDS}.", examples=["leaf"])
    label: str = Field(examples=["No green screen available"])
    description: str = Field(
        description="One sentence in the sufferer's own words. This is the text the prefilter embeds.",
        examples=["I can't get a clean cutout without a green screen and I'm doing it frame by frame"],
    )
    pain_phrases: list[str] = Field(
        default=[],
        description="Verbatim phrasings a frustrated person would type. Empty on non-leaf nodes.",
        examples=[["cut myself out of a video without a green screen", "wasted 3 hours rotoscoping"]],
    )
    negative_terms: list[str] = Field(
        default=[],
        description="Words that signal the wrong meaning of an ambiguous phrase; used to drop false matches.",
        examples=[["photoshop", "still image"]],
    )
    icp_hint: str | None = Field(default=None, examples=["solo video editor doing client work"])


class Result(BaseModel):
    """One ranked post for one leaf.

    `final` is the composite rank: `pain_match * icp_match * recency_decay *
    log1p(engagement)`, but with a hard gate -- if `is_seeking` is false (a vendor ad, a
    think-piece) or `actionable` is false (a dead or saturated thread), `final` is
    exactly 0.0 regardless of how well the pain matched.
    """

    url: str = Field(examples=["https://quora.com/how-do-i-remove-a-video-background-without-a-green-screen"])
    platform: str = Field(description=f"One of: {PLATFORMS}.", examples=["quora"])
    title: str | None = None
    posted_at: datetime | None = Field(
        default=None, description="Best available post date. Often null on Quora, which hides it from logged-out clients."
    )
    engagement: int = Field(description="Platform-specific: upvotes (Quora) or comment/reaction count (LinkedIn).", examples=[132])
    final: float = Field(description="Composite rank score. 0.0 means gated out (not a lead), not merely weak.", examples=[0.6412])
    is_seeking: bool = Field(description="True = someone wanting the solution. False = selling, announcing, or opining.", examples=[True])
    pain_match: int = Field(description="0-100: how closely the post's actual problem matches this leaf.", examples=[88])
    icp_match: int = Field(description="0-100: how well the author matches the ICP. 50 when unknown.", examples=[70])
    actionable: bool = Field(description="True if a reply now could still be useful (recent, thinly answered, a real person).", examples=[True])
    reason: str | None = Field(default=None, examples=["Asks for a green-screen-free cutout for client work, tried rotoscoping and gave up"])
    reply_angle: str | None = Field(default=None, description="How a useful reply would open. Empty when the post is not a lead.")
    node_id: str = Field(examples=["3f2c1b90-1a2b-4c3d-9e8f-aabbccddeeff"])
    node_label: str = Field(examples=["No green screen available"])


class LeafResults(BaseModel):
    """Ranked results for one leaf.

    Grouping is the point, not a convenience: which leaf converts tells you where the
    demand actually is, which is worth more than any individual link. Groups are
    returned strongest-leaf-first.
    """

    node_id: str = Field(examples=["3f2c1b90-1a2b-4c3d-9e8f-aabbccddeeff"])
    node_label: str = Field(examples=["No green screen available"])
    node_description: str = Field(examples=["I can't get a clean cutout without a green screen and I'm doing it frame by frame"])
    count: int = Field(description="Number of results in this group.", examples=[7])
    results: list[Result]


class RunUrls(BaseModel):
    """Just the ranked result URLs plus the run's status.

    The minimal shape for "give me the links": `status` tells you whether the run is
    still working or `done`, and `urls` is the flat, de-duplicated, best-first list.
    """

    status: str = Field(
        description=f"The run's current stage. One of: {RUN_STATUSES}.",
        examples=["done"],
    )
    urls: list[str] = Field(
        description="Distinct result URLs, ranked best-first. Partial while the run is still scoring.",
        examples=[
            [
                "https://quora.com/how-do-i-remove-a-video-background-without-a-green-screen",
                "https://linkedin.com/posts/someone_looking-for-a-tool-to-cut-out-video-activity-123",
            ]
        ],
    )


class ErrorDetail(BaseModel):
    detail: str = Field(examples=["run not found"])
