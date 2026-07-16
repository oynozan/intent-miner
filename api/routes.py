from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query

from api.schemas import (
    CreateRun,
    ErrorDetail,
    LeafResults,
    Node,
    Result,
    RunCreated,
    RunStatus,
)
from pipeline import repo

router = APIRouter(tags=["runs"])

_NOT_FOUND = {404: {"model": ErrorDetail, "description": "No run exists with that id."}}

_RunId = Path(description="The run id returned by POST /runs.", examples=["ff95e261-9667-4451-a7bf-f01a97b2c544"])


@router.post(
    "/runs",
    response_model=RunCreated,
    status_code=202,
    summary="Start a run",
    response_description="The run was accepted and is now processing asynchronously.",
)
def create_run(body: CreateRun) -> RunCreated:
    """Kick off a run and return its id immediately.

    The response is **202 Accepted**, not 200: the pipeline (expand → discover → fetch
    → prefilter → score) runs asynchronously in the workers. Poll `GET /runs/{run_id}`
    to follow it, then read `GET /runs/{run_id}/results` once `status` is `done`.
    """
    from pipeline.actors import start_run

    run_id = repo.create_run(body.input_text, body.icp)
    start_run.send(run_id)
    return RunCreated(run_id=run_id)


@router.get(
    "/runs/{run_id}",
    response_model=RunStatus,
    responses=_NOT_FOUND,
    summary="Get run status and stats",
)
def get_run(run_id: str = _RunId) -> RunStatus:
    """Current stage of the run plus its per-stage `stats`.

    Poll this until `status` is `done` (or `failed`). `stats` grows as stages complete,
    so it doubles as a live progress view -- e.g. `candidates_discovered` appears after
    discovery, `kill_rate` after the prefilter, `leads` at the end.
    """
    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return RunStatus(
        run_id=str(run["id"]),
        status=run["status"],
        stats=run["stats"],
        created_at=run["created_at"],
        finished_at=run["finished_at"],
    )


@router.get(
    "/runs/{run_id}/tree",
    response_model=list[Node],
    responses=_NOT_FOUND,
    summary="Get the pain tree",
)
def get_tree(run_id: str = _RunId) -> list[Node]:
    """The tree the run expanded the solution into: root → branches (jobs-to-be-done) →
    leaves (specific pains).

    Available as soon as `expand` completes, before results exist. Useful for seeing how
    the solution was translated into pain language, and for picking a `node_id` to filter
    results by. Nodes are returned ordered by depth then label.
    """
    if not repo.get_run(run_id):
        raise HTTPException(404, "run not found")
    return [
        Node(
            id=str(n["id"]),
            parent_id=str(n["parent_id"]) if n["parent_id"] else None,
            depth=n["depth"],
            kind=n["kind"],
            label=n["label"],
            description=n["description"],
            pain_phrases=n["pain_phrases"] or [],
            negative_terms=n["negative_terms"] or [],
            icp_hint=n["icp_hint"],
        )
        for n in repo.tree(run_id)
    ]


@router.get(
    "/runs/{run_id}/results",
    response_model=list[LeafResults],
    responses=_NOT_FOUND,
    summary="Get ranked results, grouped by leaf",
)
def get_results(
    run_id: str = _RunId,
    node_id: str | None = Query(
        default=None,
        description="Restrict to one leaf (a node id from GET /runs/{run_id}/tree). Omit for all leaves.",
        examples=["3f2c1b90-1a2b-4c3d-9e8f-aabbccddeeff"],
    ),
    min_score: float = Query(
        default=0.0,
        ge=0.0,
        description=(
            "Minimum `final` score. Vendor posts and dead threads are gated to exactly "
            "0.0, so a small positive value (e.g. 0.01) filters to genuine leads."
        ),
        examples=[0.01],
    ),
) -> list[LeafResults]:
    """Ranked links, grouped by leaf, strongest leaf first.

    Best read after `status` is `done`; on an in-progress run it returns whatever has
    been scored so far. Each group is one pain (leaf); within a group, results are
    ordered by `final` descending. See the `Result` schema for what makes a post a lead
    versus a gated-out ad.
    """
    if not repo.get_run(run_id):
        raise HTTPException(404, "run not found")

    rows = repo.results(run_id, node_id=node_id, min_score=min_score)

    groups: dict[str, LeafResults] = {}
    for row in rows:
        nid = str(row["node_id"])
        if nid not in groups:
            groups[nid] = LeafResults(
                node_id=nid,
                node_label=row["node_label"],
                node_description=row["node_description"],
                count=0,
                results=[],
            )
        groups[nid].results.append(
            Result(
                url=row["url"],
                platform=row["platform"],
                title=row["title"],
                posted_at=row["posted_at"],
                engagement=row["engagement"],
                final=row["final"],
                is_seeking=row["is_seeking"],
                pain_match=row["pain_match"],
                icp_match=row["icp_match"],
                actionable=row["actionable"],
                reason=row["reason"],
                reply_angle=row["reply_angle"],
                node_id=nid,
                node_label=row["node_label"],
            )
        )
        groups[nid].count += 1

    # Strongest leaf first: which branch converts is the finding, not the individual link.
    return sorted(
        groups.values(),
        key=lambda g: max((r.final for r in g.results), default=0.0),
        reverse=True,
    )
