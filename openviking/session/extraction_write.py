# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Freeze an extraction payload into one Influence View.

Nothing in this module calls a model or constructs a model client. A caller
that already holds blocks, claims, coverage, and retrieval subgraphs uses
:func:`freeze_extracted_view`. That path freezes ``construction_mode=adaptive``
and lets ``HighSeverityViolation`` propagate. It does not write a file and it
does not switch to the deterministic fallback.

:func:`degrade_to_fallback_view` is the explicit alternative. The caller
invokes it after catching ``HighSeverityViolation`` when they want the ledger
fallback (``construction_mode=fallback``).

:func:`extraction_index_builder` adapts an injected provider to the trajectory
index ``Builder`` shape ``(session_id, task_run_id, outcome_status)``. The
commit hook passes that builder only when a provider was injected.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

from openviking.session.influence_view import (
    CONSTRUCTION_MODE_ADAPTIVE,
    PURPOSE_POST_RUN_INDEX,
    CoverageManifest,
    InfluenceBlock,
    InfluenceClaim,
    InfluenceViewDraft,
    InfluenceViewRevision,
    RetrievalSubgraph,
    build_minimal_fallback_view,
)

# ``subgraph_id`` is handed to a later model with a diagnostic hit.
# ``RetrievalSubgraph`` already says the id carries none of these.
_EMBEDDED_ID_FIELDS = (
    "session_id",
    "view_id",
    "task_run_id",
    "revision_id",
)


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty str")
    return value


def _reject_embedded_subgraph_ids(
    subgraphs: Sequence[RetrievalSubgraph],
    identities: Mapping[str, str],
) -> None:
    """Refuse a subgraph id that spells out a view identity.

    Match is a case-sensitive substring of the full identity string. The
    check runs before ``freeze`` so a rejected id never becomes a revision.
    """
    for index, subgraph in enumerate(subgraphs):
        if not isinstance(subgraph, RetrievalSubgraph):
            raise ValueError(f"retrieval_subgraphs[{index}] must be a RetrievalSubgraph")
        for field in _EMBEDDED_ID_FIELDS:
            secret = identities[field]
            if secret in subgraph.subgraph_id:
                raise ValueError(f"subgraph_id {subgraph.subgraph_id!r} must not embed {field}")


class ExtractionPayload:
    """Blocks, claims, coverage, and retrieval subgraphs from an extraction.

    There is no ``subtask_id`` field. Constructing this object does not call
    a model.
    """

    __slots__ = ("blocks", "claims", "coverage", "retrieval_subgraphs")

    def __init__(
        self,
        blocks: Sequence[InfluenceBlock],
        claims: Sequence[InfluenceClaim],
        coverage: CoverageManifest,
        retrieval_subgraphs: Sequence[RetrievalSubgraph] = (),
    ) -> None:
        block_items = tuple(blocks)
        claim_items = tuple(claims)
        subgraph_items = tuple(retrieval_subgraphs)
        for index, block in enumerate(block_items):
            if not isinstance(block, InfluenceBlock):
                raise TypeError(f"blocks[{index}] must be an InfluenceBlock")
        for index, claim in enumerate(claim_items):
            if not isinstance(claim, InfluenceClaim):
                raise TypeError(f"claims[{index}] must be an InfluenceClaim")
        if not isinstance(coverage, CoverageManifest):
            raise TypeError("coverage must be a CoverageManifest")
        for index, subgraph in enumerate(subgraph_items):
            if not isinstance(subgraph, RetrievalSubgraph):
                raise TypeError(f"retrieval_subgraphs[{index}] must be a RetrievalSubgraph")
        self.blocks = block_items
        self.claims = claim_items
        self.coverage = coverage
        self.retrieval_subgraphs = subgraph_items


ExtractionProvider = Callable[[str, str, str], ExtractionPayload]


def freeze_extracted_view(
    *,
    session_id: str,
    task_run_id: str,
    view_id: str,
    revision_id: str,
    frozen_at: str,
    blocks: Sequence[InfluenceBlock],
    claims: Sequence[InfluenceClaim],
    coverage: CoverageManifest,
    retrieval_subgraphs: Sequence[RetrievalSubgraph] = (),
) -> InfluenceViewRevision:
    """Validate one extraction payload and freeze a single adaptive revision.

    ``HighSeverityViolation`` propagates. This function does not catch it,
    does not write a partial revision, and does not call
    :func:`build_minimal_fallback_view`. ``construction_mode`` is ``adaptive``.
    """
    identities = {
        "session_id": _require_str(session_id, "session_id"),
        "task_run_id": _require_str(task_run_id, "task_run_id"),
        "view_id": _require_str(view_id, "view_id"),
        "revision_id": _require_str(revision_id, "revision_id"),
    }
    stamp = _require_str(frozen_at, "frozen_at")
    if not isinstance(coverage, CoverageManifest):
        raise TypeError("coverage must be a CoverageManifest")
    subgraphs = tuple(retrieval_subgraphs)
    _reject_embedded_subgraph_ids(subgraphs, identities)
    draft = InfluenceViewDraft(
        view_id=identities["view_id"],
        purpose=PURPOSE_POST_RUN_INDEX,
        session_id=identities["session_id"],
        task_run_id=identities["task_run_id"],
        coverage=coverage,
        blocks=tuple(blocks),
        claims=tuple(claims),
        retrieval_subgraphs=subgraphs,
    )
    return draft.freeze(
        construction_mode=CONSTRUCTION_MODE_ADAPTIVE,
        revision_id=identities["revision_id"],
        frozen_at=stamp,
        enforce_checks=True,
    )


def degrade_to_fallback_view(
    session: Mapping[str, Any],
    *,
    clock: Callable[[], datetime] | None = None,
) -> InfluenceViewRevision:
    """Caller-chosen degradation after a refused extraction freeze.

    :func:`freeze_extracted_view` does not call this. The returned revision
    uses ``construction_mode=fallback`` and has no retrieval subgraphs.
    """
    return build_minimal_fallback_view(session, clock=clock)


def extraction_index_builder(
    provider: ExtractionProvider,
    *,
    frozen_at: str,
) -> Callable[[str, str, str], InfluenceViewRevision]:
    """Return a trajectory-index ``Builder`` that freezes ``provider`` output.

    The provider is called as ``provider(session_id, task_run_id, outcome_status)``
    and must return an :class:`ExtractionPayload`. View identity matches the
    post-run commit hook: ``view_id`` is ``{session_id}:post_run_index`` and
    ``revision_id`` is ``post-run:{session_id}:{task_run_id}``. ``frozen_at``
    is the commit timestamp the hook already stored on the job.

    A refused freeze propagates. This builder does not degrade to fallback.
    """
    if not callable(provider):
        raise TypeError("extraction provider must be callable")
    stamp = _require_str(frozen_at, "frozen_at")

    def builder(session_id: str, task_run_id: str, outcome_status: str) -> InfluenceViewRevision:
        payload = provider(session_id, task_run_id, outcome_status)
        if not isinstance(payload, ExtractionPayload):
            raise TypeError("extraction provider must return an ExtractionPayload")
        return freeze_extracted_view(
            session_id=session_id,
            task_run_id=task_run_id,
            view_id=f"{session_id}:{PURPOSE_POST_RUN_INDEX}",
            revision_id=f"post-run:{session_id}:{task_run_id}",
            frozen_at=stamp,
            blocks=payload.blocks,
            claims=payload.claims,
            coverage=payload.coverage,
            retrieval_subgraphs=payload.retrieval_subgraphs,
        )

    return builder
