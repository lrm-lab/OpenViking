# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Freeze an extraction payload into one Influence View.

Nothing in this module calls a model or constructs a model client. A caller
that already holds blocks, claims, coverage, and retrieval subgraphs uses
:func:`freeze_extracted_view`. That path freezes ``construction_mode=native_single_pass_aig``
and lets ``HighSeverityViolation`` propagate. It does not write a file and it
does not switch to the deterministic fallback.

:func:`degrade_to_fallback_view` is the explicit alternative. The caller
invokes it after catching ``HighSeverityViolation`` when they want the ledger
fallback (``construction_mode=fallback``).

:func:`extraction_index_builder` adapts an injected provider to the trajectory
index ``Builder`` shape ``(session_id, task_run_id, outcome_status)``. The
commit hook passes that builder only when a provider was injected.

:func:`freeze_native_edge_view` is the expand-path writer. It freezes blocks
and native edges and does not construct an ``InfluenceClaim`` or cut grounding.
:class:`NativeEdgeViewStore` refuses to open or update a frozen claim view.
It also refuses a revision whose identity carries replay provenance, and
``read`` will not return one that was injected into the production index.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from openviking.session.aig_native_protocol import (
    OBJECT_KIND_NATIVE_EDGE_INDEX,
    SCHEMA_VERSION_AIG_NATIVE_V2,
    ProtocolContractError,
    is_replay_provenance_identity,
    protocol_content_hash,
    read_native_edge_index,
)
from openviking.session.influence_view import (
    CONSTRUCTION_MODE_NATIVE_SINGLE_PASS,
    PURPOSE_FAILURE_DIAGNOSIS,
    PURPOSE_POST_RUN_INDEX,
    PURPOSES,
    AigNativeEdge,
    AigNativeViewDraft,
    AigNativeViewRevision,
    CoverageManifest,
    InfluenceBlock,
    InfluenceClaim,
    InfluenceViewDraft,
    InfluenceViewRevision,
    RetrievalSubgraph,
    ViewSupersedeIndex,
    build_minimal_fallback_view,
)
from openviking.session.legacy_write_gate import require_legacy_audit_opt_in

# Production commit writes native-edge revisions here, not into the claim index.
NATIVE_EDGE_INDEX_FILENAME = "native-edges.json"

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


class NativeEdgeWriteError(ValueError):
    """The native-edge writer refused a claim object or a legacy view update."""

    def __init__(self, message: str, *, code: str) -> None:
        self.code = code
        super().__init__(message)


class NativeExtractionPayload:
    """Blocks, native edges, and coverage. There is no claim slot."""

    __slots__ = ("blocks", "edges", "coverage")

    def __init__(
        self,
        blocks: Sequence[InfluenceBlock],
        edges: Sequence[AigNativeEdge],
        coverage: CoverageManifest,
    ) -> None:
        block_items = tuple(blocks)
        edge_items = tuple(edges)
        for index, block in enumerate(block_items):
            if isinstance(block, InfluenceClaim):
                raise NativeEdgeWriteError(
                    "refusing to map an InfluenceClaim onto a native edge",
                    code="claim_mapping_forbidden",
                )
            if not isinstance(block, InfluenceBlock):
                raise TypeError(f"blocks[{index}] must be an InfluenceBlock")
        for index, edge in enumerate(edge_items):
            if isinstance(edge, InfluenceClaim):
                raise NativeEdgeWriteError(
                    "refusing to map an InfluenceClaim onto a native edge",
                    code="claim_mapping_forbidden",
                )
            if not isinstance(edge, AigNativeEdge):
                raise TypeError(f"edges[{index}] must be an AigNativeEdge")
        if not isinstance(coverage, CoverageManifest):
            raise TypeError("coverage must be a CoverageManifest")
        self.blocks = block_items
        self.edges = edge_items
        self.coverage = coverage


@dataclass(frozen=True)
class LegacyClaimViewAudit:
    """Audit record of one frozen claim view. It has no edge projection.

    The record cannot update the view. There is no flag to turn writing on.
    """

    revision_id: str
    content_hash: str
    claim_ids: tuple[str, ...]
    block_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_ids", tuple(self.claim_ids))
        object.__setattr__(self, "block_ids", tuple(self.block_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "content_hash": self.content_hash,
            "claim_ids": list(self.claim_ids),
            "block_ids": list(self.block_ids),
        }


def audit_legacy_claim_view(revision: InfluenceViewRevision) -> LegacyClaimViewAudit:
    """Read a frozen claim view. This does not convert claims into edges."""
    if not isinstance(revision, InfluenceViewRevision):
        raise TypeError("legacy audit requires an InfluenceViewRevision")
    return LegacyClaimViewAudit(
        revision_id=revision.revision_id,
        content_hash=revision.content_hash,
        claim_ids=tuple(claim.claim_id for claim in revision.claims),
        block_ids=tuple(block.block_id for block in revision.blocks),
    )


def open_legacy_claim_view_for_audit(path: str | Path) -> tuple[LegacyClaimViewAudit, ...]:
    """Open a claim index for audit. The file is not rewritten."""
    index = ViewSupersedeIndex(Path(path))
    found: list[LegacyClaimViewAudit] = []
    for purpose in (PURPOSE_POST_RUN_INDEX, PURPOSE_FAILURE_DIAGNOSIS):
        for revision in index.revisions_for_purpose(purpose):
            found.append(audit_legacy_claim_view(revision))
    return tuple(found)


def _reject_mapped_claim(edges: Sequence[Any]) -> tuple[AigNativeEdge, ...]:
    items = tuple(edges)
    for index, edge in enumerate(items):
        if isinstance(edge, InfluenceClaim):
            raise NativeEdgeWriteError(
                "refusing to map an InfluenceClaim onto a native edge",
                code="claim_mapping_forbidden",
            )
        if not isinstance(edge, AigNativeEdge):
            raise TypeError(f"edges[{index}] must be an AigNativeEdge")
    return items


def freeze_native_edge_view(
    *,
    session_id: str,
    task_run_id: str,
    view_id: str,
    revision_id: str,
    frozen_at: str,
    blocks: Sequence[InfluenceBlock],
    edges: Sequence[AigNativeEdge],
    coverage: CoverageManifest,
    purpose: str = PURPOSE_POST_RUN_INDEX,
    construction_mode: str = CONSTRUCTION_MODE_NATIVE_SINGLE_PASS,
    known_violations: Sequence[Mapping[str, Any]] | None = None,
    paper_sidecar_hash: str = "",
) -> AigNativeViewRevision:
    """Freeze one native-edge extraction. No claim and no cut grounding is created.

    ``purpose`` defaults to ``post_run_index``. ``failure_diagnosis`` uses the
    same structural checks as ``validate_native_view``. ``HighSeverityViolation``
    propagates. ``construction_mode`` defaults to ``native_single_pass_aig``.
    Paper-graph compiles pass ``adaptive_agentic_aig`` plus optional audit fields.
    """
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {sorted(PURPOSES)}, got {purpose!r}")
    identities = {
        "session_id": _require_str(session_id, "session_id"),
        "task_run_id": _require_str(task_run_id, "task_run_id"),
        "view_id": _require_str(view_id, "view_id"),
        "revision_id": _require_str(revision_id, "revision_id"),
    }
    stamp = _require_str(frozen_at, "frozen_at")
    if not isinstance(coverage, CoverageManifest):
        raise TypeError("coverage must be a CoverageManifest")
    block_items = tuple(blocks)
    for index, block in enumerate(block_items):
        if not isinstance(block, InfluenceBlock):
            raise TypeError(f"blocks[{index}] must be an InfluenceBlock")
    edge_items = _reject_mapped_claim(edges)
    draft = AigNativeViewDraft(
        view_id=identities["view_id"],
        purpose=purpose,  # type: ignore[arg-type]
        session_id=identities["session_id"],
        task_run_id=identities["task_run_id"],
        coverage=coverage,
        blocks=block_items,
        edges=edge_items,
    )
    return draft.freeze(
        construction_mode=construction_mode,  # type: ignore[arg-type]
        revision_id=identities["revision_id"],
        frozen_at=stamp,
        enforce_checks=True,
        known_violations=known_violations,
        paper_sidecar_hash=paper_sidecar_hash,
    )


class NativeEdgeViewStore:
    """Append-only store of native-edge revisions.

    A legacy claim index opened at ``path`` is refused. When ``legacy_index``
    already holds a claim view for the same session and purpose, ``write``
    refuses instead of superseding that view.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        legacy_index: ViewSupersedeIndex | None = None,
    ) -> None:
        self.path = Path(path)
        if legacy_index is not None and not isinstance(legacy_index, ViewSupersedeIndex):
            raise TypeError("legacy_index must be a ViewSupersedeIndex")
        self._legacy = legacy_index
        self._revisions: list[AigNativeViewRevision] = []
        self._load()

    def write(self, revision: AigNativeViewRevision) -> None:
        if isinstance(revision, InfluenceViewRevision):
            raise NativeEdgeWriteError(
                "native writer refuses to update a frozen claim view",
                code="legacy_view_update_forbidden",
            )
        if not isinstance(revision, AigNativeViewRevision):
            raise TypeError("revision must be an AigNativeViewRevision")
        if _is_replay_provenance(revision):
            raise NativeEdgeWriteError(
                "production writer excludes replay provenance",
                code="replay_provenance_excluded",
            )
        self._reject_legacy_update(revision)
        if any(existing.revision_id == revision.revision_id for existing in self._revisions):
            raise NativeEdgeWriteError(
                f"duplicate native revision_id {revision.revision_id}",
                code="duplicate_revision",
            )
        self._revisions.append(revision)
        self._save()

    def read(self, revision_id: str) -> AigNativeViewRevision:
        key = _require_str(revision_id, "revision_id")
        for revision in self._revisions:
            if revision.revision_id == key:
                return revision
        raise KeyError(key)

    def revisions(self) -> tuple[AigNativeViewRevision, ...]:
        """Revisions in write order. This does not rewrite the file."""
        return tuple(self._revisions)

    def _reject_legacy_update(self, revision: AigNativeViewRevision) -> None:
        legacy = self._legacy
        if legacy is None:
            return
        for purpose in (PURPOSE_POST_RUN_INDEX, PURPOSE_FAILURE_DIAGNOSIS):
            for existing in legacy.revisions_for_purpose(purpose):
                same_view = (
                    existing.session_id == revision.session_id
                    and existing.purpose == revision.purpose
                )
                if existing.revision_id == revision.revision_id or same_view:
                    raise NativeEdgeWriteError(
                        "native writer refuses to update a frozen claim view "
                        f"{existing.revision_id}",
                        code="legacy_view_update_forbidden",
                    )

    def _load(self) -> None:
        if not self.path.is_file():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise NativeEdgeWriteError(
                "native-edge store must be a dict",
                code="stored_record_invalid",
            )
        if payload.get("object_kind") != OBJECT_KIND_NATIVE_EDGE_INDEX or "keys" in payload:
            raise NativeEdgeWriteError(
                "legacy claim index is read-only; native writer refuses to open it",
                code="legacy_index_read_only",
            )
        try:
            document = read_native_edge_index(payload)
        except ProtocolContractError as exc:
            if "schema_version" in str(exc):
                raise NativeEdgeWriteError(
                    "native-edge store schema_version is not aig-native-v2",
                    code="unsupported_schema_version",
                ) from exc
            raise NativeEdgeWriteError(
                "native-edge store record is invalid",
                code="stored_record_invalid",
            ) from exc
        for item in document.revisions:
            if _is_replay_provenance(item.revision):
                raise NativeEdgeWriteError(
                    "replay provenance contaminates the production index; "
                    "repair the index by hand before opening it",
                    code="replay_provenance_contaminates_production",
                )
        self._revisions = [item.revision for item in document.revisions]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {"revisions": [revision.protocol_document() for revision in self._revisions]}
        payload = {
            "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
            "object_kind": OBJECT_KIND_NATIVE_EDGE_INDEX,
            "content_hash": protocol_content_hash(
                SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_NATIVE_EDGE_INDEX, body
            ),
            "body": body,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(self.path)


def _is_replay_provenance(revision: AigNativeViewRevision) -> bool:
    return is_replay_provenance_identity(
        revision_id=revision.revision_id,
        session_id=revision.session_id,
        task_run_id=revision.task_run_id,
        view_id=revision.view_id,
    )


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
    legacy_audit_opt_in: bool = False,
) -> InfluenceViewRevision:
    """Validate one extraction payload and freeze a single-pass revision.

    ``HighSeverityViolation`` propagates. This function does not catch it,
    does not write a partial revision, and does not call
    :func:`build_minimal_fallback_view`. ``construction_mode`` is ``native_single_pass_aig``.

    The default refuses. ``legacy_audit_opt_in=True`` is the explicit
    legacy-audit tool. Production commit does not pass it.
    """
    require_legacy_audit_opt_in(legacy_audit_opt_in)
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
        construction_mode=CONSTRUCTION_MODE_NATIVE_SINGLE_PASS,
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
    legacy_audit_opt_in: bool = False,
) -> Callable[[str, str, str], InfluenceViewRevision]:
    """Return a trajectory-index ``Builder`` that freezes ``provider`` output.

    The provider is called as ``provider(session_id, task_run_id, outcome_status)``
    and must return an :class:`ExtractionPayload`. ``view_id`` matches the
    post-run commit hook (``{session_id}:post_run_index``). ``revision_id`` is
    ``post-run:{session_id}:{task_run_id}:adaptive`` so it does not collide
    with the Phase 1 fallback id ``post-run:{session_id}:{task_run_id}``.
    ``frozen_at`` is the commit timestamp the hook already stored on the job.

    A refused freeze propagates. This builder does not degrade to fallback.
    The default refuses before the provider runs. ``legacy_audit_opt_in=True``
    is the explicit legacy-audit tool. The commit hook does not call this.
    """
    if not callable(provider):
        raise TypeError("extraction provider must be callable")
    stamp = _require_str(frozen_at, "frozen_at")

    def builder(session_id: str, task_run_id: str, outcome_status: str) -> InfluenceViewRevision:
        require_legacy_audit_opt_in(legacy_audit_opt_in)
        payload = provider(session_id, task_run_id, outcome_status)
        if not isinstance(payload, ExtractionPayload):
            raise TypeError("extraction provider must return an ExtractionPayload")
        return freeze_extracted_view(
            session_id=session_id,
            task_run_id=task_run_id,
            view_id=f"{session_id}:{PURPOSE_POST_RUN_INDEX}",
            revision_id=f"post-run:{session_id}:{task_run_id}:adaptive",
            frozen_at=stamp,
            blocks=payload.blocks,
            claims=payload.claims,
            coverage=payload.coverage,
            retrieval_subgraphs=payload.retrieval_subgraphs,
            legacy_audit_opt_in=True,
        )

    return builder
