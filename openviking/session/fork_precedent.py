# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Fork precedent retrieval for a Reader proposal (ADR-0043 P06).

``ForkPrecedentService.submit_proposal`` is the Reader entry. It refuses the
write until ``retrieve`` has stored a server record for that same proposal,
principal, and session. ``ForkNodeOverlayStore.submit_proposal`` stays the
persistence primitive and does not consult this module.

Tiers
-----
``high-confidence`` is a branch or pattern whose counted support includes
production-direct or replay-direct evidence. ``exploratory`` is analogical,
observational, or model-only support. ``negative/retired`` is a subject whose
latest counted row is challenged, invalidated, retired, or superseded. That
last tier is auditable and is not a candidate. Retired rows are not put back
into the candidate list.

Filters run here, not in the caller. A hit is kept only when the fork's
registered ACL intersects the principal, its first observation time is at or
before ``temporal_cutoff``, and it is neither the current session nor the
stable id of the proposal being prepared. Evidence counted for a tier is only
what ``observe_fork`` had already stamped at or before the cutoff.

Hits carry ``tier`` and ``source="historical_precedent"``. They are not
ledger rows. This module does not append BranchEvaluation records, and it
does not read legacy Claim or cut forks. Relevance is structural: the same
anchor, an overlapping factual block, or a pattern target boundary equal to
the proposal anchor. High-relevance candidates require an adopted, rejected,
or not-applicable reason. Contextual candidates and negative rows do not.

Replay, guidance, and group-chat dispatch are out of scope. A precedent whose
base revision or observing session carries replay provenance is omitted. A
proposal grounded on a replay revision is refused.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from openviking.session.aig_native_protocol import (
    DIRECT_EVIDENCE_TIERS,
    EVIDENCE_TIER_PRODUCTION_DIRECT,
    EVIDENCE_TIER_REPLAY_DIRECT,
    EXPLORATORY_EVIDENCE_TIERS,
    NEGATIVE_EVALUATION_STATUSES,
    SCHEMA_VERSION_AIG_NATIVE_V2,
    STEP_KIND_FACTUAL,
    BranchEvaluation,
    ForkProposal,
    ProtocolContractError,
    is_replay_revision_id,
    is_replay_session_id,
    is_safe_id,
    protocol_content_hash,
    read_fork_proposal,
)
from openviking.session.append_only import append_jsonl, read_json_object, write_new_json
from openviking.session.fork_node_overlay import (
    ForkNodeOverlayStore,
    ForkOverlayError,
    StoredForkOverlay,
    fork_node_stable_id,
    require_label_set,
)
from openviking.session.strategy_pattern import StrategyPatternStore

FORK_PRECEDENT_DIRNAME = "fork-precedents"
SOURCE_HISTORICAL = "historical_precedent"
RECORD_PRECEDENT = "fork_precedent"
RECORD_RETRIEVAL = "fork_precedent_retrieval"
RECORD_DISPOSITION = "precedent_disposition"
RECORD_SOURCE = "precedent_source"
RECORD_STAMP = "precedent_stamp"
TIER_HIGH = "high-confidence"
TIER_EXPLORATORY = "exploratory"
TIER_NEGATIVE = "negative/retired"
RELEVANCE_HIGH = "high"
RELEVANCE_CONTEXTUAL = "contextual"
SUBJECT_BRANCH = "branch"
SUBJECT_PATTERN = "strategy_pattern"

_DISPOSITIONS = frozenset({"adopted", "rejected", "not-applicable"})
_TIERS = frozenset({TIER_HIGH, TIER_EXPLORATORY, TIER_NEGATIVE})
_RELEVANCE = frozenset({RELEVANCE_HIGH, RELEVANCE_CONTEXTUAL})
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
_HEX_ID = re.compile(r"^[0-9a-f]{64}$")
_HIT_KEYS = frozenset(
    {
        "base_revision_id",
        "candidate",
        "disposition_required",
        "evidence_tier",
        "fork_node_id",
        "precedent_id",
        "rank",
        "record_kind",
        "recorded_at",
        "relevance",
        "session_id",
        "source",
        "status",
        "subject_id",
        "subject_kind",
        "tier",
    }
)
_DISPOSITION_KEYS = frozenset({"disposition", "precedent_id", "reason"})


class ForkPrecedentError(ValueError):
    """A precedent retrieval or Reader submission was refused."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ForkPrecedent:
    """One historical precedent. ``source`` keeps it distinct from current evidence."""

    precedent_id: str
    subject_kind: str
    subject_id: str
    tier: str
    evidence_tier: str
    source: str
    record_kind: str
    status: str
    fork_node_id: str
    base_revision_id: str
    session_id: str
    recorded_at: str
    relevance: str
    candidate: bool
    disposition_required: bool
    rank: int


@dataclass(frozen=True)
class PrecedentRetrievalRecord:
    """Server snapshot. Callers do not supply the tier lists."""

    retrieval_id: str
    session_id: str
    proposal_content_hash: str
    base_revision_id: str
    anchor_block_id: str
    principal_labels: tuple[str, ...]
    temporal_cutoff: str
    high_confidence: tuple[ForkPrecedent, ...]
    exploratory: tuple[ForkPrecedent, ...]
    negative_retired: tuple[ForkPrecedent, ...]

    @property
    def candidates(self) -> tuple[ForkPrecedent, ...]:
        """High-confidence and exploratory hits, in that order. Negative rows are omitted."""
        return self.high_confidence + self.exploratory


@dataclass(frozen=True)
class PrecedentDisposition:
    """Reader treatment of one precedent from the retrieval snapshot."""

    precedent_id: str
    disposition: str
    reason: str


@dataclass(frozen=True)
class PrecedentDispositionRecord:
    """Dispositions stored with the proposal. Not a ledger row."""

    proposal_content_hash: str
    retrieval_id: str
    fork_node_id: str
    session_id: str
    dispositions: tuple[PrecedentDisposition, ...]


@dataclass(frozen=True)
class ForkPrecedentCommit:
    """Overlay commit plus the disposition record written for that proposal."""

    stored: StoredForkOverlay
    deduplicated: bool
    retrieval_id: str
    disposition: PrecedentDispositionRecord


@dataclass(frozen=True)
class _Source:
    fork_node_id: str
    session_id: str
    recorded_at: str


@dataclass(frozen=True)
class _Stamp:
    evaluation_id: str
    fork_node_id: str
    recorded_at: str


class ForkPrecedentService:
    """Read-side precedent index under ``fork-precedents``.

    Observation times live here. The evidence ledger is only read.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._dir = self._root / FORK_PRECEDENT_DIRNAME
        self._index_path = self._dir / "index.jsonl"
        self._lock = threading.Lock()

    def observe_fork(
        self,
        overlay_store: ForkNodeOverlayStore,
        pattern_store: StrategyPatternStore,
        fork_node_id: str,
        *,
        session_id: str,
        recorded_at: str,
        principal_labels: Sequence[str],
    ) -> None:
        """Stamp current ledger rows for one visible fork.

        The first observation fixes the fork's session and entry time. A later
        call from the same session may stamp rows added after that entry. It
        does not move the entry time earlier and does not change the session.
        Rows already stamped keep their first time.
        """
        _require_session(session_id)
        _require_recorded_at(recorded_at)
        _require_stores(overlay_store, pattern_store)
        stored = overlay_store.read_overlay_by_id(
            fork_node_id, principal_labels=principal_labels
        )
        if stored is None:
            raise ForkPrecedentError("source_unknown")
        imagined, pattern_ids = _subjects(stored.overlay, pattern_store)
        evaluation_ids = [
            item.evaluation_id
            for item in pattern_store.audit_evaluations()
            if _belongs(item, imagined, pattern_ids)
        ]
        with self._lock:
            self._observe_unlocked(fork_node_id, session_id, recorded_at, evaluation_ids)

    def retrieve(
        self,
        overlay_store: ForkNodeOverlayStore,
        pattern_store: StrategyPatternStore,
        payload: Mapping[str, Any],
        *,
        principal_labels: Sequence[str],
        session_id: str,
        temporal_cutoff: str,
    ) -> PrecedentRetrievalRecord:
        """Build and store the precedent snapshot for one proposal.

        The same proposal, principal, cutoff, and corpus return the stored
        record. The caller cannot pass tier lists or turn status filtering off.
        """
        _require_session(session_id)
        _require_cutoff(temporal_cutoff)
        _require_stores(overlay_store, pattern_store)
        proposal = _read_proposal(overlay_store, payload, principal_labels)
        if is_replay_revision_id(proposal.base_revision_id):
            raise ForkPrecedentError("replay_provenance_excluded")
        visible = overlay_store.visible_overlays(principal_labels=principal_labels)
        patterns = pattern_store.list_patterns()
        memberships = {
            pattern.pattern_id: pattern_store.memberships(pattern.pattern_id)
            for pattern in patterns
        }
        evaluations = pattern_store.audit_evaluations()
        with self._lock:
            sources = self._load_sources()
            stamps = self._load_stamps()
            record = _assemble(
                proposal=proposal,
                session_id=session_id,
                principal_labels=require_label_set(principal_labels, code="acl_denied"),
                temporal_cutoff=temporal_cutoff,
                visible=visible,
                patterns=patterns,
                memberships=memberships,
                evaluations=evaluations,
                sources=sources,
                stamps=stamps,
            )
            self._store_retrieval(record)
            return record

    def submit_proposal(
        self,
        overlay_store: ForkNodeOverlayStore,
        payload: Mapping[str, Any],
        *,
        retrieval_id: str,
        dispositions: Sequence[Mapping[str, str]],
        principal_labels: Sequence[str],
        session_id: str,
        recorded_at: str,
        authorize: Any = None,
    ) -> ForkPrecedentCommit:
        """Persist a Reader proposal only when its retrieval snapshot is on disk.

        High-relevance candidates in that snapshot need a reason. The snapshot
        is not recomputed here, so a precedent observed after retrieval cannot
        appear as a requirement. Disposition rows are written beside the
        proposal and are not ledger rows.
        """
        _require_session(session_id)
        _require_recorded_at(recorded_at)
        if not isinstance(overlay_store, ForkNodeOverlayStore):
            raise ForkPrecedentError("store_required")
        if not isinstance(retrieval_id, str) or not retrieval_id:
            raise ForkPrecedentError("retrieval_required")
        principal = require_label_set(principal_labels, code="acl_denied")
        with self._lock:
            record = self._load_retrieval(retrieval_id)
        if frozenset(record.principal_labels) != principal or record.session_id != session_id:
            raise ForkPrecedentError("retrieval_mismatch")
        proposal = _read_proposal(overlay_store, payload, principal_labels)
        if proposal.content_hash != record.proposal_content_hash:
            raise ForkPrecedentError("retrieval_mismatch")
        chosen = _parse_dispositions(dispositions, record)
        with self._lock:
            existing = self._read_disposition(proposal.content_hash)
            if existing is not None and (
                existing.retrieval_id != record.retrieval_id
                or not _same_dispositions(existing, chosen)
            ):
                raise ForkPrecedentError("disposition_conflict")
        commit = overlay_store.submit_proposal(
            payload,
            principal_labels=principal_labels,
            authorize=authorize,
            session_id=session_id,
        )
        disposition = PrecedentDispositionRecord(
            proposal_content_hash=proposal.content_hash,
            retrieval_id=record.retrieval_id,
            fork_node_id=commit.stored.fork_node_id,
            session_id=session_id,
            dispositions=chosen,
        )
        with self._lock:
            stored = self._read_disposition(proposal.content_hash)
            if stored is None:
                self._write_disposition(disposition)
                stored = disposition
            elif stored.retrieval_id != record.retrieval_id or not _same_dispositions(
                stored, chosen
            ):
                raise ForkPrecedentError("disposition_conflict")
            self._ensure_source(commit.stored.fork_node_id, session_id, recorded_at)
        return ForkPrecedentCommit(
            stored=commit.stored,
            deduplicated=commit.deduplicated,
            retrieval_id=record.retrieval_id,
            disposition=stored,
        )

    def read_dispositions(self, proposal_content_hash: str) -> PrecedentDispositionRecord | None:
        """Return the dispositions stored for one proposal hash, if present."""
        with self._lock:
            return self._read_disposition(proposal_content_hash)

    def read_retrieval(self, retrieval_id: str) -> PrecedentRetrievalRecord:
        """Return one stored snapshot. A missing id is ``retrieval_not_found``."""
        if not isinstance(retrieval_id, str) or _HEX_ID.fullmatch(retrieval_id) is None:
            raise ForkPrecedentError("retrieval_required")
        with self._lock:
            return self._load_retrieval(retrieval_id)

    def _observe_unlocked(
        self,
        fork_node_id: str,
        session_id: str,
        recorded_at: str,
        evaluation_ids: Sequence[str],
    ) -> None:
        existing = self._read_source(fork_node_id)
        if existing is None:
            self._write_source(
                _Source(
                    fork_node_id=fork_node_id,
                    session_id=session_id,
                    recorded_at=recorded_at,
                )
            )
        elif existing.session_id != session_id or recorded_at < existing.recorded_at:
            raise ForkPrecedentError("source_conflict")
        for evaluation_id in evaluation_ids:
            if self._read_stamp(evaluation_id) is not None:
                continue
            self._write_stamp(
                _Stamp(
                    evaluation_id=evaluation_id,
                    fork_node_id=fork_node_id,
                    recorded_at=recorded_at,
                )
            )

    def _ensure_source(self, fork_node_id: str, session_id: str, recorded_at: str) -> None:
        if self._read_source(fork_node_id) is not None:
            return
        self._write_source(
            _Source(
                fork_node_id=fork_node_id,
                session_id=session_id,
                recorded_at=recorded_at,
            )
        )

    def _store_retrieval(self, record: PrecedentRetrievalRecord) -> None:
        path = self._retrieval_path(record.retrieval_id)
        if path is None:
            raise ForkPrecedentError("stored_record_invalid")
        if path.is_file():
            loaded = self._load_retrieval(record.retrieval_id)
            if loaded != record:
                raise ForkPrecedentError("stored_record_invalid")
            return
        body = _retrieval_body(record)
        digest = protocol_content_hash(
            SCHEMA_VERSION_AIG_NATIVE_V2, RECORD_RETRIEVAL, body
        )
        if digest != record.retrieval_id:
            raise ForkPrecedentError("stored_record_invalid")
        self._write_new(
            path,
            {
                "body": body,
                "content_hash": digest,
                "object_kind": RECORD_RETRIEVAL,
                "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
            },
        )
        self._append_index(
            {
                "proposal_content_hash": record.proposal_content_hash,
                "record_kind": RECORD_RETRIEVAL,
                "retrieval_id": record.retrieval_id,
                "session_id": record.session_id,
            }
        )

    def _write_disposition(self, record: PrecedentDispositionRecord) -> None:
        path = self._disposition_path(record.proposal_content_hash)
        if path is None:
            raise ForkPrecedentError("stored_record_invalid")
        body = _disposition_body(record)
        digest = protocol_content_hash(
            SCHEMA_VERSION_AIG_NATIVE_V2, RECORD_DISPOSITION, body
        )
        self._write_new(
            path,
            {
                "body": body,
                "content_hash": digest,
                "object_kind": RECORD_DISPOSITION,
                "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
            },
        )
        self._append_index(
            {
                "fork_node_id": record.fork_node_id,
                "proposal_content_hash": record.proposal_content_hash,
                "record_kind": RECORD_DISPOSITION,
                "retrieval_id": record.retrieval_id,
            }
        )

    def _write_source(self, source: _Source) -> None:
        path = self._source_path(source.fork_node_id)
        self._write_new(
            path,
            {
                "fork_node_id": source.fork_node_id,
                "record_kind": RECORD_SOURCE,
                "recorded_at": source.recorded_at,
                "session_id": source.session_id,
            },
        )
        self._append_index(
            {
                "fork_node_id": source.fork_node_id,
                "record_kind": RECORD_SOURCE,
                "recorded_at": source.recorded_at,
            }
        )

    def _write_stamp(self, stamp: _Stamp) -> None:
        path = self._stamp_path(stamp.evaluation_id)
        self._write_new(
            path,
            {
                "evaluation_id": stamp.evaluation_id,
                "fork_node_id": stamp.fork_node_id,
                "record_kind": RECORD_STAMP,
                "recorded_at": stamp.recorded_at,
            },
        )
        self._append_index(
            {
                "evaluation_id": stamp.evaluation_id,
                "fork_node_id": stamp.fork_node_id,
                "record_kind": RECORD_STAMP,
                "recorded_at": stamp.recorded_at,
            }
        )

    def _load_retrieval(self, retrieval_id: str) -> PrecedentRetrievalRecord:
        path = self._retrieval_path(retrieval_id)
        if path is None or not path.is_file():
            raise ForkPrecedentError("retrieval_not_found")
        payload = _read_json(path)
        body = payload.get("body")
        if (
            payload.get("schema_version") != SCHEMA_VERSION_AIG_NATIVE_V2
            or payload.get("object_kind") != RECORD_RETRIEVAL
            or not isinstance(body, dict)
        ):
            raise ForkPrecedentError("retrieval_tampered")
        digest = protocol_content_hash(
            SCHEMA_VERSION_AIG_NATIVE_V2, RECORD_RETRIEVAL, body
        )
        if payload.get("content_hash") != digest or retrieval_id != digest:
            raise ForkPrecedentError("retrieval_tampered")
        return _record_from_body(retrieval_id, body)

    def _read_disposition(self, proposal_content_hash: str) -> PrecedentDispositionRecord | None:
        path = self._disposition_path(proposal_content_hash)
        if path is None or not path.is_file():
            return None
        payload = _read_json(path)
        body = payload.get("body")
        if not isinstance(body, dict):
            raise ForkPrecedentError("stored_record_invalid")
        digest = protocol_content_hash(
            SCHEMA_VERSION_AIG_NATIVE_V2, RECORD_DISPOSITION, body
        )
        if payload.get("content_hash") != digest:
            raise ForkPrecedentError("stored_record_invalid")
        return _disposition_from_body(body)

    def _read_source(self, fork_node_id: str) -> _Source | None:
        path = self._source_path(fork_node_id)
        if not path.is_file():
            return None
        return _source_from_payload(_read_json(path), fork_node_id)

    def _read_stamp(self, evaluation_id: str) -> _Stamp | None:
        path = self._stamp_path(evaluation_id)
        if not path.is_file():
            return None
        return _stamp_from_payload(_read_json(path), evaluation_id)

    def _load_sources(self) -> dict[str, _Source]:
        directory = self._dir / "sources"
        if not directory.is_dir():
            return {}
        found: dict[str, _Source] = {}
        for path in sorted(directory.glob("*.json")):
            payload = _read_json(path)
            fork_node_id = payload.get("fork_node_id")
            if not isinstance(fork_node_id, str):
                raise ForkPrecedentError("stored_record_invalid")
            found[fork_node_id] = _source_from_payload(payload, fork_node_id)
        return found

    def _load_stamps(self) -> dict[str, _Stamp]:
        directory = self._dir / "stamps"
        if not directory.is_dir():
            return {}
        found: dict[str, _Stamp] = {}
        for path in sorted(directory.glob("*.json")):
            payload = _read_json(path)
            evaluation_id = payload.get("evaluation_id")
            if not isinstance(evaluation_id, str):
                raise ForkPrecedentError("stored_record_invalid")
            found[evaluation_id] = _stamp_from_payload(payload, evaluation_id)
        return found

    def _retrieval_path(self, retrieval_id: str) -> Path | None:
        if not isinstance(retrieval_id, str) or _HEX_ID.fullmatch(retrieval_id) is None:
            return None
        return self._dir / "retrievals" / f"{retrieval_id}.json"

    def _disposition_path(self, proposal_content_hash: str) -> Path | None:
        if (
            not isinstance(proposal_content_hash, str)
            or _HEX_ID.fullmatch(proposal_content_hash) is None
        ):
            return None
        return self._dir / "dispositions" / f"{proposal_content_hash}.json"

    def _source_path(self, fork_node_id: str) -> Path:
        digest = hashlib.sha256(fork_node_id.encode("utf-8")).hexdigest()
        return self._dir / "sources" / f"{digest}.json"

    def _stamp_path(self, evaluation_id: str) -> Path:
        digest = hashlib.sha256(evaluation_id.encode("utf-8")).hexdigest()
        return self._dir / "stamps" / f"{digest}.json"

    def _append_index(self, row: Mapping[str, Any]) -> None:
        append_jsonl(self._index_path, row)

    def _write_new(self, path: Path, payload: Mapping[str, Any]) -> None:
        write_new_json(path, payload, ForkPrecedentError)


def _assemble(
    *,
    proposal: ForkProposal,
    session_id: str,
    principal_labels: frozenset[str],
    temporal_cutoff: str,
    visible: Sequence[StoredForkOverlay],
    patterns: Sequence[Any],
    memberships: Mapping[str, Sequence[Any]],
    evaluations: Sequence[BranchEvaluation],
    sources: Mapping[str, _Source],
    stamps: Mapping[str, _Stamp],
) -> PrecedentRetrievalRecord:
    self_id = fork_node_stable_id(
        proposal.base_revision_id,
        proposal.anchor_block_id,
        proposal.base_revision_block_content_hash,
    )
    eligible: dict[str, tuple[StoredForkOverlay, _Source]] = {}
    for stored in visible:
        source = sources.get(stored.fork_node_id)
        if source is None or source.recorded_at > temporal_cutoff:
            continue
        if is_replay_revision_id(stored.overlay.base_revision_id) or is_replay_session_id(
            source.session_id
        ):
            continue
        if source.session_id == session_id or stored.fork_node_id == self_id:
            continue
        eligible[stored.fork_node_id] = (stored, source)
    high: list[ForkPrecedent] = []
    exploratory: list[ForkPrecedent] = []
    negative: list[ForkPrecedent] = []
    for fork_node_id in sorted(eligible):
        stored, source = eligible[fork_node_id]
        imagined = _imagined_ids(stored.overlay)
        rows = _counted_rows(
            evaluations,
            stamps,
            temporal_cutoff,
            eligible,
            subject_kind="imagined_node",
            subject_ids=imagined,
            stamp_fork_id=fork_node_id,
        )
        classified = _classify(rows)
        if classified is None:
            continue
        _append_classified(
            high,
            exploratory,
            negative,
            _hit(
                subject_kind=SUBJECT_BRANCH,
                subject_id=fork_node_id,
                fork_node_id=fork_node_id,
                base_revision_id=stored.overlay.base_revision_id,
                session_id=source.session_id,
                recorded_at=classified[3],
                classified=classified,
                relevance=(
                    RELEVANCE_HIGH
                    if _branch_relevant(stored.overlay, proposal)
                    else RELEVANCE_CONTEXTUAL
                ),
            ),
        )
    for pattern in sorted(patterns, key=lambda item: item.pattern_id):
        members = [
            item.fork_node_id
            for item in memberships.get(pattern.pattern_id, ())
            if item.fork_node_id in eligible
        ]
        if not members:
            continue
        rows = _counted_rows(
            evaluations,
            stamps,
            temporal_cutoff,
            eligible,
            subject_kind=SUBJECT_PATTERN,
            subject_ids=frozenset({pattern.pattern_id}),
            stamp_fork_id=None,
            member_forks=frozenset(members),
        )
        classified = _classify(rows)
        if classified is None:
            continue
        latest_fork = rows[-1][2]
        stored, source = eligible[latest_fork]
        relevant = pattern.target_boundary == proposal.anchor_block_id or any(
            _branch_relevant(eligible[fork_id][0].overlay, proposal) for fork_id in members
        )
        _append_classified(
            high,
            exploratory,
            negative,
            _hit(
                subject_kind=SUBJECT_PATTERN,
                subject_id=pattern.pattern_id,
                fork_node_id=latest_fork,
                base_revision_id=stored.overlay.base_revision_id,
                session_id=source.session_id,
                recorded_at=classified[3],
                classified=classified,
                relevance=RELEVANCE_HIGH if relevant else RELEVANCE_CONTEXTUAL,
            ),
        )
    record_body_hits = (
        _rank(high),
        _rank(exploratory),
        _rank(negative),
    )
    body = {
        "anchor_block_id": proposal.anchor_block_id,
        "base_revision_id": proposal.base_revision_id,
        "exploratory": [_hit_dict(hit) for hit in record_body_hits[1]],
        "high_confidence": [_hit_dict(hit) for hit in record_body_hits[0]],
        "negative_retired": [_hit_dict(hit) for hit in record_body_hits[2]],
        "principal_labels": sorted(principal_labels),
        "proposal_content_hash": proposal.content_hash,
        "session_id": session_id,
        "temporal_cutoff": temporal_cutoff,
    }
    retrieval_id = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, RECORD_RETRIEVAL, body
    )
    return PrecedentRetrievalRecord(
        retrieval_id=retrieval_id,
        session_id=session_id,
        proposal_content_hash=proposal.content_hash,
        base_revision_id=proposal.base_revision_id,
        anchor_block_id=proposal.anchor_block_id,
        principal_labels=tuple(sorted(principal_labels)),
        temporal_cutoff=temporal_cutoff,
        high_confidence=record_body_hits[0],
        exploratory=record_body_hits[1],
        negative_retired=record_body_hits[2],
    )


def _counted_rows(
    evaluations: Sequence[BranchEvaluation],
    stamps: Mapping[str, _Stamp],
    temporal_cutoff: str,
    eligible: Mapping[str, tuple[StoredForkOverlay, _Source]],
    *,
    subject_kind: str,
    subject_ids: frozenset[str],
    stamp_fork_id: str | None,
    member_forks: frozenset[str] | None = None,
) -> list[tuple[BranchEvaluation, str, str]]:
    rows: list[tuple[BranchEvaluation, str, str]] = []
    for item in evaluations:
        if item.subject_kind != subject_kind or item.subject_id not in subject_ids:
            continue
        stamp = stamps.get(item.evaluation_id)
        if stamp is None or stamp.recorded_at > temporal_cutoff:
            continue
        if stamp.fork_node_id not in eligible:
            continue
        if stamp_fork_id is not None and stamp.fork_node_id != stamp_fork_id:
            continue
        if member_forks is not None and stamp.fork_node_id not in member_forks:
            continue
        rows.append((item, stamp.recorded_at, stamp.fork_node_id))
    return rows


def _classify(
    rows: Sequence[tuple[BranchEvaluation, str, str]],
) -> tuple[str, str, str, str] | None:
    if not rows:
        return None
    latest = rows[-1][0]
    latest_at = rows[-1][1]
    if latest.status in NEGATIVE_EVALUATION_STATUSES:
        return (TIER_NEGATIVE, latest.evidence_tier, latest.status, latest_at)
    direct = [
        item[0]
        for item in rows
        if item[0].status not in NEGATIVE_EVALUATION_STATUSES
        and item[0].evidence_tier in DIRECT_EVIDENCE_TIERS
    ]
    if direct:
        evidence = (
            EVIDENCE_TIER_PRODUCTION_DIRECT
            if any(item.evidence_tier == EVIDENCE_TIER_PRODUCTION_DIRECT for item in direct)
            else EVIDENCE_TIER_REPLAY_DIRECT
        )
        return (TIER_HIGH, evidence, latest.status, latest_at)
    exploratory = [
        item[0]
        for item in rows
        if item[0].status not in NEGATIVE_EVALUATION_STATUSES
        and item[0].evidence_tier in EXPLORATORY_EVIDENCE_TIERS
    ]
    if not exploratory:
        return None
    return (TIER_EXPLORATORY, exploratory[-1].evidence_tier, latest.status, latest_at)


def _hit(
    *,
    subject_kind: str,
    subject_id: str,
    fork_node_id: str,
    base_revision_id: str,
    session_id: str,
    recorded_at: str,
    classified: tuple[str, str, str, str],
    relevance: str,
) -> ForkPrecedent:
    tier, evidence_tier, status, _when = classified
    candidate = tier != TIER_NEGATIVE
    prefix = "pattern" if subject_kind == SUBJECT_PATTERN else "branch"
    return ForkPrecedent(
        precedent_id=f"{prefix}:{subject_id}",
        subject_kind=subject_kind,
        subject_id=subject_id,
        tier=tier,
        evidence_tier=evidence_tier,
        source=SOURCE_HISTORICAL,
        record_kind=RECORD_PRECEDENT,
        status=status,
        fork_node_id=fork_node_id,
        base_revision_id=base_revision_id,
        session_id=session_id,
        recorded_at=recorded_at,
        relevance=relevance,
        candidate=candidate,
        disposition_required=candidate and relevance == RELEVANCE_HIGH,
        rank=0,
    )


def _rank(hits: Sequence[ForkPrecedent]) -> tuple[ForkPrecedent, ...]:
    ordered = sorted(hits, key=lambda hit: hit.precedent_id)
    ordered = sorted(ordered, key=lambda hit: hit.recorded_at, reverse=True)
    ordered = sorted(ordered, key=lambda hit: 0 if hit.relevance == RELEVANCE_HIGH else 1)
    return tuple(replace(hit, rank=index) for index, hit in enumerate(ordered, start=1))


def _append_classified(
    high: list[ForkPrecedent],
    exploratory: list[ForkPrecedent],
    negative: list[ForkPrecedent],
    hit: ForkPrecedent,
) -> None:
    if hit.tier == TIER_HIGH:
        high.append(hit)
    elif hit.tier == TIER_EXPLORATORY:
        exploratory.append(hit)
    else:
        negative.append(hit)


def _branch_relevant(overlay: Any, proposal: ForkProposal) -> bool:
    if overlay.anchor_block_id == proposal.anchor_block_id:
        return True
    return bool(_factual_ids(overlay) & _factual_ids(proposal))


def _factual_ids(document: Any) -> set[str]:
    return {
        step.block_id
        for step in document.branch_steps
        if step.step_kind == STEP_KIND_FACTUAL and isinstance(step.block_id, str)
    }


def _imagined_ids(overlay: Any) -> frozenset[str]:
    return frozenset(
        step.imagined_node_id
        for step in overlay.branch_steps
        if step.step_kind == "imagined_node" and isinstance(step.imagined_node_id, str)
    )


def _subjects(overlay: Any, pattern_store: StrategyPatternStore) -> tuple[frozenset[str], frozenset[str]]:
    imagined = _imagined_ids(overlay)
    pattern_ids = {
        pattern.pattern_id
        for pattern in pattern_store.list_patterns()
        if any(
            item.fork_node_id == overlay.fork_node_id
            for item in pattern_store.memberships(pattern.pattern_id)
        )
    }
    return imagined, frozenset(pattern_ids)


def _belongs(
    item: BranchEvaluation, imagined: frozenset[str], pattern_ids: frozenset[str]
) -> bool:
    if item.subject_kind == "imagined_node":
        return item.subject_id in imagined
    if item.subject_kind == SUBJECT_PATTERN:
        return item.subject_id in pattern_ids
    return False


def _read_proposal(
    overlay_store: ForkNodeOverlayStore,
    payload: Mapping[str, Any],
    principal_labels: Sequence[str],
) -> ForkProposal:
    if not isinstance(payload, Mapping):
        raise ProtocolContractError("protocol document must be a dict")
    body = payload.get("body")
    revision_id = body.get("base_revision_id") if isinstance(body, Mapping) else None
    if not isinstance(revision_id, str) or not revision_id:
        raise ForkOverlayError("base_revision_missing")
    catalog = overlay_store.registered_catalog(
        revision_id, principal_labels=principal_labels
    )
    return read_fork_proposal(payload, catalog)


def _parse_dispositions(
    dispositions: Sequence[Mapping[str, str]],
    record: PrecedentRetrievalRecord,
) -> tuple[PrecedentDisposition, ...]:
    if isinstance(dispositions, (str, bytes)) or not isinstance(dispositions, Sequence):
        raise ForkPrecedentError("disposition_invalid")
    known = {
        hit.precedent_id
        for hit in (
            record.high_confidence + record.exploratory + record.negative_retired
        )
    }
    required = {
        hit.precedent_id
        for hit in record.candidates
        if hit.disposition_required
    }
    found: dict[str, PrecedentDisposition] = {}
    for item in dispositions:
        if not isinstance(item, Mapping) or set(item) != _DISPOSITION_KEYS:
            raise ForkPrecedentError("disposition_invalid")
        precedent_id = item.get("precedent_id")
        disposition = item.get("disposition")
        reason = item.get("reason")
        if not isinstance(precedent_id, str) or not precedent_id:
            raise ForkPrecedentError("disposition_invalid")
        if disposition not in _DISPOSITIONS:
            raise ForkPrecedentError("disposition_invalid")
        if not isinstance(reason, str) or not reason.strip():
            raise ForkPrecedentError("disposition_reason_required")
        if precedent_id in found:
            raise ForkPrecedentError("disposition_duplicate")
        if precedent_id not in known:
            raise ForkPrecedentError("disposition_unknown_precedent")
        found[precedent_id] = PrecedentDisposition(
            precedent_id=precedent_id,
            disposition=disposition,
            reason=reason.strip(),
        )
    if not required.issubset(found):
        raise ForkPrecedentError("disposition_required")
    return tuple(found[key] for key in sorted(found))


def _same_dispositions(
    stored: PrecedentDispositionRecord, chosen: Sequence[PrecedentDisposition]
) -> bool:
    left = {(item.precedent_id, item.disposition, item.reason) for item in stored.dispositions}
    right = {(item.precedent_id, item.disposition, item.reason) for item in chosen}
    return left == right


def _retrieval_body(record: PrecedentRetrievalRecord) -> dict[str, Any]:
    return {
        "anchor_block_id": record.anchor_block_id,
        "base_revision_id": record.base_revision_id,
        "exploratory": [_hit_dict(hit) for hit in record.exploratory],
        "high_confidence": [_hit_dict(hit) for hit in record.high_confidence],
        "negative_retired": [_hit_dict(hit) for hit in record.negative_retired],
        "principal_labels": list(record.principal_labels),
        "proposal_content_hash": record.proposal_content_hash,
        "session_id": record.session_id,
        "temporal_cutoff": record.temporal_cutoff,
    }


def _hit_dict(hit: ForkPrecedent) -> dict[str, Any]:
    return {
        "base_revision_id": hit.base_revision_id,
        "candidate": hit.candidate,
        "disposition_required": hit.disposition_required,
        "evidence_tier": hit.evidence_tier,
        "fork_node_id": hit.fork_node_id,
        "precedent_id": hit.precedent_id,
        "rank": hit.rank,
        "record_kind": hit.record_kind,
        "recorded_at": hit.recorded_at,
        "relevance": hit.relevance,
        "session_id": hit.session_id,
        "source": hit.source,
        "status": hit.status,
        "subject_id": hit.subject_id,
        "subject_kind": hit.subject_kind,
        "tier": hit.tier,
    }


def _record_from_body(retrieval_id: str, body: Mapping[str, Any]) -> PrecedentRetrievalRecord:
    expected = {
        "anchor_block_id",
        "base_revision_id",
        "exploratory",
        "high_confidence",
        "negative_retired",
        "principal_labels",
        "proposal_content_hash",
        "session_id",
        "temporal_cutoff",
    }
    if set(body) != expected:
        raise ForkPrecedentError("stored_record_invalid")
    labels = body.get("principal_labels")
    if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
        raise ForkPrecedentError("stored_record_invalid")
    return PrecedentRetrievalRecord(
        retrieval_id=retrieval_id,
        session_id=_require_text(body.get("session_id"), "session_id"),
        proposal_content_hash=_require_text(
            body.get("proposal_content_hash"), "proposal_content_hash"
        ),
        base_revision_id=_require_text(body.get("base_revision_id"), "base_revision_id"),
        anchor_block_id=_require_text(body.get("anchor_block_id"), "anchor_block_id"),
        principal_labels=tuple(labels),
        temporal_cutoff=_require_text(body.get("temporal_cutoff"), "temporal_cutoff"),
        high_confidence=_hits_from(body.get("high_confidence"), TIER_HIGH),
        exploratory=_hits_from(body.get("exploratory"), TIER_EXPLORATORY),
        negative_retired=_hits_from(body.get("negative_retired"), TIER_NEGATIVE),
    )


def _hits_from(value: Any, tier: str) -> tuple[ForkPrecedent, ...]:
    if not isinstance(value, list):
        raise ForkPrecedentError("stored_record_invalid")
    hits = tuple(_hit_from(item, tier) for item in value)
    if [hit.rank for hit in hits] != list(range(1, len(hits) + 1)):
        raise ForkPrecedentError("stored_record_invalid")
    return hits


def _hit_from(value: Any, tier: str) -> ForkPrecedent:
    if not isinstance(value, Mapping) or set(value) != _HIT_KEYS:
        raise ForkPrecedentError("stored_record_invalid")
    if value.get("tier") != tier or value.get("source") != SOURCE_HISTORICAL:
        raise ForkPrecedentError("stored_record_invalid")
    if value.get("record_kind") != RECORD_PRECEDENT:
        raise ForkPrecedentError("stored_record_invalid")
    if value.get("relevance") not in _RELEVANCE or value.get("tier") not in _TIERS:
        raise ForkPrecedentError("stored_record_invalid")
    candidate = value.get("candidate")
    required = value.get("disposition_required")
    rank = value.get("rank")
    if not isinstance(candidate, bool) or not isinstance(required, bool):
        raise ForkPrecedentError("stored_record_invalid")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ForkPrecedentError("stored_record_invalid")
    if candidate == (tier == TIER_NEGATIVE) or (required and not candidate):
        raise ForkPrecedentError("stored_record_invalid")
    return ForkPrecedent(
        precedent_id=_require_text(value.get("precedent_id"), "precedent_id"),
        subject_kind=_require_text(value.get("subject_kind"), "subject_kind"),
        subject_id=_require_text(value.get("subject_id"), "subject_id"),
        tier=tier,
        evidence_tier=_require_text(value.get("evidence_tier"), "evidence_tier"),
        source=SOURCE_HISTORICAL,
        record_kind=RECORD_PRECEDENT,
        status=_require_text(value.get("status"), "status"),
        fork_node_id=_require_text(value.get("fork_node_id"), "fork_node_id"),
        base_revision_id=_require_text(value.get("base_revision_id"), "base_revision_id"),
        session_id=_require_text(value.get("session_id"), "session_id"),
        recorded_at=_require_text(value.get("recorded_at"), "recorded_at"),
        relevance=str(value.get("relevance")),
        candidate=candidate,
        disposition_required=required,
        rank=rank,
    )


def _disposition_body(record: PrecedentDispositionRecord) -> dict[str, Any]:
    return {
        "dispositions": [
            {
                "disposition": item.disposition,
                "precedent_id": item.precedent_id,
                "reason": item.reason,
            }
            for item in record.dispositions
        ],
        "fork_node_id": record.fork_node_id,
        "proposal_content_hash": record.proposal_content_hash,
        "record_kind": RECORD_DISPOSITION,
        "retrieval_id": record.retrieval_id,
        "session_id": record.session_id,
        "source": SOURCE_HISTORICAL,
    }


def _disposition_from_body(body: Mapping[str, Any]) -> PrecedentDispositionRecord:
    raw = body.get("dispositions")
    if not isinstance(raw, list):
        raise ForkPrecedentError("stored_record_invalid")
    items: list[PrecedentDisposition] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ForkPrecedentError("stored_record_invalid")
        items.append(
            PrecedentDisposition(
                precedent_id=_require_text(item.get("precedent_id"), "precedent_id"),
                disposition=_require_text(item.get("disposition"), "disposition"),
                reason=_require_text(item.get("reason"), "reason"),
            )
        )
    return PrecedentDispositionRecord(
        proposal_content_hash=_require_text(
            body.get("proposal_content_hash"), "proposal_content_hash"
        ),
        retrieval_id=_require_text(body.get("retrieval_id"), "retrieval_id"),
        fork_node_id=_require_text(body.get("fork_node_id"), "fork_node_id"),
        session_id=_require_text(body.get("session_id"), "session_id"),
        dispositions=tuple(items),
    )


def _source_from_payload(payload: Mapping[str, Any], fork_node_id: str) -> _Source:
    if payload.get("fork_node_id") != fork_node_id:
        raise ForkPrecedentError("stored_record_invalid")
    return _Source(
        fork_node_id=fork_node_id,
        session_id=_require_text(payload.get("session_id"), "session_id"),
        recorded_at=_require_text(payload.get("recorded_at"), "recorded_at"),
    )


def _stamp_from_payload(payload: Mapping[str, Any], evaluation_id: str) -> _Stamp:
    if payload.get("evaluation_id") != evaluation_id:
        raise ForkPrecedentError("stored_record_invalid")
    return _Stamp(
        evaluation_id=evaluation_id,
        fork_node_id=_require_text(payload.get("fork_node_id"), "fork_node_id"),
        recorded_at=_require_text(payload.get("recorded_at"), "recorded_at"),
    )


def _require_stores(
    overlay_store: ForkNodeOverlayStore, pattern_store: StrategyPatternStore
) -> None:
    if not isinstance(overlay_store, ForkNodeOverlayStore) or not isinstance(
        pattern_store, StrategyPatternStore
    ):
        raise ForkPrecedentError("store_required")


def _require_session(session_id: str) -> None:
    if not is_safe_id(session_id):
        raise ForkPrecedentError("session_required")


def _require_cutoff(value: str) -> None:
    if not isinstance(value, str) or _TIMESTAMP.fullmatch(value) is None:
        raise ForkPrecedentError("temporal_cutoff_required")


def _require_recorded_at(value: str) -> None:
    if not isinstance(value, str) or _TIMESTAMP.fullmatch(value) is None:
        raise ForkPrecedentError("recorded_at_invalid")


def _require_text(value: Any, _field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ForkPrecedentError("stored_record_invalid")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return read_json_object(path, ForkPrecedentError)
