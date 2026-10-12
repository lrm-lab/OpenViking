# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Read-time experience-network edges (ADR-0044).

Edges are a typed view of append-only records. This module has no edge
table and no write API. A new link type requires an ADR; the closed set is
``anchors``, ``factual_ref``, ``supports``, ``refutes``, ``instance_of``,
``has_member``, ``guides``, and ``executed_as``.

``anchors`` runs from a fork overlay to its P04 anchor block. ``factual_ref``
runs from that fork to each factual branch step's block. ``supports`` and
``refutes`` run from the latest BranchEvaluation to its subject: ``active``
and ``retired`` support, ``challenged``, ``invalidated``, and ``superseded``
refute. ``instance_of`` runs from an imagined node to its P05 pattern;
``has_member`` is the opposite direction. ``guides`` runs from an approved
guidance id to its fork. ``executed_as`` runs from a GuidanceExecutionReport
to an imagined node whose alignment is not ``not_executed``.

Each hop rechecks ACL with the same non-empty intersection rule as
``retrieval_facade._labels_visible``. Replay identities (``replay-rev:``,
``replay-session:``, ``replay-run:``, ``replay-view:``) are not hop targets.
One retrieval context allows at most 8 hops and 32 cards by default.
Memory uses 4 and 16 and drops ``retired``. Exceeding either limit refuses
the call with ``hop_budget_exhausted`` and does not return a truncated page.
``retired`` stays reachable on the reader profile with ``status: retired``.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openviking.session.aig_native_protocol import (
    ALIGNMENT_NOT_EXECUTED,
    REPLAY_REVISION_PREFIX,
    REPLAY_SESSION_PREFIX,
    REPLAY_TASK_RUN_PREFIX,
    REPLAY_VIEW_PREFIX,
    STEP_KIND_FACTUAL,
    GuidanceApproval,
    GuidanceExecutionReport,
    ProtocolContractError,
    canonical_dumps,
    read_guidance_execution_report,
)
from openviking.session.append_only import append_jsonl, read_json_object
from openviking.session.fork_node_overlay import (
    ForkNodeOverlayStore,
    ForkOverlayError,
    StoredForkOverlay,
    require_label_set,
)
from openviking.session.guidance_approval import (
    GUIDANCE_APPROVAL_DIRNAME,
    GuidanceApprovalError,
    GuidanceApprovalService,
)
from openviking.session.remediation_dispatch import REMEDIATION_RUN_DIRNAME
from openviking.session.strategy_pattern import StrategyPatternStore

EXPERIENCE_LINK_AUDIT_DIRNAME = "experience-link-audit"
PROFILE_READER = "reader"
PROFILE_MEMORY = "memory"
DEFAULT_READER_MAX_HOPS = 8
DEFAULT_READER_MAX_CARDS = 32
DEFAULT_MEMORY_MAX_HOPS = 4
DEFAULT_MEMORY_MAX_CARDS = 16
LINK_TYPES = (
    "anchors",
    "factual_ref",
    "supports",
    "refutes",
    "instance_of",
    "has_member",
    "guides",
    "executed_as",
)
KIND_FORK = "fork_node"
KIND_BLOCK = "block"
KIND_IMAGINED = "imagined_node"
KIND_PATTERN = "strategy_pattern"
KIND_EVALUATION = "branch_evaluation"
KIND_GUIDANCE = "guidance"
KIND_REPORT = "execution_report"
_LINK_SET = frozenset(LINK_TYPES)
_REFUTE_STATUSES = frozenset({"challenged", "invalidated", "superseded"})
_SUPPORT_STATUSES = frozenset({"active", "retired"})
_REPLAY_PREFIXES = (
    REPLAY_REVISION_PREFIX,
    REPLAY_SESSION_PREFIX,
    REPLAY_TASK_RUN_PREFIX,
    REPLAY_VIEW_PREFIX,
)
_SUBJECT_KIND = {"imagined_node": KIND_IMAGINED, "strategy_pattern": KIND_PATTERN}


class ExperienceLinkError(ValueError):
    """A hop was refused. ``code`` is stable."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class _Edge:
    link_type: str
    source_id: str
    target_id: str
    source_kind: str
    target_kind: str
    status: str


@dataclass
class _Context:
    context_id: str
    profile: str
    hops: int
    cards: int
    max_hops: int
    max_cards: int
    include_retired: bool
    exhausted: bool
    unresolved: tuple[str, ...]


def _labels_visible(labels: Sequence[str], principal: set[str]) -> bool:
    """Same intersection rule as ``retrieval_facade._labels_visible``."""
    return bool(set(labels).intersection(principal))


def _replay_identity(value: object) -> bool:
    return isinstance(value, str) and any(value.startswith(prefix) for prefix in _REPLAY_PREFIXES)


def _link_for_status(status: str) -> str | None:
    if status in _REFUTE_STATUSES:
        return "refutes"
    if status in _SUPPORT_STATUSES:
        return "supports"
    return None


class ExperienceLinkService:
    """One-hop expander over the append-only fork, pattern, and guidance logs."""

    def __init__(
        self,
        root: str | Path,
        *,
        overlays: ForkNodeOverlayStore,
        patterns: StrategyPatternStore,
        guidance: GuidanceApprovalService | None = None,
        reader_max_hops: int = DEFAULT_READER_MAX_HOPS,
        reader_max_cards: int = DEFAULT_READER_MAX_CARDS,
        memory_max_hops: int = DEFAULT_MEMORY_MAX_HOPS,
        memory_max_cards: int = DEFAULT_MEMORY_MAX_CARDS,
    ) -> None:
        if not isinstance(overlays, ForkNodeOverlayStore):
            raise ExperienceLinkError("store_required")
        if not isinstance(patterns, StrategyPatternStore):
            raise ExperienceLinkError("store_required")
        if guidance is not None and not isinstance(guidance, GuidanceApprovalService):
            raise ExperienceLinkError("store_required")
        self._root = Path(root)
        self._overlays = overlays
        self._patterns = patterns
        self._guidance = guidance
        self._limits = {
            PROFILE_READER: (_budget(reader_max_hops), _budget(reader_max_cards), True),
            PROFILE_MEMORY: (_budget(memory_max_hops), _budget(memory_max_cards), False),
        }
        self._contexts: dict[str, _Context] = {}
        self._seq = 0
        self._prev = ""
        self._audit_seq = 0
        self._audit_path = self._root / EXPERIENCE_LINK_AUDIT_DIRNAME / "hops.jsonl"
        self._lock = threading.Lock()

    def expand(
        self,
        subject_id: str,
        *,
        link_types: Sequence[str] | None = None,
        principal_labels: Sequence[str],
        profile: str = PROFILE_READER,
        context_id: str | None = None,
        unresolved: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Return one hop of derived edges and peer cards.

        Pass ``context_id`` back to keep the same retrieval budget. Omitting
        it opens a new context. ``unresolved`` is recorded on the audit row.
        """
        if not isinstance(subject_id, str) or not subject_id:
            raise ExperienceLinkError("subject_required")
        chosen = _parse_link_types(link_types)
        recorded = _parse_unresolved(unresolved)
        try:
            principal = require_label_set(principal_labels, code="acl_denied")
        except ForkOverlayError as exc:
            raise ExperienceLinkError(exc.code) from exc
        with self._lock:
            context = self._context(context_id, profile, recorded)
            if _replay_identity(subject_id):
                self._audit(
                    context,
                    subject_id,
                    chosen,
                    recorded,
                    ok=False,
                    error="replay_provenance_excluded",
                    cards=0,
                )
                raise ExperienceLinkError("replay_provenance_excluded")
            edges = self._incident(subject_id, chosen, principal, context.include_retired)
            if (
                context.exhausted
                or context.hops + 1 > context.max_hops
                or context.cards + len(edges) > context.max_cards
            ):
                context.exhausted = True
                self._audit(
                    context,
                    subject_id,
                    chosen,
                    recorded,
                    ok=False,
                    error="hop_budget_exceeded",
                    cards=0,
                )
                raise ExperienceLinkError("hop_budget_exceeded")
            context.hops += 1
            context.cards += len(edges)
            context.unresolved = recorded
            self._audit(context, subject_id, chosen, recorded, ok=True, error="", cards=len(edges))
            return {
                "cards": context.cards,
                "context_id": context.context_id,
                "edges": [_card(subject_id, edge) for edge in edges],
                "hops": context.hops,
                "profile": context.profile,
            }

    def _context(
        self, context_id: str | None, profile: str, unresolved: tuple[str, ...]
    ) -> _Context:
        if profile not in self._limits:
            raise ExperienceLinkError("unknown_profile")
        if context_id is None:
            self._seq += 1
            hops, cards, retired = self._limits[profile]
            created = _Context(
                context_id=hashlib.sha256(f"{profile}:{self._seq}".encode()).hexdigest(),
                profile=profile,
                hops=0,
                cards=0,
                max_hops=hops,
                max_cards=cards,
                include_retired=retired,
                exhausted=False,
                unresolved=unresolved,
            )
            self._contexts[created.context_id] = created
            return created
        if not isinstance(context_id, str) or context_id not in self._contexts:
            raise ExperienceLinkError("context_not_found")
        found = self._contexts[context_id]
        if found.profile != profile:
            raise ExperienceLinkError("context_mismatch")
        return found

    def _incident(
        self,
        subject_id: str,
        link_types: frozenset[str],
        principal: frozenset[str],
        include_retired: bool,
    ) -> list[_Edge]:
        rows = [
            edge
            for edge in self._derive(principal)
            if edge.link_type in link_types
            and (subject_id == edge.source_id or subject_id == edge.target_id)
        ]
        if not include_retired:
            rows = [edge for edge in rows if edge.status != "retired"]
        rows.sort(key=lambda edge: (edge.link_type, edge.source_id, edge.target_id, edge.status))
        return rows

    def _derive(self, principal: frozenset[str]) -> list[_Edge]:
        overlays = self._visible_overlays(principal)
        imagined_fork: dict[str, str] = {}
        edges: list[_Edge] = []
        for stored in overlays:
            for step in stored.overlay.branch_steps:
                if step.step_kind == STEP_KIND_FACTUAL and isinstance(step.block_id, str):
                    if _replay_identity(step.block_id):
                        continue
                    edges.append(
                        _Edge(
                            "factual_ref",
                            stored.fork_node_id,
                            step.block_id,
                            KIND_FORK,
                            KIND_BLOCK,
                            "active",
                        )
                    )
                elif (
                    isinstance(step.imagined_node_id, str)
                    and not _replay_identity(step.imagined_node_id)
                ):
                    imagined_fork[step.imagined_node_id] = stored.fork_node_id
            anchor = stored.overlay.anchor_block_id
            if not _replay_identity(anchor):
                edges.append(
                    _Edge(
                        "anchors",
                        stored.fork_node_id,
                        anchor,
                        KIND_FORK,
                        KIND_BLOCK,
                        "active",
                    )
                )
        latest = _latest_evaluations(self._patterns)
        visible_patterns = _visible_patterns(self._patterns, imagined_fork)
        for (subject_kind, subject_id), row in latest.items():
            peer_kind = _SUBJECT_KIND.get(subject_kind)
            if peer_kind is None:
                continue
            if subject_kind == "imagined_node" and subject_id not in imagined_fork:
                continue
            if subject_kind == "strategy_pattern" and subject_id not in visible_patterns:
                continue
            link_type = _link_for_status(row.status)
            if link_type is None or _replay_identity(row.evaluation_id):
                continue
            edges.append(
                _Edge(
                    link_type,
                    row.evaluation_id,
                    subject_id,
                    KIND_EVALUATION,
                    peer_kind,
                    row.status,
                )
            )
        for pattern_id in sorted(visible_patterns):
            if _replay_identity(pattern_id):
                continue
            for member in self._patterns.memberships(pattern_id):
                imagined = member.imagined_node_id
                if imagined not in imagined_fork or _replay_identity(imagined):
                    continue
                status = _node_status(latest, imagined)
                edges.append(
                    _Edge(
                        "instance_of",
                        imagined,
                        pattern_id,
                        KIND_IMAGINED,
                        KIND_PATTERN,
                        status,
                    )
                )
                edges.append(
                    _Edge(
                        "has_member",
                        pattern_id,
                        imagined,
                        KIND_PATTERN,
                        KIND_IMAGINED,
                        status,
                    )
                )
        fork_ids = {item.fork_node_id for item in overlays}
        approvals = self._approvals(set(principal), fork_ids)
        for approval in approvals:
            edges.append(
                _Edge(
                    "guides",
                    approval.guidance_id,
                    approval.source_fork_id,
                    KIND_GUIDANCE,
                    KIND_FORK,
                    "active",
                )
            )
        guidance_ids = {item.guidance_id for item in approvals}
        for report in self._reports():
            if report.guidance_id not in guidance_ids:
                continue
            if _replay_identity(report.parent_session_id) or _replay_identity(report.content_hash):
                continue
            if _replay_identity(report.source_fork_id):
                continue
            for step in report.steps:
                if step.alignment == ALIGNMENT_NOT_EXECUTED:
                    continue
                imagined = step.imagined_node_id
                if imagined not in imagined_fork or _replay_identity(imagined):
                    continue
                edges.append(
                    _Edge(
                        "executed_as",
                        report.content_hash,
                        imagined,
                        KIND_REPORT,
                        KIND_IMAGINED,
                        _node_status(latest, imagined),
                    )
                )
        return _dedupe(edges)

    def _visible_overlays(self, principal: frozenset[str]) -> list[StoredForkOverlay]:
        found: list[StoredForkOverlay] = []
        for stored in self._overlays.visible_overlays(principal_labels=tuple(sorted(principal))):
            if _replay_identity(stored.overlay.base_revision_id):
                continue
            if _replay_identity(stored.fork_node_id):
                continue
            found.append(stored)
        return found

    def _approvals(self, principal: set[str], fork_ids: set[str]) -> list[GuidanceApproval]:
        if self._guidance is None:
            return []
        directory = self._root / GUIDANCE_APPROVAL_DIRNAME / "approvals"
        if not directory.is_dir():
            return []
        found: list[GuidanceApproval] = []
        for path in sorted(directory.glob("*.json")):
            try:
                approval = self._guidance.read_approval(path.stem)
            except GuidanceApprovalError as exc:
                raise ExperienceLinkError(exc.code) from exc
            if approval is None:
                continue
            if _replay_identity(approval.guidance_id) or _replay_identity(approval.source_fork_id):
                continue
            if (
                _replay_identity(approval.base_revision_id)
                or approval.source_fork_id not in fork_ids
            ):
                continue
            try:
                basis = self._guidance.read_basis(approval.content_hash)
            except GuidanceApprovalError as exc:
                raise ExperienceLinkError(exc.code) from exc
            if not _labels_visible(basis.principal_labels, principal):
                continue
            found.append(approval)
        return found

    def _reports(self) -> list[GuidanceExecutionReport]:
        directory = self._root / REMEDIATION_RUN_DIRNAME / "reports"
        if not directory.is_dir():
            return []
        found: list[GuidanceExecutionReport] = []
        for path in sorted(directory.glob("*.json")):
            payload = read_json_object(path, ExperienceLinkError)
            try:
                found.append(read_guidance_execution_report(payload))
            except ProtocolContractError as exc:
                raise ExperienceLinkError("stored_record_invalid") from exc
        return found

    def _audit(
        self,
        context: _Context,
        subject_id: str,
        link_types: frozenset[str],
        unresolved: tuple[str, ...],
        *,
        ok: bool,
        error: str,
        cards: int,
    ) -> None:
        body = {
            "cards": cards,
            "context_id": context.context_id,
            "error_code": error,
            "hops": context.hops,
            "link_types": sorted(link_types),
            "ok": ok,
            "profile": context.profile,
            "subject_id": subject_id,
            "unresolved": list(unresolved),
        }
        payload: dict[str, Any] = {
            "body": body,
            "prev_content_hash": self._prev,
            "record_kind": "experience_link_hop",
            "seq": self._audit_seq,
        }
        digest = hashlib.sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()
        payload["content_hash"] = digest
        append_jsonl(self._audit_path, payload)
        self._prev = digest
        self._audit_seq += 1


def _budget(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ExperienceLinkError("invalid_budget")
    return value


def _parse_link_types(link_types: Sequence[str] | None) -> frozenset[str]:
    if link_types is None:
        return frozenset(LINK_TYPES)
    if isinstance(link_types, (str, bytes)) or not isinstance(link_types, Sequence):
        raise ExperienceLinkError("unknown_link_type")
    if any(not isinstance(item, str) or item not in _LINK_SET for item in link_types):
        raise ExperienceLinkError("unknown_link_type")
    return frozenset(link_types)


def _parse_unresolved(unresolved: Sequence[str]) -> tuple[str, ...]:
    if isinstance(unresolved, (str, bytes)) or not isinstance(unresolved, Sequence):
        raise ExperienceLinkError("unresolved_required")
    if any(not isinstance(item, str) for item in unresolved):
        raise ExperienceLinkError("unresolved_required")
    return tuple(unresolved)


def _latest_evaluations(store: StrategyPatternStore) -> dict[tuple[str, str], Any]:
    latest: dict[tuple[str, str], Any] = {}
    for item in store.audit_evaluations():
        if _replay_identity(item.evaluation_id) or _replay_identity(item.subject_id):
            continue
        if _replay_identity(item.evidence_ref):
            continue
        latest[(item.subject_kind, item.subject_id)] = item
    return latest


def _node_status(latest: Mapping[tuple[str, str], Any], imagined_id: str) -> str:
    row = latest.get(("imagined_node", imagined_id))
    if row is None:
        return "active"
    return str(row.status)


def _visible_patterns(store: StrategyPatternStore, imagined_fork: Mapping[str, str]) -> set[str]:
    found: set[str] = set()
    for pattern in store.list_patterns():
        if _replay_identity(pattern.pattern_id):
            continue
        for member in store.memberships(pattern.pattern_id):
            if (
                member.imagined_node_id in imagined_fork
                and not _replay_identity(member.imagined_node_id)
            ):
                found.add(pattern.pattern_id)
                break
    return found


def _dedupe(edges: list[_Edge]) -> list[_Edge]:
    seen: set[tuple[str, str, str]] = set()
    rows: list[_Edge] = []
    for edge in edges:
        key = (edge.link_type, edge.source_id, edge.target_id)
        if key in seen:
            continue
        seen.add(key)
        rows.append(edge)
    return rows


def _card(subject_id: str, edge: _Edge) -> dict[str, Any]:
    outgoing = subject_id == edge.source_id
    peer_id = edge.target_id if outgoing else edge.source_id
    peer_kind = edge.target_kind if outgoing else edge.source_kind
    return {
        "direction": "out" if outgoing else "in",
        "link_type": edge.link_type,
        "peer": {
            "acl_visible": True,
            "id": peer_id,
            "kind": peer_kind,
            "status": edge.status,
        },
        "source_id": edge.source_id,
        "status": edge.status,
        "target_id": edge.target_id,
    }
