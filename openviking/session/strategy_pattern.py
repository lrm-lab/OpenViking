# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Branch steps, counterfactual strategy patterns, and the evidence ledger.

ADR-0043 P05. A Reader ``StrategyPatternProposal`` is input only. The server
derives ``pattern_id`` from the versioned controlled fields. Semantic
neighbors stay separate patterns; this module does not merge or rank them.

Factual steps must be blocks of the overlay's base revision, and imagined
nodes stay counterfactual and branch-local. Those checks are
``validate_v2_fork_grounding``. An imagined node file is written once.
Later evidence is an appended ``BranchEvaluation``, not a change to that file.
``BranchEvaluation.subject_id`` is only the node id, so an imagined-node id
cannot be reused on a second fork; that conflict keeps evidence on one branch.

Analogical evidence may record an estimated confidence. The ledger refuses
to store that evidence as production-direct or replay-direct. Retired rows
stay readable and are left out of ``precedent_candidates``. That list is not
ranked and is not ACL-filtered.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openviking.session.aig_native_protocol import (
    DIRECT_EVIDENCE_TIERS,
    EVALUATION_STATUS_ACTIVE,
    OBJECT_KIND_BRANCH_EVALUATION,
    OBJECT_KIND_DIRECT_EVIDENCE_ADMISSION,
    OBJECT_KIND_PRODUCTION_DIRECT_ADMISSION,
    SCHEMA_VERSION_AIG_NATIVE_V2,
    STEP_KIND_FACTUAL,
    STEP_KIND_IMAGINED,
    BaseRevisionCatalog,
    BranchEvaluation,
    CounterfactualStrategyPattern,
    DirectEvidenceAdmission,
    ForkNodeOverlay,
    ProductionDirectAdmission,
    ProtocolContractError,
    StrategyPatternProposal,
    canonical_dumps,
    protocol_content_hash,
    read_branch_evaluation,
    read_direct_evidence_admission,
    read_production_direct_admission,
    read_strategy_pattern,
    read_strategy_pattern_proposal,
    strategy_pattern_document,
    strategy_pattern_id,
    validate_v2_fork_grounding,
)
from openviking.session.append_only import (
    append_jsonl,
    read_json_object,
    read_jsonl_objects,
    write_new_json,
)
from openviking.session.fork_node_overlay import (
    StoredForkOverlay,
    fork_overlay_for,
)

STRATEGY_PATTERN_DIRNAME = "strategy-patterns"
_HEX_ID = re.compile(r"^[0-9a-f]{64}$")
_RECORD_BRANCH = "branch_snapshot"
_RECORD_IMAGINED = "imagined_node"
_RECORD_PATTERN = "strategy_pattern"
_RECORD_MEMBERSHIP = "pattern_membership"
_RECORD_EVALUATION = "branch_evaluation"


class StrategyPatternError(ValueError):
    """A pattern or evidence-ledger write was refused."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class FrozenImaginedNode:
    """Branch-local imagined node. Evidence does not live on this record."""

    imagined_node_id: str
    fork_node_id: str
    base_revision_id: str
    counterfactual: bool
    branch_local: bool


@dataclass(frozen=True)
class FrozenBranch:
    """Steps frozen from one overlay. Factual ids are refs, not copied blocks."""

    fork_node_id: str
    base_revision_id: str
    factual_block_ids: tuple[str, ...]
    imagined_nodes: tuple[FrozenImaginedNode, ...]


@dataclass(frozen=True)
class PatternMembership:
    """One imagined node accumulated onto a pattern. Not an identity field."""

    pattern_id: str
    fork_node_id: str
    imagined_node_id: str
    proposal_content_hash: str
    semantic_summary: str | None


@dataclass(frozen=True)
class PrecedentCandidate:
    """Read-only subject whose latest evaluation is not retired.

    The list is ordered by subject kind and id. It is not a relevance rank
    and it does not apply ACL.
    """

    subject_kind: str
    subject_id: str
    latest_status: str
    estimated_confidence: float | None
    has_direct_evidence: bool


class StrategyPatternStore:
    """Append-only pattern and evidence store under ``strategy-patterns``."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._dir = self._root / STRATEGY_PATTERN_DIRNAME
        self._index_path = self._dir / "index.jsonl"
        self._lock = threading.Lock()

    def freeze_branch(
        self,
        subject: ForkNodeOverlay | StoredForkOverlay,
        catalog: BaseRevisionCatalog,
    ) -> FrozenBranch:
        """Persist branch steps. Imagined nodes are written once."""
        overlay = _as_overlay(subject)
        with self._lock:
            return self._freeze_unlocked(overlay, catalog)

    def accept_proposal(
        self,
        subject: Any,
        catalog: BaseRevisionCatalog,
        payload: Mapping[str, Any],
    ) -> CounterfactualStrategyPattern:
        """Derive a pattern from a Reader proposal and accumulate an exact match.

        A second core for the same imagined node conflicts. A different
        semantic summary with the same core joins the existing pattern.
        """
        overlay = _as_overlay(subject)
        if not isinstance(payload, Mapping):
            raise StrategyPatternError("proposal_required")
        proposal = read_strategy_pattern_proposal(payload)
        validate_v2_fork_grounding(overlay, catalog)
        if proposal.fork_node_id != overlay.fork_node_id:
            raise StrategyPatternError("fork_mismatch")
        if proposal.imagined_node_id not in _imagined_ids(overlay):
            raise StrategyPatternError("imagined_node_not_on_branch")
        document = strategy_pattern_document(proposal)
        pattern = read_strategy_pattern(document)
        with self._lock:
            self._reject_rebind(proposal.imagined_node_id, pattern.pattern_id)
            self._freeze_unlocked(overlay, catalog)
            self._store_pattern(pattern, document)
            self._store_membership(proposal, pattern.pattern_id)
            return pattern

    def find_exact_pattern(
        self,
        *,
        intervention_type: str,
        target_boundary: str,
        mechanism: str,
        expected_metric: str,
        capability: str,
        applicability_conditions: Sequence[Mapping[str, str]],
    ) -> CounterfactualStrategyPattern | None:
        """Return the pattern with this exact core, or None. Does not create one."""
        pattern_id = strategy_pattern_id(
            intervention_type=intervention_type,
            target_boundary=target_boundary,
            mechanism=mechanism,
            expected_metric=expected_metric,
            capability=capability,
            applicability_conditions=applicability_conditions,
        )
        with self._lock:
            return self._read_pattern(pattern_id)

    def list_patterns(self) -> tuple[CounterfactualStrategyPattern, ...]:
        """Stored patterns in id order. No scores and no neighbor merge."""
        with self._lock:
            patterns: list[CounterfactualStrategyPattern] = []
            seen: set[str] = set()
            for row in self._read_index():
                if row.get("record_kind") != _RECORD_PATTERN:
                    continue
                pattern_id = row.get("pattern_id")
                if not isinstance(pattern_id, str) or pattern_id in seen:
                    continue
                seen.add(pattern_id)
                pattern = self._read_pattern(pattern_id)
                if pattern is None:
                    raise StrategyPatternError("stored_record_invalid")
                patterns.append(pattern)
            patterns.sort(key=lambda pattern: pattern.pattern_id)
            return tuple(patterns)

    def memberships(self, pattern_id: str) -> tuple[PatternMembership, ...]:
        """Imagined nodes accumulated onto ``pattern_id``, in append order."""
        with self._lock:
            found: list[PatternMembership] = []
            for row in self._read_index():
                if row.get("record_kind") != _RECORD_MEMBERSHIP:
                    continue
                if row.get("pattern_id") != pattern_id:
                    continue
                summary = row.get("semantic_summary")
                if summary is not None and not isinstance(summary, str):
                    raise StrategyPatternError("stored_record_invalid")
                found.append(
                    PatternMembership(
                        pattern_id=pattern_id,
                        fork_node_id=_row_str(row, "fork_node_id"),
                        imagined_node_id=_row_str(row, "imagined_node_id"),
                        proposal_content_hash=_row_str(row, "proposal_content_hash"),
                        semantic_summary=summary,
                    )
                )
            return tuple(found)

    def append_evaluation(
        self,
        payload: Mapping[str, Any],
        *,
        estimated_confidence: float | None = None,
        admission: Mapping[str, Any] | None = None,
    ) -> BranchEvaluation:
        """Append one BranchEvaluation. The same id cannot change tier.

        A direct tier must carry a ``direct_evidence_admission`` or a
        ``production_direct_admission`` document. The ledger re-reads that
        document before writing. A direct tier whose subject or tier does
        not match the admission is refused. A production admission matches
        only an active imagined-node row; a negative status is not a
        refutation this gate will store. Analogical, observational, and
        model-only rows do not need an admission.
        """
        if not isinstance(payload, Mapping):
            raise StrategyPatternError("evaluation_required")
        evaluation = read_branch_evaluation(payload)
        confidence = _confidence_for(evaluation, estimated_confidence)
        with self._lock:
            return self._append_unlocked(evaluation, payload, confidence, admission)

    def append_similar_fork_evidence(
        self,
        *,
        evaluation_id: str,
        subject_kind: str,
        subject_id: str,
        evidence_ref: str,
        estimated_confidence: float,
        evidence_tier: str = "analogical",
        status: str = "active",
    ) -> BranchEvaluation:
        """Append analogical evidence and record estimated confidence.

        A direct tier is refused before any write. The stored tier stays
        analogical.
        """
        if evidence_tier in DIRECT_EVIDENCE_TIERS:
            raise StrategyPatternError("analogical_promotion_forbidden")
        if evidence_tier != "analogical":
            raise StrategyPatternError("analogical_tier_required")
        confidence = _require_confidence(estimated_confidence)
        body = {
            "evaluation_id": evaluation_id,
            "evidence_ref": evidence_ref,
            "evidence_tier": "analogical",
            "status": status,
            "subject_id": subject_id,
            "subject_kind": subject_kind,
        }
        envelope = {
            "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
            "object_kind": OBJECT_KIND_BRANCH_EVALUATION,
            "body": body,
            "content_hash": protocol_content_hash(
                SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_BRANCH_EVALUATION, body
            ),
        }
        return self.append_evaluation(envelope, estimated_confidence=confidence)

    def estimated_confidence(self, subject_kind: str, subject_id: str) -> float | None:
        """Latest analogical confidence for the subject. Direct rows do not set it."""
        with self._lock:
            found: float | None = None
            for row in self._evaluation_rows(subject_kind, subject_id):
                if "estimated_confidence" not in row:
                    continue
                found = _stored_confidence(row.get("estimated_confidence"))
            return found

    def imagined_subject_known(self, imagined_node_id: str) -> bool:
        """True when this imagined node was frozen onto a branch."""
        with self._lock:
            return self._subject_known("imagined_node", imagined_node_id)

    def has_direct_evidence(self, subject_kind: str, subject_id: str) -> bool:
        """True only when a non-retired direct tier is already appended."""
        with self._lock:
            return any(
                _row_counts_as_direct(row)
                for row in self._evaluation_rows(subject_kind, subject_id)
            )

    def precedent_candidates(self) -> tuple[PrecedentCandidate, ...]:
        """Subjects whose latest evaluation is not retired.

        Retired rows remain in ``audit_evaluations``. This method does not
        rank and does not take an ACL principal.
        """
        with self._lock:
            grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for row in self._read_index():
                if row.get("record_kind") != _RECORD_EVALUATION:
                    continue
                kind = row.get("subject_kind")
                subject_id = row.get("subject_id")
                if not isinstance(kind, str) or not isinstance(subject_id, str):
                    raise StrategyPatternError("stored_record_invalid")
                grouped.setdefault((kind, subject_id), []).append(row)
            candidates: list[PrecedentCandidate] = []
            for key in sorted(grouped):
                rows = grouped[key]
                status = rows[-1].get("status")
                if not isinstance(status, str):
                    raise StrategyPatternError("stored_record_invalid")
                if status == "retired":
                    continue
                confidence: float | None = None
                for row in rows:
                    if "estimated_confidence" in row:
                        confidence = _stored_confidence(row.get("estimated_confidence"))
                has_direct = any(_row_counts_as_direct(row) for row in rows)
                candidates.append(
                    PrecedentCandidate(
                        subject_kind=key[0],
                        subject_id=key[1],
                        latest_status=status,
                        estimated_confidence=confidence,
                        has_direct_evidence=has_direct,
                    )
                )
            return tuple(candidates)

    def audit_evaluations(self) -> tuple[BranchEvaluation, ...]:
        """Every appended evaluation, including retired rows, in append order."""
        with self._lock:
            found: list[BranchEvaluation] = []
            for row in self._read_index():
                if row.get("record_kind") != _RECORD_EVALUATION:
                    continue
                evaluation_id = row.get("evaluation_id")
                if not isinstance(evaluation_id, str):
                    raise StrategyPatternError("stored_record_invalid")
                evaluation = self._load_evaluation(evaluation_id)
                if evaluation.content_hash != row.get("content_hash"):
                    raise StrategyPatternError("stored_record_invalid")
                found.append(evaluation)
            return tuple(found)

    def read_evaluation(self, evaluation_id: str) -> BranchEvaluation | None:
        """Return one ledger row, including a retired row. Unknown ids are None."""
        with self._lock:
            rows = [
                row
                for row in self._read_index()
                if row.get("record_kind") == _RECORD_EVALUATION
                and row.get("evaluation_id") == evaluation_id
            ]
            if not rows:
                return None
            evaluation = self._load_evaluation(evaluation_id)
            if evaluation.content_hash != rows[0].get("content_hash"):
                raise StrategyPatternError("stored_record_invalid")
            return evaluation

    def _freeze_unlocked(
        self, overlay: ForkNodeOverlay, catalog: BaseRevisionCatalog
    ) -> FrozenBranch:
        validate_v2_fork_grounding(overlay, catalog)
        factual, imagined = _classify_steps(overlay)
        if len(imagined) != len(set(imagined)):
            raise StrategyPatternError("imagined_node_conflict")
        self._reject_foreign_imagined_ids(overlay.fork_node_id, imagined)
        snapshot = _branch_snapshot(overlay, factual, imagined)
        existing = self._read_if_present(self._branch_path(overlay.fork_node_id))
        if existing is not None:
            if existing != snapshot:
                raise StrategyPatternError("branch_conflict")
            for node_id in imagined:
                stored = self._read_if_present(self._imagined_path(overlay.fork_node_id, node_id))
                if stored is None:
                    raise StrategyPatternError("stored_record_invalid")
                if stored != _imagined_record(overlay, node_id):
                    raise StrategyPatternError("imagined_node_conflict")
            return _frozen_branch(overlay, factual, imagined)
        self._write_new(self._branch_path(overlay.fork_node_id), snapshot)
        for node_id in imagined:
            self._write_new(
                self._imagined_path(overlay.fork_node_id, node_id),
                _imagined_record(overlay, node_id),
            )
        self._append_index(
            {
                "base_revision_id": overlay.base_revision_id,
                "fork_node_id": overlay.fork_node_id,
                "record_kind": _RECORD_BRANCH,
            }
        )
        for node_id in imagined:
            self._append_index(
                {
                    "base_revision_id": overlay.base_revision_id,
                    "fork_node_id": overlay.fork_node_id,
                    "imagined_node_id": node_id,
                    "record_kind": _RECORD_IMAGINED,
                }
            )
        return _frozen_branch(overlay, factual, imagined)

    def _reject_foreign_imagined_ids(self, fork_node_id: str, imagined: Sequence[str]) -> None:
        for row in self._read_index():
            if row.get("record_kind") != _RECORD_IMAGINED:
                continue
            if row.get("imagined_node_id") not in imagined:
                continue
            if row.get("fork_node_id") != fork_node_id:
                raise StrategyPatternError("imagined_node_conflict")

    def _reject_rebind(self, imagined_node_id: str, pattern_id: str) -> None:
        for row in self._read_index():
            if row.get("record_kind") != _RECORD_MEMBERSHIP:
                continue
            if row.get("imagined_node_id") != imagined_node_id:
                continue
            if row.get("pattern_id") != pattern_id:
                raise StrategyPatternError("pattern_binding_conflict")

    def _store_pattern(
        self, pattern: CounterfactualStrategyPattern, document: Mapping[str, Any]
    ) -> None:
        path = self._pattern_path(pattern.pattern_id)
        if path is None:
            raise StrategyPatternError("stored_record_invalid")
        if path.is_file():
            if _load_pattern(path) != pattern:
                raise StrategyPatternError("pattern_conflict")
            return
        self._write_new(path, document)
        self._append_index(
            {
                "content_hash": pattern.content_hash,
                "pattern_id": pattern.pattern_id,
                "record_kind": _RECORD_PATTERN,
            }
        )

    def _store_membership(self, proposal: StrategyPatternProposal, pattern_id: str) -> None:
        for row in self._read_index():
            if row.get("record_kind") != _RECORD_MEMBERSHIP:
                continue
            same = (
                row.get("pattern_id") == pattern_id
                and row.get("fork_node_id") == proposal.fork_node_id
                and row.get("imagined_node_id") == proposal.imagined_node_id
                and row.get("proposal_content_hash") == proposal.content_hash
            )
            if same:
                return
        row: dict[str, Any] = {
            "fork_node_id": proposal.fork_node_id,
            "imagined_node_id": proposal.imagined_node_id,
            "pattern_id": pattern_id,
            "proposal_content_hash": proposal.content_hash,
            "record_kind": _RECORD_MEMBERSHIP,
        }
        if proposal.semantic_summary is not None:
            row["semantic_summary"] = proposal.semantic_summary
        self._append_index(row)

    def _append_unlocked(
        self,
        evaluation: BranchEvaluation,
        payload: Mapping[str, Any],
        confidence: float | None,
        admission: Mapping[str, Any] | None,
    ) -> BranchEvaluation:
        if not self._subject_known(evaluation.subject_kind, evaluation.subject_id):
            raise StrategyPatternError("subject_unknown")
        existing = [
            row
            for row in self._read_index()
            if row.get("record_kind") == _RECORD_EVALUATION
            and row.get("evaluation_id") == evaluation.evaluation_id
        ]
        if existing:
            if any(row.get("content_hash") != evaluation.content_hash for row in existing):
                raise StrategyPatternError("evaluation_conflict")
            if any(not _confidence_matches(row, confidence) for row in existing):
                raise StrategyPatternError("evaluation_conflict")
            return evaluation
        if evaluation.evidence_tier in DIRECT_EVIDENCE_TIERS:
            parsed = _parsed_admission(admission)
            if parsed is None:
                raise StrategyPatternError("direct_admission_required")
            if not _admission_matches(evaluation, parsed):
                raise StrategyPatternError("direct_admission_mismatch")
        if evaluation.evidence_tier in DIRECT_EVIDENCE_TIERS and self._analogical_evidence(
            evaluation.evidence_ref
        ):
            raise StrategyPatternError("analogical_promotion_forbidden")
        self._write_new(self._evaluation_path(evaluation.evaluation_id), payload)
        row = {
            "content_hash": evaluation.content_hash,
            "evaluation_id": evaluation.evaluation_id,
            "evidence_ref": evaluation.evidence_ref,
            "evidence_tier": evaluation.evidence_tier,
            "record_kind": _RECORD_EVALUATION,
            "status": evaluation.status,
            "subject_id": evaluation.subject_id,
            "subject_kind": evaluation.subject_kind,
        }
        if confidence is not None:
            row["estimated_confidence"] = confidence
        self._append_index(row)
        return evaluation

    def _subject_known(self, subject_kind: str, subject_id: str) -> bool:
        if subject_kind == "imagined_node":
            return any(
                row.get("record_kind") == _RECORD_IMAGINED
                and row.get("imagined_node_id") == subject_id
                for row in self._read_index()
            )
        if subject_kind == "strategy_pattern":
            path = self._pattern_path(subject_id)
            return path is not None and path.is_file()
        return False

    def _analogical_evidence(self, evidence_ref: str) -> bool:
        return any(
            row.get("record_kind") == _RECORD_EVALUATION
            and row.get("evidence_ref") == evidence_ref
            and row.get("evidence_tier") == "analogical"
            for row in self._read_index()
        )

    def _evaluation_rows(self, subject_kind: str, subject_id: str) -> list[dict[str, Any]]:
        return [
            row
            for row in self._read_index()
            if row.get("record_kind") == _RECORD_EVALUATION
            and row.get("subject_kind") == subject_kind
            and row.get("subject_id") == subject_id
        ]

    def _read_pattern(self, pattern_id: str) -> CounterfactualStrategyPattern | None:
        path = self._pattern_path(pattern_id)
        if path is None or not path.is_file():
            return None
        pattern = _load_pattern(path)
        if pattern.pattern_id != pattern_id:
            raise StrategyPatternError("stored_record_invalid")
        return pattern

    def _load_evaluation(self, evaluation_id: str) -> BranchEvaluation:
        path = self._evaluation_path(evaluation_id)
        try:
            evaluation = read_branch_evaluation(_read_json(path))
        except ProtocolContractError as exc:
            raise StrategyPatternError("stored_record_invalid") from exc
        if evaluation.evaluation_id != evaluation_id:
            raise StrategyPatternError("stored_record_invalid")
        return evaluation

    def _read_index(self) -> list[dict[str, Any]]:
        return read_jsonl_objects(self._index_path, StrategyPatternError)

    def _read_if_present(self, path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        return _read_json(path)

    def _branch_path(self, fork_node_id: str) -> Path:
        digest = hashlib.sha256(fork_node_id.encode("utf-8")).hexdigest()
        return self._dir / "branches" / f"{digest}.json"

    def _imagined_path(self, fork_node_id: str, imagined_node_id: str) -> Path:
        digest = hashlib.sha256(
            canonical_dumps(
                {"fork_node_id": fork_node_id, "imagined_node_id": imagined_node_id}
            ).encode("utf-8")
        ).hexdigest()
        return self._dir / "imagined" / f"{digest}.json"

    def _pattern_path(self, pattern_id: str) -> Path | None:
        if not isinstance(pattern_id, str) or _HEX_ID.fullmatch(pattern_id) is None:
            return None
        return self._dir / "patterns" / f"{pattern_id}.json"

    def _evaluation_path(self, evaluation_id: str) -> Path:
        digest = hashlib.sha256(evaluation_id.encode("utf-8")).hexdigest()
        return self._dir / "evaluations" / f"{digest}.json"

    def _append_index(self, row: Mapping[str, Any]) -> None:
        append_jsonl(self._index_path, row)

    def _write_new(self, path: Path, payload: Mapping[str, Any]) -> None:
        write_new_json(path, payload, StrategyPatternError)


def _as_overlay(subject: Any) -> ForkNodeOverlay:
    return fork_overlay_for(subject, "pattern", StrategyPatternError)


def _imagined_ids(overlay: ForkNodeOverlay) -> set[str]:
    return {
        step.imagined_node_id
        for step in overlay.branch_steps
        if step.step_kind == STEP_KIND_IMAGINED and isinstance(step.imagined_node_id, str)
    }


def _classify_steps(overlay: ForkNodeOverlay) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not overlay.branch_steps:
        raise StrategyPatternError("branch_required")
    factual: list[str] = []
    imagined: list[str] = []
    for step in overlay.branch_steps:
        if step.step_kind == STEP_KIND_FACTUAL:
            if not isinstance(step.block_id, str) or not step.block_id:
                raise StrategyPatternError("branch_step_invalid")
            factual.append(step.block_id)
        elif step.step_kind == STEP_KIND_IMAGINED:
            if not isinstance(step.imagined_node_id, str) or not step.imagined_node_id:
                raise StrategyPatternError("branch_step_invalid")
            imagined.append(step.imagined_node_id)
        else:
            raise StrategyPatternError("branch_step_invalid")
    return tuple(factual), tuple(imagined)


def _branch_snapshot(
    overlay: ForkNodeOverlay, factual: Sequence[str], imagined: Sequence[str]
) -> dict[str, Any]:
    return {
        "anchor_block_id": overlay.anchor_block_id,
        "base_revision_block_content_hash": overlay.base_revision_block_content_hash,
        "base_revision_id": overlay.base_revision_id,
        "factual_block_ids": list(factual),
        "fork_node_id": overlay.fork_node_id,
        "imagined_node_ids": list(imagined),
    }


def _imagined_record(overlay: ForkNodeOverlay, imagined_node_id: str) -> dict[str, Any]:
    return {
        "base_revision_id": overlay.base_revision_id,
        "branch_local": True,
        "counterfactual": True,
        "fork_node_id": overlay.fork_node_id,
        "imagined_node_id": imagined_node_id,
        "step_kind": STEP_KIND_IMAGINED,
    }


def _frozen_branch(
    overlay: ForkNodeOverlay, factual: Sequence[str], imagined: Sequence[str]
) -> FrozenBranch:
    nodes = tuple(
        FrozenImaginedNode(
            imagined_node_id=node_id,
            fork_node_id=overlay.fork_node_id,
            base_revision_id=overlay.base_revision_id,
            counterfactual=True,
            branch_local=True,
        )
        for node_id in imagined
    )
    return FrozenBranch(
        fork_node_id=overlay.fork_node_id,
        base_revision_id=overlay.base_revision_id,
        factual_block_ids=tuple(factual),
        imagined_nodes=nodes,
    )


def _confidence_for(
    evaluation: BranchEvaluation, estimated_confidence: float | None
) -> float | None:
    if estimated_confidence is None:
        return None
    confidence = _require_confidence(estimated_confidence)
    if evaluation.evidence_tier in DIRECT_EVIDENCE_TIERS:
        raise StrategyPatternError("analogical_promotion_forbidden")
    if evaluation.evidence_tier != "analogical":
        raise StrategyPatternError("analogical_tier_required")
    return confidence


def _require_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrategyPatternError("invalid_confidence")
    if value != value or value < 0 or value > 1:
        raise StrategyPatternError("invalid_confidence")
    return float(value)


def _stored_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrategyPatternError("stored_record_invalid")
    if value != value or value < 0 or value > 1:
        raise StrategyPatternError("stored_record_invalid")
    return float(value)


def _confidence_matches(row: Mapping[str, Any], confidence: float | None) -> bool:
    if "estimated_confidence" not in row:
        return confidence is None
    if confidence is None:
        return False
    return _stored_confidence(row.get("estimated_confidence")) == confidence


def _row_str(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise StrategyPatternError("stored_record_invalid")
    return value


def _load_pattern(path: Path) -> CounterfactualStrategyPattern:
    try:
        return read_strategy_pattern(_read_json(path))
    except ProtocolContractError as exc:
        raise StrategyPatternError("stored_record_invalid") from exc


def _read_json(path: Path) -> dict[str, Any]:
    return read_json_object(path, StrategyPatternError)


def _row_counts_as_direct(row: Mapping[str, Any]) -> bool:
    return row.get("evidence_tier") in DIRECT_EVIDENCE_TIERS and row.get("status") != "retired"


def _admission_matches(
    evaluation: BranchEvaluation,
    admission: DirectEvidenceAdmission | ProductionDirectAdmission,
) -> bool:
    """True when the admission's subject and tier are this direct row."""
    if admission.evidence_tier != evaluation.evidence_tier:
        return False
    if isinstance(admission, ProductionDirectAdmission):
        return (
            evaluation.status == EVALUATION_STATUS_ACTIVE
            and evaluation.subject_kind == "imagined_node"
            and admission.imagined_node_id == evaluation.subject_id
            and bool(admission.actual_block_id)
        )
    if evaluation.subject_kind == "imagined_node":
        return (
            admission.first_action_direct
            and admission.imagined_node_id == evaluation.subject_id
        )
    if evaluation.subject_kind == "strategy_pattern":
        return admission.outcome_direct
    return False


def _parsed_admission(
    admission: Mapping[str, Any] | None,
) -> DirectEvidenceAdmission | ProductionDirectAdmission | None:
    """Re-read the admission document. A bare object is not a verified record."""
    if not isinstance(admission, Mapping):
        return None
    kind = admission.get("object_kind")
    try:
        if kind == OBJECT_KIND_DIRECT_EVIDENCE_ADMISSION:
            return read_direct_evidence_admission(admission)
        if kind == OBJECT_KIND_PRODUCTION_DIRECT_ADMISSION:
            return read_production_direct_admission(admission)
    except ProtocolContractError:
        return None
    return None
