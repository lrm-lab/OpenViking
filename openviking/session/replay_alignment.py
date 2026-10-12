# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Replay eligibility, an isolated Replay AIG, and first-action admission.

ADR-0043 P07. The server derives ReplayEligibility from controlled fields,
appends each re-judgment, and freezes replay output as its own native-edge
revision. That revision carries base-revision and replay-run provenance and
is stored under ``replay-alignment/``. It is not written to the production
native-edge store.

non_replayable judgments take the analogical route. That route may append
only analogical or model-only ledger rows. An analogical row is never
rewritten as direct evidence. This module does not write production-direct
evidence; a replay admission writes replay-direct only.

The Coforge sandbox replay worker is the implementer of ``replay_request``
and ``replay_result``. This module does not call it, open a socket, or start
a model. P09 and P12 attach a real worker. Until then,
``apply_scripted_replay`` consumes a local result document.
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
    OBJECT_KIND_BRANCH_EVALUATION,
    OUTCOME_ALIGNMENT_DEPTHS,
    REPLAY_SESSION_PREFIX,
    REPLAY_TASK_RUN_PREFIX,
    REPLAY_VIEW_PREFIX,
    ROUTE_ANALOGICAL,
    ROUTE_REPLAY,
    SCHEMA_VERSION_AIG_NATIVE_V2,
    STEP_KIND_IMAGINED,
    BranchEvaluation,
    DirectEvidenceAdmission,
    FirstActionAlignment,
    ForkNodeOverlay,
    ProtocolContractError,
    ReplayAigDocument,
    ReplayEligibilityAssessment,
    ReplayRequest,
    ReplayResult,
    direct_evidence_admission_document,
    judge_direct_admission,
    protocol_content_hash,
    read_branch_evaluation,
    read_direct_evidence_admission,
    read_replay_aig,
    read_replay_eligibility_assessment,
    read_replay_request,
    read_replay_result,
    replay_aig_document,
    replay_aig_revision_id,
    replay_eligibility_assessment_document,
    replay_request_document,
    replay_result_document,
)
from openviking.session.append_only import (
    append_jsonl,
    read_json_object,
    read_jsonl_objects,
    write_new_json,
)
from openviking.session.extraction_write import freeze_native_edge_view
from openviking.session.fork_node_overlay import StoredForkOverlay, fork_overlay_for
from openviking.session.influence_view import (
    AigNativeEdge,
    CoverageManifest,
    InfluenceBlock,
)
from openviking.session.strategy_pattern import StrategyPatternError, StrategyPatternStore

REPLAY_ALIGNMENT_DIRNAME = "replay-alignment"
_RECORD_ASSESSMENT = "replay_eligibility_assessment"
_ANALOGICAL_TIERS = frozenset({"analogical", "model-only"})
_HEX_ID = re.compile(r"^[0-9a-f]{64}$")


class ReplayAlignmentError(ValueError):
    """A replay judgment, freeze, or evidence admission was refused."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ScriptedReplayOutcome:
    """Local replay chain. ``replay_aig`` is absent when the result did not complete."""

    request: ReplayRequest
    result: ReplayResult
    replay_aig: ReplayAigDocument | None
    admission: DirectEvidenceAdmission | None
    evaluations: tuple[BranchEvaluation, ...]


class ReplayAlignmentService:
    """Append-only eligibility history and the isolated Replay AIG store."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._dir = self._root / REPLAY_ALIGNMENT_DIRNAME
        self._eligibility_index = self._dir / "eligibility" / "index.jsonl"
        self._lock = threading.RLock()

    def assess(
        self,
        overlay: ForkNodeOverlay | StoredForkOverlay,
        controls: Mapping[str, Any],
        *,
        evidence_ref: str,
        checked_at: str,
    ) -> ReplayEligibilityAssessment:
        """Append one judgment. The same evidence and controls return the first row."""
        subject = _as_overlay(overlay)
        if not isinstance(controls, Mapping):
            raise ReplayAlignmentError("controls_invalid")
        try:
            document = replay_eligibility_assessment_document(
                fork_node_id=subject.fork_node_id,
                base_revision_id=subject.base_revision_id,
                anchor_block_id=subject.anchor_block_id,
                evidence_ref=evidence_ref,
                checked_at=checked_at,
                controls=controls,
            )
            fresh = read_replay_eligibility_assessment(document)
        except ProtocolContractError as exc:
            raise ReplayAlignmentError("controls_invalid") from exc
        with self._lock:
            path = self._assessment_path(fresh.assessment_id)
            if path.is_file():
                return read_replay_eligibility_assessment(self._read_json(path))
            self._write_new(path, document)
            self._append_eligibility(
                {
                    "assessment_id": fresh.assessment_id,
                    "checked_at": fresh.checked_at,
                    "eligibility": fresh.eligibility,
                    "evidence_ref": fresh.evidence_ref,
                    "fork_node_id": fresh.fork_node_id,
                    "record_kind": _RECORD_ASSESSMENT,
                    "route": fresh.route,
                }
            )
            return fresh

    def history(self, fork_node_id: str) -> tuple[ReplayEligibilityAssessment, ...]:
        """Judgments for one fork, in append order. Retired rows are not deleted."""
        with self._lock:
            found: list[ReplayEligibilityAssessment] = []
            for row in self._read_index():
                if row.get("record_kind") != _RECORD_ASSESSMENT:
                    continue
                if row.get("fork_node_id") != fork_node_id:
                    continue
                assessment_id = row.get("assessment_id")
                if not isinstance(assessment_id, str) or not assessment_id:
                    raise ReplayAlignmentError("stored_record_invalid")
                found.append(
                    read_replay_eligibility_assessment(
                        self._read_json(self._assessment_path(assessment_id))
                    )
                )
            return tuple(found)

    def latest(self, fork_node_id: str) -> ReplayEligibilityAssessment | None:
        """The newest judgment, or None when the fork has not been assessed."""
        rows = self.history(fork_node_id)
        if not rows:
            return None
        return rows[-1]

    def build_replay_request(
        self, assessment: ReplayEligibilityAssessment, *, replay_run_id: str
    ) -> dict[str, Any]:
        """Seal a worker request. non_replayable stays on the analogical route."""
        if not isinstance(assessment, ReplayEligibilityAssessment):
            raise ReplayAlignmentError("assessment_required")
        if assessment.route != ROUTE_REPLAY or assessment.eligibility == "non_replayable":
            raise ReplayAlignmentError("analogical_route_required")
        try:
            return replay_request_document(assessment, replay_run_id=replay_run_id)
        except ProtocolContractError as exc:
            raise ReplayAlignmentError("replay_request_invalid") from exc

    def build_replay_result(
        self,
        request: Mapping[str, Any],
        *,
        status: str,
        observed_block_ids: Sequence[str] = (),
        alignment: Mapping[str, Any] | None = None,
        step_alignments: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Seal a local result. This does not contact the Coforge sandbox replay worker."""
        if not isinstance(request, Mapping):
            raise ReplayAlignmentError("replay_request_required")
        try:
            parsed = read_replay_request(request)
            return replay_result_document(
                parsed,
                status=status,
                observed_block_ids=observed_block_ids,
                alignment=alignment,
                step_alignments=step_alignments,
            )
        except ProtocolContractError as exc:
            raise ReplayAlignmentError("replay_result_invalid") from exc

    def append_analogical_route(
        self,
        overlay: ForkNodeOverlay | StoredForkOverlay,
        pattern_store: StrategyPatternStore,
        *,
        evaluation_id: str,
        subject_kind: str,
        subject_id: str,
        evidence_ref: str,
        evidence_tier: str = "analogical",
        estimated_confidence: float | None = None,
        status: str = "active",
    ) -> BranchEvaluation:
        """Append analogical or model-only evidence for the latest judgment.

        A non_replayable fork rejects every other tier. A replayable fork still
        rejects direct tiers here; those rows come only from an admission.
        """
        subject = _as_overlay(overlay)
        if not isinstance(pattern_store, StrategyPatternStore):
            raise ReplayAlignmentError("store_required")
        latest = self.latest(subject.fork_node_id)
        if latest is None:
            raise ReplayAlignmentError("assessment_required")
        if latest.route == ROUTE_ANALOGICAL and evidence_tier not in _ANALOGICAL_TIERS:
            raise ReplayAlignmentError("analogical_route_required")
        if evidence_tier in DIRECT_EVIDENCE_TIERS:
            raise ReplayAlignmentError("direct_admission_required")
        if evidence_tier not in _ANALOGICAL_TIERS:
            raise ReplayAlignmentError("analogical_tier_required")
        document = _evaluation_document(
            evaluation_id=evaluation_id,
            subject_kind=subject_kind,
            subject_id=subject_id,
            evidence_ref=evidence_ref,
            evidence_tier=evidence_tier,
            status=status,
        )
        try:
            return pattern_store.append_evaluation(
                document, estimated_confidence=estimated_confidence
            )
        except StrategyPatternError as exc:
            raise ReplayAlignmentError(exc.code) from exc

    def apply_scripted_replay(
        self,
        overlay: ForkNodeOverlay | StoredForkOverlay,
        request_payload: Mapping[str, Any],
        result_payload: Mapping[str, Any],
        *,
        blocks: Sequence[InfluenceBlock],
        edges: Sequence[AigNativeEdge],
        coverage: CoverageManifest,
        frozen_at: str,
        pattern_store: StrategyPatternStore,
        pattern_id: str | None,
    ) -> ScriptedReplayOutcome:
        """Drive eligibility's current request through a local result.

        A completed result freezes a Replay AIG, records the admission, and
        appends replay-direct rows only when the server gate allows them.
        Matching only the first action does not append an outcome row.
        """
        subject = _as_overlay(overlay)
        if not isinstance(pattern_store, StrategyPatternStore):
            raise ReplayAlignmentError("store_required")
        try:
            request = read_replay_request(request_payload)
            result = read_replay_result(result_payload)
        except ProtocolContractError as exc:
            raise ReplayAlignmentError("replay_result_invalid") from exc
        if (
            result.request_id != request.request_id
            or result.replay_run_id != request.replay_run_id
            or result.fork_node_id != request.fork_node_id
            or result.base_revision_id != request.base_revision_id
            or request.fork_node_id != subject.fork_node_id
            or request.base_revision_id != subject.base_revision_id
            or request.anchor_block_id != subject.anchor_block_id
        ):
            raise ReplayAlignmentError("replay_request_mismatch")
        latest = self.latest(subject.fork_node_id)
        if latest is None or latest.assessment_id != request.assessment_id:
            raise ReplayAlignmentError("assessment_mismatch")
        if latest.route != ROUTE_REPLAY or latest.eligibility == "non_replayable":
            raise ReplayAlignmentError("analogical_route_required")
        if request.eligibility != latest.eligibility:
            raise ReplayAlignmentError("assessment_mismatch")
        if result.status != "completed":
            return ScriptedReplayOutcome(request, result, None, None, ())
        alignment = result.first_action_alignment
        if alignment is None:
            raise ReplayAlignmentError("replay_result_invalid")
        block_ids = tuple(block.block_id for block in blocks)
        if block_ids != result.observed_block_ids:
            raise ReplayAlignmentError("observed_blocks_mismatch")
        first = _first_imagined(subject)
        decision = judge_direct_admission(
            alignment=alignment.alignment,
            depth=alignment.depth,
            intervention_type=alignment.intervention_type,
            target_boundary=alignment.target_boundary,
            required_capabilities=alignment.required_capabilities,
            safety_control_params=alignment.safety_control_params,
            expected_intervention_type=latest.intervention_type,
            expected_target_boundary=latest.target_boundary,
            expected_required_capabilities=latest.required_capabilities,
            expected_safety_control_params=latest.safety_control_params,
            equivalence_explanation=alignment.equivalence_explanation,
            route=latest.route,
            first_imagined_node=alignment.imagined_node_id == first,
            step_alignments=result.step_alignments,
            branch_steps=subject.branch_steps,
            replay_block_ids=block_ids,
        )
        if decision.outcome_direct:
            _require_outcome_pattern(pattern_store, pattern_id, latest)
        replay_aig = self._freeze_replay(
            assessment=latest,
            replay_run_id=request.replay_run_id,
            blocks=blocks,
            edges=edges,
            coverage=coverage,
            frozen_at=frozen_at,
        )
        revision_ids = tuple(block.block_id for block in replay_aig.revision.blocks)
        if revision_ids != block_ids:
            raise ReplayAlignmentError("observed_blocks_mismatch")
        if (
            alignment.actual_block_id is not None
            and alignment.actual_block_id not in set(revision_ids)
        ):
            raise ReplayAlignmentError("actual_block_not_in_replay_aig")
        admission, admission_document = self._store_admission(
            latest,
            request,
            alignment,
            replay_aig,
            first,
            step_alignments=result.step_alignments,
            branch_steps=subject.branch_steps,
            replay_block_ids=revision_ids,
        )
        evaluations: list[BranchEvaluation] = []
        if admission.first_action_direct:
            evaluations.append(
                self._append_replay_direct(
                    pattern_store,
                    admission_document,
                    subject_kind="imagined_node",
                    subject_id=alignment.imagined_node_id,
                    suffix="first-action",
                )
            )
        if admission.outcome_direct:
            if pattern_id is None:
                raise ReplayAlignmentError("outcome_subject_required")
            evaluations.append(
                self._append_replay_direct(
                    pattern_store,
                    admission_document,
                    subject_kind="strategy_pattern",
                    subject_id=pattern_id,
                    suffix="outcome",
                )
            )
        elif (
            alignment.depth in OUTCOME_ALIGNMENT_DEPTHS and admission.first_action_direct
        ):
            raise ReplayAlignmentError("outcome_alignment_unverified")
        return ScriptedReplayOutcome(
            request, result, replay_aig, admission, tuple(evaluations)
        )

    def read_replay(self, revision_id: str) -> ReplayAigDocument:
        """Return one stored Replay AIG. This is not a production query."""
        path = self._aig_path(revision_id)
        if not path.is_file():
            raise KeyError(revision_id)
        with self._lock:
            return read_replay_aig(self._read_json(path))

    def read_admission(self, admission_id: str) -> DirectEvidenceAdmission:
        """Return one stored admission, including its equivalence explanation."""
        path = self._admission_path(admission_id)
        if not path.is_file():
            raise KeyError(admission_id)
        with self._lock:
            return read_direct_evidence_admission(self._read_json(path))

    def _freeze_replay(
        self,
        *,
        assessment: ReplayEligibilityAssessment,
        replay_run_id: str,
        blocks: Sequence[InfluenceBlock],
        edges: Sequence[AigNativeEdge],
        coverage: CoverageManifest,
        frozen_at: str,
    ) -> ReplayAigDocument:
        revision_id = replay_aig_revision_id(
            base_revision_id=assessment.base_revision_id,
            replay_run_id=replay_run_id,
            source_fork_id=assessment.fork_node_id,
        )
        revision = freeze_native_edge_view(
            session_id=f"{REPLAY_SESSION_PREFIX}{replay_run_id}",
            task_run_id=f"{REPLAY_TASK_RUN_PREFIX}{replay_run_id}",
            view_id=f"{REPLAY_VIEW_PREFIX}{replay_run_id}",
            revision_id=revision_id,
            frozen_at=frozen_at,
            blocks=blocks,
            edges=edges,
            coverage=coverage,
        )
        try:
            document = replay_aig_document(
                base_revision_id=assessment.base_revision_id,
                replay_run_id=replay_run_id,
                source_fork_id=assessment.fork_node_id,
                revision=revision,
            )
            parsed = read_replay_aig(document)
        except ProtocolContractError as exc:
            raise ReplayAlignmentError("replay_aig_invalid") from exc
        with self._lock:
            path = self._aig_path(parsed.revision_id)
            self._write_same(path, document)
            return parsed

    def _store_admission(
        self,
        assessment: ReplayEligibilityAssessment,
        request: ReplayRequest,
        alignment: FirstActionAlignment,
        replay_aig: ReplayAigDocument,
        first_imagined: str | None,
        *,
        step_alignments: Sequence[Any],
        branch_steps: Sequence[Any],
        replay_block_ids: Sequence[str],
    ) -> tuple[DirectEvidenceAdmission, dict[str, Any]]:
        try:
            document = direct_evidence_admission_document(
                fork_node_id=assessment.fork_node_id,
                base_revision_id=assessment.base_revision_id,
                replay_run_id=request.replay_run_id,
                replay_revision_id=replay_aig.revision_id,
                alignment=alignment,
                expected_intervention_type=assessment.intervention_type,
                expected_target_boundary=assessment.target_boundary,
                expected_required_capabilities=assessment.required_capabilities,
                expected_safety_control_params=assessment.safety_control_params,
                first_imagined_node=alignment.imagined_node_id == first_imagined,
                route=assessment.route,
                step_alignments=step_alignments,
                branch_steps=branch_steps,
                replay_block_ids=replay_block_ids,
            )
            parsed = read_direct_evidence_admission(document)
        except ProtocolContractError as exc:
            raise ReplayAlignmentError("admission_invalid") from exc
        with self._lock:
            self._write_same(self._admission_path(parsed.admission_id), document)
            return parsed, document

    def _append_replay_direct(
        self,
        pattern_store: StrategyPatternStore,
        admission: Mapping[str, Any],
        *,
        subject_kind: str,
        subject_id: str,
        suffix: str,
    ) -> BranchEvaluation:
        try:
            parsed_admission = read_direct_evidence_admission(admission)
        except ProtocolContractError as exc:
            raise ReplayAlignmentError("admission_invalid") from exc
        evidence_ref = f"replay-direct:{parsed_admission.admission_id}:{suffix}"
        document = _evaluation_document(
            evaluation_id=evidence_ref,
            subject_kind=subject_kind,
            subject_id=subject_id,
            evidence_ref=evidence_ref,
            evidence_tier="replay-direct",
            status="active",
        )
        try:
            stored = read_branch_evaluation(document)
            if stored.evidence_tier != "replay-direct":
                raise ReplayAlignmentError("production_direct_forbidden")
            return pattern_store.append_evaluation(document, admission=admission)
        except StrategyPatternError as exc:
            raise ReplayAlignmentError(exc.code) from exc

    def _assessment_path(self, assessment_id: str) -> Path:
        return self._dir / "eligibility" / f"{_require_hex_id(assessment_id)}.json"

    def _admission_path(self, admission_id: str) -> Path:
        return self._dir / "admissions" / f"{_require_hex_id(admission_id)}.json"

    def _aig_path(self, revision_id: str) -> Path:
        digest = hashlib.sha256(revision_id.encode("utf-8")).hexdigest()
        return self._dir / "aigs" / f"{digest}.json"

    def _append_eligibility(self, row: Mapping[str, Any]) -> None:
        append_jsonl(self._eligibility_index, row)

    def _read_index(self) -> list[dict[str, Any]]:
        return read_jsonl_objects(self._eligibility_index, ReplayAlignmentError)

    def _read_json(self, path: Path) -> dict[str, Any]:
        return read_json_object(path, ReplayAlignmentError)

    def _write_new(self, path: Path, payload: Mapping[str, Any]) -> None:
        write_new_json(path, payload, ReplayAlignmentError)

    def _write_same(self, path: Path, payload: Mapping[str, Any]) -> None:
        if path.is_file():
            if self._read_json(path) != payload:
                raise ReplayAlignmentError("replay_conflict")
            return
        self._write_new(path, payload)


def _as_overlay(subject: ForkNodeOverlay | StoredForkOverlay) -> ForkNodeOverlay:
    return fork_overlay_for(subject, "replay", ReplayAlignmentError)


def _require_hex_id(value: str) -> str:
    if not isinstance(value, str) or _HEX_ID.fullmatch(value) is None:
        raise ReplayAlignmentError("stored_record_invalid")
    return value


def _first_imagined(overlay: ForkNodeOverlay) -> str | None:
    for step in overlay.branch_steps:
        if step.step_kind == STEP_KIND_IMAGINED and step.imagined_node_id:
            return step.imagined_node_id
    return None


def _require_outcome_pattern(
    pattern_store: StrategyPatternStore,
    pattern_id: str | None,
    assessment: ReplayEligibilityAssessment,
) -> None:
    if not isinstance(pattern_id, str) or not pattern_id:
        raise ReplayAlignmentError("outcome_subject_required")
    pattern = next(
        (item for item in pattern_store.list_patterns() if item.pattern_id == pattern_id),
        None,
    )
    if pattern is None:
        raise ReplayAlignmentError("outcome_subject_required")
    if (
        pattern.intervention_type != assessment.intervention_type
        or pattern.target_boundary != assessment.target_boundary
        or pattern.capability not in assessment.required_capabilities
    ):
        raise ReplayAlignmentError("pattern_controls_mismatch")


def _evaluation_document(
    *,
    evaluation_id: str,
    subject_kind: str,
    subject_id: str,
    evidence_ref: str,
    evidence_tier: str,
    status: str,
) -> dict[str, Any]:
    body = {
        "evaluation_id": evaluation_id,
        "evidence_ref": evidence_ref,
        "evidence_tier": evidence_tier,
        "status": status,
        "subject_id": subject_id,
        "subject_kind": subject_kind,
    }
    return {
        "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
        "object_kind": OBJECT_KIND_BRANCH_EVALUATION,
        "body": body,
        "content_hash": protocol_content_hash(
            SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_BRANCH_EVALUATION, body
        ),
    }
