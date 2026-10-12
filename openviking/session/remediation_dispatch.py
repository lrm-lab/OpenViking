# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Linked child remediation runs and execution reports (ADR-0043 P09).

The Coforge worker is the real executor of a remediation dispatch. P12
connects that worker. This module seals the versioned request and
confirmation and drives a scripted child run. It does not call the worker
or start a model.

Reader, Memory, Builder, Critic, and Refiner are not executors. The
selected target agent is the only executor. Dispatch creates a child run
only after every server gate passes. A completed failure run is not opened
by this module.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from openviking.session.aig_native_protocol import (
    ALIGNMENT_NOT_EXECUTED,
    DIRECT_ALIGNMENTS,
    EVALUATION_STATUS_ACTIVE,
    EVIDENCE_TIER_PRODUCTION_DIRECT,
    OBJECT_KIND_BRANCH_EVALUATION,
    SCHEMA_VERSION_AIG_NATIVE_V2,
    BranchEvaluation,
    ChildRemediationRun,
    DispatchGateDecision,
    GuidanceAdjustment,
    GuidanceApproval,
    GuidanceExecutionReport,
    ProductionDirectAdmission,
    ProtocolContractError,
    RemediationConstraint,
    RemediationDispatchConfirmation,
    RemediationDispatchRequest,
    child_remediation_run_document,
    child_run_output_document,
    dispatch_gate_document,
    guidance_adjustment_document,
    production_direct_admission_document,
    is_sha256_hex,
    protocol_content_hash,
    protocol_envelope,
    read_blocked_pre_execution,
    read_child_remediation_run,
    read_child_run_output,
    read_dispatch_gate_decision,
    read_guidance_adjustment,
    read_guidance_execution_report,
    read_production_direct_admission,
    read_remediation_constraint,
    read_remediation_dispatch_confirmation,
    read_remediation_dispatch_request,
    remediation_constraint_document,
    remediation_dispatch_confirmation_document,
    remediation_dispatch_request_document,
)
from openviking.session.append_only import append_jsonl, read_json_object, read_jsonl_objects, write_new_json
from openviking.session.fork_node_overlay import ForkOverlayError, require_label_set
from openviking.session.guidance_approval import GuidanceApprovalError, GuidanceApprovalService
from openviking.session.strategy_pattern import StrategyPatternError, StrategyPatternStore

REMEDIATION_RUN_DIRNAME = "remediation-runs"
_CALLER_CONTROL_KEYS = frozenset(
    {
        "agent_capabilities",
        "budget",
        "executor_agent_id",
        "executor_role",
        "idempotency_key",
        "isolation",
        "permission",
        "requested_side_effects",
        "side_effect_allowlist",
    }
)
_RECORD_BLOCKED = "blocked_pre_execution"
_RECORD_RUN = "child_remediation_run"


class RemediationDispatchError(ValueError):
    """A dispatch, child run, or execution report was refused."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ScriptedRemediationOutcome:
    """Local dispatch result. ``blocked`` means no child run was created."""

    decision: DispatchGateDecision
    blocked: bool
    deduplicated: bool
    request: RemediationDispatchRequest | None
    confirmation: RemediationDispatchConfirmation | None
    child_run: ChildRemediationRun | None
    report: GuidanceExecutionReport | None
    adjustment: GuidanceAdjustment | None
    admissions: tuple[ProductionDirectAdmission, ...]
    evaluations: tuple[BranchEvaluation, ...]
    constraints: tuple[RemediationConstraint, ...]


class RemediationDispatchService:
    """Append-only remediation store under ``remediation-runs``."""

    def __init__(self, root: str | Path, *, guidance: GuidanceApprovalService) -> None:
        if not isinstance(guidance, GuidanceApprovalService):
            raise RemediationDispatchError("store_required")
        self._root = Path(root)
        self._dir = self._root / REMEDIATION_RUN_DIRNAME
        self._index_path = self._dir / "index.jsonl"
        self._guidance = guidance
        self._lock = threading.Lock()

    def reject_parent_run(self, subject: Any) -> None:
        """Refuse a completed failure run. The argument is not inspected."""
        del subject
        raise RemediationDispatchError("parent_run_sealed")

    def audit_blocked(self) -> tuple[DispatchGateDecision, ...]:
        """Blocked gate audits in append order."""
        with self._lock:
            found: list[DispatchGateDecision] = []
            for row in self._read_index():
                if row.get("record_kind") != _RECORD_BLOCKED:
                    continue
                decision_id = _row_hex(row, "decision_id")
                found.append(read_blocked_pre_execution(self._read_json(self._blocked_path(decision_id))))
            return tuple(found)

    def dispatch_scripted(
        self,
        *,
        approval_content_hash: str,
        principal_labels: Sequence[str],
        parent_session_id: str,
        controls: Mapping[str, Any],
        outputs: Sequence[Mapping[str, Any]] | None = None,
        report: Mapping[str, Any] | None = None,
        pattern_store: StrategyPatternStore | None = None,
        adjustment: Mapping[str, Any] | None = None,
        orchestration_run_id: str | None = None,
    ) -> ScriptedRemediationOutcome:
        """Gate a scripted child run. Any failed gate is audited and creates nothing."""
        caller = _caller_controls(controls)
        labels = _caller_labels(principal_labels)
        if orchestration_run_id is not None and not is_sha256_hex(orchestration_run_id):
            raise RemediationDispatchError("dispatch_controls_invalid")
        try:
            approval = self._guidance.read_approval(approval_content_hash)
            basis = self._guidance.read_basis(approval_content_hash)
        except GuidanceApprovalError as exc:
            raise RemediationDispatchError(exc.code) from exc
        if approval is None:
            raise RemediationDispatchError("approval_required")
        fingerprint = _fingerprint(
            caller,
            labels,
            parent_session_id,
            _report_hash(report),
            _output_keys(outputs),
            _adjustment_fingerprint(adjustment),
        )
        with self._lock:
            existing_rows = self._run_rows(approval.guidance_content_hash, parent_session_id)
            matched = [row for row in existing_rows if row.get("fingerprint") == fingerprint]
            if len(matched) > 1:
                raise RemediationDispatchError("stored_record_invalid")
            if len(matched) == 1:
                return self._reload(matched[0], pattern_store, deduplicated=True)
            if orchestration_run_id is None:
                idempotency_state = "conflict" if existing_rows else "clear"
            else:
                scoped = [
                    row
                    for row in existing_rows
                    if row.get("orchestration_run_id") == orchestration_run_id
                ]
                if len(scoped) > 1:
                    raise RemediationDispatchError("stored_record_invalid")
                idempotency_state = "conflict" if scoped else "clear"
            gate_controls = _gate_controls(
                approval,
                basis.session_id,
                basis.principal_labels,
                labels,
                parent_session_id,
                caller,
                idempotency_state,
            )
            try:
                decision_document = dispatch_gate_document(gate_controls)
            except ProtocolContractError as exc:
                raise RemediationDispatchError("dispatch_controls_invalid") from exc
            if decision_document["object_kind"] == _RECORD_BLOCKED:
                decision = read_blocked_pre_execution(decision_document)
                return self._store_blocked(decision, decision_document)
            decision = read_dispatch_gate_decision(decision_document)
            if report is None:
                raise RemediationDispatchError("execution_report_required")
            try:
                parsed_report = read_guidance_execution_report(report)
            except ProtocolContractError as exc:
                raise RemediationDispatchError("execution_report_invalid") from exc
            if (
                parsed_report.guidance_id != approval.guidance_id
                or parsed_report.source_fork_id != approval.source_fork_id
                or parsed_report.parent_session_id != parent_session_id
            ):
                raise RemediationDispatchError("execution_report_invalid")
            expected_nodes = tuple(step.imagined_node_id for step in approval.steps)
            if tuple(step.imagined_node_id for step in parsed_report.steps) != expected_nodes:
                raise RemediationDispatchError("execution_report_invalid")
            if outputs is None:
                raise RemediationDispatchError("outputs_required")
            child_session_id = _child_session_id(
                parent_session_id,
                approval.guidance_content_hash,
                caller["idempotency_key"],
            )
            output_ids = _validated_output_ids(outputs, parent_session_id, child_session_id)
            block_ids = tuple(
                item["output_id"]
                for item in outputs
                if isinstance(item, Mapping) and item.get("output_kind") == "block"
            )
            for step in parsed_report.steps:
                if step.alignment == ALIGNMENT_NOT_EXECUTED:
                    continue
                if step.actual_block_id not in block_ids:
                    raise RemediationDispatchError("actual_block_not_in_child_run")
            parsed_adjustment, adjustment_document = _parse_adjustment(
                adjustment,
                approval,
                parent_session_id,
            )
            direct_steps = [
                step for step in parsed_report.steps if step.alignment in DIRECT_ALIGNMENTS
            ]
            if direct_steps:
                if not isinstance(pattern_store, StrategyPatternStore):
                    raise RemediationDispatchError("store_required")
                for step in direct_steps:
                    if not pattern_store.imagined_subject_known(step.imagined_node_id):
                        raise RemediationDispatchError("subject_unknown")
            request_document = remediation_dispatch_request_document(
                decision, target_role=approval.target_role
            )
            request = read_remediation_dispatch_request(request_document)
            run_document = child_remediation_run_document(
                parent_session_id=parent_session_id,
                source_fork_id=approval.source_fork_id,
                source_guidance_id=approval.guidance_id,
                guidance_content_hash=approval.guidance_content_hash,
                request_id=request.request_id,
                selected_agent_id=approval.selected_agent_id,
                child_session_id=child_session_id,
                output_ids=output_ids,
            )
            child_run = read_child_remediation_run(run_document)
            output_documents = [
                _output_document(
                    item,
                    child_run_id=child_run.child_run_id,
                    child_session_id=child_session_id,
                    parent_session_id=parent_session_id,
                    source_fork_id=approval.source_fork_id,
                    source_guidance_id=approval.guidance_id,
                )
                for item in outputs
            ]
            confirmation_document = remediation_dispatch_confirmation_document(
                request_id=request.request_id,
                child_run_id=child_run.child_run_id,
            )
            confirmation = read_remediation_dispatch_confirmation(confirmation_document)
            self._write_same(self._gate_path(decision.decision_id), decision_document)
            self._write_same(self._request_path(request.request_id), request_document)
            self._write_same(self._run_path(child_run.child_run_id), run_document)
            for document in output_documents:
                parsed_output = read_child_run_output(document)
                self._write_same(self._output_path(parsed_output.content_hash), document)
            self._write_same(self._report_path(parsed_report.content_hash), dict(report))
            adjustment_id = None
            if parsed_adjustment is not None:
                self._write_same(
                    self._adjustment_path(parsed_adjustment.adjustment_id),
                    adjustment_document,
                )
                adjustment_id = parsed_adjustment.adjustment_id
            self._write_same(
                self._confirmation_path(confirmation.confirmation_id),
                confirmation_document,
            )
            admissions: list[ProductionDirectAdmission] = []
            admission_documents: list[dict[str, Any]] = []
            constraints: list[RemediationConstraint] = []
            evaluations: list[BranchEvaluation] = []
            evaluation_ids: list[str] = []
            constraint_ids: list[str] = []
            child_blocks = tuple(sorted(block_ids))
            for step in parsed_report.steps:
                if step.alignment in DIRECT_ALIGNMENTS:
                    if step.actual_block_id is None:
                        raise RemediationDispatchError("actual_block_not_in_child_run")
                    admission_document = production_direct_admission_document(
                        imagined_node_id=step.imagined_node_id,
                        actual_block_id=step.actual_block_id,
                        alignment=step.alignment,
                        source_fork_id=approval.source_fork_id,
                        source_guidance_id=approval.guidance_id,
                        parent_session_id=parent_session_id,
                        child_run_id=child_run.child_run_id,
                        report_content_hash=parsed_report.content_hash,
                        child_block_ids=child_blocks,
                    )
                    admission = read_production_direct_admission(admission_document)
                    self._write_same(self._admission_path(admission.admission_id), admission_document)
                    evaluation_id = (
                        f"{EVIDENCE_TIER_PRODUCTION_DIRECT}:{admission.admission_id}:"
                        f"{step.imagined_node_id}"
                    )
                    evaluation_document = _evaluation_document(
                        evaluation_id=evaluation_id,
                        subject_id=step.imagined_node_id,
                        evidence_ref=evaluation_id,
                    )
                    try:
                        stored = cast(StrategyPatternStore, pattern_store).append_evaluation(
                            evaluation_document,
                            admission=admission_document,
                        )
                    except StrategyPatternError as exc:
                        raise RemediationDispatchError(exc.code) from exc
                    if (
                        stored.evidence_tier != EVIDENCE_TIER_PRODUCTION_DIRECT
                        or stored.status != EVALUATION_STATUS_ACTIVE
                    ):
                        raise RemediationDispatchError("direct_refutation_forbidden")
                    admissions.append(admission)
                    admission_documents.append(admission_document)
                    evaluations.append(stored)
                    evaluation_ids.append(evaluation_id)
                else:
                    constraint_document = remediation_constraint_document(
                        imagined_node_id=step.imagined_node_id,
                        alignment=step.alignment,
                        applicability=approval.applicability.to_dict(),
                        statement=f"constraint:{step.alignment}",
                        source_fork_id=approval.source_fork_id,
                        source_guidance_id=approval.guidance_id,
                        parent_session_id=parent_session_id,
                        child_run_id=child_run.child_run_id,
                        report_content_hash=parsed_report.content_hash,
                    )
                    constraint = read_remediation_constraint(constraint_document)
                    self._write_same(self._constraint_path(constraint.constraint_id), constraint_document)
                    constraints.append(constraint)
                    constraint_ids.append(constraint.constraint_id)
            self._append_index(
                {
                    "adjustment_id": adjustment_id,
                    "admission_ids": [item.admission_id for item in admissions],
                    "child_run_id": child_run.child_run_id,
                    "confirmation_id": confirmation.confirmation_id,
                    "constraint_ids": constraint_ids,
                    "decision_id": decision.decision_id,
                    "evaluation_ids": evaluation_ids,
                    "fingerprint": fingerprint,
                    "guidance_content_hash": approval.guidance_content_hash,
                    "orchestration_run_id": orchestration_run_id,
                    "parent_session_id": parent_session_id,
                    "record_kind": _RECORD_RUN,
                    "report_content_hash": parsed_report.content_hash,
                    "request_id": request.request_id,
                    "source_fork_id": approval.source_fork_id,
                    "source_guidance_id": approval.guidance_id,
                }
            )
            return ScriptedRemediationOutcome(
                decision=decision,
                blocked=False,
                deduplicated=False,
                request=request,
                confirmation=confirmation,
                child_run=child_run,
                report=parsed_report,
                adjustment=parsed_adjustment,
                admissions=tuple(admissions),
                evaluations=tuple(evaluations),
                constraints=tuple(constraints),
            )

    def _store_blocked(
        self, decision: DispatchGateDecision, document: Mapping[str, Any]
    ) -> ScriptedRemediationOutcome:
        path = self._blocked_path(decision.decision_id)
        deduplicated = path.is_file()
        self._write_same(path, document)
        if not deduplicated:
            self._append_index(
                {
                    "content_hash": decision.content_hash,
                    "decision_id": decision.decision_id,
                    "failed_gate": decision.failed_gate,
                    "guidance_content_hash": decision.guidance_content_hash,
                    "parent_session_id": decision.parent_session_id,
                    "reason": decision.reasons[0] if decision.reasons else "",
                    "record_kind": _RECORD_BLOCKED,
                }
            )
        return ScriptedRemediationOutcome(
            decision=decision,
            blocked=True,
            deduplicated=deduplicated,
            request=None,
            confirmation=None,
            child_run=None,
            report=None,
            adjustment=None,
            admissions=(),
            evaluations=(),
            constraints=(),
        )

    def _reload(
        self,
        row: Mapping[str, Any],
        pattern_store: StrategyPatternStore | None,
        *,
        deduplicated: bool,
    ) -> ScriptedRemediationOutcome:
        decision_id = _row_hex(row, "decision_id")
        request_id = _row_hex(row, "request_id")
        child_run_id = _row_hex(row, "child_run_id")
        confirmation_id = _row_hex(row, "confirmation_id")
        report_hash = _row_hex(row, "report_content_hash")
        decision = read_dispatch_gate_decision(self._read_json(self._gate_path(decision_id)))
        request = read_remediation_dispatch_request(self._read_json(self._request_path(request_id)))
        child_run = read_child_remediation_run(self._read_json(self._run_path(child_run_id)))
        confirmation = read_remediation_dispatch_confirmation(
            self._read_json(self._confirmation_path(confirmation_id))
        )
        report = read_guidance_execution_report(self._read_json(self._report_path(report_hash)))
        adjustment = None
        if row.get("adjustment_id") is not None:
            adjustment_id = _row_hex(row, "adjustment_id")
            adjustment = read_guidance_adjustment(
                self._read_json(self._adjustment_path(adjustment_id))
            )
        admissions = tuple(
            read_production_direct_admission(self._read_json(self._admission_path(item)))
            for item in _row_hex_list(row, "admission_ids")
        )
        constraints = tuple(
            read_remediation_constraint(self._read_json(self._constraint_path(item)))
            for item in _row_hex_list(row, "constraint_ids")
        )
        evaluation_ids = _row_str_list(row, "evaluation_ids")
        evaluations: list[BranchEvaluation] = []
        if evaluation_ids:
            if not isinstance(pattern_store, StrategyPatternStore):
                raise RemediationDispatchError("store_required")
            for evaluation_id in evaluation_ids:
                stored = pattern_store.read_evaluation(evaluation_id)
                if stored is None:
                    raise RemediationDispatchError("stored_record_invalid")
                evaluations.append(stored)
        return ScriptedRemediationOutcome(
            decision=decision,
            blocked=False,
            deduplicated=deduplicated,
            request=request,
            confirmation=confirmation,
            child_run=child_run,
            report=report,
            adjustment=adjustment,
            admissions=admissions,
            evaluations=tuple(evaluations),
            constraints=constraints,
        )

    def _run_rows(self, guidance_content_hash: str, parent_session_id: str) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for row in self._read_index():
            if row.get("record_kind") != _RECORD_RUN:
                continue
            if (
                row.get("guidance_content_hash") == guidance_content_hash
                and row.get("parent_session_id") == parent_session_id
            ):
                found.append(row)
        return found

    def _blocked_path(self, decision_id: str) -> Path:
        return self._dir / "blocked" / f"{decision_id}.json"

    def _gate_path(self, decision_id: str) -> Path:
        return self._dir / "gates" / f"{decision_id}.json"

    def _request_path(self, request_id: str) -> Path:
        return self._dir / "requests" / f"{request_id}.json"

    def _run_path(self, child_run_id: str) -> Path:
        return self._dir / "runs" / f"{child_run_id}.json"

    def _output_path(self, content_hash: str) -> Path:
        return self._dir / "outputs" / f"{content_hash}.json"

    def _report_path(self, content_hash: str) -> Path:
        return self._dir / "reports" / f"{content_hash}.json"

    def _adjustment_path(self, adjustment_id: str) -> Path:
        return self._dir / "adjustments" / f"{adjustment_id}.json"

    def _confirmation_path(self, confirmation_id: str) -> Path:
        return self._dir / "confirmations" / f"{confirmation_id}.json"

    def _admission_path(self, admission_id: str) -> Path:
        return self._dir / "admissions" / f"{admission_id}.json"

    def _constraint_path(self, constraint_id: str) -> Path:
        return self._dir / "constraints" / f"{constraint_id}.json"

    def _append_index(self, row: Mapping[str, Any]) -> None:
        append_jsonl(self._index_path, row)

    def _read_index(self) -> list[dict[str, Any]]:
        return read_jsonl_objects(self._index_path, RemediationDispatchError)

    def _read_json(self, path: Path) -> dict[str, Any]:
        return read_json_object(path, RemediationDispatchError)

    def _write_same(self, path: Path, payload: Mapping[str, Any]) -> None:
        if path.is_file():
            if self._read_json(path) != dict(payload):
                raise RemediationDispatchError("stored_record_invalid")
            return
        write_new_json(path, payload, RemediationDispatchError)


def _caller_controls(controls: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(controls, Mapping) or set(controls) != _CALLER_CONTROL_KEYS:
        raise RemediationDispatchError("dispatch_controls_invalid")
    parsed: dict[str, Any] = {}
    for key in (
        "agent_capabilities",
        "requested_side_effects",
        "side_effect_allowlist",
    ):
        value = controls.get(key)
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            raise RemediationDispatchError("dispatch_controls_invalid")
        if len(value) != len(set(value)):
            raise RemediationDispatchError("dispatch_controls_invalid")
        parsed[key] = sorted(value)
    for key in ("budget", "executor_agent_id", "executor_role", "idempotency_key", "isolation", "permission"):
        value = controls.get(key)
        if not isinstance(value, str) or not value:
            raise RemediationDispatchError("dispatch_controls_invalid")
        parsed[key] = value
    return parsed


def _caller_labels(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RemediationDispatchError("dispatch_controls_invalid")
    if len(value) == 0:
        return ()
    try:
        labels = require_label_set(value, code="dispatch_controls_invalid")
    except ForkOverlayError as exc:
        raise RemediationDispatchError(exc.code) from exc
    return tuple(sorted(labels))


def _gate_controls(
    approval: GuidanceApproval,
    bound_session_id: str,
    bound_principal_labels: Sequence[str],
    principal_labels: Sequence[str],
    parent_session_id: str,
    caller: Mapping[str, Any],
    idempotency_state: str,
) -> dict[str, Any]:
    return {
        "agent_capabilities": list(caller["agent_capabilities"]),
        "bound_principal_labels": sorted(bound_principal_labels),
        "bound_session_id": bound_session_id,
        "budget": caller["budget"],
        "capability_limits": sorted(approval.capability_limits),
        "executor_agent_id": caller["executor_agent_id"],
        "executor_role": caller["executor_role"],
        "guidance_content_hash": approval.guidance_content_hash,
        "guidance_id": approval.guidance_id,
        "idempotency_key": caller["idempotency_key"],
        "idempotency_state": idempotency_state,
        "isolation": caller["isolation"],
        "parent_session_id": parent_session_id,
        "permission": caller["permission"],
        "principal_labels": list(principal_labels),
        "requested_side_effects": list(caller["requested_side_effects"]),
        "selected_agent_id": approval.selected_agent_id,
        "side_effect_allowlist": list(caller["side_effect_allowlist"]),
        "source_fork_id": approval.source_fork_id,
        "source_guidance_id": approval.guidance_id,
    }


def _fingerprint(
    caller: Mapping[str, Any],
    labels: Sequence[str],
    parent_session_id: str,
    report_hash: str | None,
    output_keys: tuple[str, ...],
    adjustment_hash: str | None,
) -> str:
    body = {
        "adjustment_hash": adjustment_hash,
        "agent_capabilities": list(caller["agent_capabilities"]),
        "budget": caller["budget"],
        "executor_agent_id": caller["executor_agent_id"],
        "executor_role": caller["executor_role"],
        "idempotency_key": caller["idempotency_key"],
        "isolation": caller["isolation"],
        "output_keys": list(output_keys),
        "parent_session_id": parent_session_id,
        "permission": caller["permission"],
        "principal_labels": list(labels),
        "report_content_hash": report_hash,
        "requested_side_effects": list(caller["requested_side_effects"]),
        "side_effect_allowlist": list(caller["side_effect_allowlist"]),
    }
    return protocol_content_hash(SCHEMA_VERSION_AIG_NATIVE_V2, "dispatch_fingerprint", body)


def _report_hash(report: Mapping[str, Any] | None) -> str | None:
    if report is None:
        return None
    try:
        return read_guidance_execution_report(report).content_hash
    except ProtocolContractError:
        return "invalid"


def _output_keys(outputs: Sequence[Mapping[str, Any]] | None) -> tuple[str, ...]:
    if outputs is None:
        return ()
    keys: list[str] = []
    for item in outputs:
        if not isinstance(item, Mapping):
            return ("invalid",)
        kind = item.get("output_kind")
        output_id = item.get("output_id")
        session = item.get("child_session_id")
        keys.append(f"{kind}:{output_id}:{session}")
    return tuple(sorted(keys))


def _adjustment_fingerprint(adjustment: Mapping[str, Any] | None) -> str | None:
    if adjustment is None:
        return None
    if not isinstance(adjustment, Mapping):
        return "invalid"
    return protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        "adjustment_fingerprint",
        dict(adjustment),
    )


def _child_session_id(parent_session_id: str, guidance_content_hash: str, idempotency_key: str) -> str:
    digest = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        "child_session_identity",
        {
            "guidance_content_hash": guidance_content_hash,
            "idempotency_key": idempotency_key,
            "parent_session_id": parent_session_id,
        },
    )
    return "child-" + digest[:32]


def _validated_output_ids(
    outputs: Sequence[Mapping[str, Any]],
    parent_session_id: str,
    child_session_id: str,
) -> tuple[str, ...]:
    if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Sequence) or not outputs:
        raise RemediationDispatchError("outputs_required")
    ids: list[str] = []
    for item in outputs:
        if not isinstance(item, Mapping):
            raise RemediationDispatchError("outputs_required")
        supplied_session = item.get("child_session_id")
        if supplied_session == parent_session_id:
            raise RemediationDispatchError("parent_run_sealed")
        if supplied_session is not None and supplied_session != child_session_id:
            raise RemediationDispatchError("output_provenance_required")
        if "parent_session_id" in item and item["parent_session_id"] != parent_session_id:
            raise RemediationDispatchError("output_provenance_required")
        output_id = item.get("output_id")
        if not isinstance(output_id, str) or not output_id:
            raise RemediationDispatchError("outputs_required")
        ids.append(output_id)
    if len(ids) != len(set(ids)):
        raise RemediationDispatchError("outputs_required")
    return tuple(sorted(ids))


def _output_document(
    item: Mapping[str, Any],
    *,
    child_run_id: str,
    child_session_id: str,
    parent_session_id: str,
    source_fork_id: str,
    source_guidance_id: str,
) -> dict[str, Any]:
    for field, expected in (
        ("source_fork_id", source_fork_id),
        ("source_guidance_id", source_guidance_id),
        ("parent_session_id", parent_session_id),
    ):
        if field in item and item[field] != expected:
            raise RemediationDispatchError("output_provenance_required")
    try:
        return child_run_output_document(
            output_kind=item.get("output_kind"),
            output_id=item.get("output_id"),
            child_run_id=child_run_id,
            child_session_id=child_session_id,
            parent_session_id=parent_session_id,
            source_fork_id=source_fork_id,
            source_guidance_id=source_guidance_id,
        )
    except ProtocolContractError as exc:
        raise RemediationDispatchError("output_provenance_required") from exc


def _parse_adjustment(
    adjustment: Mapping[str, Any] | None,
    approval: GuidanceApproval,
    parent_session_id: str,
) -> tuple[GuidanceAdjustment | None, dict[str, Any] | None]:
    if adjustment is None:
        return None, None
    if not isinstance(adjustment, Mapping):
        raise RemediationDispatchError("adjustment_forbidden")
    required = {
        "author_agent_id",
        "capability_limits",
        "revised_steps",
        "target_role",
    }
    if set(adjustment) != required:
        raise RemediationDispatchError("adjustment_forbidden")
    limits = adjustment.get("capability_limits")
    revised = adjustment.get("revised_steps")
    if not isinstance(limits, list) or not isinstance(revised, list):
        raise RemediationDispatchError("adjustment_forbidden")
    try:
        document = guidance_adjustment_document(
            guidance_content_hash=approval.guidance_content_hash,
            guidance_id=approval.guidance_id,
            author_agent_id=adjustment.get("author_agent_id"),
            bound_agent_id=approval.selected_agent_id,
            parent_session_id=parent_session_id,
            target_role=adjustment.get("target_role"),
            bound_target_role=approval.target_role,
            capability_limits=sorted(str(item) for item in limits),
            bound_capability_limits=sorted(approval.capability_limits),
            bound_imagined_node_ids=sorted(step.imagined_node_id for step in approval.steps),
            revised_steps=revised,
        )
    except (ProtocolContractError, TypeError) as exc:
        raise RemediationDispatchError("adjustment_forbidden") from exc
    return read_guidance_adjustment(document), document


def _evaluation_document(*, evaluation_id: str, subject_id: str, evidence_ref: str) -> dict[str, Any]:
    body = {
        "evaluation_id": evaluation_id,
        "evidence_ref": evidence_ref,
        "evidence_tier": EVIDENCE_TIER_PRODUCTION_DIRECT,
        "status": EVALUATION_STATUS_ACTIVE,
        "subject_id": subject_id,
        "subject_kind": "imagined_node",
    }
    return protocol_envelope(SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_BRANCH_EVALUATION, body)


def _row_hex(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not is_sha256_hex(value):
        raise RemediationDispatchError("stored_id_invalid")
    return str(value)


def _row_hex_list(row: Mapping[str, Any], field: str) -> tuple[str, ...]:
    value = row.get(field)
    if not isinstance(value, list) or not all(is_sha256_hex(item) for item in value):
        raise RemediationDispatchError("stored_id_invalid")
    return tuple(str(item) for item in value)


def _row_str_list(row: Mapping[str, Any], field: str) -> tuple[str, ...]:
    value = row.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RemediationDispatchError("stored_record_invalid")
    return tuple(value)
