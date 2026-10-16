# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Frozen OutcomeContract, Reader assessment, and OutcomeEvidencePolicy.

The contract is frozen at task start. Reader submits a TaskOutcomeAssessment
on a trusted AIG report. OutcomeEvidencePolicy version ``outcome-evidence-v1``
adjudicates ``success``, ``confirmed_failure``, or ``uncertain``.

Explicit evaluator, test, and acceptance signals outrank Reader inference.
Missing, conflicting, or provisional criteria stay ``uncertain``. This module
does not dispatch remediation: ``uncertain`` forbids follow-up, and
``confirmed_failure`` is only a recorded decision. Automatic dispatch lives
in ``failure_gate.FailureGateOrchestrator``.

Reader success with no external signal is recorded as success. Reader failure
with no external signal stays uncertain. That asymmetry is fail-safe: failure
is not confirmed from Reader inference alone, and success does not start
remediation. ``FailureGateOrchestrator`` is the automatic dispatcher;
``remediation_gate`` still refuses every direct caller.

Session telemetry and ``index_outcome_status`` are not evaluator evidence.
``session._publish_telemetry_summary_best_effort`` publishes metrics without
changing the task outcome; this adapter does not read that summary. A real
session adapter is a P10/P12 concern. Callers pass ``evaluator_results`` or an
``OutcomeEvaluatorSource``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from openviking.session.aig_native_protocol import (
    OutcomeContract,
    ProtocolContractError,
    TaskOutcomeAssessment,
    is_safe_id,
    read_outcome_contract,
    read_task_outcome_assessment,
)
from openviking.session.aig_role_chain import AigConstructionRun
from openviking.session.append_only import append_jsonl, write_exclusive

OUTCOME_EVIDENCE_POLICY_VERSION = "outcome-evidence-v1"
DECISION_SUCCESS = "success"
DECISION_CONFIRMED_FAILURE = "confirmed_failure"
DECISION_UNCERTAIN = "uncertain"
FOLLOW_UP_FORBIDDEN = "forbidden"
FOLLOW_UP_RECORDED_ONLY = "recorded_only"
REASON_CRITERIA_MISSING = "criteria_missing"
REASON_CRITERIA_CONFLICT = "criteria_conflict"
REASON_CRITERIA_NOT_EXPLICIT = "criteria_not_explicit"
REASON_EXTERNAL_FAIL = "external_fail"
REASON_EXTERNAL_PASS = "external_pass"
REASON_READER_SUCCESS = "reader_success"
REASON_READER_INFERENCE_ONLY = "reader_inference_only"
REASON_READER_UNCERTAIN = "reader_uncertain"

_DECISIONS = frozenset({DECISION_SUCCESS, DECISION_CONFIRMED_FAILURE, DECISION_UNCERTAIN})
_SIGNAL_KINDS = frozenset({"evaluator", "test", "acceptance"})
_SIGNALS = frozenset({"pass", "fail"})
OUTCOME_SIGNAL_KINDS = _SIGNAL_KINDS
OUTCOME_SIGNALS = _SIGNALS
_RESULT_KEYS = frozenset({"binding_id", "signal_kind", "signal", "evidence_ref"})
_TELEMETRY_KEYS = frozenset({"summary", "tokens", "telemetry", "telemetry_id"})
_NAMESPACE = "outcome-contracts"
_RECORD_CONTRACT = "outcome_contract"
_RECORD_ASSESSMENT = "task_outcome_assessment"


class OutcomeContractError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@runtime_checkable
class OutcomeEvaluatorSource(Protocol):
    """Evaluator results a later adapter will expose.

    P10/P12 bind a real session to this attribute. Nothing here reads a
    session field that this protocol does not name.
    """

    outcome_evaluator_results: Sequence[Mapping[str, Any]] | None


@dataclass(frozen=True)
class CriticViolations:
    """Critic graph notes. They are not a task-outcome assessment."""

    items: tuple[Any, ...] = ()


@dataclass(frozen=True)
class ExternalOutcomeEvidence:
    """One evaluator, test, or acceptance signal. Not a telemetry summary."""

    binding_id: str
    signal_kind: str
    signal: str
    evidence_ref: str


@dataclass(frozen=True)
class OutcomeAdjudication:
    """Policy decision. ``recorded_only`` does not start remediation."""

    policy_version: str
    decision: str
    follow_up: str
    reason: str


def adjudicate_outcome(
    assessment: TaskOutcomeAssessment | CriticViolations,
    contract: OutcomeContract,
    external_evidence: Sequence[ExternalOutcomeEvidence] = (),
) -> OutcomeAdjudication:
    """Apply ``outcome-evidence-v1``.

    Each contract criterion needs a disposition. A missing key, an extra key,
    or a non-explicit value is uncertain before any external signal is read.
    An empty acceptance list is missing criteria. Bound signals must cover
    every contract binding and must not disagree. Uncovered or conflicting
    signals are uncertain. A unanimous fail is ``confirmed_failure`` even
    when Reader said success. A unanimous pass is ``success`` even when
    Reader said failure.

    With no bindings, Reader failure is inference and stays uncertain.
    Reader success with no external signal is recorded and does not
    remediate. That asymmetry is fail-safe: a failure is not confirmed from
    Reader inference alone. Automatic dispatch is not decided here.
    """
    if isinstance(assessment, CriticViolations):
        raise OutcomeContractError("critic_violation_is_not_failure")
    if not isinstance(assessment, TaskOutcomeAssessment):
        raise OutcomeContractError("assessment_required")
    if not isinstance(contract, OutcomeContract):
        raise OutcomeContractError("contract_required")
    if assessment.author_role != "reader":
        raise OutcomeContractError("outcome_write_forbidden")
    evidence = _validate_evidence(external_evidence)
    criteria_reason = _criterion_uncertainty(assessment, contract)
    if criteria_reason is not None:
        return _adjudication(DECISION_UNCERTAIN, criteria_reason)
    classified = _classify_signals(contract, evidence)
    if classified == "conflict":
        return _adjudication(DECISION_UNCERTAIN, REASON_CRITERIA_CONFLICT)
    if classified == "missing":
        return _adjudication(DECISION_UNCERTAIN, REASON_CRITERIA_MISSING)
    if classified == "fail":
        return _adjudication(DECISION_CONFIRMED_FAILURE, REASON_EXTERNAL_FAIL)
    if classified == "pass":
        return _adjudication(DECISION_SUCCESS, REASON_EXTERNAL_PASS)
    if assessment.verdict == "success":
        return _adjudication(DECISION_SUCCESS, REASON_READER_SUCCESS)
    if assessment.verdict == "failure":
        return _adjudication(DECISION_UNCERTAIN, REASON_READER_INFERENCE_ONLY)
    return _adjudication(DECISION_UNCERTAIN, REASON_READER_UNCERTAIN)


def remediation_gate(adjudication: OutcomeAdjudication) -> None:
    """Refuse follow-up. Direct callers never dispatch a remediation run.

    ``FailureGateOrchestrator`` is the post-commit gate. It does not call
    this function. Confirmed failure stays ``recorded_only`` on the
    adjudication itself.
    """
    if not isinstance(adjudication, OutcomeAdjudication):
        raise OutcomeContractError("adjudication_required")
    if adjudication.policy_version != OUTCOME_EVIDENCE_POLICY_VERSION:
        raise OutcomeContractError("unsupported_policy_version")
    if adjudication.decision not in _DECISIONS:
        raise OutcomeContractError("adjudication_required")
    if (
        adjudication.decision == DECISION_CONFIRMED_FAILURE
        and adjudication.follow_up != FOLLOW_UP_FORBIDDEN
    ):
        raise OutcomeContractError("remediation_not_in_scope")
    raise OutcomeContractError("automatic_remediation_forbidden")


def external_evidence_from_session(
    session: OutcomeEvaluatorSource | None = None,
    evaluator_results: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[ExternalOutcomeEvidence, ...]:
    """Map explicit evaluator results. Telemetry and index status are ignored.

    ``index_outcome_status`` only prioritizes trajectory indexing. A failed
    index hint is not an evaluator fail. Pass ``evaluator_results`` or an
    ``OutcomeEvaluatorSource``. P10/P12 attach the real session adapter.
    This function does not read an attribute the object does not define.
    """
    if evaluator_results is None:
        if not isinstance(session, OutcomeEvaluatorSource):
            raise OutcomeContractError("evaluator_source_required")
        raw = session.outcome_evaluator_results
        if raw is None:
            return ()
        evaluator_results = raw
    if isinstance(evaluator_results, (str, bytes)) or not isinstance(evaluator_results, Sequence):
        raise OutcomeContractError("invalid_external_evidence")
    parsed: list[ExternalOutcomeEvidence] = []
    for item in evaluator_results:
        parsed.append(_parse_result(item))
    parsed.sort(
        key=lambda evidence: (
            evidence.binding_id,
            evidence.signal_kind,
            evidence.signal,
            evidence.evidence_ref,
        )
    )
    return tuple(parsed)


class OutcomeContractStore:
    """Append-only contract and assessment store under ``outcome-contracts``."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._dir = self._root / _NAMESPACE
        self._index_path = self._dir / "index.jsonl"
        self._lock = threading.Lock()

    def freeze(self, document: OutcomeContract, envelope: Mapping[str, Any]) -> OutcomeContract:
        if not isinstance(document, OutcomeContract):
            raise OutcomeContractError("contract_required")
        _require_safe_id(document.contract_id)
        _require_safe_id(document.task_id)
        with self._lock:
            by_id = self._one(self._matching(_RECORD_CONTRACT, contract_id=document.contract_id))
            by_task = self._one(self._matching(_RECORD_CONTRACT, task_id=document.task_id))
            if by_id is not None and by_id.get("content_hash") != document.content_hash:
                raise OutcomeContractError("contract_frozen")
            if by_task is not None and by_task.get("contract_id") != document.contract_id:
                raise OutcomeContractError("task_contract_exists")
            if by_id is not None:
                return self._load_contract(document.contract_id, document.content_hash)
            self._write_new(self._contract_path(document.contract_id), envelope)
            self._append_index(
                {
                    "content_hash": document.content_hash,
                    "contract_id": document.contract_id,
                    "record_kind": _RECORD_CONTRACT,
                    "task_id": document.task_id,
                }
            )
            return self._load_contract(document.contract_id, document.content_hash)

    def read_contract(self, contract_id: str) -> OutcomeContract | None:
        with self._lock:
            row = self._one(self._matching(_RECORD_CONTRACT, contract_id=contract_id))
            if row is None:
                return None
            return self._load_contract(contract_id, str(row.get("content_hash")))

    def read_task_contract(self, task_id: str) -> OutcomeContract | None:
        with self._lock:
            row = self._one(self._matching(_RECORD_CONTRACT, task_id=task_id))
            if row is None:
                return None
            contract_id = row.get("contract_id")
            if not isinstance(contract_id, str):
                raise OutcomeContractError("stored_record_invalid")
            return self._load_contract(contract_id, str(row.get("content_hash")))

    def append_assessment(
        self,
        assessment: TaskOutcomeAssessment,
        envelope: Mapping[str, Any],
        *,
        task_id: str,
        contract_id: str,
    ) -> TaskOutcomeAssessment:
        if not isinstance(assessment, TaskOutcomeAssessment):
            raise OutcomeContractError("assessment_required")
        _require_safe_id(task_id)
        _require_safe_id(contract_id)
        with self._lock:
            task_row = self._one(self._matching(_RECORD_CONTRACT, task_id=task_id))
            if task_row is None or task_row.get("contract_id") != contract_id:
                raise OutcomeContractError("contract_not_frozen")
            existing = self._one(
                self._matching(
                    _RECORD_ASSESSMENT,
                    task_id=task_id,
                    report_hash=assessment.report_hash,
                )
            )
            if existing is not None:
                same_hash = existing.get("content_hash") == assessment.content_hash
                same_contract = existing.get("contract_id") == contract_id
                if not same_hash or not same_contract:
                    raise OutcomeContractError("assessment_conflict")
                return self._load_assessment(assessment.content_hash)
            path = self._assessment_path(assessment.content_hash)
            if not path.is_file():
                self._write_new(path, envelope)
            self._append_index(
                {
                    "content_hash": assessment.content_hash,
                    "contract_id": contract_id,
                    "record_kind": _RECORD_ASSESSMENT,
                    "report_hash": assessment.report_hash,
                    "task_id": task_id,
                }
            )
            return self._load_assessment(assessment.content_hash)

    def read_assessment(self, task_id: str, report_hash: str) -> TaskOutcomeAssessment | None:
        with self._lock:
            row = self._one(
                self._matching(_RECORD_ASSESSMENT, task_id=task_id, report_hash=report_hash)
            )
            if row is None:
                return None
            content_hash = row.get("content_hash")
            if not isinstance(content_hash, str):
                raise OutcomeContractError("stored_record_invalid")
            return self._load_assessment(content_hash)

    def _contract_path(self, contract_id: str) -> Path:
        return self._dir / "contracts" / f"{contract_id}.json"

    def _assessment_path(self, content_hash: str) -> Path:
        return self._dir / "assessments" / f"{content_hash}.json"

    def _matching(self, record_kind: str, **wanted: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in self._read_index():
            if row.get("record_kind") != record_kind:
                continue
            if all(row.get(key) == value for key, value in wanted.items()):
                rows.append(row)
        return rows

    def _one(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not rows:
            return None
        hashes = {row.get("content_hash") for row in rows}
        if len(hashes) != 1:
            raise OutcomeContractError("outcome_index_conflict")
        return rows[0]

    def _read_index(self) -> list[dict[str, Any]]:
        if not self._index_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in self._index_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise OutcomeContractError("outcome_index_corrupt") from exc
            if not isinstance(row, dict):
                raise OutcomeContractError("outcome_index_corrupt")
            rows.append(row)
        return rows

    def _append_index(self, row: Mapping[str, Any]) -> None:
        append_jsonl(self._index_path, row)

    def _write_new(self, path: Path, envelope: Mapping[str, Any]) -> None:
        try:
            write_exclusive(path, envelope)
        except FileExistsError as exc:
            raise OutcomeContractError("stored_record_exists") from exc

    def _load_contract(self, contract_id: str, expected_hash: str) -> OutcomeContract:
        document = _read_contract_file(self._contract_path(contract_id))
        if document.contract_id != contract_id or document.content_hash != expected_hash:
            raise OutcomeContractError("stored_hash_mismatch")
        return document

    def _load_assessment(self, content_hash: str) -> TaskOutcomeAssessment:
        assessment = _read_assessment_file(self._assessment_path(content_hash))
        if assessment.content_hash != content_hash:
            raise OutcomeContractError("stored_hash_mismatch")
        return assessment


class OutcomeAssessmentService:
    """Server entry. Builder, Critic, and Refiner cannot submit an outcome."""

    def __init__(self, store: OutcomeContractStore) -> None:
        if not isinstance(store, OutcomeContractStore):
            raise OutcomeContractError("store_required")
        self._store = store

    def freeze_contract(self, payload: Mapping[str, Any]) -> OutcomeContract:
        if not isinstance(payload, Mapping):
            raise OutcomeContractError("contract_required")
        document = read_outcome_contract(payload)
        body = payload.get("body")
        if not isinstance(body, dict):
            raise OutcomeContractError("contract_required")
        envelope = {
            "body": body,
            "content_hash": document.content_hash,
            "object_kind": document.object_kind,
            "schema_version": document.schema_version,
        }
        return self._store.freeze(document, envelope)

    def submit_assessment(
        self,
        run: AigConstructionRun,
        session_id: str,
        payload: Mapping[str, Any],
        *,
        task_id: str,
        contract_id: str,
    ) -> TaskOutcomeAssessment:
        report_hash = run.authorize_outcome_write(session_id)
        if not isinstance(payload, Mapping):
            raise OutcomeContractError("assessment_required")
        assessment = read_task_outcome_assessment(payload)
        if assessment.author_role != "reader":
            raise OutcomeContractError("outcome_write_forbidden")
        if assessment.report_hash != report_hash:
            raise OutcomeContractError("report_hash_mismatch")
        body = payload.get("body")
        if not isinstance(body, dict):
            raise OutcomeContractError("assessment_required")
        envelope = {
            "body": body,
            "content_hash": assessment.content_hash,
            "object_kind": assessment.object_kind,
            "schema_version": assessment.schema_version,
        }
        return self._store.append_assessment(
            assessment,
            envelope,
            task_id=task_id,
            contract_id=contract_id,
        )

    def adjudicate(
        self,
        *,
        task_id: str,
        report_hash: str,
        external_evidence: Sequence[ExternalOutcomeEvidence] = (),
    ) -> OutcomeAdjudication:
        contract = self._store.read_task_contract(task_id)
        if contract is None:
            raise OutcomeContractError("contract_not_frozen")
        assessment = self._store.read_assessment(task_id, report_hash)
        if assessment is None:
            raise OutcomeContractError("assessment_not_submitted")
        return adjudicate_outcome(assessment, contract, external_evidence)


def _criterion_uncertainty(
    assessment: TaskOutcomeAssessment, contract: OutcomeContract
) -> str | None:
    """Return an uncertain reason, or None when every criterion is explicit."""
    expected = {item.criterion_id for item in contract.acceptance_criteria}
    observed = {item.criterion_id: item.disposition for item in assessment.criterion_dispositions}
    if not expected:
        return REASON_CRITERIA_MISSING
    if not expected <= set(observed):
        return REASON_CRITERIA_MISSING
    if set(observed) != expected:
        return REASON_CRITERIA_CONFLICT
    values = [observed[criterion_id] for criterion_id in sorted(expected)]
    if any(value == "missing" for value in values):
        return REASON_CRITERIA_MISSING
    if any(value == "conflict" for value in values):
        return REASON_CRITERIA_CONFLICT
    if any(value != "explicit" for value in values):
        return REASON_CRITERIA_NOT_EXPLICIT
    if not assessment.evidence_refs:
        return REASON_CRITERIA_MISSING
    return None


def _adjudication(decision: str, reason: str) -> OutcomeAdjudication:
    follow_up = (
        FOLLOW_UP_RECORDED_ONLY if decision == DECISION_CONFIRMED_FAILURE else FOLLOW_UP_FORBIDDEN
    )
    return OutcomeAdjudication(
        policy_version=OUTCOME_EVIDENCE_POLICY_VERSION,
        decision=decision,
        follow_up=follow_up,
        reason=reason,
    )


def _validate_evidence(
    external_evidence: Sequence[ExternalOutcomeEvidence],
) -> tuple[ExternalOutcomeEvidence, ...]:
    if isinstance(external_evidence, (str, bytes)) or not isinstance(external_evidence, Sequence):
        raise OutcomeContractError("invalid_external_evidence")
    evidence: list[ExternalOutcomeEvidence] = []
    for item in external_evidence:
        if not isinstance(item, ExternalOutcomeEvidence):
            raise OutcomeContractError("invalid_external_evidence")
        if item.signal_kind not in _SIGNAL_KINDS or item.signal not in _SIGNALS:
            raise OutcomeContractError("invalid_external_evidence")
        if not item.binding_id or not item.evidence_ref:
            raise OutcomeContractError("invalid_external_evidence")
        evidence.append(item)
    return tuple(evidence)


def _classify_signals(
    contract: OutcomeContract, evidence: Sequence[ExternalOutcomeEvidence]
) -> str:
    bindings = {item.binding_id: item.kind for item in contract.evaluator_bindings}
    if not bindings and not evidence:
        return "reader"
    grouped: dict[str, set[str]] = {binding_id: set() for binding_id in bindings}
    for item in evidence:
        expected = bindings.get(item.binding_id)
        if expected is None or expected != item.signal_kind:
            return "conflict"
        grouped[item.binding_id].add(item.signal)
    if not bindings:
        return "conflict"
    if any(len(signals) > 1 for signals in grouped.values()):
        return "conflict"
    if any(not signals for signals in grouped.values()):
        return "missing"
    observed = {next(iter(signals)) for signals in grouped.values()}
    if observed == {"fail"}:
        return "fail"
    if observed == {"pass"}:
        return "pass"
    return "conflict"


def _parse_result(item: Any) -> ExternalOutcomeEvidence:
    if not isinstance(item, Mapping):
        raise OutcomeContractError("invalid_external_evidence")
    if _TELEMETRY_KEYS.intersection(item):
        raise OutcomeContractError("telemetry_is_not_outcome_evidence")
    if set(item) != _RESULT_KEYS:
        raise OutcomeContractError("invalid_external_evidence")
    binding_id = item.get("binding_id")
    signal_kind = item.get("signal_kind")
    signal = item.get("signal")
    evidence_ref = item.get("evidence_ref")
    if (
        not isinstance(binding_id, str)
        or not binding_id
        or not isinstance(signal_kind, str)
        or signal_kind not in _SIGNAL_KINDS
        or not isinstance(signal, str)
        or signal not in _SIGNALS
        or not isinstance(evidence_ref, str)
        or not evidence_ref
    ):
        raise OutcomeContractError("invalid_external_evidence")
    return ExternalOutcomeEvidence(
        binding_id=binding_id,
        signal_kind=signal_kind,
        signal=signal,
        evidence_ref=evidence_ref,
    )


def _require_safe_id(value: str) -> str:
    if not is_safe_id(value):
        raise OutcomeContractError("invalid_id")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise OutcomeContractError("stored_record_missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OutcomeContractError("stored_record_invalid") from exc
    if not isinstance(payload, dict):
        raise OutcomeContractError("stored_record_invalid")
    return payload


def _read_contract_file(path: Path) -> OutcomeContract:
    try:
        return read_outcome_contract(_read_json(path))
    except ProtocolContractError as exc:
        raise OutcomeContractError("stored_record_invalid") from exc


def _read_assessment_file(path: Path) -> TaskOutcomeAssessment:
    try:
        return read_task_outcome_assessment(_read_json(path))
    except ProtocolContractError as exc:
        raise OutcomeContractError("stored_record_invalid") from exc
