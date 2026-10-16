# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Versioned Coforge coordinator boundary (ADR-0043 P10).

Coforge owns group-chat routing, ``@memory diagnose`` delivery, and the
worker that will consume these documents. This module is the OpenViking
contract only: a closed kind set, a canonical content hash, and a
fail-closed reader. It does not route chat, call a model, or open a
network connection. P12 connects the Coforge coordinator to
``FailureGateOrchestrator.diagnose``.

``@memory retrieve`` is not a document of this boundary. A request whose
intent is anything other than ``diagnose`` is refused.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openviking.session.aig_native_protocol import (
    ProtocolContractError,
    UnsupportedSchemaVersion,
    is_safe_id,
    is_sha256_hex,
    protocol_content_hash,
    protocol_envelope,
)
from openviking.session.fork_node_overlay import ForkOverlayError, require_label_set
from openviking.session.outcome_contract import (
    DECISION_CONFIRMED_FAILURE,
    DECISION_SUCCESS,
    DECISION_UNCERTAIN,
    OUTCOME_EVIDENCE_POLICY_VERSION,
)

COORDINATOR_SCHEMA_VERSION = "coordinator-boundary-v1"
KIND_DIAGNOSE_REQUEST = "coordinator_diagnose_request"
KIND_FAILURE_GATE_EVENT = "coordinator_failure_gate_event"
KIND_DELIVERY_BINDING = "coordinator_delivery_binding"
TRIGGER_COMMIT = "commit"
TRIGGER_DIAGNOSE = "memory_diagnose"
ACTION_RETAINED = "aig_retained"
ACTION_ADVICE = "diagnostic_advice"
ACTION_DISPATCHED = "dispatched"
ACTION_BLOCKED = "blocked_pre_execution"
INTENT_DIAGNOSE = "diagnose"
# Closed kind set for the diagnose HTTP face. session_acl issues ``user:{id}``
# only. Group, role, and wildcard labels default-deny.
COORDINATOR_ALLOWED_LABEL_KINDS = frozenset({"user"})

_DOCUMENT_KINDS = frozenset({KIND_DIAGNOSE_REQUEST, KIND_FAILURE_GATE_EVENT})
_TRIGGERS = frozenset({TRIGGER_COMMIT, TRIGGER_DIAGNOSE})
_ACTIONS = frozenset({ACTION_RETAINED, ACTION_ADVICE, ACTION_DISPATCHED, ACTION_BLOCKED})
_DECISIONS = frozenset({DECISION_SUCCESS, DECISION_CONFIRMED_FAILURE, DECISION_UNCERTAIN})
_ENVELOPE_KEYS = frozenset({"schema_version", "object_kind", "content_hash", "body"})
_REQUEST_KEYS = frozenset(
    {"commit_id", "intent", "principal_labels", "session_id", "source_digest", "task_id"}
)
_EVENT_KEYS = frozenset(
    {
        "action",
        "commit_id",
        "decision",
        "orchestration_run_id",
        "policy_version",
        "reason",
        "request_content_hash",
        "task_id",
        "trigger",
    }
)


@dataclass(frozen=True)
class CoordinatorDiagnoseRequest:
    """One ``@memory diagnose`` request. Retrieve is not this document."""

    schema_version: str
    object_kind: str
    content_hash: str
    commit_id: str
    intent: str
    principal_labels: tuple[str, ...]
    session_id: str
    source_digest: str
    task_id: str


@dataclass(frozen=True)
class CoordinatorFailureGateEvent:
    """One gate outcome for Coforge to observe. It does not start a worker."""

    schema_version: str
    object_kind: str
    content_hash: str
    action: str
    commit_id: str
    decision: str
    orchestration_run_id: str
    policy_version: str
    reason: str
    request_content_hash: str
    task_id: str
    trigger: str


def coordinator_diagnose_request_document(
    *,
    commit_id: str,
    session_id: str,
    task_id: str,
    source_digest: str,
    principal_labels: Sequence[str],
) -> dict[str, Any]:
    """Build a diagnose request. Labels are stored in sorted order."""
    body = {
        "commit_id": _require_id(commit_id, "commit_id"),
        "intent": INTENT_DIAGNOSE,
        "principal_labels": sorted(_collect_labels(principal_labels)),
        "session_id": _require_id(session_id, "session_id"),
        "source_digest": _require_hex(source_digest, "source_digest"),
        "task_id": _require_id(task_id, "task_id"),
    }
    return protocol_envelope(COORDINATOR_SCHEMA_VERSION, KIND_DIAGNOSE_REQUEST, body)


def coordinator_failure_gate_event_document(
    *,
    action: str,
    commit_id: str,
    decision: str,
    orchestration_run_id: str,
    reason: str,
    request_content_hash: str,
    task_id: str,
    trigger: str,
) -> dict[str, Any]:
    """Build a gate event. ``policy_version`` is the outcome policy, not a caller field."""
    body = {
        "action": _require_choice(action, "action", _ACTIONS),
        "commit_id": _require_id(commit_id, "commit_id"),
        "decision": _require_choice(decision, "decision", _DECISIONS),
        "orchestration_run_id": _require_hex(orchestration_run_id, "orchestration_run_id"),
        "policy_version": OUTCOME_EVIDENCE_POLICY_VERSION,
        "reason": _require_text(reason, "reason"),
        "request_content_hash": _require_hex(request_content_hash, "request_content_hash"),
        "task_id": _require_id(task_id, "task_id"),
        "trigger": _require_choice(trigger, "trigger", _TRIGGERS),
    }
    return protocol_envelope(COORDINATOR_SCHEMA_VERSION, KIND_FAILURE_GATE_EVENT, body)


def coordinator_delivery_binding_hash(
    *,
    commit_id: str,
    source_digest: str,
    task_id: str,
    trigger: str,
) -> str:
    """Hash a commit delivery. This kind is not a readable transport document."""
    body = {
        "commit_id": commit_id,
        "source_digest": source_digest,
        "task_id": task_id,
        "trigger": trigger,
    }
    return protocol_content_hash(COORDINATOR_SCHEMA_VERSION, KIND_DELIVERY_BINDING, body)


def read_coordinator_document(
    payload: Mapping[str, Any],
) -> CoordinatorDiagnoseRequest | CoordinatorFailureGateEvent:
    """Parse one boundary document. Unknown versions and kinds fail closed."""
    if not isinstance(payload, Mapping):
        raise ProtocolContractError("coordinator document must be a dict")
    version = payload.get("schema_version")
    if version != COORDINATOR_SCHEMA_VERSION:
        raise UnsupportedSchemaVersion(
            f"unsupported schema_version {version!r}; refusing to guess or downgrade"
        )
    kind = payload.get("object_kind")
    if kind not in _DOCUMENT_KINDS:
        raise ProtocolContractError(
            f"schema_version {version!r} does not accept object_kind {kind!r}"
        )
    if set(payload) != _ENVELOPE_KEYS:
        raise ProtocolContractError("coordinator document keys are not the envelope")
    body = payload.get("body")
    if not isinstance(body, dict):
        raise ProtocolContractError("body must be a dict")
    expected = _REQUEST_KEYS if kind == KIND_DIAGNOSE_REQUEST else _EVENT_KEYS
    if set(body) != expected:
        raise ProtocolContractError(f"object_kind {kind!r} body keys do not match")
    digest = payload.get("content_hash")
    if digest != protocol_content_hash(COORDINATOR_SCHEMA_VERSION, str(kind), body):
        raise ProtocolContractError("content_hash mismatch")
    if kind == KIND_DIAGNOSE_REQUEST:
        return _read_request(body, str(digest))
    return _read_event(body, str(digest))


def read_coordinator_diagnose_request(payload: Mapping[str, Any]) -> CoordinatorDiagnoseRequest:
    """Parse a diagnose request. Events and other kinds are refused."""
    parsed = read_coordinator_document(payload)
    if not isinstance(parsed, CoordinatorDiagnoseRequest):
        raise ProtocolContractError(f"object_kind must be {KIND_DIAGNOSE_REQUEST!r}")
    return parsed


def coordinator_boundary_labels(value: Any) -> frozenset[str]:
    """Return labels this boundary accepts. Unknown kinds default-deny.

    ``session_acl`` issues ``user:{user_id}``. ``user:*`` is a storage
    wildcard, not an owner label, and is refused.
    """
    try:
        labels = require_label_set(value, code="acl_denied")
    except ForkOverlayError as exc:
        raise ProtocolContractError(exc.code) from exc
    kept = frozenset(item for item in labels if _is_boundary_label(item))
    if not kept:
        raise ProtocolContractError("acl_denied")
    return kept


def _read_request(body: Mapping[str, Any], digest: str) -> CoordinatorDiagnoseRequest:
    intent = body.get("intent")
    if intent != INTENT_DIAGNOSE:
        raise ProtocolContractError("intent must be diagnose")
    return CoordinatorDiagnoseRequest(
        schema_version=COORDINATOR_SCHEMA_VERSION,
        object_kind=KIND_DIAGNOSE_REQUEST,
        content_hash=digest,
        commit_id=_require_id(body.get("commit_id"), "commit_id"),
        intent=INTENT_DIAGNOSE,
        principal_labels=tuple(_label_list(body.get("principal_labels"))),
        session_id=_require_id(body.get("session_id"), "session_id"),
        source_digest=_require_hex(body.get("source_digest"), "source_digest"),
        task_id=_require_id(body.get("task_id"), "task_id"),
    )


def _read_event(body: Mapping[str, Any], digest: str) -> CoordinatorFailureGateEvent:
    if body.get("policy_version") != OUTCOME_EVIDENCE_POLICY_VERSION:
        raise ProtocolContractError("unsupported policy_version")
    return CoordinatorFailureGateEvent(
        schema_version=COORDINATOR_SCHEMA_VERSION,
        object_kind=KIND_FAILURE_GATE_EVENT,
        content_hash=digest,
        action=_require_choice(body.get("action"), "action", _ACTIONS),
        commit_id=_require_id(body.get("commit_id"), "commit_id"),
        decision=_require_choice(body.get("decision"), "decision", _DECISIONS),
        orchestration_run_id=_require_hex(body.get("orchestration_run_id"), "orchestration_run_id"),
        policy_version=OUTCOME_EVIDENCE_POLICY_VERSION,
        reason=_require_text(body.get("reason"), "reason"),
        request_content_hash=_require_hex(body.get("request_content_hash"), "request_content_hash"),
        task_id=_require_id(body.get("task_id"), "task_id"),
        trigger=_require_choice(body.get("trigger"), "trigger", _TRIGGERS),
    )


def _is_boundary_label(label: str) -> bool:
    kind, separator, rest = label.partition(":")
    if not separator or kind not in COORDINATOR_ALLOWED_LABEL_KINDS:
        return False
    return bool(rest) and "*" not in rest


def _collect_labels(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) == 0:
        raise ProtocolContractError("principal_labels must be a non-empty list")
    try:
        labels = require_label_set(value, code="principal_labels must be exact labels")
    except ForkOverlayError as exc:
        raise ProtocolContractError(exc.code) from exc
    if len(value) != len(labels):
        raise ProtocolContractError("principal_labels must be unique")
    return list(value)


def _label_list(value: Any) -> list[str]:
    labels = _collect_labels(value)
    if labels != sorted(labels):
        raise ProtocolContractError("principal_labels must be sorted")
    return labels


def _require_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not is_safe_id(value):
        raise ProtocolContractError(f"{field} must be a safe id")
    return value


def _require_hex(value: Any, field: str) -> str:
    if not is_sha256_hex(value):
        raise ProtocolContractError(f"{field} must be a sha256 hex digest")
    return str(value)


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolContractError(f"{field} must be a non-empty string")
    return value


def _require_choice(value: Any, field: str, choices: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ProtocolContractError(f"{field} must be one of {sorted(choices)}")
    return value
