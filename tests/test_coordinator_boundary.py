# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""P10: versioned Coforge coordinator boundary.

The reader accepts the golden request and event. Unknown versions, unknown
kinds, and tampered hashes fail closed. Group-chat routing is not this module.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from openviking.session.aig_native_protocol import (
    ProtocolContractError,
    UnsupportedSchemaVersion,
    protocol_content_hash,
    read_protocol_document,
)
from openviking.session.coordinator_boundary import (
    COORDINATOR_SCHEMA_VERSION,
    KIND_DIAGNOSE_REQUEST,
    KIND_FAILURE_GATE_EVENT,
    coordinator_boundary_labels,
    coordinator_diagnose_request_document,
    coordinator_failure_gate_event_document,
    read_coordinator_document,
)
from openviking.session.outcome_contract import (
    DECISION_UNCERTAIN,
    OUTCOME_EVIDENCE_POLICY_VERSION,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "adr0043"
_REQUEST = FIXTURE_DIR / "coordinator_diagnose_request.json"
_EVENT = FIXTURE_DIR / "coordinator_failure_gate_event.json"


def _spec_hash(kind: str, body: dict) -> str:
    return protocol_content_hash(COORDINATOR_SCHEMA_VERSION, kind, body)


def test_golden_request_and_event_roundtrip() -> None:
    request = json.loads(_REQUEST.read_text(encoding="utf-8"))
    event = json.loads(_EVENT.read_text(encoding="utf-8"))
    parsed_request = read_coordinator_document(request)
    parsed_event = read_coordinator_document(event)
    assert parsed_request == read_coordinator_document(json.loads(json.dumps(request)))
    assert parsed_event == read_coordinator_document(json.loads(json.dumps(event)))
    assert parsed_request.object_kind == KIND_DIAGNOSE_REQUEST
    assert parsed_request.intent == "diagnose"
    assert parsed_request.commit_id == "commit-diagnose-1"
    assert parsed_request.content_hash == _spec_hash(KIND_DIAGNOSE_REQUEST, request["body"])
    assert parsed_event.object_kind == KIND_FAILURE_GATE_EVENT
    assert parsed_event.decision == DECISION_UNCERTAIN
    assert parsed_event.action == "diagnostic_advice"
    assert parsed_event.trigger == "memory_diagnose"
    assert parsed_event.request_content_hash == parsed_request.content_hash
    assert parsed_event.content_hash == _spec_hash(KIND_FAILURE_GATE_EVENT, event["body"])
    built = coordinator_diagnose_request_document(
        commit_id=parsed_request.commit_id,
        session_id=parsed_request.session_id,
        task_id=parsed_request.task_id,
        source_digest=parsed_request.source_digest,
        principal_labels=parsed_request.principal_labels,
    )
    assert built == request
    built_event = coordinator_failure_gate_event_document(
        action=parsed_event.action,
        commit_id=parsed_event.commit_id,
        decision=parsed_event.decision,
        orchestration_run_id=parsed_event.orchestration_run_id,
        reason=parsed_event.reason,
        request_content_hash=parsed_event.request_content_hash,
        task_id=parsed_event.task_id,
        trigger=parsed_event.trigger,
    )
    assert built_event == event


def test_reader_fail_closed_on_version_kind_hash_and_intent() -> None:
    request = json.loads(_REQUEST.read_text(encoding="utf-8"))
    unknown_version = json.loads(json.dumps(request))
    unknown_version["schema_version"] = "coordinator-boundary-v9"
    unknown_version["content_hash"] = protocol_content_hash(
        "coordinator-boundary-v9", unknown_version["object_kind"], unknown_version["body"]
    )
    with pytest.raises(UnsupportedSchemaVersion):
        read_coordinator_document(unknown_version)

    aig_version = json.loads(json.dumps(request))
    aig_version["schema_version"] = "aig-native-v2"
    aig_version["content_hash"] = protocol_content_hash(
        "aig-native-v2", aig_version["object_kind"], aig_version["body"]
    )
    with pytest.raises(UnsupportedSchemaVersion):
        read_coordinator_document(aig_version)

    unknown_kind = json.loads(json.dumps(request))
    unknown_kind["object_kind"] = "coordinator_retrieve_request"
    unknown_kind["content_hash"] = _spec_hash(
        "coordinator_retrieve_request", unknown_kind["body"]
    )
    with pytest.raises(ProtocolContractError):
        read_coordinator_document(unknown_kind)

    tampered = json.loads(json.dumps(request))
    tampered["body"]["commit_id"] = "commit-rewritten"
    with pytest.raises(ProtocolContractError, match="content_hash"):
        read_coordinator_document(tampered)

    extra = json.loads(json.dumps(request))
    extra["body"]["route"] = "room-1"
    extra["content_hash"] = _spec_hash(KIND_DIAGNOSE_REQUEST, extra["body"])
    with pytest.raises(ProtocolContractError):
        read_coordinator_document(extra)

    retrieve = coordinator_diagnose_request_document(
        commit_id="commit-diagnose-1",
        session_id="sess-diagnose",
        task_id="task-diagnose",
        source_digest="ab" * 32,
        principal_labels=("user:carol",),
    )
    retrieve["body"]["intent"] = "retrieve"
    retrieve["content_hash"] = _spec_hash(KIND_DIAGNOSE_REQUEST, retrieve["body"])
    with pytest.raises(ProtocolContractError, match="diagnose"):
        read_coordinator_document(retrieve)

    with pytest.raises(ProtocolContractError):
        read_protocol_document(request)


def test_boundary_labels_accept_user_and_refuse_other_kinds() -> None:
    assert coordinator_boundary_labels(["user:carol"]) == frozenset({"user:carol"})
    with pytest.raises(ProtocolContractError, match="acl_denied"):
        coordinator_boundary_labels(["group:eng"])
    with pytest.raises(ProtocolContractError, match="acl_denied"):
        coordinator_boundary_labels(["user:*"])
    with pytest.raises(ProtocolContractError, match="acl_denied"):
        coordinator_boundary_labels(["role:coordinator"])


def test_boundary_names_coforge_as_the_implementer() -> None:
    source = inspect.getsource(
        __import__("openviking.session.coordinator_boundary", fromlist=["x"])
    )
    assert "Coforge" in source
    assert "P12" in source
    for banned in ("socket", "urllib", "httpx", "subprocess"):
        assert banned not in source
    assert OUTCOME_EVIDENCE_POLICY_VERSION == "outcome-evidence-v1"
