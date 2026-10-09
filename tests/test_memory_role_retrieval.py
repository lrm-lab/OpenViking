# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Role-projected discovery entry: Memory envelope, replay, and facade wiring."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest

from openviking.session.memory_role_retrieval import (
    EXACT_CAPABILITY_ABSENT,
    EXACT_LANE_NOT_IN_DISCOVERY_ENTRY,
    PLAN_DRIFT,
    SCHEMA_VERSION,
    USE_CLASS_POLICY,
    BoundContext,
    RetrievalContractError,
    RoleRetrievalService,
    public_card_from_retrieval,
)
from openviking.session.retrieval_facade import RetrievalCard
from openviking.session.retrieval_wiring import retrieve_projected

SCOPE = "scope-1"
SNAP = "snap-1"
LEASE = "lease-1"
_MISSING = object()

ENVELOPE_KEYS = frozenset(
    {
        "schemaVersion",
        "operationId",
        "planHash",
        "duplicate",
        "resolvedContext",
        "policyVersions",
        "auditRef",
        "memoryCandidateCards",
        "analogicalPriorCards",
        "discoveryAudit",
    }
)
AUDIT_KEYS = frozenset(
    {"executionState", "resultState", "reasonCode", "returnedCount", "truncated"}
)
SEMANTIC_ROLES = ("reader", "critic", "refiner", "builder", "diagnosis")
EXACT_KEYS = (
    "exact",
    "current_log_exact",
    "currentLogExact",
    "currentLogExactMaxHits",
    "currentEvidenceRefs",
    "current_evidence_refs",
    "exactLaneAudit",
    "exact_lane_audit",
    "exactQuota",
    "exactLease",
    "exactDeadline",
    "exactCancel",
    "exactTruncation",
    "exactCompleteness",
    "Exact",
    "CURRENT_LOG_EXACT",
    "exact-quota",
)
UNKNOWN_FIELDS = (
    "principal",
    "deadline",
    "principalId",
    "deadlineAt",
    "role",
    "cancellation",
    "adapterHandle",
    "scopeRef",
    "snapshotRef",
    "budgetLeaseRef",
    "intent",
    "operation_id",
    "exactly",
    "query",
    "topK",
    "limit",
)
ROLE_CODES = (
    ("memory", EXACT_CAPABILITY_ABSENT),
    *((role, EXACT_LANE_NOT_IN_DISCOVERY_ENTRY) for role in SEMANTIC_ROLES),
)
FORBIDDEN = (
    "currentEvidenceRefs",
    "current_evidence_refs",
    "currentEvidence",
    "current_evidence",
    "current_log_exact",
    "currentLogExact",
    "exactLaneAudit",
    "exact_lane_audit",
    "exactQuota",
    "exactLease",
    "exactDeadline",
    "exactCancel",
    "exactTruncation",
    "exactCompleteness",
    "acl_labels",
    "aclLabels",
    "score_breakdown",
    "scoreBreakdown",
    "nativeScore",
    "principal",
    "deadline",
    "stackTrace",
    "timestamp",
    "elapsed",
    "excludedCount",
    "not_dispatched",
    "terminated",
    "ACL-SECRET",
    "SCORE-SENTINEL",
    "PRINCIPAL-LABEL",
    "POINTER-SECRET",
    "BODY-SECRET",
    "SECRET-ELIGIBLE",
)


class Harness:
    def __init__(self) -> None:
        self.cards: list[Any] = []
        self.raw: Any = _MISSING
        self.discover_calls: list[tuple[Any, Any]] = []
        self.authority_calls: list[str] = []
        self.scope = SCOPE
        self.snap = SNAP
        self.lease = LEASE
        self.authority_override = None
        self.service = RoleRetrievalService()

    def authority(self, role: str) -> Any:
        self.authority_calls.append(role)
        if self.authority_override is not None:
            return self.authority_override(role)
        return BoundContext(self.scope, self.snap, self.lease)

    def discover(self, intent: str | None, pattern: Mapping[str, str] | None) -> Any:
        self.discover_calls.append((intent, pattern))
        if self.raw is not _MISSING:
            return self.raw
        copied: list[Any] = []
        for card in self.cards:
            copied.append(dict(card) if isinstance(card, Mapping) else card)
        return copied

    def retrieve(self, role: str, request: Mapping[str, Any] | Any) -> dict[str, Any]:
        return self.service.retrieve(
            role=role,
            request=request,
            authority=self.authority,
            discover=self.discover,
        )


class RecordingFacade:
    def __init__(self, cards: list[Any]) -> None:
        self.cards = cards
        self.calls: list[tuple[Any, Any, Any]] = []

    def retrieve(self, query: Any, principal_labels: Any, profile: Any = None) -> Any:
        self.calls.append((query, principal_labels, profile))
        return SimpleNamespace(
            cards=tuple(self.cards),
            truncated_by_rerank_clamp=True,
            selection_audit={"eligible": "SECRET-ELIGIBLE"},
        )


@pytest.fixture
def harness() -> Harness:
    return Harness()


def assert_rejected(
    harness: Harness,
    role: str,
    request: Any,
    code: str,
    *,
    dispatch: bool = False,
) -> None:
    discover_before = len(harness.discover_calls)
    authority_before = len(harness.authority_calls)
    with pytest.raises(RetrievalContractError) as caught:
        harness.retrieve(role, request)
    assert caught.value.code == code
    assert str(caught.value) == code
    assert isinstance(caught.value, ValueError)
    if dispatch:
        assert len(harness.discover_calls) > discover_before
        assert len(harness.authority_calls) > authority_before
        return
    assert len(harness.discover_calls) == discover_before
    assert len(harness.authority_calls) == authority_before


def assert_public_envelope(
    env: dict[str, Any],
    *,
    operation_id: str,
    duplicate: bool = False,
    absent: tuple[str, ...] = (),
) -> None:
    assert set(env) == ENVELOPE_KEYS
    assert env["schemaVersion"] == SCHEMA_VERSION == "RetrievalEnvelopeV1"
    assert env["operationId"] == operation_id
    assert re.fullmatch(r"[0-9a-f]{64}", env["planHash"])
    assert env["duplicate"] is duplicate
    assert set(env["resolvedContext"]) == {"scopeRef", "snapshotRef", "budgetLeaseRef"}
    assert env["resolvedContext"] == {
        "scopeRef": SCOPE,
        "snapshotRef": SNAP,
        "budgetLeaseRef": LEASE,
    }
    assert env["policyVersions"] == {"useClass": USE_CLASS_POLICY}
    assert USE_CLASS_POLICY == "memory-use-class-pending-central-v1"
    assert env["auditRef"] == f"audit-{env['planHash'][:16]}"
    audit = env["discoveryAudit"]
    assert set(audit) == AUDIT_KEYS
    assert audit["executionState"] == "completed"
    assert audit["reasonCode"] is None
    assert audit["truncated"] is False
    candidates = env["memoryCandidateCards"]
    priors = env["analogicalPriorCards"]
    assert isinstance(candidates, list)
    assert isinstance(priors, list)
    for card in candidates:
        assert set(card) == {"id", "kind", "useClass"}
        assert card["useClass"] == "memory_candidate"
    for card in priors:
        assert set(card) == {"id", "kind", "useClass"}
        assert card["useClass"] == "analogical_prior"
    assert audit["returnedCount"] == len(candidates) + len(priors)
    if audit["returnedCount"] == 0:
        assert audit["resultState"] == "empty"
    else:
        assert audit["resultState"] == "hits"
    kinds = {card["kind"] for card in [*candidates, *priors]}
    assert kinds.isdisjoint({"episode", "insight"})
    encoded = json.dumps(env, ensure_ascii=False)
    for token in (*FORBIDDEN, *absent):
        assert token not in encoded


def facade_card(kind: str, payload_ref: str, *, channel: str = "verified", rank: int = 0) -> RetrievalCard:
    return RetrievalCard(
        kind=kind,  # type: ignore[arg-type]
        channel=channel,  # type: ignore[arg-type]
        in_channel_rank=rank,
        status_label="active",
        source_pointer={"marker": "POINTER-SECRET"},
        payload_ref=payload_ref,
        score_breakdown={"bm25": "SCORE-SENTINEL"},
        acl_labels=("ACL-SECRET-LABEL",),
        text="BODY-SECRET",
    )


def test_contract_codes_are_stable() -> None:
    assert EXACT_CAPABILITY_ABSENT == "exact_capability_absent_for_memory_agent"
    assert EXACT_LANE_NOT_IN_DISCOVERY_ENTRY == "exact_lane_not_in_discovery_entry"
    assert PLAN_DRIFT == "plan_drift"


@pytest.mark.parametrize("role,code", ROLE_CODES)
@pytest.mark.parametrize("key", EXACT_KEYS)
def test_exact_keys_rejected_before_discover(harness: Harness, role: str, code: str, key: str) -> None:
    assert_rejected(
        harness,
        role,
        {"intentText": "guide", "operationId": "op", key: "5xx"},
        code,
    )


def test_exact_only_payload_is_not_treated_as_an_empty_plan(harness: Harness) -> None:
    assert_rejected(harness, "memory", {"exact": "5xx"}, EXACT_CAPABILITY_ABSENT)
    assert_rejected(
        harness,
        "reader",
        {"current_log_exact": {"expression": "5xx", "syntax": "regex"}},
        EXACT_LANE_NOT_IN_DISCOVERY_ENTRY,
    )


def test_nested_exact_is_rejected_before_discover(harness: Harness) -> None:
    legacy = {
        "intentText": "guide",
        "legacy": [{"current_log_exact": {"expression": "5xx", "syntax": "literal"}}],
    }
    assert_rejected(harness, "memory", legacy, EXACT_CAPABILITY_ABSENT)
    assert_rejected(
        harness,
        "diagnosis",
        {
            "operationId": "op",
            "intentText": "guide",
            "legacy": [{"current_log_exact": {"expression": "5xx", "syntax": "literal"}}],
        },
        EXACT_LANE_NOT_IN_DISCOVERY_ENTRY,
    )


def test_exact_wins_over_unknown_and_empty_checks(harness: Harness) -> None:
    request = {
        "principal": "self",
        "deadline": "2026-10-09",
        "dependencyPattern": {"exact": "5xx", "wantedArtifact": "wheel"},
    }
    assert_rejected(harness, "memory", request, EXACT_CAPABILITY_ABSENT)
    assert_rejected(
        harness,
        "builder",
        {**request, "operationId": "op"},
        EXACT_LANE_NOT_IN_DISCOVERY_ENTRY,
    )


def test_intent_text_may_mention_exact(harness: Harness) -> None:
    env = harness.retrieve(
        "memory",
        {"operationId": "op", "intentText": "use exact rollback"},
    )
    assert_public_envelope(env, operation_id="op")
    assert harness.discover_calls == [("use exact rollback", None)]
    assert harness.authority_calls == ["memory"]


@pytest.mark.parametrize("role", ("memory", "reader"))
@pytest.mark.parametrize("field", UNKNOWN_FIELDS)
def test_self_reported_and_unknown_fields_are_rejected(
    harness: Harness, role: str, field: str
) -> None:
    assert_rejected(
        harness,
        role,
        {"intentText": "guide", "operationId": "op", field: "self-reported"},
        "unknown_field",
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"intentText": None},
        {"dependencyPattern": None},
        {"dependencyPattern": {}},
        {"intentText": None, "dependencyPattern": None},
        {"operationId": "op-empty"},
        {"operationId": ""},
        {"dependencyPattern": {"wantedArtifact": None, "wantedEffect": None}},
    ],
)
def test_memory_empty_plan_is_rejected_before_discover(
    harness: Harness, payload: dict[str, Any]
) -> None:
    assert_rejected(harness, "memory", payload, "empty_discovery_plan")


@pytest.mark.parametrize(
    "payload",
    [
        {"operationId": "op-empty"},
        {"operationId": "op-empty", "intentText": None},
        {"operationId": "op-empty", "dependencyPattern": {}},
        {"operationId": "op-empty", "dependencyPattern": {"wantedEffect": None}},
    ],
)
def test_semantic_empty_plan_is_rejected_before_discover(
    harness: Harness, payload: dict[str, Any]
) -> None:
    assert_rejected(harness, "refiner", payload, "empty_discovery_plan")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"intentText": "guide"},
        {"operationId": None, "intentText": "guide"},
        {"operationId": "  ", "intentText": "guide"},
        {"operationId": 7, "intentText": "guide"},
    ],
)
def test_semantic_plan_requires_operation_id(harness: Harness, payload: dict[str, Any]) -> None:
    assert_rejected(harness, "critic", payload, "invalid_operation_id")


@pytest.mark.parametrize(
    "payload,code",
    [
        ({"intentText": ""}, "invalid_intent"),
        ({"intentText": "   "}, "invalid_intent"),
        ({"intentText": 1}, "invalid_intent"),
        ({"dependencyPattern": "wheel"}, "invalid_dependency_pattern"),
        ({"dependencyPattern": {"wantedArtifact": ""}}, "invalid_dependency_pattern"),
        ({"dependencyPattern": {"wantedArtifact": "  "}}, "invalid_dependency_pattern"),
        ({"dependencyPattern": {"wantedArtifact": 1}}, "invalid_dependency_pattern"),
        ({"dependencyPattern": ["wheel"]}, "invalid_dependency_pattern"),
        (
            {"intentText": "guide", "dependencyPattern": {"wantedArtifact": "a", "extra": "b"}},
            "unknown_field",
        ),
        ({"intentText": "guide", "operationId": ""}, "invalid_operation_id"),
        ({"intentText": "guide", "operationId": 4}, "invalid_operation_id"),
        ({"intentText": "guide", "operationId": "   "}, "invalid_operation_id"),
    ],
)
def test_memory_rejects_malformed_plan_fields(
    harness: Harness, payload: dict[str, Any], code: str
) -> None:
    assert_rejected(harness, "memory", payload, code)


@pytest.mark.parametrize("payload", ["guide", ["exact"], 1, None, b"{}"])
def test_request_must_be_a_mapping(harness: Harness, payload: Any) -> None:
    assert_rejected(harness, "memory", payload, "invalid_request")
    assert_rejected(harness, "reader", payload, "invalid_request")


@pytest.mark.parametrize("role", ["planner", "memory-agent", ""])
@pytest.mark.parametrize(
    "payload",
    [
        {"intentText": "guide"},
        {"intentText": "guide", "exact": "5xx"},
        {"exact": "5xx"},
    ],
)
def test_unknown_role_is_rejected_before_discover(
    harness: Harness, role: str, payload: dict[str, Any]
) -> None:
    assert_rejected(harness, role, payload, "unknown_role")


def test_memory_completed_empty_audit_is_not_a_rejection(harness: Harness) -> None:
    env = harness.retrieve("memory", {"operationId": "op", "intentText": "guide"})
    assert_public_envelope(env, operation_id="op")
    assert env["memoryCandidateCards"] == []
    assert env["analogicalPriorCards"] == []
    assert env["discoveryAudit"]["resultState"] == "empty"
    assert env["discoveryAudit"]["returnedCount"] == 0
    assert harness.discover_calls == [("guide", None)]


def test_pending_kind_map_drops_episode_insight_and_private_fields(harness: Harness) -> None:
    harness.cards = [
        {"id": " f1 ", "kind": "fork", "acl_labels": ["ACL-SECRET"], "score": "SCORE-SENTINEL"},
        {"id": "b1", "kind": "branch", "useClass": "memory_candidate"},
        {"id": "s1", "kind": "skill"},
        {"id": "r1", "kind": "skill_revision", "text": "BODY-SECRET"},
        {"id": "p1", "kind": "block_projection"},
        {"id": "c1", "kind": "claim_projection"},
        {"id": "g1", "kind": "segment"},
        {"id": "a1", "kind": "atom"},
        {"id": "e1", "kind": "episode", "useClass": "memory_candidate", "score": "SCORE-SENTINEL"},
        {"id": "i1", "kind": "insight", "acl_labels": ["ACL-SECRET"]},
        {"id": "h1", "kind": "historical_subgraph"},
    ]
    env = harness.retrieve(
        "memory",
        {
            "operationId": "op-cards",
            "intentText": " INTENT-SENTINEL ",
            "dependencyPattern": {"wantedEffect": " keep ", "wantedArtifact": " ARTIFACT-MARKER "},
        },
    )
    assert_public_envelope(
        env,
        operation_id="op-cards",
        absent=("INTENT-SENTINEL", "ARTIFACT-MARKER"),
    )
    assert [card["id"] for card in env["memoryCandidateCards"]] == ["f1", "b1", "s1", "r1"]
    assert [card["kind"] for card in env["memoryCandidateCards"]] == [
        "fork",
        "branch",
        "skill",
        "skill_revision",
    ]
    assert [card["id"] for card in env["analogicalPriorCards"]] == ["p1", "c1", "g1", "a1"]
    assert env["discoveryAudit"]["returnedCount"] == 8
    assert harness.discover_calls == [
        (
            "INTENT-SENTINEL",
            {"wantedArtifact": "ARTIFACT-MARKER", "wantedEffect": "keep"},
        )
    ]


def test_discovery_entry_does_not_apply_corpus_seat_allocation(harness: Harness) -> None:
    harness.cards = [{"id": f"f{i}", "kind": "fork"} for i in range(9)]
    harness.cards += [{"id": f"p{i}", "kind": "block_projection"} for i in range(5)]
    env = harness.retrieve("memory", {"operationId": "op", "intentText": "guide"})
    assert len(env["memoryCandidateCards"]) == 9
    assert len(env["analogicalPriorCards"]) == 3
    assert env["discoveryAudit"]["returnedCount"] == 12
    assert env["discoveryAudit"]["truncated"] is True


def test_null_pattern_field_is_treated_as_absent(harness: Harness) -> None:
    harness.retrieve(
        "memory",
        {
            "operationId": "op",
            "dependencyPattern": {"wantedArtifact": "wheel", "wantedEffect": None},
        },
    )
    assert harness.discover_calls == [(None, {"wantedArtifact": "wheel"})]


def test_pattern_only_memory_plan_dispatches_discovery(harness: Harness) -> None:
    harness.cards = [{"id": "a1", "kind": "atom"}]
    env = harness.retrieve(
        "memory",
        {"operationId": "op", "dependencyPattern": {"wantedArtifact": " 预案 "}},
    )
    assert_public_envelope(env, operation_id="op")
    assert env["memoryCandidateCards"] == []
    assert env["analogicalPriorCards"][0]["kind"] == "atom"
    assert harness.discover_calls == [(None, {"wantedArtifact": "预案"})]


@pytest.mark.parametrize("role", SEMANTIC_ROLES)
def test_authorized_semantic_roles_use_the_discovery_envelope(harness: Harness, role: str) -> None:
    harness.cards = [
        {"id": "f1", "kind": "fork"},
        {"id": "e1", "kind": "episode"},
        {"id": "p1", "kind": "claim_projection"},
    ]
    env = harness.retrieve(role, {"operationId": f"op-{role}", "intentText": "guide"})
    assert_public_envelope(env, operation_id=f"op-{role}")
    assert [card["id"] for card in env["memoryCandidateCards"]] == ["f1"]
    assert [card["id"] for card in env["analogicalPriorCards"]] == ["p1"]
    assert harness.authority_calls == [role]
    assert harness.discover_calls == [("guide", None)]


def test_diagnosis_pattern_only_plan(harness: Harness) -> None:
    env = harness.retrieve(
        "diagnosis",
        {"operationId": " op-d ", "dependencyPattern": {"wantedEffect": " stop "}},
    )
    assert_public_envelope(env, operation_id="op-d")
    assert harness.discover_calls == [(None, {"wantedEffect": "stop"})]


def test_memory_can_omit_operation_id(harness: Harness) -> None:
    first = harness.retrieve("memory", {"intentText": "guide"})
    second = harness.retrieve("memory", {"intentText": "guide", "operationId": None})
    assert first["operationId"]
    assert second["operationId"]
    assert first["operationId"] != second["operationId"]
    assert first["planHash"] == second["planHash"]
    assert first["duplicate"] is False
    assert second["duplicate"] is False
    assert len(harness.discover_calls) == 2


def test_same_plan_replays_without_discover(harness: Harness) -> None:
    harness.cards = [{"id": "f1", "kind": "fork", "score": "SCORE-SENTINEL"}]
    first = harness.retrieve("memory", {"operationId": " op ", "intentText": " guide "})
    harness.cards = [{"id": "other", "kind": "branch"}]
    second = harness.retrieve("memory", {"operationId": "op", "intentText": "guide"})
    third = harness.retrieve(
        "memory",
        {"operationId": "op", "intentText": "guide", "dependencyPattern": None},
    )
    assert first["duplicate"] is False
    assert first["operationId"] == "op"
    assert second["duplicate"] is True
    assert third["duplicate"] is True
    assert first["planHash"] == second["planHash"] == third["planHash"]
    assert [card["id"] for card in second["memoryCandidateCards"]] == ["f1"]
    assert len(harness.discover_calls) == 1
    assert harness.authority_calls == ["memory", "memory", "memory"]
    assert_public_envelope(second, operation_id="op", duplicate=True)


def test_replay_is_isolated_from_caller_mutation(harness: Harness) -> None:
    harness.cards = [{"id": "f1", "kind": "fork"}]
    request = {"operationId": "op", "intentText": "guide"}
    first = harness.retrieve("memory", request)
    first["duplicate"] = True
    first["resolvedContext"]["scopeRef"] = "hacked"
    first["memoryCandidateCards"].append(
        {"id": "x", "kind": "fork", "useClass": "memory_candidate"}
    )
    first["discoveryAudit"]["returnedCount"] = 99
    second = harness.retrieve("memory", request)
    assert second["duplicate"] is True
    assert second["resolvedContext"]["scopeRef"] == SCOPE
    assert [card["id"] for card in second["memoryCandidateCards"]] == ["f1"]
    assert second["discoveryAudit"]["returnedCount"] == 1
    assert len(harness.discover_calls) == 1


def test_pattern_whitespace_and_key_order_do_not_drift(harness: Harness) -> None:
    first = harness.retrieve(
        "memory",
        {
            "operationId": "op",
            "dependencyPattern": {"wantedEffect": " safer ", "wantedArtifact": " plan "},
        },
    )
    second = harness.retrieve(
        "memory",
        {
            "operationId": "op",
            "dependencyPattern": {"wantedArtifact": "plan", "wantedEffect": "safer"},
        },
    )
    assert second["duplicate"] is True
    assert first["planHash"] == second["planHash"]
    assert harness.discover_calls == [(None, {"wantedArtifact": "plan", "wantedEffect": "safer"})]


def test_different_roles_do_not_share_a_plan_hash(harness: Harness) -> None:
    harness.cards = [{"id": "f1", "kind": "fork"}]
    first = harness.retrieve("memory", {"operationId": "op-a", "intentText": "guide"})
    harness.cards = [{"id": "f2", "kind": "branch"}]
    second = harness.retrieve("reader", {"operationId": "op-b", "intentText": "guide"})
    assert first["duplicate"] is False
    assert second["duplicate"] is False
    assert first["planHash"] != second["planHash"]
    assert [card["id"] for card in first["memoryCandidateCards"]] == ["f1"]
    assert [card["id"] for card in second["memoryCandidateCards"]] == ["f2"]
    assert len(harness.discover_calls) == 2


def test_same_role_and_plan_hash_matches_across_operation_ids(harness: Harness) -> None:
    first = harness.retrieve("memory", {"operationId": "op-a", "intentText": "guide"})
    second = harness.retrieve("memory", {"operationId": "op-b", "intentText": "guide"})
    assert first["planHash"] == second["planHash"]
    assert first["operationId"] != second["operationId"]


def test_plan_drift_does_not_discover_or_replace_the_stored_plan(harness: Harness) -> None:
    harness.cards = [{"id": "f1", "kind": "fork"}]
    request = {
        "operationId": "op",
        "intentText": "guide",
        "dependencyPattern": {"wantedArtifact": "wheel"},
    }
    first = harness.retrieve("memory", request)
    with pytest.raises(RetrievalContractError) as caught:
        harness.retrieve(
            "memory",
            {"operationId": "op", "dependencyPattern": {"wantedArtifact": "tire"}},
        )
    assert caught.value.code == PLAN_DRIFT
    with pytest.raises(RetrievalContractError) as caught:
        harness.retrieve("memory", {"operationId": "op", "intentText": "other"})
    assert caught.value.code == PLAN_DRIFT
    harness.snap = "snap-2"
    with pytest.raises(RetrievalContractError) as caught:
        harness.retrieve("memory", request)
    assert caught.value.code == PLAN_DRIFT
    harness.snap = SNAP
    with pytest.raises(RetrievalContractError) as caught:
        harness.retrieve("reader", request)
    assert caught.value.code == PLAN_DRIFT
    assert len(harness.discover_calls) == 1
    replay = harness.retrieve("memory", request)
    assert replay["duplicate"] is True
    assert replay["planHash"] == first["planHash"]
    assert [card["id"] for card in replay["memoryCandidateCards"]] == ["f1"]
    assert len(harness.discover_calls) == 1


def test_plan_hash_binds_role_query_and_server_refs(harness: Harness) -> None:
    intent = "切流后错误率升高时如何安全回退"
    env = harness.retrieve(
        "memory",
        {
            "operationId": "op-hash",
            "intentText": f"  {intent}  ",
            "dependencyPattern": {"wantedEffect": " 安全回退 ", "wantedArtifact": " 预案 "},
        },
    )
    plan = {
        "role": "memory",
        "intentText": intent,
        "dependencyPattern": {"wantedArtifact": "预案", "wantedEffect": "安全回退"},
        "scopeRef": SCOPE,
        "snapshotRef": SNAP,
        "budgetLeaseRef": LEASE,
    }
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    assert env["planHash"] == digest
    assert_public_envelope(env, operation_id="op-hash")
    assert intent not in json.dumps(env, ensure_ascii=False)
    assert harness.authority_calls == ["memory"]


def test_rejected_exact_does_not_poison_the_operation(harness: Harness) -> None:
    assert_rejected(
        harness,
        "memory",
        {"operationId": "op", "intentText": "guide", "exact": "5xx"},
        EXACT_CAPABILITY_ABSENT,
    )
    env = harness.retrieve("memory", {"operationId": "op", "intentText": "guide"})
    assert env["duplicate"] is False
    replay = harness.retrieve("memory", {"operationId": "op", "intentText": "guide"})
    assert replay["duplicate"] is True
    assert len(harness.discover_calls) == 1


def test_rejected_empty_plan_does_not_poison_the_operation(harness: Harness) -> None:
    assert_rejected(harness, "memory", {"operationId": "op"}, "empty_discovery_plan")
    env = harness.retrieve("memory", {"operationId": "op", "intentText": "guide"})
    assert env["duplicate"] is False
    assert len(harness.discover_calls) == 1


@pytest.mark.parametrize(
    "card",
    [
        "fork",
        {"kind": "fork"},
        {"id": "  ", "kind": "fork"},
        {"id": 1, "kind": "fork"},
        {"id": "f1", "kind": None},
        {"id": "f1"},
        {"useClass": "current_evidence"},
    ],
)
def test_bad_cards_do_not_enter_the_envelope(harness: Harness, card: Any) -> None:
    harness.cards = [card]
    code = "adapter_self_upgrade" if card == {"useClass": "current_evidence"} else "invalid_card"
    assert_rejected(
        harness,
        "memory",
        {"operationId": "op", "intentText": "guide"},
        code,
        dispatch=True,
    )


@pytest.mark.parametrize("raw", [None, "fork", b"fork", {"id": "f1", "kind": "fork"}])
def test_discover_must_return_a_card_sequence(harness: Harness, raw: Any) -> None:
    harness.raw = raw
    assert_rejected(
        harness,
        "memory",
        {"operationId": "op", "intentText": "guide"},
        "invalid_card",
        dispatch=True,
    )


@pytest.mark.parametrize(
    "card",
    [
        {"id": "f1", "kind": "fork", "useClass": "analogical_prior"},
        {"id": "f1", "kind": "fork", "useClass": "current_evidence"},
        {"id": "g1", "kind": "segment", "useClass": "memory_candidate"},
        {"id": "e1", "kind": "episode", "useClass": "current_evidence"},
    ],
)
def test_adapter_self_upgrade_is_rejected(harness: Harness, card: dict[str, str]) -> None:
    harness.cards = [card]
    assert_rejected(
        harness,
        "memory",
        {"operationId": "op", "intentText": "guide"},
        "adapter_self_upgrade",
        dispatch=True,
    )


def test_self_upgrade_does_not_poison_the_operation(harness: Harness) -> None:
    harness.cards = [{"id": "f1", "kind": "fork", "useClass": "current_evidence"}]
    assert_rejected(
        harness,
        "reader",
        {"operationId": "op", "intentText": "guide"},
        "adapter_self_upgrade",
        dispatch=True,
    )
    harness.cards = [{"id": "f1", "kind": "fork"}]
    env = harness.retrieve("reader", {"operationId": "op", "intentText": "guide"})
    assert env["duplicate"] is False
    assert [card["id"] for card in env["memoryCandidateCards"]] == ["f1"]
    assert len(harness.discover_calls) == 2


def test_authority_must_bind_server_refs(harness: Harness) -> None:
    harness.authority_override = lambda role: {"scopeRef": SCOPE}
    with pytest.raises(RetrievalContractError) as caught:
        harness.retrieve("memory", {"operationId": "op", "intentText": "guide"})
    assert caught.value.code == "authority_unbound"
    assert harness.authority_calls == ["memory"]
    assert harness.discover_calls == []


def test_blank_server_refs_are_unbound(harness: Harness) -> None:
    harness.authority_override = lambda role: BoundContext("  ", SNAP, LEASE)
    with pytest.raises(RetrievalContractError) as caught:
        harness.retrieve("memory", {"operationId": "op", "intentText": "guide"})
    assert caught.value.code == "authority_unbound"
    assert harness.discover_calls == []
    harness.authority_override = None
    env = harness.retrieve("memory", {"operationId": "op", "intentText": "guide"})
    assert env["duplicate"] is False


def test_replay_fails_closed_when_refs_no_longer_resolve(harness: Harness) -> None:
    first = harness.retrieve("memory", {"operationId": "op", "intentText": "guide"})
    harness.authority_override = lambda role: None
    with pytest.raises(RetrievalContractError) as caught:
        harness.retrieve("memory", {"operationId": "op", "intentText": "guide"})
    assert caught.value.code == "authority_unbound"
    assert len(harness.discover_calls) == 1
    harness.authority_override = None
    replay = harness.retrieve("memory", {"operationId": "op", "intentText": "guide"})
    assert replay["duplicate"] is True
    assert replay["planHash"] == first["planHash"]


def test_public_card_drops_acl_scores_and_body() -> None:
    mapped = public_card_from_retrieval(facade_card("fork", "fork:f1@r1"))
    assert mapped == {"id": "fork:f1@r1", "kind": "fork"}


def test_public_card_keeps_only_a_claimed_use_class() -> None:
    card = SimpleNamespace(
        payload_ref="f1",
        kind="fork",
        useClass="current_evidence",
        acl_labels=("ACL-SECRET-LABEL",),
        score_breakdown={"bm25": "SCORE-SENTINEL"},
        text="BODY-SECRET",
    )
    assert public_card_from_retrieval(card) == {
        "id": "f1",
        "kind": "fork",
        "useClass": "current_evidence",
    }


def test_retrieve_projected_is_keyword_only_and_reuses_the_callers_service() -> None:
    signature = inspect.signature(retrieve_projected)
    assert list(signature.parameters) == [
        "facade",
        "service",
        "role",
        "request",
        "authority",
        "principal_labels",
        "exact_searcher",
    ]
    for name in ("role", "request", "authority", "principal_labels", "exact_searcher"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["exact_searcher"].default is None
    assert "RoleRetrievalService(" not in inspect.getsource(retrieve_projected)
    with pytest.raises(TypeError):
        retrieve_projected(  # type: ignore[misc]
            object(),
            RoleRetrievalService(),
            "memory",
            {},
            lambda role: None,
            (),
        )


def test_retrieve_projected_rejects_a_fresh_store_substitute() -> None:
    facade = RecordingFacade([])

    def explode(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("facade called")

    facade.retrieve = explode  # type: ignore[method-assign]
    with pytest.raises(TypeError):
        retrieve_projected(
            facade,
            object(),  # type: ignore[arg-type]
            role="memory",
            request={"intentText": "guide"},
            authority=lambda role: BoundContext(SCOPE, SNAP, LEASE),
            principal_labels=(),
        )


def test_wiring_projects_cards_and_dedupes_on_the_same_service() -> None:
    fork = facade_card("fork", "fork:f1@r1")
    prior = facade_card("block_projection", "claim-card-31", channel="provisional", rank=1)
    episode = SimpleNamespace(
        payload_ref="episode-1",
        kind="episode",
        acl_labels=("ACL-SECRET-LABEL",),
        score_breakdown={"trigram": "SCORE-SENTINEL"},
        text="BODY-SECRET",
        useClass="memory_candidate",
    )
    revision = SimpleNamespace(
        payload_ref="skill-rev-17",
        kind="skill_revision",
        acl_labels=("ACL-SECRET-LABEL",),
        score="SCORE-SENTINEL",
    )
    facade = RecordingFacade([episode, fork, revision, prior])
    service = RoleRetrievalService()
    labels = ["PRINCIPAL-LABEL-SECRET", "team-a"]
    request = {
        "operationId": "op-wire",
        "intentText": "  INTENT-SENTINEL  ",
        "dependencyPattern": {"wantedEffect": "keep", "wantedArtifact": "ARTIFACT-MARKER"},
    }

    def authority(role: str) -> BoundContext:
        assert role == "memory"
        return BoundContext(SCOPE, SNAP, LEASE)

    first = retrieve_projected(
        facade,
        service,
        role="memory",
        request=request,
        authority=authority,
        principal_labels=labels,
    )
    assert_public_envelope(
        first,
        operation_id="op-wire",
        absent=(
            "INTENT-SENTINEL",
            "ARTIFACT-MARKER",
            "truncated_by_rerank_clamp",
            "selection_audit",
        ),
    )
    assert [card["id"] for card in first["memoryCandidateCards"]] == [
        "fork:f1@r1",
        "skill-rev-17",
    ]
    assert [card["id"] for card in first["analogicalPriorCards"]] == ["claim-card-31"]
    assert first["discoveryAudit"]["truncated"] is False
    assert len(facade.calls) == 1
    assert facade.calls[0][0] == "INTENT-SENTINEL"
    assert isinstance(facade.calls[0][0], str)
    assert tuple(facade.calls[0][1]) == tuple(labels)
    assert facade.calls[0][2] is None
    assert "ARTIFACT-MARKER" not in facade.calls[0][0]

    second = retrieve_projected(
        facade,
        service,
        role="memory",
        request={"operationId": "op-wire", "intentText": "INTENT-SENTINEL", "dependencyPattern": {
            "wantedArtifact": "ARTIFACT-MARKER",
            "wantedEffect": "keep",
        }},
        authority=authority,
        principal_labels=labels,
    )
    assert second["duplicate"] is True
    assert second["planHash"] == first["planHash"]
    assert len(facade.calls) == 1

    with pytest.raises(RetrievalContractError) as caught:
        retrieve_projected(
            facade,
            service,
            role="memory",
            request={"operationId": "op-wire", "intentText": "other"},
            authority=authority,
            principal_labels=labels,
        )
    assert caught.value.code == PLAN_DRIFT
    assert len(facade.calls) == 1

    other = RoleRetrievalService()
    third = retrieve_projected(
        facade,
        other,
        role="memory",
        request=request,
        authority=authority,
        principal_labels=("team-b",),
    )
    assert third["duplicate"] is False
    assert third["planHash"] == first["planHash"]
    assert len(facade.calls) == 2
    assert tuple(facade.calls[1][1]) == ("team-b",)


def test_wiring_pattern_only_queries_with_an_empty_string() -> None:
    facade = RecordingFacade([])
    env = retrieve_projected(
        facade,
        RoleRetrievalService(),
        role="reader",
        request={"operationId": "op-p", "dependencyPattern": {"wantedEffect": "roll back"}},
        authority=lambda role: BoundContext(SCOPE, SNAP, LEASE),
        principal_labels=("team-a",),
    )
    assert_public_envelope(env, operation_id="op-p")
    assert facade.calls == [("", ("team-a",), None)]


def test_wiring_refuses_exact_before_facade_retrieve() -> None:
    facade = RecordingFacade([facade_card("fork", "fork:f1@r1")])
    service = RoleRetrievalService()

    def authority(_role: str) -> BoundContext:
        return BoundContext(SCOPE, SNAP, LEASE)

    with pytest.raises(RetrievalContractError) as caught:
        retrieve_projected(
            facade,
            service,
            role="memory",
            request={"intentText": "guide", "current_log_exact": "5xx", "operationId": "op"},
            authority=authority,
            principal_labels=("team-a",),
        )
    assert caught.value.code == EXACT_CAPABILITY_ABSENT
    with pytest.raises(RetrievalContractError) as caught:
        retrieve_projected(
            facade,
            service,
            role="reader",
            request={"intentText": "guide", "exact": "schema mismatch", "operationId": "op"},
            authority=authority,
            principal_labels=("team-a",),
        )
    assert caught.value.code == EXACT_LANE_NOT_IN_DISCOVERY_ENTRY
    assert facade.calls == []


def test_wiring_rejects_adapter_current_evidence_without_leaking_private_fields() -> None:
    facade = RecordingFacade(
        [
            SimpleNamespace(
                payload_ref="f1",
                kind="fork",
                useClass="current_evidence",
                acl_labels=("ACL-SECRET-LABEL",),
                score_breakdown={"bm25": "SCORE-SENTINEL"},
            )
        ]
    )
    service = RoleRetrievalService()
    with pytest.raises(RetrievalContractError) as caught:
        retrieve_projected(
            facade,
            service,
            role="memory",
            request={"operationId": "op", "intentText": "guide"},
            authority=lambda role: BoundContext(SCOPE, SNAP, LEASE),
            principal_labels=("PRINCIPAL-LABEL-SECRET",),
        )
    assert caught.value.code == "adapter_self_upgrade"
    assert str(caught.value) == "adapter_self_upgrade"
    assert len(facade.calls) == 1
    facade.cards = [facade_card("fork", "fork:f1@r1")]
    env = retrieve_projected(
        facade,
        service,
        role="memory",
        request={"operationId": "op", "intentText": "guide"},
        authority=lambda role: BoundContext(SCOPE, SNAP, LEASE),
        principal_labels=("PRINCIPAL-LABEL-SECRET",),
    )
    assert env["duplicate"] is False
    assert env["memoryCandidateCards"] == [
        {"id": "fork:f1@r1", "kind": "fork", "useClass": "memory_candidate"}
    ]
    assert len(facade.calls) == 2
