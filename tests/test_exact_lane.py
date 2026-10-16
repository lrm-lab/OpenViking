# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Current-log exact lane: clause trigger, ACL-before-limit, and no backfill."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openviking.session.exact_lane import execute_exact_lane
from openviking.session.memory_role_retrieval import (
    EXACT_CAPABILITY_ABSENT,
    PLAN_DRIFT,
    BoundContext,
    RetrievalContractError,
    RoleRetrievalService,
)
from openviking.session.retrieval_wiring import retrieve_projected

SCOPE = "scope-1"
SNAP = "snap-1"
LEASE = "lease-1"
SOURCE = Path(__file__).parents[1] / "openviking" / "session" / "exact_lane.py"


def _clause(**overrides: Any) -> dict[str, Any]:
    clause = {"expression": "ab", "syntax": "literal"}
    clause.update(overrides)
    return clause


def _record(**overrides: Any) -> dict[str, Any]:
    record = {
        "sourceSessionRef": "session-a",
        "aoId": "ao-1",
        "sequence": 1,
        "matchedField": "observation",
        "text": "ab ab",
        "visible": True,
        "inScope": True,
        "withdrawn": False,
    }
    record.update(overrides)
    return record


def _request(**overrides: Any) -> dict[str, Any]:
    request = {
        "operationId": "op-exact",
        "intentText": "guide",
        "query": {"exact": _clause()},
    }
    request.update(overrides)
    return request


class _Calls:
    def __init__(
        self, cards: list[dict[str, Any]] | None = None, records: list[dict[str, Any]] | None = None
    ) -> None:
        self.cards = [] if cards is None else cards
        self.records = [] if records is None else records
        self.discovered: list[tuple[Any, Any]] = []
        self.searched: list[Any] = []

    def authority(self, _role: str) -> BoundContext:
        return BoundContext(SCOPE, SNAP, LEASE)

    def discover(self, intent: str | None, pattern: Any) -> list[dict[str, Any]]:
        self.discovered.append((intent, pattern))
        return list(self.cards)

    def search(self, clause: Any, bound: BoundContext) -> list[dict[str, Any]]:
        self.searched.append((clause, bound))
        return list(self.records)


def _retrieve(calls: _Calls, request: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return RoleRetrievalService().retrieve(
        role=kwargs.pop("role", "reader"),
        request=request,
        authority=calls.authority,
        discover=calls.discover,
        exact_searcher=kwargs.pop("exact_searcher", calls.search),
        **kwargs,
    )


def test_exact_lane_module_does_not_import_host_re() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert "re" not in imported


def test_literal_ref_coalesces_spans_and_hashes_the_searched_text() -> None:
    text = "ab ab"
    lane = execute_exact_lane(
        _clause(),
        [_record(text=text)],
        limit=20,
        snapshot_ref=SNAP,
    )
    assert lane["exactLaneAudit"]["executionState"] == "completed"
    assert lane["exactLaneAudit"]["resultState"] == "hits"
    assert lane["exactLaneAudit"]["returnedCount"] == 1
    assert lane["exactLaneAudit"]["truncated"] is False
    ref = lane["currentEvidenceRefs"][0]
    assert ref["sourceSessionRef"] == "session-a"
    assert ref["aoId"] == "ao-1"
    assert ref["matchedField"] == "observation"
    assert ref["snapshotRef"] == SNAP
    assert ref["contentHash"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert ref["totalOccurrences"] == 2
    assert ref["spansTruncated"] is False
    assert [(span["start"], span["end"], span["fragment"]) for span in ref["spans"]] == [
        (0, 2, "ab"),
        (3, 5, "ab"),
    ]
    assert "_sessionOrder" not in ref
    assert "ab ab" not in json.dumps(ref["spans"])


def test_ninth_occurrence_sets_spans_truncated_without_a_ninth_span() -> None:
    text = "ab" * 9
    lane = execute_exact_lane(_clause(), [_record(text=text)], limit=20, snapshot_ref=SNAP)
    ref = lane["currentEvidenceRefs"][0]
    assert ref["totalOccurrences"] == 9
    assert len(ref["spans"]) == 8
    assert ref["spansTruncated"] is True
    assert lane["exactLaneAudit"]["truncated"] is False


def test_literal_does_not_treat_metacharacters_as_regex() -> None:
    lane = execute_exact_lane(
        _clause(expression="a.b"),
        [_record(text="axb a.b")],
        limit=20,
        snapshot_ref=SNAP,
    )
    ref = lane["currentEvidenceRefs"][0]
    assert ref["totalOccurrences"] == 1
    assert ref["spans"] == [{"start": 4, "end": 7, "fragment": "a.b"}]


def test_case_fold_is_opt_in() -> None:
    sensitive = execute_exact_lane(
        _clause(expression="Ab"),
        [_record(text="xxabxx")],
        limit=20,
        snapshot_ref=SNAP,
    )
    folded = execute_exact_lane(
        _clause(expression="Ab", caseSensitive=False),
        [_record(text="xxabxx")],
        limit=20,
        snapshot_ref=SNAP,
    )
    assert sensitive["currentEvidenceRefs"] == []
    assert folded["currentEvidenceRefs"][0]["spans"][0]["fragment"] == "ab"


def test_invisible_hits_do_not_consume_the_public_limit() -> None:
    visible = _record(sourceSessionRef="visible", aoId="ao-vis", text="ab")
    hidden = _record(sourceSessionRef="hidden", aoId="ao-hid", text="ab", visible=False)
    withdrawn = _record(
        sourceSessionRef="gone", aoId="ao-gone", text="ab", withdrawn=True, sequence=0
    )
    outside = _record(sourceSessionRef="out", aoId="ao-out", text="ab", inScope=False, sequence=0)
    lane = execute_exact_lane(
        _clause(),
        [hidden, withdrawn, outside, visible],
        limit=1,
        snapshot_ref=SNAP,
    )
    assert lane["exactLaneAudit"]["returnedCount"] == 1
    assert lane["exactLaneAudit"]["truncated"] is False
    body = json.dumps(lane)
    assert "hidden" not in body
    assert "ao-hid" not in body
    assert "filtered" not in body


def test_limit_applies_after_authorization_and_canonical_order() -> None:
    records = [
        _record(
            sourceSessionRef="session-b", aoId="b", sequence=1, matchedField="action", text="ab"
        ),
        _record(
            sourceSessionRef="session-a", aoId="a2", sequence=2, matchedField="action", text="ab"
        ),
        _record(
            sourceSessionRef="session-a", aoId="am", sequence=1, matchedField="message", text="ab"
        ),
        _record(
            sourceSessionRef="session-a", aoId="a1", sequence=1, matchedField="action", text="ab"
        ),
        _record(sourceSessionRef="hidden", aoId="hid", text="ab", visible=False),
    ]
    lane = execute_exact_lane(
        _clause(fields=["action", "message"]), records, limit=2, snapshot_ref=SNAP
    )
    assert [ref["aoId"] for ref in lane["currentEvidenceRefs"]] == ["b", "a1"]
    assert lane["exactLaneAudit"]["returnedCount"] == 2
    assert lane["exactLaneAudit"]["truncated"] is True
    assert "hid" not in json.dumps(lane)


def test_regex_without_re2_terminates_without_searching_or_partial_hits() -> None:
    calls = _Calls(cards=[{"id": "f1", "kind": "fork"}], records=[_record()])
    env = _retrieve(
        calls,
        _request(query={"exact": _clause(expression="ab+", syntax="regex")}),
    )
    assert calls.searched == []
    assert calls.discovered == [("guide", None)]
    assert env["currentEvidenceRefs"] == []
    assert env["exactLaneAudit"]["executionState"] == "terminated"
    assert env["exactLaneAudit"]["resultState"] == "none"
    assert env["exactLaneAudit"]["reasonCode"] == "dependency_unavailable"
    assert env["exactLaneAudit"]["returnedCount"] == 0
    assert env["memoryCandidateCards"] == [
        {"id": "f1", "kind": "fork", "useClass": "memory_candidate"}
    ]


def test_invalid_regex_is_rejected_before_dispatch() -> None:
    class _Engine:
        IGNORECASE = 1

        def compile(self, _expression: str, _flags: int = 0) -> Any:
            raise ValueError("unsupported")

    calls = _Calls(records=[_record()])
    with pytest.raises(RetrievalContractError) as caught:
        _retrieve(
            calls,
            _request(query={"exact": _clause(syntax="regex")}),
            regex_engine=_Engine(),
        )
    assert caught.value.code == "invalid_exact_clause"
    assert calls.discovered == []
    assert calls.searched == []


def test_injected_linear_engine_can_match_without_host_re() -> None:
    class _Match:
        def __init__(self, start: int, end: int) -> None:
            self._start = start
            self._end = end

        def start(self) -> int:
            return self._start

        def end(self) -> int:
            return self._end

    class _Pattern:
        def finditer(self, text: str) -> Any:
            found = text.find("err")
            if found >= 0:
                yield _Match(found, found + 3)

    class _Engine:
        IGNORECASE = 1

        def compile(self, expression: str, flags: int = 0) -> _Pattern:
            assert expression == "err"
            assert flags == 0
            return _Pattern()

    calls = _Calls(records=[_record(text="prefix err suffix")])
    env = _retrieve(
        calls,
        _request(query={"exact": _clause(expression="err", syntax="regex")}),
        regex_engine=_Engine(),
    )
    assert env["exactLaneAudit"]["executionState"] == "completed"
    assert env["currentEvidenceRefs"][0]["spans"][0]["fragment"] == "err"
    assert calls.searched


def test_memory_query_exact_is_rejected_before_either_lane() -> None:
    calls = _Calls(records=[_record()])
    with pytest.raises(RetrievalContractError) as caught:
        _retrieve(calls, _request(), role="memory")
    assert caught.value.code == EXACT_CAPABILITY_ABSENT
    assert calls.discovered == []
    assert calls.searched == []


def test_clause_presence_dispatches_exact_without_feeding_discovery() -> None:
    calls = _Calls(
        cards=[{"id": "f1", "kind": "fork"}],
        records=[_record(text="ab")],
    )
    env = _retrieve(calls, _request(query={"exact": _clause(expression="SCHEMA")}))
    assert calls.discovered == [("guide", None)]
    assert calls.searched[0][0]["expression"] == "SCHEMA"
    assert calls.searched[0][1].snapshot_ref == SNAP
    assert env["memoryCandidateCards"][0]["id"] == "f1"
    assert env["currentEvidenceRefs"] == []
    assert env["exactLaneAudit"]["resultState"] == "empty"
    assert "SCHEMA" not in json.dumps(env)


def test_empty_exact_does_not_backfill_discovery_and_discovery_does_not_backfill_exact() -> None:
    calls = _Calls(
        cards=[{"id": f"f{i}", "kind": "fork"} for i in range(15)],
        records=[],
    )
    env = _retrieve(calls, _request())
    assert len(env["memoryCandidateCards"]) == 12
    assert env["analogicalPriorCards"] == []
    assert env["discoveryAudit"]["truncated"] is True
    assert env["discoveryAudit"]["returnedCount"] == 12
    assert env["currentEvidenceRefs"] == []
    assert env["exactLaneAudit"]["returnedCount"] == 0
    assert env["exactLaneAudit"]["truncated"] is False

    exact_only = _Calls(records=[_record(text="ab") for _ in range(3)])
    exact_env = _retrieve(
        exact_only,
        {"operationId": "op-only", "query": {"exact": _clause()}},
    )
    assert exact_only.discovered == []
    assert exact_env["discoveryAudit"] == {
        "executionState": "not_dispatched",
        "resultState": "none",
        "reasonCode": "not_selected",
        "returnedCount": 0,
        "truncated": False,
    }
    assert len(exact_env["currentEvidenceRefs"]) == 1
    assert exact_env["memoryCandidateCards"] == []


def test_effective_limit_enters_the_plan_hash_and_replays() -> None:
    calls = _Calls(records=[_record(text="ab")])
    service = RoleRetrievalService()
    request = _request(
        operationId="op-hash", intentText=None, query={"exact": _clause(expression="ab")}
    )
    del request["intentText"]
    first = service.retrieve(
        role="reader",
        request=request,
        authority=calls.authority,
        discover=calls.discover,
        exact_searcher=calls.search,
    )
    plan = {
        "role": "reader",
        "intentText": None,
        "dependencyPattern": None,
        "scopeRef": SCOPE,
        "snapshotRef": SNAP,
        "budgetLeaseRef": LEASE,
        "exact": {
            "expression": "ab",
            "syntax": "literal",
            "fields": ["action", "observation", "message"],
            "caseSensitive": True,
        },
        "currentLogExactMaxHits": 20,
    }
    digest = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert first["planHash"] == digest
    explicit = service.retrieve(
        role="reader",
        request={**request, "currentLogExactMaxHits": 20},
        authority=calls.authority,
        discover=calls.discover,
        exact_searcher=calls.search,
    )
    assert explicit["duplicate"] is True
    assert len(calls.searched) == 1
    with pytest.raises(RetrievalContractError) as caught:
        service.retrieve(
            role="reader",
            request={**request, "currentLogExactMaxHits": 5},
            authority=calls.authority,
            discover=calls.discover,
            exact_searcher=calls.search,
        )
    assert caught.value.code == PLAN_DRIFT


def test_limit_without_a_clause_or_outside_1_to_20_is_invalid() -> None:
    calls = _Calls()
    with pytest.raises(RetrievalContractError) as caught:
        _retrieve(calls, {"operationId": "op", "intentText": "guide", "currentLogExactMaxHits": 5})
    assert caught.value.code == "exact_limit_without_clause"
    for limit in (0, 21):
        with pytest.raises(RetrievalContractError) as caught:
            _retrieve(
                calls,
                _request(currentLogExactMaxHits=limit),
            )
        assert caught.value.code == "invalid_exact_limit"
    assert calls.discovered == []
    assert calls.searched == []


def test_searcher_failure_keeps_discovery_and_publishes_no_exact_hits() -> None:
    calls = _Calls(cards=[{"id": "f1", "kind": "fork"}], records=[_record()])

    def explode(_clause: Any, _bound: BoundContext) -> list[dict[str, Any]]:
        raise RuntimeError("resolver down")

    env = _retrieve(calls, _request(), exact_searcher=explode)
    assert env["memoryCandidateCards"][0]["id"] == "f1"
    assert env["currentEvidenceRefs"] == []
    assert env["exactLaneAudit"]["executionState"] == "terminated"
    assert env["exactLaneAudit"]["reasonCode"] == "internal_error"
    assert env["exactLaneAudit"]["resultState"] == "none"
    assert "resolver down" not in json.dumps(env)


def test_field_order_and_case_default_are_canonical() -> None:
    calls = _Calls(records=[])
    service = RoleRetrievalService()
    base = {
        "operationId": "op-canon",
        "query": {
            "exact": {"expression": "ab", "syntax": "literal", "fields": ["message", "action"]}
        },
    }
    first = service.retrieve(
        role="critic",
        request=base,
        authority=calls.authority,
        discover=calls.discover,
        exact_searcher=calls.search,
    )
    second = service.retrieve(
        role="critic",
        request={
            "operationId": "op-canon",
            "query": {
                "exact": {
                    "expression": "ab",
                    "syntax": "literal",
                    "fields": ["action", "message"],
                    "caseSensitive": True,
                }
            },
        },
        authority=calls.authority,
        discover=calls.discover,
        exact_searcher=calls.search,
    )
    assert second["duplicate"] is True
    assert second["planHash"] == first["planHash"]
    assert len(calls.searched) == 1


def test_wiring_keeps_exact_expression_out_of_facade_retrieve() -> None:
    class _Facade:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def retrieve(self, query: str, _labels: Any, _profile: Any = None) -> Any:
            self.calls.append(query)
            return SimpleNamespace(cards=())

    facade = _Facade()
    searched: list[str] = []

    def search(clause: Any, _bound: BoundContext) -> list[dict[str, Any]]:
        searched.append(clause["expression"])
        return [_record(text="SCHEMA")]

    env = retrieve_projected(
        facade,
        RoleRetrievalService(),
        role="reader",
        request=_request(query={"exact": _clause(expression="SCHEMA")}),
        authority=lambda _role: BoundContext(SCOPE, SNAP, LEASE),
        principal_labels=("team-a",),
        exact_searcher=search,
    )
    assert facade.calls == ["guide"]
    assert searched == ["SCHEMA"]
    assert env["currentEvidenceRefs"][0]["spans"][0]["fragment"] == "SCHEMA"

    refused = _Facade()
    with pytest.raises(RetrievalContractError) as caught:
        retrieve_projected(
            refused,
            RoleRetrievalService(),
            role="memory",
            request=_request(),
            authority=lambda _role: BoundContext(SCOPE, SNAP, LEASE),
            principal_labels=("team-a",),
            exact_searcher=search,
        )
    assert caught.value.code == EXACT_CAPABILITY_ABSENT
    assert refused.calls == []
