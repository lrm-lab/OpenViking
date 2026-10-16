# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Paper-exact Phi(G): root, sink, cycle, and reachability."""

from __future__ import annotations

from openviking.session.paper_graph_phi import (
    CONNECTIVITY_POLICY_VERSION,
    VALIDATOR_PROFILE,
    evaluate_phi,
)


def _ids(graph: dict) -> set[str]:
    report = evaluate_phi(graph)
    return {item.violation_id for item in report.violations}


def test_missing_task_root_is_structural() -> None:
    report = evaluate_phi({"blocks": [{"block_id": "end-1", "kind": "conclusion"}], "edges": []})
    found = [item for item in report.violations if item.violation_id == "phi:missing_task_root"]
    assert len(found) == 1
    assert found[0].object_id == "graph"
    assert found[0].kind == "structural"
    assert found[0].severity == "high"
    assert found[0].scope == "graph"
    assert found[0].evidence


def test_missing_conclusion_sink_is_structural() -> None:
    report = evaluate_phi({"blocks": [{"block_id": "task-1", "kind": "task"}], "edges": []})
    found = [
        item for item in report.violations if item.violation_id == "phi:missing_conclusion_sink"
    ]
    assert len(found) == 1
    assert found[0].kind == "structural"
    assert found[0].severity == "high"
    assert found[0].scope == "graph"


def test_cycle_is_reported() -> None:
    graph = {
        "blocks": [
            {"block_id": "a", "kind": "agent"},
            {"block_id": "b", "kind": "agent"},
        ],
        "edges": [{"from": "a", "to": "b"}, {"from": "b", "to": "a"}],
    }
    cycle = [
        item
        for item in evaluate_phi(graph).violations
        if item.violation_id.startswith("phi:cycle:")
    ]
    assert len(cycle) == 1
    assert cycle[0].kind == "structural"
    assert cycle[0].severity == "high"
    assert set(cycle[0].involved_ids) == {"a", "b"}


def test_unreachable_node_is_reported() -> None:
    graph = {
        "blocks": [
            {"block_id": "task-1", "kind": "task"},
            {"block_id": "agent-1", "kind": "agent"},
            {"block_id": "end-1", "kind": "conclusion"},
        ],
        "edges": [{"from": "task-1", "to": "end-1"}],
    }
    report = evaluate_phi(graph)
    found = [item for item in report.violations if item.violation_id == "phi:unreachable:agent-1"]
    assert len(found) == 1
    assert found[0].scope == "graph"
    assert found[0].object_id == "agent-1"
    assert found[0].severity == "high"


def test_legal_chain_passes_and_records_connectivity_policy() -> None:
    graph = {
        "blocks": [
            {"block_id": "task-1", "kind": "task"},
            {"block_id": "agent-1", "kind": "agent"},
            {"block_id": "end-1", "kind": "conclusion"},
        ],
        "edges": [
            {"from": "task-1", "to": "agent-1"},
            {"from": "agent-1", "to": "end-1"},
        ],
    }
    report = evaluate_phi(graph)
    assert report.violations == ()
    assert report.profile == VALIDATOR_PROFILE
    assert report.connectivity_policy_version == CONNECTIVITY_POLICY_VERSION
    assert report.connectivity_policy_version == "paper-exact-connectivity-v1"
    assert "does not define" in report.connectivity_interpretation
    assert _ids(graph) == set()
