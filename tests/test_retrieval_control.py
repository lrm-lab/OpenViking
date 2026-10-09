# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Control tree: sibling isolation, parent fan-out, and hidden terminated hits."""

from __future__ import annotations

import json

import pytest

from openviking.session.memory_role_retrieval import EXACT_CAPABILITY_ABSENT
from openviking.session.retrieval_control import (
    DISCOVERY_LANE,
    EXACT_LANE,
    ControlError,
    ControlPolicy,
    RetrievalControl,
)

BOTH = frozenset({EXACT_LANE, DISCOVERY_LANE})
POLICY = ControlPolicy(
    exact_window_ms=3_000,
    discovery_window_ms=6_000,
    exact_budget=5,
    discovery_budget=5,
)


def _tree(**overrides: object) -> RetrievalControl:
    args = {
        "role": "reader",
        "parent_lease_ref": "parent-lease",
        "parent_deadline_ms": 8_000,
        "now_ms": 0,
        "dispatched": BOTH,
        "policy": POLICY,
    }
    args.update(overrides)
    return RetrievalControl(**args)  # type: ignore[arg-type]


def test_child_budgets_do_not_transfer_and_quotas_stay_put() -> None:
    tree = _tree()
    tree.stage(DISCOVERY_LANE, "card-1")
    tree.complete(DISCOVERY_LANE)
    tree.stage(EXACT_LANE, "SECRET-HIT")
    tree.charge(EXACT_LANE, 5, 100)
    view = tree.public_view()
    assert view["completeness"] == "partial"
    assert view["lanes"][EXACT_LANE]["reasonCode"] == "budget_exhausted"
    assert view["lanes"][EXACT_LANE]["returnedCount"] == 0
    assert view["lanes"][DISCOVERY_LANE]["executionState"] == "completed"
    assert view["results"][DISCOVERY_LANE] == ["card-1"]
    assert "SECRET-HIT" not in json.dumps(view)
    assert tree.native_audit(EXACT_LANE)["withheldHitIds"] == ["SECRET-HIT"]
    assert tree.lanes[DISCOVERY_LANE].budget == 5


def test_child_deadline_stops_one_lane_and_remaining_time_does_not_reset() -> None:
    tree = _tree()
    assert tree.remaining_ms(EXACT_LANE, 0) == 3_000
    assert tree.remaining_ms(EXACT_LANE, 2_000) == 1_000
    assert tree.remaining_ms(EXACT_LANE, 1_000) == 1_000
    tree.checkpoint(EXACT_LANE, 3_000)
    tree.checkpoint(DISCOVERY_LANE, 3_000)
    assert tree.is_running(EXACT_LANE) is False
    assert tree.is_running(DISCOVERY_LANE) is True
    tree.stage(DISCOVERY_LANE, "card-1")
    tree.complete(DISCOVERY_LANE)
    view = tree.public_view()
    assert view["lanes"][EXACT_LANE]["reasonCode"] == "deadline_exceeded"
    assert view["results"][DISCOVERY_LANE] == ["card-1"]
    assert "deadline_ms" not in json.dumps(view)


def test_parent_cancel_fans_out_and_keeps_a_finished_sibling() -> None:
    tree = _tree()
    tree.stage(DISCOVERY_LANE, "card-1")
    tree.complete(DISCOVERY_LANE)
    tree.stage(EXACT_LANE, "SECRET-HIT")
    tree.parent_signal("client_abort", 1_000)
    view = tree.public_view()
    assert view["lanes"][EXACT_LANE]["reasonCode"] == "cancelled"
    assert view["lanes"][EXACT_LANE]["returnedCount"] == 0
    assert view["results"][DISCOVERY_LANE] == ["card-1"]
    assert tree.expansion["cancelled"] is False
    assert tree.expansion["budget_lease_ref"] != tree.native_audit(DISCOVERY_LANE)["leaseRef"]


def test_first_terminal_cause_wins() -> None:
    tree = _tree()
    tree.child_signal(EXACT_LANE, "dependency_unavailable", 10)
    tree.parent_signal("shutdown", 20)
    assert tree.public_view()["lanes"][EXACT_LANE]["reasonCode"] == "dependency_unavailable"
    assert tree.native_audit(EXACT_LANE)["secondarySignals"] == ["cancelled"]
    assert tree.public_view()["lanes"][DISCOVERY_LANE]["reasonCode"] == "cancelled"


def test_semantic_only_discovery_is_complete_without_exact() -> None:
    tree = _tree(dispatched=frozenset({DISCOVERY_LANE}))
    tree.complete(DISCOVERY_LANE)
    view = tree.public_view()
    assert view["completeness"] == "complete"
    assert EXACT_LANE not in view["lanes"]


def test_all_terminated_lanes_are_unavailable() -> None:
    tree = _tree()
    tree.child_signal(EXACT_LANE, "dependency_unavailable", 1)
    tree.child_signal(DISCOVERY_LANE, "dependency_unavailable", 1)
    assert tree.public_view()["completeness"] == "unavailable"
    assert tree.public_view()["results"] == {}


def test_memory_cannot_receive_an_exact_control_surface() -> None:
    with pytest.raises(ControlError) as caught:
        _tree(role="memory", dispatched=BOTH)
    assert caught.value.code == EXACT_CAPABILITY_ABSENT
    memory = _tree(role="memory", dispatched=frozenset({DISCOVERY_LANE}))
    memory.complete(DISCOVERY_LANE)
    body = json.dumps(memory.public_view())
    assert set(memory.public_view()["lanes"]) == {DISCOVERY_LANE}
    assert EXACT_LANE not in body
    assert "timestamp" not in body
    assert "elapsed" not in body
