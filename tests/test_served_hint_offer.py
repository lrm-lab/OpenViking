# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Hint and Offer cite only served memory candidates."""

from __future__ import annotations

import pytest

from openviking.session.served_hint_offer import HintError, ServedDeliveryLedger


def _envelope(**overrides: object) -> dict[str, object]:
    envelope: dict[str, object] = {
        "envelope_ref": "env-1",
        "operation_id": "op-1",
        "plan_hash": "p" * 64,
        "lanes": {
            "memory_discovery": {"executionState": "completed", "reasonCode": None},
            "current_log_exact": {"executionState": "completed", "reasonCode": None},
        },
        "memory_candidates": [
            {"id": "skill-1", "kind": "skill_revision", "use_class": "memory_candidate"}
        ],
        "analogical_priors": [
            {"id": "prior-1", "kind": "block_projection", "use_class": "analogical_prior"}
        ],
    }
    envelope.update(overrides)
    return envelope


def test_unserved_prior_retired_and_terminated_discovery_cannot_drive_a_hint() -> None:
    ledger = ServedDeliveryLedger()
    with pytest.raises(HintError) as caught:
        ledger.publish(_envelope(), ["missing"], kind="hint", target="agent", lineage="task")
    assert caught.value.code == "source_not_served"
    with pytest.raises(HintError) as caught:
        ledger.publish(_envelope(), ["prior-1"], kind="hint", target="agent", lineage="task")
    assert caught.value.code == "source_not_served"
    retired = _envelope(
        memory_candidates=[{"id": "ep-1", "kind": "episode", "use_class": "memory_candidate"}]
    )
    with pytest.raises(HintError) as caught:
        ledger.publish(retired, ["ep-1"], kind="offer", target="agent", lineage="task")
    assert caught.value.code == "retired_from_memory_discovery"
    terminated = _envelope(
        lanes={
            "memory_discovery": {"executionState": "terminated", "reasonCode": "budget_exhausted"},
            "current_log_exact": {"executionState": "completed", "reasonCode": None},
        }
    )
    with pytest.raises(HintError) as caught:
        ledger.publish(terminated, ["skill-1"], kind="hint", target="agent", lineage="task")
    assert caught.value.code == "discovery_not_completed"
    assert ledger.deliveries == []


def test_partial_hint_does_not_swallow_a_later_complete_hint() -> None:
    ledger = ServedDeliveryLedger()
    partial_lanes = {
        "memory_discovery": {"executionState": "completed", "reasonCode": None},
        "current_log_exact": {"executionState": "terminated", "reasonCode": "deadline_exceeded"},
    }
    partial = ledger.publish(
        _envelope(lanes=partial_lanes, operation_id="op-partial"),
        ["skill-1"],
        kind="hint",
        target="agent",
        lineage="task",
    )
    complete = ledger.publish(
        _envelope(operation_id="op-complete"),
        ["skill-1"],
        kind="hint",
        target="agent",
        lineage="task",
    )
    assert partial["duplicate"] is False
    assert complete["duplicate"] is False
    assert partial["delivery"]["id"] != complete["delivery"]["id"]
    assert partial["delivery"]["provenance"]["retrievalCompleteness"] == "partial"
    assert complete["delivery"]["provenance"]["retrievalCompleteness"] == "complete"
    assert partial["delivery"]["provenance"]["incompleteLanes"] == ["current_log_exact"]
    again = ledger.publish(
        _envelope(lanes=partial_lanes, operation_id="op-later-partial"),
        ["skill-1"],
        kind="hint",
        target="agent",
        lineage="task",
    )
    assert again["duplicate"] is True
    assert again["delivery"] is partial["delivery"]
    assert complete["delivery"]["provenance"]["operationId"] == "op-complete"
    fresh = ServedDeliveryLedger()
    fresh.publish(
        _envelope(),
        ["skill-1"],
        kind="hint",
        target="agent",
        lineage="task",
    )
    with pytest.raises(HintError) as caught:
        fresh.publish(
            _envelope(lanes=partial_lanes),
            ["skill-1"],
            kind="hint",
            target="agent",
            lineage="task",
        )
    assert caught.value.code == "complete_hint_stands"


def test_same_completeness_refreshes_without_rewriting_or_counting_utility() -> None:
    ledger = ServedDeliveryLedger()
    first = ledger.publish(
        _envelope(),
        ["skill-1"],
        kind="offer",
        target="agent",
        lineage="task",
    )
    second = ledger.publish(
        _envelope(operation_id="op-replay"),
        ["skill-1"],
        kind="offer",
        target="agent",
        lineage="task",
    )
    assert second["duplicate"] is True
    assert second["refreshed"] is True
    assert second["utility_counted"] is False
    assert second["delivery"] is first["delivery"]
    assert first["delivery"]["provenance"]["operationId"] == "op-1"
    assert len(ledger.deliveries) == 1


def test_hidden_or_withdrawn_card_fails_recheck_and_exact_only_is_not_enough() -> None:
    ledger = ServedDeliveryLedger()

    def withdrawn(card_id: str) -> str:
        return "withdrawn" if card_id == "skill-1" else "visible"

    with pytest.raises(HintError) as caught:
        ledger.publish(
            _envelope(),
            ["skill-1"],
            kind="hint",
            target="agent",
            lineage="task",
            status_of=withdrawn,
        )
    assert caught.value.code == "source_not_currently_visible"
    exact_only = _envelope(
        lanes={"current_log_exact": {"executionState": "completed", "reasonCode": None}},
        memory_candidates=[],
    )
    with pytest.raises(HintError) as caught:
        ledger.publish(exact_only, ["skill-1"], kind="hint", target="agent", lineage="task")
    assert caught.value.code == "discovery_not_completed"


def test_composer_cannot_declare_completeness() -> None:
    ledger = ServedDeliveryLedger()
    envelope = _envelope()
    envelope["retrievalCompleteness"] = "complete"
    with pytest.raises(HintError) as caught:
        ledger.publish(envelope, ["skill-1"], kind="hint", target="agent", lineage="task")
    assert caught.value.code == "completeness_not_self_reported"
