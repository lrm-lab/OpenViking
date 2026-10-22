# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Cycle 006/007 grill decisions: per-agent segments, delivery AOs, DAG overview, judge report."""

from __future__ import annotations

from pathlib import Path

import pytest

from openviking.message import Message, TextPart, ToolPart
from openviking.session.aig_tool_surface import AigToolSurface, AoSnapshot, ToolSurfaceError
from openviking.session.ao_ledger_recorder import AOLedgerRecorder
from openviking.session.outcome_contract import OutcomeContractError, external_evidence_from_session
from openviking.session.outcome_signal import OutcomeSignal, outcome_overview


def _tool_part(tool_id: str) -> ToolPart:
    return ToolPart(
        tool_id=tool_id,
        tool_name="read_file",
        tool_input={"path": f"{tool_id}.md"},
        tool_output="ok",
        tool_status="completed",
    )


def _turn(peer: str, msg_id: str, tools: list[str], source: str | None) -> Message:
    return Message(
        id=msg_id,
        role="assistant",
        peer_id=peer,
        parts=[*[_tool_part(t) for t in tools], TextPart(text=f"{peer} says {msg_id}")],
        source_message_ids=[source] if source else None,
    )


def _recorded(tmp_path: Path) -> AOLedgerRecorder:
    recorder = AOLedgerRecorder(tmp_path / "s", "s")
    recorder.record_channel_delivery(participant_id="bob", event_id="e1", text="hi", cutting=False)
    recorder.record_messages([_turn("alice", "m1", ["t1", "t2"], "e1")])
    recorder.record_channel_delivery(participant_id="bob", event_id="e1", text="hi")
    recorder.record_messages([_turn("bob", "m2", ["t3"], "e2")])
    return recorder


def test_tool_and_send_aos_share_one_participant_segment(tmp_path: Path) -> None:
    rows = list(_recorded(tmp_path).ledger.records())
    alice = [r for r in rows if r.action.get("participant_id") == "alice"]
    assert len(alice) == 3
    assert {r.action["segment_id"] for r in alice} == {"alice#1"}
    assert alice[-1].action["kind"] == "channel_message"  # send closes the interval


def test_receive_opens_next_segment_and_notice_does_not(tmp_path: Path) -> None:
    rows = list(_recorded(tmp_path).ledger.records())
    notice = next(r for r in rows if r.action.get("kind") == "channel_notice")
    receive = next(r for r in rows if r.action.get("kind") == "channel_receive")
    assert "segment_id" not in notice.action
    bob_tool = next(r for r in rows if r.action.get("tool") == "read_file" and r.action["participant_id"] == "bob")
    assert receive.action["segment_id"] == bob_tool.action["segment_id"] == "bob#1"


def test_delivery_dedup_and_restart_keeps_segment_counter(tmp_path: Path) -> None:
    recorder = _recorded(tmp_path)
    assert recorder.record_channel_delivery(participant_id="bob", event_id="e1", text="hi") is None
    reloaded = AOLedgerRecorder(tmp_path / "s", "s")
    reloaded.record_messages([_turn("alice", "m3", ["t9"], "e3")])
    last = list(reloaded.ledger.records())[-2]
    assert last.action["segment_id"] == "alice#2"


def test_overview_returns_dag_and_rows_only_on_request(tmp_path: Path) -> None:
    records = tuple(_recorded(tmp_path).ledger.records())
    surface = AigToolSurface(AoSnapshot(records, watermark="w"), root=tmp_path, run_id="r")
    overview = surface.get_overview()
    assert "rows" not in overview
    dag = overview["interaction_dag"]
    segs = {(s["participant"], s["segment"]): s for s in dag["segments"]}
    assert segs[("alice", "alice#1")]["row_count"] == 3
    assert segs[("alice", "alice#1")]["tools"] == ["read_file"]
    kinds = {(e["kind"], e["from"], e["to"]) for e in dag["edges"]}
    assert ("emit", "alice#1", "e1") in kinds
    assert ("deliver", "e1", "bob#1") in kinds
    event = next(e for e in dag["events"] if e["event_id"] == "e1")
    assert event["sender"] == "alice"
    assert event["receivers"]["bob"] == {"notified": True, "read": True}
    assert "rows" in surface.call("get_overview", {"include_rows": True})
    with pytest.raises(ToolSurfaceError):
        surface.call("get_overview", {"include_rows": "yes"})


def test_judge_feedback_is_capped_optional_and_outside_digest() -> None:
    base = {
        "binding_id": "b",
        "signal_kind": "evaluator",
        "signal": "fail",
        "evidence_ref": "ev",
    }
    ok = external_evidence_from_session(evaluator_results=[{**base, "feedback": "KEY FINDINGS: x"}])
    assert ok
    with pytest.raises(OutcomeContractError):
        external_evidence_from_session(evaluator_results=[{**base, "feedback": "x" * 7000}])
    with pytest.raises(OutcomeContractError):
        external_evidence_from_session(evaluator_results=[{**base, "feedback": 3}])
    outcome = OutcomeSignal("fail", "verifier", {"reward": 0.2, "feedback": "KEY FINDINGS: x"})
    summary = outcome_overview(outcome)
    assert summary["feedback"] == "KEY FINDINGS: x"
    assert "KEY FINDINGS" not in summary["digest"]
    assert "feedback" not in outcome_overview(OutcomeSignal("fail", "verifier", {"reward": 0.2}))
