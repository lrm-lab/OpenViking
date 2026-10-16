# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Refiner edits stay inside the named violation."""

from __future__ import annotations

from pathlib import Path

import pytest

from openviking.session.aig_tool_surface import AigToolSurface, AoSnapshot
from openviking.session.aig_toolloop_refiner import (
    REFINER_SYSTEM_PROMPT,
    FakeRefinerProvider,
    RefinerAction,
    RefinerError,
    RefinerSession,
    refiner_catalog,
    run_refiner_round,
)
from openviking.session.ao_ledger import AORecord
from openviking.session.paper_graph_phi import PaperViolation, evaluate_phi


def _record(index: int, text: str) -> AORecord:
    return AORecord(
        ao_id=f"ao-{index}",
        session_id="sess-refiner",
        sequence=index + 1,
        action={"participant_id": "worker", "segment_id": "seg", "text": text},
        observation={"text": text},
        skill_invocations=[],
        message_ref=None,
        created_at="2026-10-11T00:00:00.000Z",
        captured_state="live",
    )


def _open(tmp_path: Path, count: int, run_id: str) -> AigToolSurface:
    snapshot = AoSnapshot(
        tuple(_record(index, f"row {index}") for index in range(count)),
        watermark="wm-refiner",
        question="q",
    )
    return AigToolSurface(snapshot, root=tmp_path, run_id=run_id)


def _semantic(object_id: str = "agent-1") -> PaperViolation:
    return PaperViolation(
        violation_id=f"critic:0:semantic:{object_id}",
        object_id=object_id,
        kind="semantic",
        severity="high",
        evidence="summary contradicts the row",
        evidence_ao_refs=(1,),
        note="summary contradicts the row",
        scope="objects",
        involved_ids=(object_id,),
    )


def _legal(tmp_path: Path) -> AigToolSurface:
    surface = _open(tmp_path, 3, "run-legal")
    surface.add_block(
        "task",
        {"content": "the question", "operative_brief": "start here", "step_refs": [0]},
        block_id="task-1",
    )
    surface.add_block(
        "agent",
        {
            "agent_name": "worker",
            "summary": "did the work",
            "input": "do the task",
            "output": "a draft",
            "authorship": "authored",
            "step_refs": [1],
        },
        block_id="agent-1",
    )
    surface.add_block("conclusion", {"content": "the answer", "step_refs": [2]}, block_id="end-1")
    surface.add_inheritance_note("task-1", "agent-1", "the question", "the worker dropped it")
    surface.add_inheritance_note("agent-1", "end-1", "the draft", "the answer published it")
    surface.submit()
    return surface


def _assert_code(action, code: str) -> None:
    with pytest.raises(RefinerError) as caught:
        action()
    assert caught.value.code == code


def test_catalog_excludes_submit_and_keeps_the_prompt(tmp_path: Path) -> None:
    surface = _legal(tmp_path)
    names = {name for name, _description in refiner_catalog()}
    assert "submit" not in names
    assert "add_block" in names
    assert "modify_block" in names
    assert "get_graph" in names
    assert "read_step" in names
    assert "read_step_range" in names
    assert "get_story" not in names
    assert "search" not in names
    session = RefinerSession(surface, (_semantic(),))
    provider = FakeRefinerProvider()
    run_refiner_round(session, provider)
    assert provider.turns[0].system_prompt == REFINER_SYSTEM_PROMPT
    assert "violation_id" in REFINER_SYSTEM_PROMPT


def test_missing_violation_id_is_rejected(tmp_path: Path) -> None:
    surface = _legal(tmp_path)
    session = RefinerSession(surface, (_semantic(),))
    _assert_code(
        lambda: session.call(
            "modify_block",
            {"block_id": "agent-1", "fields": {"summary": "rewritten"}},
        ),
        "violation_id_missing",
    )


def test_semantic_edit_must_modify_the_named_node(tmp_path: Path) -> None:
    surface = _legal(tmp_path)
    session = RefinerSession(surface, (_semantic(),))
    before = surface.get_graph()
    violation_id = "critic:0:semantic:agent-1"
    _assert_code(
        lambda: session.call(
            "modify_block",
            {"block_id": "task-1", "fields": {"content": "other"}},
            violation_id=violation_id,
        ),
        "edit_out_of_violation_scope",
    )
    _assert_code(
        lambda: session.call(
            "add_block",
            {
                "kind": "orchestrator",
                "block_id": "orch-1",
                "fields": {
                    "agent_name": "worker",
                    "summary": "routed",
                    "authorship": "relayed",
                    "step_refs": [0],
                },
            },
            violation_id=violation_id,
        ),
        "edit_out_of_violation_scope",
    )
    assert surface.get_graph() == before
    session.call(
        "modify_block",
        {"block_id": "agent-1", "fields": {"summary": "a tighter summary"}},
        violation_id=violation_id,
    )
    agent = next(block for block in surface.get_graph()["blocks"] if block["block_id"] == "agent-1")
    assert agent["summary"] == "a tighter summary"
    _assert_code(lambda: session.call("submit", violation_id=violation_id), "submit_forbidden")


def test_structural_graph_scope_allows_add_block(tmp_path: Path) -> None:
    surface = _open(tmp_path, 3, "run-root")
    surface.add_block(
        "agent",
        {
            "agent_name": "worker",
            "summary": "did the work",
            "input": "do the task",
            "output": "a draft",
            "authorship": "authored",
            "step_refs": [1],
        },
        block_id="agent-1",
    )
    surface.add_block("conclusion", {"content": "the answer", "step_refs": [2]}, block_id="end-1")
    surface.add_inheritance_note("agent-1", "end-1", "the draft", "the answer published it")
    surface.submit()
    violations = evaluate_phi(surface.get_graph()).violations
    session = RefinerSession(surface, violations)
    before = surface.get_graph()
    _assert_code(
        lambda: session.call(
            "add_block",
            {
                "kind": "task",
                "block_id": "task-1",
                "fields": {
                    "content": "the question",
                    "operative_brief": "start here",
                    "step_refs": [0],
                },
            },
            violation_id="phi:extra_root:agent-1",
        ),
        "edit_out_of_violation_scope",
    )
    assert surface.get_graph() == before
    created = session.call(
        "add_block",
        {
            "kind": "task",
            "block_id": "task-1",
            "fields": {
                "content": "the question",
                "operative_brief": "start here",
                "step_refs": [0],
            },
        },
        violation_id="phi:missing_task_root",
    )
    assert created == {"block_id": "task-1"}
    assert any(block["block_id"] == "task-1" for block in surface.get_graph()["blocks"])


def test_unresolvable_and_unknown_violation(tmp_path: Path) -> None:
    surface = _legal(tmp_path)
    session = RefinerSession(surface, (_semantic(),))
    _assert_code(
        lambda: session.declare_unresolvable("missing", "no such row"),
        "unknown_violation",
    )
    record = session.declare_unresolvable("critic:0:semantic:agent-1", "the row really says this")
    assert record == {
        "action": "unresolvable",
        "reason": "the row really says this",
        "violation_id": "critic:0:semantic:agent-1",
    }
    provider = FakeRefinerProvider(
        [
            RefinerAction(
                kind="unresolvable",
                violation_id="critic:0:semantic:agent-1",
                reason="stuck",
            )
        ]
    )
    refined = run_refiner_round(RefinerSession(surface, (_semantic(),)), provider)
    assert refined.edits == ()
    assert refined.unresolvable[0]["action"] == "unresolvable"


@pytest.mark.parametrize("fields", [{"summary": "did the work"}, {}])
def test_no_effect_edit_is_rejected_and_repeat_cannot_exhaust_round(
    tmp_path: Path, fields: dict
) -> None:
    surface = _legal(tmp_path)
    session = RefinerSession(surface, (_semantic(),))
    action = RefinerAction(
        "edit",
        "modify_block",
        {"block_id": "agent-1", "fields": fields},
        "critic:0:semantic:agent-1",
    )
    _assert_code(
        lambda: session.call(action.verb, action.arguments, violation_id=action.violation_id),
        "no_effect_edit",
    )
    provider = FakeRefinerProvider([action] * 8)
    result = run_refiner_round(session, provider)
    assert len(provider.turns) <= 3
    assert result.edits == ()
    assert result.stop_reason == "no_progress"
    assert result.unresolvable[0]["violation_id"] == action.violation_id
