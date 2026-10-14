# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ADR-0044 Wave 1: Builder tool-loop driver, budget, and retry."""

from __future__ import annotations

from pathlib import Path

import pytest

from openviking.session.aig_tool_surface import TOOL_CATALOG, AoSnapshot
from openviking.session.aig_toolloop_builder import (
    BUILDER_SYSTEM_PROMPT,
    CONSTRUCTION_ADAPTIVE_AGENTIC,
    CONSTRUCTION_FALLBACK,
    DEFAULT_MAX_CALLS,
    FakeToolloopProvider,
    ToolInvocation,
    ToolloopError,
    ToolloopTurn,
    run_toolloop,
)
from openviking.session.ao_ledger import AORecord
from openviking.session.influence_view import CONSTRUCTION_MODE_FALLBACK


def _record(index: int, text: str) -> AORecord:
    return AORecord(
        ao_id=f"ao-{index}",
        session_id="sess-loop",
        sequence=index + 1,
        action={"participant_id": "worker", "segment_id": "seg", "text": text},
        observation={"text": text},
        skill_invocations=[],
        message_ref=None,
        created_at="2026-10-11T00:00:00.000Z",
        captured_state="live",
    )


def _snapshot() -> AoSnapshot:
    return AoSnapshot(
        records=(_record(0, "ask"), _record(1, "answer")),
        watermark="wm-loop",
        question=None,
    )


def _task(steps: list[int]) -> dict[str, object]:
    return {"content": "the question", "operative_brief": "start here", "step_refs": steps}


def _agent(steps: list[int]) -> dict[str, object]:
    return {
        "agent_name": "worker",
        "summary": "did the work",
        "input": "do the task",
        "output": "a draft",
        "authorship": "authored",
        "step_refs": steps,
    }


def _end(steps: list[int]) -> dict[str, object]:
    return {"content": "the answer", "step_refs": steps}


def test_fake_provider_inspects_builds_and_submits(tmp_path: Path) -> None:
    script = [
        ("get_overview", {}),
        ("get_question", {}),
        ("read_step", {"index": 0}),
        ("add_block", {"kind": "task", "fields": _task([0]), "block_id": "task-1"}),
        ("add_block", {"kind": "conclusion", "fields": _end([1]), "block_id": "end-1"}),
        (
            "add_inheritance_note",
            {
                "from": "task-1",
                "to": "end-1",
                "inherited": "the question",
                "effect": "the answer dropped a constraint",
            },
        ),
        ("submit", {}),
    ]
    provider = FakeToolloopProvider(script)
    result = run_toolloop(
        _snapshot(),
        "what failed?",
        provider,
        root=tmp_path,
        run_id="run-loop",
    )
    assert result.status == "submitted"
    assert result.construction_mode == CONSTRUCTION_ADAPTIVE_AGENTIC
    assert result.construction_mode == "adaptive_agentic_aig"
    assert result.fallback_marker is None
    assert result.draft is not None
    assert result.draft["submitted"] is True
    assert [block["block_id"] for block in result.draft["blocks"]] == ["task-1", "end-1"]
    assert result.draft["edges"][0]["link_kind"] == "adjacent"
    assert len(result.audit) == len(script)
    assert all(item.ok and item.error_code is None for item in result.audit)
    assert result.audit[0].verb == "get_overview"
    assert result.audit[0].result_summary == "rows=2"
    assert result.audit[-1].result_summary == "submitted"
    assert provider.turns[0].system_prompt == BUILDER_SYSTEM_PROMPT
    assert provider.turns[0].tools == TOOL_CATALOG
    assert len(provider.turns[0].tools) == 19
    assert provider.turns[1].history[0]["verb"] == "get_overview"
    assert provider.turns[1].history[0]["ok"] is True
    assert provider.turns[1].history[0]["error_code"] is None


def test_dead_loop_exhausts_call_budget(tmp_path: Path) -> None:
    provider = FakeToolloopProvider()
    result = run_toolloop(
        _snapshot(),
        "what failed?",
        provider,
        root=tmp_path,
        run_id="run-budget",
        max_calls=3,
    )
    assert result.status == "budget_exhausted"
    assert result.construction_mode == "fallback"
    assert result.construction_mode == CONSTRUCTION_FALLBACK
    assert result.construction_mode != CONSTRUCTION_ADAPTIVE_AGENTIC
    assert result.draft is None
    marker = result.fallback_marker
    assert marker is not None
    assert marker["accepted_as_aig"] is False
    assert marker["reason"] == "budget_exhausted"
    assert marker["deterministic_view"] == "build_minimal_fallback_view"
    assert marker["calls_used"] == 3
    assert marker["construction_mode"] == "fallback"
    assert len(result.audit) == 3
    assert all(item.verb == "get_overview" for item in result.audit)
    assert result.audit[0].ok and result.audit[1].ok
    assert result.audit[2].ok is False
    assert result.audit[2].error_code == "repeat_call"
    assert DEFAULT_MAX_CALLS == 64
    assert CONSTRUCTION_FALLBACK == CONSTRUCTION_MODE_FALLBACK


def test_wall_clock_budget_stops_the_loop(tmp_path: Path) -> None:
    clock = {"now": 0.0}

    def now() -> float:
        return clock["now"]

    class Slow(FakeToolloopProvider):
        def __call__(self, turn: ToolloopTurn) -> ToolInvocation:
            chosen = super().__call__(turn)
            clock["now"] += 10.0
            return chosen

    result = run_toolloop(
        _snapshot(),
        "what failed?",
        Slow(),
        root=tmp_path,
        run_id="run-clock",
        max_seconds=15,
        clock=now,
    )
    assert result.status == "budget_exhausted"
    assert result.draft is None
    assert len(result.audit) == 2
    assert result.fallback_marker is not None
    assert result.fallback_marker["calls_used"] == 2


def test_rejection_returns_the_error_code_for_retry(tmp_path: Path) -> None:
    class Repair:
        def __init__(self) -> None:
            self.codes: list[str | None] = []

        def __call__(self, turn: ToolloopTurn) -> ToolInvocation:
            if turn.history:
                self.codes.append(turn.history[-1]["error_code"])
            step = len(turn.history)
            if step == 0:
                return ToolInvocation(
                    "add_block",
                    {"kind": "task", "block_id": "task-1", "fields": _task([0])},
                )
            if step == 1:
                return ToolInvocation(
                    "add_block",
                    {"kind": "agent", "block_id": "agent-1", "fields": _agent([0])},
                )
            if step == 2:
                return ToolInvocation(
                    "add_block",
                    {"kind": "agent", "block_id": "agent-1", "fields": _agent([1])},
                )
            return ToolInvocation("submit", {})

    provider = Repair()
    result = run_toolloop(
        _snapshot(),
        "what failed?",
        provider,
        root=tmp_path,
        run_id="run-repair",
    )
    assert "step_refs_overlap" in provider.codes
    assert result.status == "submitted"
    assert result.construction_mode == "adaptive_agentic_aig"
    assert result.draft is not None
    agent = next(block for block in result.draft["blocks"] if block["kind"] == "agent")
    assert agent["step_refs"] == [1]
    rejected = next(item for item in result.audit if not item.ok)
    assert rejected.error_code == "step_refs_overlap"
    assert rejected.verb == "add_block"


def test_repeated_backward_pair_feedback_ignores_note_wording(tmp_path: Path) -> None:
    provider = FakeToolloopProvider(
        [
            ("add_block", {"kind": "task", "block_id": "t", "fields": _task([0])}),
            ("add_block", {"kind": "conclusion", "block_id": "c", "fields": _end([1])}),
            ("add_inheritance_note", {"from": "c", "to": "t", "inherited": "a", "effect": "b"}),
            ("add_inheritance_note", {"from": "c", "to": "t", "inherited": "c", "effect": "d"}),
            ("submit", {}),
        ]
    )
    result = run_toolloop(
        _snapshot(), "what failed?", provider, root=tmp_path, run_id="pair-repeat"
    )
    assert result.status == "submitted"
    feedback = provider.turns[-1].history[-1]
    assert feedback["error_code"] == "edge_not_forward"
    assert feedback["result"]["same_pair_rejections"] == 2
    assert "Changing inherited/effect text" in feedback["result"]["progress_hint"]
    assert result.draft["edges"] == []


def test_partial_graph_is_not_returned_as_an_aig(tmp_path: Path) -> None:
    provider = FakeToolloopProvider(
        [("add_block", {"kind": "task", "fields": _task([0]), "block_id": "task-1"})]
    )
    result = run_toolloop(
        _snapshot(),
        "what failed?",
        provider,
        root=tmp_path,
        run_id="run-partial",
        max_calls=1,
    )
    assert result.status == "budget_exhausted"
    assert result.draft is None
    assert result.construction_mode == "fallback"
    assert result.audit[0].ok is True
    assert result.audit[0].verb == "add_block"


def test_invalid_budget_and_question(tmp_path: Path) -> None:
    provider = FakeToolloopProvider()
    with pytest.raises(ToolloopError) as budget:
        run_toolloop(
            _snapshot(), "what failed?", provider, root=tmp_path, run_id="run-bad", max_calls=0
        )
    assert budget.value.code == "invalid_budget"
    with pytest.raises(ToolloopError) as clock:
        run_toolloop(
            _snapshot(),
            "what failed?",
            provider,
            root=tmp_path,
            run_id="run-bad",
            clock=lambda: "no",
        )
    assert clock.value.code == "invalid_budget"
    with pytest.raises(ToolloopError) as missing:
        run_toolloop(_snapshot(), "  ", provider, root=tmp_path, run_id="run-bad")
    assert missing.value.code == "question_missing"


def test_alternating_redundant_reads_exit_without_spending_whole_budget(tmp_path: Path) -> None:
    from benchmark.causal_memory.aig0045_verdict_smoke import synthetic_materials

    spec = synthetic_materials()[0]
    snapshot = AoSnapshot(spec.records, watermark="wm-stalled", question=spec.question)
    provider = FakeToolloopProvider([("get_question", {}), ("get_overview", {})] * 20)
    result = run_toolloop(
        snapshot, spec.question, provider, root=tmp_path, run_id="alternating-stall", max_calls=64
    )
    assert result.status == "no_progress"
    assert len(result.audit) < 20
