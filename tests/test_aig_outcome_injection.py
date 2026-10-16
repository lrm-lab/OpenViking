# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ADR-0045 Wave E1: snapshot outcome injection and dual-narrative prompts."""

from __future__ import annotations

from pathlib import Path

import pytest

from openviking.session.aig_native_protocol import TOOLLOOP_BUILD_VERBS
from openviking.session.aig_tool_surface import (
    AigToolSurface,
    AoSnapshot,
    ToolSurfaceError,
    snapshot_fingerprint,
)
from openviking.session.aig_toolloop_builder import (
    BUILDER_SUCCESS_PROMPT,
    BUILDER_SYSTEM_PROMPT,
    FakeToolloopProvider,
    builder_prompt_for_mode,
    run_toolloop,
)
from openviking.session.aig_toolloop_critic import (
    CRITIC_SUCCESS_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    FakeCriticProvider,
    audit_semantics,
    critic_prompt_for_mode,
)
from openviking.session.aig_toolloop_reader import (
    READER_OUTPUT_SCHEMA,
    READER_SUCCESS_PROMPT,
    READER_SYSTEM_PROMPT,
    reader_prompt_for_mode,
)
from openviking.session.aig_toolloop_refiner import (
    REFINER_SUCCESS_PROMPT,
    REFINER_SYSTEM_PROMPT,
    FakeRefinerProvider,
    RefinerSession,
    refiner_prompt_for_mode,
    run_refiner_round,
)
from openviking.session.ao_ledger import AORecord
from openviking.session.outcome_signal import OutcomeSignal, outcome_overview
from openviking.session.paper_graph_phi import PaperViolation


def _record(index: int, text: str = "row") -> AORecord:
    return AORecord(
        ao_id=f"ao-{index}",
        session_id="sess-outcome",
        sequence=index + 1,
        action={"participant_id": "worker", "segment_id": "seg", "text": text},
        observation={"text": text},
        skill_invocations=[],
        message_ref=None,
        created_at="2026-10-11T00:00:00.000Z",
        captured_state="live",
    )


def _records() -> tuple[AORecord, ...]:
    return (_record(0, "ask"), _record(1, "answer"))


def _outcome(signal: str = "fail", *, source: str = "verifier") -> OutcomeSignal:
    return OutcomeSignal(
        signal=signal,
        source=source,
        raw={"reward": 0.0 if signal == "fail" else 1.0, "error": None},
    )


def _snapshot(*, outcome: OutcomeSignal | None = None) -> AoSnapshot:
    return AoSnapshot(
        records=_records(),
        watermark="wm-outcome",
        question="what failed?",
        outcome=outcome,
    )


def test_snapshot_rejects_a_non_signal_outcome() -> None:
    with pytest.raises(ToolSurfaceError) as caught:
        AoSnapshot(records=_records(), watermark="wm-1", outcome="fail")  # type: ignore[arg-type]
    assert caught.value.code == "invalid_snapshot"


def test_snapshot_fingerprint_changes_with_outcome() -> None:
    records = _records()
    empty = snapshot_fingerprint(records)
    none = snapshot_fingerprint(records, None)
    fail = snapshot_fingerprint(records, _outcome("fail"))
    success = snapshot_fingerprint(records, _outcome("pass"))
    inferred = snapshot_fingerprint(records, _outcome("fail", source="model_inferred"))
    assert empty == none
    assert fail != empty
    assert success != fail
    assert inferred != fail
    bound_fail = AoSnapshot(records, watermark="wm-1", outcome=_outcome("fail"))
    bound_pass = AoSnapshot(records, watermark="wm-1", outcome=_outcome("pass"))
    assert snapshot_fingerprint(bound_fail.records, bound_fail.outcome) != snapshot_fingerprint(
        bound_pass.records, bound_pass.outcome
    )


def test_get_overview_omits_outcome_when_absent(tmp_path: Path) -> None:
    surface = AigToolSurface(_snapshot(), root=tmp_path, run_id="run-none")
    overview = surface.get_overview()
    assert "outcome" not in overview
    assert overview["row_count"] == 2
    assert overview["watermark"] == "wm-outcome"


def test_get_overview_summarizes_outcome(tmp_path: Path) -> None:
    outcome = _outcome("pass")
    surface = AigToolSurface(_snapshot(outcome=outcome), root=tmp_path, run_id="run-sum")
    overview = surface.get_overview()
    summary = overview["outcome"]
    expected = outcome_overview(outcome)
    assert summary == expected
    assert set(summary) == {"signal", "source", "digest"}
    assert summary["signal"] == "pass"
    assert summary["source"] == "verifier"
    assert isinstance(summary["digest"], str)
    assert 0 < len(summary["digest"]) <= 200
    assert "\n" not in summary["digest"]


def test_outcome_does_not_offer_a_build_verb(tmp_path: Path) -> None:
    assert "outcome" not in TOOLLOOP_BUILD_VERBS
    surface = AigToolSurface(_snapshot(outcome=_outcome()), root=tmp_path, run_id="run-ro")
    with pytest.raises(ToolSurfaceError) as top:
        surface.call("add_block", {"kind": "task", "fields": {}, "outcome": {"signal": "fail"}})
    assert top.value.code == "outcome_readonly"
    with pytest.raises(ToolSurfaceError) as fields:
        surface.call(
            "add_block",
            {
                "kind": "task",
                "fields": {
                    "content": "q",
                    "operative_brief": "start",
                    "step_refs": [0],
                    "outcome": {"signal": "fail"},
                },
            },
        )
    assert fields.value.code == "outcome_readonly"
    graph = surface.get_graph()
    assert "outcome" not in graph
    overview = surface.get_overview()
    assert overview["outcome"]["signal"] == "fail"


def test_eight_prompts_exist_and_differ_by_mode() -> None:
    pairs = (
        (BUILDER_SYSTEM_PROMPT, BUILDER_SUCCESS_PROMPT, "失败轨迹", "成功轨迹"),
        (CRITIC_SYSTEM_PROMPT, CRITIC_SUCCESS_PROMPT, "下游错误确实呈现", "下游成功确实源于该复用"),
        (REFINER_SYSTEM_PROMPT, REFINER_SUCCESS_PROMPT, "违规清单", "成功轨迹"),
        (READER_SYSTEM_PROMPT, READER_SUCCESS_PROMPT, "下游错误", "下游成功贡献"),
    )
    for fail, success, fail_mark, success_mark in pairs:
        assert isinstance(fail, str) and isinstance(success, str)
        assert fail != success
        assert fail_mark in fail
        assert success_mark in success
        assert fail_mark not in success or fail_mark == "违规清单"
    assert "这次复用如何促成最终正确结果" in BUILDER_SUCCESS_PROMPT
    assert "成功轨迹仍可有未达标记录" in BUILDER_SUCCESS_PROMPT
    assert "mark_defect_recorded" in BUILDER_SUCCESS_PROMPT
    assert "关键贡献节点" in READER_SUCCESS_PROMPT
    assert "归因对象" in READER_SUCCESS_PROMPT


def test_reader_output_schema_string_is_identical() -> None:
    assert READER_OUTPUT_SCHEMA in READER_SYSTEM_PROMPT
    assert READER_OUTPUT_SCHEMA in READER_SUCCESS_PROMPT
    fail_at = READER_SYSTEM_PROMPT.index(READER_OUTPUT_SCHEMA)
    success_at = READER_SUCCESS_PROMPT.index(READER_OUTPUT_SCHEMA)
    fail_schema = READER_SYSTEM_PROMPT[fail_at : fail_at + len(READER_OUTPUT_SCHEMA)]
    success_schema = READER_SUCCESS_PROMPT[success_at : success_at + len(READER_OUTPUT_SCHEMA)]
    assert fail_schema == success_schema == READER_OUTPUT_SCHEMA


def test_prompt_selectors_are_fail_closed() -> None:
    assert builder_prompt_for_mode("fail") == BUILDER_SYSTEM_PROMPT
    assert builder_prompt_for_mode("success") == BUILDER_SUCCESS_PROMPT
    assert builder_prompt_for_mode(None) == BUILDER_SYSTEM_PROMPT
    assert builder_prompt_for_mode("bogus") == BUILDER_SYSTEM_PROMPT
    assert critic_prompt_for_mode(None) == CRITIC_SYSTEM_PROMPT
    assert critic_prompt_for_mode("success") == CRITIC_SUCCESS_PROMPT
    assert refiner_prompt_for_mode("success") == REFINER_SUCCESS_PROMPT
    assert refiner_prompt_for_mode("FAIL") == REFINER_SYSTEM_PROMPT
    assert reader_prompt_for_mode("success") == READER_SUCCESS_PROMPT
    assert reader_prompt_for_mode(None) == READER_SYSTEM_PROMPT


def test_run_toolloop_default_mode_matches_fail_prompt(tmp_path: Path) -> None:
    script = [("get_overview", {}), ("submit", {})]
    default_provider = FakeToolloopProvider(script)
    fail_provider = FakeToolloopProvider(script)
    success_provider = FakeToolloopProvider(script)
    snapshot = _snapshot()
    default = run_toolloop(
        snapshot, "what failed?", default_provider, root=tmp_path, run_id="run-default"
    )
    explicit = run_toolloop(
        snapshot,
        "what failed?",
        fail_provider,
        root=tmp_path,
        run_id="run-fail",
        mode="fail",
    )
    run_toolloop(
        snapshot,
        "what failed?",
        success_provider,
        root=tmp_path,
        run_id="run-success",
        mode="success",
    )
    assert default_provider.turns[0].system_prompt == BUILDER_SYSTEM_PROMPT
    assert fail_provider.turns[0].system_prompt == BUILDER_SYSTEM_PROMPT
    assert success_provider.turns[0].system_prompt == BUILDER_SUCCESS_PROMPT
    assert default_provider.turns[0].tools == fail_provider.turns[0].tools
    assert default.status == explicit.status == "submitted"
    assert [item.verb for item in default.audit] == [item.verb for item in explicit.audit]


def test_run_toolloop_preserves_outcome_on_the_bound_snapshot(tmp_path: Path) -> None:
    provider = FakeToolloopProvider([("get_overview", {}), ("submit", {})])
    result = run_toolloop(
        _snapshot(outcome=_outcome("pass")),
        "what passed?",
        provider,
        root=tmp_path,
        run_id="run-keep",
        mode="success",
    )
    assert result.status == "submitted"
    overview = provider.turns[1].history[0]["result"]
    assert overview["outcome"]["signal"] == "pass"


def test_critic_and_refiner_honor_mode(tmp_path: Path) -> None:
    snapshot = _snapshot()
    surface = AigToolSurface(snapshot, root=tmp_path, run_id="run-roles")
    surface.add_block(
        "agent",
        {
            "agent_name": "worker",
            "summary": "did the work",
            "input": "do the task",
            "output": "a draft",
            "authorship": "authored",
            "step_refs": [0],
        },
        block_id="agent-1",
    )
    surface.add_block("conclusion", {"content": "the answer", "step_refs": [1]}, block_id="end-1")
    critic = FakeCriticProvider([{"violations": []}])
    audit_semantics(surface, (), critic, mode="success")
    assert critic.turns[0].system_prompt == CRITIC_SUCCESS_PROMPT
    surface.submit()
    violation = PaperViolation(
        violation_id="critic:0:semantic:agent-1",
        object_id="agent-1",
        kind="semantic",
        severity="low",
        evidence="note",
        evidence_ao_refs=(0,),
        note="note",
        scope="objects",
        involved_ids=("agent-1",),
    )
    refiner = FakeRefinerProvider()
    run_refiner_round(RefinerSession(surface, (violation,)), refiner, mode="success")
    assert refiner.turns[0].system_prompt == REFINER_SUCCESS_PROMPT
