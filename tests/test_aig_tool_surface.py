# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ADR-0044 Wave 1: T_log, T_build, hard invariants, and durable replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openviking.session.aig_native_protocol import (
    TOOLLOOP_BUILD_VERBS,
    ProtocolContractError,
    read_protocol_document,
    read_toolloop_draft_op,
    toolloop_draft_op_document,
)
from openviking.session.aig_tool_surface import (
    SEARCH_MATCH_LIMIT,
    T_LOG_VERBS,
    TOOL_CATALOG,
    AigToolSurface,
    AoSnapshot,
    SnapshotAgent,
    ToolSurfaceError,
    replay_toolloop_draft,
)
from openviking.session.ao_ledger import AORecord
from openviking.session.paper_graph_compile import compile_paper_graph

_SECRET = "SECRET-PROMPT"
_DERIVED_SECRET = "SECRET-DERIVED"


def _record(
    index: int,
    text: str,
    *,
    participant: str = "agent-a",
    segment: str = "seg-1",
    role: str | None = None,
    capability: str | None = None,
    system_prompt: str | None = None,
    observation: str | None = None,
) -> AORecord:
    action: dict[str, object] = {
        "participant_id": participant,
        "segment_id": segment,
        "text": text,
    }
    if role is not None:
        action["role"] = role
    if capability is not None:
        action["capability"] = capability
    if system_prompt is not None:
        action["system_prompt"] = system_prompt
    return AORecord(
        ao_id=f"ao-{index}",
        session_id="sess-tool",
        sequence=index + 1,
        action=action,
        observation={"text": text if observation is None else observation},
        skill_invocations=[],
        message_ref=None,
        created_at="2026-10-11T00:00:00.000Z",
        captured_state="live",
    )


def _snapshot(
    texts: list[str],
    *,
    question: str | None = "what failed?",
    watermark: str = "wm-1",
    agents: tuple[SnapshotAgent, ...] = (),
    records: tuple[AORecord, ...] | None = None,
) -> AoSnapshot:
    rows = (
        records
        if records is not None
        else tuple(_record(i, text) for i, text in enumerate(texts))
    )
    return AoSnapshot(rows, watermark=watermark, question=question, agents=agents)


def _open(
    tmp_path: Path,
    texts: list[str],
    *,
    run_id: str = "run-1",
    question: str | None = "what failed?",
    watermark: str = "wm-1",
    agents: tuple[SnapshotAgent, ...] = (),
    records: tuple[AORecord, ...] | None = None,
) -> tuple[AigToolSurface, AoSnapshot]:
    snapshot = _snapshot(
        texts,
        question=question,
        watermark=watermark,
        agents=agents,
        records=records,
    )
    return AigToolSurface(snapshot, root=tmp_path, run_id=run_id), snapshot


def _task(steps: list[int]) -> dict[str, object]:
    return {"content": "the question", "operative_brief": "start here", "step_refs": steps}


def _orch(steps: list[int]) -> dict[str, object]:
    return {
        "agent_name": "agent-a",
        "summary": "routed the work",
        "authorship": "relayed",
        "step_refs": steps,
    }


def _agent(steps: list[int], **extra: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "agent_name": "agent-a",
        "summary": "did the work",
        "input": "do the task",
        "output": "a draft",
        "authorship": "authored",
        "step_refs": steps,
    }
    fields.update(extra)
    return fields


def _end(steps: list[int]) -> dict[str, object]:
    return {"content": "the answer", "step_refs": steps}


def _note(block_id: str) -> dict[str, str]:
    return {
        "block_id": block_id,
        "inherited": "the prior result",
        "effect": "it misled the next step",
    }


def _populate(tmp_path: Path) -> tuple[AigToolSurface, AoSnapshot]:
    surface, snapshot = _open(tmp_path, ["q", "plan", "work", "answer"])
    surface.add_block("task", _task([0]), block_id="task-1")
    surface.add_block("orchestrator", _orch([1]), block_id="orch-1")
    surface.add_block("agent", _agent([2]), block_id="agent-1")
    surface.add_block("conclusion", _end([3]), block_id="end-1")
    surface.add_inheritance_note(
        "task-1", "orch-1", "the question", "the route dropped a constraint"
    )
    surface.add_inheritance_note(
        "orch-1", "agent-1", "the plan", "the worker followed the bad plan"
    )
    surface.add_inheritance_note("agent-1", "end-1", "the draft", "the answer published it")
    surface.add_skip_attribution(
        "task-1", "agent-1", "the question", "the worker skipped the route"
    )
    return surface, snapshot


def _log_text(surface: AigToolSurface) -> str:
    path = surface.draft_log_path
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _assert_code(action, code: str) -> None:
    with pytest.raises(ToolSurfaceError) as caught:
        action()
    assert caught.value.code == code


def test_catalog_covers_nineteen_verbs() -> None:
    names = [name for name, _description in TOOL_CATALOG]
    assert names == [*T_LOG_VERBS, *sorted(TOOLLOOP_BUILD_VERBS, key=names.index)]
    assert len(T_LOG_VERBS) == 10
    assert len(TOOLLOOP_BUILD_VERBS) == 9
    assert len(names) == 19


def test_overview_full_segment_and_step_bounds(tmp_path: Path) -> None:
    surface, _snapshot = _open(tmp_path, ["alpha", "beta"])
    overview = surface.get_overview()
    assert overview["row_count"] == 2
    assert overview["watermark"] == "wm-1"
    assert overview["rows"][0]["participant"] == "agent-a"
    assert overview["rows"][0]["segment"] == "seg-1"
    assert overview["rows"][0]["index"] == 0
    assert (
        overview["rows"][0]["size"] > overview["rows"][1]["size"]
        or overview["rows"][0]["size"] > 0
    )
    full = surface.read_full_segment("seg-1")
    assert "alpha" in full
    assert "beta" in full
    assert surface.call("read_full_segment", {"s": "seg-1"}) == full
    step = surface.read_step(0)
    assert step["index"] == 0
    assert step["record"]["action"]["text"] == "alpha"
    assert surface.read_step_range(0, 1)[1]["record"]["action"]["text"] == "beta"
    assert surface.call("read_step", {"i": 1})["index"] == 1
    _assert_code(lambda: surface.read_step(-1), "step_out_of_range")
    _assert_code(lambda: surface.read_step(2), "step_out_of_range")
    _assert_code(lambda: surface.read_step_range(1, 0), "invalid_range")
    _assert_code(lambda: surface.read_step_range(0, 3), "step_out_of_range")
    _assert_code(lambda: surface.call("read_step", {"index": "0"}), "invalid_args")
    _assert_code(lambda: surface.call("get_overview", {"extra": 1}), "invalid_args")
    _assert_code(lambda: surface.call("not_a_verb"), "unknown_verb")


def test_read_full_segment_filters_and_reports_unknown(tmp_path: Path) -> None:
    records = (
        _record(0, "alpha", segment="seg-a"),
        _record(1, "beta", segment="seg-b"),
        _record(2, "gamma", segment="seg-a"),
    )
    surface, _snapshot = _open(tmp_path, [], records=records)
    seg_a = surface.read_full_segment("seg-a")
    assert "alpha" in seg_a
    assert "gamma" in seg_a
    assert "beta" not in seg_a
    assert "[2]" in seg_a
    assert "[1]" not in seg_a
    assert "beta" in surface.read_full_segment("seg-b")
    with pytest.raises(ToolSurfaceError) as caught:
        surface.read_full_segment("seg-x")
    assert caught.value.code == "unknown_segment"
    assert caught.value.detail == {"available_segments": ["seg-a", "seg-b"]}
    _assert_code(lambda: surface.call("read_full_segment", {}), "invalid_args")
    _assert_code(
        lambda: surface.call("read_full_segment", {"segment": 1}), "invalid_args"
    )


def test_bulk_reads_truncate_but_read_step_stays_full(tmp_path: Path) -> None:
    long_action = "A" * 5000
    long_obs = "B" * 5000
    records = (
        _record(0, "alpha", observation=long_obs),
        _record(1, long_action),
    )
    records[0].action["text"] = long_action
    surface, _snapshot = _open(tmp_path, [], records=records)

    full = surface.read_full_segment("seg-1")
    assert "…[truncated" in full
    assert "A" * 5000 not in full
    assert "B" * 5000 not in full
    for line in full.split("\n\n"):
        assert len(line) < 2 * 2000 + 400  # two capped fields plus row header

    ranged = surface.read_step_range(0, 1)
    for row in ranged:
        assert isinstance(row["record"]["action"], dict)
        assert isinstance(row["record"]["observation"], dict)
        text = row["record"]["observation"].get("text") or ""
        if len(text):
            assert len(text) < 5000 or "…[truncated" in text
    assert ranged[0]["record"]["observation"]["text"].startswith("B" * 2000)
    assert "…[truncated" in ranged[0]["record"]["observation"]["text"]
    assert ranged[1]["record"]["action"]["text"].startswith("A" * 2000)
    assert "…[truncated" in ranged[1]["record"]["action"]["text"]

    step = surface.read_step(0)
    assert step["record"]["observation"]["text"] == long_obs
    step1 = surface.read_step(1)
    assert step1["record"]["action"]["text"] == long_action

    overview = surface.get_overview()
    assert overview["rows"][0]["size"] > 2 * 2000

    matches = surface.call("search", {"pattern": "B{4500}"})
    assert len(matches) >= 1


def test_search_caps_matches_and_honors_flags(tmp_path: Path) -> None:
    noisy = _record(0, "quiet", observation="ZZ" * (SEARCH_MATCH_LIMIT + 1))
    cased = _record(1, "quiet", observation="Needle")
    surface, _snapshot = _open(tmp_path, [], records=(noisy, cased))
    found = surface.search("ZZ")
    assert len(found["matches"]) == SEARCH_MATCH_LIMIT
    assert found["truncated"] is True
    assert {item["index"] for item in found["matches"]} == {0}
    assert found["matches"][0]["snippet"] == "ZZ"
    sensitive = surface.search("needle")
    assert sensitive["matches"] == []
    assert sensitive["truncated"] is False
    folded = surface.search("needle", flags="i")
    assert len(folded["matches"]) == 1
    assert folded["matches"][0]["index"] == 1
    assert folded["matches"][0]["snippet"] == "Needle"
    _assert_code(lambda: surface.search("["), "invalid_pattern")
    _assert_code(lambda: surface.search(""), "invalid_pattern")
    _assert_code(lambda: surface.search("ZZ", flags="z"), "invalid_flags")


def test_list_agents_hides_system_prompt(tmp_path: Path) -> None:
    agent = SnapshotAgent("worker", "coder", "tools", system_prompt=_SECRET)
    explicit, _snapshot = _open(tmp_path, ["alpha"], agents=(agent,), run_id="run-explicit")
    listed = explicit.list_agents()
    assert listed == [{"identity": "worker", "role": "coder", "capability": "tools"}]
    rendered = json.dumps(listed)
    assert _SECRET not in rendered
    assert "system_prompt" not in rendered
    explicit.add_block("task", _task([0]), block_id="task-1")
    disk = explicit.draft_log_path.read_text(encoding="utf-8")
    disk += (explicit.run_directory / "meta.json").read_text(encoding="utf-8")
    assert _SECRET not in disk
    assert "system_prompt" not in disk

    derived_row = _record(
        0,
        "alpha",
        participant="planner",
        role="coordinator",
        capability="routing",
        system_prompt=_DERIVED_SECRET,
    )
    derived, _snapshot = _open(
        tmp_path, [], records=(derived_row,), run_id="run-derived", agents=()
    )
    catalog = derived.list_agents()
    assert catalog == [
        {"identity": "planner", "role": "coordinator", "capability": "routing"}
    ]
    assert _DERIVED_SECRET not in json.dumps(catalog)
    assert "system_prompt" not in json.dumps(catalog)


def test_question_and_block_kind_vocabulary(tmp_path: Path) -> None:
    missing, _bound = _open(tmp_path, ["alpha"], question=None, run_id="run-q")
    _assert_code(missing.get_question, "question_missing")
    present, _bound = _open(tmp_path, ["alpha"], question="why did it fail?", run_id="run-q2")
    assert present.get_question() == {"question": "why did it fail?"}
    kinds = present.list_block_kinds()
    assert set(kinds["kinds"]) == {"task", "orchestrator", "agent", "conclusion"}
    assert kinds["kinds"]["task"]["required"] == ["content", "operative_brief", "step_refs"]
    assert kinds["kinds"]["agent"]["required"] == [
        "agent_name",
        "summary",
        "input",
        "output",
        "authorship",
        "step_refs",
    ]
    assert kinds["kinds"]["agent"]["optional"] == ["defect_recorded"]
    assert "input" not in kinds["kinds"]["conclusion"]["required"]
    assert kinds["authorship"] == ["authored", "relayed", "executed"]
    _assert_code(
        lambda: AigToolSurface(_snapshot(["alpha"]), root=tmp_path, run_id="../x"),
        "invalid_run_id",
    )


def test_build_verbs_story_graph_and_submit(tmp_path: Path) -> None:
    surface, snapshot = _open(tmp_path, ["q", "plan", "work", "spare", "answer"])
    surface.add_block("task", _task([0]), block_id="task-1")
    created = surface.add_block(
        "orchestrator",
        _orch([1]),
        block_id="orch-1",
        from_refs=[_note("task-1")],
    )
    assert created == {"block_id": "orch-1"}
    surface.add_block("agent", _agent([2]), block_id="agent-1")
    surface.add_inheritance_note(
        "orch-1", "agent-1", "the plan", "the worker followed the bad plan"
    )
    surface.add_block("agent", _agent([3]), block_id="spare-1")
    surface.add_inheritance_note("agent-1", "spare-1", "the draft", "the spare copied it")
    surface.add_block("conclusion", _end([4]), block_id="end-1")
    surface.add_skip_attribution(
        "task-1", "agent-1", "the question", "the worker skipped the route"
    )
    surface.modify_block("agent-1", {"summary": "worked on the wrong question"})
    marked = surface.mark_defect_recorded("agent-1", "output missed the operative brief")
    assert marked["defect_recorded"] == "output missed the operative brief"
    surface.delete_inheritance_note("agent-1", "spare-1")
    deleted = surface.delete_block("spare-1")
    assert deleted == {"block_id": "spare-1"}
    surface.delete_skip_attribution("task-1", "agent-1")
    surface.add_skip_attribution(
        "task-1", "agent-1", "the question", "the worker skipped the route"
    )
    surface.add_skip_attribution(
        "agent-1", "end-1", "the wrong answer", "the conclusion published it"
    )
    graph = surface.get_graph()
    assert [block["block_id"] for block in graph["blocks"]] == [
        "task-1",
        "orch-1",
        "agent-1",
        "end-1",
    ]
    assert graph["submitted"] is False
    agent = next(block for block in graph["blocks"] if block["block_id"] == "agent-1")
    assert agent["summary"] == "worked on the wrong question"
    assert agent["defect_recorded"] == "output missed the operative brief"
    assert {edge["link_kind"] for edge in graph["edges"]} <= {"adjacent", "skip"}
    assert any(edge["from"] == "task-1" and edge["to"] == "orch-1" for edge in graph["edges"])
    story = surface.get_story()
    assert "the question" in story
    assert "the conclusion published it" in story
    assert "`agent-1`" in story
    assert "submitted: false" in story
    json.dumps(graph)
    sealed = surface.submit()
    assert sealed == {"submitted": True}
    assert surface.get_graph()["submitted"] is True
    assert "submitted: true" in surface.get_story()
    assert surface.draft_log_path.stat().st_mode & 0o777 == 0o444
    assert surface.run_directory.stat().st_mode & 0o777 == 0o555
    _assert_code(surface.submit, "already_submitted")
    _assert_code(
        lambda: surface.add_block("task", _task([0]), block_id="task-2"),
        "already_submitted",
    )
    reopened = AigToolSurface(snapshot, root=tmp_path, run_id="run-1")
    assert reopened.get_graph() == surface.get_graph()
    assert reopened.read_step(0)["index"] == 0
    _assert_code(
        lambda: reopened.add_block("task", _task([0]), block_id="task-9"),
        "already_submitted",
    )


def test_submit_does_not_require_root_or_sink(tmp_path: Path) -> None:
    surface, _snapshot = _open(tmp_path, ["only"])
    surface.add_block("agent", _agent([0]), block_id="agent-1")
    surface.submit()
    graph = surface.get_graph()
    assert graph["submitted"] is True
    assert [block["kind"] for block in graph["blocks"]] == ["agent"]
    assert graph["edges"] == []


def test_backward_edge_reports_both_spans_without_mutation(tmp_path: Path) -> None:
    surface, snapshot = _populate(tmp_path)
    before = surface.get_graph()
    with pytest.raises(ToolSurfaceError) as caught:
        surface.add_inheritance_note("end-1", "task-1", "later", "backwards")
    assert caught.value.code == "edge_not_forward"
    detail = caught.value.detail
    assert detail["source"] == {"block_id": "end-1", "span": [3, 3]}
    assert detail["target"] == {"block_id": "task-1", "span": [0, 0]}
    assert "source.span[0] < target.span[0]" in detail["required_order"]
    assert "effect" in detail["hint"]
    assert surface.get_graph() == before
    reopened = AigToolSurface(snapshot, root=tmp_path, run_id="run-1")
    assert reopened.get_graph() == before


def test_hard_invariants_reject_without_mutation(tmp_path: Path) -> None:
    surface, _snapshot = _populate(tmp_path)
    before = surface.get_graph()
    lines = _log_text(surface)
    count = surface.operation_count
    cases = [
        ("duplicate_task", lambda: surface.add_block("task", _task([0]), block_id="task-2")),
        (
            "duplicate_conclusion",
            lambda: surface.add_block("conclusion", _end([3]), block_id="end-2"),
        ),
        ("step_out_of_range", lambda: surface.add_block("agent", _agent([99]), block_id="agent-x")),
        ("step_refs_overlap", lambda: surface.add_block("agent", _agent([2]), block_id="agent-x")),
        (
            "step_refs_not_contiguous",
            lambda: surface.add_block("agent", _agent([0, 2]), block_id="agent-x"),
        ),
        ("step_refs_empty", lambda: surface.add_block("agent", _agent([]), block_id="agent-x")),
        (
            "invalid_step_refs",
            lambda: surface.add_block("agent", _agent([True]), block_id="agent-x"),
        ),
        (
            "edge_not_forward",
            lambda: surface.add_inheritance_note("end-1", "task-1", "later", "backwards"),
        ),
        (
            "not_adjacent",
            lambda: surface.add_inheritance_note("task-1", "end-1", "the question", "skipped"),
        ),
        (
            "duplicate_edge",
            lambda: surface.add_inheritance_note("task-1", "orch-1", "again", "again"),
        ),
        ("kind_immutable", lambda: surface.modify_block("task-1", {"kind": "agent"})),
        ("id_immutable", lambda: surface.modify_block("task-1", {"block_id": "other"})),
        ("unknown_block", lambda: surface.modify_block("missing", {"content": "x"})),
        ("unknown_field", lambda: surface.modify_block("task-1", {"foo": "bar"})),
        ("invalid_authorship", lambda: surface.modify_block("agent-1", {"authorship": "written"})),
        ("missing_field", lambda: surface.modify_block("task-1", {"content": ""})),
        (
            "missing_field",
            lambda: surface.add_block(
                "agent",
                {
                    "agent_name": "w",
                    "summary": "s",
                    "input": "i",
                    "authorship": "authored",
                    "step_refs": [0],
                },
                block_id="agent-x",
            ),
        ),
        ("edge_kind_mismatch", lambda: surface.delete_inheritance_note("task-1", "agent-1")),
        ("edge_kind_mismatch", lambda: surface.delete_skip_attribution("task-1", "orch-1")),
        ("edge_missing", lambda: surface.delete_inheritance_note("task-1", "end-1")),
        ("defect_not_agent", lambda: surface.mark_defect_recorded("task-1", "missed")),
        ("missing_note", lambda: surface.mark_defect_recorded("agent-1", "  ")),
        ("missing_note", lambda: surface.add_inheritance_note("orch-1", "end-1", "", "effect")),
        ("invalid_kind", lambda: surface.add_block("nope", _task([0]), block_id="task-x")),
        ("invalid_block_id", lambda: surface.add_block("agent", _agent([0]), block_id="bad id")),
        ("duplicate_block_id", lambda: surface.add_block("agent", _agent([0]), block_id="task-1")),
        (
            "unknown_block",
            lambda: surface.add_block(
                "agent",
                _agent([0]),
                block_id="agent-x",
                from_refs=[_note("missing")],
            ),
        ),
    ]
    for code, action in cases:
        _assert_code(action, code)
        assert surface.get_graph() == before
        assert _log_text(surface) == lines
        assert surface.operation_count == count


def test_not_skip_and_modify_link_conflict(tmp_path: Path) -> None:
    surface, _snapshot = _open(tmp_path, ["q", "a", "c", "tail"])
    surface.add_block("task", _task([0]), block_id="task-1")
    surface.add_block("conclusion", _end([1]), block_id="end-1")
    _assert_code(
        lambda: surface.add_skip_attribution("task-1", "end-1", "the question", "too close"),
        "not_skip",
    )
    moved, _snapshot = _open(tmp_path, ["q", "b", "gap", "c", "later"], run_id="run-move")
    moved.add_block("task", _task([0]), block_id="a")
    moved.add_block("agent", _agent([1]), block_id="b")
    moved.add_block("conclusion", _end([3]), block_id="c")
    moved.add_inheritance_note("a", "b", "the question", "b reused it badly")
    before = moved.get_graph()
    _assert_code(lambda: moved.modify_block("b", {"step_refs": [5]}), "step_out_of_range")
    _assert_code(lambda: moved.modify_block("b", {"step_refs": [4]}), "link_kind_conflict")
    assert moved.get_graph() == before


def test_replay_is_idempotent_and_corruption_fails(tmp_path: Path) -> None:
    surface, snapshot = _populate(tmp_path)
    first = surface.get_graph()
    lines = _log_text(surface)
    assert replay_toolloop_draft(tmp_path, "run-1", snapshot) == first
    again = AigToolSurface(snapshot, root=tmp_path, run_id="run-1")
    assert again.get_graph() == first
    assert _log_text(again) == lines
    op = read_toolloop_draft_op(json.loads(lines.splitlines()[0]))
    assert op.verb == "add_block"
    assert op.seq == 0
    assert read_protocol_document(json.loads(lines.splitlines()[0])).verb == "add_block"
    tampered = lines
    payload = json.loads(tampered.splitlines()[0])
    payload["content_hash"] = "0" * 64
    broken = json.dumps(payload) + "\n" + "\n".join(tampered.splitlines()[1:]) + "\n"
    _rewrite(surface.draft_log_path, broken)
    with pytest.raises(ToolSurfaceError) as caught:
        AigToolSurface(snapshot, root=tmp_path, run_id="run-1")
    assert caught.value.code == "stored_record_invalid"
    other = _snapshot(["q", "plan", "work", "answer"], watermark="wm-other")
    _restore(surface.draft_log_path, lines)
    with pytest.raises(ToolSurfaceError) as mismatch:
        AigToolSurface(other, root=tmp_path, run_id="run-1")
    assert mismatch.value.code == "snapshot_mismatch"


def test_replay_rejects_a_broken_sequence(tmp_path: Path) -> None:
    surface, snapshot = _open(tmp_path, ["only"])
    surface.add_block("task", _task([0]), block_id="task-1")
    first = json.loads(_log_text(surface).splitlines()[0])
    extra = toolloop_draft_op_document(
        run_id="run-1",
        seq=5,
        verb="submit",
        args={},
        prev_content_hash=first["content_hash"],
    )
    _rewrite(surface.draft_log_path, _log_text(surface) + json.dumps(extra) + "\n")
    with pytest.raises(ToolSurfaceError) as caught:
        AigToolSurface(snapshot, root=tmp_path, run_id="run-1")
    assert caught.value.code == "draft_chain_broken"


def test_protocol_op_rejects_unknown_verb() -> None:
    with pytest.raises(ProtocolContractError):
        toolloop_draft_op_document(
            run_id="run-1",
            seq=0,
            verb="get_overview",
            args={},
            prev_content_hash="ab" * 32,
        )


def test_gap_requires_skip_attribution(tmp_path: Path) -> None:
    surface, snapshot = _open(tmp_path, ["a", "gap", "b"])
    surface.add_block("task", _task([0]), block_id="a")
    surface.add_block("conclusion", _end([2]), block_id="b")
    with pytest.raises(ToolSurfaceError) as caught:
        surface.add_inheritance_note("a", "b", "the prior result", "it jumped the gap")
    assert caught.value.code == "not_adjacent"
    assert caught.value.detail is not None
    assert caught.value.detail["source_id"] == "a"
    assert caught.value.detail["target_id"] == "b"
    assert caught.value.detail["skipped_rows"] == [1]
    assert "add_skip_attribution" in caught.value.detail["hint"]
    surface.add_skip_attribution("a", "b", "the prior result", "it jumped the gap")
    graph = surface.get_graph()
    assert graph["edges"][0]["link_kind"] == "skip"
    surface.submit()
    compiled = compile_paper_graph(
        surface.get_graph(),
        snapshot,
        root=tmp_path,
        run_id="run-gap-compile",
        session_id="sess-tool",
        task_run_id="run-gap-compile",
        view_id="view-run-gap",
        revision_id="rev-run-gap",
        frozen_at="2026-10-11T00:00:00.000Z",
    )
    edges = {
        (edge.source_block_id, edge.target_block_id): edge
        for edge in compiled.revision.edges
    }
    assert edges[("a", "b")].relation_type == "skip"


def test_contiguous_spans_remain_adjacent(tmp_path: Path) -> None:
    surface, _snapshot = _open(tmp_path, ["a", "b"])
    surface.add_block("task", _task([0]), block_id="a")
    surface.add_block("conclusion", _end([1]), block_id="b")
    surface.add_inheritance_note("a", "b", "the prior result", "the next used it")
    graph = surface.get_graph()
    assert graph["edges"][0]["link_kind"] == "adjacent"
    _assert_code(
        lambda: surface.add_skip_attribution("a", "b", "the prior result", "too close"),
        "duplicate_edge",
    )


def test_link_kind_conflict_reports_skipped_rows(tmp_path: Path) -> None:
    surface, _snapshot = _open(tmp_path, ["a", "mid", "b"])
    surface.add_block("task", _task([0]), block_id="a")
    surface.add_block("conclusion", _end([1]), block_id="b")
    surface.add_inheritance_note("a", "b", "the prior result", "the next used it")
    with pytest.raises(ToolSurfaceError) as caught:
        surface.modify_block("b", {"step_refs": [2]})
    assert caught.value.code == "link_kind_conflict"
    assert caught.value.detail is not None
    assert caught.value.detail["skipped_rows"] == [1]
    assert caught.value.detail["source_id"] == "a"
    assert caught.value.detail["target_id"] == "b"


def test_participant_mismatch_offers_modify_repair(tmp_path: Path) -> None:
    surface, _snapshot = _open(
        tmp_path,
        [],
        records=(_record(0, "row", participant="task-agent"),),
    )
    fields = _agent([0], agent_name="task")
    with pytest.raises(ToolSurfaceError) as caught:
        surface.add_block("agent", fields, block_id="agent-1")
    assert caught.value.code == "participant_mismatch"
    detail = caught.value.detail
    assert detail is not None
    assert detail["expected"] == "task-agent"
    assert detail["got"] == "task"
    assert detail["block_id"] == "agent-1"
    assert detail["hint"] == "agent_name must equal the rows' participant exactly"
    repairs = detail["repairs"]
    assert repairs == [
        {
            "apply_with": "modify_block",
            "block_id": "agent-1",
            "fields": {"agent_name": "task-agent"},
        }
    ]
    surface.add_block(
        "agent", _agent([0], agent_name="task-agent"), block_id="agent-1"
    )
    with pytest.raises(ToolSurfaceError) as again:
        surface.modify_block("agent-1", {"agent_name": "task"})
    assert again.value.code == "participant_mismatch"
    repair = again.value.detail["repairs"][0]
    surface.modify_block(repair["block_id"], repair["fields"])
    agent = next(
        block for block in surface.get_graph()["blocks"] if block["block_id"] == "agent-1"
    )
    assert agent["agent_name"] == "task-agent"


def test_participant_mixed_rows_rejected(tmp_path: Path) -> None:
    surface, _snapshot = _open(
        tmp_path,
        [],
        records=(
            _record(0, "first", participant="agent-a"),
            _record(1, "second", participant="agent-b"),
        ),
    )
    with pytest.raises(ToolSurfaceError) as caught:
        surface.add_block("task", _task([0, 1]), block_id="task-1")
    assert caught.value.code == "participant_mixed_rows"
    assert caught.value.detail is not None
    assert caught.value.detail["block_id"] == "task-1"
    assert caught.value.detail["participants"] == {"agent-a": [0], "agent-b": [1]}


def _rewrite(path: Path, text: str) -> None:
    path.chmod(0o644)
    path.write_text(text, encoding="utf-8")


def _restore(path: Path, text: str) -> None:
    _rewrite(path, text)
    path.chmod(0o444)
