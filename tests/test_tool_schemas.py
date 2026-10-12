# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ADR-0045 Wave E3: typed tool schemas and repeat-call circuit breaker."""

from __future__ import annotations

from pathlib import Path

from benchmark.causal_memory.lenovo_gemma_provider import openai_tools_from_catalog
from openviking.session.aig_tool_surface import (
    AUTHORSHIP_VALUES,
    TOOL_CATALOG,
    TOOL_EXACT_ARGS,
    TOOL_SCHEMAS,
    _OPTIONAL_FIELDS,
    _REQUIRED_FIELDS,
    AoSnapshot,
    AigToolSurface,
    ToolSurfaceError,
)
from openviking.session.aig_toolloop_builder import (
    CONSTRUCTION_FALLBACK,
    FakeToolloopProvider,
    ToolInvocation,
    ToolloopTurn,
    run_toolloop,
)
from openviking.session.ao_ledger import AORecord

_SCHEMA_ALIASES = {
    "read_step": frozenset({"i"}),
    "read_step_range": frozenset({"a", "b"}),
}
_SCHEMA_REQUIRED: dict[str, list[str]] = {
    "add_block": ["kind", "fields"],
    "add_inheritance_note": ["from", "to", "inherited", "effect"],
    "add_skip_attribution": ["from", "to", "inherited", "effect"],
    "delete_block": ["block_id"],
    "delete_inheritance_note": ["from", "to"],
    "delete_skip_attribution": ["from", "to"],
    "get_graph": [],
    "get_overview": [],
    "get_question": [],
    "get_story": [],
    "list_agents": [],
    "list_block_kinds": [],
    "mark_defect_recorded": ["block_id", "note"],
    "modify_block": ["block_id", "fields"],
    "read_full_log": [],
    "read_step": ["index"],
    "read_step_range": ["start", "end"],
    "search": ["pattern"],
    "submit": [],
}


def _record(index: int, text: str) -> AORecord:
    return AORecord(
        ao_id=f"ao-{index}",
        session_id="sess-schema",
        sequence=index + 1,
        action={"participant_id": "worker", "segment_id": "seg", "text": text},
        observation={"text": text},
        skill_invocations=[],
        message_ref=None,
        created_at="2026-10-12T00:00:00.000Z",
        captured_state="live",
    )


def _snapshot() -> AoSnapshot:
    return AoSnapshot(
        records=(_record(0, "ask"), _record(1, "answer")),
        watermark="wm-schema",
        question=None,
    )


def _walk_objects(node: object) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            found.append(node)
        for value in node.values():
            found.extend(_walk_objects(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_walk_objects(item))
    return found


def test_schema_keys_match_exact_args_and_required_partition() -> None:
    catalog = [name for name, _text in TOOL_CATALOG]
    assert len(catalog) == 19
    assert set(TOOL_SCHEMAS) == set(catalog) == set(TOOL_EXACT_ARGS)
    for verb in catalog:
        allowed, _server_required = TOOL_EXACT_ARGS[verb]
        schema = TOOL_SCHEMAS[verb]
        advertised = set(allowed) - _SCHEMA_ALIASES.get(verb, frozenset())
        assert set(schema["properties"]) == advertised
        assert schema["required"] == _SCHEMA_REQUIRED[verb]
        assert schema["additionalProperties"] is False
        assert schema["type"] == "object"


def test_add_block_fields_cover_required_and_kind_enum() -> None:
    fields = TOOL_SCHEMAS["add_block"]["properties"]["fields"]
    needed: set[str] = set()
    for keys in _REQUIRED_FIELDS.values():
        needed.update(keys)
    needed.update(_OPTIONAL_FIELDS.get("agent", ()))
    assert needed <= set(fields["properties"])
    assert set(TOOL_SCHEMAS["add_block"]["properties"]["kind"]["enum"]) == set(_REQUIRED_FIELDS)
    assert fields["properties"]["authorship"]["enum"] == list(AUTHORSHIP_VALUES)
    assert fields["properties"]["step_refs"]["items"]["type"] == "integer"
    assert fields["properties"]["defect_recorded"]["type"] == "string"
    assert fields["additionalProperties"] is False
    modify_fields = TOOL_SCHEMAS["modify_block"]["properties"]["fields"]
    assert set(modify_fields["properties"]) == set(fields["properties"])


def test_openai_tools_from_catalog_are_fully_typed() -> None:
    tools = openai_tools_from_catalog()
    names = [item["function"]["name"] for item in tools]
    assert names == [verb for verb, _text in TOOL_CATALOG]
    assert len(tools) == 19
    for item in tools:
        parameters = item["function"]["parameters"]
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False
        assert "properties" in parameters
        assert "required" in parameters
        verb = item["function"]["name"]
        assert parameters["required"] == TOOL_SCHEMAS[verb]["required"]
        if "kind" in parameters["properties"]:
            assert parameters["properties"]["kind"]["enum"] == list(
                TOOL_SCHEMAS[verb]["properties"]["kind"]["enum"]
            )
        for obj in _walk_objects(parameters):
            assert "properties" in obj, f"{verb} has a bare object parameter"
            assert obj["additionalProperties"] is False


def test_repeat_call_rejects_third_identical_without_mutating(
    tmp_path: Path,
) -> None:
    script = [
        ("get_overview", {}),
        ("get_overview", {}),
        ("get_overview", {}),
        ("add_block", {"kind": "task", "block_id": "task-1", "fields": _task([0])}),
        ("add_block", {"kind": "conclusion", "block_id": "end-1", "fields": _end([1])}),
        ("submit", {}),
    ]
    provider = FakeToolloopProvider(script)
    result = run_toolloop(
        _snapshot(),
        "what failed?",
        provider,
        root=tmp_path,
        run_id="run-trip",
    )
    assert result.status == "submitted"
    third = result.audit[2]
    assert third.verb == "get_overview"
    assert third.ok is False
    assert third.error_code == "repeat_call"
    assert third.repeat_count == 3
    history = provider.turns[3].history[2]
    assert history["error_code"] == "repeat_call"
    assert history["result"]["message"] == (
        "identical call repeated (3/5); take a different action — the run "
        "terminates at 5 identical calls"
    )
    assert [block["block_id"] for block in result.draft["blocks"]] == ["task-1", "end-1"]


def test_repeat_loop_stops_on_fifth_identical_call(tmp_path: Path) -> None:
    provider = FakeToolloopProvider([("get_overview", {})] * 5)
    result = run_toolloop(
        _snapshot(),
        "what failed?",
        provider,
        root=tmp_path,
        run_id="run-stop",
        max_calls=64,
    )
    assert result.status == "repeat_loop"
    assert result.construction_mode == CONSTRUCTION_FALLBACK
    assert result.draft is None
    assert result.fallback_marker is not None
    assert result.fallback_marker["reason"] == "repeat_loop"
    assert result.fallback_marker["repeat_count"] == 5
    assert result.fallback_marker["accepted_as_aig"] is False
    assert len(result.audit) == 5
    assert [item.repeat_count for item in result.audit] == [1, 2, 3, 4, 5]
    assert result.audit[0].ok and result.audit[1].ok
    assert all(item.error_code == "repeat_call" for item in result.audit[2:])


def test_interleaved_or_changed_args_do_not_trip(tmp_path: Path) -> None:
    script = [
        ("get_overview", {}),
        ("read_step", {"index": 0}),
        ("get_overview", {}),
        ("read_step", {"index": 1}),
        ("get_overview", {}),
        ("add_block", {"kind": "task", "block_id": "task-1", "fields": _task([0])}),
        ("add_block", {"kind": "conclusion", "block_id": "end-1", "fields": _end([1])}),
        ("submit", {}),
    ]
    provider = FakeToolloopProvider(script)
    result = run_toolloop(
        _snapshot(),
        "what failed?",
        provider,
        root=tmp_path,
        run_id="run-reset",
    )
    assert result.status == "submitted"
    assert all(item.error_code is None for item in result.audit)
    assert max(item.repeat_count for item in result.audit) == 1


def test_same_verb_different_args_resets_repeat_count(tmp_path: Path) -> None:
    class Scripted:
        def __init__(self) -> None:
            self.turns: list[ToolloopTurn] = []

        def __call__(self, turn: ToolloopTurn) -> ToolInvocation:
            self.turns.append(turn)
            step = len(turn.history)
            if step < 3:
                return ToolInvocation("read_step", {"index": step % 2})
            if step == 3:
                return ToolInvocation(
                    "add_block",
                    {"kind": "task", "block_id": "task-1", "fields": _task([0])},
                )
            if step == 4:
                return ToolInvocation(
                    "add_block",
                    {"kind": "conclusion", "block_id": "end-1", "fields": _end([1])},
                )
            return ToolInvocation("submit", {})

    result = run_toolloop(
        _snapshot(),
        "what failed?",
        Scripted(),
        root=tmp_path,
        run_id="run-args",
    )
    assert result.status == "submitted"
    assert all(item.error_code != "repeat_call" for item in result.audit)


def _task(steps: list[int]) -> dict[str, object]:
    return {"content": "the question", "operative_brief": "start here", "step_refs": steps}


def _end(steps: list[int]) -> dict[str, object]:
    return {"content": "the answer", "step_refs": steps}


def _surface(tmp_path: Path) -> AigToolSurface:
    return AigToolSurface(_snapshot(), root=tmp_path, run_id="schema-detail")


def test_step_refs_overlap_error_names_conflicting_blocks(tmp_path: Path) -> None:
    surface = _surface(tmp_path)
    surface.call(
        "add_block",
        {
            "kind": "task",
            "block_id": "t",
            "fields": {"content": "q", "operative_brief": "b", "step_refs": [0, 1]},
        },
    )
    try:
        surface.call(
            "add_block",
            {
                "kind": "conclusion",
                "block_id": "c",
                "fields": {"content": "done", "step_refs": [1]},
            },
        )
        raise AssertionError("expected step_refs_overlap")
    except ToolSurfaceError as info:
        assert info.code == "step_refs_overlap"
        assert info.detail is not None
        detail = info.detail
    assert detail["shared_rows"] == [1]
    assert {item["block_id"] for item in detail["blocks"]} == {"t", "c"}
    assert "disjoint" in detail["hint"]
    overview = surface.get_overview()
    assert overview["draft_blocks"] == [{"block_id": "t", "kind": "task", "span": [0, 1]}]


def test_overview_hides_draft_blocks_until_first_add(tmp_path: Path) -> None:
    surface = _surface(tmp_path)
    assert "draft_blocks" not in surface.get_overview()


def test_builder_history_carries_overlap_detail_to_provider(tmp_path: Path) -> None:
    class CapturingProvider:
        def __init__(self) -> None:
            self.turns: list[ToolloopTurn] = []
            self.script = [
                ToolInvocation(
                    "add_block",
                    {
                        "kind": "task",
                        "block_id": "t",
                        "fields": {
                            "content": "q",
                            "operative_brief": "b",
                            "step_refs": [0, 1],
                        },
                    },
                ),
                ToolInvocation(
                    "add_block",
                    {
                        "kind": "conclusion",
                        "block_id": "c",
                        "fields": {"content": "done", "step_refs": [1]},
                    },
                ),
                ToolInvocation("submit", {}),
            ]

        def __call__(self, turn: ToolloopTurn) -> ToolInvocation:
            self.turns.append(turn)
            return self.script[len(self.turns) - 1]

    provider = CapturingProvider()
    result = run_toolloop(
        _snapshot(),
        "what happened",
        provider,
        root=tmp_path,
        run_id="schema-history",
    )
    assert result.status == "submitted"
    rejected = provider.turns[-1].history[-1]
    assert rejected["verb"] == "add_block"
    assert rejected["ok"] is False
    assert rejected["error_code"] == "step_refs_overlap"
    assert isinstance(rejected["result"], dict)
    assert rejected["result"]["shared_rows"] == [1]


def test_step_refs_empty_error_teaches_the_shrink_move(tmp_path: Path) -> None:
    surface = _surface(tmp_path)
    surface.call(
        "add_block",
        {
            "kind": "task",
            "block_id": "t",
            "fields": {"content": "q", "operative_brief": "b", "step_refs": [0, 1]},
        },
    )
    try:
        surface.call(
            "modify_block",
            {"block_id": "t", "fields": {"content": "q", "step_refs": []}},
        )
        raise AssertionError("expected step_refs_empty")
    except ToolSurfaceError as info:
        assert info.code == "step_refs_empty"
        assert info.detail is not None
        assert "smaller" in info.detail["hint"]


def test_step_refs_overlap_offers_ready_repair(tmp_path: Path) -> None:
    snapshot = AoSnapshot(
        records=(_record(0, "ask"), _record(1, "open"), _record(2, "answer")),
        watermark="wm-schema-3",
        question=None,
    )
    surface = AigToolSurface(snapshot, root=tmp_path, run_id="schema-repair")
    surface.call(
        "add_block",
        {
            "kind": "task",
            "block_id": "t",
            "fields": {"content": "q", "operative_brief": "b", "step_refs": [0]},
        },
    )
    surface.call(
        "add_block",
        {
            "kind": "agent",
            "block_id": "a",
            "fields": {
                "agent_name": "w",
                "authorship": "authored",
                "input": "i",
                "output": "o",
                "summary": "s",
                "step_refs": [1, 2],
            },
        },
    )
    try:
        surface.call(
            "add_block",
            {
                "kind": "conclusion",
                "block_id": "c",
                "fields": {"content": "done", "step_refs": [2]},
            },
        )
        raise AssertionError("expected step_refs_overlap")
    except ToolSurfaceError as info:
        assert info.code == "step_refs_overlap"
        assert info.detail is not None
        detail = info.detail
    # a=[1,2] keeps row 1 when the shared row 2 is freed; c=[2] has nothing
    # left, so exactly one repair is offered.
    assert detail["repairs"] == [
        {
            "apply_with": "modify_block",
            "block_id": "a",
            "kind": "agent",
            "new_step_refs": [1],
        }
    ]
    surface.call(
        "modify_block", {"block_id": "a", "fields": {"step_refs": [1], "summary": "s"}}
    )
    surface.call(
        "add_block",
        {
            "kind": "conclusion",
            "block_id": "c",
            "fields": {"content": "done", "step_refs": [2]},
        },
    )
    overview = surface.get_overview()
    assert [item["block_id"] for item in overview["draft_blocks"]] == ["t", "a", "c"]
