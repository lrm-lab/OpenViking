# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Reader navigation: transfer rule, frozen report hash, and anchor candidates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openviking.session.aig_native_protocol import TOOLLOOP_BUILD_VERBS
from openviking.session.aig_role_chain import AigConstructionRun
from openviking.session.aig_tool_surface import AigToolSurface, AoSnapshot
from openviking.session.aig_toolloop_reader import (
    READER_SYSTEM_PROMPT,
    READER_VERBS,
    DeterministicReader,
    FakeReaderProvider,
    ReaderError,
    ReaderInvocation,
    ReaderView,
    anchor_candidates,
    load_frozen_paper_graph,
    reader_catalog,
    run_reader,
    transfer_responsibility,
)
from openviking.session.ao_ledger import AORecord
from openviking.session.paper_graph_compile import compile_paper_graph

_FROZEN_AT = "2026-10-11T00:00:00.000Z"
_EFFECT_C = "executed this flawed heuristic and produced 0"
_EFFECT_END = "accepted that 0 at face value"
_EFFECT_SKIP = "leaned on the prior zero framing to terminate"
_SESSION = "sess-reader"


def _record(index: int, participant: str, text: str) -> AORecord:
    return AORecord(
        ao_id=f"ao-{index}",
        session_id=_SESSION,
        sequence=index + 1,
        action={"participant_id": participant, "segment_id": "seg", "text": text},
        observation={"text": text},
        skill_invocations=[],
        message_ref=None,
        created_at=_FROZEN_AT,
        captured_state="live",
    )


def _compiled(tmp_path: Path, *, log_effect: bool = True):
    records = (
        _record(0, "Task_Framer", "count hep-lat articles"),
        _record(1, "Verification_Expert", "python script using entry_id"),
        _record(2, "Computer_terminal", _EFFECT_C if log_effect else "exitcode 0"),
        _record(3, "Closer", _EFFECT_END if log_effect else "final answer 0"),
    )
    snapshot = AoSnapshot(records, watermark="wm-reader", question="how many hep-lat articles")
    surface = AigToolSurface(snapshot, root=tmp_path, run_id="run-reader-draft")
    surface.add_block(
        "task",
        {"content": "count hep-lat articles", "operative_brief": "start here", "step_refs": [0]},
        block_id="task-1",
    )
    surface.add_block(
        "agent",
        {
            "agent_name": "Verification_Expert",
            "authorship": "authored",
            "input": "plan from step 0",
            "output": "python script using entry_id",
            "step_refs": [1],
            "summary": "proposed a flawed ps substring check",
        },
        block_id="agent-b",
    )
    surface.add_block(
        "agent",
        {
            "agent_name": "Computer_terminal",
            "authorship": "executed",
            "input": "the script",
            "output": f"exitcode 0; {_EFFECT_C}",
            "step_refs": [2],
            "summary": "ran the script and printed 0",
        },
        block_id="agent-c",
    )
    surface.add_block(
        "conclusion",
        {"content": f"Final answer 0. {_EFFECT_END}", "step_refs": [3]},
        block_id="end-1",
    )
    surface.add_inheritance_note(
        "task-1", "agent-b", "the question", "the plan dropped the format field"
    )
    surface.add_inheritance_note("agent-b", "agent-c", "the flawed script", _EFFECT_C)
    surface.add_inheritance_note("agent-c", "end-1", "the printed zero", _EFFECT_END)
    surface.add_skip_attribution("agent-b", "end-1", "the script", _EFFECT_SKIP)
    surface.mark_defect_recorded("agent-b", "ps detection used entry_id")
    surface.mark_defect_recorded("agent-c", "executed the flawed script unchanged")
    surface.submit()
    compiled = compile_paper_graph(
        surface.get_graph(),
        snapshot,
        root=tmp_path,
        run_id="run-reader",
        session_id=_SESSION,
        task_run_id="run-reader",
        view_id="view-run-reader",
        revision_id="rev-run-reader",
        frozen_at=_FROZEN_AT,
    )
    return surface, snapshot, compiled


def _open_run(surface: AigToolSurface, run_id: str = "run-reader") -> AigConstructionRun:
    run = AigConstructionRun(run_id)
    run.submit_draft(run.session_for("builder"), surface.get_graph())
    return run


def test_transfer_stops_at_the_block_whose_effect_the_log_confirms(tmp_path: Path) -> None:
    _surface, snapshot, compiled = _compiled(tmp_path)
    graph = load_frozen_paper_graph(compiled.revision, snapshot, sidecar_path=compiled.sidecar_path)
    report = transfer_responsibility(graph, snapshot)
    assert report.critical_block_id == "agent-b"
    assert report.responsible_agent == "Verification_Expert"
    assert report.decisive_ao_refs == (1,)
    assert report.unresolved_areas == ()
    assert "责任转移" in READER_SYSTEM_PROMPT
    assert anchor_candidates(graph, report) == ("agent-b", "agent-c")


def test_log_must_confirm_the_effect_before_responsibility_moves(tmp_path: Path) -> None:
    _surface, _snapshot, compiled = _compiled(tmp_path)
    quiet = AoSnapshot(
        (
            _record(0, "Task_Framer", "count hep-lat articles"),
            _record(1, "Verification_Expert", "python script using entry_id"),
            _record(2, "Computer_terminal", "exitcode 0"),
            _record(3, "Closer", "final answer 0"),
        ),
        watermark="wm-reader",
        question="how many hep-lat articles",
    )
    graph = load_frozen_paper_graph(
        compiled.revision, _snapshot, sidecar_path=compiled.sidecar_path
    )
    report = transfer_responsibility(graph, quiet)
    assert report.critical_block_id == "end-1"
    assert anchor_candidates(graph, report) == ("end-1", "agent-b", "agent-c")


def test_deterministic_reader_freezes_the_role_chain_hash(tmp_path: Path) -> None:
    surface, snapshot, compiled = _compiled(tmp_path)
    run = _open_run(surface)
    provider = DeterministicReader()
    result = run_reader(
        compiled.revision,
        snapshot,
        provider,
        root=tmp_path,
        run_id="run-reader",
        construction_run=run,
        sidecar_path=compiled.sidecar_path,
    )
    assert result.report.critical_block_id == "agent-b"
    assert result.anchor_candidates == ("agent-b", "agent-c")
    assert result.reader_report_hash == run.reader_report_hash
    assert run.phase == "SYNTHESIZE_PROPOSE"
    assert result.budget_exhausted is False
    assert provider.turns[0].system_prompt.startswith(READER_SYSTEM_PROMPT)
    assert {name for name, _text in provider.turns[0].tools} == READER_VERBS
    assert result.tool_calls >= 2


def test_budget_exhaustion_records_unresolved_and_still_freezes(tmp_path: Path) -> None:
    surface, snapshot, compiled = _compiled(tmp_path)
    run = _open_run(surface, "run-budget")
    result = run_reader(
        compiled.revision,
        snapshot,
        DeterministicReader(),
        root=tmp_path,
        run_id="run-budget",
        construction_run=run,
        sidecar_path=compiled.sidecar_path,
        max_calls=1,
    )
    assert result.budget_exhausted is True
    assert result.report.unresolved_areas == ("budget_exhausted",)
    assert result.report.critical_block_id == ""
    assert result.anchor_candidates == ("agent-b", "agent-c")
    assert result.reader_report_hash == run.reader_report_hash


def test_catalog_rejects_build_verbs_and_a_bad_sidecar(tmp_path: Path) -> None:
    surface, snapshot, compiled = _compiled(tmp_path)
    graph = load_frozen_paper_graph(compiled.revision, snapshot, sidecar_path=compiled.sidecar_path)
    view = ReaderView(snapshot, graph, root=tmp_path, run_id="run-view")
    names = {name for name, _text in reader_catalog()}
    assert names == READER_VERBS
    assert names.isdisjoint(TOOLLOOP_BUILD_VERBS)
    with pytest.raises(ReaderError) as caught:
        view.call("add_block", {"kind": "task", "fields": {}})
    assert caught.value.code == "build_verb_forbidden"
    bogus = tmp_path / "bogus.json"
    bogus.write_text("{}", encoding="utf-8")
    with pytest.raises(ReaderError) as missing:
        load_frozen_paper_graph(compiled.revision, snapshot, sidecar_path=bogus)
    assert missing.value.code == "sidecar_mismatch"
    with pytest.raises(ReaderError) as required:
        load_frozen_paper_graph(compiled.revision, snapshot)
    assert required.value.code == "sidecar_required"


def test_fake_provider_report_must_match_the_block(tmp_path: Path) -> None:
    surface, snapshot, compiled = _compiled(tmp_path)
    graph = load_frozen_paper_graph(compiled.revision, snapshot, sidecar_path=compiled.sidecar_path)
    body = {
        "critical_block_id": "agent-b",
        "decisive_ao_refs": [1],
        "reason": "the script introduced the bad heuristic",
        "responsible_agent": "Verification_Expert",
        "unresolved_areas": ["ps field"],
    }
    run = _open_run(surface, "run-fake")
    result = run_reader(
        compiled.revision,
        snapshot,
        FakeReaderProvider(
            [
                ReaderInvocation("tool", "read_step", {"index": 1}),
                ReaderInvocation("report", report=body),
            ]
        ),
        root=tmp_path,
        run_id="run-fake",
        construction_run=run,
        sidecar_path=compiled.sidecar_path,
    )
    assert result.report.unresolved_areas == ("ps field",)
    assert json.loads(json.dumps(result.report.to_dict()))["critical_block_id"] == "agent-b"
    refused = FakeReaderProvider(
        [ReaderInvocation("report", report={**body, "responsible_agent": "other"})]
    )
    other = _open_run(surface, "run-bad")
    with pytest.raises(ReaderError) as caught:
        run_reader(
            compiled.revision,
            snapshot,
            refused,
            root=tmp_path,
            run_id="run-bad-log",
            construction_run=other,
            sidecar_path=compiled.sidecar_path,
        )
    assert caught.value.code == "invalid_reader_output"
    assert anchor_candidates(graph, result.report)[0] == "agent-b"


def test_reader_deduplicates_repeated_evidence_and_requests_final_report(tmp_path: Path) -> None:
    surface, snapshot, compiled = _compiled(tmp_path)
    body = {
        "critical_block_id": "agent-b",
        "decisive_ao_refs": [1],
        "reason": "read AO 1",
        "responsible_agent": "Verification_Expert",
        "unresolved_areas": [],
    }
    provider = FakeReaderProvider(
        [
            ReaderInvocation("tool", "read_step", {"index": 1}),
            ReaderInvocation("tool", "read_step", {"index": 1}),
            ReaderInvocation("report", report=body),
        ]
    )
    result = run_reader(
        compiled.revision,
        snapshot,
        provider,
        root=tmp_path,
        run_id="dedup",
        construction_run=_open_run(surface, "dedup"),
        sidecar_path=compiled.sidecar_path,
        reader_policy="guarded",
    )
    assert result.report.decisive_ao_refs == (1,)
    assert provider.turns[-1].read_ao_refs == (1,)
    assert "record" in provider.turns[-1].history[0]["result"]
    assert "record" not in provider.turns[-1].history[1]["result"]
    assert provider.turns[-1].history[1]["result"]["duplicate_of"] == 0


def test_guarded_reader_rejects_graph_only_citations(tmp_path: Path) -> None:
    surface, snapshot, compiled = _compiled(tmp_path)
    provider = FakeReaderProvider(
        [
            ReaderInvocation(
                "report",
                report={
                    "critical_block_id": "agent-b",
                    "decisive_ao_refs": [1],
                    "reason": "guess",
                    "responsible_agent": "Verification_Expert",
                    "unresolved_areas": [],
                },
            )
        ]
    )
    with pytest.raises(ReaderError, match="unread_decisive_refs"):
        run_reader(
            compiled.revision,
            snapshot,
            provider,
            root=tmp_path,
            run_id="unread",
            construction_run=_open_run(surface, "unread"),
            sidecar_path=compiled.sidecar_path,
            reader_policy="guarded",
        )


def test_guarded_reader_stops_stalled_reads_with_final_opportunity(tmp_path: Path) -> None:
    surface, snapshot, compiled = _compiled(tmp_path)
    provider = FakeReaderProvider([ReaderInvocation("tool", "read_step", {"index": 1})] * 16)
    result = run_reader(
        compiled.revision,
        snapshot,
        provider,
        root=tmp_path,
        run_id="stall",
        construction_run=_open_run(surface, "stall"),
        sidecar_path=compiled.sidecar_path,
        reader_policy="guarded",
    )
    assert result.tool_calls == 4
    assert provider.turns[-1].final_only is True
    assert result.stop_reason == "no_progress"
    assert result.report.unresolved_areas == ("no_progress",)
    assert len(result.reads) == 4


def test_guarded_reader_can_declare_unresolved_without_inventing_citations(tmp_path: Path) -> None:
    surface, snapshot, compiled = _compiled(tmp_path)
    provider = FakeReaderProvider(
        [
            ReaderInvocation(
                "report",
                report={
                    "critical_block_id": "",
                    "decisive_ao_refs": [],
                    "responsible_agent": "",
                    "reason": "No evidence read",
                    "unresolved_areas": ["need to verify effects"],
                },
            )
        ]
    )
    result = run_reader(
        compiled.revision,
        snapshot,
        provider,
        root=tmp_path,
        run_id="unresolved",
        construction_run=_open_run(surface, "unresolved"),
        sidecar_path=compiled.sidecar_path,
        reader_policy="guarded",
    )
    assert result.report.decisive_ao_refs == ()
    assert result.report.unresolved_areas == ("need to verify effects",)
    assert result.read_ao_refs == ()


def _guard_report():
    return {
        "critical_block_id": "agent-b",
        "decisive_ao_refs": [1],
        "reason": "AO 1 contains the proposed script",
        "responsible_agent": "Verification_Expert",
        "unresolved_areas": [],
    }


def test_guarded_range_read_establishes_evidence_coverage(tmp_path: Path) -> None:
    surface, snapshot, compiled = _compiled(tmp_path)
    provider = FakeReaderProvider(
        [
            ReaderInvocation("tool", "read_step_range", {"start": 0, "end": 2}),
            ReaderInvocation("report", report=_guard_report()),
        ]
    )
    result = run_reader(
        compiled.revision,
        snapshot,
        provider,
        root=tmp_path,
        run_id="range-coverage",
        construction_run=_open_run(surface, "range-coverage"),
        sidecar_path=compiled.sidecar_path,
    )
    assert result.read_ao_refs == (0, 1, 2)
    assert result.report.decisive_ao_refs == (1,)


def test_search_results_do_not_establish_decisive_evidence(tmp_path: Path) -> None:
    surface, snapshot, compiled = _compiled(tmp_path)
    provider = FakeReaderProvider(
        [
            ReaderInvocation("tool", "search", {"pattern": "python"}),
            ReaderInvocation("report", report=_guard_report()),
        ]
    )
    with pytest.raises(ReaderError, match="unread_decisive_refs"):
        run_reader(
            compiled.revision,
            snapshot,
            provider,
            root=tmp_path,
            run_id="search-coverage",
            construction_run=_open_run(surface, "search-coverage"),
            sidecar_path=compiled.sidecar_path,
        )
    assert provider.turns[-1].history[0]["ok"] is True
    assert provider.turns[-1].read_ao_refs == ()


def test_final_budget_slot_accepts_report_from_previously_read_evidence(tmp_path: Path) -> None:
    surface, snapshot, compiled = _compiled(tmp_path)
    provider = FakeReaderProvider(
        [
            *(ReaderInvocation("tool", "read_step", {"index": i}) for i in [1, 2, 3]),
            ReaderInvocation("report", report=_guard_report()),
        ]
    )
    result = run_reader(
        compiled.revision,
        snapshot,
        provider,
        root=tmp_path,
        run_id="final-slot",
        construction_run=_open_run(surface, "final-slot"),
        sidecar_path=compiled.sidecar_path,
        max_calls=4,
    )
    assert result.tool_calls == 3
    assert result.budget_exhausted is False
    assert provider.turns[-1].remaining_calls == 1
    assert provider.turns[-1].final_only is True
    assert result.report.decisive_ao_refs == (1,)


def test_failed_reads_stop_without_polluting_evidence_coverage(tmp_path: Path) -> None:
    surface, snapshot, compiled = _compiled(tmp_path)
    provider = FakeReaderProvider([ReaderInvocation("tool", "read_step", {"index": 999})] * 16)
    result = run_reader(
        compiled.revision,
        snapshot,
        provider,
        root=tmp_path,
        run_id="failed-reads",
        construction_run=_open_run(surface, "failed-reads"),
        sidecar_path=compiled.sidecar_path,
    )
    assert result.tool_calls == 3
    assert result.read_ao_refs == ()
    assert all(not row["ok"] for row in result.reads)
    assert result.stop_reason == "no_progress"
    assert result.report.unresolved_areas == ("no_progress",)
