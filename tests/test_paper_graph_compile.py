# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Paper-graph compile and the critic/refiner loop around it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from openviking.session.aig_native_protocol import canonical_dumps, read_native_edge_revision
from openviking.session.aig_role_chain import AigConstructionRun
from openviking.session.aig_tool_surface import AigToolSurface, AoSnapshot
from openviking.session.aig_toolloop_critic import FakeCriticProvider
from openviking.session.aig_toolloop_orchestrator import run_critic_refiner
from openviking.session.aig_toolloop_refiner import FakeRefinerProvider, RefinerAction
from openviking.session.ao_ledger import AORecord
from openviking.session.influence_view import CONSTRUCTION_MODE_ADAPTIVE_AGENTIC
from openviking.session.paper_graph_compile import (
    PaperCompileError,
    compile_paper_graph,
    join_mechanism,
    render_paper_graph,
    sidecar_digest,
)

_FROZEN_AT = "2026-10-11T00:00:00.000Z"
_DEFECT = "the script never checked a format field"


def _record(
    index: int,
    text: str,
    *,
    participant: str,
    session_id: str,
) -> AORecord:
    return AORecord(
        ao_id=f"ao-{index}",
        session_id=session_id,
        sequence=index + 1,
        action={"participant_id": participant, "segment_id": "seg", "text": text},
        observation={"text": text},
        skill_invocations=[],
        message_ref=None,
        created_at=_FROZEN_AT,
        captured_state="live",
    )


def _legal_surface(tmp_path: Path, run_id: str) -> tuple[AigToolSurface, AoSnapshot]:
    records = tuple(
        _record(index, f"row {index}", participant="worker", session_id="sess-loop")
        for index in range(3)
    )
    snapshot = AoSnapshot(records, watermark="wm-loop", question="what failed?")
    surface = AigToolSurface(snapshot, root=tmp_path, run_id=run_id)
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
    return surface, snapshot


def _violation(severity: str) -> dict[str, object]:
    return {
        "evidence_ao_refs": [1],
        "kind": "semantic",
        "note": "summary contradicts the row",
        "object_id": "agent-1",
        "severity": severity,
    }


def _chain(path: Path) -> list[dict]:
    rows: list[dict] = []
    previous = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        body = {key: value for key, value in row.items() if key != "content_hash"}
        digest = hashlib.sha256(canonical_dumps(body).encode("utf-8")).hexdigest()
        assert row["content_hash"] == digest
        assert row["prev_content_hash"] == previous
        previous = digest
        rows.append(row)
    return rows


def test_empty_set_compiles_without_entering_the_loop(tmp_path: Path) -> None:
    surface, snapshot = _legal_surface(tmp_path, "run-early")
    critic = FakeCriticProvider([{"violations": []}])
    refiner = FakeRefinerProvider()
    result = run_critic_refiner(
        surface,
        snapshot,
        critic=critic,
        refiner=refiner,
        root=tmp_path,
        run_id="run-early",
        frozen_at=_FROZEN_AT,
    )
    assert result.entered_loop is False
    assert result.revision_rounds == 0
    assert result.status == "frozen"
    assert result.run.critic_rounds == 0
    assert result.run.refiner_rounds == 0
    assert refiner.turns == []
    assert len(critic.turns) == 1
    assert result.revision is not None
    assert result.revision.construction_mode == CONSTRUCTION_MODE_ADAPTIVE_AGENTIC
    assert "known_violations" not in result.revision.protocol_body()
    assert result.fallback_marker is None
    rows = _chain(result.trajectory_path)
    assert [row["record_kind"] for row in rows] == ["phi", "critic", "decision"]
    assert result.run.submit_draft(result.run.session_for("builder"), surface.get_graph())
    assert result.run.critique(result.run.session_for("critic"), []) == 1


def test_high_severity_exhausts_rounds_into_fallback(tmp_path: Path) -> None:
    surface, snapshot = _legal_surface(tmp_path, "run-high")
    critic = FakeCriticProvider([{"violations": [_violation("high")]}])
    refiner = FakeRefinerProvider()
    result = run_critic_refiner(
        surface,
        snapshot,
        critic=critic,
        refiner=refiner,
        root=tmp_path,
        run_id="run-high",
        frozen_at=_FROZEN_AT,
    )
    assert result.status == "fallback"
    assert result.entered_loop is True
    assert result.revision_rounds == 3
    assert result.revision is None
    assert result.sidecar_path is None
    assert result.run.critic_rounds == 3
    assert result.run.refiner_rounds == 3
    assert result.run.construction_mode == "fallback"
    assert result.run.trust == "provisional"
    assert result.fallback_marker is not None
    assert result.fallback_marker["accepted_as_aig"] is False
    assert result.fallback_marker["deterministic_view"] == "build_minimal_fallback_view"
    assert result.fallback_marker["reason"] == "high_severity_violations"
    assert len(critic.turns) == 4
    rows = _chain(result.trajectory_path)
    assert sum(row["record_kind"] == "refine_round" for row in rows) == 3
    assert rows[-1]["body"]["status"] == "fallback"
    sidecar = tmp_path / "toolloop-audit" / "run-high" / "rev-run-high.paper-graph.json"
    assert not sidecar.exists()


def test_low_severity_freezes_with_known_violations(tmp_path: Path) -> None:
    surface, snapshot = _legal_surface(tmp_path, "run-low")
    critic = FakeCriticProvider([{"violations": [_violation("low")]}])
    refiner = FakeRefinerProvider()
    result = run_critic_refiner(
        surface,
        snapshot,
        critic=critic,
        refiner=refiner,
        root=tmp_path,
        run_id="run-low",
        frozen_at=_FROZEN_AT,
    )
    assert result.status == "frozen"
    assert result.entered_loop is True
    assert result.revision_rounds == 3
    assert result.revision is not None
    assert result.revision.construction_mode == CONSTRUCTION_MODE_ADAPTIVE_AGENTIC
    assert result.run.construction_mode == "native_single_pass_aig"
    assert result.run.trust == "draft"
    assert result.known_violations
    assert all(item["severity"] == "low" for item in result.known_violations)
    body = result.revision.protocol_body()
    assert body["known_violations"][0]["violation_id"] == "critic:0:semantic:agent-1"
    assert body["paper_sidecar_hash"] == result.sidecar_hash
    parsed = read_native_edge_revision(result.revision.protocol_document())
    assert parsed.revision.construction_mode == CONSTRUCTION_MODE_ADAPTIVE_AGENTIC
    assert parsed.revision.known_violations[0]["severity"] == "low"
    rows = _chain(result.trajectory_path)
    assert sum(row["record_kind"] == "refine_round" for row in rows) == 3
    assert rows[-1]["body"]["status"] == "frozen"


def test_all_unresolvable_stops_before_another_round(tmp_path: Path) -> None:
    surface, snapshot = _legal_surface(tmp_path, "run-stuck")
    critic = FakeCriticProvider([{"violations": [_violation("high")]}])
    refiner = FakeRefinerProvider(
        [
            RefinerAction(
                kind="unresolvable",
                violation_id="critic:0:semantic:agent-1",
                reason="the row really says this",
            )
        ]
    )
    result = run_critic_refiner(
        surface,
        snapshot,
        critic=critic,
        refiner=refiner,
        root=tmp_path,
        run_id="run-stuck",
        frozen_at=_FROZEN_AT,
    )
    assert result.status == "fallback"
    assert result.revision_rounds == 1
    assert result.revision is None
    assert len(critic.turns) == 1
    assert result.run.critic_rounds == 1
    rows = _chain(result.trajectory_path)
    assert sum(row["record_kind"] == "refine_round" for row in rows) == 1


def _appendix_graph() -> dict:
    return {
        "blocks": [
            {
                "block_id": "A",
                "kind": "agent",
                "agent_name": "Data_Extraction_Expert",
                "summary": "Restated the manager task to count hep-lat articles.",
                "input": "Count hep-lat Arxiv articles from January 2020.",
                "output": "Carried forward a prior result of 0.",
                "authorship": "relayed",
                "step_refs": [0],
            },
            {
                "block_id": "B",
                "kind": "agent",
                "agent_name": "Verification_Expert",
                "summary": "Proposed a script that treats ps as a substring of entry_id.",
                "input": "Plan from step 0.",
                "output": "Python code that counts ps via substring.",
                "authorship": "authored",
                "step_refs": [1],
            },
            {
                "block_id": "C",
                "kind": "agent",
                "agent_name": "Computer_terminal",
                "summary": "Executed the script and printed 0.",
                "input": "Python script from step 1.",
                "output": "exitcode 0; stdout 0.",
                "authorship": "executed",
                "step_refs": [2],
                "defect_recorded": _DEFECT,
            },
            {
                "block_id": "D",
                "kind": "agent",
                "agent_name": "Verification_Expert",
                "summary": "Treated the printed 0 as confirmation and stopped.",
                "input": "Computer_terminal result of 0 from step 2.",
                "output": "Final answer 0.",
                "authorship": "relayed",
                "step_refs": [3, 4],
            },
        ],
        "edges": [
            {
                "from": "B",
                "to": "C",
                "link_kind": "adjacent",
                "inherited": "Step 1 defined ps-detection as a substring of entry_id.",
                "effect": "Step 2 executed that heuristic and produced 0.",
            },
            {
                "from": "C",
                "to": "D",
                "link_kind": "adjacent",
                "inherited": "Step 2 returned 0 from the entry_id substring method.",
                "effect": "Step 3 accepted that 0 without checking the method.",
            },
            {
                "from": "A",
                "to": "D",
                "link_kind": "skip",
                "inherited": "Step 0 propagated the prior 0 framing.",
                "effect": "Step 3 used that framing to terminate at 0.",
            },
        ],
    }


def _appendix_snapshot() -> AoSnapshot:
    names = (
        "Data_Extraction_Expert",
        "Verification_Expert",
        "Computer_terminal",
        "Verification_Expert",
        "Verification_Expert",
    )
    records = tuple(
        _record(index, f"step {index}", participant=name, session_id="sess-paper")
        for index, name in enumerate(names)
    )
    return AoSnapshot(records, watermark="wm-paper", question="count the articles")


def test_appendix_f_shape_compiles_and_round_trips(tmp_path: Path) -> None:
    graph = _appendix_graph()
    snapshot = _appendix_snapshot()
    compiled = compile_paper_graph(
        graph,
        snapshot,
        root=tmp_path,
        run_id="run-paper",
        session_id="sess-paper",
        task_run_id="run-paper",
        view_id="view-run-paper",
        revision_id="rev-run-paper",
        frozen_at=_FROZEN_AT,
    )
    blocks = {block.block_id: block for block in compiled.revision.blocks}
    assert compiled.id_map == {"A": "A", "B": "B", "C": "C", "D": "D"}
    assert blocks["A"].participant_id == "Data_Extraction_Expert"
    assert (blocks["A"].ao_start_seq, blocks["A"].ao_end_seq) == (1, 1)
    assert blocks["A"].authorship == "relayed"
    assert blocks["A"].block_kind == "agent"
    assert blocks["A"].role == "agent_action"
    assert blocks["A"].defect_recorded is None
    assert "defect_recorded" not in blocks["A"].to_dict()
    assert blocks["B"].participant_id == "Verification_Expert"
    assert blocks["B"].authorship == "authored"
    assert (blocks["B"].ao_start_seq, blocks["B"].ao_end_seq) == (2, 2)
    assert blocks["C"].participant_id == "Computer_terminal"
    assert blocks["C"].authorship == "executed"
    assert blocks["C"].defect_recorded == _DEFECT
    assert blocks["C"].to_dict()["defect_recorded"] == {"note": _DEFECT}
    assert (blocks["C"].ao_start_seq, blocks["C"].ao_end_seq) == (3, 3)
    assert blocks["D"].participant_id == "Verification_Expert"
    assert blocks["D"].authorship == "relayed"
    assert (blocks["D"].ao_start_seq, blocks["D"].ao_end_seq) == (4, 5)
    assert len(blocks["D"].source_refs) == 2
    edges = {
        (edge.source_block_id, edge.target_block_id): edge for edge in compiled.revision.edges
    }
    assert edges[("B", "C")].relation_type == "influence"
    assert edges[("C", "D")].relation_type == "influence"
    assert edges[("A", "D")].relation_type == "skip"
    for paper in graph["edges"]:
        native = edges[(paper["from"], paper["to"])]
        assert native.mechanism == join_mechanism(paper["inherited"], paper["effect"])
    assert compiled.revision.construction_mode == CONSTRUCTION_MODE_ADAPTIVE_AGENTIC
    assert "known_violations" not in compiled.revision.protocol_body()
    document = json.loads(compiled.sidecar_path.read_text(encoding="utf-8"))
    assert document["content_hash"] == compiled.sidecar_hash
    assert compiled.revision.paper_sidecar_hash == sidecar_digest(document["body"])
    assert document["body"]["id_map"] == compiled.id_map
    rendered = render_paper_graph(compiled.revision, snapshot)
    by_id = {node["block_id"]: node for node in rendered["blocks"]}
    for node in graph["blocks"]:
        got = by_id[node["block_id"]]
        for key, value in node.items():
            assert got[key] == value
    rendered_edges = sorted(rendered["edges"], key=lambda item: (item["from"], item["to"]))
    expected_edges = sorted(graph["edges"], key=lambda item: (item["from"], item["to"]))
    assert rendered_edges == expected_edges
    parsed = read_native_edge_revision(compiled.revision.protocol_document())
    again = {block.block_id: block for block in parsed.revision.blocks}
    assert again["C"].defect_recorded == _DEFECT
    assert again["C"].authorship == "executed"
    assert again["A"].authorship == "relayed"
    assert again["B"].authorship == "authored"
    assert again["D"].authorship == "relayed"


def test_invalid_authorship_is_rejected(tmp_path: Path) -> None:
    snapshot = AoSnapshot(
        (_record(0, "row", participant="worker", session_id="sess-paper"),),
        watermark="wm-paper",
    )
    graph = {
        "blocks": [
            {
                "block_id": "A",
                "kind": "agent",
                "agent_name": "worker",
                "summary": "did the work",
                "input": "do the task",
                "output": "a draft",
                "authorship": "nope",
                "step_refs": [0],
            }
        ],
        "edges": [],
    }
    with pytest.raises(PaperCompileError) as caught:
        compile_paper_graph(
            graph,
            snapshot,
            root=tmp_path,
            run_id="run-bad",
            session_id="sess-paper",
            task_run_id="run-bad",
            view_id="view-run-bad",
            revision_id="rev-run-bad",
            frozen_at=_FROZEN_AT,
        )
    assert caught.value.code == "invalid_authorship"
