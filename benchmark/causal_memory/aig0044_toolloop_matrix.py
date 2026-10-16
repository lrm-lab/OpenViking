#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""ADR-0044 Wave 3 structural matrix. Zero LLM.

A synthetic appendix-F trajectory drives build, critique, refine, compile,
reader navigation, T_fork, and derived experience hops. Scores below are
booleans and content-addressed ids. Session ids, retrieval ids, and temp
paths are not part of the report. Do not edit the frozen rules to absorb a miss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openviking.session.aig_native_protocol import (  # noqa: E402
    OBJECT_KIND_AGENT_CANDIDATE_CATALOG,
    OBJECT_KIND_BRANCH_EVALUATION,
    OBJECT_KIND_FORK_PROPOSAL,
    OBJECT_KIND_GUIDED_COUNTERFACTUAL_BRANCH,
    OBJECT_KIND_OUTCOME_CONTRACT,
    OBJECT_KIND_STRATEGY_PATTERN_PROPOSAL,
    OBJECT_KIND_TASK_OUTCOME_ASSESSMENT,
    SCHEMA_VERSION_AIG_NATIVE_V2,
    protocol_envelope,
)
from openviking.session.aig_role_chain import AigConstructionRun  # noqa: E402
from openviking.session.aig_tool_surface import (  # noqa: E402
    TOOL_CATALOG,
    AigToolSurface,
    AoSnapshot,
    ToolSurfaceError,
)
from openviking.session.aig_toolloop_builder import (  # noqa: E402
    FakeToolloopProvider,
    run_toolloop,
)
from openviking.session.aig_toolloop_critic import FakeCriticProvider  # noqa: E402
from openviking.session.aig_toolloop_orchestrator import run_critic_refiner  # noqa: E402
from openviking.session.aig_toolloop_reader import (  # noqa: E402
    DeterministicReader,
    run_reader,
)
from openviking.session.aig_toolloop_refiner import (  # noqa: E402
    FakeRefinerProvider,
    RefinerAction,
    RefinerError,
    RefinerSession,
)
from openviking.session.ao_ledger import AORecord  # noqa: E402
from openviking.session.experience_links import (  # noqa: E402
    DEFAULT_MEMORY_MAX_CARDS,
    DEFAULT_MEMORY_MAX_HOPS,
    DEFAULT_READER_MAX_CARDS,
    DEFAULT_READER_MAX_HOPS,
    ExperienceLinkError,
    ExperienceLinkService,
)
from openviking.session.extraction_write import freeze_native_edge_view  # noqa: E402
from openviking.session.fork_node_overlay import ForkNodeOverlayStore  # noqa: E402
from openviking.session.fork_precedent import ForkPrecedentService  # noqa: E402
from openviking.session.guidance_approval import GuidanceApprovalService  # noqa: E402
from openviking.session.influence_view import (  # noqa: E402
    CONSTRUCTION_MODE_ADAPTIVE_AGENTIC,
    CONSTRUCTION_MODE_NATIVE_SINGLE_PASS,
)
from openviking.session.outcome_contract import (  # noqa: E402
    ExternalOutcomeEvidence,
    OutcomeAssessmentService,
    OutcomeContractStore,
)
from openviking.session.paper_graph_compile import (  # noqa: E402
    render_paper_graph,
    sidecar_digest,
)
from openviking.session.paper_graph_phi import PaperViolation  # noqa: E402
from openviking.session.strategy_pattern import StrategyPatternStore  # noqa: E402
from openviking.session.t_fork_surface import TForkError, TForkSurface  # noqa: E402

SCHEMA = "aig0044-toolloop-matrix-v1"
FROZEN_AT = "2026-10-11T00:00:00.000Z"
SESSION = "sess-aig0044"
CAROL = "user:carol"
BOB = "user:bob"
CUTOFF = "2026-10-10T00:00:01.000Z"
RECORDED = "2026-10-10T00:00:02.000Z"
EFFECT_TA = "dropped the format field"
EFFECT_AB = "relayed the prior zero framing"
EFFECT_BC = "executed this flawed heuristic and produced 0"
EFFECT_CE = "accepted that 0 at face value"
EFFECT_SKIP = "leaned on the prior zero framing to terminate"
DEFECT_B = "ps detection used entry_id"
DEFECT_C = "executed the flawed script unchanged"
FROZEN_RULES = (
    "Frozen before the matrix runs. A miss is a miss.",
    "Build: the 19-verb catalog must each succeed once; "
    "step_out_of_range, step_refs_overlap, edge_not_forward, missing_field, "
    "not_adjacent, and not_skip are rejected on the call that violates them; "
    "a one-call budget returns construction_mode=fallback and "
    "build_minimal_fallback_view.",
    "Critique/refine: an empty violation set freezes without entering the loop; "
    "three rounds of a remaining high violation fall back; three rounds of a "
    "remaining low violation freeze with known_violations; an unknown "
    "violation_id is rejected.",
    "Compile: the paper graph freezes as adaptive_agentic_aig; the sidecar hash "
    "matches the revision; render_paper_graph round-trips the paper fields.",
    "Read: responsibility walks incoming edges and stops at agent-b "
    "(Verification_Expert, step 2). Anchor candidates are agent-b then agent-c.",
    "T_fork: search, submit_fork_proposal, and submit_guidance_plan are "
    "phase_locked until confirmed_failure. The chain is retrieve, proposal, "
    "guidance. The same proposal returns the same overlay. A fourth distinct "
    "submit fails closed with submit_budget_exhausted.",
    "Hops: anchors leave the fork toward its block. invalidated becomes refutes. "
    "A non-intersecting principal sees no cards. replay-rev: is not a hop. "
    "Reader budget is 8 hops and 32 cards; memory is 4 and 16 and hides retired. "
    "Either limit fails closed with hop_budget_exceeded. retired stays visible "
    "on the reader profile with status retired.",
    "native_single_pass_aig stays the construction-run default and the "
    "freeze_native_edge_view default.",
    "The report omits session uuids, retrieval ids, and filesystem paths.",
)


def _record(index: int, participant: str, text: str, *, session_id: str = SESSION) -> AORecord:
    return AORecord(
        ao_id=f"ao-{index}",
        session_id=session_id,
        sequence=index + 1,
        action={"participant_id": participant, "segment_id": "seg", "text": text},
        observation={"text": text},
        skill_invocations=[],
        message_ref=None,
        created_at=FROZEN_AT,
        captured_state="live",
    )


def _envelope(kind: str, body: dict[str, Any]) -> dict[str, Any]:
    return protocol_envelope(SCHEMA_VERSION_AIG_NATIVE_V2, kind, body)


def _code(call) -> str:
    try:
        call()
    except (ToolSurfaceError, RefinerError, TForkError, ExperienceLinkError) as exc:
        return exc.code
    raise AssertionError("expected a coded error")


def _paper_surface(root: Path) -> tuple[AigToolSurface, AoSnapshot]:
    records = (
        _record(0, "Task_Framer", "count hep-lat articles from January 2020"),
        _record(1, "Data_Extraction_Expert", "Carried forward a prior result of 0"),
        _record(2, "Verification_Expert", "Python code that counts ps via substring of entry_id"),
        _record(3, "Computer_terminal", f"exitcode 0; {EFFECT_BC}"),
        _record(4, "Closer", f"Final answer 0. {EFFECT_CE}"),
    )
    snapshot = AoSnapshot(records, watermark="wm-aig0044", question="how many hep-lat articles")
    surface = AigToolSurface(snapshot, root=root, run_id="run-paper")
    surface.add_block(
        "task",
        {
            "content": "count hep-lat articles from January 2020",
            "operative_brief": "start from the manager request",
            "step_refs": [0],
        },
        block_id="task-1",
    )
    surface.add_block(
        "agent",
        {
            "agent_name": "Data_Extraction_Expert",
            "authorship": "relayed",
            "input": "Count hep-lat Arxiv articles from January 2020.",
            "output": "Carried forward a prior result of 0.",
            "step_refs": [1],
            "summary": "Restated the manager task and carried a prior zero.",
        },
        block_id="agent-a",
    )
    surface.add_block(
        "agent",
        {
            "agent_name": "Verification_Expert",
            "authorship": "authored",
            "input": "Plan from the extraction step.",
            "output": "Python code that counts ps via substring of entry_id.",
            "step_refs": [2],
            "summary": "Proposed a script that treats ps as a substring of entry_id.",
        },
        block_id="agent-b",
    )
    surface.add_block(
        "agent",
        {
            "agent_name": "Computer_terminal",
            "authorship": "executed",
            "input": "The script from the verification step.",
            "output": f"exitcode 0; {EFFECT_BC}",
            "step_refs": [3],
            "summary": "Ran the script and printed 0.",
        },
        block_id="agent-c",
    )
    surface.add_block(
        "conclusion",
        {"content": f"Final answer 0. {EFFECT_CE}", "step_refs": [4]},
        block_id="end-1",
    )
    surface.add_inheritance_note("task-1", "agent-a", "the manager request", EFFECT_TA)
    surface.add_inheritance_note("agent-a", "agent-b", "the prior zero", EFFECT_AB)
    surface.add_inheritance_note("agent-b", "agent-c", "the flawed script", EFFECT_BC)
    surface.add_inheritance_note("agent-c", "end-1", "the printed zero", EFFECT_CE)
    surface.add_skip_attribution("agent-a", "end-1", "the prior zero framing", EFFECT_SKIP)
    surface.mark_defect_recorded("agent-b", DEFECT_B)
    surface.mark_defect_recorded("agent-c", DEFECT_C)
    surface.submit()
    return surface, snapshot


def _legal(root: Path, run_id: str, session_id: str) -> tuple[AigToolSurface, AoSnapshot]:
    records = tuple(
        _record(index, "worker", f"row {index}", session_id=session_id) for index in range(3)
    )
    snapshot = AoSnapshot(records, watermark=f"wm-{run_id}", question="what failed?")
    surface = AigToolSurface(snapshot, root=root, run_id=run_id)
    surface.add_block(
        "task",
        {"content": "the question", "operative_brief": "start here", "step_refs": [0]},
        block_id="task-1",
    )
    surface.add_block(
        "agent",
        {
            "agent_name": "worker",
            "authorship": "authored",
            "input": "do the task",
            "output": "a draft",
            "step_refs": [1],
            "summary": "did the work",
        },
        block_id="agent-1",
    )
    surface.add_block("conclusion", {"content": "the answer", "step_refs": [2]}, block_id="end-1")
    surface.add_inheritance_note("task-1", "agent-1", "the question", "the worker dropped it")
    surface.add_inheritance_note("agent-1", "end-1", "the draft", "the answer published it")
    surface.submit()
    return surface, snapshot


def _semantic(severity: str) -> dict[str, Any]:
    return {
        "violations": [
            {
                "evidence_ao_refs": [1],
                "kind": "semantic",
                "note": "summary contradicts the row",
                "object_id": "agent-1",
                "severity": severity,
            }
        ]
    }


def _refiner() -> FakeRefinerProvider:
    return FakeRefinerProvider(
        [
            RefinerAction(
                kind="edit",
                verb="modify_block",
                violation_id="critic:0:semantic:agent-1",
                arguments={
                    "block_id": "agent-1",
                    "fields": {"summary": "revised after the critic note"},
                },
            )
        ]
    )


def _verb_script() -> list[tuple[str, dict[str, Any]]]:
    task = {"content": "the question", "operative_brief": "start here", "step_refs": [0]}
    orch = {
        "agent_name": "lead",
        "authorship": "relayed",
        "step_refs": [1],
        "summary": "routed the work",
    }
    agent = {
        "agent_name": "worker",
        "authorship": "authored",
        "input": "do the task",
        "output": "a draft",
        "step_refs": [2],
        "summary": "did the work",
    }
    spare = {
        "agent_name": "spare",
        "authorship": "relayed",
        "input": "the draft",
        "output": "a copy",
        "step_refs": [3],
        "summary": "copied the draft",
    }
    end = {"content": "the answer", "step_refs": [4]}
    return [
        ("get_overview", {}),
        ("read_full_segment", {"segment": "seg"}),
        ("read_step", {"index": 0}),
        ("read_step_range", {"start": 0, "end": 1}),
        ("list_agents", {}),
        ("get_question", {}),
        ("search", {"pattern": "q"}),
        ("list_block_kinds", {}),
        ("get_story", {}),
        ("get_graph", {}),
        ("add_block", {"kind": "task", "block_id": "task-1", "fields": task}),
        ("add_block", {"kind": "orchestrator", "block_id": "orch-1", "fields": orch}),
        (
            "add_inheritance_note",
            {
                "effect": "the route dropped a constraint",
                "from": "task-1",
                "inherited": "the question",
                "to": "orch-1",
            },
        ),
        ("add_block", {"kind": "agent", "block_id": "agent-1", "fields": agent}),
        (
            "add_inheritance_note",
            {
                "effect": "the worker followed the bad plan",
                "from": "orch-1",
                "inherited": "the plan",
                "to": "agent-1",
            },
        ),
        ("add_block", {"kind": "agent", "block_id": "spare-1", "fields": spare}),
        (
            "add_skip_attribution",
            {
                "effect": "the worker skipped the route",
                "from": "task-1",
                "inherited": "the question",
                "to": "agent-1",
            },
        ),
        (
            "modify_block",
            {"block_id": "agent-1", "fields": {"summary": "worked on the wrong question"}},
        ),
        (
            "mark_defect_recorded",
            {"block_id": "agent-1", "note": "output missed the operative brief"},
        ),
        ("delete_inheritance_note", {"from": "orch-1", "to": "agent-1"}),
        ("delete_block", {"block_id": "spare-1"}),
        ("delete_skip_attribution", {"from": "task-1", "to": "agent-1"}),
        ("add_block", {"kind": "conclusion", "block_id": "end-1", "fields": end}),
        ("submit", {}),
    ]


def _build(root: Path) -> dict[str, Any]:
    texts = ["q", "plan", "work", "spare", "answer"]
    records = tuple(
        _record(index, "worker", text, session_id="sess-verbs") for index, text in enumerate(texts)
    )
    snapshot = AoSnapshot(records, watermark="wm-verbs", question="what failed?")
    covered = run_toolloop(
        snapshot,
        "what failed?",
        FakeToolloopProvider(_verb_script()),
        root=root,
        run_id="run-verbs",
    )
    ok = {item.verb for item in covered.audit if item.ok}
    catalog = [name for name, _text in TOOL_CATALOG]
    failed = [(item.verb, item.error_code) for item in covered.audit if not item.ok]
    if failed or covered.status != "submitted" or set(catalog) - ok:
        raise AssertionError(
            f"verb coverage failed status={covered.status} "
            f"missing={set(catalog) - ok} failed={failed}"
        )
    hard = _hard_invariants(root)
    budget = run_toolloop(
        snapshot,
        "what failed?",
        FakeToolloopProvider([("get_overview", {})]),
        root=root,
        run_id="run-budget",
        max_calls=1,
    )
    marker = budget.fallback_marker or {}
    if budget.status != "budget_exhausted" or budget.construction_mode != "fallback":
        raise AssertionError(f"budget status={budget.status} mode={budget.construction_mode}")
    if marker.get("deterministic_view") != "build_minimal_fallback_view":
        raise AssertionError(f"fallback marker={marker.get('deterministic_view')}")
    return {
        "budget_fallback": True,
        "fallback_view": marker["deterministic_view"],
        "hard_invariants": hard,
        "verb_count": len(catalog),
        "verbs_ok": catalog,
    }


def _hard_invariants(root: Path) -> dict[str, bool]:
    records = tuple(
        _record(index, "worker", text, session_id="sess-hard")
        for index, text in enumerate(["q", "plan", "work", "answer"])
    )
    snapshot = AoSnapshot(records, watermark="wm-hard", question="what failed?")
    surface = AigToolSurface(snapshot, root=root, run_id="run-hard")
    surface.add_block(
        "task",
        {"content": "the question", "operative_brief": "start here", "step_refs": [0]},
        block_id="task-1",
    )
    surface.add_block(
        "orchestrator",
        {
            "agent_name": "lead",
            "authorship": "relayed",
            "step_refs": [1],
            "summary": "routed the work",
        },
        block_id="orch-1",
    )
    surface.add_block(
        "agent",
        {
            "agent_name": "worker",
            "authorship": "authored",
            "input": "do the task",
            "output": "a draft",
            "step_refs": [2],
            "summary": "did the work",
        },
        block_id="agent-1",
    )
    surface.add_block("conclusion", {"content": "the answer", "step_refs": [3]}, block_id="end-1")
    surface.add_inheritance_note(
        "task-1", "orch-1", "the question", "the route dropped a constraint"
    )
    surface.add_inheritance_note(
        "orch-1", "agent-1", "the plan", "the worker followed the bad plan"
    )
    surface.add_inheritance_note("agent-1", "end-1", "the draft", "the answer published it")
    agent = {
        "agent_name": "worker",
        "authorship": "authored",
        "input": "do the task",
        "output": "a draft",
        "step_refs": [2],
        "summary": "did the work",
    }
    found = {
        "edge_not_forward": _code(
            lambda: surface.add_inheritance_note("end-1", "task-1", "later", "backwards")
        )
        == "edge_not_forward",
        "missing_field": _code(lambda: surface.modify_block("task-1", {"content": ""}))
        == "missing_field",
        "not_adjacent": _code(
            lambda: surface.add_inheritance_note("task-1", "end-1", "the question", "skipped")
        )
        == "not_adjacent",
        "step_out_of_range": _code(
            lambda: surface.add_block("agent", {**agent, "step_refs": [99]}, block_id="agent-x")
        )
        == "step_out_of_range",
        "step_refs_overlap": _code(lambda: surface.add_block("agent", agent, block_id="agent-x"))
        == "step_refs_overlap",
    }
    close = AigToolSurface(
        AoSnapshot(
            (
                _record(0, "worker", "q", session_id="sess-skip"),
                _record(1, "worker", "a", session_id="sess-skip"),
            ),
            watermark="wm-skip",
            question="what failed?",
        ),
        root=root,
        run_id="run-skip",
    )
    close.add_block(
        "task",
        {"content": "the question", "operative_brief": "start here", "step_refs": [0]},
        block_id="task-1",
    )
    close.add_block("conclusion", {"content": "the answer", "step_refs": [1]}, block_id="end-1")
    found["not_skip"] = (
        _code(lambda: close.add_skip_attribution("task-1", "end-1", "the question", "too close"))
        == "not_skip"
    )
    return found


def _loops(root: Path) -> dict[str, Any]:
    empty_surface, empty_snapshot = _paper_surface(root)
    empty = run_critic_refiner(
        empty_surface,
        empty_snapshot,
        critic=FakeCriticProvider(),
        refiner=FakeRefinerProvider(),
        root=root,
        run_id="run-paper",
        frozen_at=FROZEN_AT,
    )
    if empty.entered_loop or empty.status != "frozen" or empty.revision is None:
        raise AssertionError(f"empty critic status={empty.status} entered={empty.entered_loop}")
    high_surface, high_snapshot = _legal(root, "run-high", "sess-high")
    high = run_critic_refiner(
        high_surface,
        high_snapshot,
        critic=FakeCriticProvider([_semantic("high")]),
        refiner=_refiner(),
        root=root,
        run_id="run-high",
        frozen_at=FROZEN_AT,
    )
    if high.status != "fallback" or high.revision_rounds != 3 or high.revision is not None:
        raise AssertionError(f"high status={high.status} rounds={high.revision_rounds}")
    low_surface, low_snapshot = _legal(root, "run-low", "sess-low")
    low = run_critic_refiner(
        low_surface,
        low_snapshot,
        critic=FakeCriticProvider([_semantic("low")]),
        refiner=_refiner(),
        root=root,
        run_id="run-low",
        frozen_at=FROZEN_AT,
    )
    if low.status != "frozen" or low.revision_rounds != 3 or low.revision is None:
        raise AssertionError(f"low status={low.status} rounds={low.revision_rounds}")
    known = low.known_violations
    if not known or known[0].get("severity") != "low":
        raise AssertionError(f"low known={known}")
    unknown_surface, _snapshot = _legal(root, "run-unknown", "sess-unknown")
    session = RefinerSession(
        unknown_surface,
        (
            PaperViolation(
                violation_id="critic:0:semantic:agent-1",
                object_id="agent-1",
                kind="semantic",
                severity="low",
                evidence="summary contradicts the row",
                scope="objects",
                involved_ids=("agent-1",),
            ),
        ),
    )
    if _code(lambda: session.declare_unresolvable("missing", "no such row")) != "unknown_violation":
        raise AssertionError("unknown violation was accepted")
    return {
        "compile": _compile_view(empty, empty_snapshot),
        "empty_entered_loop": False,
        "empty_rounds": empty.revision_rounds,
        "empty_status": empty.status,
        "high_mode": high.construction_mode,
        "high_rounds": high.revision_rounds,
        "high_status": high.status,
        "low_known_severity": known[0]["severity"],
        "low_known_violation_id": known[0]["violation_id"],
        "low_rounds": low.revision_rounds,
        "low_status": low.status,
        "pipeline": (empty, empty_surface, empty_snapshot),
        "unknown_violation": True,
    }


def _compile_view(result, snapshot: AoSnapshot) -> dict[str, Any]:
    revision = result.revision
    document = json.loads(Path(result.sidecar_path).read_text(encoding="utf-8"))
    if document["content_hash"] != result.sidecar_hash:
        raise AssertionError("sidecar file hash drifted")
    if revision.paper_sidecar_hash != sidecar_digest(document["body"]):
        raise AssertionError("revision sidecar hash drifted")
    if revision.construction_mode != CONSTRUCTION_MODE_ADAPTIVE_AGENTIC:
        raise AssertionError(revision.construction_mode)
    rendered = render_paper_graph(revision, snapshot)
    paper = {
        block["block_id"]: block
        for block in json.loads(Path(result.sidecar_path).read_text(encoding="utf-8"))["body"][
            "nodes"
        ]
    }
    by_id = {node["block_id"]: node for node in rendered["blocks"]}
    for block_id, node in paper.items():
        got = by_id[block_id]
        if node["kind"] == "task":
            expected = {
                "agent_name": got["agent_name"],
                "input": node["operative_brief"],
                "kind": "task",
                "output": "",
                "step_refs": node["step_refs"],
                "summary": node["content"],
            }
        elif node["kind"] == "conclusion":
            expected = {
                "agent_name": got["agent_name"],
                "input": "",
                "kind": "conclusion",
                "output": node["content"],
                "step_refs": node["step_refs"],
                "summary": node["content"],
            }
        else:
            expected = {
                "agent_name": node["agent_name"],
                "authorship": node["authorship"],
                "input": node["input"],
                "kind": node["kind"],
                "output": node["output"],
                "step_refs": node["step_refs"],
                "summary": node["summary"],
            }
            if "defect_recorded" in node:
                expected["defect_recorded"] = node["defect_recorded"]
        for key, value in expected.items():
            if got.get(key) != value:
                raise AssertionError(f"round trip {block_id}.{key}")
    paper_edges = sorted(document["body"]["edges"], key=lambda item: (item["from"], item["to"]))
    rendered_edges = sorted(rendered["edges"], key=lambda item: (item["from"], item["to"]))
    if paper_edges != rendered_edges:
        raise AssertionError("round trip edges drifted")
    native = {block.block_id: block for block in revision.blocks}
    if native["agent-b"].participant_id != "Verification_Expert":
        raise AssertionError("agent-b participant")
    if native["agent-b"].authorship != "authored" or native["agent-b"].defect_recorded != DEFECT_B:
        raise AssertionError("agent-b fields")
    if native["agent-c"].authorship != "executed" or native["agent-c"].defect_recorded != DEFECT_C:
        raise AssertionError("agent-c fields")
    edges = {
        (edge.source_block_id, edge.target_block_id): edge.relation_type for edge in revision.edges
    }
    if edges[("agent-b", "agent-c")] != "influence" or edges[("agent-a", "end-1")] != "skip":
        raise AssertionError(f"edge kinds {edges}")
    return {
        "block_ids": sorted(native),
        "construction_mode": revision.construction_mode,
        "round_trip": True,
        "sidecar_hash": result.sidecar_hash,
    }


def _read(root: Path, compiled, surface: AigToolSurface, snapshot: AoSnapshot) -> dict[str, Any]:
    run = compiled.run
    run.submit_draft(run.session_for("builder"), surface.get_graph())
    result = run_reader(
        compiled.revision,
        snapshot,
        DeterministicReader(),
        root=root,
        run_id="run-paper",
        construction_run=run,
        sidecar_path=compiled.sidecar_path,
    )
    report = result.report
    if report.critical_block_id != "agent-b" or report.responsible_agent != "Verification_Expert":
        raise AssertionError(
            f"reader stopped at {report.critical_block_id} {report.responsible_agent}"
        )
    if report.decisive_ao_refs != (2,) or result.anchor_candidates != ("agent-b", "agent-c"):
        raise AssertionError(f"refs={report.decisive_ao_refs} anchors={result.anchor_candidates}")
    if result.reader_report_hash != run.reader_report_hash or run.phase != "SYNTHESIZE_PROPOSE":
        raise AssertionError("reader hash was not frozen")
    return {
        "anchor_candidates": list(result.anchor_candidates),
        "critical_block_id": report.critical_block_id,
        "decisive_ao_refs": list(report.decisive_ao_refs),
        "reader_report_hash": result.reader_report_hash,
        "responsible_agent": report.responsible_agent,
        "run": run,
    }


def _proposal(catalog, document_hash: str, **overrides: object) -> dict[str, Any]:
    body: dict[str, Any] = {
        "anchor_block_id": "agent-b",
        "author_role": "reader",
        "base_revision_block_content_hash": catalog.block_content_hash,
        "base_revision_id": catalog.base_revision_id,
        "branch_steps": [
            {"block_id": "agent-a", "step_kind": "factual_block_ref"},
            {"imagined_node_id": "img-retry", "step_kind": "imagined_node"},
        ],
        "report_hash": document_hash,
    }
    body.update(overrides)
    return _envelope(OBJECT_KIND_FORK_PROPOSAL, body)


def _evidence() -> dict[str, Any]:
    return {
        "ao_id": "ao-eval-1",
        "archive_commit_watermark": None,
        "captured_state": "committed",
        "evidence_content_hash": "hash-eval-1",
        "evidence_role": "hindsight_attribution",
        "source_archive_id": None,
        "source_read_snapshot_watermark": "wm-aig0044",
        "source_sequence": 4,
        "source_session_id": SESSION,
    }


def _branch(catalog, report_hash: str, retrieval_id: str, fork_id: str) -> dict[str, Any]:
    return _envelope(
        OBJECT_KIND_GUIDED_COUNTERFACTUAL_BRANCH,
        {
            "applicability": {
                "action_skill_role": "applicable",
                "anchor_state": "partially_applicable",
                "failure_mode": "applicable",
                "task_decision_type": "applicable",
            },
            "author_role": "reader",
            "base_revision_id": catalog.base_revision_id,
            "report_hash": report_hash,
            "required_capabilities": ["replay-sandbox"],
            "retrieval_id": retrieval_id,
            "risk": "retry may repeat the failed rollout",
            "source_fork_id": fork_id,
            "steps": [
                {
                    "evidence_refs": [_evidence()],
                    "imagined_node_id": "img-retry",
                    "statement": "retry the count with a format-field check",
                    "step_id": "step-retry",
                }
            ],
            "target_role": "task_agent",
            "task_id": "task-1",
        },
    )


def _catalog(task_id: str = "task-1") -> dict[str, Any]:
    return _envelope(
        OBJECT_KIND_AGENT_CANDIDATE_CATALOG,
        {
            "candidates": [
                {
                    "acl_labels": [CAROL],
                    "agent_id": "agent-a",
                    "available": True,
                    "capabilities": ["replay-sandbox", "rollback"],
                    "roles": ["task_agent"],
                    "task_member": True,
                }
            ],
            "task_id": task_id,
        },
    )


def _pattern_proposal(fork_id: str, imagined: str, mechanism: str) -> dict[str, Any]:
    return _envelope(
        OBJECT_KIND_STRATEGY_PATTERN_PROPOSAL,
        {
            "applicability_conditions": [
                {"condition_id": "anchor-state", "statement": "anchor block is the failed action"},
                {"condition_id": "failure-mode", "statement": "the script counted the wrong field"},
            ],
            "author_role": "reader",
            "capability": "replay-sandbox",
            "expected_metric": "count matches the format field",
            "fork_node_id": fork_id,
            "imagined_node_id": imagined,
            "intervention_type": "narrow_retry",
            "mechanism": mechanism,
            "semantic_summary": mechanism,
            "target_boundary": "agent-b",
        },
    )


def _evaluation(
    subject_id: str, evaluation_id: str, evidence_ref: str, status: str
) -> dict[str, Any]:
    return _envelope(
        OBJECT_KIND_BRANCH_EVALUATION,
        {
            "evaluation_id": evaluation_id,
            "evidence_ref": evidence_ref,
            "evidence_tier": "analogical",
            "status": status,
            "subject_id": subject_id,
            "subject_kind": "imagined_node",
        },
    )


def _fork_and_hops(
    root: Path, revision, run: AigConstructionRun, report_hash: str
) -> dict[str, Any]:
    outcomes = OutcomeAssessmentService(OutcomeContractStore(root))
    outcomes.freeze_contract(
        _envelope(
            OBJECT_KIND_OUTCOME_CONTRACT,
            {
                "acceptance_criteria": [
                    {
                        "criterion_id": "crit-error-budget",
                        "statement": "the count uses the format field",
                    }
                ],
                "contract_id": "contract-task-1",
                "evaluator_bindings": [
                    {"binding_id": "bind-eval-1", "kind": "evaluator", "ref": "eval://hep-lat"}
                ],
                "frozen_at": "2026-10-10T00:00:00.000Z",
                "goals": ["count hep-lat articles"],
                "policy_constraints": [
                    {"constraint_id": "sla-latency", "kind": "sla", "statement": "finish the count"}
                ],
                "source": "orchestrator",
                "task_id": "task-1",
            },
        )
    )
    outcomes.submit_assessment(
        run,
        run.session_for("reader"),
        _envelope(
            OBJECT_KIND_TASK_OUTCOME_ASSESSMENT,
            {
                "author_role": "reader",
                "criteria_dispositions": {"crit-error-budget": "explicit"},
                "evidence_refs": [_evidence()],
                "report_hash": report_hash,
                "verdict": "failure",
            },
        ),
        task_id="task-1",
        contract_id="contract-task-1",
    )
    overlays = ForkNodeOverlayStore(root)
    patterns = StrategyPatternStore(root)
    precedents = ForkPrecedentService(root)
    guidance = GuidanceApprovalService(
        root, outcomes=outcomes, precedents=precedents, overlays=overlays
    )
    catalog = overlays.register_base_revision(revision, acl_labels=(CAROL,))
    common = {
        "run": run,
        "session_id": run.session_for("reader"),
        "task_id": "task-1",
        "report_hash": report_hash,
        "principal_labels": (CAROL,),
        "overlays": overlays,
        "patterns": patterns,
        "precedents": precedents,
        "guidance": guidance,
        "outcomes": outcomes,
    }
    locked = TForkSurface(root, external_evidence=(), **common)
    proposal = _proposal(catalog, revision.content_hash)
    query = {"proposal": proposal, "temporal_cutoff": CUTOFF}
    if _code(lambda: locked.search_precedents(query)) != "phase_locked":
        raise AssertionError("action verbs were visible before confirmed failure")
    if "search_precedents" in {name for name, _text in locked.catalog()}:
        raise AssertionError("locked catalog listed an action verb")
    opened = TForkSurface(root, external_evidence=(_fail_signal(),), **common)
    found = opened.search_precedents(query)
    if found["high_confidence"]:
        raise AssertionError("self citation leaked into high confidence")
    first = opened.submit_fork_proposal(
        proposal, [], retrieval_id=found["retrieval_id"], recorded_at=RECORDED
    )
    again = opened.submit_fork_proposal(
        proposal, [], retrieval_id=found["retrieval_id"], recorded_at=RECORDED
    )
    if again["fork_node_id"] != first["fork_node_id"] or again["deduplicated"] is not True:
        raise AssertionError("resubmit did not return the same overlay")
    guided = opened.submit_guidance_plan(
        {
            "catalog": _catalog(),
            "proposal": _branch(catalog, report_hash, found["retrieval_id"], first["fork_node_id"]),
        }
    )
    if (
        guided["selected_agent_id"] != "agent-a"
        or guided["source_fork_id"] != first["fork_node_id"]
    ):
        raise AssertionError(f"guidance={guided}")
    conflict = _proposal(
        catalog,
        revision.content_hash,
        branch_steps=[
            {"block_id": "agent-a", "step_kind": "factual_block_ref"},
            {"imagined_node_id": "img-other", "step_kind": "imagined_node"},
        ],
    )
    conflict_found = opened.search_precedents({"proposal": conflict, "temporal_cutoff": CUTOFF})
    if (
        _code(
            lambda: opened.submit_fork_proposal(
                conflict, [], retrieval_id=conflict_found["retrieval_id"], recorded_at=RECORDED
            )
        )
        != "overlay_conflict"
    ):
        raise AssertionError("same-anchor conflict was stored")
    retired_proposal = _proposal(
        catalog,
        revision.content_hash,
        anchor_block_id="agent-c",
        branch_steps=[
            {"block_id": "agent-b", "step_kind": "factual_block_ref"},
            {"imagined_node_id": "img-retired", "step_kind": "imagined_node"},
        ],
    )
    retired_found = opened.search_precedents(
        {"proposal": retired_proposal, "temporal_cutoff": CUTOFF}
    )
    retired_commit = opened.submit_fork_proposal(
        retired_proposal, [], retrieval_id=retired_found["retrieval_id"], recorded_at=RECORDED
    )
    blocked = _proposal(
        catalog,
        revision.content_hash,
        branch_steps=[
            {"block_id": "agent-a", "step_kind": "factual_block_ref"},
            {"imagined_node_id": "img-budget", "step_kind": "imagined_node"},
        ],
    )
    if (
        _code(
            lambda: opened.submit_fork_proposal(
                blocked, [], retrieval_id="ignored", recorded_at=RECORDED
            )
        )
        != "submit_budget_exhausted"
    ):
        raise AssertionError("submit budget stayed open")
    still = opened.submit_fork_proposal(
        proposal, [], retrieval_id=found["retrieval_id"], recorded_at=RECORDED
    )
    if still["fork_node_id"] != first["fork_node_id"]:
        raise AssertionError("idempotent submit was rejected after the budget")
    hops = _hops(
        root,
        overlays,
        patterns,
        guidance,
        catalog,
        revision,
        first["fork_node_id"],
        retired_commit["fork_node_id"],
        guided["guidance_id"],
    )
    return {
        "fork_node_id": first["fork_node_id"],
        "guidance_selected_agent": guided["selected_agent_id"],
        "idempotent_resubmit": True,
        "phase_locked_before_failure": True,
        "submit_budget": "submit_budget_exhausted",
        **hops,
    }


def _fail_signal() -> ExternalOutcomeEvidence:
    return ExternalOutcomeEvidence(
        binding_id="bind-eval-1",
        signal_kind="evaluator",
        signal="fail",
        evidence_ref="evidence:eval-1",
    )


def _edge(result: dict[str, Any], link_type: str) -> dict[str, Any]:
    rows = [edge for edge in result["edges"] if edge["link_type"] == link_type]
    if len(rows) != 1:
        raise AssertionError(f"{link_type} count={len(rows)}")
    return rows[0]


def _hops(
    root: Path,
    overlays,
    patterns,
    guidance,
    catalog,
    revision,
    fork_id: str,
    retired_fork: str,
    guidance_id: str,
) -> dict[str, Any]:
    links = ExperienceLinkService(root, overlays=overlays, patterns=patterns, guidance=guidance)
    anchors = _edge(
        links.expand(
            fork_id,
            link_types=["anchors"],
            principal_labels=(CAROL,),
            unresolved=("ps field",),
        ),
        "anchors",
    )
    if anchors["direction"] != "out" or anchors["target_id"] != "agent-b":
        raise AssertionError(f"anchor edge {anchors}")
    factual = _edge(
        links.expand(
            fork_id,
            link_types=["factual_ref"],
            principal_labels=(CAROL,),
            unresolved=("ps field",),
        ),
        "factual_ref",
    )
    if factual["direction"] != "out" or factual["target_id"] != "agent-a":
        raise AssertionError(f"factual edge {factual}")
    stored = overlays.read_overlay_by_id(fork_id, principal_labels=(CAROL,))
    patterns.accept_proposal(
        stored,
        catalog,
        _pattern_proposal(fork_id, "img-retry", "check the format field before counting"),
    )
    patterns.append_evaluation(_evaluation("img-retry", "eval-active", "evidence:active", "active"))
    supported = _edge(
        links.expand(
            "img-retry",
            link_types=["supports", "refutes"],
            principal_labels=(CAROL,),
            unresolved=("ps field",),
        ),
        "supports",
    )
    if supported["direction"] != "in":
        raise AssertionError("supports direction")
    patterns.append_evaluation(
        _evaluation("img-retry", "eval-invalid", "evidence:invalid", "invalidated")
    )
    refuted = _edge(
        links.expand(
            "img-retry",
            link_types=["supports", "refutes"],
            principal_labels=(CAROL,),
            unresolved=("ps field",),
        ),
        "refutes",
    )
    if refuted["status"] != "invalidated":
        raise AssertionError(refuted["status"])
    retired_stored = overlays.read_overlay_by_id(retired_fork, principal_labels=(CAROL,))
    patterns.accept_proposal(
        retired_stored,
        catalog,
        _pattern_proposal(retired_fork, "img-retired", "retire the old substring check"),
    )
    patterns.append_evaluation(
        _evaluation("img-retired", "eval-retired", "evidence:retired", "retired")
    )
    retired = _edge(
        links.expand(
            "img-retired",
            link_types=["supports"],
            principal_labels=(CAROL,),
            profile="reader",
            unresolved=("old",),
        ),
        "supports",
    )
    memory = links.expand(
        "img-retired",
        link_types=["supports"],
        principal_labels=(CAROL,),
        profile="memory",
        unresolved=("old",),
    )
    if retired["status"] != "retired" or memory["edges"]:
        raise AssertionError("retired flag did not follow the profile")
    hidden = links.expand(fork_id, link_types=["anchors"], principal_labels=(BOB,), unresolved=())
    if hidden["edges"]:
        raise AssertionError("bob saw carol's fork")
    if _code(
        lambda: links.expand("replay-rev:hidden", principal_labels=(CAROL,), unresolved=())
    ) != ("replay_provenance_excluded"):
        raise AssertionError("replay subject was reachable")
    replay_fork = _plant_replay(revision, overlays)
    replay_edges = links.expand(
        replay_fork, link_types=["anchors"], principal_labels=(CAROL,), unresolved=()
    )
    if replay_edges["edges"]:
        raise AssertionError("a replay revision stayed reachable")
    guided = _edge(
        links.expand(
            guidance_id,
            link_types=["guides"],
            principal_labels=(CAROL,),
            unresolved=("ps field",),
        ),
        "guides",
    )
    if guided["direction"] != "out" or guided["target_id"] != fork_id:
        raise AssertionError(f"guides edge {guided}")
    tight = ExperienceLinkService(root, overlays=overlays, patterns=patterns, reader_max_cards=1)
    if (
        _code(lambda: tight.expand(fork_id, principal_labels=(CAROL,), unresolved=()))
        != "hop_budget_exceeded"
    ):
        raise AssertionError("card budget returned a truncated page")
    hopper = ExperienceLinkService(root, overlays=overlays, patterns=patterns)
    context_id = None
    for _index in range(DEFAULT_READER_MAX_HOPS):
        step = hopper.expand(
            "nobody", principal_labels=(CAROL,), context_id=context_id, unresolved=()
        )
        context_id = step["context_id"]
    if step["hops"] != DEFAULT_READER_MAX_HOPS:
        raise AssertionError(f"hops={step['hops']}")
    if (
        _code(
            lambda: hopper.expand(
                "nobody", principal_labels=(CAROL,), context_id=context_id, unresolved=()
            )
        )
        != "hop_budget_exceeded"
    ):
        raise AssertionError("hop budget stayed open")
    if (DEFAULT_READER_MAX_HOPS, DEFAULT_READER_MAX_CARDS) != (8, 32):
        raise AssertionError("reader budget defaults moved")
    if (DEFAULT_MEMORY_MAX_HOPS, DEFAULT_MEMORY_MAX_CARDS) != (4, 16):
        raise AssertionError("memory budget defaults moved")
    audit = (root / "t-fork-audit" / "attempts.jsonl").read_text(encoding="utf-8")
    if "ps field" not in (root / "experience-link-audit" / "hops.jsonl").read_text(
        encoding="utf-8"
    ):
        raise AssertionError("unresolved context was not audited")
    if "phase_locked" not in audit:
        raise AssertionError("phase lock was not audited")
    return {
        "acl_filtered": True,
        "anchor_direction": anchors["direction"],
        "anchor_target": anchors["target_id"],
        "card_budget": "hop_budget_exceeded",
        "factual_target": factual["target_id"],
        "guides_target_is_fork": True,
        "hop_budget": "hop_budget_exceeded",
        "memory_max_cards": DEFAULT_MEMORY_MAX_CARDS,
        "memory_max_hops": DEFAULT_MEMORY_MAX_HOPS,
        "reader_max_cards": DEFAULT_READER_MAX_CARDS,
        "reader_max_hops": DEFAULT_READER_MAX_HOPS,
        "refutes_status": refuted["status"],
        "replay_code": "replay_provenance_excluded",
        "replay_overlay_hidden": True,
        "retired_memory_hidden": True,
        "retired_status": retired["status"],
        "supports_before_invalidation": True,
    }


def _plant_replay(root_revision, overlays) -> str:
    replay = freeze_native_edge_view(
        session_id=SESSION,
        task_run_id="run-replay",
        view_id="view-replay",
        revision_id="replay-rev:hidden",
        frozen_at=FROZEN_AT,
        blocks=root_revision.blocks,
        edges=root_revision.edges,
        coverage=root_revision.coverage,
    )
    replay_catalog = overlays.register_base_revision(replay, acl_labels=(CAROL,))
    stored = overlays.submit_proposal(
        _proposal(replay_catalog, replay.content_hash), principal_labels=(CAROL,)
    )
    return stored.stored.fork_node_id


def _single_pass(revision) -> dict[str, str]:
    run = AigConstructionRun("run-single")
    if run.construction_mode != CONSTRUCTION_MODE_NATIVE_SINGLE_PASS:
        raise AssertionError(run.construction_mode)
    frozen = freeze_native_edge_view(
        session_id=SESSION,
        task_run_id="run-single",
        view_id="view-single",
        revision_id="rev-single",
        frozen_at=FROZEN_AT,
        blocks=revision.blocks,
        edges=revision.edges,
        coverage=revision.coverage,
    )
    if frozen.construction_mode != CONSTRUCTION_MODE_NATIVE_SINGLE_PASS:
        raise AssertionError(frozen.construction_mode)
    return {
        "freeze_default": frozen.construction_mode,
        "run_default": run.construction_mode,
    }


def _report(root: Path) -> dict[str, Any]:
    built = _build(root)
    looped = _loops(root)
    compiled, surface, snapshot = looped.pop("pipeline")
    compiled_view = looped.pop("compile")
    read = _read(root, compiled, surface, snapshot)
    run = read.pop("run")
    forked = _fork_and_hops(root, compiled.revision, run, read["reader_report_hash"])
    single = _single_pass(compiled.revision)
    return {
        "assertions": {
            **built,
            **dict(looped.items()),
            **compiled_view,
            **read,
            **forked,
            **single,
        },
        "frozen_at": FROZEN_AT,
        "schema": SCHEMA,
    }


def _text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="ADR-0044 Wave 3 structural matrix")
    parser.add_argument("--out", type=Path, default=Path("/tmp/aig0044-toolloop/report.json"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for line in FROZEN_RULES:
        print(line)
    try:
        with tempfile.TemporaryDirectory(prefix="aig0044-") as directory:
            payload = _report(Path(directory))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None
    rendered = _text(payload)
    if args.resume:
        if not args.out.is_file():
            print(f"FAIL resume missing {args.out}", file=sys.stderr)
            raise SystemExit(1)
        previous = args.out.read_text(encoding="utf-8")
        if previous != rendered:
            print("FAIL resume report drifted", file=sys.stderr)
            raise SystemExit(1)
        print(f"RESUME {args.out}")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        print(f"WROTE {args.out}")
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    print(f"REPORT_SHA256 {digest}")
    print(f"CRITICAL {payload['assertions']['critical_block_id']}")
    print(f"FORK {payload['assertions']['fork_node_id']}")


if __name__ == "__main__":
    main()
