# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""T_fork verbs: phase lock, precedent chain, idempotency, and submit budget."""

from __future__ import annotations

from pathlib import Path

from openviking.session.aig_role_chain import AigConstructionRun
from openviking.session.outcome_contract import OutcomeAssessmentService, OutcomeContractStore
from openviking.session.t_fork_surface import ACTION_VERBS, TForkError, TForkSurface
from tests.test_fork_node_overlay import _proposal
from tests.test_guidance_approval import _branch, _candidate, _catalog, _world
from tests.test_outcome_contract import _signal

_CAROL = "user:carol"
_CUTOFF = "2026-10-10T00:00:01.000Z"
_RECORDED = "2026-10-10T00:00:02.000Z"


def _surface(world, tmp_path: Path, *, evidence: tuple = ()) -> TForkSurface:
    outcomes = OutcomeAssessmentService(OutcomeContractStore(tmp_path))
    return TForkSurface(
        tmp_path,
        run=world.run,
        session_id=world.sessions["reader"],
        task_id=world.contract.task_id,
        report_hash=world.report_hash,
        principal_labels=(_CAROL,),
        overlays=world.overlays,
        patterns=world.patterns,
        precedents=world.precedents,
        guidance=world.service,
        outcomes=outcomes,
        external_evidence=evidence,
    )


def _code(call) -> str:
    try:
        call()
    except TForkError as exc:
        return exc.code
    raise AssertionError("expected a coded error")


def test_action_verbs_stay_locked_until_confirmed_failure(tmp_path: Path) -> None:
    world = _world(tmp_path)
    locked = _surface(world, tmp_path)
    names = [name for name, _text in locked.catalog()]
    assert names == ["expand_experience_links"]
    query = {"proposal": world.fork_payload, "temporal_cutoff": _CUTOFF}
    assert _code(lambda: locked.search_precedents(query)) == "phase_locked"
    assert _code(
        lambda: locked.submit_fork_proposal(
            world.fork_payload, [], retrieval_id="ab", recorded_at=_RECORDED
        )
    ) == ("phase_locked")
    opened = locked.expand_experience_links(world.fork_node_id, unresolved=("reader-gap",))
    assert opened["edges"]
    audit = (tmp_path / "t-fork-audit" / "attempts.jsonl").read_text(encoding="utf-8")
    assert "reader-gap" in audit
    assert "phase_locked" in audit
    assert (
        _code(lambda: locked.call("expand_experience_links", {"subject_id": "x"}))
        == "unresolved_required"
    )


def test_retrieve_proposal_guidance_idempotency_and_submit_budget(tmp_path: Path) -> None:
    world = _world(tmp_path)
    surface = _surface(world, tmp_path, evidence=(_signal("fail"),))
    names = [name for name, _text in surface.catalog()]
    assert names[0] == "expand_experience_links"
    assert tuple(names[1:]) == ACTION_VERBS
    plan = {"catalog": _catalog(world, [_candidate()]), "proposal": _branch(world)}
    assert _code(lambda: surface.submit_guidance_plan(plan)) == "fork_required_first"
    found = surface.search_precedents({"proposal": world.fork_payload, "temporal_cutoff": _CUTOFF})
    assert found["retrieval_id"] == world.retrieval.retrieval_id
    assert found["high_confidence"] == []
    assert found["anchor_block_id"] == "b-2"
    first = surface.submit_fork_proposal(
        world.fork_payload,
        [],
        retrieval_id=found["retrieval_id"],
        recorded_at=_RECORDED,
    )
    assert first["fork_node_id"] == world.fork_node_id
    assert first["deduplicated"] is True
    again = surface.submit_fork_proposal(
        world.fork_payload,
        [],
        retrieval_id=found["retrieval_id"],
        recorded_at=_RECORDED,
    )
    assert again["fork_node_id"] == first["fork_node_id"]
    assert again["deduplicated"] is True
    guided = surface.submit_guidance_plan(plan)
    assert guided["source_fork_id"] == world.fork_node_id
    assert guided["selected_agent_id"] == "agent-a"
    document_hash = world.fork_payload["body"]["report_hash"]
    conflict = _proposal(
        world.base,
        document_hash=document_hash,
        branch_steps=[
            {"block_id": "b-1", "step_kind": "factual_block_ref"},
            {"imagined_node_id": "img-other", "step_kind": "imagined_node"},
        ],
    )
    conflict_retrieval = surface.search_precedents(
        {"proposal": conflict, "temporal_cutoff": _CUTOFF}
    )
    assert (
        _code(
            lambda: surface.submit_fork_proposal(
                conflict,
                [],
                retrieval_id=conflict_retrieval["retrieval_id"],
                recorded_at=_RECORDED,
            )
        )
        == "overlay_conflict"
    )
    fresh = _proposal(
        world.base,
        document_hash=document_hash,
        anchor_block_id="b-1",
        branch_steps=[
            {"block_id": "b-2", "step_kind": "factual_block_ref"},
            {"imagined_node_id": "img-b1", "step_kind": "imagined_node"},
        ],
    )
    fresh_retrieval = surface.search_precedents({"proposal": fresh, "temporal_cutoff": _CUTOFF})
    stored = surface.submit_fork_proposal(
        fresh,
        [],
        retrieval_id=fresh_retrieval["retrieval_id"],
        recorded_at=_RECORDED,
    )
    assert stored["deduplicated"] is False
    assert stored["fork_node_id"] != world.fork_node_id
    blocked = _proposal(
        world.base,
        document_hash=document_hash,
        branch_steps=[
            {"block_id": "b-1", "step_kind": "factual_block_ref"},
            {"imagined_node_id": "img-budget", "step_kind": "imagined_node"},
        ],
    )
    assert (
        _code(
            lambda: surface.submit_fork_proposal(
                blocked,
                [],
                retrieval_id="ignored",
                recorded_at=_RECORDED,
            )
        )
        == "submit_budget_exhausted"
    )
    still = surface.submit_fork_proposal(
        world.fork_payload,
        [],
        retrieval_id=found["retrieval_id"],
        recorded_at=_RECORDED,
    )
    assert still["fork_node_id"] == world.fork_node_id
    assert still["deduplicated"] is True
    early = TForkSurface(
        tmp_path / "early",
        run=AigConstructionRun("run-early-fork"),
        session_id="sess-early",
        task_id="task-early",
        report_hash="ab" * 32,
        principal_labels=(_CAROL,),
        overlays=world.overlays,
        patterns=world.patterns,
        precedents=world.precedents,
        guidance=world.service,
        outcomes=OutcomeAssessmentService(OutcomeContractStore(tmp_path)),
    )
    assert early.catalog() == ()
    assert _code(lambda: early.expand_experience_links("b-2", unresolved=())) == "phase_locked"
