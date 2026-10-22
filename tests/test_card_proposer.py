# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ADR-0046: post-freeze skill_card proposals from the production pipeline."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path

from openviking.session.aig_production_pipeline import AigRoleProviders, run_production_aig
from openviking.session.aig_toolloop_builder import FakeToolloopProvider
from openviking.session.aig_toolloop_critic import FakeCriticProvider
from openviking.session.aig_toolloop_reader import DeterministicReader
from openviking.session.aig_toolloop_refiner import FakeRefinerProvider
from openviking.session.card_proposer import ProposerTurn, outcome_task_score, run_card_proposer
from openviking.session.experience_cards import ExperienceCardStore
from openviking.session.fork_node_overlay import ForkNodeOverlayStore
from openviking.session.outcome_signal import OutcomeSignal
from tests.test_aig_production_pipeline import _script, _snapshot

_OWNER = "user:alice"


def _pass(reward: float, source: str = "verifier") -> OutcomeSignal:
    return OutcomeSignal(signal="pass", source=source, raw={"reward": reward})


def _providers() -> AigRoleProviders:
    return AigRoleProviders(
        builder=FakeToolloopProvider(_script()),
        critic=FakeCriticProvider(),
        refiner=FakeRefinerProvider(),
        reader=DeterministicReader(),
    )


def _proposal(turn: ProposerTurn, **over) -> dict:
    blocks = list(turn.block_aos)
    first, last = blocks[0], blocks[-1]
    body = {
        "kind": "skill_card",
        "base_revision_id": turn.base_revision_id,
        "base_revision_block_content_hash": turn.base_revision_block_content_hash,
        "report_hash": turn.report_hash,
        "boundary": "只适用于修复解析器类任务",
        "gamma": {"deliverable": "text", "abstract_text": "修复解析器并核对输出", "anchors": [first]},
        "steps": [
            {"step_id": "s1", "step_kind": "produce", "statement": "定位分支", "block_id": first},
            {"step_id": "s2", "step_kind": "verify", "statement": "核对输出", "block_id": last,
             "issue_kind": "incorrect", "check_how": "对照期望输出",
             "violation_sign": "输出为零"},
        ],
    }
    body.update(over)
    return body


def _run(tmp_path: Path, outcome: OutcomeSignal, provider, cards: Path):
    snapshot = replace(_snapshot(), outcome=outcome)
    hook = partial(run_card_proposer_adapter, provider=provider, card_root=cards)
    return run_production_aig(
        snapshot, root=tmp_path, task_run_id="task-p", frozen_at="2026-10-14T00:00:00Z",
        providers=_providers(), card_hook=hook,
    )


def run_card_proposer_adapter(revision, snapshot, read, *, provider, card_root):
    return run_card_proposer(
        revision, snapshot, read, provider, card_root=card_root, owner_label=_OWNER
    )


def test_score_gate_only_accepts_verifier_pass_at_threshold() -> None:
    assert outcome_task_score(_pass(0.8)) == 0.8
    assert outcome_task_score(_pass(0.59)) is None
    assert outcome_task_score(_pass(0.9, "model_inferred")) is None
    assert outcome_task_score(OutcomeSignal("fail", "verifier", {"reward": 0.9})) is None
    assert outcome_task_score(None) is None


def test_pipeline_writes_a_card_and_binds_score(tmp_path: Path) -> None:
    cards = tmp_path / "user-root"
    seen: list[ProposerTurn] = []

    def provider(turn: ProposerTurn):
        seen.append(turn)
        return [_proposal(turn, task_score=0.1, author_role="critic")]

    result = _run(tmp_path, _pass(0.85), provider, cards)
    assert result.status == "frozen" and len(result.card_ids) == 1
    store = ExperienceCardStore(cards, ForkNodeOverlayStore(cards))
    card = store.read(result.card_ids[0], principal_labels=[_OWNER])
    assert card is not None and card.task_score == 0.85 and card.author_role == "reader"
    assert seen[0].allowed_kinds == ("skill_card",)
    assert store.find(principal_labels=["user:bob"]).hits == ()


def test_low_score_failed_and_bad_proposals_never_change_the_pipeline(tmp_path: Path) -> None:
    calls: list[int] = []

    def provider(turn: ProposerTurn):
        calls.append(1)
        return [_proposal(turn)]

    low = _run(tmp_path / "a", _pass(0.4), provider, tmp_path / "cards-a")
    assert low.status == "frozen" and low.card_ids == () and calls == []

    def bad(turn: ProposerTurn):
        return [_proposal(turn, steps=[{"step_id": "s1", "step_kind": "produce",
                                        "statement": "x", "block_id": "no-such-block"}]),
                {**_proposal(turn), "kind": "fork"}]

    out = _run(tmp_path / "b", _pass(0.9), bad, tmp_path / "cards-b")
    assert out.status == "frozen" and out.card_ids == ()

    def boom(turn: ProposerTurn):
        raise RuntimeError("provider down")

    out = _run(tmp_path / "c", _pass(0.9), boom, tmp_path / "cards-c")
    assert out.status == "frozen" and out.card_ids == ()
