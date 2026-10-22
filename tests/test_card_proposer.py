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


# ---- commit wiring ------------------------------------------------------

def _result(signal: str = "pass", **extra) -> dict:
    return {"binding_id": "b1", "signal_kind": "evaluator", "signal": signal,
            "evidence_ref": "evidence:1", **extra}


def test_evaluator_reward_is_optional_validated_and_carried() -> None:
    import pytest

    from openviking.session.aig_commit_wiring import commit_outcome_signal
    from openviking.session.outcome_contract import (
        OutcomeContractError,
        external_evidence_from_session,
    )
    from types import SimpleNamespace

    with_reward = SimpleNamespace(outcome_evaluator_results=[_result(reward=0.8)])
    signal = commit_outcome_signal(with_reward)
    assert signal is not None and signal.raw["reward"] == 0.8
    assert outcome_task_score(signal) == 0.8
    plain = commit_outcome_signal(SimpleNamespace(outcome_evaluator_results=[_result()]))
    assert plain is not None and "reward" not in plain.raw
    assert outcome_task_score(plain) is None
    two = SimpleNamespace(outcome_evaluator_results=[
        _result(reward=0.9), {**_result(reward=0.7), "binding_id": "b2"}])
    assert commit_outcome_signal(two).raw["reward"] == 0.7  # weakest evaluator decides
    for bad in (1.5, -0.1, True, "0.9"):
        with pytest.raises(OutcomeContractError):
            external_evidence_from_session(evaluator_results=[_result(reward=bad)])
    with pytest.raises(OutcomeContractError):
        external_evidence_from_session(evaluator_results=[_result(extra=1)])


def test_vlm_card_proposer_parses_strictly() -> None:
    import asyncio
    import json
    import time

    import pytest

    from openviking.session.aig_vlm_roles import VLMCardProposer

    class Handle:
        def __init__(self, content: str) -> None:
            self.content = content

        async def get_completion_async(self, **_kw):
            await asyncio.sleep(0)
            return self.content

    turn = ProposerTurn("p", {}, "r", "h", "d", {}, (), "pass", ("skill_card",))
    ok = VLMCardProposer(Handle(json.dumps({"proposals": [{"kind": "skill_card"}]})),
                         time.monotonic() + 30)
    assert ok(turn) == [{"kind": "skill_card"}]
    for body in ('{"proposals": "x"}', '{"proposals": [1]}', '{"other": []}', "not json"):
        with pytest.raises(Exception):
            VLMCardProposer(Handle(body), time.monotonic() + 30)(turn)


def test_card_hook_is_off_by_default_and_needs_a_user_root(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from openviking.session.aig_commit_wiring import card_hook_for_session, user_root_for_session

    session = SimpleNamespace(session_id="s", ctx=None, _viking_fs=None)
    assert card_hook_for_session(session, object()) is None
    session.experience_cards_enabled = True
    assert user_root_for_session(session) is None
    assert card_hook_for_session(session, object()) is None  # no user root, no hook


def test_card_hook_env_opt_in_enables_cards_for_every_session(monkeypatch) -> None:
    from types import SimpleNamespace

    from openviking.session import aig_commit_wiring

    monkeypatch.setattr(aig_commit_wiring, "user_root_for_session", lambda session: Path("/root"))
    monkeypatch.setattr(aig_commit_wiring, "session_owner_label", lambda session: "user:alice")
    session = SimpleNamespace(session_id="s")
    monkeypatch.delenv("OPENVIKING_EXPERIENCE_CARDS", raising=False)
    assert aig_commit_wiring.card_hook_for_session(session, object()) is None
    monkeypatch.setenv("OPENVIKING_EXPERIENCE_CARDS", "1")
    assert aig_commit_wiring.card_hook_for_session(session, object()) is not None


def test_card_hook_runs_through_a_session_and_writes_under_the_user_root(tmp_path: Path) -> None:
    import asyncio
    import json
    from types import SimpleNamespace

    from openviking.session import aig_vlm_roles
    from openviking.session.aig_commit_wiring import card_hook_for_session

    user_root = tmp_path / "user-alice"
    session = SimpleNamespace(
        session_id="s",
        ctx=SimpleNamespace(user=SimpleNamespace(user_id="alice")),
        _viking_fs=SimpleNamespace(_uri_to_path=lambda uri, ctx=None: str(user_root)),
        experience_cards_enabled=True,
    )
    captured: dict = {}

    class Handle:
        async def get_completion_async(self, **kw):
            await asyncio.sleep(0)
            state = json.loads(kw["messages"][1]["content"])
            captured["state"] = state
            first, last = list(state["block_aos"])[0], list(state["block_aos"])[-1]
            return json.dumps({"proposals": [_proposal_from_state(state, first, last)]})

    hook = card_hook_for_session(session, Handle())
    assert hook is not None and aig_vlm_roles.VLMCardProposer
    snapshot = replace(_snapshot(), outcome=_pass(0.9))
    result = run_production_aig(
        snapshot, root=tmp_path / "session", task_run_id="task-h",
        frozen_at="2026-10-14T00:00:00Z", providers=_providers(), card_hook=hook,
    )
    assert result.status == "frozen" and len(result.card_ids) == 1
    store = ExperienceCardStore(user_root, ForkNodeOverlayStore(user_root))
    assert store.read(result.card_ids[0], principal_labels=["user:alice"]) is not None
    assert captured["state"]["allowed_kinds"] == ["skill_card"]


def _proposal_from_state(state: dict, first: str, last: str) -> dict:
    return {
        "kind": "skill_card",
        "base_revision_id": state["base_revision_id"],
        "base_revision_block_content_hash": state["base_revision_block_content_hash"],
        "report_hash": state["report_hash"],
        "boundary": "仅适用于解析器修复",
        "gamma": {"deliverable": "text", "abstract_text": "修复解析器", "anchors": [first]},
        "steps": [
            {"step_id": "s1", "step_kind": "produce", "statement": "定位", "block_id": first},
            {"step_id": "s2", "step_kind": "verify", "statement": "核对", "block_id": last,
             "issue_kind": "incorrect", "check_how": "对照输出", "violation_sign": "为零"},
        ],
    }
