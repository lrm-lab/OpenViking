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


def _cite(turn: ProposerTurn, block: str) -> list[str]:
    """An AO inside ``block`` that the Reader read, or nothing."""
    read = set(turn.read_aos)
    return [ao for ao in turn.block_aos[block] if ao in read][:1]


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
            {"step_id": "s1", "step_kind": "produce", "statement": "定位分支", "block_id": first,
             "ao_refs": _cite(turn, first)},
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


def _fail(source: str = "verifier", **raw) -> OutcomeSignal:
    return OutcomeSignal(signal="fail", source=source, raw={"reward": 0.1, **raw})


def _fork(turn: ProposerTurn, **over) -> dict:
    blocks = list(turn.block_aos)
    first = blocks[0]
    body = {
        "kind": "fork",
        "anchor_block_id": first,
        "base_revision_id": turn.base_revision_id,
        "base_revision_block_content_hash": turn.base_revision_block_content_hash,
        "report_hash": turn.report_hash,
        "boundary": "只适用于解析器类任务",
        "gamma": {"deliverable": "text", "issue_kind": "incorrect",
                  "abstract_text": "解析器遇到空输入时直接返回零", "anchors": [blocks[-1]]},
        "steps": [{"step_id": "f1", "step_kind": "produce", "statement": "先处理空输入再解析",
                   "block_id": first, "ao_refs": _cite(turn, first)}],
        # The model does not get to pick these.
        "fork_node_id": "f" * 64, "violates": {"card_id": "x", "step_id": "y"},
        "diagnosis_report_id": "d",
    }
    body.update(over)
    return body


def test_failed_verifier_run_writes_a_fork_overlay_and_card(tmp_path: Path) -> None:
    cards = tmp_path / "user-root"
    seen: list[ProposerTurn] = []

    def provider(turn: ProposerTurn):
        seen.append(turn)
        return [_fork(turn)]

    out = _run(tmp_path, _fail(feedback="缺少空输入处理"), provider, cards)
    assert out.status == "frozen" and len(out.card_ids) == 1
    turn = seen[0]
    assert turn.allowed_kinds == ("fork",) and turn.outcome_feedback == "缺少空输入处理"
    assert turn.outcome_digest.startswith("fail")
    overlays = ForkNodeOverlayStore(cards)
    store = ExperienceCardStore(cards, overlays)
    card = store.read(out.card_ids[0], principal_labels=[_OWNER])
    assert card is not None and card.kind == "fork" and card.task_score is None
    assert card.fork_node_id != "f" * 64 and card.violates is None and card.diagnosis_report_id is None
    assert card.gamma.anchors[-1] == list(turn.block_aos)[0]  # the anchor is added by the server
    stored = overlays.read_overlay_by_id(card.fork_node_id, principal_labels=[_OWNER])
    assert stored is not None and stored.overlay.anchor_block_id == list(turn.block_aos)[0]
    assert [s.imagined_node_id for s in stored.overlay.branch_steps] == ["img-f1"]
    found = store.find(principal_labels=[_OWNER], kind="fork")
    assert [h.card.card_id for h in found.hits] == [card.card_id]


def test_fork_needs_a_verifier_fail_and_a_skill_needs_a_pass(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def provider(turn: ProposerTurn):
        calls.append(turn.allowed_kinds)
        return [_fork(turn), _proposal(turn)]

    inferred = _run(tmp_path / "a", _fail("model_inferred"), provider, tmp_path / "ca")
    assert inferred.card_ids == () and calls == []
    failed = _run(tmp_path / "b", _fail(), provider, tmp_path / "cb")
    # The skill_card in the same list is refused on a failed run; the fork lands.
    assert len(failed.card_ids) == 1 and calls == [("fork",)]
    passed = _run(tmp_path / "c", _pass(0.9), provider, tmp_path / "cc")
    assert len(passed.card_ids) == 1 and calls[-1] == ("skill_card",)


def test_card_without_a_cited_ao_or_with_a_bad_anchor_is_not_stored(tmp_path: Path) -> None:
    def uncited(turn: ProposerTurn):
        body = _proposal(turn)
        body["steps"][0].pop("ao_refs")
        fork = _fork(turn)
        fork["steps"][0].pop("ao_refs")
        return [body, fork]

    assert _run(tmp_path / "a", _pass(0.9), uncited, tmp_path / "ca").card_ids == ()
    assert _run(tmp_path / "b", _fail(), uncited, tmp_path / "cb").card_ids == ()

    def bad_anchor(turn: ProposerTurn):
        return [_fork(turn, anchor_block_id="no-such-block"), _fork(turn, anchor_block_id=None)]

    cards = tmp_path / "cc"
    assert _run(tmp_path / "c", _fail(), bad_anchor, cards).card_ids == ()
    # A refused card leaves no overlay behind.
    overlay_dir = cards / "fork-node-overlays" / "overlays"
    assert not overlay_dir.exists() or list(overlay_dir.glob("*.json")) == []


def test_vlm_card_proposer_sends_an_uncited_card_back_once(tmp_path: Path) -> None:
    import asyncio
    import json
    import time

    from openviking.session.aig_vlm_roles import VLMCardProposer

    good = {"kind": "skill_card", "steps": [{"step_id": "s1", "block_id": "b", "ao_refs": ["ao-0"]}]}
    bad = {"kind": "skill_card", "steps": [{"step_id": "s1", "block_id": "b"}]}
    wrong = {"kind": "skill_card", "steps": [{"step_id": "s1", "block_id": "b", "ao_refs": ["ao-9"]}]}

    class Handle:
        def __init__(self, replies: list[dict]) -> None:
            self.replies = replies
            self.messages: list[list[dict]] = []

        async def get_completion_async(self, **kw):
            await asyncio.sleep(0)
            self.messages.append(kw["messages"])
            return json.dumps({"proposals": [self.replies[len(self.messages) - 1]]})

    turn = ProposerTurn("p", {}, "r", "h", "d", {"b": ("ao-0", "ao-1")}, ("ao-0",), "pass",
                        ("skill_card",))
    handle = Handle([bad, good])
    assert VLMCardProposer(handle, time.monotonic() + 30)(turn) == [good]
    assert len(handle.messages) == 2 and "no step cites an AO" in handle.messages[1][-1]["content"]
    handle = Handle([wrong, good])
    assert VLMCardProposer(handle, time.monotonic() + 30)(turn) == [good]
    assert "AOs you may cite there: ['ao-0']" in handle.messages[1][-1]["content"]


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
    read = set(state["read_aos"])
    cite = [ao for ao in state["block_aos"][first] if ao in read][:1]
    return {
        "kind": "skill_card",
        "base_revision_id": state["base_revision_id"],
        "base_revision_block_content_hash": state["base_revision_block_content_hash"],
        "report_hash": state["report_hash"],
        "boundary": "仅适用于解析器修复",
        "gamma": {"deliverable": "text", "abstract_text": "修复解析器", "anchors": [first]},
        "steps": [
            {"step_id": "s1", "step_kind": "produce", "statement": "定位", "block_id": first,
             "ao_refs": cite},
            {"step_id": "s2", "step_kind": "verify", "statement": "核对", "block_id": last,
             "issue_kind": "incorrect", "check_how": "对照输出", "violation_sign": "为零"},
        ],
    }


def _trace_handle(replies: list[str]):
    import asyncio

    class Handle:
        def __init__(self) -> None:
            self.n = 0

        async def get_completion_async(self, **_kw):
            await asyncio.sleep(0)
            reply = replies[min(self.n, len(replies) - 1)]
            self.n += 1
            return reply

    return Handle()


def _trace_turn() -> ProposerTurn:
    return ProposerTurn("p", {}, "r", "h", "d", {"b": ("ao-0",)}, ("ao-0",), "pass", ("skill_card",))


def test_provider_records_every_attempt_and_why_it_was_rejected() -> None:
    import json
    import time

    from openviking.session.aig_vlm_roles import VLMCardProposer

    bad = json.dumps({"proposals": [{"kind": "skill_card", "steps": [{"step_id": "s", "block_id": "b"}]}]})
    empty = json.dumps({"proposals": []})
    provider = VLMCardProposer(_trace_handle([bad, empty]), time.monotonic() + 30)
    assert provider(_trace_turn()) == []
    assert [a["result"] for a in provider.attempts] == ["rejected", "accepted"]
    assert "no step cites an AO" in provider.attempts[0]["error"]
    assert provider.attempts[1]["proposals"] == 0


def test_run_result_tells_empty_answer_from_exhausted_retries(tmp_path: Path) -> None:
    import json
    import time

    from openviking.session.aig_vlm_roles import VLMCardProposer

    def run(replies: list[str]):
        cycle = run_production_aig(
            replace(_snapshot(), outcome=_pass(0.9)),
            root=tmp_path / str(len(list(tmp_path.iterdir()))), task_run_id="t",
            providers=_providers(),
            card_hook=lambda rev, snap, read: run_card_proposer(
                rev, snap, read, VLMCardProposer(_trace_handle(replies), time.monotonic() + 30),
                card_root=tmp_path / f"cards-{len(replies)}-{replies[0][:3]}", owner_label=_OWNER),
        )
        return cycle

    empty = run([json.dumps({"proposals": []})])
    assert empty.status == "frozen" and empty.card_ids == ()
    assert empty.card_trace["verdict"] == "model_returned_empty"
    assert empty.card_trace["attempts"][0]["result"] == "accepted"

    exhausted = run(["not json"])
    assert exhausted.status == "frozen", (exhausted.status, exhausted.reason)
    assert exhausted.card_trace["verdict"] == "retries_exhausted"
    assert [a["result"] for a in exhausted.card_trace["attempts"]] == ["rejected"] * 3
    assert exhausted.card_trace["error"].startswith("JSONDecodeError")
