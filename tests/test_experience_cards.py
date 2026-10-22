# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ADR-0046: skill_card and fork experience cards."""

from __future__ import annotations

from pathlib import Path

import pytest

from openviking.session.experience_cards import (
    CARD_KIND_FORK,
    CARD_KIND_SKILL,
    ExperienceCardError,
    ExperienceCardStore,
)
from openviking.session.fork_node_overlay import ForkNodeOverlayStore
from tests.test_fork_node_overlay import _native_revision, _proposal

_OWNER = ["user:alice"]
_OTHER = ["user:bob"]


def _setup(tmp_path: Path):
    overlays = ForkNodeOverlayStore(tmp_path)
    revision = _native_revision()
    catalog = overlays.register_base_revision(revision, acl_labels=_OWNER)
    cards = ExperienceCardStore(tmp_path, overlays)
    return overlays, revision, catalog, cards


def _skill(catalog, revision, **over) -> dict:
    b1, b2 = catalog.block_ids[:2]
    body = {
        "kind": CARD_KIND_SKILL,
        "author_role": "reader",
        "base_revision_id": catalog.base_revision_id,
        "base_revision_block_content_hash": catalog.block_content_hash,
        "report_hash": revision.content_hash,
        "task_score": 0.8,
        "boundary": "只适用于需要交付 pdf 报告的任务",
        "gamma": {
            "deliverable": "pdf",
            "occupation": "Financial Analyst",
            "issue_kind": "missing_section",
            "abstract_text": "生成 pdf 财务报告并检查章节完整",
            "anchors": [b1],
        },
        "steps": [
            {"step_id": "s1", "step_kind": "produce", "statement": "写出报告", "block_id": b1},
            {
                "step_id": "s2", "step_kind": "verify", "statement": "核对章节",
                "block_id": b2, "issue_kind": "missing_section",
                "check_how": "逐项对照任务要求的章节清单",
                "violation_sign": "缺少任一必需章节", "ao_refs": ["ao-2"],
            },
        ],
    }
    body.update(over)
    return body


def _submit(cards, payload, labels=_OWNER, **kw):
    return cards.submit(payload, principal_labels=labels, block_aos={"b-1": ["ao-1"], "b-2": ["ao-2"]}, **kw)


def test_skill_card_roundtrip_and_idempotent(tmp_path: Path) -> None:
    _, rev, cat, cards = _setup(tmp_path)
    card, dup = _submit(cards, _skill(cat, rev))
    assert not dup and card.kind == CARD_KIND_SKILL
    again, dup2 = _submit(cards, _skill(cat, rev))
    assert dup2 and again.card_id == card.card_id
    assert cards.read(card.card_id, principal_labels=_OWNER) == card


def test_score_and_verify_gates(tmp_path: Path) -> None:
    _, rev, cat, cards = _setup(tmp_path)
    with pytest.raises(ExperienceCardError) as low:
        _submit(cards, _skill(cat, rev, task_score=0.5))
    assert low.value.code == "score_gate"
    only_produce = _skill(cat, rev)
    only_produce["steps"] = only_produce["steps"][:1]
    with pytest.raises(ExperienceCardError) as none:
        _submit(cards, only_produce)
    assert none.value.code == "no_verify_or_recovery"
    only_produce["steps"][0]["recovery"] = True
    _submit(cards, only_produce)


def test_block_and_ao_grounding(tmp_path: Path) -> None:
    _, rev, cat, cards = _setup(tmp_path)
    bad_block = _skill(cat, rev)
    bad_block["steps"][0]["block_id"] = "b-missing"
    with pytest.raises(ExperienceCardError) as e1:
        _submit(cards, bad_block)
    assert e1.value.code == "anchor_unknown"
    bad_ao = _skill(cat, rev)
    bad_ao["steps"][1]["ao_refs"] = ["ao-1"]  # ao-1 belongs to b-1, not b-2
    with pytest.raises(ExperienceCardError) as e2:
        _submit(cards, bad_ao)
    assert e2.value.code == "ao_not_in_block"
    with pytest.raises(ExperienceCardError) as e3:
        cards.submit(_skill(cat, rev), principal_labels=_OWNER)  # no AO index
    assert e3.value.code == "ao_index_required"
    with pytest.raises(ExperienceCardError) as e4:
        _submit(cards, _skill(cat, rev), read_aos=["ao-1"])
    assert e4.value.code == "ao_not_read"
    no_ao = _skill(cat, rev)
    del no_ao["steps"][1]["ao_refs"]
    cards.submit(no_ao, principal_labels=_OWNER)  # AO refs are optional


def test_acl_and_author(tmp_path: Path) -> None:
    _, rev, cat, cards = _setup(tmp_path)
    with pytest.raises(ExperienceCardError) as denied:
        _submit(cards, _skill(cat, rev), labels=_OTHER)
    assert denied.value.code == "acl_denied"
    with pytest.raises(ExperienceCardError) as role:
        _submit(cards, _skill(cat, rev, author_role="critic"))
    assert role.value.code == "author_role"
    card, _ = _submit(cards, _skill(cat, rev))
    assert cards.read(card.card_id, principal_labels=_OTHER) is None
    assert cards.visible_cards(principal_labels=_OTHER) == ()


def _fork_overlay(overlays, catalog, revision):
    from openviking.session.fork_node_overlay import ForkNodeOverlay  # noqa: F401

    payload = _proposal(catalog, document_hash=revision.content_hash)
    return overlays.submit_proposal(payload, principal_labels=_OWNER).stored


def test_fork_card_links_overlay_and_violates(tmp_path: Path) -> None:
    overlays, rev, cat, cards = _setup(tmp_path)
    skill, _ = _submit(cards, _skill(cat, rev))
    stored = _fork_overlay(overlays, cat, rev)
    fork = {
        "kind": CARD_KIND_FORK, "author_role": "reader",
        "base_revision_id": cat.base_revision_id,
        "base_revision_block_content_hash": cat.block_content_hash,
        "report_hash": rev.content_hash, "boundary": "仅在缺少章节时",
        "fork_node_id": stored.fork_node_id, "diagnosis_report_id": "diag-1",
        "violates": {"card_id": skill.card_id, "step_id": "s2"},
        "gamma": {
            "deliverable": "pdf", "issue_kind": "missing_section",
            "abstract_text": "交付前发现缺少章节时补写", "anchors": [cat.block_ids[-1]],
        },
        "steps": [{"step_id": "f1", "step_kind": "produce", "statement": "补写章节",
                   "block_id": cat.block_ids[0]}],
    }
    card, _ = _submit(cards, fork)
    assert cards.forks_violating(skill.card_id, "s2", principal_labels=_OWNER) == (card,)
    assert cards.repair_group("diag-1", principal_labels=_OWNER) == (card,)

    bad_step = {**fork, "violates": {"card_id": skill.card_id, "step_id": "s1"}}
    with pytest.raises(ExperienceCardError) as e:
        _submit(cards, bad_step)
    assert e.value.code == "violates_step_unknown"
    no_overlay = {**fork, "fork_node_id": "0" * 64}
    with pytest.raises(ExperienceCardError) as e2:
        _submit(cards, no_overlay)
    assert e2.value.code == "fork_overlay_missing"
    wrong_anchor = {**fork, "gamma": {**fork["gamma"], "anchors": [cat.block_ids[0]]}}
    with pytest.raises(ExperienceCardError) as e3:
        _submit(cards, wrong_anchor)
    assert e3.value.code == "fork_anchor_mismatch"


def test_find_filters_boosts_and_relaxes(tmp_path: Path) -> None:
    _, rev, cat, cards = _setup(tmp_path)
    pdf, _ = _submit(cards, _skill(cat, rev))
    xl = _skill(cat, rev)
    xl["gamma"] = {**xl["gamma"], "deliverable": "xlsx", "occupation": "Accountant",
                   "abstract_text": "制作 xlsx 对账表并校验公式"}
    xlsx, _ = _submit(cards, xl)

    res = cards.find(principal_labels=_OWNER, deliverable="xlsx", query="对账表")
    assert [h.card.card_id for h in res.hits] == [xlsx.card_id] and not res.relaxed

    res = cards.find(principal_labels=_OWNER, deliverable="docx", query="报告")
    assert res.relaxed and len(res.hits) == 2

    res = cards.find(principal_labels=_OWNER, occupation="accountant")
    assert res.hits[0].card.card_id == xlsx.card_id
    res = cards.find(principal_labels=_OWNER, kind=CARD_KIND_FORK)
    assert res.hits == () and not res.relaxed
    assert cards.find(principal_labels=_OTHER).hits == ()
    assert pdf.card_id != xlsx.card_id


def test_http_face_submit_and_find(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from openviking.server.routers.experience_cards import (
        CardFindRequest,
        CardSubmitRequest,
        find_cards,
        submit_card,
    )

    _, rev, cat, cards = _setup(tmp_path)
    ctx = SimpleNamespace(user=SimpleNamespace(user_id="alice"))
    sub = submit_card(
        ctx,
        CardSubmitRequest(
            proposal=_skill(cat, rev), block_aos={"b-1": ["ao-1"], "b-2": ["ao-2"]}
        ),
        store=cards,
    )
    assert sub["kind"] == CARD_KIND_SKILL and not sub["deduplicated"]
    out = find_cards(ctx, CardFindRequest(query="pdf 财务报告", deliverable="pdf"), store=cards)
    assert out["relaxed"] is False and out["cards"][0]["card_id"] == sub["card_id"]
    bob = SimpleNamespace(user=SimpleNamespace(user_id="bob"))
    assert find_cards(bob, CardFindRequest(), store=cards)["cards"] == []


# ---- Reader tool surface (submit_experience_card) -------------------------

def _surface_with_cards(tmp_path: Path, *, verdict: str, signal: str, score: float | None):
    from openviking.session.outcome_contract import (
        OutcomeAssessmentService,
        OutcomeContractStore,
    )
    from openviking.session.t_fork_surface import TForkSurface
    from tests.test_guidance_approval import _world
    from tests.test_outcome_contract import _signal

    world = _world(tmp_path, verdict=verdict)
    cards = ExperienceCardStore(tmp_path, world.overlays)
    surface = TForkSurface(
        tmp_path,
        run=world.run,
        session_id=world.sessions["reader"],
        task_id=world.contract.task_id,
        report_hash=world.report_hash,
        principal_labels=("user:carol",),
        overlays=world.overlays,
        patterns=world.patterns,
        precedents=world.precedents,
        guidance=world.service,
        outcomes=OutcomeAssessmentService(OutcomeContractStore(tmp_path)),
        external_evidence=(_signal(signal),),
        cards=cards,
        task_score=score,
        block_aos={"b-1": ["ao-1"], "b-2": ["ao-2"]},
    )
    catalog = world.overlays.registered_catalog("rev-guidance-1", principal_labels=["user:carol"])
    doc = world.overlays.registered_document_hash("rev-guidance-1", principal_labels=["user:carol"])
    return world, surface, cards, catalog, doc


class _Rev:
    def __init__(self, content_hash: str) -> None:
        self.content_hash = content_hash


def test_surface_gates_cards_by_outcome_and_binds_score(tmp_path: Path) -> None:
    from openviking.session.t_fork_surface import CARD_VERB, TForkError

    world, surface, cards, cat, doc = _surface_with_cards(
        tmp_path, verdict="success", signal="pass", score=0.9
    )
    assert CARD_VERB in [n for n, _ in surface.catalog()]
    proposal = _skill(cat, _Rev(doc), task_score=0.1, author_role="critic")
    out = surface.call(CARD_VERB, {"proposal": proposal})
    assert out["kind"] == CARD_KIND_SKILL and not out["deduplicated"]
    stored = cards.read(out["card_id"], principal_labels=["user:carol"])
    assert stored is not None and stored.task_score == 0.9 and stored.author_role == "reader"
    again = surface.call(CARD_VERB, {"proposal": proposal})
    assert again["deduplicated"] is True
    fork = {**proposal, "kind": CARD_KIND_FORK, "fork_node_id": world.fork_node_id}
    with pytest.raises(TForkError) as locked:
        surface.call(CARD_VERB, {"proposal": fork})
    assert locked.value.code == "phase_locked"  # a fork needs a confirmed failure


def test_surface_blocks_low_score_and_wrong_report(tmp_path: Path) -> None:
    from openviking.session.t_fork_surface import CARD_VERB, TForkError

    _, surface, _, cat, doc = _surface_with_cards(
        tmp_path, verdict="success", signal="pass", score=0.5
    )
    with pytest.raises(TForkError) as low:
        surface.call(CARD_VERB, {"proposal": _skill(cat, _Rev(doc))})
    assert low.value.code == "score_gate"


def test_surface_rejects_a_proposal_bound_to_another_revision_report(tmp_path: Path) -> None:
    from openviking.session.t_fork_surface import CARD_VERB, TForkError

    _, surface, _, cat, _doc = _surface_with_cards(
        tmp_path, verdict="success", signal="pass", score=0.9
    )
    with pytest.raises(TForkError) as bad:
        surface.call(CARD_VERB, {"proposal": _skill(cat, _Rev("f" * 64))})
    assert bad.value.code == "report_hash_mismatch"


def test_surface_fork_card_after_confirmed_failure(tmp_path: Path) -> None:
    from openviking.session.t_fork_surface import CARD_VERB, TForkError

    world, surface, cards, cat, doc = _surface_with_cards(
        tmp_path, verdict="failure", signal="fail", score=0.2
    )
    base = _skill(cat, _Rev(doc))
    with pytest.raises(TForkError) as locked:
        surface.call(CARD_VERB, {"proposal": base})
    assert locked.value.code == "phase_locked"  # no skill_card from a failed run
    fork = {
        "kind": CARD_KIND_FORK, "base_revision_id": cat.base_revision_id,
        "base_revision_block_content_hash": cat.block_content_hash,
        "report_hash": doc, "boundary": "仅在缺少章节时",
        "fork_node_id": world.fork_node_id,
        "gamma": {"deliverable": "pdf", "abstract_text": "补写缺失章节",
                  "anchors": [cat.block_ids[-1]]},
        "steps": [{"step_id": "f1", "step_kind": "produce", "statement": "补写",
                   "block_id": cat.block_ids[0]}],
    }
    out = surface.call(CARD_VERB, {"proposal": fork})
    assert out["kind"] == CARD_KIND_FORK
    assert cards.read(out["card_id"], principal_labels=["user:carol"]) is not None
