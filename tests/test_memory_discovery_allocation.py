# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Q153/Q154 allocation: floors, pause hole, exploration insert, stable hashes."""

from __future__ import annotations

import pytest

from openviking.session.memory_discovery_allocation import (
    CANDIDATE_CAP,
    PRIOR_CAP,
    AllocationError,
    allocate_discovery,
    stable_rotation,
)

PLAN = "plan-allocation-1"


def _card(card_id: str, corpus: str, rank: int, **overrides: object) -> dict[str, object]:
    card: dict[str, object] = {
        "id": card_id,
        "kind": "fork" if corpus == "fork_branch" else "skill_revision",
        "corpus": corpus,
        "corpus_rank": rank,
        "use_class": "memory_candidate",
        "canonical_identity": card_id,
    }
    card.update(overrides)
    return card


def _forks(count: int, prefix: str = "f") -> list[dict[str, object]]:
    return [_card(f"{prefix}{i}", "fork_branch", i) for i in range(1, count + 1)]


def _skills(count: int, prefix: str = "s") -> list[dict[str, object]]:
    return [
        _card(f"{prefix}{i}", "skill_revision", i, kind="skill_revision")
        for i in range(1, count + 1)
    ]


def _projections(count: int) -> list[dict[str, object]]:
    return [
        _card(
            f"p{i}",
            "projection",
            i,
            kind="block_projection",
            use_class="analogical_prior",
        )
        for i in range(1, count + 1)
    ]


def test_caps_are_the_accepted_envelope() -> None:
    assert CANDIDATE_CAP == 8
    assert PRIOR_CAP == 4


def test_one_corpus_stops_at_the_ceiling_and_does_not_take_prior_seats() -> None:
    result = allocate_discovery([*_forks(10)], plan_hash=PLAN)
    assert [card.id for card in result.memory_candidates] == ["f1", "f2", "f3", "f4"]
    assert result.memory_candidates[0].allocation_reason == "floor"
    assert {card.allocation_reason for card in result.memory_candidates[1:]} == {"flex"}
    assert result.analogical_priors == ()
    assert result.candidate_truncated is True


def test_exhausted_corpus_backfills_within_the_same_class() -> None:
    result = allocate_discovery([*_forks(6), *_skills(1)], plan_hash=PLAN)
    by_corpus: dict[str, list[str]] = {}
    for card in result.memory_candidates:
        by_corpus.setdefault(card.corpus, []).append(card.allocation_reason)
    assert by_corpus["skill_revision"] == ["floor"]
    assert by_corpus["fork_branch"][0] == "floor"
    assert by_corpus["fork_branch"][1:] == ["same_class_backfill"] * 3
    assert len(by_corpus["fork_branch"]) == 4
    assert result.analogical_priors == ()


def test_two_corpora_fill_candidate_cap_without_passing_ceiling() -> None:
    result = allocate_discovery([*_forks(6), *_skills(6)], plan_hash=PLAN)
    counts: dict[str, int] = {}
    for card in result.memory_candidates:
        counts[card.corpus] = counts.get(card.corpus, 0) + 1
    assert counts == {"fork_branch": 4, "skill_revision": 4}
    assert len(result.memory_candidates) == 8
    assert [card.public_rank for card in result.memory_candidates] == list(range(1, 9))


def test_insight_and_episode_add_no_seat_and_do_not_change_membership() -> None:
    base = [*_forks(3), *_skills(1), *_projections(2)]
    retired = [
        _card("i1", "insight", 1, kind="insight"),
        _card("e1", "episode", 1, kind="episode", use_class="analogical_prior"),
    ]
    without = allocate_discovery(base, plan_hash=PLAN)
    with_retired = allocate_discovery([*base, *retired], plan_hash=PLAN)
    assert [card.id for card in with_retired.memory_candidates] == [
        card.id for card in without.memory_candidates
    ]
    assert [card.id for card in with_retired.analogical_priors] == [
        card.id for card in without.analogical_priors
    ]
    assert with_retired.allocation_hash == without.allocation_hash
    published = [card.id for card in with_retired.memory_candidates]
    published.extend(card.id for card in with_retired.analogical_priors)
    assert "i1" not in published
    assert "e1" not in published


def test_fact_pause_reserves_one_prior_seat_without_a_placeholder() -> None:
    cards = [
        *_forks(5),
        *_skills(5),
        *_projections(5),
        _card("seg", "segment_atom", 1, kind="segment", use_class="analogical_prior"),
    ]
    active = allocate_discovery(cards, plan_hash=PLAN, fact_paused=False)
    paused = allocate_discovery(cards, plan_hash=PLAN, fact_paused=True)
    assert paused.pause_reserved is True
    assert active.pause_reserved is False
    assert "seg" in {card.id for card in active.analogical_priors}
    assert len(active.analogical_priors) == 4
    assert [card.id for card in paused.analogical_priors] == ["p1", "p2", "p3"]
    assert len(paused.memory_candidates) == 8
    assert len(paused.memory_candidates) + len(paused.analogical_priors) == 11
    assert all("placeholder" not in card.id for card in paused.analogical_priors)
    assert "seg" not in {card.id for card in paused.analogical_priors}


def test_exploration_is_inserted_after_the_first_exploitation_round() -> None:
    cards = [
        *_forks(3),
        _card(
            "fx",
            "fork_branch",
            9,
            exploration_stratum="provisional_fork",
            canonical_identity="fx",
        ),
    ]
    result = allocate_discovery(cards, plan_hash=PLAN)
    ids = [card.id for card in result.memory_candidates]
    assert ids == ["f1", "fx", "f2", "f3"]
    exploration = result.memory_candidates[1]
    assert exploration.selection_mode == "exploration"
    assert exploration.allocation_reason == "exploration"
    assert exploration.interleave_reason == "exploration_after_first_round"
    assert exploration.public_rank == 2
    assert result.memory_candidates[0].allocation_reason == "floor"


def test_exploration_without_exploitation_is_public_rank_one() -> None:
    cards = [_card("fx", "fork_branch", 1, exploration_stratum="provisional_fork")]
    result = allocate_discovery(cards, plan_hash=PLAN)
    assert [card.id for card in result.memory_candidates] == ["fx"]
    assert result.memory_candidates[0].public_rank == 1
    assert result.memory_candidates[0].selection_mode == "exploration"


def test_skill_exploration_gate_is_ten_percent() -> None:
    cold = _card(
        "cold",
        "skill_revision",
        1,
        kind="skill_revision",
        exploration_stratum="cold_start_skill",
    )
    closed = allocate_discovery(
        [_forks(1)[0], cold],
        plan_hash=PLAN,
        skill_explorations=0,
        skill_exploitations=8,
    )
    assert [card.selection_mode for card in closed.memory_candidates if card.id == "cold"] == [
        "exploitation"
    ]
    opened = allocate_discovery(
        [cold],
        plan_hash=PLAN,
        skill_explorations=0,
        skill_exploitations=9,
    )
    assert opened.memory_candidates[0].selection_mode == "exploration"
    held = allocate_discovery(
        [cold, *_forks(1)],
        plan_hash=PLAN,
        skill_explorations=1,
        skill_exploitations=9,
    )
    assert [card.selection_mode for card in held.memory_candidates if card.id == "cold"] == [
        "exploitation"
    ]


def test_same_plan_replays_membership_and_order_hashes() -> None:
    cards = [*_forks(4), *_skills(2), *_projections(3)]
    first = allocate_discovery(cards, plan_hash=PLAN)
    second = allocate_discovery(cards, plan_hash=PLAN)
    assert first.allocation_hash == second.allocation_hash
    assert first.final_order_hash == second.final_order_hash
    assert [card.id for card in first.memory_candidates] == [
        card.id for card in second.memory_candidates
    ]
    assert first.candidate_rotation == stable_rotation(
        PLAN, "memory_candidate", ["fork_branch", "skill_revision"]
    )


def test_rotation_is_stable_and_changes_with_the_plan() -> None:
    corpora = ["fork_branch", "skill_revision", "projection"]
    first = stable_rotation(PLAN, "memory_candidate", corpora)
    assert stable_rotation(PLAN, "memory_candidate", list(reversed(corpora))) == first
    orders = {
        stable_rotation(f"{PLAN}-{index}", "memory_candidate", corpora) for index in range(12)
    }
    assert len(orders) > 1


def test_duplicate_identity_keeps_the_better_corpus_rank() -> None:
    cards = [
        _card("f1", "fork_branch", 3, canonical_identity="same"),
        _card("f1-dup", "fork_branch", 1, canonical_identity="same"),
    ]
    result = allocate_discovery(cards, plan_hash=PLAN)
    assert [card.id for card in result.memory_candidates] == ["f1-dup"]


def test_current_evidence_claim_is_rejected() -> None:
    with pytest.raises(AllocationError) as caught:
        allocate_discovery(
            [_card("f1", "fork_branch", 1, use_class="current_evidence")],
            plan_hash=PLAN,
        )
    assert caught.value.code == "adapter_self_upgrade"
