# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Served-anchor expansion: unserved anchors fail, and a bad hop leaves E0 frozen."""

from __future__ import annotations

import pytest

from openviking.session.served_anchor_expansion import (
    CHAIN_TTL_SECONDS,
    DEFAULT_MAX_STEPS,
    HARD_MAX_STEPS,
    UNIQUE_SERVED_CAP,
    ExpansionChain,
    ExpansionError,
    freeze_direct,
    reject_redirect,
)


def _served(
    card_id: str, kind: str = "skill_revision", use_class: str = "memory_candidate"
) -> dict[str, object]:
    return {
        "id": card_id,
        "kind": kind,
        "corpus": "skill_revision" if kind == "skill_revision" else kind,
        "use_class": use_class,
        "public_rank": 1,
    }


def _chain(cards: list[dict[str, object]] | None = None, **kwargs: object) -> ExpansionChain:
    parent = freeze_direct(
        operation_id="op-e0",
        plan_hash="p" * 64,
        final_order_hash="f" * 64,
        memory_candidates=cards if cards is not None else [_served("skill-1")],
    )
    return ExpansionChain(parent, opened_at=1_000.0, clock=lambda: 1_000.0, **kwargs)  # type: ignore[arg-type]


def _expand(chain: ExpansionChain, neighbors, **overrides: object):
    request = {
        "operationId": "op-e1",
        "parentEnvelopeRef": chain.parent.envelope_ref,
        "anchorCardRef": "skill-1",
        "relation": "similar",
        "expansionChainRef": chain.ref,
    }
    request.update(overrides)
    return chain.expand(request, neighbors)


def test_chain_limits_match_the_accepted_bounds() -> None:
    assert CHAIN_TTL_SECONDS == 600
    assert DEFAULT_MAX_STEPS == 4
    assert HARD_MAX_STEPS == 8
    assert UNIQUE_SERVED_CAP == 30


def test_unserved_card_is_not_an_anchor() -> None:
    chain = _chain()
    called = False

    def neighbors(*_args: object) -> list[dict[str, object]]:
        nonlocal called
        called = True
        return []

    with pytest.raises(ExpansionError) as caught:
        _expand(chain, neighbors, anchorCardRef="not-served")
    assert caught.value.code == "anchor_not_served"
    assert called is False
    assert list(chain.envelopes) == [chain.parent.envelope_ref]


def test_failed_hop_does_not_change_the_parent() -> None:
    chain = _chain()
    parent_hash = chain.parent.plan_hash
    parent_cards = chain.parent.memory_candidates

    def neighbors(*_args: object) -> list[dict[str, object]]:
        raise RuntimeError("graph down")

    with pytest.raises(ExpansionError) as caught:
        _expand(chain, neighbors)
    assert caught.value.code == "expansion_terminated"
    assert chain.parent.plan_hash == parent_hash
    assert chain.parent.memory_candidates == parent_cards
    assert list(chain.envelopes) == [chain.parent.envelope_ref]


def test_step_cap_and_unique_served_cap() -> None:
    chain = _chain(max_steps=DEFAULT_MAX_STEPS)

    def empty(*_args: object) -> list[dict[str, object]]:
        return []

    for index in range(DEFAULT_MAX_STEPS):
        _expand(chain, empty, operationId=f"op-{index}")
    with pytest.raises(ExpansionError) as caught:
        _expand(chain, empty, operationId="op-over")
    assert caught.value.code == "chain_step_cap"

    wide = _chain([_served(f"s{index}") for index in range(UNIQUE_SERVED_CAP)])
    called = False

    def neighbors(*_args: object) -> list[dict[str, object]]:
        nonlocal called
        called = True
        return []

    with pytest.raises(ExpansionError) as caught:
        _expand(wide, neighbors, anchorCardRef="s0")
    assert caught.value.code == "chain_served_cap"
    assert called is False


def test_new_cards_that_would_pass_thirty_are_not_published() -> None:
    chain = _chain([_served(f"s{index}") for index in range(28)])

    def neighbors(*_args: object) -> list[dict[str, object]]:
        return [
            {
                "id": f"new-{index}",
                "kind": "skill_revision",
                "corpus": "skill_revision",
                "use_class": "memory_candidate",
            }
            for index in range(4)
        ]

    with pytest.raises(ExpansionError) as caught:
        _expand(chain, neighbors, anchorCardRef="s0")
    assert caught.value.code == "chain_served_cap"
    assert list(chain.envelopes) == [chain.parent.envelope_ref]


def test_episode_and_insight_cannot_enter_a_new_hop() -> None:
    chain = _chain(
        [
            _served("skill-1"),
            _served("ep-1", kind="episode", use_class="analogical_prior"),
        ]
    )
    with pytest.raises(ExpansionError) as caught:
        _expand(chain, lambda *_args: [], anchorCardRef="ep-1", relation="similar")
    assert caught.value.code == "retired_from_memory_discovery"

    def neighbors(_anchor: str, edge: str, _child: object) -> list[dict[str, object]]:
        assert edge == "similar"
        return [
            {
                "id": "insight-9",
                "kind": "insight",
                "corpus": "insight",
                "use_class": "memory_candidate",
            },
            {
                "id": "skill-2",
                "kind": "skill_revision",
                "corpus": "skill_revision",
                "use_class": "memory_candidate",
            },
        ]

    derived = _expand(chain, neighbors)
    assert [card.id for card in derived.memory_candidates] == ["skill-2"]
    assert derived.analogical_priors == ()
    with pytest.raises(ExpansionError) as caught:
        _expand(chain, neighbors, operationId="op-old", relation="episodeInsight")
    assert caught.value.code == "retired_from_memory_discovery"


def test_skills_relation_runs_grounding_before_lineage_and_stops_clean() -> None:
    chain = _chain()
    seen: list[str] = []

    def neighbors(_anchor: str, edge: str, _child: object) -> list[dict[str, object]]:
        seen.append(edge)
        if edge == "skillLineage":
            raise RuntimeError("lineage down")
        return [
            {
                "id": "skill-2",
                "kind": "skill_revision",
                "corpus": "skill_revision",
                "use_class": "memory_candidate",
            }
        ]

    with pytest.raises(ExpansionError) as caught:
        _expand(chain, neighbors, relation="skills")
    assert caught.value.code == "expansion_terminated"
    assert seen == ["skillGrounding", "skillLineage"]
    assert list(chain.envelopes) == [chain.parent.envelope_ref]


def test_agent_cannot_supply_authority_or_redirect() -> None:
    chain = _chain()
    with pytest.raises(ExpansionError) as caught:
        _expand(chain, lambda *_args: [], principal="self")
    assert caught.value.code == "agent_supplied_authority"
    with pytest.raises(ExpansionError) as caught:
        reject_redirect()
    assert caught.value.code == "redirect_removed"
    checkpoint = chain.checkpoint()
    assert checkpoint["hint_authority"] is False
    assert checkpoint["served_refs"] == ["skill-1"]


def test_duplicate_replay_does_not_walk_the_graph_again() -> None:
    chain = _chain()
    calls = 0

    def neighbors(*_args: object) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        return []

    first = _expand(chain, neighbors)
    second = _expand(chain, neighbors)
    assert calls == 1
    assert second.duplicate is True
    assert second.envelope_ref == first.envelope_ref
    assert chain.parent.plan_hash == "p" * 64


def test_chain_expires_after_ten_minutes() -> None:
    parent = freeze_direct(
        operation_id="op-e0",
        plan_hash="p" * 64,
        final_order_hash="f" * 64,
        memory_candidates=[_served("skill-1")],
    )
    chain = ExpansionChain(
        parent,
        opened_at=0.0,
        clock=lambda: CHAIN_TTL_SECONDS + 1,
    )
    with pytest.raises(ExpansionError) as caught:
        _expand(chain, lambda *_args: [])
    assert caught.value.code == "chain_expired"


def test_operation_limit_must_be_from_one_to_twelve() -> None:
    chain = _chain()
    for limit in (0, 13):
        with pytest.raises(ExpansionError) as caught:
            _expand(chain, lambda *_args: [], requestedLimit=limit, operationId=f"op-{limit}")
        assert caught.value.code == "invalid_expansion_limit"
