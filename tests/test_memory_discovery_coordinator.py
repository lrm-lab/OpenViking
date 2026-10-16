# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Classifier contract for the outer memory-discovery coordinator.

Expected cards are written as literals. They are not produced by calling the
classifier, or any helper that re-implements it, a second time.
"""

from __future__ import annotations

from dataclasses import asdict, fields

import pytest

from openviking.session.memory_discovery_coordinator import (
    RETIRED_FROM_MEMORY_DISCOVERY,
    MemoryDiscoveryCoordinator,
    MemoryDiscoveryRejection,
)

_POLICY = (
    "memory-corpus-topology/Q147-B"
    "+memory-use-class-policy/Q148-B"
    "+memory-coforge-episode-insight-retirement/Q160-B"
)


def _card(
    card_id: str,
    kind: str,
    status: str,
    corpus: str,
    corpus_rank: int,
    **extra: object,
) -> dict[str, object]:
    card: dict[str, object] = {
        "id": card_id,
        "kind": kind,
        "status": status,
        "corpus": corpus,
        "corpus_rank": corpus_rank,
        "provenance_complete": True,
        "visible": True,
    }
    card.update(extra)
    return card


def _mixed_corpora() -> dict[str, dict[str, object]]:
    return {
        "skill_revision": {
            "pre_truncated": False,
            "cards": [
                _card(
                    "sr-current",
                    "skill_revision",
                    "active",
                    "skill_revision",
                    1,
                    current=True,
                    score=0.99,
                    excluded_count=4,
                ),
                _card("sk-current", "skill", "active", "skill", 2, current=True),
                _card("sr-old", "skill_revision", "active", "skill_revision", 3, current=False),
                _card(
                    "insight-head",
                    "insight",
                    "active",
                    "insight",
                    4,
                    current=True,
                    current_head=True,
                ),
                _card(
                    "episode-live",
                    "episode",
                    "active",
                    "episode",
                    5,
                    current_head=True,
                ),
                _card(
                    "sr-hidden",
                    "skill_revision",
                    "active",
                    "skill_revision",
                    6,
                    current=True,
                    visible=False,
                ),
                _card(
                    "sr-withdrawn-status",
                    "skill_revision",
                    "withdrawn",
                    "skill_revision",
                    7,
                    current=True,
                ),
                _card(
                    "sr-withdrawn-flag",
                    "skill_revision",
                    "active",
                    "skill_revision",
                    8,
                    current=True,
                    withdrawn=True,
                ),
                _card(
                    "sr-invalid",
                    "skill_revision",
                    "invalid",
                    "skill_revision",
                    9,
                    current=True,
                ),
                _card(
                    "sr-noprov",
                    "skill_revision",
                    "active",
                    "skill_revision",
                    10,
                    current=True,
                    provenance_complete=False,
                ),
            ],
        },
        "fork_branch": {
            "cards": [
                _card("fork-val", "fork", "validated", "fork_branch", 2),
                _card("branch-val", "branch", "validated", "fork_branch", 1),
                _card(
                    "fork-prov",
                    "fork",
                    "provisional",
                    "fork_branch",
                    3,
                    machine_warning=True,
                ),
                _card(
                    "branch-prov",
                    "branch",
                    "provisional",
                    "fork_branch",
                    4,
                    machine_warning=True,
                ),
                _card(
                    "fork-plain",
                    "fork",
                    "provisional",
                    "fork_branch",
                    5,
                    machine_warning=False,
                ),
                _card("branch-nowarn", "branch", "provisional", "fork_branch", 6),
            ],
        },
        "navigation": {
            "pre_truncated": False,
            "cards": [
                _card("block-1", "block_projection", "active", "projection", 1),
                _card("claim-1", "claim_projection", "active", "projection", 2),
                _card("proj-1", "projection", "active", "projection", 3),
                _card("seg-1", "segment", "observed", "segment_atom", 1),
                _card("atom-1", "atom", "observed", "segment_atom", 2),
                _card("seg-hidden", "segment", "observed", "segment_atom", 3, visible=False),
                _card(
                    "atom-noprov",
                    "atom",
                    "observed",
                    "segment_atom",
                    4,
                    provenance_complete=False,
                ),
                _card("proj-withdrawn", "projection", "withdrawn", "projection", 4),
                _card("seg-invalid", "segment", "invalid", "segment_atom", 5),
                _card("proj-invalidated", "projection", "invalidated", "projection", 5),
            ],
        },
    }


def test_classifies_skills_forks_and_priors_without_seat_caps() -> None:
    result = MemoryDiscoveryCoordinator().classify(_mixed_corpora())

    assert asdict(result) == {
        "policy_version": _POLICY,
        "memory_candidates": (
            {
                "id": "sr-current",
                "kind": "skill_revision",
                "corpus": "skill_revision",
                "corpus_rank": 1,
                "use_class": "memory_candidate",
            },
            {
                "id": "sk-current",
                "kind": "skill",
                "corpus": "skill",
                "corpus_rank": 2,
                "use_class": "memory_candidate",
            },
            {
                "id": "fork-val",
                "kind": "fork",
                "corpus": "fork_branch",
                "corpus_rank": 2,
                "use_class": "memory_candidate",
            },
            {
                "id": "branch-val",
                "kind": "branch",
                "corpus": "fork_branch",
                "corpus_rank": 1,
                "use_class": "memory_candidate",
            },
            {
                "id": "fork-prov",
                "kind": "fork",
                "corpus": "fork_branch",
                "corpus_rank": 3,
                "use_class": "memory_candidate",
            },
            {
                "id": "branch-prov",
                "kind": "branch",
                "corpus": "fork_branch",
                "corpus_rank": 4,
                "use_class": "memory_candidate",
            },
        ),
        "analogical_priors": (
            {
                "id": "block-1",
                "kind": "block_projection",
                "corpus": "projection",
                "corpus_rank": 1,
                "use_class": "analogical_prior",
            },
            {
                "id": "claim-1",
                "kind": "claim_projection",
                "corpus": "projection",
                "corpus_rank": 2,
                "use_class": "analogical_prior",
            },
            {
                "id": "proj-1",
                "kind": "projection",
                "corpus": "projection",
                "corpus_rank": 3,
                "use_class": "analogical_prior",
            },
            {
                "id": "seg-1",
                "kind": "segment",
                "corpus": "segment_atom",
                "corpus_rank": 1,
                "use_class": "analogical_prior",
            },
            {
                "id": "atom-1",
                "kind": "atom",
                "corpus": "segment_atom",
                "corpus_rank": 2,
                "use_class": "analogical_prior",
            },
        ),
    }
    assert [field.name for field in fields(result)] == [
        "policy_version",
        "memory_candidates",
        "analogical_priors",
    ]
    published = (*result.memory_candidates, *result.analogical_priors)
    assert len(published) == 11
    for card in published:
        assert card.use_class in {"memory_candidate", "analogical_prior"}
        assert card.use_class != "current_evidence"
        assert [item.name for item in fields(card)] == [
            "id",
            "kind",
            "corpus",
            "corpus_rank",
            "use_class",
        ]


def test_episode_and_insight_are_retired_before_use_class() -> None:
    assert RETIRED_FROM_MEMORY_DISCOVERY == "retired_from_memory_discovery"
    result = MemoryDiscoveryCoordinator().classify(
        {
            "retired": {
                "cards": [
                    _card(
                        "insight-head",
                        "insight",
                        "active",
                        "insight",
                        1,
                        current=True,
                        current_head=True,
                    ),
                    _card("episode-live", "episode", "active", "episode", 1, current_head=True),
                ],
            }
        }
    )

    assert asdict(result) == {
        "policy_version": _POLICY,
        "memory_candidates": (),
        "analogical_priors": (),
    }
    blob = repr(asdict(result))
    assert "insight-head" not in blob
    assert "episode-live" not in blob
    assert "retired_from_memory_discovery" not in blob
    assert "excluded" not in blob


def test_public_result_does_not_leak_excluded_rows_or_counts() -> None:
    result = MemoryDiscoveryCoordinator().classify(_mixed_corpora())
    blob = repr(asdict(result))

    assert "excluded_count" not in blob
    assert "excluded" not in blob
    for hidden in (
        "sr-old",
        "sr-hidden",
        "sr-withdrawn-status",
        "sr-withdrawn-flag",
        "sr-invalid",
        "sr-noprov",
        "fork-plain",
        "branch-nowarn",
        "seg-hidden",
        "atom-noprov",
        "proj-withdrawn",
        "seg-invalid",
        "proj-invalidated",
        "insight-head",
        "episode-live",
    ):
        assert hidden not in blob
    assert len(result.memory_candidates) == 6
    assert len(result.analogical_priors) == 5


def test_provider_pretruncation_rejects_the_whole_call() -> None:
    corpora = _mixed_corpora()
    corpora["navigation"]["pre_truncated"] = True
    corpora["skill_revision"]["cards"].append(
        _card(
            "sr-claim",
            "skill_revision",
            "active",
            "skill_revision",
            11,
            current=True,
            claimed_use_class="current_evidence",
        )
    )

    with pytest.raises(MemoryDiscoveryRejection) as caught:
        MemoryDiscoveryCoordinator().classify(corpora)

    assert caught.value.code == "provider_pretruncation"
    assert caught.value.args == ("provider_pretruncation",)
    assert not hasattr(caught.value, "memory_candidates")
    assert not hasattr(caught.value, "analogical_priors")


def test_retired_only_corpus_is_an_empty_success_when_not_pretruncated() -> None:
    result = MemoryDiscoveryCoordinator().classify(
        {
            "episode": {
                "pre_truncated": False,
                "cards": [_card("episode-live", "episode", "active", "episode", 1)],
            }
        }
    )

    assert result.policy_version == _POLICY
    assert result.memory_candidates == ()
    assert result.analogical_priors == ()


@pytest.mark.parametrize(
    ("kind", "status", "extra", "claimed"),
    [
        ("skill_revision", "active", {"current": True}, "analogical_prior"),
        ("skill_revision", "active", {"current": True}, "current_evidence"),
        ("skill_revision", "active", {"current": True}, "excluded"),
        ("fork", "validated", {}, "analogical_prior"),
        ("fork", "provisional", {"machine_warning": False}, "memory_candidate"),
        ("branch", "provisional", {}, "memory_candidate"),
        ("segment", "observed", {}, "memory_candidate"),
        ("insight", "active", {"current_head": True, "current": True}, "memory_candidate"),
        ("insight", "active", {"current_head": True}, "analogical_prior"),
        ("episode", "active", {"current_head": True}, "analogical_prior"),
        ("skill_revision", "active", {"current": True, "visible": False}, "memory_candidate"),
        (
            "fork",
            "validated",
            {"provenance_complete": False},
            "memory_candidate",
        ),
    ],
)
def test_mismatched_claim_rejects_the_whole_call(
    kind: str,
    status: str,
    extra: dict[str, object],
    claimed: str,
) -> None:
    bad = _card("bad", kind, status, "mixed", 1, claimed_use_class=claimed, **extra)
    sibling = _card("keep-me", "skill", "active", "skill", 1, current=True)

    with pytest.raises(MemoryDiscoveryRejection) as caught:
        MemoryDiscoveryCoordinator().classify({"mixed": {"cards": [sibling, bad]}})

    assert caught.value.code == "adapter_self_upgrade"
    assert not hasattr(caught.value, "memory_candidates")


def test_matching_claims_keep_only_the_public_cards() -> None:
    result = MemoryDiscoveryCoordinator().classify(
        {
            "skill_revision": {
                "cards": [
                    _card(
                        "sr-current",
                        "skill_revision",
                        "active",
                        "skill_revision",
                        1,
                        current=True,
                        claimed_use_class="memory_candidate",
                    ),
                    _card(
                        "sr-hidden",
                        "skill_revision",
                        "active",
                        "skill_revision",
                        2,
                        current=True,
                        visible=False,
                        claimed_use_class="excluded",
                    ),
                ],
            },
            "navigation": {
                "cards": [
                    _card(
                        "atom-1",
                        "atom",
                        "observed",
                        "segment_atom",
                        1,
                        claimed_use_class="analogical_prior",
                    ),
                ],
            },
        }
    )

    assert asdict(result) == {
        "policy_version": _POLICY,
        "memory_candidates": (
            {
                "id": "sr-current",
                "kind": "skill_revision",
                "corpus": "skill_revision",
                "corpus_rank": 1,
                "use_class": "memory_candidate",
            },
        ),
        "analogical_priors": (
            {
                "id": "atom-1",
                "kind": "atom",
                "corpus": "segment_atom",
                "corpus_rank": 1,
                "use_class": "analogical_prior",
            },
        ),
    }


def test_more_than_eight_candidates_and_four_priors_are_all_returned() -> None:
    skills = [
        _card(f"skill-{rank}", "skill_revision", "active", "skill_revision", rank, current=True)
        for rank in (1, 2, 3, 4, 5, 6, 7, 8, 9)
    ]
    skills.append(
        _card("skill-hidden", "skill_revision", "active", "skill_revision", 10, visible=False)
    )
    priors = [
        _card(f"atom-{rank}", "atom", "observed", "segment_atom", rank) for rank in (1, 2, 3, 4, 5)
    ]
    result = MemoryDiscoveryCoordinator().classify(
        {
            "skill_revision": {"cards": skills},
            "segment_atom": {"cards": priors},
        }
    )

    assert [card.id for card in result.memory_candidates] == [
        "skill-1",
        "skill-2",
        "skill-3",
        "skill-4",
        "skill-5",
        "skill-6",
        "skill-7",
        "skill-8",
        "skill-9",
    ]
    assert [card.corpus_rank for card in result.memory_candidates] == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ]
    assert [card.use_class for card in result.memory_candidates] == ["memory_candidate"] * 9
    assert [card.id for card in result.analogical_priors] == [
        "atom-1",
        "atom-2",
        "atom-3",
        "atom-4",
        "atom-5",
    ]
    assert [card.use_class for card in result.analogical_priors] == ["analogical_prior"] * 5
    assert len(result.memory_candidates) + len(result.analogical_priors) == 14


def test_output_keeps_input_order_rather_than_sorting_by_rank() -> None:
    result = MemoryDiscoveryCoordinator().classify(
        {
            "skill_revision": [
                _card("sr-3", "skill_revision", "active", "skill_revision", 3, current=True),
                _card("sr-1", "skill_revision", "active", "skill_revision", 1, current=True),
                _card("sr-2", "skill", "active", "skill", 2, current=True),
            ]
        }
    )

    assert [(card.id, card.corpus_rank) for card in result.memory_candidates] == [
        ("sr-3", 3),
        ("sr-1", 1),
        ("sr-2", 2),
    ]
    assert result.analogical_priors == ()
