# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Discovery allocation and interleave (Q153-B, Q154-B, Q160-B).

After use-class classification, candidate seats are at most 8 and prior
seats at most 4. A corpus with eligible cards gets floor 1 and ceiling 4.
Episode and Insight are dropped before seats are assigned and do not leave
a replacement seat. A paused fact corpus leaves one prior seat empty.

The exploration seat, when filled, is inside the candidate cap. Interleave
only permutes the selected set: one zipper round, then the exploration
card, then the rest of the zipper.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

ALLOCATION_POLICY_VERSION = "memory-discovery-allocation/Q153-B"
INTERLEAVE_POLICY_VERSION = "memory-discovery-interleave/Q154-B"
CANDIDATE_CAP = 8
PRIOR_CAP = 4
CORPUS_FLOOR = 1
CORPUS_CEILING = 4
CANDIDATE = "memory_candidate"
PRIOR = "analogical_prior"
RETIRED_KINDS = frozenset({"episode", "insight"})
RETIRED_CORPORA = frozenset({"episode", "insight"})
FACT_KINDS = frozenset({"segment", "atom"})
FACT_CORPORA = frozenset({"segment_atom", "segment/atom", "fact"})
PROVISIONAL_FORK = "provisional_fork"
COLD_START_SKILL = "cold_start_skill"


class AllocationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class OrderedCard:
    id: str
    kind: str
    corpus: str
    use_class: str
    public_rank: int
    corpus_local_rank: int
    selection_mode: str
    allocation_reason: str
    interleave_reason: str


@dataclass(frozen=True)
class DiscoveryAllocation:
    memory_candidates: tuple[OrderedCard, ...]
    analogical_priors: tuple[OrderedCard, ...]
    allocation_hash: str
    final_order_hash: str
    pause_reserved: bool
    candidate_rotation: tuple[str, ...]
    prior_rotation: tuple[str, ...]
    policy_versions: Mapping[str, str]
    candidate_truncated: bool
    prior_truncated: bool


def allocate_discovery(
    cards: Sequence[Mapping[str, Any]],
    *,
    plan_hash: str,
    fact_paused: bool = False,
    skill_explorations: int = 0,
    skill_exploitations: int = 0,
) -> DiscoveryAllocation:
    """Select seats, then zipper them. Membership does not depend on display order."""
    if not isinstance(plan_hash, str) or not plan_hash:
        raise AllocationError("invalid_plan_hash")
    if type(fact_paused) is not bool:
        raise AllocationError("invalid_fact_pause")
    _count(skill_explorations, "skill_explorations")
    _count(skill_exploitations, "skill_exploitations")
    prepared = _prepare(cards, fact_paused=fact_paused)
    candidates = [card for card in prepared if card["use_class"] == CANDIDATE]
    priors = [card for card in prepared if card["use_class"] == PRIOR]
    exploration = _draw_exploration(
        candidates,
        plan_hash=plan_hash,
        skill_explorations=skill_explorations,
        skill_exploitations=skill_exploitations,
    )
    if exploration is not None:
        candidates = [card for card in candidates if card["key"] != exploration["key"]]
    exploitation_cap = CANDIDATE_CAP - (1 if exploration is not None else 0)
    ceilings = _ceilings(candidates, exploration)
    candidate_rotation = stable_rotation(
        plan_hash, CANDIDATE, [card["corpus"] for card in candidates]
    )
    selected_candidates, candidate_short = _water_fill(
        candidates,
        cap=exploitation_cap,
        rotation=candidate_rotation,
        ceilings=ceilings,
    )
    if exploration is not None:
        exploration = {
            **exploration,
            "allocation_reason": "exploration",
            "selection_mode": "exploration",
        }
    prior_pool = priors
    prior_cap = PRIOR_CAP - (1 if fact_paused else 0)
    prior_rotation = stable_rotation(plan_hash, PRIOR, [card["corpus"] for card in prior_pool])
    selected_priors, prior_short = _water_fill(
        prior_pool,
        cap=prior_cap,
        rotation=prior_rotation,
        ceilings=dict.fromkeys(prior_rotation, CORPUS_CEILING),
    )
    ordered_candidates = _interleave(selected_candidates, candidate_rotation, exploration)
    ordered_priors = _interleave(selected_priors, prior_rotation, None)
    allocation_hash = _allocation_hash(
        plan_hash,
        fact_paused,
        ordered_candidates,
        ordered_priors,
    )
    final_order_hash = _final_order_hash(allocation_hash, ordered_candidates, ordered_priors)
    return DiscoveryAllocation(
        memory_candidates=tuple(ordered_candidates),
        analogical_priors=tuple(ordered_priors),
        allocation_hash=allocation_hash,
        final_order_hash=final_order_hash,
        pause_reserved=fact_paused,
        candidate_rotation=candidate_rotation,
        prior_rotation=prior_rotation,
        policy_versions={
            "allocation": ALLOCATION_POLICY_VERSION,
            "interleave": INTERLEAVE_POLICY_VERSION,
        },
        candidate_truncated=candidate_short
        or _unseated(candidates, selected_candidates, exploration),
        prior_truncated=prior_short or len(selected_priors) < len(prior_pool),
    )


def stable_rotation(plan_hash: str, use_class: str, corpora: Sequence[str]) -> tuple[str, ...]:
    """Order corpora from the plan hash, the allocation policy, and the use class."""
    unique = tuple(dict.fromkeys(corpora))
    return tuple(
        sorted(
            unique,
            key=lambda corpus: (
                _digest(plan_hash, ALLOCATION_POLICY_VERSION, use_class, corpus),
                corpus,
            ),
        )
    )


def _prepare(cards: Sequence[Mapping[str, Any]], *, fact_paused: bool) -> list[dict[str, Any]]:
    if isinstance(cards, (str, bytes)) or not isinstance(cards, Sequence):
        raise AllocationError("invalid_cards")
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in cards:
        if not isinstance(raw, Mapping):
            raise AllocationError("invalid_card")
        card = _normalize(raw)
        if card is None:
            continue
        if fact_paused and card["fact"]:
            continue
        current = best.get(card["key"])
        if current is None or (card["corpus_rank"], card["id"]) < (
            current["corpus_rank"],
            current["id"],
        ):
            best[card["key"]] = card
    return list(best.values())


def _normalize(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    kind = raw.get("kind")
    corpus = raw.get("corpus")
    use_class = raw.get("use_class")
    card_id = raw.get("id")
    rank = raw.get("corpus_rank")
    if kind in RETIRED_KINDS or corpus in RETIRED_CORPORA:
        return None
    if use_class == "current_evidence":
        raise AllocationError("adapter_self_upgrade")
    if use_class not in (CANDIDATE, PRIOR):
        raise AllocationError("invalid_use_class")
    if not isinstance(card_id, str) or not card_id:
        raise AllocationError("invalid_card")
    if not isinstance(kind, str) or not isinstance(corpus, str) or not corpus:
        raise AllocationError("invalid_card")
    if type(rank) is not int or rank < 1:
        raise AllocationError("invalid_card")
    identity = raw.get("canonical_identity", card_id)
    if not isinstance(identity, str) or not identity:
        raise AllocationError("invalid_card")
    stratum = raw.get("exploration_stratum")
    if stratum not in (None, PROVISIONAL_FORK, COLD_START_SKILL):
        raise AllocationError("invalid_exploration_stratum")
    return {
        "id": card_id,
        "kind": kind,
        "corpus": corpus,
        "use_class": use_class,
        "corpus_rank": rank,
        "identity": identity,
        "stratum": stratum,
        "fact": kind in FACT_KINDS or corpus in FACT_CORPORA,
        "key": (corpus, identity),
        "selection_mode": "exploitation",
        "allocation_reason": "floor",
    }


def _draw_exploration(
    cards: Sequence[Mapping[str, Any]],
    *,
    plan_hash: str,
    skill_explorations: int,
    skill_exploitations: int,
) -> dict[str, Any] | None:
    allowed = {
        PROVISIONAL_FORK: [card for card in cards if card["stratum"] == PROVISIONAL_FORK],
    }
    if _skill_gate_open(skill_explorations, skill_exploitations):
        allowed[COLD_START_SKILL] = [card for card in cards if card["stratum"] == COLD_START_SKILL]
    strata = [name for name, pool in allowed.items() if pool]
    if not strata:
        return None
    stratum = min(
        strata,
        key=lambda name: (_digest(plan_hash, ALLOCATION_POLICY_VERSION, "stratum", name), name),
    )
    pool = allowed[stratum]
    return min(
        pool,
        key=lambda card: (
            _digest(plan_hash, ALLOCATION_POLICY_VERSION, "card", card["identity"]),
            card["identity"],
        ),
    )


def _skill_gate_open(explorations: int, exploitations: int) -> bool:
    """Next skill exploration stays at or under 10 percent of the rolling window."""
    return (explorations + 1) * 10 <= (explorations + exploitations + 1)


def _ceilings(
    cards: Sequence[Mapping[str, Any]],
    exploration: Mapping[str, Any] | None,
) -> dict[str, int]:
    ceilings = {card["corpus"]: CORPUS_CEILING for card in cards}
    if exploration is not None:
        corpus = exploration["corpus"]
        ceilings[corpus] = ceilings.get(corpus, CORPUS_CEILING) - 1
    return ceilings


def _water_fill(
    cards: Sequence[Mapping[str, Any]],
    *,
    cap: int,
    rotation: Sequence[str],
    ceilings: Mapping[str, int],
) -> tuple[list[dict[str, Any]], bool]:
    queues: dict[str, list[dict[str, Any]]] = {corpus: [] for corpus in rotation}
    for card in cards:
        queues.setdefault(card["corpus"], []).append(card)
    for queue in queues.values():
        queue.sort(key=lambda card: (card["corpus_rank"], card["identity"]))
    selected: list[dict[str, Any]] = []
    taken = dict.fromkeys(queues, 0)
    for corpus in rotation:
        if len(selected) >= cap:
            break
        if taken[corpus] >= ceilings.get(corpus, CORPUS_CEILING) or not queues[corpus]:
            continue
        selected.append(_take(queues[corpus], "floor"))
        taken[corpus] += 1
    while len(selected) < cap:
        ready = [
            corpus
            for corpus in rotation
            if queues[corpus] and taken[corpus] < ceilings.get(corpus, CORPUS_CEILING)
        ]
        if not ready:
            break
        exhausted = any(not queues[corpus] for corpus in rotation)
        reason = "same_class_backfill" if exhausted else "flex"
        for corpus in ready:
            if len(selected) >= cap:
                break
            selected.append(_take(queues[corpus], reason))
            taken[corpus] += 1
    return selected, len(selected) >= cap and any(queues.values())


def _take(queue: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    card = queue.pop(0)
    return {**card, "allocation_reason": reason, "selection_mode": "exploitation"}


def _interleave(
    selected: Sequence[Mapping[str, Any]],
    rotation: Sequence[str],
    exploration: Mapping[str, Any] | None,
) -> list[OrderedCard]:
    queues: dict[str, list[Mapping[str, Any]]] = {corpus: [] for corpus in rotation}
    for card in selected:
        queues.setdefault(card["corpus"], []).append(card)
    for queue in queues.values():
        queue.sort(key=lambda card: (card["corpus_rank"], card["identity"]))
    ordered: list[Mapping[str, Any]] = []
    for corpus in rotation:
        if queues.get(corpus):
            ordered.append({**queues[corpus].pop(0), "interleave_reason": "rank_depth_zipper"})
    if exploration is not None:
        ordered.append({**exploration, "interleave_reason": "exploration_after_first_round"})
    while any(queues.values()):
        for corpus in rotation:
            if queues.get(corpus):
                ordered.append({**queues[corpus].pop(0), "interleave_reason": "rank_depth_zipper"})
    return [
        OrderedCard(
            id=card["id"],
            kind=card["kind"],
            corpus=card["corpus"],
            use_class=card["use_class"],
            public_rank=index,
            corpus_local_rank=card["corpus_rank"],
            selection_mode=card["selection_mode"],
            allocation_reason=card["allocation_reason"],
            interleave_reason=card["interleave_reason"],
        )
        for index, card in enumerate(ordered, start=1)
    ]


def _unseated(
    pool: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    exploration: Mapping[str, Any] | None,
) -> bool:
    chosen = {card["key"] for card in selected}
    if exploration is not None:
        chosen.add(exploration["key"])
    return any(card["key"] not in chosen for card in pool)


def _allocation_hash(
    plan_hash: str,
    fact_paused: bool,
    candidates: Sequence[OrderedCard],
    priors: Sequence[OrderedCard],
) -> str:
    payload = {
        "policy": ALLOCATION_POLICY_VERSION,
        "planHash": plan_hash,
        "factPaused": fact_paused,
        "candidates": sorted(card.id for card in candidates),
        "priors": sorted(card.id for card in priors),
    }
    return _json_hash(payload)


def _final_order_hash(
    allocation_hash: str,
    candidates: Sequence[OrderedCard],
    priors: Sequence[OrderedCard],
) -> str:
    payload = {
        "policy": INTERLEAVE_POLICY_VERSION,
        "allocationHash": allocation_hash,
        "candidates": [card.id for card in candidates],
        "priors": [card.id for card in priors],
    }
    return _json_hash(payload)


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _digest(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _count(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise AllocationError(f"invalid_{name}")
