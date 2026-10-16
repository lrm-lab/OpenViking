# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Outer memory-discovery classifier (Q147-B, Q148-B, Q160-B).

Typed corpora hand the coordinator their native cards, still in corpus order
and not yet publicly truncated. This type rechecks visibility, withdrawal,
invalidity, and provenance, then assigns at most one use class.

Episode and Insight are removed before that assignment. They are not
candidates, priors, or excluded public rows. Seat allocation is a later step
and is intentionally not done here. Nothing in production retrieval imports
this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

POLICY_VERSION = (
    "memory-corpus-topology/Q147-B"
    "+memory-use-class-policy/Q148-B"
    "+memory-coforge-episode-insight-retirement/Q160-B"
)

PROVIDER_PRETRUNCATION = "provider_pretruncation"
ADAPTER_SELF_UPGRADE = "adapter_self_upgrade"
RETIRED_FROM_MEMORY_DISCOVERY = "retired_from_memory_discovery"

MEMORY_CANDIDATE = "memory_candidate"
ANALOGICAL_PRIOR = "analogical_prior"
EXCLUDED = "excluded"
CURRENT_EVIDENCE = "current_evidence"

_PUBLIC_USE_CLASSES = frozenset({MEMORY_CANDIDATE, ANALOGICAL_PRIOR})
_ASSIGNABLE = frozenset({MEMORY_CANDIDATE, ANALOGICAL_PRIOR, EXCLUDED})
_RETIRED_KINDS = frozenset({"episode", "insight"})
_SKILL_KINDS = frozenset({"skill", "skill_revision"})
_FORK_KINDS = frozenset({"fork", "branch"})
_PRIOR_KINDS = frozenset(
    {
        "block_projection",
        "claim_projection",
        "projection",
        "segment",
        "atom",
    }
)
_INVALID_STATUSES = frozenset({"invalid", "invalidated"})


class MemoryDiscoveryRejection(ValueError):
    """Whole-call refusal. No public card list is returned."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class PublicMemoryCard:
    """One public card. It carries a single use class and its corpus identity."""

    id: str
    kind: str
    corpus: str
    corpus_rank: int
    use_class: str

    def __post_init__(self) -> None:
        if self.use_class not in _PUBLIC_USE_CLASSES:
            raise ValueError(f"unsupported use_class: {self.use_class}")


@dataclass(frozen=True)
class MemoryDiscoveryClassification:
    """Public classification envelope. Excluded rows and their count are omitted."""

    policy_version: str
    memory_candidates: tuple[PublicMemoryCard, ...]
    analogical_priors: tuple[PublicMemoryCard, ...]


class MemoryDiscoveryCoordinator:
    """Classify native typed-corpus cards. Do not allocate the 8/4 public seats."""

    def classify(self, corpora: Mapping[str, Any]) -> MemoryDiscoveryClassification:
        """Classify cards grouped by corpus name.

        Each value is either a sequence of native cards, or a mapping with
        ``cards`` and optional ``pre_truncated``. A single ``pre_truncated=True``
        refuses the whole call with ``provider_pretruncation``.
        """
        groups = _groups(corpora)
        if any(pre_truncated for pre_truncated, _cards in groups):
            raise MemoryDiscoveryRejection(PROVIDER_PRETRUNCATION)

        candidates: list[PublicMemoryCard] = []
        priors: list[PublicMemoryCard] = []
        for _pre_truncated, cards in groups:
            for card in cards:
                disposition = _disposition(card)
                _reject_self_upgrade(card, disposition)
                if disposition == MEMORY_CANDIDATE:
                    candidates.append(_public_card(card, MEMORY_CANDIDATE))
                elif disposition == ANALOGICAL_PRIOR:
                    priors.append(_public_card(card, ANALOGICAL_PRIOR))
        return MemoryDiscoveryClassification(
            policy_version=POLICY_VERSION,
            memory_candidates=tuple(candidates),
            analogical_priors=tuple(priors),
        )


def _groups(
    corpora: Mapping[str, Any],
) -> list[tuple[bool, list[Mapping[str, Any]]]]:
    if not isinstance(corpora, Mapping):
        raise TypeError("corpora must be a mapping of corpus name to native cards")
    groups: list[tuple[bool, list[Mapping[str, Any]]]] = []
    for name, payload in corpora.items():
        if not isinstance(name, str):
            raise TypeError("corpus name must be a string")
        pre_truncated, cards = _payload(payload)
        groups.append((pre_truncated, cards))
    return groups


def _payload(payload: Any) -> tuple[bool, list[Mapping[str, Any]]]:
    if isinstance(payload, Mapping):
        pre_truncated = payload.get("pre_truncated") is True
        raw_cards = payload.get("cards", ())
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        pre_truncated = False
        raw_cards = payload
    else:
        raise TypeError("corpus payload must be a card sequence or a cards mapping")
    if isinstance(raw_cards, (str, bytes)) or not isinstance(raw_cards, Sequence):
        raise TypeError("cards must be a sequence of native cards")
    cards: list[Mapping[str, Any]] = []
    for card in raw_cards:
        if not isinstance(card, Mapping):
            raise TypeError("native card must be a mapping")
        cards.append(card)
    return pre_truncated, cards


def _disposition(card: Mapping[str, Any]) -> str:
    """Assign a disposition. Retirement runs before any use class."""
    kind = card.get("kind")
    if kind in _RETIRED_KINDS:
        return RETIRED_FROM_MEMORY_DISCOVERY
    if card.get("visible") is not True or _is_withdrawn(card):
        return EXCLUDED
    if card.get("status") in _INVALID_STATUSES:
        return EXCLUDED
    if card.get("provenance_complete") is not True:
        return EXCLUDED

    status = card.get("status")
    if kind in _SKILL_KINDS and status == "active" and card.get("current") is True:
        return MEMORY_CANDIDATE
    if kind in _FORK_KINDS and status == "validated":
        return MEMORY_CANDIDATE
    if kind in _FORK_KINDS and status == "provisional":
        if card.get("machine_warning") is True:
            return MEMORY_CANDIDATE
        return EXCLUDED
    if kind in _PRIOR_KINDS:
        return ANALOGICAL_PRIOR
    return EXCLUDED


def _is_withdrawn(card: Mapping[str, Any]) -> bool:
    if card.get("withdrawn") is True:
        return True
    return card.get("status") == "withdrawn"


def _reject_self_upgrade(card: Mapping[str, Any], disposition: str) -> None:
    if "claimed_use_class" not in card:
        return
    claimed = card["claimed_use_class"]
    if claimed is None:
        return
    legal_disposition = disposition in _ASSIGNABLE
    if not legal_disposition or claimed == CURRENT_EVIDENCE or claimed != disposition:
        raise MemoryDiscoveryRejection(ADAPTER_SELF_UPGRADE)


def _public_card(card: Mapping[str, Any], use_class: str) -> PublicMemoryCard:
    if use_class == CURRENT_EVIDENCE or use_class not in _PUBLIC_USE_CLASSES:
        raise MemoryDiscoveryRejection(ADAPTER_SELF_UPGRADE)
    return PublicMemoryCard(
        id=card["id"],
        kind=card["kind"],
        corpus=card["corpus"],
        corpus_rank=card["corpus_rank"],
        use_class=use_class,
    )
