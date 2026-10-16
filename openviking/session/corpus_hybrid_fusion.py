# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Per-corpus hybrid fusion (Q156-B).

Each typed corpus unions its own BM25 top-32 with its own vector top-32 and
orders that union by equal RRF. Ranks are 1-based positions in the channel
list. A missing channel contributes zero. Canonical identity breaks ties.
Vector failure degrades only that corpus to its BM25 order.

This module does not allocate seats and does not score diagnostic subgraphs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

FUSION_POLICY_VERSION = "memory-hybrid-fusion/Q156-B"
RRF_K = 60
CHANNEL_LIMIT = 32
_PROOF_FIELDS = ("snapshot_ref", "index_version", "model_version", "policy_version")


@dataclass(frozen=True)
class CorpusFusion:
    """Corpus-local order. Raw scores, ranks, and proof details stay out."""

    corpus: str
    ids: tuple[str, ...]
    policy_version: str
    degraded: bool


def fuse_corpora(corpora: Mapping[str, Mapping[str, Any]]) -> tuple[CorpusFusion, ...]:
    """Fuse each corpus independently. One failure does not rewrite its siblings."""
    if not isinstance(corpora, Mapping):
        raise TypeError("corpora must be a mapping")
    fused: list[CorpusFusion] = []
    for name, payload in corpora.items():
        if not isinstance(name, str) or not name:
            raise TypeError("corpus name must be a non-empty string")
        if not isinstance(payload, Mapping):
            raise TypeError("corpus payload must be a mapping")
        fused.append(_fuse_one(name, payload))
    return tuple(fused)


def _fuse_one(corpus: str, payload: Mapping[str, Any]) -> CorpusFusion:
    bm25 = _channel(payload.get("bm25"))
    vector_ok = (
        _proof_is_complete(payload.get("proof")) and payload.get("vector_failed") is not True
    )
    if not vector_ok:
        return CorpusFusion(
            corpus=corpus,
            ids=bm25,
            policy_version=FUSION_POLICY_VERSION,
            degraded=True,
        )
    vector = _channel(payload.get("vector"))
    ranks: dict[str, list[int | None]] = {}
    for rank, identity in enumerate(bm25, start=1):
        ranks.setdefault(identity, [None, None])[0] = rank
    for rank, identity in enumerate(vector, start=1):
        slot = ranks.setdefault(identity, [None, None])
        if slot[1] is None:
            slot[1] = rank
    ordered = tuple(sorted(ranks, key=lambda identity: (-_rrf(ranks[identity]), identity)))
    return CorpusFusion(
        corpus=corpus,
        ids=ordered,
        policy_version=FUSION_POLICY_VERSION,
        degraded=False,
    )


def _channel(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("channel must be a sequence of identities")
    seen: list[str] = []
    for identity in raw:
        if not isinstance(identity, str) or not identity:
            raise TypeError("channel identity must be a non-empty string")
        if identity in seen:
            continue
        seen.append(identity)
        if len(seen) == CHANNEL_LIMIT:
            break
    return tuple(seen)


def _proof_is_complete(proof: Any) -> bool:
    if not isinstance(proof, Mapping) or proof.get("complete") is not True:
        return False
    return all(isinstance(proof.get(field), str) and proof[field] for field in _PROOF_FIELDS)


def _rrf(ranks: Sequence[int | None]) -> float:
    score = 0.0
    for rank in ranks:
        if rank is not None:
            score += 1.0 / (RRF_K + rank)
    return score
