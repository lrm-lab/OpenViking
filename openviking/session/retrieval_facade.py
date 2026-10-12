# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Typed hybrid retrieval facade for the causal-memory lane (Q112-A / Q122-A).

Three corpus adapters share one card envelope. Storage and identity stay
separate: fork/branch, block/claim projection, and segment/atom facts are
not collapsed into one corpus. Verified/provisional slotting (8/4, with
backfill) applies only to stateful cards. When the provisional quota is
``n >= 2``, those seats are ``n - 1`` exploitation cards in in-channel rank
order plus one seeded exploration card drawn from the provisional candidates
exploitation did not take. Segment/atom cards stay on the fact channel.
Dependency-pattern hits expand inside one frozen view at read time and do
not write a stored object.

Initial rank inside each corpus is BM25. ``RetrievalProfile.rank_mode`` may
select the hybrid tier, which mixes that score with a real vector from the
injected embedder. The mix weights are an unconfirmed stand-in; see
``HYBRID_FUSION_POLICY_VERSION``. A missing embedder, or a vector whose dimension
is 0, keeps this call on BM25 and still returns the BM25 hits. The projection
``embedding_stub`` is not a vector. ``FEATURE_SPEC_VERSION`` stays on the
result. While ``FACT_LANE_PAUSED`` is set, ``retrieve`` does not call the
segment/atom adapter and does not give those empty seats to another corpus.
The query cap remains ``CAUSAL_MEMORY_CAP``.

Fork cards carry the latest revision's seven sections on ``RetrievalCard.text``.
Branch cards carry a bounded action synopsis built from AO-ledger synopses
for ``ao_sequence``. Guidance and raw AO text stay off the card. Projection
and fact cards keep an empty ``text``. Rank documents for fork and branch
stay the serialized revision, so ``score_breakdown`` does not gain a text term.

``expand_claim_neighborhood(card_id=...)`` is the Memory Agent entry. It
resolves the claim projection back to a frozen view revision and claim id,
and that resolution fails closed on ``principal_labels``. The default empty
sequence sees no card. A card the principal cannot see raises the same
``unknown card id`` error as an id that is not indexed. Diagnostic agents
not calling it is an agent-side fence.

``retrieve_historical_subgraphs(view_revision_id, subgraph_id, principal_labels)``
is the diagnostic read. The server resolves that subgraph on the named frozen
revision and requires ``is_failure``. The corpus is retrieval subgraphs on
other ``post_run_index`` revisions. Ranking is BM25 on the shared subgraph
text (``SUBGRAPH_TEXT_POLICY_VERSION``); there is no vector tier.
The read uses ``ViewSupersedeIndex.revisions_for_purpose`` and does not call
``ViewSupersedeIndex.to_dict``. It does not write forks and ``retrieve`` does
not call it. The querying revision's session must have ACL labels visible to
the principal; a missing label set or a mismatch raises and does not return
an empty hit list. A revision id that matches nothing, or more than one
frozen revision, also raises. The lookup does not pick one of several matches.

Online retrieval applies principal ACL inside each adapter before the quota
is spent, then clamps the reranked list back to the causal-memory cap.
Offline management reads are not part of this path and online code must not
call them: ``materialize_strength(None)`` (strength with no principal),
``CitationLedger.events()``, ``CausalBridgeStore.list_bridges``, and
``ProjectionRegistry.to_dict``. Those dumps skip principal filtering.
Online projection lookup goes through ``ProjectionRegistry.search``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from openviking.session.ao_ledger import AOLedger, LEDGER_FILENAME
from openviking.session.ao_ledger_reader import list_ao_ledger
from openviking.session.bm25 import BM25Index
from openviking.session.causal_bridges import (
    PATH_MAX_ENTITIES,
    PATH_MAX_HYPOTHESIS_HOPS,
    check_path_budget,
)
from openviking.session.causal_experiences import (
    FORK_CANDIDATE_ROLE,
    INDEX_FILENAME,
    NAMESPACE_DIRNAME,
    SEVEN_SECTION_KEYS,
    CausalExperiencesStore,
    ForkStatus,
    InfluenceGrounding,
    branch_retrieval_channel,
)
from openviking.session.influence_projection import (
    InfluenceProjection,
    ProjectionRegistry,
    SessionAcl,
)
from openviking.session.influence_view import (
    PURPOSE_POST_RUN_INDEX,
    PURPOSES,
    RELATION_INFLUENCE,
    InfluenceBlock,
    InfluenceClaim,
    InfluenceViewRevision,
    RetrievalSubgraph,
    ViewSupersedeIndex,
)

FEATURE_SPEC_VERSION = "ranker-feature-spec-v1"
SELECTION_POLICY_VERSION = "provisional-selection-v1"
NO_ELIGIBLE_EXPLORATION_REASON = "no_eligible_exploration_candidate"

CAUSAL_MEMORY_CAP = 12
DEFAULT_FORK_BRANCH_QUOTA = 6
DEFAULT_PROJECTION_QUOTA = 4
DEFAULT_SEGMENT_ATOM_QUOTA = 2
VERIFIED_SLOTS = 8
PROVISIONAL_SLOTS = 4
FACT_LANE_PAUSED = True
RANK_MODE_BM25 = "bm25"
RANK_MODE_HYBRID = "hybrid"
RANK_MODES = frozenset((RANK_MODE_BM25, RANK_MODE_HYBRID))
# Pending parameter. CONSENSUS decision #95 / narrative 86 only require
# fusing BM25 with a real vector into one rank. They do not set weights or a
# fusion formula. These values still need design confirmation.
# HYBRID_FUSION_POLICY_VERSION marks this unconfirmed choice.
# Stand-in until then: the vector weight is above one half, so a real vector
# can move a weaker BM25 hit ahead of a stronger one. Equal min-max weights
# tie a pure two-document reversal, and the stable sort keeps write order.
HYBRID_FUSION_POLICY_VERSION = "hybrid-fusion-pending-v0"
HYBRID_BM25_WEIGHT = 0.25
HYBRID_VECTOR_WEIGHT = 0.75
# Branch cards carry a short action synopsis, not guidance or raw AO text.
# The first N ``ao_sequence`` ids are resolved against the session AO ledger.
# Each synopsis line and the joined summary are clipped. Rank blobs stay the
# serialized fork revision, so this text is not a score feature.
BRANCH_SYNOPSIS_LIMIT = 8
BRANCH_SYNOPSIS_LINE_CHARS = 180
BRANCH_SUMMARY_CHARS = 720
# Diagnostic subgraph query and historical documents share one BM25 string.
# Prose stays in stored member order. A bounded typed-token section is
# appended after it. SUBGRAPH_TEXT_POLICY_VERSION is that accepted policy.
# There is no vector tier, hybrid fusion, or structural score on this path.
SUBGRAPH_TEXT_POLICY_VERSION = "diagnostic-subgraph-text-v1"
DIAGNOSTIC_SUBGRAPH_HIT_CAP = 3
_RELATION_PROFILE_COUNT_CAP = 99

CardKind = Literal[
    "fork",
    "branch",
    "block_projection",
    "claim_projection",
    "segment",
    "atom",
    "native_block",
    "native_edge",
    "fork_overlay",
    "strategy_pattern",
    "guidance",
]
CardChannel = Literal["verified", "provisional", "fact"]
SelectionMode = Literal["exploitation", "exploration"]
RankMode = Literal["bm25", "hybrid"]
EmbedFn = Callable[[str], Sequence[float]]

CARD_KINDS = frozenset(
    (
        "fork",
        "branch",
        "block_projection",
        "claim_projection",
        "segment",
        "atom",
        "native_block",
        "native_edge",
        "fork_overlay",
        "strategy_pattern",
        "guidance",
    )
)
CARD_CHANNELS = frozenset(("verified", "provisional", "fact"))
SELECTION_MODES = frozenset(("exploitation", "exploration"))
STATEFUL_CHANNELS = frozenset(("verified", "provisional"))
FACT_KINDS = frozenset(("segment", "atom"))

AclFilter = Callable[["RetrievalCard", Sequence[str]], bool]
RerankFn = Callable[[Sequence["RetrievalCard"], Mapping[str, Any]], Sequence["RetrievalCard"]]


class QuotaError(ValueError):
    """Raised when a retrieval profile exceeds the causal-memory lane cap."""

    def __init__(self, message: str, *, total: int, cap: int = CAUSAL_MEMORY_CAP) -> None:
        self.total = total
        self.cap = cap
        super().__init__(message)


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return dict(value)


def _json_copy(value: Any, field: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON-serializable") from exc


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty str")
    return value


def _require_rank(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("in_channel_rank must be an int >= 0")
    return value


def _normalize_labels(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("acl_labels must be a sequence of str")
    labels: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _require_str(item, "acl_labels")
        if text in seen:
            continue
        seen.add(text)
        labels.append(text)
    return tuple(labels)


@dataclass(frozen=True)
class RetrievalCard:
    """Cross-corpus envelope. ``kind`` keeps each corpus identity intact.

    ``text`` is display body and is part of card content (``to_dict`` /
    equality). Fork cards carry the seven-section snapshot. Branch cards
    carry a bounded action synopsis. Projection and fact cards leave it
    empty; their match text stays off the card.
    """

    kind: CardKind
    channel: CardChannel
    in_channel_rank: int
    status_label: str
    source_pointer: dict[str, Any]
    payload_ref: str
    score_breakdown: dict[str, Any]
    acl_labels: tuple[str, ...] = ()
    selection_mode: SelectionMode = "exploitation"
    text: str = ""

    def __post_init__(self) -> None:
        if self.kind not in CARD_KINDS:
            raise ValueError(f"kind must be one of {sorted(CARD_KINDS)}, got {self.kind!r}")
        if self.channel not in CARD_CHANNELS:
            raise ValueError(
                f"channel must be one of {sorted(CARD_CHANNELS)}, got {self.channel!r}"
            )
        if self.selection_mode not in SELECTION_MODES:
            raise ValueError(
                "selection_mode must be one of "
                f"{sorted(SELECTION_MODES)}, got {self.selection_mode!r}"
            )
        object.__setattr__(self, "in_channel_rank", _require_rank(self.in_channel_rank))
        object.__setattr__(self, "status_label", _require_str(self.status_label, "status_label"))
        object.__setattr__(self, "payload_ref", _require_str(self.payload_ref, "payload_ref"))
        pointer = _json_copy(_require_mapping(self.source_pointer, "source_pointer"), "source_pointer")
        if not isinstance(pointer, dict):
            raise ValueError("source_pointer must be a dict")
        breakdown = _json_copy(
            _require_mapping(self.score_breakdown, "score_breakdown"), "score_breakdown"
        )
        if not isinstance(breakdown, dict):
            raise ValueError("score_breakdown must be a dict")
        object.__setattr__(self, "source_pointer", pointer)
        object.__setattr__(self, "score_breakdown", breakdown)
        object.__setattr__(self, "acl_labels", _normalize_labels(self.acl_labels))
        if not isinstance(self.text, str):
            raise ValueError("text must be a str")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "channel": self.channel,
            "in_channel_rank": self.in_channel_rank,
            "status_label": self.status_label,
            "source_pointer": _json_copy(self.source_pointer, "source_pointer"),
            "payload_ref": self.payload_ref,
            "score_breakdown": _json_copy(self.score_breakdown, "score_breakdown"),
            "acl_labels": list(self.acl_labels),
            "selection_mode": self.selection_mode,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalCard:
        payload = _require_mapping(data, "retrieval_card")
        labels = payload.get("acl_labels", [])
        if not isinstance(labels, list):
            raise ValueError("acl_labels must be a list")
        raw_text = payload.get("text", "")
        if not isinstance(raw_text, str):
            raise ValueError("text must be a str")
        return cls(
            kind=payload.get("kind"),  # type: ignore[arg-type]
            channel=payload.get("channel"),  # type: ignore[arg-type]
            in_channel_rank=payload.get("in_channel_rank", 0),
            status_label=str(payload.get("status_label") or ""),
            source_pointer=_require_mapping(payload.get("source_pointer"), "source_pointer"),
            payload_ref=str(payload.get("payload_ref") or ""),
            score_breakdown=_require_mapping(payload.get("score_breakdown"), "score_breakdown"),
            acl_labels=tuple(str(item) for item in labels),
            selection_mode=payload.get("selection_mode", "exploitation"),
            text=raw_text,
        )


@dataclass(frozen=True)
class RetrievalQuery:
    """Query text plus an optional dependency-pattern signature (Q122-A)."""

    text: str = ""
    dependency_pattern: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("text must be a str")
        pattern = self.dependency_pattern
        if pattern is None:
            return
        copied = _require_mapping(pattern, "dependency_pattern")
        normalized: dict[str, str] = {}
        for key in ("wanted_artifact", "wanted_effect"):
            if key not in copied or copied[key] is None:
                continue
            normalized[key] = _require_str(copied[key], f"dependency_pattern.{key}")
        object.__setattr__(self, "dependency_pattern", normalized or None)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "dependency_pattern": self.dependency_pattern}


def coerce_query(value: RetrievalQuery | Mapping[str, Any] | str | None) -> RetrievalQuery:
    if value is None:
        return RetrievalQuery()
    if isinstance(value, RetrievalQuery):
        return value
    if isinstance(value, str):
        return RetrievalQuery(text=value)
    payload = _require_mapping(value, "query")
    pattern = payload.get("dependency_pattern")
    return RetrievalQuery(
        text=str(payload.get("text") or ""),
        dependency_pattern=None if pattern is None else dict(pattern),
    )


@dataclass(frozen=True)
class RetrievalProfile:
    """Per-corpus budgets inside the causal-memory lane. Their sum cannot exceed 12.

    ``provisional_slots`` is the provisional channel quota (default 4). It is
    not a fourth corpus budget and is not added to that sum. For ``n >= 2``
    the quota is filled as ``n - 1`` exploitation seats plus one exploration
    seat.
    """

    fork_branch: int = DEFAULT_FORK_BRANCH_QUOTA
    projection: int = DEFAULT_PROJECTION_QUOTA
    segment_atom: int = DEFAULT_SEGMENT_ATOM_QUOTA
    provisional_slots: int = PROVISIONAL_SLOTS
    rank_mode: RankMode = RANK_MODE_BM25

    def __post_init__(self) -> None:
        object.__setattr__(self, "rank_mode", _coerce_rank_mode(self.rank_mode))

    def to_dict(self) -> dict[str, int | str]:
        return {
            "fork_branch": self.fork_branch,
            "projection": self.projection,
            "segment_atom": self.segment_atom,
            "provisional_slots": self.provisional_slots,
            "rank_mode": self.rank_mode,
        }


def _coerce_rank_mode(value: Any) -> RankMode:
    if not isinstance(value, str) or value not in RANK_MODES:
        raise ValueError(f"rank_mode must be one of {sorted(RANK_MODES)}, got {value!r}")
    return value  # type: ignore[return-value]


def _quota_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QuotaError(f"{field} must be an int", total=0)
    if value < 0:
        raise QuotaError(f"{field} must be >= 0", total=value)
    return value


def coerce_profile(value: RetrievalProfile | Mapping[str, Any] | None) -> RetrievalProfile:
    if value is None:
        profile = RetrievalProfile()
    elif isinstance(value, RetrievalProfile):
        profile = value
    else:
        payload = _require_mapping(value, "profile")
        profile = RetrievalProfile(
            fork_branch=_quota_int(payload.get("fork_branch", DEFAULT_FORK_BRANCH_QUOTA), "fork_branch"),
            projection=_quota_int(payload.get("projection", DEFAULT_PROJECTION_QUOTA), "projection"),
            segment_atom=_quota_int(
                payload.get("segment_atom", DEFAULT_SEGMENT_ATOM_QUOTA), "segment_atom"
            ),
            provisional_slots=_quota_int(
                payload.get("provisional_slots", PROVISIONAL_SLOTS), "provisional_slots"
            ),
            rank_mode=_coerce_rank_mode(payload.get("rank_mode", RANK_MODE_BM25)),
        )
    fork_branch = _quota_int(profile.fork_branch, "fork_branch")
    projection = _quota_int(profile.projection, "projection")
    segment_atom = _quota_int(profile.segment_atom, "segment_atom")
    provisional_slots = _quota_int(profile.provisional_slots, "provisional_slots")
    rank_mode = _coerce_rank_mode(profile.rank_mode)
    total = fork_branch + projection + segment_atom
    if total > CAUSAL_MEMORY_CAP:
        raise QuotaError(
            f"retrieval profile sums to {total}, above the causal-memory cap {CAUSAL_MEMORY_CAP}",
            total=total,
        )
    return RetrievalProfile(
        fork_branch=fork_branch,
        projection=projection,
        segment_atom=segment_atom,
        provisional_slots=provisional_slots,
        rank_mode=rank_mode,
    )


@dataclass(frozen=True)
class AssembledPath:
    """Read-time claim neighborhood. Not a stored object."""

    view_revision_id: str
    claim_id: str
    entities: tuple[str, ...]
    edges: tuple[tuple[str, str, str], ...]
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_revision_id": self.view_revision_id,
            "claim_id": self.claim_id,
            "entities": list(self.entities),
            "edges": [
                {"claim_id": claim_id, "source_block_id": source, "target_block_id": target}
                for claim_id, source, target in self.edges
            ],
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class SelectionAudit:
    """Logged draw for the provisional exploration seat (Q87-A)."""

    eligible_set: tuple[str, ...]
    selection_propensity: float | None
    seed: str
    selection_policy_version: str
    backfill: bool
    backfill_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible_set": list(self.eligible_set),
            "selection_propensity": self.selection_propensity,
            "seed": self.seed,
            "selection_policy_version": self.selection_policy_version,
            "backfill": self.backfill,
            "backfill_reason": self.backfill_reason,
        }


@dataclass(frozen=True)
class RetrievalResult:
    cards: tuple[RetrievalCard, ...]
    feature_version: str
    truncated_by_rerank_clamp: bool = False
    selection_audit: SelectionAudit | None = None
    rank_tier: str = RANK_MODE_BM25

    def to_dict(self) -> dict[str, Any]:
        audit = None if self.selection_audit is None else self.selection_audit.to_dict()
        return {
            "cards": [card.to_dict() for card in self.cards],
            "feature_version": self.feature_version,
            "truncated_by_rerank_clamp": self.truncated_by_rerank_clamp,
            "selection_audit": audit,
            "rank_tier": self.rank_tier,
        }


class CorpusAdapter(Protocol):
    """One corpus search. Results keep that corpus's kind."""

    name: str

    def search(
        self,
        query: RetrievalQuery,
        limit: int,
        principal_labels: Sequence[str],
    ) -> list[RetrievalCard]:
        """Return at most ``limit`` cards visible to ``principal_labels``."""


class RerankHook(Protocol):
    def rerank(
        self, cards: Sequence[RetrievalCard], context: Mapping[str, Any]
    ) -> Sequence[RetrievalCard]:
        """Return cards in reranked order. The default hook is identity."""


class IdentityRerank:
    """Placeholder ranker. Records no mutation; feature version lives on the facade."""

    def rerank(
        self, cards: Sequence[RetrievalCard], context: Mapping[str, Any]
    ) -> Sequence[RetrievalCard]:
        del context
        return list(cards)


def default_acl_filter(card: RetrievalCard, principal_labels: Sequence[str]) -> bool:
    """Keep a card when its labels intersect the principal's labels."""
    if isinstance(principal_labels, (str, bytes)) or not isinstance(principal_labels, Sequence):
        raise ValueError("principal_labels must be a sequence of str")
    principal = set(principal_labels)
    return bool(principal.intersection(card.acl_labels))


class _RankPlan:
    """Requested tier for one retrieve call. ``actual`` may fall back to BM25."""

    def __init__(self, requested: str, embedder: EmbedFn | None) -> None:
        self.requested = _coerce_rank_mode(requested)
        self.embedder = embedder
        self.actual = RANK_MODE_BM25
        self.query_vector: list[float] | None = None
        self.downgraded = False

    def prepare(self, query_text: str) -> None:
        """Choose the tier actually used. Empty queries and dim-0 vectors stay BM25."""
        self.actual = RANK_MODE_BM25
        self.query_vector = None
        self.downgraded = False
        if self.requested != RANK_MODE_HYBRID or not query_text or self.embedder is None:
            return
        vector = _real_vector(self.embedder(query_text))
        if vector is None:
            return
        self.actual = RANK_MODE_HYBRID
        self.query_vector = vector


def _prepare_rank_plan(query: RetrievalQuery, plan: _RankPlan | None) -> _RankPlan:
    if plan is not None:
        return plan
    owned = _RankPlan(RANK_MODE_BM25, None)
    owned.prepare(query.text)
    return owned


def _document_text(*parts: str) -> str:
    return "\n".join(parts)


def _pending_score() -> dict[str, Any]:
    return {"bm25": 0.0, "lexical": 0.0, "rank_tier": RANK_MODE_BM25}


def _real_vector(value: object) -> list[float] | None:
    """A usable vector has dimension > 0. The projection stub does not qualify."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    if len(value) == 0:
        return None
    coords: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        coords.append(number)
    return coords


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        dot += left_value * right_value
        left_sq += left_value * left_value
        right_sq += right_value * right_value
    if left_sq == 0.0 or right_sq == 0.0:
        return 0.0
    return dot / (math.sqrt(left_sq) * math.sqrt(right_sq))


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [1.0] * len(values)
    span = high - low
    return [(value - low) / span for value in values]


def _score_breakdown(
    bm25: float,
    plan: _RankPlan,
    *,
    tier: str,
    vector: float | None = None,
    combined: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"bm25": bm25, "lexical": bm25, "rank_tier": tier}
    if plan.requested != tier:
        payload["requested_rank_tier"] = plan.requested
    if vector is not None:
        payload["vector"] = vector
        payload["bm25_weight"] = HYBRID_BM25_WEIGHT
        payload["vector_weight"] = HYBRID_VECTOR_WEIGHT
    if combined is not None:
        payload["combined"] = combined
    return payload


def _order_by(scores: Sequence[float]) -> list[int]:
    positions = list(range(len(scores)))
    positions.sort(key=lambda pos: scores[pos], reverse=True)
    return positions


def _similarities(
    plan: _RankPlan, documents: Sequence[str], kept: Sequence[int]
) -> list[float] | None:
    embedder = plan.embedder
    query_vector = plan.query_vector
    if embedder is None or query_vector is None:
        return None
    scored: list[float] = []
    for index in kept:
        document_vector = _real_vector(embedder(documents[index]))
        if document_vector is None:
            return None
        similarity = _cosine(query_vector, document_vector)
        if similarity is None:
            return None
        scored.append(similarity)
    return scored


def _rank_hits(
    cards: Sequence[RetrievalCard],
    documents: Sequence[str],
    query: RetrievalQuery,
    plan: _RankPlan,
) -> list[RetrievalCard]:
    """Order cards by BM25, or by a BM25/vector mix when this call has real vectors.

    An empty query keeps every card at the same score and in the given order.
    A non-empty query drops BM25 zeros. Hybrid fusion does not bring them back.
    """
    if len(cards) != len(documents):
        raise ValueError("cards and documents must be the same length")
    if not cards:
        return []
    bm25_scores = BM25Index(documents).scores(query.text)
    if not query.text:
        return [
            replace(card, score_breakdown=_score_breakdown(0.0, plan, tier=RANK_MODE_BM25))
            for card in cards
        ]
    kept = [index for index, score in enumerate(bm25_scores) if score > 0.0]
    similarities: list[float] | None = None
    if plan.actual == RANK_MODE_HYBRID:
        similarities = _similarities(plan, documents, kept)
        if similarities is None:
            plan.actual = RANK_MODE_BM25
            plan.query_vector = None
            plan.downgraded = True
    if similarities is None:
        bm25_kept = [bm25_scores[index] for index in kept]
        return [
            replace(
                cards[kept[pos]],
                score_breakdown=_score_breakdown(bm25_kept[pos], plan, tier=RANK_MODE_BM25),
            )
            for pos in _order_by(bm25_kept)
        ]
    bm25_kept = [bm25_scores[index] for index in kept]
    norm_bm25 = _minmax(bm25_kept)
    norm_vector = _minmax(similarities)
    combined = [
        HYBRID_BM25_WEIGHT * bm25_norm + HYBRID_VECTOR_WEIGHT * vector_norm
        for bm25_norm, vector_norm in zip(norm_bm25, norm_vector, strict=True)
    ]
    ranked: list[RetrievalCard] = []
    for pos in _order_by(combined):
        ranked.append(
            replace(
                cards[kept[pos]],
                score_breakdown=_score_breakdown(
                    bm25_kept[pos],
                    plan,
                    tier=RANK_MODE_HYBRID,
                    vector=similarities[pos],
                    combined=combined[pos],
                ),
            )
        )
    return ranked


def _rank_document_indexes(
    query_text: str, documents: Sequence[str]
) -> list[tuple[int, dict[str, Any]]]:
    """BM25 order for documents that are not retrieval cards.

    An empty query keeps every document at score 0 in the given order.
    A non-empty query drops BM25 zeros. Equal scores keep the given order.
    This helper does not embed. Hybrid fusion is not applied.
    """
    plan = _RankPlan(RANK_MODE_BM25, None)
    plan.prepare(query_text)
    if not documents:
        return []
    scores = BM25Index(documents).scores(query_text)
    if not query_text:
        return [
            (index, _score_breakdown(0.0, plan, tier=RANK_MODE_BM25))
            for index in range(len(documents))
        ]
    kept = [index for index, score in enumerate(scores) if score > 0.0]
    bm25_kept = [scores[index] for index in kept]
    return [
        (kept[pos], _score_breakdown(bm25_kept[pos], plan, tier=RANK_MODE_BM25))
        for pos in _order_by(bm25_kept)
    ]


def _channel_for_fork_status(status: str | None) -> CardChannel | None:
    if status == ForkStatus.VALIDATED:
        return "verified"
    if status == ForkStatus.PROVISIONAL:
        return "provisional"
    return None


def _status_label_for_fork(status: str) -> str:
    if status == ForkStatus.PROVISIONAL:
        return FORK_CANDIDATE_ROLE
    return status


def _clip_text(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3] + "..."


def _fork_retrieval_text(sections: Any) -> str:
    """Seven-section snapshot in ``SEVEN_SECTION_KEYS`` order. Not guidance."""
    if not isinstance(sections, Mapping):
        return ""
    lines: list[str] = []
    for key in SEVEN_SECTION_KEYS:
        value = sections.get(key)
        if isinstance(value, str) and value:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _synopsis_from_observation(observation: Mapping[str, Any]) -> str:
    """Title plus summary only. ``sample`` is a raw excerpt and stays off the card."""
    title = observation.get("title")
    title_text = title.strip() if isinstance(title, str) else ""
    summary = observation.get("summary")
    body = ""
    if isinstance(summary, str):
        body = summary.strip()
    elif isinstance(summary, list):
        lines = [item.strip() for item in summary if isinstance(item, str) and item.strip()]
        body = "; ".join(lines)
    if title_text and body:
        return f"{title_text}: {body}"
    return title_text or body


def _branch_retrieval_text(branch: Mapping[str, Any], synopses: Mapping[str, str]) -> str:
    """Join the first N ledger synopses. Fall back when none of them resolve."""
    branch_id = str(branch.get("branch_id") or "")
    status = branch.get("evidence_status")
    status_text = status if isinstance(status, str) else ""
    sequence = branch.get("ao_sequence")
    parts: list[str] = []
    if isinstance(sequence, list):
        for ao_id in sequence[:BRANCH_SYNOPSIS_LIMIT]:
            if not isinstance(ao_id, str) or not ao_id:
                continue
            synopsis = synopses.get(ao_id, "")
            if not synopsis:
                continue
            parts.append(_clip_text(synopsis, BRANCH_SYNOPSIS_LINE_CHARS))
    if not parts:
        return f"{branch_id} {status_text}".strip()
    return _clip_text(" | ".join(parts), BRANCH_SUMMARY_CHARS)


def _limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be an int >= 0")
    return limit


def _principal_set(principal_labels: Sequence[str]) -> set[str]:
    if isinstance(principal_labels, (str, bytes)) or not isinstance(principal_labels, Sequence):
        raise ValueError("principal_labels must be a sequence of str")
    return set(principal_labels)


def _labels_visible(labels: Sequence[str], principal: set[str]) -> bool:
    return bool(principal.intersection(labels))


class ForkBranchAdapter:
    """Wraps ``CausalExperiencesStore`` reads. Latest revision only.

    Validated revisions enter the verified channel. Provisional revisions are
    consumed as ForkCandidate and enter the provisional channel. Draft and
    invalidated revisions stay out of retrieval. A branch card uses
    ``branch_retrieval_channel`` on its own evidence status; it does not
    inherit the parent revision channel. ACL labels are the fork workspace id:
    the store has no separate label column. Invisible cards are dropped before
    the quota is spent.

    Online code must not call ``materialize_strength(None)``, ``events()``,
    ``list_bridges``, or ``ProjectionRegistry.to_dict``. Those are offline
    management dumps and do not apply principal ACL.

    Fork card ``text`` is the latest revision's seven sections. Branch card
    ``text`` is a bounded synopsis of ``ao_sequence`` looked up in the session
    AO ledger (``session_dir`` + ``session_id``). Guidance and raw AO bodies
    are not copied onto the card. When that ledger has none of the first
    ``BRANCH_SYNOPSIS_LIMIT`` ids, the branch text falls back to
    ``branch_id`` and ``evidence_status``. Rank documents stay the serialized
    revision either way.
    """

    name = "fork_branch"

    def __init__(
        self,
        root: str | Path,
        store: CausalExperiencesStore | None = None,
        *,
        session_dir: str | Path | None = None,
        session_id: str | None = None,
    ) -> None:
        self._root = Path(root)
        self._store = store if store is not None else CausalExperiencesStore(self._root)
        self._session_dir = None if session_dir is None else Path(session_dir)
        self._session_id = session_id if isinstance(session_id, str) and session_id else None

    def search(
        self,
        query: RetrievalQuery | Mapping[str, Any] | None,
        limit: int,
        principal_labels: Sequence[str] = (),
        *,
        rank_plan: _RankPlan | None = None,
    ) -> list[RetrievalCard]:
        """ACL-filter fork and branch cards, then keep at most ``limit`` visible ones.

        Document text is the serialized latest revision, the same blob previously
        passed to the substring hit test. ACL drops a revision before it enters
        the BM25 corpus or spends quota.
        """
        parsed = coerce_query(query)
        cap = _limit(limit)
        principal = _principal_set(principal_labels)
        plan = _prepare_rank_plan(parsed, rank_plan)
        cards: list[RetrievalCard] = []
        documents: list[str] = []
        synopses: dict[str, str] | None = None
        for fork_node_id in self._fork_ids():
            loaded = self._store.get_fork_node(fork_node_id)
            if loaded is None:
                continue
            latest = loaded.get("latest_revision")
            if not isinstance(latest, dict):
                continue
            channel = _channel_for_fork_status(
                latest.get("status") if isinstance(latest.get("status"), str) else None
            )
            if channel is None:
                continue
            status = str(latest["status"])
            raw_workspace = latest.get("workspace_id")
            if not isinstance(raw_workspace, str) or not raw_workspace:
                continue
            labels = _normalize_labels((raw_workspace,))
            if not _labels_visible(labels, principal):
                continue
            blob = json.dumps(latest, ensure_ascii=False, sort_keys=True)
            revision_id = str(latest.get("revision_id") or "")
            cards.append(
                RetrievalCard(
                    kind="fork",
                    channel=channel,
                    in_channel_rank=0,
                    status_label=_status_label_for_fork(status),
                    source_pointer={
                        "corpus": self.name,
                        "fork_node_id": fork_node_id,
                        "revision_id": revision_id,
                    },
                    payload_ref=f"fork:{fork_node_id}@{revision_id}",
                    score_breakdown=_pending_score(),
                    acl_labels=labels,
                    text=_fork_retrieval_text(latest.get("seven_sections")),
                )
            )
            documents.append(blob)
            branches = latest.get("branches") if isinstance(latest.get("branches"), list) else []
            ordered = sorted(
                (item for item in branches if isinstance(item, dict)),
                key=lambda item: str(item.get("branch_id") or ""),
            )
            for branch in ordered:
                branch_id = str(branch.get("branch_id") or "")
                if not branch_id:
                    continue
                if synopses is None:
                    synopses = self._ao_synopses()
                branch_channel = branch_retrieval_channel(branch, status)
                if branch_channel not in CARD_CHANNELS or branch_channel == "fact":
                    continue
                cards.append(
                    RetrievalCard(
                        kind="branch",
                        channel=branch_channel,  # type: ignore[arg-type]
                        in_channel_rank=0,
                        status_label=_status_label_for_fork(status),
                        source_pointer={
                            "corpus": self.name,
                            "fork_node_id": fork_node_id,
                            "revision_id": revision_id,
                            "branch_id": branch_id,
                            "evidence_status": branch.get("evidence_status"),
                        },
                        payload_ref=f"branch:{fork_node_id}@{revision_id}/{branch_id}",
                        score_breakdown=_pending_score(),
                        acl_labels=labels,
                        text=_branch_retrieval_text(branch, synopses),
                    )
                )
                documents.append(blob)
        return _rank_hits(cards, documents, parsed, plan)[:cap]

    def _fork_ids(self) -> list[str]:
        index_path = self._root / NAMESPACE_DIRNAME / INDEX_FILENAME
        if not index_path.is_file():
            return []
        ordered: list[str] = []
        seen: set[str] = set()
        for line in index_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            fork_node_id = row.get("fork_node_id")
            if isinstance(fork_node_id, str) and fork_node_id and fork_node_id not in seen:
                seen.add(fork_node_id)
                ordered.append(fork_node_id)
        return ordered

    def _ao_synopses(self) -> dict[str, str]:
        """Map AO id to synopsis text. Missing ledger yields an empty map."""
        if self._session_dir is None or not self._session_id:
            return {}
        if not (self._session_dir / LEDGER_FILENAME).is_file():
            return {}
        found: dict[str, str] = {}
        for record in AOLedger(self._session_dir, self._session_id).records():
            synopsis = _synopsis_from_observation(record.observation)
            if synopsis:
                found[record.ao_id] = synopsis
        return found


def _claim_fields(
    card: InfluenceProjection, views: Mapping[str, InfluenceViewRevision]
) -> dict[str, str] | None:
    if card.kind != "claim":
        return None
    view = views.get(card.source_pointer.view_revision_id)
    if view is not None:
        for claim in view.claims:
            if claim.claim_id == card.source_pointer.block_or_claim_id:
                return {
                    "carried_artifact": claim.carried_artifact,
                    "downstream_effect": claim.downstream_effect,
                }
    summary = card.summary
    marker = " -> "
    if marker in summary:
        artifact, effect = summary.split(marker, 1)
        if artifact and effect:
            return {"carried_artifact": artifact, "downstream_effect": effect}
    return {"carried_artifact": "", "downstream_effect": ""}


def _pattern_matches(fields: Mapping[str, str], pattern: Mapping[str, str]) -> bool:
    artifact = pattern.get("wanted_artifact")
    effect = pattern.get("wanted_effect")
    if artifact and fields.get("carried_artifact") != artifact:
        return False
    if effect and fields.get("downstream_effect") != effect:
        return False
    return True


def _remember_ao_id(ao_id: str, ordered: list[str], seen: set[str]) -> None:
    if ao_id and ao_id not in seen:
        seen.add(ao_id)
        ordered.append(ao_id)


def _projection_source_ao_ids(
    view: InfluenceViewRevision, kind: str, entity_id: str
) -> list[str]:
    """AO ids behind one block or claim. Used only to resolve match text."""
    if kind == "block":
        for block in view.blocks:
            if block.block_id == entity_id:
                return [ref.ao_id for ref in block.source_refs if ref.ao_id]
        return []
    if kind != "claim":
        return []
    for claim in view.claims:
        if claim.claim_id != entity_id:
            continue
        ordered: list[str] = []
        seen: set[str] = set()
        for ref in (*claim.source_evidence_refs, *claim.target_evidence_refs):
            _remember_ao_id(ref.ao_id, ordered, seen)
        linked = {claim.source_block_id, claim.target_block_id}
        for block in view.blocks:
            if block.block_id in linked:
                for ref in block.source_refs:
                    _remember_ao_id(ref.ao_id, ordered, seen)
        return ordered
    return []


class InfluenceProjectionAdapter:
    """Wraps ``ProjectionRegistry``. Card status maps onto the retrieval channel.

    Claim signature fields are read from the bound frozen view (the projection
    card stores them only inside its summary). Neighborhood expansion walks
    claim edges of that same view and never writes the registry. Online search
    calls ``ProjectionRegistry.search`` with ``principal_labels`` and never
    ``to_dict``. Card-id neighborhood expansion uses the same intersection:
    the default principal sees nothing, and an invisible card is an unknown
    id. ``materialize_strength(None)``, ``events()``, and ``list_bridges``
    are offline management paths; online code must not call them.

    ``content_resolver`` maps an AO id to that AO's body text. Resolved text
    is concatenated into the BM25 document only. It is not written onto the
    returned card or its source pointer. A missing resolver, an unbound view,
    or an empty resolution leaves the document as the summary, card id, and
    semantic type. Neighborhood walks only ``relation_type=influence``.
    """

    name = "projection"

    def __init__(
        self,
        registry: ProjectionRegistry,
        views: Sequence[InfluenceViewRevision] | None = None,
        content_resolver: Callable[[str], str] | None = None,
    ) -> None:
        self._registry = registry
        self._views: dict[str, InfluenceViewRevision] = {}
        self._content_resolver = content_resolver
        for revision in views or ():
            self.bind_view(revision)

    def bind_view(self, revision: InfluenceViewRevision) -> None:
        if not isinstance(revision, InfluenceViewRevision):
            raise ValueError("revision must be an InfluenceViewRevision")
        self._views[revision.revision_id] = revision

    def search(
        self,
        query: RetrievalQuery | Mapping[str, Any] | None,
        limit: int,
        principal_labels: Sequence[str] = (),
        *,
        rank_plan: _RankPlan | None = None,
    ) -> list[RetrievalCard]:
        """ACL via ``ProjectionRegistry.search``, then keep at most ``limit`` hits.

        Does not call ``ProjectionRegistry.to_dict``. That dump, along with
        ``materialize_strength(None)``, ``events()``, and ``list_bridges``, is
        an offline management path and must not be used online. ``embedding_stub``
        is not read; dimension 0 there is not a vector tier.
        """
        parsed = coerce_query(query)
        cap = _limit(limit)
        plan = _prepare_rank_plan(parsed, rank_plan)
        pattern = parsed.dependency_pattern
        raw_cards = list(self._registry.search(principal_labels=principal_labels))
        cards: list[RetrievalCard] = []
        documents: list[str] = []
        for card in raw_cards:
            fields = _claim_fields(card, self._views)
            if card.kind == "claim" and pattern is not None:
                if fields is None or not _pattern_matches(fields, pattern):
                    continue
            kind: CardKind = "claim_projection" if card.kind == "claim" else "block_projection"
            pointer: dict[str, Any] = {
                "corpus": self.name,
                "view_revision_id": card.source_pointer.view_revision_id,
                "block_or_claim_id": card.source_pointer.block_or_claim_id,
                "card_id": card.card_id,
                "session_id": card.session_id,
            }
            if fields is not None:
                pointer["carried_artifact"] = fields["carried_artifact"]
                pointer["downstream_effect"] = fields["downstream_effect"]
            cards.append(
                RetrievalCard(
                    kind=kind,
                    channel=card.status,  # type: ignore[arg-type]
                    in_channel_rank=0,
                    status_label=card.status,
                    source_pointer=pointer,
                    payload_ref=card.card_id,
                    score_breakdown=_pending_score(),
                    acl_labels=card.acl_labels,
                )
            )
            documents.append(
                _document_text(
                    card.summary,
                    card.card_id,
                    card.semantic_type,
                    *self._resolved_match_parts(card),
                )
            )
        return _rank_hits(cards, documents, parsed, plan)[:cap]

    def _resolved_match_parts(self, card: InfluenceProjection) -> tuple[str, ...]:
        """Body snippets for the BM25 document. Never copied onto the card."""
        resolver = self._content_resolver
        if resolver is None:
            return ()
        view = self._views.get(card.source_pointer.view_revision_id)
        if view is None:
            return ()
        parts: list[str] = []
        for ao_id in _projection_source_ao_ids(
            view, card.kind, card.source_pointer.block_or_claim_id
        ):
            text = resolver(ao_id)
            if isinstance(text, str) and text:
                parts.append(text)
        return tuple(parts)

    def expand_claim_neighborhood(
        self,
        view_revision_id: str | None = None,
        claim_id: str | None = None,
        max_entities: int = PATH_MAX_ENTITIES,
        *,
        card_id: str | None = None,
        principal_labels: Sequence[str] = (),
    ) -> AssembledPath:
        """Assemble a bounded path along influence edges of one frozen view.

        This entry is for the Memory Agent. OpenViking does not decide
        whether a diagnostic agent may call it; that fence sits on the
        agent side.

        ``card_id`` is the public argument. The server resolves that claim
        projection to its frozen view revision and claim id, then walks
        the same path as the compatibility ``(view_revision_id, claim_id)``
        form. Resolution intersects ``principal_labels`` with the card's
        labels. The default ``()`` sees nothing. A missing id and an id the
        principal cannot see raise the same ``ValueError`` (``unknown card
        id``). A visible card that is not a claim projection names its kind
        and is not guessed into another walk. The compatibility form does
        not resolve a card id and does not apply this principal.

        Claims whose ``relation_type`` is not influence, including ``skip``,
        are left off the adjacency before the walk. A skip seed still places
        both endpoint blocks in the start set, and its own edge is not
        returned. The walk is bidirectional, stops at ``max_entities`` (never
        above ``PATH_MAX_ENTITIES``), and stops when the next block is already
        on the path. Nothing is persisted.
        """
        if card_id is not None:
            if view_revision_id is not None or claim_id is not None:
                raise ValueError(
                    "card_id cannot be combined with view_revision_id or claim_id"
                )
            view_revision_id, claim_id = self._resolve_claim_card(
                card_id, principal_labels
            )
        elif not view_revision_id or not claim_id:
            raise ValueError("pass card_id, or both view_revision_id and claim_id")
        if isinstance(max_entities, bool) or not isinstance(max_entities, int):
            raise ValueError("max_entities must be an int")
        if max_entities < 1:
            raise ValueError("max_entities must be >= 1")
        if max_entities > PATH_MAX_ENTITIES:
            raise ValueError(
                f"max_entities cannot exceed the global ceiling {PATH_MAX_ENTITIES}"
            )
        revision_id = _require_str(view_revision_id, "view_revision_id")
        seed_id = _require_str(claim_id, "claim_id")
        view = self._views.get(revision_id)
        if view is None:
            raise ValueError(f"unknown view revision {revision_id}")
        claims = [claim for claim in view.claims if isinstance(claim, InfluenceClaim)]
        seed = next((claim for claim in claims if claim.claim_id == seed_id), None)
        if seed is None:
            raise ValueError(f"unknown claim {seed_id} on view {revision_id}")

        entities: list[str] = []
        seen: set[str] = set()
        edges: list[tuple[str, str, str]] = []
        truncated = False

        def _try_add(block_id: str) -> bool:
            nonlocal truncated
            if block_id in seen:
                truncated = True
                return False
            if len(seen) >= max_entities:
                truncated = True
                return False
            seen.add(block_id)
            entities.append(block_id)
            return True

        _try_add(seed.source_block_id)
        _try_add(seed.target_block_id)
        both_present = seed.source_block_id in seen and seed.target_block_id in seen
        if both_present and seed.relation_type == RELATION_INFLUENCE:
            edges.append((seed.claim_id, seed.source_block_id, seed.target_block_id))
        elif seed.target_block_id not in seen:
            truncated = True

        adjacency: dict[str, list[InfluenceClaim]] = {}
        for claim in claims:
            if claim.relation_type != RELATION_INFLUENCE:
                continue
            adjacency.setdefault(claim.source_block_id, []).append(claim)
            adjacency.setdefault(claim.target_block_id, []).append(claim)
        for bucket in adjacency.values():
            bucket.sort(key=lambda claim: claim.claim_id)

        visited_claims = {seed.claim_id}
        queue: deque[str] = deque(block_id for block_id in entities)
        while queue and not (len(seen) >= max_entities and truncated):
            node = queue.popleft()
            for claim in adjacency.get(node, ()):
                if claim.claim_id in visited_claims:
                    continue
                other = (
                    claim.target_block_id if claim.source_block_id == node else claim.source_block_id
                )
                if other in seen:
                    visited_claims.add(claim.claim_id)
                    truncated = True
                    continue
                if len(seen) >= max_entities:
                    truncated = True
                    continue
                visited_claims.add(claim.claim_id)
                seen.add(other)
                entities.append(other)
                edges.append((claim.claim_id, claim.source_block_id, claim.target_block_id))
                queue.append(other)

        usages = [
            {
                "kind": "evidence",
                "bridge_id": edge_claim,
                "source_ref": {"type": "block", "id": source},
                "target_ref": {"type": "block", "id": target},
                "relation_type": "claim",
            }
            for edge_claim, source, target in edges
        ]
        violations = check_path_budget(
            entities,
            usages,
            max_entities=max_entities,
            max_hypothesis_hops=PATH_MAX_HYPOTHESIS_HOPS,
        )
        if violations:
            raise ValueError(
                "assembled claim path exceeds the path budget: "
                + ", ".join(item.code for item in violations)
            )
        return AssembledPath(
            view_revision_id=revision_id,
            claim_id=seed_id,
            entities=tuple(entities),
            edges=tuple(edges),
            truncated=truncated,
        )

    def _resolve_claim_card(
        self, card_id: str, principal_labels: Sequence[str] = ()
    ) -> tuple[str, str]:
        """Reverse-lookup a claim projection visible to ``principal_labels``.

        Does not accept block, fork, or branch ids. The card is loaded, then
        its labels are intersected with the principal. The default principal
        is empty and sees nothing. A missing id and an id that does not meet
        the principal raise the same ``unknown card id`` error, before the
        kind is named. A visible card of another kind still names that kind.
        """
        principal = _principal_set(principal_labels)
        found = self._registry.get(card_id)
        if found is None or not _labels_visible(found.acl_labels, principal):
            raise ValueError(f"unknown card id {card_id!r}")
        if found.kind != "claim":
            raise ValueError(
                f"card {card_id!r} is a {found.kind} projection, not a claim projection"
            )
        return found.source_pointer.view_revision_id, found.source_pointer.block_or_claim_id


class SegmentAtomAdapter:
    """Wraps ``list_ao_ledger``. Facts use channel ``fact`` and skip 8/4 slotting.

    ``action.record_kind == "segment"`` selects a segment card; every other AO
    is an atom. Labels come only from ``action.acl_labels``. An action with no
    labels is rejected and stays invisible: the session id is not used as a
    stand-in label, and the row does not consume quota.
    """

    name = "segment_atom"

    def __init__(self, session_dir: str | Path, session_id: str) -> None:
        self._session_dir = Path(session_dir)
        self._session_id = _require_str(session_id, "session_id")

    def search(
        self,
        query: RetrievalQuery | Mapping[str, Any] | None,
        limit: int,
        principal_labels: Sequence[str] = (),
        *,
        rank_plan: _RankPlan | None = None,
    ) -> list[RetrievalCard]:
        """Drop unlabeled or non-intersecting facts, then keep at most ``limit``.

        The facade does not call this while the fact lane is paused. The
        document is the action/observation blob plus the AO id.
        """
        parsed = coerce_query(query)
        cap = _limit(limit)
        principal = _principal_set(principal_labels)
        plan = _prepare_rank_plan(parsed, rank_plan)
        cards: list[RetrievalCard] = []
        documents: list[str] = []
        for item in self._items():
            action = item.get("action") if isinstance(item.get("action"), dict) else {}
            observation = item.get("observation") if isinstance(item.get("observation"), dict) else {}
            record_kind = action.get("record_kind")
            kind: CardKind = "segment" if record_kind == "segment" else "atom"
            raw_labels = action.get("acl_labels")
            if not isinstance(raw_labels, list) or not raw_labels:
                continue
            labels = _normalize_labels(tuple(str(label) for label in raw_labels))
            if not _labels_visible(labels, principal):
                continue
            ao_id = str(item.get("ao_id") or "")
            blob = json.dumps({"action": action, "observation": observation}, ensure_ascii=False)
            cards.append(
                RetrievalCard(
                    kind=kind,
                    channel="fact",
                    in_channel_rank=0,
                    status_label="fact",
                    source_pointer={
                        "corpus": self.name,
                        "session_id": self._session_id,
                        "ao_id": ao_id,
                        "sequence": item.get("sequence"),
                        "captured_state": item.get("captured_state"),
                    },
                    payload_ref=f"{kind}:{self._session_id}/{ao_id}",
                    score_breakdown=_pending_score(),
                    acl_labels=labels,
                )
            )
            documents.append(_document_text(blob, ao_id))
        return _rank_hits(cards, documents, parsed, plan)[:cap]

    def _items(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        cursor: str | None = None
        watermark: str | None = None
        while True:
            page = list_ao_ledger(
                self._session_dir,
                self._session_id,
                cursor=cursor,
                limit=200,
                snapshot_watermark=watermark,
                include_live=True,
            )
            items.extend(item for item in page.items if isinstance(item, dict))
            if not page.next_cursor:
                return items
            cursor = page.next_cursor
            watermark = page.snapshot_watermark


def _nonempty_text(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _canonical_token_value(value: object) -> str:
    """Lowercase token value. Other characters become underscores.

    Empty, whitespace-only, and non-text values are the explicit ``unknown``
    token. Underscores produced from punctuation are kept.
    """
    if not isinstance(value, str):
        return "unknown"
    stripped = value.strip().lower()
    if not stripped:
        return "unknown"
    normalized = "".join(
        char if ("a" <= char <= "z") or ("0" <= char <= "9") or char == "_" else "_"
        for char in stripped
    )
    return normalized or "unknown"


def _relation_profile_token(claims: Sequence[InfluenceClaim]) -> str:
    """Canonical relation counts, sorted by normalized relation name.

    No member claims yields ``relation_profile:unknown``. A count above
    ``_RELATION_PROFILE_COUNT_CAP`` is written as that cap.
    """
    if not claims:
        return "relation_profile:unknown"
    counts: dict[str, int] = {}
    for claim in claims:
        name = _canonical_token_value(claim.relation_type)
        counts[name] = counts.get(name, 0) + 1
    body = ",".join(
        f"{name}:{min(counts[name], _RELATION_PROFILE_COUNT_CAP)}" for name in sorted(counts)
    )
    return f"relation_profile:{body}"


def _subgraph_members(
    revision: InfluenceViewRevision, subgraph: RetrievalSubgraph
) -> tuple[tuple[InfluenceBlock, ...], tuple[InfluenceClaim, ...]]:
    """Blocks and claims named by this subgraph, in stored id order.

    Ids that are not on the revision are skipped. Other subgraphs on the
    same revision are not included.
    """
    blocks = {block.block_id: block for block in revision.blocks}
    claims = {claim.claim_id: claim for claim in revision.claims}
    chosen_blocks = tuple(
        blocks[block_id] for block_id in subgraph.block_ids if block_id in blocks
    )
    chosen_claims = tuple(
        claims[claim_id] for claim_id in subgraph.claim_ids if claim_id in claims
    )
    return chosen_blocks, chosen_claims


def _subgraph_retrieval_text(
    revision: InfluenceViewRevision, subgraph: RetrievalSubgraph
) -> str:
    """Shared diagnostic document. See ``SUBGRAPH_TEXT_POLICY_VERSION``.

    Query and candidate both use this string. Prose stays in stored member
    order: block summary, input, output, then claim carried_artifact and
    downstream_effect. Typed tokens are appended after that prose, one field
    per line: subgraph failure, then each block's role, kind, and failure,
    then each claim's relation and failure, then the relation profile.

    Block and claim records have no failure field, so those lines are
    ``unknown`` and are not omitted. Ids, participant ids, guidance, evidence
    refs, ACL, raw AO text, and AO range numbers stay out of this string.
    """
    blocks, claims = _subgraph_members(revision, subgraph)
    parts: list[str] = []
    for block in blocks:
        for value in (block.summary, block.input, block.output):
            text = _nonempty_text(value)
            if text:
                parts.append(text)
    for claim in claims:
        for value in (claim.carried_artifact, claim.downstream_effect):
            text = _nonempty_text(value)
            if text:
                parts.append(text)
    failure = "true" if subgraph.failure_marked else "false"
    parts.append(f"subgraph_failure:{failure}")
    for block in blocks:
        parts.append(f"block_role:{_canonical_token_value(block.role)}")
        parts.append(f"block_kind:{_canonical_token_value(block.block_kind)}")
        parts.append("block_failure:unknown")
    for claim in claims:
        parts.append(f"claim_relation:{_canonical_token_value(claim.relation_type)}")
        parts.append("claim_failure:unknown")
    parts.append(_relation_profile_token(claims))
    return "\n".join(parts)


def _position_inside_subgraph(
    grounding: InfluenceGrounding, subgraph: RetrievalSubgraph
) -> bool:
    """Stand-in position parser. Never raises.

    ``block`` matches only ``block:<id>`` against ``block_ids``.
    ``edge`` matches only ``edge:<id>`` against ``claim_ids``.
    The id is the substring after the first colon, with no whitespace trim.
    ``cut``, a missing colon, an empty id, or a prefix that is not the kind
    are outside the subgraph.
    """
    kind = grounding.position_kind
    if kind != "block" and kind != "edge":
        return False
    prefix, separator, ident = grounding.position_ref.partition(":")
    if separator != ":" or not ident:
        return False
    if kind == "block":
        return prefix == "block" and ident in subgraph.block_ids
    return prefix == "edge" and ident in subgraph.claim_ids


def _diagnostic_fork_text(revision: Mapping[str, Any], synopses: Mapping[str, str]) -> str:
    """Seven sections plus branch synopses. Not guidance and not raw AO."""
    parts: list[str] = []
    sections = _fork_retrieval_text(revision.get("seven_sections"))
    if sections:
        parts.append(sections)
    branches = revision.get("branches")
    if not isinstance(branches, list):
        return "\n".join(parts)
    ordered = sorted(
        (item for item in branches if isinstance(item, dict)),
        key=lambda item: str(item.get("branch_id") or ""),
    )
    for branch in ordered:
        if not str(branch.get("branch_id") or ""):
            continue
        synopsis = _branch_retrieval_text(branch, synopses)
        if synopsis:
            parts.append(synopsis)
    return "\n".join(parts)


@dataclass(frozen=True)
class DiagnosticForkHit:
    """One stored fork revision whose grounding falls inside the hit subgraph.

    ``text`` is the seven-section snapshot plus branch action synopses.
    """

    fork_node_id: str
    revision_id: str
    workspace_id: str
    text: str
    influence_grounding: InfluenceGrounding


@dataclass(frozen=True)
class DiagnosticSubgraphHit:
    """One historical retrieval subgraph plus the forks grounded inside it.

    Blocks and claims are that subgraph's own members. A sibling subgraph on
    the same frozen view is not part of this hit. ``score_breakdown`` uses the
    same BM25 keys as causal-memory card ranking.
    """

    view_revision_id: str
    session_id: str
    subgraph: RetrievalSubgraph
    blocks: tuple[InfluenceBlock, ...]
    claims: tuple[InfluenceClaim, ...]
    forks: tuple[DiagnosticForkHit, ...]
    score_breakdown: dict[str, Any]

    def __post_init__(self) -> None:
        breakdown = _json_copy(
            _require_mapping(self.score_breakdown, "score_breakdown"), "score_breakdown"
        )
        if not isinstance(breakdown, dict):
            raise ValueError("score_breakdown must be a dict")
        object.__setattr__(self, "score_breakdown", breakdown)
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "claims", tuple(self.claims))
        object.__setattr__(self, "forks", tuple(self.forks))


class RetrievalSubgraphAdapter:
    """Historical ``post_run_index`` retrieval subgraphs.

    ``name`` is ``retrieval_subgraph``. Candidates are filtered with the
    injected ``session_acl`` callback before they enter the BM25 corpus. A
    session with no labels is invisible. The querying revision's session is
    held to that rule and raises when it has no labels or those labels are
    not visible to the principal; the call does not return an empty hit list.
    The session id is not used as a label. Fork visibility uses the revision
    ``workspace_id``, the same label the fork/branch adapter uses. This
    adapter does not write forks and is not called by
    ``HybridRetrievalFacade.retrieve``.
    """

    name = "retrieval_subgraph"

    def __init__(
        self,
        views: ViewSupersedeIndex,
        forks: ForkBranchAdapter,
        *,
        session_acl: SessionAcl | None = None,
        session_dirs: Mapping[str, str | Path] | None = None,
    ) -> None:
        if not isinstance(views, ViewSupersedeIndex):
            raise ValueError("views must be a ViewSupersedeIndex")
        if not isinstance(forks, ForkBranchAdapter):
            raise ValueError("forks must be a ForkBranchAdapter")
        self._views = views
        self._forks = forks
        self._session_acl = session_acl
        dirs: dict[str, Path] = {}
        if session_dirs is not None:
            if isinstance(session_dirs, (str, bytes)) or not isinstance(session_dirs, Mapping):
                raise ValueError("session_dirs must be a mapping of session_id to path")
            for session_id, path in session_dirs.items():
                dirs[_require_str(session_id, "session_dirs")] = Path(path)
        if forks._session_id and forks._session_dir is not None:
            dirs.setdefault(forks._session_id, Path(forks._session_dir))
        self._session_dirs = dirs

    def retrieve_historical_subgraphs(
        self,
        view_revision_id: str,
        subgraph_id: str,
        principal_labels: Sequence[str],
    ) -> tuple[DiagnosticSubgraphHit, ...]:
        """Rank other post-run subgraphs against one failure subgraph.

        The subgraph must exist on ``view_revision_id`` and ``is_failure``.
        Otherwise this raises ``ValueError``. The querying revision's session
        must be visible to ``principal_labels``. No labels, or labels that do
        not meet the principal, raise ``ValueError`` and do not return an
        empty hit list. At most ``DIAGNOSTIC_SUBGRAPH_HIT_CAP`` hits are
        returned. Fewer matches are not backfilled. The querying revision is
        not a hit.
        """
        principal = _principal_set(principal_labels)
        current = self._revision(view_revision_id)
        self._require_query_session_visible(current.session_id, principal)
        query_subgraph = self._require_failure_subgraph(current, subgraph_id)
        query_text = _subgraph_retrieval_text(current, query_subgraph)
        candidates: list[tuple[InfluenceViewRevision, RetrievalSubgraph]] = []
        documents: list[str] = []
        for revision in self._views.revisions_for_purpose(PURPOSE_POST_RUN_INDEX):
            if revision.revision_id == current.revision_id:
                continue
            labels = self._session_labels(revision.session_id)
            if not labels or not _labels_visible(labels, principal):
                continue
            for subgraph in revision.retrieval_subgraphs:
                candidates.append((revision, subgraph))
                documents.append(_subgraph_retrieval_text(revision, subgraph))
        ranked = _rank_document_indexes(query_text, documents)[:DIAGNOSTIC_SUBGRAPH_HIT_CAP]
        synopses = self._synopses()
        hits: list[DiagnosticSubgraphHit] = []
        for index, breakdown in ranked:
            revision, subgraph = candidates[index]
            blocks, claims = _subgraph_members(revision, subgraph)
            hits.append(
                DiagnosticSubgraphHit(
                    view_revision_id=revision.revision_id,
                    session_id=revision.session_id,
                    subgraph=subgraph,
                    blocks=blocks,
                    claims=claims,
                    forks=self._forks_inside(revision, subgraph, principal, synopses),
                    score_breakdown=breakdown,
                )
            )
        return tuple(hits)

    def _revision(self, view_revision_id: str) -> InfluenceViewRevision:
        identity = _require_str(view_revision_id, "view_revision_id")
        matches: list[InfluenceViewRevision] = []
        for purpose in sorted(PURPOSES):
            for revision in self._views.revisions_for_purpose(purpose):
                if revision.revision_id == identity:
                    matches.append(revision)
        if not matches:
            raise ValueError(f"view revision {identity!r} is not in the supersede index")
        if len(matches) != 1:
            raise ValueError(
                f"view revision {identity!r} is not unique in the supersede index"
            )
        return matches[0]

    def _require_query_session_visible(self, session_id: str, principal: set[str]) -> None:
        """Raise unless this session's labels are visible to ``principal``.

        Historical candidates already drop a session with no labels. The
        querying session used to skip that check. Missing labels and a
        principal mismatch both raise. Neither case returns an empty hit list.
        """
        labels = self._session_labels(session_id)
        if not labels:
            raise ValueError(f"querying session {session_id!r} has no ACL labels")
        if not _labels_visible(labels, principal):
            raise ValueError(
                f"querying session {session_id!r} is not visible to the principal"
            )

    def _require_failure_subgraph(
        self, revision: InfluenceViewRevision, subgraph_id: str
    ) -> RetrievalSubgraph:
        identity = _require_str(subgraph_id, "subgraph_id")
        for subgraph in revision.retrieval_subgraphs:
            if subgraph.subgraph_id != identity:
                continue
            if not subgraph.is_failure:
                raise ValueError(f"subgraph {identity!r} is not failure-marked")
            return subgraph
        raise ValueError(
            f"subgraph {identity!r} is not on view revision {revision.revision_id!r}"
        )

    def _session_labels(self, session_id: str) -> tuple[str, ...]:
        if self._session_acl is None:
            return ()
        raw = self._session_acl(session_id)
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise ValueError("session_acl must return a sequence of str")
        return _normalize_labels(raw)

    def _synopses(self) -> dict[str, str]:
        found: dict[str, str] = {}
        for session_id in sorted(self._session_dirs):
            directory = self._session_dirs[session_id]
            if not (directory / LEDGER_FILENAME).is_file():
                continue
            for record in AOLedger(directory, session_id).records():
                synopsis = _synopsis_from_observation(record.observation)
                if synopsis:
                    found[record.ao_id] = synopsis
        return found

    def _forks_inside(
        self,
        revision: InfluenceViewRevision,
        subgraph: RetrievalSubgraph,
        principal: set[str],
        synopses: Mapping[str, str],
    ) -> tuple[DiagnosticForkHit, ...]:
        hits: list[DiagnosticForkHit] = []
        for fork_node_id in self._forks._fork_ids():
            loaded = self._forks._store.get_fork_node(fork_node_id)
            if loaded is None:
                continue
            revisions = loaded.get("revisions")
            if not isinstance(revisions, list):
                continue
            for item in revisions:
                if not isinstance(item, dict):
                    continue
                parsed = _fork_grounded_in(
                    item, revision.revision_id, subgraph, principal, synopses
                )
                if parsed is not None:
                    hits.append(parsed)
        return tuple(hits)


def _fork_grounded_in(
    revision: Mapping[str, Any],
    view_revision_id: str,
    subgraph: RetrievalSubgraph,
    principal: set[str],
    synopses: Mapping[str, str],
) -> DiagnosticForkHit | None:
    raw_workspace = revision.get("workspace_id")
    if not isinstance(raw_workspace, str) or not raw_workspace:
        return None
    labels = _normalize_labels((raw_workspace,))
    if not _labels_visible(labels, principal):
        return None
    raw_grounding = revision.get("influence_grounding")
    if not isinstance(raw_grounding, Mapping):
        return None
    try:
        grounding = InfluenceGrounding.from_dict(dict(raw_grounding))
    except ValueError:
        return None
    if grounding.view_revision_id != view_revision_id:
        return None
    if not _position_inside_subgraph(grounding, subgraph):
        return None
    fork_node_id = revision.get("fork_node_id")
    revision_id = revision.get("revision_id")
    if not isinstance(fork_node_id, str) or not fork_node_id:
        return None
    if not isinstance(revision_id, str) or not revision_id:
        return None
    return DiagnosticForkHit(
        fork_node_id=fork_node_id,
        revision_id=revision_id,
        workspace_id=raw_workspace,
        text=_diagnostic_fork_text(revision, synopses),
        influence_grounding=grounding,
    )


def _assign_ranks(cards: Sequence[RetrievalCard], channel: str) -> list[RetrievalCard]:
    group = [card for card in cards if card.channel == channel]
    return [replace(card, in_channel_rank=index) for index, card in enumerate(group, start=1)]


def _candidate_id(card: RetrievalCard) -> str:
    return card.payload_ref


def _derive_query_seed(query: RetrievalQuery) -> str:
    """Seed from the query envelope: sha256 digest, first 8 bytes, hex."""
    payload = json.dumps(
        query.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).digest()[:8].hex()


def _normalize_seed(seed: str | int | None) -> str | None:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, (str, int)):
        raise ValueError("seed must be a str, int, or None")
    text = str(seed)
    if not text:
        raise ValueError("seed must be non-empty when provided")
    return text


def _exploration_index(seed: str, eligible_count: int) -> int:
    """Uniform index in ``[0, eligible_count)`` from a versioned seed digest.

    Replay: ``eligible_set[_exploration_index(seed, len(eligible_set))]``.
    """
    material = f"{SELECTION_POLICY_VERSION}\n{seed}\n{eligible_count}".encode("utf-8")
    draw = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return draw % eligible_count


def _select_provisional(
    ranked: Sequence[RetrievalCard],
    slots: int,
    seed: str,
) -> tuple[list[RetrievalCard], list[RetrievalCard], SelectionAudit]:
    """Fill provisional seats: ``n - 1`` exploitation plus one seeded exploration.

    ``ranked`` is in-channel rank order. Exploitation keeps that prefix.
    Exploration is drawn uniformly from provisional cards exploitation did
    not take (already ACL- and status-gated by the adapters). When that
    eligible set is empty, the exploration seat stays open for the existing
    exploitation backfill and the audit records ``backfill``. Unchosen cards
    stay in rank order so a verified-slot shortage can still absorb them.
    """
    if slots < 2:
        primary = [replace(card, selection_mode="exploitation") for card in ranked[:slots]]
        overflow = [replace(card, selection_mode="exploitation") for card in ranked[slots:]]
        audit = SelectionAudit(
            eligible_set=(),
            selection_propensity=None,
            seed=seed,
            selection_policy_version=SELECTION_POLICY_VERSION,
            backfill=False,
        )
        return primary, overflow, audit

    exploit_count = slots - 1
    exploitation = list(ranked[:exploit_count])
    eligible = list(ranked[exploit_count:])
    if not eligible:
        primary = [replace(card, selection_mode="exploitation") for card in exploitation]
        audit = SelectionAudit(
            eligible_set=(),
            selection_propensity=None,
            seed=seed,
            selection_policy_version=SELECTION_POLICY_VERSION,
            backfill=True,
            backfill_reason=NO_ELIGIBLE_EXPLORATION_REASON,
        )
        return primary, [], audit

    chosen = eligible[_exploration_index(seed, len(eligible))]
    overflow = [
        replace(card, selection_mode="exploitation") for card in eligible if card is not chosen
    ]
    primary = [
        *[replace(card, selection_mode="exploitation") for card in exploitation],
        replace(chosen, selection_mode="exploration"),
    ]
    audit = SelectionAudit(
        eligible_set=tuple(_candidate_id(card) for card in eligible),
        selection_propensity=1.0 / len(eligible),
        seed=seed,
        selection_policy_version=SELECTION_POLICY_VERSION,
        backfill=False,
    )
    return primary, overflow, audit


def _allocate_stateful(
    cards: Sequence[RetrievalCard],
    *,
    provisional_slots: int,
    seed: str,
) -> tuple[list[RetrievalCard], SelectionAudit]:
    """Fill 8 verified slots and ``provisional_slots`` provisional slots.

    A short channel is backfilled from the other channel's overflow.
    Backfilled cards keep the channel and in-channel rank they had before
    slotting. Fact cards never enter this function. Verified cards stay
    ``selection_mode="exploitation"``.
    """
    verified = [
        replace(card, selection_mode="exploitation") for card in _assign_ranks(cards, "verified")
    ]
    provisional = _assign_ranks(cards, "provisional")
    primary_verified = verified[:VERIFIED_SLOTS]
    primary_provisional, provisional_overflow, audit = _select_provisional(
        provisional, provisional_slots, seed
    )
    verified_deficit = VERIFIED_SLOTS - len(primary_verified)
    provisional_deficit = provisional_slots - len(primary_provisional)
    backfill_verified_slots = provisional_overflow[:verified_deficit]
    backfill_provisional_slots = verified[VERIFIED_SLOTS : VERIFIED_SLOTS + provisional_deficit]
    return [
        *primary_verified,
        *backfill_verified_slots,
        *primary_provisional,
        *backfill_provisional_slots,
    ], audit


class HybridRetrievalFacade:
    """Active retrieve reads native edges, overlays, patterns, and guidance.

    ``audit_legacy`` is the explicit Claim/Cut/legacy-fork audit path.
    ``retrieve`` does not call it. Online code must not call
    ``materialize_strength(None)``, ``events()``, ``list_bridges``, or
    ``ProjectionRegistry.to_dict``. Those are offline management paths and
    do not apply principal ACL. Diagnostic subgraph retrieval is
    ``retrieve_historical_subgraphs`` and is not part of ``retrieve``.
    """

    def __init__(
        self,
        fork_branch: ForkBranchAdapter,
        projection: InfluenceProjectionAdapter,
        segment_atom: SegmentAtomAdapter,
        *,
        rerank: RerankHook | RerankFn | None = None,
        acl_filter: AclFilter | None = None,
        feature_version: str = FEATURE_SPEC_VERSION,
        seed: str | int | None = None,
        fact_lane_paused: bool | None = None,
        embedder: EmbedFn | None = None,
        retrieval_subgraph: RetrievalSubgraphAdapter | None = None,
        active_corpus: Any = None,
    ) -> None:
        self.fork_branch = fork_branch
        self.projection = projection
        self.segment_atom = segment_atom
        self.active_corpus = active_corpus
        if retrieval_subgraph is not None and not isinstance(
            retrieval_subgraph, RetrievalSubgraphAdapter
        ):
            raise ValueError("retrieval_subgraph must be a RetrievalSubgraphAdapter or None")
        self.retrieval_subgraph = retrieval_subgraph
        self.rerank = rerank if rerank is not None else IdentityRerank()
        self.acl_filter = acl_filter if acl_filter is not None else default_acl_filter
        self.feature_version = _require_str(feature_version, "feature_version")
        self.seed = _normalize_seed(seed)
        if fact_lane_paused is None:
            self.fact_lane_paused = FACT_LANE_PAUSED
        elif isinstance(fact_lane_paused, bool):
            self.fact_lane_paused = fact_lane_paused
        else:
            raise ValueError("fact_lane_paused must be a bool or None")
        self.embedder = embedder

    def retrieve(
        self,
        query: RetrievalQuery | Mapping[str, Any] | str | None,
        principal_labels: Sequence[str],
        profile: RetrievalProfile | Mapping[str, Any] | None = None,
    ) -> RetrievalResult:
        """Active retrieval. Claim, cut, and legacy fork cards are not returned.

        The fact lane and the claim projection adapter are not consulted.
        ``audit_legacy`` keeps that older path for explicit audit reads.
        """
        from openviking.session.active_retrieval import retrieve_active

        return retrieve_active(
            self.active_corpus,
            query,
            principal_labels,
            profile,
            rerank=self.rerank,
            acl_filter=self.acl_filter,
            feature_version=self.feature_version,
            seed=self.seed,
            embedder=self.embedder,
        )

    def audit_legacy(
        self,
        query: RetrievalQuery | Mapping[str, Any] | str | None,
        principal_labels: Sequence[str],
        profile: RetrievalProfile | Mapping[str, Any] | None = None,
    ) -> RetrievalResult:
        """Legacy-audit retrieval over claim projections and legacy forks.

        This is not the active path. ``retrieve`` does not call it.
        While the fact lane is paused this does not call ``segment_atom.search``
        and does not backfill those seats. ``materialize_strength(None)``,
        ``events()``, ``list_bridges``, and ``ProjectionRegistry.to_dict`` are
        offline management paths. This method does not call them, and it does
        not call ``retrieval_subgraph``.
        """
        parsed = coerce_query(query)
        budgets = coerce_profile(profile)
        _principal_set(principal_labels)
        seed = self.seed if self.seed is not None else _derive_query_seed(parsed)
        plan = _RankPlan(
            budgets.rank_mode,
            self.embedder if budgets.rank_mode == RANK_MODE_HYBRID else None,
        )
        plan.prepare(parsed.text)
        fork_cards, projection_cards, fact_cards = self._lane_cards(
            parsed, budgets, principal_labels, plan
        )
        if plan.downgraded:
            plan.actual = RANK_MODE_BM25
            plan.query_vector = None
            plan.embedder = None
            plan.downgraded = False
            fork_cards, projection_cards, fact_cards = self._lane_cards(
                parsed, budgets, principal_labels, plan
            )
        stateful, audit = _allocate_stateful(
            [*fork_cards, *projection_cards],
            provisional_slots=budgets.provisional_slots,
            seed=seed,
        )
        facts = [
            replace(card, selection_mode="exploitation")
            for card in _assign_ranks(fact_cards, "fact")
        ]
        selected = [*stateful, *facts]
        context: dict[str, Any] = {
            "feature_version": self.feature_version,
            "query": parsed.to_dict(),
            "profile": budgets.to_dict(),
            "rank_tier": plan.actual,
            "requested_rank_tier": plan.requested,
        }
        reranked = self._call_rerank(selected, context)
        truncated = len(reranked) > CAUSAL_MEMORY_CAP
        clamped = reranked[:CAUSAL_MEMORY_CAP]
        return RetrievalResult(
            cards=tuple(clamped),
            feature_version=self.feature_version,
            truncated_by_rerank_clamp=truncated,
            selection_audit=audit,
            rank_tier=plan.actual,
        )

    def _lane_cards(
        self,
        query: RetrievalQuery,
        budgets: RetrievalProfile,
        principal_labels: Sequence[str],
        plan: _RankPlan,
    ) -> tuple[list[RetrievalCard], list[RetrievalCard], list[RetrievalCard]]:
        fork_cards = self._visible(
            self.fork_branch.search(
                query, budgets.fork_branch, principal_labels, rank_plan=plan
            ),
            principal_labels,
        )
        projection_cards = self._visible(
            self.projection.search(
                query, budgets.projection, principal_labels, rank_plan=plan
            ),
            principal_labels,
        )
        if self.fact_lane_paused:
            fact_cards: list[RetrievalCard] = []
        else:
            fact_cards = self._visible(
                self.segment_atom.search(
                    query, budgets.segment_atom, principal_labels, rank_plan=plan
                ),
                principal_labels,
            )
        return fork_cards, projection_cards, fact_cards

    def audit_expand_claim_neighborhood(
        self,
        view_revision_id: str | None = None,
        claim_id: str | None = None,
        max_entities: int = PATH_MAX_ENTITIES,
        *,
        card_id: str | None = None,
        principal_labels: Sequence[str] = (),
    ) -> AssembledPath:
        """Legacy-audit claim neighborhood. ``retrieve`` does not call this."""
        return self.expand_claim_neighborhood(
            view_revision_id,
            claim_id,
            max_entities=max_entities,
            card_id=card_id,
            principal_labels=principal_labels,
        )

    def expand_claim_neighborhood(
        self,
        view_revision_id: str | None = None,
        claim_id: str | None = None,
        max_entities: int = PATH_MAX_ENTITIES,
        *,
        card_id: str | None = None,
        principal_labels: Sequence[str] = (),
    ) -> AssembledPath:
        """Legacy-audit claim neighborhood. Not an active retrieve object.

        OpenViking only provides this entry. Whether a diagnostic agent may
        call it is an agent-side fence and is not checked here.

        ``card_id`` is the public argument: the server resolves that claim
        projection's frozen view revision and claim id, then walks influence
        edges only. Resolution fails closed on ``principal_labels``; the
        default ``()`` sees no card, and an invisible id raises the same
        ``unknown card id`` error as a missing id. ``(view_revision_id,
        claim_id)`` remains for internal and compatibility callers and does
        not apply this principal. A visible card that is not a claim
        projection raises ``ValueError``. Nothing is persisted.
        """
        return self.projection.expand_claim_neighborhood(
            view_revision_id,
            claim_id,
            max_entities=max_entities,
            card_id=card_id,
            principal_labels=principal_labels,
        )

    def retrieve_historical_subgraphs(
        self,
        view_revision_id: str,
        subgraph_id: str,
        principal_labels: Sequence[str],
    ) -> tuple[DiagnosticSubgraphHit, ...]:
        """Read historical subgraphs for one failure subgraph on this revision.

        The caller names the current view revision and the subgraph id on it.
        The server loads that subgraph. A missing id, an id from another
        revision, or a subgraph that is not failure-marked raises
        ``ValueError``. This does not write forks and does not change markers.
        """
        if self.retrieval_subgraph is None:
            raise ValueError("retrieval subgraph adapter is not configured")
        return self.retrieval_subgraph.retrieve_historical_subgraphs(
            view_revision_id, subgraph_id, principal_labels
        )

    def _visible(
        self, cards: Sequence[RetrievalCard], principal_labels: Sequence[str]
    ) -> list[RetrievalCard]:
        return [card for card in cards if self.acl_filter(card, principal_labels)]

    def _call_rerank(
        self, cards: Sequence[RetrievalCard], context: Mapping[str, Any]
    ) -> list[RetrievalCard]:
        hook = self.rerank
        if isinstance(hook, IdentityRerank) or hasattr(hook, "rerank"):
            reranked = hook.rerank(cards, context)  # type: ignore[union-attr]
        else:
            reranked = hook(cards, context)  # type: ignore[operator]
        return [card if isinstance(card, RetrievalCard) else RetrievalCard.from_dict(card) for card in reranked]
