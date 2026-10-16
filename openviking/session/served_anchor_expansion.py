# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Served-anchor expansion (Q155-B, Q160-B).

Initial discovery stays a frozen E0. A later hop names one served card and
one public relation. The derived envelope is allocated on its own. A failed
hop does not change the parent envelope and publishes no partial hits.
Episode and Insight cannot be anchors or targets.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openviking.session.memory_discovery_allocation import allocate_discovery

EXPANSION_POLICY_VERSION = "memory-expansion-placement/Q155-B"
CHAIN_TTL_SECONDS = 10 * 60
DEFAULT_MAX_STEPS = 4
HARD_MAX_STEPS = 8
UNIQUE_SERVED_CAP = 30
DEFAULT_OPERATION_LIMIT = 12
PUBLIC_RELATIONS = frozenset({"similar", "skills"})
RETIRED_RELATIONS = frozenset({"episodeInsight", "related", "collaborators", "interactions"})
RETIRED_KINDS = frozenset({"episode", "insight"})
SKILL_EDGE_ORDER = ("skillGrounding", "skillLineage")
FORBIDDEN_AUTHORITY = frozenset(
    {
        "principal",
        "deadline",
        "budget",
        "snapshot",
        "snapshotRef",
        "scopeRef",
        "budgetLeaseRef",
        "workspace",
        "acl",
        "intentText",
        "query",
        "exact",
    }
)


class ExpansionError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ServedCard:
    id: str
    kind: str
    corpus: str
    use_class: str
    public_rank: int


@dataclass(frozen=True)
class FrozenEnvelope:
    envelope_ref: str
    operation_id: str
    plan_hash: str
    final_order_hash: str
    completed: bool
    memory_candidates: tuple[ServedCard, ...]
    analogical_priors: tuple[ServedCard, ...]
    duplicate: bool = False

    def served(self) -> tuple[ServedCard, ...]:
        return self.memory_candidates + self.analogical_priors


NeighborSource = Callable[[str, str, Mapping[str, str]], Sequence[Mapping[str, Any]]]
StatusSource = Callable[[str], str]
Clock = Callable[[], float]


def freeze_direct(
    *,
    operation_id: str,
    plan_hash: str,
    final_order_hash: str,
    memory_candidates: Sequence[Mapping[str, Any]],
    analogical_priors: Sequence[Mapping[str, Any]] = (),
) -> FrozenEnvelope:
    """Freeze a completed direct-discovery envelope. This does not expand."""
    return FrozenEnvelope(
        envelope_ref=f"env-{_digest(operation_id, plan_hash)[:16]}",
        operation_id=operation_id,
        plan_hash=plan_hash,
        final_order_hash=final_order_hash,
        completed=True,
        memory_candidates=tuple(_served(card) for card in memory_candidates),
        analogical_priors=tuple(_served(card) for card in analogical_priors),
    )


class ExpansionChain:
    """One task chain. Close only records a checkpoint."""

    def __init__(
        self,
        parent: FrozenEnvelope,
        *,
        opened_at: float,
        max_steps: int = DEFAULT_MAX_STEPS,
        clock: Clock | None = None,
    ) -> None:
        if type(max_steps) is not int or not 1 <= max_steps <= HARD_MAX_STEPS:
            raise ExpansionError("invalid_chain_steps")
        if not parent.completed:
            raise ExpansionError("parent_not_completed")
        self.ref = f"chain-{_digest(parent.envelope_ref, str(opened_at))[:16]}"
        self.opened_at = opened_at
        self.max_steps = max_steps
        self.clock = clock or (lambda: opened_at)
        self.steps = 0
        self.envelopes: dict[str, FrozenEnvelope] = {parent.envelope_ref: parent}
        self._order = [parent.envelope_ref]
        self._operations: dict[str, tuple[str, FrozenEnvelope]] = {}

    @property
    def parent(self) -> FrozenEnvelope:
        return self.envelopes[self._order[0]]

    def unique_served(self) -> frozenset[str]:
        return frozenset(card.id for ref in self._order for card in self.envelopes[ref].served())

    def expand(
        self,
        request: Mapping[str, Any],
        neighbors: NeighborSource,
        *,
        status_of: StatusSource | None = None,
        fact_paused: bool = False,
    ) -> FrozenEnvelope:
        parent = self._prepare(request)
        operation_id = _required_text(request, "operationId")
        relation = _required_text(request, "relation")
        anchor_id = _required_text(request, "anchorCardRef")
        limit = _limit(request.get("requestedLimit", DEFAULT_OPERATION_LIMIT))
        if relation in RETIRED_RELATIONS:
            raise ExpansionError("retired_from_memory_discovery")
        if relation not in PUBLIC_RELATIONS:
            raise ExpansionError("invalid_relation")
        anchor = _find_served(parent, anchor_id)
        if anchor is None:
            raise ExpansionError("anchor_not_served")
        if anchor.kind in RETIRED_KINDS:
            raise ExpansionError("retired_from_memory_discovery")
        if (status_of or _visible)(anchor.id) != "visible":
            raise ExpansionError("anchor_unavailable")
        request_hash = _request_hash(parent, anchor_id, relation, limit)
        previous = self._operations.get(operation_id)
        if previous is not None:
            seen_hash, envelope = previous
            if seen_hash != request_hash:
                raise ExpansionError("plan_drift")
            return FrozenEnvelope(
                envelope_ref=envelope.envelope_ref,
                operation_id=envelope.operation_id,
                plan_hash=envelope.plan_hash,
                final_order_hash=envelope.final_order_hash,
                completed=envelope.completed,
                memory_candidates=envelope.memory_candidates,
                analogical_priors=envelope.analogical_priors,
                duplicate=True,
            )
        if self.steps >= self.max_steps:
            raise ExpansionError("chain_step_cap")
        if len(self.unique_served()) >= UNIQUE_SERVED_CAP:
            raise ExpansionError("chain_served_cap")
        child = _child_binding(parent, anchor_id, relation)
        plan_hash = _plan_hash(parent, anchor_id, relation, limit, child["snapshot_ref"])
        self.steps += 1
        try:
            native = _collect(neighbors, anchor_id, relation, child, limit)
        except ExpansionError:
            raise
        except Exception as exc:
            raise ExpansionError("expansion_terminated") from exc
        if len(self.unique_served().union(card["id"] for card in native)) > UNIQUE_SERVED_CAP:
            raise ExpansionError("chain_served_cap")
        allocated = allocate_discovery(native, plan_hash=plan_hash, fact_paused=fact_paused)
        envelope = FrozenEnvelope(
            envelope_ref=f"env-{plan_hash[:16]}",
            operation_id=operation_id,
            plan_hash=plan_hash,
            final_order_hash=allocated.final_order_hash,
            completed=True,
            memory_candidates=tuple(_from_ordered(card) for card in allocated.memory_candidates),
            analogical_priors=tuple(_from_ordered(card) for card in allocated.analogical_priors),
        )
        self.envelopes[envelope.envelope_ref] = envelope
        self._order.append(envelope.envelope_ref)
        self._operations[operation_id] = (request_hash, envelope)
        return envelope

    def checkpoint(self) -> dict[str, Any]:
        return {
            "chain_ref": self.ref,
            "served_refs": sorted(self.unique_served()),
            "hint_authority": False,
        }

    def _prepare(self, request: Mapping[str, Any]) -> FrozenEnvelope:
        if not isinstance(request, Mapping):
            raise ExpansionError("invalid_request")
        if FORBIDDEN_AUTHORITY.intersection(request):
            raise ExpansionError("agent_supplied_authority")
        if self.clock() - self.opened_at > CHAIN_TTL_SECONDS:
            raise ExpansionError("chain_expired")
        if request.get("expansionChainRef") != self.ref:
            raise ExpansionError("unknown_chain")
        parent_ref = request.get("parentEnvelopeRef")
        parent = self.envelopes.get(parent_ref) if isinstance(parent_ref, str) else None
        if parent is None or not parent.completed:
            raise ExpansionError("parent_not_completed")
        return parent


def reject_redirect() -> None:
    """Redirect is not an expansion command. A new query needs a new retrieve."""
    raise ExpansionError("redirect_removed")


def _collect(
    neighbors: NeighborSource,
    anchor_id: str,
    relation: str,
    child: Mapping[str, str],
    limit: int,
) -> list[dict[str, Any]]:
    edges = SKILL_EDGE_ORDER if relation == "skills" else (relation,)
    ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for ordinal, edge in enumerate(edges):
        rows = neighbors(anchor_id, edge, child)
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise ExpansionError("expansion_terminated")
        for row in rows:
            card = _native(row, edge, ordinal)
            if card is None:
                continue
            ranked.append((_order_key(card), card))
    ranked.sort(key=lambda item: item[0])
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _key, card in ranked:
        if card["canonical_identity"] in seen:
            continue
        seen.add(card["canonical_identity"])
        chosen.append(card)
        if len(chosen) == limit:
            break
    for index, card in enumerate(chosen, start=1):
        card["corpus_rank"] = index
    return chosen


def _native(row: Mapping[str, Any], edge: str, ordinal: int) -> dict[str, Any] | None:
    if not isinstance(row, Mapping):
        raise ExpansionError("expansion_terminated")
    kind = row.get("kind")
    if kind in RETIRED_KINDS:
        return None
    card_id = row.get("id")
    corpus = row.get("corpus")
    use_class = row.get("use_class")
    if not isinstance(card_id, str) or not isinstance(corpus, str) or not isinstance(kind, str):
        raise ExpansionError("expansion_terminated")
    if use_class not in ("memory_candidate", "analogical_prior"):
        raise ExpansionError("expansion_terminated")
    identity = row.get("canonical_identity", card_id)
    score = row.get("score")
    return {
        "id": card_id,
        "kind": kind,
        "corpus": corpus,
        "use_class": use_class,
        "canonical_identity": identity if isinstance(identity, str) and identity else card_id,
        "corpus_rank": 1,
        "edge": edge,
        "edge_ordinal": ordinal,
        "score": score if isinstance(score, (int, float)) and type(score) is not bool else None,
    }


def _order_key(card: Mapping[str, Any]) -> tuple[Any, ...]:
    score = card["score"]
    score_key = -score if isinstance(score, (int, float)) else 0
    return (
        card["edge_ordinal"],
        score_key,
        card["canonical_identity"],
    )


def _served(card: Mapping[str, Any]) -> ServedCard:
    return ServedCard(
        id=str(card["id"]),
        kind=str(card["kind"]),
        corpus=str(card["corpus"]),
        use_class=str(card["use_class"]),
        public_rank=int(card["public_rank"]),
    )


def _from_ordered(card: Any) -> ServedCard:
    return ServedCard(
        id=card.id,
        kind=card.kind,
        corpus=card.corpus,
        use_class=card.use_class,
        public_rank=card.public_rank,
    )


def _find_served(parent: FrozenEnvelope, anchor_id: str) -> ServedCard | None:
    for card in parent.served():
        if card.id == anchor_id:
            return card
    return None


def _visible(card_id: str) -> str:
    del card_id
    return "visible"


def _limit(raw: Any) -> int:
    if type(raw) is not int or not 1 <= raw <= DEFAULT_OPERATION_LIMIT:
        raise ExpansionError("invalid_expansion_limit")
    return raw


def _required_text(request: Mapping[str, Any], key: str) -> str:
    value = request.get(key)
    if not isinstance(value, str) or not value:
        raise ExpansionError("invalid_request")
    return value


def _child_binding(parent: FrozenEnvelope, anchor_id: str, relation: str) -> dict[str, str]:
    token = _digest(parent.final_order_hash, anchor_id, relation, uuid.uuid4().hex)
    return {
        "snapshot_ref": f"child-snap-{token[:16]}",
        "budget_lease_ref": f"child-budget-{token[:16]}",
        "deadline_ref": f"child-deadline-{token[:16]}",
        "cancel_ref": f"child-cancel-{token[:16]}",
        "parent_envelope_ref": parent.envelope_ref,
        "anchor": anchor_id,
    }


def _request_hash(parent: FrozenEnvelope, anchor_id: str, relation: str, limit: int) -> str:
    payload = {
        "policy": EXPANSION_POLICY_VERSION,
        "parentFinalOrderHash": parent.final_order_hash,
        "anchor": anchor_id,
        "relation": relation,
        "requestedLimit": limit,
    }
    return _digest(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _plan_hash(
    parent: FrozenEnvelope,
    anchor_id: str,
    relation: str,
    limit: int,
    snapshot_ref: str,
) -> str:
    payload = {
        "policy": EXPANSION_POLICY_VERSION,
        "parentFinalOrderHash": parent.final_order_hash,
        "anchor": anchor_id,
        "relation": relation,
        "requestedLimit": limit,
        "childSnapshotRef": snapshot_ref,
    }
    return _digest(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _digest(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
