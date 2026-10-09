# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Served-only Hint and Offer grounding (Q146-B, Q155-B, Q160-B).

A Hint or Offer may cite only memory candidates that a completed discovery
lane actually served. Priors, current evidence, terminated discovery, and
retired Episode or Insight cannot start a new delivery. Completeness is
part of the dedup identity, so a later complete Hint is not swallowed by
an earlier partial one. Offer does not count as skill utility.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

SCHEMA_VERSION = "memory-hint-retrieval-provenance.v1"
DISCOVERY = "memory_discovery"
EXACT = "current_log_exact"
RETIRED_KINDS = frozenset({"episode", "insight"})

StatusOf = Callable[[str], str]


class HintError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ServedDeliveryLedger:
    """Append-only deliveries. A refresh does not rewrite an older provenance."""

    def __init__(self) -> None:
        self.deliveries: list[dict[str, Any]] = []
        self._by_key: dict[str, dict[str, Any]] = {}

    def publish(
        self,
        envelope: Mapping[str, Any],
        source_ids: Sequence[str],
        *,
        kind: str,
        target: str,
        lineage: str,
        status_of: StatusOf | None = None,
    ) -> dict[str, Any]:
        if kind not in {"hint", "offer"}:
            raise HintError("invalid_delivery")
        if "retrievalCompleteness" in envelope:
            raise HintError("completeness_not_self_reported")
        provenance = build_provenance(envelope, source_ids, status_of or _visible)
        dedup = _dedup_key(target, lineage, provenance)
        if provenance["retrievalCompleteness"] == "partial":
            complete_key = _dedup_key(
                target,
                lineage,
                {**provenance, "retrievalCompleteness": "complete", "incompleteLanes": []},
            )
            if complete_key in self._by_key and dedup not in self._by_key:
                raise HintError("complete_hint_stands")
        existing = self._by_key.get(dedup)
        if existing is not None:
            return {
                "delivery": existing,
                "duplicate": True,
                "refreshed": True,
                "utility_counted": False,
            }
        delivery = {
            "id": f"{kind}-{_digest(dedup)[:16]}",
            "kind": kind,
            "provenance": provenance,
            "utility_counted": False,
        }
        self.deliveries.append(delivery)
        self._by_key[dedup] = delivery
        return {"delivery": delivery, "duplicate": False, "refreshed": False, "utility_counted": False}


def build_provenance(
    envelope: Mapping[str, Any],
    source_ids: Sequence[str],
    status_of: StatusOf,
) -> dict[str, Any]:
    lanes = envelope.get("lanes")
    if not isinstance(lanes, Mapping) or DISCOVERY not in lanes:
        raise HintError("discovery_not_completed")
    discovery = lanes[DISCOVERY]
    if not isinstance(discovery, Mapping) or discovery.get("executionState") != "completed":
        raise HintError("discovery_not_completed")
    if discovery.get("reasonCode") == "cancelled":
        raise HintError("retrieval_cancelled")
    derived = _completeness(lanes)
    if derived == "unavailable":
        raise HintError("retrieval_unavailable")
    candidates = _cards(envelope.get("memory_candidates"))
    priors = _cards(envelope.get("analogical_priors"))
    served = {card["id"]: card for card in candidates if card["use_class"] == "memory_candidate"}
    prior_ids = {card["id"] for card in priors}
    if isinstance(source_ids, (str, bytes)) or not isinstance(source_ids, Sequence) or not source_ids:
        raise HintError("empty_sources")
    chosen: list[str] = []
    for source_id in source_ids:
        if not isinstance(source_id, str) or source_id in prior_ids or source_id not in served:
            raise HintError("source_not_served")
        card = served[source_id]
        if card["kind"] in RETIRED_KINDS:
            raise HintError("retired_from_memory_discovery")
        if status_of(source_id) != "visible":
            raise HintError("source_not_currently_visible")
        if source_id not in chosen:
            chosen.append(source_id)
    if not chosen:
        raise HintError("empty_sources")
    incomplete = sorted(
        name
        for name, audit in lanes.items()
        if isinstance(audit, Mapping) and audit.get("executionState") == "terminated"
    )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "retrievalEnvelopeRef": envelope["envelope_ref"],
        "operationId": envelope["operation_id"],
        "planHash": envelope["plan_hash"],
        "retrievalCompleteness": derived,
        "contributingLanes": [DISCOVERY],
        "incompleteLanes": incomplete,
        "laneAuditHash": _digest(_canonical(dict(lanes))),
        "sourceCardRefs": chosen,
    }


def _completeness(lanes: Mapping[str, Any]) -> str:
    states = [
        audit.get("executionState")
        for audit in lanes.values()
        if isinstance(audit, Mapping) and audit.get("executionState") != "not_dispatched"
    ]
    if states and all(state == "completed" for state in states):
        return "complete"
    if any(state == "completed" for state in states) and any(state == "terminated" for state in states):
        return "partial"
    return "unavailable"


def _cards(raw: Any) -> list[dict[str, str]]:
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise HintError("invalid_envelope")
    cards: list[dict[str, str]] = []
    for card in raw:
        if not isinstance(card, Mapping):
            raise HintError("invalid_envelope")
        cards.append(
            {
                "id": str(card["id"]),
                "kind": str(card["kind"]),
                "use_class": str(card["use_class"]),
            }
        )
    return cards


def _dedup_key(target: str, lineage: str, provenance: Mapping[str, Any]) -> str:
    payload = {
        "target": target,
        "lineage": lineage,
        "sources": sorted(provenance["sourceCardRefs"]),
        "completeness": provenance["retrievalCompleteness"],
        "incompleteLanes": sorted(provenance["incompleteLanes"]),
    }
    return _digest(_canonical(payload))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _visible(_card_id: str) -> str:
    return "visible"
