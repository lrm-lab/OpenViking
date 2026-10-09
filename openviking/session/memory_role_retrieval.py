# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Role-projected retrieval entry (Q134–Q139, Q159) and exact dispatch.

Memory submits a discovery plan and receives candidate and prior cards.
Forged exact fields on a Memory plan are rejected before any discovery
call. Authorized non-Memory roles dispatch ``query.exact`` into the
current-log exact lane and semantic fields into discovery. The two lanes
do not share result seats.
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openviking.session.exact_lane import (
    DEFAULT_EXACT_LIMIT,
    ExactInputError,
    compile_regex,
    execute_exact_lane,
    parse_exact_clause,
    resolve_regex_engine,
)

EXACT_CAPABILITY_ABSENT = "exact_capability_absent_for_memory_agent"
EXACT_LANE_NOT_IN_DISCOVERY_ENTRY = "exact_lane_not_in_discovery_entry"
PLAN_DRIFT = "plan_drift"
SCHEMA_VERSION = "RetrievalEnvelopeV1"
USE_CLASS_POLICY = "memory-use-class-pending-central-v1"

MEMORY_ROLE = "memory"
EXACT_ROLES = frozenset({"reader", "critic", "refiner", "builder", "diagnosis"})
DISCOVERY_PUBLIC_CAP = 12

_EXACT_KEYS = frozenset(
    {
        "exact",
        "current_log_exact",
        "currentLogExact",
        "currentLogExactMaxHits",
        "currentEvidence",
        "current_evidence",
        "currentEvidenceRef",
        "current_evidence_ref",
        "currentEvidenceRefs",
        "current_evidence_refs",
        "exactLaneAudit",
        "exact_lane_audit",
        "exactQuota",
        "exact_quota",
        "exactLease",
        "exact_lease",
        "exactDeadline",
        "exact_deadline",
        "exactCancel",
        "exact_cancel",
        "exactTruncation",
        "exact_truncation",
        "exactCompleteness",
        "exact_completeness",
    }
)
_MEMORY_FIELDS = frozenset({"intentText", "dependencyPattern", "operationId"})
_PATTERN_FIELDS = frozenset({"wantedArtifact", "wantedEffect"})
_KIND_USE = {
    "fork": "memory_candidate",
    "branch": "memory_candidate",
    "skill_revision": "memory_candidate",
    "skill": "memory_candidate",
    "block_projection": "analogical_prior",
    "claim_projection": "analogical_prior",
    "segment": "analogical_prior",
    "atom": "analogical_prior",
}


class RetrievalContractError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class BoundContext:
    scope_ref: str
    snapshot_ref: str
    budget_lease_ref: str


Discover = Callable[[str | None, Mapping[str, str] | None], Sequence[Mapping[str, Any]]]
Authority = Callable[[str], BoundContext]
ExactSearcher = Callable[[Mapping[str, Any], BoundContext], Sequence[Mapping[str, Any]]]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fold_key(key: str) -> str:
    return key.replace("-", "").replace("_", "").lower()


_EXACT_FOLDED = frozenset(_fold_key(key) for key in _EXACT_KEYS)


def _is_exact_key(key: Any) -> bool:
    return isinstance(key, str) and _fold_key(key) in _EXACT_FOLDED


def _contains_exact(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _is_exact_key(key) or _contains_exact(item):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_exact(item) for item in value)
    return False


def _pattern(raw: Any) -> dict[str, str] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise RetrievalContractError("invalid_dependency_pattern")
    unknown = set(raw) - _PATTERN_FIELDS
    if unknown:
        raise RetrievalContractError("unknown_field")
    pattern: dict[str, str] = {}
    for key in ("wantedArtifact", "wantedEffect"):
        if key not in raw or raw[key] is None:
            continue
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise RetrievalContractError("invalid_dependency_pattern")
        pattern[key] = raw[key].strip()
    return pattern or None


def _intent(raw: Any) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise RetrievalContractError("invalid_intent")
    return raw.strip()


class RoleRetrievalService:
    """One in-process idempotency store for role-projected retrieve calls."""

    def __init__(self) -> None:
        self._seen: dict[str, tuple[str, dict[str, Any]]] = {}

    def retrieve(
        self,
        *,
        role: str,
        request: Mapping[str, Any],
        authority: Authority,
        discover: Discover,
        exact_searcher: ExactSearcher | None = None,
        regex_engine: Any | None = None,
    ) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise RetrievalContractError("invalid_request")
        if role == MEMORY_ROLE:
            if _contains_exact(request):
                raise RetrievalContractError(EXACT_CAPABILITY_ABSENT)
            return self._memory(request, authority, discover)
        if role not in EXACT_ROLES:
            raise RetrievalContractError("unknown_role")
        clause, limit, remainder, error = _peel_exact(request)
        if error == "forged_exact":
            raise RetrievalContractError(EXACT_LANE_NOT_IN_DISCOVERY_ENTRY)
        if error is not None:
            raise RetrievalContractError(error)
        return self._semantic(
            role,
            remainder,
            authority,
            discover,
            clause=clause,
            exact_limit=limit,
            exact_searcher=exact_searcher,
            regex_engine=regex_engine,
        )

    def _memory(
        self,
        request: Mapping[str, Any],
        authority: Authority,
        discover: Discover,
    ) -> dict[str, Any]:
        unknown = set(request) - _MEMORY_FIELDS
        if unknown:
            raise RetrievalContractError("unknown_field")
        intent = _intent(request.get("intentText"))
        pattern = _pattern(request.get("dependencyPattern"))
        if intent is None and pattern is None:
            raise RetrievalContractError("empty_discovery_plan")
        operation_id = request.get("operationId")
        if operation_id is None:
            operation_id = str(uuid.uuid4())
        elif not isinstance(operation_id, str) or not operation_id.strip():
            raise RetrievalContractError("invalid_operation_id")
        else:
            operation_id = operation_id.strip()
        return self._finish(
            role=MEMORY_ROLE,
            operation_id=operation_id,
            intent=intent,
            pattern=pattern,
            authority=authority,
            discover=discover,
        )

    def _semantic(
        self,
        role: str,
        request: Mapping[str, Any],
        authority: Authority,
        discover: Discover,
        *,
        clause: Mapping[str, Any] | None,
        exact_limit: int | None,
        exact_searcher: ExactSearcher | None,
        regex_engine: Any | None,
    ) -> dict[str, Any]:
        allowed = _MEMORY_FIELDS
        if set(request) - allowed:
            raise RetrievalContractError("unknown_field")
        operation_id = request.get("operationId")
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise RetrievalContractError("invalid_operation_id")
        intent = _intent(request.get("intentText"))
        pattern = _pattern(request.get("dependencyPattern"))
        if intent is None and pattern is None and clause is None:
            raise RetrievalContractError("empty_discovery_plan")
        compiled, regex_unavailable = _prepare_regex(clause, regex_engine)
        return self._finish(
            role=role,
            operation_id=operation_id.strip(),
            intent=intent,
            pattern=pattern,
            authority=authority,
            discover=discover,
            clause=clause,
            exact_limit=exact_limit,
            exact_searcher=exact_searcher,
            compiled_pattern=compiled,
            regex_unavailable=regex_unavailable,
        )

    def _finish(
        self,
        *,
        role: str,
        operation_id: str,
        intent: str | None,
        pattern: Mapping[str, str] | None,
        authority: Authority,
        discover: Discover,
        clause: Mapping[str, Any] | None = None,
        exact_limit: int | None = None,
        exact_searcher: ExactSearcher | None = None,
        compiled_pattern: Any | None = None,
        regex_unavailable: bool = False,
    ) -> dict[str, Any]:
        bound = authority(role)
        if not isinstance(bound, BoundContext):
            raise RetrievalContractError("authority_unbound")
        for ref in (bound.scope_ref, bound.snapshot_ref, bound.budget_lease_ref):
            if not isinstance(ref, str) or not ref.strip():
                raise RetrievalContractError("authority_unbound")
        plan: dict[str, Any] = {
            "role": role,
            "intentText": intent,
            "dependencyPattern": dict(pattern) if pattern else None,
            "scopeRef": bound.scope_ref,
            "snapshotRef": bound.snapshot_ref,
            "budgetLeaseRef": bound.budget_lease_ref,
        }
        if clause is not None:
            plan["exact"] = {
                "expression": clause["expression"],
                "syntax": clause["syntax"],
                "fields": list(clause["fields"]),
                "caseSensitive": clause["caseSensitive"],
            }
            plan["currentLogExactMaxHits"] = exact_limit
        plan_hash = hashlib.sha256(_canonical(plan).encode("utf-8")).hexdigest()
        previous = self._seen.get(operation_id)
        if previous is not None:
            seen_hash, envelope = previous
            if seen_hash != plan_hash:
                raise RetrievalContractError(PLAN_DRIFT)
            replay = copy.deepcopy(envelope)
            replay["duplicate"] = True
            return replay
        if intent is None and pattern is None:
            candidates: list[dict[str, str]] = []
            priors: list[dict[str, str]] = []
            discovery_audit = {
                "executionState": "not_dispatched",
                "resultState": "none",
                "reasonCode": "not_selected",
                "returnedCount": 0,
                "truncated": False,
            }
        else:
            found = discover(intent, dict(pattern) if pattern else None)
            if isinstance(found, (str, bytes)) or not isinstance(found, Sequence):
                raise RetrievalContractError("invalid_card")
            candidates, priors, truncated = _cap_discovery(_classify(found))
            returned = len(candidates) + len(priors)
            discovery_audit = {
                "executionState": "completed",
                "resultState": "hits" if returned else "empty",
                "reasonCode": None,
                "returnedCount": returned,
                "truncated": truncated,
            }
        envelope: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "operationId": operation_id,
            "planHash": plan_hash,
            "duplicate": False,
            "resolvedContext": {
                "scopeRef": bound.scope_ref,
                "snapshotRef": bound.snapshot_ref,
                "budgetLeaseRef": bound.budget_lease_ref,
            },
            "policyVersions": {"useClass": USE_CLASS_POLICY},
            "auditRef": f"audit-{plan_hash[:16]}",
            "memoryCandidateCards": candidates,
            "analogicalPriorCards": priors,
            "discoveryAudit": discovery_audit,
        }
        if clause is not None:
            envelope.update(
                _run_exact(
                    clause,
                    bound,
                    exact_limit if exact_limit is not None else DEFAULT_EXACT_LIMIT,
                    exact_searcher,
                    compiled_pattern,
                    regex_unavailable,
                )
            )
        stored = copy.deepcopy(envelope)
        self._seen[operation_id] = (plan_hash, stored)
        return copy.deepcopy(stored)


def _classify(cards: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    public: list[dict[str, str]] = []
    for card in cards:
        if not isinstance(card, Mapping):
            raise RetrievalContractError("invalid_card")
        claimed = card.get("useClass")
        if claimed == "current_evidence":
            raise RetrievalContractError("adapter_self_upgrade")
        kind = card.get("kind")
        if not isinstance(kind, str):
            raise RetrievalContractError("invalid_card")
        use = _KIND_USE.get(kind)
        if use is None:
            continue
        if claimed is not None and claimed != use:
            raise RetrievalContractError("adapter_self_upgrade")
        card_id = card.get("id")
        if not isinstance(card_id, str) or not card_id.strip():
            raise RetrievalContractError("invalid_card")
        public.append({"id": card_id.strip(), "kind": kind, "useClass": use})
    return public


def _peel_exact(
    request: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, int | None, dict[str, Any], str | None]:
    """Split a legal ``query.exact`` from forged exact payloads.

    The error ``forged_exact`` keeps the pre-dispatch rejection used for
    legacy and misspelled exact keys. A typed limit without a clause, and
    a malformed clause, are schema errors instead.
    """
    if not _contains_exact(request):
        return None, None, dict(request), None
    data = dict(request)
    saw_limit = False
    limit: int | None = None
    if "currentLogExactMaxHits" in data:
        saw_limit = True
        raw_limit = data.pop("currentLogExactMaxHits")
        if type(raw_limit) is not int:
            return None, None, dict(request), "forged_exact"
        if not 1 <= raw_limit <= DEFAULT_EXACT_LIMIT:
            return None, None, dict(request), "invalid_exact_limit"
        limit = raw_limit
    clause = None
    query = data.get("query")
    if isinstance(query, Mapping) and "exact" in query:
        extra = set(query) - {"exact"}
        if any(_is_exact_key(key) for key in extra):
            return None, None, dict(request), "forged_exact"
        if extra:
            return None, None, dict(request), "unknown_field"
        try:
            clause = parse_exact_clause(query["exact"])
        except ExactInputError as exc:
            return None, None, dict(request), exc.code
        data.pop("query")
    if _contains_exact(data):
        return None, None, dict(request), "forged_exact"
    if saw_limit and clause is None:
        return None, None, dict(request), "exact_limit_without_clause"
    if clause is not None and limit is None:
        limit = DEFAULT_EXACT_LIMIT
    return clause, limit, data, None


def _prepare_regex(
    clause: Mapping[str, Any] | None,
    engine: Any | None,
) -> tuple[Any | None, bool]:
    if clause is None or clause["syntax"] != "regex":
        return None, False
    resolved = resolve_regex_engine(engine)
    if resolved is None:
        return None, True
    try:
        return compile_regex(clause, resolved), False
    except ExactInputError as exc:
        raise RetrievalContractError(exc.code) from None


def _run_exact(
    clause: Mapping[str, Any],
    bound: BoundContext,
    limit: int,
    searcher: ExactSearcher | None,
    compiled_pattern: Any | None,
    dependency_unavailable: bool,
) -> dict[str, Any]:
    if dependency_unavailable or searcher is None:
        return execute_exact_lane(
            clause,
            [],
            limit=limit,
            snapshot_ref=bound.snapshot_ref,
            dependency_unavailable=True,
        )
    try:
        records = searcher(clause, bound)
    except Exception:
        records = None
    return execute_exact_lane(
        clause,
        records,
        limit=limit,
        snapshot_ref=bound.snapshot_ref,
        compiled_pattern=compiled_pattern,
        dependency_unavailable=False,
    )


def _cap_discovery(
    cards: Sequence[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], bool]:
    candidates = [card for card in cards if card["useClass"] == "memory_candidate"]
    priors = [card for card in cards if card["useClass"] == "analogical_prior"]
    combined = [("candidate", card) for card in candidates] + [
        ("prior", card) for card in priors
    ]
    truncated = len(combined) > DISCOVERY_PUBLIC_CAP
    kept = combined[:DISCOVERY_PUBLIC_CAP]
    return (
        [card for kind, card in kept if kind == "candidate"],
        [card for kind, card in kept if kind == "prior"],
        truncated,
    )


def public_card_from_retrieval(card: Any) -> dict[str, Any]:
    """Map a facade card onto the discovery classifier input.

    ACL labels and scores stay off the mapped card. A claimed ``useClass``
    is kept only so the classifier can reject adapter self-upgrade; the
    public envelope still receives the server-assigned class.
    """
    mapped: dict[str, Any] = {"id": str(card.payload_ref), "kind": str(card.kind)}
    claimed = getattr(card, "useClass", None)
    if claimed is not None:
        mapped["useClass"] = claimed
    return mapped
