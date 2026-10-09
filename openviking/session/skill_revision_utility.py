# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Skill revision utility ledger (Q126-A through skill-utility/Q133-A).

Pure domain record of revision consumption. It does not allocate discovery
seats and it is not called from production retrieval or execution paths.
Exploration quota is exposed for a later discovery-wide seat to read.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

CHAIN_STAGES: tuple[str, ...] = (
    "eligible",
    "offered",
    "selected",
    "invoked",
    "applied",
    "followthrough",
    "objective_outcome",
)

GRADE_OBSERVATIONAL = "observational"
GRADE_COUNTERFACTUAL = "counterfactual"
GRADE_CONFOUNDED = "confounded_set"
GRADE_SELF_REPORT = "self_report"
_UTILITY_GRADES = frozenset({GRADE_OBSERVATIONAL, GRADE_COUNTERFACTUAL})

STATUS_ACTIVE = "active"
STATUS_NEEDS_REVALIDATION = "needs_revalidation"
STATUS_ELIGIBLE = "eligible"
STATUS_QUARANTINED = "quarantined"
BAND_BASELINE = "baseline"
BAND_DEMOTED = "demoted"

MODE_EXPLORATION = "exploration"
MODE_EXPLOITATION = "exploitation"
_MODES = frozenset({MODE_EXPLORATION, MODE_EXPLOITATION})

NON_UTILITY_KINDS = frozenset({"offer", "exposure", "citation", "self_report"})
FOLLOWTHROUGH_OBSERVED = "observed"
_FOLLOWTHROUGH_VALUES = frozenset({"observed", "not_observed", "partial", "contradicted"})
_OUTCOME_VALUES = frozenset({"positive", "negative"})

OBSERVATIONAL_HALF_LIFE_DAYS = 30.0
COUNTERFACTUAL_HALF_LIFE_DAYS = 180.0
SHRINKAGE_STRENGTH = 2
LCB_Z = 1.64
NEUTRAL_PRIOR = 0.5

EXPLORATION_SHARE_CAP = 0.10
MIN_ATTRIBUTABLE_OUTCOMES = 5
MIN_TASK_LINEAGES = 3
PROVISIONAL_MEMORY_EXPLOITATION_SLOTS = 3
PROVISIONAL_MEMORY_EXPLORATION_SLOTS = 1

DEMOTE_LCB = 0.40
RESTORE_LCB = 0.60
MIN_HYSTERESIS_SAMPLES = 5

DEFAULT_POLICY_VERSION = "skill-utility-v1"

DIRECT_QUARANTINE_ACTORS = frozenset(
    {
        "server_verifier",
        "security_policy",
        "authoritative_revocation",
        "admin",
    }
)
PROPOSAL_ONLY_ACTORS = frozenset(
    {
        "memory",
        "diagnosis",
        "builder",
        "critic",
        "refiner",
        "reader",
        "task_agent",
    }
)
DRIFT_KINDS = frozenset(
    {
        "context_cluster",
        "tool",
        "model",
        "schema",
        "evidence",
        "acl",
        "compatibility",
    }
)
SEVERE_REASONS = frozenset(
    {
        "severe_security",
        "permission_boundary",
        "provenance_integrity",
        "authoritative_revocation",
        "executable_incompatibility",
    }
)

_CONTROL_STRING_FIELDS = (
    "anchor_state_snapshot",
    "task_input",
    "agent_model_prompt_policy",
    "permissions",
    "budget",
    "evaluator",
)
_DAY_SECONDS = 86400.0


class SkillUtilityError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _require_str(value: Any, code: str) -> str:
    if not isinstance(value, str) or value == "":
        raise SkillUtilityError(code)
    return value


def _require_aware(value: Any, code: str = "invalid_timestamp") -> datetime:
    if not isinstance(value, datetime):
        raise SkillUtilityError(code)
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise SkillUtilityError("naive_timestamp")
    return value.astimezone(timezone.utc)


def _require_finite(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SkillUtilityError(code)
    number = float(value)
    if not math.isfinite(number):
        raise SkillUtilityError(code)
    return number


def utility_lcb(n: float, s: float, g: float) -> float:
    """Conservative lower bound after shrinkage toward ``g``.

    ``shrunk = (n * (s / n) + 2 * g) / (n + 2)``
    ``se = sqrt(shrunk * (1 - shrunk) / n)``
    ``LCB = shrunk - 1.64 * se``
    """
    n = _require_finite(n, "invalid_sample")
    s = _require_finite(s, "invalid_sample")
    g = _require_finite(g, "invalid_prior")
    if n <= 0 or s < 0 or s > n + 1e-9:
        raise SkillUtilityError("invalid_sample")
    if s > n:
        s = n
    if g < 0 or g > 1:
        raise SkillUtilityError("invalid_prior")
    return _lcb_parts(n, s, g)[2]


def exploration_priority(n: float, s: float, g: float) -> float:
    """Higher standard error is more exploration-eligible. It is not an exploit bonus."""
    n = _require_finite(n, "invalid_sample")
    if n <= 0:
        return math.inf
    s = _require_finite(s, "invalid_sample")
    g = _require_finite(g, "invalid_prior")
    if s < 0 or s > n + 1e-9:
        raise SkillUtilityError("invalid_sample")
    if s > n:
        s = n
    if g < 0 or g > 1:
        raise SkillUtilityError("invalid_prior")
    return _lcb_parts(n, s, g)[1]


def evidence_weight(
    grade: str,
    age: timedelta,
    *,
    half_life_days: float | None = None,
) -> float:
    """``weight = 2 ** (-age_days / half_life)``. Half-life weight is 0.5."""
    if not isinstance(age, timedelta):
        raise SkillUtilityError("invalid_age")
    if age.total_seconds() < 0:
        raise SkillUtilityError("negative_age")
    if half_life_days is None:
        half_life_days = _default_half_life(grade)
    else:
        half_life_days = _require_finite(half_life_days, "invalid_half_life")
        if half_life_days <= 0:
            raise SkillUtilityError("invalid_half_life")
    age_days = age.total_seconds() / _DAY_SECONDS
    return 2.0 ** (-age_days / half_life_days)


def projected_exploration_share(exploration_count: int, exploitation_count: int) -> float:
    """Share if one more exploration is admitted: ``(e + 1) / (e + x + 1)``."""
    if (
        isinstance(exploration_count, bool)
        or isinstance(exploitation_count, bool)
        or not isinstance(exploration_count, int)
        or not isinstance(exploitation_count, int)
        or exploration_count < 0
        or exploitation_count < 0
    ):
        raise SkillUtilityError("invalid_selection_count")
    return (exploration_count + 1) / (exploration_count + exploitation_count + 1)


def rank_exploitation(
    candidates: Sequence[ExploitationCandidate],
) -> tuple[ExploitationRank, ...]:
    """Sort by ``utility_lcb`` descending. Mean and uncertainty bonus are not keys."""
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise SkillUtilityError("invalid_candidate")
    rows: list[ExploitationRank] = []
    for candidate in candidates:
        if not isinstance(candidate, ExploitationCandidate):
            raise SkillUtilityError("invalid_candidate")
        _shrunk, se, lcb, mean = _validated_parts(candidate.n, candidate.s, candidate.g)
        rows.append(
            ExploitationRank(
                skill_revision_id=candidate.skill_revision_id,
                context_cluster=candidate.context_cluster,
                n=float(candidate.n),
                s=float(candidate.s),
                g=float(candidate.g),
                posterior_mean=mean,
                utility_lcb=lcb,
                uncertainty=se,
                effect_band=candidate.effect_band,
            )
        )
    rows.sort(key=lambda row: (-row.utility_lcb, row.skill_revision_id))
    return tuple(rows)


def _lcb_parts(n: float, s: float, g: float) -> tuple[float, float, float, float]:
    shrunk = (n * (s / n) + SHRINKAGE_STRENGTH * g) / (n + SHRINKAGE_STRENGTH)
    se = math.sqrt(shrunk * (1 - shrunk) / n)
    lcb = shrunk - LCB_Z * se
    return shrunk, se, lcb, s / n


def _validated_parts(n: float, s: float, g: float) -> tuple[float, float, float, float]:
    utility_lcb(n, s, g)
    return _lcb_parts(float(n), float(s), float(g))


def _default_half_life(grade: str) -> float:
    if grade == GRADE_OBSERVATIONAL:
        return OBSERVATIONAL_HALF_LIFE_DAYS
    if grade == GRADE_COUNTERFACTUAL:
        return COUNTERFACTUAL_HALF_LIFE_DAYS
    raise SkillUtilityError("weight_undefined")


@dataclass(frozen=True)
class UtilityPolicy:
    version: str
    observational_half_life_days: float = OBSERVATIONAL_HALF_LIFE_DAYS
    counterfactual_half_life_days: float = COUNTERFACTUAL_HALF_LIFE_DAYS

    def __post_init__(self) -> None:
        _require_str(self.version, "invalid_policy")
        for name in ("observational_half_life_days", "counterfactual_half_life_days"):
            value = _require_finite(getattr(self, name), "invalid_half_life")
            if value <= 0:
                raise SkillUtilityError("invalid_half_life")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ControlContract:
    """Frozen paired-control fields. A satisfied contract grades a unique outcome as counterfactual."""

    anchor_state_snapshot: str
    task_input: str
    agent_model_prompt_policy: str
    tool_dependency_versions: tuple[tuple[str, str], ...] | Mapping[str, str]
    permissions: str
    budget: str
    evaluator: str

    def __post_init__(self) -> None:
        for name in _CONTROL_STRING_FIELDS:
            _require_str(getattr(self, name), "invalid_control_contract")
        versions = self.tool_dependency_versions
        if isinstance(versions, Mapping):
            items = versions.items()
        elif isinstance(versions, tuple):
            items = versions
        else:
            raise SkillUtilityError("invalid_control_contract")
        normalized: list[tuple[str, str]] = []
        for key, value in items:
            if not isinstance(key, str) or key == "" or not isinstance(value, str):
                raise SkillUtilityError("invalid_control_contract")
            normalized.append((key, value))
        object.__setattr__(self, "tool_dependency_versions", tuple(sorted(normalized)))


@dataclass(frozen=True)
class ChainEvent:
    event_id: str
    invocation_id: str
    stage: str
    skill_revision_id: str
    task_id: str
    run_id: str
    task_lineage_id: str
    context_cluster: str
    outcome_policy_version: str
    occurred_at: datetime
    outcome_id: str | None = None
    outcome_value: str | None = None
    followthrough: str | None = None
    control_contract: ControlContract | None = None
    relative_difference: float | None = None
    self_reported: bool = False


@dataclass(frozen=True)
class NonUtilityEvent:
    event_id: str
    kind: str
    skill_revision_id: str
    occurred_at: datetime
    payload: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class EvidenceView:
    outcome_id: str
    grade: str
    skill_revision_ids: tuple[str, ...]
    invocation_ids: tuple[str, ...]
    task_lineage_id: str | None
    context_cluster: str | None
    context_clusters: tuple[str, ...]
    success: bool | None
    utility_credited: bool
    outcome_value: str | None
    occurred_at: datetime
    outcome_policy_version: str
    weight: float | None
    status: str
    relative_difference: float | None


@dataclass(frozen=True)
class LocalPosterior:
    skill_revision_id: str
    context_cluster: str
    grade: str
    n: float
    s: float
    g: float
    posterior_mean: float | None
    utility_lcb: float | None
    uncertainty: float | None


@dataclass(frozen=True)
class ExploitationCandidate:
    skill_revision_id: str
    n: float
    s: float
    g: float
    uncertainty_bonus: float = 0.0
    context_cluster: str = ""
    effect_band: str = BAND_BASELINE


@dataclass(frozen=True)
class ExploitationRank:
    skill_revision_id: str
    context_cluster: str
    n: float
    s: float
    g: float
    posterior_mean: float
    utility_lcb: float
    uncertainty: float
    effect_band: str


@dataclass(frozen=True)
class ExplorationCandidate:
    skill_revision_id: str
    priority: float
    may_exploit: bool


@dataclass(frozen=True)
class ExplorationQuota:
    share_cap: float
    exploration_count: int
    exploitation_count: int
    next_exploration_share: float
    next_exploration_allowed: bool
    min_attributable_outcomes: int
    min_task_lineages: int
    provisional_memory_exploitation_slots: int
    provisional_memory_exploration_slots: int
    consumes_provisional_memory_quota: bool


@dataclass(frozen=True)
class RevisionView:
    skill_revision_id: str
    published_at: datetime
    ttl_days: None
    expires_at: None
    online_status: str
    may_exploit: bool
    attributable_outcomes: int
    task_lineages: int


@dataclass(frozen=True)
class SelectionAudit:
    selection_id: str
    mode: str
    invocation_id: str
    skill_revision_id: str
    eligible_set: tuple[str, ...]
    seed: str
    selection_propensity: float
    invocation_cap: int
    occurred_at: datetime
    projected_share: float


@dataclass(frozen=True)
class GovernanceDecision:
    action: str
    actor: str
    skill_revision_id: str
    reason: str
    evidence_refs: tuple[str, ...]
    online_status: str
    changed_online_status: bool
    occurred_at: datetime
    idempotency_key: str


@dataclass(frozen=True)
class StatusTransition:
    skill_revision_id: str
    action: str
    actor: str
    reason: str
    occurred_at: datetime
    evidence_refs: tuple[str, ...]
    severe: bool
    expires_at: None
    idempotency_key: str


@dataclass
class _Invocation:
    invocation_id: str
    skill_revision_id: str
    task_id: str
    run_id: str
    task_lineage_id: str
    context_cluster: str
    outcome_policy_version: str
    stages: dict[str, ChainEvent]


@dataclass(frozen=True)
class _Revision:
    skill_revision_id: str
    published_at: datetime


@dataclass(frozen=True)
class _Drift:
    drift_id: str
    kind: str
    occurred_at: datetime
    skill_revision_id: str | None
    context_cluster: str | None


@dataclass(frozen=True)
class _Revalidation:
    outcome_id: str
    occurred_at: datetime
    note: str


class SkillUtilityLedger:
    """Append-only consumption ledger for one skill revision at a time."""

    def __init__(self, policy: UtilityPolicy | None = None) -> None:
        default = policy if policy is not None else UtilityPolicy(version=DEFAULT_POLICY_VERSION)
        if not isinstance(default, UtilityPolicy):
            raise SkillUtilityError("invalid_policy")
        self._policies: dict[str, UtilityPolicy] = {default.version: default}
        self._revisions: dict[str, _Revision] = {}
        self._events: list[ChainEvent] = []
        self._raw: list[dict[str, Any]] = []
        self._invocations: dict[str, _Invocation] = {}
        self._non_utility: list[NonUtilityEvent] = []
        self._drifts: list[_Drift] = []
        self._revalidations: list[_Revalidation] = []
        self._transitions: list[StatusTransition] = []
        self._proposals: list[GovernanceDecision] = []
        self._selections: list[SelectionAudit] = []
        self._idempotency: dict[str, tuple[tuple[Any, ...], GovernanceDecision]] = {}

    def register_policy(self, policy: UtilityPolicy) -> None:
        if not isinstance(policy, UtilityPolicy):
            raise SkillUtilityError("invalid_policy")
        existing = self._policies.get(policy.version)
        if existing is not None and existing != policy:
            raise SkillUtilityError("policy_frozen")
        self._policies[policy.version] = policy

    def publish_active(self, skill_revision_id: str, *, published_at: datetime) -> RevisionView:
        revision_id = _require_str(skill_revision_id, "invalid_revision")
        published = _require_aware(published_at)
        if revision_id in self._revisions:
            raise SkillUtilityError("already_published")
        self._revisions[revision_id] = _Revision(revision_id, published)
        return self.revision(revision_id)

    def revision(self, skill_revision_id: str, *, as_of: datetime | None = None) -> RevisionView:
        revision = self._require_revision(skill_revision_id)
        outcomes, lineages = self._attributable(revision.skill_revision_id, as_of)
        return RevisionView(
            skill_revision_id=revision.skill_revision_id,
            published_at=revision.published_at,
            ttl_days=None,
            expires_at=None,
            online_status=self.online_status(revision.skill_revision_id, as_of=as_of),
            may_exploit=self.may_exploit(revision.skill_revision_id, as_of=as_of),
            attributable_outcomes=outcomes,
            task_lineages=lineages,
        )

    def is_online_eligible(self, skill_revision_id: str, *, as_of: datetime | None = None) -> bool:
        revision = self._require_revision(skill_revision_id)
        if as_of is not None:
            moment = _require_aware(as_of)
            if moment < revision.published_at:
                return False
        return self.online_status(skill_revision_id, as_of=as_of) == STATUS_ELIGIBLE

    def may_exploit(self, skill_revision_id: str, *, as_of: datetime | None = None) -> bool:
        if not self.is_online_eligible(skill_revision_id, as_of=as_of):
            return False
        outcomes, lineages = self._attributable(skill_revision_id, as_of)
        return outcomes >= MIN_ATTRIBUTABLE_OUTCOMES and lineages >= MIN_TASK_LINEAGES

    def online_status(self, skill_revision_id: str, *, as_of: datetime | None = None) -> str:
        self._require_revision(skill_revision_id)
        moment = _require_aware(as_of) if as_of is not None else None
        status = STATUS_ELIGIBLE
        for transition in self._transitions:
            if transition.skill_revision_id != skill_revision_id:
                continue
            if moment is not None and transition.occurred_at > moment:
                continue
            if transition.action == "quarantine":
                status = STATUS_QUARANTINED
            elif transition.action == "release":
                status = STATUS_ELIGIBLE
        return status

    def append_event(
        self,
        *,
        invocation_id: str,
        stage: str,
        skill_revision_id: str,
        task_id: str,
        run_id: str,
        task_lineage_id: str,
        context_cluster: str,
        outcome_policy_version: str,
        occurred_at: datetime,
        outcome_id: str | None = None,
        outcome_value: str | None = None,
        followthrough: str | None = None,
        control_contract: ControlContract | Mapping[str, Any] | None = None,
        relative_difference: float | None = None,
        self_reported: bool = False,
    ) -> ChainEvent:
        invocation = _require_str(invocation_id, "invalid_invocation")
        stage_name = _require_str(stage, "invalid_stage")
        if stage_name not in CHAIN_STAGES:
            raise SkillUtilityError("invalid_stage")
        revision_id = _require_str(skill_revision_id, "invalid_revision")
        self._require_revision(revision_id)
        binding = {
            "skill_revision_id": revision_id,
            "task_id": _require_str(task_id, "invalid_binding"),
            "run_id": _require_str(run_id, "invalid_binding"),
            "task_lineage_id": _require_str(task_lineage_id, "invalid_binding"),
            "context_cluster": _require_str(context_cluster, "invalid_binding"),
            "outcome_policy_version": _require_str(outcome_policy_version, "invalid_policy"),
        }
        if binding["outcome_policy_version"] not in self._policies:
            raise SkillUtilityError("unknown_policy")
        moment = _require_aware(occurred_at)
        if not isinstance(self_reported, bool):
            raise SkillUtilityError("invalid_field")
        contract = _coerce_contract(control_contract)
        difference = _coerce_difference(relative_difference)
        state = self._invocations.get(invocation)
        self._check_stage_transition(state, stage_name, binding)
        outcome_key = self._validated_outcome_id(state, stage_name, outcome_id)
        follow_value = self._validated_followthrough(stage_name, followthrough)
        result_value = self._validated_outcome_value(stage_name, outcome_value)
        self._validated_contract(stage_name, result_value, contract, difference, self_reported)
        if state is not None and outcome_key is not None:
            self._reject_duplicate_outcome(revision_id, invocation, outcome_key)
        event = ChainEvent(
            event_id=f"evt-{len(self._events) + 1:04d}",
            invocation_id=invocation,
            stage=stage_name,
            occurred_at=moment,
            outcome_id=outcome_key,
            outcome_value=result_value,
            followthrough=follow_value,
            control_contract=contract,
            relative_difference=difference,
            self_reported=self_reported,
            **binding,
        )
        raw = _event_snapshot(event)
        if state is None:
            state = _Invocation(invocation_id=invocation, stages={}, **binding)
            self._invocations[invocation] = state
        state.stages[stage_name] = event
        self._events.append(event)
        self._raw.append(raw)
        return event

    def events(self) -> tuple[ChainEvent, ...]:
        return tuple(self._events)

    def raw_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(_copy_raw(item) for item in self._raw)

    def record_non_utility(
        self,
        *,
        kind: str,
        skill_revision_id: str,
        occurred_at: datetime,
        payload: Mapping[str, str] | None = None,
    ) -> NonUtilityEvent:
        kind_name = _require_str(kind, "invalid_kind")
        if kind_name not in NON_UTILITY_KINDS:
            raise SkillUtilityError("invalid_kind")
        revision_id = _require_str(skill_revision_id, "invalid_revision")
        self._require_revision(revision_id)
        moment = _require_aware(occurred_at)
        stored_payload = _freeze_payload(payload)
        event = NonUtilityEvent(
            event_id=f"nu-{len(self._non_utility) + 1:04d}",
            kind=kind_name,
            skill_revision_id=revision_id,
            occurred_at=moment,
            payload=stored_payload,
        )
        self._non_utility.append(event)
        return event

    def non_utility_events(self) -> tuple[NonUtilityEvent, ...]:
        return tuple(self._non_utility)

    def evidence_views(
        self,
        *,
        as_of: datetime | None = None,
        policy_version: str | None = None,
    ) -> tuple[EvidenceView, ...]:
        moment = _require_aware(as_of) if as_of is not None else None
        if policy_version is not None:
            self._require_policy(policy_version)
        groups: dict[str, list[_Invocation]] = {}
        for state in self._invocations.values():
            applied = state.stages.get("applied")
            if applied is None or applied.outcome_id is None:
                continue
            groups.setdefault(applied.outcome_id, []).append(state)
        views: list[EvidenceView] = []
        for outcome_id in sorted(groups):
            view = self._view_for(outcome_id, groups[outcome_id], moment, policy_version)
            if view is not None:
                views.append(view)
        views.sort(key=lambda item: (item.occurred_at, item.outcome_id))
        return tuple(views)

    def evidence(
        self,
        outcome_id: str,
        *,
        as_of: datetime | None = None,
        policy_version: str | None = None,
    ) -> EvidenceView:
        key = _require_str(outcome_id, "unknown_outcome")
        for view in self.evidence_views(as_of=as_of, policy_version=policy_version):
            if view.outcome_id == key:
                return view
        raise SkillUtilityError("unknown_outcome")

    def projected_weight(
        self,
        outcome_id: str,
        *,
        as_of: datetime,
        policy_version: str | None = None,
    ) -> float:
        view = self.evidence(outcome_id, as_of=as_of, policy_version=policy_version)
        if view.weight is None:
            raise SkillUtilityError("weight_undefined")
        return view.weight

    def credited_positive_utility(self, skill_revision_id: str) -> float:
        revision_id = _require_str(skill_revision_id, "invalid_revision")
        total = 0.0
        for view in self.evidence_views():
            if view.utility_credited and view.success and view.skill_revision_ids == (revision_id,):
                total += 1.0
        return total

    def local_posterior(
        self,
        skill_revision_id: str,
        context_cluster: str,
        *,
        grade: str,
        as_of: datetime,
        policy_version: str | None = None,
    ) -> LocalPosterior:
        revision_id = _require_str(skill_revision_id, "invalid_revision")
        self._require_revision(revision_id)
        cluster = _require_str(context_cluster, "invalid_binding")
        if grade not in _UTILITY_GRADES:
            raise SkillUtilityError("invalid_grade")
        moment = _require_aware(as_of)
        n, s = self._weighted(revision_id, grade, moment, policy_version, cluster=cluster)
        g = self._global_mean(revision_id, grade, moment, policy_version, cluster)
        mean = None
        lcb = None
        uncertainty = None
        if n > 0:
            _shrunk, uncertainty, lcb, mean = _lcb_parts(n, s, g)
        return LocalPosterior(
            skill_revision_id=revision_id,
            context_cluster=cluster,
            grade=grade,
            n=n,
            s=s,
            g=g,
            posterior_mean=mean,
            utility_lcb=lcb,
            uncertainty=uncertainty,
        )

    def effect_band(
        self,
        skill_revision_id: str,
        context_cluster: str,
        *,
        as_of: datetime,
        grade: str = GRADE_OBSERVATIONAL,
    ) -> str:
        revision_id = _require_str(skill_revision_id, "invalid_revision")
        self._require_revision(revision_id)
        cluster = _require_str(context_cluster, "invalid_binding")
        if grade not in _UTILITY_GRADES:
            raise SkillUtilityError("invalid_grade")
        moment = _require_aware(as_of)
        samples = [
            view
            for view in self._active_credited(revision_id, grade, moment, None)
            if view.context_cluster == cluster
        ]
        samples.sort(key=lambda view: (view.occurred_at, view.outcome_id))
        g = self._global_mean(revision_id, grade, moment, None, cluster)
        band = BAND_BASELINE
        n = 0.0
        s = 0.0
        for view in samples:
            n += 1.0
            if view.success:
                s += 1.0
            if n < MIN_HYSTERESIS_SAMPLES:
                continue
            lcb = utility_lcb(n, s, g)
            if band == BAND_BASELINE and lcb < DEMOTE_LCB:
                band = BAND_DEMOTED
            elif band == BAND_DEMOTED and lcb > RESTORE_LCB:
                band = BAND_BASELINE
        return band

    def exploitation_ranking(
        self,
        context_cluster: str,
        *,
        as_of: datetime,
    ) -> tuple[ExploitationRank, ...]:
        cluster = _require_str(context_cluster, "invalid_binding")
        moment = _require_aware(as_of)
        candidates: list[ExploitationCandidate] = []
        for revision_id in self._revisions:
            if not self.may_exploit(revision_id, as_of=moment):
                continue
            chosen = self._exploit_posterior(revision_id, cluster, moment)
            if chosen is None or chosen.utility_lcb is None:
                continue
            candidates.append(
                ExploitationCandidate(
                    skill_revision_id=revision_id,
                    n=chosen.n,
                    s=chosen.s,
                    g=chosen.g,
                    context_cluster=cluster,
                    effect_band=self.effect_band(
                        revision_id,
                        cluster,
                        as_of=moment,
                        grade=chosen.grade,
                    ),
                )
            )
        return rank_exploitation(candidates)

    def exploration_candidates(self, *, as_of: datetime) -> tuple[ExplorationCandidate, ...]:
        moment = _require_aware(as_of)
        rows: list[ExplorationCandidate] = []
        for revision_id in self._revisions:
            if not self.is_online_eligible(revision_id, as_of=moment):
                continue
            rows.append(
                ExplorationCandidate(
                    skill_revision_id=revision_id,
                    priority=self._exploration_priority(revision_id, moment),
                    may_exploit=self.may_exploit(revision_id, as_of=moment),
                )
            )
        rows.sort(key=lambda row: (-row.priority, row.skill_revision_id))
        return tuple(rows)

    def exploration_quota(self) -> ExplorationQuota:
        explorations, exploitations = self._selection_counts()
        share = projected_exploration_share(explorations, exploitations)
        return ExplorationQuota(
            share_cap=EXPLORATION_SHARE_CAP,
            exploration_count=explorations,
            exploitation_count=exploitations,
            next_exploration_share=share,
            next_exploration_allowed=share <= EXPLORATION_SHARE_CAP,
            min_attributable_outcomes=MIN_ATTRIBUTABLE_OUTCOMES,
            min_task_lineages=MIN_TASK_LINEAGES,
            provisional_memory_exploitation_slots=PROVISIONAL_MEMORY_EXPLOITATION_SLOTS,
            provisional_memory_exploration_slots=PROVISIONAL_MEMORY_EXPLORATION_SLOTS,
            consumes_provisional_memory_quota=False,
        )

    def commit_selection(
        self,
        *,
        mode: str,
        invocation_id: str,
        skill_revision_id: str,
        eligible_set: Sequence[str],
        seed: str,
        selection_propensity: float,
        invocation_cap: int,
        occurred_at: datetime,
    ) -> SelectionAudit:
        mode_name = _require_str(mode, "invalid_mode")
        if mode_name not in _MODES:
            raise SkillUtilityError("invalid_mode")
        invocation = _require_str(invocation_id, "invalid_invocation")
        revision_id = _require_str(skill_revision_id, "invalid_revision")
        self._require_revision(revision_id)
        members = _eligible_set(eligible_set, revision_id)
        seed_value = _require_str(seed, "invalid_seed")
        propensity = _require_finite(selection_propensity, "invalid_propensity")
        if propensity <= 0 or propensity > 1:
            raise SkillUtilityError("invalid_propensity")
        if (
            isinstance(invocation_cap, bool)
            or not isinstance(invocation_cap, int)
            or invocation_cap < 1
        ):
            raise SkillUtilityError("invalid_invocation_cap")
        moment = _require_aware(occurred_at)
        if any(item.invocation_id == invocation for item in self._selections):
            raise SkillUtilityError("duplicate_selection")
        if not self.is_online_eligible(revision_id, as_of=moment):
            raise SkillUtilityError("not_eligible")
        explorations, exploitations = self._selection_counts()
        share = projected_exploration_share(explorations, exploitations)
        if mode_name == MODE_EXPLOITATION:
            if not self.may_exploit(revision_id, as_of=moment):
                raise SkillUtilityError("exploitation_not_allowed")
        elif share > EXPLORATION_SHARE_CAP:
            raise SkillUtilityError("exploration_share_exceeded")
        audit = SelectionAudit(
            selection_id=f"sel-{len(self._selections) + 1:04d}",
            mode=mode_name,
            invocation_id=invocation,
            skill_revision_id=revision_id,
            eligible_set=members,
            seed=seed_value,
            selection_propensity=propensity,
            invocation_cap=invocation_cap,
            occurred_at=moment,
            projected_share=share,
        )
        self._selections.append(audit)
        return audit

    def selections(self) -> tuple[SelectionAudit, ...]:
        return tuple(self._selections)

    def declare_drift(
        self,
        *,
        kind: str,
        occurred_at: datetime,
        skill_revision_id: str | None = None,
        context_cluster: str | None = None,
    ) -> None:
        kind_name = _require_str(kind, "invalid_drift")
        if kind_name not in DRIFT_KINDS:
            raise SkillUtilityError("invalid_drift")
        revision_id = (
            None
            if skill_revision_id is None
            else _require_str(skill_revision_id, "invalid_revision")
        )
        cluster = (
            None if context_cluster is None else _require_str(context_cluster, "invalid_binding")
        )
        if revision_id is None and cluster is None:
            raise SkillUtilityError("invalid_drift")
        if revision_id is not None:
            self._require_revision(revision_id)
        moment = _require_aware(occurred_at)
        self._drifts.append(
            _Drift(
                drift_id=f"drift-{len(self._drifts) + 1:04d}",
                kind=kind_name,
                occurred_at=moment,
                skill_revision_id=revision_id,
                context_cluster=cluster,
            )
        )

    def revalidate_evidence(self, outcome_id: str, *, occurred_at: datetime, note: str) -> None:
        key = _require_str(outcome_id, "unknown_outcome")
        self.evidence(key)
        moment = _require_aware(occurred_at)
        text = _require_str(note, "invalid_note")
        self._revalidations.append(_Revalidation(outcome_id=key, occurred_at=moment, note=text))

    def submit_quarantine(
        self,
        *,
        actor: str,
        skill_revision_id: str,
        reason: str,
        occurred_at: datetime,
        idempotency_key: str,
        evidence_refs: Sequence[str] | None = None,
    ) -> GovernanceDecision:
        return self._govern(
            action="quarantine",
            actor=actor,
            skill_revision_id=skill_revision_id,
            reason=reason,
            occurred_at=occurred_at,
            idempotency_key=idempotency_key,
            evidence_refs=evidence_refs,
        )

    def release_quarantine(
        self,
        *,
        actor: str,
        skill_revision_id: str,
        reason: str,
        occurred_at: datetime,
        idempotency_key: str,
        evidence_refs: Sequence[str] | None = None,
    ) -> GovernanceDecision:
        actor_name = _require_str(actor, "actor_not_authorized")
        if actor_name not in DIRECT_QUARANTINE_ACTORS:
            raise SkillUtilityError("actor_not_authorized")
        revision_id = _require_str(skill_revision_id, "invalid_revision")
        self._require_revision(revision_id)
        key = _require_str(idempotency_key, "invalid_idempotency_key")
        if key not in self._idempotency and self.online_status(revision_id) != STATUS_QUARANTINED:
            raise SkillUtilityError("not_quarantined")
        return self._govern(
            action="release",
            actor=actor_name,
            skill_revision_id=revision_id,
            reason=reason,
            occurred_at=occurred_at,
            idempotency_key=idempotency_key,
            evidence_refs=evidence_refs,
            actor_checked=True,
        )

    def quarantine_transitions(self, skill_revision_id: str) -> tuple[StatusTransition, ...]:
        revision_id = _require_str(skill_revision_id, "invalid_revision")
        return tuple(item for item in self._transitions if item.skill_revision_id == revision_id)

    def quarantine_proposals(self, skill_revision_id: str) -> tuple[GovernanceDecision, ...]:
        revision_id = _require_str(skill_revision_id, "invalid_revision")
        return tuple(item for item in self._proposals if item.skill_revision_id == revision_id)

    def _govern(
        self,
        *,
        action: str,
        actor: str,
        skill_revision_id: str,
        reason: str,
        occurred_at: datetime,
        idempotency_key: str,
        evidence_refs: Sequence[str] | None,
        actor_checked: bool = False,
    ) -> GovernanceDecision:
        actor_name = _require_str(actor, "actor_not_authorized")
        if not actor_checked:
            if actor_name in DIRECT_QUARANTINE_ACTORS:
                pass
            elif actor_name in PROPOSAL_ONLY_ACTORS:
                if action != "quarantine":
                    raise SkillUtilityError("actor_not_authorized")
            else:
                raise SkillUtilityError("actor_not_authorized")
        revision_id = _require_str(skill_revision_id, "invalid_revision")
        self._require_revision(revision_id)
        reason_text = _require_str(reason, "invalid_reason")
        moment = _require_aware(occurred_at)
        key = _require_str(idempotency_key, "invalid_idempotency_key")
        refs = _evidence_refs(evidence_refs)
        proposal_only = actor_name in PROPOSAL_ONLY_ACTORS and action == "quarantine"
        if proposal_only and not refs:
            raise SkillUtilityError("evidence_refs_required")
        fingerprint = (action, actor_name, revision_id, reason_text, refs)
        existing = self._idempotency.get(key)
        if existing is not None:
            if existing[0] != fingerprint:
                raise SkillUtilityError("idempotency_conflict")
            return existing[1]
        if proposal_only:
            decision = GovernanceDecision(
                action="proposed",
                actor=actor_name,
                skill_revision_id=revision_id,
                reason=reason_text,
                evidence_refs=refs,
                online_status=self.online_status(revision_id),
                changed_online_status=False,
                occurred_at=moment,
                idempotency_key=key,
            )
            self._proposals.append(decision)
            self._idempotency[key] = (fingerprint, decision)
            return decision
        before = self.online_status(revision_id)
        stored_action = "quarantine" if action == "quarantine" else "release"
        self._transitions.append(
            StatusTransition(
                skill_revision_id=revision_id,
                action=stored_action,
                actor=actor_name,
                reason=reason_text,
                occurred_at=moment,
                evidence_refs=refs,
                severe=reason_text in SEVERE_REASONS,
                expires_at=None,
                idempotency_key=key,
            )
        )
        after = self.online_status(revision_id)
        decision = GovernanceDecision(
            action="quarantined" if stored_action == "quarantine" else "released",
            actor=actor_name,
            skill_revision_id=revision_id,
            reason=reason_text,
            evidence_refs=refs,
            online_status=after,
            changed_online_status=before != after,
            occurred_at=moment,
            idempotency_key=key,
        )
        self._idempotency[key] = (fingerprint, decision)
        return decision

    def _require_revision(self, skill_revision_id: str) -> _Revision:
        revision = self._revisions.get(skill_revision_id)
        if revision is None:
            raise SkillUtilityError("unknown_revision")
        return revision

    def _require_policy(self, version: str) -> UtilityPolicy:
        policy = self._policies.get(version)
        if policy is None:
            raise SkillUtilityError("unknown_policy")
        return policy

    def _check_stage_transition(
        self,
        state: _Invocation | None,
        stage: str,
        binding: Mapping[str, str],
    ) -> None:
        if state is None:
            if stage != CHAIN_STAGES[0]:
                raise SkillUtilityError("skipped_stage")
            return
        for name, value in binding.items():
            if getattr(state, name) != value:
                raise SkillUtilityError("binding_mismatch")
        if stage in state.stages:
            raise SkillUtilityError("duplicate_stage")
        if len(state.stages) >= len(CHAIN_STAGES):
            raise SkillUtilityError("chain_complete")
        expected = CHAIN_STAGES[len(state.stages)]
        if stage != expected:
            raise SkillUtilityError("skipped_stage")

    def _validated_outcome_id(
        self,
        state: _Invocation | None,
        stage: str,
        outcome_id: str | None,
    ) -> str | None:
        stage_index = CHAIN_STAGES.index(stage)
        applied_index = CHAIN_STAGES.index("applied")
        if stage_index < applied_index:
            if outcome_id is not None:
                raise SkillUtilityError("invalid_field")
            return None
        key = _require_str(outcome_id, "invalid_outcome")
        if state is not None:
            applied = state.stages.get("applied")
            if applied is not None and applied.outcome_id != key:
                raise SkillUtilityError("binding_mismatch")
        return key

    def _validated_followthrough(self, stage: str, followthrough: str | None) -> str | None:
        if stage != "followthrough":
            if followthrough is not None:
                raise SkillUtilityError("invalid_field")
            return None
        if followthrough not in _FOLLOWTHROUGH_VALUES:
            raise SkillUtilityError("invalid_followthrough")
        return followthrough

    def _validated_outcome_value(self, stage: str, outcome_value: str | None) -> str | None:
        if stage != "objective_outcome":
            if outcome_value is not None:
                raise SkillUtilityError("invalid_field")
            return None
        if outcome_value not in _OUTCOME_VALUES:
            raise SkillUtilityError("invalid_outcome")
        return outcome_value

    def _validated_contract(
        self,
        stage: str,
        outcome_value: str | None,
        contract: ControlContract | None,
        difference: float | None,
        self_reported: bool,
    ) -> None:
        if stage != "objective_outcome":
            if contract is not None or difference is not None or self_reported:
                raise SkillUtilityError("invalid_field")
            return
        if contract is None and difference is not None:
            raise SkillUtilityError("invalid_control_contract")
        if contract is not None and difference is None:
            raise SkillUtilityError("invalid_control_contract")
        if contract is None or difference is None or outcome_value is None:
            return
        if outcome_value == "positive" and difference <= 0:
            raise SkillUtilityError("relative_difference_mismatch")
        if outcome_value == "negative" and difference >= 0:
            raise SkillUtilityError("relative_difference_mismatch")

    def _reject_duplicate_outcome(
        self, revision_id: str, invocation_id: str, outcome_id: str
    ) -> None:
        for other_id, other in self._invocations.items():
            if other_id == invocation_id or other.skill_revision_id != revision_id:
                continue
            applied = other.stages.get("applied")
            if applied is not None and applied.outcome_id == outcome_id:
                raise SkillUtilityError("duplicate_outcome")

    def _view_for(
        self,
        outcome_id: str,
        states: list[_Invocation],
        as_of: datetime | None,
        policy_version: str | None,
    ) -> EvidenceView | None:
        objectives = [
            state.stages["objective_outcome"]
            for state in states
            if "objective_outcome" in state.stages
        ]
        if not objectives:
            return None
        objectives.sort(key=lambda event: (event.occurred_at, event.event_id))
        primary = objectives[0]
        if as_of is not None and primary.occurred_at > as_of:
            return None
        revision_ids = tuple(sorted({state.skill_revision_id for state in states}))
        clusters = tuple(sorted({state.context_cluster for state in states}))
        invocation_ids = tuple(sorted(state.invocation_id for state in states))
        objective_states = [state for state in states if "objective_outcome" in state.stages]
        lineage = objective_states[0].task_lineage_id if len(objective_states) == 1 else None
        cluster = clusters[0] if len(clusters) == 1 else None
        reported = any(event.self_reported for event in objectives)
        has_contract = any(event.control_contract is not None for event in objectives)
        difference = primary.relative_difference
        if len(revision_ids) > 1:
            grade = GRADE_CONFOUNDED
            credited = False
            success = None
            outcome_value = None
            difference = None
        elif reported:
            grade = GRADE_SELF_REPORT
            credited = False
            success = primary.outcome_value == "positive"
            outcome_value = primary.outcome_value
        else:
            follow_ok = all(
                state.stages["followthrough"].followthrough == FOLLOWTHROUGH_OBSERVED
                for state in objective_states
            )
            grade = GRADE_COUNTERFACTUAL if has_contract else GRADE_OBSERVATIONAL
            credited = follow_ok
            success = primary.outcome_value == "positive"
            outcome_value = primary.outcome_value
        status = self._revalidation_status(
            outcome_id,
            primary.occurred_at,
            revision_ids,
            clusters,
        )
        weight = None
        if grade in _UTILITY_GRADES and as_of is not None:
            version = policy_version or primary.outcome_policy_version
            weight = evidence_weight(
                grade,
                as_of - primary.occurred_at,
                half_life_days=self._half_life(version, grade),
            )
        return EvidenceView(
            outcome_id=outcome_id,
            grade=grade,
            skill_revision_ids=revision_ids,
            invocation_ids=invocation_ids,
            task_lineage_id=lineage,
            context_cluster=cluster,
            context_clusters=clusters,
            success=success,
            utility_credited=credited,
            outcome_value=outcome_value,
            occurred_at=primary.occurred_at,
            outcome_policy_version=primary.outcome_policy_version,
            weight=weight,
            status=status,
            relative_difference=difference,
        )

    def _half_life(self, policy_version: str, grade: str) -> float:
        policy = self._require_policy(policy_version)
        if grade == GRADE_OBSERVATIONAL:
            return policy.observational_half_life_days
        if grade == GRADE_COUNTERFACTUAL:
            return policy.counterfactual_half_life_days
        raise SkillUtilityError("weight_undefined")

    def _revalidation_status(
        self,
        outcome_id: str,
        occurred_at: datetime,
        revision_ids: tuple[str, ...],
        clusters: tuple[str, ...],
    ) -> str:
        latest: datetime | None = None
        for drift in self._drifts:
            if occurred_at >= drift.occurred_at:
                continue
            if drift.skill_revision_id is not None and drift.skill_revision_id not in revision_ids:
                continue
            if drift.context_cluster is not None and drift.context_cluster not in clusters:
                continue
            if latest is None or drift.occurred_at > latest:
                latest = drift.occurred_at
        if latest is None:
            return STATUS_ACTIVE
        for record in self._revalidations:
            if record.outcome_id == outcome_id and record.occurred_at >= latest:
                return STATUS_ACTIVE
        return STATUS_NEEDS_REVALIDATION

    def _active_credited(
        self,
        revision_id: str,
        grade: str,
        as_of: datetime,
        policy_version: str | None,
    ) -> list[EvidenceView]:
        views: list[EvidenceView] = []
        for view in self.evidence_views(as_of=as_of, policy_version=policy_version):
            if (
                view.utility_credited
                and view.grade == grade
                and view.status == STATUS_ACTIVE
                and view.skill_revision_ids == (revision_id,)
            ):
                views.append(view)
        return views

    def _weighted(
        self,
        revision_id: str,
        grade: str,
        as_of: datetime,
        policy_version: str | None,
        *,
        cluster: str | None,
    ) -> tuple[float, float]:
        n = 0.0
        s = 0.0
        for view in self._active_credited(revision_id, grade, as_of, policy_version):
            if cluster is not None and view.context_cluster != cluster:
                continue
            weight = view.weight if view.weight is not None else 0.0
            n += weight
            if view.success:
                s += weight
        return n, s

    def _global_mean(
        self,
        revision_id: str,
        grade: str,
        as_of: datetime,
        policy_version: str | None,
        cluster: str,
    ) -> float:
        n = 0.0
        s = 0.0
        for view in self._active_credited(revision_id, grade, as_of, policy_version):
            if view.context_cluster == cluster:
                continue
            weight = view.weight if view.weight is not None else 0.0
            n += weight
            if view.success:
                s += weight
        if n <= 0:
            return NEUTRAL_PRIOR
        return s / n

    def _exploit_posterior(
        self,
        revision_id: str,
        cluster: str,
        as_of: datetime,
    ) -> LocalPosterior | None:
        chosen: LocalPosterior | None = None
        for grade in (GRADE_OBSERVATIONAL, GRADE_COUNTERFACTUAL):
            posterior = self.local_posterior(
                revision_id,
                cluster,
                grade=grade,
                as_of=as_of,
            )
            if posterior.utility_lcb is None:
                continue
            if chosen is None or posterior.utility_lcb < chosen.utility_lcb:
                chosen = posterior
        return chosen

    def _exploration_priority(self, revision_id: str, as_of: datetime) -> float:
        uncertainties: list[float] = []
        for grade in (GRADE_OBSERVATIONAL, GRADE_COUNTERFACTUAL):
            n, s = self._weighted(revision_id, grade, as_of, None, cluster=None)
            if n > 0:
                uncertainties.append(exploration_priority(n, s, NEUTRAL_PRIOR))
        if not uncertainties:
            return math.inf
        return max(uncertainties)

    def _attributable(self, revision_id: str, as_of: datetime | None) -> tuple[int, int]:
        lineages: set[str] = set()
        count = 0
        for view in self.evidence_views(as_of=as_of):
            if (
                view.utility_credited
                and view.grade in _UTILITY_GRADES
                and view.skill_revision_ids == (revision_id,)
                and view.task_lineage_id is not None
            ):
                count += 1
                lineages.add(view.task_lineage_id)
        return count, len(lineages)

    def _selection_counts(self) -> tuple[int, int]:
        explorations = sum(1 for item in self._selections if item.mode == MODE_EXPLORATION)
        exploitations = sum(1 for item in self._selections if item.mode == MODE_EXPLOITATION)
        return explorations, exploitations


def _coerce_contract(
    contract: ControlContract | Mapping[str, Any] | None,
) -> ControlContract | None:
    if contract is None:
        return None
    if isinstance(contract, ControlContract):
        return contract
    if isinstance(contract, Mapping):
        try:
            return ControlContract(
                anchor_state_snapshot=contract["anchor_state_snapshot"],
                task_input=contract["task_input"],
                agent_model_prompt_policy=contract["agent_model_prompt_policy"],
                tool_dependency_versions=contract["tool_dependency_versions"],
                permissions=contract["permissions"],
                budget=contract["budget"],
                evaluator=contract["evaluator"],
            )
        except KeyError as exc:
            raise SkillUtilityError("invalid_control_contract") from exc
    raise SkillUtilityError("invalid_control_contract")


def _coerce_difference(value: float | None) -> float | None:
    if value is None:
        return None
    return _require_finite(value, "invalid_control_contract")


def _freeze_payload(payload: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    if payload is None:
        return ()
    if not isinstance(payload, Mapping):
        raise SkillUtilityError("invalid_payload")
    items: list[tuple[str, str]] = []
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise SkillUtilityError("invalid_payload")
        items.append((key, value))
    return tuple(sorted(items))


def _eligible_set(eligible_set: Sequence[str], revision_id: str) -> tuple[str, ...]:
    if isinstance(eligible_set, (str, bytes)) or not isinstance(eligible_set, Sequence):
        raise SkillUtilityError("invalid_eligible_set")
    members = tuple(eligible_set)
    if (
        not members
        or any(not isinstance(item, str) or item == "" for item in members)
        or len(members) != len(set(members))
    ):
        raise SkillUtilityError("invalid_eligible_set")
    if revision_id not in members:
        raise SkillUtilityError("revision_not_eligible")
    return members


def _evidence_refs(evidence_refs: Sequence[str] | None) -> tuple[str, ...]:
    if evidence_refs is None:
        return ()
    if isinstance(evidence_refs, (str, bytes)) or not isinstance(evidence_refs, Sequence):
        raise SkillUtilityError("evidence_refs_required")
    refs = tuple(evidence_refs)
    if any(not isinstance(item, str) or item == "" for item in refs):
        raise SkillUtilityError("evidence_refs_required")
    return refs


def _event_snapshot(event: ChainEvent) -> dict[str, Any]:
    contract = event.control_contract
    return {
        "event_id": event.event_id,
        "invocation_id": event.invocation_id,
        "stage": event.stage,
        "skill_revision_id": event.skill_revision_id,
        "task_id": event.task_id,
        "run_id": event.run_id,
        "task_lineage_id": event.task_lineage_id,
        "context_cluster": event.context_cluster,
        "outcome_policy_version": event.outcome_policy_version,
        "occurred_at": event.occurred_at.isoformat().replace("+00:00", "Z"),
        "outcome_id": event.outcome_id,
        "outcome_value": event.outcome_value,
        "followthrough": event.followthrough,
        "relative_difference": event.relative_difference,
        "self_reported": event.self_reported,
        "control_contract": None
        if contract is None
        else {
            "anchor_state_snapshot": contract.anchor_state_snapshot,
            "task_input": contract.task_input,
            "agent_model_prompt_policy": contract.agent_model_prompt_policy,
            "tool_dependency_versions": [list(item) for item in contract.tool_dependency_versions],
            "permissions": contract.permissions,
            "budget": contract.budget,
            "evaluator": contract.evaluator,
        },
    }


def _copy_raw(item: dict[str, Any]) -> dict[str, Any]:
    contract = item["control_contract"]
    copied = dict(item)
    if contract is not None:
        copied["control_contract"] = {
            "anchor_state_snapshot": contract["anchor_state_snapshot"],
            "task_input": contract["task_input"],
            "agent_model_prompt_policy": contract["agent_model_prompt_policy"],
            "tool_dependency_versions": [
                list(pair) for pair in contract["tool_dependency_versions"]
            ],
            "permissions": contract["permissions"],
            "budget": contract["budget"],
            "evaluator": contract["evaluator"],
        }
    return copied
