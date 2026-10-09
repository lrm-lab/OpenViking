# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Skill revision utility ledger (Q126-A through skill-utility/Q133-A)."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from openviking.session.skill_revision_utility import (
    BAND_BASELINE,
    BAND_DEMOTED,
    CHAIN_STAGES,
    DEMOTE_LCB,
    DIRECT_QUARANTINE_ACTORS,
    GRADE_CONFOUNDED,
    GRADE_COUNTERFACTUAL,
    GRADE_OBSERVATIONAL,
    GRADE_SELF_REPORT,
    MIN_ATTRIBUTABLE_OUTCOMES,
    MIN_TASK_LINEAGES,
    PROPOSAL_ONLY_ACTORS,
    RESTORE_LCB,
    STATUS_ELIGIBLE,
    STATUS_NEEDS_REVALIDATION,
    STATUS_QUARANTINED,
    ControlContract,
    ExploitationCandidate,
    SkillUtilityError,
    SkillUtilityLedger,
    UtilityPolicy,
    evidence_weight,
    exploration_priority,
    projected_exploration_share,
    rank_exploitation,
    utility_lcb,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
POLICY = "skill-utility-v1"
CLUSTER = "cluster-a"


def _ledger() -> SkillUtilityLedger:
    return SkillUtilityLedger()


def _publish(ledger: SkillUtilityLedger, revision: str, at: datetime = T0) -> None:
    ledger.publish_active(revision, published_at=at)


def _contract() -> ControlContract:
    return ControlContract(
        anchor_state_snapshot="snap",
        task_input="task",
        agent_model_prompt_policy="model",
        tool_dependency_versions={"tool": "1"},
        permissions="read",
        budget="10",
        evaluator="eval",
    )


def _chain(
    ledger: SkillUtilityLedger,
    revision: str,
    invocation: str,
    at: datetime,
    *,
    lineage: str = "L1",
    cluster: str = CLUSTER,
    outcome: str = "positive",
    outcome_id: str | None = None,
    contract: ControlContract | None = None,
    relative: float | None = None,
    followthrough: str = "observed",
    self_reported: bool = False,
    policy: str = POLICY,
) -> str:
    outcome_key = outcome_id or f"out-{invocation}"
    common = {
        "skill_revision_id": revision,
        "task_id": f"task-{invocation}",
        "run_id": f"run-{invocation}",
        "task_lineage_id": lineage,
        "context_cluster": cluster,
        "outcome_policy_version": policy,
        "occurred_at": at,
    }
    for stage in CHAIN_STAGES:
        extra: dict[str, object] = {}
        if stage in ("applied", "followthrough", "objective_outcome"):
            extra["outcome_id"] = outcome_key
        if stage == "followthrough":
            extra["followthrough"] = followthrough
        if stage == "objective_outcome":
            extra["outcome_value"] = outcome
            extra["self_reported"] = self_reported
            if contract is not None:
                extra["control_contract"] = contract
                extra["relative_difference"] = relative
        ledger.append_event(invocation_id=invocation, stage=stage, **common, **extra)
    return outcome_key


def _literal_lcb(n: float, s: float, g: float) -> float:
    shrunk = (n * (s / n) + 2 * g) / (n + 2)
    se = math.sqrt(shrunk * (1 - shrunk) / n)
    return shrunk - 1.64 * se


def test_chain_is_append_only_and_rejects_skips() -> None:
    assert CHAIN_STAGES == (
        "eligible",
        "offered",
        "selected",
        "invoked",
        "applied",
        "followthrough",
        "objective_outcome",
    )
    ledger = _ledger()
    _publish(ledger, "rev")
    first = ledger.append_event(
        invocation_id="inv",
        stage="eligible",
        skill_revision_id="rev",
        task_id="task-inv",
        run_id="run-inv",
        task_lineage_id="L1",
        context_cluster=CLUSTER,
        outcome_policy_version=POLICY,
        occurred_at=T0,
    )
    with pytest.raises(SkillUtilityError, match="skipped_stage"):
        ledger.append_event(
            invocation_id="inv",
            stage="selected",
            skill_revision_id="rev",
            task_id="task-inv",
            run_id="run-inv",
            task_lineage_id="L1",
            context_cluster=CLUSTER,
            outcome_policy_version=POLICY,
            occurred_at=T0,
        )
    assert len(ledger.events()) == 1
    with pytest.raises(SkillUtilityError, match="skipped_stage"):
        ledger.append_event(
            invocation_id="other",
            stage="objective_outcome",
            skill_revision_id="rev",
            task_id="task-other",
            run_id="run-other",
            task_lineage_id="L1",
            context_cluster=CLUSTER,
            outcome_policy_version=POLICY,
            occurred_at=T0,
            outcome_id="out-other",
            outcome_value="positive",
        )
    ledger.append_event(
        invocation_id="inv",
        stage="offered",
        skill_revision_id="rev",
        task_id="task-inv",
        run_id="run-inv",
        task_lineage_id="L1",
        context_cluster=CLUSTER,
        outcome_policy_version=POLICY,
        occurred_at=T0,
    )
    with pytest.raises(SkillUtilityError, match="skipped_stage"):
        ledger.append_event(
            invocation_id="inv",
            stage="invoked",
            skill_revision_id="rev",
            task_id="task-inv",
            run_id="run-inv",
            task_lineage_id="L1",
            context_cluster=CLUSTER,
            outcome_policy_version=POLICY,
            occurred_at=T0,
        )
    with pytest.raises(SkillUtilityError, match="duplicate_stage"):
        ledger.append_event(
            invocation_id="inv",
            stage="offered",
            skill_revision_id="rev",
            task_id="task-inv",
            run_id="run-inv",
            task_lineage_id="L1",
            context_cluster=CLUSTER,
            outcome_policy_version=POLICY,
            occurred_at=T0,
        )
    with pytest.raises(SkillUtilityError, match="binding_mismatch"):
        ledger.append_event(
            invocation_id="inv",
            stage="selected",
            skill_revision_id="rev",
            task_id="other-task",
            run_id="run-inv",
            task_lineage_id="L1",
            context_cluster=CLUSTER,
            outcome_policy_version=POLICY,
            occurred_at=T0,
        )
    assert [event.stage for event in ledger.events()] == ["eligible", "offered"]
    with pytest.raises(FrozenInstanceError):
        first.stage = "applied"  # type: ignore[misc]
    copied = ledger.raw_events()
    copied[0]["stage"] = "mutated"
    assert ledger.raw_events()[0]["stage"] == "eligible"
    assert ledger.events()[0] is first


def test_offer_exposure_citation_and_self_report_do_not_count() -> None:
    ledger = _ledger()
    _publish(ledger, "rev")
    for kind in ("offer", "exposure", "citation", "self_report"):
        ledger.record_non_utility(
            kind=kind,
            skill_revision_id="rev",
            occurred_at=T0,
            payload={"outcome": "positive", "utility": "1"},
        )
    ledger.append_event(
        invocation_id="partial",
        stage="eligible",
        skill_revision_id="rev",
        task_id="task-partial",
        run_id="run-partial",
        task_lineage_id="L1",
        context_cluster=CLUSTER,
        outcome_policy_version=POLICY,
        occurred_at=T0,
    )
    ledger.append_event(
        invocation_id="partial",
        stage="offered",
        skill_revision_id="rev",
        task_id="task-partial",
        run_id="run-partial",
        task_lineage_id="L1",
        context_cluster=CLUSTER,
        outcome_policy_version=POLICY,
        occurred_at=T0,
    )
    _chain(ledger, "rev", "self", T0, self_reported=True)
    _chain(ledger, "rev", "partial-follow", T0, followthrough="partial")
    posterior = ledger.local_posterior("rev", CLUSTER, grade=GRADE_OBSERVATIONAL, as_of=T0)
    assert posterior.n == 0
    assert posterior.s == 0
    assert ledger.credited_positive_utility("rev") == 0
    assert ledger.revision("rev").attributable_outcomes == 0
    assert len(ledger.non_utility_events()) == 4
    self_view = ledger.evidence("out-self")
    assert self_view.grade == GRADE_SELF_REPORT
    assert self_view.utility_credited is False


def test_unique_applied_outcome_is_observational() -> None:
    ledger = _ledger()
    _publish(ledger, "rev")
    outcome_id = _chain(ledger, "rev", "inv-1", T0, lineage="L1")
    view = ledger.evidence(outcome_id)
    assert view.grade == GRADE_OBSERVATIONAL
    assert view.utility_credited is True
    assert view.success is True
    assert view.skill_revision_ids == ("rev",)
    assert ledger.credited_positive_utility("rev") == 1
    posterior = ledger.local_posterior("rev", CLUSTER, grade=GRADE_OBSERVATIONAL, as_of=T0)
    assert posterior.n == 1
    assert posterior.s == 1
    assert posterior.g == 0.5
    counterfactual = ledger.local_posterior("rev", CLUSTER, grade=GRADE_COUNTERFACTUAL, as_of=T0)
    assert counterfactual.n == 0
    objective = next(event for event in ledger.events() if event.stage == "objective_outcome")
    assert objective.invocation_id == "inv-1"
    assert objective.task_id == "task-inv-1"
    assert objective.run_id == "run-inv-1"
    assert objective.context_cluster == CLUSTER
    assert objective.task_lineage_id == "L1"
    assert objective.outcome_policy_version == POLICY


def test_control_contract_is_counterfactual_and_grades_stay_separate() -> None:
    ledger = _ledger()
    _publish(ledger, "rev")
    observational = _chain(ledger, "rev", "obs", T0, outcome_id="obs")
    counterfactual = _chain(
        ledger,
        "rev",
        "cf",
        T0,
        outcome_id="cf",
        contract=_contract(),
        relative=0.4,
    )
    assert ledger.evidence(observational).grade == GRADE_OBSERVATIONAL
    assert ledger.evidence(counterfactual).grade == GRADE_COUNTERFACTUAL
    assert ledger.evidence(counterfactual).relative_difference == 0.4
    obs = ledger.local_posterior("rev", CLUSTER, grade=GRADE_OBSERVATIONAL, as_of=T0)
    cf = ledger.local_posterior("rev", CLUSTER, grade=GRADE_COUNTERFACTUAL, as_of=T0)
    assert obs.n == 1 and obs.s == 1
    assert cf.n == 1 and cf.s == 1
    assert ledger.credited_positive_utility("rev") == 2


def test_multiple_applied_revisions_are_confounded_set() -> None:
    ledger = _ledger()
    _publish(ledger, "rev-a")
    _publish(ledger, "rev-b")
    _chain(ledger, "rev-a", "a1", T0, outcome_id="shared")
    before = ledger.raw_events()
    assert ledger.evidence("shared").grade == GRADE_OBSERVATIONAL
    assert ledger.credited_positive_utility("rev-a") == 1
    _chain(ledger, "rev-b", "b1", T0, outcome_id="shared")
    view = ledger.evidence("shared")
    assert view.grade == GRADE_CONFOUNDED
    assert view.utility_credited is False
    assert set(view.skill_revision_ids) == {"rev-a", "rev-b"}
    assert set(view.invocation_ids) == {"a1", "b1"}
    assert ledger.credited_positive_utility("rev-a") == 0
    assert ledger.credited_positive_utility("rev-b") == 0
    assert ledger.local_posterior("rev-a", CLUSTER, grade=GRADE_OBSERVATIONAL, as_of=T0).n == 0
    assert ledger.local_posterior("rev-b", CLUSTER, grade=GRADE_OBSERVATIONAL, as_of=T0).n == 0
    assert [
        event["outcome_value"]
        for event in ledger.raw_events()
        if event["stage"] == "objective_outcome"
    ] == ["positive", "positive"]
    assert ledger.raw_events()[: len(before)] == before


def test_decay_projects_half_life_without_rewriting_events() -> None:
    ledger = _ledger()
    _publish(ledger, "rev")
    observational = _chain(ledger, "rev", "obs", T0, outcome_id="obs")
    counterfactual = _chain(
        ledger,
        "rev",
        "cf",
        T0,
        outcome_id="cf",
        contract=_contract(),
        relative=0.2,
    )
    objective = next(
        event
        for event in ledger.events()
        if event.stage == "objective_outcome" and event.outcome_id == observational
    )
    before = ledger.raw_events()
    later = T0 + timedelta(days=30)
    assert evidence_weight(GRADE_OBSERVATIONAL, timedelta(days=30)) == 0.5
    assert evidence_weight(GRADE_COUNTERFACTUAL, timedelta(days=180)) == 0.5
    assert ledger.projected_weight(observational, as_of=later) == 0.5
    assert ledger.projected_weight(counterfactual, as_of=T0 + timedelta(days=180)) == 0.5
    assert ledger.projected_weight(counterfactual, as_of=later) == pytest.approx(2 ** (-30 / 180))
    aged = ledger.local_posterior("rev", CLUSTER, grade=GRADE_OBSERVATIONAL, as_of=later)
    assert aged.n == 0.5
    assert aged.s == 0.5
    assert aged.utility_lcb == pytest.approx(utility_lcb(0.5, 0.5, 0.5))
    ledger.register_policy(
        UtilityPolicy(version="skill-utility-v2", observational_half_life_days=10)
    )
    assert (
        ledger.projected_weight(
            observational,
            as_of=later,
            policy_version="skill-utility-v2",
        )
        == 0.125
    )
    assert ledger.projected_weight(observational, as_of=later) == 0.5
    assert ledger.raw_events() == before
    assert ledger.events()[ledger.events().index(objective)] is objective
    assert objective.outcome_value == "positive"
    assert objective.occurred_at == T0
    assert objective.outcome_policy_version == POLICY
    with pytest.raises(SkillUtilityError, match="policy_frozen"):
        ledger.register_policy(
            UtilityPolicy(version="skill-utility-v2", observational_half_life_days=12)
        )


def test_drift_needs_revalidation_until_revalidated() -> None:
    ledger = _ledger()
    _publish(ledger, "rev")
    outcome_ids = [
        _chain(ledger, "rev", f"d-{index}", T0, lineage=f"L{index % 3}") for index in range(5)
    ]
    before = ledger.raw_events()
    assert [row.skill_revision_id for row in ledger.exploitation_ranking(CLUSTER, as_of=T0)] == [
        "rev"
    ]
    ledger.declare_drift(kind="schema", occurred_at=T0 + timedelta(days=1), skill_revision_id="rev")
    view = ledger.evidence(outcome_ids[0], as_of=T0 + timedelta(days=30))
    assert view.status == STATUS_NEEDS_REVALIDATION
    assert view.outcome_value == "positive"
    assert ledger.projected_weight(outcome_ids[0], as_of=T0 + timedelta(days=30)) == 0.5
    assert ledger.exploitation_ranking(CLUSTER, as_of=T0 + timedelta(days=1)) == ()
    for outcome_id in outcome_ids:
        ledger.revalidate_evidence(outcome_id, occurred_at=T0 + timedelta(days=2), note="rechecked")
    assert ledger.evidence(outcome_ids[0], as_of=T0 + timedelta(days=2)).status == "active"
    assert [
        row.skill_revision_id
        for row in ledger.exploitation_ranking(CLUSTER, as_of=T0 + timedelta(days=2))
    ] == ["rev"]
    ledger.declare_drift(kind="model", occurred_at=T0 + timedelta(days=3), skill_revision_id="rev")
    assert ledger.evidence(outcome_ids[0]).status == STATUS_NEEDS_REVALIDATION
    assert ledger.exploitation_ranking(CLUSTER, as_of=T0 + timedelta(days=3)) == ()
    assert ledger.raw_events() == before
    assert ledger.is_online_eligible("rev", as_of=T0 + timedelta(days=3650))
    assert ledger.revision("rev").ttl_days is None


def test_revision_has_no_ttl_and_cold_start_gate() -> None:
    ledger = _ledger()
    _publish(ledger, "fresh")
    view = ledger.revision("fresh")
    assert view.ttl_days is None
    assert view.expires_at is None
    assert view.online_status == STATUS_ELIGIBLE
    assert ledger.is_online_eligible("fresh", as_of=T0 + timedelta(days=3650))
    assert ledger.may_exploit("fresh") is False
    ledger.record_non_utility(
        kind="exposure", skill_revision_id="fresh", occurred_at=T0, payload={"utility": "1"}
    )
    ledger.record_non_utility(kind="citation", skill_revision_id="fresh", occurred_at=T0)
    assert ledger.may_exploit("fresh") is False

    _publish(ledger, "one-lineage")
    for index in range(5):
        _chain(ledger, "one-lineage", f"one-{index}", T0, lineage="only")
    assert ledger.revision("one-lineage").attributable_outcomes == 5
    assert ledger.revision("one-lineage").task_lineages == 1
    assert ledger.may_exploit("one-lineage") is False

    _publish(ledger, "gated")
    for index in range(4):
        _chain(ledger, "gated", f"g-{index}", T0, lineage=f"L{index % 3}")
    assert ledger.revision("gated").attributable_outcomes == 4
    assert ledger.revision("gated").task_lineages == 3
    assert ledger.may_exploit("gated") is False
    _chain(ledger, "gated", "g-4", T0, lineage="L1")
    assert ledger.revision("gated").attributable_outcomes == MIN_ATTRIBUTABLE_OUTCOMES
    assert ledger.revision("gated").task_lineages == MIN_TASK_LINEAGES
    assert ledger.may_exploit("gated") is True

    _publish(ledger, "confounded")
    _publish(ledger, "partner")
    for index in range(5):
        _chain(
            ledger,
            "confounded",
            f"c-{index}",
            T0,
            lineage=f"L{index % 3}",
            outcome_id=f"mix-{index}",
        )
        _chain(
            ledger, "partner", f"p-{index}", T0, lineage=f"L{index % 3}", outcome_id=f"mix-{index}"
        )
    assert ledger.may_exploit("confounded") is False
    assert ledger.may_exploit("partner") is False
    assert ledger.credited_positive_utility("confounded") == 0
    assert ledger.credited_positive_utility("partner") == 0


def test_utility_lcb_formula_and_rank_not_mean() -> None:
    n, s, g = 4, 4, 0.5
    shrunk = (n * (s / n) + 2 * g) / (n + 2)
    se = math.sqrt(shrunk * (1 - shrunk) / n)
    expected = shrunk - 1.64 * se
    assert utility_lcb(n, s, g) == pytest.approx(expected)
    assert utility_lcb(4, 4, 0.5) == pytest.approx(0.527737, abs=1e-4)

    n_many, s_many, g_many = 40, 28, 0.5
    shrunk_many = (n_many * (s_many / n_many) + 2 * g_many) / (n_many + 2)
    se_many = math.sqrt(shrunk_many * (1 - shrunk_many) / n_many)
    many_lcb = shrunk_many - 1.64 * se_many
    assert utility_lcb(40, 28, 0.5) == pytest.approx(many_lcb)
    assert many_lcb > expected
    assert (s / n) > (s_many / n_many)

    ranked = rank_exploitation(
        [
            ExploitationCandidate("few", 4, 4, 0.5, uncertainty_bonus=50),
            ExploitationCandidate("many", 40, 28, 0.5, uncertainty_bonus=0),
        ]
    )
    assert [row.skill_revision_id for row in ranked] == ["many", "few"]
    assert ranked[0].utility_lcb == pytest.approx(many_lcb)
    assert ranked[1].utility_lcb == pytest.approx(0.527737, abs=1e-4)
    assert ranked[1].posterior_mean > ranked[0].posterior_mean
    assert ranked[1].uncertainty > ranked[0].uncertainty


def test_high_uncertainty_does_not_raise_exploit_rank() -> None:
    def standard_error(n: float, s: float, g: float) -> float:
        shrunk = (n * (s / n) + 2 * g) / (n + 2)
        return math.sqrt(shrunk * (1 - shrunk) / n)

    assert exploration_priority(4, 2, 0.5) == pytest.approx(standard_error(4, 2, 0.5))
    assert standard_error(4, 2, 0.5) > standard_error(40, 36, 0.5)
    assert (2 / 4) < (36 / 40)
    ranked = rank_exploitation(
        [
            ExploitationCandidate("noisy", 4, 2, 0.5, uncertainty_bonus=100),
            ExploitationCandidate("sure", 40, 36, 0.5),
        ]
    )
    assert [row.skill_revision_id for row in ranked] == ["sure", "noisy"]
    assert ranked[0].utility_lcb > ranked[1].utility_lcb


def test_ledger_ranks_exploitation_by_lcb() -> None:
    ledger = _ledger()
    _publish(ledger, "few")
    _publish(ledger, "many")
    for index in range(4):
        _chain(
            ledger,
            "few",
            f"few-a-{index}",
            T0,
            lineage="L0" if index % 2 == 0 else "L1",
            cluster=CLUSTER,
        )
    _chain(ledger, "few", "few-b-0", T0, lineage="L2", cluster="cluster-b", outcome="positive")
    _chain(ledger, "few", "few-b-1", T0, lineage="L2", cluster="cluster-b", outcome="negative")
    for index in range(40):
        _chain(
            ledger,
            "many",
            f"many-a-{index}",
            T0,
            lineage=f"L{index % 3}",
            cluster=CLUSTER,
            outcome="positive" if index < 28 else "negative",
        )
    _chain(ledger, "many", "many-b-0", T0, lineage="L0", cluster="cluster-b", outcome="positive")
    _chain(ledger, "many", "many-b-1", T0, lineage="L1", cluster="cluster-b", outcome="negative")

    assert ledger.may_exploit("few") is True
    assert ledger.may_exploit("many") is True
    ranking = ledger.exploitation_ranking(CLUSTER, as_of=T0)
    assert [row.skill_revision_id for row in ranking] == ["many", "few"]
    few = ranking[1]
    many = ranking[0]
    assert few.n == 4
    assert few.s == 4
    assert few.g == 0.5
    assert few.posterior_mean == 1
    assert many.n == 40
    assert many.s == 28
    assert many.g == 0.5
    assert many.posterior_mean == pytest.approx(0.7)
    assert few.posterior_mean > many.posterior_mean
    assert few.utility_lcb == pytest.approx(0.527737, abs=1e-4)
    assert few.utility_lcb == pytest.approx(_literal_lcb(4, 4, 0.5))
    assert many.utility_lcb == pytest.approx(_literal_lcb(40, 28, 0.5))
    assert many.utility_lcb > few.utility_lcb
    assert few.uncertainty > many.uncertainty

    lone = _ledger()
    _publish(lone, "lone")
    for index in range(4):
        _chain(lone, "lone", f"lone-{index}", T0)
    posterior = lone.local_posterior("lone", CLUSTER, grade=GRADE_OBSERVATIONAL, as_of=T0)
    assert posterior.n == 4
    assert posterior.s == 4
    assert posterior.g == 0.5
    assert posterior.utility_lcb == pytest.approx(0.527737, abs=1e-4)
    assert lone.may_exploit("lone") is False
    assert lone.exploitation_ranking(CLUSTER, as_of=T0) == ()


def test_exploration_share_cap_and_cold_revision_priority() -> None:
    ledger = _ledger()
    _publish(ledger, "mature")
    _publish(ledger, "cold")
    for index in range(5):
        _chain(ledger, "mature", f"m-{index}", T0, lineage=f"L{index % 3}")
    for index in range(9):
        ledger.commit_selection(
            mode="exploitation",
            invocation_id=f"exploit-{index}",
            skill_revision_id="mature",
            eligible_set=("mature", "cold"),
            seed="seed",
            selection_propensity=0.5,
            invocation_cap=20,
            occurred_at=T0,
        )
    quota = ledger.exploration_quota()
    assert quota.exploitation_count == 9
    assert quota.exploration_count == 0
    assert quota.next_exploration_share == (0 + 1) / (9 + 1)
    assert quota.next_exploration_share == pytest.approx(0.10)
    assert projected_exploration_share(0, 9) == (0 + 1) / (9 + 1)
    assert quota.next_exploration_allowed is True
    assert quota.share_cap == 0.10
    assert quota.min_attributable_outcomes == 5
    assert quota.min_task_lineages == 3
    assert quota.provisional_memory_exploitation_slots == 3
    assert quota.provisional_memory_exploration_slots == 1
    assert quota.consumes_provisional_memory_quota is False

    audit = ledger.commit_selection(
        mode="exploration",
        invocation_id="explore-1",
        skill_revision_id="cold",
        eligible_set=("mature", "cold"),
        seed="seed-explore",
        selection_propensity=0.1,
        invocation_cap=4,
        occurred_at=T0,
    )
    assert audit.mode == "exploration"
    assert audit.seed == "seed-explore"
    assert audit.eligible_set == ("mature", "cold")
    assert audit.selection_propensity == 0.1
    assert audit.invocation_cap == 4
    assert audit.projected_share == pytest.approx(0.10)
    quota = ledger.exploration_quota()
    assert quota.exploration_count == 1
    assert quota.exploitation_count == 9
    assert quota.next_exploration_share == (1 + 1) / (10 + 1)
    assert projected_exploration_share(1, 9) == (1 + 1) / (10 + 1)
    assert quota.next_exploration_share > 0.10
    assert quota.next_exploration_allowed is False
    with pytest.raises(SkillUtilityError, match="exploration_share_exceeded"):
        ledger.commit_selection(
            mode="exploration",
            invocation_id="explore-2",
            skill_revision_id="cold",
            eligible_set=("mature", "cold"),
            seed="seed-explore",
            selection_propensity=0.1,
            invocation_cap=4,
            occurred_at=T0,
        )
    assert ledger.exploration_quota().exploration_count == 1
    with pytest.raises(SkillUtilityError, match="exploitation_not_allowed"):
        ledger.commit_selection(
            mode="exploitation",
            invocation_id="exploit-cold",
            skill_revision_id="cold",
            eligible_set=("mature", "cold"),
            seed="seed",
            selection_propensity=0.5,
            invocation_cap=20,
            occurred_at=T0,
        )
    assert ledger.exploration_quota().exploitation_count == 9

    explore = ledger.exploration_candidates(as_of=T0)
    assert [row.skill_revision_id for row in explore] == ["cold", "mature"]
    assert explore[0].may_exploit is False
    assert explore[0].priority == math.inf
    assert math.isfinite(explore[1].priority)
    ranking = ledger.exploitation_ranking(CLUSTER, as_of=T0)
    assert [row.skill_revision_id for row in ranking] == ["mature"]


def test_actor_sets_are_disjoint() -> None:
    assert DIRECT_QUARANTINE_ACTORS == frozenset(
        {"server_verifier", "security_policy", "authoritative_revocation", "admin"}
    )
    assert PROPOSAL_ONLY_ACTORS == frozenset(
        {"memory", "diagnosis", "builder", "critic", "refiner", "reader", "task_agent"}
    )
    assert DIRECT_QUARANTINE_ACTORS.isdisjoint(PROPOSAL_ONLY_ACTORS)


@pytest.mark.parametrize("actor", sorted(PROPOSAL_ONLY_ACTORS))
def test_proposal_actor_cannot_change_online_status(actor: str) -> None:
    ledger = _ledger()
    _publish(ledger, "rev")
    with pytest.raises(SkillUtilityError, match="evidence_refs_required"):
        ledger.submit_quarantine(
            actor=actor,
            skill_revision_id="rev",
            reason="severe_security",
            occurred_at=T0,
            idempotency_key=f"{actor}-empty",
            evidence_refs=(),
        )
    assert ledger.online_status("rev") == STATUS_ELIGIBLE
    decision = ledger.submit_quarantine(
        actor=actor,
        skill_revision_id="rev",
        reason="severe_security",
        occurred_at=T0,
        idempotency_key=f"{actor}-proposal",
        evidence_refs=("evidence-1",),
    )
    assert decision.action == "proposed"
    assert decision.changed_online_status is False
    assert decision.online_status == STATUS_ELIGIBLE
    assert ledger.is_online_eligible("rev") is True
    assert ledger.quarantine_transitions("rev") == ()
    assert len(ledger.quarantine_proposals("rev")) == 1
    again = ledger.submit_quarantine(
        actor=actor,
        skill_revision_id="rev",
        reason="severe_security",
        occurred_at=T0,
        idempotency_key=f"{actor}-proposal",
        evidence_refs=("evidence-1",),
    )
    assert again == decision
    assert len(ledger.quarantine_proposals("rev")) == 1


@pytest.mark.parametrize("actor", sorted(DIRECT_QUARANTINE_ACTORS))
def test_direct_actor_quarantines(actor: str) -> None:
    ledger = _ledger()
    _publish(ledger, "rev")
    for index in range(5):
        _chain(ledger, "rev", f"q-{index}", T0, lineage=f"L{index % 3}")
    assert ledger.may_exploit("rev") is True
    decision = ledger.submit_quarantine(
        actor=actor,
        skill_revision_id="rev",
        reason="severe_security",
        occurred_at=T0,
        idempotency_key=f"{actor}-quarantine",
        evidence_refs=(),
    )
    assert decision.action == "quarantined"
    assert decision.changed_online_status is True
    assert ledger.online_status("rev") == STATUS_QUARANTINED
    assert ledger.is_online_eligible("rev") is False
    assert ledger.may_exploit("rev") is False
    again = ledger.submit_quarantine(
        actor=actor,
        skill_revision_id="rev",
        reason="severe_security",
        occurred_at=T0,
        idempotency_key=f"{actor}-quarantine",
        evidence_refs=(),
    )
    assert again == decision
    transitions = ledger.quarantine_transitions("rev")
    assert len(transitions) == 1
    assert transitions[0].expires_at is None
    assert transitions[0].severe is True


def test_negative_utility_does_not_quarantine() -> None:
    ledger = _ledger()
    _publish(ledger, "rev")
    times = [T0 + timedelta(seconds=index) for index in range(8)]
    for index, at in enumerate(times):
        _chain(ledger, "rev", f"neg-{index}", at, outcome="negative")
    assert ledger.effect_band("rev", CLUSTER, as_of=times[3]) == BAND_BASELINE
    assert ledger.effect_band("rev", CLUSTER, as_of=times[4]) == BAND_DEMOTED
    assert ledger.online_status("rev") == STATUS_ELIGIBLE
    assert ledger.is_online_eligible("rev") is True
    assert ledger.quarantine_transitions("rev") == ()
    assert ledger.quarantine_proposals("rev") == ()
    posterior = ledger.local_posterior("rev", CLUSTER, grade=GRADE_OBSERVATIONAL, as_of=times[-1])
    assert posterior.n == pytest.approx(8, abs=1e-3)
    assert posterior.s == 0
    assert ledger.revision("rev", as_of=times[-1]).attributable_outcomes == 8


def test_severe_security_is_not_washed_by_time() -> None:
    ledger = _ledger()
    _publish(ledger, "rev")
    _chain(ledger, "rev", "before", T0)
    ledger.submit_quarantine(
        actor="security_policy",
        skill_revision_id="rev",
        reason="severe_security",
        occurred_at=T0 + timedelta(hours=1),
        idempotency_key="severe",
        evidence_refs=("sec-1",),
    )
    _chain(ledger, "rev", "after", T0 + timedelta(hours=2), outcome="positive")
    far = T0 + timedelta(days=3650)
    assert ledger.online_status("rev", as_of=far) == STATUS_QUARANTINED
    assert ledger.is_online_eligible("rev", as_of=far) is False
    transition = ledger.quarantine_transitions("rev")[0]
    assert transition.expires_at is None
    assert transition.severe is True
    with pytest.raises(SkillUtilityError, match="actor_not_authorized"):
        ledger.release_quarantine(
            actor="task_agent",
            skill_revision_id="rev",
            reason="please",
            occurred_at=far,
            idempotency_key="release-agent",
            evidence_refs=("sec-1",),
        )
    assert ledger.is_online_eligible("rev", as_of=far) is False

    _publish(ledger, "other")
    ledger.submit_quarantine(
        actor="admin",
        skill_revision_id="other",
        reason="severe_security",
        occurred_at=T0,
        idempotency_key="other-q",
    )
    released = ledger.release_quarantine(
        actor="admin",
        skill_revision_id="other",
        reason="revalidated",
        occurred_at=T0 + timedelta(days=1),
        idempotency_key="other-release",
        evidence_refs=("review-1",),
    )
    assert released.action == "released"
    assert ledger.is_online_eligible("other", as_of=T0 + timedelta(days=400)) is True
    assert ledger.is_online_eligible("rev", as_of=far) is False


def test_hysteresis_uses_asymmetric_thresholds() -> None:
    assert RESTORE_LCB > DEMOTE_LCB
    assert utility_lcb(6, 5, 0.5) > DEMOTE_LCB
    assert utility_lcb(6, 5, 0.5) < RESTORE_LCB
    assert utility_lcb(7, 5, 0.5) < DEMOTE_LCB
    assert utility_lcb(15, 10, 0.5) > DEMOTE_LCB
    assert utility_lcb(15, 10, 0.5) < RESTORE_LCB
    assert utility_lcb(23, 18, 0.5) > RESTORE_LCB

    ledger = _ledger()
    _publish(ledger, "down")
    moment = T0
    for index in range(5):
        moment = T0 + timedelta(seconds=index)
        _chain(ledger, "down", f"down-s-{index}", moment, outcome="positive")
    assert ledger.effect_band("down", CLUSTER, as_of=moment) == BAND_BASELINE
    moment = T0 + timedelta(seconds=5)
    _chain(ledger, "down", "down-f-0", moment, outcome="negative")
    assert ledger.effect_band("down", CLUSTER, as_of=moment) == BAND_BASELINE
    moment = T0 + timedelta(seconds=6)
    _chain(ledger, "down", "down-f-1", moment, outcome="negative")
    assert ledger.effect_band("down", CLUSTER, as_of=moment) == BAND_DEMOTED
    assert ledger.online_status("down") == STATUS_ELIGIBLE

    _publish(ledger, "up")
    for index in range(5):
        moment = T0 + timedelta(seconds=index)
        _chain(ledger, "up", f"up-f-{index}", moment, outcome="negative")
    assert ledger.effect_band("up", CLUSTER, as_of=moment) == BAND_DEMOTED
    for index in range(10):
        moment = T0 + timedelta(seconds=5 + index)
        _chain(ledger, "up", f"up-s-{index}", moment, outcome="positive")
    assert ledger.effect_band("up", CLUSTER, as_of=moment) == BAND_DEMOTED
    for index in range(8):
        moment = T0 + timedelta(seconds=15 + index)
        _chain(ledger, "up", f"up-s2-{index}", moment, outcome="positive")
    assert ledger.effect_band("up", CLUSTER, as_of=moment) == BAND_BASELINE
    assert ledger.quarantine_transitions("up") == ()
    assert ledger.quarantine_transitions("down") == ()


def test_shrinks_toward_other_context_mean() -> None:
    ledger = _ledger()
    _publish(ledger, "rev")
    for index in range(4):
        _chain(ledger, "rev", f"east-{index}", T0, cluster="east", outcome="positive")
        _chain(ledger, "rev", f"west-{index}", T0, cluster="west", outcome="negative")
    east = ledger.local_posterior("rev", "east", grade=GRADE_OBSERVATIONAL, as_of=T0)
    west = ledger.local_posterior("rev", "west", grade=GRADE_OBSERVATIONAL, as_of=T0)
    assert east.n == 4 and east.s == 4 and east.g == 0
    assert east.utility_lcb == pytest.approx(utility_lcb(4, 4, 0))
    assert west.n == 4 and west.s == 0 and west.g == 1
    assert west.utility_lcb == pytest.approx(utility_lcb(4, 0, 1))
    assert west.utility_lcb is not None and west.g is not None
    assert west.utility_lcb < west.g


def test_unknown_actor_is_rejected() -> None:
    ledger = _ledger()
    _publish(ledger, "rev")
    with pytest.raises(SkillUtilityError, match="actor_not_authorized"):
        ledger.submit_quarantine(
            actor="user",
            skill_revision_id="rev",
            reason="severe_security",
            occurred_at=T0,
            idempotency_key="user",
            evidence_refs=("evidence-1",),
        )
    assert ledger.online_status("rev") == STATUS_ELIGIBLE
