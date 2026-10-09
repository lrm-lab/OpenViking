# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""One observable path across the Q109–Q160 slices.

Recorded run, from ``OpenViking``:

``python -m pytest tests/test_retrieval_acceptance.py tests/test_retrieval_control.py tests/test_served_hint_offer.py tests/test_served_anchor_expansion.py tests/test_memory_discovery_allocation.py tests/test_corpus_hybrid_fusion.py tests/test_exact_lane.py tests/test_memory_role_retrieval.py tests/test_aig_role_chain.py tests/test_influence_projection.py tests/test_subgraph_retrieval.py -q --no-cov``

Result: 329 passed. This does not measure whether memory improves task success.
The hybrid facade still scores with ``hybrid-fusion-pending-v0``; corpus RRF
stays in its own module until that pending choice is replaced. Hint, offer,
allocation, expansion, and the control tree are composed here; they are not
yet called from ``RoleRetrievalService`` or ``HybridRetrievalFacade``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from openviking.session.aig_role_chain import AigConstructionRun
from openviking.session.corpus_hybrid_fusion import fuse_corpora
from openviking.session.influence_projection import ProjectionRegistry
from openviking.session.influence_view import (
    AORange,
    CoverageManifest,
    InfluenceBlock,
    InfluenceClaim,
    InfluenceViewDraft,
    PURPOSE_FAILURE_DIAGNOSIS,
    PURPOSE_POST_RUN_INDEX,
    RetrievalSubgraph,
    SegmentRef,
    SourceRef,
    build_minimal_fallback_view,
)
from openviking.session.memory_discovery_allocation import allocate_discovery
from openviking.session.memory_role_retrieval import (
    EXACT_CAPABILITY_ABSENT,
    BoundContext,
    RetrievalContractError,
    RoleRetrievalService,
)
from openviking.session.retrieval_control import (
    DISCOVERY_LANE,
    EXACT_LANE,
    ControlPolicy,
    RetrievalControl,
)
from openviking.session.retrieval_facade import (
    HYBRID_FUSION_POLICY_VERSION,
    SUBGRAPH_TEXT_POLICY_VERSION,
    _subgraph_retrieval_text,
)
from openviking.session.served_anchor_expansion import ExpansionChain, ExpansionError, freeze_direct
from openviking.session.served_hint_offer import HintError, ServedDeliveryLedger
from openviking.session.skill_revision_utility import SkillUtilityLedger

_FROZEN = "2026-01-02T00:00:00.000Z"
_PLAN = "a" * 64


def _clock() -> datetime:
    return datetime(2026, 1, 2, tzinfo=timezone.utc)


def _fallback(session_id: str, revision_id: str, purpose: str = PURPOSE_POST_RUN_INDEX):
    return build_minimal_fallback_view(
        {
            "session_id": session_id,
            "task_run_id": f"task-{revision_id}",
            "purpose": purpose,
            "revision_id": revision_id,
            "view_id": f"view-{revision_id}",
            "frozen_at": _FROZEN,
            "ao_records": [
                {
                    "ao_id": f"ao-{revision_id}",
                    "sequence": 1,
                    "participant_id": "agent-a",
                    "segment_id": "seg-a",
                    "session_id": session_id,
                    "content_hash": f"hash-{revision_id}",
                }
            ],
        }
    )


def _block(session_id: str, block_id: str, sequence: int) -> InfluenceBlock:
    ref = SourceRef(
        ao_id=f"ao-{session_id}-{sequence}",
        sequence=sequence,
        participant_id="agent-a",
        segment_id="seg-a",
        session_id=session_id,
        content_hash=f"hash-{session_id}-{sequence}",
    )
    return InfluenceBlock(
        block_id=block_id,
        role="agent_action",
        participant_id="agent-a",
        segment_ref=SegmentRef(segment_id="seg-a", session_id=session_id),
        ao_start_seq=sequence,
        ao_end_seq=sequence,
        summary="deploy switch traffic",
        input="",
        output="",
        authorship="agent-a",
        granularity_reason="one tool call",
        source_content_hashes=(ref.content_hash,),
        source_refs=(ref,),
    )


def _revision(session_id: str, revision_id: str, relation: str, failure: bool):
    blocks = [_block(session_id, "b-deploy", 1), _block(session_id, "b-inspect", 2)]
    refs = tuple(ref for block in blocks for ref in block.source_refs)
    coverage = CoverageManifest(
        total_ao_range=AORange(start_seq=1, end_seq=2),
        inspected_refs=refs,
        included_ranges=(
            AORange(start_seq=1, end_seq=1, segment_id="seg-a"),
            AORange(start_seq=2, end_seq=2, segment_id="seg-a"),
        ),
        omitted_ranges=(),
        snapshot_watermark="wm-1",
    )
    claim = InfluenceClaim(
        claim_id="c-link",
        source_block_id="b-deploy",
        target_block_id="b-inspect",
        carried_artifact="packet shift",
        downstream_effect="status climbed",
        relation_type=relation,
    )
    subgraph = RetrievalSubgraph(
        subgraph_id="sg-cutover",
        block_ids=("b-deploy", "b-inspect"),
        claim_ids=("c-link",),
        failure_marked=failure,
    )
    draft = InfluenceViewDraft(
        view_id=f"view-{revision_id}",
        purpose=PURPOSE_POST_RUN_INDEX,
        session_id=session_id,
        task_run_id=f"task-{revision_id}",
        coverage=coverage,
        blocks=blocks,
        claims=(claim,),
        retrieval_subgraphs=(subgraph,),
    )
    return draft.freeze(revision_id=revision_id, frozen_at=_FROZEN, enforce_checks=False)


def test_accepted_retrieval_aig_and_skill_path(tmp_path) -> None:
    run = AigConstructionRun("run-accept")
    assert run.session_for("diagnosis") == run.session_for("reader")
    run.submit_draft(run.session_for("builder"), {"body": "frozen draft"})
    run.critique(run.session_for("critic"), [])
    report_hash = run.read_verify(run.session_for("reader"), {"phase": "READ_VERIFY"})
    assert run.freeze_report(run.session_for("reader")) == report_hash
    assert run.phase == "SYNTHESIZE_PROPOSE"
    assert run.trust == "frozen"

    notified: list[str] = []
    registry = ProjectionRegistry(
        tmp_path / "projections.json",
        clock=_clock,
        revalidation_callback=notified.append,
    )
    registry.register(_fallback("sess-1", "rev-1"), acl_labels=("team-a",))
    diagnosis = _fallback("sess-1", "rev-diag", PURPOSE_FAILURE_DIAGNOSIS)
    registry.register(diagnosis, acl_labels=("team-a",))
    registry.register(_fallback("sess-1", "rev-2"), acl_labels=("team-a",))
    visible = {card.source_pointer.view_revision_id for card in registry.search(principal_labels=("team-a",))}
    assert "rev-1" not in visible
    assert "rev-2" in visible
    assert "rev-diag" in visible
    registry.handle_acl_change("sess-1", ("team-b",))
    assert {"rev-2", "rev-diag"} <= set(notified)
    assert len(notified) == len(set(notified))
    assert registry.search(principal_labels=("team-a",)) == ()

    service = RoleRetrievalService()
    discovered: list[tuple[object, object]] = []

    def authority(_role: str) -> BoundContext:
        return BoundContext("scope", "snap", "lease")

    def discover(intent, pattern):
        discovered.append((intent, pattern))
        return [
            {"id": "skill-1", "kind": "skill_revision"},
            {"id": "prior-1", "kind": "block_projection"},
            {"id": "ep-1", "kind": "episode"},
        ]

    def search(_clause, _bound):
        return [
            {
                "sourceSessionRef": "sess-1",
                "aoId": "ao-1",
                "matchedField": "message",
                "sequence": 1,
                "text": "SECRET-HIT",
                "visible": True,
                "inScope": True,
                "withdrawn": False,
            }
        ]

    with pytest.raises(RetrievalContractError) as caught:
        service.retrieve(
            role="memory",
            request={"operationId": "op-memory", "intentText": "guide", "exact": "SECRET-HIT"},
            authority=authority,
            discover=discover,
            exact_searcher=search,
        )
    assert caught.value.code == EXACT_CAPABILITY_ABSENT
    assert discovered == []

    envelope = service.retrieve(
        role="reader",
        request={
            "operationId": "op-reader",
            "intentText": "guide",
            "query": {"exact": {"expression": "SECRET-HIT", "syntax": "literal"}},
        },
        authority=authority,
        discover=discover,
        exact_searcher=search,
    )
    assert [card["id"] for card in envelope["memoryCandidateCards"]] == ["skill-1"]
    assert [card["id"] for card in envelope["analogicalPriorCards"]] == ["prior-1"]
    assert envelope["exactLaneAudit"]["returnedCount"] == 1
    assert "ep-1" not in json.dumps(envelope["memoryCandidateCards"])

    policy = ControlPolicy(
        exact_window_ms=3_000,
        discovery_window_ms=6_000,
        exact_budget=5,
        discovery_budget=5,
    )
    control = RetrievalControl(
        role="reader",
        parent_lease_ref="parent-lease",
        parent_deadline_ms=8_000,
        now_ms=0,
        dispatched=frozenset({EXACT_LANE, DISCOVERY_LANE}),
        policy=policy,
    )
    control.stage(DISCOVERY_LANE, "skill-1")
    control.complete(DISCOVERY_LANE)
    control.stage(EXACT_LANE, "SECRET-HIT")
    control.charge(EXACT_LANE, 5, 100)
    view = control.public_view()
    assert view["completeness"] == "partial"
    assert view["results"][DISCOVERY_LANE] == ["skill-1"]
    assert "SECRET-HIT" not in json.dumps(view)
    assert control.lanes[EXACT_LANE].withheld == ["SECRET-HIT"]
    assert control.lanes[DISCOVERY_LANE].budget == 5

    allocated = allocate_discovery(
        [
            {"id": "skill-1", "kind": "skill_revision", "corpus": "skill", "use_class": "memory_candidate", "corpus_rank": 1},
            {
                "id": "cold-1",
                "kind": "skill",
                "corpus": "skill",
                "use_class": "memory_candidate",
                "corpus_rank": 2,
                "exploration_stratum": "cold_start_skill",
            },
            {"id": "ep-1", "kind": "episode", "corpus": "episode", "use_class": "memory_candidate", "corpus_rank": 1},
            {"id": "in-1", "kind": "insight", "corpus": "insight", "use_class": "memory_candidate", "corpus_rank": 1},
            {"id": "prior-1", "kind": "block_projection", "corpus": "projection", "use_class": "analogical_prior", "corpus_rank": 1},
        ],
        plan_hash=_PLAN,
        skill_explorations=0,
        skill_exploitations=0,
    )
    seated = {card.id for card in allocated.memory_candidates}
    assert "ep-1" not in seated and "in-1" not in seated
    assert "skill-1" in seated
    assert all(card.selection_mode != "exploration" for card in allocated.memory_candidates)

    proof = {
        "complete": True,
        "snapshot_ref": "snap",
        "index_version": "idx",
        "model_version": "model",
        "policy_version": "policy",
    }
    fused = {item.corpus: item for item in fuse_corpora({
        "skill": {"bm25": ["skill-1"], "vector": ["other"], "proof": proof, "vector_failed": True},
        "projection": {"bm25": ["prior-1"], "vector": ["prior-1"], "proof": proof},
    })}
    assert fused["skill"].degraded is True and fused["skill"].ids == ("skill-1",)
    assert fused["projection"].degraded is False and fused["projection"].ids == ("prior-1",)
    assert HYBRID_FUSION_POLICY_VERSION == "hybrid-fusion-pending-v0"

    influence = _revision("sess-influence", "rev-influence", "influence", True)
    skip = _revision("sess-skip", "rev-skip", "skip", False)
    influence_text = _subgraph_retrieval_text(influence, influence.retrieval_subgraphs[0])
    skip_text = _subgraph_retrieval_text(skip, skip.retrieval_subgraphs[0])
    assert influence_text.split("\nsubgraph_failure:", 1)[0] == skip_text.split("\nsubgraph_failure:", 1)[0]
    assert "claim_relation:influence" in influence_text
    assert "claim_relation:skip" in skip_text
    assert SUBGRAPH_TEXT_POLICY_VERSION == "diagnostic-subgraph-text-v1"

    parent = freeze_direct(
        operation_id="op-e0",
        plan_hash=envelope["planHash"],
        final_order_hash=allocated.final_order_hash,
        memory_candidates=[{"id": "skill-1", "kind": "skill_revision", "corpus": "skill", "use_class": "memory_candidate", "public_rank": 1}],
        analogical_priors=[{"id": "prior-1", "kind": "block_projection", "corpus": "projection", "use_class": "analogical_prior", "public_rank": 1}],
    )
    chain = ExpansionChain(parent, opened_at=1_000.0, clock=lambda: 1_000.0)

    def neighbors(*_args: object) -> list[dict[str, object]]:
        raise AssertionError("unserved anchor must not call the neighbor source")

    with pytest.raises(ExpansionError) as expansion_error:
        chain.expand(
            {
                "operationId": "op-e1",
                "parentEnvelopeRef": parent.envelope_ref,
                "anchorCardRef": "not-served",
                "relation": "similar",
                "expansionChainRef": chain.ref,
            },
            neighbors,
        )
    assert expansion_error.value.code == "anchor_not_served"

    deliveries = ServedDeliveryLedger()
    hint = deliveries.publish(
        {
            "envelope_ref": parent.envelope_ref,
            "operation_id": "op-reader",
            "plan_hash": envelope["planHash"],
            "lanes": view["lanes"],
            "memory_candidates": [
                {"id": "skill-1", "kind": "skill_revision", "use_class": "memory_candidate"}
            ],
            "analogical_priors": [
                {"id": "prior-1", "kind": "block_projection", "use_class": "analogical_prior"}
            ],
        },
        ["skill-1"],
        kind="hint",
        target="agent",
        lineage="task",
    )
    assert hint["utility_counted"] is False
    assert hint["delivery"]["provenance"]["retrievalCompleteness"] == "partial"
    with pytest.raises(HintError) as hint_error:
        deliveries.publish(
            {
                "envelope_ref": parent.envelope_ref,
                "operation_id": "op-reader",
                "plan_hash": envelope["planHash"],
                "lanes": view["lanes"],
                "memory_candidates": [
                    {"id": "skill-1", "kind": "skill_revision", "use_class": "memory_candidate"}
                ],
            },
            ["prior-1"],
            kind="offer",
            target="agent",
            lineage="task",
        )
    assert hint_error.value.code == "source_not_served"

    utility = SkillUtilityLedger()
    published = _clock()
    utility.publish_active("skill-1", published_at=published)
    proposed = utility.submit_quarantine(
        actor="memory",
        skill_revision_id="skill-1",
        reason="needs review",
        occurred_at=published,
        idempotency_key="proposal-1",
        evidence_refs=("env-1",),
    )
    assert proposed.action == "proposed"
    assert utility.revision("skill-1").online_status != "quarantined"
    quarantined = utility.submit_quarantine(
        actor="server_verifier",
        skill_revision_id="skill-1",
        reason="policy hold",
        occurred_at=published,
        idempotency_key="quarantine-1",
    )
    assert quarantined.online_status == "quarantined"
