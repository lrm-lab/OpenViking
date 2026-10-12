# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Contract tests for the ADR-0043 versioned protocol fixtures.

This file stays separate from the influence-view and extraction suites.
CONTRIBUTING says not to add a test file by default and to extend an existing
contract test; these fixtures lock a cross-repository protocol (schema
version, content hash, and v2 rejection) that those suites do not own.

The canonical hash below is the specification. It does not call the
implementation under test. Readers must accept the on-disk golden fixtures
and reject unknown versions, cross-version relabeling, and Claim/Cut fields
on v2 writes. No test starts a model client or a group-chat service.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from openviking.session.aig_native_protocol import (
    OBJECT_KIND_BLOCKED_PRE_EXECUTION,
    OBJECT_KIND_DIRECT_EVIDENCE_ADMISSION,
    OBJECT_KIND_DIRECT_EVIDENCE_ADMISSION_IDENTITY,
    OBJECT_KIND_REMEDIATION_DISPATCH_CONFIRMATION_IDENTITY,
    OBJECT_KIND_REMEDIATION_DISPATCH_REQUEST_IDENTITY,
    OBJECT_KIND_REPLAY_AIG,
    OBJECT_KIND_REPLAY_ELIGIBILITY_ASSESSMENT,
    OBJECT_KIND_REPLAY_ELIGIBILITY_IDENTITY,
    OBJECT_KIND_STRATEGY_PATTERN,
    OBJECT_KIND_STRATEGY_PATTERN_IDENTITY,
    OBJECT_KIND_STRATEGY_PATTERN_PROPOSAL,
    PATTERN_IDENTITY_VERSION,
    SCHEMA_VERSION_AIG_NATIVE_V2,
    SCHEMA_VERSION_LEGACY_CLAIM_CUT,
    SCHEMA_VERSIONS,
    CriterionDisposition,
    ProtocolContractError,
    SchemaVersion,
    UnsupportedSchemaVersion,
    base_revision_catalog_from_view,
    canonical_hash_input,
    protocol_content_hash,
    read_blocked_pre_execution,
    read_branch_evaluation,
    read_first_action_alignment,
    read_fork_node_overlay,
    read_fork_proposal,
    guidance_approval_document,
    guidance_publication_document,
    guidance_rejection_document,
    read_agent_candidate_catalog,
    read_guidance_approval,
    read_guidance_document,
    read_guidance_execution_report,
    read_guidance_publication,
    read_guidance_rejection,
    read_guided_counterfactual_branch,
    read_legacy_claim_cut,
    read_native_edge_draft,
    read_native_edge_revision,
    read_direct_evidence_admission,
    read_outcome_contract,
    read_protocol_document,
    read_remediation_dispatch_confirmation,
    read_remediation_dispatch_request,
    read_replay_aig,
    read_replay_eligibility_assessment,
    read_replay_eligibility,
    read_strategy_pattern,
    read_strategy_pattern_proposal,
    read_task_outcome_assessment,
    read_v2_fixture_set,
    strategy_pattern_document,
    strategy_pattern_id,
)
from openviking.session.outcome_contract import (
    DECISION_CONFIRMED_FAILURE,
    OUTCOME_EVIDENCE_POLICY_VERSION,
    REASON_EXTERNAL_FAIL,
)
from openviking.session.causal_experiences import (
    ApplicabilityCheck,
    EvidenceRef,
    InfluenceGrounding,
)
from openviking.session.influence_view import (
    AORange,
    CoverageManifest,
    InfluenceBlock,
    InfluenceClaim,
    InfluenceViewRevision,
    SegmentRef,
    SourceRef,
    compute_content_hash,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "adr0043"
LEGACY_CONTENT_HASH = "13ccf85f06557bfec5d7790b765769b13d22497374ec7675ac3ebe9adfc5e0ce"
V2_CONTENT_HASH = "acd7c028ec344e09c8b8c769eb821d00c0cb1571236ed44387738454b107cfa8"
NATIVE_EDGE_DRAFT_HASH = "65356beed0f70b458dc7bb3be81dbf5cfae94a16c58389f046b0e13f6e2ab4ae"
NATIVE_EDGE_REVISION_HASH = "fab0b5eaa7867e9a38f56a75c30e9e3d9ac2f123799a9d3cd6a6e6cffc0e8c45"
OUTCOME_CONTRACT_HASH = "e7950f3e439877c6421f8935d8daab06ca3515fddbf2c25e6237cb8f436d7587"
STRATEGY_PATTERN_PROPOSAL_HASH = (
    "44b08f0b14d4623bb09ff4395613cb3a13f85cf1eb3911e33ae8283561f2b700"
)
STRATEGY_PATTERN_HASH = "17e2a73ec7b19380ed47f31a147c0d399d63cc519e3770a56505ffaf4e098822"
STRATEGY_PATTERN_ID = "eddcf51b921a4cf01fef9ddc99673e9111460516a0b032d5b96e29ab7357b9b7"
REPLAY_ELIGIBILITY_ASSESSMENT_HASH = (
    "bcfbb10fb6b8e6d462eb3fb01911426aa08f5a20cc32059816c446cfd05a5e9e"
)
REPLAY_AIG_HASH = "7625f9250b790656558e8c1aeda12a81068ece2a8ec65df281b8000d58f34c9d"
DIRECT_EVIDENCE_ADMISSION_HASH = (
    "21fd3c830c7a29d2d01bef1275ca94c90d34ffdf3b7fbb01cc0fbb20f4831792"
)
V2_CLAIM_CUT_KEYS = frozenset(
    {
        "claim_id",
        "claim_ids",
        "claims",
        "cut",
        "cut_id",
        "cuts",
        "influence_claim",
        "influence_claims",
        "influence_grounding",
        "position_kind",
        "position_ref",
    }
)


def _spec_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _spec_content_hash(schema_version: str, object_kind: str, body: dict[str, Any]) -> str:
    """Specified hash input: version and object kind tag the body."""
    payload = {
        "schema_version": schema_version,
        "object_kind": object_kind,
        "body": body,
    }
    return hashlib.sha256(_spec_dumps(payload).encode("utf-8")).hexdigest()


def _spec_block_content_hash(blocks: tuple[InfluenceBlock, ...]) -> str:
    """Stdlib recomputation of the block-only catalog digest.

    Input is each block's ``to_dict()`` record, sorted by ``block_id``.
    This does not call the protocol hash helper.
    """
    records = [block.to_dict() for block in blocks]
    records.sort(key=lambda record: record["block_id"])
    return hashlib.sha256(_spec_dumps(records).encode("utf-8")).hexdigest()


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _document(schema_version: str, object_kind: str, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "object_kind": object_kind,
        "body": body,
        "content_hash": _spec_content_hash(schema_version, object_kind, body),
    }


def _contains_key(value: Any, forbidden: frozenset[str]) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in forbidden:
                return key
            found = _contains_key(item, forbidden)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _contains_key(item, forbidden)
            if found is not None:
                return found
    return None


@pytest.fixture
def legacy_payload() -> dict[str, Any]:
    return _load("legacy_claim_cut.json")


@pytest.fixture
def v2_payload() -> dict[str, Any]:
    return _load("v2_fixture_set.json")


def test_schema_versions_are_closed() -> None:
    assert SCHEMA_VERSIONS == frozenset(
        {"legacy-claim-cut-v1", "aig-native-v2"}
    )
    assert SchemaVersion.LEGACY_CLAIM_CUT_V1.value == SCHEMA_VERSION_LEGACY_CLAIM_CUT
    assert SchemaVersion.AIG_NATIVE_V2.value == SCHEMA_VERSION_AIG_NATIVE_V2
    assert frozenset(item.value for item in SchemaVersion) == SCHEMA_VERSIONS


def test_canonical_hash_input_tags_version_and_kind(legacy_payload: dict[str, Any]) -> None:
    tagged = canonical_hash_input(
        legacy_payload["schema_version"],
        legacy_payload["object_kind"],
        legacy_payload["body"],
    )
    assert set(tagged) == {"schema_version", "object_kind", "body"}
    assert tagged["body"] == legacy_payload["body"]
    assert protocol_content_hash(
        legacy_payload["schema_version"],
        legacy_payload["object_kind"],
        legacy_payload["body"],
    ) == _spec_content_hash(
        legacy_payload["schema_version"],
        legacy_payload["object_kind"],
        legacy_payload["body"],
    )
    assert legacy_payload["content_hash"] == LEGACY_CONTENT_HASH


def test_legacy_and_v2_content_hashes_do_not_collide(
    legacy_payload: dict[str, Any], v2_payload: dict[str, Any]
) -> None:
    assert legacy_payload["content_hash"] == LEGACY_CONTENT_HASH
    assert v2_payload["content_hash"] == V2_CONTENT_HASH
    assert LEGACY_CONTENT_HASH != V2_CONTENT_HASH
    assert _spec_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, "legacy_claim_cut", legacy_payload["body"]
    ) != LEGACY_CONTENT_HASH
    assert _spec_content_hash(
        SCHEMA_VERSION_LEGACY_CLAIM_CUT, "v2_fixture_set", v2_payload["body"]
    ) != V2_CONTENT_HASH

    legacy_text = _spec_dumps(
        {
            "schema_version": legacy_payload["schema_version"],
            "object_kind": legacy_payload["object_kind"],
            "body": legacy_payload["body"],
        }
    )
    assert "claim_id" in legacy_text
    assert '"position_kind":"cut"' in legacy_text

    proposal = v2_payload["body"]["documents"]["fork_proposal"]
    proposal_text = _spec_dumps(
        {
            "schema_version": proposal["schema_version"],
            "object_kind": proposal["object_kind"],
            "body": proposal["body"],
        }
    )
    assert "anchor_block_id" in proposal_text
    assert _contains_key(proposal["body"], V2_CLAIM_CUT_KEYS) is None
    assert _contains_key(v2_payload["body"], V2_CLAIM_CUT_KEYS) is None

    child_hashes = []
    for kind, document in v2_payload["body"]["documents"].items():
        assert document["object_kind"] == kind
        assert document["schema_version"] == SCHEMA_VERSION_AIG_NATIVE_V2
        digest = _spec_content_hash(
            document["schema_version"], document["object_kind"], document["body"]
        )
        assert document["content_hash"] == digest
        assert digest not in {LEGACY_CONTENT_HASH, V2_CONTENT_HASH}
        child_hashes.append(digest)
    assert len(child_hashes) == len(set(child_hashes)) == 8

    same_body = v2_payload["body"]["documents"]["branch_evaluation"]["body"]
    assert protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, "branch_evaluation", same_body
    ) != protocol_content_hash(SCHEMA_VERSION_AIG_NATIVE_V2, "fork_proposal", same_body)


def test_legacy_fixture_reader_is_deterministic(legacy_payload: dict[str, Any]) -> None:
    first = read_legacy_claim_cut(legacy_payload)
    second = read_protocol_document(json.loads(json.dumps(legacy_payload)))
    assert first == second
    assert isinstance(first.claim, InfluenceClaim)
    assert isinstance(first.cut_grounding, InfluenceGrounding)
    assert first.schema_version == SCHEMA_VERSION_LEGACY_CLAIM_CUT
    assert first.content_hash == LEGACY_CONTENT_HASH
    assert first.claim.claim_id == "claim-rate-to-rollback"
    assert first.claim.source_block_id == "blk-rate"
    assert first.claim.target_block_id == "blk-rollback"
    assert first.claim.relation_type == "influence"
    assert first.claim.status == "hypothesis"
    assert first.cut_grounding.position_kind == "cut"
    assert first.cut_grounding.position_ref == "cut:blk-rate:blk-rollback"
    assert first.cut_grounding.view_revision_id == "rev-legacy-1"
    assert not hasattr(first, "anchor_block_id")
    assert not hasattr(first, "fork_node_overlay")


def test_v2_fixture_reader_is_deterministic(v2_payload: dict[str, Any]) -> None:
    first = read_v2_fixture_set(v2_payload)
    second = read_protocol_document(json.loads(json.dumps(v2_payload)))
    assert first == second
    assert first.schema_version == SCHEMA_VERSION_AIG_NATIVE_V2
    assert first.content_hash == V2_CONTENT_HASH
    assert first.base_revision.base_revision_id == "rev-aig-frozen-1"
    assert first.base_revision.block_ids == ("blk-task", "blk-action")
    assert first.base_revision.block_content_hash == (
        "61b024e1db7c82bc1a3e521938ab76d1da6707c1318bd0ef8dcb09b51645c436"
    )
    assert first.fork_proposal.anchor_block_id == "blk-action"
    assert first.fork_proposal.author_role == "reader"
    assert (
        first.fork_proposal.base_revision_block_content_hash
        == first.base_revision.block_content_hash
    )
    assert first.fork_proposal.anchor_block_id in first.base_revision.block_ids
    assert first.fork_node_overlay.fork_node_id == "fork-overlay-1"
    assert first.fork_node_overlay.anchor_block_id in first.base_revision.block_ids
    factual, imagined, skipped_node = first.fork_node_overlay.branch_steps
    assert factual.step_kind == "factual_block_ref"
    assert factual.block_id == "blk-task"
    assert factual.block_id in first.base_revision.block_ids
    assert imagined.step_kind == "imagined_node"
    assert imagined.imagined_node_id == "img-narrow-retry"
    assert imagined.imagined_node_id not in first.base_revision.block_ids
    assert skipped_node.imagined_node_id == "img-skipped"
    defined = {
        step.imagined_node_id
        for step in first.fork_node_overlay.branch_steps
        if step.imagined_node_id is not None
    }
    assert {
        step.imagined_node_id for step in first.guidance_execution_report.steps
    } <= defined
    assert not hasattr(first.fork_node_overlay, "source_refs")
    assert first.task_outcome_assessment.author_role == "reader"
    assert first.task_outcome_assessment.verdict == "failure"
    assert first.task_outcome_assessment.criterion_dispositions == (
        CriterionDisposition("crit-error-budget", "explicit"),
    )
    assert isinstance(first.task_outcome_assessment.evidence_refs[0], EvidenceRef)
    assert first.replay_eligibility.eligibility == "conditionally_replayable"
    assert isinstance(first.guidance_document.applicability, ApplicabilityCheck)
    assert first.guidance_document.applicability.anchor_state == "partially_applicable"
    assert first.guidance_document.capability_limits == ("replay-sandbox",)
    assert first.guidance_document.source_fork_id == "fork-overlay-1"
    assert first.guidance_document.target_role == "task_agent"
    assert first.branch_evaluation.evidence_tier == "analogical"
    assert first.branch_evaluation.status == "active"
    assert first.branch_evaluation.evidence_ref == "evidence:ao-eval-1"
    alignment = first.first_action_alignment
    assert alignment.actual_block_id == "blk-replay-action"
    assert alignment.alignment == "equivalent"
    assert alignment.depth == "prefix"
    assert alignment.intervention_type == "narrow_retry"
    assert alignment.target_boundary == "blk-action"
    assert alignment.required_capabilities == ("replay-sandbox",)
    assert alignment.safety_control_params == {"isolation": "sandbox"}
    assert alignment.equivalence_explanation is None
    executed, skipped = first.guidance_execution_report.steps
    assert executed.alignment == "functionally_equivalent"
    assert executed.actual_block_id == "blk-child-action"
    assert skipped.alignment == "not_executed"
    assert skipped.actual_block_id is None
    assert skipped.imagined_node_id == "img-skipped"


@pytest.mark.parametrize(
    "version",
    ["v3", "3", "aig-native-v1", "legacy", "V2", "", "aig-native-v2 ", 2, True],
)
def test_unknown_schema_version_fails_closed(version: object) -> None:
    payload = {
        "schema_version": version,
        "object_kind": "v2_fixture_set",
        "body": {},
        "content_hash": "ignored",
    }
    with pytest.raises(UnsupportedSchemaVersion, match="unsupported schema_version"):
        read_protocol_document(payload)
    with pytest.raises(UnsupportedSchemaVersion, match="refusing to guess or downgrade"):
        read_legacy_claim_cut(payload)
    with pytest.raises(UnsupportedSchemaVersion, match="refusing to guess or downgrade"):
        read_v2_fixture_set(payload)


def test_missing_schema_version_fails_closed() -> None:
    with pytest.raises(UnsupportedSchemaVersion, match="unsupported schema_version missing"):
        read_protocol_document({"object_kind": "legacy_claim_cut", "body": {}})


def test_integer_version_is_not_treated_as_v2() -> None:
    with pytest.raises(UnsupportedSchemaVersion, match="unsupported schema_version 2"):
        read_protocol_document({"schema_version": 2, "object_kind": "v2_fixture_set"})


def test_readers_do_not_downgrade_across_versions(
    legacy_payload: dict[str, Any], v2_payload: dict[str, Any]
) -> None:
    with pytest.raises(ProtocolContractError, match="refuses schema_version 'aig-native-v2'"):
        read_legacy_claim_cut(v2_payload)
    with pytest.raises(ProtocolContractError) as legacy_on_v2:
        read_legacy_claim_cut(v2_payload)
    assert type(legacy_on_v2.value) is ProtocolContractError

    with pytest.raises(ProtocolContractError, match="refuses schema_version 'legacy-claim-cut-v1'"):
        read_v2_fixture_set(legacy_payload)
    with pytest.raises(ProtocolContractError) as v2_on_legacy:
        read_v2_fixture_set(legacy_payload)
    assert type(v2_on_legacy.value) is ProtocolContractError


def test_relabeling_legacy_payload_as_v2_is_rejected(legacy_payload: dict[str, Any]) -> None:
    relabeled = _document(
        SCHEMA_VERSION_AIG_NATIVE_V2, "legacy_claim_cut", legacy_payload["body"]
    )
    assert relabeled["content_hash"] != legacy_payload["content_hash"]
    with pytest.raises(ProtocolContractError):
        read_protocol_document(relabeled)
    with pytest.raises(ProtocolContractError):
        read_v2_fixture_set(relabeled)

    disguised = _document(
        SCHEMA_VERSION_AIG_NATIVE_V2, "v2_fixture_set", legacy_payload["body"]
    )
    with pytest.raises(ProtocolContractError, match="Claim/Cut field"):
        read_v2_fixture_set(disguised)


def test_content_hash_mismatch_is_rejected(legacy_payload: dict[str, Any]) -> None:
    tampered = copy.deepcopy(legacy_payload)
    tampered["body"]["claim"]["downstream_effect"] = "rewritten without updating the hash"
    with pytest.raises(ProtocolContractError, match="content_hash mismatch"):
        read_legacy_claim_cut(tampered)


def test_v2_fork_grounding_is_limited_to_blocks_in_the_base_revision(
    v2_payload: dict[str, Any],
) -> None:
    parsed = read_v2_fixture_set(v2_payload)
    catalog = parsed.base_revision
    proposal = v2_payload["body"]["documents"]["fork_proposal"]
    overlay = v2_payload["body"]["documents"]["fork_node_overlay"]
    assert read_fork_proposal(proposal, catalog) == parsed.fork_proposal
    assert read_fork_node_overlay(overlay, catalog) == parsed.fork_node_overlay

    def _proposal_with(**changes: object) -> dict[str, Any]:
        body = copy.deepcopy(proposal["body"])
        body.update(changes)
        return _document(SCHEMA_VERSION_AIG_NATIVE_V2, "fork_proposal", body)

    with pytest.raises(ProtocolContractError, match="not a Block in base revision"):
        read_fork_proposal(_proposal_with(anchor_block_id="blk-from-other-revision"), catalog)
    with pytest.raises(ProtocolContractError, match="not a Block in base revision"):
        read_fork_proposal(_proposal_with(anchor_block_id="claim-rate-to-rollback"), catalog)
    with pytest.raises(ProtocolContractError, match="does not match base revision"):
        read_fork_proposal(_proposal_with(base_revision_id="rev-other"), catalog)
    with pytest.raises(
        ProtocolContractError, match="base_revision block_content_hash does not match"
    ):
        read_fork_proposal(
            _proposal_with(base_revision_block_content_hash="0" * 64),
            catalog,
        )

    foreign_step = copy.deepcopy(proposal["body"])
    foreign_step["branch_steps"] = [
        {"step_kind": "factual_block_ref", "block_id": "blk-from-other-revision"}
    ]
    with pytest.raises(ProtocolContractError, match="not a Block in base revision"):
        read_fork_proposal(
            _document(SCHEMA_VERSION_AIG_NATIVE_V2, "fork_proposal", foreign_step),
            catalog,
        )

    collided = copy.deepcopy(proposal["body"])
    collided["branch_steps"] = [
        {"step_kind": "imagined_node", "imagined_node_id": "blk-action"}
    ]
    with pytest.raises(ProtocolContractError, match="factual Block"):
        read_fork_proposal(
            _document(SCHEMA_VERSION_AIG_NATIVE_V2, "fork_proposal", collided),
            catalog,
        )

    with pytest.raises(ProtocolContractError, match="base revision catalog"):
        read_protocol_document(proposal)


def test_v2_write_rejects_claim_and_cut_fields(v2_payload: dict[str, Any]) -> None:
    parsed = read_v2_fixture_set(v2_payload)
    proposal = v2_payload["body"]["documents"]["fork_proposal"]

    def _with_extra(extra: dict[str, Any]) -> dict[str, Any]:
        body = copy.deepcopy(proposal["body"])
        body.update(extra)
        return _document(SCHEMA_VERSION_AIG_NATIVE_V2, "fork_proposal", body)

    with pytest.raises(ProtocolContractError, match="Claim/Cut field 'claim_id'"):
        read_fork_proposal(
            _with_extra({"claim_id": "claim-rate-to-rollback"}),
            parsed.base_revision,
        )
    with pytest.raises(ProtocolContractError, match="Claim/Cut field 'influence_grounding'"):
        read_fork_proposal(
            _with_extra(
                {
                    "influence_grounding": {
                        "position_kind": "cut",
                        "position_ref": "cut:blk-rate:blk-rollback",
                    }
                }
            ),
            parsed.base_revision,
        )

    stepped = copy.deepcopy(proposal["body"])
    stepped["branch_steps"] = [
        {"step_kind": "factual_block_ref", "block_id": "blk-task", "position_kind": "edge"}
    ]
    with pytest.raises(ProtocolContractError, match="Claim/Cut field 'position_kind'"):
        read_fork_proposal(
            _document(SCHEMA_VERSION_AIG_NATIVE_V2, "fork_proposal", stepped),
            parsed.base_revision,
        )


def test_outcome_assessment_is_uncertain_without_explicit_criteria(
    v2_payload: dict[str, Any],
) -> None:
    base = copy.deepcopy(v2_payload["body"]["documents"]["task_outcome_assessment"]["body"])

    def _assessment(**overrides: object) -> dict[str, Any]:
        body = copy.deepcopy(base)
        body.update(overrides)
        return _document(SCHEMA_VERSION_AIG_NATIVE_V2, "task_outcome_assessment", body)

    for disposition in ("missing", "conflict", "provisional"):
        parsed = read_task_outcome_assessment(
            _assessment(
                criteria_dispositions={"crit-error-budget": disposition},
                verdict="uncertain",
                evidence_refs=[],
            )
        )
        assert parsed.verdict == "uncertain"
        assert parsed.criterion_dispositions == (
            CriterionDisposition("crit-error-budget", disposition),
        )
        with pytest.raises(ProtocolContractError, match="uncertain"):
            read_task_outcome_assessment(
                _assessment(
                    criteria_dispositions={"crit-error-budget": disposition},
                    verdict="failure",
                )
            )
    with pytest.raises(ProtocolContractError, match="criteria_dispositions"):
        read_task_outcome_assessment(_assessment(criteria_dispositions="explicit"))

    explicit = read_task_outcome_assessment(_assessment(verdict="success"))
    assert explicit.verdict == "success"
    uncertain = read_task_outcome_assessment(_assessment(verdict="uncertain"))
    assert uncertain.verdict == "uncertain"
    with pytest.raises(ProtocolContractError, match="evidence_refs"):
        read_task_outcome_assessment(_assessment(verdict="failure", evidence_refs=[]))


@pytest.mark.parametrize("role", ["builder", "critic", "refiner"])
def test_only_reader_can_author_fork_proposal(v2_payload: dict[str, Any], role: str) -> None:
    parsed = read_v2_fixture_set(v2_payload)
    body = copy.deepcopy(v2_payload["body"]["documents"]["fork_proposal"]["body"])
    body["author_role"] = role
    with pytest.raises(ProtocolContractError, match="ForkProposal author_role must be reader"):
        read_fork_proposal(
            _document(SCHEMA_VERSION_AIG_NATIVE_V2, "fork_proposal", body),
            parsed.base_revision,
        )


@pytest.mark.parametrize("role", ["builder", "critic", "refiner"])
def test_only_reader_can_author_outcome_assessment(
    v2_payload: dict[str, Any], role: str
) -> None:
    body = copy.deepcopy(v2_payload["body"]["documents"]["task_outcome_assessment"]["body"])
    body["author_role"] = role
    with pytest.raises(ProtocolContractError, match="author_role must be reader"):
        read_task_outcome_assessment(
            _document(SCHEMA_VERSION_AIG_NATIVE_V2, "task_outcome_assessment", body)
        )


def test_branch_evaluation_keeps_recorded_tier_and_lifecycle(
    v2_payload: dict[str, Any],
) -> None:
    evaluation = read_branch_evaluation(v2_payload["body"]["documents"]["branch_evaluation"])
    assert evaluation.evidence_tier == "analogical"
    assert evaluation.status == "active"
    assert evaluation.evidence_ref == "evidence:ao-eval-1"

    promoted = copy.deepcopy(v2_payload["body"]["documents"]["branch_evaluation"]["body"])
    promoted["promoted_tier"] = "production-direct"
    with pytest.raises(ProtocolContractError, match="unexpected field 'promoted_tier'"):
        read_branch_evaluation(
            _document(SCHEMA_VERSION_AIG_NATIVE_V2, "branch_evaluation", promoted)
        )
    disguised = copy.deepcopy(promoted)
    del disguised["promoted_tier"]
    disguised["evidence_tier"] = "analogical-direct"
    with pytest.raises(ProtocolContractError, match="evidence_tier"):
        read_branch_evaluation(
            _document(SCHEMA_VERSION_AIG_NATIVE_V2, "branch_evaluation", disguised)
        )

    pattern = _document(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        "branch_evaluation",
        {
            "evaluation_id": "eval-pattern-1",
            "subject_kind": "strategy_pattern",
            "subject_id": "pattern-narrow-retry",
            "evidence_tier": "observational",
            "evidence_ref": "evidence:pattern-1",
        },
    )
    parsed = read_branch_evaluation(pattern)
    assert parsed.subject_kind == "strategy_pattern"
    assert parsed.evidence_tier == "observational"
    assert parsed.status == "active"


@pytest.mark.parametrize(
    "status",
    ["active", "challenged", "invalidated", "retired", "superseded"],
)
def test_branch_evaluation_status_is_append_only_lifecycle(
    v2_payload: dict[str, Any], status: str
) -> None:
    body = copy.deepcopy(v2_payload["body"]["documents"]["branch_evaluation"]["body"])
    body["status"] = status
    parsed = read_branch_evaluation(
        _document(SCHEMA_VERSION_AIG_NATIVE_V2, "branch_evaluation", body)
    )
    assert parsed.status == status


def test_branch_evaluation_rejects_unknown_status(v2_payload: dict[str, Any]) -> None:
    body = copy.deepcopy(v2_payload["body"]["documents"]["branch_evaluation"]["body"])
    body["status"] = "deleted"
    with pytest.raises(ProtocolContractError, match="status"):
        read_branch_evaluation(
            _document(SCHEMA_VERSION_AIG_NATIVE_V2, "branch_evaluation", body)
        )


def test_first_action_alignment_records_submitter_observation() -> None:
    def _alignment(
        alignment: str,
        depth: str,
        actual: str | None,
        *,
        explanation: str | None = None,
        include_explanation: bool = False,
    ) -> Any:
        body: dict[str, Any] = {
            "imagined_node_id": "img-narrow-retry",
            "actual_block_id": actual,
            "alignment": alignment,
            "depth": depth,
            "intervention_type": "narrow_retry",
            "target_boundary": "blk-action",
            "required_capabilities": ["replay-sandbox"],
            "safety_control_params": {"isolation": "sandbox"},
        }
        if include_explanation:
            body["equivalence_explanation"] = explanation
        return read_first_action_alignment(
            _document(SCHEMA_VERSION_AIG_NATIVE_V2, "first_action_alignment", body)
        )

    prefix = _alignment("equivalent", "prefix", "blk-replay-action")
    assert prefix.depth == "prefix"
    assert prefix.alignment == "equivalent"
    assert prefix.actual_block_id == "blk-replay-action"
    assert prefix.equivalence_explanation is None
    first_only = _alignment("equivalent", "first_action", "blk-replay-action")
    assert first_only.depth == "first_action"
    full = _alignment(
        "functionally_equivalent",
        "full_branch",
        "blk-replay-action",
        explanation="same retry, narrower tool args",
        include_explanation=True,
    )
    assert full.depth == "full_branch"
    assert full.equivalence_explanation == "same retry, narrower tool args"
    divergent = _alignment("divergent", "full_branch", "blk-other")
    assert divergent.alignment == "divergent"
    assert divergent.actual_block_id == "blk-other"
    skipped = _alignment("not_executed", "first_action", None)
    assert skipped.actual_block_id is None
    assert skipped.depth == "first_action"
    with pytest.raises(ProtocolContractError, match="actual_block_id"):
        _alignment("equivalent", "prefix", None)
    with pytest.raises(ProtocolContractError, match="actual_block_id"):
        _alignment("not_executed", "first_action", "blk-replay-action")
    with pytest.raises(ProtocolContractError, match="safety_control_params"):
        body = {
            "imagined_node_id": "img-narrow-retry",
            "actual_block_id": "blk-replay-action",
            "alignment": "equivalent",
            "depth": "prefix",
            "intervention_type": "narrow_retry",
            "target_boundary": "blk-action",
            "required_capabilities": ["replay-sandbox"],
            "safety_control_params": ["sandbox"],
        }
        read_first_action_alignment(
            _document(SCHEMA_VERSION_AIG_NATIVE_V2, "first_action_alignment", body)
        )


def test_execution_report_records_divergence_and_skipped_steps() -> None:
    report = read_guidance_execution_report(
        _document(
            SCHEMA_VERSION_AIG_NATIVE_V2,
            "guidance_execution_report",
            {
                "guidance_id": "guidance-1",
                "source_fork_id": "fork-overlay-1",
                "parent_session_id": "sess-1",
                "steps": [
                    {
                        "imagined_node_id": "img-narrow-retry",
                        "alignment": "divergent",
                        "actual_block_id": "blk-other",
                    },
                    {
                        "imagined_node_id": "img-skipped",
                        "alignment": "not_executed",
                        "actual_block_id": None,
                    },
                ],
            },
        )
    )
    assert [step.alignment for step in report.steps] == ["divergent", "not_executed"]
    assert report.steps[0].actual_block_id == "blk-other"
    assert report.steps[1].actual_block_id is None
    with pytest.raises(ProtocolContractError, match="actual_block_id"):
        read_guidance_execution_report(
            _document(
                SCHEMA_VERSION_AIG_NATIVE_V2,
                "guidance_execution_report",
                {
                    "guidance_id": "guidance-1",
                    "source_fork_id": "fork-overlay-1",
                    "parent_session_id": "sess-1",
                    "steps": [
                        {
                            "imagined_node_id": "img-skipped",
                            "alignment": "not_executed",
                            "actual_block_id": "blk-other",
                        }
                    ],
                },
            )
        )


@pytest.mark.parametrize(
    "eligibility",
    ["replayable", "conditionally_replayable", "non_replayable"],
)
def test_replay_eligibility_values(v2_payload: dict[str, Any], eligibility: str) -> None:
    body = copy.deepcopy(v2_payload["body"]["documents"]["replay_eligibility"]["body"])
    body["eligibility"] = eligibility
    parsed = read_replay_eligibility(
        _document(SCHEMA_VERSION_AIG_NATIVE_V2, "replay_eligibility", body)
    )
    assert parsed.eligibility == eligibility


@pytest.mark.parametrize("role", ["agent_action", "reader", "builder", "memory"])
def test_guidance_target_role_rejects_block_and_construction_roles(
    v2_payload: dict[str, Any], role: str
) -> None:
    body = copy.deepcopy(v2_payload["body"]["documents"]["guidance_document"]["body"])
    body["target_role"] = role
    with pytest.raises(ProtocolContractError, match="target_role"):
        read_guidance_document(
            _document(SCHEMA_VERSION_AIG_NATIVE_V2, "guidance_document", body)
        )


def test_v2_fixture_set_rejects_unresolved_branch_refs(v2_payload: dict[str, Any]) -> None:
    dangling = copy.deepcopy(v2_payload)
    report = dangling["body"]["documents"]["guidance_execution_report"]
    report["body"]["steps"][1]["imagined_node_id"] = "img-not-defined"
    _rehash_fixture_set(dangling)
    with pytest.raises(ProtocolContractError, match="not defined on fork_node_overlay"):
        read_v2_fixture_set(dangling)

    foreign = copy.deepcopy(v2_payload)
    overlay = foreign["body"]["documents"]["fork_node_overlay"]
    overlay["body"]["branch_steps"][0]["block_id"] = "blk-from-other-revision"
    _rehash_fixture_set(foreign)
    with pytest.raises(ProtocolContractError, match="not a Block in base revision"):
        read_v2_fixture_set(foreign)


def _rehash_fixture_set(payload: dict[str, Any]) -> None:
    documents = payload["body"]["documents"]
    for document in documents.values():
        document["content_hash"] = _spec_content_hash(
            document["schema_version"], document["object_kind"], document["body"]
        )
    payload["content_hash"] = _spec_content_hash(
        payload["schema_version"], payload["object_kind"], payload["body"]
    )


def test_replay_eligibility_rejects_unknown_value(v2_payload: dict[str, Any]) -> None:
    body = copy.deepcopy(v2_payload["body"]["documents"]["replay_eligibility"]["body"])
    body["eligibility"] = "maybe"
    with pytest.raises(ProtocolContractError, match="eligibility"):
        read_replay_eligibility(
            _document(SCHEMA_VERSION_AIG_NATIVE_V2, "replay_eligibility", body)
        )


def test_real_revision_catalog_excludes_claims_and_accepts_its_blocks() -> None:
    task = _block("blk-real-task", 1, role="task", block_kind="task")
    action = _block("blk-real-action", 2, role="agent_action", block_kind="tool_call")
    claim = InfluenceClaim(
        claim_id="claim-real",
        source_block_id="blk-real-task",
        target_block_id="blk-real-action",
        carried_artifact="patch",
        downstream_effect="tests passed",
        relation_type="influence",
    )
    blocks = (task, action)
    coverage = _coverage(blocks)
    revision = InfluenceViewRevision(
        revision_id="rev-real-1",
        view_id="view-real-1",
        purpose="failure_diagnosis",
        session_id="sess-1",
        task_run_id="task-real-1",
        blocks=blocks,
        claims=(claim,),
        coverage=coverage,
        critic_rounds=0,
        patch_provenance=(),
        construction_mode="adaptive",
        content_hash=compute_content_hash(blocks, (claim,), coverage),
        frozen_at="2026-10-10T00:00:00.000Z",
    )
    catalog = base_revision_catalog_from_view(revision)
    block_hash = _spec_block_content_hash(blocks)
    assert catalog.base_revision_id == revision.revision_id
    assert catalog.block_content_hash == block_hash
    assert catalog.block_content_hash != revision.content_hash
    assert catalog.block_ids == ("blk-real-task", "blk-real-action")
    assert "claim-real" not in catalog.block_ids

    other_claim = InfluenceClaim(
        claim_id="claim-other",
        source_block_id="blk-real-task",
        target_block_id="blk-real-action",
        carried_artifact="patch",
        downstream_effect="tests failed",
        relation_type="influence",
    )
    other_revision = InfluenceViewRevision(
        revision_id="rev-real-1",
        view_id="view-real-1",
        purpose="failure_diagnosis",
        session_id="sess-1",
        task_run_id="task-real-1",
        blocks=(action, task),
        claims=(other_claim,),
        coverage=coverage,
        critic_rounds=0,
        patch_provenance=(),
        construction_mode="adaptive",
        content_hash=compute_content_hash((action, task), (other_claim,), coverage),
        frozen_at="2026-10-10T00:00:00.000Z",
    )
    other_catalog = base_revision_catalog_from_view(other_revision)
    assert other_revision.content_hash != revision.content_hash
    assert other_catalog.block_content_hash == block_hash
    assert other_catalog.block_ids == ("blk-real-action", "blk-real-task")

    proposal = _document(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        "fork_proposal",
        {
            "base_revision_id": revision.revision_id,
            "base_revision_block_content_hash": block_hash,
            "anchor_block_id": "blk-real-action",
            "author_role": "reader",
            "report_hash": "cf899129e58b27f8d77082b97e54eb9ee59f99d5fe25843472d5e30a1cc8ec70",
            "branch_steps": [
                {"step_kind": "factual_block_ref", "block_id": "blk-real-task"},
                {"step_kind": "imagined_node", "imagined_node_id": "img-real"},
            ],
        },
    )
    parsed = read_fork_proposal(proposal, catalog)
    assert parsed.anchor_block_id == "blk-real-action"
    claimed = copy.deepcopy(proposal)
    claimed["body"] = copy.deepcopy(proposal["body"])
    claimed["body"]["anchor_block_id"] = "claim-real"
    claimed["content_hash"] = _spec_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, "fork_proposal", claimed["body"]
    )
    with pytest.raises(ProtocolContractError, match="not a Block in base revision"):
        read_fork_proposal(claimed, catalog)


def _block(block_id: str, sequence: int, *, role: str, block_kind: str) -> InfluenceBlock:
    ref = SourceRef(
        ao_id=f"ao-{sequence}",
        sequence=sequence,
        participant_id="agent-a",
        segment_id="seg-a",
        session_id="sess-1",
        content_hash=f"hash-{sequence}",
    )
    return InfluenceBlock(
        block_id=block_id,
        role=role,  # type: ignore[arg-type]
        participant_id="agent-a",
        segment_ref=SegmentRef(segment_id="seg-a", session_id="sess-1"),
        ao_start_seq=sequence,
        ao_end_seq=sequence,
        summary=block_id,
        input="in",
        output="out",
        authorship="agent-a",
        granularity_reason="one action",
        source_content_hashes=(ref.content_hash,),
        source_refs=(ref,),
        block_kind=block_kind,
    )


@pytest.fixture
def native_edge_draft_payload() -> dict[str, Any]:
    return _load("native_edge_draft.json")


@pytest.fixture
def native_edge_revision_payload() -> dict[str, Any]:
    return _load("native_edge_revision.json")


def _revision_with_agent_authorship(construction_mode: str) -> dict[str, Any]:
    """Clone the golden native-edge revision and retarget one agent block."""
    payload = copy.deepcopy(_load("native_edge_revision.json"))
    block = payload["body"]["blocks"][0]
    block["block_kind"] = "agent"
    block["authorship"] = "agent-7 撰写"
    payload["body"]["construction_mode"] = construction_mode
    payload["content_hash"] = _spec_content_hash(
        payload["schema_version"], payload["object_kind"], payload["body"]
    )
    return payload


def test_single_pass_revision_accepts_free_string_agent_authorship() -> None:
    payload = _revision_with_agent_authorship("native_single_pass_aig")
    parsed = read_native_edge_revision(payload)
    agent = parsed.revision.blocks[0]
    assert agent.block_kind == "agent"
    assert agent.authorship == "agent-7 撰写"
    assert parsed.revision.construction_mode == "native_single_pass_aig"


def test_paper_revision_rejects_free_string_agent_authorship() -> None:
    payload = _revision_with_agent_authorship("adaptive_agentic_aig")
    with pytest.raises(ProtocolContractError, match="invalid_authorship"):
        read_native_edge_revision(payload)


def test_native_edge_fixtures_roundtrip_without_claim_lifecycle(
    native_edge_draft_payload: dict[str, Any],
    native_edge_revision_payload: dict[str, Any],
    legacy_payload: dict[str, Any],
) -> None:
    assert native_edge_draft_payload["content_hash"] == NATIVE_EDGE_DRAFT_HASH
    assert native_edge_revision_payload["content_hash"] == NATIVE_EDGE_REVISION_HASH
    assert NATIVE_EDGE_DRAFT_HASH not in {LEGACY_CONTENT_HASH, V2_CONTENT_HASH}
    assert NATIVE_EDGE_REVISION_HASH not in {
        LEGACY_CONTENT_HASH,
        V2_CONTENT_HASH,
        NATIVE_EDGE_DRAFT_HASH,
    }
    for payload in (native_edge_draft_payload, native_edge_revision_payload):
        assert payload["schema_version"] == SCHEMA_VERSION_AIG_NATIVE_V2
        assert payload["content_hash"] == _spec_content_hash(
            payload["schema_version"], payload["object_kind"], payload["body"]
        )
        assert _contains_key(payload["body"], V2_CLAIM_CUT_KEYS) is None

    draft = read_native_edge_draft(native_edge_draft_payload)
    draft_again = read_protocol_document(json.loads(json.dumps(native_edge_draft_payload)))
    assert draft == draft_again
    assert draft.content_hash == NATIVE_EDGE_DRAFT_HASH
    assert draft.draft.edges[0].edge_id == "edge-rate-to-rollback"
    assert draft.draft.edges[0].relation_type == "influence"
    assert draft.draft.edges[0].mechanism == "rate change inherited by the rollback decision"
    assert not hasattr(draft.draft, "claims")
    assert [block.block_id for block in draft.draft.blocks] == ["blk-rate", "blk-rollback"]

    revision = read_native_edge_revision(native_edge_revision_payload)
    revision_again = read_protocol_document(json.loads(json.dumps(native_edge_revision_payload)))
    assert revision == revision_again
    assert revision.content_hash == NATIVE_EDGE_REVISION_HASH
    assert revision.revision.protocol_body() == native_edge_revision_payload["body"]
    assert revision.revision.content_hash == NATIVE_EDGE_REVISION_HASH
    assert not hasattr(revision.revision, "claims")
    assert not hasattr(revision.revision, "retrieval_subgraphs")
    assert revision.revision.edges[0].source_block_id == "blk-rate"
    assert revision.revision.edges[0].target_block_id == "blk-rollback"

    with pytest.raises(ProtocolContractError):
        read_native_edge_revision(legacy_payload)
    with pytest.raises(ProtocolContractError):
        read_legacy_claim_cut(native_edge_revision_payload)


def test_native_edge_reader_rejects_claim_lifecycle_and_legacy_relabel(
    native_edge_revision_payload: dict[str, Any],
    legacy_payload: dict[str, Any],
) -> None:
    with_status = copy.deepcopy(native_edge_revision_payload)
    with_status["body"]["edges"][0]["status"] = "hypothesis"
    with_status["content_hash"] = _spec_content_hash(
        with_status["schema_version"], with_status["object_kind"], with_status["body"]
    )
    with pytest.raises(ProtocolContractError, match="claim lifecycle field 'status'"):
        read_native_edge_revision(with_status)

    with_claim = copy.deepcopy(native_edge_revision_payload)
    with_claim["body"]["claims"] = []
    with_claim["content_hash"] = _spec_content_hash(
        with_claim["schema_version"], with_claim["object_kind"], with_claim["body"]
    )
    with pytest.raises(ProtocolContractError):
        read_native_edge_revision(with_claim)

    relabeled = _document(
        SCHEMA_VERSION_LEGACY_CLAIM_CUT,
        "native_edge_revision",
        native_edge_revision_payload["body"],
    )
    with pytest.raises(ProtocolContractError):
        read_protocol_document(relabeled)
    disguised = _document(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        "native_edge_revision",
        legacy_payload["body"],
    )
    with pytest.raises(ProtocolContractError):
        read_native_edge_revision(disguised)

    tampered = copy.deepcopy(native_edge_revision_payload)
    tampered["body"]["edges"][0]["mechanism"] = "rewritten without updating the hash"
    with pytest.raises(ProtocolContractError, match="content_hash mismatch"):
        read_native_edge_revision(tampered)


def test_native_edge_documents_are_outside_the_fork_fixture_set(
    v2_payload: dict[str, Any],
) -> None:
    documents = v2_payload["body"]["documents"]
    assert "native_edge_draft" not in documents
    assert "native_edge_revision" not in documents
    assert "outcome_contract" not in documents
    assert len(documents) == 8


def test_outcome_contract_fixture_is_outside_the_fork_set_and_fail_closed() -> None:
    payload = _load("outcome_contract.json")
    assert payload["content_hash"] == OUTCOME_CONTRACT_HASH
    assert payload["content_hash"] == _spec_content_hash(
        payload["schema_version"], payload["object_kind"], payload["body"]
    )
    parsed = read_outcome_contract(payload)
    assert read_protocol_document(json.loads(json.dumps(payload))) == parsed
    assert parsed.object_kind == "outcome_contract"
    assert parsed.schema_version == SCHEMA_VERSION_AIG_NATIVE_V2

    v2_documents = _load("v2_fixture_set.json")["body"]["documents"]
    assert "outcome_contract" not in v2_documents
    assert len(v2_documents) == 8

    with pytest.raises(ProtocolContractError, match="schema_version"):
        read_outcome_contract(_load("legacy_claim_cut.json"))
    with pytest.raises(ProtocolContractError, match="schema_version"):
        read_legacy_claim_cut(payload)


def _spec_pattern_id(body: dict[str, Any]) -> str:
    """Specified identity digest. Semantic summary and proposal ids are not inputs."""
    conditions = sorted(body["applicability_conditions"], key=lambda item: item["condition_id"])
    identity = {
        "applicability_conditions": conditions,
        "capability": body["capability"],
        "expected_metric": body["expected_metric"],
        "identity_version": "strategy-pattern-identity-v1",
        "intervention_type": body["intervention_type"],
        "mechanism": body["mechanism"],
        "target_boundary": body["target_boundary"],
    }
    return _spec_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_STRATEGY_PATTERN_IDENTITY, identity
    )


def test_strategy_pattern_fixtures_are_outside_the_fork_set_and_fail_closed() -> None:
    proposal_payload = _load("strategy_pattern_proposal.json")
    pattern_payload = _load("strategy_pattern.json")
    assert proposal_payload["content_hash"] == STRATEGY_PATTERN_PROPOSAL_HASH
    assert pattern_payload["content_hash"] == STRATEGY_PATTERN_HASH
    assert proposal_payload["content_hash"] == _spec_content_hash(
        proposal_payload["schema_version"],
        proposal_payload["object_kind"],
        proposal_payload["body"],
    )
    assert pattern_payload["content_hash"] == _spec_content_hash(
        pattern_payload["schema_version"],
        pattern_payload["object_kind"],
        pattern_payload["body"],
    )
    assert PATTERN_IDENTITY_VERSION == "strategy-pattern-identity-v1"
    assert OBJECT_KIND_STRATEGY_PATTERN_PROPOSAL == "strategy_pattern_proposal"
    assert OBJECT_KIND_STRATEGY_PATTERN == "strategy_pattern"
    assert OBJECT_KIND_STRATEGY_PATTERN_IDENTITY == "strategy_pattern_identity"

    proposal = read_strategy_pattern_proposal(proposal_payload)
    pattern = read_strategy_pattern(pattern_payload)
    assert read_protocol_document(json.loads(json.dumps(proposal_payload))) == proposal
    assert read_protocol_document(json.loads(json.dumps(pattern_payload))) == pattern
    assert proposal.author_role == "reader"
    assert proposal.semantic_summary == "a narrower retry after timeout"
    assert proposal.imagined_node_id == "img-narrow-retry"
    assert pattern.pattern_id == STRATEGY_PATTERN_ID
    assert pattern.pattern_id == _spec_pattern_id(proposal_payload["body"])
    assert pattern.identity_version == PATTERN_IDENTITY_VERSION
    assert "semantic_summary" not in pattern_payload["body"]
    assert pattern == read_strategy_pattern(strategy_pattern_document(proposal))
    reversed_conditions = list(reversed(proposal_payload["body"]["applicability_conditions"]))
    assert strategy_pattern_id(
        intervention_type=proposal.intervention_type,
        target_boundary=proposal.target_boundary,
        mechanism=proposal.mechanism,
        expected_metric=proposal.expected_metric,
        capability=proposal.capability,
        applicability_conditions=reversed_conditions,
    ) == pattern.pattern_id

    v2_documents = _load("v2_fixture_set.json")["body"]["documents"]
    assert "strategy_pattern" not in v2_documents
    assert "strategy_pattern_proposal" not in v2_documents
    assert len(v2_documents) == 8

    with pytest.raises(ProtocolContractError, match="schema_version"):
        read_strategy_pattern_proposal(_load("legacy_claim_cut.json"))
    with pytest.raises(ProtocolContractError, match="schema_version"):
        read_legacy_claim_cut(pattern_payload)
    identity_only = _document(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        OBJECT_KIND_STRATEGY_PATTERN_IDENTITY,
        {
            "applicability_conditions": pattern_payload["body"]["applicability_conditions"],
            "capability": pattern_payload["body"]["capability"],
            "expected_metric": pattern_payload["body"]["expected_metric"],
            "identity_version": PATTERN_IDENTITY_VERSION,
            "intervention_type": pattern_payload["body"]["intervention_type"],
            "mechanism": pattern_payload["body"]["mechanism"],
            "target_boundary": pattern_payload["body"]["target_boundary"],
        },
    )
    with pytest.raises(ProtocolContractError, match="object_kind"):
        read_protocol_document(identity_only)

    chosen = copy.deepcopy(pattern_payload["body"])
    chosen["pattern_id"] = "a" * 64
    with pytest.raises(ProtocolContractError, match="pattern_id"):
        read_strategy_pattern(
            _document(SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_STRATEGY_PATTERN, chosen)
        )
    smuggled = copy.deepcopy(proposal_payload["body"])
    smuggled["pattern_id"] = pattern.pattern_id
    with pytest.raises(ProtocolContractError, match="unexpected field"):
        read_strategy_pattern_proposal(
            _document(
                SCHEMA_VERSION_AIG_NATIVE_V2,
                OBJECT_KIND_STRATEGY_PATTERN_PROPOSAL,
                smuggled,
            )
        )
    claimed = copy.deepcopy(proposal_payload["body"])
    claimed["claim_id"] = "claim-smuggled"
    with pytest.raises(ProtocolContractError, match="Claim/Cut"):
        read_strategy_pattern_proposal(
            _document(
                SCHEMA_VERSION_AIG_NATIVE_V2,
                OBJECT_KIND_STRATEGY_PATTERN_PROPOSAL,
                claimed,
            )
        )


def test_replay_gate_fixtures_are_outside_the_fork_set_and_fail_closed() -> None:
    assessment_payload = _load("replay_eligibility_assessment.json")
    replay_aig_payload = _load("replay_aig.json")
    admission_payload = _load("direct_evidence_admission.json")
    pinned = {
        "replay_eligibility_assessment": (
            assessment_payload,
            REPLAY_ELIGIBILITY_ASSESSMENT_HASH,
            read_replay_eligibility_assessment,
        ),
        "replay_aig": (replay_aig_payload, REPLAY_AIG_HASH, read_replay_aig),
        "direct_evidence_admission": (
            admission_payload,
            DIRECT_EVIDENCE_ADMISSION_HASH,
            read_direct_evidence_admission,
        ),
    }
    v2_documents = _load("v2_fixture_set.json")["body"]["documents"]
    assert len(v2_documents) == 8
    for kind, (payload, digest, reader) in pinned.items():
        assert payload["object_kind"] == kind
        assert payload["content_hash"] == digest
        assert payload["content_hash"] == _spec_content_hash(
            payload["schema_version"], payload["object_kind"], payload["body"]
        )
        parsed = reader(payload)
        assert read_protocol_document(json.loads(json.dumps(payload))) == parsed
        assert parsed.schema_version == SCHEMA_VERSION_AIG_NATIVE_V2
        assert kind not in v2_documents
        with pytest.raises(ProtocolContractError, match="schema_version"):
            reader(_load("legacy_claim_cut.json"))
        claimed = copy.deepcopy(payload["body"])
        claimed["claim_id"] = "claim-smuggled"
        with pytest.raises(ProtocolContractError, match="Claim/Cut"):
            reader(_document(SCHEMA_VERSION_AIG_NATIVE_V2, kind, claimed))

    assessment = read_replay_eligibility_assessment(assessment_payload)
    assert assessment.eligibility == "replayable"
    assert assessment.route == "replay"
    replay_aig = read_replay_aig(replay_aig_payload)
    assert replay_aig.provenance == "replay"
    assert replay_aig.revision_id.startswith("replay-rev:")
    admission = read_direct_evidence_admission(admission_payload)
    assert admission.outcome_direct is True
    assert admission.first_action_direct is True
    assert admission.evidence_tier == "replay-direct"
    assert admission.step_alignments[0].imagined_node_id == "img-narrow-retry"
    assert admission.branch_imagined_ids == ("img-narrow-retry",)
    assert admission.replay_block_ids == ("blk-replay-action",)

    for kind, body in (
        (
            OBJECT_KIND_REPLAY_ELIGIBILITY_IDENTITY,
            {"fork_node_id": "fork-overlay-1"},
        ),
        (
            OBJECT_KIND_DIRECT_EVIDENCE_ADMISSION_IDENTITY,
            {"imagined_node_id": "img-narrow-retry"},
        ),
    ):
        with pytest.raises(ProtocolContractError, match="object_kind"):
            read_protocol_document(_document(SCHEMA_VERSION_AIG_NATIVE_V2, kind, body))

    assert OBJECT_KIND_REPLAY_ELIGIBILITY_ASSESSMENT == "replay_eligibility_assessment"
    assert OBJECT_KIND_REPLAY_AIG == "replay_aig"
    assert OBJECT_KIND_DIRECT_EVIDENCE_ADMISSION == "direct_evidence_admission"


GUIDED_BRANCH_HASH = "893800bd825fcc4991a27506ed99ff3fce19eb1c94896aa4c1be3f9bfa0e81e0"
AGENT_CATALOG_HASH = "6901aa89c13c3fbf6a4cad2ae7728a954ee0d12be6c63d445aaaca5bdc9192b6"
GUIDANCE_APPROVAL_HASH = "e44834edf32dae25000b9b4410c7d295ed1bc2a16d4c79e801c24ad7d7d84e84"
GUIDANCE_PUBLICATION_HASH = "7dd7c490404bed5e1ecfcbbe24647abc890c6ada0365ddc34cb23928aef3846b"
GUIDANCE_REJECTION_HASH = "27263e03b6374020ae8b8270d4b7af06d2612d26a45b86598e9e65a434528d6a"
GUIDANCE_APPROVAL_ID = "d916a96c1ed736228cd30a62ef73fa560e325cbe2e19dd830093377e11940dc7"
GUIDANCE_PROJECTION_HASH = "91b38305bb07d19a6f1058689060f4c8799649a2c89bf6bfe01c1a3753d6bea6"


def test_guidance_approval_fixtures_are_outside_the_fork_set_and_fail_closed() -> None:
    proposal_payload = _load("guided_counterfactual_branch.json")
    catalog_payload = _load("agent_candidate_catalog.json")
    approval_payload = _load("guidance_approval.json")
    publication_payload = _load("guidance_publication.json")
    rejection_payload = _load("guidance_rejection.json")
    pinned = {
        "guided_counterfactual_branch": (
            proposal_payload,
            GUIDED_BRANCH_HASH,
            read_guided_counterfactual_branch,
        ),
        "agent_candidate_catalog": (
            catalog_payload,
            AGENT_CATALOG_HASH,
            read_agent_candidate_catalog,
        ),
        "guidance_approval": (
            approval_payload,
            GUIDANCE_APPROVAL_HASH,
            read_guidance_approval,
        ),
        "guidance_publication": (
            publication_payload,
            GUIDANCE_PUBLICATION_HASH,
            read_guidance_publication,
        ),
        "guidance_rejection": (
            rejection_payload,
            GUIDANCE_REJECTION_HASH,
            read_guidance_rejection,
        ),
    }
    v2_documents = _load("v2_fixture_set.json")["body"]["documents"]
    assert len(v2_documents) == 8
    for kind, (payload, digest, reader) in pinned.items():
        assert payload["object_kind"] == kind
        assert payload["content_hash"] == digest
        assert payload["content_hash"] == _spec_content_hash(
            payload["schema_version"], payload["object_kind"], payload["body"]
        )
        parsed = reader(payload)
        assert read_protocol_document(json.loads(json.dumps(payload))) == parsed
        assert parsed.schema_version == SCHEMA_VERSION_AIG_NATIVE_V2
        assert kind not in v2_documents
        with pytest.raises(ProtocolContractError, match="schema_version"):
            reader(_load("legacy_claim_cut.json"))
        claimed = copy.deepcopy(payload["body"])
        claimed["claim_id"] = "claim-smuggled"
        with pytest.raises(ProtocolContractError, match="Claim/Cut"):
            reader(_document(SCHEMA_VERSION_AIG_NATIVE_V2, kind, claimed))

    proposal = read_guided_counterfactual_branch(proposal_payload)
    assert proposal.author_role == "reader"
    assert proposal.steps[0].evidence_refs
    identity = {
        "candidate_catalog_hash": catalog_payload["content_hash"],
        "proposal_content_hash": proposal.content_hash,
        "selected_agent_id": "agent-a",
    }
    assert approval_payload["body"]["guidance_id"] == GUIDANCE_APPROVAL_ID
    assert approval_payload["body"]["guidance_id"] == _spec_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, "guidance_approval_identity", identity
    )
    projected = {
        "applicability": approval_payload["body"]["applicability"],
        "base_revision_id": approval_payload["body"]["base_revision_id"],
        "capability_limits": approval_payload["body"]["capability_limits"],
        "evidence_refs": approval_payload["body"]["evidence_refs"],
        "guidance_id": approval_payload["body"]["guidance_id"],
        "report_hash": approval_payload["body"]["report_hash"],
        "risk": approval_payload["body"]["risk"],
        "source_fork_id": approval_payload["body"]["source_fork_id"],
        "target_role": approval_payload["body"]["target_role"],
    }
    assert approval_payload["body"]["guidance_content_hash"] == GUIDANCE_PROJECTION_HASH
    assert approval_payload["body"]["guidance_content_hash"] == _spec_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, "guidance_document", projected
    )
    assert approval_payload["body"]["decision"] == DECISION_CONFIRMED_FAILURE
    assert approval_payload["body"]["policy_version"] == OUTCOME_EVIDENCE_POLICY_VERSION
    assert approval_payload["body"]["reason"] == REASON_EXTERNAL_FAIL
    approval = read_guidance_approval(
        guidance_approval_document(
            proposal,
            candidate_catalog_hash=catalog_payload["content_hash"],
            selected_agent_id="agent-a",
            policy_version=OUTCOME_EVIDENCE_POLICY_VERSION,
            decision=DECISION_CONFIRMED_FAILURE,
            reason=REASON_EXTERNAL_FAIL,
            outcome_evidence=approval_payload["body"]["outcome_evidence"],
        )
    )
    assert approval == read_guidance_approval(approval_payload)
    assert guidance_publication_document(
        approval,
        publisher_labels=("user:carol",),
        published_at="2026-10-10T00:00:03.000Z",
    ) == publication_payload
    assert guidance_rejection_document(proposal.content_hash) == rejection_payload
    with pytest.raises(ProtocolContractError, match="object_kind"):
        read_protocol_document(
            _document(
                SCHEMA_VERSION_AIG_NATIVE_V2,
                "guidance_approval_identity",
                identity,
            )
        )
    forged_author = copy.deepcopy(proposal_payload["body"])
    forged_author["author_role"] = "memory"
    with pytest.raises(ProtocolContractError, match="author_role"):
        read_guided_counterfactual_branch(
            _document(SCHEMA_VERSION_AIG_NATIVE_V2, "guided_counterfactual_branch", forged_author)
        )
    unevidenced = copy.deepcopy(proposal_payload["body"])
    unevidenced["steps"][0]["evidence_refs"] = []
    with pytest.raises(ProtocolContractError, match="evidence_refs"):
        read_guided_counterfactual_branch(
            _document(SCHEMA_VERSION_AIG_NATIVE_V2, "guided_counterfactual_branch", unevidenced)
        )
    success = copy.deepcopy(approval_payload["body"])
    success["decision"] = "success"
    with pytest.raises(ProtocolContractError, match="confirmed_failure"):
        read_guidance_approval(
            _document(SCHEMA_VERSION_AIG_NATIVE_V2, "guidance_approval", success)
        )
    legacy_guidance = copy.deepcopy(_load("v2_fixture_set.json")["body"]["documents"]["guidance_document"]["body"])
    legacy_guidance["selected_agent_id"] = "agent-a"
    with pytest.raises(ProtocolContractError, match="unexpected field"):
        read_guidance_document(
            _document(SCHEMA_VERSION_AIG_NATIVE_V2, "guidance_document", legacy_guidance)
        )


DISPATCH_REQUEST_HASH = "8dd95fa9c5b4e4b929bfb7e077d807f7e24ccb9187515c45c77d73607dd13819"
DISPATCH_CONFIRMATION_HASH = "1cf2b81e198a60a8643a2994a88344d18d1202c9744ca91e0785285dea79eb50"
BLOCKED_PRE_EXECUTION_HASH = "843a4fcff35d356c320f8c7aa0f44aa50a9e43f6facd3d089a6508fb41907b01"


def test_remediation_dispatch_fixtures_are_outside_the_fork_set_and_fail_closed() -> None:
    request_payload = _load("remediation_dispatch_request.json")
    confirmation_payload = _load("remediation_dispatch_confirmation.json")
    blocked_payload = _load("blocked_pre_execution.json")
    pinned = {
        "remediation_dispatch_request": (
            request_payload,
            DISPATCH_REQUEST_HASH,
            read_remediation_dispatch_request,
        ),
        "remediation_dispatch_confirmation": (
            confirmation_payload,
            DISPATCH_CONFIRMATION_HASH,
            read_remediation_dispatch_confirmation,
        ),
        "blocked_pre_execution": (
            blocked_payload,
            BLOCKED_PRE_EXECUTION_HASH,
            read_blocked_pre_execution,
        ),
    }
    v2_documents = _load("v2_fixture_set.json")["body"]["documents"]
    assert len(v2_documents) == 8
    for kind, (payload, digest, reader) in pinned.items():
        assert payload["object_kind"] == kind
        assert payload["content_hash"] == digest
        assert payload["content_hash"] == _spec_content_hash(
            payload["schema_version"], payload["object_kind"], payload["body"]
        )
        parsed = reader(payload)
        assert read_protocol_document(json.loads(json.dumps(payload))) == parsed
        assert parsed.schema_version == SCHEMA_VERSION_AIG_NATIVE_V2
        assert kind not in v2_documents
        with pytest.raises(ProtocolContractError, match="schema_version"):
            reader(_load("legacy_claim_cut.json"))
        claimed = copy.deepcopy(payload["body"])
        claimed["claim_id"] = "claim-smuggled"
        with pytest.raises(ProtocolContractError, match="Claim/Cut"):
            reader(_document(SCHEMA_VERSION_AIG_NATIVE_V2, kind, claimed))

    request = read_remediation_dispatch_request(request_payload)
    confirmation = read_remediation_dispatch_confirmation(confirmation_payload)
    assert confirmation.request_id == request.request_id
    assert confirmation.executor == "scripted"
    assert confirmation.status == "accepted"
    blocked = read_blocked_pre_execution(blocked_payload)
    assert blocked.object_kind == OBJECT_KIND_BLOCKED_PRE_EXECUTION
    assert blocked.outcome == "blocked_pre_execution"
    assert blocked.failed_gate == "budget"
    assert "budget:budget_not_compliant" in blocked.reasons
    forged = copy.deepcopy(blocked_payload)
    forged["body"]["outcome"] = "admitted"
    forged["content_hash"] = _spec_content_hash(
        forged["schema_version"], forged["object_kind"], forged["body"]
    )
    with pytest.raises(ProtocolContractError, match="outcome"):
        read_blocked_pre_execution(forged)
    coforge = copy.deepcopy(confirmation_payload)
    coforge["body"]["executor"] = "coforge"
    coforge_identity = {
        key: coforge["body"][key] for key in coforge["body"] if key != "confirmation_id"
    }
    coforge["body"]["confirmation_id"] = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        OBJECT_KIND_REMEDIATION_DISPATCH_CONFIRMATION_IDENTITY,
        coforge_identity,
    )
    coforge["content_hash"] = _spec_content_hash(
        coforge["schema_version"], coforge["object_kind"], coforge["body"]
    )
    with pytest.raises(ProtocolContractError, match="executor"):
        read_remediation_dispatch_confirmation(coforge)
    for kind in (
        OBJECT_KIND_REMEDIATION_DISPATCH_REQUEST_IDENTITY,
        OBJECT_KIND_REMEDIATION_DISPATCH_CONFIRMATION_IDENTITY,
    ):
        with pytest.raises(ProtocolContractError, match="object_kind"):
            read_protocol_document(_document(SCHEMA_VERSION_AIG_NATIVE_V2, kind, {"request_id": "x"}))


def _coverage(blocks: tuple[InfluenceBlock, ...]) -> CoverageManifest:
    refs = tuple(ref for block in blocks for ref in block.source_refs)
    return CoverageManifest(
        total_ao_range=AORange(start_seq=refs[0].sequence, end_seq=refs[-1].sequence),
        inspected_refs=refs,
        included_ranges=tuple(
            AORange(
                start_seq=block.ao_start_seq,
                end_seq=block.ao_end_seq,
                segment_id=block.segment_ref.segment_id,
            )
            for block in blocks
        ),
        omitted_ranges=(),
        snapshot_watermark="wm-1",
    )
