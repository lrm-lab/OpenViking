# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""P07: ReplayEligibility, an isolated Replay AIG, and first-action admission.

The Coforge sandbox replay worker is not started. Request and result documents
are local fixtures. No test opens a socket, calls a model, or writes the
production native-edge store from the replay path.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openviking.session.aig_native_protocol import (
    OBJECT_KIND_DIRECT_EVIDENCE_ADMISSION,
    OBJECT_KIND_NATIVE_EDGE_INDEX,
    OBJECT_KIND_REPLAY_AIG,
    OBJECT_KIND_REPLAY_ELIGIBILITY_ASSESSMENT,
    OBJECT_KIND_REPLAY_REQUEST,
    OBJECT_KIND_REPLAY_REQUEST_IDENTITY,
    OBJECT_KIND_REPLAY_RESULT,
    REPLAY_REVISION_PREFIX,
    REPLAY_SESSION_PREFIX,
    SCHEMA_VERSION_AIG_NATIVE_V2,
    STEP_KIND_IMAGINED,
    BranchStepRef,
    GuidanceStepAlignment,
    ProtocolContractError,
    derive_replay_eligibility,
    judge_direct_admission,
    protocol_content_hash,
    read_protocol_document,
    read_replay_request,
    read_replay_result,
)
from openviking.session.extraction_write import (
    NativeEdgeViewStore,
    NativeEdgeWriteError,
    freeze_native_edge_view,
)
from openviking.session.fork_node_overlay import ForkNodeOverlayStore, LegacyForkGrounding
from openviking.session.fork_precedent import ForkPrecedentError
from openviking.session.influence_view import (
    AigNativeEdge,
    AORange,
    CoverageManifest,
    InfluenceBlock,
    SegmentRef,
    SourceRef,
)
from openviking.session.replay_alignment import (
    ReplayAlignmentError,
    ReplayAlignmentService,
)
from openviking.session.strategy_pattern import StrategyPatternStore
from tests.test_fork_node_overlay import _coverage, _native_revision, _proposal
from tests.test_fork_precedent import _ALICE, _T1, _T2, _env, _plant, _query, _revision
from tests.test_strategy_pattern import (
    _core,
    _direct_admission,
    _evaluation_files,
    _proposal_payload,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "adr0043"
_OWNER = "user:alice"
_FROZEN = "2026-10-10T00:00:00.000Z"
_T0 = "2026-10-10T00:00:00.000Z"
_T3 = "2026-10-10T00:00:03.000Z"
_RUN = "run-1"


def _spec_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _spec_hash(object_kind: str, body: dict) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
        "object_kind": object_kind,
        "body": body,
    }
    return hashlib.sha256(_spec_dumps(payload).encode("utf-8")).hexdigest()


def _document(object_kind: str, body: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
        "object_kind": object_kind,
        "body": body,
        "content_hash": _spec_hash(object_kind, body),
    }


def _controls(**overrides: object) -> dict:
    controls: dict = {
        "intervention_type": "narrow_retry",
        "target_boundary": "b-2",
        "required_capabilities": ["replay-sandbox"],
        "safety_control_params": {"isolation": "sandbox", "network": "denied"},
        "anchor_state": "recoverable",
        "dependencies": "available",
        "permissions": "sufficient",
        "side_effect_isolation": "isolated",
        "budget": "compliant",
        "evaluator": "bound",
    }
    controls.update(overrides)
    return controls


def _alignment(**overrides: object) -> dict:
    body: dict = {
        "imagined_node_id": "img-retry",
        "actual_block_id": "rb-1",
        "alignment": "functionally_equivalent",
        "depth": "prefix",
        "intervention_type": "narrow_retry",
        "target_boundary": "b-2",
        "required_capabilities": ["replay-sandbox"],
        "safety_control_params": {"isolation": "sandbox", "network": "denied"},
        "equivalence_explanation": "same retry with a narrower timeout",
    }
    body.update(overrides)
    if body.get("equivalence_explanation") is None:
        body.pop("equivalence_explanation")
    return _document("first_action_alignment", body)


def _step_alignment(
    imagined: str = "img-retry",
    block: str = "rb-1",
    alignment: str = "equivalent",
) -> dict:
    return {
        "actual_block_id": block,
        "alignment": alignment,
        "imagined_node_id": imagined,
    }


def _covered_branch(*imagined_ids: str) -> tuple[BranchStepRef, ...]:
    return tuple(
        BranchStepRef(step_kind=STEP_KIND_IMAGINED, imagined_node_id=node_id)
        for node_id in imagined_ids
    )


def _replay_blocks(session_id: str) -> tuple[list[InfluenceBlock], tuple[AigNativeEdge, ...]]:
    def block(block_id: str, sequence: int) -> InfluenceBlock:
        source = SourceRef(
            ao_id=f"ao-{block_id}",
            sequence=sequence,
            participant_id="agent-a",
            segment_id="seg-replay",
            session_id=session_id,
            content_hash=f"hash-{block_id}",
        )
        return InfluenceBlock(
            block_id=block_id,
            role="agent_action",
            participant_id="agent-a",
            segment_ref=SegmentRef(segment_id="seg-replay", session_id=session_id),
            ao_start_seq=sequence,
            ao_end_seq=sequence,
            summary=block_id,
            input="in",
            output="out",
            authorship="agent-a",
            granularity_reason="one replay tool call",
            source_content_hashes=(source.content_hash,),
            source_refs=(source,),
            block_kind="tool_call",
        )

    blocks = [block("rb-1", 1), block("rb-2", 2)]
    edges = (
        AigNativeEdge(
            edge_id="e-replay",
            source_block_id="rb-1",
            target_block_id="rb-2",
            relation_type="influence",
            mechanism="later replay block inherits the earlier result",
        ),
    )
    return blocks, edges


def _world(tmp_path: Path) -> SimpleNamespace:
    revision = _native_revision("rev-prod-1")
    native_path = tmp_path / "native-views.json"
    production = NativeEdgeViewStore(native_path)
    production.write(revision)
    overlays = ForkNodeOverlayStore(tmp_path)
    catalog = overlays.register_base_revision(revision, acl_labels=(_OWNER,))
    commit = overlays.submit_proposal(
        _proposal(catalog, document_hash=revision.content_hash),
        principal_labels=(_OWNER,),
    )
    patterns = StrategyPatternStore(tmp_path)
    pattern = patterns.accept_proposal(
        commit.stored,
        catalog,
        _proposal_payload(commit.stored.overlay, "img-retry", _core(target_boundary="b-2")),
    )
    return SimpleNamespace(
        revision=revision,
        native_path=native_path,
        production=production,
        overlays=overlays,
        catalog=catalog,
        commit=commit,
        patterns=patterns,
        pattern=pattern,
        service=ReplayAlignmentService(tmp_path),
    )


def _code(call) -> str:
    try:
        call()
    except ReplayAlignmentError as exc:
        return exc.code
    raise AssertionError("expected ReplayAlignmentError")


def test_derive_replay_eligibility_is_server_side_and_closed() -> None:
    replayable, route, reasons = derive_replay_eligibility(_controls())
    assert replayable == "replayable"
    assert route == "replay"
    assert reasons == ("controls_recoverable",)

    conditional, route, reasons = derive_replay_eligibility(_controls(budget="unbounded"))
    assert conditional == "conditionally_replayable"
    assert route == "replay"
    assert reasons == ("budget_not_compliant",)

    blocked, route, reasons = derive_replay_eligibility(
        _controls(side_effect_isolation="unisolated", evaluator="unbound")
    )
    assert blocked == "non_replayable"
    assert route == "analogical"
    assert reasons == ("evaluator_unbound", "side_effects_unisolated")

    missing, route, reasons = derive_replay_eligibility(
        _controls(required_capabilities=[], safety_control_params={"network": "denied"})
    )
    assert missing == "non_replayable"
    assert route == "analogical"
    assert "capabilities_missing" in reasons
    assert "isolation_not_sandbox" in reasons
    with pytest.raises(ProtocolContractError):
        derive_replay_eligibility({**_controls(), "eligibility": "replayable"})


def test_direct_admission_consumes_audited_alignment_fields() -> None:
    covered = (GuidanceStepAlignment("img-retry", "equivalent", "rb-1"),)
    admitted = judge_direct_admission(
        alignment="functionally_equivalent",
        depth="full_branch",
        intervention_type="narrow_retry",
        target_boundary="b-2",
        required_capabilities=("replay-sandbox",),
        safety_control_params={"isolation": "sandbox"},
        expected_intervention_type="narrow_retry",
        expected_target_boundary="b-2",
        expected_required_capabilities=("replay-sandbox",),
        expected_safety_control_params={"isolation": "sandbox"},
        equivalence_explanation="same retry, narrower args",
        route="replay",
        first_imagined_node=True,
        step_alignments=covered,
        branch_steps=_covered_branch("img-retry"),
        replay_block_ids=("rb-1",),
    )
    assert admitted.first_action_direct is True
    assert admitted.outcome_direct is True
    assert admitted.evidence_tier == "replay-direct"
    assert admitted.reasons == ("first_action_direct", "outcome_aligned")

    claimed = judge_direct_admission(
        alignment="functionally_equivalent",
        depth="full_branch",
        intervention_type="narrow_retry",
        target_boundary="b-2",
        required_capabilities=("replay-sandbox",),
        safety_control_params={"isolation": "sandbox"},
        expected_intervention_type="narrow_retry",
        expected_target_boundary="b-2",
        expected_required_capabilities=("replay-sandbox",),
        expected_safety_control_params={"isolation": "sandbox"},
        equivalence_explanation="same retry, narrower args",
        route="replay",
        first_imagined_node=True,
    )
    assert claimed.first_action_direct is True
    assert claimed.outcome_direct is False
    assert claimed.evidence_tier == "replay-direct"
    assert claimed.reasons == ("first_action_direct", "outcome_alignment_unverified")

    outside = judge_direct_admission(
        alignment="equivalent",
        depth="prefix",
        intervention_type="narrow_retry",
        target_boundary="b-2",
        required_capabilities=("replay-sandbox",),
        safety_control_params={"isolation": "sandbox"},
        expected_intervention_type="narrow_retry",
        expected_target_boundary="b-2",
        expected_required_capabilities=("replay-sandbox",),
        expected_safety_control_params={"isolation": "sandbox"},
        equivalence_explanation=None,
        route="replay",
        first_imagined_node=True,
        step_alignments=(GuidanceStepAlignment("img-later", "equivalent", "rb-1"),),
        branch_steps=_covered_branch("img-retry", "img-later"),
        replay_block_ids=("rb-1",),
    )
    assert outside.outcome_direct is False
    assert outside.reasons == ("first_action_direct", "outcome_alignment_unverified")

    first_only = judge_direct_admission(
        alignment="equivalent",
        depth="first_action",
        intervention_type="narrow_retry",
        target_boundary="b-2",
        required_capabilities=("replay-sandbox",),
        safety_control_params={"isolation": "sandbox"},
        expected_intervention_type="narrow_retry",
        expected_target_boundary="b-2",
        expected_required_capabilities=("replay-sandbox",),
        expected_safety_control_params={"isolation": "sandbox"},
        equivalence_explanation=None,
        route="replay",
        first_imagined_node=True,
    )
    assert first_only.first_action_direct is True
    assert first_only.outcome_direct is False
    assert "first_action_does_not_verify_outcome" in first_only.reasons

    unexplained = judge_direct_admission(
        alignment="functionally_equivalent",
        depth="prefix",
        intervention_type="narrow_retry",
        target_boundary="b-2",
        required_capabilities=("replay-sandbox",),
        safety_control_params={"isolation": "sandbox"},
        expected_intervention_type="narrow_retry",
        expected_target_boundary="b-2",
        expected_required_capabilities=("replay-sandbox",),
        expected_safety_control_params={"isolation": "sandbox"},
        equivalence_explanation=None,
        route="replay",
        first_imagined_node=True,
    )
    assert unexplained.first_action_direct is False
    assert unexplained.outcome_direct is False
    assert unexplained.reasons == ("explanation_required",)

    mismatched = judge_direct_admission(
        alignment="equivalent",
        depth="prefix",
        intervention_type="wide_retry",
        target_boundary="b-2",
        required_capabilities=("replay-sandbox",),
        safety_control_params={"isolation": "sandbox"},
        expected_intervention_type="narrow_retry",
        expected_target_boundary="b-2",
        expected_required_capabilities=("replay-sandbox",),
        expected_safety_control_params={"isolation": "sandbox"},
        equivalence_explanation=None,
        route="replay",
        first_imagined_node=True,
    )
    assert mismatched.evidence_tier == "none"
    assert mismatched.reasons == ("audited_field_mismatch",)

    later_node = judge_direct_admission(
        alignment="equivalent",
        depth="full_branch",
        intervention_type="narrow_retry",
        target_boundary="b-2",
        required_capabilities=("replay-sandbox",),
        safety_control_params={"isolation": "sandbox"},
        expected_intervention_type="narrow_retry",
        expected_target_boundary="b-2",
        expected_required_capabilities=("replay-sandbox",),
        expected_safety_control_params={"isolation": "sandbox"},
        equivalence_explanation=None,
        route="replay",
        first_imagined_node=False,
    )
    assert later_node.first_action_direct is False
    assert later_node.outcome_direct is False
    assert later_node.reasons == ("not_first_action",)


def test_replay_contract_golden_fixtures_fail_closed() -> None:
    request_payload = json.loads((FIXTURE_DIR / "replay_request.json").read_text(encoding="utf-8"))
    result_payload = json.loads((FIXTURE_DIR / "replay_result.json").read_text(encoding="utf-8"))
    assert request_payload["content_hash"] == _spec_hash(
        OBJECT_KIND_REPLAY_REQUEST, request_payload["body"]
    )
    assert result_payload["content_hash"] == _spec_hash(
        OBJECT_KIND_REPLAY_RESULT, result_payload["body"]
    )
    request = read_protocol_document(request_payload)
    result = read_protocol_document(result_payload)
    assert request == read_replay_request(request_payload)
    assert result == read_replay_result(result_payload)
    assert request.object_kind == OBJECT_KIND_REPLAY_REQUEST
    assert request.eligibility == "replayable"
    assert result.object_kind == OBJECT_KIND_REPLAY_RESULT
    assert result.status == "completed"
    assert result.first_action_alignment is not None
    assert result.first_action_alignment.depth == "prefix"
    assert result.step_alignments == (
        GuidanceStepAlignment("img-narrow-retry", "equivalent", "blk-replay-action"),
    )
    identity = {key: value for key, value in request_payload["body"].items() if key != "request_id"}
    assert request.request_id == _spec_hash(OBJECT_KIND_REPLAY_REQUEST_IDENTITY, identity)

    unknown = copy.deepcopy(request_payload)
    unknown["schema_version"] = "replay-v9"
    unknown["content_hash"] = _spec_hash("replay-v9", unknown["body"])
    with pytest.raises(ProtocolContractError, match="schema_version"):
        read_protocol_document(unknown)

    extra = copy.deepcopy(request_payload)
    extra["body"]["worker_url"] = "http://127.0.0.1/replay"
    extra["content_hash"] = _spec_hash(OBJECT_KIND_REPLAY_REQUEST, extra["body"])
    with pytest.raises(ProtocolContractError, match="unexpected field"):
        read_replay_request(extra)

    claimed = copy.deepcopy(result_payload)
    claimed["body"]["first_action_alignment"]["body"]["claim_id"] = "claim-smuggled"
    with pytest.raises(ProtocolContractError, match="Claim/Cut"):
        read_replay_result(claimed)

    missing_steps = copy.deepcopy(result_payload)
    del missing_steps["body"]["step_alignments"]
    missing_steps["content_hash"] = _spec_hash(OBJECT_KIND_REPLAY_RESULT, missing_steps["body"])
    with pytest.raises(ProtocolContractError, match="step_alignments"):
        read_replay_result(missing_steps)

    blocked = copy.deepcopy(request_payload["body"])
    blocked["eligibility"] = "non_replayable"
    identity = {key: value for key, value in blocked.items() if key != "request_id"}
    blocked["request_id"] = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_REPLAY_REQUEST_IDENTITY, identity
    )
    with pytest.raises(ProtocolContractError, match="eligibility"):
        read_replay_request(_document(OBJECT_KIND_REPLAY_REQUEST, blocked))

    with pytest.raises(ProtocolContractError, match="object_kind"):
        read_protocol_document(
            _document(OBJECT_KIND_REPLAY_REQUEST_IDENTITY, {"replay_run_id": "run-1"})
        )
    assert (
        OBJECT_KIND_REPLAY_REQUEST
        not in json.loads((FIXTURE_DIR / "v2_fixture_set.json").read_text(encoding="utf-8"))[
            "body"
        ]["documents"]
    )


def test_eligibility_reassessment_appends_and_routes_non_replayable(tmp_path: Path) -> None:
    world = _world(tmp_path)
    fork_id = world.commit.stored.fork_node_id
    service = world.service
    first = service.assess(
        world.commit.stored,
        _controls(side_effect_isolation="unisolated"),
        evidence_ref="fork-created",
        checked_at=_T0,
    )
    assert first.eligibility == "non_replayable"
    assert first.route == "analogical"
    stored_path = next((tmp_path / "replay-alignment").rglob(f"{first.assessment_id}.json"))
    original = stored_path.read_bytes()

    repeated = service.assess(
        world.commit.stored,
        _controls(side_effect_isolation="unisolated"),
        evidence_ref="fork-created",
        checked_at=_T3,
    )
    assert repeated == first
    assert len(service.history(fork_id)) == 1
    assert stored_path.read_bytes() == original

    analogical = service.append_analogical_route(
        world.commit.stored,
        world.patterns,
        evaluation_id="eval-analog",
        subject_kind="imagined_node",
        subject_id="img-retry",
        evidence_ref="evidence:analog",
        evidence_tier="analogical",
        estimated_confidence=0.4,
    )
    assert analogical.evidence_tier == "analogical"
    model_only = service.append_analogical_route(
        world.commit.stored,
        world.patterns,
        evaluation_id="eval-model",
        subject_kind="imagined_node",
        subject_id="img-retry",
        evidence_ref="evidence:model",
        evidence_tier="model-only",
    )
    assert model_only.evidence_tier == "model-only"
    for tier in ("replay-direct", "production-direct", "observational"):
        assert (
            _code(
                lambda tier=tier: service.append_analogical_route(
                    world.commit.stored,
                    world.patterns,
                    evaluation_id=f"eval-{tier}",
                    subject_kind="imagined_node",
                    subject_id="img-retry",
                    evidence_ref=f"evidence:{tier}",
                    evidence_tier=tier,
                )
            )
            == "analogical_route_required"
        )
    assert (
        _code(lambda: service.build_replay_request(first, replay_run_id=_RUN))
        == "analogical_route_required"
    )

    opened = service.assess(
        world.commit.stored,
        _controls(),
        evidence_ref="env:isolation-fixed",
        checked_at=_T3,
    )
    assert opened.eligibility == "replayable"
    assert opened.route == "replay"
    assert stored_path.read_bytes() == original
    assert [item.assessment_id for item in service.history(fork_id)] == [
        first.assessment_id,
        opened.assessment_id,
    ]
    assert service.latest(fork_id) == opened
    assert ReplayAlignmentService(tmp_path).history(fork_id) == service.history(fork_id)
    assert json.loads(original)["body"]["route"] == "analogical"
    assert [item.evidence_tier for item in world.patterns.audit_evaluations()] == [
        "analogical",
        "model-only",
    ]


def test_scripted_replay_chain_keeps_production_aig_clean(tmp_path: Path) -> None:
    world = _world(tmp_path)
    before = world.native_path.read_bytes()
    service = world.service
    created = service.assess(
        world.commit.stored,
        _controls(),
        evidence_ref="fork-created",
        checked_at=_T0,
    )
    rechecked = service.assess(
        world.commit.stored,
        _controls(),
        evidence_ref="env:recheck",
        checked_at=_T3,
    )
    assert created.assessment_id != rechecked.assessment_id
    assert rechecked.eligibility == "replayable"
    service.append_analogical_route(
        world.commit.stored,
        world.patterns,
        evaluation_id="eval-analog",
        subject_kind="imagined_node",
        subject_id="img-retry",
        evidence_ref="evidence:analog",
        estimated_confidence=0.2,
    )
    analogical_path = next(
        path
        for path in _evaluation_files(tmp_path)
        if json.loads(path.read_text(encoding="utf-8"))["body"]["evidence_ref"] == "evidence:analog"
    )
    analogical_bytes = analogical_path.read_bytes()
    world.patterns.append_evaluation(
        _document(
            "branch_evaluation",
            {
                "evaluation_id": "eval-prod",
                "subject_kind": "imagined_node",
                "subject_id": "img-retry",
                "evidence_tier": "replay-direct",
                "evidence_ref": "evidence:prod",
                "status": "active",
            },
        ),
        admission=_direct_admission("img-retry"),
    )

    request = service.build_replay_request(rechecked, replay_run_id=_RUN)
    session_id = f"{REPLAY_SESSION_PREFIX}{_RUN}"
    blocks, edges = _replay_blocks(session_id)
    result = service.build_replay_result(
        request,
        status="completed",
        observed_block_ids=["rb-1", "rb-2"],
        alignment=_alignment(),
        step_alignments=[_step_alignment()],
    )
    outcome = service.apply_scripted_replay(
        world.commit.stored,
        request,
        result,
        blocks=blocks,
        edges=edges,
        coverage=_coverage(blocks),
        frozen_at=_FROZEN,
        pattern_store=world.patterns,
        pattern_id=world.pattern.pattern_id,
    )

    assert world.native_path.read_bytes() == before
    assert world.production.read(world.revision.revision_id) == world.revision
    assert outcome.replay_aig is not None
    assert outcome.replay_aig.provenance == "replay"
    assert outcome.replay_aig.base_revision_id == world.revision.revision_id
    assert outcome.replay_aig.revision_id.startswith(REPLAY_REVISION_PREFIX)
    assert outcome.replay_aig.revision.revision_id == outcome.replay_aig.revision_id
    assert outcome.replay_aig.revision.session_id == session_id
    assert outcome.replay_aig.revision.task_run_id == f"replay-run:{_RUN}"
    assert outcome.replay_aig.object_kind == OBJECT_KIND_REPLAY_AIG
    assert service.read_replay(outcome.replay_aig.revision_id) == outcome.replay_aig
    with pytest.raises(KeyError):
        world.production.read(outcome.replay_aig.revision_id)
    with pytest.raises(NativeEdgeWriteError) as rejected:
        world.production.write(outcome.replay_aig.revision)
    assert rejected.value.code == "replay_provenance_excluded"
    assert world.native_path.read_bytes() == before

    assert outcome.admission is not None
    assert outcome.admission.equivalence_explanation == "same retry with a narrower timeout"
    assert outcome.admission.first_action_direct is True
    assert outcome.admission.outcome_direct is True
    assert outcome.admission.evidence_tier == "replay-direct"
    assert service.read_admission(outcome.admission.admission_id) == outcome.admission
    assert [item.evidence_tier for item in outcome.evaluations] == [
        "replay-direct",
        "replay-direct",
    ]
    assert {item.subject_kind for item in outcome.evaluations} == {
        "imagined_node",
        "strategy_pattern",
    }
    recorded = world.patterns.audit_evaluations()
    assert [item.evidence_tier for item in recorded] == [
        "analogical",
        "replay-direct",
        "replay-direct",
        "replay-direct",
    ]
    assert "evidence:analog" not in {item.evidence_ref for item in outcome.evaluations}
    assert analogical_path.read_bytes() == analogical_bytes
    assert json.loads(analogical_bytes)["body"]["evidence_tier"] == "analogical"
    assert (
        read_protocol_document(
            json.loads((FIXTURE_DIR / "replay_request.json").read_text(encoding="utf-8"))
        ).object_kind
        == OBJECT_KIND_REPLAY_REQUEST
    )


def test_first_action_match_does_not_verify_branch_outcome(tmp_path: Path) -> None:
    world = _world(tmp_path)
    service = world.service
    assessment = service.assess(
        world.commit.stored,
        _controls(),
        evidence_ref="fork-created",
        checked_at=_T0,
    )
    request = service.build_replay_request(assessment, replay_run_id=_RUN)
    blocks, edges = _replay_blocks(f"{REPLAY_SESSION_PREFIX}{_RUN}")
    result = service.build_replay_result(
        request,
        status="completed",
        observed_block_ids=["rb-1", "rb-2"],
        alignment=_alignment(
            alignment="equivalent",
            depth="first_action",
            equivalence_explanation=None,
        ),
    )
    before = {item.evaluation_id for item in world.patterns.audit_evaluations()}
    outcome = service.apply_scripted_replay(
        world.commit.stored,
        request,
        result,
        blocks=blocks,
        edges=edges,
        coverage=_coverage(blocks),
        frozen_at=_FROZEN,
        pattern_store=world.patterns,
        pattern_id=world.pattern.pattern_id,
    )
    assert outcome.admission is not None
    assert outcome.admission.first_action_direct is True
    assert outcome.admission.outcome_direct is False
    assert outcome.admission.equivalence_explanation is None
    assert [item.subject_kind for item in outcome.evaluations] == ["imagined_node"]
    assert outcome.evaluations[0].evidence_tier == "replay-direct"
    added = [
        item for item in world.patterns.audit_evaluations() if item.evaluation_id not in before
    ]
    assert [item.subject_kind for item in added] == ["imagined_node"]
    assert world.pattern.pattern_id not in {item.subject_id for item in added}

    second_request = service.build_replay_request(assessment, replay_run_id="run-divergent")
    divergent = service.build_replay_result(
        second_request,
        status="completed",
        observed_block_ids=["rb-1", "rb-2"],
        alignment=_alignment(
            alignment="divergent",
            depth="full_branch",
            actual_block_id="rb-2",
            equivalence_explanation=None,
        ),
    )
    blocks, edges = _replay_blocks(f"{REPLAY_SESSION_PREFIX}run-divergent")
    refused = service.apply_scripted_replay(
        world.commit.stored,
        second_request,
        divergent,
        blocks=blocks,
        edges=edges,
        coverage=_coverage(blocks),
        frozen_at=_FROZEN,
        pattern_store=world.patterns,
        pattern_id=world.pattern.pattern_id,
    )
    assert refused.replay_aig is not None
    assert refused.admission is not None
    assert refused.admission.evidence_tier == "none"
    assert refused.evaluations == ()
    assert refused.admission.first_action_direct is False
    assert refused.admission.outcome_direct is False


def test_later_imagined_node_is_not_a_first_action(tmp_path: Path) -> None:
    revision = _native_revision("rev-two-nodes")
    overlays = ForkNodeOverlayStore(tmp_path)
    catalog = overlays.register_base_revision(revision, acl_labels=(_OWNER,))
    payload = _proposal(
        catalog,
        document_hash=revision.content_hash,
        branch_steps=[
            {"block_id": "b-1", "step_kind": "factual_block_ref"},
            {"imagined_node_id": "img-first", "step_kind": "imagined_node"},
            {"imagined_node_id": "img-second", "step_kind": "imagined_node"},
        ],
    )
    commit = overlays.submit_proposal(payload, principal_labels=(_OWNER,))
    patterns = StrategyPatternStore(tmp_path)
    patterns.freeze_branch(commit.stored, catalog)
    service = ReplayAlignmentService(tmp_path)
    assessment = service.assess(
        commit.stored,
        _controls(),
        evidence_ref="fork-created",
        checked_at=_T0,
    )
    request = service.build_replay_request(assessment, replay_run_id=_RUN)
    blocks, edges = _replay_blocks(f"{REPLAY_SESSION_PREFIX}{_RUN}")
    result = service.build_replay_result(
        request,
        status="completed",
        observed_block_ids=["rb-1", "rb-2"],
        alignment=_alignment(
            imagined_node_id="img-second",
            alignment="equivalent",
            depth="full_branch",
            equivalence_explanation=None,
        ),
    )
    outcome = service.apply_scripted_replay(
        commit.stored,
        request,
        result,
        blocks=blocks,
        edges=edges,
        coverage=_coverage(blocks),
        frozen_at=_FROZEN,
        pattern_store=patterns,
        pattern_id=None,
    )
    assert outcome.admission is not None
    assert outcome.admission.reasons == ("not_first_action",)
    assert outcome.evaluations == ()
    assert outcome.admission.outcome_direct is False


def test_prefix_claim_without_step_alignment_does_not_write_outcome(tmp_path: Path) -> None:
    world = _world(tmp_path)
    service = world.service
    assessment = service.assess(
        world.commit.stored,
        _controls(),
        evidence_ref="fork-created",
        checked_at=_T0,
    )
    request = service.build_replay_request(assessment, replay_run_id=_RUN)
    blocks, edges = _replay_blocks(f"{REPLAY_SESSION_PREFIX}{_RUN}")
    result = service.build_replay_result(
        request,
        status="completed",
        observed_block_ids=["rb-1", "rb-2"],
        alignment=_alignment(),
        step_alignments=[],
    )
    before = {item.evaluation_id for item in world.patterns.audit_evaluations()}
    with pytest.raises(ReplayAlignmentError) as unverified:
        service.apply_scripted_replay(
            world.commit.stored,
            request,
            result,
            blocks=blocks,
            edges=edges,
            coverage=_coverage(blocks),
            frozen_at=_FROZEN,
            pattern_store=world.patterns,
            pattern_id=world.pattern.pattern_id,
        )
    assert unverified.value.code == "outcome_alignment_unverified"
    added = [
        item for item in world.patterns.audit_evaluations() if item.evaluation_id not in before
    ]
    assert [item.subject_kind for item in added] == ["imagined_node"]
    assert world.pattern.pattern_id not in {item.subject_id for item in added}


def test_full_branch_requires_every_imagined_step(tmp_path: Path) -> None:
    revision = _native_revision("rev-two-nodes")
    overlays = ForkNodeOverlayStore(tmp_path)
    catalog = overlays.register_base_revision(revision, acl_labels=(_OWNER,))
    payload = _proposal(
        catalog,
        document_hash=revision.content_hash,
        branch_steps=[
            {"block_id": "b-1", "step_kind": "factual_block_ref"},
            {"imagined_node_id": "img-first", "step_kind": "imagined_node"},
            {"imagined_node_id": "img-second", "step_kind": "imagined_node"},
        ],
    )
    commit = overlays.submit_proposal(payload, principal_labels=(_OWNER,))
    patterns = StrategyPatternStore(tmp_path)
    patterns.freeze_branch(commit.stored, catalog)
    service = ReplayAlignmentService(tmp_path)
    assessment = service.assess(
        commit.stored,
        _controls(),
        evidence_ref="fork-created",
        checked_at=_T0,
    )
    request = service.build_replay_request(assessment, replay_run_id=_RUN)
    blocks, edges = _replay_blocks(f"{REPLAY_SESSION_PREFIX}{_RUN}")
    result = service.build_replay_result(
        request,
        status="completed",
        observed_block_ids=["rb-1", "rb-2"],
        alignment=_alignment(
            imagined_node_id="img-first",
            alignment="equivalent",
            depth="full_branch",
            equivalence_explanation=None,
        ),
        step_alignments=[_step_alignment("img-first", "rb-1")],
    )
    before = {item.evaluation_id for item in patterns.audit_evaluations()}
    with pytest.raises(ReplayAlignmentError) as unverified:
        service.apply_scripted_replay(
            commit.stored,
            request,
            result,
            blocks=blocks,
            edges=edges,
            coverage=_coverage(blocks),
            frozen_at=_FROZEN,
            pattern_store=patterns,
            pattern_id=None,
        )
    assert unverified.value.code == "outcome_alignment_unverified"
    added = [item for item in patterns.audit_evaluations() if item.evaluation_id not in before]
    assert [item.subject_kind for item in added] == ["imagined_node"]
    assert added[0].subject_id == "img-first"


def test_history_and_admission_ids_must_be_hex(tmp_path: Path) -> None:
    world = _world(tmp_path)
    fork_id = world.commit.stored.fork_node_id
    world.service.assess(
        world.commit.stored,
        _controls(),
        evidence_ref="fork-created",
        checked_at=_T0,
    )
    index = tmp_path / "replay-alignment" / "eligibility" / "index.jsonl"
    index.write_text(
        index.read_text(encoding="utf-8")
        + json.dumps(
            {
                "assessment_id": "../secrets",
                "fork_node_id": fork_id,
                "record_kind": "replay_eligibility_assessment",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ReplayAlignmentError) as history:
        world.service.history(fork_id)
    assert history.value.code == "stored_record_invalid"
    with pytest.raises(ReplayAlignmentError) as admission:
        world.service.read_admission("../secrets")
    assert admission.value.code == "stored_record_invalid"


def test_replay_refuses_a_legacy_fork(tmp_path: Path) -> None:
    service = ReplayAlignmentService(tmp_path)
    legacy = LegacyForkGrounding(
        fork_node_id="legacy-1",
        revision_id="legacy-rev",
        position_kind="cut",
        position_ref="cut:a:b",
        view_revision_id="rev-legacy",
        content_hash="hash-legacy",
        semantic_summary_snapshot="old cut grounding",
        grounded_at="2026-10-10T00:00:00.000Z",
        workspace_id="ws-legacy",
    )
    with pytest.raises(ReplayAlignmentError) as excluded:
        service.assess(legacy, _controls(), evidence_ref="legacy", checked_at=_T0)
    assert excluded.value.code == "legacy_fork_excluded"


def test_production_store_fails_closed_on_injected_replay_provenance(tmp_path: Path) -> None:
    world = _world(tmp_path)
    before = world.native_path.read_bytes()
    session_id = f"{REPLAY_SESSION_PREFIX}injected"
    blocks, edges = _replay_blocks(session_id)
    coverage = _coverage(blocks)
    replay = freeze_native_edge_view(
        session_id=session_id,
        task_run_id="replay-run:injected",
        view_id="replay-view:injected",
        revision_id=REPLAY_REVISION_PREFIX + ("ab" * 32),
        frozen_at=_FROZEN,
        blocks=blocks,
        edges=edges,
        coverage=coverage,
    )
    with pytest.raises(NativeEdgeWriteError) as rejected:
        world.production.write(replay)
    assert rejected.value.code == "replay_provenance_excluded"
    assert world.native_path.read_bytes() == before

    payload = json.loads(before.decode("utf-8"))
    payload["body"]["revisions"].append(replay.protocol_document())
    payload["content_hash"] = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_NATIVE_EDGE_INDEX, payload["body"]
    )
    world.native_path.write_text(json.dumps(payload), encoding="utf-8")
    contaminated = world.native_path.read_bytes()
    with pytest.raises(NativeEdgeWriteError) as contaminated_open:
        NativeEdgeViewStore(world.native_path)
    assert contaminated_open.value.code == "replay_provenance_contaminates_production"
    assert world.native_path.read_bytes() == contaminated


def test_precedent_retrieve_excludes_replay_provenance(tmp_path: Path) -> None:
    overlays, patterns, precedents = _env(tmp_path)
    production = _revision("rev-prod", ("b-fact", "b-anchor"))
    replay = _revision(REPLAY_REVISION_PREFIX + ("cd" * 32), ("b-fact", "b-anchor"))
    hidden = _revision("rev-hidden", ("b-fact", "b-anchor"))
    query_revision = _revision("rev-query", ("b-fact", "b-anchor"))
    production_catalog = overlays.register_base_revision(production, acl_labels=(_ALICE,))
    replay_catalog = overlays.register_base_revision(replay, acl_labels=(_ALICE,))
    hidden_catalog = overlays.register_base_revision(hidden, acl_labels=(_ALICE,))
    query_catalog = overlays.register_base_revision(query_revision, acl_labels=(_ALICE,))
    produced, _pattern = _plant(
        overlays,
        patterns,
        precedents,
        production,
        production_catalog,
        anchor="b-anchor",
        factual=("b-fact",),
        imagined="img-prod",
        session_id="sess-past",
        recorded_at=_T1,
        principal=(_ALICE,),
        node_rows=(("analogical", "active"),),
    )
    replay_commit, _ignored = _plant(
        overlays,
        patterns,
        precedents,
        replay,
        replay_catalog,
        anchor="b-anchor",
        factual=("b-fact",),
        imagined="img-replay",
        session_id="sess-past",
        recorded_at=_T1,
        principal=(_ALICE,),
        node_rows=(("replay-direct", "active"),),
    )
    hidden_commit, _ignored = _plant(
        overlays,
        patterns,
        precedents,
        hidden,
        hidden_catalog,
        anchor="b-anchor",
        factual=("b-fact",),
        imagined="img-hidden",
        session_id=f"{REPLAY_SESSION_PREFIX}hidden",
        recorded_at=_T1,
        principal=(_ALICE,),
        node_rows=(("replay-direct", "active"),),
    )
    record = precedents.retrieve(
        overlays,
        patterns,
        _query(query_catalog, query_revision, imagined="img-new"),
        principal_labels=(_ALICE,),
        session_id="sess-reader",
        temporal_cutoff=_T2,
    )
    visible = {
        hit.fork_node_id
        for hit in (*record.high_confidence, *record.exploratory, *record.negative_retired)
    }
    assert produced.stored.fork_node_id in visible
    assert replay_commit.stored.fork_node_id not in visible
    assert hidden_commit.stored.fork_node_id not in visible
    with pytest.raises(ForkPrecedentError) as rejected:
        precedents.retrieve(
            overlays,
            patterns,
            _query(replay_catalog, replay, imagined="img-query"),
            principal_labels=(_ALICE,),
            session_id="sess-reader",
            temporal_cutoff=_T2,
        )
    assert rejected.value.code == "replay_provenance_excluded"


def test_failed_replay_result_does_not_freeze_or_append(tmp_path: Path) -> None:
    world = _world(tmp_path)
    before = world.native_path.read_bytes()
    assessment = world.service.assess(
        world.commit.stored,
        _controls(budget="exceeded", evaluator="unbound"),
        evidence_ref="fork-created",
        checked_at=_T0,
    )
    assert assessment.eligibility == "conditionally_replayable"
    request = world.service.build_replay_request(assessment, replay_run_id=_RUN)
    assert read_replay_request(request).eligibility == "conditionally_replayable"
    result = world.service.build_replay_result(request, status="blocked")
    outcome = world.service.apply_scripted_replay(
        world.commit.stored,
        request,
        result,
        blocks=[],
        edges=(),
        coverage=CoverageManifest(
            total_ao_range=AORange(start_seq=1, end_seq=1),
            inspected_refs=(),
            included_ranges=(),
            omitted_ranges=(),
            snapshot_watermark="unused",
        ),
        frozen_at=_FROZEN,
        pattern_store=world.patterns,
        pattern_id=world.pattern.pattern_id,
    )
    assert outcome.replay_aig is None
    assert outcome.admission is None
    assert outcome.evaluations == ()
    assert world.native_path.read_bytes() == before
    assert list((tmp_path / "replay-alignment").glob("aigs/*")) == []


def test_module_names_the_coforge_worker_as_a_later_boundary() -> None:
    import openviking.session.aig_native_protocol as protocol
    import openviking.session.replay_alignment as replay_alignment

    source = Path(replay_alignment.__file__).read_text(encoding="utf-8")
    assert "Coforge sandbox replay worker" in source
    assert "P09" in source and "P12" in source
    for banned in ("import socket", "import urllib", "import httpx", "import requests", "openai"):
        assert banned not in source
    protocol_source = Path(protocol.__file__).read_text(encoding="utf-8")
    assert "Coforge sandbox replay worker" in protocol_source
    assert OBJECT_KIND_REPLAY_ELIGIBILITY_ASSESSMENT == "replay_eligibility_assessment"
    assert OBJECT_KIND_DIRECT_EVIDENCE_ADMISSION == "direct_evidence_admission"
