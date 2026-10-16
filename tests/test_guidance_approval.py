# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""P08: Reader guidance proposals, frozen approval, and target selection.

Only a frozen Reader report that OutcomeEvidencePolicy marks as confirmed
failure can propose auto-executable guidance, and only when that proposal
cites a server precedent snapshot. The server freezes the GuidanceDocument
and the selected agent. Memory can publish that snapshot and cannot add
actions or rewrite the target. This module does not start a remediation run.
"""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openviking.session.aig_native_protocol import (
    SCHEMA_VERSION_AIG_NATIVE_V2,
    AgentCandidate,
    AgentCandidateCatalog,
    ApplicabilityCheck,
    GuidedCounterfactualBranch,
    ProtocolContractError,
    guidance_approval_document,
    protocol_content_hash,
)
from openviking.session.aig_role_chain import AigConstructionRun, AigContractError
from openviking.session.fork_node_overlay import ForkNodeOverlayStore, ForkOverlayError
from openviking.session.fork_precedent import ForkPrecedentError, ForkPrecedentService
from openviking.session.guidance_approval import (
    GUIDANCE_APPROVAL_DIRNAME,
    GuidanceApprovalError,
    GuidanceApprovalService,
    select_target_agent,
)
from openviking.session.outcome_contract import (
    DECISION_CONFIRMED_FAILURE,
    DECISION_SUCCESS,
    OutcomeAssessmentService,
    OutcomeContractError,
    OutcomeContractStore,
)
from openviking.session.strategy_pattern import StrategyPatternStore
from tests.test_fork_node_overlay import _native_revision, _proposal
from tests.test_outcome_contract import (
    _EVIDENCE,
    _assessment_body,
    _contract_body,
    _document,
    _frozen_run,
    _signal,
)

_CAROL = "user:carol"
_BOB = "user:bob"
_CUTOFF = "2026-10-10T00:00:01.000Z"
_PUBLISHED = "2026-10-10T00:00:03.000Z"
_APPLICABILITY = {
    "action_skill_role": "applicable",
    "anchor_state": "partially_applicable",
    "failure_mode": "applicable",
    "task_decision_type": "applicable",
}


def _code(call) -> str:
    try:
        call()
    except GuidanceApprovalError as exc:
        return exc.code
    except (AigContractError, OutcomeContractError, ForkOverlayError, ForkPrecedentError) as exc:
        return exc.code
    raise AssertionError("expected a coded error")


def _step(
    evidence: dict | None = None,
    *,
    step_id: str = "step-retry",
    imagined: str = "img-narrow-retry",
    statement: str = "retry the call with a narrower timeout",
) -> dict:
    return {
        "evidence_refs": [copy.deepcopy(evidence or _EVIDENCE)],
        "imagined_node_id": imagined,
        "statement": statement,
        "step_id": step_id,
    }


def _candidate(agent_id: str = "agent-a", **overrides: object) -> dict:
    item: dict = {
        "acl_labels": [_CAROL],
        "agent_id": agent_id,
        "available": True,
        "capabilities": ["replay-sandbox", "rollback"],
        "roles": ["task_agent"],
        "task_member": True,
    }
    item.update(overrides)
    return item


def _world(
    tmp_path: Path,
    *,
    verdict: str = "failure",
    retrieve_principal: tuple[str, ...] = (_CAROL,),
    submit_principal: tuple[str, ...] = (_CAROL,),
    acl: tuple[str, ...] = (_CAROL,),
):
    run, sessions, report_hash = _frozen_run()
    outcomes = OutcomeAssessmentService(OutcomeContractStore(tmp_path))
    contract = outcomes.freeze_contract(_document("outcome_contract", _contract_body()))
    outcomes.submit_assessment(
        run,
        sessions["reader"],
        _document(
            "task_outcome_assessment",
            _assessment_body(report_hash, verdict=verdict),
        ),
        task_id=contract.task_id,
        contract_id=contract.contract_id,
    )
    overlays = ForkNodeOverlayStore(tmp_path)
    patterns = StrategyPatternStore(tmp_path)
    precedents = ForkPrecedentService(tmp_path)
    revision = _native_revision("rev-guidance-1")
    base = overlays.register_base_revision(revision, acl_labels=acl)
    fork_payload = _proposal(base, document_hash=revision.content_hash)
    stored = overlays.submit_proposal(fork_payload, principal_labels=submit_principal)
    retrieval = precedents.retrieve(
        overlays,
        patterns,
        fork_payload,
        principal_labels=retrieve_principal,
        session_id=sessions["reader"],
        temporal_cutoff=_CUTOFF,
    )
    service = GuidanceApprovalService(
        tmp_path,
        outcomes=outcomes,
        precedents=precedents,
        overlays=overlays,
    )
    return SimpleNamespace(
        run=run,
        sessions=sessions,
        report_hash=report_hash,
        contract=contract,
        base=base,
        stored=stored,
        fork_node_id=stored.stored.fork_node_id,
        fork_payload=fork_payload,
        retrieval=retrieval,
        precedents=precedents,
        patterns=patterns,
        service=service,
        overlays=overlays,
    )


def _branch(world, **overrides: object) -> dict:
    body: dict = {
        "applicability": copy.deepcopy(_APPLICABILITY),
        "author_role": "reader",
        "base_revision_id": world.base.base_revision_id,
        "report_hash": world.report_hash,
        "required_capabilities": ["replay-sandbox"],
        "retrieval_id": world.retrieval.retrieval_id,
        "risk": "retry may repeat the failed rollout",
        "source_fork_id": world.fork_node_id,
        "steps": [_step()],
        "target_role": "task_agent",
        "task_id": world.contract.task_id,
    }
    body.update(overrides)
    return _document("guided_counterfactual_branch", body)


def _catalog(world, candidates: list[dict]) -> dict:
    return _document(
        "agent_candidate_catalog",
        {"candidates": candidates, "task_id": world.contract.task_id},
    )


def _files(tmp_path: Path, kind: str) -> list[Path]:
    directory = tmp_path / GUIDANCE_APPROVAL_DIRNAME / kind
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def _publish(service, approval_hash: str, **kwargs: object):
    render = kwargs.pop("render", None)
    return service.publish(
        approval_hash,
        principal_labels=kwargs.pop("principal_labels", (_CAROL,)),
        published_at=kwargs.pop("published_at", _PUBLISHED),
        render=render,
    )


def _propose(world, payload: dict | None = None, evidence: tuple = (_signal("fail"),)):
    return world.service.propose(
        world.run,
        world.sessions["reader"],
        payload if payload is not None else _branch(world),
        principal_labels=(_CAROL,),
        external_evidence=evidence,
    )


def test_only_confirmed_failure_with_a_precedent_can_propose(tmp_path: Path) -> None:
    world = _world(tmp_path)
    assert world.precedents.read_retrieval(world.retrieval.retrieval_id) == world.retrieval
    committed = _propose(world)
    again = _propose(world)
    assert committed.decision == DECISION_CONFIRMED_FAILURE
    assert committed.proposal.author_role == "reader"
    assert again.deduplicated is True
    assert again.proposal == committed.proposal
    assert len(_files(tmp_path, "proposals")) == 1

    success = _world(tmp_path / "success", verdict="success")
    assert _code(lambda: _propose(success, evidence=(_signal("pass"),))) == (
        "outcome_not_confirmed_failure"
    )
    assert _files(tmp_path / "success", "proposals") == []

    uncertain = _world(tmp_path / "uncertain")
    assert _code(lambda: _propose(uncertain, evidence=())) == "outcome_not_confirmed_failure"
    assert _files(tmp_path / "uncertain", "proposals") == []

    missing = _branch(world, retrieval_id="ab" * 32)
    assert _code(lambda: _propose(world, missing)) == "retrieval_required"
    other_snapshot = world.precedents.retrieve(
        world.overlays,
        world.patterns,
        world.fork_payload,
        principal_labels=(_CAROL,),
        session_id="sess-other",
        temporal_cutoff=_CUTOFF,
    )
    mismatched = _branch(world, retrieval_id=other_snapshot.retrieval_id)
    assert _code(lambda: _propose(world, mismatched)) == "retrieval_mismatch"
    unknown_fork = _branch(world, source_fork_id="fork-missing")
    assert _code(lambda: _propose(world, unknown_fork)) == "source_fork_unknown"
    assert (
        _code(
            lambda: world.service.propose(
                world.run,
                world.sessions["reader"],
                _branch(world),
                principal_labels=(),
                external_evidence=(_signal("fail"),),
            )
        )
        == "acl_denied"
    )
    assert (
        _code(lambda: _propose(world, evidence=(_signal("fail", evidence_ref="evidence:other"),)))
        == "adjudication_conflict"
    )
    assert len(_files(tmp_path, "proposals")) == 1


def test_builder_critic_refiner_and_memory_cannot_propose(tmp_path: Path) -> None:
    world = _world(tmp_path)
    payload = _branch(world)
    for role in ("builder", "critic", "refiner"):
        assert (
            _code(
                lambda role=role: world.service.propose(
                    world.run,
                    world.sessions[role],
                    payload,
                    principal_labels=(_CAROL,),
                    external_evidence=(_signal("fail"),),
                )
            )
            == "outcome_write_forbidden"
        )
    forged = _branch(world, author_role="memory")
    with pytest.raises(ProtocolContractError, match="author_role"):
        world.service.propose(
            world.run,
            world.sessions["reader"],
            forged,
            principal_labels=(_CAROL,),
            external_evidence=(_signal("fail"),),
        )
    bare = AigConstructionRun("run-open")
    sessions = {role: bare.session_for(role) for role in ("builder", "critic", "refiner", "reader")}
    bare.submit_draft(sessions["builder"], {"graph": []})
    bare.critique(sessions["critic"], [{"code": "gap", "severity": "high"}])
    bare.refine(sessions["refiner"], {"graph": ["n"]})
    bare.read_verify(sessions["reader"], {"verdict": "supported"})
    assert (
        _code(
            lambda: world.service.propose(
                bare,
                sessions["reader"],
                payload,
                principal_labels=(_CAROL,),
                external_evidence=(_signal("fail"),),
            )
        )
        == "report_not_frozen"
    )
    assert _files(tmp_path, "proposals") == []
    with pytest.raises(AigContractError, match="memory_session_not_shared"):
        bare.session_for("memory")


def test_server_freezes_the_selected_agent_without_widening(tmp_path: Path) -> None:
    world = _world(tmp_path)
    proposed = _propose(world)
    catalog = _catalog(world, [_candidate("agent-a")])
    first = world.service.approve(
        proposed.proposal.content_hash,
        catalog,
        principal_labels=(_CAROL,),
    )
    second = world.service.approve(
        proposed.proposal.content_hash,
        catalog,
        principal_labels=(_CAROL,),
    )
    assert first.deduplicated is False
    assert second.deduplicated is True
    assert second.approval == first.approval
    assert first.approval.selected_agent_id == "agent-a"
    assert first.approval.target_role == "task_agent"
    assert first.approval.capability_limits == ("replay-sandbox",)
    assert first.approval.steps == proposed.proposal.steps
    assert first.guidance.capability_limits == ("replay-sandbox",)
    assert first.guidance.target_role == "task_agent"
    assert first.guidance.content_hash == first.approval.guidance_content_hash
    assert "steps" not in inspect.signature(guidance_approval_document).parameters
    stored = json.loads(
        (
            tmp_path
            / GUIDANCE_APPROVAL_DIRNAME
            / "guidance"
            / f"{first.guidance.content_hash}.json"
        ).read_text(encoding="utf-8")
    )
    assert "selected_agent_id" not in stored["body"]
    assert "steps" not in stored["body"]
    assert stored["object_kind"] == "guidance_document"
    other_task = _document(
        "agent_candidate_catalog",
        {"candidates": [_candidate("agent-a")], "task_id": "task-other"},
    )
    assert (
        _code(
            lambda: world.service.approve(
                proposed.proposal.content_hash,
                other_task,
                principal_labels=(_CAROL,),
            )
        )
        == "catalog_mismatch"
    )

    narrow = _world(tmp_path / "capability")
    held = _propose(
        narrow,
        _branch(narrow, required_capabilities=["replay-sandbox", "rollback"]),
    )
    assert (
        _code(
            lambda: narrow.service.approve(
                held.proposal.content_hash,
                _catalog(narrow, [_candidate("agent-a", capabilities=["replay-sandbox"])]),
                principal_labels=(_CAROL,),
            )
        )
        == "no_eligible_target"
    )
    assert (
        _code(
            lambda: narrow.service.approve(
                held.proposal.content_hash,
                _catalog(narrow, [_candidate("agent-a", acl_labels=[_BOB])]),
                principal_labels=(_CAROL,),
            )
        )
        == "no_eligible_target"
    )
    assert _files(tmp_path / "capability", "approvals") == []


def test_reselection_writes_a_new_version(tmp_path: Path) -> None:
    world = _world(tmp_path)
    proposed = _propose(world)
    first_catalog = _catalog(world, [_candidate("agent-a")])
    first = world.service.approve(
        proposed.proposal.content_hash,
        first_catalog,
        principal_labels=(_CAROL,),
    )
    before = (
        tmp_path / GUIDANCE_APPROVAL_DIRNAME / "approvals" / f"{first.approval.content_hash}.json"
    ).read_bytes()
    second_catalog = _catalog(
        world,
        [
            _candidate("agent-a", available=False),
            _candidate("agent-b"),
        ],
    )
    second = world.service.approve(
        proposed.proposal.content_hash,
        second_catalog,
        principal_labels=(_CAROL,),
    )
    assert second.approval.guidance_id != first.approval.guidance_id
    assert second.approval.selected_agent_id == "agent-b"
    assert second.approval.content_hash != first.approval.content_hash
    assert (
        tmp_path / GUIDANCE_APPROVAL_DIRNAME / "approvals" / f"{first.approval.content_hash}.json"
    ).read_bytes() == before
    assert world.service.read_approval(first.approval.content_hash).selected_agent_id == "agent-a"
    assert len(_files(tmp_path, "approvals")) == 2


def test_selector_fail_closed_does_not_rank_or_relax() -> None:
    proposal = GuidedCounterfactualBranch(
        schema_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        object_kind="guided_counterfactual_branch",
        content_hash="ab" * 32,
        author_role="reader",
        report_hash="cd" * 32,
        task_id="task-1",
        source_fork_id="fork-1",
        base_revision_id="rev-1",
        retrieval_id="ef" * 32,
        target_role="task_agent",
        required_capabilities=("replay-sandbox", "rollback"),
        applicability=ApplicabilityCheck(
            task_decision_type="applicable",
            anchor_state="applicable",
            failure_mode="applicable",
            action_skill_role="applicable",
        ),
        risk="retry may repeat the failed rollout",
        steps=(),
    )
    principal = frozenset({_CAROL})

    def candidate(agent_id: str, **overrides: object) -> AgentCandidate:
        values: dict = {
            "agent_id": agent_id,
            "roles": ("task_agent",),
            "capabilities": ("replay-sandbox", "rollback"),
            "acl_labels": (_CAROL,),
            "task_member": True,
            "available": True,
        }
        values.update(overrides)
        return AgentCandidate(**values)

    def choose(candidates: tuple[AgentCandidate, ...]) -> str:
        catalog = AgentCandidateCatalog(
            schema_version=SCHEMA_VERSION_AIG_NATIVE_V2,
            object_kind="agent_candidate_catalog",
            content_hash="11" * 32,
            task_id="task-1",
            candidates=candidates,
        )
        return select_target_agent(proposal, catalog, principal)

    assert choose((candidate("agent-b"), candidate("agent-a"))) == "agent-a"
    for broken in (
        candidate("agent-a", capabilities=("replay-sandbox",)),
        candidate("agent-a", acl_labels=(_BOB,)),
        candidate("agent-a", task_member=False),
        candidate("agent-a", available=False),
        candidate("agent-a", roles=()),
    ):
        assert _code(lambda broken=broken: choose((broken,))) == "no_eligible_target"


def test_memory_publishes_only_the_approved_snapshot(tmp_path: Path) -> None:
    world = _world(tmp_path)
    proposed = _propose(world)
    assert _code(lambda: _publish(world.service, "ab" * 32)) == "approval_required"
    approval = world.service.approve(
        proposed.proposal.content_hash,
        _catalog(world, [_candidate("agent-a")]),
        principal_labels=(_CAROL,),
    )
    published = _publish(world.service, approval.approval.content_hash)
    again = _publish(world.service, approval.approval.content_hash)
    assert published.publication.publisher_labels == (_CAROL,)
    assert published.publication.published_at == _PUBLISHED
    envelope = published.envelope
    assert again.deduplicated is True
    assert again.publication == published.publication
    assert published.publication.selected_agent_id == "agent-a"
    assert published.publication.steps == approval.approval.steps
    assert published.publication.decision == DECISION_CONFIRMED_FAILURE
    assert published.publication.guidance_content_hash == approval.guidance.content_hash
    assert not hasattr(published.publication, "child_run_id")
    assert _files(tmp_path, "executions") == []
    kinds = {
        json.loads(path.read_text(encoding="utf-8")).get("object_kind")
        for path in (tmp_path / GUIDANCE_APPROVAL_DIRNAME).rglob("*.json")
    }
    assert "guidance_execution_report" not in kinds

    def mutated(field: str, value: object, *, reproject: bool = False) -> dict:
        body = copy.deepcopy(envelope["body"])
        body[field] = value
        if reproject:
            projected = {
                "applicability": body["applicability"],
                "base_revision_id": body["base_revision_id"],
                "capability_limits": body["capability_limits"],
                "evidence_refs": body["evidence_refs"],
                "guidance_id": body["guidance_id"],
                "report_hash": body["report_hash"],
                "risk": body["risk"],
                "source_fork_id": body["source_fork_id"],
                "target_role": body["target_role"],
            }
            body["guidance_content_hash"] = protocol_content_hash(
                SCHEMA_VERSION_AIG_NATIVE_V2, "guidance_document", projected
            )
        return _document("guidance_publication", body)

    extra = _step(step_id="step-extra", imagined="img-extra", statement="add an unproposed retry")
    approval_bytes = (
        tmp_path
        / GUIDANCE_APPROVAL_DIRNAME
        / "approvals"
        / f"{approval.approval.content_hash}.json"
    ).read_bytes()
    assert (
        _code(
            lambda: _publish(
                world.service,
                approval.approval.content_hash,
                render=mutated("steps", list(envelope["body"]["steps"]) + [extra]),
            )
        )
        == "action_not_in_proposal"
    )
    assert (
        _code(
            lambda: _publish(
                world.service,
                approval.approval.content_hash,
                render=mutated("selected_agent_id", "agent-b"),
            )
        )
        == "target_rewrite_forbidden"
    )
    assert (
        _code(
            lambda: _publish(
                world.service,
                approval.approval.content_hash,
                render=mutated(
                    "capability_limits",
                    ["replay-sandbox", "rollback"],
                    reproject=True,
                ),
            )
        )
        == "unevidenced_action"
    )
    assert (
        _code(
            lambda: _publish(
                world.service,
                approval.approval.content_hash,
                render=mutated("reason", "reader_inference_only"),
            )
        )
        == "basis_rewrite_forbidden"
    )
    with pytest.raises(ProtocolContractError, match="confirmed_failure"):
        _publish(
            world.service,
            approval.approval.content_hash,
            render=mutated("decision", DECISION_SUCCESS),
        )
    assert (
        tmp_path
        / GUIDANCE_APPROVAL_DIRNAME
        / "approvals"
        / f"{approval.approval.content_hash}.json"
    ).read_bytes() == approval_bytes
    assert (
        world.service.read_publication(approval.approval.content_hash, principal_labels=(_CAROL,))
        == published.publication
    )


def test_rejected_and_unapproved_proposals_cannot_be_published(tmp_path: Path) -> None:
    world = _world(tmp_path)
    proposed = _propose(world)
    rejection = world.service.reject(proposed.proposal.content_hash)
    assert rejection.reason == "guidance_rejected"
    assert world.service.reject(proposed.proposal.content_hash) == rejection
    assert (
        _code(
            lambda: world.service.approve(
                proposed.proposal.content_hash,
                _catalog(world, [_candidate("agent-a")]),
                principal_labels=(_CAROL,),
            )
        )
        == "proposal_rejected"
    )
    assert (
        _code(lambda: _publish(world.service, proposed.proposal.content_hash))
        == "approval_required"
    )
    assert _files(tmp_path, "approvals") == []

    other = _world(tmp_path / "kept")
    held = _propose(other)
    frozen = other.service.approve(
        held.proposal.content_hash,
        _catalog(other, [_candidate("agent-a")]),
        principal_labels=(_CAROL,),
    )
    _publish(other.service, frozen.approval.content_hash)
    assert _code(lambda: other.service.reject(held.proposal.content_hash)) == "approval_frozen"
    assert (
        other.service.read_publication(frozen.approval.content_hash, principal_labels=(_CAROL,))
        is not None
    )


def test_publish_records_the_principal_and_caller_timestamp(tmp_path: Path) -> None:
    world = _world(tmp_path)
    proposed = _propose(world)
    approval = world.service.approve(
        proposed.proposal.content_hash,
        _catalog(world, [_candidate("agent-a")]),
        principal_labels=(_CAROL,),
    )
    content_hash = approval.approval.content_hash
    assert _code(lambda: _publish(world.service, content_hash, principal_labels=())) == (
        "publisher_required"
    )
    assert (
        _code(lambda: _publish(world.service, content_hash, principal_labels=(" user:carol",)))
        == "publisher_required"
    )
    assert _code(lambda: _publish(world.service, content_hash, published_at="")) == (
        "published_at_required"
    )
    assert _code(lambda: _publish(world.service, content_hash, published_at="now")) == (
        "published_at_required"
    )
    first = _publish(world.service, content_hash)
    assert (
        _code(
            lambda: _publish(world.service, content_hash, published_at="2026-10-10T00:00:04.000Z")
        )
        == "publication_mismatch"
    )
    assert (
        world.service.read_publication(content_hash, principal_labels=(_CAROL,))
        == first.publication
    )
    second = _publish(
        world.service,
        content_hash,
        principal_labels=(_BOB,),
        published_at="2026-10-10T00:00:04.000Z",
    )
    assert second.deduplicated is False
    assert second.publication.publisher_labels == (_BOB,)
    assert second.publication.published_at == "2026-10-10T00:00:04.000Z"
    assert second.publication.content_hash != first.publication.content_hash
    assert second.publication.approval_content_hash == content_hash
    assert (
        world.service.read_publication(content_hash, principal_labels=(_CAROL,))
        == first.publication
    )
    assert len(_files(tmp_path, "publications")) == 2


def test_illegal_outcome_evidence_has_its_own_code(tmp_path: Path) -> None:
    world = _world(tmp_path)
    assert (
        _code(
            lambda: world.service.propose(
                world.run,
                world.sessions["reader"],
                _branch(world),
                principal_labels=(_CAROL,),
                external_evidence="fail",
            )
        )
        == "outcome_evidence_invalid"
    )
    assert (
        _code(
            lambda: world.service.propose(
                world.run,
                world.sessions["reader"],
                _branch(world),
                principal_labels=(_CAROL,),
                external_evidence=("not-evidence",),
            )
        )
        == "outcome_evidence_invalid"
    )
    assert _files(tmp_path, "proposals") == []
