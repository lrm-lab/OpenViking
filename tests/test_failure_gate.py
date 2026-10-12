# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""P10: post-commit failure gate and automatic diagnosis orchestration.

Commit and ``@memory diagnose`` share one construction pipeline. Only a
confirmed failure creates a Fork, Guidance, and a scripted remediation run.
The same commit, trigger, and input fingerprint replays. A different trigger
or fingerprint is a new orchestration version and does not duplicate those
objects. No model, network, or Coforge worker is called.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from openviking.session.aig_native_protocol import SCHEMA_VERSION_AIG_NATIVE_V2, protocol_content_hash
from openviking.session.aig_role_chain import AigConstructionRun, AigContractError
from openviking.session.coordinator_boundary import (
    TRIGGER_COMMIT,
    TRIGGER_DIAGNOSE,
    coordinator_diagnose_request_document,
)
from openviking.session.extraction_write import freeze_native_edge_view
from openviking.session.failure_gate import (
    AIG_CONSTRUCTION_PIPELINE_VERSION,
    FRESHNESS_RECORD_VERSION,
    ACTION_ADVICE,
    ACTION_DISPATCHED,
    ACTION_RETAINED,
    AUDIT_DIAGNOSE_REQUESTED,
    AUDIT_REPLAYED,
    AUDIT_REFUSED,
    AUDIT_VIEW_CONSTRUCTED,
    AUDIT_VIEW_REUSED,
    FailureGateDelivery,
    FailureGateError,
    FailureGateOrchestrator,
    bind_role_chain_authorizer,
    view_is_fresh,
)
from openviking.session.fork_node_overlay import FORK_NODE_OVERLAY_DIRNAME
from openviking.session.guidance_approval import GUIDANCE_APPROVAL_DIRNAME, GuidanceApprovalService
from openviking.session.influence_view import (
    PURPOSE_POST_RUN_INDEX,
    AORange,
    AigNativeEdge,
    CoverageManifest,
    InfluenceBlock,
    SegmentRef,
    SourceRef,
)
from openviking.session.outcome_contract import (
    DECISION_CONFIRMED_FAILURE,
    DECISION_SUCCESS,
    DECISION_UNCERTAIN,
    REASON_CRITERIA_NOT_EXPLICIT,
    REASON_EXTERNAL_FAIL,
    REASON_EXTERNAL_PASS,
    OutcomeContractError,
)
from openviking.session.remediation_dispatch import REMEDIATION_RUN_DIRNAME
from openviking.session.strategy_pattern import StrategyPatternStore

_CAROL = "user:carol"
_CUTOFF = "2026-10-10T00:00:01.000Z"
_RECORDED = "2026-10-10T00:00:02.000Z"
_FROZEN = "2026-10-10T00:00:00.000Z"
_EVIDENCE = {
    "ao_id": "ao-eval-1",
    "archive_commit_watermark": None,
    "captured_state": "committed",
    "evidence_content_hash": "hash-eval-1",
    "evidence_role": "hindsight_attribution",
    "source_archive_id": None,
    "source_read_snapshot_watermark": "wm-1",
    "source_sequence": 4,
    "source_session_id": "sess-1",
}
_APPLICABILITY = {
    "action_skill_role": "applicable",
    "anchor_state": "partially_applicable",
    "failure_mode": "applicable",
    "task_decision_type": "applicable",
}


def _document(kind: str, body: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
        "object_kind": kind,
        "body": body,
        "content_hash": protocol_content_hash(SCHEMA_VERSION_AIG_NATIVE_V2, kind, body),
    }


def _contract(task_id: str) -> dict:
    return _document(
        "outcome_contract",
        {
            "acceptance_criteria": [
                {"criterion_id": "crit-error-budget", "statement": "error rate stays under 1%"}
            ],
            "contract_id": f"contract-{task_id}",
            "evaluator_bindings": [
                {"binding_id": "bind-eval-1", "kind": "evaluator", "ref": "eval://error-budget"}
            ],
            "frozen_at": _FROZEN,
            "goals": ["ship the rollback within the error budget"],
            "policy_constraints": [
                {"constraint_id": "sla-latency", "kind": "sla", "statement": "p99 under 200ms"}
            ],
            "source": "orchestrator",
            "task_id": task_id,
        },
    )


def _signal(signal: str) -> dict:
    return {
        "binding_id": "bind-eval-1",
        "evidence_ref": "evidence:eval-1",
        "signal": signal,
        "signal_kind": "evaluator",
    }


def _ref(session_id: str, sequence: int) -> SourceRef:
    return SourceRef(
        ao_id=f"ao-{sequence}",
        sequence=sequence,
        participant_id="agent-a",
        segment_id="seg-a",
        session_id=session_id,
        content_hash=f"hash-{sequence}",
    )


def _block(session_id: str, block_id: str, sequence: int) -> InfluenceBlock:
    source = _ref(session_id, sequence)
    return InfluenceBlock(
        block_id=block_id,
        role="agent_action",
        participant_id="agent-a",
        segment_ref=SegmentRef(segment_id="seg-a", session_id=session_id),
        ao_start_seq=sequence,
        ao_end_seq=sequence,
        summary=block_id,
        input="in",
        output="out",
        authorship="agent-a",
        granularity_reason="one tool call",
        source_content_hashes=(source.content_hash,),
        source_refs=(source,),
        block_kind="tool_call",
    )


def _revision(request):
    blocks = [_block(request.session_id, "b-1", 1), _block(request.session_id, "b-2", 2)]
    refs = tuple(ref for block in blocks for ref in block.source_refs)
    coverage = CoverageManifest(
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
        snapshot_watermark="wm-failure-gate",
    )
    return freeze_native_edge_view(
        session_id=request.session_id,
        task_run_id=request.task_id,
        view_id=f"view-{request.task_id}",
        revision_id=f"rev-{request.task_id}",
        frozen_at=_FROZEN,
        blocks=blocks,
        edges=(
            AigNativeEdge(
                edge_id="e-1",
                source_block_id="b-1",
                target_block_id="b-2",
                relation_type="influence",
                mechanism="later block inherits the earlier result",
            ),
        ),
        coverage=coverage,
        purpose=PURPOSE_POST_RUN_INDEX,
    )


def _orchestrator(tmp_path: Path, calls: list):
    def build(request):
        calls.append((request.pipeline_version, request.task_id, request.source_digest))
        return _revision(request)

    return FailureGateOrchestrator(tmp_path, build_revision=build)


def _plan() -> dict:
    return {
        "applicability": dict(_APPLICABILITY),
        "required_capabilities": ["replay-sandbox"],
        "risk": "retry may repeat the failed rollout",
        "steps": [
            {
                "evidence_refs": [dict(_EVIDENCE)],
                "imagined_node_id": "img-retry",
                "statement": "retry the call with a narrower timeout",
                "step_id": "step-retry",
            }
        ],
        "target_role": "task_agent",
    }


def _catalog() -> dict:
    return {
        "candidates": [
            {
                "acl_labels": [_CAROL],
                "agent_id": "agent-a",
                "available": True,
                "capabilities": ["replay-sandbox", "rollback"],
                "roles": ["task_agent"],
                "task_member": True,
            }
        ]
    }


def _controls(**overrides: object) -> dict:
    body: dict = {
        "agent_capabilities": ["replay-sandbox"],
        "budget": "compliant",
        "executor_agent_id": "agent-a",
        "executor_role": "task_agent",
        "idempotency_key": "caller-placeholder",
        "isolation": "isolated",
        "permission": "granted",
        "requested_side_effects": [],
        "side_effect_allowlist": [],
    }
    body.update(overrides)
    return body


def _delivery(
    *,
    task_id: str,
    commit_id: str,
    source_digest: str,
    verdict: str,
    dispositions: str,
    signal: str | None,
    trigger: str = TRIGGER_COMMIT,
    script: bool = False,
    controls: dict | None = None,
    anchor_block_id: str | None = None,
    evaluator_results: tuple | None = None,
    guidance_plan: dict | None = None,
) -> FailureGateDelivery:
    evidence = () if signal is None else (_signal(signal),)
    if evaluator_results is not None:
        evidence = evaluator_results
    return FailureGateDelivery(
        trigger=trigger,
        commit_id=commit_id,
        task_id=task_id,
        session_id=f"sess-{task_id}",
        source_digest=source_digest,
        principal_labels=(_CAROL,),
        contract=_contract(task_id),
        assessment_body={
            "criteria_dispositions": {"crit-error-budget": dispositions},
            "evidence_refs": [dict(_EVIDENCE)],
            "verdict": verdict,
        },
        evaluator_results=evidence,
        temporal_cutoff=_CUTOFF,
        recorded_at=_RECORDED,
        anchor_block_id=(anchor_block_id if anchor_block_id is not None else "b-2")
        if script
        else None,
        branch_steps=(
            (
                {"block_id": "b-1", "step_kind": "factual_block_ref"},
                {"imagined_node_id": "img-retry", "step_kind": "imagined_node"},
            )
            if script
            else ()
        ),
        guidance_plan=guidance_plan if guidance_plan is not None else (_plan() if script else {"ignored": True}),
        catalog=_catalog() if script else None,
        dispatch_controls=_controls() if controls is None and script else controls,
        dispatch_outputs=(
            ({"output_id": "ao-child-1", "output_kind": "ao"}, {"output_id": "blk-child-1", "output_kind": "block"})
            if script
            else ()
        ),
        dispatch_steps=(
            (
                {
                    "actual_block_id": "blk-child-1",
                    "alignment": "equivalent",
                    "imagined_node_id": "img-retry",
                },
            )
            if script
            else ()
        ),
    )


def _json_count(root: Path, *parts: str) -> int:
    directory = root.joinpath(*parts)
    if not directory.is_dir():
        return 0
    return len(list(directory.glob("*.json")))


def _forks(root: Path) -> int:
    return _json_count(root, FORK_NODE_OVERLAY_DIRNAME, "overlays")


def _guidance(root: Path) -> int:
    return _json_count(root, GUIDANCE_APPROVAL_DIRNAME, "approvals")


def _runs(root: Path) -> int:
    return _json_count(root, REMEDIATION_RUN_DIRNAME, "runs")


def _diagnose(orchestrator: FailureGateOrchestrator, delivery: FailureGateDelivery):
    document = coordinator_diagnose_request_document(
        commit_id=delivery.commit_id,
        session_id=delivery.session_id,
        task_id=delivery.task_id,
        source_digest=delivery.source_digest,
        principal_labels=delivery.principal_labels,
    )
    diagnosed = FailureGateDelivery(
        trigger=TRIGGER_DIAGNOSE,
        commit_id=delivery.commit_id,
        task_id=delivery.task_id,
        session_id=delivery.session_id,
        source_digest=delivery.source_digest,
        principal_labels=delivery.principal_labels,
        contract=delivery.contract,
        assessment_body=delivery.assessment_body,
        evaluator_results=delivery.evaluator_results,
        temporal_cutoff=delivery.temporal_cutoff,
        recorded_at=delivery.recorded_at,
        anchor_block_id=delivery.anchor_block_id,
        branch_steps=delivery.branch_steps,
        guidance_plan=delivery.guidance_plan,
        catalog=delivery.catalog,
        dispatch_controls=delivery.dispatch_controls,
        dispatch_outputs=delivery.dispatch_outputs,
        dispatch_steps=delivery.dispatch_steps,
        dispositions=delivery.dispositions,
    )
    return orchestrator.diagnose(document, diagnosed), document


def test_role_chain_authorizer_rejects_non_readers_and_returns_the_view_hash() -> None:
    run = AigConstructionRun("run-gate")
    sessions = {role: run.session_for(role) for role in ("builder", "critic", "refiner", "reader")}
    run.submit_draft(sessions["builder"], {"graph": []})
    run.critique(sessions["critic"], [])
    report_hash = run.read_verify(sessions["reader"], {"verdict": "supported"})
    view_hash = "ab" * 32
    authorizer = bind_role_chain_authorizer(
        run, reader_report_hash=report_hash, view_content_hash=view_hash
    )
    with pytest.raises(AigContractError) as unfrozen:
        authorizer(sessions["reader"])
    assert unfrozen.value.code == "report_not_frozen"
    assert run.freeze_report(sessions["reader"]) == report_hash
    assert authorizer(sessions["reader"]) == view_hash
    for role in ("builder", "critic", "refiner"):
        with pytest.raises(AigContractError) as denied:
            authorizer(sessions[role])
        assert denied.value.code == "outcome_write_forbidden"
    snapshot = run.frozen_snapshot()
    restored = AigConstructionRun.restore_frozen(snapshot)
    restored_authorizer = bind_role_chain_authorizer(
        restored, reader_report_hash=report_hash, view_content_hash=view_hash
    )
    assert restored_authorizer(snapshot.sessions["reader"]) == view_hash
    with pytest.raises(AigContractError) as restored_denied:
        restored_authorizer(snapshot.sessions["builder"])
    assert restored_denied.value.code == "outcome_write_forbidden"


def test_success_commit_and_diagnose_share_one_fresh_view_and_create_nothing(
    tmp_path: Path,
) -> None:
    calls: list = []
    orchestrator = _orchestrator(tmp_path, calls)
    delivery = _delivery(
        task_id="task-success",
        commit_id="commit-success",
        source_digest="aa" * 32,
        verdict="success",
        dispositions="explicit",
        signal="pass",
        script=False,
    )
    first = orchestrator.deliver(delivery)
    diagnosed, document = _diagnose(orchestrator, delivery)
    assert first.decision == DECISION_SUCCESS
    assert first.reason == REASON_EXTERNAL_PASS
    assert first.action == ACTION_RETAINED
    assert first.reused_view is False
    assert first.deduplicated is False
    assert first.fork_node_id is None
    assert first.guidance_content_hash is None
    assert first.child_run_id is None
    assert diagnosed.decision == DECISION_SUCCESS
    assert diagnosed.action == ACTION_RETAINED
    assert diagnosed.deduplicated is False
    assert diagnosed.orchestration_run_id != first.orchestration_run_id
    assert diagnosed.reused_view is True
    assert diagnosed.coordinator_event is not None
    assert diagnosed.coordinator_event["body"]["request_content_hash"] == document["content_hash"]
    assert diagnosed.coordinator_event["body"]["trigger"] == TRIGGER_DIAGNOSE
    assert calls == [(AIG_CONSTRUCTION_PIPELINE_VERSION, "task-success", "aa" * 32)]
    assert _forks(tmp_path) == 0
    assert _guidance(tmp_path) == 0
    assert _runs(tmp_path) == 0
    view = orchestrator.read_frozen_view(
        task_id="task-success", session_id="sess-task-success", source_digest="aa" * 32
    )
    assert view is not None
    assert view.content_hash == first.view_content_hash
    assert view.task_run_id == "task-success"
    kinds = [row["event_kind"] for row in orchestrator.audit_events()]
    assert AUDIT_VIEW_CONSTRUCTED in kinds
    assert AUDIT_VIEW_REUSED in kinds
    assert kinds.count(ACTION_RETAINED) == 2
    assert AUDIT_DIAGNOSE_REQUESTED in kinds
    assert AUDIT_REPLAYED not in kinds
    assert ACTION_ADVICE not in kinds


def test_confirmed_failure_dispatches_once_and_diagnose_reuses_the_view(tmp_path: Path) -> None:
    calls: list = []
    orchestrator = _orchestrator(tmp_path, calls)
    delivery = _delivery(
        task_id="task-failure",
        commit_id="commit-failure",
        source_digest="bb" * 32,
        verdict="failure",
        dispositions="explicit",
        signal="fail",
        script=True,
    )
    first = orchestrator.deliver(delivery)
    assert first.decision == DECISION_CONFIRMED_FAILURE
    assert first.reason == REASON_EXTERNAL_FAIL
    assert first.action == ACTION_DISPATCHED
    assert first.blocked is False
    assert first.fork_node_id
    assert first.guidance_content_hash
    assert first.child_run_id
    assert first.deduplicated is False
    assert StrategyPatternStore(tmp_path).has_direct_evidence("imagined_node", "img-retry") is True
    replay = orchestrator.deliver(
        _delivery(
            task_id="task-failure",
            commit_id="commit-failure",
            source_digest="bb" * 32,
            verdict="failure",
            dispositions="explicit",
            signal="fail",
            script=True,
            controls=_controls(idempotency_key="different-caller-key"),
        )
    )
    assert replay.deduplicated is True
    assert replay.reused_view is True
    assert replay.fork_node_id == first.fork_node_id
    assert replay.guidance_content_hash == first.guidance_content_hash
    assert replay.child_run_id == first.child_run_id
    diagnosed, _document = _diagnose(orchestrator, delivery)
    assert diagnosed.deduplicated is False
    assert diagnosed.orchestration_run_id != first.orchestration_run_id
    assert diagnosed.reused_view is True
    assert diagnosed.action == ACTION_DISPATCHED
    assert diagnosed.child_run_id == first.child_run_id
    assert diagnosed.guidance_content_hash == first.guidance_content_hash
    assert calls == [(AIG_CONSTRUCTION_PIPELINE_VERSION, "task-failure", "bb" * 32)]
    assert _forks(tmp_path) == 1
    assert _guidance(tmp_path) == 1
    assert _runs(tmp_path) == 1
    changed = _delivery(
        task_id="task-failure",
        commit_id="commit-failure",
        source_digest="bb" * 32,
        verdict="success",
        dispositions="explicit",
        signal="pass",
        script=True,
    )
    with pytest.raises(OutcomeContractError) as conflict:
        orchestrator.deliver(changed)
    assert conflict.value.code == "assessment_conflict"
    refused = [row for row in orchestrator.audit_events() if row.get("outcome") == AUDIT_REFUSED]
    assert refused[-1]["reason"] == "assessment_conflict"
    assert _forks(tmp_path) == 1
    assert _runs(tmp_path) == 1


def test_uncertain_diagnose_records_advice_and_does_not_dispatch(tmp_path: Path) -> None:
    calls: list = []
    orchestrator = _orchestrator(tmp_path, calls)
    delivery = _delivery(
        task_id="task-uncertain",
        commit_id="commit-uncertain",
        source_digest="cc" * 32,
        verdict="uncertain",
        dispositions="provisional",
        signal="fail",
        trigger=TRIGGER_DIAGNOSE,
        script=True,
    )
    diagnosed, document = _diagnose(orchestrator, delivery)
    assert diagnosed.decision == DECISION_UNCERTAIN
    assert diagnosed.reason == REASON_CRITERIA_NOT_EXPLICIT
    assert diagnosed.action == ACTION_ADVICE
    assert diagnosed.fork_node_id is None
    assert diagnosed.child_run_id is None
    assert diagnosed.coordinator_event["body"]["action"] == ACTION_ADVICE
    assert diagnosed.coordinator_event["body"]["decision"] == DECISION_UNCERTAIN
    assert diagnosed.coordinator_event["body"]["request_content_hash"] == document["content_hash"]
    replay = orchestrator.deliver(
        _delivery(
            task_id="task-uncertain",
            commit_id="commit-uncertain",
            source_digest="cc" * 32,
            verdict="uncertain",
            dispositions="provisional",
            signal="fail",
            script=True,
        )
    )
    assert replay.deduplicated is False
    assert replay.orchestration_run_id != diagnosed.orchestration_run_id
    assert replay.reused_view is True
    assert replay.action == ACTION_ADVICE
    assert replay.fork_node_id is None
    again, _again_document = _diagnose(orchestrator, delivery)
    assert again.deduplicated is True
    assert again.orchestration_run_id == diagnosed.orchestration_run_id
    assert calls == [(AIG_CONSTRUCTION_PIPELINE_VERSION, "task-uncertain", "cc" * 32)]
    assert _forks(tmp_path) == 0
    assert _guidance(tmp_path) == 0
    assert _runs(tmp_path) == 0
    kinds = [row["event_kind"] for row in orchestrator.audit_events()]
    assert kinds.count(ACTION_ADVICE) == 2
    assert AUDIT_DIAGNOSE_REQUESTED in kinds
    assert orchestrator.read_frozen_view(
        task_id="task-uncertain",
        session_id="sess-task-uncertain",
        source_digest="cc" * 32,
    )


def test_retry_after_a_mid_flight_failure_does_not_duplicate_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list = []
    orchestrator = _orchestrator(tmp_path, calls)
    delivery = _delivery(
        task_id="task-retry",
        commit_id="commit-retry",
        source_digest="dd" * 32,
        verdict="failure",
        dispositions="explicit",
        signal="fail",
        script=True,
    )
    original = GuidanceApprovalService.propose
    seen = {"n": 0}

    def wrapped(self, *args, **kwargs):
        seen["n"] += 1
        if seen["n"] == 1:
            raise RuntimeError("injected")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GuidanceApprovalService, "propose", wrapped)
    with pytest.raises(RuntimeError, match="injected"):
        orchestrator.deliver(delivery)
    refused = [row for row in orchestrator.audit_events() if row.get("outcome") == AUDIT_REFUSED]
    assert len(refused) == 1
    assert refused[0]["reason"] == "store_failed"
    assert _forks(tmp_path) == 1
    assert _guidance(tmp_path) == 0
    assert _runs(tmp_path) == 0
    finished = orchestrator.deliver(delivery)
    assert finished.action == ACTION_DISPATCHED
    assert finished.deduplicated is False
    assert finished.reused_view is True
    assert _forks(tmp_path) == 1
    assert _guidance(tmp_path) == 1
    assert _runs(tmp_path) == 1
    assert len(calls) == 1
    again = orchestrator.deliver(delivery)
    assert again.deduplicated is True
    assert again.child_run_id == finished.child_run_id
    assert again.fork_node_id == finished.fork_node_id
    assert _forks(tmp_path) == 1
    assert _guidance(tmp_path) == 1
    assert _runs(tmp_path) == 1


def test_a_tampered_freshness_record_fails_closed(tmp_path: Path) -> None:
    calls: list = []
    orchestrator = _orchestrator(tmp_path, calls)
    delivery = _delivery(
        task_id="task-fresh",
        commit_id="commit-fresh",
        source_digest="ee" * 32,
        verdict="success",
        dispositions="explicit",
        signal="pass",
    )
    first = orchestrator.deliver(delivery)
    stored = next((tmp_path / "failure-gate" / "views").glob("*.json"))
    record = json.loads(stored.read_text(encoding="utf-8"))
    assert view_is_fresh(
        record,
        session_id="sess-task-fresh",
        task_id="task-fresh",
        source_digest="ee" * 32,
    )
    assert record["freshness_version"] == FRESHNESS_RECORD_VERSION
    assert record["pipeline_version"] == AIG_CONSTRUCTION_PIPELINE_VERSION
    stale = json.loads(json.dumps(record))
    stale["trust"] = "provisional"
    assert view_is_fresh(
        stale,
        session_id="sess-task-fresh",
        task_id="task-fresh",
        source_digest="ee" * 32,
    ) is False
    other_digest = json.loads(json.dumps(record))
    other_digest["source_digest"] = "ff" * 32
    assert view_is_fresh(
        other_digest,
        session_id="sess-task-fresh",
        task_id="task-fresh",
        source_digest="ee" * 32,
    ) is False
    record["trust"] = "provisional"
    stored.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(FailureGateError) as tampered:
        orchestrator.deliver(delivery)
    assert tampered.value.code == "stored_record_invalid"
    assert len(calls) == 1
    assert first.view_content_hash


def test_a_failed_dispatch_gate_creates_no_child_run(tmp_path: Path) -> None:
    calls: list = []
    orchestrator = _orchestrator(tmp_path, calls)
    delivery = _delivery(
        task_id="task-blocked",
        commit_id="commit-blocked",
        source_digest="ab" * 32,
        verdict="failure",
        dispositions="explicit",
        signal="fail",
        script=True,
        controls=_controls(permission="denied"),
    )
    result = orchestrator.deliver(delivery)
    assert result.blocked is True
    assert result.action == "blocked_pre_execution"
    assert result.child_run_id is None
    assert _runs(tmp_path) == 0
    assert _json_count(tmp_path, REMEDIATION_RUN_DIRNAME, "blocked") == 1
    replay = orchestrator.deliver(delivery)
    assert replay.deduplicated is True
    assert replay.blocked is True
    assert _json_count(tmp_path, REMEDIATION_RUN_DIRNAME, "blocked") == 1
    assert len(calls) == 1


def test_orchestrator_does_not_import_a_network_or_the_session_class() -> None:
    source = inspect.getsource(__import__("openviking.session.failure_gate", fromlist=["x"]))
    for banned in ("socket", "urllib", "httpx", "subprocess", "InfluenceClaim", "session.py"):
        assert banned not in source
    assert "authorize_outcome_write" in source
    assert "ForkPrecedentService" in source
    assert "GuidanceApprovalService" in source
    assert "RemediationDispatchService" in source
    assert "Coforge" in source


def test_pre_gate_refusals_leave_one_terminal_audit(tmp_path: Path) -> None:
    calls: list = []
    orchestrator = _orchestrator(tmp_path, calls)
    missing_script = _delivery(
        task_id="task-noscript",
        commit_id="commit-noscript",
        source_digest="11" * 32,
        verdict="failure",
        dispositions="explicit",
        signal="fail",
        script=False,
    )
    with pytest.raises(FailureGateError) as missing:
        orchestrator.deliver(missing_script)
    assert missing.value.code == "remediation_script_required"
    with pytest.raises(FailureGateError) as repeated:
        orchestrator.deliver(missing_script)
    assert repeated.value.code == "remediation_script_required"
    refused = [row for row in orchestrator.audit_events() if row.get("outcome") == AUDIT_REFUSED]
    assert len(refused) == 1
    assert refused[0]["reason"] == "remediation_script_required"
    assert _forks(tmp_path) == 0
    assert _guidance(tmp_path) == 0
    assert _runs(tmp_path) == 0

    missing_anchor = _delivery(
        task_id="task-anchor",
        commit_id="commit-anchor",
        source_digest="22" * 32,
        verdict="failure",
        dispositions="explicit",
        signal="fail",
        script=True,
        anchor_block_id="missing-anchor",
    )
    with pytest.raises(FailureGateError) as anchor:
        orchestrator.deliver(missing_anchor)
    assert anchor.value.code == "anchor_missing"
    anchor_rows = [
        row
        for row in orchestrator.audit_events()
        if row.get("outcome") == AUDIT_REFUSED and row.get("reason") == "anchor_missing"
    ]
    assert len(anchor_rows) == 1
    assert _forks(tmp_path) == 0
    assert _guidance(tmp_path) == 0
    assert _runs(tmp_path) == 0


def test_rediagnose_versions_evidence_without_duplicating_guidance(tmp_path: Path) -> None:
    calls: list = []
    orchestrator = _orchestrator(tmp_path, calls)
    first_delivery = _delivery(
        task_id="task-rediagnose",
        commit_id="commit-rediagnose",
        source_digest="33" * 32,
        verdict="failure",
        dispositions="explicit",
        signal="fail",
        script=True,
    )
    first, _first_document = _diagnose(orchestrator, first_delivery)
    assert first.action == ACTION_DISPATCHED
    assert first.fork_node_id
    assert first.guidance_content_hash
    assert first.child_run_id
    same, _same_document = _diagnose(orchestrator, first_delivery)
    assert same.deduplicated is True
    assert same.orchestration_run_id == first.orchestration_run_id

    second_delivery = _delivery(
        task_id="task-rediagnose",
        commit_id="commit-rediagnose",
        source_digest="33" * 32,
        verdict="failure",
        dispositions="explicit",
        signal="fail",
        script=True,
        evaluator_results=(
            {
                "binding_id": "bind-eval-1",
                "evidence_ref": "evidence:eval-2",
                "signal": "fail",
                "signal_kind": "evaluator",
            },
        ),
    )
    second, _second_document = _diagnose(orchestrator, second_delivery)
    assert second.deduplicated is False
    assert second.orchestration_run_id != first.orchestration_run_id
    assert second.decision == DECISION_CONFIRMED_FAILURE
    assert second.fork_node_id == first.fork_node_id
    assert second.guidance_content_hash == first.guidance_content_hash
    assert second.child_run_id == first.child_run_id
    assert _forks(tmp_path) == 1
    assert _guidance(tmp_path) == 1
    assert _runs(tmp_path) == 1

    changed_plan = _plan()
    changed_plan["steps"] = [
        {
            "evidence_refs": [dict(_EVIDENCE)],
            "imagined_node_id": "img-retry",
            "statement": "retry the call with a bounded backoff",
            "step_id": "step-retry",
        }
    ]
    third_delivery = _delivery(
        task_id="task-rediagnose",
        commit_id="commit-rediagnose",
        source_digest="33" * 32,
        verdict="failure",
        dispositions="explicit",
        signal="fail",
        script=True,
        evaluator_results=(
            {
                "binding_id": "bind-eval-1",
                "evidence_ref": "evidence:eval-3",
                "signal": "fail",
                "signal_kind": "evaluator",
            },
        ),
        guidance_plan=changed_plan,
    )
    third, _third_document = _diagnose(orchestrator, third_delivery)
    assert third.orchestration_run_id not in {
        first.orchestration_run_id,
        second.orchestration_run_id,
    }
    assert third.fork_node_id == first.fork_node_id
    assert third.guidance_content_hash != first.guidance_content_hash
    assert third.child_run_id != first.child_run_id
    assert _forks(tmp_path) == 1
    assert _guidance(tmp_path) == 2
    assert _runs(tmp_path) == 2
    versions = list((tmp_path / "failure-gate" / "runs").glob("*.json"))
    assert len(versions) == 3
    assert calls == [(AIG_CONSTRUCTION_PIPELINE_VERSION, "task-rediagnose", "33" * 32)]
