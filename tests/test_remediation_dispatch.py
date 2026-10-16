# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""P09: linked child remediation runs and GuidanceExecutionReport.

Dispatch stays fail-closed until every server gate passes. The scripted
entry writes a child run and never touches the completed failure run. The
Coforge worker is not called.
"""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest

from openviking.session.aig_native_protocol import (
    DIRECT_ALIGNMENTS,
    EVALUATION_STATUS_ACTIVE,
    NEGATIVE_EVALUATION_STATUSES,
    OBJECT_KIND_BLOCKED_PRE_EXECUTION,
    SCHEMA_VERSION_AIG_NATIVE_V2,
    ProtocolContractError,
    guidance_execution_report_document,
    production_direct_admission_document,
    protocol_content_hash,
    read_production_direct_admission,
    read_protocol_document,
)
from openviking.session.guidance_approval import GUIDANCE_APPROVAL_DIRNAME
from openviking.session.remediation_dispatch import (
    REMEDIATION_RUN_DIRNAME,
    RemediationDispatchError,
    RemediationDispatchService,
)
from openviking.session.strategy_pattern import (
    STRATEGY_PATTERN_DIRNAME,
    StrategyPatternStore,
)
from tests.test_guidance_approval import (
    _BOB,
    _CAROL,
    _branch,
    _candidate,
    _catalog,
    _propose,
    _step,
    _world,
)
from tests.test_strategy_pattern import _catalog as _pattern_catalog
from tests.test_strategy_pattern import _direct_admission, _evaluation, _overlay

_GATES = ("acl", "permission", "budget", "security", "capability", "idempotency")


def _code(call) -> str:
    try:
        call()
    except RemediationDispatchError as exc:
        return exc.code
    raise AssertionError("expected a coded error")


def _controls(**overrides: object) -> dict:
    body: dict = {
        "agent_capabilities": ["replay-sandbox"],
        "budget": "compliant",
        "executor_agent_id": "agent-a",
        "executor_role": "task_agent",
        "idempotency_key": "remediation-1",
        "isolation": "isolated",
        "permission": "granted",
        "requested_side_effects": [],
        "side_effect_allowlist": [],
    }
    body.update(overrides)
    return body


def _approved(tmp_path: Path, *, imagined: str = "img-retry", steps: list[dict] | None = None):
    world = _world(tmp_path)
    payload = _branch(world, steps=steps or [_step(imagined=imagined)])
    proposed = _propose(world, payload)
    approved = world.service.approve(
        proposed.proposal.content_hash,
        _catalog(world, [_candidate("agent-a")]),
        principal_labels=(_CAROL,),
    )
    world.patterns.freeze_branch(world.stored.stored, world.base)
    service = RemediationDispatchService(tmp_path, guidance=world.service)
    return world, approved, service


def _report(approval, parent: str, steps: list[dict]) -> dict:
    return guidance_execution_report_document(
        guidance_id=approval.guidance_id,
        source_fork_id=approval.source_fork_id,
        parent_session_id=parent,
        steps=steps,
    )


def _block(output_id: str = "blk-child-1") -> dict:
    return {"output_id": output_id, "output_kind": "block"}


def _ao(output_id: str = "ao-child-1") -> dict:
    return {"output_id": output_id, "output_kind": "ao"}


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def _dispatch(service, world, approved, **overrides):
    parent = world.sessions["reader"]
    approval = approved.approval
    report = overrides.pop(
        "report",
        _report(
            approval,
            parent,
            [
                {
                    "actual_block_id": "blk-child-1",
                    "alignment": "equivalent",
                    "imagined_node_id": approval.steps[0].imagined_node_id,
                }
            ],
        ),
    )
    outputs = overrides.pop("outputs", [_ao(), _block()])
    return service.dispatch_scripted(
        approval_content_hash=approval.content_hash,
        principal_labels=overrides.pop("principal_labels", (_CAROL,)),
        parent_session_id=overrides.pop("parent_session_id", parent),
        controls=overrides.pop("controls", _controls()),
        outputs=outputs,
        report=report,
        pattern_store=overrides.pop("pattern_store", world.patterns),
        adjustment=overrides.pop("adjustment", None),
    )


def _run_files(tmp_path: Path) -> list[Path]:
    directory = tmp_path / REMEDIATION_RUN_DIRNAME / "runs"
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def _blocked_files(tmp_path: Path) -> list[Path]:
    directory = tmp_path / REMEDIATION_RUN_DIRNAME / "blocked"
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def test_each_failed_gate_is_fail_closed_and_audited(tmp_path: Path) -> None:
    world, approved, service = _approved(tmp_path)
    parent = world.sessions["reader"]
    cases = {
        "acl": {"principal_labels": (_BOB,)},
        "permission": {"controls": _controls(executor_role="reader")},
        "budget": {"controls": _controls(budget="exceeded")},
        "security": {"controls": _controls(isolation="unisolated")},
        "capability": {"controls": _controls(agent_capabilities=["rollback"])},
        "idempotency": {},
    }
    first = _dispatch(service, world, approved)
    assert first.blocked is False
    assert len(_run_files(tmp_path)) == 1
    conflict = _dispatch(
        service,
        world,
        approved,
        controls=_controls(idempotency_key="remediation-2"),
    )
    assert conflict.blocked is True
    assert conflict.decision.object_kind == OBJECT_KIND_BLOCKED_PRE_EXECUTION
    assert conflict.decision.outcome == "blocked_pre_execution"
    assert conflict.decision.failed_gate == "idempotency"
    assert any(item.startswith("idempotency:") for item in conflict.decision.reasons)
    assert conflict.child_run is None
    assert len(_run_files(tmp_path)) == 1

    for gate, kwargs in cases.items():
        if gate == "idempotency":
            continue
        root = tmp_path / gate
        held_world, held, held_service = _approved(root)
        before_runs = _run_files(root)
        outcome = _dispatch(held_service, held_world, held, **kwargs)
        assert outcome.blocked is True
        assert outcome.decision.failed_gate == gate
        assert outcome.decision.object_kind == OBJECT_KIND_BLOCKED_PRE_EXECUTION
        assert outcome.child_run is None
        assert outcome.request is None
        assert _run_files(root) == before_runs
        assert len(_blocked_files(root)) == 1
        stored = json.loads(_blocked_files(root)[0].read_text(encoding="utf-8"))
        assert stored["object_kind"] == "blocked_pre_execution"
        assert stored["body"]["failed_gate"] == gate
        assert read_protocol_document(stored) == outcome.decision
        assert held_service.audit_blocked()[0].decision_id == outcome.decision.decision_id
    assert first.child_run is not None
    assert parent == first.child_run.parent_session_id
    assert set(_GATES) == {item.gate for item in first.decision.gates}
    assert all(item.passed for item in first.decision.gates)


def test_empty_acl_and_allowlist_miss_do_not_create_a_run(tmp_path: Path) -> None:
    world, approved, service = _approved(tmp_path)
    empty = _dispatch(service, world, approved, principal_labels=())
    assert empty.decision.failed_gate == "acl"
    assert empty.child_run is None
    denied = _dispatch(
        service,
        world,
        approved,
        controls=_controls(
            requested_side_effects=["network"],
            side_effect_allowlist=["filesystem"],
        ),
    )
    assert denied.decision.failed_gate == "security"
    assert "security:side_effect_not_allowlisted" in denied.decision.reasons
    assert _run_files(tmp_path) == []
    scripted = _dispatch(
        service,
        world,
        approved,
        controls=_controls(
            requested_side_effects=["filesystem"],
            side_effect_allowlist=["filesystem"],
        ),
    )
    assert scripted.decision.failed_gate == "security"
    assert "security:scripted_side_effect_forbidden" in scripted.decision.reasons
    assert len(_blocked_files(tmp_path)) == 3


def test_exact_retry_is_idempotent_and_a_conflict_is_audited(tmp_path: Path) -> None:
    world, approved, service = _approved(tmp_path)
    first = _dispatch(service, world, approved)
    again = _dispatch(service, world, approved)
    assert again.deduplicated is True
    assert again.child_run == first.child_run
    assert len(_run_files(tmp_path)) == 1
    assert len(world.patterns.audit_evaluations()) == 1


def test_child_run_keeps_provenance_and_does_not_touch_the_failure_run(
    tmp_path: Path,
) -> None:
    world, approved, service = _approved(tmp_path)
    parent = world.sessions["reader"]
    before_run = copy.deepcopy(world.run.__dict__)
    before_files = _snapshot(tmp_path)
    guidance_path = (
        tmp_path / GUIDANCE_APPROVAL_DIRNAME / "guidance" / f"{approved.guidance.content_hash}.json"
    )
    guidance_bytes = guidance_path.read_bytes()
    outcome = _dispatch(service, world, approved)
    assert world.run.__dict__ == before_run
    after = _snapshot(tmp_path)
    for path, data in before_files.items():
        if path.endswith("strategy-patterns/index.jsonl"):
            assert after[path].startswith(data)
            continue
        assert after[path] == data, path
    for path in after:
        if path in before_files:
            continue
        assert path.startswith(f"{REMEDIATION_RUN_DIRNAME}/") or path.startswith(
            f"{STRATEGY_PATTERN_DIRNAME}/evaluations/"
        )
    assert guidance_path.read_bytes() == guidance_bytes
    child = outcome.child_run
    assert child is not None
    assert child.parent_session_id == parent
    assert child.source_fork_id == approved.guidance.source_fork_id
    assert child.source_guidance_id == approved.guidance.guidance_id
    assert child.child_session_id != parent
    stored_outputs = list((tmp_path / REMEDIATION_RUN_DIRNAME / "outputs").glob("*.json"))
    assert stored_outputs
    for path in stored_outputs:
        body = json.loads(path.read_text(encoding="utf-8"))["body"]
        assert body["parent_session_id"] == parent
        assert body["source_fork_id"] == approved.guidance.source_fork_id
        assert body["source_guidance_id"] == approved.guidance.guidance_id
        assert body["child_session_id"] != parent
    assert _code(lambda: service.reject_parent_run(world.run)) == "parent_run_sealed"


def test_output_session_cannot_be_the_parent_session(tmp_path: Path) -> None:
    world, approved, service = _approved(tmp_path)
    parent = world.sessions["reader"]
    before = _snapshot(tmp_path)
    assert (
        _code(
            lambda: _dispatch(
                service,
                world,
                approved,
                outputs=[
                    {
                        "output_id": "blk-parent",
                        "output_kind": "block",
                        "child_session_id": parent,
                    }
                ],
            )
        )
        == "parent_run_sealed"
    )
    assert _run_files(tmp_path) == []
    assert _blocked_files(tmp_path) == []
    assert _snapshot(tmp_path) == before


def test_execution_report_maps_blocks_and_rejects_a_missing_block(tmp_path: Path) -> None:
    world, approved, service = _approved(tmp_path)
    parent = world.sessions["reader"]
    imagined = approved.approval.steps[0].imagined_node_id
    assert (
        _code(
            lambda: _dispatch(
                service,
                world,
                approved,
                report=_report(
                    approved.approval,
                    parent,
                    [
                        {
                            "actual_block_id": "blk-not-frozen",
                            "alignment": "equivalent",
                            "imagined_node_id": imagined,
                        }
                    ],
                ),
            )
        )
        == "actual_block_not_in_child_run"
    )
    assert _run_files(tmp_path) == []
    assert world.patterns.has_direct_evidence("imagined_node", imagined) is False
    with pytest.raises(ProtocolContractError, match="actual_block_id"):
        guidance_execution_report_document(
            guidance_id=approved.guidance.guidance_id,
            source_fork_id=approved.guidance.source_fork_id,
            parent_session_id=parent,
            steps=[
                {
                    "actual_block_id": "blk-child-1",
                    "alignment": "not_executed",
                    "imagined_node_id": imagined,
                }
            ],
        )


def test_adjustment_is_an_appendix_and_cannot_widen_permission(tmp_path: Path) -> None:
    world, approved, service = _approved(tmp_path)
    guidance_path = (
        tmp_path / GUIDANCE_APPROVAL_DIRNAME / "guidance" / f"{approved.guidance.content_hash}.json"
    )
    before = guidance_path.read_bytes()
    imagined = approved.approval.steps[0].imagined_node_id
    outcome = _dispatch(
        service,
        world,
        approved,
        adjustment={
            "author_agent_id": "agent-a",
            "capability_limits": ["replay-sandbox"],
            "revised_steps": [
                {"imagined_node_id": imagined, "statement": "retry with the same capability"}
            ],
            "target_role": "task_agent",
        },
    )
    assert outcome.adjustment is not None
    assert outcome.adjustment.guidance_content_hash == approved.guidance.content_hash
    assert guidance_path.read_bytes() == before
    assert json.loads(before)["body"]["capability_limits"] == ["replay-sandbox"]
    widened = _approved(tmp_path / "wide")
    wide_path = (
        tmp_path
        / "wide"
        / GUIDANCE_APPROVAL_DIRNAME
        / "guidance"
        / f"{widened[1].guidance.content_hash}.json"
    )
    wide_before = wide_path.read_bytes()
    assert (
        _code(
            lambda: _dispatch(
                widened[2],
                widened[0],
                widened[1],
                adjustment={
                    "author_agent_id": "agent-a",
                    "capability_limits": ["replay-sandbox", "shell"],
                    "revised_steps": [{"imagined_node_id": imagined, "statement": "open a shell"}],
                    "target_role": "task_agent",
                },
            )
        )
        == "adjustment_forbidden"
    )
    assert wide_path.read_bytes() == wide_before
    assert _run_files(tmp_path / "wide") == []
    assert (
        _code(
            lambda: _dispatch(
                widened[2],
                widened[0],
                widened[1],
                adjustment={
                    "author_agent_id": "agent-b",
                    "capability_limits": ["replay-sandbox"],
                    "revised_steps": [{"imagined_node_id": imagined, "statement": "same words"}],
                    "target_role": "task_agent",
                },
            )
        )
        == "adjustment_forbidden"
    )
    assert wide_path.read_bytes() == wide_before


def test_divergent_and_not_executed_are_not_direct_refutations(tmp_path: Path) -> None:
    world, approved, service = _approved(
        tmp_path,
        steps=[
            _step(imagined="img-retry", step_id="step-retry"),
            _step(imagined="img-later", step_id="step-later", statement="stop the rollout"),
        ],
    )
    parent = world.sessions["reader"]
    approval = approved.approval
    outcome = _dispatch(
        service,
        world,
        approved,
        report=_report(
            approval,
            parent,
            [
                {
                    "actual_block_id": "blk-child-1",
                    "alignment": "divergent",
                    "imagined_node_id": "img-retry",
                },
                {
                    "actual_block_id": None,
                    "alignment": "not_executed",
                    "imagined_node_id": "img-later",
                },
            ],
        ),
        outputs=[_block(), _ao()],
    )
    assert outcome.admissions == ()
    assert outcome.evaluations == ()
    assert {item.alignment for item in outcome.constraints} == {"divergent", "not_executed"}
    assert world.patterns.has_direct_evidence("imagined_node", "img-retry") is False
    for row in world.patterns.audit_evaluations():
        assert row.evidence_tier not in {"production-direct", "replay-direct"}
        assert row.status not in NEGATIVE_EVALUATION_STATUSES
    imagined_path = next((tmp_path / STRATEGY_PATTERN_DIRNAME / "imagined").glob("*.json"))
    imagined_bytes = imagined_path.read_bytes()
    assert b"invalidated" not in imagined_bytes
    assert b"challenged" not in imagined_bytes


def test_aligned_child_block_enters_the_ledger_as_production_direct(tmp_path: Path) -> None:
    catalog = _pattern_catalog()
    overlay = _overlay(catalog)
    store = StrategyPatternStore(tmp_path)
    store.freeze_branch(overlay, catalog)
    admission = production_direct_admission_document(
        imagined_node_id="img-retry",
        actual_block_id="blk-child-1",
        alignment="equivalent",
        source_fork_id="fork-a",
        source_guidance_id="guidance-1",
        parent_session_id="sess-parent",
        child_run_id="child-run-1",
        report_content_hash="ab" * 32,
        child_block_ids=("blk-child-1",),
    )
    parsed = read_production_direct_admission(admission)
    assert parsed.evidence_tier == "production-direct"
    assert parsed.alignment in DIRECT_ALIGNMENTS
    stored = store.append_evaluation(
        _evaluation(
            "imagined_node",
            "img-retry",
            "eval-prod-legal",
            f"production-direct:{parsed.admission_id}",
            "production-direct",
        ),
        admission=admission,
    )
    assert stored.evidence_tier == "production-direct"
    assert stored.status == EVALUATION_STATUS_ACTIVE
    assert store.has_direct_evidence("imagined_node", "img-retry") is True
    challenged = _evaluation(
        "imagined_node",
        "img-retry",
        "eval-prod-refute",
        "evidence:refute",
        "production-direct",
        status="invalidated",
    )
    with pytest.raises(Exception) as exc_info:
        store.append_evaluation(challenged, admission=admission)
    assert getattr(exc_info.value, "code", "") == "direct_admission_mismatch"
    replay_tier = _evaluation(
        "imagined_node",
        "img-retry",
        "eval-replay-via-prod",
        "evidence:replay",
        "replay-direct",
    )
    with pytest.raises(Exception) as replay_exc:
        store.append_evaluation(replay_tier, admission=admission)
    assert getattr(replay_exc.value, "code", "") == "direct_admission_mismatch"
    with pytest.raises(Exception) as prod_exc:
        store.append_evaluation(
            _evaluation(
                "imagined_node",
                "img-retry",
                "eval-prod-via-replay",
                "evidence:prod",
                "production-direct",
            ),
            admission=_direct_admission("img-retry"),
        )
    assert getattr(prod_exc.value, "code", "") == "direct_admission_mismatch"
    with pytest.raises(ProtocolContractError):
        production_direct_admission_document(
            imagined_node_id="img-retry",
            actual_block_id="blk-child-1",
            alignment="divergent",
            source_fork_id="fork-a",
            source_guidance_id="guidance-1",
            parent_session_id="sess-parent",
            child_run_id="child-run-1",
            report_content_hash="ab" * 32,
            child_block_ids=("blk-child-1",),
        )


def test_confirmed_failure_reaches_production_direct_without_external_calls(
    tmp_path: Path,
) -> None:
    world, approved, service = _approved(tmp_path)
    parent = world.sessions["reader"]
    before = _snapshot(tmp_path)
    outcome = _dispatch(service, world, approved)
    assert outcome.blocked is False
    assert outcome.request is not None
    assert outcome.request.object_kind == "remediation_dispatch_request"
    assert outcome.confirmation is not None
    assert outcome.confirmation.executor == "scripted"
    assert outcome.confirmation.status == "accepted"
    assert outcome.child_run is not None
    assert outcome.report is not None
    assert outcome.admissions[0].evidence_tier == "production-direct"
    assert outcome.admissions[0].actual_block_id == "blk-child-1"
    assert outcome.evaluations[0].evidence_tier == "production-direct"
    assert outcome.evaluations[0].status == EVALUATION_STATUS_ACTIVE
    assert outcome.evaluations[0].subject_id == "img-retry"
    imagined = approved.approval.steps[0].imagined_node_id
    assert world.patterns.has_direct_evidence("imagined_node", imagined) is True
    after = _snapshot(tmp_path)
    for path, data in before.items():
        if path.endswith("strategy-patterns/index.jsonl"):
            assert after[path].startswith(data)
            continue
        assert after[path] == data
    source = inspect.getsource(
        __import__("openviking.session.remediation_dispatch", fromlist=["x"])
    )
    for banned in ("socket", "urllib", "httpx", "subprocess", "AigConstructionRun", "ao_ledger"):
        assert banned not in source
    assert "Coforge" in source
    assert "P12" in source
    request_doc = json.loads(
        next((tmp_path / REMEDIATION_RUN_DIRNAME / "requests").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert request_doc["content_hash"] == protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        request_doc["object_kind"],
        request_doc["body"],
    )
    assert read_protocol_document(request_doc) == outcome.request
    assert parent == outcome.child_run.parent_session_id


def test_non_executor_roles_cannot_pass_the_permission_gate(tmp_path: Path) -> None:
    world, approved, service = _approved(tmp_path)
    for role in ("builder", "critic", "refiner", "reader", "memory"):
        outcome = _dispatch(
            service,
            world,
            approved,
            controls=_controls(executor_role=role, idempotency_key=f"role-{role}"),
        )
        assert outcome.blocked is True
        assert outcome.decision.failed_gate == "permission"
        assert outcome.child_run is None
    assert _run_files(tmp_path) == []


def test_functionally_equivalent_is_production_direct(tmp_path: Path) -> None:
    world, approved, service = _approved(tmp_path)
    parent = world.sessions["reader"]
    imagined = approved.approval.steps[0].imagined_node_id
    outcome = _dispatch(
        service,
        world,
        approved,
        report=_report(
            approved.approval,
            parent,
            [
                {
                    "actual_block_id": "blk-child-1",
                    "alignment": "functionally_equivalent",
                    "imagined_node_id": imagined,
                }
            ],
        ),
    )
    assert outcome.admissions[0].alignment == "functionally_equivalent"
    assert outcome.evaluations[0].evidence_tier == "production-direct"
    assert outcome.constraints == ()


def test_missing_report_does_not_create_a_run(tmp_path: Path) -> None:
    world, approved, service = _approved(tmp_path)
    assert (
        _code(
            lambda: service.dispatch_scripted(
                approval_content_hash=approved.approval.content_hash,
                principal_labels=(_CAROL,),
                parent_session_id=world.sessions["reader"],
                controls=_controls(),
                outputs=[_block()],
                report=None,
                pattern_store=world.patterns,
            )
        )
        == "execution_report_required"
    )
    assert _run_files(tmp_path) == []
    assert _blocked_files(tmp_path) == []


def test_index_ids_are_checked_before_they_become_paths(tmp_path: Path) -> None:
    world, approved, service = _approved(tmp_path)
    blocked = _dispatch(service, world, approved, principal_labels=(_BOB,))
    assert blocked.blocked is True
    index = tmp_path / REMEDIATION_RUN_DIRNAME / "index.jsonl"
    rows = [
        json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rows[0]["decision_id"] = "../escaped"
    index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert _code(service.audit_blocked) == "stored_id_invalid"
    assert not (tmp_path / "escaped.json").exists()

    held_world, held, held_service = _approved(tmp_path / "reload")
    _dispatch(held_service, held_world, held)
    reload_index = tmp_path / "reload" / REMEDIATION_RUN_DIRNAME / "index.jsonl"
    reload_rows = [
        json.loads(line)
        for line in reload_index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reload_rows[0]["child_run_id"] = "../escaped"
    reload_index.write_text(
        "".join(json.dumps(row) + "\n" for row in reload_rows),
        encoding="utf-8",
    )
    assert _code(lambda: _dispatch(held_service, held_world, held)) == "stored_id_invalid"
    assert not (tmp_path / "escaped.json").exists()
