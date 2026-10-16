# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Reader outcome assessment, frozen OutcomeContract, and evidence policy.

P02 records a decision. It does not dispatch remediation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openviking.session.aig_native_protocol import (
    SCHEMA_VERSION_AIG_NATIVE_V2,
    SCHEMA_VERSION_LEGACY_CLAIM_CUT,
    CriterionDisposition,
    OutcomeContract,
    ProtocolContractError,
    TaskOutcomeAssessment,
    UnsupportedSchemaVersion,
    read_outcome_contract,
    read_task_outcome_assessment,
)
from openviking.session.aig_role_chain import AigConstructionRun, AigContractError
from openviking.session.outcome_contract import (
    DECISION_CONFIRMED_FAILURE,
    DECISION_SUCCESS,
    DECISION_UNCERTAIN,
    FOLLOW_UP_FORBIDDEN,
    FOLLOW_UP_RECORDED_ONLY,
    OUTCOME_EVIDENCE_POLICY_VERSION,
    REASON_CRITERIA_CONFLICT,
    REASON_CRITERIA_MISSING,
    REASON_CRITERIA_NOT_EXPLICIT,
    REASON_EXTERNAL_FAIL,
    REASON_EXTERNAL_PASS,
    REASON_READER_INFERENCE_ONLY,
    REASON_READER_SUCCESS,
    REASON_READER_UNCERTAIN,
    CriticViolations,
    ExternalOutcomeEvidence,
    OutcomeAdjudication,
    OutcomeAssessmentService,
    OutcomeContractError,
    OutcomeContractStore,
    OutcomeEvaluatorSource,
    adjudicate_outcome,
    external_evidence_from_session,
    remediation_gate,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "adr0043"
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


def _spec_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _spec_hash(schema_version: str, object_kind: str, body: dict[str, Any]) -> str:
    payload = {"schema_version": schema_version, "object_kind": object_kind, "body": body}
    return hashlib.sha256(_spec_dumps(payload).encode("utf-8")).hexdigest()


def _document(
    object_kind: str,
    body: dict[str, Any],
    schema_version: str = SCHEMA_VERSION_AIG_NATIVE_V2,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "object_kind": object_kind,
        "body": body,
        "content_hash": _spec_hash(schema_version, object_kind, body),
    }


def _contract_body(**overrides: object) -> dict[str, Any]:
    body: dict[str, Any] = {
        "acceptance_criteria": [
            {"criterion_id": "crit-error-budget", "statement": "error rate stays under 1%"}
        ],
        "contract_id": "contract-task-1",
        "evaluator_bindings": [
            {"binding_id": "bind-eval-1", "kind": "evaluator", "ref": "eval://error-budget"}
        ],
        "frozen_at": "2026-10-10T00:00:00.000Z",
        "goals": ["ship the rollback within the error budget"],
        "policy_constraints": [
            {"constraint_id": "sla-latency", "kind": "sla", "statement": "p99 under 200ms"}
        ],
        "source": "orchestrator",
        "task_id": "task-1",
    }
    body.update(overrides)
    return body


def _assessment_body(report_hash: str, **overrides: object) -> dict[str, Any]:
    body: dict[str, Any] = {
        "author_role": "reader",
        "criteria_dispositions": {"crit-error-budget": "explicit"},
        "evidence_refs": [copy.deepcopy(_EVIDENCE)],
        "report_hash": report_hash,
        "verdict": "failure",
    }
    body.update(overrides)
    return body


def _contract(**overrides: object) -> OutcomeContract:
    return read_outcome_contract(_document("outcome_contract", _contract_body(**overrides)))


def _assessment(report_hash: str = "report-hash-1", **overrides: object) -> TaskOutcomeAssessment:
    return read_task_outcome_assessment(
        _document("task_outcome_assessment", _assessment_body(report_hash, **overrides))
    )


def _signal(
    signal: str,
    binding_id: str = "bind-eval-1",
    signal_kind: str = "evaluator",
    evidence_ref: str = "evidence:eval-1",
) -> ExternalOutcomeEvidence:
    return ExternalOutcomeEvidence(
        binding_id=binding_id,
        signal_kind=signal_kind,
        signal=signal,
        evidence_ref=evidence_ref,
    )


def _frozen_run(run_id: str = "run-outcome") -> tuple[AigConstructionRun, dict[str, str], str]:
    run = AigConstructionRun(run_id)
    sessions = {role: run.session_for(role) for role in ("builder", "critic", "refiner", "reader")}
    run.submit_draft(sessions["builder"], {"graph": []})
    run.critique(sessions["critic"], [{"code": "gap", "severity": "high"}])
    run.refine(sessions["refiner"], {"graph": ["n"]})
    report_hash = run.read_verify(sessions["reader"], {"verdict": "supported"})
    assert run.freeze_report(sessions["reader"]) == report_hash
    return run, sessions, report_hash


def _assert_blocked(adjudication: OutcomeAdjudication, code: str) -> None:
    with pytest.raises(OutcomeContractError) as exc_info:
        remediation_gate(adjudication)
    assert exc_info.value.code == code


def test_outcome_contract_reader_is_fail_closed() -> None:
    golden = json.loads((FIXTURE_DIR / "outcome_contract.json").read_text(encoding="utf-8"))
    parsed = read_outcome_contract(golden)
    again = read_outcome_contract(json.loads(json.dumps(golden)))
    assert parsed == again
    assert parsed.content_hash == _spec_hash(
        golden["schema_version"], golden["object_kind"], golden["body"]
    )
    assert parsed.source == "orchestrator"
    assert parsed.goals == ("ship the rollback within the error budget",)
    assert parsed.acceptance_criteria[0].criterion_id == "crit-error-budget"
    assert [item.kind for item in parsed.evaluator_bindings] == ["evaluator", "test"]
    assert {item.kind for item in parsed.policy_constraints} == {"policy", "sla"}

    claimed = copy.deepcopy(golden)
    claimed["body"]["claim_id"] = "claim-1"
    with pytest.raises(ProtocolContractError, match="Claim/Cut field 'claim_id'"):
        read_outcome_contract(claimed)

    tampered = copy.deepcopy(golden)
    tampered["body"]["goals"] = ["rewritten after the hash"]
    with pytest.raises(ProtocolContractError, match="content_hash mismatch"):
        read_outcome_contract(tampered)

    unknown = _document("outcome_contract", _contract_body(), schema_version="aig-native-v9")
    with pytest.raises(UnsupportedSchemaVersion):
        read_outcome_contract(unknown)

    relabeled = _document(
        "outcome_contract", _contract_body(), schema_version=SCHEMA_VERSION_LEGACY_CLAIM_CUT
    )
    with pytest.raises(ProtocolContractError, match="schema_version"):
        read_outcome_contract(relabeled)

    with pytest.raises(ProtocolContractError, match="goals"):
        read_outcome_contract(_document("outcome_contract", _contract_body(goals=[])))
    with pytest.raises(ProtocolContractError, match="source"):
        read_outcome_contract(_document("outcome_contract", _contract_body(source="memory")))
    with pytest.raises(ProtocolContractError, match="duplicate criterion_id"):
        read_outcome_contract(
            _document(
                "outcome_contract",
                _contract_body(
                    acceptance_criteria=[
                        {"criterion_id": "crit-error-budget", "statement": "under 1%"},
                        {"criterion_id": "crit-error-budget", "statement": "under 2%"},
                    ]
                ),
            )
        )
    with pytest.raises(ProtocolContractError, match="contract_id"):
        read_outcome_contract(
            _document("outcome_contract", _contract_body(contract_id="../escape"))
        )


def test_non_reader_cannot_store_an_outcome(tmp_path: Path) -> None:
    run, sessions, report_hash = _frozen_run()
    store = OutcomeContractStore(tmp_path)
    service = OutcomeAssessmentService(store)
    contract = service.freeze_contract(_document("outcome_contract", _contract_body()))
    payload = _document("task_outcome_assessment", _assessment_body(report_hash))
    for role in ("builder", "critic", "refiner"):
        with pytest.raises(AigContractError) as exc_info:
            service.submit_assessment(
                run,
                sessions[role],
                {"author_role": "reader"},
                task_id=contract.task_id,
                contract_id=contract.contract_id,
            )
        assert exc_info.value.code == "outcome_write_forbidden"
        with pytest.raises(AigContractError) as exc_info:
            service.submit_assessment(
                run,
                sessions[role],
                payload,
                task_id=contract.task_id,
                contract_id=contract.contract_id,
            )
        assert exc_info.value.code == "outcome_write_forbidden"
    assert store.read_assessment(contract.task_id, report_hash) is None


def test_critic_violations_are_not_a_task_failure() -> None:
    violations = CriticViolations(({"code": "gap", "severity": "high"},))
    with pytest.raises(OutcomeContractError) as exc_info:
        adjudicate_outcome(violations, _contract(), ())
    assert exc_info.value.code == "critic_violation_is_not_failure"
    with pytest.raises(OutcomeContractError) as exc_info:
        adjudicate_outcome([{"code": "gap"}], _contract(), ())  # type: ignore[arg-type]
    assert exc_info.value.code == "assessment_required"

    forged = TaskOutcomeAssessment(
        schema_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        object_kind="task_outcome_assessment",
        content_hash="a" * 64,
        report_hash="b" * 64,
        author_role="critic",
        verdict="failure",
        criterion_dispositions=(CriterionDisposition("crit-error-budget", "explicit"),),
        evidence_refs=(),
    )
    with pytest.raises(OutcomeContractError) as exc_info:
        adjudicate_outcome(forged, _contract(), (_signal("fail"),))
    assert exc_info.value.code == "outcome_write_forbidden"


def test_reader_assessment_binds_report_hash_evidence_and_criteria(tmp_path: Path) -> None:
    run, sessions, report_hash = _frozen_run("run-bind")
    store = OutcomeContractStore(tmp_path)
    service = OutcomeAssessmentService(store)
    contract = service.freeze_contract(_document("outcome_contract", _contract_body()))
    payload = _document("task_outcome_assessment", _assessment_body(report_hash, verdict="failure"))

    bare = OutcomeAssessmentService(OutcomeContractStore(tmp_path / "bare"))
    with pytest.raises(OutcomeContractError) as exc_info:
        bare.submit_assessment(
            run,
            sessions["reader"],
            payload,
            task_id=contract.task_id,
            contract_id=contract.contract_id,
        )
    assert exc_info.value.code == "contract_not_frozen"

    stored = service.submit_assessment(
        run,
        sessions["reader"],
        payload,
        task_id=contract.task_id,
        contract_id=contract.contract_id,
    )
    assert stored.author_role == "reader"
    assert stored.report_hash == report_hash
    assert stored.criterion_dispositions == (CriterionDisposition("crit-error-budget", "explicit"),)
    assert stored.evidence_refs[0].ao_id == "ao-eval-1"
    assert stored.evidence_refs[0].evidence_content_hash == "hash-eval-1"
    assert stored.content_hash == payload["content_hash"]

    again = service.submit_assessment(
        run,
        run.session_for("diagnosis"),
        payload,
        task_id=contract.task_id,
        contract_id=contract.contract_id,
    )
    assert again == stored

    mismatched = _document(
        "task_outcome_assessment", _assessment_body("other-report", verdict="failure")
    )
    with pytest.raises(OutcomeContractError) as exc_info:
        service.submit_assessment(
            run,
            sessions["reader"],
            mismatched,
            task_id=contract.task_id,
            contract_id=contract.contract_id,
        )
    assert exc_info.value.code == "report_hash_mismatch"
    assert store.read_assessment(contract.task_id, report_hash) == stored

    critic_author = _assessment_body(report_hash, verdict="success")
    critic_author["author_role"] = "critic"
    with pytest.raises(ProtocolContractError, match="author_role must be reader"):
        service.submit_assessment(
            run,
            sessions["reader"],
            _document("task_outcome_assessment", critic_author),
            task_id=contract.task_id,
            contract_id=contract.contract_id,
        )

    conflict = _document(
        "task_outcome_assessment", _assessment_body(report_hash, verdict="success")
    )
    with pytest.raises(OutcomeContractError) as exc_info:
        service.submit_assessment(
            run,
            sessions["reader"],
            conflict,
            task_id=contract.task_id,
            contract_id=contract.contract_id,
        )
    assert exc_info.value.code == "assessment_conflict"
    assert store.read_assessment(contract.task_id, report_hash) == stored


def test_external_signal_overrides_reader_inference() -> None:
    failure = _assessment(verdict="failure")
    success = _assessment(verdict="success")
    contract = _contract()

    confirmed = adjudicate_outcome(success, contract, (_signal("fail"), _signal("fail")))
    assert confirmed == OutcomeAdjudication(
        policy_version=OUTCOME_EVIDENCE_POLICY_VERSION,
        decision=DECISION_CONFIRMED_FAILURE,
        follow_up=FOLLOW_UP_RECORDED_ONLY,
        reason=REASON_EXTERNAL_FAIL,
    )
    _assert_blocked(confirmed, "remediation_not_in_scope")

    overridden = adjudicate_outcome(failure, contract, (_signal("pass"),))
    assert overridden.decision == DECISION_SUCCESS
    assert overridden.follow_up == FOLLOW_UP_FORBIDDEN
    assert overridden.reason == REASON_EXTERNAL_PASS
    _assert_blocked(overridden, "automatic_remediation_forbidden")

    mixed = adjudicate_outcome(
        failure,
        _contract(
            evaluator_bindings=[
                {"binding_id": "bind-eval-1", "kind": "evaluator", "ref": "eval://error-budget"},
                {"binding_id": "bind-test-1", "kind": "test", "ref": "test://rollback-suite"},
            ]
        ),
        (
            _signal("fail"),
            _signal(
                "pass",
                binding_id="bind-test-1",
                signal_kind="test",
                evidence_ref="evidence:test-1",
            ),
        ),
    )
    assert mixed.decision == DECISION_UNCERTAIN
    assert mixed.follow_up == FOLLOW_UP_FORBIDDEN
    assert mixed.reason == REASON_CRITERIA_CONFLICT
    _assert_blocked(mixed, "automatic_remediation_forbidden")


def test_reader_inference_alone_cannot_confirm_failure() -> None:
    unbound = _contract(evaluator_bindings=[])
    failure = adjudicate_outcome(_assessment(verdict="failure"), unbound, ())
    assert failure.decision == DECISION_UNCERTAIN
    assert failure.follow_up == FOLLOW_UP_FORBIDDEN
    assert failure.reason == REASON_READER_INFERENCE_ONLY
    _assert_blocked(failure, "automatic_remediation_forbidden")

    success = adjudicate_outcome(_assessment(verdict="success"), unbound, ())
    assert success.decision == DECISION_SUCCESS
    assert success.follow_up == FOLLOW_UP_FORBIDDEN
    assert success.reason == REASON_READER_SUCCESS

    uncertain = adjudicate_outcome(_assessment(verdict="uncertain"), unbound, ())
    assert uncertain.decision == DECISION_UNCERTAIN
    assert uncertain.reason == REASON_READER_UNCERTAIN
    _assert_blocked(uncertain, "automatic_remediation_forbidden")


@pytest.mark.parametrize(
    ("disposition", "reason"),
    [
        ("missing", REASON_CRITERIA_MISSING),
        ("conflict", REASON_CRITERIA_CONFLICT),
        ("provisional", REASON_CRITERIA_NOT_EXPLICIT),
    ],
)
def test_missing_or_conflicting_criteria_are_uncertain(disposition: str, reason: str) -> None:
    assessment = _assessment(
        verdict="uncertain",
        criteria_dispositions={"crit-error-budget": disposition},
        evidence_refs=[],
    )
    adjudication = adjudicate_outcome(assessment, _contract(), (_signal("fail"),))
    assert adjudication.decision == DECISION_UNCERTAIN
    assert adjudication.follow_up == FOLLOW_UP_FORBIDDEN
    assert adjudication.reason == reason
    assert adjudication.policy_version == OUTCOME_EVIDENCE_POLICY_VERSION
    _assert_blocked(adjudication, "automatic_remediation_forbidden")


def test_criterion_set_must_match_the_contract() -> None:
    missing_key = adjudicate_outcome(
        _assessment(
            verdict="uncertain",
            criteria_dispositions={},
            evidence_refs=[],
        ),
        _contract(),
        (_signal("fail"),),
    )
    assert missing_key.decision == DECISION_UNCERTAIN
    assert missing_key.reason == REASON_CRITERIA_MISSING

    extra_key = adjudicate_outcome(
        _assessment(
            criteria_dispositions={
                "crit-error-budget": "explicit",
                "crit-other": "explicit",
            },
        ),
        _contract(),
        (),
    )
    assert extra_key.decision == DECISION_UNCERTAIN
    assert extra_key.reason == REASON_CRITERIA_CONFLICT

    two = _contract(
        acceptance_criteria=[
            {"criterion_id": "crit-error-budget", "statement": "error rate stays under 1%"},
            {"criterion_id": "crit-latency", "statement": "p99 stays under 200ms"},
        ]
    )
    partial = adjudicate_outcome(
        _assessment(
            verdict="failure",
            criteria_dispositions={"crit-error-budget": "explicit"},
        ),
        two,
        (_signal("fail"),),
    )
    assert partial.decision == DECISION_UNCERTAIN
    assert partial.reason == REASON_CRITERIA_MISSING
    mixed = adjudicate_outcome(
        _assessment(
            verdict="uncertain",
            criteria_dispositions={
                "crit-error-budget": "explicit",
                "crit-latency": "conflict",
            },
            evidence_refs=[],
        ),
        two,
        (_signal("fail"),),
    )
    assert mixed.decision == DECISION_UNCERTAIN
    assert mixed.reason == REASON_CRITERIA_CONFLICT


def test_missing_contract_criteria_or_bindings_stay_uncertain() -> None:
    explicit_failure = _assessment(verdict="failure")
    empty = adjudicate_outcome(
        explicit_failure,
        _contract(acceptance_criteria=[]),
        (_signal("fail"),),
    )
    assert empty.decision == DECISION_UNCERTAIN
    assert empty.reason == REASON_CRITERIA_MISSING
    _assert_blocked(empty, "automatic_remediation_forbidden")

    partial = adjudicate_outcome(
        explicit_failure,
        _contract(
            evaluator_bindings=[
                {"binding_id": "bind-eval-1", "kind": "evaluator", "ref": "eval://error-budget"},
                {"binding_id": "bind-test-1", "kind": "test", "ref": "test://rollback-suite"},
            ]
        ),
        (_signal("fail"),),
    )
    assert partial.decision == DECISION_UNCERTAIN
    assert partial.reason == REASON_CRITERIA_MISSING
    _assert_blocked(partial, "automatic_remediation_forbidden")

    mismatched = adjudicate_outcome(
        explicit_failure,
        _contract(),
        (_signal("fail", signal_kind="test"),),
    )
    assert mismatched.decision == DECISION_UNCERTAIN
    assert mismatched.reason == REASON_CRITERIA_CONFLICT

    unbound = adjudicate_outcome(
        explicit_failure,
        _contract(),
        (_signal("fail", binding_id="bind-other"),),
    )
    assert unbound.decision == DECISION_UNCERTAIN
    assert unbound.reason == REASON_CRITERIA_CONFLICT
    _assert_blocked(unbound, "automatic_remediation_forbidden")


def test_policy_is_deterministic() -> None:
    contract = _contract(
        evaluator_bindings=[
            {"binding_id": "bind-eval-1", "kind": "evaluator", "ref": "eval://error-budget"},
            {"binding_id": "bind-test-1", "kind": "test", "ref": "test://rollback-suite"},
        ]
    )
    assessment = _assessment(verdict="success")
    forward = (
        _signal("fail", evidence_ref="evidence:b"),
        _signal("fail", binding_id="bind-test-1", signal_kind="test", evidence_ref="evidence:a"),
    )
    reverse = tuple(reversed(forward))
    first = adjudicate_outcome(assessment, contract, forward)
    second = adjudicate_outcome(assessment, contract, reverse)
    assert first == second
    assert first.decision == DECISION_CONFIRMED_FAILURE
    assert first.policy_version == "outcome-evidence-v1"


def test_policy_constraints_are_frozen_and_not_executed() -> None:
    contract = _contract(evaluator_bindings=[])
    assert contract.policy_constraints[0].kind == "sla"
    assert contract.policy_constraints[0].statement == "p99 under 200ms"
    adjudication = adjudicate_outcome(_assessment(verdict="success"), contract, ())
    assert adjudication.decision == DECISION_SUCCESS
    assert adjudication.follow_up == FOLLOW_UP_FORBIDDEN


def test_contract_store_is_append_only(tmp_path: Path) -> None:
    payload = _document("outcome_contract", _contract_body())
    service = OutcomeAssessmentService(OutcomeContractStore(tmp_path))
    first = service.freeze_contract(payload)
    assert service.freeze_contract(json.loads(json.dumps(payload))) == first

    rewritten = _contract_body(goals=["a different goal after start"])
    with pytest.raises(OutcomeContractError) as exc_info:
        service.freeze_contract(_document("outcome_contract", rewritten))
    assert exc_info.value.code == "contract_frozen"

    other = _contract_body(contract_id="contract-task-2")
    with pytest.raises(OutcomeContractError) as exc_info:
        service.freeze_contract(_document("outcome_contract", other))
    assert exc_info.value.code == "task_contract_exists"

    reopened = OutcomeContractStore(tmp_path)
    assert reopened.read_contract(first.contract_id) == first
    assert reopened.read_task_contract(first.task_id) == first
    index = (tmp_path / "outcome-contracts" / "index.jsonl").read_text(encoding="utf-8")
    assert index.count(first.content_hash) == 1
    assert "a different goal after start" not in index


class _EvaluatorSession:
    """Test double for ``OutcomeEvaluatorSource``. Telemetry is not read."""

    def __init__(self, results: list[dict[str, str]] | None) -> None:
        self.outcome_evaluator_results = results
        self.index_outcome_status = "failed"
        self.telemetry_snapshot = SimpleNamespace(
            summary={"tokens": {"llm": {}}, "status": "error"}
        )


def test_session_telemetry_is_not_evaluator_evidence() -> None:
    assert isinstance(_EvaluatorSession(None), OutcomeEvaluatorSource)
    session = _EvaluatorSession(None)
    assert external_evidence_from_session(session) == ()
    with pytest.raises(OutcomeContractError) as missing_source:
        external_evidence_from_session(
            SimpleNamespace(index_outcome_status="failed")  # type: ignore[arg-type]
        )
    assert missing_source.value.code == "evaluator_source_required"
    with pytest.raises(OutcomeContractError) as exc_info:
        external_evidence_from_session(
            session, [{"summary": {"tokens": {"llm": {}}}, "status": "failed"}]
        )
    assert exc_info.value.code == "telemetry_is_not_outcome_evidence"
    with pytest.raises(OutcomeContractError) as exc_info:
        external_evidence_from_session(
            session,
            [
                {
                    "binding_id": "bind-eval-1",
                    "signal_kind": "evaluator",
                    "signal": "failed",
                    "evidence_ref": "evidence:eval-1",
                }
            ],
        )
    assert exc_info.value.code == "invalid_external_evidence"

    ignored = adjudicate_outcome(
        _assessment(verdict="failure"),
        _contract(evaluator_bindings=[]),
        external_evidence_from_session(session),
    )
    assert ignored.decision == DECISION_UNCERTAIN
    assert ignored.reason == REASON_READER_INFERENCE_ONLY

    session.outcome_evaluator_results = [
        {
            "binding_id": "bind-eval-1",
            "signal_kind": "evaluator",
            "signal": "fail",
            "evidence_ref": "evidence:eval-1",
        }
    ]
    evidence = external_evidence_from_session(session)
    assert evidence == (_signal("fail"),)
    confirmed = adjudicate_outcome(_assessment(verdict="success"), _contract(), evidence)
    assert confirmed.decision == DECISION_CONFIRMED_FAILURE
    assert confirmed.follow_up == FOLLOW_UP_RECORDED_ONLY


def test_adjudication_does_not_dispatch(tmp_path: Path) -> None:
    run, sessions, report_hash = _frozen_run("run-dispatch")
    service = OutcomeAssessmentService(OutcomeContractStore(tmp_path))
    contract = service.freeze_contract(_document("outcome_contract", _contract_body()))
    service.submit_assessment(
        run,
        sessions["reader"],
        _document("task_outcome_assessment", _assessment_body(report_hash, verdict="success")),
        task_id=contract.task_id,
        contract_id=contract.contract_id,
    )
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    adjudication = service.adjudicate(
        task_id=contract.task_id,
        report_hash=report_hash,
        external_evidence=external_evidence_from_session(
            SimpleNamespace(index_outcome_status="failed"),
            [
                {
                    "binding_id": "bind-eval-1",
                    "signal_kind": "evaluator",
                    "signal": "fail",
                    "evidence_ref": "evidence:eval-1",
                }
            ],
        ),
    )
    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert adjudication.decision == DECISION_CONFIRMED_FAILURE
    assert adjudication.follow_up == FOLLOW_UP_RECORDED_ONLY
    assert before == after
    assert not hasattr(service, "dispatch")
    assert not hasattr(service, "remediate")
    _assert_blocked(adjudication, "remediation_not_in_scope")

    with pytest.raises(OutcomeContractError) as exc_info:
        service.adjudicate(task_id="missing-task", report_hash=report_hash)
    assert exc_info.value.code == "contract_not_frozen"
    with pytest.raises(OutcomeContractError) as exc_info:
        service.adjudicate(task_id=contract.task_id, report_hash="missing-report")
    assert exc_info.value.code == "assessment_not_submitted"
