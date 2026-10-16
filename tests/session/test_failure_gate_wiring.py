# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""P10 commit-path wiring: Session.commit → FailureGateOrchestrator.deliver.

These tests stay next to the Phase 2 seam (``test_phase2_causal_seam``) rather
than inside ``test_failure_gate.py``. That file owns the orchestrator. This
file owns the adapter, the frozen-view builder, and the skip / swallow
decisions. Fake builder plus scripted session products; no model is called.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.session.coordinator_boundary import (
    ACTION_ADVICE,
    ACTION_RETAINED,
    TRIGGER_COMMIT,
    TRIGGER_DIAGNOSE,
    coordinator_diagnose_request_document,
    read_coordinator_diagnose_request,
)
from openviking.session.extraction_write import NATIVE_EDGE_INDEX_FILENAME, NativeEdgeViewStore
from openviking.session.failure_gate import (
    AIG_CONSTRUCTION_PIPELINE_VERSION,
    AUDIT_ADJUDICATED,
    AUDIT_REFUSED,
    AUDIT_REPLAYED,
    AUDIT_VIEW_CONSTRUCTED,
    ConstructionRequest,
    FailureGateOrchestrator,
)
from openviking.session.failure_gate_wiring import (
    build_commit_delivery,
    build_diagnose_delivery,
    commit_id_for_archive,
    diagnose_coordinator_request,
    latest_native_aig_revision,
    maybe_deliver_commit_failure_gate,
    session_frozen_view_builder,
)
from openviking.session.fork_node_overlay import FORK_NODE_OVERLAY_DIRNAME
from openviking.session.guidance_approval import GUIDANCE_APPROVAL_DIRNAME
from openviking.session.outcome_contract import (
    DECISION_CONFIRMED_FAILURE,
    DECISION_SUCCESS,
    DECISION_UNCERTAIN,
    REASON_CRITERIA_NOT_EXPLICIT,
    REASON_EXTERNAL_FAIL,
    REASON_EXTERNAL_PASS,
)
from openviking.session.remediation_dispatch import REMEDIATION_RUN_DIRNAME
from openviking.session.session import Session
from openviking.session.trajectory_index_hook import VIEWS_FILENAME
from openviking_cli.session.user_id import UserIdentifier
from tests.test_failure_gate import _contract, _revision, _signal
from tests.test_trajectory_index_hook import SESSION_ID, SESSION_URI, _DummyFS

SESSION = "sess-gate-1"
TASK = "task-gate-1"
COMMIT = "archive_001"
_DIGEST = "aa" * 32


def _ctx() -> RequestContext:
    return RequestContext(user=UserIdentifier("default", "default"), role=Role.ROOT)


def _session(session_dir: Path, **extra: object) -> SimpleNamespace:
    uri = f"viking://user/default/sessions/{SESSION}"
    return SimpleNamespace(
        session_id=SESSION,
        _session_uri=uri,
        ctx=_ctx(),
        _viking_fs=SimpleNamespace(_uri_to_path=lambda _uri, ctx=None: str(session_dir)),
        **extra,
    )


def _seed_native(session_dir: Path, task_id: str = TASK):
    request = ConstructionRequest(
        pipeline_version=AIG_CONSTRUCTION_PIPELINE_VERSION,
        session_id=SESSION,
        task_id=task_id,
        source_digest=_DIGEST,
    )
    revision = _revision(request)
    NativeEdgeViewStore(session_dir / NATIVE_EDGE_INDEX_FILENAME).write(revision)
    return revision


def _counting_builder(session_dir: Path, calls: list):
    inner = session_frozen_view_builder(session_dir)

    def build(request):
        calls.append((request.task_id, request.source_digest))
        return inner(request)

    return build


def _audit_kinds(session_dir: Path) -> list[str]:
    return [
        row["event_kind"]
        for row in FailureGateOrchestrator(
            session_dir, session_frozen_view_builder(session_dir)
        ).audit_events()
    ]


def _audit_rows(session_dir: Path) -> tuple[dict, ...]:
    return FailureGateOrchestrator(
        session_dir, session_frozen_view_builder(session_dir)
    ).audit_events()


def _json_count(root: Path, *parts: str) -> int:
    directory = root.joinpath(*parts)
    if not directory.is_dir():
        return 0
    return len(list(directory.glob("*.json")))


@pytest.fixture
def causal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "openviking.session.failure_gate_wiring.is_causal_mode_enabled",
        lambda *_args, **_kwargs: True,
    )


def test_commit_id_prefers_the_archive_directory_name() -> None:
    assert (
        commit_id_for_archive(f"viking://user/default/sessions/{SESSION}/history/archive_001", TASK)
        == "archive_001"
    )
    assert commit_id_for_archive("", TASK) == TASK


def test_confirmed_failure_records_outcome_and_does_not_dispatch(
    tmp_path: Path, causal: None
) -> None:
    revision = _seed_native(tmp_path)
    calls: list = []
    session = _session(
        tmp_path,
        outcome_contract=_contract(TASK),
        outcome_evaluator_results=(_signal("fail"),),
    )
    result = maybe_deliver_commit_failure_gate(
        session,
        task_id=TASK,
        commit_id=COMMIT,
        build_revision=_counting_builder(tmp_path, calls),
    )
    assert result is not None
    assert result.trigger == TRIGGER_COMMIT
    assert result.decision == DECISION_CONFIRMED_FAILURE
    assert result.reason == REASON_EXTERNAL_FAIL
    assert result.action == ACTION_RETAINED
    assert result.deduplicated is False
    assert result.fork_node_id is None
    assert result.guidance_content_hash is None
    assert result.child_run_id is None
    assert result.blocked is False
    assert result.coordinator_event is not None
    assert result.coordinator_event["body"]["trigger"] == TRIGGER_COMMIT
    assert calls == [(TASK, revision.content_hash)]
    kinds = _audit_kinds(tmp_path)
    assert AUDIT_VIEW_CONSTRUCTED in kinds
    assert AUDIT_ADJUDICATED in kinds
    assert ACTION_RETAINED in kinds
    assert AUDIT_REFUSED not in kinds
    assert _json_count(tmp_path, "failure-gate", "runs") == 1
    assert _json_count(tmp_path, FORK_NODE_OVERLAY_DIRNAME, "overlays") == 0
    assert _json_count(tmp_path, GUIDANCE_APPROVAL_DIRNAME, "approvals") == 0
    assert _json_count(tmp_path, REMEDIATION_RUN_DIRNAME, "runs") == 0
    rows = [row for row in _audit_rows(tmp_path) if row["event_kind"] == AUDIT_ADJUDICATED]
    assert rows[0]["trigger"] == TRIGGER_COMMIT
    assert rows[0]["orchestration_run_id"] == result.orchestration_run_id


@pytest.mark.parametrize(
    ("signal", "dispositions", "verdict", "decision", "reason", "action"),
    [
        ("pass", "explicit", "success", DECISION_SUCCESS, REASON_EXTERNAL_PASS, ACTION_RETAINED),
        (
            "fail",
            "provisional",
            "uncertain",
            DECISION_UNCERTAIN,
            REASON_CRITERIA_NOT_EXPLICIT,
            ACTION_ADVICE,
        ),
    ],
)
def test_success_and_uncertain_do_not_propose_or_dispatch(
    tmp_path: Path,
    causal: None,
    signal: str,
    dispositions: str,
    verdict: str,
    decision: str,
    reason: str,
    action: str,
) -> None:
    _seed_native(tmp_path)
    session = _session(
        tmp_path,
        outcome_contract=_contract(TASK),
        outcome_evaluator_results=(_signal(signal),),
        outcome_assessment_body={
            "criteria_dispositions": {"crit-error-budget": dispositions},
            "evidence_refs": (
                []
                if dispositions != "explicit"
                else [
                    {
                        "ao_id": "ao-1",
                        "archive_commit_watermark": None,
                        "captured_state": "committed",
                        "evidence_content_hash": "hash-1",
                        "evidence_role": "hindsight_attribution",
                        "source_archive_id": None,
                        "source_read_snapshot_watermark": "wm-failure-gate",
                        "source_sequence": 1,
                        "source_session_id": SESSION,
                    }
                ]
            ),
            "verdict": verdict,
        },
    )
    result = maybe_deliver_commit_failure_gate(session, task_id=TASK, commit_id=COMMIT)
    assert result is not None
    assert result.decision == decision
    assert result.reason == reason
    assert result.action == action
    assert result.fork_node_id is None
    assert result.child_run_id is None
    assert result.trigger == TRIGGER_COMMIT
    assert action in _audit_kinds(tmp_path)
    assert _json_count(tmp_path, FORK_NODE_OVERLAY_DIRNAME, "overlays") == 0
    assert _json_count(tmp_path, GUIDANCE_APPROVAL_DIRNAME, "approvals") == 0
    assert _json_count(tmp_path, REMEDIATION_RUN_DIRNAME, "runs") == 0


def test_gate_exception_does_not_fail_commit_and_writes_refusal(
    tmp_path: Path, causal: None
) -> None:
    _seed_native(tmp_path)
    session = _session(
        tmp_path,
        outcome_contract=_contract(TASK),
        outcome_evaluator_results=(_signal("pass"),),
    )

    def _boom(_request):
        raise RuntimeError("injected builder failure")

    result = maybe_deliver_commit_failure_gate(
        session, task_id=TASK, commit_id=COMMIT, build_revision=_boom
    )
    assert result is None
    refused = [row for row in _audit_rows(tmp_path) if row.get("outcome") == AUDIT_REFUSED]
    assert len(refused) == 1
    assert refused[0]["reason"] == "store_failed"
    assert refused[0]["trigger"] == TRIGGER_COMMIT
    assert _json_count(tmp_path, "failure-gate", "runs") == 0


def test_replay_hits_the_existing_run(tmp_path: Path, causal: None) -> None:
    revision = _seed_native(tmp_path)
    calls: list = []
    builder = _counting_builder(tmp_path, calls)
    session = _session(
        tmp_path,
        outcome_contract=_contract(TASK),
        outcome_evaluator_results=(_signal("fail"),),
    )
    first = maybe_deliver_commit_failure_gate(
        session, task_id=TASK, commit_id=COMMIT, build_revision=builder
    )
    second = maybe_deliver_commit_failure_gate(
        session, task_id=TASK, commit_id=COMMIT, build_revision=builder
    )
    assert first is not None and second is not None
    assert first.deduplicated is False
    assert second.deduplicated is True
    assert second.reused_view is True
    assert second.orchestration_run_id == first.orchestration_run_id
    assert second.decision == first.decision
    assert calls == [(TASK, revision.content_hash)]
    kinds = _audit_kinds(tmp_path)
    assert kinds.count(AUDIT_REPLAYED) == 1
    assert _json_count(tmp_path, "failure-gate", "runs") == 1


def test_causal_mode_off_is_a_zero_side_effect_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openviking.session.failure_gate_wiring.is_causal_mode_enabled",
        lambda *_args, **_kwargs: False,
    )
    _seed_native(tmp_path)
    calls: list = []
    result = maybe_deliver_commit_failure_gate(
        _session(tmp_path, outcome_contract=_contract(TASK)),
        task_id=TASK,
        commit_id=COMMIT,
        build_revision=_counting_builder(tmp_path, calls),
    )
    assert result is None
    assert calls == []
    assert not (tmp_path / "failure-gate").exists()


def test_missing_config_is_a_zero_side_effect_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _missing(*_args, **_kwargs):
        raise FileNotFoundError("ov.conf")

    monkeypatch.setattr(
        "openviking.session.failure_gate_wiring.is_causal_mode_enabled",
        _missing,
    )
    _seed_native(tmp_path)
    result = maybe_deliver_commit_failure_gate(
        _session(tmp_path, outcome_contract=_contract(TASK)),
        task_id=TASK,
        commit_id=COMMIT,
    )
    assert result is None
    assert not (tmp_path / "failure-gate").exists()


def test_session_without_aig_pipeline_skips(tmp_path: Path, causal: None) -> None:
    calls: list = []
    result = maybe_deliver_commit_failure_gate(
        _session(tmp_path),
        task_id=TASK,
        commit_id=COMMIT,
        build_revision=_counting_builder(tmp_path, calls),
    )
    assert result is None
    assert calls == []
    assert not (tmp_path / "failure-gate").exists()
    assert latest_native_aig_revision(tmp_path, SESSION, TASK) is None


def test_builder_reads_the_hook_revision_and_does_not_rewrite(tmp_path: Path, causal: None) -> None:
    stored = _seed_native(tmp_path)
    builder = session_frozen_view_builder(tmp_path)
    loaded = builder(
        ConstructionRequest(
            pipeline_version=AIG_CONSTRUCTION_PIPELINE_VERSION,
            session_id=SESSION,
            task_id=TASK,
            source_digest=stored.content_hash,
        )
    )
    assert loaded.content_hash == stored.content_hash
    assert loaded.revision_id == stored.revision_id
    assert not (tmp_path / VIEWS_FILENAME).exists()
    delivery = build_commit_delivery(
        _session(tmp_path, outcome_contract=_contract(TASK)),
        stored,
        task_id=TASK,
        commit_id=COMMIT,
        session_dir=tmp_path,
    )
    assert delivery.trigger == TRIGGER_COMMIT
    assert delivery.auto_remediate is False
    assert delivery.source_digest == stored.content_hash
    assert delivery.principal_labels == ("user:default",)


def test_phase2_seam_invokes_the_commit_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, str]] = []

    def _fake(session, *, task_id, commit_id, **_kwargs):
        del session
        seen.append((task_id, commit_id))
        return None

    monkeypatch.setattr(
        "openviking.session.failure_gate_wiring.maybe_deliver_commit_failure_gate",
        _fake,
    )
    root = tmp_path / "session"
    root.mkdir()
    session = Session(
        viking_fs=_DummyFS(root, SESSION_URI),
        session_id=SESSION_ID,
        session_uri=SESSION_URI,
    )
    archive_uri = f"{SESSION_URI}/history/archive_001"
    session._run_post_extraction_causal_seam("task-seam-1", archive_uri)
    assert seen == [("task-seam-1", "archive_001")]
    assert not (root / "failure-gate").exists()


def _diagnose_document(*, source_digest: str = "ab" * 32):
    return coordinator_diagnose_request_document(
        commit_id=COMMIT,
        session_id=SESSION,
        task_id=TASK,
        source_digest=source_digest,
        principal_labels=("user:default",),
    )


def test_diagnose_delivery_uses_request_ids_and_session_products(
    tmp_path: Path,
) -> None:
    revision = _seed_native(tmp_path)
    document = _diagnose_document(source_digest="cd" * 32)
    parsed = read_coordinator_diagnose_request(document)
    session = _session(
        tmp_path,
        outcome_contract=_contract(TASK),
        outcome_evaluator_results=(_signal("pass"),),
    )
    delivery = build_diagnose_delivery(session, revision, parsed, session_dir=tmp_path)
    assert delivery.trigger == TRIGGER_DIAGNOSE
    assert delivery.commit_id == COMMIT
    assert delivery.session_id == SESSION
    assert delivery.task_id == TASK
    assert delivery.source_digest == "cd" * 32
    assert delivery.principal_labels == ("user:default",)
    assert delivery.auto_remediate is False
    assert delivery.guidance_plan is None


def test_diagnose_delivery_attaches_script_only_when_session_carries_it(
    tmp_path: Path,
) -> None:
    revision = _seed_native(tmp_path)
    document = _diagnose_document()
    parsed = read_coordinator_diagnose_request(document)
    plan = {
        "applicability": {},
        "required_capabilities": [],
        "risk": "r",
        "steps": [],
        "target_role": "task_agent",
    }
    session = _session(
        tmp_path,
        outcome_contract=_contract(TASK),
        failure_gate_remediation={
            "anchor_block_id": "b-2",
            "guidance_plan": plan,
            "catalog": {"candidates": []},
        },
    )
    delivery = build_diagnose_delivery(session, revision, parsed, session_dir=tmp_path)
    assert delivery.auto_remediate is True
    assert delivery.anchor_block_id == "b-2"
    assert delivery.guidance_plan == plan
    assert delivery.catalog == {"candidates": []}


def test_diagnose_coordinator_request_replays_the_same_event(tmp_path: Path) -> None:
    _seed_native(tmp_path)
    document = _diagnose_document()
    session = _session(
        tmp_path,
        outcome_contract=_contract(TASK),
        outcome_evaluator_results=(_signal("pass"),),
    )
    first = diagnose_coordinator_request(session, document)
    second = diagnose_coordinator_request(session, document)
    assert first.coordinator_event == second.coordinator_event
    assert second.deduplicated is True
    assert first.trigger == TRIGGER_DIAGNOSE
