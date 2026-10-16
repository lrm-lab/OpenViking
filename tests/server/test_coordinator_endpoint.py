# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""HTTP face for ``POST /api/v1/coordinator/diagnose`` (ADR-0043 P12).

Zero-LLM: golden document, fail-closed reader, ACL, and orchestrator
idempotency. Session products are scripted; no model is called.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from openviking.server.auth import get_request_context
from openviking.server.identity import RequestContext, Role
from openviking.server.models import ERROR_CODE_TO_HTTP_STATUS, ErrorInfo, Response
from openviking.server.routers import coordinator as coordinator_module
from openviking.session.aig_native_protocol import protocol_envelope
from openviking.session.coordinator_boundary import (
    ACTION_BLOCKED,
    ACTION_DISPATCHED,
    ACTION_RETAINED,
    COORDINATOR_SCHEMA_VERSION,
    KIND_DIAGNOSE_REQUEST,
    KIND_FAILURE_GATE_EVENT,
    read_coordinator_document,
)
from openviking.session.extraction_write import NATIVE_EDGE_INDEX_FILENAME, NativeEdgeViewStore
from openviking.session.failure_gate import AIG_CONSTRUCTION_PIPELINE_VERSION, ConstructionRequest
from openviking_cli.exceptions import OpenVikingError
from openviking_cli.session.user_id import UserIdentifier
from tests.test_failure_gate import (
    _EVIDENCE,
    _catalog,
    _contract,
    _controls,
    _plan,
    _revision,
    _signal,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "adr0043"
    / "coordinator_diagnose_request.json"
)
_GOLDEN_SESSION = "sess-diagnose"
_GOLDEN_TASK = "task-diagnose"


def _golden() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _ctx(user_id: str = "carol") -> RequestContext:
    return RequestContext(user=UserIdentifier("default", user_id), role=Role.USER)


def _seed_native(session_dir: Path, session_id: str, task_id: str):
    request = ConstructionRequest(
        pipeline_version=AIG_CONSTRUCTION_PIPELINE_VERSION,
        session_id=session_id,
        task_id=task_id,
        source_digest="ab" * 32,
    )
    revision = _revision(request)
    NativeEdgeViewStore(session_dir / NATIVE_EDGE_INDEX_FILENAME).write(revision)
    return revision


def _session(
    session_dir: Path,
    *,
    session_id: str = _GOLDEN_SESSION,
    owner: str = "carol",
    **extra: object,
) -> SimpleNamespace:
    uri = f"viking://user/{owner}/sessions/{session_id}"
    return SimpleNamespace(
        session_id=session_id,
        _session_uri=uri,
        ctx=_ctx(owner),
        _viking_fs=SimpleNamespace(_uri_to_path=lambda _uri, ctx=None: str(session_dir)),
        **extra,
    )


def _script(**control_overrides: object) -> dict[str, Any]:
    return {
        "anchor_block_id": "b-2",
        "branch_steps": (
            {"block_id": "b-1", "step_kind": "factual_block_ref"},
            {"imagined_node_id": "img-retry", "step_kind": "imagined_node"},
        ),
        "guidance_plan": _plan(),
        "catalog": _catalog(),
        "dispatch_controls": _controls(**control_overrides),
        "dispatch_outputs": (
            {"output_id": "ao-child-1", "output_kind": "ao"},
            {"output_id": "blk-child-1", "output_kind": "block"},
        ),
        "dispatch_steps": (
            {
                "actual_block_id": "blk-child-1",
                "alignment": "equivalent",
                "imagined_node_id": "img-retry",
            },
        ),
    }


def _client(session: Any, ctx: RequestContext, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(
        coordinator_module,
        "get_service",
        lambda: SimpleNamespace(
            sessions=SimpleNamespace(session=lambda *_args, **_kwargs: session)
        ),
    )
    app = FastAPI()
    app.include_router(coordinator_module.router)
    app.dependency_overrides[get_request_context] = lambda: ctx

    @app.exception_handler(OpenVikingError)
    async def _ov_error(_request: Request, exc: OpenVikingError):
        return JSONResponse(
            status_code=ERROR_CODE_TO_HTTP_STATUS.get(exc.code, 500),
            content=Response(
                status="error",
                error=ErrorInfo(code=exc.code, message=exc.message, details=exc.details),
            ).model_dump(),
        )

    return TestClient(app)


def _post(client: TestClient, payload: dict[str, Any]):
    return client.post("/api/v1/coordinator/diagnose", json=payload)


def test_openapi_lists_the_diagnose_route() -> None:
    app = FastAPI()
    app.include_router(coordinator_module.router)
    paths = app.openapi()["paths"]
    assert "/api/v1/coordinator/diagnose" in paths
    assert "post" in paths["/api/v1/coordinator/diagnose"]


def test_golden_document_returns_a_hashed_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_native(tmp_path, _GOLDEN_SESSION, _GOLDEN_TASK)
    session = _session(
        tmp_path,
        outcome_contract=_contract(_GOLDEN_TASK),
        outcome_evaluator_results=(_signal("pass"),),
    )
    client = _client(session, _ctx("carol"), monkeypatch)
    response = _post(client, _golden())
    assert response.status_code == 200, response.text
    event = response.json()["result"]
    parsed = read_coordinator_document(event)
    assert parsed.object_kind == KIND_FAILURE_GATE_EVENT
    assert parsed.action == ACTION_RETAINED
    assert parsed.content_hash == event["content_hash"]
    assert parsed.request_content_hash == _golden()["content_hash"]
    assert parsed.trigger == "memory_diagnose"


def test_replay_returns_the_same_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_native(tmp_path, _GOLDEN_SESSION, _GOLDEN_TASK)
    session = _session(
        tmp_path,
        outcome_contract=_contract(_GOLDEN_TASK),
        outcome_evaluator_results=(_signal("pass"),),
    )
    client = _client(session, _ctx("carol"), monkeypatch)
    first = _post(client, _golden())
    second = _post(client, _golden())
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["result"] == second.json()["result"]


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda doc: {**doc, "schema_version": "coordinator-boundary-v9"},
            "unsupported_schema_version",
        ),
        (
            lambda doc: {**doc, "object_kind": KIND_FAILURE_GATE_EVENT},
            "object_kind_invalid",
        ),
        (
            lambda doc: {**doc, "content_hash": "00" + doc["content_hash"][2:]},
            "content_hash_mismatch",
        ),
    ],
)
def test_invalid_envelope_is_4xx_and_does_not_diagnose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    code: str,
) -> None:
    calls: list[object] = []

    def _boom(*_args, **_kwargs):
        calls.append(1)
        raise AssertionError("diagnose must not run")

    monkeypatch.setattr(coordinator_module, "diagnose_coordinator_request", _boom)
    session = _session(tmp_path)
    client = _client(session, _ctx("carol"), monkeypatch)
    response = _post(client, mutate(_golden()))
    assert response.status_code == 400
    assert response.json()["error"]["details"]["code"] == code
    assert calls == []


def test_intent_and_illegal_id_are_4xx_and_do_not_diagnose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        coordinator_module,
        "diagnose_coordinator_request",
        lambda *_args, **_kwargs: calls.append(1),
    )
    session = _session(tmp_path)
    client = _client(session, _ctx("carol"), monkeypatch)
    retrieve_body = dict(_golden()["body"])
    retrieve_body["intent"] = "retrieve"
    retrieve = protocol_envelope(COORDINATOR_SCHEMA_VERSION, KIND_DIAGNOSE_REQUEST, retrieve_body)
    illegal_body = dict(_golden()["body"])
    illegal_body["session_id"] = "sess diagnose"
    illegal = protocol_envelope(COORDINATOR_SCHEMA_VERSION, KIND_DIAGNOSE_REQUEST, illegal_body)
    for payload, code in (
        (retrieve, "intent_must_be_diagnose"),
        (illegal, "invalid_id"),
    ):
        response = _post(client, payload)
        assert response.status_code == 400, response.text
        assert response.json()["error"]["details"]["code"] == code
    assert calls == []


def test_missing_coordinator_label_is_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        coordinator_module,
        "diagnose_coordinator_request",
        lambda *_args, **_kwargs: calls.append(1),
    )
    session = _session(tmp_path)
    client = _client(session, _ctx("dave"), monkeypatch)
    response = _post(client, _golden())
    assert response.status_code == 403
    assert response.json()["error"]["details"]["code"] == "acl_denied"
    assert calls == []


def test_invisible_session_acl_is_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        coordinator_module,
        "diagnose_coordinator_request",
        lambda *_args, **_kwargs: calls.append(1),
    )
    session = _session(tmp_path, owner="eve")
    client = _client(session, _ctx("carol"), monkeypatch)
    response = _post(client, _golden())
    assert response.status_code == 403
    assert response.json()["error"]["details"]["code"] == "session_acl_denied"
    assert calls == []


def test_confirmed_failure_dispatches_when_script_is_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_native(tmp_path, _GOLDEN_SESSION, _GOLDEN_TASK)
    session = _session(
        tmp_path,
        outcome_contract=_contract(_GOLDEN_TASK),
        outcome_evaluator_results=(_signal("fail"),),
        outcome_assessment_body={
            "criteria_dispositions": {"crit-error-budget": "explicit"},
            "evidence_refs": [dict(_EVIDENCE)],
            "verdict": "failure",
        },
        failure_gate_remediation=_script(),
    )
    client = _client(session, _ctx("carol"), monkeypatch)
    response = _post(client, _golden())
    assert response.status_code == 200, response.text
    event = read_coordinator_document(response.json()["result"])
    assert event.action == ACTION_DISPATCHED
    assert event.decision == "confirmed_failure"


def test_confirmed_failure_blocks_when_a_dispatch_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_native(tmp_path, _GOLDEN_SESSION, _GOLDEN_TASK)
    session = _session(
        tmp_path,
        outcome_contract=_contract(_GOLDEN_TASK),
        outcome_evaluator_results=(_signal("fail"),),
        outcome_assessment_body={
            "criteria_dispositions": {"crit-error-budget": "explicit"},
            "evidence_refs": [dict(_EVIDENCE)],
            "verdict": "failure",
        },
        failure_gate_remediation=_script(budget="exceeded"),
    )
    client = _client(session, _ctx("carol"), monkeypatch)
    response = _post(client, _golden())
    assert response.status_code == 200, response.text
    event = read_coordinator_document(response.json()["result"])
    assert event.action == ACTION_BLOCKED
    assert event.decision == "confirmed_failure"
