# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Coordinator diagnose HTTP face (ADR-0043 P12).

``POST /api/v1/coordinator/diagnose`` is the OpenViking end of the
Coforge coordinator-boundary contract. The body is one
``coordinator-boundary-v1`` envelope. Validation is fail-closed: a
document that is not a diagnose request never reaches
``FailureGateOrchestrator.diagnose``.

ACL
---
The closed label-kind set is ``COORDINATOR_ALLOWED_LABEL_KINDS``
(``user`` only; see ``coordinator_boundary``). Default deny:

* the HTTP caller must present a boundary-allowed label that also
  appears on the request document;
* the target session's stored owner (meta or URI, not caller ctx) must
  be visible to that caller. An unresolvable owner is invisible.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, Depends

from openviking.server.auth import get_request_context
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext
from openviking.server.models import Response
from openviking.session.aig_native_protocol import (
    ProtocolContractError,
    UnsupportedSchemaVersion,
)
from openviking.session.coordinator_boundary import (
    COORDINATOR_SCHEMA_VERSION,
    KIND_DIAGNOSE_REQUEST,
    CoordinatorDiagnoseRequest,
    coordinator_boundary_labels,
    read_coordinator_diagnose_request,
)
from openviking.session.failure_gate import FailureGateError
from openviking.session.failure_gate_wiring import diagnose_coordinator_request
from openviking.session.session_acl import session_stored_owner_label
from openviking_cli.exceptions import InvalidArgumentError, OpenVikingError

router = APIRouter(prefix="/api/v1/coordinator", tags=["coordinator"])


@router.post("/diagnose")
async def coordinator_diagnose(
    payload: dict[str, Any],
    ctx: RequestContext = Depends(get_request_context),
) -> dict[str, Any]:
    """Accept one ``coordinator_diagnose_request`` and return a gate event."""
    parsed = _read_diagnose_request(payload)
    _require_caller_labels(ctx, parsed)
    session = _load_target_session(ctx, parsed)
    _require_session_visible(ctx, session)
    try:
        result = diagnose_coordinator_request(session, payload)
    except FailureGateError as exc:
        raise InvalidArgumentError(exc.code, details={"code": exc.code}) from exc
    except ProtocolContractError as exc:
        raise InvalidArgumentError(str(exc), details={"code": _contract_code(str(exc))}) from exc
    event = result.coordinator_event
    if not isinstance(event, Mapping):
        raise InvalidArgumentError(
            "coordinator_event_required",
            details={"code": "coordinator_event_required"},
        )
    return Response(status="ok", result=dict(event)).model_dump(exclude_none=True)


def _read_diagnose_request(payload: Any) -> CoordinatorDiagnoseRequest:
    if not isinstance(payload, Mapping):
        raise InvalidArgumentError(
            "coordinator document must be a dict",
            details={"code": "coordinator_document_invalid"},
        )
    version = payload.get("schema_version")
    if version != COORDINATOR_SCHEMA_VERSION:
        raise InvalidArgumentError(
            f"unsupported schema_version {version!r}; refusing to guess or downgrade",
            details={"code": "unsupported_schema_version"},
        )
    kind = payload.get("object_kind")
    if kind != KIND_DIAGNOSE_REQUEST:
        raise InvalidArgumentError(
            f"object_kind must be {KIND_DIAGNOSE_REQUEST!r}",
            details={"code": "object_kind_invalid"},
        )
    try:
        return read_coordinator_diagnose_request(payload)
    except UnsupportedSchemaVersion as exc:
        raise InvalidArgumentError(
            str(exc), details={"code": "unsupported_schema_version"}
        ) from exc
    except ProtocolContractError as exc:
        raise InvalidArgumentError(str(exc), details={"code": _contract_code(str(exc))}) from exc


def _require_caller_labels(ctx: RequestContext, request: CoordinatorDiagnoseRequest) -> None:
    try:
        caller = coordinator_boundary_labels((f"user:{ctx.user.user_id}",))
        document = coordinator_boundary_labels(request.principal_labels)
    except ProtocolContractError as exc:
        raise OpenVikingError(
            str(exc),
            code="PERMISSION_DENIED",
            details={"code": "acl_denied"},
        ) from exc
    if caller.isdisjoint(document):
        raise OpenVikingError(
            "coordinator label required",
            code="PERMISSION_DENIED",
            details={"code": "acl_denied"},
        )


def _require_session_visible(ctx: RequestContext, session: Any) -> None:
    try:
        caller = coordinator_boundary_labels((f"user:{ctx.user.user_id}",))
        owner = session_stored_owner_label(session)
        session_labels = coordinator_boundary_labels((owner,))
    except (ProtocolContractError, ValueError) as exc:
        raise OpenVikingError(
            "target session is not visible",
            code="PERMISSION_DENIED",
            details={"code": "session_acl_denied"},
        ) from exc
    if caller.isdisjoint(session_labels):
        raise OpenVikingError(
            "target session is not visible",
            code="PERMISSION_DENIED",
            details={"code": "session_acl_denied"},
        )


def _load_target_session(ctx: RequestContext, request: CoordinatorDiagnoseRequest) -> Any:
    labels = coordinator_boundary_labels(request.principal_labels)
    users = sorted(item for item in labels if item.startswith("user:"))
    if len(users) != 1:
        raise InvalidArgumentError(
            "principal_labels must name one user",
            details={"code": "acl_denied"},
        )
    user_id = users[0].split(":", 1)[1]
    session_uri = f"viking://user/{user_id}/sessions/{request.session_id}"
    return get_service().sessions.session(ctx, request.session_id, session_uri=session_uri)


def _contract_code(message: str) -> str:
    if "content_hash" in message:
        return "content_hash_mismatch"
    if "intent" in message:
        return "intent_must_be_diagnose"
    if "safe id" in message:
        return "invalid_id"
    if "object_kind" in message:
        return "object_kind_invalid"
    if "schema_version" in message:
        return "unsupported_schema_version"
    return "coordinator_document_invalid"
