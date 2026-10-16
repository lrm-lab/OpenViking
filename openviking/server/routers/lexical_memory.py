# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Lexical-only ingestion and read-only graph navigation."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from openviking.core.uri_validation import validate_request_viking_uri
from openviking.server.auth import get_request_context, require_role
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext, Role
from openviking.server.models import Response
from openviking_cli.exceptions import InvalidArgumentError

router = APIRouter(prefix="/api/v1/search", tags=["search"])


class LexicalResourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uri: str
    text: str = Field(min_length=1)
    kind: str = "document"


class LexicalBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resources: list[LexicalResourceRequest] = Field(min_length=1, max_length=100)


class GraphExpandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    chain_ref: str = Field(min_length=1, max_length=256)
    uri: str
    relation: str
    limit: int = Field(default=10, ge=1, le=10)


@router.post("/index_lexical")
async def index_lexical(
    body: LexicalResourceRequest | LexicalBatchRequest,
    ctx: RequestContext = require_role(Role.ROOT, Role.ADMIN),
):
    rows = body.resources if isinstance(body, LexicalBatchRequest) else [body]
    resources = []
    for row in rows:
        uri = validate_request_viking_uri(row.uri, ctx)
        if not uri.startswith("viking://resources/"):
            raise InvalidArgumentError("lexical ingestion requires a shared resource URI")
        resources.append({"uri": uri, "text": row.text, "kind": row.kind})
    try:
        result = await get_service().lexical.put_resources(ctx, resources)
    except ValueError as exc:
        raise InvalidArgumentError(str(exc)) from exc
    return Response(
        status="ok",
        result={"resources": result, "retrieval_mode": "bm25"}
        if isinstance(body, LexicalBatchRequest)
        else result[0],
    )


@router.post("/expand")
async def expand(body: GraphExpandRequest, ctx: RequestContext = Depends(get_request_context)):
    uri = validate_request_viking_uri(body.uri, ctx)
    try:
        result = await get_service().lexical.expand(
            ctx,
            chain_ref=body.chain_ref,
            uri=uri,
            relation=body.relation,
            operation_id=body.operation_id,
            limit=body.limit,
        )
    except ValueError as exc:
        raise InvalidArgumentError(str(exc)) from exc
    return Response(status="ok", result=result)


@router.get("/index_lexical")
async def index_status(ctx: RequestContext = require_role(Role.ROOT, Role.ADMIN)):
    counts = await asyncio.to_thread(get_service().lexical.index(ctx).counts)
    return Response(status="ok", result={"retrieval_mode": "bm25", **counts})


class PublishCausalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


@router.post("/publish_causal")
async def publish_causal(
    body: PublishCausalRequest, ctx: RequestContext = require_role(Role.ROOT, Role.ADMIN)
):
    from pathlib import Path

    from openviking.core.namespace import canonical_user_root
    from openviking.session.active_retrieval import corpus_for_session
    from openviking.session.session_acl import session_stored_owner_label
    from openviking.session.trajectory_index_hook import session_directory
    from openviking_cli.exceptions import PermissionDeniedError

    service = get_service()
    session = service.sessions.session(ctx, body.session_id)
    await session.load()
    owner = f"user:{ctx.user.user_id}"
    try:
        stored_owner = session_stored_owner_label(session)
    except ValueError as exc:
        raise PermissionDeniedError("stored session owner is unavailable") from exc
    if stored_owner != owner:
        raise PermissionDeniedError("only the stored session owner may publish causal summaries")
    directory = session_directory(session)
    if directory is None:
        raise InvalidArgumentError("causal session storage is unavailable")
    filesystem = service.lexical.filesystem()
    user_root = Path(filesystem._uri_to_path(canonical_user_root(ctx), ctx=ctx))
    corpus = corpus_for_session(directory, user_root, owner_label=owner)
    result = await service.lexical.publish_corpus(ctx, corpus, owner)
    return Response(status="ok", result=result)
