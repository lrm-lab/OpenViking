# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""HTTP face for ADR-0046 experience cards: Reader proposals in, find out."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from openviking.core.namespace import canonical_user_root
from openviking.server.auth import require_role
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext, Role
from openviking.server.models import Response
from openviking.session.experience_cards import (
    ExperienceCardError,
    ExperienceCardStore,
    card_to_dict,
)
from openviking.session.fork_node_overlay import ForkNodeOverlayStore
from openviking_cli.exceptions import InvalidArgumentError

router = APIRouter(prefix="/api/v1/experience", tags=["experience"])


class CardSubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proposal: dict[str, Any]
    block_aos: dict[str, list[str]] | None = None
    read_aos: list[str] | None = None


class CardFindRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = ""
    kind: str | None = None
    deliverable: str | None = None
    occupation: str | None = None
    issue_kind: str | None = None
    limit: int = Field(default=8, ge=1, le=20)


def _store(ctx: RequestContext) -> ExperienceCardStore:
    filesystem = get_service().lexical.filesystem()
    user_root = Path(filesystem._uri_to_path(canonical_user_root(ctx), ctx=ctx))
    return ExperienceCardStore(user_root, ForkNodeOverlayStore(user_root))


def _labels(ctx: RequestContext) -> list[str]:
    return [f"user:{ctx.user.user_id}"]


def submit_card(
    ctx: RequestContext, body: CardSubmitRequest, store: ExperienceCardStore | None = None
) -> dict[str, Any]:
    store = store or _store(ctx)
    try:
        card, duplicate = store.submit(
            body.proposal,
            principal_labels=_labels(ctx),
            block_aos=body.block_aos,
            read_aos=body.read_aos,
        )
    except ExperienceCardError as exc:
        raise InvalidArgumentError(str(exc)) from exc
    return {"card_id": card.card_id, "kind": card.kind, "deduplicated": duplicate}


def find_cards(
    ctx: RequestContext, body: CardFindRequest, store: ExperienceCardStore | None = None
) -> dict[str, Any]:
    store = store or _store(ctx)
    try:
        found = store.find(
            principal_labels=_labels(ctx),
            query=body.query,
            kind=body.kind,
            deliverable=body.deliverable,
            occupation=body.occupation,
            issue_kind=body.issue_kind,
            limit=body.limit,
        )
    except ExperienceCardError as exc:
        raise InvalidArgumentError(str(exc)) from exc
    return {
        "relaxed": found.relaxed,
        "cards": [{"score": h.score, **card_to_dict(h.card)} for h in found.hits],
    }


@router.post("/cards")
async def post_card(
    body: CardSubmitRequest, ctx: RequestContext = require_role(Role.ROOT, Role.ADMIN)
):
    return Response(status="ok", result=await asyncio.to_thread(submit_card, ctx, body))


@router.post("/find")
async def post_find(
    body: CardFindRequest, ctx: RequestContext = require_role(Role.ROOT, Role.ADMIN, Role.USER)
):
    return Response(status="ok", result=await asyncio.to_thread(find_cards, ctx, body))
