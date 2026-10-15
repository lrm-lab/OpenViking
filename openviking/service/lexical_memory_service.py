# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Account-bound lexical memory use cases over the existing authorized filesystem."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openviking.server.identity import RequestContext
from openviking.service.lexical_memory import LexicalMemoryIndex
from openviking.service.lexical_retrieval import LexicalMemoryRetrieval
from openviking_cli.exceptions import NotFoundError, PermissionDeniedError


class LexicalMemoryService:
    def __init__(self, root: Path, filesystem: Callable[[], Any]):
        self.root = root
        self.filesystem = filesystem
        self._retrievals: dict[str, LexicalMemoryRetrieval] = {}
        self._write_locks: dict[str, asyncio.Lock] = {}

    def _write_lock(self, account_id: str) -> asyncio.Lock:
        return self._write_locks.setdefault(account_id, asyncio.Lock())

    async def delete_account(self, account_id: str) -> None:
        async with self._write_lock(account_id):
            self._retrievals.pop(account_id, None)
            path = self.root / (hashlib.sha256(account_id.encode()).hexdigest() + ".sqlite3")
            for suffix in ("", "-wal", "-shm"):
                await asyncio.to_thread(Path(str(path) + suffix).unlink, missing_ok=True)

    def index(self, ctx: RequestContext) -> LexicalMemoryIndex:
        return self._retrieval(ctx).index

    def _retrieval(self, ctx: RequestContext) -> LexicalMemoryRetrieval:
        if ctx.account_id not in self._retrievals:
            self._retrievals[ctx.account_id] = LexicalMemoryRetrieval(
                LexicalMemoryIndex(self.root, ctx.account_id)
            )
        return self._retrievals[ctx.account_id]

    def _can_read(self, ctx: RequestContext):
        async def can_read(uri: str) -> bool:
            try:
                await self.filesystem().stat(uri, ctx=ctx, skip_count=True)
                text = await self.filesystem().read_file(uri, ctx=ctx)
                hit = await asyncio.to_thread(self.index(ctx).get, uri)
                return (
                    hit is not None
                    and hit["content_hash"] == "sha256:" + hashlib.sha256(text.encode()).hexdigest()
                )
            except (PermissionDeniedError, NotFoundError):
                return False

        return can_read

    @staticmethod
    def _principal(ctx: RequestContext) -> str:
        return f"{ctx.account_id}:{ctx.user.user_id}"

    async def find(
        self,
        query: str,
        ctx: RequestContext,
        *,
        target_uri: str = "viking://resources",
        limit: int = 10,
    ) -> dict[str, Any]:
        return await self._retrieval(ctx).find(
            query,
            principal=self._principal(ctx),
            can_read=self._can_read(ctx),
            target_uri=target_uri,
            limit=limit,
        )

    async def expand(
        self,
        ctx: RequestContext,
        *,
        chain_ref: str,
        uri: str,
        relation: str,
        operation_id: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        return await self._retrieval(ctx).expand(
            chain_ref,
            uri,
            relation,
            principal=self._principal(ctx),
            can_read=self._can_read(ctx),
            operation_id=operation_id,
            limit=limit,
        )

    async def put_resource(
        self, ctx: RequestContext, *, uri: str, text: str, kind: str
    ) -> dict[str, Any]:
        result = await self.put_resources(ctx, [{"uri": uri, "text": text, "kind": kind}])
        return result[0]

    async def put_resources(
        self, ctx: RequestContext, resources: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        async with self._write_lock(ctx.account_id):
            return await self._put_resources(ctx, resources)

    async def _put_resources(
        self, ctx: RequestContext, resources: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        seen = set()
        for row in resources:
            if not row["uri"].startswith("viking://resources/") or not row["text"].strip():
                raise ValueError(
                    "lexical resources require a shared resource URI and nonempty text"
                )
            if row["uri"] in seen:
                raise ValueError("duplicate resource URI")
            seen.add(row["uri"])
            if row["kind"] not in {"document", "experience"}:
                raise ValueError("graph nodes must be published from frozen causal storage")
        semaphore = asyncio.Semaphore(16)

        async def write(row):
            async with semaphore:
                await self.filesystem().write_file(row["uri"], row["text"], ctx=ctx)

        await asyncio.gather(*(write(row) for row in resources))
        hashes = await asyncio.to_thread(
            self.index(ctx).upsert_many,
            [(row["uri"], row["text"], {"kind": row["kind"]}) for row in resources],
        )
        return [
            {"uri": row["uri"], "content_hash": hashes[row["uri"]], "retrieval_mode": "bm25"}
            for row in resources
        ]

    async def publish_corpus(self, ctx: RequestContext, corpus, owner_label: str) -> dict[str, Any]:
        from openviking.service.causal_memory_projection import causal_memory_projection

        rows = causal_memory_projection(corpus, owner_label)
        async with self._write_lock(ctx.account_id):
            for uri, text, _metadata in rows:
                await self.filesystem().write_file(uri, text, ctx=ctx)
            await asyncio.to_thread(self.index(ctx).upsert_many, rows)
        return {"resources": len(rows), "retrieval_mode": "bm25"}
