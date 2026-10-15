# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""BM25 discovery and bounded, permission-checked expansion of served graph nodes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from openviking.service.lexical_memory import LexicalMemoryIndex
from openviking.session.served_anchor_expansion import (
    CHAIN_TTL_SECONDS,
    DEFAULT_MAX_STEPS,
    UNIQUE_SERVED_CAP,
)

GRAPH_RELATIONS = (
    "incoming",
    "outgoing",
    "source",
    "target",
    "forks",
    "anchor",
    "branch_steps",
    "next",
    "previous",
    "observed_block",
)
CanRead = Callable[[str], Awaitable[bool]]


@dataclass
class _Chain:
    principal: str
    generation: int
    expires: float
    served: dict[str, str]
    steps: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    operations: dict[str, tuple[str, dict[str, Any]]] = field(default_factory=dict)


class LexicalMemoryRetrieval:
    def __init__(self, index: LexicalMemoryIndex, *, clock: Callable[[], float] = time.monotonic):
        self.index = index
        self.clock = clock
        self._chains: dict[str, _Chain] = {}

    async def find(
        self,
        query: str,
        *,
        principal: str,
        can_read: CanRead,
        target_uri: str = "viking://resources",
        limit: int = 10,
    ) -> dict[str, Any]:
        if not 1 <= limit <= 10:
            raise ValueError("invalid result limit")
        generation = await asyncio.to_thread(lambda: self.index.generation)
        hits: list[dict[str, Any]] = []
        offset = 0
        while len(hits) < limit:
            candidates = await asyncio.to_thread(
                self.index.find, query, target_uri=target_uri, limit=50, offset=offset
            )
            for candidate in candidates:
                if await can_read(candidate["uri"]):
                    hits.append(candidate)
                    if len(hits) == limit:
                        break
            if len(candidates) < 50:
                break
            offset += 50
        if await asyncio.to_thread(lambda: self.index.generation) != generation:
            raise ValueError("snapshot_stale")
        now = self.clock()
        self._chains = {key: chain for key, chain in self._chains.items() if chain.expires > now}
        chain_ref = secrets.token_urlsafe(24)
        self._chains[chain_ref] = _Chain(
            principal,
            generation,
            now + CHAIN_TTL_SECONDS,
            {hit["uri"]: hit["content_hash"] for hit in hits},
        )
        return self._response(chain_ref, hits, 0)

    async def expand(
        self,
        chain_ref: str,
        uri: str,
        relation: str,
        *,
        principal: str,
        can_read: CanRead,
        operation_id: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        chain = self._chains.get(chain_ref)
        if chain is None:
            raise ValueError("chain_unavailable")
        async with chain.lock:
            return await self._expand(
                chain_ref,
                uri,
                relation,
                principal=principal,
                can_read=can_read,
                operation_id=operation_id,
                limit=limit,
            )

    async def _expand(
        self,
        chain_ref: str,
        uri: str,
        relation: str,
        *,
        principal: str,
        can_read: CanRead,
        operation_id: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        chain = self._chains.get(chain_ref)
        if chain is None or chain.principal != principal or chain.expires <= self.clock():
            raise ValueError("chain_unavailable")
        if relation not in GRAPH_RELATIONS or not 1 <= limit <= 10:
            raise ValueError("invalid_expansion")
        if uri not in chain.served:
            raise ValueError("anchor_not_served")
        if await asyncio.to_thread(lambda: self.index.generation) != chain.generation:
            raise ValueError("snapshot_stale")
        fingerprint = hashlib.sha256(json.dumps([uri, relation, limit]).encode()).hexdigest()
        if not await can_read(uri):
            raise ValueError("anchor_unavailable")
        anchor = await asyncio.to_thread(self.index.get, uri)
        if anchor is None or anchor["content_hash"] != chain.served[uri]:
            raise ValueError("anchor_unavailable")
        if anchor["metadata"].get("kind") not in {
            "native_block",
            "native_edge",
            "fork_overlay",
            "fork_step",
        }:
            raise ValueError("anchor_not_graph")
        if operation_id in chain.operations:
            saved_hash, response = chain.operations[operation_id]
            if saved_hash != fingerprint:
                raise ValueError("operation_conflict")
            for hit in response["resources"]:
                if not await can_read(hit["uri"]):
                    raise ValueError("replayed_target_unavailable")
            return {**response, "duplicate": True}
        if chain.steps >= DEFAULT_MAX_STEPS:
            raise ValueError("chain_step_cap")
        hits: list[dict[str, Any]] = []
        for target_uri in anchor["metadata"].get("links", {}).get(relation, []):
            if target_uri in chain.served or any(hit["uri"] == target_uri for hit in hits):
                continue
            if not await can_read(target_uri):
                continue
            target = await asyncio.to_thread(self.index.get, target_uri)
            if target is not None:
                hits.append(target)
            if len(hits) == limit:
                break
        if len(chain.served) + len(hits) > UNIQUE_SERVED_CAP:
            raise ValueError("chain_served_cap")
        if await asyncio.to_thread(lambda: self.index.generation) != chain.generation:
            raise ValueError("snapshot_stale")
        chain.steps += 1
        chain.served.update({hit["uri"]: hit["content_hash"] for hit in hits})
        response = self._response(chain_ref, hits, chain.steps)
        chain.operations[operation_id] = (fingerprint, response)
        return response

    @staticmethod
    def _response(chain_ref: str, hits: list[dict[str, Any]], step: int) -> dict[str, Any]:
        relations = {
            "native_block": ["incoming", "outgoing", "forks"],
            "native_edge": ["source", "target"],
            "fork_overlay": ["anchor", "branch_steps"],
            "fork_step": ["next", "previous", "observed_block"],
        }
        public = []
        for hit in hits:
            metadata = hit.get("metadata", {})
            safe = {
                key: metadata[key] for key in ("kind", "status", "source_ref") if key in metadata
            }
            safe["relations"] = relations.get(metadata.get("kind"), [])
            public.append({**hit, "metadata": safe})
        return {
            "resources": public,
            "total": len(public),
            "retrieval_mode": "bm25",
            "chain_ref": chain_ref,
            "step": step,
            "remaining_hops": DEFAULT_MAX_STEPS - step,
            "exhausted": not hits,
            "duplicate": False,
        }
