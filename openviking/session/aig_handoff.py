# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Deterministic, version-bound role handoffs; claims never become raw evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from openviking.session.aig_native_protocol import canonical_dumps


def graph_index(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Complete topology and block handles, without model-authored prose."""
    return {
        "blocks": [
            {k: b[k] for k in ("block_id", "kind", "step_refs") if k in b}
            for b in graph.get("blocks", [])
        ],
        "edges": [
            {"edge_index": i, **{k: e[k] for k in ("from", "to", "link_kind") if k in e}}
            for i, e in enumerate(graph.get("edges", []))
        ],
        "submitted": bool(graph.get("submitted")),
    }


def build_handoff(
    graph: Mapping[str, Any],
    *,
    role: str,
    snapshot: Any = None,
    violations: Sequence[Mapping[str, Any]] = (),
    edits: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Export facts, upstream claims, and pending checks as separate namespaces."""
    if role not in {"builder", "critic", "refiner", "reader"}:
        raise ValueError("invalid_handoff_role")
    facts: dict[str, Any] = {"evidence_tools": ["read_step", "read_step_range"]}
    if snapshot is not None:
        facts.update(
            {
                "watermark": snapshot.watermark,
                "question": snapshot.question,
                "ao_index": [
                    {
                        "index": i,
                        "ao_id": r.ao_id,
                        "sha256": hashlib.sha256(canonical_dumps(r.to_dict()).encode()).hexdigest(),
                    }
                    for i, r in enumerate(snapshot.records)
                ],
            }
        )
    packet = {
        "schema": "aig-role-handoff/v1",
        "role": role,
        "graph_hash": hashlib.sha256(canonical_dumps(graph).encode()).hexdigest(),
        "graph_index": graph_index(graph),
        "raw_facts": facts,
        "upstream_claims": {"available_via": "get_graph_block"}
        if role == "reader"
        else {"graph": graph, "violations": list(violations), "edits": list(edits)},
        "pending_verification": {
            "edge_indices": list(range(len(graph.get("edges", [])))),
            "rule": "Read original AO before attributing an effect; upstream claims are unverified.",
        },
    }
    return json.loads(canonical_dumps(packet))
