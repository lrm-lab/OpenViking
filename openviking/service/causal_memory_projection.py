# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Publish owner-approved frozen AIG/Fork summaries and explicit navigation links."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from openviking.session.active_retrieval import ActiveRetrievalCorpus
from openviking.session.aig_native_protocol import (
    BaseRevisionCatalog,
    block_content_hash,
    validate_v2_fork_grounding,
)
from openviking.session.fork_node_overlay import ForkOverlayError, guard_v2_fork_consumer


def causal_memory_projection(
    corpus: ActiveRetrievalCorpus, owner_label: str
) -> list[tuple[str, str, dict[str, Any]]]:
    nodes: dict[str, tuple[str, dict[str, Any]]] = {}
    revisions = {}
    blocks = {}

    def uri(ref: str) -> str:
        return (
            "viking://resources/causal/"
            + hashlib.sha256(
                json.dumps([owner_label, ref], separators=(",", ":")).encode()
            ).hexdigest()
            + ".md"
        )

    def revision_namespace(revision) -> str:
        identity = [revision.session_id, revision.task_run_id, revision.revision_id]
        return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()

    def add(ref: str, kind: str, text: str, status: str):
        key = uri(ref)
        nodes[key] = (
            f"{kind} ({status})\nSource: {ref}\n\n{text}",
            {"kind": kind, "status": status, "source_ref": ref, "links": {}},
        )
        return key

    def link(source: str, relation: str, target: str):
        nodes[source][1]["links"].setdefault(relation, []).append(target)

    for bound in corpus.native_views:
        if owner_label not in bound.acl_labels or not bound.revision.frozen_at:
            continue
        revision = bound.revision
        revisions.setdefault(revision.revision_id, []).append(revision)
        namespace = revision_namespace(revision)
        for block in revision.blocks:
            ref = f"native-block:{namespace}:{block.block_id}"
            blocks[(namespace, block.block_id)] = add(
                ref, "native_block", block.summary, "recorded"
            )
        for edge in revision.edges:
            ref = f"native-edge:{namespace}:{edge.edge_id}"
            key = add(
                ref,
                "native_edge",
                f"Recorded AIG relation: {edge.relation_type}.\n{edge.mechanism}\nThis recorded explanation is not independent proof of causality.",
                "recorded",
            )
            source = blocks[(namespace, edge.source_block_id)]
            target = blocks[(namespace, edge.target_block_id)]
            link(key, "source", source)
            link(key, "target", target)
            link(source, "outgoing", key)
            link(target, "incoming", key)
    overlays = []
    if corpus.overlay_store:
        try:
            overlays = corpus.overlay_store.visible_overlays(principal_labels=(owner_label,))
        except ForkOverlayError as error:
            if error.code != "acl_denied":
                raise
    for stored in overlays:
        guard_v2_fork_consumer(stored, "v2_retrieval")
        overlay = stored.overlay
        matches = revisions.get(overlay.base_revision_id, [])
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError("ambiguous fork base revision across sessions")
        revision = matches[0]
        namespace = revision_namespace(revision)
        catalog = BaseRevisionCatalog(
            revision.revision_id,
            block_content_hash(revision.blocks),
            tuple(b.block_id for b in revision.blocks),
        )
        validate_v2_fork_grounding(overlay, catalog)
        anchor = blocks[(namespace, overlay.anchor_block_id)]
        summary = next(b.summary for b in revision.blocks if b.block_id == overlay.anchor_block_id)
        fork = add(
            f"fork-overlay:{namespace}:{overlay.fork_node_id}",
            "fork_overlay",
            f"Alternative branch anchored at: {summary}.\nCounterfactual branch; its steps are not observed execution.",
            "counterfactual",
        )
        link(fork, "anchor", anchor)
        link(anchor, "forks", fork)
        previous = None
        for index, step in enumerate(overlay.branch_steps):
            label = step.block_id or step.imagined_node_id
            key = add(
                f"fork-step:{namespace}:{overlay.fork_node_id}:{index}",
                "fork_step",
                f"Branch step {index}: {step.step_kind} {label}.\nMembership in a Fork is not evidence that the step executed.",
                "counterfactual",
            )
            link(fork, "branch_steps", key)
            if step.block_id:
                link(key, "observed_block", blocks[(namespace, step.block_id)])
            if previous:
                link(previous, "next", key)
                link(key, "previous", previous)
            previous = key
    return [(key, text, metadata) for key, (text, metadata) in nodes.items()]
