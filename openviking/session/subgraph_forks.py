# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Write one ordinary fork for one failure-marked retrieval subgraph.

Diagnosis calls this after commit-time extraction. Memory tools do not.
Each call writes one provisional fork through
``CausalExperiencesStore.submit_fork_draft`` and
``persist_provisional_draft`` (which calls ``upsert_fork_node``). There is
no batch write, no new fork kind, and no stored group of failures.

Implementation choices the design leaves open:

* The anchor is ``failure_evidence_refs[0]`` as ``anchor_kind="ao"``.
  ``contemporaneous_basis`` cannot be that same AO (its sequence is the
  anchor bound), so the caller must pass it. ``workspace_id``,
  ``diagnosis_run_id``, and ``model_version`` are not on the frozen view;
  the caller must pass those too. Empty stand-ins are not filled in.
  ``workspace_labels`` is the caller's label set for this session.
  ``workspace_id`` must be one of them. An empty or missing set is rejected
  before any write. This module does not take an ACL callback.
* ``hindsight_attribution`` and ``used_skills`` default to an empty list
  when the caller omits them. That is the schema's absence, not a fabricated
  ref or skill. Guidance stays on ``branches[].guidance``.
* The default grounding position is the one block in the subgraph whose
  ``source_refs`` contain that failure AO (``ao_id``, session, and
  sequence). Zero or several matches raise. The position is a block, not
  an edge: the failure evidence is an AO, and a block is what covers an AO.
* A caller-supplied ``position_kind="block"`` must name a block on the
  revision whose ``source_refs`` cover ``failure_evidence_refs[0]``
  (``_block_covers_evidence``: same ``ao_id``, session, and sequence).
  A miss raises before any draft is submitted, so nothing is written.
  Other position kinds are not checked against that AO. They still have
  to sit inside the subgraph and yield a non-empty summary snapshot.
  ``validate_view`` still does not resolve evidence against a ledger.
* ``semantic_summary_snapshot`` is the block ``summary``, or for an edge the
  claim's ``carried_artifact`` and ``downstream_effect`` joined by a newline.
  An empty snapshot raises.
* ``content_hash`` is the frozen revision's ``content_hash``.
  ``grounded_at`` and the fork ``created_at`` are that revision's
  ``frozen_at``. A wall clock would change the upsert payload on retry and
  turn an identical write into an idempotency conflict.
* ``fork_node_id`` is a uuid5 of workspace plus the canonical anchor key.
  ``revision_id`` is a uuid5 of the frozen revision id plus the subgraph id.
  The idempotency key is only that pair. A second write of the same pair
  with the same payload hits. Different payload conflicts and does not
  create another fork. Two subgraphs whose failure AOs differ get two
  anchor identities and therefore two fork nodes.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from openviking.session.causal_experiences import (
    ANCHOR_KIND_AO,
    SEVEN_SECTION_KEYS,
    CausalExperiencesStore,
    EvidenceRef,
    ForkRevision,
    InfluenceGrounding,
    _canonical_dumps,
    _require_str,
    canonical_anchor_key,
)
from openviking.session.influence_view import (
    InfluenceBlock,
    InfluenceClaim,
    InfluenceViewRevision,
    RetrievalSubgraph,
)
from openviking.session.legacy_write_gate import require_legacy_audit_opt_in
from openviking.session.retrieval_facade import _position_inside_subgraph

_NODE_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "openviking.failure-subgraph-fork-node")
_REVISION_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "openviking.failure-subgraph-fork-revision")
_UNRESOLVED_SUMMARY = "unresolved"


def failure_subgraphs_owed_forks(
    revision: InfluenceViewRevision,
) -> tuple[RetrievalSubgraph, ...]:
    """Failure-marked subgraphs on ``revision``, in ``retrieval_subgraphs`` order."""
    if not isinstance(revision, InfluenceViewRevision):
        raise ValueError("revision must be an InfluenceViewRevision")
    return tuple(item for item in revision.retrieval_subgraphs if item.is_failure)


def failure_subgraph_fork_idempotency_key(view_revision_id: str, subgraph_id: str) -> str:
    """Stable upsert key for one failure subgraph on one frozen revision."""
    return _canonical_dumps(
        {
            "op": "failure_subgraph_fork",
            "view_revision_id": _require_str(view_revision_id, "view_revision_id"),
            "subgraph_id": _require_str(subgraph_id, "subgraph_id"),
        }
    )


def default_failure_grounding_position(
    revision: InfluenceViewRevision,
    subgraph_id: str,
) -> tuple[str, str]:
    """Block position covering ``failure_evidence_refs[0]`` inside the subgraph.

    A block covers the AO when one of its ``source_refs`` has the same
    ``ao_id``, ``session_id``, and ``sequence`` as that evidence. The scan
    follows ``block_ids`` order and keeps the first match per block id.
    Exactly one covering block is accepted. The result is
    ``("block", "block:<block_id>")``.
    """
    subgraph = _require_failure_subgraph(revision, subgraph_id)
    evidence = _failure_anchor_evidence(subgraph)
    blocks = {block.block_id: block for block in revision.blocks}
    matches: list[str] = []
    seen: set[str] = set()
    for block_id in subgraph.block_ids:
        if block_id in seen:
            continue
        seen.add(block_id)
        block = blocks.get(block_id)
        if block is None or not _block_covers_evidence(block, evidence):
            continue
        matches.append(block_id)
    if len(matches) != 1:
        raise ValueError(
            f"failure evidence ao {evidence.ao_id!r} is covered by {len(matches)} "
            f"blocks in subgraph {subgraph.subgraph_id!r}; expected exactly one"
        )
    return ("block", f"block:{matches[0]}")


def _require_workspace_in_labels(workspace_id: str, workspace_labels: Sequence[str]) -> None:
    """Reject a workspace id that the caller did not list as visible.

    Fail closed: a missing set, a non-sequence, or an empty set does not write.
    """
    if isinstance(workspace_labels, (str, bytes)) or not isinstance(workspace_labels, Sequence):
        raise ValueError("workspace_labels must be a non-empty sequence of str")
    labels: list[str] = []
    for item in workspace_labels:
        if not isinstance(item, str) or not item:
            raise ValueError("workspace_labels entries must be non-empty str")
        labels.append(item)
    if workspace_id not in labels:
        raise ValueError(f"workspace_id {workspace_id!r} is not in workspace_labels")


def write_failure_subgraph_fork(
    store: CausalExperiencesStore,
    revision: InfluenceViewRevision,
    subgraph_id: str,
    *,
    seven_sections: Mapping[str, str],
    branches: Sequence[Mapping[str, Any]],
    position_kind: str,
    position_ref: str,
    workspace_id: str,
    workspace_labels: Sequence[str],
    diagnosis_run_id: str,
    model_version: str,
    contemporaneous_basis: Sequence[Mapping[str, Any] | EvidenceRef],
    hindsight_attribution: Sequence[Mapping[str, Any] | EvidenceRef] | None = None,
    used_skills: Sequence[Mapping[str, Any]] | None = None,
    legacy_audit_opt_in: bool = False,
) -> ForkRevision:
    """Write one provisional fork for one failure subgraph on ``revision``.

    A second call with the same revision and subgraph id and the same payload
    returns that fork. This function does not accept a list of subgraph ids
    and does not return a group of forks.

    ``workspace_id`` must belong to ``workspace_labels``. Otherwise this
    raises and does not write.

    The default refuses before any draft is submitted.
    ``legacy_audit_opt_in=True`` is the explicit legacy-audit tool.
    """
    require_legacy_audit_opt_in(legacy_audit_opt_in)
    if not isinstance(store, CausalExperiencesStore):
        raise ValueError("store must be a CausalExperiencesStore")
    subgraph = _require_failure_subgraph(revision, subgraph_id)
    evidence = _failure_anchor_evidence(subgraph)
    if contemporaneous_basis is None:
        raise ValueError(
            "contemporaneous_basis is required; failure_evidence_refs[0] is the "
            "anchor, so its source_sequence is not a contemporaneous basis"
        )
    anchor = {
        "anchor_kind": ANCHOR_KIND_AO,
        "ao_id": evidence.ao_id,
        "source_session_id": evidence.source_session_id,
        "anchor_sequence": evidence.source_sequence,
    }
    workspace = _require_str(workspace_id, "workspace_id")
    _require_workspace_in_labels(workspace, workspace_labels)
    diagnosis = _require_str(diagnosis_run_id, "diagnosis_run_id")
    model = _require_str(model_version, "model_version")
    kind = _require_str(position_kind, "position_kind")
    ref = _require_str(position_ref, "position_ref")
    summary = _semantic_summary_snapshot(revision, kind, ref)
    # The stand-in snapshot only feeds the R4a position parser. A miss raises
    # before any write, and a hit with no real summary raises as well.
    grounding = InfluenceGrounding(
        view_revision_id=revision.revision_id,
        position_kind=kind,  # type: ignore[arg-type]
        position_ref=ref,
        semantic_summary_snapshot=summary if summary is not None else _UNRESOLVED_SUMMARY,
        content_hash=revision.content_hash,
        grounded_at=revision.frozen_at,
    )
    if not _position_inside_subgraph(grounding, subgraph):
        raise ValueError(
            f"grounding position {kind}:{ref!r} is not inside subgraph {subgraph.subgraph_id!r}"
        )
    # Block positions must cover the anchor. Edge and any other kind that
    # the inside-subgraph parser accepts keep the rules above and below.
    _require_block_position_covers_anchor(revision, subgraph, evidence, kind, ref)
    if summary is None:
        raise ValueError(
            f"cannot derive semantic_summary_snapshot for grounding position {kind}:{ref!r}"
        )
    grounding = InfluenceGrounding(
        view_revision_id=revision.revision_id,
        position_kind=kind,  # type: ignore[arg-type]
        position_ref=ref,
        semantic_summary_snapshot=summary,
        content_hash=revision.content_hash,
        grounded_at=revision.frozen_at,
    )
    fork_node_id = str(
        uuid.uuid5(
            _NODE_NAMESPACE,
            _canonical_dumps({"workspace_id": workspace, "anchor": canonical_anchor_key(anchor)}),
        )
    )
    revision_id = str(
        uuid.uuid5(
            _REVISION_NAMESPACE,
            _canonical_dumps(
                {
                    "view_revision_id": revision.revision_id,
                    "subgraph_id": subgraph.subgraph_id,
                }
            ),
        )
    )
    payload = {
        "fork_node_id": fork_node_id,
        "revision_id": revision_id,
        "created_at": revision.frozen_at,
        "workspace_id": workspace,
        "anchor": anchor,
        "seven_sections": _seven_sections(seven_sections),
        "contemporaneous_basis": _evidence_payload(contemporaneous_basis, "contemporaneous_basis"),
        "hindsight_attribution": _evidence_payload(
            () if hindsight_attribution is None else hindsight_attribution,
            "hindsight_attribution",
        ),
        "used_skills": _skill_payload(used_skills),
        "branches": _branch_payload(branches),
        "diagnosis_run_id": diagnosis,
        "model_version": model,
    }
    draft = store.submit_fork_draft(payload, influence_grounding=grounding)
    return store.persist_provisional_draft(
        draft,
        idempotency_key=failure_subgraph_fork_idempotency_key(
            revision.revision_id, subgraph.subgraph_id
        ),
    )


def _require_failure_subgraph(
    revision: InfluenceViewRevision, subgraph_id: str
) -> RetrievalSubgraph:
    if not isinstance(revision, InfluenceViewRevision):
        raise ValueError("revision must be an InfluenceViewRevision")
    identity = _require_str(subgraph_id, "subgraph_id")
    matches = [item for item in revision.retrieval_subgraphs if item.subgraph_id == identity]
    if not matches:
        raise ValueError(f"subgraph {identity!r} is not on view revision {revision.revision_id!r}")
    if len(matches) != 1:
        raise ValueError(
            f"subgraph {identity!r} is not unique on view revision {revision.revision_id!r}"
        )
    subgraph = matches[0]
    if not subgraph.is_failure:
        raise ValueError(f"subgraph {identity!r} is not failure-marked")
    return subgraph


def _failure_anchor_evidence(subgraph: RetrievalSubgraph):
    if not subgraph.failure_evidence_refs:
        raise ValueError(
            f"subgraph {subgraph.subgraph_id!r} has no failure_evidence_refs; "
            "cannot derive an ao anchor"
        )
    return subgraph.failure_evidence_refs[0]


def _require_block_position_covers_anchor(
    revision: InfluenceViewRevision,
    subgraph: RetrievalSubgraph,
    evidence: Any,
    position_kind: str,
    position_ref: str,
) -> None:
    """Reject a block position that does not cover the failure anchor.

    ``position_kind="block"`` requires ``position_ref`` ``block:<block_id>``
    and ``_block_covers_evidence`` on that block. The check runs only after
    the position is already inside the subgraph, and before any store write.
    Any other ``position_kind`` returns without looking at the anchor.
    """
    if position_kind != "block":
        return
    prefix, separator, ident = position_ref.partition(":")
    block = None
    if separator == ":" and prefix == "block" and ident:
        for item in revision.blocks:
            if item.block_id == ident:
                block = item
                break
    if block is None or not _block_covers_evidence(block, evidence):
        raise ValueError(
            f"block position {position_ref!r} does not cover failure evidence "
            f"ao {evidence.ao_id!r} on subgraph {subgraph.subgraph_id!r}"
        )


def _block_covers_evidence(block: InfluenceBlock, evidence: Any) -> bool:
    for ref in block.source_refs:
        if (
            ref.ao_id == evidence.ao_id
            and ref.session_id == evidence.source_session_id
            and ref.sequence == evidence.source_sequence
        ):
            return True
    return False


def _semantic_summary_snapshot(
    revision: InfluenceViewRevision, position_kind: str, position_ref: str
) -> str | None:
    prefix, separator, ident = position_ref.partition(":")
    if separator != ":" or not ident:
        return None
    if position_kind == "block" and prefix == "block":
        for block in revision.blocks:
            if block.block_id == ident:
                text = block.summary.strip()
                return text or None
        return None
    if position_kind == "edge" and prefix == "edge":
        for claim in revision.claims:
            if claim.claim_id == ident:
                return _claim_summary(claim)
        return None
    return None


def _claim_summary(claim: InfluenceClaim) -> str | None:
    parts = [claim.carried_artifact.strip(), claim.downstream_effect.strip()]
    text = "\n".join(part for part in parts if part)
    return text or None


def _seven_sections(sections: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(sections, Mapping):
        raise ValueError("seven_sections must be a dict")
    missing = [key for key in SEVEN_SECTION_KEYS if key not in sections]
    if missing:
        raise ValueError(f"seven_sections missing keys: {missing}")
    return {key: sections[key] for key in SEVEN_SECTION_KEYS}


def _evidence_payload(items: Sequence[Any], field: str) -> list[dict[str, Any]]:
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise ValueError(f"{field} must be a list")
    payload: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if isinstance(item, EvidenceRef):
            payload.append(item.to_dict())
            continue
        if isinstance(item, Mapping):
            payload.append(dict(item))
            continue
        raise ValueError(f"{field}[{index}] must be an evidence ref")
    return payload


def _branch_payload(branches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(branches, (str, bytes)) or not isinstance(branches, Sequence):
        raise ValueError("branches must be a list")
    payload: list[dict[str, Any]] = []
    for index, item in enumerate(branches):
        if not isinstance(item, Mapping):
            raise ValueError(f"branches[{index}] must be a dict")
        payload.append(dict(item))
    return payload


def _skill_payload(skills: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if skills is None:
        return []
    if isinstance(skills, (str, bytes)) or not isinstance(skills, Sequence):
        raise ValueError("used_skills must be a list")
    payload: list[dict[str, Any]] = []
    for index, item in enumerate(skills):
        if not isinstance(item, Mapping):
            raise ValueError(f"used_skills[{index}] must be a dict")
        payload.append(dict(item))
    return payload
