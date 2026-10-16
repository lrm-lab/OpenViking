# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Typed structural signature for one retrieval subgraph.

The production diagnostic channel ranks ``_subgraph_retrieval_text`` and
drops roles, relation types, the failure marker, ids, and raw AO text.
This module reads those discarded fields and nothing else: no block
summary, input, or output, no claim prose, and no seven-section text.

Rules in this file are frozen. The runner prints them before any score.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openviking.session.causal_experiences import POSITION_KINDS
from openviking.session.influence_view import (
    BLOCK_ROLES,
    RELATION_INFLUENCE,
    RELATION_SKIP,
    InfluenceClaim,
    InfluenceViewRevision,
    RetrievalSubgraph,
)
from openviking.session.retrieval_facade import DIAGNOSTIC_SUBGRAPH_HIT_CAP
from openviking.session.subgraph_forks import default_failure_grounding_position

# Frozen before scores. Do not edit these to absorb a miss.
SCHEMA = "r7b-structural-signature-v1"
ROLE_ORDER = tuple(sorted(BLOCK_ROLES))
BOUNDARY_VALUES = ("call_cut", "result_cut", "text_end")
INCIDENCE_VALUES = (
    "influence_upstream",
    "influence_downstream",
    "influence_both",
    "skip_only",
)
TRUNCATION_UNKNOWN = "unknown"
WORKSPACE_HASH_CHARS = 16
SKILL_KIND_HASH_CHARS = 16

# Suggested weights from R7b-design.md §3, adopted as the frozen table.
# ``workspace_id`` is not a weighted field: a mismatch drops the candidate.
FIELD_WEIGHTS: tuple[tuple[str, int], ...] = (
    ("boundary", 2),
    ("anchor_kind", 1),
    ("anchor_incidence", 2),
    ("relation_profile", 1),
    ("role_profile", 1),
    ("coverage_crossing", 2),
    ("member_gap", 1),
    ("truncation_lines", 1),
    ("grounding_kind", 1),
    ("evidence_status", 1),
    ("overlaps_anchor", 2),
    ("used_skill_kinds", 1),
)
TIE_BREAK = ("session_id", "subgraph_id")
CAP = DIAGNOSTIC_SUBGRAPH_HIT_CAP

BOUNDARY_RULE = (
    "boundary is derived from the tail member and the claim that links the "
    "previous member to it. Members are the subgraph blocks sorted by "
    "(ao_end_seq, block_id). The tail is the last member. "
    "text_end: the tail role is not agent_action. "
    "result_cut: the tail role is agent_action and the subgraph claim whose "
    "source is the previous member and whose target is the tail has "
    "relation_type influence. "
    "call_cut: the tail role is agent_action and that influence link is absent. "
    "No summary, input, output, carried_artifact, or downstream_effect is read."
)
INCIDENCE_RULE = (
    "anchor_incidence uses the block returned by "
    "default_failure_grounding_position. influence_upstream: that block is the "
    "target of a subgraph claim with relation_type influence and not a source. "
    "influence_downstream: source only. influence_both: both. "
    "skip_only: it is not an endpoint of any influence claim in the subgraph."
)
RELATION_PROFILE_RULE = (
    "relation_profile is (influence_n, skip_n) over subgraph claims, plus one "
    "row per block_id in subgraph order: (index, source_influence, source_skip, "
    "target_influence, target_skip). The index is the position in block_ids, "
    "not the block id string. Claims whose relation_type is neither influence "
    "nor skip are counted in neither column."
)
ROLE_PROFILE_RULE = (
    "role_profile is the count of subgraph member blocks for each role in "
    f"sorted(BLOCK_ROLES) order {ROLE_ORDER}."
)
COVERAGE_RULE = (
    "coverage_crossing is true when some subgraph member span [ao_start_seq, "
    "ao_end_seq] touches an omitted range: member_end + 1 >= omitted_start and "
    "omitted_end + 1 >= member_start. That is overlap or adjacency. A gap of "
    "one or more sequences is not a crossing. The omitted reason string is not read."
)
MEMBER_GAP_RULE = (
    "member_gap is the maximum of (next.ao_start_seq - prev.ao_end_seq - 1) "
    "over subgraph members sorted by (ao_start_seq, block_id). Zero when the "
    "subgraph has fewer than two members."
)
TRUNCATION_RULE = (
    "truncation_lines is the integer count of bad JSON lines at the tail when "
    "an AO ledger is attached to the revision. InfluenceViewRevision does not "
    "carry a ledger, and seven-section text is not read, so the value is the "
    f"closed token {TRUNCATION_UNKNOWN!r} on every signature in this wave."
)
WORKSPACE_RULE = (
    "workspace_id on the signature is sha256(utf-8 label).hexdigest()[:16]. "
    "The raw label is not stored. Two signatures with different hashes are "
    "incomparable and are dropped before the cap. Equal hashes contribute no "
    "weight; they only gate comparison."
)
FORK_FIELD_RULE = (
    "Candidate-only fields are read from forks the caller passes in, already "
    "restricted to those grounded inside the subgraph. grounding_kind is the "
    "sorted tuple of influence_grounding.position_kind values (empty if none). "
    "evidence_status is the sorted tuple of branch evidence_status values. "
    "overlaps_anchor is true when any branch ao_sequence entry equals "
    "failure_evidence_refs[0].ao_id. used_skill_kinds is the sorted tuple of "
    "sha256(skill_uri).hexdigest()[:16]. Seven-section text, guidance, and "
    "branch prose are not read."
)
MATCH_RULE = (
    "Score is the sum of FIELD_WEIGHTS for fields whose values are equal. "
    "A workspace hash mismatch scores None and the candidate is dropped. "
    f"Ties break by {TIE_BREAK} ascending. Cap is the production "
    f"DIAGNOSTIC_SUBGRAPH_HIT_CAP ({CAP}). No prose, no model, no RNG."
)

POLICY_LINES: tuple[str, ...] = (
    f"SCHEMA {SCHEMA}",
    f"FIELD_WEIGHTS {FIELD_WEIGHTS}",
    f"TIE_BREAK {TIE_BREAK}",
    f"CAP {CAP}",
    f"ROLE_ORDER {ROLE_ORDER}",
    f"BOUNDARY_VALUES {BOUNDARY_VALUES}",
    f"INCIDENCE_VALUES {INCIDENCE_VALUES}",
    BOUNDARY_RULE,
    INCIDENCE_RULE,
    RELATION_PROFILE_RULE,
    ROLE_PROFILE_RULE,
    COVERAGE_RULE,
    MEMBER_GAP_RULE,
    TRUNCATION_RULE,
    WORKSPACE_RULE,
    FORK_FIELD_RULE,
    MATCH_RULE,
)


def print_signature_policy() -> None:
    """Print the frozen signature rules. Callers must do this before scores."""
    print("R7B STRUCTURAL SIGNATURE (frozen before scores)")
    for line in POLICY_LINES:
        print(line)


def workspace_hash(label: str) -> str:
    """Closed id for an ACL label. The raw label is not returned."""
    if not isinstance(label, str) or not label:
        raise ValueError("workspace label must be a non-empty str")
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return digest[:WORKSPACE_HASH_CHARS]


def _members(revision: InfluenceViewRevision, subgraph: RetrievalSubgraph) -> tuple[Any, ...]:
    by_id = {block.block_id: block for block in revision.blocks}
    chosen = tuple(by_id[block_id] for block_id in subgraph.block_ids if block_id in by_id)
    return tuple(sorted(chosen, key=lambda block: (block.ao_start_seq, block.block_id)))


def _subgraph_claims(
    revision: InfluenceViewRevision, subgraph: RetrievalSubgraph
) -> tuple[InfluenceClaim, ...]:
    by_id = {claim.claim_id: claim for claim in revision.claims}
    return tuple(by_id[claim_id] for claim_id in subgraph.claim_ids if claim_id in by_id)


def boundary_of(revision: InfluenceViewRevision, subgraph: RetrievalSubgraph) -> str:
    """Closed boundary token. See ``BOUNDARY_RULE``."""
    members = _members(revision, subgraph)
    if not members:
        raise ValueError(f"subgraph {subgraph.subgraph_id!r} has no blocks")
    tail = max(members, key=lambda block: (block.ao_end_seq, block.block_id))
    if tail.role != "agent_action":
        return "text_end"
    ordered = tuple(sorted(members, key=lambda block: (block.ao_end_seq, block.block_id)))
    prev = None
    for block in ordered:
        if block.block_id == tail.block_id:
            break
        prev = block
    if prev is None:
        return "call_cut"
    for claim in _subgraph_claims(revision, subgraph):
        if (
            claim.source_block_id == prev.block_id
            and claim.target_block_id == tail.block_id
            and claim.relation_type == RELATION_INFLUENCE
        ):
            return "result_cut"
    return "call_cut"


def _anchor_block_id(revision: InfluenceViewRevision, subgraph: RetrievalSubgraph) -> str:
    kind, ref = default_failure_grounding_position(revision, subgraph.subgraph_id)
    if kind not in POSITION_KINDS:
        raise ValueError(f"position kind {kind!r} is outside POSITION_KINDS")
    prefix, separator, ident = ref.partition(":")
    if kind != "block" or separator != ":" or prefix != "block" or not ident:
        raise ValueError(
            f"default_failure_grounding_position returned {(kind, ref)!r}; "
            "this wave's anchor_kind table expects a block position"
        )
    return ident


def anchor_incidence_of(
    revision: InfluenceViewRevision, subgraph: RetrievalSubgraph, anchor_block_id: str
) -> str:
    """Closed incidence token. See ``INCIDENCE_RULE``."""
    upstream = False
    downstream = False
    for claim in _subgraph_claims(revision, subgraph):
        if claim.relation_type != RELATION_INFLUENCE:
            continue
        if claim.target_block_id == anchor_block_id:
            upstream = True
        if claim.source_block_id == anchor_block_id:
            downstream = True
    if upstream and downstream:
        return "influence_both"
    if upstream:
        return "influence_upstream"
    if downstream:
        return "influence_downstream"
    return "skip_only"


def relation_profile_of(
    revision: InfluenceViewRevision, subgraph: RetrievalSubgraph
) -> tuple[Any, ...]:
    """Counts only. Block id strings are not included. See ``RELATION_PROFILE_RULE``."""
    claims = _subgraph_claims(revision, subgraph)
    influence_n = sum(1 for claim in claims if claim.relation_type == RELATION_INFLUENCE)
    skip_n = sum(1 for claim in claims if claim.relation_type == RELATION_SKIP)
    rows: list[tuple[int, int, int, int, int]] = []
    for index, block_id in enumerate(subgraph.block_ids):
        source_influence = 0
        source_skip = 0
        target_influence = 0
        target_skip = 0
        for claim in claims:
            if claim.source_block_id == block_id:
                if claim.relation_type == RELATION_INFLUENCE:
                    source_influence += 1
                elif claim.relation_type == RELATION_SKIP:
                    source_skip += 1
            if claim.target_block_id == block_id:
                if claim.relation_type == RELATION_INFLUENCE:
                    target_influence += 1
                elif claim.relation_type == RELATION_SKIP:
                    target_skip += 1
        rows.append((index, source_influence, source_skip, target_influence, target_skip))
    return (influence_n, skip_n, tuple(rows))


def role_profile_of(
    revision: InfluenceViewRevision, subgraph: RetrievalSubgraph
) -> tuple[int, ...]:
    """Role counts in ``ROLE_ORDER``. See ``ROLE_PROFILE_RULE``."""
    counts = dict.fromkeys(ROLE_ORDER, 0)
    by_id = {block.block_id: block for block in revision.blocks}
    for block_id in subgraph.block_ids:
        block = by_id.get(block_id)
        if block is None:
            continue
        counts[block.role] = counts.get(block.role, 0) + 1
    return tuple(counts[role] for role in ROLE_ORDER)


def coverage_crossing_of(revision: InfluenceViewRevision, subgraph: RetrievalSubgraph) -> bool:
    """See ``COVERAGE_RULE``. The omitted reason is not read."""
    by_id = {block.block_id: block for block in revision.blocks}
    for block_id in subgraph.block_ids:
        block = by_id.get(block_id)
        if block is None:
            continue
        for omitted in revision.coverage.omitted_ranges:
            if (
                block.ao_end_seq + 1 >= omitted.start_seq
                and omitted.end_seq + 1 >= block.ao_start_seq
            ):
                return True
    return False


def member_gap_of(revision: InfluenceViewRevision, subgraph: RetrievalSubgraph) -> int:
    """See ``MEMBER_GAP_RULE``."""
    members = _members(revision, subgraph)
    if len(members) < 2:
        return 0
    gap = 0
    for prev, nxt in zip(members, members[1:], strict=False):
        gap = max(gap, nxt.ao_start_seq - prev.ao_end_seq - 1)
    return gap


def _skill_kind(skill_uri: str) -> str:
    digest = hashlib.sha256(skill_uri.encode("utf-8")).hexdigest()
    return digest[:SKILL_KIND_HASH_CHARS]


def _fork_fields(
    subgraph: RetrievalSubgraph, forks: Sequence[Any]
) -> tuple[tuple[str, ...], tuple[str, ...], bool, tuple[str, ...]]:
    """Closed fork fields. See ``FORK_FIELD_RULE``. Prose is not read."""
    if not subgraph.failure_evidence_refs:
        anchor_ao = ""
    else:
        anchor_ao = subgraph.failure_evidence_refs[0].ao_id
    kinds: list[str] = []
    statuses: list[str] = []
    skills: list[str] = []
    overlaps = False
    for fork in forks:
        grounding = fork.influence_grounding
        if grounding is None:
            continue
        kind = grounding.position_kind
        if kind not in POSITION_KINDS:
            raise ValueError(f"fork position_kind {kind!r} is outside POSITION_KINDS")
        kinds.append(kind)
        branches = fork.branches
        if not isinstance(branches, Sequence) or isinstance(branches, (str, bytes)):
            raise ValueError("fork.branches must be a sequence")
        for branch in branches:
            if not isinstance(branch, Mapping):
                raise ValueError("fork branch must be a mapping")
            status = branch.get("evidence_status")
            if not isinstance(status, str):
                raise ValueError("branch evidence_status must be a str")
            statuses.append(status)
            sequence = branch.get("ao_sequence") or []
            if isinstance(sequence, Sequence) and not isinstance(sequence, (str, bytes)):
                if anchor_ao and any(item == anchor_ao for item in sequence):
                    overlaps = True
        used = fork.used_skills
        if not isinstance(used, Sequence) or isinstance(used, (str, bytes)):
            raise ValueError("fork.used_skills must be a sequence")
        for skill in used:
            if not isinstance(skill, Mapping):
                raise ValueError("used_skills entry must be a mapping")
            uri = skill.get("skill_uri")
            if not isinstance(uri, str):
                raise ValueError("used_skills.skill_uri must be a str")
            skills.append(_skill_kind(uri))
    return (tuple(sorted(kinds)), tuple(sorted(statuses)), overlaps, tuple(sorted(set(skills))))


@dataclass(frozen=True)
class StructuralSignature:
    """Closed vocabulary, integers, and hashes. No natural-language prose."""

    boundary: str
    anchor_kind: str
    anchor_incidence: str
    relation_profile: tuple[Any, ...]
    role_profile: tuple[int, ...]
    coverage_crossing: bool
    member_gap: int
    truncation_lines: str | int
    workspace_id: str
    grounding_kind: tuple[str, ...]
    evidence_status: tuple[str, ...]
    overlaps_anchor: bool
    used_skill_kinds: tuple[str, ...]

    def to_json_obj(self) -> dict[str, Any]:
        influence_n, skip_n, rows = self.relation_profile
        return {
            "anchor_incidence": self.anchor_incidence,
            "anchor_kind": self.anchor_kind,
            "boundary": self.boundary,
            "coverage_crossing": self.coverage_crossing,
            "evidence_status": list(self.evidence_status),
            "grounding_kind": list(self.grounding_kind),
            "member_gap": self.member_gap,
            "overlaps_anchor": self.overlaps_anchor,
            "relation_profile": {
                "endpoints": [
                    {
                        "index": row[0],
                        "source_influence": row[1],
                        "source_skip": row[2],
                        "target_influence": row[3],
                        "target_skip": row[4],
                    }
                    for row in rows
                ],
                "influence_n": influence_n,
                "skip_n": skip_n,
            },
            "role_profile": {
                role: self.role_profile[index] for index, role in enumerate(ROLE_ORDER)
            },
            "truncation_lines": self.truncation_lines,
            "used_skill_kinds": list(self.used_skill_kinds),
            "workspace_id": self.workspace_id,
        }


def structural_signature(
    revision: InfluenceViewRevision,
    subgraph: RetrievalSubgraph,
    *,
    workspace_id: str,
    forks: Sequence[Any] = (),
) -> StructuralSignature:
    """Signature of one subgraph. ``forks`` are already inside the subgraph.

    ``workspace_id`` is the raw ACL label. The signature stores its hash.
    """
    if not isinstance(revision, InfluenceViewRevision):
        raise ValueError("revision must be an InfluenceViewRevision")
    if not isinstance(subgraph, RetrievalSubgraph):
        raise ValueError("subgraph must be a RetrievalSubgraph")
    kind, _ref = default_failure_grounding_position(revision, subgraph.subgraph_id)
    anchor_id = _anchor_block_id(revision, subgraph)
    grounding_kind, evidence_status, overlaps, skills = _fork_fields(subgraph, forks)
    return StructuralSignature(
        boundary=boundary_of(revision, subgraph),
        anchor_kind=kind,
        anchor_incidence=anchor_incidence_of(revision, subgraph, anchor_id),
        relation_profile=relation_profile_of(revision, subgraph),
        role_profile=role_profile_of(revision, subgraph),
        coverage_crossing=coverage_crossing_of(revision, subgraph),
        member_gap=member_gap_of(revision, subgraph),
        truncation_lines=TRUNCATION_UNKNOWN,
        workspace_id=workspace_hash(workspace_id),
        grounding_kind=grounding_kind,
        evidence_status=evidence_status,
        overlaps_anchor=overlaps,
        used_skill_kinds=skills,
    )


def structural_score(query: StructuralSignature, candidate: StructuralSignature) -> int | None:
    """Weighted field-equality. ``None`` means the workspaces differ."""
    if query.workspace_id != candidate.workspace_id:
        return None
    total = 0
    for name, weight in FIELD_WEIGHTS:
        if getattr(query, name) == getattr(candidate, name):
            total += weight
    return total


def field_contrast(query: StructuralSignature, candidate: StructuralSignature) -> dict[str, Any]:
    """Which weighted fields match the query, and which do not."""
    matched: list[dict[str, Any]] = []
    differed: list[dict[str, Any]] = []
    for name, weight in FIELD_WEIGHTS:
        row = {
            "candidate": _jsonable(getattr(candidate, name)),
            "field": name,
            "query": _jsonable(getattr(query, name)),
            "weight": weight,
        }
        if getattr(query, name) == getattr(candidate, name):
            matched.append(row)
        else:
            differed.append(row)
    return {"differed": differed, "matched": matched}


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value
