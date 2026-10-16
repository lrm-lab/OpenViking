# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Legacy-audit coverage of the hybrid retrieval facade (slice S13).

These tests call ``audit_legacy``. Active retrieve, which does not return
Claim, Cut, or legacy fork cards, is covered by ``test_p11_contract.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from openviking.session.ao_ledger import AOLedger
from openviking.session.bm25 import DEFAULT_B, DEFAULT_K1, BM25Index, tokenize
from openviking.session.causal_bridges import PATH_MAX_ENTITIES, check_path_budget
from openviking.session.causal_experiences import (
    SEVEN_SECTION_KEYS,
    CausalExperiencesStore,
    ServerChecks,
)
from openviking.session.influence_projection import ProjectionRegistry
from openviking.session.influence_view import (
    PURPOSE_POST_RUN_INDEX,
    RELATION_INFLUENCE,
    RELATION_SKIP,
    AORange,
    CoverageManifest,
    InfluenceBlock,
    InfluenceClaim,
    InfluenceViewDraft,
    SegmentRef,
    SourceRef,
)
from openviking.session.retrieval_facade import (
    BRANCH_SUMMARY_CHARS,
    BRANCH_SYNOPSIS_LIMIT,
    CAUSAL_MEMORY_CAP,
    DEFAULT_SEGMENT_ATOM_QUOTA,
    FACT_LANE_PAUSED,
    FEATURE_SPEC_VERSION,
    NO_ELIGIBLE_EXPLORATION_REASON,
    SELECTION_POLICY_VERSION,
    ForkBranchAdapter,
    HybridRetrievalFacade,
    InfluenceProjectionAdapter,
    QuotaError,
    RetrievalCard,
    RetrievalProfile,
    SegmentAtomAdapter,
)

_CAUSAL = {"skill_trajectory_mode": "causal"}
_CHECKS = ServerChecks(anchor_exists=True, refs_exist=True, provenance_valid=True)
_EVIDENCE = {
    "observed_branch_refs": [{"branch_id": "br-obs", "evidence_status": "real", "ref": "run-1"}],
    "independent_control_refs": [{"ref_kind": "independent_control", "ref_id": "ctrl-1"}],
    "anchor_valid": {"valid": True, "checked_at": "2026-01-02T00:00:00.000Z"},
    "applicability_check": {
        "task_decision_type": "applicable",
        "anchor_state": "applicable",
        "failure_mode": "applicable",
        "action_skill_role": "applicable",
    },
    "counter_example_check": {"conclusion": "no_counter_example", "refs": ["run-1"]},
    "no_high_severity_contradictions": True,
    "control_contract": {
        "anchor_state_snapshot": "snap-1",
        "task_input": "task-input-1",
        "agent_model_prompt_policy": "policy-v1",
        "tool_dependency_versions": {"toolkit": "1.0.0"},
        "permissions": "read-only",
        "budget": "1000",
        "evaluator": "evaluator-v1",
        "frozen_at": "2026-01-02T00:00:00.000Z",
    },
}
_PRINCIPAL = ("team-a",)
_CREATED = "2026-01-02T00:00:00.000Z"


def _sections() -> dict[str, str]:
    return {key: f"{key} body" for key in SEVEN_SECTION_KEYS}


def _text_sections(text: str) -> dict[str, str]:
    sections = _sections()
    sections["task_and_goals"] = text
    return sections


def _append_text(ledger: AOLedger, text: str, *, record_kind: str = "atom") -> str:
    record = ledger.append(
        {"tool": "read_file", "record_kind": record_kind, "acl_labels": ["team-a"]},
        {"kind": "text", "summary": text},
    )
    return record.ao_id


def _evidence_ref() -> dict[str, object]:
    return {
        "ao_id": "ao-e1",
        "source_session_id": "sess-1",
        "source_archive_id": "arch-1",
        "archive_commit_watermark": "commit-1",
        "source_sequence": 4,
        "source_read_snapshot_watermark": "read-wm-1",
        "evidence_role": "contemporaneous_basis",
        "evidence_content_hash": "hash-ao-e1",
        "captured_state": "committed",
    }


def _store(root: Path) -> CausalExperiencesStore:
    return CausalExperiencesStore(root, config=_CAUSAL)


def _commit_fork(
    store: CausalExperiencesStore,
    *,
    ao_id: str,
    workspace: str,
    status: str,
    branches: list[dict[str, object]] | None = None,
    sections: dict[str, str] | None = None,
) -> str:
    draft = store.submit_fork_draft(
        {
            "workspace_id": workspace,
            "anchor": {
                "anchor_kind": "ao",
                "ao_id": ao_id,
                "source_session_id": "sess-1",
                "anchor_sequence": 10,
            },
            "seven_sections": sections or _sections(),
            "contemporaneous_basis": [_evidence_ref()],
            "hindsight_attribution": [],
            "used_skills": [],
            "branches": branches or [],
            "diagnosis_run_id": f"diag-{ao_id}",
            "model_version": "diag-model-v1",
        }
    )
    provisional = store.commit_provisional(draft, _CHECKS)
    if status == "validated":
        store.append_validated(provisional.fork_node_id, _EVIDENCE)
    elif status != "provisional":
        raise AssertionError(status)
    return provisional.fork_node_id


def _ref(sequence: int, *, session_id: str = "sess-1") -> SourceRef:
    return SourceRef(
        ao_id=f"ao-{session_id}-{sequence}",
        sequence=sequence,
        participant_id="agent-a",
        segment_id="seg-a",
        session_id=session_id,
        content_hash=f"hash-{session_id}-{sequence}",
    )


def _block(
    block_id: str,
    sequence: int,
    *,
    session_id: str = "sess-1",
    summary: str | None = None,
) -> InfluenceBlock:
    ref = _ref(sequence, session_id=session_id)
    return InfluenceBlock(
        block_id=block_id,
        role="agent_action",
        participant_id="agent-a",
        segment_ref=SegmentRef(segment_id="seg-a", session_id=session_id),
        ao_start_seq=sequence,
        ao_end_seq=sequence,
        summary=summary if summary is not None else f"summary-{block_id}",
        input="",
        output="",
        authorship="agent-a",
        granularity_reason="one action",
        source_content_hashes=(ref.content_hash,),
        source_refs=(ref,),
    )


def _coverage(blocks: list[InfluenceBlock]) -> CoverageManifest:
    refs = tuple(ref for block in blocks for ref in block.source_refs)
    return CoverageManifest(
        total_ao_range=AORange(
            start_seq=min(ref.sequence for ref in refs),
            end_seq=max(ref.sequence for ref in refs),
        ),
        inspected_refs=refs,
        included_ranges=tuple(
            AORange(
                start_seq=block.ao_start_seq,
                end_seq=block.ao_end_seq,
                segment_id="seg-a",
            )
            for block in blocks
        ),
        omitted_ranges=(),
        snapshot_watermark="wm-1",
    )


def _freeze(
    blocks: list[InfluenceBlock],
    claims: list[InfluenceClaim],
    *,
    revision_id: str,
    session_id: str = "sess-1",
    enforce_checks: bool = True,
):
    draft = InfluenceViewDraft(
        view_id=f"view-{revision_id}",
        purpose=PURPOSE_POST_RUN_INDEX,
        session_id=session_id,
        task_run_id=f"task-{revision_id}",
        coverage=_coverage(blocks),
        blocks=blocks,
        claims=claims,
    )
    return draft.freeze(
        revision_id=revision_id,
        frozen_at=_CREATED,
        enforce_checks=enforce_checks,
    )


def _facade(
    root: Path,
    store: CausalExperiencesStore,
    registry: ProjectionRegistry,
    session_dir: Path,
    session_id: str,
    *,
    views: list | None = None,
    seed: str | int | None = None,
    fact_lane_paused: bool | None = None,
    embedder=None,
) -> HybridRetrievalFacade:
    projection = InfluenceProjectionAdapter(registry, views=views)
    return HybridRetrievalFacade(
        ForkBranchAdapter(root, store, session_dir=session_dir, session_id=session_id),
        projection,
        SegmentAtomAdapter(session_dir, session_id),
        seed=seed,
        fact_lane_paused=fact_lane_paused,
        embedder=embedder,
    )


def _fork_lane(
    tmp_path: Path,
    name: str,
    *,
    validated: int = 0,
    provisional: int = 0,
    hidden_provisional: int = 0,
) -> tuple[Path, CausalExperiencesStore, Path, list[str], list[str], list[str]]:
    root = tmp_path / name
    store = _store(root)
    hidden_ids = [
        _commit_fork(
            store,
            ao_id=f"ao-h-{index}",
            workspace="secret",
            status="provisional",
        )
        for index in range(hidden_provisional)
    ]
    validated_ids = [
        _commit_fork(
            store,
            ao_id=f"ao-v-{index}",
            workspace="team-a",
            status="validated",
        )
        for index in range(validated)
    ]
    provisional_ids = [
        _commit_fork(
            store,
            ao_id=f"ao-p-{index}",
            workspace="team-a",
            status="provisional",
        )
        for index in range(provisional)
    ]
    session_dir = root / "sess"
    AOLedger(session_dir, "sess")
    return root, store, session_dir, validated_ids, provisional_ids, hidden_ids


def _open_lane(
    root: Path,
    store: CausalExperiencesStore,
    session_dir: Path,
    *,
    seed: str | int | None = None,
) -> HybridRetrievalFacade:
    return _facade(
        root,
        store,
        ProjectionRegistry(root / "projections.json"),
        session_dir,
        "sess",
        seed=seed,
    )


def _fork_id(card: RetrievalCard) -> str:
    return str(card.source_pointer["fork_node_id"])


def _expected_query_seed(text: str, pattern: dict[str, str] | None = None) -> str:
    payload = json.dumps(
        {"text": text, "dependency_pattern": pattern},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).digest()[:8].hex()


def _append_ao(ledger: AOLedger, *, labels: list[str], record_kind: str = "atom") -> str:
    record = ledger.append(
        {"tool": "read_file", "record_kind": record_kind, "acl_labels": labels},
        {"kind": "text", "summary": [record_kind]},
    )
    return record.ao_id


def test_corpus_quotas_and_profile_over_cap(tmp_path: Path) -> None:
    root = tmp_path / "lane"
    store = _store(root)
    fork_ids = [
        _commit_fork(store, ao_id=f"ao-fork-{index}", workspace="team-a", status="validated")
        for index in range(8)
    ]
    blocks = [_block(f"b{index}", index) for index in range(1, 7)]
    revision = _freeze(blocks, [], revision_id="rev-quota")
    registry = ProjectionRegistry(root / "projections.json")
    registry.register(revision, acl_labels=_PRINCIPAL)
    session_dir = root / "sess-facts"
    ledger = AOLedger(session_dir, "sess-facts")
    for _ in range(4):
        _append_ao(ledger, labels=["team-a"])

    facade = _facade(root, store, registry, session_dir, "sess-facts", views=[revision])
    assert facade.fact_lane_paused is FACT_LANE_PAUSED is True
    assert DEFAULT_SEGMENT_ATOM_QUOTA == 2
    assert CAUSAL_MEMORY_CAP == 12
    result = facade.audit_legacy({}, _PRINCIPAL, RetrievalProfile())
    forks = [card for card in result.cards if card.kind == "fork"]
    projections = [card for card in result.cards if card.kind.endswith("_projection")]
    facts = [card for card in result.cards if card.kind in {"segment", "atom"}]
    assert len(fork_ids) == 8
    assert len(forks) == 6
    assert [card.source_pointer["fork_node_id"] for card in forks] == fork_ids[:6]
    assert len(projections) == 4
    assert facts == []
    assert len(result.cards) == 10
    assert len(result.cards) <= 10
    assert all(card.channel != "fact" for card in result.cards)
    assert result.rank_tier == "bm25"
    assert result.feature_version == FEATURE_SPEC_VERSION == facade.feature_version
    assert RetrievalCard.from_dict(result.cards[0].to_dict()) == result.cards[0]

    resumed = _facade(
        root,
        store,
        registry,
        session_dir,
        "sess-facts",
        views=[revision],
        fact_lane_paused=False,
    )
    resumed_result = resumed.audit_legacy({}, _PRINCIPAL, RetrievalProfile())
    resumed_forks = [card for card in resumed_result.cards if card.kind == "fork"]
    resumed_projections = [
        card for card in resumed_result.cards if card.kind.endswith("_projection")
    ]
    resumed_facts = [card for card in resumed_result.cards if card.kind in {"segment", "atom"}]
    assert len(resumed_forks) == 6
    assert len(resumed_projections) == 4
    assert len(resumed_facts) == 2
    assert {card.kind for card in resumed_facts} == {"atom"}
    assert {card.channel for card in resumed_facts} == {"fact"}
    assert len(resumed_result.cards) == 12

    with pytest.raises(QuotaError) as raised:
        facade.audit_legacy(
            {},
            _PRINCIPAL,
            RetrievalProfile(fork_branch=7, projection=4, segment_atom=2),
        )
    assert raised.value.total == 13
    assert raised.value.cap == 12


def test_branch_cards_follow_parent_channel(tmp_path: Path) -> None:
    root = tmp_path / "branches"
    store = _store(root)
    branch = {
        "branch_id": "br-1",
        "evidence_status": "real",
        "ao_sequence": ["ao-step"],
        "guidance": {"summary": "retry the read"},
    }
    _commit_fork(
        store,
        ao_id="ao-prov",
        workspace="team-a",
        status="provisional",
        branches=[branch],
    )
    _commit_fork(
        store,
        ao_id="ao-val",
        workspace="team-a",
        status="validated",
        branches=[branch],
    )
    registry = ProjectionRegistry(root / "projections.json")
    session_dir = root / "sess"
    AOLedger(session_dir, "sess")
    facade = _facade(root, store, registry, session_dir, "sess")
    result = facade.audit_legacy(
        {}, _PRINCIPAL, RetrievalProfile(fork_branch=4, projection=0, segment_atom=0)
    )
    assert [(card.kind, card.channel, card.status_label) for card in result.cards] == [
        ("fork", "verified", "validated"),
        ("branch", "verified", "validated"),
        ("fork", "provisional", "ForkCandidate"),
        ("branch", "provisional", "ForkCandidate"),
    ]


def test_channel_backfill_keeps_in_channel_rank(tmp_path: Path) -> None:
    verified_root = tmp_path / "verified-backfill"
    verified_store = _store(verified_root)
    validated_ids = [
        _commit_fork(
            verified_store,
            ao_id=f"ao-v-{index}",
            workspace="team-a",
            status="validated",
        )
        for index in range(9)
    ]
    provisional_ids = [
        _commit_fork(
            verified_store,
            ao_id=f"ao-p-{index}",
            workspace="team-a",
            status="provisional",
        )
        for index in range(3)
    ]
    registry = ProjectionRegistry(verified_root / "projections.json")
    session_dir = verified_root / "sess"
    AOLedger(session_dir, "sess")
    facade = _facade(verified_root, verified_store, registry, session_dir, "sess")
    result = facade.audit_legacy(
        {},
        _PRINCIPAL,
        RetrievalProfile(fork_branch=12, projection=0, segment_atom=0),
    )
    by_fork = {card.source_pointer["fork_node_id"]: card for card in result.cards}
    assert by_fork[validated_ids[8]].channel == "verified"
    assert by_fork[validated_ids[8]].in_channel_rank == 9
    assert by_fork[validated_ids[8]].status_label == "validated"
    assert [by_fork[fork_id].in_channel_rank for fork_id in provisional_ids] == [1, 2, 3]
    assert {by_fork[fork_id].channel for fork_id in provisional_ids} == {"provisional"}

    short_root = tmp_path / "provisional-backfill"
    short_store = _store(short_root)
    for index in range(2):
        _commit_fork(short_store, ao_id=f"ao-sv-{index}", workspace="team-a", status="validated")
    provisional_ids = [
        _commit_fork(
            short_store,
            ao_id=f"ao-sp-{index}",
            workspace="team-a",
            status="provisional",
        )
        for index in range(6)
    ]
    short_registry = ProjectionRegistry(short_root / "projections.json")
    short_session = short_root / "sess"
    AOLedger(short_session, "sess")
    short_facade = _facade(short_root, short_store, short_registry, short_session, "sess")
    backfilled = short_facade.audit_legacy(
        {},
        _PRINCIPAL,
        RetrievalProfile(fork_branch=8, projection=0, segment_atom=0),
    )
    provisional_cards = [card for card in backfilled.cards if card.channel == "provisional"]
    assert sorted(card.in_channel_rank for card in provisional_cards) == [1, 2, 3, 4, 5, 6]
    kept = {card.source_pointer["fork_node_id"]: card for card in provisional_cards}
    assert kept[provisional_ids[4]].in_channel_rank == 5
    assert kept[provisional_ids[5]].in_channel_rank == 6
    assert kept[provisional_ids[5]].status_label == "ForkCandidate"
    assert len([card for card in backfilled.cards if card.channel == "verified"]) == 2


def test_acl_filters_non_intersecting_labels(tmp_path: Path) -> None:
    root = tmp_path / "acl"
    store = _store(root)
    visible_fork = _commit_fork(store, ao_id="ao-visible", workspace="team-a", status="validated")
    hidden_fork = _commit_fork(store, ao_id="ao-hidden", workspace="secret", status="validated")
    visible_rev = _freeze(
        [_block("b-vis", 1, session_id="sess-vis")],
        [],
        revision_id="rev-vis",
        session_id="sess-vis",
    )
    hidden_rev = _freeze(
        [_block("b-hid", 1, session_id="sess-hid")],
        [],
        revision_id="rev-hid",
        session_id="sess-hid",
    )
    registry = ProjectionRegistry(root / "projections.json")
    registry.register(visible_rev, acl_labels=("team-a",))
    registry.register(hidden_rev, acl_labels=("secret",))
    session_dir = root / "sess-acl"
    ledger = AOLedger(session_dir, "sess-acl")
    visible_ao = _append_ao(ledger, labels=["team-a"])
    hidden_ao = _append_ao(ledger, labels=["secret"])

    facade = _facade(
        root, store, registry, session_dir, "sess-acl", views=[visible_rev, hidden_rev]
    )
    result = facade.audit_legacy({}, _PRINCIPAL)
    pointers = [card.source_pointer for card in result.cards]
    assert any(item.get("fork_node_id") == visible_fork for item in pointers)
    assert all(item.get("fork_node_id") != hidden_fork for item in pointers)
    assert any(item.get("view_revision_id") == "rev-vis" for item in pointers)
    assert all(item.get("view_revision_id") != "rev-hid" for item in pointers)
    assert all(item.get("ao_id") not in {visible_ao, hidden_ao} for item in pointers)
    assert all(card.kind not in {"segment", "atom"} for card in result.cards)
    assert result.cards
    assert all(set(card.acl_labels) & set(_PRINCIPAL) for card in result.cards)

    resumed = _facade(
        root,
        store,
        registry,
        session_dir,
        "sess-acl",
        views=[visible_rev, hidden_rev],
        fact_lane_paused=False,
    )
    resumed_pointers = [card.source_pointer for card in resumed.audit_legacy({}, _PRINCIPAL).cards]
    assert any(item.get("ao_id") == visible_ao for item in resumed_pointers)
    assert all(item.get("ao_id") != hidden_ao for item in resumed_pointers)


def test_dependency_pattern_and_bounded_neighborhood(tmp_path: Path) -> None:
    root = tmp_path / "deps"
    store = _store(root)
    blocks = [_block(f"b{index}", index) for index in range(1, 4)]
    match = InfluenceClaim(
        claim_id="c-match",
        source_block_id="b1",
        target_block_id="b2",
        carried_artifact="patch",
        downstream_effect="tests passed",
        relation_type="influence",
    )
    other = InfluenceClaim(
        claim_id="c-other",
        source_block_id="b2",
        target_block_id="b3",
        carried_artifact="notes",
        downstream_effect="no change",
        relation_type="influence",
    )
    revision = _freeze(blocks, [match, other], revision_id="rev-dep")
    registry = ProjectionRegistry(root / "projections.json")
    registry.register(revision, acl_labels=_PRINCIPAL, validated_positions=("c-match",))
    registry.promote_to_verified(
        revision.revision_id,
        "c-match",
        {"fork_node_id": "fk-static-1", "fork_revision_id": "fr-1"},
    )
    session_dir = root / "sess"
    AOLedger(session_dir, "sess")
    facade = _facade(root, store, registry, session_dir, "sess", views=[revision])
    result = facade.audit_legacy(
        {"dependency_pattern": {"wanted_artifact": "patch", "wanted_effect": "tests passed"}},
        _PRINCIPAL,
        RetrievalProfile(fork_branch=0, projection=4, segment_atom=0),
    )
    claims = [card for card in result.cards if card.kind == "claim_projection"]
    assert len(claims) == 1
    assert claims[0].source_pointer["carried_artifact"] == "patch"
    assert claims[0].source_pointer["downstream_effect"] == "tests passed"
    assert claims[0].channel == "verified"
    assert claims[0].status_label == "verified"
    assert all(card.source_pointer.get("block_or_claim_id") != "c-other" for card in result.cards)

    chain_blocks = [_block(f"n{index}", index) for index in range(1, 15)]
    chain_claims = [
        InfluenceClaim(
            claim_id=f"edge-{index}",
            source_block_id=f"n{index}",
            target_block_id=f"n{index + 1}",
            carried_artifact=f"art-{index}",
            downstream_effect=f"eff-{index}",
            relation_type="influence",
        )
        for index in range(1, 14)
    ]
    chain = _freeze(chain_blocks, chain_claims, revision_id="rev-chain")
    cycle_blocks = [_block(f"c{index}", index, session_id="sess-cycle") for index in range(1, 4)]
    cycle_claims = [
        InfluenceClaim(
            claim_id="cy-1",
            source_block_id="c1",
            target_block_id="c2",
            carried_artifact="loop",
            downstream_effect="step",
            relation_type="influence",
        ),
        InfluenceClaim(
            claim_id="cy-2",
            source_block_id="c2",
            target_block_id="c3",
            carried_artifact="loop",
            downstream_effect="step",
            relation_type="influence",
        ),
        InfluenceClaim(
            claim_id="cy-3",
            source_block_id="c3",
            target_block_id="c1",
            carried_artifact="loop",
            downstream_effect="step",
            relation_type="influence",
        ),
    ]
    cycle = _freeze(
        cycle_blocks,
        cycle_claims,
        revision_id="rev-cycle",
        session_id="sess-cycle",
        enforce_checks=False,  # deliberately invalid view: read-side truncation is defense-in-depth
    )
    facade.projection.bind_view(chain)
    facade.projection.bind_view(cycle)
    registry_bytes = (root / "projections.json").read_bytes()
    causal_before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / "causal-experiences").rglob("*")
        if path.is_file()
    }

    neighborhood = facade.expand_claim_neighborhood("rev-chain", "edge-1", max_entities=12)
    assert len(neighborhood.entities) == PATH_MAX_ENTITIES == 12
    assert len(neighborhood.entities) == len(set(neighborhood.entities))
    assert neighborhood.truncated is True
    assert neighborhood.entities[0] == "n1"
    assert (
        check_path_budget(
            list(neighborhood.entities),
            [
                {
                    "kind": "evidence",
                    "bridge_id": claim_id,
                    "source_ref": {"type": "block", "id": source},
                    "target_ref": {"type": "block", "id": target},
                    "relation_type": "claim",
                }
                for claim_id, source, target in neighborhood.edges
            ],
            max_entities=12,
        )
        == []
    )

    cycled = facade.expand_claim_neighborhood("rev-cycle", "cy-1")
    assert cycled.truncated is True
    assert cycled.entities == ("c1", "c2", "c3")
    assert len(cycled.entities) == len(set(cycled.entities)) <= 12
    assert (root / "projections.json").read_bytes() == registry_bytes
    causal_after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / "causal-experiences").rglob("*")
        if path.is_file()
    }
    assert causal_after == causal_before


def test_fact_channel_does_not_consume_state_slots(tmp_path: Path) -> None:
    root = tmp_path / "facts"
    store = _store(root)
    validated_ids = [
        _commit_fork(store, ao_id=f"ao-fact-{index}", workspace="team-a", status="validated")
        for index in range(10)
    ]
    registry = ProjectionRegistry(root / "projections.json")
    session_dir = root / "sess-facts"
    ledger = AOLedger(session_dir, "sess-facts")
    segment_id = _append_ao(ledger, labels=["team-a"], record_kind="segment")
    atom_id = _append_ao(ledger, labels=["team-a"], record_kind="atom")
    facade = _facade(root, store, registry, session_dir, "sess-facts", fact_lane_paused=False)
    result = facade.audit_legacy(
        {},
        _PRINCIPAL,
        RetrievalProfile(fork_branch=10, projection=0, segment_atom=2),
    )
    verified = [card for card in result.cards if card.channel == "verified"]
    facts = [card for card in result.cards if card.kind in {"segment", "atom"}]
    assert [card.source_pointer["fork_node_id"] for card in verified] == validated_ids
    assert [card.in_channel_rank for card in verified] == list(range(1, 11))
    assert len(facts) == 2
    assert {card.channel for card in facts} == {"fact"}
    assert {card.status_label for card in facts} == {"fact"}
    assert {card.kind for card in facts} == {"segment", "atom"}
    assert {card.source_pointer["ao_id"] for card in facts} == {segment_id, atom_id}
    assert all(card.in_channel_rank >= 1 for card in facts)
    assert not any(card.channel in {"verified", "provisional"} for card in facts)


def test_acl_does_not_consume_quota(tmp_path: Path) -> None:
    root = tmp_path / "acl-quota"
    store = _store(root)
    for index in range(12):
        _commit_fork(store, ao_id=f"ao-hidden-{index}", workspace="secret", status="validated")
    visible = _commit_fork(store, ao_id="ao-visible-late", workspace="team-a", status="validated")
    registry = ProjectionRegistry(root / "projections.json")
    session_dir = root / "sess"
    AOLedger(session_dir, "sess")
    facade = _facade(root, store, registry, session_dir, "sess")
    result = facade.audit_legacy(
        {},
        _PRINCIPAL,
        RetrievalProfile(fork_branch=12, projection=0, segment_atom=0),
    )
    fork_ids = [card.source_pointer["fork_node_id"] for card in result.cards if card.kind == "fork"]
    assert visible in fork_ids
    assert all(
        card.source_pointer.get("fork_node_id") != visible or card.acl_labels == _PRINCIPAL
        for card in result.cards
    )
    assert all(set(card.acl_labels) & set(_PRINCIPAL) for card in result.cards)


class _OverflowRerank:
    def rerank(self, cards, context):
        del context
        seed = cards[0]
        return [seed for _ in range(13)]


def test_rerank_clamp_restores_cap(tmp_path: Path) -> None:
    root = tmp_path / "rerank-cap"
    store = _store(root)
    _commit_fork(store, ao_id="ao-one", workspace="team-a", status="validated")
    registry = ProjectionRegistry(root / "projections.json")
    session_dir = root / "sess"
    AOLedger(session_dir, "sess")
    projection = InfluenceProjectionAdapter(registry)
    facade = HybridRetrievalFacade(
        ForkBranchAdapter(root, store),
        projection,
        SegmentAtomAdapter(session_dir, "sess"),
        rerank=_OverflowRerank(),
    )
    result = facade.audit_legacy(
        {},
        _PRINCIPAL,
        RetrievalProfile(fork_branch=6, projection=0, segment_atom=0),
    )
    assert len(result.cards) <= 12
    assert len(result.cards) == 12
    assert result.truncated_by_rerank_clamp is True
    assert {card.in_channel_rank for card in result.cards} == {1}


def test_branch_channel_follows_own_evidence(tmp_path: Path) -> None:
    root = tmp_path / "branch-channel"
    store = _store(root)
    _commit_fork(
        store,
        ao_id="ao-mixed",
        workspace="team-a",
        status="validated",
        branches=[
            {
                "branch_id": "br-imag",
                "evidence_status": "imagined_unverified",
                "ao_sequence": ["ao-imag"],
                "guidance": None,
            },
            {
                "branch_id": "br-real",
                "evidence_status": "real",
                "ao_sequence": ["ao-real"],
                "guidance": None,
            },
        ],
    )
    registry = ProjectionRegistry(root / "projections.json")
    session_dir = root / "sess"
    AOLedger(session_dir, "sess")
    facade = _facade(root, store, registry, session_dir, "sess")
    result = facade.audit_legacy(
        {},
        _PRINCIPAL,
        RetrievalProfile(fork_branch=6, projection=0, segment_atom=0),
    )
    branches = {
        card.source_pointer["branch_id"]: card for card in result.cards if card.kind == "branch"
    }
    assert branches["br-imag"].channel == "provisional"
    assert branches["br-real"].channel == "verified"


def test_projection_adapter_hides_invisible_cards(tmp_path: Path) -> None:
    root = tmp_path / "proj-acl"
    visible_rev = _freeze(
        [_block("b-vis", 1, session_id="sess-vis")],
        [],
        revision_id="rev-vis",
        session_id="sess-vis",
    )
    hidden_rev = _freeze(
        [_block("b-hid", 1, session_id="sess-hid")],
        [],
        revision_id="rev-hid",
        session_id="sess-hid",
    )
    registry = ProjectionRegistry(root / "projections.json")
    registry.register(visible_rev, acl_labels=("team-a",))
    registry.register(hidden_rev, acl_labels=("secret",))
    adapter = InfluenceProjectionAdapter(registry, views=[visible_rev, hidden_rev])
    cards = adapter.search({}, 10, _PRINCIPAL)
    registry_ids = {card.card_id for card in registry.search(principal_labels=_PRINCIPAL)}
    assert {card.payload_ref for card in cards} == registry_ids
    assert registry_ids
    assert all(card.source_pointer["view_revision_id"] != "rev-hid" for card in cards)
    dumped = {entry["card"]["card_id"] for entry in registry.to_dict()["entries"]}
    assert len(dumped) > len(registry_ids)


def test_provisional_quota_is_three_exploitation_plus_one_exploration(tmp_path: Path) -> None:
    root, store, session_dir, validated_ids, provisional_ids, _hidden = _fork_lane(
        tmp_path,
        "three-plus-one",
        validated=8,
        provisional=4,
    )
    facade = _open_lane(root, store, session_dir, seed="slot-seed")
    profile = RetrievalProfile(fork_branch=12, projection=0, segment_atom=0)
    result = facade.audit_legacy({}, _PRINCIPAL, profile)
    again = facade.audit_legacy({}, _PRINCIPAL, profile)

    assert len(result.cards) == 12
    verified = [card for card in result.cards if card.channel == "verified"]
    provisional = [card for card in result.cards if card.channel == "provisional"]
    assert [card.in_channel_rank for card in verified] == list(range(1, 9))
    assert [_fork_id(card) for card in verified] == validated_ids
    assert all(card.selection_mode == "exploitation" for card in verified)
    assert [card.selection_mode for card in provisional] == [
        "exploitation",
        "exploitation",
        "exploitation",
        "exploration",
    ]
    assert [card.in_channel_rank for card in provisional] == [1, 2, 3, 4]
    assert [_fork_id(card) for card in provisional[:3]] == provisional_ids[:3]
    assert _fork_id(provisional[3]) == provisional_ids[3]
    assert _fork_id(provisional[3]) not in set(provisional_ids[:3])

    audit = result.selection_audit
    assert audit is not None
    assert audit.backfill is False
    assert audit.backfill_reason is None
    assert audit.seed == "slot-seed"
    assert audit.selection_policy_version == SELECTION_POLICY_VERSION
    assert audit.eligible_set == (provisional[3].payload_ref,)
    assert audit.selection_propensity == 1.0
    assert provisional[3].payload_ref in audit.eligible_set
    assert RetrievalCard.from_dict(provisional[3].to_dict()) == provisional[3]
    assert provisional[3].to_dict()["selection_mode"] == "exploration"
    assert result.to_dict()["selection_audit"]["eligible_set"] == [provisional[3].payload_ref]
    assert [(card.payload_ref, card.selection_mode) for card in result.cards] == [
        (card.payload_ref, card.selection_mode) for card in again.cards
    ]
    assert again.selection_audit == audit


def test_exploration_selection_is_seeded_and_audited(tmp_path: Path) -> None:
    root, store, session_dir, _validated, provisional_ids, _hidden = _fork_lane(
        tmp_path,
        "seeded",
        provisional=8,
    )
    profile = RetrievalProfile(fork_branch=8, projection=0, segment_atom=0)

    def explore(seed: str) -> RetrievalCard:
        result = _open_lane(root, store, session_dir, seed=seed).audit_legacy(
            {}, _PRINCIPAL, profile
        )
        assert len(result.cards) <= 12
        explored = [card for card in result.cards if card.selection_mode == "exploration"]
        assert len(explored) == 1
        audit = result.selection_audit
        assert audit is not None
        assert audit.backfill is False
        assert audit.seed == seed
        assert audit.selection_policy_version == SELECTION_POLICY_VERSION
        assert len(audit.eligible_set) == 5
        assert audit.selection_propensity == 1.0 / len(audit.eligible_set)
        assert explored[0].payload_ref in audit.eligible_set
        by_ref = {card.payload_ref: card for card in result.cards}
        eligible_ids = {_fork_id(by_ref[ref]) for ref in audit.eligible_set}
        assert eligible_ids == set(provisional_ids[3:])
        assert _fork_id(explored[0]) in set(provisional_ids[3:])
        prefix = [
            card
            for card in result.cards
            if card.channel == "provisional" and card.in_channel_rank <= 3
        ]
        assert {_fork_id(card) for card in prefix} == set(provisional_ids[:3])
        assert {card.selection_mode for card in prefix} == {"exploitation"}
        return explored[0]

    first = explore("seed-a")
    assert explore("seed-a").payload_ref == first.payload_ref
    assert any(explore(f"seed-{index}").payload_ref != first.payload_ref for index in range(24))


def test_exploration_backfill_when_no_eligible_candidate(tmp_path: Path) -> None:
    root, store, session_dir, validated_ids, provisional_ids, _hidden = _fork_lane(
        tmp_path,
        "backfill",
        validated=9,
        provisional=3,
    )
    result = _open_lane(root, store, session_dir, seed="backfill-seed").audit_legacy(
        {},
        _PRINCIPAL,
        RetrievalProfile(fork_branch=12, projection=0, segment_atom=0),
    )
    assert len(result.cards) <= 12
    provisional = [card for card in result.cards if card.channel == "provisional"]
    assert [_fork_id(card) for card in provisional] == provisional_ids
    assert [card.in_channel_rank for card in provisional] == [1, 2, 3]
    assert {card.selection_mode for card in result.cards} == {"exploitation"}
    ninth = next(card for card in result.cards if card.in_channel_rank == 9)
    assert ninth.channel == "verified"
    assert ninth.selection_mode == "exploitation"
    assert _fork_id(ninth) == validated_ids[8]
    audit = result.selection_audit
    assert audit is not None
    assert audit.backfill is True
    assert audit.backfill_reason == NO_ELIGIBLE_EXPLORATION_REASON
    assert audit.eligible_set == ()
    assert audit.selection_propensity is None
    assert audit.seed == "backfill-seed"
    assert result.to_dict()["selection_audit"]["backfill"] is True
    assert result.to_dict()["selection_audit"]["backfill_reason"] == NO_ELIGIBLE_EXPLORATION_REASON


def test_raised_provisional_quota_keeps_single_exploration_seat(tmp_path: Path) -> None:
    root, store, session_dir, _validated, provisional_ids, _hidden = _fork_lane(
        tmp_path,
        "raised-quota",
        provisional=8,
    )
    result = _open_lane(root, store, session_dir, seed="raised").audit_legacy(
        {},
        _PRINCIPAL,
        {
            "fork_branch": 8,
            "projection": 0,
            "segment_atom": 0,
            "provisional_slots": 6,
        },
    )
    assert len(result.cards) <= 12
    explored = [card for card in result.cards if card.selection_mode == "exploration"]
    assert len(explored) == 1
    assert explored[0].in_channel_rank >= 6
    assert _fork_id(explored[0]) in set(provisional_ids[5:])
    prefix = [
        card for card in result.cards if card.channel == "provisional" and card.in_channel_rank <= 5
    ]
    assert {_fork_id(card) for card in prefix} == set(provisional_ids[:5])
    assert {card.selection_mode for card in prefix} == {"exploitation"}
    audit = result.selection_audit
    assert audit is not None
    assert audit.backfill is False
    assert len(audit.eligible_set) == 3
    assert audit.selection_propensity == 1.0 / 3
    assert explored[0].payload_ref in audit.eligible_set
    by_ref = {card.payload_ref: card for card in result.cards}
    assert {_fork_id(by_ref[ref]) for ref in audit.eligible_set} == set(provisional_ids[5:])


def test_query_derived_seed_and_default_selection_mode(tmp_path: Path) -> None:
    root = tmp_path / "derived-seed"
    store = _store(root)
    registry = ProjectionRegistry(root / "projections.json")
    session_dir = root / "sess-facts"
    ledger = AOLedger(session_dir, "sess-facts")
    _append_ao(ledger, labels=["team-a"])
    facade = _facade(root, store, registry, session_dir, "sess-facts", fact_lane_paused=False)
    profile = RetrievalProfile(fork_branch=0, projection=0, segment_atom=1)
    empty = facade.audit_legacy({}, _PRINCIPAL, profile)
    alpha = facade.audit_legacy({"text": "alpha"}, _PRINCIPAL, profile)
    beta = facade.audit_legacy({"text": "beta"}, _PRINCIPAL, profile)
    patterned = facade.audit_legacy(
        {"text": "alpha", "dependency_pattern": {"wanted_artifact": "patch"}},
        _PRINCIPAL,
        profile,
    )
    assert empty.selection_audit is not None
    assert empty.selection_audit.seed == _expected_query_seed("")
    assert len(empty.selection_audit.seed) == 16
    assert alpha.selection_audit is not None
    assert beta.selection_audit is not None
    assert patterned.selection_audit is not None
    assert alpha.selection_audit.seed == _expected_query_seed("alpha")
    assert beta.selection_audit.seed == _expected_query_seed("beta")
    assert alpha.selection_audit.seed != beta.selection_audit.seed
    assert patterned.selection_audit.seed == _expected_query_seed(
        "alpha", {"wanted_artifact": "patch"}
    )
    assert patterned.selection_audit.seed != alpha.selection_audit.seed
    repeated = facade.audit_legacy({"text": "alpha"}, _PRINCIPAL, profile)
    assert repeated.selection_audit == alpha.selection_audit
    facts = [card for card in empty.cards if card.channel == "fact"]
    assert len(facts) == 1
    assert facts[0].selection_mode == "exploitation"
    seeded = _facade(
        root,
        store,
        registry,
        session_dir,
        "sess-facts",
        seed="explicit-seed",
        fact_lane_paused=False,
    )
    seeded_alpha = seeded.audit_legacy({"text": "alpha"}, _PRINCIPAL, profile)
    seeded_beta = seeded.audit_legacy({"text": "beta"}, _PRINCIPAL, profile)
    assert seeded_alpha.selection_audit is not None
    assert seeded_beta.selection_audit is not None
    assert seeded_alpha.selection_audit.seed == "explicit-seed"
    assert seeded_beta.selection_audit.seed == "explicit-seed"


def test_exploration_eligible_set_excludes_acl_hidden(tmp_path: Path) -> None:
    root, store, session_dir, _validated, visible_ids, hidden_ids = _fork_lane(
        tmp_path,
        "acl-explore",
        provisional=6,
        hidden_provisional=4,
    )
    result = _open_lane(root, store, session_dir, seed="acl-seed").audit_legacy(
        {},
        _PRINCIPAL,
        RetrievalProfile(fork_branch=6, projection=0, segment_atom=0),
    )
    assert len(result.cards) <= 12
    assert hidden_ids
    assert all(_fork_id(card) not in set(hidden_ids) for card in result.cards)
    assert {_fork_id(card) for card in result.cards} == set(visible_ids)
    explored = [card for card in result.cards if card.selection_mode == "exploration"]
    assert len(explored) == 1
    assert _fork_id(explored[0]) in set(visible_ids[3:])
    audit = result.selection_audit
    assert audit is not None
    assert audit.backfill is False
    by_ref = {card.payload_ref: card for card in result.cards}
    eligible_ids = {_fork_id(by_ref[ref]) for ref in audit.eligible_set}
    assert eligible_ids == set(visible_ids[3:])
    assert eligible_ids.isdisjoint(hidden_ids)
    assert all(set(card.acl_labels) & set(_PRINCIPAL) for card in result.cards)


def test_text_match_is_case_insensitive(tmp_path: Path) -> None:
    root = tmp_path / "casefold"
    store = _store(root)
    registry = ProjectionRegistry(root / "projections.json")
    session_dir = root / "sess-case"
    ledger = AOLedger(session_dir, "sess-case")
    upper = ledger.append(
        {"tool": "read_file", "record_kind": "atom", "acl_labels": ["team-a"]},
        {"kind": "text", "summary": "CAROLINE met yesterday"},
    )
    lower = ledger.append(
        {"tool": "read_file", "record_kind": "atom", "acl_labels": ["team-a"]},
        {"kind": "text", "summary": "caroline wrote the note"},
    )
    facade = _facade(root, store, registry, session_dir, "sess-case", fact_lane_paused=False)
    profile = RetrievalProfile(fork_branch=0, projection=0, segment_atom=2)
    expected = {upper.ao_id, lower.ao_id}

    lower_query = facade.audit_legacy("caroline", _PRINCIPAL, profile)
    assert {card.source_pointer["ao_id"] for card in lower_query.cards} == expected
    upper_query = facade.audit_legacy("CAROLINE", _PRINCIPAL, profile)
    assert {card.source_pointer["ao_id"] for card in upper_query.cards} == expected
    empty = facade.audit_legacy("", _PRINCIPAL, profile)
    assert {card.source_pointer["ao_id"] for card in empty.cards} == expected
    assert facade.audit_legacy("melanie", _PRINCIPAL, profile).cards == ()


def test_content_resolver_matches_block_body_without_changing_facts(tmp_path: Path) -> None:
    root = tmp_path / "resolver"
    store = _store(root)
    block = _block("b-body", 1)
    revision = _freeze([block], [], revision_id="rev-body")
    registry = ProjectionRegistry(root / "projections.json")
    registry.register(revision, acl_labels=_PRINCIPAL)
    session_dir = root / "sess-res"
    ledger = AOLedger(session_dir, "sess-res")
    fact = ledger.append(
        {"tool": "read_file", "record_kind": "atom", "acl_labels": ["team-a"]},
        {"kind": "text", "summary": "notebook stays on the fact channel"},
    )
    body = "Caroline met Melanie at the cafe"
    ao_id = block.source_refs[0].ao_id

    def resolve(key: str) -> str:
        if key == ao_id:
            return body
        return ""

    profile = RetrievalProfile(fork_branch=0, projection=4, segment_atom=2)
    plain = _facade(
        root,
        store,
        registry,
        session_dir,
        "sess-res",
        views=[revision],
        fact_lane_paused=False,
    )
    assert plain.audit_legacy("caroline", _PRINCIPAL, profile).cards == ()
    plain_facts = plain.audit_legacy("notebook", _PRINCIPAL, profile)
    assert [card.kind for card in plain_facts.cards] == ["atom"]
    assert plain_facts.cards[0].source_pointer["ao_id"] == fact.ao_id

    projection = InfluenceProjectionAdapter(registry, views=[revision], content_resolver=resolve)
    by_body = projection.search("caroline", 4, _PRINCIPAL)
    by_summary = projection.search("summary-b-body", 4, _PRINCIPAL)
    assert len(by_body) == 1
    assert by_body[0].kind == "block_projection"
    body_payload = by_body[0].to_dict()
    summary_payload = by_summary[0].to_dict()
    assert body_payload.pop("score_breakdown")["bm25"] > 0
    assert summary_payload.pop("score_breakdown")["bm25"] > 0
    assert body_payload == summary_payload
    assert "cafe" not in json.dumps(by_body[0].to_dict(), ensure_ascii=False)

    facade = HybridRetrievalFacade(
        ForkBranchAdapter(root, store),
        projection,
        SegmentAtomAdapter(session_dir, "sess-res"),
        fact_lane_paused=False,
    )
    found = facade.audit_legacy("caroline", _PRINCIPAL, profile)
    assert [card.kind for card in found.cards] == ["block_projection"]
    assert all(card.channel != "fact" for card in found.cards)
    resolved_facts = facade.audit_legacy("notebook", _PRINCIPAL, profile)
    assert [card.to_dict() for card in resolved_facts.cards] == [
        card.to_dict() for card in plain_facts.cards
    ]

    empty_resolver = InfluenceProjectionAdapter(
        registry, views=[revision], content_resolver=lambda _ao_id: ""
    )
    assert empty_resolver.search("caroline", 4, _PRINCIPAL) == []
    unbound = InfluenceProjectionAdapter(registry, content_resolver=resolve)
    assert unbound.search("caroline", 4, _PRINCIPAL) == []
    assert len(unbound.search("summary-b-body", 4, _PRINCIPAL)) == 1


def test_content_resolver_matches_claim_via_linked_block_refs(tmp_path: Path) -> None:
    root = tmp_path / "resolver-claim"
    source = _block("b-src", 1)
    target = _block("b-dst", 2)
    claim = InfluenceClaim(
        claim_id="c-link",
        source_block_id="b-src",
        target_block_id="b-dst",
        carried_artifact="fallback-span",
        downstream_effect="observed outcome",
        relation_type="influence",
    )
    revision = _freeze([source, target], [claim], revision_id="rev-claim-body")
    registry = ProjectionRegistry(root / "projections.json")
    registry.register(revision, acl_labels=_PRINCIPAL)
    body = "Caroline sent the patch"
    source_ao = source.source_refs[0].ao_id

    def resolve(key: str) -> str:
        if key == source_ao:
            return body
        return ""

    adapter = InfluenceProjectionAdapter(registry, views=[revision], content_resolver=resolve)
    claims = [
        card
        for card in adapter.search("caroline", 10, _PRINCIPAL)
        if card.kind == "claim_projection"
    ]
    assert len(claims) == 1
    assert claims[0].source_pointer["block_or_claim_id"] == "c-link"
    assert "patch" not in json.dumps(claims[0].to_dict(), ensure_ascii=False)
    bare = InfluenceProjectionAdapter(registry, views=[revision])
    assert bare.search("caroline", 10, _PRINCIPAL) == []


class _ExplodingFacts:
    name = "segment_atom"

    def search(self, *args, **kwargs):
        raise AssertionError("segment_atom.search must not run while the fact lane is paused")


def test_fact_lane_pause_skips_search_and_does_not_backfill(tmp_path: Path) -> None:
    root = tmp_path / "pause"
    store = _store(root)
    matching = [
        _commit_fork(
            store,
            ao_id=f"ao-z-{index}",
            workspace="team-a",
            status="validated",
            sections=_text_sections("zephyr"),
        )
        for index in range(8)
    ]
    registry = ProjectionRegistry(root / "projections.json")
    session_dir = root / "sess"
    AOLedger(session_dir, "sess")
    facade = HybridRetrievalFacade(
        ForkBranchAdapter(root, store),
        InfluenceProjectionAdapter(registry),
        _ExplodingFacts(),  # type: ignore[arg-type]
    )
    assert facade.fact_lane_paused is True
    result = facade.audit_legacy(
        {},
        _PRINCIPAL,
        RetrievalProfile(fork_branch=6, projection=4, segment_atom=2),
    )
    forks = [card for card in result.cards if card.kind == "fork"]
    assert [card.source_pointer["fork_node_id"] for card in forks] == matching[:6]
    assert len(forks) == 6
    assert len(matching) == 8
    assert all(card.kind not in {"segment", "atom"} for card in result.cards)
    assert all(card.channel != "fact" for card in result.cards)
    assert len(result.cards) <= 10
    assert result.feature_version == FEATURE_SPEC_VERSION
    assert CAUSAL_MEMORY_CAP == 12


def test_bm25_term_and_document_frequency_change_rank() -> None:
    assert DEFAULT_K1 == 1.2
    assert DEFAULT_B == 0.75
    frequency = BM25Index(["apple apple apple", "apple"], k1=1.2, b=0.75)
    scores = frequency.scores("apple")
    heavy = (
        math.log(1 + (2 - 2 + 0.5) / (2 + 0.5))
        * (3 * 2.2)
        / (3 + 1.2 * (1 - 0.75 + 0.75 * (3 / 2)))
    )
    light = (
        math.log(1 + (2 - 2 + 0.5) / (2 + 0.5))
        * (1 * 2.2)
        / (1 + 1.2 * (1 - 0.75 + 0.75 * (1 / 2)))
    )
    assert scores[0] == pytest.approx(heavy)
    assert scores[1] == pytest.approx(light)
    assert scores[0] > scores[1]

    rarity = BM25Index(["rare", "common", "common", "common"])
    mixed = rarity.scores("rare common")
    assert mixed[0] > mixed[1]
    assert mixed[1] == pytest.approx(mixed[2]) == pytest.approx(mixed[3])
    assert rarity.scores("absent") == [0.0, 0.0, 0.0, 0.0]
    assert rarity.scores("") == [0.0, 0.0, 0.0, 0.0]


def test_bm25_cjk_unigrams_rank_q127_release_example(tmp_path: Path) -> None:
    """Q127 A: a shared CJK character overlaps; an English doc with no shared term stays out."""
    query = "这次发布必须蓝绿"
    exact = "这次发布必须蓝绿"
    english = "green environment cutover"
    green_once = "文档里只出现绿灯"
    fork_text = "蓝绿发布"

    assert tokenize(query) == ["这", "次", "发", "布", "必", "须", "蓝", "绿"]
    assert tokenize(fork_text) == ["蓝", "绿", "发", "布"]
    assert tokenize(green_once).count("绿") == 1
    assert tokenize("blue-green 蓝绿 v2") == ["blue", "green", "蓝", "绿", "v2"]
    assert tokenize("\u3400\u4dbf\u4e00\u9fff") == ["\u3400", "\u4dbf", "\u4e00", "\u9fff"]

    scores = BM25Index([exact, english, green_once, fork_text]).scores(query)
    assert scores[1] == 0.0
    assert scores[0] > scores[3] > scores[2] > 0.0

    root = tmp_path / "q127"
    store = _store(root)
    registry = ProjectionRegistry(root / "projections.json")
    session_dir = root / "sess-q127"
    ledger = AOLedger(session_dir, "sess-q127")
    exact_id = _append_text(ledger, exact)
    english_id = _append_text(ledger, english)
    green_id = _append_text(ledger, green_once)
    fork_id = _append_text(ledger, fork_text)
    result = _facade(
        root,
        store,
        registry,
        session_dir,
        "sess-q127",
        fact_lane_paused=False,
    ).audit_legacy(
        query,
        _PRINCIPAL,
        RetrievalProfile(fork_branch=0, projection=0, segment_atom=4),
    )
    ids = [card.source_pointer["ao_id"] for card in result.cards]
    assert english_id not in ids
    assert ids[0] == exact_id
    assert ids.index(fork_id) < ids.index(green_id)
    assert result.cards[ids.index(fork_id)].score_breakdown["bm25"] > 0
    assert result.rank_tier == "bm25"


def test_bm25_english_words_stay_whole_tokens() -> None:
    assert tokenize("APPLE apple_2") == ["apple", "apple_2"]
    assert tokenize("Don't stop.") == ["don", "t", "stop"]
    corpus = ["apple apple apple", "pineapple sauce", "apple"]
    scores = BM25Index(corpus, k1=DEFAULT_K1, b=DEFAULT_B).scores("apple")
    assert scores[1] == 0.0
    assert scores[0] > scores[2] > 0.0
    assert BM25Index(corpus).scores("") == [0.0, 0.0, 0.0]
    assert BM25Index(corpus).scores("zephyr") == [0.0, 0.0, 0.0]


def test_bm25_retrieval_orders_hits_and_keeps_per_corpus_quotas(tmp_path: Path) -> None:
    root = tmp_path / "bm25"
    store = _store(root)
    quiet = _commit_fork(
        store,
        ao_id="ao-quiet",
        workspace="team-a",
        status="validated",
        sections=_text_sections("unrelated ledger note"),
    )
    weak_fork = _commit_fork(
        store,
        ao_id="ao-weak-fork",
        workspace="team-a",
        status="validated",
        sections=_text_sections("zephyr"),
    )
    strong_fork = _commit_fork(
        store,
        ao_id="ao-strong-fork",
        workspace="team-a",
        status="validated",
        sections=_text_sections("zephyr " * 40),
    )
    weak_block = _block("b-weak", 1, summary="zephyr")
    strong_block = _block("b-strong", 2, summary="zephyr " * 40)
    revision = _freeze([weak_block, strong_block], [], revision_id="rev-bm25")
    registry = ProjectionRegistry(root / "projections.json")
    registry.register(revision, acl_labels=_PRINCIPAL)
    session_dir = root / "sess-bm25"
    ledger = AOLedger(session_dir, "sess-bm25")
    first = _append_text(ledger, "alpha note")
    second = _append_text(ledger, "beta note")
    third = _append_text(ledger, "gamma note")
    weak_fact = _append_text(ledger, "zephyr")
    strong_fact = _append_text(ledger, "zephyr " * 40)
    for _ in range(3):
        _append_text(ledger, "commonword")
    rare_fact = _append_text(ledger, "rareword commonword")
    facade = _facade(
        root,
        store,
        registry,
        session_dir,
        "sess-bm25",
        views=[revision],
        fact_lane_paused=False,
    )

    empty = facade.audit_legacy(
        "",
        _PRINCIPAL,
        RetrievalProfile(fork_branch=0, projection=0, segment_atom=3),
    )
    assert [card.source_pointer["ao_id"] for card in empty.cards] == [first, second, third]
    assert {card.score_breakdown["bm25"] for card in empty.cards} == {0.0}
    assert empty.rank_tier == "bm25"

    shared = facade.audit_legacy(
        "alpha",
        _PRINCIPAL,
        RetrievalProfile(fork_branch=0, projection=0, segment_atom=3),
    )
    assert [card.source_pointer["ao_id"] for card in shared.cards] == [first]
    assert shared.cards[0].score_breakdown["bm25"] > 0
    assert shared.cards[0].score_breakdown["lexical"] == shared.cards[0].score_breakdown["bm25"]
    assert shared.cards[0].score_breakdown["rank_tier"] == "bm25"

    assert (
        facade.audit_legacy(
            "carol",
            _PRINCIPAL,
            RetrievalProfile(fork_branch=0, projection=0, segment_atom=4),
        ).cards
        == ()
    )
    caroline = _append_text(ledger, "caroline met melanie")
    folded = facade.audit_legacy(
        "CAROLINE",
        _PRINCIPAL,
        RetrievalProfile(fork_branch=0, projection=0, segment_atom=4),
    )
    assert [card.source_pointer["ao_id"] for card in folded.cards] == [caroline]
    assert (
        facade.audit_legacy(
            "quinquereme",
            _PRINCIPAL,
            RetrievalProfile(fork_branch=0, projection=0, segment_atom=4),
        ).cards
        == ()
    )

    by_frequency = facade.audit_legacy(
        "zephyr",
        _PRINCIPAL,
        RetrievalProfile(fork_branch=1, projection=1, segment_atom=1),
    )
    assert [card.kind for card in by_frequency.cards] == ["fork", "block_projection", "atom"]
    assert by_frequency.cards[0].source_pointer["fork_node_id"] == strong_fork
    assert by_frequency.cards[0].source_pointer["fork_node_id"] != weak_fork
    assert quiet not in {card.source_pointer.get("fork_node_id") for card in by_frequency.cards}
    assert by_frequency.cards[1].source_pointer["block_or_claim_id"] == "b-strong"
    assert by_frequency.cards[2].source_pointer["ao_id"] == strong_fact
    assert by_frequency.cards[2].source_pointer["ao_id"] != weak_fact

    rare_first = facade.audit_legacy(
        "rareword commonword",
        _PRINCIPAL,
        RetrievalProfile(fork_branch=0, projection=0, segment_atom=4),
    )
    rare_ids = [card.source_pointer["ao_id"] for card in rare_first.cards]
    assert rare_ids[0] == rare_fact
    assert len(rare_ids) == 4

    forks_only = facade.audit_legacy(
        "zephyr",
        _PRINCIPAL,
        RetrievalProfile(fork_branch=1, projection=0, segment_atom=0),
    )
    assert [card.kind for card in forks_only.cards] == ["fork"]
    assert forks_only.cards[0].source_pointer["fork_node_id"] == strong_fork


def test_hybrid_tier_mixes_real_vectors_or_stays_on_bm25(tmp_path: Path) -> None:
    root = tmp_path / "hybrid"
    store = _store(root)
    registry = ProjectionRegistry(root / "projections.json")
    session_dir = root / "sess-hybrid"
    ledger = AOLedger(session_dir, "sess-hybrid")
    heavy = _append_text(ledger, "bm25-heavy " + "zephyr " * 30)
    aligned = _append_text(ledger, "vector-hit zephyr")
    bait = _append_text(ledger, "perfect vector bait")
    profile = {
        "fork_branch": 0,
        "projection": 0,
        "segment_atom": 3,
        "rank_mode": "hybrid",
    }

    def embed(text: str) -> list[float]:
        folded = text.casefold()
        if "bm25-heavy" in folded:
            return [0.0, 1.0]
        if folded.strip() == "zephyr":
            return [1.0, 0.0]
        return [1.0, 0.0]

    fused = _facade(
        root,
        store,
        registry,
        session_dir,
        "sess-hybrid",
        fact_lane_paused=False,
        embedder=embed,
    ).audit_legacy("zephyr", _PRINCIPAL, profile)
    fused_ids = [card.source_pointer["ao_id"] for card in fused.cards]
    assert fused_ids[0] == aligned
    assert heavy in fused_ids
    assert bait not in fused_ids
    assert fused.rank_tier == "hybrid"
    assert fused.cards[0].score_breakdown["rank_tier"] == "hybrid"
    assert fused.cards[0].score_breakdown["bm25"] > 0
    assert "lexical" in fused.cards[0].score_breakdown
    assert fused.cards[0].score_breakdown["vector"] > fused.cards[1].score_breakdown["vector"]
    assert "combined" in fused.cards[0].score_breakdown

    dim0 = _facade(
        root,
        store,
        registry,
        session_dir,
        "sess-hybrid",
        fact_lane_paused=False,
        embedder=lambda _text: [],
    ).audit_legacy("zephyr", _PRINCIPAL, profile)
    dim0_ids = [card.source_pointer["ao_id"] for card in dim0.cards]
    assert dim0_ids
    assert dim0_ids[0] == heavy
    assert bait not in dim0_ids
    assert dim0.rank_tier == "bm25"
    assert dim0.cards[0].score_breakdown["rank_tier"] == "bm25"
    assert dim0.cards[0].score_breakdown["requested_rank_tier"] == "hybrid"
    assert "vector" not in dim0.cards[0].score_breakdown

    unavailable = _facade(
        root,
        store,
        registry,
        session_dir,
        "sess-hybrid",
        fact_lane_paused=False,
    ).audit_legacy("zephyr", _PRINCIPAL, profile)
    assert unavailable.rank_tier == "bm25"
    assert [card.source_pointer["ao_id"] for card in unavailable.cards][0] == heavy
    assert unavailable.cards

    block = _block("b-stub", 1, summary="zephyr projection")
    stub_revision = _freeze([block], [], revision_id="rev-stub")
    stub_registry = ProjectionRegistry(root / "stub.json")
    stub_registry.register(stub_revision, acl_labels=_PRINCIPAL)
    stubbed = HybridRetrievalFacade(
        ForkBranchAdapter(root, store),
        InfluenceProjectionAdapter(stub_registry, views=[stub_revision]),
        SegmentAtomAdapter(session_dir, "sess-hybrid"),
    ).audit_legacy(
        "zephyr",
        _PRINCIPAL,
        {"fork_branch": 0, "projection": 2, "segment_atom": 2, "rank_mode": "hybrid"},
    )
    assert stubbed.cards
    assert stubbed.rank_tier == "bm25"
    assert [card.kind for card in stubbed.cards] == ["block_projection"]
    assert stubbed.cards[0].score_breakdown["rank_tier"] == "bm25"
    assert "vector" not in stubbed.cards[0].score_breakdown


def test_rank_mode_has_no_vector_only_tier(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rank_mode"):
        RetrievalProfile(rank_mode="vector")  # type: ignore[arg-type]
    root = tmp_path / "mode"
    store = _store(root)
    session_dir = root / "sess"
    AOLedger(session_dir, "sess")
    facade = _facade(
        root, store, ProjectionRegistry(root / "projections.json"), session_dir, "sess"
    )
    with pytest.raises(ValueError, match="rank_mode"):
        facade.audit_legacy({}, _PRINCIPAL, {"rank_mode": "vector"})


def test_neighborhood_walks_influence_edges_only(tmp_path: Path) -> None:
    blocks = [_block(f"n{index}", index) for index in range(1, 6)]
    blocks.extend(_block(f"p{index}", 5 + index) for index in range(1, 4))
    claims = [
        InfluenceClaim(
            claim_id="e-12",
            source_block_id="n1",
            target_block_id="n2",
            carried_artifact="patch",
            downstream_effect="applied",
            relation_type=RELATION_INFLUENCE,
        ),
        InfluenceClaim(
            claim_id="e-skip",
            source_block_id="n2",
            target_block_id="n3",
            carried_artifact="gap",
            downstream_effect="omitted",
            relation_type=RELATION_SKIP,
        ),
        InfluenceClaim(
            claim_id="e-34",
            source_block_id="n3",
            target_block_id="n4",
            carried_artifact="rate",
            downstream_effect="rose",
            relation_type=RELATION_INFLUENCE,
        ),
        InfluenceClaim(
            claim_id="e-skip2",
            source_block_id="n4",
            target_block_id="n5",
            carried_artifact="tail",
            downstream_effect="unread",
            relation_type=RELATION_SKIP,
        ),
        InfluenceClaim(
            claim_id="e-p12",
            source_block_id="p1",
            target_block_id="p2",
            carried_artifact="upstream",
            downstream_effect="kept",
            relation_type=RELATION_INFLUENCE,
        ),
        InfluenceClaim(
            claim_id="e-p23",
            source_block_id="p2",
            target_block_id="p3",
            carried_artifact="downstream",
            downstream_effect="kept",
            relation_type=RELATION_INFLUENCE,
        ),
    ]
    revision = _freeze(blocks, claims, revision_id="rev-skip", enforce_checks=False)
    root = tmp_path / "walk"
    store = _store(root)
    session_dir = root / "sess"
    AOLedger(session_dir, "sess")
    adapter = InfluenceProjectionAdapter(
        ProjectionRegistry(root / "projections.json"), views=[revision]
    )
    facade = HybridRetrievalFacade(
        ForkBranchAdapter(root, store),
        adapter,
        SegmentAtomAdapter(session_dir, "sess"),
    )

    stopped = facade.expand_claim_neighborhood("rev-skip", "e-12")
    assert stopped.entities[:2] == ("n1", "n2")
    assert "n3" not in stopped.entities
    assert "n5" not in stopped.entities
    assert {edge[0] for edge in stopped.edges} == {"e-12"}

    seeded = adapter.expand_claim_neighborhood("rev-skip", "e-skip")
    assert seeded.entities[:2] == ("n2", "n3")
    assert "n1" in seeded.entities
    assert "n4" in seeded.entities
    assert "n5" not in seeded.entities
    edge_ids = {edge[0] for edge in seeded.edges}
    assert edge_ids == {"e-12", "e-34"}
    assert "e-skip" not in edge_ids
    assert "e-skip2" not in edge_ids

    upstream = facade.expand_claim_neighborhood("rev-skip", "e-p23")
    assert "p1" in upstream.entities
    assert "p2" in upstream.entities
    assert "p3" in upstream.entities
    assert {edge[0] for edge in upstream.edges} == {"e-p12", "e-p23"}
    assert len(upstream.entities) <= PATH_MAX_ENTITIES


def test_fork_and_branch_cards_carry_text(tmp_path: Path) -> None:
    root = tmp_path / "card-text"
    store = _store(root)
    session_dir = root / "sess-text"
    ledger = AOLedger(session_dir, "sess-text")
    raw_body = "RAW-AO-BODY-7c21"
    guidance_text = "GUIDANCE-FULL-TEXT-9f3a"
    first = ledger.append(
        {"tool": "edit", "arguments": {"body": raw_body}},
        {
            "kind": "text",
            "title": "edit",
            "summary": ["set timeout to 30"],
            "sample": raw_body,
        },
    )
    second = ledger.append(
        {"tool": "test", "arguments": {"body": raw_body}},
        {"kind": "text", "summary": "rerun the suite", "sample": raw_body},
    )
    counted = [
        ledger.append(
            {"tool": "step", "arguments": {"body": raw_body}},
            {"kind": "text", "summary": f"step-{index:02d}", "sample": raw_body},
        )
        for index in range(BRANCH_SYNOPSIS_LIMIT + 1)
    ]
    long = ledger.append(
        {"tool": "step", "arguments": {"body": raw_body}},
        {"kind": "text", "summary": "LONGMARK-" + ("y" * 400), "sample": raw_body},
    )
    sections = _sections()
    sections["open_issues"] = "fork-seven-open-issue-marker"
    _commit_fork(
        store,
        ao_id="ao-anchor",
        workspace="team-a",
        status="validated",
        sections=sections,
        branches=[
            {
                "branch_id": "br-real",
                "evidence_status": "real",
                "ao_sequence": [first.ao_id, "ao-not-in-ledger", second.ao_id],
                "guidance": {"instruction": guidance_text},
            },
            {
                "branch_id": "br-missing",
                "evidence_status": "imagined_unverified",
                "ao_sequence": ["ao-not-in-ledger"],
                "guidance": {"instruction": guidance_text},
            },
            {
                "branch_id": "br-count",
                "evidence_status": "real",
                "ao_sequence": [record.ao_id for record in counted],
                "guidance": {"instruction": guidance_text},
            },
            {
                "branch_id": "br-long",
                "evidence_status": "real",
                "ao_sequence": [long.ao_id],
                "guidance": {"instruction": guidance_text},
            },
        ],
    )
    registry = ProjectionRegistry(root / "projections.json")
    facade = _facade(root, store, registry, session_dir, "sess-text")
    profile = RetrievalProfile(fork_branch=6, projection=0, segment_atom=0)
    result = facade.audit_legacy({}, _PRINCIPAL, profile)
    forks = [card for card in result.cards if card.kind == "fork"]
    assert len(forks) == 1
    fork = forks[0]
    assert fork.text == "\n".join(f"{key}: {sections[key]}" for key in SEVEN_SECTION_KEYS)
    assert "fork-seven-open-issue-marker" in fork.text
    fork_dump = json.dumps(fork.to_dict(), ensure_ascii=False)
    assert guidance_text not in fork_dump
    assert raw_body not in fork_dump
    assert RetrievalCard.from_dict(fork.to_dict()) == fork
    changed = fork.to_dict()
    changed["text"] = fork.text + "\nextra"
    assert RetrievalCard.from_dict(changed) != fork
    content = fork.to_dict()
    content.pop("score_breakdown")
    replayed = RetrievalCard.from_dict(fork.to_dict()).to_dict()
    replayed.pop("score_breakdown")
    assert content == replayed
    assert "text" in content

    branches = {
        card.source_pointer["branch_id"]: card for card in result.cards if card.kind == "branch"
    }
    assert branches["br-real"].text == "edit: set timeout to 30 | rerun the suite"
    assert branches["br-missing"].text == "br-missing imagined_unverified"
    assert branches["br-count"].text == " | ".join(
        f"step-{index:02d}" for index in range(BRANCH_SYNOPSIS_LIMIT)
    )
    assert f"step-{BRANCH_SYNOPSIS_LIMIT:02d}" not in branches["br-count"].text
    assert len(branches["br-long"].text) <= BRANCH_SUMMARY_CHARS
    assert branches["br-long"].text.endswith("...")
    assert branches["br-long"].text.startswith("LONGMARK-")
    for card in branches.values():
        dumped = json.dumps(card.to_dict(), ensure_ascii=False)
        assert guidance_text not in dumped
        assert raw_body not in dumped
        assert "fork-seven-open-issue-marker" not in card.text
        assert set(card.score_breakdown) == {"bm25", "lexical", "rank_tier"}
    assert set(fork.score_breakdown) == {"bm25", "lexical", "rank_tier"}

    bare = ForkBranchAdapter(root, store)
    bare_branches = {
        card.source_pointer["branch_id"]: card
        for card in bare.search({}, 6, _PRINCIPAL)
        if card.kind == "branch"
    }
    assert bare_branches["br-real"].text == "br-real real"


def test_expand_claim_neighborhood_by_card_id(tmp_path: Path) -> None:
    blocks = [_block(f"n{index}", index) for index in range(1, 6)]
    blocks.extend(_block(f"p{index}", 5 + index) for index in range(1, 4))
    claims = [
        InfluenceClaim(
            claim_id="e-12",
            source_block_id="n1",
            target_block_id="n2",
            carried_artifact="patch",
            downstream_effect="applied",
            relation_type=RELATION_INFLUENCE,
        ),
        InfluenceClaim(
            claim_id="e-skip",
            source_block_id="n2",
            target_block_id="n3",
            carried_artifact="gap",
            downstream_effect="omitted",
            relation_type=RELATION_SKIP,
        ),
        InfluenceClaim(
            claim_id="e-34",
            source_block_id="n3",
            target_block_id="n4",
            carried_artifact="rate",
            downstream_effect="rose",
            relation_type=RELATION_INFLUENCE,
        ),
        InfluenceClaim(
            claim_id="e-skip2",
            source_block_id="n4",
            target_block_id="n5",
            carried_artifact="tail",
            downstream_effect="unread",
            relation_type=RELATION_SKIP,
        ),
    ]
    revision = _freeze(blocks, claims, revision_id="rev-skip", enforce_checks=False)
    root = tmp_path / "card-expand"
    store = _store(root)
    registry = ProjectionRegistry(root / "projections.json")
    registry.register(revision, acl_labels=_PRINCIPAL)
    session_dir = root / "sess"
    AOLedger(session_dir, "sess")
    facade = _facade(root, store, registry, session_dir, "sess", views=[revision])
    profile = RetrievalProfile(fork_branch=0, projection=4, segment_atom=0)

    recalled = facade.audit_legacy("omitted", _PRINCIPAL, profile)
    claim_cards = [card for card in recalled.cards if card.kind == "claim_projection"]
    assert len(claim_cards) == 1
    claim_card = claim_cards[0]
    assert claim_card.text == ""
    assert claim_card.payload_ref == claim_card.source_pointer["card_id"]
    assert set(claim_card.source_pointer) == {
        "corpus",
        "view_revision_id",
        "block_or_claim_id",
        "card_id",
        "session_id",
        "carried_artifact",
        "downstream_effect",
    }
    assert set(claim_card.to_dict()) == {
        "kind",
        "channel",
        "in_channel_rank",
        "status_label",
        "source_pointer",
        "payload_ref",
        "score_breakdown",
        "acl_labels",
        "selection_mode",
        "text",
    }

    expanded = facade.expand_claim_neighborhood(
        card_id=claim_card.payload_ref, principal_labels=_PRINCIPAL
    )
    direct = facade.expand_claim_neighborhood("rev-skip", "e-skip")
    assert expanded == direct
    owned = facade.projection.expand_claim_neighborhood(
        card_id=claim_card.payload_ref, principal_labels=_PRINCIPAL
    )
    assert owned == expanded
    assert expanded.entities[:2] == ("n2", "n3")
    assert "n1" in expanded.entities
    assert "n4" in expanded.entities
    assert "n5" not in expanded.entities
    assert {edge[0] for edge in expanded.edges} == {"e-12", "e-34"}
    assert "e-skip" not in {edge[0] for edge in expanded.edges}
    assert "e-skip2" not in {edge[0] for edge in expanded.edges}

    block_hits = facade.audit_legacy("n1", _PRINCIPAL, profile)
    block_cards = [card for card in block_hits.cards if card.kind == "block_projection"]
    assert len(block_cards) == 1
    assert block_cards[0].text == ""
    with pytest.raises(ValueError, match="not a claim projection"):
        facade.expand_claim_neighborhood(
            card_id=block_cards[0].payload_ref, principal_labels=_PRINCIPAL
        )
    missing_id = "claim-card-00000000"
    with pytest.raises(ValueError) as missing:
        facade.expand_claim_neighborhood(card_id=missing_id, principal_labels=_PRINCIPAL)
    with pytest.raises(ValueError) as hidden:
        facade.expand_claim_neighborhood(
            card_id=claim_card.payload_ref, principal_labels=("outsider",)
        )
    with pytest.raises(ValueError) as hidden_block:
        facade.expand_claim_neighborhood(
            card_id=block_cards[0].payload_ref, principal_labels=("outsider",)
        )
    with pytest.raises(ValueError) as default_denied:
        facade.expand_claim_neighborhood(card_id=claim_card.payload_ref)
    assert str(missing.value) == f"unknown card id {missing_id!r}"
    assert str(hidden.value) == f"unknown card id {claim_card.payload_ref!r}"
    assert str(hidden_block.value) == f"unknown card id {block_cards[0].payload_ref!r}"
    assert str(default_denied.value) == f"unknown card id {claim_card.payload_ref!r}"
    assert str(hidden.value).replace(claim_card.payload_ref, "ID") == str(missing.value).replace(
        missing_id, "ID"
    )
    with pytest.raises(ValueError, match="unknown card id"):
        facade.expand_claim_neighborhood(card_id="fork:fk-1@rev-1")
    with pytest.raises(ValueError, match="card_id"):
        facade.expand_claim_neighborhood("rev-skip", "e-skip", card_id=claim_card.payload_ref)
