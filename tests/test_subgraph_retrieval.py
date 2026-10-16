# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Diagnostic read of historical retrieval subgraphs (R4a)."""

from __future__ import annotations

from pathlib import Path

import pytest

from openviking.session.ao_ledger import AOLedger
from openviking.session.causal_experiences import (
    INDEX_FILENAME,
    INTERVENTION_INDEX_FILENAME,
    NAMESPACE_DIRNAME,
    SEVEN_SECTION_KEYS,
    CausalExperiencesStore,
    InfluenceGrounding,
    ServerChecks,
)
from openviking.session.influence_projection import ProjectionRegistry
from openviking.session.influence_view import (
    AORange,
    CoverageManifest,
    InfluenceBlock,
    InfluenceClaim,
    InfluenceEvidenceRef,
    InfluenceViewDraft,
    RetrievalSubgraph,
    SegmentRef,
    SourceRef,
    ViewSupersedeIndex,
)
from openviking.session.retrieval_facade import (
    DIAGNOSTIC_SUBGRAPH_HIT_CAP,
    RANK_MODE_BM25,
    SUBGRAPH_TEXT_POLICY_VERSION,
    ForkBranchAdapter,
    HybridRetrievalFacade,
    InfluenceProjectionAdapter,
    RetrievalSubgraphAdapter,
    SegmentAtomAdapter,
    _rank_document_indexes,
    _subgraph_retrieval_text,
)

_CAUSAL = {"skill_trajectory_mode": "causal"}
_CHECKS = ServerChecks(anchor_exists=True, refs_exist=True, provenance_valid=True)
_PRINCIPAL = ("team-a",)
_SIMILAR = "cutover error rate elevated"
_NOISE = "notify mailbox digest"
_FROZEN = "2026-01-02T00:00:00.000Z"
_GUIDANCE = "GUIDANCE-FULL-TEXT-DO-NOT-LEAK"
_RAW_AO = "RAW-AO-BODY-DO-NOT-LEAK"
_SEVEN = "错误率升高就切回蓝"
_SYNOPSIS = "切回蓝环境"


def _allow(_session_id: str) -> tuple[str, ...]:
    return _PRINCIPAL


def _ref(session_id: str, sequence: int) -> SourceRef:
    return SourceRef(
        ao_id=f"ao-{session_id}-{sequence}",
        sequence=sequence,
        participant_id="agent-a",
        segment_id="seg-a",
        session_id=session_id,
        content_hash=f"hash-{session_id}-{sequence}",
    )


def _block(session_id: str, block_id: str, sequence: int, summary: str) -> InfluenceBlock:
    ref = _ref(session_id, sequence)
    return InfluenceBlock(
        block_id=block_id,
        role="agent_action",
        participant_id="agent-a",
        segment_ref=SegmentRef(segment_id="seg-a", session_id=session_id),
        ao_start_seq=sequence,
        ao_end_seq=sequence,
        summary=summary,
        input="",
        output="",
        authorship="agent-a",
        granularity_reason="one tool call",
        source_content_hashes=(ref.content_hash,),
        source_refs=(ref,),
    )


def _coverage(blocks: list[InfluenceBlock]) -> CoverageManifest:
    refs = tuple(ref for block in blocks for ref in block.source_refs)
    return CoverageManifest(
        total_ao_range=AORange(
            start_seq=min(ref.sequence for ref in refs), end_seq=max(ref.sequence for ref in refs)
        ),
        inspected_refs=refs,
        included_ranges=tuple(
            AORange(
                start_seq=block.ao_start_seq,
                end_seq=block.ao_end_seq,
                segment_id=block.segment_ref.segment_id,
            )
            for block in blocks
        ),
        omitted_ranges=(),
        snapshot_watermark="wm-1",
    )


def _similar_blocks(session_id: str) -> list[InfluenceBlock]:
    return [
        _block(session_id, "b-cutover", 1, _SIMILAR),
        _block(session_id, "b-rate", 2, _SIMILAR),
    ]


def _cutover(failure_marked: bool = True) -> RetrievalSubgraph:
    return RetrievalSubgraph(
        subgraph_id="sg-cutover",
        block_ids=("b-cutover", "b-rate"),
        claim_ids=(),
        failure_marked=failure_marked,
    )


def _freeze(
    session_id: str,
    revision_id: str,
    blocks: list[InfluenceBlock],
    subgraphs: list[RetrievalSubgraph],
    *,
    purpose: str = "post_run_index",
    claims: list[InfluenceClaim] | None = None,
) -> InfluenceViewDraft:
    draft = InfluenceViewDraft(
        view_id=f"view-{session_id}",
        purpose=purpose,  # type: ignore[arg-type]
        session_id=session_id,
        task_run_id=f"task-{session_id}",
        coverage=_coverage(blocks),
        blocks=blocks,
        claims=claims or [],
        retrieval_subgraphs=subgraphs,
    )
    return draft.freeze(revision_id=revision_id, enforce_checks=False, frozen_at=_FROZEN)


def _index(tmp_path: Path) -> ViewSupersedeIndex:
    return ViewSupersedeIndex(tmp_path / "views.json")


def _facade(
    tmp_path: Path,
    index: ViewSupersedeIndex,
    *,
    session_acl=_allow,
    session_dirs: dict[str, Path] | None = None,
    store: CausalExperiencesStore | None = None,
) -> HybridRetrievalFacade:
    root = tmp_path / "store"
    experiences = store if store is not None else CausalExperiencesStore(root, config=_CAUSAL)
    forks = ForkBranchAdapter(root, experiences)
    adapter = RetrievalSubgraphAdapter(
        index,
        forks,
        session_acl=session_acl,
        session_dirs=session_dirs,
    )
    return HybridRetrievalFacade(
        forks,
        InfluenceProjectionAdapter(ProjectionRegistry(tmp_path / "projections.json")),
        SegmentAtomAdapter(tmp_path / "unused-sess", "unused"),
        retrieval_subgraph=adapter,
    )


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


def _sections(**overrides: str) -> dict[str, str]:
    sections = {key: f"{key} body" for key in SEVEN_SECTION_KEYS}
    sections.update(overrides)
    return sections


def _grounding(**overrides: str) -> InfluenceGrounding:
    fields = {
        "view_revision_id": "rev-hist",
        "position_kind": "block",
        "position_ref": "block:b-cutover",
        "semantic_summary_snapshot": "frozen summary",
        "content_hash": "sha256:grounding",
        "grounded_at": _FROZEN,
    }
    fields.update(overrides)
    return InfluenceGrounding(**fields)  # type: ignore[arg-type]


def test_revisions_for_purpose_lists_every_session_in_order(tmp_path: Path) -> None:
    index = _index(tmp_path)
    index.register(_freeze("sess-b", "rev-b1", _similar_blocks("sess-b"), [_cutover()]))
    index.register(_freeze("sess-a", "rev-a", _similar_blocks("sess-a"), [_cutover()]))
    index.register(_freeze("sess-b", "rev-b2", _similar_blocks("sess-b"), [_cutover()]))
    index.register(
        _freeze(
            "sess-a",
            "rev-diag",
            _similar_blocks("sess-a"),
            [_cutover()],
            purpose="failure_diagnosis",
        )
    )

    reloaded = ViewSupersedeIndex(index.path)
    listed = reloaded.revisions_for_purpose("post_run_index")
    assert [item.revision_id for item in listed] == ["rev-a", "rev-b1", "rev-b2"]
    assert [item.revision_id for item in reloaded.revisions_for_purpose("failure_diagnosis")] == [
        "rev-diag"
    ]


def test_returns_at_most_three_and_skips_self_side_and_other_purpose(tmp_path: Path) -> None:
    index = _index(tmp_path)
    index.register(
        _freeze("aaa-current", "rev-current", _similar_blocks("aaa-current"), [_cutover()])
    )
    index.register(
        _freeze(
            "aaa-fail-div",
            "rev-diag",
            _similar_blocks("aaa-fail-div"),
            [_cutover()],
            purpose="failure_diagnosis",
        )
    )
    side_blocks = _similar_blocks("hist-a") + [_block("hist-a", "b-notify", 3, _NOISE)]
    index.register(
        _freeze(
            "hist-a",
            "rev-a",
            side_blocks,
            [
                _cutover(),
                RetrievalSubgraph(
                    subgraph_id="sg-notify",
                    block_ids=("b-notify",),
                    claim_ids=(),
                    failure_marked=False,
                ),
            ],
        )
    )
    for session_id, revision_id in (
        ("hist-b", "rev-b"),
        ("hist-c", "rev-c"),
        ("hist-d", "rev-d"),
    ):
        index.register(_freeze(session_id, revision_id, _similar_blocks(session_id), [_cutover()]))

    facade = _facade(tmp_path, index)
    hits = facade.retrieve_historical_subgraphs("rev-current", "sg-cutover", _PRINCIPAL)
    identity = [(hit.view_revision_id, hit.subgraph.subgraph_id) for hit in hits]
    assert identity == [("rev-a", "sg-cutover"), ("rev-b", "sg-cutover"), ("rev-c", "sg-cutover")]
    assert [block.block_id for block in hits[0].blocks] == ["b-cutover", "b-rate"]
    assert all(_NOISE not in block.summary for hit in hits for block in hit.blocks)
    assert all(hit.session_id != "aaa-current" for hit in hits)
    assert all(hit.view_revision_id != "rev-diag" for hit in hits)
    for hit in hits:
        assert hit.score_breakdown["rank_tier"] == RANK_MODE_BM25
        assert hit.score_breakdown["lexical"] == hit.score_breakdown["bm25"]
        assert hit.score_breakdown["bm25"] > 0


def test_fewer_than_three_does_not_backfill_zero_score(tmp_path: Path) -> None:
    index = _index(tmp_path)
    index.register(
        _freeze("zzz-current", "rev-current", _similar_blocks("zzz-current"), [_cutover()])
    )
    index.register(_freeze("hist-a", "rev-a", _similar_blocks("hist-a"), [_cutover()]))
    index.register(_freeze("hist-b", "rev-b", _similar_blocks("hist-b"), [_cutover()]))
    index.register(
        _freeze(
            "hist-noise",
            "rev-noise",
            [_block("hist-noise", "b-notify", 1, _NOISE)],
            [
                RetrievalSubgraph(
                    subgraph_id="sg-notify",
                    block_ids=("b-notify",),
                    claim_ids=(),
                    failure_marked=True,
                )
            ],
        )
    )
    hits = _facade(tmp_path, index).retrieve_historical_subgraphs(
        "rev-current", "sg-cutover", _PRINCIPAL
    )
    # Typed field names overlap, so the prose-disjoint subgraph is a positive
    # hit. It does not outrank the prose matches, and a true zero is still dropped.
    assert [hit.view_revision_id for hit in hits] == ["rev-a", "rev-b", "rev-noise"]
    assert hits[0].score_breakdown["bm25"] == hits[1].score_breakdown["bm25"]
    assert hits[1].score_breakdown["bm25"] > hits[2].score_breakdown["bm25"] > 0
    ranked = _rank_document_indexes("alpha beta", ["alpha beta", "zzzz unrelated"])
    assert [index for index, _breakdown in ranked] == [0]
    assert len(ranked[:DIAGNOSTIC_SUBGRAPH_HIT_CAP]) == 1


def test_unmarked_subgraph_raises(tmp_path: Path) -> None:
    index = _index(tmp_path)
    index.register(
        _freeze(
            "aaa-current",
            "rev-current",
            _similar_blocks("aaa-current"),
            [_cutover(failure_marked=False)],
        )
    )
    facade = _facade(tmp_path, index)
    with pytest.raises(ValueError, match="not failure-marked"):
        facade.retrieve_historical_subgraphs("rev-current", "sg-cutover", _PRINCIPAL)


def test_missing_subgraph_and_foreign_id_raise(tmp_path: Path) -> None:
    index = _index(tmp_path)
    index.register(
        _freeze("aaa-current", "rev-current", _similar_blocks("aaa-current"), [_cutover()])
    )
    index.register(_freeze("hist-a", "rev-a", _similar_blocks("hist-a"), [_cutover()]))
    facade = _facade(tmp_path, index)
    with pytest.raises(ValueError, match="not in the supersede index"):
        facade.retrieve_historical_subgraphs("rev-missing", "sg-cutover", _PRINCIPAL)
    with pytest.raises(ValueError, match="not on view revision"):
        facade.retrieve_historical_subgraphs("rev-current", "sg-other", _PRINCIPAL)
    with pytest.raises(ValueError, match="not on view revision"):
        facade.retrieve_historical_subgraphs("rev-current", "sg-notify", _PRINCIPAL)


def test_repeated_calls_keep_hit_order(tmp_path: Path) -> None:
    index = _index(tmp_path)
    index.register(
        _freeze("zzz-current", "rev-current", _similar_blocks("zzz-current"), [_cutover()])
    )
    index.register(_freeze("hist-a", "rev-a", _similar_blocks("hist-a"), [_cutover()]))
    index.register(_freeze("hist-b", "rev-b", _similar_blocks("hist-b"), [_cutover()]))
    facade = _facade(tmp_path, index)
    first = facade.retrieve_historical_subgraphs("rev-current", "sg-cutover", _PRINCIPAL)
    second = facade.retrieve_historical_subgraphs("rev-current", "sg-cutover", _PRINCIPAL)
    assert [hit.view_revision_id for hit in first] == [hit.view_revision_id for hit in second]
    assert [hit.score_breakdown for hit in first] == [hit.score_breakdown for hit in second]
    assert [hit.view_revision_id for hit in first] == ["rev-a", "rev-b"]


def test_superseded_revision_stays_in_the_corpus(tmp_path: Path) -> None:
    index = _index(tmp_path)
    index.register(
        _freeze("zzz-current", "rev-current", _similar_blocks("zzz-current"), [_cutover()])
    )
    index.register(_freeze("hist-old", "rev-old", _similar_blocks("hist-old"), [_cutover()]))
    index.register(
        _freeze(
            "hist-old",
            "rev-new",
            [_block("hist-old", "b-notify", 1, _NOISE)],
            [
                RetrievalSubgraph(
                    subgraph_id="sg-notify",
                    block_ids=("b-notify",),
                    claim_ids=(),
                    failure_marked=False,
                )
            ],
        )
    )
    latest = index.latest("hist-old", "post_run_index")
    assert latest is not None
    assert latest.revision_id == "rev-new"
    hits = _facade(tmp_path, index).retrieve_historical_subgraphs(
        "rev-current", "sg-cutover", _PRINCIPAL
    )
    assert [hit.view_revision_id for hit in hits] == ["rev-old", "rev-new"]
    assert hits[0].score_breakdown["bm25"] > hits[1].score_breakdown["bm25"] > 0


def test_acl_drops_unlabeled_and_mismatched_before_quota(tmp_path: Path) -> None:
    index = _index(tmp_path)
    index.register(
        _freeze("zzz-current", "rev-current", _similar_blocks("zzz-current"), [_cutover()])
    )
    index.register(_freeze("aaa-hidden", "rev-hidden", _similar_blocks("aaa-hidden"), [_cutover()]))
    index.register(
        _freeze("aaa-foreign", "rev-foreign", _similar_blocks("aaa-foreign"), [_cutover()])
    )
    for session_id, revision_id in (
        ("hist-a", "rev-a"),
        ("hist-b", "rev-b"),
        ("hist-c", "rev-c"),
    ):
        index.register(_freeze(session_id, revision_id, _similar_blocks(session_id), [_cutover()]))

    def session_acl(session_id: str) -> tuple[str, ...]:
        if session_id == "aaa-hidden":
            return ()
        if session_id == "aaa-foreign":
            return ("other-team",)
        return ("team-a",)

    hits = _facade(tmp_path, index, session_acl=session_acl).retrieve_historical_subgraphs(
        "rev-current",
        "sg-cutover",
        ("team-a", "aaa-hidden"),
    )
    assert [hit.view_revision_id for hit in hits] == ["rev-a", "rev-b", "rev-c"]


def test_grounded_forks_follow_position_and_omit_guidance_and_raw_ao(tmp_path: Path) -> None:
    session_id = "hist-ground"
    blocks = _similar_blocks(session_id) + [_block(session_id, "b-notify", 3, _NOISE)]
    claim = InfluenceClaim(
        claim_id="c-rate",
        source_block_id="b-cutover",
        target_block_id="b-rate",
        carried_artifact="green traffic",
        downstream_effect="five xx elevated",
        relation_type="influence",
    )
    query_blocks = _similar_blocks("zzz-current")
    query_claim = InfluenceClaim(
        claim_id="c-rate",
        source_block_id="b-cutover",
        target_block_id="b-rate",
        carried_artifact="green traffic",
        downstream_effect="five xx elevated",
        relation_type="influence",
    )
    index = _index(tmp_path)
    index.register(
        _freeze(
            "zzz-current",
            "rev-current",
            query_blocks,
            [
                RetrievalSubgraph(
                    subgraph_id="sg-cutover",
                    block_ids=("b-cutover", "b-rate"),
                    claim_ids=("c-rate",),
                    failure_marked=True,
                )
            ],
            claims=[query_claim],
        )
    )
    index.register(
        _freeze(
            session_id,
            "rev-hist",
            blocks,
            [
                RetrievalSubgraph(
                    subgraph_id="sg-cutover",
                    block_ids=("b-cutover", "b-rate"),
                    claim_ids=("c-rate",),
                    failure_marked=True,
                ),
                RetrievalSubgraph(
                    subgraph_id="sg-notify",
                    block_ids=("b-notify",),
                    claim_ids=(),
                    failure_marked=False,
                ),
            ],
            claims=[claim],
        )
    )
    root = tmp_path / "store"
    store = CausalExperiencesStore(root, config=_CAUSAL)
    session_dir = tmp_path / "ledgers" / session_id
    ledger = AOLedger(session_dir, session_id)
    record = ledger.append(
        {"tool": "edit", "arguments": {"body": _RAW_AO}},
        {
            "kind": "text",
            "title": "rollback",
            "summary": _SYNOPSIS,
            "sample": _RAW_AO,
        },
    )
    sections = _sections(key_facts_and_decisions=_SEVEN)
    branches = [
        {
            "branch_id": "br-observed",
            "evidence_status": "real",
            "ao_sequence": [record.ao_id],
            "guidance": {"instruction": _GUIDANCE},
        }
    ]

    def commit(ao_id: str, workspace: str, **grounding_overrides: str):
        draft = store.submit_fork_draft(
            {
                "workspace_id": workspace,
                "anchor": {
                    "anchor_kind": "ao",
                    "ao_id": ao_id,
                    "source_session_id": session_id,
                    "anchor_sequence": 10,
                },
                "seven_sections": sections,
                "contemporaneous_basis": [_evidence_ref()],
                "hindsight_attribution": [],
                "used_skills": [],
                "branches": branches,
                "diagnosis_run_id": f"diag-{ao_id}",
                "model_version": "diag-model-v1",
            },
            influence_grounding=_grounding(**grounding_overrides),
        )
        return store.commit_provisional(draft, _CHECKS)

    inside_block = commit("ao-block", "team-a")
    inside_edge = commit("ao-edge", "team-a", position_kind="edge", position_ref="edge:c-rate")
    commit("ao-outside-block", "team-a", position_ref="block:b-notify")
    commit("ao-outside-edge", "team-a", position_kind="edge", position_ref="edge:c-other")
    commit("ao-cut", "team-a", position_kind="cut", position_ref="block:b-cutover")
    commit("ao-bare", "team-a", position_ref="b-cutover")
    commit("ao-garbage", "team-a", position_ref="???")
    commit("ao-prefix", "team-a", position_ref="edge:c-rate")
    commit("ao-empty-id", "team-a", position_ref="block:")
    commit("ao-other-view", "team-a", view_revision_id="rev-current")
    commit("ao-other-workspace", "other-team")

    facade = _facade(
        tmp_path,
        index,
        session_dirs={session_id: session_dir},
        store=store,
    )
    fork_index = root / NAMESPACE_DIRNAME / INDEX_FILENAME
    intervention = root / NAMESPACE_DIRNAME / INTERVENTION_INDEX_FILENAME
    before_forks = fork_index.read_text(encoding="utf-8")
    before_intervention = intervention.read_text(encoding="utf-8")
    hits = facade.retrieve_historical_subgraphs("rev-current", "sg-cutover", _PRINCIPAL)
    assert fork_index.read_text(encoding="utf-8") == before_forks
    assert intervention.read_text(encoding="utf-8") == before_intervention
    assert [hit.subgraph.subgraph_id for hit in hits] == ["sg-cutover", "sg-notify"]
    hit = hits[0]
    assert hit.view_revision_id == "rev-hist"
    assert [block.block_id for block in hit.blocks] == ["b-cutover", "b-rate"]
    assert [item.claim_id for item in hit.claims] == ["c-rate"]
    assert [item.fork_node_id for item in hit.forks] == [
        inside_block.fork_node_id,
        inside_edge.fork_node_id,
    ]
    text = hit.forks[0].text
    assert f"key_facts_and_decisions: {_SEVEN}" in text
    assert "session_title:" in text
    assert f"rollback: {_SYNOPSIS}" in text
    assert _GUIDANCE not in text
    assert _RAW_AO not in text
    for fork in hit.forks:
        assert _GUIDANCE not in fork.text
        assert _RAW_AO not in fork.text


def test_online_read_does_not_call_supersede_to_dict(tmp_path: Path) -> None:
    index = _index(tmp_path)
    index.register(
        _freeze("zzz-current", "rev-current", _similar_blocks("zzz-current"), [_cutover()])
    )
    index.register(_freeze("hist-a", "rev-a", _similar_blocks("hist-a"), [_cutover()]))

    def _boom(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("ViewSupersedeIndex.to_dict is not an online read")

    index.to_dict = _boom  # type: ignore[method-assign]
    hits = _facade(tmp_path, index).retrieve_historical_subgraphs(
        "rev-current", "sg-cutover", _PRINCIPAL
    )
    assert [hit.view_revision_id for hit in hits] == ["rev-a"]


_SENTINEL = "opqzxq9unique"
_RANGE_START = 814270
_RANGE_END = 814271


def _typed_block(
    session_id: str,
    block_id: str,
    summary: str,
    *,
    role: str,
    block_kind: str,
    sequence: int,
    input_text: str = "",
    output_text: str = "",
    participant_id: str = "agent-a",
    ao_id: str | None = None,
) -> InfluenceBlock:
    ref = SourceRef(
        ao_id=ao_id or f"ao-{session_id}-{sequence}",
        sequence=sequence,
        participant_id=participant_id,
        segment_id="seg-a",
        session_id=session_id,
        content_hash=f"hash-{session_id}-{sequence}",
    )
    return InfluenceBlock(
        block_id=block_id,
        role=role,  # type: ignore[arg-type]
        participant_id=participant_id,
        segment_ref=SegmentRef(segment_id="seg-a", session_id=session_id),
        ao_start_seq=sequence,
        ao_end_seq=sequence,
        summary=summary,
        input=input_text,
        output=output_text,
        authorship="agent-a",
        granularity_reason="one tool call",
        source_content_hashes=(ref.content_hash,),
        source_refs=(ref,),
        block_kind=block_kind,
    )


def _evidence(ao_id: str, sequence: int) -> InfluenceEvidenceRef:
    return InfluenceEvidenceRef(
        ao_id=ao_id,
        source_session_id="sess-evidence",
        source_archive_id="arch-evidence",
        archive_commit_watermark="commit-evidence",
        source_sequence=sequence,
        source_read_snapshot_watermark="read-wm-evidence",
        evidence_role="contemporaneous_basis",
        evidence_content_hash=f"hash-{ao_id}",
        captured_state="committed",
    )


def test_typed_tokens_follow_prose_and_keep_ids_out() -> None:
    assert SUBGRAPH_TEXT_POLICY_VERSION == "diagnostic-subgraph-text-v1"
    prose = (
        "deploy switch traffic",
        "prep notes",
        "traffic moved",
        "inspect status",
        "status climbed",
        "packet shift",
        "error climbed",
    )
    assert _SENTINEL not in "\n".join(prose)
    assert str(_RANGE_START) not in "\n".join(prose)
    task = _typed_block(
        "sess-typed",
        f"block-{_SENTINEL}",
        prose[0],
        role="task",
        block_kind="task",
        sequence=_RANGE_START,
        input_text=prose[1],
        output_text=prose[2],
        participant_id=f"participant-{_SENTINEL}",
        ao_id=f"ao-{_SENTINEL}",
    )
    action = _typed_block(
        "sess-typed",
        "b-inspect",
        prose[3],
        role="agent_action",
        block_kind="tool_call",
        sequence=_RANGE_END,
        output_text=prose[4],
        participant_id=f"participant-{_SENTINEL}-2",
        ao_id=f"ao-{_SENTINEL}-2",
    )
    # Revision stores the action block first. Member order follows claim/block ids.
    influence = InfluenceClaim(
        claim_id=f"claim-{_SENTINEL}",
        source_block_id=task.block_id,
        target_block_id=action.block_id,
        carried_artifact=prose[5],
        downstream_effect=prose[6],
        relation_type="Influence",
        source_evidence_refs=(_evidence(f"evidence-{_SENTINEL}", _RANGE_START),),
    )
    skip = InfluenceClaim(
        claim_id="c-skip",
        source_block_id=task.block_id,
        target_block_id=action.block_id,
        carried_artifact="announcement copy",
        downstream_effect="mailbox note",
        relation_type="skip-link",
    )
    subgraph = RetrievalSubgraph(
        subgraph_id=f"sg-{_SENTINEL}",
        block_ids=(task.block_id, action.block_id),
        claim_ids=(influence.claim_id, skip.claim_id),
        failure_marked=True,
        failure_evidence_refs=(_evidence(f"failure-{_SENTINEL}", _RANGE_END),),
    )
    revision = _freeze(
        "sess-typed",
        "rev-typed",
        [action, task],
        [subgraph],
        claims=[skip, influence],
    )
    text = _subgraph_retrieval_text(revision, subgraph)
    assert _SENTINEL not in text
    assert str(_RANGE_START) not in text
    assert str(_RANGE_END) not in text
    assert "agent_action" not in text.split("subgraph_failure:", 1)[0]
    assert text.count("agent_action") == text.count("block_role:agent_action")
    assert text == "\n".join(
        [
            *prose,
            "announcement copy",
            "mailbox note",
            "subgraph_failure:true",
            "block_role:task",
            "block_kind:task",
            "block_failure:unknown",
            "block_role:agent_action",
            "block_kind:tool_call",
            "block_failure:unknown",
            "claim_relation:influence",
            "claim_failure:unknown",
            "claim_relation:skip_link",
            "claim_failure:unknown",
            "relation_profile:influence:1,skip_link:1",
        ]
    )

    bare = _freeze("sess-bare", "rev-bare", _similar_blocks("sess-bare"), [_cutover()])
    bare_text = _subgraph_retrieval_text(bare, bare.retrieval_subgraphs[0])
    assert bare_text.endswith(
        "\n".join(
            [
                "subgraph_failure:true",
                "block_role:agent_action",
                "block_kind:unknown",
                "block_failure:unknown",
                "block_role:agent_action",
                "block_kind:unknown",
                "block_failure:unknown",
                "relation_profile:unknown",
            ]
        )
    )


def test_relation_profile_sorts_names_and_caps_counts() -> None:
    block = _block("sess-cap", "b-one", 1, "deploy switch traffic")
    claims = [
        InfluenceClaim(
            claim_id=f"c-skip-{index}",
            source_block_id="b-one",
            target_block_id="b-one",
            carried_artifact="packet shift",
            downstream_effect="status climbed",
            relation_type="skip",
        )
        for index in range(2)
    ]
    claims.extend(
        InfluenceClaim(
            claim_id=f"c-influence-{index}",
            source_block_id="b-one",
            target_block_id="b-one",
            carried_artifact="packet shift",
            downstream_effect="status climbed",
            relation_type="influence",
        )
        for index in range(100)
    )
    claims.append(
        InfluenceClaim(
            claim_id="c-marks",
            source_block_id="b-one",
            target_block_id="b-one",
            carried_artifact="packet shift",
            downstream_effect="status climbed",
            relation_type="@@@",
        )
    )
    subgraph = RetrievalSubgraph(
        subgraph_id="sg-cap",
        block_ids=("b-one",),
        claim_ids=tuple(claim.claim_id for claim in claims),
        failure_marked=False,
    )
    revision = _freeze("sess-cap", "rev-cap", [block], [subgraph], claims=claims)
    text = _subgraph_retrieval_text(revision, subgraph)
    assert text.splitlines()[-1] == "relation_profile:___:1,influence:99,skip:2"
    assert "claim_relation:___" in text
    assert "claim_failure:unknown" in text


def test_same_prose_ranks_failure_influence_ahead_of_skip(tmp_path: Path) -> None:
    def blocks(session_id: str) -> list[InfluenceBlock]:
        return [
            _block(session_id, "b-deploy", 1, "deploy switch traffic"),
            _block(session_id, "b-inspect", 2, "inspect status"),
        ]

    def claim(relation: str) -> InfluenceClaim:
        return InfluenceClaim(
            claim_id="c-link",
            source_block_id="b-deploy",
            target_block_id="b-inspect",
            carried_artifact="packet shift",
            downstream_effect="status climbed",
            relation_type=relation,
        )

    def subgraph(failure_marked: bool) -> RetrievalSubgraph:
        return RetrievalSubgraph(
            subgraph_id="sg-cutover",
            block_ids=("b-deploy", "b-inspect"),
            claim_ids=("c-link",),
            failure_marked=failure_marked,
        )

    index = _index(tmp_path)
    skip_revision = _freeze(
        "a-skip",
        "rev-skip",
        blocks("a-skip"),
        [subgraph(False)],
        claims=[claim("skip")],
    )
    influence_revision = _freeze(
        "b-influence",
        "rev-influence",
        blocks("b-influence"),
        [subgraph(True)],
        claims=[claim("influence")],
    )
    query_revision = _freeze(
        "zzz-query",
        "rev-query",
        blocks("zzz-query"),
        [subgraph(True)],
        claims=[claim("influence")],
    )
    index.register(skip_revision)
    index.register(influence_revision)
    index.register(query_revision)

    query_text = _subgraph_retrieval_text(query_revision, query_revision.retrieval_subgraphs[0])
    influence_text = _subgraph_retrieval_text(
        influence_revision, influence_revision.retrieval_subgraphs[0]
    )
    skip_text = _subgraph_retrieval_text(skip_revision, skip_revision.retrieval_subgraphs[0])
    assert query_text == influence_text
    prose = query_text.split("\nsubgraph_failure:", 1)[0]
    assert prose == skip_text.split("\nsubgraph_failure:", 1)[0]
    assert "claim_relation:influence" in query_text
    assert "subgraph_failure:true" in query_text
    assert "claim_relation:skip" in skip_text
    assert "subgraph_failure:false" in skip_text

    hits = _facade(tmp_path, index).retrieve_historical_subgraphs(
        "rev-query", "sg-cutover", _PRINCIPAL
    )
    assert [hit.view_revision_id for hit in hits] == ["rev-influence", "rev-skip"]
    assert hits[0].score_breakdown["rank_tier"] == RANK_MODE_BM25
    assert "vector" not in hits[0].score_breakdown
    assert hits[0].score_breakdown["bm25"] > hits[1].score_breakdown["bm25"] > 0
    assert len(hits) <= DIAGNOSTIC_SUBGRAPH_HIT_CAP
