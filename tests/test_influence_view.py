# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Unit tests for the Influence View layer (slice S7)."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from openviking.session.influence_view import (
    AORange,
    CoverageManifest,
    DETERMINISTIC_FALLBACK_REASON,
    SYNTHETIC_ROOT_SINK_REASON,
    HighSeverityViolation,
    InfluenceBlock,
    InfluenceClaim,
    InfluenceEvidenceRef,
    InfluenceViewDraft,
    InfluenceViewRevision,
    OmittedRange,
    RetrievalSubgraph,
    SegmentRef,
    SourceRef,
    ViewSupersedeIndex,
    Violation,
    build_minimal_fallback_view,
    compute_content_hash,
    validate_view,
)

SESSION = "sess-1"
TASK = "task-run-1"


def _ref(
    sequence: int,
    *,
    participant_id: str = "agent-a",
    segment_id: str = "seg-a",
    session_id: str = SESSION,
    ao_id: str | None = None,
) -> SourceRef:
    return SourceRef(
        ao_id=ao_id or f"ao-{sequence}",
        sequence=sequence,
        participant_id=participant_id,
        segment_id=segment_id,
        session_id=session_id,
        content_hash=f"hash-{sequence}",
    )


def _block(
    block_id: str,
    sequences: list[int],
    *,
    role: str = "agent_action",
    participant_id: str = "agent-a",
    segment_id: str = "seg-a",
    session_id: str = SESSION,
    granularity_reason: str = "one tool call",
    summary: str = "did the step",
) -> InfluenceBlock:
    refs = tuple(
        _ref(
            sequence,
            participant_id=participant_id,
            segment_id=segment_id,
            session_id=session_id,
        )
        for sequence in sequences
    )
    return InfluenceBlock(
        block_id=block_id,
        role=role,  # type: ignore[arg-type]
        participant_id=participant_id,
        segment_ref=SegmentRef(segment_id=segment_id, session_id=session_id),
        ao_start_seq=min(sequences),
        ao_end_seq=max(sequences),
        summary=summary,
        input="in",
        output="out",
        authorship=participant_id,
        granularity_reason=granularity_reason,
        source_content_hashes=tuple(ref.content_hash for ref in refs),
        source_refs=refs,
    )


def _coverage(blocks: list[InfluenceBlock], *, omitted: tuple[OmittedRange, ...] = ()) -> CoverageManifest:
    refs = tuple(ref for block in blocks for ref in block.source_refs)
    sequences = [ref.sequence for ref in refs] or [1]
    return CoverageManifest(
        total_ao_range=AORange(start_seq=min(sequences), end_seq=max(sequences)),
        inspected_refs=refs,
        included_ranges=tuple(
            AORange(
                start_seq=block.ao_start_seq,
                end_seq=block.ao_end_seq,
                segment_id=block.segment_ref.segment_id,
            )
            for block in blocks
        ),
        omitted_ranges=omitted,
        snapshot_watermark="wm-1",
    )


def _draft(
    blocks: list[InfluenceBlock],
    claims: list[InfluenceClaim] | None = None,
    *,
    purpose: str = "post_run_index",
    omitted: tuple[OmittedRange, ...] = (),
    clock=None,
) -> InfluenceViewDraft:
    return InfluenceViewDraft(
        view_id="view-1",
        purpose=purpose,  # type: ignore[arg-type]
        session_id=SESSION,
        task_run_id=TASK,
        coverage=_coverage(blocks, omitted=omitted),
        blocks=blocks,
        claims=claims or [],
        clock=clock,
    )


def _claim(claim_id: str, source: str, target: str, *, relation_type: str = "influence") -> InfluenceClaim:
    return InfluenceClaim(
        claim_id=claim_id,
        source_block_id=source,
        target_block_id=target,
        carried_artifact="patch",
        downstream_effect="tests passed",
        relation_type=relation_type,
    )


def _evidence(sequence: int) -> InfluenceEvidenceRef:
    return InfluenceEvidenceRef(
        ao_id=f"ao-{sequence}",
        source_session_id=SESSION,
        source_archive_id=None,
        archive_commit_watermark=None,
        source_sequence=sequence,
        source_read_snapshot_watermark="wm-1",
        evidence_role="contemporaneous_basis",
        evidence_content_hash=f"hash-{sequence}",
        captured_state="committed",
    )


def test_block_rejects_cross_participant_segment_gap_and_empty_reason() -> None:
    with pytest.raises(ValueError, match="single participant"):
        InfluenceBlock(
            block_id="b-cross-p",
            role="agent_action",
            participant_id="agent-a",
            segment_ref=SegmentRef(segment_id="seg-a", session_id=SESSION),
            ao_start_seq=1,
            ao_end_seq=2,
            summary="s",
            input="",
            output="",
            authorship="agent-a",
            granularity_reason="split by participant",
            source_content_hashes=("hash-1", "hash-2"),
            source_refs=(
                _ref(1, participant_id="agent-a"),
                _ref(2, participant_id="agent-b"),
            ),
        )

    with pytest.raises(ValueError, match="single segment"):
        InfluenceBlock(
            block_id="b-cross-s",
            role="agent_action",
            participant_id="agent-a",
            segment_ref=SegmentRef(segment_id="seg-a", session_id=SESSION),
            ao_start_seq=1,
            ao_end_seq=2,
            summary="s",
            input="",
            output="",
            authorship="agent-a",
            granularity_reason="split by segment",
            source_content_hashes=("hash-1", "hash-2"),
            source_refs=(
                _ref(1, segment_id="seg-a"),
                _ref(2, segment_id="seg-b"),
            ),
        )

    with pytest.raises(ValueError, match="contiguous"):
        InfluenceBlock(
            block_id="b-gap",
            role="agent_action",
            participant_id="agent-a",
            segment_ref=SegmentRef(segment_id="seg-a", session_id=SESSION),
            ao_start_seq=1,
            ao_end_seq=3,
            summary="s",
            input="",
            output="",
            authorship="agent-a",
            granularity_reason="merged a gap",
            source_content_hashes=("hash-1", "hash-3"),
            source_refs=(_ref(1), _ref(3)),
        )

    with pytest.raises(ValueError, match="granularity_reason"):
        _block("b-empty-reason", [1], granularity_reason="  ")


def test_claim_temporal_inversion_and_cycle_are_violations() -> None:
    early = _block("early", [1, 2], segment_id="seg-a")
    late = _block("late", [4, 5], segment_id="seg-b", participant_id="agent-b")
    inverted = _draft([early, late], [_claim("c-back", "late", "early")])
    inverted_codes = {item.code for item in validate_view(inverted)}
    assert "temporal_order" in inverted_codes
    temporal = next(item for item in validate_view(inverted) if item.code == "temporal_order")
    assert temporal.severity == "high"
    assert temporal.location == "claim:c-back"

    forward = _draft([early, late], [_claim("c-ok", "early", "late")])
    assert [item.code for item in validate_view(forward)] == []

    cycle = _draft(
        [early, late],
        [_claim("c-1", "early", "late"), _claim("c-2", "late", "early")],
    )
    cycle_hits = [item for item in validate_view(cycle) if item.code == "cycle"]
    assert cycle_hits
    assert all(item.severity == "high" for item in cycle_hits)


def test_freeze_is_immutable_and_content_hash_is_stable() -> None:
    block = _block("b1", [1, 2])
    first = _draft([block]).freeze(frozen_at="2026-09-24T00:00:00.000Z", revision_id="rev-1")
    second = _draft([block]).freeze(frozen_at="2026-09-24T01:00:00.000Z", revision_id="rev-2")
    assert first.content_hash == second.content_hash
    assert first.content_hash == compute_content_hash(first.blocks, first.claims, first.coverage)
    changed = _draft([_block("b1", [1, 2], summary="different")]).freeze(
        frozen_at="2026-09-24T00:00:00.000Z",
        revision_id="rev-3",
    )
    assert changed.content_hash != first.content_hash

    with pytest.raises(FrozenInstanceError):
        first.construction_mode = "fallback"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.blocks = ()  # type: ignore[misc]

    restored = type(first).from_dict(first.to_dict())
    assert restored == first


def test_critic_rounds_cap_rejects_fourth_edit() -> None:
    draft = _draft([_block("b1", [1])])
    violation = Violation(
        code="granularity",
        severity="low",
        message="split the block",
        location="block:b1",
    )
    for index in range(3):
        draft.apply_critic_patch(violation, note=f"round-{index + 1}")
    assert draft.critic_rounds == 3
    assert [item.round_index for item in draft.patch_provenance] == [1, 2, 3]
    assert draft.patch_provenance[0].violation_code == "granularity"
    with pytest.raises(ValueError, match="critic_rounds"):
        draft.apply_critic_patch(violation, note="round-4")
    with pytest.raises(ValueError, match="critic_rounds"):
        draft.add_block(_block("b2", [2], segment_id="seg-b"))
    with pytest.raises(ValueError, match="critic_rounds"):
        draft.add_claim(_claim("c-new", "b1", "b2"))


def test_fallback_builder_emits_valid_fallback_view() -> None:
    session = {
        "session_id": SESSION,
        "task_run_id": TASK,
        "purpose": "post_run_index",
        "snapshot_watermark": "wm-fallback",
        "view_id": "view-fb",
        "revision_id": "rev-fb",
        "frozen_at": "2026-09-24T02:00:00.000Z",
        "ao_records": [
            {
                "ao_id": "ao-1",
                "sequence": 1,
                "participant_id": "agent-a",
                "segment_id": "seg-a",
                "content_hash": "hash-1",
            },
            {
                "ao_id": "ao-2",
                "sequence": 2,
                "participant_id": "agent-a",
                "segment_id": "seg-a",
                "content_hash": "hash-2",
            },
            {
                "ao_id": "ao-3",
                "sequence": 3,
                "participant_id": "agent-b",
                "segment_id": "seg-b",
                "content_hash": "hash-3",
            },
        ],
    }
    revision = build_minimal_fallback_view(session)
    assert revision.construction_mode == "fallback"
    assert revision.claims == ()
    assert len(revision.blocks) == 2
    assert {block.granularity_reason for block in revision.blocks} == {DETERMINISTIC_FALLBACK_REASON}
    assert [block.segment_ref.segment_id for block in revision.blocks] == ["seg-a", "seg-b"]
    assert revision.blocks[0].ao_start_seq == 1
    assert revision.blocks[0].ao_end_seq == 2
    assert revision.blocks[0].source_refs[0].ao_id == "ao-1"
    assert validate_view(revision) == []
    again = build_minimal_fallback_view(session)
    assert again.content_hash == revision.content_hash


def test_supersede_keeps_history_and_points_latest_at_the_new_revision(tmp_path: Path) -> None:
    index_path = tmp_path / "influence-view-index.json"
    index = ViewSupersedeIndex(index_path)
    first = _draft([_block("b1", [1])]).freeze(
        revision_id="rev-old",
        frozen_at="2026-09-24T00:00:00.000Z",
    )
    second = _draft([_block("b1", [1, 2])]).freeze(
        revision_id="rev-new",
        frozen_at="2026-09-24T03:00:00.000Z",
    )
    root = _block("root", [1], role="task", participant_id="task", segment_id="seg-root")
    mid = _block("b1", [2])
    sink = _block("sink", [3], role="conclusion", participant_id="judge", segment_id="seg-sink")
    other = _draft(
        [root, mid, sink],
        [_claim("c1", "root", "b1"), _claim("c2", "b1", "sink")],
        purpose="failure_diagnosis",
    ).freeze(
        revision_id="rev-diag",
        frozen_at="2026-09-24T04:00:00.000Z",
    )
    index.register(first)
    index.register(second)
    index.register(other)

    assert index.latest(SESSION, "post_run_index") == second
    assert [item.revision_id for item in index.history(SESSION, "post_run_index")] == [
        "rev-old",
        "rev-new",
    ]
    assert index.latest(SESSION, "failure_diagnosis") == other

    reloaded = ViewSupersedeIndex(index_path)
    assert reloaded.latest(SESSION, "post_run_index") == second
    assert [item.revision_id for item in reloaded.history(SESSION, "post_run_index")] == [
        "rev-old",
        "rev-new",
    ]


def test_failure_diagnosis_requires_root_sink_weak_subgraph() -> None:
    root = _block("root", [1], role="task", participant_id="task", segment_id="seg-root")
    mid = _block("mid", [2], participant_id="agent-a", segment_id="seg-a")
    sink = _block("sink", [3], role="conclusion", participant_id="judge", segment_id="seg-sink")
    stray = _block("stray", [4], participant_id="agent-b", segment_id="seg-b")
    connected = _draft(
        [root, mid, sink],
        [_claim("c1", "root", "mid"), _claim("c2", "mid", "sink")],
        purpose="failure_diagnosis",
    )
    assert validate_view(connected) == []

    with_stray = _draft(
        [root, mid, sink, stray],
        [_claim("c1", "root", "mid"), _claim("c2", "mid", "sink")],
        purpose="failure_diagnosis",
    )
    stray_hits = [
        item for item in validate_view(with_stray) if item.code == "block_outside_root_sink_subgraph"
    ]
    assert [item.location for item in stray_hits] == ["block:stray"]

    post_run = _draft([mid, stray], purpose="post_run_index")
    assert validate_view(post_run) == []


def test_omitted_range_without_reason_is_a_violation() -> None:
    block = _block("b1", [1])
    draft = _draft(
        [block],
        omitted=(OmittedRange(start_seq=2, end_seq=2, reason=""),),
    )
    hits = [item for item in validate_view(draft) if item.code == "omitted_range_missing_reason"]
    assert len(hits) == 1
    assert hits[0].severity == "high"
    assert hits[0].location == "coverage.omitted_ranges[0]"


def test_json_roundtrip_preserves_claim_status_and_critic_verdict() -> None:
    block_a = _block("b1", [1])
    block_b = _block("b2", [3], segment_id="seg-b", participant_id="agent-b")
    claim = InfluenceClaim(
        claim_id="c1",
        source_block_id="b1",
        target_block_id="b2",
        carried_artifact="diff",
        downstream_effect="build failed",
        relation_type="skip",
        source_evidence_refs=(_evidence(1),),
        target_evidence_refs=(_evidence(3),),
        status="supported",
        critic_verdict="no_structural_issue",
    )
    draft = _draft(
        [block_a, block_b],
        omitted=(OmittedRange(start_seq=2, end_seq=2, reason="unread side path"),),
    )
    draft.add_claim(
        InfluenceClaim(
            claim_id="c1",
            source_block_id="b1",
            target_block_id="b2",
            carried_artifact="diff",
            downstream_effect="build failed",
            relation_type="skip",
            source_evidence_refs=(_evidence(1),),
            target_evidence_refs=(_evidence(3),),
        )
    )
    loaded = InfluenceViewDraft.from_dict(draft.to_dict())
    assert loaded.to_dict() == draft.to_dict()

    revision = InfluenceViewDraft(
        view_id="view-1",
        purpose="post_run_index",
        session_id=SESSION,
        task_run_id=TASK,
        coverage=draft.coverage,
        blocks=[block_a, block_b],
        claims=[claim],
    ).freeze(revision_id="rev-rt", frozen_at="2026-09-24T05:00:00.000Z")
    assert revision.claims[0].status == "supported"
    assert revision.claims[0].critic_verdict == "no_structural_issue"
    assert type(revision).from_dict(revision.to_dict()) == revision


def test_freeze_rejects_high_severity_violations() -> None:
    draft = _draft([_block("b1", [1])], purpose="failure_diagnosis")
    with pytest.raises(HighSeverityViolation) as caught:
        draft.freeze(revision_id="rev-bad", frozen_at="2026-09-24T06:00:00.000Z")
    assert caught.value.violations
    assert all(item.severity == "high" for item in caught.value.violations)
    assert {item.code for item in caught.value.violations} >= {
        "missing_task_root",
        "missing_outcome_sink",
    }
    bypassed = draft.freeze(
        revision_id="rev-bypass",
        frozen_at="2026-09-24T06:00:00.000Z",
        enforce_checks=False,
    )
    assert bypassed.revision_id == "rev-bypass"
    assert validate_view(bypassed)


def test_failure_diagnosis_fallback_freezes_with_synthetic_root_and_sink() -> None:
    session = {
        "session_id": SESSION,
        "task_run_id": TASK,
        "purpose": "failure_diagnosis",
        "snapshot_watermark": "wm-fallback",
        "view_id": "view-fb-diag",
        "revision_id": "rev-fb-diag",
        "frozen_at": "2026-09-24T07:00:00.000Z",
        "ao_records": [
            {
                "ao_id": "ao-1",
                "sequence": 1,
                "participant_id": "agent-a",
                "segment_id": "seg-a",
                "content_hash": "hash-1",
            },
            {
                "ao_id": "ao-3",
                "sequence": 3,
                "participant_id": "agent-b",
                "segment_id": "seg-b",
                "content_hash": "hash-3",
            },
        ],
    }
    revision = build_minimal_fallback_view(session)
    assert revision.construction_mode == "fallback"
    assert validate_view(revision) == []
    synthetic = [
        block for block in revision.blocks if block.granularity_reason == SYNTHETIC_ROOT_SINK_REASON
    ]
    assert [block.role for block in synthetic] == ["task", "conclusion"]
    assert {block.role for block in revision.blocks} >= {"task", "agent_action", "conclusion"}
    again = build_minimal_fallback_view(session)
    assert again.content_hash == revision.content_hash


def test_injected_clock_stamps_persisted_frozen_at(tmp_path: Path) -> None:
    fixed = datetime(2026, 1, 2, 3, 4, 5, 678000, tzinfo=timezone.utc)
    expected = "2026-01-02T03:04:05.678Z"

    def clock() -> datetime:
        return fixed

    draft = _draft([_block("b1", [1, 2])], clock=clock)
    restored = InfluenceViewDraft.from_dict(draft.to_dict(), clock=clock)
    revision = restored.freeze()
    assert revision.frozen_at == expected

    index_path = tmp_path / "influence-view-index.json"
    ViewSupersedeIndex(index_path).register(revision)
    loaded = ViewSupersedeIndex(index_path).latest(SESSION, "post_run_index")
    assert loaded is not None
    assert loaded.frozen_at == expected
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    assert raw["keys"][0]["revisions"][0]["frozen_at"] == expected

    fallback = build_minimal_fallback_view(
        {
            "session_id": SESSION,
            "task_run_id": TASK,
            "purpose": "post_run_index",
            "snapshot_watermark": "wm-fallback",
            "view_id": "view-fb-clock",
            "revision_id": "rev-fb-clock",
            "ao_records": [
                {
                    "ao_id": "ao-1",
                    "sequence": 1,
                    "participant_id": "agent-a",
                    "segment_id": "seg-a",
                    "content_hash": "hash-1",
                }
            ],
        },
        clock=clock,
    )
    assert fallback.frozen_at == expected


def _subgraph(
    subgraph_id: str,
    block_ids: tuple[str, ...],
    claim_ids: tuple[str, ...] = (),
    *,
    failure_marked: bool = False,
    failure_evidence_refs: tuple[InfluenceEvidenceRef, ...] = (),
) -> RetrievalSubgraph:
    return RetrievalSubgraph(
        subgraph_id=subgraph_id,
        block_ids=block_ids,
        claim_ids=claim_ids,
        failure_marked=failure_marked,
        failure_evidence_refs=failure_evidence_refs,
    )


def _span() -> tuple[InfluenceBlock, InfluenceBlock, InfluenceClaim]:
    early = _block("b-script", [1, 2], segment_id="seg-a")
    late = _block("b-rate", [4, 5], segment_id="seg-b", participant_id="agent-b")
    return early, late, _claim("c-cutover", "b-script", "b-rate")


def _codes(view: InfluenceViewDraft | InfluenceViewRevision) -> set[str]:
    return {item.code for item in validate_view(view)}


def test_retrieval_subgraph_roundtrip_keeps_failure_marker_and_evidence() -> None:
    early, late, claim = _span()
    evidence = _evidence(5)
    draft = _draft([early, late], [claim])
    draft.add_retrieval_subgraph(
        _subgraph(
            "sg-cutover",
            ("b-script", "b-rate"),
            ("c-cutover",),
            failure_marked=True,
            failure_evidence_refs=(evidence,),
        )
    )
    assert _codes(draft) == set()
    revision = draft.freeze(revision_id="rev-sg", frozen_at="2026-09-24T08:00:00.000Z")
    restored = InfluenceViewRevision.from_dict(json.loads(json.dumps(revision.to_dict())))
    assert restored == revision
    assert restored.retrieval_subgraphs[0].failure_marked is True
    assert restored.retrieval_subgraphs[0].is_failure is True
    assert restored.retrieval_subgraphs[0].failure_evidence_refs == (evidence,)
    assert restored.retrieval_subgraphs[0].to_dict()["failure_evidence_refs"][0]["ao_id"] == "ao-5"

    legacy = revision.to_dict()
    legacy.pop("retrieval_subgraphs")
    loaded = InfluenceViewRevision.from_dict(legacy)
    assert loaded.retrieval_subgraphs == ()
    assert loaded.content_hash == revision.content_hash


def test_content_hash_covers_retrieval_subgraphs() -> None:
    early, late, claim = _span()
    plain = _subgraph("sg-cutover", ("b-script", "b-rate"), ("c-cutover",))
    marked = _subgraph(
        "sg-cutover",
        ("b-script", "b-rate"),
        ("c-cutover",),
        failure_marked=True,
        failure_evidence_refs=(_evidence(5),),
    )
    coverage = _coverage([early, late])
    empty_hash = compute_content_hash((early, late), (claim,), coverage)
    plain_hash = compute_content_hash((early, late), (claim,), coverage, (plain,))
    marked_hash = compute_content_hash((early, late), (claim,), coverage, (marked,))
    assert plain_hash != empty_hash
    assert marked_hash != plain_hash
    assert compute_content_hash((early, late), (claim,), coverage, (marked,)) == marked_hash

    stamp = "2026-09-24T08:00:00.000Z"
    without = _draft([early, late], [claim]).freeze(revision_id="rev-sg", frozen_at=stamp)
    with_plain = _draft([early, late], [claim])
    with_plain.add_retrieval_subgraph(plain)
    frozen_plain = with_plain.freeze(revision_id="rev-sg", frozen_at=stamp)
    assert frozen_plain.content_hash == plain_hash
    assert frozen_plain != without
    again = _draft([early, late], [claim])
    again.add_retrieval_subgraph(plain)
    assert again.freeze(revision_id="rev-sg", frozen_at=stamp) == frozen_plain


def test_retrieval_subgraph_unknown_member_is_a_violation() -> None:
    early, late, claim = _span()
    legal = _draft([early, late], [claim])
    legal.add_retrieval_subgraph(_subgraph("sg-ok", ("b-script", "b-rate"), ("c-cutover",)))
    assert _codes(legal) == set()

    illegal = _draft([early, late], [claim])
    illegal.add_retrieval_subgraph(
        _subgraph("sg-block", ("b-script", "b-rate", "missing-block"), ("c-cutover",))
    )
    illegal.add_retrieval_subgraph(
        _subgraph("sg-claim", ("b-script", "b-rate"), ("c-cutover", "missing-claim"))
    )
    hits = [item for item in validate_view(illegal) if item.code == "subgraph_unknown_member"]
    assert {item.location for item in hits} == {
        "subgraph:sg-block/block_ids/missing-block",
        "subgraph:sg-claim/claim_ids/missing-claim",
    }
    assert _codes(illegal) == {"subgraph_unknown_member"}


def test_retrieval_subgraph_claim_endpoint_must_lie_inside_blocks() -> None:
    early, late, claim = _span()
    legal = _draft([early, late], [claim])
    legal.add_retrieval_subgraph(_subgraph("sg-ok", ("b-script", "b-rate"), ("c-cutover",)))
    assert _codes(legal) == set()

    illegal = _draft([early, late], [claim])
    illegal.add_retrieval_subgraph(_subgraph("sg-half", ("b-script",), ("c-cutover",)))
    hits = [item for item in validate_view(illegal) if item.code == "subgraph_claim_endpoint_outside"]
    assert len(hits) == 1
    assert hits[0].location == "subgraph:sg-half/claim_ids/c-cutover"
    assert _codes(illegal) == {"subgraph_claim_endpoint_outside"}


def test_duplicate_retrieval_subgraph_id_is_a_violation() -> None:
    early, late, claim = _span()
    legal = _draft([early, late], [claim])
    legal.add_retrieval_subgraph(_subgraph("sg-a", ("b-script", "b-rate"), ("c-cutover",)))
    legal.add_retrieval_subgraph(_subgraph("sg-b", ("b-rate",), ()))
    assert _codes(legal) == set()

    illegal = _draft([early, late], [claim])
    illegal.add_retrieval_subgraph(_subgraph("sg-a", ("b-script", "b-rate"), ("c-cutover",)))
    illegal.add_retrieval_subgraph(_subgraph("sg-a", ("b-script",), ()))
    hits = [item for item in validate_view(illegal) if item.code == "duplicate_subgraph_id"]
    assert len(hits) == 1
    assert hits[0].location == "subgraph:sg-a"
    assert _codes(illegal) == {"duplicate_subgraph_id"}


def test_retrieval_subgraph_over_block_limit_is_a_violation() -> None:
    def many(count: int) -> list[InfluenceBlock]:
        return [_block(f"b{index}", [index], segment_id=f"seg-{index}") for index in range(1, count + 1)]

    allowed = many(12)
    legal = _draft(allowed)
    legal.add_retrieval_subgraph(_subgraph("sg-fit", tuple(block.block_id for block in allowed)))
    assert _codes(legal) == set()
    legal.freeze(revision_id="rev-fit", frozen_at="2026-09-24T08:00:00.000Z")

    oversized = many(13)
    illegal = _draft(oversized)
    illegal.add_retrieval_subgraph(_subgraph("sg-over", tuple(block.block_id for block in oversized)))
    hits = [item for item in validate_view(illegal) if item.code == "subgraph_block_limit"]
    assert len(hits) == 1
    assert hits[0].location == "subgraph:sg-over/block_ids"
    assert _codes(illegal) == {"subgraph_block_limit"}
    with pytest.raises(HighSeverityViolation) as caught:
        illegal.freeze(revision_id="rev-over", frozen_at="2026-09-24T08:00:00.000Z")
    assert {item.code for item in caught.value.violations} == {"subgraph_block_limit"}


def test_retrieval_subgraph_failure_evidence_must_match_marker() -> None:
    early, late, claim = _span()
    evidence = (_evidence(5),)
    legal = _draft([early, late], [claim])
    legal.add_retrieval_subgraph(
        _subgraph("sg-open", ("b-script", "b-rate"), ("c-cutover",))
    )
    legal.add_retrieval_subgraph(
        _subgraph(
            "sg-failed",
            ("b-script", "b-rate"),
            ("c-cutover",),
            failure_marked=True,
            failure_evidence_refs=evidence,
        )
    )
    assert _codes(legal) == set()
    assert legal.retrieval_subgraphs[0].is_failure is False
    assert legal.retrieval_subgraphs[1].is_failure is True

    missing = _draft([early, late], [claim])
    missing.add_retrieval_subgraph(
        _subgraph("sg-bare", ("b-script", "b-rate"), ("c-cutover",), failure_marked=True)
    )
    bare_hits = [item for item in validate_view(missing) if item.code == "subgraph_failure_evidence"]
    assert len(bare_hits) == 1
    assert bare_hits[0].location == "subgraph:sg-bare/failure_evidence_refs"

    extra = _draft([early, late], [claim])
    extra.add_retrieval_subgraph(
        _subgraph(
            "sg-extra",
            ("b-script", "b-rate"),
            ("c-cutover",),
            failure_evidence_refs=evidence,
        )
    )
    extra_hits = [item for item in validate_view(extra) if item.code == "subgraph_failure_evidence"]
    assert len(extra_hits) == 1
    assert extra_hits[0].location == "subgraph:sg-extra/failure_evidence_refs"
    assert extra.retrieval_subgraphs[0].is_failure is False


def test_conclusion_block_without_marker_is_not_a_failure_subgraph() -> None:
    action = _block("b-cutover", [1, 2], segment_id="seg-a")
    outcome = _block(
        "b-rate",
        [4],
        role="conclusion",
        participant_id="agent-b",
        segment_id="seg-b",
        summary="error rate rose",
    )
    claim = _claim("c-rate", "b-cutover", "b-rate")
    draft = _draft([action, outcome], [claim])
    draft.add_retrieval_subgraph(_subgraph("sg-rate", ("b-cutover", "b-rate"), ("c-rate",)))
    assert validate_view(draft) == []
    revision = draft.freeze(revision_id="rev-rate", frozen_at="2026-09-24T08:00:00.000Z")
    subgraph = revision.retrieval_subgraphs[0]
    roles = {
        block.role
        for block in revision.blocks
        if block.block_id in subgraph.block_ids
    }
    assert "conclusion" in roles
    assert outcome.summary == "error rate rose"
    assert subgraph.failure_marked is False
    assert subgraph.is_failure is False


def test_retrieval_subgraph_serialization_omits_subtask_wording() -> None:
    early, late, claim = _span()
    draft = _draft([early, late], [claim])
    draft.add_retrieval_subgraph(
        _subgraph(
            "sg-cutover",
            ("b-script", "b-rate"),
            ("c-cutover",),
            failure_marked=True,
            failure_evidence_refs=(_evidence(5),),
        )
    )
    revision = draft.freeze(revision_id="rev-sg", frozen_at="2026-09-24T08:00:00.000Z")
    text = json.dumps(revision.to_dict())
    assert "subtask" not in text


def test_fallback_view_leaves_retrieval_subgraphs_empty() -> None:
    records = [
        {
            "ao_id": "ao-1",
            "sequence": 1,
            "participant_id": "agent-a",
            "segment_id": "seg-a",
            "content_hash": "hash-1",
        }
    ]
    post = build_minimal_fallback_view(
        {
            "session_id": SESSION,
            "task_run_id": TASK,
            "purpose": "post_run_index",
            "snapshot_watermark": "wm-fallback",
            "view_id": "view-fb",
            "revision_id": "rev-fb",
            "frozen_at": "2026-09-24T02:00:00.000Z",
            "ao_records": records,
        }
    )
    diagnosis = build_minimal_fallback_view(
        {
            "session_id": SESSION,
            "task_run_id": TASK,
            "purpose": "failure_diagnosis",
            "snapshot_watermark": "wm-fallback",
            "view_id": "view-fb-diag",
            "revision_id": "rev-fb-diag",
            "frozen_at": "2026-09-24T07:00:00.000Z",
            "ao_records": records,
        }
    )
    assert post.retrieval_subgraphs == ()
    assert diagnosis.retrieval_subgraphs == ()
    assert any(block.role == "conclusion" for block in diagnosis.blocks)
    assert validate_view(post) == []
    assert validate_view(diagnosis) == []
