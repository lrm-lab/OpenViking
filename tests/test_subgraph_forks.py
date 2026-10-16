# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""One ordinary fork per failure-marked retrieval subgraph (R4b)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from openviking.session.causal_experiences import (
    SEVEN_SECTION_KEYS,
    CausalExperiencesStore,
    ForkRevision,
    influence_position_key,
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
    ForkBranchAdapter,
    HybridRetrievalFacade,
    InfluenceProjectionAdapter,
    RetrievalSubgraphAdapter,
    SegmentAtomAdapter,
)
from openviking.session.subgraph_forks import (
    default_failure_grounding_position,
    failure_subgraphs_owed_forks,
    write_failure_subgraph_fork,
)

_CAUSAL = {"skill_trajectory_mode": "causal"}
_FROZEN = "2026-01-02T00:00:00.000Z"
_PRINCIPAL = ("team-a",)
_SESSION = "hist-sess"
_GUIDANCE = "切回蓝"


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


def _block(
    session_id: str,
    block_id: str,
    sequence: int,
    summary: str,
    *,
    role: str = "agent_action",
) -> InfluenceBlock:
    ref = _ref(session_id, sequence)
    return InfluenceBlock(
        block_id=block_id,
        role=role,  # type: ignore[arg-type]
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


def _evidence(session_id: str, sequence: int) -> InfluenceEvidenceRef:
    return InfluenceEvidenceRef(
        ao_id=f"ao-{session_id}-{sequence}",
        source_session_id=session_id,
        source_archive_id=None,
        archive_commit_watermark=None,
        source_sequence=sequence,
        source_read_snapshot_watermark="wm-1",
        evidence_role="hindsight_attribution",
        evidence_content_hash=f"hash-{session_id}-{sequence}",
        captured_state="committed",
    )


def _basis(anchor_sequence: int) -> dict[str, object]:
    sequence = anchor_sequence - 1
    return {
        "ao_id": f"ao-basis-{sequence}",
        "source_session_id": _SESSION,
        "source_archive_id": None,
        "archive_commit_watermark": None,
        "source_sequence": sequence,
        "source_read_snapshot_watermark": "wm-1",
        "evidence_role": "contemporaneous_basis",
        "evidence_content_hash": f"hash-basis-{sequence}",
        "captured_state": "committed",
    }


def _sections(decision: str) -> dict[str, str]:
    sections = {key: f"{key} text" for key in SEVEN_SECTION_KEYS}
    sections["key_facts_and_decisions"] = decision
    return sections


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
                segment_id=block.segment_ref.segment_id,
            )
            for block in blocks
        ),
        omitted_ranges=(),
        snapshot_watermark="wm-1",
    )


def _freeze(
    session_id: str,
    revision_id: str,
    blocks: list[InfluenceBlock],
    subgraphs: list[RetrievalSubgraph],
    *,
    claims: list[InfluenceClaim] | None = None,
) -> InfluenceViewDraft:
    draft = InfluenceViewDraft(
        view_id=f"view-{session_id}",
        purpose="post_run_index",
        session_id=session_id,
        task_run_id=f"task-{session_id}",
        coverage=_coverage(blocks),
        blocks=blocks,
        claims=claims or [],
        retrieval_subgraphs=subgraphs,
    )
    return draft.freeze(revision_id=revision_id, enforce_checks=False, frozen_at=_FROZEN)


def _release_blocks() -> list[InfluenceBlock]:
    return [
        _block(_SESSION, "b-cutover", 1, ""),
        _block(_SESSION, "b-rate", 2, "错误率升高"),
        _block(_SESSION, "b-migrate", 3, ""),
        _block(_SESSION, "b-schema", 4, "schema 不匹配"),
        _block(_SESSION, "b-notify", 5, "公告已发送"),
        _block(_SESSION, "b-release", 6, "发布失败", role="conclusion"),
    ]


def _release_claim() -> InfluenceClaim:
    return InfluenceClaim(
        claim_id="c-schema",
        source_block_id="b-migrate",
        target_block_id="b-schema",
        carried_artifact="migration artifact",
        downstream_effect="schema mismatch effect",
        relation_type="influence",
    )


def _release_subgraphs() -> list[RetrievalSubgraph]:
    return [
        RetrievalSubgraph(
            subgraph_id="sg-notify",
            block_ids=("b-notify",),
            claim_ids=(),
            failure_marked=False,
        ),
        RetrievalSubgraph(
            subgraph_id="sg-cutover",
            block_ids=("b-cutover", "b-rate"),
            claim_ids=(),
            failure_marked=True,
            failure_evidence_refs=(_evidence(_SESSION, 2),),
        ),
        RetrievalSubgraph(
            subgraph_id="sg-release",
            block_ids=("b-release",),
            claim_ids=(),
            failure_marked=False,
        ),
        RetrievalSubgraph(
            subgraph_id="sg-migrate",
            block_ids=("b-migrate", "b-schema"),
            claim_ids=("c-schema",),
            failure_marked=True,
            failure_evidence_refs=(_evidence(_SESSION, 4),),
        ),
    ]


def _release_revision():
    return _freeze(
        _SESSION,
        "rev-hist",
        _release_blocks(),
        _release_subgraphs(),
        claims=[_release_claim()],
    )


def _store(tmp_path: Path) -> CausalExperiencesStore:
    return CausalExperiencesStore(tmp_path / "store", config=_CAUSAL)


def _revision_files(root: Path) -> list[Path]:
    return sorted((root / "causal-experiences" / "forks").glob("*/*/*.json"))


def _write(store: CausalExperiencesStore, revision, subgraph_id: str, **overrides):
    anchor_sequence = 2 if subgraph_id == "sg-cutover" else 4
    fields = {
        "seven_sections": _sections("错误率升高就切回蓝"),
        "branches": [
            {
                "branch_id": "br-cutover",
                "evidence_status": "real",
                "ao_sequence": ["ao-step"],
                "guidance": {"instruction": _GUIDANCE},
            }
        ],
        "position_kind": "block",
        "position_ref": "block:b-rate",
        "workspace_id": "team-a",
        "workspace_labels": _PRINCIPAL,
        "diagnosis_run_id": "diag-cutover",
        "model_version": "diag-model-v1",
        "contemporaneous_basis": [_basis(anchor_sequence)],
    }
    fields.update(overrides)
    fields.setdefault("legacy_audit_opt_in", True)
    return write_failure_subgraph_fork(store, revision, subgraph_id, **fields)


def test_failure_subgraph_fork_refuses_unless_legacy_audit_opt_in(tmp_path: Path) -> None:
    from openviking.session.legacy_write_gate import LegacyWriteRefused

    revision = _release_revision()
    store = _store(tmp_path)
    with pytest.raises(LegacyWriteRefused):
        write_failure_subgraph_fork(
            store,
            revision,
            "sg-cutover",
            seven_sections=_sections("错误率升高就切回蓝"),
            branches=[],
            position_kind="block",
            position_ref="block:b-rate",
            workspace_id="team-a",
            workspace_labels=_PRINCIPAL,
            diagnosis_run_id="diag-cutover",
            model_version="diag-model-v1",
            contemporaneous_basis=[_basis(2)],
        )
    assert _revision_files(tmp_path / "store") == []
    written = _write(store, revision, "sg-cutover")
    assert written.fork_node_id


def test_one_failure_subgraph_writes_one_ordinary_fork(tmp_path: Path) -> None:
    revision = _release_revision()
    store = _store(tmp_path)
    kind, ref = default_failure_grounding_position(revision, "sg-cutover")
    written = _write(store, revision, "sg-cutover", position_kind=kind, position_ref=ref)

    assert isinstance(written, ForkRevision)
    assert not isinstance(written, tuple)
    assert written.status == "provisional"
    assert written.anchor == {
        "anchor_kind": "ao",
        "ao_id": "ao-hist-sess-2",
        "source_session_id": _SESSION,
        "anchor_sequence": 2,
    }
    assert written.contemporaneous_basis[0].source_sequence < written.anchor["anchor_sequence"]
    assert written.contemporaneous_basis[0].ao_id == "ao-basis-1"
    grounding = written.influence_grounding
    assert grounding is not None
    assert grounding.view_revision_id == "rev-hist"
    assert (grounding.position_kind, grounding.position_ref) == ("block", "block:b-rate")
    assert grounding.semantic_summary_snapshot == "错误率升高"
    assert grounding.content_hash == revision.content_hash
    assert grounding.grounded_at == revision.frozen_at
    assert written.created_at == revision.frozen_at
    assert written.branches[0]["guidance"] == {"instruction": _GUIDANCE}
    assert set(written.seven_sections) == set(SEVEN_SECTION_KEYS)
    files = _revision_files(tmp_path / "store")
    assert len(files) == 1
    disk = files[0].read_text(encoding="utf-8")
    assert "subtask" not in disk
    assert "cause_set" not in disk
    rows = store.query_intervention_history(
        "rev-hist", influence_position_key(grounding), _PRINCIPAL
    )
    assert [row["fork_node_id"] for row in rows] == [written.fork_node_id]


def test_two_failure_subgraphs_write_two_independent_forks(tmp_path: Path) -> None:
    revision = _release_revision()
    owed = failure_subgraphs_owed_forks(revision)
    assert [item.subgraph_id for item in owed] == ["sg-cutover", "sg-migrate"]
    store = _store(tmp_path)
    cutover_kind, cutover_ref = default_failure_grounding_position(revision, "sg-cutover")
    cutover = _write(
        store,
        revision,
        "sg-cutover",
        position_kind=cutover_kind,
        position_ref=cutover_ref,
        seven_sections=_sections("错误率升高就切回蓝"),
    )
    assert default_failure_grounding_position(revision, "sg-migrate") == ("block", "block:b-schema")
    migrate = _write(
        store,
        revision,
        "sg-migrate",
        position_kind="edge",
        position_ref="edge:c-schema",
        seven_sections=_sections("先修 schema"),
        branches=[
            {
                "branch_id": "br-migrate",
                "evidence_status": "real",
                "ao_sequence": ["ao-schema"],
                "guidance": {"instruction": "先修 schema"},
            }
        ],
        diagnosis_run_id="diag-migrate",
    )

    assert cutover.fork_node_id != migrate.fork_node_id
    assert cutover.revision_id != migrate.revision_id
    assert cutover.influence_grounding is not None
    assert migrate.influence_grounding is not None
    assert cutover.influence_grounding.position_ref == "block:b-rate"
    assert migrate.influence_grounding.position_ref == "edge:c-schema"
    assert migrate.influence_grounding.semantic_summary_snapshot == (
        "migration artifact\nschema mismatch effect"
    )
    assert cutover.seven_sections["key_facts_and_decisions"] == "错误率升高就切回蓝"
    assert migrate.seven_sections["key_facts_and_decisions"] == "先修 schema"
    assert len(_revision_files(tmp_path / "store")) == 2
    rendered = json.dumps(cutover.to_dict(), ensure_ascii=False) + json.dumps(
        migrate.to_dict(), ensure_ascii=False
    )
    assert "subtask" not in rendered
    assert "cause_set" not in rendered

    index = ViewSupersedeIndex(tmp_path / "views.json")
    index.register(revision)
    query_blocks = [_block("zzz-current", "b-query", 1, "错误率升高 schema 不匹配")]
    index.register(
        _freeze(
            "zzz-current",
            "rev-current",
            query_blocks,
            [
                RetrievalSubgraph(
                    subgraph_id="sg-query",
                    block_ids=("b-query",),
                    claim_ids=(),
                    failure_marked=True,
                    failure_evidence_refs=(_evidence("zzz-current", 1),),
                )
            ],
        )
    )
    root = tmp_path / "store"
    forks = ForkBranchAdapter(root, store)
    adapter = RetrievalSubgraphAdapter(index, forks, session_acl=_allow)
    facade = HybridRetrievalFacade(
        forks,
        InfluenceProjectionAdapter(ProjectionRegistry(tmp_path / "projections.json")),
        SegmentAtomAdapter(tmp_path / "unused-sess", "unused"),
        retrieval_subgraph=adapter,
    )
    hits = facade.retrieve_historical_subgraphs("rev-current", "sg-query", _PRINCIPAL)
    found = {item.fork_node_id for hit in hits for item in hit.forks}
    assert found == {cutover.fork_node_id, migrate.fork_node_id}
    for hit in hits:
        assert len(hit.forks) <= 1
    bundled = {tuple(item.fork_node_id for item in hit.forks) for hit in hits}
    assert (cutover.fork_node_id, migrate.fork_node_id) not in bundled
    assert (migrate.fork_node_id, cutover.fork_node_id) not in bundled


def test_unmarked_subgraph_is_not_written(tmp_path: Path) -> None:
    revision = _release_revision()
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="not failure-marked"):
        _write(store, revision, "sg-notify", position_ref="block:b-notify")
    assert _revision_files(tmp_path / "store") == []


def test_marked_subgraph_without_evidence_does_not_invent_an_anchor(tmp_path: Path) -> None:
    block = _block(_SESSION, "b-rate", 2, "错误率升高")
    revision = _freeze(
        _SESSION,
        "rev-bare",
        [block],
        [
            RetrievalSubgraph(
                subgraph_id="sg-cutover",
                block_ids=("b-rate",),
                claim_ids=(),
                failure_marked=True,
            )
        ],
    )
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="failure_evidence_refs"):
        _write(store, revision, "sg-cutover")
    assert _revision_files(tmp_path / "store") == []


def _store_tree(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {path.relative_to(root).as_posix() for path in root.rglob("*")}


def test_forged_anchor_with_caller_position_is_not_written(tmp_path: Path) -> None:
    """A caller-supplied block position must cover failure_evidence_refs[0]."""
    revision = _release_revision()
    forged_subgraphs = []
    for subgraph in revision.retrieval_subgraphs:
        if subgraph.subgraph_id != "sg-cutover":
            forged_subgraphs.append(subgraph)
            continue
        forged = replace(
            subgraph.failure_evidence_refs[0],
            ao_id="ao-forged-not-on-any-block",
        )
        forged_subgraphs.append(replace(subgraph, failure_evidence_refs=(forged,)))
    forged_revision = _freeze(
        _SESSION,
        "rev-forged-anchor",
        list(revision.blocks),
        forged_subgraphs,
        claims=list(revision.claims),
    )
    store = _store(tmp_path)
    root = tmp_path / "store"
    before = _store_tree(root)
    with pytest.raises(ValueError, match="does not cover failure evidence"):
        _write(
            store,
            forged_revision,
            "sg-cutover",
            position_kind="block",
            position_ref="block:b-rate",
        )
    assert _store_tree(root) == before
    assert _revision_files(root) == []


def test_grounding_outside_subgraph_is_rejected(tmp_path: Path) -> None:
    revision = _release_revision()
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="not inside"):
        _write(store, revision, "sg-cutover", position_ref="block:b-notify")
    with pytest.raises(ValueError, match="not inside"):
        _write(store, revision, "sg-cutover", position_ref="???")
    with pytest.raises(ValueError, match="not inside"):
        _write(store, revision, "sg-cutover", position_ref="block:")
    assert _revision_files(tmp_path / "store") == []


def test_cut_position_is_rejected(tmp_path: Path) -> None:
    revision = _release_revision()
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="not inside"):
        _write(
            store,
            revision,
            "sg-cutover",
            position_kind="cut",
            position_ref="block:b-rate",
        )
    assert _revision_files(tmp_path / "store") == []


def test_repeat_write_is_idempotent_and_keeps_one_fork(tmp_path: Path) -> None:
    revision = _release_revision()
    store = _store(tmp_path)
    kind, ref = default_failure_grounding_position(revision, "sg-cutover")
    first = _write(store, revision, "sg-cutover", position_kind=kind, position_ref=ref)
    second = _write(store, revision, "sg-cutover", position_kind=kind, position_ref=ref)
    assert second.fork_node_id == first.fork_node_id
    assert second.revision_id == first.revision_id
    loaded = store.get_fork_node(first.fork_node_id)
    assert loaded is not None
    assert len(loaded["revisions"]) == 1
    assert len(_revision_files(tmp_path / "store")) == 1

    changed = _sections("错误率升高就切回蓝")
    changed["open_issues"] = "a different open issue"
    with pytest.raises(ValueError, match="idempotency key conflict"):
        _write(
            store,
            revision,
            "sg-cutover",
            position_kind=kind,
            position_ref=ref,
            seven_sections=changed,
        )
    loaded = store.get_fork_node(first.fork_node_id)
    assert loaded is not None
    assert len(loaded["revisions"]) == 1
    assert loaded["latest_revision"]["seven_sections"]["open_issues"] == "open_issues text"
    assert len(_revision_files(tmp_path / "store")) == 1


def test_task_conclusion_does_not_add_a_fork(tmp_path: Path) -> None:
    revision = _release_revision()
    conclusion = next(block for block in revision.blocks if block.block_id == "b-release")
    assert conclusion.role == "conclusion"
    assert conclusion.summary == "发布失败"
    owed = [item.subgraph_id for item in failure_subgraphs_owed_forks(revision)]
    assert "sg-release" not in owed
    store = _store(tmp_path)
    _write(
        store,
        revision,
        "sg-cutover",
        position_kind="block",
        position_ref="block:b-rate",
    )
    _write(
        store,
        revision,
        "sg-migrate",
        position_kind="edge",
        position_ref="edge:c-schema",
        seven_sections=_sections("先修 schema"),
        diagnosis_run_id="diag-migrate",
    )
    assert len(_revision_files(tmp_path / "store")) == 2
    with pytest.raises(ValueError, match="not failure-marked"):
        _write(store, revision, "sg-release", position_ref="block:b-release")
    assert len(_revision_files(tmp_path / "store")) == 2


def test_serialized_fork_omits_subtask(tmp_path: Path) -> None:
    revision = _release_revision()
    store = _store(tmp_path)
    written = _write(
        store,
        revision,
        "sg-cutover",
        position_kind="block",
        position_ref="block:b-rate",
    )
    text = json.dumps(written.to_dict(), ensure_ascii=False)
    assert "subtask" not in text
    disk = _revision_files(tmp_path / "store")[0].read_text(encoding="utf-8")
    assert "subtask" not in disk


def test_default_position_requires_exactly_one_covering_block() -> None:
    shared = _ref(_SESSION, 2)
    first = InfluenceBlock(
        block_id="b-one",
        role="agent_action",
        participant_id="agent-a",
        segment_ref=SegmentRef(segment_id="seg-a", session_id=_SESSION),
        ao_start_seq=2,
        ao_end_seq=2,
        summary="错误率升高",
        input="",
        output="",
        authorship="agent-a",
        granularity_reason="one tool call",
        source_content_hashes=(shared.content_hash,),
        source_refs=(shared,),
    )
    second = InfluenceBlock(
        block_id="b-two",
        role="agent_action",
        participant_id="agent-a",
        segment_ref=SegmentRef(segment_id="seg-a", session_id=_SESSION),
        ao_start_seq=2,
        ao_end_seq=2,
        summary="另一块",
        input="",
        output="",
        authorship="agent-a",
        granularity_reason="one tool call",
        source_content_hashes=(shared.content_hash,),
        source_refs=(shared,),
    )
    both = _freeze(
        _SESSION,
        "rev-both",
        [first, second],
        [
            RetrievalSubgraph(
                subgraph_id="sg-cutover",
                block_ids=("b-one", "b-two"),
                claim_ids=(),
                failure_marked=True,
                failure_evidence_refs=(_evidence(_SESSION, 2),),
            )
        ],
    )
    with pytest.raises(ValueError, match="covered by 2"):
        default_failure_grounding_position(both, "sg-cutover")

    missing = _freeze(
        _SESSION,
        "rev-none",
        [_block(_SESSION, "b-rate", 2, "错误率升高")],
        [
            RetrievalSubgraph(
                subgraph_id="sg-cutover",
                block_ids=("b-rate",),
                claim_ids=(),
                failure_marked=True,
                failure_evidence_refs=(_evidence("other-sess", 2),),
            )
        ],
    )
    with pytest.raises(ValueError, match="covered by 0"):
        default_failure_grounding_position(missing, "sg-cutover")


def test_query_session_without_labels_or_visibility_is_rejected(tmp_path: Path) -> None:
    index = ViewSupersedeIndex(tmp_path / "views.json")
    current_blocks = [_block("zzz-current", "b-query", 1, "错误率升高")]
    index.register(
        _freeze(
            "zzz-current",
            "rev-current",
            current_blocks,
            [
                RetrievalSubgraph(
                    subgraph_id="sg-query",
                    block_ids=("b-query",),
                    claim_ids=(),
                    failure_marked=True,
                    failure_evidence_refs=(_evidence("zzz-current", 1),),
                )
            ],
        )
    )
    index.register(
        _freeze(
            "hist-visible",
            "rev-hist",
            [_block("hist-visible", "b-rate", 1, "错误率升高")],
            [
                RetrievalSubgraph(
                    subgraph_id="sg-cutover",
                    block_ids=("b-rate",),
                    claim_ids=(),
                    failure_marked=True,
                    failure_evidence_refs=(_evidence("hist-visible", 1),),
                )
            ],
        )
    )
    root = tmp_path / "store"
    store = CausalExperiencesStore(root, config=_CAUSAL)
    forks = ForkBranchAdapter(root, store)

    def unlabeled(session_id: str) -> tuple[str, ...]:
        if session_id == "zzz-current":
            return ()
        return _PRINCIPAL

    def hidden(session_id: str) -> tuple[str, ...]:
        if session_id == "zzz-current":
            return ("other-team",)
        return _PRINCIPAL

    unlabeled_adapter = RetrievalSubgraphAdapter(index, forks, session_acl=unlabeled)
    with pytest.raises(ValueError, match="no ACL labels"):
        unlabeled_adapter.retrieve_historical_subgraphs("rev-current", "sg-query", _PRINCIPAL)

    hidden_adapter = RetrievalSubgraphAdapter(index, forks, session_acl=hidden)
    with pytest.raises(ValueError, match="not visible to the principal"):
        hidden_adapter.retrieve_historical_subgraphs("rev-current", "sg-query", _PRINCIPAL)


def test_revision_lookup_distinguishes_missing_from_not_unique(tmp_path: Path) -> None:
    index = ViewSupersedeIndex(tmp_path / "views.json")
    for session_id in ("sess-a", "sess-b"):
        block = _block(session_id, "b-rate", 1, "错误率升高")
        index.register(
            _freeze(
                session_id,
                "rev-shared",
                [block],
                [
                    RetrievalSubgraph(
                        subgraph_id="sg-cutover",
                        block_ids=("b-rate",),
                        claim_ids=(),
                        failure_marked=True,
                        failure_evidence_refs=(_evidence(session_id, 1),),
                    )
                ],
            )
        )
    root = tmp_path / "store"
    store = CausalExperiencesStore(root, config=_CAUSAL)
    adapter = RetrievalSubgraphAdapter(index, ForkBranchAdapter(root, store), session_acl=_allow)
    with pytest.raises(ValueError, match="not unique in the supersede index") as shared:
        adapter.retrieve_historical_subgraphs("rev-shared", "sg-cutover", _PRINCIPAL)
    assert "is not in the supersede index" not in str(shared.value)

    with pytest.raises(ValueError, match="is not in the supersede index") as missing:
        adapter.retrieve_historical_subgraphs("rev-missing", "sg-cutover", _PRINCIPAL)
    assert "not unique" not in str(missing.value)


def test_workspace_outside_labels_is_not_written(tmp_path: Path) -> None:
    revision = _release_revision()
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="workspace_labels"):
        _write(
            store,
            revision,
            "sg-cutover",
            position_kind="block",
            position_ref="block:b-rate",
            workspace_labels=("other-team",),
        )
    assert _revision_files(tmp_path / "store") == []


def test_empty_or_missing_workspace_labels_are_not_written(tmp_path: Path) -> None:
    revision = _release_revision()
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="workspace_labels"):
        _write(
            store,
            revision,
            "sg-cutover",
            position_kind="block",
            position_ref="block:b-rate",
            workspace_labels=(),
        )
    with pytest.raises(ValueError, match="workspace_labels"):
        _write(
            store,
            revision,
            "sg-cutover",
            position_kind="block",
            position_ref="block:b-rate",
            workspace_labels=None,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        write_failure_subgraph_fork(
            store,
            revision,
            "sg-cutover",
            seven_sections=_sections("错误率升高就切回蓝"),
            branches=[
                {
                    "branch_id": "br-cutover",
                    "evidence_status": "real",
                    "ao_sequence": ["ao-step"],
                    "guidance": {"instruction": _GUIDANCE},
                }
            ],
            position_kind="block",
            position_ref="block:b-rate",
            workspace_id="team-a",
            diagnosis_run_id="diag-cutover",
            model_version="diag-model-v1",
            contemporaneous_basis=[_basis(2)],
        )
    assert _revision_files(tmp_path / "store") == []


def test_workspace_inside_labels_is_written(tmp_path: Path) -> None:
    revision = _release_revision()
    store = _store(tmp_path)
    written = _write(
        store,
        revision,
        "sg-cutover",
        position_kind="block",
        position_ref="block:b-rate",
        workspace_labels=("other-team", "team-a"),
    )
    assert written.workspace_id == "team-a"
    files = _revision_files(tmp_path / "store")
    assert len(files) == 1
    disk = files[0].read_text(encoding="utf-8")
    assert "subtask" not in disk
