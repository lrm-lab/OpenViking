# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""P04: ForkNode overlay persistence and legacy dual-read.

Reader proposals are validated and stored as overlays. Legacy edge/cut forks
stay readable and are not mapped into the v2 writer.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from openviking.session.aig_native_protocol import (
    OBJECT_KIND_FORK_PROPOSAL,
    SCHEMA_VERSION_AIG_NATIVE_V2,
    ForkNodeOverlay,
    ProtocolContractError,
    base_revision_catalog_from_view,
    protocol_content_hash,
    read_fork_proposal,
)
from openviking.session.causal_experiences import (
    SEVEN_SECTION_KEYS,
    CausalExperiencesStore,
    ForkNodeRevision,
)
from openviking.session.extraction_write import NativeEdgeViewStore, freeze_native_edge_view
from openviking.session.fork_node_overlay import (
    FORK_NODE_OVERLAY_DIRNAME,
    ForkDualRead,
    ForkOverlayCommit,
    ForkOverlayError,
    LegacyForkGrounding,
    StoredForkOverlay,
    catalog_for_base_revision,
    dual_read_fork,
    fork_node_revision_id,
    fork_node_stable_id,
    fork_proposal_idempotency_key,
    guard_v2_fork_consumer,
    read_legacy_fork_grounding,
    refuse_legacy_migration,
)
from openviking.session.influence_view import (
    PURPOSE_POST_RUN_INDEX,
    AORange,
    AigNativeEdge,
    CoverageManifest,
    InfluenceBlock,
    InfluenceClaim,
    InfluenceViewDraft,
    SegmentRef,
    SourceRef,
)

_CAUSAL = {"skill_trajectory_mode": "causal"}
_FROZEN = "2026-10-10T00:00:00.000Z"
_OWNER = "user:alice"
_OTHER = "user:bob"
_V2_CONSUMERS = ("pattern", "replay", "guided_branch", "v2_writer")


def _spec_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _bound_id(kind: str, revision_id: str, anchor: str, block_hash: str) -> str:
    """Identity contract: both ids are the digest of the same context."""
    payload = {
        "anchor_block_id": anchor,
        "base_revision_block_content_hash": block_hash,
        "base_revision_id": revision_id,
        "identity_kind": kind,
        "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
    }
    return hashlib.sha256(_spec_dumps(payload).encode("utf-8")).hexdigest()


def _expected_idempotency(
    proposal_hash: str, revision_id: str, anchor: str, block_hash: str
) -> str:
    return _spec_dumps(
        {
            "anchor_block_id": anchor,
            "base_revision_block_content_hash": block_hash,
            "base_revision_id": revision_id,
            "op": "fork_node_overlay",
            "proposal_content_hash": proposal_hash,
            "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
        }
    )


def _ref(sequence: int) -> SourceRef:
    return SourceRef(
        ao_id=f"ao-{sequence}",
        sequence=sequence,
        participant_id="agent-a",
        segment_id="seg-a",
        session_id="sess-overlay",
        content_hash=f"hash-{sequence}",
    )


def _block(block_id: str, sequence: int, *, summary: str | None = None) -> InfluenceBlock:
    source = _ref(sequence)
    return InfluenceBlock(
        block_id=block_id,
        role="agent_action",
        participant_id="agent-a",
        segment_ref=SegmentRef(segment_id="seg-a", session_id="sess-overlay"),
        ao_start_seq=sequence,
        ao_end_seq=sequence,
        summary=block_id if summary is None else summary,
        input="in",
        output="out",
        authorship="agent-a",
        granularity_reason="one tool call",
        source_content_hashes=(source.content_hash,),
        source_refs=(source,),
        block_kind="tool_call",
    )


def _coverage(blocks: list[InfluenceBlock]) -> CoverageManifest:
    refs = tuple(ref for block in blocks for ref in block.source_refs)
    return CoverageManifest(
        total_ao_range=AORange(start_seq=refs[0].sequence, end_seq=refs[-1].sequence),
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
        snapshot_watermark="wm-overlay",
    )


def _native_revision(
    revision_id: str = "rev-overlay-1",
    *,
    summary: str | None = None,
    mechanism: str = "later block inherits the earlier result",
):
    blocks = [_block("b-1", 1, summary=summary), _block("b-2", 2)]
    return freeze_native_edge_view(
        session_id="sess-overlay",
        task_run_id="task-overlay",
        view_id="view-overlay",
        revision_id=revision_id,
        frozen_at=_FROZEN,
        blocks=blocks,
        edges=(
            AigNativeEdge(
                edge_id="e-1",
                source_block_id="b-1",
                target_block_id="b-2",
                relation_type="influence",
                mechanism=mechanism,
            ),
        ),
        coverage=_coverage(blocks),
    )


def _claim_revision(blocks: list[InfluenceBlock]):
    claim = InfluenceClaim(
        claim_id="claim-1",
        source_block_id=blocks[0].block_id,
        target_block_id=blocks[1].block_id,
        carried_artifact="patch",
        downstream_effect="tests passed",
        relation_type="influence",
        status="hypothesis",
    )
    draft = InfluenceViewDraft(
        view_id="view-claim",
        purpose=PURPOSE_POST_RUN_INDEX,
        session_id="sess-overlay",
        task_run_id="task-overlay",
        coverage=_coverage(blocks),
        blocks=blocks,
        claims=(claim,),
    )
    return draft.freeze(revision_id="rev-claim-blocks", frozen_at=_FROZEN)


def _proposal_body(catalog, *, document_hash: str, **overrides: object) -> dict:
    body: dict = {
        "anchor_block_id": catalog.block_ids[-1],
        "author_role": "reader",
        "base_revision_block_content_hash": catalog.block_content_hash,
        "base_revision_id": catalog.base_revision_id,
        "branch_steps": [
            {"block_id": catalog.block_ids[0], "step_kind": "factual_block_ref"},
            {"imagined_node_id": "img-retry", "step_kind": "imagined_node"},
        ],
        "report_hash": document_hash,
    }
    body.update(overrides)
    return body


def _proposal(catalog, *, document_hash: str, **overrides: object) -> dict:
    body = _proposal_body(catalog, document_hash=document_hash, **overrides)
    return {
        "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
        "object_kind": OBJECT_KIND_FORK_PROPOSAL,
        "body": body,
        "content_hash": protocol_content_hash(
            SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_FORK_PROPOSAL, body
        ),
    }


def _store(tmp_path: Path):
    from openviking.session.fork_node_overlay import ForkNodeOverlayStore

    return ForkNodeOverlayStore(tmp_path)


def _index_lines(tmp_path: Path) -> list[dict]:
    path = tmp_path / FORK_NODE_OVERLAY_DIRNAME / "index.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _overlay_files(tmp_path: Path) -> list[Path]:
    directory = tmp_path / FORK_NODE_OVERLAY_DIRNAME / "overlays"
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def _legacy_payload(fork_node_id: str, *, position_kind: str, position_ref: str) -> dict:
    sections = {key: f"{key} body" for key in SEVEN_SECTION_KEYS}
    return {
        "fork_node_id": fork_node_id,
        "revision_id": fork_node_id + "-rev",
        "workspace_id": "ws-legacy",
        "anchor": {
            "anchor_kind": "ao",
            "ao_id": "ao-" + fork_node_id,
            "source_session_id": "sess-legacy",
            "anchor_sequence": 10,
        },
        "seven_sections": sections,
        "contemporaneous_basis": [
            {
                "ao_id": "ao-basis",
                "source_session_id": "sess-legacy",
                "source_archive_id": "arch-1",
                "archive_commit_watermark": "commit-1",
                "source_sequence": 4,
                "source_read_snapshot_watermark": "read-wm-1",
                "evidence_role": "contemporaneous_basis",
                "evidence_content_hash": "hash-basis",
                "captured_state": "committed",
            }
        ],
        "hindsight_attribution": [],
        "used_skills": [],
        "branches": [],
        "diagnosis_run_id": "diag-" + fork_node_id,
        "created_at": _FROZEN,
        "model_version": "diag-model-v1",
        "status": "provisional",
        "influence_grounding": {
            "view_revision_id": "rev-legacy-1",
            "position_kind": position_kind,
            "position_ref": position_ref,
            "semantic_summary_snapshot": f"legacy {position_kind} snapshot",
            "content_hash": "legacy-view-hash-" + position_kind,
            "grounded_at": _FROZEN,
        },
    }


def test_identity_binds_revision_and_anchor_block_context() -> None:
    revision = _native_revision()
    catalog = catalog_for_base_revision(revision)
    stable = fork_node_stable_id(
        catalog.base_revision_id, "b-2", catalog.block_content_hash
    )
    revision_identity = fork_node_revision_id(
        catalog.base_revision_id, "b-2", catalog.block_content_hash
    )
    assert stable == _bound_id(
        "fork_node_stable", catalog.base_revision_id, "b-2", catalog.block_content_hash
    )
    assert revision_identity == _bound_id(
        "fork_node_revision", catalog.base_revision_id, "b-2", catalog.block_content_hash
    )
    assert stable != revision_identity
    changed_blocks = _native_revision("rev-overlay-1", summary="anchor text changed")
    changed = catalog_for_base_revision(changed_blocks)
    assert changed.block_content_hash != catalog.block_content_hash
    assert fork_node_stable_id(
        changed.base_revision_id, "b-2", changed.block_content_hash
    ) != stable
    other_anchor = fork_node_stable_id(
        catalog.base_revision_id, "b-1", catalog.block_content_hash
    )
    assert other_anchor != stable
    other_revision = fork_node_stable_id(
        "rev-other", "b-2", catalog.block_content_hash
    )
    assert other_revision != stable


def test_native_catalog_uses_block_content_hash_and_ignores_edges() -> None:
    first = _native_revision(mechanism="later block inherits the earlier result")
    second = _native_revision(mechanism="a different mechanism note")
    left = catalog_for_base_revision(first)
    right = catalog_for_base_revision(second)
    assert left.block_content_hash == right.block_content_hash
    assert left.block_ids == ("b-1", "b-2")
    blocks = [_block("b-1", 1), _block("b-2", 2)]
    claim_view = _claim_revision(blocks)
    claim_catalog = base_revision_catalog_from_view(claim_view)
    native_catalog = catalog_for_base_revision(_native_revision("rev-claim-blocks"))
    assert native_catalog.block_content_hash == claim_catalog.block_content_hash
    assert "claim" not in native_catalog.block_content_hash


def test_submit_persists_overlay_without_rewriting_base_or_creating_claims(tmp_path: Path) -> None:
    revision = _native_revision()
    native_path = tmp_path / "native-views.json"
    NativeEdgeViewStore(native_path).write(revision)
    before = native_path.read_bytes()
    block_ids = tuple(block.block_id for block in revision.blocks)
    store = _store(tmp_path)
    catalog = store.register_base_revision(revision, acl_labels=(_OWNER,))
    payload = _proposal(catalog, document_hash=revision.content_hash)
    commit = store.submit_proposal(payload, principal_labels=(_OWNER,))
    assert isinstance(commit, ForkOverlayCommit)
    assert commit.deduplicated is False
    stored = commit.stored
    assert isinstance(stored, StoredForkOverlay)
    assert isinstance(stored.overlay, ForkNodeOverlay)
    assert stored.fork_node_id == _bound_id(
        "fork_node_stable",
        catalog.base_revision_id,
        catalog.block_ids[-1],
        catalog.block_content_hash,
    )
    assert stored.revision_id == _bound_id(
        "fork_node_revision",
        catalog.base_revision_id,
        catalog.block_ids[-1],
        catalog.block_content_hash,
    )
    assert stored.overlay.fork_node_id == stored.fork_node_id
    assert stored.overlay.anchor_block_id == "b-2"
    assert stored.overlay.base_revision_id == revision.revision_id
    assert stored.overlay.base_revision_block_content_hash == catalog.block_content_hash
    assert stored.idempotency_key == _expected_idempotency(
        payload["content_hash"],
        catalog.base_revision_id,
        "b-2",
        catalog.block_content_hash,
    )
    assert stored.idempotency_key == fork_proposal_idempotency_key(
        read_fork_proposal(payload, catalog)
    )
    imagined = stored.overlay.branch_steps[1]
    assert imagined.step_kind == "imagined_node"
    assert imagined.imagined_node_id == "img-retry"
    assert imagined.imagined_node_id not in catalog.block_ids
    assert tuple(block.block_id for block in revision.blocks) == block_ids
    assert native_path.read_bytes() == before
    reloaded = NativeEdgeViewStore(native_path).read(revision.revision_id)
    assert [block.block_id for block in reloaded.blocks] == ["b-1", "b-2"]
    assert not hasattr(reloaded, "claims")
    document = json.loads(_overlay_files(tmp_path)[0].read_text(encoding="utf-8"))
    body = document["document"]["body"]
    assert document["document"]["object_kind"] == "fork_node_overlay"
    assert "influence_grounding" not in json.dumps(document)
    assert "claim_id" not in json.dumps(document)
    assert "position_kind" not in json.dumps(document)
    assert "img-retry" not in catalog.block_ids
    assert body["fork_node_id"] == stored.fork_node_id
    catalog_file = next((tmp_path / FORK_NODE_OVERLAY_DIRNAME / "base-revisions").glob("*.json"))
    recorded = json.loads(catalog_file.read_text(encoding="utf-8"))
    assert set(recorded) == {
        "acl_labels",
        "base_revision_id",
        "block_content_hash",
        "block_ids",
        "document_content_hash",
    }
    assert recorded["document_content_hash"] == revision.content_hash
    assert recorded["block_ids"] == ["b-1", "b-2"]
    assert not (tmp_path / "causal-experiences").exists()
    source = inspect.getsource(__import__("openviking.session.fork_node_overlay", fromlist=["x"]))
    assert "InfluenceClaim(" not in source
    assert "InfluenceGrounding(" not in source
    assert "upsert_fork_node" not in source


def test_same_proposal_is_idempotent_and_a_different_one_conflicts(tmp_path: Path) -> None:
    revision = _native_revision()
    store = _store(tmp_path)
    catalog = store.register_base_revision(revision, acl_labels=(_OWNER,))
    payload = _proposal(catalog, document_hash=revision.content_hash)
    first = store.submit_proposal(payload, principal_labels=(_OWNER, _OTHER))
    second = store.submit_proposal(payload, principal_labels=(_OWNER,))
    assert second.deduplicated is True
    assert second.stored == first.stored
    assert len(_overlay_files(tmp_path)) == 1
    overlay_rows = [row for row in _index_lines(tmp_path) if row["record_kind"] == "fork_node_overlay"]
    assert len(overlay_rows) == 1
    other = _proposal(
        catalog,
        document_hash=revision.content_hash,
        branch_steps=[
            {"block_id": catalog.block_ids[0], "step_kind": "factual_block_ref"},
            {"imagined_node_id": "img-other", "step_kind": "imagined_node"},
        ],
    )
    with pytest.raises(ForkOverlayError) as conflict:
        store.submit_proposal(other, principal_labels=(_OWNER,))
    assert conflict.value.code == "overlay_conflict"
    assert len(_overlay_files(tmp_path)) == 1
    assert json.loads(_overlay_files(tmp_path)[0].read_text(encoding="utf-8"))[
        "proposal_content_hash"
    ] == payload["content_hash"]
    reopened = _store(tmp_path)
    loaded = reopened.read_overlay(
        catalog.base_revision_id, "b-2", principal_labels=(_OWNER,)
    )
    assert loaded == first.stored
    other_anchor = _proposal(catalog, document_hash=revision.content_hash, anchor_block_id="b-1")
    other_commit = store.submit_proposal(other_anchor, principal_labels=(_OWNER,))
    assert other_commit.stored.fork_node_id != first.stored.fork_node_id
    assert len(_overlay_files(tmp_path)) == 2


def test_server_revalidates_acl_existence_membership_hash_and_branch(tmp_path: Path) -> None:
    revision = _native_revision()
    store = _store(tmp_path)
    catalog = store.register_base_revision(revision, acl_labels=(_OWNER,))
    payload = _proposal(catalog, document_hash=revision.content_hash)

    with pytest.raises(ForkOverlayError) as denied:
        store.submit_proposal(payload, principal_labels=(_OTHER,))
    assert denied.value.code == "acl_denied"
    with pytest.raises(ForkOverlayError) as empty:
        store.submit_proposal(payload, principal_labels=())
    assert empty.value.code == "acl_denied"
    with pytest.raises(ForkOverlayError) as principal_type:
        store.submit_proposal(payload, principal_labels=_OWNER)  # type: ignore[arg-type]
    assert principal_type.value.code == "acl_denied"
    assert _overlay_files(tmp_path) == []

    missing = _proposal(catalog, document_hash=revision.content_hash, base_revision_id="rev-missing")
    with pytest.raises(ForkOverlayError) as absent:
        store.submit_proposal(missing, principal_labels=(_OWNER,))
    assert absent.value.code == "base_revision_missing"

    bad_anchor = _proposal(catalog, document_hash=revision.content_hash, anchor_block_id="blk-missing")
    with pytest.raises(ProtocolContractError, match="is not a Block"):
        store.submit_proposal(bad_anchor, principal_labels=(_OWNER,))
    bad_hash = _proposal(catalog, document_hash=revision.content_hash, base_revision_block_content_hash="a" * 64)
    with pytest.raises(ProtocolContractError, match="block_content_hash does not match"):
        store.submit_proposal(bad_hash, principal_labels=(_OWNER,))
    foreign_step = _proposal(
        catalog,
        document_hash=revision.content_hash,
        branch_steps=[
            {"block_id": "blk-from-other-revision", "step_kind": "factual_block_ref"}
        ],
    )
    with pytest.raises(ProtocolContractError, match="factual_block_ref"):
        store.submit_proposal(foreign_step, principal_labels=(_OWNER,))
    collided = _proposal(
        catalog,
        document_hash=revision.content_hash,
        branch_steps=[{"imagined_node_id": "b-1", "step_kind": "imagined_node"}],
    )
    with pytest.raises(ProtocolContractError, match="not branch-local"):
        store.submit_proposal(collided, principal_labels=(_OWNER,))
    builder = _proposal(catalog, document_hash=revision.content_hash, author_role="builder")
    with pytest.raises(ProtocolContractError, match="author_role must be reader"):
        store.submit_proposal(builder, principal_labels=(_OWNER,))
    claim_body = _proposal_body(catalog, document_hash=revision.content_hash)
    claim_body["cut"] = "cut:claim-1"
    claim_payload = {
        "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
        "object_kind": OBJECT_KIND_FORK_PROPOSAL,
        "body": claim_body,
        "content_hash": protocol_content_hash(
            SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_FORK_PROPOSAL, claim_body
        ),
    }
    with pytest.raises(ProtocolContractError, match="rejects Claim/Cut field 'cut'"):
        store.submit_proposal(claim_payload, principal_labels=(_OWNER,))
    assert _overlay_files(tmp_path) == []

    with pytest.raises(ForkOverlayError) as labels:
        store.register_base_revision(revision, acl_labels=())
    assert labels.value.code == "acl_labels_required"
    with pytest.raises(ForkOverlayError) as relabel:
        store.register_base_revision(revision, acl_labels=(_OTHER,))
    assert relabel.value.code == "acl_conflict"
    rewritten = _native_revision("rev-overlay-1", summary="rewritten block")
    with pytest.raises(ForkOverlayError) as hash_conflict:
        store.register_base_revision(rewritten, acl_labels=(_OWNER,))
    assert hash_conflict.value.code == "base_revision_conflict"
    with pytest.raises(ForkOverlayError) as not_revision:
        store.register_base_revision({"base_revision_id": "rev-overlay-1"}, acl_labels=(_OWNER,))  # type: ignore[arg-type]
    assert not_revision.value.code == "revision_required"
    again = store.register_base_revision(revision, acl_labels=(_OWNER,))
    assert again == catalog
    assert len([row for row in _index_lines(tmp_path) if row["record_kind"] == "base_revision"]) == 1


def test_legacy_edge_and_cut_forks_stay_read_only(tmp_path: Path) -> None:
    legacy = CausalExperiencesStore(tmp_path / "legacy", config=_CAUSAL)
    cut_id = "legacy-cut-fork"
    edge_id = "legacy-edge-fork"
    legacy.upsert_fork_node(
        _legacy_payload(cut_id, position_kind="cut", position_ref="cut:claim-9"),
        idempotency_key="legacy-cut-once",
    )
    legacy.upsert_fork_node(
        _legacy_payload(edge_id, position_kind="edge", position_ref="edge:claim-9"),
        idempotency_key="legacy-edge-once",
    )
    overlay = _store(tmp_path / "overlay")
    principal = ("ws-legacy",)
    cut = read_legacy_fork_grounding(legacy, cut_id, principal_labels=principal)
    edge = read_legacy_fork_grounding(legacy, edge_id, principal_labels=principal)
    assert isinstance(cut, LegacyForkGrounding)
    assert cut.position_kind == "cut"
    assert cut.position_ref == "cut:claim-9"
    assert cut.content_hash == "legacy-view-hash-cut"
    assert cut.semantic_summary_snapshot == "legacy cut snapshot"
    assert edge.position_kind == "edge"
    assert edge.position_ref == "edge:claim-9"
    assert not (tmp_path / "overlay" / FORK_NODE_OVERLAY_DIRNAME).exists()

    with pytest.raises(ForkOverlayError) as hidden:
        read_legacy_fork_grounding(legacy, cut_id, principal_labels=(_OTHER,))
    assert hidden.value.code == "acl_denied"

    dual = dual_read_fork(
        cut_id,
        overlay_store=overlay,
        legacy_store=legacy,
        principal_labels=principal,
    )
    assert isinstance(dual, ForkDualRead)
    assert dual.kind == "legacy"
    assert dual.overlay is None
    assert dual.legacy == cut
    for consumer in _V2_CONSUMERS:
        with pytest.raises(ForkOverlayError) as excluded:
            guard_v2_fork_consumer(dual.legacy, consumer)
        assert excluded.value.code == "legacy_fork_excluded"
    with pytest.raises(ForkOverlayError) as writer:
        overlay.submit_proposal(dual.legacy, principal_labels=principal)  # type: ignore[arg-type]
    assert writer.value.code == "legacy_fork_excluded"
    with pytest.raises(ForkOverlayError) as migrated:
        refuse_legacy_migration(cut)
    assert migrated.value.code == "legacy_migration_forbidden"
    raw_revision = legacy.get_fork_node(cut_id)
    assert raw_revision is not None
    with pytest.raises(ForkOverlayError) as old_object:
        overlay.submit_proposal(
            ForkNodeRevision.from_dict(raw_revision["latest_revision"]),
            principal_labels=principal,
        )
    assert old_object.value.code == "legacy_fork_excluded"
    assert _overlay_files(tmp_path / "overlay") == []

    revision = _native_revision()
    catalog = overlay.register_base_revision(revision, acl_labels=principal)
    commit = overlay.submit_proposal(_proposal(catalog, document_hash=revision.content_hash), principal_labels=principal)
    for consumer in _V2_CONSUMERS:
        assert guard_v2_fork_consumer(commit.stored, consumer) is None
        assert guard_v2_fork_consumer(commit.stored.overlay, consumer) is None
    overlay_dual = dual_read_fork(
        commit.stored.fork_node_id,
        overlay_store=overlay,
        legacy_store=legacy,
        principal_labels=principal,
    )
    assert overlay_dual is not None
    assert overlay_dual.kind == "overlay"
    assert overlay_dual.legacy is None
    assert overlay_dual.overlay == commit.stored


def test_report_hash_binds_the_registered_document_and_authorize_gates_write(
    tmp_path: Path,
) -> None:
    native = _native_revision()
    store = _store(tmp_path)
    catalog = store.register_base_revision(native, acl_labels=(_OWNER,))
    unbound = _proposal(catalog, document_hash=native.content_hash, report_hash="report-hash-1")
    with pytest.raises(ForkOverlayError) as mismatch:
        store.submit_proposal(unbound, principal_labels=(_OWNER,))
    assert mismatch.value.code == "report_hash_mismatch"
    assert _overlay_files(tmp_path) == []

    blocks = [_block("b-1", 1), _block("b-2", 2)]
    claim = _claim_revision(blocks)
    claim_store = _store(tmp_path / "claim-overlay")
    claim_catalog = claim_store.register_base_revision(claim, acl_labels=(_OWNER,))
    claim_commit = claim_store.submit_proposal(
        _proposal(claim_catalog, document_hash=claim.content_hash),
        principal_labels=(_OWNER,),
    )
    assert claim_commit.stored.overlay.base_revision_id == claim.revision_id

    def deny(session_id: str) -> str:
        raise RuntimeError(f"denied {session_id}")

    with pytest.raises(RuntimeError, match="denied sess-reader"):
        store.submit_proposal(
            _proposal(catalog, document_hash=native.content_hash),
            principal_labels=(_OWNER,),
            authorize=deny,
            session_id="sess-reader",
        )
    assert _overlay_files(tmp_path) == []

    def wrong(_session_id: str) -> str:
        return "f" * 64

    with pytest.raises(ForkOverlayError) as authorized_mismatch:
        store.submit_proposal(
            _proposal(catalog, document_hash=native.content_hash),
            principal_labels=(_OWNER,),
            authorize=wrong,
            session_id="sess-reader",
        )
    assert authorized_mismatch.value.code == "report_hash_mismatch"

    with pytest.raises(ForkOverlayError) as missing_session:
        store.submit_proposal(
            _proposal(catalog, document_hash=native.content_hash),
            principal_labels=(_OWNER,),
            authorize=lambda _session_id: native.content_hash,
        )
    assert missing_session.value.code == "session_required"
    assert _overlay_files(tmp_path) == []

    seen: list[str] = []

    def allow(session_id: str) -> str:
        seen.append(session_id)
        return native.content_hash

    commit = store.submit_proposal(
        _proposal(catalog, document_hash=native.content_hash),
        principal_labels=(_OWNER,),
        authorize=allow,
        session_id="sess-reader",
    )
    assert seen == ["sess-reader"]
    assert commit.deduplicated is False
