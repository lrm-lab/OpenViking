# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""P11: active retrieval, Claim/Cut write contraction, and explicit migration."""

from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path

import pytest

from openviking.session.active_retrieval import (
    LEGACY_ACTIVITY_KINDS,
    ActiveRetrievalCorpus,
    BoundNativeView,
)
from openviking.session.aig_native_protocol import (
    SCHEMA_VERSION_AIG_NATIVE_V2,
    GuidancePublication,
    read_legacy_claim_cut,
)
from openviking.session.causal_experiences import ApplicabilityCheck, CausalExperiencesStore
from openviking.session.extraction_write import (
    NativeEdgeViewStore,
    NativeEdgeWriteError,
    NativeExtractionPayload,
    audit_legacy_claim_view,
    freeze_extracted_view,
    freeze_native_edge_view,
)
from openviking.session.fork_node_overlay import ForkNodeOverlayStore, catalog_for_base_revision
from openviking.session.influence_projection import ProjectionRegistry
from openviking.session.influence_view import AigNativeEdge, InfluenceClaim
from openviking.session.legacy_fork_migration import (
    LegacyMigrationError,
    migrate_legacy_fork_to_overlay,
)
from openviking.session.legacy_write_gate import LegacyWriteRefused
from openviking.session.memory_role_retrieval import (
    MEMORY_ROLE,
    BoundContext,
    RoleRetrievalService,
    public_card_from_retrieval,
)
from openviking.session.retrieval_facade import (
    ForkBranchAdapter,
    HybridRetrievalFacade,
    InfluenceProjectionAdapter,
    SegmentAtomAdapter,
)
from openviking.session.strategy_pattern import StrategyPatternStore
from openviking.session.trajectory_index_hook import maybe_index_session_trajectory
from tests.test_aig_native_protocol import LEGACY_CONTENT_HASH
from tests.test_extraction_write import (
    HOOK_SESSION,
    HOOK_TASK,
    _bound_session,
    _causal_config,
    _freeze_kwargs,
    _happy_parts,
    _seed_ledger,
)
from tests.test_fork_node_overlay import _CAUSAL, _legacy_payload, _native_revision, _proposal
from tests.test_influence_projection import _revision_with_claim
from tests.test_strategy_pattern import _proposal_payload

_ROOT = Path(__file__).resolve().parents[1]
_SESSION = _ROOT / "openviking" / "session"
_PRINCIPAL = ("user:alice",)
_V2_MODULES = (
    "active_retrieval.py",
    "legacy_fork_migration.py",
    "failure_gate.py",
    "fork_precedent.py",
    "strategy_pattern.py",
    "guidance_approval.py",
    "remediation_dispatch.py",
    "replay_alignment.py",
    "outcome_contract.py",
    "coordinator_boundary.py",
    "fork_node_overlay.py",
    "append_only.py",
)
_FORBIDDEN = {
    "write_failure_subgraph_fork",
    "write_diagnosis_fork",
    "extraction_index_builder",
    "freeze_extracted_view",
    "InfluenceClaim",
}


def _applicability() -> ApplicabilityCheck:
    return ApplicabilityCheck(
        task_decision_type="applicable",
        anchor_state="applicable",
        failure_mode="applicable",
        action_skill_role="applicable",
    )


def _guidance() -> GuidancePublication:
    return GuidancePublication(
        schema_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        object_kind="guidance_publication",
        content_hash="publication-hash",
        approval_content_hash="a" * 64,
        guidance_content_hash="b" * 64,
        guidance_id="guide-narrow-retry",
        selected_agent_id="agent-a",
        target_role="task_agent",
        capability_limits=("replay-sandbox",),
        report_hash="c" * 64,
        source_fork_id="fork-source",
        base_revision_id="rev-overlay-1",
        steps=(),
        evidence_refs=(),
        policy_version="outcome-evidence-v1",
        decision="confirmed_failure",
        reason="retry with a narrower timeout",
        outcome_evidence=(),
        risk="low",
        applicability=_applicability(),
        publisher_labels=_PRINCIPAL,
        published_at="2026-10-10T00:00:00.000Z",
    )


def _facade(
    store_root: Path,
    *,
    store: CausalExperiencesStore,
    registry: ProjectionRegistry,
    views: list,
    corpus: ActiveRetrievalCorpus | None,
) -> HybridRetrievalFacade:
    return HybridRetrievalFacade(
        ForkBranchAdapter(store_root, store, session_dir=store_root, session_id="sess-1"),
        InfluenceProjectionAdapter(registry, views=views),
        SegmentAtomAdapter(store_root, "sess-1"),
        active_corpus=corpus,
    )


def test_active_retrieve_excludes_legacy_objects_and_ranks_new_projections(tmp_path: Path) -> None:
    revision = _native_revision()
    overlays = ForkNodeOverlayStore(tmp_path / "overlay")
    catalog = overlays.register_base_revision(revision, acl_labels=_PRINCIPAL)
    proposal = _proposal(catalog, document_hash=revision.content_hash)
    stored = overlays.submit_proposal(proposal, principal_labels=_PRINCIPAL).stored
    patterns = StrategyPatternStore(tmp_path / "overlay")
    pattern = patterns.accept_proposal(
        stored, catalog_for_base_revision(revision), _proposal_payload(stored.overlay, "img-retry")
    )
    claim_view = _revision_with_claim()
    registry = ProjectionRegistry(tmp_path / "cards.json")
    registry.register(claim_view, acl_labels=_PRINCIPAL)
    legacy_root = tmp_path / "legacy"
    legacy = CausalExperiencesStore(legacy_root, config=_CAUSAL)
    legacy_body = _legacy_payload(
        "legacy-cut-fork", position_kind="cut", position_ref="cut:claim-9"
    )
    legacy_body["workspace_id"] = "user:alice"
    legacy.upsert_fork_node(legacy_body, idempotency_key="legacy-once")
    facade = _facade(
        legacy_root,
        store=legacy,
        registry=registry,
        views=[claim_view],
        corpus=ActiveRetrievalCorpus(
            native_views=(BoundNativeView(revision, _PRINCIPAL),),
            overlay_store=overlays,
            pattern_store=patterns,
            guidance_publications=(_guidance(),),
        ),
    )
    active = facade.retrieve({}, _PRINCIPAL)
    kinds = {card.kind for card in active.cards}
    assert kinds.isdisjoint(LEGACY_ACTIVITY_KINDS)
    assert {"native_block", "native_edge", "fork_overlay", "strategy_pattern", "guidance"} <= kinds
    assert pattern.pattern_id in {card.source_pointer.get("pattern_id") for card in active.cards}
    for card in active.cards:
        assert "claim_id" not in card.source_pointer
        assert card.source_pointer.get("position_kind") != "cut"
    again = facade.retrieve({}, _PRINCIPAL)
    assert [card.payload_ref for card in again.cards] == [card.payload_ref for card in active.cards]
    audit = facade.audit_legacy({}, _PRINCIPAL)
    audit_kinds = {card.kind for card in audit.cards}
    assert "claim_projection" in audit_kinds
    assert "fork" in audit_kinds
    source = inspect.getsource(HybridRetrievalFacade.retrieve)
    assert "self.projection" not in source
    assert "self.fork_branch" not in source
    assert "expand_claim_neighborhood" not in source

    def authority(_role: str) -> BoundContext:
        return BoundContext("scope", "snap", "lease")

    envelope = RoleRetrievalService().retrieve(
        role=MEMORY_ROLE,
        request={"intentText": "retry", "operationId": "op-p11"},
        authority=authority,
        discover=lambda _intent, _pattern: [
            public_card_from_retrieval(card) for card in active.cards
        ],
    )
    prompted = {
        card["kind"] for card in envelope["memoryCandidateCards"] + envelope["analogicalPriorCards"]
    }
    assert prompted.isdisjoint(LEGACY_ACTIVITY_KINDS)
    assert "native_block" in prompted
    assert "guidance" in prompted


def test_commit_native_provider_writes_edges_and_claim_provider_writes_nothing(
    tmp_path: Path,
) -> None:
    from tests.test_extraction_write import _block, _coverage

    blocks = [
        _block("b-1", [1], session_id=HOOK_SESSION),
        _block("b-2", [2], session_id=HOOK_SESSION),
    ]
    payload = NativeExtractionPayload(
        blocks,
        (
            AigNativeEdge(
                edge_id="e-1",
                source_block_id="b-1",
                target_block_id="b-2",
                relation_type="influence",
                mechanism="later block inherits the earlier result",
            ),
        ),
        _coverage(blocks),
    )

    def _native(session_id: str, task_run_id: str, outcome_status: str) -> NativeExtractionPayload:
        del session_id, task_run_id, outcome_status
        return payload

    _seed_ledger(tmp_path, HOOK_SESSION)
    session = _bound_session(tmp_path, HOOK_SESSION)
    config = _causal_config()
    maybe_index_session_trajectory(session, HOOK_TASK, config=config, extraction_provider=_native)
    maybe_index_session_trajectory(session, HOOK_TASK, config=config, extraction_provider=_native)
    stored = NativeEdgeViewStore(tmp_path / "native-edges.json")
    revisions = stored.revisions()
    assert len(revisions) == 1
    assert revisions[0].revision_id == f"post-run:{HOOK_SESSION}:{HOOK_TASK}:native"
    text = (tmp_path / "native-edges.json").read_text(encoding="utf-8")
    assert "claim_id" not in text
    assert '"position_kind": "cut"' not in text
    assert not (tmp_path / "views.json").exists()

    def _claims(session_id: str, task_run_id: str, outcome_status: str):
        del task_run_id, outcome_status
        from tests.test_extraction_write import _payload_for

        return _payload_for(session_id)

    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    _seed_ledger(claim_root, HOOK_SESSION)
    maybe_index_session_trajectory(
        _bound_session(claim_root, HOOK_SESSION),
        HOOK_TASK,
        config=config,
        extraction_provider=_claims,
    )
    assert not (claim_root / "native-edges.json").exists()
    assert not (claim_root / "views.json").exists()


def test_explicit_migration_records_source_and_does_not_batch_rewrite(tmp_path: Path) -> None:
    revision = _native_revision()
    overlays = ForkNodeOverlayStore(tmp_path / "overlay")
    catalog = overlays.register_base_revision(revision, acl_labels=("ws-legacy",))
    proposal = _proposal(catalog, document_hash=revision.content_hash)
    legacy = CausalExperiencesStore(tmp_path / "legacy", config=_CAUSAL)
    cut_id = "legacy-cut-fork"
    edge_id = "legacy-edge-fork"
    legacy.upsert_fork_node(
        _legacy_payload(cut_id, position_kind="cut", position_ref="cut:claim-9"),
        idempotency_key="cut-once",
    )
    legacy.upsert_fork_node(
        _legacy_payload(edge_id, position_kind="edge", position_ref="edge:claim-9"),
        idempotency_key="edge-once",
    )
    before_cut = copy.deepcopy(legacy.get_fork_node(cut_id))
    before_edge = copy.deepcopy(legacy.get_fork_node(edge_id))
    fixture = _ROOT / "tests" / "fixtures" / "adr0043" / "legacy_claim_cut.json"
    fixture_bytes = fixture.read_bytes()
    first = migrate_legacy_fork_to_overlay(
        legacy_store=legacy,
        overlay_store=overlays,
        fork_node_id=cut_id,
        migration_reason="reader supplied a block anchor for audit",
        proposal=proposal,
        principal_labels=("ws-legacy",),
    )
    assert first.deduplicated is False
    assert first.migration.schema_version == SCHEMA_VERSION_AIG_NATIVE_V2
    assert first.migration.legacy_fork_id == cut_id
    assert first.migration.migration_reason == "reader supplied a block anchor for audit"
    assert first.migration.grounding_snapshot["position_kind"] == "cut"
    assert first.migration.grounding_snapshot["position_ref"] == "cut:claim-9"
    assert first.migration.grounding_snapshot["content_hash"] == "legacy-view-hash-cut"
    assert first.overlay.overlay.anchor_block_id == catalog.block_ids[-1]
    assert first.overlay.fork_node_id != cut_id
    overlay_dir = tmp_path / "overlay" / "fork-node-overlays" / "overlays"
    overlay_files = sorted(overlay_dir.glob("*.json"))
    overlay_text = "".join(path.read_text(encoding="utf-8") for path in overlay_files)
    assert len(overlay_files) == 1
    assert "claim_id" not in overlay_text
    second = migrate_legacy_fork_to_overlay(
        legacy_store=legacy,
        overlay_store=overlays,
        fork_node_id=cut_id,
        migration_reason="reader supplied a block anchor for audit",
        proposal=proposal,
        principal_labels=("ws-legacy",),
    )
    assert second.deduplicated is True
    assert second.migration.content_hash == first.migration.content_hash
    assert second.overlay.fork_node_id == first.overlay.fork_node_id
    with pytest.raises(LegacyMigrationError) as conflict:
        migrate_legacy_fork_to_overlay(
            legacy_store=legacy,
            overlay_store=overlays,
            fork_node_id=cut_id,
            migration_reason="a different explicit reason",
            proposal=proposal,
            principal_labels=("ws-legacy",),
        )
    assert conflict.value.code == "migration_conflict"
    assert legacy.get_fork_node(cut_id) == before_cut
    assert legacy.get_fork_node(edge_id) == before_edge
    assert list((tmp_path / "overlay" / "fork-node-overlays" / "overlays").glob("*.json"))
    migrations = list((tmp_path / "overlay" / "fork-node-overlays" / "migrations").glob("*.json"))
    assert len(migrations) == 1
    assert fixture.read_bytes() == fixture_bytes
    document = read_legacy_claim_cut(json.loads(fixture_bytes))
    assert document.content_hash == LEGACY_CONTENT_HASH
    assert document.cut_grounding.position_kind == "cut"
    blocks, claims, coverage, subgraphs = _happy_parts("sess-audit")
    audited = audit_legacy_claim_view(
        freeze_extracted_view(
            **_freeze_kwargs(blocks, claims, coverage, subgraphs, session_id="sess-audit")
        )
    )
    assert claims[0].claim_id in audited.claim_ids
    assert audited.content_hash


def test_v2_modules_do_not_depend_on_legacy_writers() -> None:
    for name in _V2_MODULES:
        tree = ast.parse((_SESSION / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported = {alias.name for alias in node.names}
                assert imported.isdisjoint(_FORBIDDEN), name
            if isinstance(node, ast.Call):
                func = node.func
                called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                assert called not in _FORBIDDEN, name
    hook = (_SESSION / "trajectory_index_hook.py").read_text(encoding="utf-8")
    session = (_SESSION / "session.py").read_text(encoding="utf-8")
    assert "extraction_index_builder" not in hook
    assert "freeze_extracted_view" not in hook
    assert "write_failure_subgraph_fork" not in hook
    assert "extraction_index_builder" not in session
    assert "write_diagnosis_fork" not in session
    package = _ROOT / "openviking"
    for path in package.rglob("*.py"):
        if path.name == "legacy_fork_migration.py":
            continue
        assert "migrate_legacy_fork_to_overlay" not in path.read_text(encoding="utf-8")


def test_native_writer_still_rejects_a_claim_object() -> None:
    revision = _native_revision()
    claim = InfluenceClaim(
        claim_id="claim-1",
        source_block_id="b-1",
        target_block_id="b-2",
        carried_artifact="patch",
        downstream_effect="tests passed",
        relation_type="influence",
    )
    with pytest.raises(NativeEdgeWriteError) as refused:
        freeze_native_edge_view(
            session_id="sess-overlay",
            task_run_id="task-overlay",
            view_id="view-overlay",
            revision_id="rev-claim-edge",
            frozen_at="2026-10-10T00:00:00.000Z",
            blocks=revision.blocks,
            edges=(claim,),  # type: ignore[arg-type]
            coverage=revision.coverage,
        )
    assert refused.value.code == "claim_mapping_forbidden"


def test_legacy_writers_default_refuse() -> None:
    with pytest.raises(LegacyWriteRefused):
        freeze_extracted_view(
            session_id="s",
            task_run_id="t",
            view_id="v",
            revision_id="r",
            frozen_at="2026-10-10T00:00:00.000Z",
            blocks=(),
            claims=(),
            coverage=None,  # type: ignore[arg-type]
        )
