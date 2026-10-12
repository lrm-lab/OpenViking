# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""P06: Fork precedent retrieval and controlled Reader input.

A Reader proposal is retrieved on the server before it can be submitted.
Results are historical precedents in three tiers. They are not current
evidence and they are not appended to the P05 ledger.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from openviking.session.aig_native_protocol import (
    ProtocolContractError,
    read_branch_evaluation,
)
from openviking.session.causal_experiences import CausalExperiencesStore
from openviking.session.extraction_write import freeze_native_edge_view
from openviking.session.fork_node_overlay import (
    ForkNodeOverlayStore,
    ForkOverlayError,
    fork_node_stable_id,
)
from openviking.session.fork_precedent import (
    FORK_PRECEDENT_DIRNAME,
    ForkPrecedentCommit,
    ForkPrecedentError,
    ForkPrecedentService,
    PrecedentRetrievalRecord,
)
from openviking.session.influence_view import AigNativeEdge
from openviking.session.strategy_pattern import STRATEGY_PATTERN_DIRNAME, StrategyPatternStore
from tests.test_fork_node_overlay import (
    _block,
    _coverage,
    _legacy_payload,
    _native_revision,
    _overlay_files,
    _proposal,
)
from tests.test_strategy_pattern import (
    _core,
    _direct_admission,
    _evaluation,
    _evaluation_files,
    _proposal_payload,
)

_READER = "sess-reader"
_PAST = "sess-past"
_OTHER = "sess-other"
_CROSS = "sess-cross"
_HIDDEN = "sess-hidden"
_CAROL = "user:carol"
_ALICE = "user:alice"
_BOB = "user:bob"
_T0 = "2026-10-10T00:00:00.000Z"
_T1 = "2026-10-10T00:00:01.000Z"
_T2 = "2026-10-10T00:00:02.000Z"
_T3 = "2026-10-10T00:00:03.000Z"
_T4 = "2026-10-10T00:00:04.000Z"
_T5 = "2026-10-10T00:00:05.000Z"
_FROZEN = "2026-10-10T00:00:00.000Z"


def _code(call) -> str:
    try:
        call()
    except ForkPrecedentError as exc:
        return exc.code
    except ForkOverlayError as exc:
        return exc.code
    raise AssertionError("expected a coded error")


def _revision(revision_id: str, block_ids: tuple[str, ...]):
    blocks = [_block(block_id, index) for index, block_id in enumerate(block_ids, start=1)]
    edges = tuple(
        AigNativeEdge(
            edge_id=f"e-{revision_id}-{index}",
            source_block_id=block_ids[index - 1],
            target_block_id=block_ids[index],
            relation_type="influence",
            mechanism="later block inherits the earlier result",
        )
        for index in range(1, len(block_ids))
    )
    return freeze_native_edge_view(
        session_id="sess-overlay",
        task_run_id="task-overlay",
        view_id=f"view-{revision_id}",
        revision_id=revision_id,
        frozen_at=_FROZEN,
        blocks=blocks,
        edges=edges,
        coverage=_coverage(blocks),
    )


def _env(tmp_path: Path):
    return (
        ForkNodeOverlayStore(tmp_path),
        StrategyPatternStore(tmp_path),
        ForkPrecedentService(tmp_path),
    )


def _query(catalog, revision, imagined: str = "img-new"):
    return _proposal(
        catalog,
        document_hash=revision.content_hash,
        anchor_block_id="b-anchor",
        branch_steps=[
            {"block_id": "b-fact", "step_kind": "factual_block_ref"},
            {"imagined_node_id": imagined, "step_kind": "imagined_node"},
        ],
    )


def _plant(
    overlays,
    patterns,
    precedents,
    revision,
    catalog,
    *,
    anchor: str,
    factual: tuple[str, ...],
    imagined: str,
    session_id: str,
    recorded_at: str,
    principal: tuple[str, ...],
    node_rows: tuple[tuple[str, str], ...] = (),
    pattern_core: dict | None = None,
    pattern_rows: tuple[tuple[str, str], ...] = (),
    observe: bool = True,
):
    steps = [
        {"block_id": block_id, "step_kind": "factual_block_ref"} for block_id in factual
    ]
    steps.append({"imagined_node_id": imagined, "step_kind": "imagined_node"})
    payload = _proposal(
        catalog,
        document_hash=revision.content_hash,
        anchor_block_id=anchor,
        branch_steps=steps,
    )
    commit = overlays.submit_proposal(payload, principal_labels=principal)
    pattern = None
    if pattern_core is None:
        patterns.freeze_branch(commit.stored, catalog)
    else:
        pattern = patterns.accept_proposal(
            commit.stored,
            catalog,
            _proposal_payload(commit.stored.overlay, imagined, pattern_core),
        )
    for index, (tier, status) in enumerate(node_rows):
        patterns.append_evaluation(
            _evaluation(
                "imagined_node",
                imagined,
                f"eval-{imagined}-n{index}",
                f"evidence:{imagined}:n{index}",
                tier,
                status,
            ),
            admission=_direct_admission(imagined) if tier == "replay-direct" else None,
        )
    if pattern is not None:
        for index, (tier, status) in enumerate(pattern_rows):
            patterns.append_evaluation(
                _evaluation(
                    "strategy_pattern",
                    pattern.pattern_id,
                    f"eval-{imagined}-p{index}",
                    f"evidence:{imagined}:p{index}",
                    tier,
                    status,
                ),
                admission=(
                    _direct_admission(imagined, outcome=True) if tier == "replay-direct" else None
                ),
            )
    if observe:
        precedents.observe_fork(
            overlays,
            patterns,
            commit.stored.fork_node_id,
            session_id=session_id,
            recorded_at=recorded_at,
            principal_labels=principal,
        )
    return commit, pattern


def _ids(hits) -> list[str]:
    return [hit.fork_node_id for hit in hits]


def _retrieval_path(root: Path, retrieval_id: str) -> Path:
    return root / FORK_PRECEDENT_DIRNAME / "retrievals" / f"{retrieval_id}.json"


def _disposition_files(root: Path) -> list[Path]:
    directory = root / FORK_PRECEDENT_DIRNAME / "dispositions"
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def _strategy_index_lines(root: Path) -> int:
    path = root / STRATEGY_PATTERN_DIRNAME / "index.jsonl"
    if not path.is_file():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line])


def test_overlay_persistence_primitive_does_not_grow_a_retrieval_parameter() -> None:
    params = inspect.signature(ForkNodeOverlayStore.submit_proposal).parameters
    assert "retrieval_id" not in params
    assert "dispositions" not in params
    retrieve = inspect.signature(ForkPrecedentService.retrieve).parameters
    for blocked in ("include_retired", "hits", "tiers", "skip_acl", "precedents"):
        assert blocked not in retrieve
    assert retrieve["temporal_cutoff"].default is inspect.Parameter.empty
    assert retrieve["principal_labels"].default is inspect.Parameter.empty
    assert retrieve["session_id"].default is inspect.Parameter.empty
    submit = inspect.signature(ForkPrecedentService.submit_proposal).parameters
    assert submit["retrieval_id"].default is inspect.Parameter.empty
    assert submit["dispositions"].default is inspect.Parameter.empty


def test_retrieve_splits_direct_exploratory_and_negative_tiers(tmp_path: Path) -> None:
    overlays, patterns, precedents = _env(tmp_path)
    revision = _revision(
        "rev-tiers",
        (
            "b-fact",
            "b-anchor",
            "b-prod",
            "b-replay",
            "b-mix",
            "b-model",
            "b-obs",
            "b-apat",
            "b-chal",
            "b-inv",
            "b-ret",
            "b-sup",
        ),
    )
    catalog = overlays.register_base_revision(revision, acl_labels=(_ALICE, _CAROL))
    principal = (_CAROL,)
    prod, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-prod",
        factual=("b-fact",),
        imagined="img-prod",
        session_id=_PAST,
        recorded_at=_T1,
        principal=principal,
        node_rows=(("replay-direct", "active"), ("replay-direct", "active")),
    )
    replay, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-replay",
        factual=("b-replay",),
        imagined="img-replay",
        session_id=_PAST,
        recorded_at=_T5,
        principal=principal,
        node_rows=(("replay-direct", "active"),),
    )
    mix, mix_pattern = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-mix",
        factual=("b-fact",),
        imagined="img-mix",
        session_id=_PAST,
        recorded_at=_T2,
        principal=principal,
        node_rows=(("analogical", "active"),),
        pattern_core=_core(intervention_type="mix-direct", target_boundary="b-anchor"),
        pattern_rows=(("replay-direct", "active"),),
    )
    model, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-model",
        factual=("b-model",),
        imagined="img-model",
        session_id=_PAST,
        recorded_at=_T4,
        principal=principal,
        node_rows=(("model-only", "active"),),
    )
    obs, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-obs",
        factual=("b-obs",),
        imagined="img-obs",
        session_id=_PAST,
        recorded_at=_T3,
        principal=principal,
        node_rows=(("observational", "active"),),
    )
    _apat, apat_pattern = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-apat",
        factual=("b-apat",),
        imagined="img-apat",
        session_id=_PAST,
        recorded_at=_T1,
        principal=principal,
        pattern_core=_core(
            intervention_type="watch-only", target_boundary="unrelated-boundary"
        ),
        pattern_rows=(("analogical", "active"),),
    )
    challenged, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-chal",
        factual=("b-fact",),
        imagined="img-chal",
        session_id=_PAST,
        recorded_at=_T1,
        principal=principal,
        node_rows=(("replay-direct", "active"), ("analogical", "challenged")),
    )
    invalidated, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-inv",
        factual=("b-fact",),
        imagined="img-inv",
        session_id=_PAST,
        recorded_at=_T1,
        principal=principal,
        node_rows=(("model-only", "invalidated"),),
    )
    retired, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-ret",
        factual=("b-fact",),
        imagined="img-ret",
        session_id=_PAST,
        recorded_at=_T1,
        principal=principal,
        node_rows=(("replay-direct", "active"), ("observational", "retired")),
    )
    superseded, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-sup",
        factual=("b-fact",),
        imagined="img-sup",
        session_id=_PAST,
        recorded_at=_T1,
        principal=principal,
        node_rows=(("observational", "superseded"),),
    )
    before_files = _evaluation_files(tmp_path)
    before_index = _strategy_index_lines(tmp_path)
    query = _query(catalog, revision)
    record = precedents.retrieve(
        overlays,
        patterns,
        query,
        principal_labels=principal,
        session_id=_READER,
        temporal_cutoff=_T5,
    )
    assert isinstance(record, PrecedentRetrievalRecord)
    again = precedents.retrieve(
        overlays,
        patterns,
        query,
        principal_labels=principal,
        session_id=_READER,
        temporal_cutoff=_T5,
    )
    assert again == record
    assert len(list((tmp_path / FORK_PRECEDENT_DIRNAME / "retrievals").glob("*.json"))) == 1
    assert _evaluation_files(tmp_path) == before_files
    assert _strategy_index_lines(tmp_path) == before_index

    assert [hit.subject_kind for hit in record.high_confidence] == [
        "strategy_pattern",
        "branch",
        "branch",
    ]
    assert [hit.subject_id for hit in record.high_confidence] == [
        mix_pattern.pattern_id,
        prod.stored.fork_node_id,
        replay.stored.fork_node_id,
    ]
    assert [hit.evidence_tier for hit in record.high_confidence] == [
        "replay-direct",
        "replay-direct",
        "replay-direct",
    ]
    assert [hit.relevance for hit in record.high_confidence] == ["high", "high", "contextual"]
    assert [hit.rank for hit in record.high_confidence] == [1, 2, 3]
    assert [hit.disposition_required for hit in record.high_confidence] == [True, True, False]
    assert record.high_confidence[0].tier == "high-confidence"
    assert record.high_confidence[1].evidence_tier == "replay-direct"
    assert all(hit.source == "historical_precedent" for hit in record.high_confidence)
    assert all(hit.record_kind == "fork_precedent" for hit in record.high_confidence)
    assert all(hit.candidate is True for hit in record.high_confidence)
    assert "estimated_confidence" not in record.high_confidence[0].__dict__

    assert [hit.subject_id for hit in record.exploratory] == [
        mix.stored.fork_node_id,
        model.stored.fork_node_id,
        obs.stored.fork_node_id,
        apat_pattern.pattern_id,
    ]
    assert [hit.evidence_tier for hit in record.exploratory] == [
        "analogical",
        "model-only",
        "observational",
        "analogical",
    ]
    assert [hit.tier for hit in record.exploratory] == ["exploratory"] * 4
    assert [hit.relevance for hit in record.exploratory] == [
        "high",
        "contextual",
        "contextual",
        "contextual",
    ]
    assert [hit.disposition_required for hit in record.exploratory] == [True, False, False, False]
    assert [hit.rank for hit in record.exploratory] == [1, 2, 3, 4]
    assert record.candidates == record.high_confidence + record.exploratory

    negative_ids = {hit.fork_node_id for hit in record.negative_retired}
    assert negative_ids == {
        challenged.stored.fork_node_id,
        invalidated.stored.fork_node_id,
        retired.stored.fork_node_id,
        superseded.stored.fork_node_id,
    }
    by_fork = {hit.fork_node_id: hit for hit in record.negative_retired}
    assert by_fork[challenged.stored.fork_node_id].status == "challenged"
    assert by_fork[challenged.stored.fork_node_id].evidence_tier == "analogical"
    assert by_fork[invalidated.stored.fork_node_id].status == "invalidated"
    assert by_fork[retired.stored.fork_node_id].status == "retired"
    assert by_fork[retired.stored.fork_node_id].evidence_tier == "observational"
    assert by_fork[superseded.stored.fork_node_id].status == "superseded"
    assert all(hit.tier == "negative/retired" for hit in record.negative_retired)
    assert all(hit.candidate is False for hit in record.negative_retired)
    assert all(hit.disposition_required is False for hit in record.negative_retired)
    assert [hit.rank for hit in record.negative_retired] == list(
        range(1, len(record.negative_retired) + 1)
    )
    candidate_forks = {hit.fork_node_id for hit in record.candidates}
    assert negative_ids.isdisjoint(candidate_forks)
    p05 = {item.subject_id for item in patterns.precedent_candidates()}
    assert {"img-chal", "img-inv", "img-sup"}.issubset(p05)
    assert "img-ret" not in p05

    stored = json.loads(_retrieval_path(tmp_path, record.retrieval_id).read_text(encoding="utf-8"))
    assert stored["object_kind"] == "fork_precedent_retrieval"
    assert stored["body"]["high_confidence"][0]["source"] == "historical_precedent"
    assert stored["body"]["high_confidence"][0]["tier"] == "high-confidence"
    with pytest.raises(ProtocolContractError):
        read_branch_evaluation(stored)
    with pytest.raises(ProtocolContractError):
        patterns.append_evaluation(stored)
    assert _evaluation_files(tmp_path) == before_files


def test_server_filters_acl_cutoff_self_and_status_before_rank(tmp_path: Path) -> None:
    overlays, patterns, precedents = _env(tmp_path)
    revision = _revision(
        "rev-filter",
        (
            "b-fact",
            "b-anchor",
            "b-new",
            "b-mid",
            "b-old",
            "b-ctx",
            "b-self",
            "b-future",
            "b-retired",
            "b-quiet",
        ),
    )
    catalog = overlays.register_base_revision(revision, acl_labels=(_ALICE, _CAROL))
    principal = (_CAROL,)
    hidden_revision = _native_revision("rev-hidden")
    hidden_catalog = overlays.register_base_revision(hidden_revision, acl_labels=(_BOB,))
    hidden, _ = _plant(
        overlays,
        patterns,
        precedents,
        hidden_revision,
        hidden_catalog,
        anchor="b-2",
        factual=("b-1",),
        imagined="img-hidden",
        session_id=_HIDDEN,
        recorded_at=_T4,
        principal=(_BOB,),
        node_rows=(("analogical", "active"),),
    )
    rel_new, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-new",
        factual=("b-fact",),
        imagined="img-new-past",
        session_id=_PAST,
        recorded_at=_T3,
        principal=principal,
        node_rows=(("analogical", "active"),),
    )
    rel_mid, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-mid",
        factual=("b-fact",),
        imagined="img-mid",
        session_id=_PAST,
        recorded_at=_T2,
        principal=principal,
        node_rows=(("analogical", "active"),),
    )
    rel_old, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-old",
        factual=("b-fact",),
        imagined="img-old",
        session_id=_PAST,
        recorded_at=_T1,
        principal=principal,
        node_rows=(("analogical", "active"),),
    )
    contextual, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-ctx",
        factual=("b-ctx",),
        imagined="img-ctx",
        session_id=_PAST,
        recorded_at=_T3,
        principal=principal,
        node_rows=(("analogical", "active"),),
    )
    self_fork, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-self",
        factual=("b-fact",),
        imagined="img-self",
        session_id=_READER,
        recorded_at=_T4,
        principal=principal,
        node_rows=(("analogical", "active"),),
    )
    future, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-future",
        factual=("b-fact",),
        imagined="img-future",
        session_id=_PAST,
        recorded_at=_T5,
        principal=principal,
        node_rows=(("analogical", "active"),),
    )
    retired, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-retired",
        factual=("b-fact",),
        imagined="img-filter-retired",
        session_id=_PAST,
        recorded_at=_T2,
        principal=principal,
        node_rows=(("analogical", "retired"),),
    )
    quiet, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-quiet",
        factual=("b-fact",),
        imagined="img-quiet",
        session_id=_PAST,
        recorded_at=_T1,
        principal=principal,
        node_rows=(("analogical", "active"),),
        observe=False,
    )
    other_revision = _native_revision("rev-cross")
    other_catalog = overlays.register_base_revision(other_revision, acl_labels=(_CAROL,))
    cross, _ = _plant(
        overlays,
        patterns,
        precedents,
        other_revision,
        other_catalog,
        anchor="b-2",
        factual=("b-1",),
        imagined="img-cross",
        session_id=_CROSS,
        recorded_at=_T0,
        principal=principal,
        node_rows=(("analogical", "active"),),
    )
    assert _code(
        lambda: precedents.observe_fork(
            overlays,
            patterns,
            hidden.stored.fork_node_id,
            session_id=_READER,
            recorded_at=_T4,
            principal_labels=principal,
        )
    ) == "acl_denied"
    query = _query(catalog, revision, imagined="img-query")
    assert _code(
        lambda: precedents.retrieve(
            overlays,
            patterns,
            query,
            principal_labels="user:carol",  # type: ignore[arg-type]
            session_id=_READER,
            temporal_cutoff=_T4,
        )
    ) == "acl_denied"
    assert not (tmp_path / FORK_PRECEDENT_DIRNAME / "retrievals").exists()
    assert _code(
        lambda: precedents.retrieve(
            overlays,
            patterns,
            query,
            principal_labels=principal,
            session_id=_READER,
            temporal_cutoff="2026-10-10",
        )
    ) == "temporal_cutoff_required"
    assert _code(
        lambda: precedents.retrieve(
            overlays,
            patterns,
            query,
            principal_labels=principal,
            session_id="",
            temporal_cutoff=_T4,
        )
    ) == "session_required"
    record = precedents.retrieve(
        overlays,
        patterns,
        query,
        principal_labels=principal,
        session_id=_READER,
        temporal_cutoff=_T4,
    )
    assert record.high_confidence == ()
    expected = [
        rel_new.stored.fork_node_id,
        rel_mid.stored.fork_node_id,
        rel_old.stored.fork_node_id,
        contextual.stored.fork_node_id,
        cross.stored.fork_node_id,
    ]
    assert _ids(record.exploratory) == expected
    assert [hit.rank for hit in record.exploratory] == [1, 2, 3, 4, 5]
    assert [hit.relevance for hit in record.exploratory] == [
        "high",
        "high",
        "high",
        "contextual",
        "contextual",
    ]
    absent = {
        hidden.stored.fork_node_id,
        self_fork.stored.fork_node_id,
        future.stored.fork_node_id,
        quiet.stored.fork_node_id,
        retired.stored.fork_node_id,
    }
    present = {hit.fork_node_id for hit in record.candidates}
    assert absent.isdisjoint(present)
    assert _ids(record.negative_retired) == [retired.stored.fork_node_id]
    assert record.negative_retired[0].candidate is False
    assert record.negative_retired[0].status == "retired"
    assert retired.stored.fork_node_id not in {hit.fork_node_id for hit in record.candidates}
    visible = overlays.visible_overlays(principal_labels=principal)
    assert hidden.stored.fork_node_id not in {item.fork_node_id for item in visible}
    assert rel_new.stored.fork_node_id in {item.fork_node_id for item in visible}
    assert _code(
        lambda: overlays.registered_catalog("rev-missing", principal_labels=principal)
    ) == "base_revision_missing"
    assert (
        overlays.registered_catalog(revision.revision_id, principal_labels=principal)
        == catalog
    )


def test_current_proposal_is_not_its_own_precedent(tmp_path: Path) -> None:
    overlays, patterns, precedents = _env(tmp_path)
    revision = _revision("rev-self", ("b-fact", "b-anchor"))
    catalog = overlays.register_base_revision(revision, acl_labels=(_CAROL,))
    payload = _query(catalog, revision)
    commit = overlays.submit_proposal(payload, principal_labels=(_CAROL,))
    assert commit.stored.fork_node_id == fork_node_stable_id(
        catalog.base_revision_id, "b-anchor", catalog.block_content_hash
    )
    patterns.freeze_branch(commit.stored, catalog)
    patterns.append_evaluation(
        _evaluation(
            "imagined_node",
            "img-new",
            "eval-self-direct",
            "evidence:self-direct",
            "replay-direct",
        ),
        admission=_direct_admission("img-new"),
    )
    precedents.observe_fork(
        overlays,
        patterns,
        commit.stored.fork_node_id,
        session_id=_OTHER,
        recorded_at=_T1,
        principal_labels=(_CAROL,),
    )
    before_files = _evaluation_files(tmp_path)
    record = precedents.retrieve(
        overlays,
        patterns,
        payload,
        principal_labels=(_CAROL,),
        session_id=_READER,
        temporal_cutoff=_T2,
    )
    assert record.high_confidence == ()
    assert record.candidates == ()
    assert all(hit.fork_node_id != commit.stored.fork_node_id for hit in record.negative_retired)
    submitted = precedents.submit_proposal(
        overlays,
        payload,
        retrieval_id=record.retrieval_id,
        dispositions=[],
        principal_labels=(_CAROL,),
        session_id=_READER,
        recorded_at=_T2,
    )
    assert isinstance(submitted, ForkPrecedentCommit)
    assert submitted.deduplicated is True
    assert submitted.stored.fork_node_id == commit.stored.fork_node_id
    assert _evaluation_files(tmp_path) == before_files


def test_later_evidence_stays_behind_the_cutoff_until_observed(tmp_path: Path) -> None:
    overlays, patterns, precedents = _env(tmp_path)
    revision = _revision("rev-time", ("b-fact", "b-anchor", "b-late"))
    catalog = overlays.register_base_revision(revision, acl_labels=(_CAROL,))
    planted, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-late",
        factual=("b-fact",),
        imagined="img-late",
        session_id=_PAST,
        recorded_at=_T1,
        principal=(_CAROL,),
        node_rows=(("analogical", "active"),),
    )
    query = _query(catalog, revision)
    early = precedents.retrieve(
        overlays,
        patterns,
        query,
        principal_labels=(_CAROL,),
        session_id=_READER,
        temporal_cutoff=_T2,
    )
    assert [hit.evidence_tier for hit in early.exploratory] == ["analogical"]
    assert early.high_confidence == ()
    before = _evaluation_files(tmp_path)
    patterns.append_evaluation(
        _evaluation(
            "imagined_node",
            "img-late",
            "eval-img-late-direct",
            "evidence:img-late:direct",
            "replay-direct",
        ),
        admission=_direct_admission("img-late"),
    )
    assert len(_evaluation_files(tmp_path)) == len(before) + 1
    still = precedents.retrieve(
        overlays,
        patterns,
        query,
        principal_labels=(_CAROL,),
        session_id=_READER,
        temporal_cutoff=_T2,
    )
    assert still == early
    assert _code(
        lambda: precedents.observe_fork(
            overlays,
            patterns,
            planted.stored.fork_node_id,
            session_id=_OTHER,
            recorded_at=_T2,
            principal_labels=(_CAROL,),
        )
    ) == "source_conflict"
    assert _code(
        lambda: precedents.observe_fork(
            overlays,
            patterns,
            planted.stored.fork_node_id,
            session_id=_PAST,
            recorded_at=_T0,
            principal_labels=(_CAROL,),
        )
    ) == "source_conflict"
    precedents.observe_fork(
        overlays,
        patterns,
        planted.stored.fork_node_id,
        session_id=_PAST,
        recorded_at=_T2,
        principal_labels=(_CAROL,),
    )
    promoted = precedents.retrieve(
        overlays,
        patterns,
        query,
        principal_labels=(_CAROL,),
        session_id=_READER,
        temporal_cutoff=_T2,
    )
    assert promoted.retrieval_id != early.retrieval_id
    assert [hit.tier for hit in promoted.high_confidence] == ["high-confidence"]
    assert promoted.high_confidence[0].evidence_tier == "replay-direct"
    assert promoted.high_confidence[0].source == "historical_precedent"
    held = precedents.retrieve(
        overlays,
        patterns,
        query,
        principal_labels=(_CAROL,),
        session_id=_READER,
        temporal_cutoff=_T1,
    )
    assert [hit.evidence_tier for hit in held.exploratory] == ["analogical"]
    assert held.high_confidence == ()
    assert len(_evaluation_files(tmp_path)) == len(before) + 1


def test_submit_requires_retrieval_and_persists_dispositions(tmp_path: Path) -> None:
    overlays, patterns, precedents = _env(tmp_path)
    revision = _revision(
        "rev-submit",
        ("b-fact", "b-anchor", "b-req", "b-req2", "b-ctx", "b-neg"),
    )
    catalog = overlays.register_base_revision(revision, acl_labels=(_CAROL,))
    principal = (_CAROL,)
    required, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-req",
        factual=("b-fact",),
        imagined="img-req",
        session_id=_PAST,
        recorded_at=_T1,
        principal=principal,
        node_rows=(("analogical", "active"),),
    )
    also, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-req2",
        factual=("b-fact",),
        imagined="img-req2",
        session_id=_PAST,
        recorded_at=_T1,
        principal=principal,
        node_rows=(("observational", "active"),),
    )
    contextual, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-ctx",
        factual=("b-ctx",),
        imagined="img-submit-ctx",
        session_id=_PAST,
        recorded_at=_T1,
        principal=principal,
        node_rows=(("analogical", "active"),),
    )
    negative, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-neg",
        factual=("b-fact",),
        imagined="img-neg",
        session_id=_PAST,
        recorded_at=_T1,
        principal=principal,
        node_rows=(("analogical", "invalidated"),),
    )
    query = _query(catalog, revision, imagined="img-submit")
    before_files = _evaluation_files(tmp_path)
    before_index = _strategy_index_lines(tmp_path)
    record = precedents.retrieve(
        overlays,
        patterns,
        query,
        principal_labels=principal,
        session_id=_READER,
        temporal_cutoff=_T2,
    )
    required_ids = {hit.precedent_id for hit in record.candidates if hit.disposition_required}
    assert required_ids == {
        f"branch:{required.stored.fork_node_id}",
        f"branch:{also.stored.fork_node_id}",
    }
    assert f"branch:{contextual.stored.fork_node_id}" not in required_ids
    assert all(hit.fork_node_id != negative.stored.fork_node_id for hit in record.candidates)
    before_overlays = _overlay_files(tmp_path)
    assert _code(
        lambda: precedents.submit_proposal(
            overlays,
            query,
            retrieval_id=record.retrieval_id,
            dispositions=[],
            principal_labels=principal,
            session_id=_READER,
            recorded_at=_T2,
        )
    ) == "disposition_required"
    assert _overlay_files(tmp_path) == before_overlays
    assert _disposition_files(tmp_path) == []
    adopted = {
        "precedent_id": f"branch:{required.stored.fork_node_id}",
        "disposition": "adopted",
        "reason": "same failed action",
    }
    rejected = {
        "precedent_id": f"branch:{also.stored.fork_node_id}",
        "disposition": "not-applicable",
        "reason": "different capability",
    }
    optional = {
        "precedent_id": f"branch:{negative.stored.fork_node_id}",
        "disposition": "rejected",
        "reason": "counterexample stays out of the plan",
    }

    def deny(_session_id: str) -> str:
        raise RuntimeError("denied")

    with pytest.raises(RuntimeError, match="denied"):
        precedents.submit_proposal(
            overlays,
            query,
            retrieval_id=record.retrieval_id,
            dispositions=[adopted, rejected],
            principal_labels=principal,
            session_id=_READER,
            recorded_at=_T2,
            authorize=deny,
        )
    assert _overlay_files(tmp_path) == before_overlays
    assert _disposition_files(tmp_path) == []
    commit = precedents.submit_proposal(
        overlays,
        query,
        retrieval_id=record.retrieval_id,
        dispositions=[optional, rejected, adopted],
        principal_labels=principal,
        session_id=_READER,
        recorded_at=_T2,
    )
    assert commit.deduplicated is False
    assert commit.retrieval_id == record.retrieval_id
    assert {
        (item.precedent_id, item.disposition, item.reason)
        for item in commit.disposition.dispositions
    } == {
        (adopted["precedent_id"], "adopted", "same failed action"),
        (optional["precedent_id"], "rejected", "counterexample stays out of the plan"),
        (rejected["precedent_id"], "not-applicable", "different capability"),
    }
    loaded = precedents.read_dispositions(query["content_hash"])
    assert loaded == commit.disposition
    assert loaded is not None
    assert loaded.fork_node_id == commit.stored.fork_node_id
    assert loaded.retrieval_id == record.retrieval_id
    text = _disposition_files(tmp_path)[0].read_text(encoding="utf-8")
    assert "historical_precedent" in text
    assert "branch_evaluation" not in text
    assert _evaluation_files(tmp_path) == before_files
    assert _strategy_index_lines(tmp_path) == before_index
    replay = precedents.submit_proposal(
        overlays,
        query,
        retrieval_id=record.retrieval_id,
        dispositions=[adopted, optional, rejected],
        principal_labels=principal,
        session_id=_READER,
        recorded_at=_T2,
    )
    assert replay.deduplicated is True
    assert replay.disposition == commit.disposition
    assert len(_overlay_files(tmp_path)) == len(before_overlays) + 1
    assert len(_disposition_files(tmp_path)) == 1
    changed = {**adopted, "reason": "a different reason"}
    assert _code(
        lambda: precedents.submit_proposal(
            overlays,
            query,
            retrieval_id=record.retrieval_id,
            dispositions=[changed, rejected, optional],
            principal_labels=principal,
            session_id=_READER,
            recorded_at=_T2,
        )
    ) == "disposition_conflict"
    assert precedents.read_dispositions(query["content_hash"]) == commit.disposition
    assert len(_overlay_files(tmp_path)) == len(before_overlays) + 1


def test_submit_rejects_a_bad_retrieval_without_writing(tmp_path: Path) -> None:
    overlays, patterns, precedents = _env(tmp_path)
    revision = _revision("rev-bad", ("b-fact", "b-anchor", "b-req"))
    catalog = overlays.register_base_revision(revision, acl_labels=(_CAROL,))
    _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-req",
        factual=("b-fact",),
        imagined="img-bad",
        session_id=_PAST,
        recorded_at=_T1,
        principal=(_CAROL,),
        node_rows=(("analogical", "active"),),
    )
    query = _query(catalog, revision, imagined="img-bad-query")
    record = precedents.retrieve(
        overlays,
        patterns,
        query,
        principal_labels=(_CAROL,),
        session_id=_READER,
        temporal_cutoff=_T2,
    )
    precedent_id = record.exploratory[0].precedent_id
    adopted = {
        "precedent_id": precedent_id,
        "disposition": "adopted",
        "reason": "use it",
    }
    assert _code(
        lambda: precedents.submit_proposal(
            overlays,
            query,
            retrieval_id="",
            dispositions=[adopted],
            principal_labels=(_CAROL,),
            session_id=_READER,
            recorded_at=_T2,
        )
    ) == "retrieval_required"
    assert _code(
        lambda: precedents.submit_proposal(
            overlays,
            query,
            retrieval_id="c" * 64,
            dispositions=[adopted],
            principal_labels=(_CAROL,),
            session_id=_READER,
            recorded_at=_T2,
        )
    ) == "retrieval_not_found"
    assert _code(
        lambda: precedents.submit_proposal(
            overlays,
            query,
            retrieval_id=record.retrieval_id,
            dispositions=[{**adopted, "reason": "   "}],
            principal_labels=(_CAROL,),
            session_id=_READER,
            recorded_at=_T2,
        )
    ) == "disposition_reason_required"
    assert _code(
        lambda: precedents.submit_proposal(
            overlays,
            query,
            retrieval_id=record.retrieval_id,
            dispositions=[{**adopted, "disposition": "accepted"}],
            principal_labels=(_CAROL,),
            session_id=_READER,
            recorded_at=_T2,
        )
    ) == "disposition_invalid"
    assert _code(
        lambda: precedents.submit_proposal(
            overlays,
            query,
            retrieval_id=record.retrieval_id,
            dispositions=[adopted, adopted],
            principal_labels=(_CAROL,),
            session_id=_READER,
            recorded_at=_T2,
        )
    ) == "disposition_duplicate"
    assert _code(
        lambda: precedents.submit_proposal(
            overlays,
            query,
            retrieval_id=record.retrieval_id,
            dispositions=[
                {
                    "precedent_id": "branch:missing",
                    "disposition": "adopted",
                    "reason": "no such precedent",
                }
            ],
            principal_labels=(_CAROL,),
            session_id=_READER,
            recorded_at=_T2,
        )
    ) == "disposition_unknown_precedent"
    other_query = _query(catalog, revision, imagined="img-other-query")
    assert _code(
        lambda: precedents.submit_proposal(
            overlays,
            other_query,
            retrieval_id=record.retrieval_id,
            dispositions=[adopted],
            principal_labels=(_CAROL,),
            session_id=_READER,
            recorded_at=_T2,
        )
    ) == "retrieval_mismatch"
    assert _code(
        lambda: precedents.submit_proposal(
            overlays,
            query,
            retrieval_id=record.retrieval_id,
            dispositions=[adopted],
            principal_labels=(_ALICE,),
            session_id=_READER,
            recorded_at=_T2,
        )
    ) == "retrieval_mismatch"
    assert _code(
        lambda: precedents.submit_proposal(
            overlays,
            query,
            retrieval_id=record.retrieval_id,
            dispositions=[adopted],
            principal_labels=(_CAROL,),
            session_id=_OTHER,
            recorded_at=_T2,
        )
    ) == "retrieval_mismatch"
    path = _retrieval_path(tmp_path, record.retrieval_id)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["body"]["exploratory"][0]["rank"] = 9
    path.write_text(json.dumps(tampered), encoding="utf-8")
    assert _code(
        lambda: precedents.submit_proposal(
            overlays,
            query,
            retrieval_id=record.retrieval_id,
            dispositions=[adopted],
            principal_labels=(_CAROL,),
            session_id=_READER,
            recorded_at=_T2,
        )
    ) == "retrieval_tampered"
    assert len(_overlay_files(tmp_path)) == 1
    assert _disposition_files(tmp_path) == []
    assert len(_evaluation_files(tmp_path)) == 1


def test_empty_retrieval_still_gates_the_proposal(tmp_path: Path) -> None:
    overlays, _patterns, precedents = _env(tmp_path)
    revision = _native_revision("rev-empty")
    catalog = overlays.register_base_revision(revision, acl_labels=(_CAROL,))
    payload = _proposal(catalog, document_hash=revision.content_hash)
    assert _code(
        lambda: precedents.submit_proposal(
            overlays,
            payload,
            retrieval_id="d" * 64,
            dispositions=[],
            principal_labels=(_CAROL,),
            session_id=_READER,
            recorded_at=_T1,
        )
    ) == "retrieval_not_found"
    assert _overlay_files(tmp_path) == []
    record = precedents.retrieve(
        overlays,
        StrategyPatternStore(tmp_path),
        payload,
        principal_labels=(_CAROL,),
        session_id=_READER,
        temporal_cutoff=_T1,
    )
    assert record.candidates == ()
    assert record.negative_retired == ()
    commit = precedents.submit_proposal(
        overlays,
        payload,
        retrieval_id=record.retrieval_id,
        dispositions=[],
        principal_labels=(_CAROL,),
        session_id=_READER,
        recorded_at=_T1,
    )
    assert commit.deduplicated is False
    assert commit.disposition.dispositions == ()
    assert not (tmp_path / STRATEGY_PATTERN_DIRNAME).exists()
    assert _disposition_files(tmp_path)


def test_legacy_claim_and_cut_forks_stay_out_of_candidates(tmp_path: Path) -> None:
    overlays, patterns, precedents = _env(tmp_path)
    revision = _native_revision("rev-legacy-neighbor")
    catalog = overlays.register_base_revision(revision, acl_labels=(_CAROL,))
    planted, _ = _plant(
        overlays,
        patterns,
        precedents,
        revision,
        catalog,
        anchor="b-2",
        factual=("b-1",),
        imagined="img-live",
        session_id=_PAST,
        recorded_at=_T1,
        principal=(_CAROL,),
        node_rows=(("analogical", "active"),),
    )
    legacy = CausalExperiencesStore(tmp_path, config={"skill_trajectory_mode": "causal"})
    legacy.upsert_fork_node(
        _legacy_payload("legacy-cut-fork", position_kind="cut", position_ref="cut:claim-9"),
        idempotency_key="legacy-cut-once",
    )
    legacy.upsert_fork_node(
        _legacy_payload("legacy-edge-fork", position_kind="edge", position_ref="edge:claim-9"),
        idempotency_key="legacy-edge-once",
    )
    assert _code(
        lambda: precedents.observe_fork(
            overlays,
            patterns,
            "legacy-cut-fork",
            session_id=_PAST,
            recorded_at=_T1,
            principal_labels=(_CAROL,),
        )
    ) == "source_unknown"
    query = _proposal(
        catalog,
        document_hash=revision.content_hash,
        anchor_block_id="b-1",
        branch_steps=[
            {"block_id": "b-2", "step_kind": "factual_block_ref"},
            {"imagined_node_id": "img-legacy-query", "step_kind": "imagined_node"},
        ],
    )
    record = precedents.retrieve(
        overlays,
        patterns,
        query,
        principal_labels=(_CAROL,),
        session_id=_READER,
        temporal_cutoff=_T1,
    )
    dumped = json.dumps(
        {
            "high": [hit.__dict__ for hit in record.high_confidence],
            "exploratory": [hit.__dict__ for hit in record.exploratory],
            "negative": [hit.__dict__ for hit in record.negative_retired],
        },
        sort_keys=True,
    )
    assert planted.stored.fork_node_id in dumped
    assert "legacy-cut-fork" not in dumped
    assert "legacy-edge-fork" not in dumped
    assert "claim_id" not in dumped
    assert "position_kind" not in dumped
    assert "influence_grounding" not in dumped
    source = inspect.getsource(
        __import__("openviking.session.fork_precedent", fromlist=["x"])
    )
    assert "CausalExperiencesStore" not in source
    assert "InfluenceClaim(" not in source
    assert "append_evaluation" not in source
    assert "append_similar_fork_evidence" not in source
