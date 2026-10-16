# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Active retrieval over native edges, overlays, patterns, and guidance.

Claim projections, cut grounding, and legacy forks are not cards on this
path. Those stay on ``HybridRetrievalFacade.audit_legacy``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openviking.session.aig_native_protocol import GuidancePublication
from openviking.session.extraction_write import (
    NATIVE_EDGE_INDEX_FILENAME,
    NativeEdgeViewStore,
)
from openviking.session.fork_node_overlay import (
    ForkNodeOverlayStore,
    ForkOverlayError,
    guard_v2_fork_consumer,
)
from openviking.session.guidance_approval import read_stored_guidance_publications
from openviking.session.influence_view import AigNativeViewRevision
from openviking.session.retrieval_facade import (
    CAUSAL_MEMORY_CAP,
    FEATURE_SPEC_VERSION,
    RANK_MODE_BM25,
    RANK_MODE_HYBRID,
    IdentityRerank,
    RetrievalCard,
    RetrievalProfile,
    RetrievalQuery,
    RetrievalResult,
    _allocate_stateful,
    _derive_query_seed,
    _document_text,
    _labels_visible,
    _pending_score,
    _principal_set,
    _rank_hits,
    _RankPlan,
    coerce_profile,
    coerce_query,
    default_acl_filter,
)
from openviking.session.session_acl import session_owner_label
from openviking.session.strategy_pattern import StrategyPatternStore

LEGACY_ACTIVITY_KINDS = frozenset(
    {"fork", "branch", "block_projection", "claim_projection", "segment", "atom"}
)
_RETIRED = "retired"
_V2_RETRIEVAL = "v2_retrieval"


@dataclass(frozen=True)
class BoundNativeView:
    """One native-edge revision plus the labels that may read it."""

    revision: AigNativeViewRevision
    acl_labels: tuple[str, ...]


@dataclass(frozen=True)
class ActiveRetrievalCorpus:
    """Sources for one active retrieve. Legacy stores are not fields."""

    native_views: tuple[BoundNativeView, ...] = ()
    overlay_store: ForkNodeOverlayStore | None = None
    pattern_store: StrategyPatternStore | None = None
    guidance_publications: tuple[GuidancePublication, ...] = ()


def corpus_for_session(
    session_dir: Path,
    user_root: Path,
    *,
    owner_label: str,
) -> ActiveRetrievalCorpus:
    """Bind on-disk native views, overlays, patterns, and guidance publications.

    A missing native-edge file is an empty view list. This function does not
    open the claim index and does not write.
    """
    native_path = session_dir / NATIVE_EDGE_INDEX_FILENAME
    views: tuple[BoundNativeView, ...] = ()
    if native_path.is_file():
        stored = NativeEdgeViewStore(native_path)
        labels = (owner_label,)
        views = tuple(BoundNativeView(revision, labels) for revision in stored.revisions())
    return ActiveRetrievalCorpus(
        native_views=views,
        overlay_store=ForkNodeOverlayStore(user_root),
        pattern_store=StrategyPatternStore(user_root),
        guidance_publications=read_stored_guidance_publications(user_root),
    )


def corpus_for_facade(*, session: Any, user_root: Path, session_dir: Path) -> ActiveRetrievalCorpus:
    """Session-owner binding used by the production facade."""
    return corpus_for_session(session_dir, user_root, owner_label=session_owner_label(session))


def retrieve_active(
    corpus: ActiveRetrievalCorpus | None,
    query: RetrievalQuery | Mapping[str, Any] | str | None,
    principal_labels: Sequence[str],
    profile: RetrievalProfile | Mapping[str, Any] | None = None,
    *,
    rerank: Any = None,
    acl_filter: Callable[[RetrievalCard, Sequence[str]], bool] | None = None,
    feature_version: str = FEATURE_SPEC_VERSION,
    seed: str | None = None,
    embedder: Any = None,
) -> RetrievalResult:
    """Rank native blocks, edges, overlays, patterns, and guidance.

    Segment/atom cards and legacy claim/fork cards are not collected.
    ``dependency_pattern`` is not applied: it is a claim signature.
    """
    parsed = coerce_query(query)
    budgets = coerce_profile(profile)
    _principal_set(principal_labels)
    chosen_seed = seed if seed is not None else _derive_query_seed(parsed)
    plan = _RankPlan(budgets.rank_mode, embedder if budgets.rank_mode == RANK_MODE_HYBRID else None)
    plan.prepare(parsed.text)
    gate = acl_filter if acl_filter is not None else default_acl_filter
    hook = rerank if rerank is not None else IdentityRerank()
    native_cards, experience_cards = _lanes(corpus, parsed, budgets, principal_labels, plan, gate)
    if plan.downgraded:
        plan.actual = RANK_MODE_BM25
        plan.query_vector = None
        plan.embedder = None
        plan.downgraded = False
        native_cards, experience_cards = _lanes(
            corpus, parsed, budgets, principal_labels, plan, gate
        )
    stateful, audit = _allocate_stateful(
        [*experience_cards, *native_cards],
        provisional_slots=budgets.provisional_slots,
        seed=chosen_seed,
    )
    selected = _without_legacy(stateful)
    context: dict[str, Any] = {
        "feature_version": feature_version,
        "query": parsed.to_dict(),
        "profile": budgets.to_dict(),
        "rank_tier": plan.actual,
        "requested_rank_tier": plan.requested,
    }
    reranked = _call_rerank(hook, selected, context)
    truncated = len(reranked) > CAUSAL_MEMORY_CAP
    clamped = _without_legacy(reranked[:CAUSAL_MEMORY_CAP])
    return RetrievalResult(
        cards=tuple(clamped),
        feature_version=feature_version,
        truncated_by_rerank_clamp=truncated,
        selection_audit=audit,
        rank_tier=plan.actual,
    )


def _lanes(
    corpus: ActiveRetrievalCorpus | None,
    query: RetrievalQuery,
    budgets: RetrievalProfile,
    principal_labels: Sequence[str],
    plan: _RankPlan,
    acl_filter: Callable[[RetrievalCard, Sequence[str]], bool],
) -> tuple[list[RetrievalCard], list[RetrievalCard]]:
    if corpus is None:
        return [], []
    native = _visible(
        _native_cards(corpus, query, plan, principal_labels),
        principal_labels,
        acl_filter,
    )[: budgets.projection]
    experience = _visible(
        _experience_cards(corpus, query, plan, principal_labels),
        principal_labels,
        acl_filter,
    )[: budgets.fork_branch]
    return native, experience


def _native_cards(
    corpus: ActiveRetrievalCorpus,
    query: RetrievalQuery,
    plan: _RankPlan,
    principal_labels: Sequence[str],
) -> list[RetrievalCard]:
    cards: list[RetrievalCard] = []
    documents: list[str] = []
    principal = _principal_set(principal_labels)
    for bound in corpus.native_views:
        if not _labels_visible(bound.acl_labels, principal):
            continue
        revision = bound.revision
        labels = bound.acl_labels
        for block in revision.blocks:
            cards.append(
                _card(
                    kind="native_block",
                    channel="verified",
                    status_label="factual",
                    payload_ref=f"native-block:{revision.revision_id}:{block.block_id}",
                    labels=labels,
                    pointer={
                        "corpus": "native_edge",
                        "view_revision_id": revision.revision_id,
                        "block_id": block.block_id,
                        "session_id": revision.session_id,
                    },
                )
            )
            documents.append(_document_text(block.summary, block.block_id, block.role))
        for edge in revision.edges:
            cards.append(
                _card(
                    kind="native_edge",
                    channel="verified",
                    status_label="factual",
                    payload_ref=f"native-edge:{revision.revision_id}:{edge.edge_id}",
                    labels=labels,
                    pointer={
                        "corpus": "native_edge",
                        "view_revision_id": revision.revision_id,
                        "edge_id": edge.edge_id,
                        "source_block_id": edge.source_block_id,
                        "target_block_id": edge.target_block_id,
                        "session_id": revision.session_id,
                    },
                )
            )
            documents.append(
                _document_text(
                    edge.mechanism,
                    edge.edge_id,
                    edge.relation_type,
                    edge.source_block_id,
                    edge.target_block_id,
                )
            )
    return _rank_hits(cards, documents, query, plan)


def _experience_cards(
    corpus: ActiveRetrievalCorpus,
    query: RetrievalQuery,
    plan: _RankPlan,
    principal_labels: Sequence[str],
) -> list[RetrievalCard]:
    cards: list[RetrievalCard] = []
    documents: list[str] = []
    _append_overlays(corpus.overlay_store, principal_labels, cards, documents)
    _append_patterns(corpus.pattern_store, corpus.overlay_store, principal_labels, cards, documents)
    _append_guidance(corpus.guidance_publications, principal_labels, cards, documents)
    return _rank_hits(cards, documents, query, plan)


def _append_overlays(
    store: ForkNodeOverlayStore | None,
    principal_labels: Sequence[str],
    cards: list[RetrievalCard],
    documents: list[str],
) -> None:
    if store is None or not _principal_set(principal_labels):
        return
    try:
        overlays = store.visible_overlays(principal_labels=principal_labels)
    except ForkOverlayError as exc:
        if exc.code == "acl_denied":
            return
        raise
    for stored in overlays:
        guard_v2_fork_consumer(stored, _V2_RETRIEVAL)
        overlay = stored.overlay
        step_ids = [step.block_id or step.imagined_node_id or "" for step in overlay.branch_steps]
        cards.append(
            _card(
                kind="fork_overlay",
                channel="provisional",
                status_label="counterfactual",
                payload_ref=f"fork-overlay:{overlay.fork_node_id}",
                labels=(),
                pointer={
                    "corpus": "fork_overlay",
                    "fork_node_id": overlay.fork_node_id,
                    "base_revision_id": overlay.base_revision_id,
                    "anchor_block_id": overlay.anchor_block_id,
                },
            )
        )
        documents.append(
            _document_text(
                overlay.fork_node_id,
                overlay.anchor_block_id,
                overlay.base_revision_id,
                *step_ids,
            )
        )


def _append_patterns(
    patterns: StrategyPatternStore | None,
    overlays: ForkNodeOverlayStore | None,
    principal_labels: Sequence[str],
    cards: list[RetrievalCard],
    documents: list[str],
) -> None:
    if patterns is None or overlays is None or not _principal_set(principal_labels):
        return
    retired = _retired_imagined_nodes(patterns)
    for pattern in patterns.list_patterns():
        memberships = patterns.memberships(pattern.pattern_id)
        if not memberships:
            continue
        visible = False
        for membership in memberships:
            if membership.imagined_node_id in retired:
                continue
            if _overlay_visible(overlays, membership.fork_node_id, principal_labels):
                visible = True
                break
        if not visible:
            continue
        cards.append(
            _card(
                kind="strategy_pattern",
                channel="provisional",
                status_label="pattern",
                payload_ref=f"strategy-pattern:{pattern.pattern_id}",
                labels=(),
                pointer={
                    "corpus": "strategy_pattern",
                    "pattern_id": pattern.pattern_id,
                    "intervention_type": pattern.intervention_type,
                    "target_boundary": pattern.target_boundary,
                },
            )
        )
        documents.append(
            _document_text(
                pattern.intervention_type,
                pattern.mechanism,
                pattern.expected_metric,
                pattern.capability,
                pattern.target_boundary,
                pattern.pattern_id,
            )
        )


def _append_guidance(
    publications: Sequence[GuidancePublication],
    principal_labels: Sequence[str],
    cards: list[RetrievalCard],
    documents: list[str],
) -> None:
    principal = _principal_set(principal_labels)
    for publication in publications:
        if not _labels_visible(publication.publisher_labels, principal):
            continue
        statements = [step.statement for step in publication.steps]
        cards.append(
            _card(
                kind="guidance",
                channel="verified",
                status_label="approved",
                payload_ref=f"guidance:{publication.guidance_id}",
                labels=publication.publisher_labels,
                pointer={
                    "corpus": "guidance",
                    "guidance_id": publication.guidance_id,
                    "source_fork_id": publication.source_fork_id,
                    "base_revision_id": publication.base_revision_id,
                    "target_role": publication.target_role,
                },
                text=publication.reason,
            )
        )
        documents.append(
            _document_text(
                publication.guidance_id,
                publication.target_role,
                publication.reason,
                publication.risk,
                *statements,
            )
        )


def _overlay_visible(
    store: ForkNodeOverlayStore, fork_node_id: str, principal_labels: Sequence[str]
) -> bool:
    try:
        found = store.read_overlay_by_id(fork_node_id, principal_labels=principal_labels)
    except ForkOverlayError as exc:
        if exc.code == "acl_denied":
            return False
        raise
    if found is None:
        return False
    guard_v2_fork_consumer(found, _V2_RETRIEVAL)
    return True


def _retired_imagined_nodes(store: StrategyPatternStore) -> set[str]:
    latest: dict[str, str] = {}
    for evaluation in store.audit_evaluations():
        if evaluation.subject_kind != "imagined_node":
            continue
        latest[evaluation.subject_id] = evaluation.status
    return {subject_id for subject_id, status in latest.items() if status == _RETIRED}


def _card(
    *,
    kind: str,
    channel: str,
    status_label: str,
    payload_ref: str,
    labels: Sequence[str],
    pointer: dict[str, Any],
    text: str = "",
) -> RetrievalCard:
    return RetrievalCard(
        kind=kind,  # type: ignore[arg-type]
        channel=channel,  # type: ignore[arg-type]
        in_channel_rank=0,
        status_label=status_label,
        source_pointer=pointer,
        payload_ref=payload_ref,
        score_breakdown=_pending_score(),
        acl_labels=tuple(labels),
        text=text,
    )


def _visible(
    cards: Sequence[RetrievalCard],
    principal_labels: Sequence[str],
    acl_filter: Callable[[RetrievalCard, Sequence[str]], bool],
) -> list[RetrievalCard]:
    principal = _principal_set(principal_labels)
    visible: list[RetrievalCard] = []
    for card in cards:
        labels = card.acl_labels
        if labels:
            if acl_filter(card, principal_labels) and _labels_visible(labels, principal):
                visible.append(card)
            continue
        visible.append(card)
    return visible


def _without_legacy(cards: Sequence[RetrievalCard]) -> list[RetrievalCard]:
    return [card for card in cards if card.kind not in LEGACY_ACTIVITY_KINDS]


def _call_rerank(
    hook: Any, cards: Sequence[RetrievalCard], context: Mapping[str, Any]
) -> list[RetrievalCard]:
    if isinstance(hook, IdentityRerank) or hasattr(hook, "rerank"):
        reranked = hook.rerank(cards, context)
    else:
        reranked = hook(cards, context)
    return [
        card if isinstance(card, RetrievalCard) else RetrievalCard.from_dict(card)
        for card in reranked
    ]
