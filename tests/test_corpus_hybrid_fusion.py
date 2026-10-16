# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Q156 per-corpus RRF. Ranks are 1-based. One corpus can degrade alone."""

from __future__ import annotations

from openviking.session.corpus_hybrid_fusion import (
    CHANNEL_LIMIT,
    FUSION_POLICY_VERSION,
    RRF_K,
    fuse_corpora,
)

PROOF = {
    "complete": True,
    "snapshot_ref": "snap-1",
    "index_version": "index-1",
    "model_version": "model-1",
    "policy_version": FUSION_POLICY_VERSION,
}


def test_rrf_constants_match_the_accepted_policy() -> None:
    assert RRF_K == 60
    assert CHANNEL_LIMIT == 32
    assert FUSION_POLICY_VERSION == "memory-hybrid-fusion/Q156-B"


def test_vector_only_hit_joins_the_union_and_dual_hit_is_one_card() -> None:
    fused = fuse_corpora(
        {
            "skill_revision": {
                "bm25": ["lexical"],
                "vector": ["semantic", "lexical"],
                "proof": PROOF,
            }
        }
    )
    order = fused[0]
    assert order.degraded is False
    assert order.policy_version == FUSION_POLICY_VERSION
    assert order.ids == ("lexical", "semantic")
    assert not hasattr(order, "scores")
    assert "snap-1" not in "".join(order.ids)


def test_equal_rrf_breaks_ties_by_canonical_identity() -> None:
    fused = fuse_corpora(
        {
            "fork_branch": {
                "bm25": ["m", "b"],
                "vector": ["m", "a"],
                "proof": PROOF,
            }
        }
    )
    # b is BM25 rank 2 and a is vector rank 2, so both score 1/(60+2).
    assert fused[0].ids[1:] == ("a", "b")


def test_channel_limit_drops_the_thirty_third_lexical_hit() -> None:
    identities = [f"id-{index:02d}" for index in range(1, 34)]
    fused = fuse_corpora({"projection": {"bm25": identities, "vector": [], "proof": PROOF}})
    assert fused[0].ids == tuple(identities[:32])
    assert "id-33" not in fused[0].ids


def test_vector_failure_degrades_only_that_corpus() -> None:
    bm25_a = ["a2", "a1"]
    bm25_b = ["b1", "b2"]
    fused = fuse_corpora(
        {
            "skill_revision": {
                "bm25": bm25_a,
                "vector": ["should-not-win"],
                "vector_failed": True,
                "proof": PROOF,
            },
            "fork_branch": {
                "bm25": bm25_b,
                "vector": ["vector-only", "b2"],
                "proof": PROOF,
            },
        }
    )
    by_corpus = {item.corpus: item for item in fused}
    assert by_corpus["skill_revision"].degraded is True
    assert by_corpus["skill_revision"].ids == tuple(bm25_a)
    assert by_corpus["fork_branch"].degraded is False
    assert by_corpus["fork_branch"].ids[0] == "b2"
    assert "vector-only" in by_corpus["fork_branch"].ids


def test_missing_proof_uses_bm25_order_even_when_vector_hits_exist() -> None:
    fused = fuse_corpora(
        {
            "segment_atom": {
                "bm25": ["first", "second"],
                "vector": ["second", "vector-only"],
                "proof": None,
            }
        }
    )
    assert fused[0].degraded is True
    assert fused[0].ids == ("first", "second")
