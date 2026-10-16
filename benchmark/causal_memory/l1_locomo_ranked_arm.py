#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""L1 zero-LLM LoCoMo arm that ranks the candidate pool globally.

Same ingest and the same fallback view as ``l1_locomo_projection_arm.py``.
Seat count stays a retrieve profile (``fork_branch`` / ``projection`` /
``segment_atom``). One revision per session. The fact lane is opened only
through ``HybridRetrievalFacade(..., fact_lane_paused=False)``; the module
constant is not written.

For each question, every keyword is its own ``facade.retrieve``. That card
list is the candidate pool. Three readings share it:

  legacy   first time a dia appears, walking keywords, then returned cards,
           then the dias one card covers.
  ranked   dia score is the highest ``score_breakdown["bm25"]`` among cards
           that cover it. Ties break by first-seen index, then dia id.
  rrf      for each keyword add ``1 / (60 + best 1-based rank)`` of the
           returned cards that cover the dia. Same tie break as ranked.

``l1_locomo_hitk.py`` and ``l1_locomo_projection_arm.py`` stay untouched.

Beyond any-hit, each reading records coverage@k (fraction of that question's
gold set inside the top k), all_gold@k (the whole gold set is inside the top
k), and mrr (1/rank of the first gold inside the top 10, else 0). ``gold_n``
is the deduped gold size. These fields are additive: existing keys stay.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from l1_locomo_projection_arm import (
    DEFAULT_DATA,
    FACTS_PROFILE,
    LABELS,
    PROJECTION_PROFILE,
    _fail,
    _pct,
    _question_keywords,
    _tokens,
    any_hit,
    build_projection,
    dias_for_card,
    ingest_conversation,
    token_ranked_dias,
)

from openviking.session.influence_view import InfluenceViewRevision
from openviking.session.retrieval_facade import (
    FACT_LANE_PAUSED,
    ForkBranchAdapter,
    HybridRetrievalFacade,
    InfluenceProjectionAdapter,
    SegmentAtomAdapter,
)

# Full-run numbers the paused fact lane and the token reference must still
# print. Same .1% strings as l1_locomo_projection_arm.py. Skipped with
# --limit-convs. Do not edit the frozen baseline to absorb a miss.
_BASELINE_P_HIT5 = "0.0%"
_BASELINE_P_HIT10 = "0.0%"
_BASELINE_P_ZERO = "100.0%"
_BASELINE_T_HIT5 = "56.0%"
_BASELINE_T_HIT10 = "61.8%"
# Absolute counts from the R6e archive. A miss means this edit moved the pool
# or the order. Do not edit these to absorb a miss. T's 56.0% is shared with
# a neighbour count, so the count is part of the gate.
_R6E_T_HIT5 = 1109
_R6E_T_HIT10 = 1225
_R6E_HIT10: tuple[tuple[str, str, int, str], ...] = (
    ("A12", "ranked", 1170, "59.0%"),
    ("A4", "ranked", 1110, "56.0%"),
    ("B4", "ranked", 1062, "53.6%"),
    ("B12", "ranked", 1108, "55.9%"),
    ("A4", "legacy", 722, "36.4%"),
    ("A12", "legacy", 302, "15.2%"),
    ("B4", "legacy", 722, "36.4%"),
    ("B12", "legacy", 314, "15.8%"),
)
_GOLD_BUCKETS = ("1", "2", ">=3")

PROFILE_A4 = {"fork_branch": 0, "projection": 0, "segment_atom": 4}
PROFILE_B12 = {"fork_branch": 0, "projection": 12, "segment_atom": 0}
RRF_K = 60

# name, profile, channel, fact_lane_paused, metrics
# B keeps the fact lane paused: its segment_atom quota is 0, so opening the
# lane only adds a search that is sliced to nothing. Projection cards stay
# the same. A opens the lane because that is the channel under test.
_ARMS: tuple[tuple[str, dict[str, int], str, bool, tuple[str, ...]], ...] = (
    ("P_legacy", FACTS_PROFILE, "atom", True, ("legacy",)),
    ("A12", FACTS_PROFILE, "atom", False, ("legacy", "ranked")),
    ("A4", PROFILE_A4, "atom", False, ("legacy", "ranked", "rrf")),
    ("B4", PROJECTION_PROFILE, "projection", True, ("legacy", "ranked", "rrf")),
    ("B12", PROFILE_B12, "projection", True, ("legacy", "ranked")),
)
_LIVE_ARMS = ("A4", "A12", "B4", "B12")
_RANKED_ARMS = ("A4", "A12", "B4", "B12")


class _Hit:
    __slots__ = ("dedup_key", "bm25", "dias", "keyword_index", "rank")

    def __init__(
        self,
        dedup_key: str,
        bm25: float,
        dias: tuple[str, ...],
        keyword_index: int,
        rank: int,
    ) -> None:
        self.dedup_key = dedup_key
        self.bm25 = bm25
        self.dias = dias
        self.keyword_index = keyword_index
        self.rank = rank


class _Metric:
    __slots__ = (
        "pool",
        "union",
        "hit5",
        "hit10",
        "ordered",
        "coverage5",
        "coverage10",
        "all_gold5",
        "all_gold10",
        "mrr",
    )

    def __init__(
        self,
        pool: int,
        union: int,
        hit5: bool,
        hit10: bool,
        ordered: tuple[str, ...],
        coverage5: float,
        coverage10: float,
        all_gold5: bool,
        all_gold10: bool,
        mrr: float,
    ) -> None:
        self.pool = pool
        self.union = union
        self.hit5 = hit5
        self.hit10 = hit10
        self.ordered = ordered
        self.coverage5 = coverage5
        self.coverage10 = coverage10
        self.all_gold5 = all_gold5
        self.all_gold10 = all_gold10
        self.mrr = mrr


def _print_policy() -> None:
    print(
        "paging: trajectory_index_hook._page_ao_records; "
        "row shape: trajectory_index_hook._view_ao_record; "
        "view: build_minimal_fallback_view purpose=post_run_index"
    )
    print(
        f"module FACT_LANE_PAUSED={FACT_LANE_PAUSED} "
        "(not assigned). A passes fact_lane_paused=False. "
        "P and B pass fact_lane_paused=True."
    )
    print(
        "dedup key: projection source_pointer['card_id']; atom payload_ref. "
        "dia: projection block source_refs ao_id (every dia on the block); "
        "atom source_pointer['ao_id']."
    )
    print(
        "legacy = first-seen dia. "
        "ranked = max score_breakdown['bm25']; ties by first-seen index, then dia id. "
        f"rrf = sum 1/({RRF_K} + best 1-based rank in that keyword's returned cards); "
        "same ties. One candidate list per arm."
    )
    print(
        "coverage@k = |gold ∩ top k| / |gold|. "
        "all_gold@k = gold ⊆ top k. "
        "mrr = 1/rank of the first gold inside the top 10, else 0. "
        "The order is already total (score, first-seen, dia id); no tie average. "
        "gold_n buckets: 1, 2, >=3."
    )


def build_facade(
    work: Path,
    session_id: str,
    registry: Any,
    revision: InfluenceViewRevision,
    ao_text: dict[str, str],
    *,
    fact_lane_paused: bool,
) -> HybridRetrievalFacade:
    def resolve(ao_id: str) -> str:
        text = ao_text.get(ao_id, "")
        return text if isinstance(text, str) else ""

    return HybridRetrievalFacade(
        fork_branch=ForkBranchAdapter(work),
        projection=InfluenceProjectionAdapter(registry, views=[revision], content_resolver=resolve),
        segment_atom=SegmentAtomAdapter(work / session_id, session_id),
        fact_lane_paused=fact_lane_paused,
    )


def _bm25(card: Any, arm: str) -> float:
    breakdown = card.score_breakdown
    if not isinstance(breakdown, dict) or "bm25" not in breakdown:
        _fail(f"{arm}: card is missing score_breakdown['bm25']")
    score = breakdown["bm25"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        _fail(f"{arm}: bm25 score is not a finite number: {score!r}")
    return float(score)


def _hit_from_card(
    card: Any,
    arm: str,
    channel: str,
    revision: InfluenceViewRevision,
    ao_to_dia: dict[str, str],
    keyword_index: int,
    rank: int,
    stats: Counter,
) -> _Hit:
    pointer = card.source_pointer if isinstance(card.source_pointer, dict) else {}
    if channel == "atom":
        if card.kind not in ("atom", "segment"):
            _fail(f"{arm}: expected an atom card, got kind={card.kind!r}")
        key = card.payload_ref
        if not isinstance(key, str) or not key:
            _fail(f"{arm}: atom card is missing payload_ref")
        ao_id = pointer.get("ao_id")
        if not isinstance(ao_id, str) or ao_id not in ao_to_dia:
            _fail(f"{arm}: atom card ao_id {ao_id!r} is not in the ingested ledger")
        dias = (ao_to_dia[ao_id],)
    else:
        if card.kind not in ("block_projection", "claim_projection"):
            _fail(f"{arm}: expected a projection card, got kind={card.kind!r}")
        key = pointer.get("card_id")
        if not isinstance(key, str) or not key:
            _fail(f"{arm}: projection card source_pointer is missing card_id")
        if card.payload_ref != key:
            stats["payload_ne_card_id"] += 1
        dias_list = dias_for_card(card, revision, ao_to_dia)
        if not dias_list:
            _fail(f"{arm}: projection card {key} covers no dia")
        dias = tuple(dias_list)
    return _Hit(key, _bm25(card, arm), dias, keyword_index, rank)


def collect_hits(
    facade: HybridRetrievalFacade,
    keywords: list[str],
    profile: dict[str, int],
    arm: str,
    channel: str,
    revision: InfluenceViewRevision,
    ao_to_dia: dict[str, str],
    stats: Counter,
) -> list[_Hit]:
    hits: list[_Hit] = []
    for keyword_index, keyword in enumerate(keywords):
        result = facade.retrieve(keyword, principal_labels=LABELS, profile=profile)
        for rank, card in enumerate(result.cards, start=1):
            hits.append(
                _hit_from_card(card, arm, channel, revision, ao_to_dia, keyword_index, rank, stats)
            )
    return hits


def _first_seen(hits: list[_Hit]) -> dict[str, int]:
    seen: dict[str, int] = {}
    for hit in hits:
        for dia in hit.dias:
            if dia not in seen:
                seen[dia] = len(seen)
    return seen


def _ranked_order(hits: list[_Hit], first: dict[str, int]) -> list[str]:
    best: dict[str, float] = {}
    for hit in hits:
        for dia in hit.dias:
            current = best.get(dia)
            if current is None or hit.bm25 > current:
                best[dia] = hit.bm25
    return sorted(first, key=lambda dia: (-best[dia], first[dia], dia))


def _rrf_order(hits: list[_Hit], first: dict[str, int]) -> list[str]:
    best_rank: dict[tuple[int, str], int] = {}
    for hit in hits:
        for dia in hit.dias:
            key = (hit.keyword_index, dia)
            current = best_rank.get(key)
            if current is None or hit.rank < current:
                best_rank[key] = hit.rank
    scores = dict.fromkeys(first, 0.0)
    for (_keyword_index, dia), rank in best_rank.items():
        scores[dia] += 1.0 / (RRF_K + rank)
    return sorted(first, key=lambda dia: (-scores[dia], first[dia], dia))


def _coverage_bundle(
    ordered: list[str] | tuple[str, ...], gold: set[str]
) -> tuple[float, float, bool, bool, float]:
    """coverage@5, coverage@10, all_gold@5, all_gold@10, mrr.

    mrr is 1/rank of the first gold inside the top 10, else 0. Truncating at
    10 is what makes ``mrr > 0`` the same fact as ``coverage@10 > 0``.
    """
    gold_n = len(gold)
    if gold_n == 0:
        _fail("coverage on an empty gold set")
    covered5 = len(gold & set(ordered[:5]))
    covered10 = len(gold & set(ordered[:10]))
    mrr = 0.0
    for rank, dia in enumerate(ordered[:10], start=1):
        if dia in gold:
            mrr = 1.0 / rank
            break
    return (
        covered5 / gold_n,
        covered10 / gold_n,
        covered5 == gold_n,
        covered10 == gold_n,
        mrr,
    )


def _gold_bucket(gold_n: int) -> str:
    if gold_n == 1:
        return "1"
    if gold_n == 2:
        return "2"
    return ">=3"


def _inconsistent(
    sample_id: str,
    label: str,
    gold: set[str],
    gold_n: int,
    ordered: list[str] | tuple[str, ...],
    hit5: bool,
    hit10: bool,
    coverage5: float,
    coverage10: float,
    all_gold5: bool,
    all_gold10: bool,
    mrr: float,
) -> str | None:
    """gold_n, coverage, all_gold, and mrr must describe the same gold set."""
    if gold_n != len(set(gold)) or gold_n < 1:
        return f"{sample_id} {label}: gold_n {gold_n} != len(set(gold)) {len(set(gold))}"
    fresh = _coverage_bundle(ordered, gold)
    stored = (coverage5, coverage10, all_gold5, all_gold10, mrr)
    if stored != fresh:
        return f"{sample_id} {label}: stored {stored} != recomputed {fresh}"
    if all_gold5 != (coverage5 == 1.0):
        return f"{sample_id} {label}: all_gold@5={all_gold5} coverage@5={coverage5}"
    if all_gold10 != (coverage10 == 1.0):
        return f"{sample_id} {label}: all_gold@10={all_gold10} coverage@10={coverage10}"
    if (mrr > 0.0) != (coverage10 > 0.0):
        return (
            f"{sample_id} {label}: mrr={mrr} coverage@10={coverage10} (mrr > 0 iff coverage@10 > 0)"
        )
    if mrr < 0.0 or mrr > 1.0:
        return f"{sample_id} {label}: mrr={mrr} is outside 0..1"
    if hit5 != (coverage5 > 0.0) or hit10 != (coverage10 > 0.0):
        return f"{sample_id} {label}: any-hit disagrees with coverage"
    if all_gold5 and not all_gold10:
        return f"{sample_id} {label}: all_gold@5 without all_gold@10"
    if coverage5 > coverage10:
        return f"{sample_id} {label}: coverage@5 {coverage5} > coverage@10 {coverage10}"
    if gold_n == 1 and (all_gold5 != hit5 or all_gold10 != hit10):
        return f"{sample_id} {label}: single-gold all_gold disagrees with any-hit"
    return None


def score_pool(hits: list[_Hit], gold: set[str], metrics: tuple[str, ...]) -> dict[str, _Metric]:
    """Rank one shared candidate list. Union size is recomputed per metric."""
    first = _first_seen(hits)
    pool = len({hit.dedup_key for hit in hits})
    orders: dict[str, list[str]] = {}
    if "legacy" in metrics:
        orders["legacy"] = list(first)
    if "ranked" in metrics:
        orders["ranked"] = _ranked_order(hits, first)
    if "rrf" in metrics:
        orders["rrf"] = _rrf_order(hits, first)
    scored: dict[str, _Metric] = {}
    for name, ordered in orders.items():
        if len(ordered) != len(set(ordered)):
            _fail(f"{name} ranking emitted a duplicate dia")
        coverage5, coverage10, all_gold5, all_gold10, mrr = _coverage_bundle(ordered, gold)
        scored[name] = _Metric(
            pool=pool,
            union=len(ordered),
            hit5=any_hit(ordered, gold, 5),
            hit10=any_hit(ordered, gold, 10),
            ordered=tuple(ordered),
            coverage5=coverage5,
            coverage10=coverage10,
            all_gold5=all_gold5,
            all_gold10=all_gold10,
            mrr=mrr,
        )
    return scored


def _add(counter: Counter, metric: _Metric) -> None:
    counter["hit5"] += int(metric.hit5)
    counter["hit10"] += int(metric.hit10)
    counter["zero"] += int(metric.union == 0)
    counter["union"] += metric.union
    counter["pool"] += metric.pool
    counter["coverage5"] += metric.coverage5
    counter["coverage10"] += metric.coverage10
    counter["all_gold5"] += int(metric.all_gold5)
    counter["all_gold10"] += int(metric.all_gold10)
    counter["mrr"] += metric.mrr


def _delta(later: int, earlier: int, total: int) -> str:
    return f"{(later - earlier) / total:+.1%}"


def _row_line(arm: str, metric: str, counter: Counter, total: int) -> None:
    print(
        f"{arm:<10} {metric:<8} "
        f"hit@5={_pct(counter['hit5'], total):>7}  "
        f"hit@10={_pct(counter['hit10'], total):>7}  "
        f"zero={_pct(counter['zero'], total):>7}  "
        f"mean_union={counter['union'] / total:.2f}  "
        f"mean_pool={counter['pool'] / total:.2f}"
    )


def _reading_labels() -> list[tuple[str, str]]:
    labels = [("T", "token")]
    for name, _profile, _channel, _paused, metrics in _ARMS:
        for metric in metrics:
            labels.append((name, metric))
    return labels


def _counter_for(bucket: dict[str, Any], arm: str, metric: str) -> Counter:
    if arm == "T":
        return bucket["T"]
    return bucket[arm][metric]


def _rich_line(arm: str, metric: str, counter: Counter, total: int) -> None:
    print(
        f"{arm:<10} {metric:<8} "
        f"hit@5={_pct(counter['hit5'], total):>7}  "
        f"hit@10={_pct(counter['hit10'], total):>7}  "
        f"cov@5={_pct(counter['coverage5'], total):>7}  "
        f"cov@10={_pct(counter['coverage10'], total):>7}  "
        f"all@5={_pct(counter['all_gold5'], total):>7}  "
        f"all@10={_pct(counter['all_gold10'], total):>7}  "
        f"mrr={counter['mrr'] / total:.4f}"
    )


def _print_tables(
    overall: dict[str, dict[str, Counter]],
    token: Counter,
    by_cat: dict[Any, dict[str, Counter]],
    by_gold: dict[str, dict[str, Any]],
    total: int,
    top10_differs: Counter,
    problems_so_far: int,
) -> None:
    print("\n=== overall (zero-LLM evidence any-hit) ===")
    print(f"questions={total}")
    for name, _profile, _channel, _paused, metrics in _ARMS:
        for metric in metrics:
            _row_line(name, metric, overall[name][metric], total)
    print(
        f"{'T':<10} {'token':<8} "
        f"hit@5={_pct(token['hit5'], total):>7}  "
        f"hit@10={_pct(token['hit10'], total):>7}"
    )

    print("\n=== seats 4 -> 12 (12 minus 4) ===")
    for label, small, large in (("atom", "A4", "A12"), ("projection", "B4", "B12")):
        for metric in ("legacy", "ranked"):
            left = overall[small][metric]
            right = overall[large][metric]
            print(
                f"{label} {metric}: "
                f"hit@5 {_pct(left['hit5'], total)} -> {_pct(right['hit5'], total)} "
                f"({_delta(right['hit5'], left['hit5'], total)})  "
                f"hit@10 {_pct(left['hit10'], total)} -> {_pct(right['hit10'], total)} "
                f"({_delta(right['hit10'], left['hit10'], total)})  "
                f"mean_union {left['union'] / total:.2f} -> {right['union'] / total:.2f}"
            )

    print("\n=== same seats, projection minus atom (B - A) ===")
    for seats, atom_arm, proj_arm in (("4", "A4", "B4"), ("12", "A12", "B12")):
        for metric in ("legacy", "ranked"):
            atom = overall[atom_arm][metric]
            proj = overall[proj_arm][metric]
            print(
                f"{seats} {metric}: "
                f"hit@5 {_pct(proj['hit5'], total)} - {_pct(atom['hit5'], total)} "
                f"= {_delta(proj['hit5'], atom['hit5'], total)}  "
                f"hit@10 {_pct(proj['hit10'], total)} - {_pct(atom['hit10'], total)} "
                f"= {_delta(proj['hit10'], atom['hit10'], total)}"
            )

    print("\n=== ranked minus legacy (same arm, same pool) ===")
    for arm in _RANKED_ARMS:
        legacy = overall[arm]["legacy"]
        ranked = overall[arm]["ranked"]
        print(
            f"{arm}: "
            f"hit@5 {_pct(legacy['hit5'], total)} -> {_pct(ranked['hit5'], total)} "
            f"({_delta(ranked['hit5'], legacy['hit5'], total)})  "
            f"hit@10 {_pct(legacy['hit10'], total)} -> {_pct(ranked['hit10'], total)} "
            f"({_delta(ranked['hit10'], legacy['hit10'], total)})  "
            f"top10_order_differs={top10_differs[arm]}/{total}"
        )

    print("\n=== rrf minus ranked (cross-check) ===")
    for arm in ("A4", "B4"):
        ranked = overall[arm]["ranked"]
        rrf = overall[arm]["rrf"]
        print(
            f"{arm}: "
            f"hit@5 {_pct(ranked['hit5'], total)} -> {_pct(rrf['hit5'], total)} "
            f"({_delta(rrf['hit5'], ranked['hit5'], total)})  "
            f"hit@10 {_pct(ranked['hit10'], total)} -> {_pct(rrf['hit10'], total)} "
            f"({_delta(rrf['hit10'], ranked['hit10'], total)})"
        )

    print("\n=== per category (hit@5 / hit@10 / empty-union) ===")
    labels = [("T", "token")]
    for name, _profile, _channel, _paused, metrics in _ARMS:
        for metric in metrics:
            if metric == "rrf":
                continue
            labels.append((name, metric))
    for cat in sorted(by_cat, key=_cat_key):
        n = by_cat[cat]["T"]["n"]
        print(f"cat {cat}: n={n}")
        for arm, metric in labels:
            if arm == "T":
                counter = by_cat[cat]["T"]
                print(
                    f"  {arm} {metric}: hit@5={_pct(counter['hit5'], n)}  "
                    f"hit@10={_pct(counter['hit10'], n)}"
                )
                continue
            counter = by_cat[cat][arm][metric]
            print(
                f"  {arm} {metric}: hit@5={_pct(counter['hit5'], n)}  "
                f"hit@10={_pct(counter['hit10'], n)}  "
                f"zero={_pct(counter['zero'], n)}  "
                f"mean_union={counter['union'] / n:.2f}  "
                f"mean_pool={counter['pool'] / n:.2f}"
            )

    labels = _reading_labels()
    print("\n=== overall (any-hit / coverage / all-gold / mrr) ===")
    print("coverage and mrr are means; any-hit and all-gold are question rates.")
    for name, _profile, _channel, _paused, metrics in _ARMS:
        for metric in metrics:
            _rich_line(name, metric, overall[name][metric], total)
    _rich_line("T", "token", token, total)

    print("\n=== by gold count (all_gold@10 / coverage@10) ===")
    for bucket_name in _GOLD_BUCKETS:
        bucket = by_gold.get(bucket_name)
        n = 0 if bucket is None else bucket["T"]["n"]
        print(f"gold_n={bucket_name}: n={n}")
        if n == 0 or bucket is None:
            continue
        for arm, metric in labels:
            counter = _counter_for(bucket, arm, metric)
            print(
                f"  {arm:<10} {metric:<8} "
                f"all@10={_pct(counter['all_gold10'], n):>7} ({counter['all_gold10']})  "
                f"cov@10={_pct(counter['coverage10'], n):>7}"
            )

    print("\n=== per category (all_gold@10) ===")
    for cat in sorted(by_cat, key=_cat_key):
        n = by_cat[cat]["T"]["n"]
        print(f"cat {cat}: n={n}")
        for arm, metric in labels:
            counter = _counter_for(by_cat[cat], arm, metric)
            print(
                f"  {arm:<10} {metric:<8} "
                f"all@10={_pct(counter['all_gold10'], n):>7} ({counter['all_gold10']})"
            )
    if problems_so_far:
        print(f"\nconsistency problems seen during the run: {problems_so_far}")


def _cat_key(category: Any) -> tuple[bool, str]:
    return (category is None, str(category))


def _check_family(scored: dict[str, _Metric], arm: str, sample_id: str) -> str | None:
    metrics = list(scored.values())
    base = metrics[0]
    for metric in metrics[1:]:
        if metric.pool != base.pool or metric.union != base.union:
            return (
                f"{sample_id} {arm}: pool/union "
                f"{metric.pool}/{metric.union} != {base.pool}/{base.union}"
            )
        if set(metric.ordered) != set(base.ordered):
            return f"{sample_id} {arm}: dia sets differ across metrics"
    for metric in metrics:
        if metric.hit5 and not metric.hit10:
            return f"{sample_id} {arm}: hit@5 is not a subset of hit@10"
    return None


def _empty_overall() -> dict[str, dict[str, Counter]]:
    overall: dict[str, dict[str, Counter]] = {}
    for name, _profile, _channel, _paused, metrics in _ARMS:
        overall[name] = {metric: Counter() for metric in metrics}
    return overall


def _remember_category(
    by_cat: dict[Any, dict[str, Counter]],
    category: Any,
    scored_by_arm: dict[str, dict[str, _Metric]],
    token_hit5: bool,
    token_hit10: bool,
    token_coverage5: float,
    token_coverage10: float,
    token_all5: bool,
    token_all10: bool,
    token_mrr: float,
) -> None:
    bucket = by_cat.setdefault(category, {})
    token_counter = bucket.setdefault("T", Counter())
    token_counter["n"] += 1
    token_counter["hit5"] += int(token_hit5)
    token_counter["hit10"] += int(token_hit10)
    token_counter["coverage5"] += token_coverage5
    token_counter["coverage10"] += token_coverage10
    token_counter["all_gold5"] += int(token_all5)
    token_counter["all_gold10"] += int(token_all10)
    token_counter["mrr"] += token_mrr
    for name, _profile, _channel, _paused, metrics in _ARMS:
        arm_bucket = bucket.setdefault(name, {})
        for metric in metrics:
            counter = arm_bucket.setdefault(metric, Counter())
            _add(counter, scored_by_arm[name][metric])


def _r6e_drift(overall: dict[str, dict[str, Counter]], token: Counter, total: int) -> list[str]:
    paused = overall["P_legacy"]["legacy"]
    specs: list[tuple[str, int, int, str]] = [
        ("P_legacy hit@5", paused["hit5"], 0, _BASELINE_P_HIT5),
        ("P_legacy hit@10", paused["hit10"], 0, _BASELINE_P_HIT10),
        ("P_legacy empty-union", paused["zero"], total, _BASELINE_P_ZERO),
        ("T hit@5", token["hit5"], _R6E_T_HIT5, _BASELINE_T_HIT5),
        ("T hit@10", token["hit10"], _R6E_T_HIT10, _BASELINE_T_HIT10),
    ]
    for arm, metric, count, pct in _R6E_HIT10:
        specs.append((f"{arm} {metric} @10", overall[arm][metric]["hit10"], count, pct))
    drifts: list[str] = []
    for label, got, expected_count, expected_pct in specs:
        got_pct = _pct(got, total)
        if got != expected_count or got_pct != expected_pct:
            drifts.append(
                f"{label} got {got_pct} ({got}) expected {expected_pct} ({expected_count})"
            )
    return drifts


def _print_r6e_gate(overall: dict[str, dict[str, Counter]], token: Counter, total: int) -> None:
    paused = overall["P_legacy"]["legacy"]

    def hit10(arm: str, metric: str) -> str:
        count = overall[arm][metric]["hit10"]
        return f"{_pct(count, total)} ({count})"

    print(
        "R6e any-hit gate: "
        f"P_legacy {_pct(paused['hit5'], total)}/{_pct(paused['hit10'], total)}/"
        f"空并集 {_pct(paused['zero'], total)}; "
        f"T {_pct(token['hit5'], total)}@5 / {_pct(token['hit10'], total)}@10; "
        f"A12 ranked @10 = {hit10('A12', 'ranked')}; "
        f"A4 ranked @10 = {hit10('A4', 'ranked')}; "
        f"B4 ranked @10 = {hit10('B4', 'ranked')}; "
        f"B12 ranked @10 = {hit10('B12', 'ranked')}; "
        f"legacy A4 {hit10('A4', 'legacy')} / "
        f"A12 {hit10('A12', 'legacy')} / "
        f"B4 {hit10('B4', 'legacy')} / "
        f"B12 {hit10('B12', 'legacy')}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out", default=None, help="write per-question JSON here")
    parser.add_argument("--limit-convs", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    if FACT_LANE_PAUSED is not True:
        _fail(f"module FACT_LANE_PAUSED is {FACT_LANE_PAUSED!r}; this arm must not flip it")

    with open(args.data, encoding="utf-8") as handle:
        conversations = json.load(handle)
    if args.limit_convs:
        conversations = conversations[: args.limit_convs]

    _print_policy()
    per_question: list[dict[str, Any]] = []
    overall = _empty_overall()
    token: Counter = Counter()
    by_cat: dict[Any, dict[str, Counter]] = {}
    by_gold: dict[str, dict[str, Any]] = {}
    top10_differs: Counter = Counter()
    stats: Counter = Counter()
    problems: list[str] = []
    total_q = 0

    for conv in conversations:
        with tempfile.TemporaryDirectory(prefix=f"l1r-{conv['sample_id']}-") as work_raw:
            work = Path(work_raw)
            session_id, atoms = ingest_conversation(conv, work)
            revision, registry = build_projection(work, session_id, atoms)
            ao_to_dia = {atom["ao_id"]: atom["dia_id"] for atom in atoms}
            ao_text = {atom["ao_id"]: atom["text"] for atom in atoms}
            facades = {
                True: build_facade(
                    work, session_id, registry, revision, ao_text, fact_lane_paused=True
                ),
                False: build_facade(
                    work, session_id, registry, revision, ao_text, fact_lane_paused=False
                ),
            }
            if (
                facades[True].fact_lane_paused is not True
                or facades[False].fact_lane_paused is not False
            ):
                _fail(f"{session_id}: fact_lane_paused constructor flag did not stick")
            conv_questions = 0
            for question in conv["qa"]:
                evidence = question.get("evidence") or []
                gold = set(evidence)
                if not gold:
                    continue
                gold_n = len(gold)
                if gold_n != len(set(gold)) or gold_n != len(set(evidence)):
                    problems.append(
                        f"{conv['sample_id']}: gold_n {gold_n} != len(set(gold)) {len(set(evidence))}"
                    )
                total_q += 1
                conv_questions += 1
                keywords = _question_keywords(question["question"])
                scored_by_arm: dict[str, dict[str, _Metric]] = {}
                for name, profile, channel, paused, metrics in _ARMS:
                    hits = collect_hits(
                        facades[paused],
                        keywords,
                        profile,
                        name,
                        channel,
                        revision,
                        ao_to_dia,
                        stats,
                    )
                    scored = score_pool(hits, gold, metrics)
                    problem = _check_family(scored, name, conv["sample_id"])
                    if problem is not None:
                        problems.append(problem)
                    scored_by_arm[name] = scored
                    for metric_name, metric in scored.items():
                        _add(overall[name][metric_name], metric)
                        problem = _inconsistent(
                            conv["sample_id"],
                            f"{name} {metric_name}",
                            gold,
                            gold_n,
                            metric.ordered,
                            metric.hit5,
                            metric.hit10,
                            metric.coverage5,
                            metric.coverage10,
                            metric.all_gold5,
                            metric.all_gold10,
                            metric.mrr,
                        )
                        if problem is not None:
                            problems.append(problem)
                    if "legacy" in scored and "ranked" in scored:
                        if scored["legacy"].ordered[:10] != scored["ranked"].ordered[:10]:
                            top10_differs[name] += 1
                token_ranked = token_ranked_dias(atoms, _tokens(question["question"]))
                token_hit5 = any_hit(token_ranked, gold, 5)
                token_hit10 = any_hit(token_ranked, gold, 10)
                t_cov5, t_cov10, t_all5, t_all10, t_mrr = _coverage_bundle(token_ranked, gold)
                if token_hit5 and not token_hit10:
                    problems.append(f"{conv['sample_id']}: T hit@5 is not a subset of hit@10")
                problem = _inconsistent(
                    conv["sample_id"],
                    "T token",
                    gold,
                    gold_n,
                    token_ranked,
                    token_hit5,
                    token_hit10,
                    t_cov5,
                    t_cov10,
                    t_all5,
                    t_all10,
                    t_mrr,
                )
                if problem is not None:
                    problems.append(problem)
                token["hit5"] += int(token_hit5)
                token["hit10"] += int(token_hit10)
                token["coverage5"] += t_cov5
                token["coverage10"] += t_cov10
                token["all_gold5"] += int(t_all5)
                token["all_gold10"] += int(t_all10)
                token["mrr"] += t_mrr
                _remember_category(
                    by_cat,
                    question.get("category"),
                    scored_by_arm,
                    token_hit5,
                    token_hit10,
                    t_cov5,
                    t_cov10,
                    t_all5,
                    t_all10,
                    t_mrr,
                )
                _remember_category(
                    by_gold,
                    _gold_bucket(gold_n),
                    scored_by_arm,
                    token_hit5,
                    token_hit10,
                    t_cov5,
                    t_cov10,
                    t_all5,
                    t_all10,
                    t_mrr,
                )
                row: dict[str, Any] = {
                    "sample_id": conv["sample_id"],
                    "category": question.get("category"),
                    "question": question["question"],
                    "gold": sorted(gold),
                    "T_hit5": token_hit5,
                    "T_hit10": token_hit10,
                }
                for name, _profile, _channel, _paused, metrics in _ARMS:
                    for metric_name in metrics:
                        metric = scored_by_arm[name][metric_name]
                        prefix = f"{name}_{metric_name}"
                        row[f"{prefix}_pool"] = metric.pool
                        row[f"{prefix}_union"] = metric.union
                        row[f"{prefix}_hit5"] = metric.hit5
                        row[f"{prefix}_hit10"] = metric.hit10
                    if "legacy" in metrics and "ranked" in metrics:
                        row[f"{name}_top10_differs"] = (
                            scored_by_arm[name]["legacy"].ordered[:10]
                            != scored_by_arm[name]["ranked"].ordered[:10]
                        )
                if len(row["gold"]) != gold_n:
                    problems.append(
                        f"{conv['sample_id']}: stored gold length {len(row['gold'])} != gold_n {gold_n}"
                    )
                row["gold_n"] = gold_n
                row["T_coverage5"] = t_cov5
                row["T_coverage10"] = t_cov10
                row["T_all_gold5"] = t_all5
                row["T_all_gold10"] = t_all10
                row["T_mrr"] = t_mrr
                for name, _profile, _channel, _paused, metrics in _ARMS:
                    for metric_name in metrics:
                        metric = scored_by_arm[name][metric_name]
                        prefix = f"{name}_{metric_name}"
                        row[f"{prefix}_coverage5"] = metric.coverage5
                        row[f"{prefix}_coverage10"] = metric.coverage10
                        row[f"{prefix}_all_gold5"] = metric.all_gold5
                        row[f"{prefix}_all_gold10"] = metric.all_gold10
                        row[f"{prefix}_mrr"] = metric.mrr
                per_question.append(row)
            print(
                f"[{conv['sample_id']}] atoms={len(atoms)} blocks={len(revision.blocks)} "
                f"mode={revision.construction_mode} claims={len(revision.claims)} "
                f"cards={len(revision.blocks)} qa_with_evidence={conv_questions}",
                flush=True,
            )

    if total_q == 0:
        _fail("no questions with evidence")

    _print_tables(overall, token, by_cat, by_gold, total_q, top10_differs, len(problems))
    print(
        f"\nprojection payload_ref != card_id: {stats['payload_ne_card_id']} "
        "(dedup still uses source_pointer['card_id'])"
    )

    if args.out:
        Path(args.out).write_text(
            json.dumps(per_question, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"per-question rows -> {args.out}")

    for arm in _LIVE_ARMS:
        zero = overall[arm]["ranked"]["zero"]
        if zero == total_q:
            problems.append(
                f"{arm} ranked empty-union is 100%. The channel returned no dia; "
                "this is a wiring failure, not a score."
            )
    if problems:
        shown = problems[:8]
        extra = len(problems) - len(shown)
        suffix = f" (+{extra} more)" if extra else ""
        _fail("; ".join(shown) + suffix)

    if not args.limit_convs:
        _print_r6e_gate(overall, token, total_q)
        drifts = _r6e_drift(overall, token, total_q)
        if drifts:
            _fail("R6e any-hit drift: " + "; ".join(drifts))
        print("P/T match the frozen baseline (P 0.0%/0.0%/empty 100.0%, T 56.0%/61.8%).")
    else:
        print("baseline gate skipped (--limit-convs).")
    print("gates passed.")


if __name__ == "__main__":
    main()
