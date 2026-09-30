#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""L1 zero-LLM LoCoMo oracle arm for adaptive-view block summaries.

Oracle upper bound on the content axis only. Human-written ``observation``
text is copied into ``InfluenceBlock.summary`` and frozen through
``freeze_extracted_view`` (``construction_mode=adaptive``, no claims, no
edges). The same questions were written by the same authors, and observation
does not cover every message, so a gain here is an optimistic ceiling on
summary text. It is not evidence for cross-message structure.

Three summaries share one view shape:

  oracle        observation text for an anchored dia, message text otherwise
  oracle_only   observation text for an anchored dia, fallback summary otherwise
  placeholder   the fallback summary on every block

Each summary is scored with and without ``content_resolver``, at projection
seats 4 and 12. Ranking and metrics are the R6e/R7a functions. This file does
not reimplement them.

``--out`` is rewritten after every conversation. ``--resume`` skips a
conversation whose checkpoint already holds every evidence question.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from l1_locomo_projection_arm import (
    DISJOINT_LABEL,
    LABELS,
    build_projection,
    ingest_conversation,
)
from l1_locomo_ranked_arm import (
    FACTS_PROFILE,
    PROFILE_A4,
    PROFILE_B12,
    PROJECTION_PROFILE,
    _Hit,
    _Metric,
    _add,
    _check_family,
    _coverage_bundle,
    _fail,
    _gold_bucket,
    _inconsistent,
    _pct,
    _print_r6e_gate,
    _question_keywords,
    _r6e_drift,
    _rich_line,
    _tokens,
    any_hit,
    build_facade,
    collect_hits,
    score_pool,
    token_ranked_dias,
)

from openviking.session.extraction_write import freeze_extracted_view
from openviking.session.influence_projection import ProjectionRegistry
from openviking.session.influence_view import (
    DETERMINISTIC_FALLBACK_REASON,
    PURPOSE_POST_RUN_INDEX,
    AORange,
    CoverageManifest,
    HighSeverityViolation,
    InfluenceBlock,
    InfluenceViewRevision,
)
from openviking.session.retrieval_facade import (
    FACT_LANE_PAUSED,
    ForkBranchAdapter,
    HybridRetrievalFacade,
    InfluenceProjectionAdapter,
    SegmentAtomAdapter,
)

SCHEMA = "r7c-oracle-v1"
FROZEN_AT = "2026-09-30T00:00:00.000Z"
METRICS = ("legacy", "ranked")
VARIANTS = ("oracle", "oracle_only", "placeholder")
CONFIGS = ("no_resolver", "with_resolver")
SEATS = (4, 12)
SEAT_PROFILE = {4: PROJECTION_PROFILE, 12: PROFILE_B12}

# R6e arm set, so ``_r6e_drift`` / ``_print_r6e_gate`` run unchanged.
# name, profile, channel, facade key, revision key
_R6E_SPECS: tuple[tuple[str, dict[str, int], str, str, str], ...] = (
    ("P_legacy", FACTS_PROFILE, "atom", "fallback_paused", "fallback"),
    ("A12", FACTS_PROFILE, "atom", "fallback_open", "fallback"),
    ("A4", PROFILE_A4, "atom", "fallback_open", "fallback"),
    ("B4", PROJECTION_PROFILE, "projection", "fallback_paused", "fallback"),
    ("B12", PROFILE_B12, "projection", "fallback_paused", "fallback"),
)

# Pins from the R7a full run. Do not edit these to absorb a miss.
_PIN_TOTAL = 1982
_PIN_A12_RANKED = {"coverage10": "54.3%", "all_gold10": "50.6%", "mrr": "0.3012"}
_PIN_T = {"coverage10": "57.2%", "all_gold10": "53.5%", "mrr": "0.4346"}
_PIN_BUCKETS: tuple[tuple[str, str, str, str, int | None], ...] = (
    # group, arm, field, percent, count (None = mean, percent only)
    ("gold:2", "A12", "all_gold10", "17.1%", 41),
    ("gold:2", "A12", "coverage10", "34.2%", None),
    ("gold:>=3", "A12", "all_gold10", "1.1%", 2),
    ("gold:>=3", "A12", "coverage10", "18.3%", None),
    ("cat:1", "A12", "all_gold10", "9.2%", 26),
    ("cat:1", "A12", "coverage10", "24.8%", None),
    ("gold:2", "T", "all_gold10", "17.5%", 42),
    ("gold:2", "T", "coverage10", "33.5%", None),
    ("gold:>=3", "T", "all_gold10", "0.0%", 0),
    ("gold:>=3", "T", "coverage10", "19.0%", None),
    ("cat:1", "T", "all_gold10", "9.6%", 27),
    ("cat:1", "T", "coverage10", "24.1%", None),
)
_PIN_GROUP_N = {"gold:1": 1559, "gold:2": 240, "gold:>=3": 183, "cat:1": 282}

_ORACLE_NOTE = (
    "Oracle upper bound on the content axis. Observations are human-written "
    "and share authors with the questions, so wording overlap makes the "
    "numbers optimistic. This view has no claims and no edges, so the numbers "
    "do not measure structural (cross-message) value."
)


def _oracle_arm_name(variant: str, config: str, seat: int) -> str:
    return f"{variant}.{config}.{seat}"


def _oracle_specs() -> tuple[tuple[str, dict[str, int], str, str, str], ...]:
    specs = []
    for variant in VARIANTS:
        for config in CONFIGS:
            for seat in SEATS:
                specs.append(
                    (
                        _oracle_arm_name(variant, config, seat),
                        SEAT_PROFILE[seat],
                        "projection",
                        f"{variant}.{config}",
                        variant,
                    )
                )
    return tuple(specs)


def _all_specs() -> tuple[tuple[str, dict[str, int], str, str, str], ...]:
    return _R6E_SPECS + _oracle_specs()


def _print_policy() -> None:
    print(
        "ORACLE UPPER BOUND (content axis only). "
        "Human observations fill adaptive block summaries. "
        "No claims, no edges. Not a structural result."
    )
    print(
        "write path: freeze_extracted_view purpose=post_run_index "
        "construction_mode=adaptive. "
        f"granularity_reason={DETERMINISTIC_FALLBACK_REASON} "
        "(copied from the fallback block the library already built). "
        "revision_id=r7c:{variant}:{session_id} (fixed, not a uuid). "
        f"frozen_at={FROZEN_AT}."
    )
    print(
        "summary: multiple observations for one dia are joined with a newline "
        "in file appearance order. Unanchored oracle blocks use the message "
        "text. Unanchored oracle_only blocks and every placeholder block use "
        "the fallback block summary (segment <segment_id> sequences a-b). "
        "session_summary is not written into any document."
    )
    print(
        "configs: no_resolver = InfluenceProjectionAdapter(registry, views=[revision]) "
        "with no content_resolver. with_resolver = build_facade (AO body in the "
        "BM25 document only). Seats 4 and 12 use the R7a projection profiles. "
        f"Oracle facades pass fact_lane_paused=False (module FACT_LANE_PAUSED="
        f"{FACT_LANE_PAUSED} is not assigned). segment_atom is 0 on those "
        "profiles, so the open fact lane adds no candidate."
    )
    print(
        "anchors: T, A12, A4, B4, B12 (fallback view + resolver, fact lane "
        "paused on P/B) go through _r6e_drift / _print_r6e_gate. "
        "placeholder+no_resolver must return 0 cards."
    )
    print(
        "metrics: imported score_pool / _coverage_bundle / _inconsistent / "
        "_check_family. ranked = max BM25; legacy = first-seen. Same pool."
    )


def _audit_observations(conversations: list[dict]) -> dict[str, Any]:
    """Dataset-level observation audit.

    ``shared_anchor_dias`` counts un-namespaced ``dia_id`` strings whose
    valid observation entries occur more than once anywhere in the file.
    The same string in two conversations counts as one shared id. That is
    the 538 figure. Concatenation itself is per conversation.
    """
    global_count: Counter[str] = Counter()
    malformed = 0
    entries = 0
    unmatched_entries = 0
    within_multi = 0
    messages = 0
    anchored_messages = 0
    per_conv: list[dict[str, Any]] = []
    for conv in conversations:
        obs = conv.get("observation")
        if not isinstance(obs, dict):
            _fail(f"{conv.get('sample_id')}: observation is not an object")
        by_dia, conv_malformed = _observation_index(obs)
        malformed += conv_malformed
        conv_entries = sum(len(texts) for texts in by_dia.values())
        entries += conv_entries
        for dia, texts in by_dia.items():
            global_count[dia] += len(texts)
        within_multi += sum(1 for texts in by_dia.values() if len(texts) > 1)
        message_ids = _message_ids(conv)
        anchored = sum(1 for dia in message_ids if dia in by_dia)
        unmatched_entries += sum(len(texts) for dia, texts in by_dia.items() if dia not in message_ids)
        messages += len(message_ids)
        anchored_messages += anchored
        per_conv.append(
            {
                "sample_id": conv["sample_id"],
                "messages": len(message_ids),
                "anchored_messages": anchored,
                "distinct_anchor_dias": len(by_dia),
                "observation_entries": conv_entries,
                "malformed_skipped": conv_malformed,
                "within_conv_shared_dias": sum(1 for texts in by_dia.values() if len(texts) > 1),
            }
        )
    distinct = len(global_count)
    shared = sum(1 for count in global_count.values() if count > 1)
    return {
        "oracle_upper_bound": True,
        "axis": "content",
        "note": _ORACLE_NOTE,
        "distinct_anchor_dias": distinct,
        "shared_anchor_dias": shared,
        "shared_anchor_dias_definition": (
            "un-namespaced dia_id strings with more than one shape-valid "
            "observation entry across the whole file"
        ),
        "within_conversation_shared_dias": within_multi,
        "malformed_skipped": malformed,
        "malformed_rule": "skip an entry whose length is not 2 or whose second item is not a str",
        "observation_entries": entries,
        "messages": messages,
        "anchored_messages": anchored_messages,
        "anchored_message_rate": anchored_messages / messages if messages else 0.0,
        "shape_valid_unmatched_entries": unmatched_entries,
        "summary_join": "newline, file appearance order",
        "unanchored_oracle": "message text",
        "unanchored_oracle_only": "fallback summary",
        "placeholder": "fallback summary on every block",
        "session_summary_used": False,
        "granularity_reason": DETERMINISTIC_FALLBACK_REASON,
        "per_conversation": per_conv,
    }


def _observation_index(obs: dict[str, Any]) -> tuple[dict[str, list[str]], int]:
    """dia_id -> observation texts in appearance order, and the skip count."""
    by_dia: dict[str, list[str]] = {}
    malformed = 0
    for speakers in obs.values():
        if not isinstance(speakers, dict):
            malformed += 1
            continue
        for items in speakers.values():
            if not isinstance(items, list):
                malformed += 1
                continue
            for item in items:
                if not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[1], str):
                    malformed += 1
                    continue
                text = item[0]
                if not isinstance(text, str):
                    malformed += 1
                    continue
                by_dia.setdefault(item[1], []).append(text)
    return by_dia, malformed


def _message_ids(conv: dict) -> list[str]:
    ids: list[str] = []
    conversation = conv["conversation"]
    sessions = sorted(
        ((key, value) for key, value in conversation.items() if re.fullmatch(r"session_\d+", key)),
        key=lambda item: int(item[0].split("_")[1]),
    )
    for _name, messages in sessions:
        for message in messages:
            ids.append(message["dia_id"])
    return ids


def _coverage_for(blocks: list[InfluenceBlock], session_id: str) -> CoverageManifest:
    refs = tuple(
        sorted(
            (ref for block in blocks for ref in block.source_refs),
            key=lambda ref: (ref.sequence, ref.ao_id),
        )
    )
    if len(refs) != len(blocks):
        _fail(f"{session_id}: coverage refs {len(refs)} != blocks {len(blocks)}")
    sequences = [ref.sequence for ref in refs]
    if len(set(sequences)) != len(sequences):
        _fail(f"{session_id}: AO sequences are not unique")
    start, end = min(sequences), max(sequences)
    if sorted(sequences) != list(range(start, end + 1)):
        _fail(f"{session_id}: AO sequences are not contiguous ({start}..{end})")
    included = tuple(
        sorted(
            (
                AORange(
                    start_seq=block.ao_start_seq,
                    end_seq=block.ao_end_seq,
                    segment_id=block.segment_ref.segment_id,
                )
                for block in blocks
            ),
            key=lambda item: (item.start_seq, item.segment_id or ""),
        )
    )
    return CoverageManifest(
        total_ao_range=AORange(start_seq=start, end_seq=end),
        inspected_refs=refs,
        included_ranges=included,
        omitted_ranges=(),
        snapshot_watermark=f"r7c:{session_id}:{end}",
    )


def _summary_for(
    variant: str,
    fallback_summary: str,
    dia_id: str,
    message_text: str,
    by_dia: dict[str, list[str]],
) -> str:
    texts = by_dia.get(dia_id) or []
    if variant == "placeholder":
        return fallback_summary
    if texts:
        return "\n".join(texts)
    if variant == "oracle":
        return message_text
    if variant == "oracle_only":
        return fallback_summary
    _fail(f"unknown summary variant {variant}")
    return ""


def _adaptive_blocks(
    session_id: str,
    variant: str,
    fallback: InfluenceViewRevision,
    atoms_by_ao: dict[str, dict],
    by_dia: dict[str, list[str]],
) -> list[InfluenceBlock]:
    blocks: list[InfluenceBlock] = []
    for source in fallback.blocks:
        if len(source.source_refs) != 1:
            _fail(
                f"{session_id}: fallback block {source.block_id} covers "
                f"{len(source.source_refs)} AOs; this arm is one message per block"
            )
        if source.granularity_reason != DETERMINISTIC_FALLBACK_REASON:
            _fail(
                f"{session_id}: fallback granularity_reason={source.granularity_reason!r}, "
                f"expected {DETERMINISTIC_FALLBACK_REASON!r}"
            )
        ref = source.source_refs[0]
        atom = atoms_by_ao.get(ref.ao_id)
        if atom is None:
            _fail(f"{session_id}: fallback AO {ref.ao_id} is not an ingested message")
        if atom["segment_id"] != ref.segment_id or atom["dia_id"] not in ref.segment_id:
            _fail(f"{session_id}: AO {ref.ao_id} segment does not match the ingested dia")
        message_text = atom["text"]
        if not isinstance(message_text, str):
            _fail(f"{session_id}: message text for {atom['dia_id']} is not a str")
        summary = _summary_for(variant, source.summary, atom["dia_id"], message_text, by_dia)
        blocks.append(
            InfluenceBlock(
                block_id=f"r7c-{ref.segment_id}-{ref.sequence}",
                role=source.role,
                participant_id=source.participant_id,
                segment_ref=source.segment_ref,
                ao_start_seq=source.ao_start_seq,
                ao_end_seq=source.ao_end_seq,
                summary=summary,
                input=source.input,
                output=source.output,
                authorship=source.authorship,
                granularity_reason=source.granularity_reason,
                source_content_hashes=source.source_content_hashes,
                source_refs=source.source_refs,
            )
        )
    return blocks


def _freeze_variant(
    work: Path,
    session_id: str,
    variant: str,
    blocks: list[InfluenceBlock],
    coverage: CoverageManifest,
    n_messages: int,
) -> tuple[InfluenceViewRevision, ProjectionRegistry]:
    try:
        revision = freeze_extracted_view(
            session_id=session_id,
            task_run_id=f"l1-{session_id}",
            view_id=f"{session_id}:{PURPOSE_POST_RUN_INDEX}",
            revision_id=f"r7c:{variant}:{session_id}",
            frozen_at=FROZEN_AT,
            blocks=blocks,
            claims=(),
            coverage=coverage,
            retrieval_subgraphs=(),
        )
    except (HighSeverityViolation, TypeError, ValueError) as exc:
        _fail(f"{session_id} {variant}: freeze_extracted_view refused: {exc}")
    if revision.construction_mode != "adaptive":
        _fail(f"{session_id} {variant}: construction_mode={revision.construction_mode!r}")
    if revision.purpose != PURPOSE_POST_RUN_INDEX:
        _fail(f"{session_id} {variant}: purpose={revision.purpose!r}")
    if revision.claims:
        _fail(f"{session_id} {variant}: expected no claims, got {len(revision.claims)}")
    if revision.retrieval_subgraphs:
        _fail(f"{session_id} {variant}: expected no subgraphs")
    if len(revision.blocks) != n_messages:
        _fail(
            f"{session_id} {variant}: blocks {len(revision.blocks)} != messages {n_messages}"
        )
    registry = ProjectionRegistry(work / f"projections-{variant}.jsonl")
    cards = registry.register(revision, acl_labels=tuple(LABELS))
    visible = registry.search(principal_labels=tuple(LABELS))
    hidden = registry.search(principal_labels=(DISJOINT_LABEL,))
    if not visible:
        _fail(f"{session_id} {variant}: registry.search(locomo) is empty")
    if len(cards) != n_messages or len(visible) != n_messages:
        _fail(
            f"{session_id} {variant}: cards={len(cards)} search(locomo)={len(visible)} "
            f"messages={n_messages}"
        )
    if hidden:
        _fail(f"{session_id} {variant}: disjoint label returned {len(hidden)} cards")
    return revision, registry


def _unresolved_facade(
    work: Path,
    session_id: str,
    registry: ProjectionRegistry,
    revision: InfluenceViewRevision,
) -> HybridRetrievalFacade:
    """Production document: summary, card id, semantic type. No resolver."""
    return HybridRetrievalFacade(
        fork_branch=ForkBranchAdapter(work),
        projection=InfluenceProjectionAdapter(registry, views=[revision]),
        segment_atom=SegmentAtomAdapter(work / session_id, session_id),
        fact_lane_paused=False,
    )


def _pack_metric(metric: _Metric) -> dict[str, Any]:
    return {
        "hit5": metric.hit5,
        "hit10": metric.hit10,
        "coverage5": metric.coverage5,
        "coverage10": metric.coverage10,
        "all_gold5": metric.all_gold5,
        "all_gold10": metric.all_gold10,
        "mrr": metric.mrr,
        "top10": list(metric.ordered[:10]),
    }


def _metric_from_packed(blob: dict[str, Any], pool: int, union: int) -> _Metric:
    return _Metric(
        pool=pool,
        union=union,
        hit5=bool(blob["hit5"]),
        hit10=bool(blob["hit10"]),
        ordered=tuple(blob["top10"]),
        coverage5=float(blob["coverage5"]),
        coverage10=float(blob["coverage10"]),
        all_gold5=bool(blob["all_gold5"]),
        all_gold10=bool(blob["all_gold10"]),
        mrr=float(blob["mrr"]),
    )


def _empty_overall(specs: tuple[tuple[str, dict[str, int], str, str, str], ...]) -> dict[str, dict[str, Counter]]:
    return {name: {metric: Counter() for metric in METRICS} for name, *_rest in specs}


def _bucket() -> dict[str, Any]:
    bucket: dict[str, Any] = {"T": Counter(), "n": 0}
    return bucket


def _arm_bucket(bucket: dict[str, Any], arm: str) -> dict[str, Counter]:
    found = bucket.get(arm)
    if found is None:
        found = {metric: Counter() for metric in METRICS}
        bucket[arm] = found
    return found


def _add_token(counter: Counter, blob: dict[str, Any]) -> None:
    counter["hit5"] += int(blob["hit5"])
    counter["hit10"] += int(blob["hit10"])
    counter["coverage5"] += blob["coverage5"]
    counter["coverage10"] += blob["coverage10"]
    counter["all_gold5"] += int(blob["all_gold5"])
    counter["all_gold10"] += int(blob["all_gold10"])
    counter["mrr"] += blob["mrr"]


def _recheck_stored(row: dict[str, Any], problems: list[str]) -> None:
    """Re-run ``_inconsistent`` from the stored top 10.

    Coverage, all-gold, and mrr only read the top 10, so a resumed row can
    be checked without the full order. ``_check_family`` needs the full dia
    set and already ran when the row was scored.
    """
    sample_id = row["sample_id"]
    gold = set(row["gold"])
    gold_n = row["gold_n"]
    token_blob = row["T"]
    problem = _inconsistent(
        sample_id,
        "T token",
        gold,
        gold_n,
        token_blob["top10"],
        bool(token_blob["hit5"]),
        bool(token_blob["hit10"]),
        float(token_blob["coverage5"]),
        float(token_blob["coverage10"]),
        bool(token_blob["all_gold5"]),
        bool(token_blob["all_gold10"]),
        float(token_blob["mrr"]),
    )
    if problem is not None:
        problems.append(problem)
    for name, stored in row["arms"].items():
        pool = int(stored["pool"])
        union = int(stored["union"])
        for metric_name in METRICS:
            metric = _metric_from_packed(stored[metric_name], pool, union)
            problem = _inconsistent(
                sample_id,
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


def _fold_row(
    row: dict[str, Any],
    specs: tuple[tuple[str, dict[str, int], str, str, str], ...],
    overall: dict[str, dict[str, Counter]],
    token: Counter,
    by_cat: dict[Any, dict[str, Any]],
    by_gold: dict[str, dict[str, Any]],
    cards_returned: Counter,
) -> None:
    gold_n = row["gold_n"]
    category = row["category"]
    token_blob = row["T"]
    _add_token(token, token_blob)
    cat_bucket = by_cat.setdefault(category, _bucket())
    gold_bucket = by_gold.setdefault(_gold_bucket(gold_n), _bucket())
    for bucket in (cat_bucket, gold_bucket):
        bucket["n"] += 1
        _add_token(bucket["T"], token_blob)
    for name, _profile, _channel, _facade, _revision in specs:
        stored = row["arms"][name]
        pool = int(stored["pool"])
        union = int(stored["union"])
        n_cards = int(stored["n_cards"])
        cards_returned[name] += n_cards
        for metric_name in METRICS:
            metric = _metric_from_packed(stored[metric_name], pool, union)
            _add(overall[name][metric_name], metric)
            for bucket in (cat_bucket, gold_bucket):
                _add(_arm_bucket(bucket, name)[metric_name], metric)


def _score_question(
    question: dict[str, Any],
    sample_id: str,
    atoms: list[dict],
    facades: dict[str, HybridRetrievalFacade],
    revisions: dict[str, InfluenceViewRevision],
    ao_to_dia: dict[str, str],
    specs: tuple[tuple[str, dict[str, int], str, str, str], ...],
    stats: Counter,
    problems: list[str],
) -> dict[str, Any]:
    evidence = question.get("evidence") or []
    gold = set(evidence)
    gold_n = len(gold)
    if gold_n != len(set(evidence)):
        problems.append(f"{sample_id}: gold_n {gold_n} != len(set(evidence))")
    keywords = _question_keywords(question["question"])
    arms: dict[str, Any] = {}
    for name, profile, channel, facade_key, revision_key in specs:
        hits: list[_Hit] = collect_hits(
            facades[facade_key],
            keywords,
            profile,
            name,
            channel,
            revisions[revision_key],
            ao_to_dia,
            stats,
        )
        scored = score_pool(hits, gold, METRICS)
        problem = _check_family(scored, name, sample_id)
        if problem is not None:
            problems.append(problem)
        pool = scored["ranked"].pool
        union = scored["ranked"].union
        for metric_name, metric in scored.items():
            problem = _inconsistent(
                sample_id,
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
            if metric.pool != pool or metric.union != union:
                problems.append(f"{sample_id} {name}: metric {metric_name} pool/union drifted")
        arms[name] = {
            "pool": pool,
            "union": union,
            "n_cards": len(hits),
            "legacy": _pack_metric(scored["legacy"]),
            "ranked": _pack_metric(scored["ranked"]),
        }
    token_ranked = token_ranked_dias(atoms, _tokens(question["question"]))
    token_hit5 = any_hit(token_ranked, gold, 5)
    token_hit10 = any_hit(token_ranked, gold, 10)
    coverage5, coverage10, all_gold5, all_gold10, mrr = _coverage_bundle(token_ranked, gold)
    if token_hit5 and not token_hit10:
        problems.append(f"{sample_id}: T hit@5 is not a subset of hit@10")
    problem = _inconsistent(
        sample_id,
        "T token",
        gold,
        gold_n,
        token_ranked,
        token_hit5,
        token_hit10,
        coverage5,
        coverage10,
        all_gold5,
        all_gold10,
        mrr,
    )
    if problem is not None:
        problems.append(problem)
    if len(sorted(gold)) != gold_n:
        problems.append(f"{sample_id}: stored gold length != gold_n {gold_n}")
    return {
        "sample_id": sample_id,
        "category": question.get("category"),
        "question": question["question"],
        "gold": sorted(gold),
        "gold_n": gold_n,
        "T": {
            "hit5": token_hit5,
            "hit10": token_hit10,
            "coverage5": coverage5,
            "coverage10": coverage10,
            "all_gold5": all_gold5,
            "all_gold10": all_gold10,
            "mrr": mrr,
            "top10": list(token_ranked[:10]),
        },
        "arms": arms,
    }


def _probe_card_keyword(
    facade: HybridRetrievalFacade,
    label: str,
) -> int:
    """How many cards the production document returns for the token ``card``.

    Card ids are ``block-card-<hex>``. The token ``card`` is therefore in
    every BM25 document even when the summary is the fallback placeholder
    and no resolver is attached. This probe is not a scored question.
    """
    result = facade.retrieve("card", principal_labels=LABELS, profile=PROJECTION_PROFILE)
    count = len(result.cards)
    print(f"  probe {label} keyword 'card' -> {count} cards", flush=True)
    return count


def _load_checkpoint(path: Path, resume: bool) -> dict[str, Any]:
    if not path.exists():
        if resume:
            print(f"resume: no checkpoint at {path}, starting fresh", flush=True)
        return {"schema": SCHEMA, "conversations": {}, "runs": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        _fail(f"checkpoint schema is not {SCHEMA}; refusing to mix arm sets")
    if not isinstance(payload.get("conversations"), dict):
        _fail("checkpoint conversations is not an object")
    if not resume:
        _fail(f"{path} already exists; pass --resume to continue it")
    return payload


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _question_count(conv: dict) -> int:
    return sum(1 for question in conv["qa"] if question.get("evidence"))


def _is_complete(entry: Any, n_questions: int, arm_names: set[str]) -> bool:
    if not isinstance(entry, dict) or entry.get("complete") is not True:
        return False
    rows = entry.get("rows")
    if not isinstance(rows, list) or len(rows) != n_questions:
        return False
    if n_questions == 0:
        return True
    arms = rows[0].get("arms")
    return isinstance(arms, dict) and arm_names <= set(arms)


def _print_tables(
    specs: tuple[tuple[str, dict[str, int], str, str, str], ...],
    overall: dict[str, dict[str, Counter]],
    token: Counter,
    by_cat: dict[Any, dict[str, Any]],
    by_gold: dict[str, dict[str, Any]],
    total: int,
    cards_returned: Counter,
    disagreements: int,
) -> None:
    if total == 0:
        _fail("no questions with evidence")
    print("\n=== anchors (any-hit / coverage / all-gold / mrr) ===")
    print("coverage and mrr are means; any-hit and all-gold are question rates.")
    for name, _profile, _channel, _facade, _revision in _R6E_SPECS:
        for metric in METRICS:
            if name == "P_legacy" and metric != "legacy":
                continue
            _rich_line(name, metric, overall[name][metric], total)
    _rich_line("T", "token", token, total)

    print("\n=== main (variant x config x seat, ranked) ===")
    for name, _profile, _channel, _facade, _revision in _oracle_specs():
        _rich_line(name, "ranked", overall[name]["ranked"], total)

    print("\n=== same arms, legacy (first-seen order of the same pool) ===")
    for name, _profile, _channel, _facade, _revision in _oracle_specs():
        _rich_line(name, "legacy", overall[name]["legacy"], total)

    labels = [("T", "token", token)]
    for name, _profile, _channel, _facade, _revision in specs:
        if name in ("P_legacy", "A4"):
            continue
        labels.append((name, "ranked", overall[name]["ranked"]))

    print("\n=== by gold count (ranked all_gold@10 / coverage@10 / any-hit@10) ===")
    for bucket_name in ("1", "2", ">=3"):
        bucket = by_gold.get(bucket_name)
        n = 0 if bucket is None else bucket["n"]
        print(f"gold_n={bucket_name}: n={n}")
        if n == 0 or bucket is None:
            continue
        _print_group(labels, bucket, n)

    merged = _merge_gold(by_gold)
    n_multi = merged["n"]
    print(f"gold_n>=2: n={n_multi}")
    if n_multi:
        _print_group(labels, merged, n_multi)

    print("\n=== cat 1 multi-hop (ranked all_gold@10 / coverage@10 / any-hit@10) ===")
    cat1 = by_cat.get(1)
    n_cat = 0 if cat1 is None else cat1["n"]
    print(f"cat 1: n={n_cat}")
    if cat1 is not None and n_cat:
        _print_group(labels, cat1, n_cat)

    print("\n=== placeholder no_resolver cards returned ===")
    for seat in SEATS:
        name = _oracle_arm_name("placeholder", "no_resolver", seat)
        print(f"  {name}: {cards_returned[name]}")
    print(f"placeholder with_resolver vs B, ranked top10 differs on {disagreements} questions")


def _print_group(
    labels: list[tuple[str, str, Counter]],
    bucket: dict[str, Any],
    n: int,
) -> None:
    for arm, metric, _counter in labels:
        counter = bucket["T"] if arm == "T" else bucket[arm][metric]
        print(
            f"  {arm:<28} {metric:<8} "
            f"all@10={_pct(counter['all_gold10'], n):>7} ({int(counter['all_gold10'])})  "
            f"cov@10={_pct(counter['coverage10'], n):>7}  "
            f"hit@10={_pct(counter['hit10'], n):>7}"
        )


def _merge_gold(by_gold: dict[str, dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {"T": Counter(), "n": 0}
    for bucket_name in ("2", ">=3"):
        bucket = by_gold.get(bucket_name)
        if not bucket:
            continue
        merged["n"] += bucket["n"]
        merged["T"] += bucket["T"]
        for key, value in bucket.items():
            if key in ("T", "n"):
                continue
            found = merged.get(key)
            if found is None:
                merged[key] = {metric: Counter(value[metric]) for metric in METRICS}
                continue
            for metric in METRICS:
                found[metric] += value[metric]
    return merged


def _mean_pin(drifts: list[str], label: str, counter: Counter, total: int, expected: dict[str, str]) -> None:
    got = {
        "coverage10": _pct(counter["coverage10"], total),
        "all_gold10": _pct(counter["all_gold10"], total),
        "mrr": f"{counter['mrr'] / total:.4f}",
    }
    for key, want in expected.items():
        if got[key] != want:
            drifts.append(f"{label} {key} got {got[key]} expected {want}")


def _group_counter(group: str, arm: str, by_gold: dict[str, dict[str, Any]], by_cat: dict[Any, dict[str, Any]]) -> tuple[Counter, int]:
    if group.startswith("gold:"):
        bucket_name = group.split(":", 1)[1]
        bucket = by_gold.get(bucket_name)
    else:
        bucket = by_cat.get(1)
    if bucket is None:
        _fail(f"missing group {group}")
    n = int(bucket["n"])
    if arm == "T":
        return bucket["T"], n
    return bucket[arm]["ranked"], n


def _section4_drift(
    overall: dict[str, dict[str, Counter]],
    token: Counter,
    total: int,
    by_gold: dict[str, dict[str, Any]],
    by_cat: dict[Any, dict[str, Any]],
    cards_returned: Counter,
) -> list[str]:
    drifts = _r6e_drift(overall, token, total)
    if total != _PIN_TOTAL:
        drifts.append(f"questions got {total} expected {_PIN_TOTAL}")
    _mean_pin(drifts, "A12 ranked", overall["A12"]["ranked"], total, _PIN_A12_RANKED)
    _mean_pin(drifts, "T", token, total, _PIN_T)
    for group, arm, field, percent, count in _PIN_BUCKETS:
        counter, n = _group_counter(group, arm, by_gold, by_cat)
        want_n = _PIN_GROUP_N[group]
        if n != want_n:
            drifts.append(f"{group} n got {n} expected {want_n}")
            continue
        got_pct = _pct(counter[field], n)
        got_count = counter[field]
        if count is None:
            if got_pct != percent:
                drifts.append(f"{group} {arm} {field} got {got_pct} expected {percent}")
        elif int(got_count) != count or got_pct != percent:
            drifts.append(
                f"{group} {arm} {field} got {got_pct} ({got_count}) expected {percent} ({count})"
            )
    for seat in SEATS:
        name = _oracle_arm_name("placeholder", "no_resolver", seat)
        got = cards_returned[name]
        if got != 0:
            drifts.append(
                f"{name} retrieved {got} cards; expected 0 "
                "(a non-zero count means some question keyword is a BM25 token of "
                "the fallback summary, the card id, or the semantic type)"
            )
    return drifts


def _placeholder_card_questions(rows: list[dict[str, Any]]) -> list[str]:
    """Questions where placeholder+no_resolver returned any card."""
    found: list[str] = []
    for row in rows:
        for seat in SEATS:
            name = _oracle_arm_name("placeholder", "no_resolver", seat)
            if int(row["arms"][name]["n_cards"]) <= 0:
                continue
            found.append(
                f"{row['sample_id']} seat {seat} n_cards={row['arms'][name]['n_cards']}: "
                f"{row['question']}"
            )
    return found


def _top10_disagreements(rows: list[dict[str, Any]]) -> int:
    differs = 0
    pairs = (
        (_oracle_arm_name("placeholder", "with_resolver", 4), "B4"),
        (_oracle_arm_name("placeholder", "with_resolver", 12), "B12"),
    )
    for row in rows:
        for left, right in pairs:
            if row["arms"][left]["ranked"]["top10"] != row["arms"][right]["ranked"]["top10"]:
                differs += 1
    return differs


def _print_annotation(audit: dict[str, Any]) -> None:
    rate = audit["anchored_message_rate"]
    print(
        "annotation: "
        f"distinct_anchor_dias={audit['distinct_anchor_dias']} "
        f"shared_anchor_dias={audit['shared_anchor_dias']} "
        f"within_conversation_shared_dias={audit['within_conversation_shared_dias']} "
        f"malformed_skipped={audit['malformed_skipped']} "
        f"anchored_messages={audit['anchored_messages']}/{audit['messages']} "
        f"({rate:.1%}) "
        f"shape_valid_unmatched_entries={audit['shape_valid_unmatched_entries']}"
    )
    print("session_summary_used=false")
    print(_ORACLE_NOTE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="/home/zhoujie22/river2_0/evol_bench/LoCoMo/data/locomo/locomo10.json")
    parser.add_argument("--out", required=True, help="checkpoint JSON, rewritten after each conversation")
    parser.add_argument("--limit-convs", type=int, default=0, help="0 = all")
    parser.add_argument("--resume", action="store_true", help="skip conversations already complete in --out")
    args = parser.parse_args()
    started = time.perf_counter()

    if FACT_LANE_PAUSED is not True:
        _fail(f"module FACT_LANE_PAUSED is {FACT_LANE_PAUSED!r}; this arm must not flip it")

    with open(args.data, encoding="utf-8") as handle:
        conversations = json.load(handle)
    audit = _audit_observations(conversations)
    if (
        audit["malformed_skipped"] != 10
        or audit["distinct_anchor_dias"] != 750
        or audit["shared_anchor_dias"] != 538
    ):
        _fail(
            "observation audit mismatch: "
            f"malformed={audit['malformed_skipped']} distinct={audit['distinct_anchor_dias']} "
            f"shared={audit['shared_anchor_dias']} "
            "(expected 10 / 750 / 538)"
        )

    _print_policy()
    _print_annotation(audit)

    out_path = Path(args.out)
    payload = _load_checkpoint(out_path, args.resume)
    payload["schema"] = SCHEMA
    payload["oracle_upper_bound"] = True
    payload["annotation"] = audit
    payload.setdefault("runs", [])
    payload.setdefault("conversations", {})

    selected = conversations[: args.limit_convs] if args.limit_convs else conversations
    specs = _all_specs()
    arm_names = {name for name, *_rest in specs}
    overall = _empty_overall(specs)
    # P_legacy only has legacy. Give it a ranked counter so the dict stays uniform;
    # _r6e_drift reads P_legacy legacy only. Scoring still requests both metrics
    # from one pool, which _check_family compares.
    token: Counter = Counter()
    by_cat: dict[Any, dict[str, Any]] = {}
    by_gold: dict[str, dict[str, Any]] = {}
    cards_returned: Counter = Counter()
    problems: list[str] = []
    stats: Counter = Counter()
    folded_rows: list[dict[str, Any]] = []
    probed = False

    for conv in selected:
        sample_id = conv["sample_id"]
        n_questions = _question_count(conv)
        entry = payload["conversations"].get(sample_id)
        if args.resume and _is_complete(entry, n_questions, arm_names):
            print(
                f"[{sample_id}] skipped（already done） questions={n_questions}",
                flush=True,
            )
            for row in entry["rows"]:
                _recheck_stored(row, problems)
                _fold_row(row, specs, overall, token, by_cat, by_gold, cards_returned)
                folded_rows.append(row)
            continue

        conv_started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"l1o-{sample_id}-") as work_raw:
            work = Path(work_raw)
            session_id, atoms = ingest_conversation(conv, work)
            fallback, fallback_registry = build_projection(work, session_id, atoms)
            by_dia, _malformed = _observation_index(conv["observation"])
            atoms_by_ao = {atom["ao_id"]: atom for atom in atoms}
            ao_to_dia = {atom["ao_id"]: atom["dia_id"] for atom in atoms}
            ao_text = {atom["ao_id"]: atom["text"] for atom in atoms}
            first_blocks = _adaptive_blocks(session_id, VARIANTS[0], fallback, atoms_by_ao, by_dia)
            coverage = _coverage_for(first_blocks, session_id)
            revisions: dict[str, InfluenceViewRevision] = {"fallback": fallback}
            registries: dict[str, ProjectionRegistry] = {}
            for variant in VARIANTS:
                blocks = (
                    first_blocks
                    if variant == VARIANTS[0]
                    else _adaptive_blocks(session_id, variant, fallback, atoms_by_ao, by_dia)
                )
                revision, registry = _freeze_variant(
                    work, session_id, variant, blocks, coverage, len(atoms)
                )
                revisions[variant] = revision
                registries[variant] = registry
            facades = {
                "fallback_paused": build_facade(
                    work, session_id, fallback_registry, fallback, ao_text, fact_lane_paused=True
                ),
                "fallback_open": build_facade(
                    work, session_id, fallback_registry, fallback, ao_text, fact_lane_paused=False
                ),
            }
            for variant in VARIANTS:
                facades[f"{variant}.with_resolver"] = build_facade(
                    work,
                    session_id,
                    registries[variant],
                    revisions[variant],
                    ao_text,
                    fact_lane_paused=False,
                )
                facades[f"{variant}.no_resolver"] = _unresolved_facade(
                    work, session_id, registries[variant], revisions[variant]
                )
            if facades["fallback_open"].fact_lane_paused is not False:
                _fail(f"{session_id}: open fact lane did not stick")
            if facades["fallback_paused"].fact_lane_paused is not True:
                _fail(f"{session_id}: paused fact lane did not stick")
            for variant in VARIANTS:
                for config in CONFIGS:
                    facade = facades[f"{variant}.{config}"]
                    if facade.fact_lane_paused is not False:
                        _fail(f"{session_id} {variant}.{config}: fact_lane_paused is not False")
            probe_counts = None
            if not probed:
                unresolved_fallback = _unresolved_facade(work, session_id, fallback_registry, fallback)
                probe_counts = {
                    "placeholder_no_resolver_card": _probe_card_keyword(
                        facades["placeholder.no_resolver"], "placeholder.no_resolver"
                    ),
                    "fallback_no_resolver_card": _probe_card_keyword(
                        unresolved_fallback, "fallback.no_resolver"
                    ),
                }
                probed = True
                payload["card_token_probe"] = probe_counts

            rows = []
            for question in conv["qa"]:
                if not question.get("evidence"):
                    continue
                rows.append(
                    _score_question(
                        question,
                        sample_id,
                        atoms,
                        facades,
                        revisions,
                        ao_to_dia,
                        specs,
                        stats,
                        problems,
                    )
                )
            anchored = sum(1 for atom in atoms if atom["dia_id"] in by_dia)
            entry = {
                "complete": True,
                "questions": len(rows),
                "messages": len(atoms),
                "blocks": len(revisions["oracle"].blocks),
                "construction_mode": revisions["oracle"].construction_mode,
                "anchored_messages": anchored,
                "elapsed_s": round(time.perf_counter() - conv_started, 3),
                "rows": rows,
            }
            payload["conversations"][sample_id] = entry
            _write_checkpoint(out_path, payload)
            for row in rows:
                _recheck_stored(row, problems)
                _fold_row(row, specs, overall, token, by_cat, by_gold, cards_returned)
                folded_rows.append(row)
            print(
                f"[{sample_id}] atoms={len(atoms)} blocks={len(atoms)} "
                f"mode=adaptive anchored={anchored}/{len(atoms)} "
                f"qa_with_evidence={len(rows)} elapsed={entry['elapsed_s']:.1f}s",
                flush=True,
            )

    total_q = len(folded_rows)
    disagreements = _top10_disagreements(folded_rows)
    _print_tables(
        specs, overall, token, by_cat, by_gold, total_q, cards_returned, disagreements
    )
    print(
        f"\nprojection payload_ref != card_id: {stats['payload_ne_card_id']} "
        "(dedup still uses source_pointer['card_id'])"
    )
    elapsed = time.perf_counter() - started
    payload["runs"].append(
        {
            "limit_convs": args.limit_convs,
            "resume": bool(args.resume),
            "questions": total_q,
            "elapsed_s": round(elapsed, 3),
            "placeholder_no_resolver_cards": {
                str(seat): cards_returned[_oracle_arm_name("placeholder", "no_resolver", seat)]
                for seat in SEATS
            },
            "placeholder_with_resolver_top10_disagreements": disagreements,
        }
    )
    _write_checkpoint(out_path, payload)
    print(f"checkpoint -> {out_path}")
    print(f"elapsed={elapsed:.1f}s")

    if problems:
        shown = problems[:8]
        extra = len(problems) - len(shown)
        suffix = f" (+{extra} more)" if extra else ""
        _fail("; ".join(shown) + suffix)

    full = not args.limit_convs and len(selected) == len(conversations)
    if full:
        _print_r6e_gate(overall, token, total_q)
        drifts = _section4_drift(overall, token, total_q, by_gold, by_cat, cards_returned)
        leaked = _placeholder_card_questions(folded_rows)
        if leaked:
            print("placeholder no_resolver questions that returned cards:")
            for line in leaked:
                print(f"  {line}")
        if drifts:
            _fail("R7c anchor drift: " + "; ".join(drifts))
        print(
            "anchors match R7a "
            "(T 56.0%@5/61.8%@10, A12 ranked @10 59.0%, "
            "A12 coverage@10 54.3% all_gold@10 50.6% mrr 0.3012, "
            "T coverage@10 57.2% all_gold@10 53.5% mrr 0.4346)."
        )
    else:
        print("baseline gate skipped (--limit-convs).")
        leaked = _placeholder_card_questions(folded_rows)
        if leaked:
            print("placeholder no_resolver questions that returned cards:")
            for line in leaked:
                print(f"  {line}")
        for seat in SEATS:
            name = _oracle_arm_name("placeholder", "no_resolver", seat)
            if cards_returned[name] != 0:
                _fail(
                    f"{name} retrieved {cards_returned[name]} cards on this slice; expected 0"
                )
    print("gates passed.")


if __name__ == "__main__":
    main()
