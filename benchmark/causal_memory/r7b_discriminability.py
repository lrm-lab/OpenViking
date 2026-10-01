#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""R7b: discriminability of typed fields on the diagnosis-side subgraph channel.

Lexical arm: production ``RetrievalSubgraphAdapter.retrieve_historical_subgraphs``.
Structural arm: weighted field equality on ``structural_signature``.

L1 is a fixture. Discriminability is not an effect size. L2 reports whether
the two arms order the real R8a revisions differently, and which fields
separate the structural scores. No model is called. No dependency is added.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Frozen before scores. Do not edit these to absorb a miss.
SCHEMA = "r7b-discriminability-v1"
LABEL_L1 = "基于夹具，判别力 ≠ 效果量"
WORKSPACE_LABEL = "sb-skillsbench"
PRINCIPAL = (WORKSPACE_LABEL,)
FROZEN_AT = "2026-01-02T00:00:00.000Z"
CHECKPOINT_DIR = Path("/tmp/sb-influence-r8b/checkpoints")
NARROW_CODES = ("0001", "0003", "0008", "0010", "0012", "0014")
EPISODE_CODES = tuple(f"{i:04d}" for i in range(1, 15))
SHUFFLE_SEED = 20261001
SHUFFLE_REPEATS = 1000
SKELETON_BLOCKS = 4
SKELETON_CLAIMS = 3
PROSE_TOKEN_MIN = 8

# Axis order and poles are frozen. Positive is the pole the query carries.
AXES: tuple[dict[str, Any], ...] = (
    {
        "id": "A",
        "field": "boundary",
        "negative": "call_cut",
        "positive": "result_cut",
    },
    {
        "id": "B",
        "field": "anchor_incidence",
        "negative": "skip_only",
        "positive": "influence_both",
    },
    {
        "id": "C",
        "field": "coverage_crossing",
        "negative": False,
        "positive": True,
    },
    {
        "id": "D",
        "field": "overlaps_anchor",
        "negative": False,
        "positive": True,
    },
)

FIXTURE_RULE = (
    "L1 pairs are rule-generated from the narrow R8a revisions, sorted by "
    "episode_dir. Sources are exactly NARROW_CODES. For each source and each "
    "axis in AXES order, copy the first 4 subgraph blocks' summary/input/output "
    "and the first 3 subgraph claims' carried_artifact/downstream_effect onto a "
    "4-block skeleton. Both poles and the query receive those same bytes. "
    "Axis A flips only the last claim's relation_type (influence vs skip) with "
    "the anchor on block index 1. Axis B keeps relations "
    "(influence, influence, skip) and moves the failure anchor from index 1 "
    "to index 3. Axis C keeps that same claim pattern and anchor index 1, and "
    "adds or omits an OmittedRange at sequence 5. Axis D keeps that structure "
    "with no omitted range; the anchor is the lowest index >= 1 whose copied "
    "summary is non-empty, and the fork branch ao_sequence either contains "
    "that block's ao_id or the token ao-off. No pair is added or dropped "
    "after this rule runs."
)
SHUFFLE_RULE = (
    f"Shuffle seed {SHUFFLE_SEED}, repeats {SHUFFLE_REPEATS}. One Random "
    "stream, Fisher-Yates via random.Random.shuffle, pairs in axis-then-source "
    "order. Each repeat permutes the two candidate signatures inside the pair. "
    "Member ids stay put. Discriminability is whether the original positive "
    "member is strictly ahead of the negative member. "
    "Random band: theoretical 0.5 ± 2*SE, SE = sqrt(0.25/n_pairs), n_pairs = "
    "len(AXES)*len(NARROW_CODES). The measured shuffle rate is the mean of the "
    f"{SHUFFLE_REPEATS} rates. The gate passes when that mean lies inside the "
    "band. Also recorded: repeat 0, and the linear 2.5 and 97.5 percentiles."
)
PROSE_SWAP_RULE = (
    "Prose-swap pairs use the same axis poles as L1. The left member and the "
    "query copy source i's prose. The right member copies source (i+1) mod n "
    "prose, sources sorted by episode_dir. Structure: left and query are the "
    "positive pole, right is the negative pole. After the baseline rank, the "
    "two members exchange prose and keep structure, and both arms are ranked "
    "again. Lexical follows prose when its top-1 member is the one whose "
    "production retrieval text equals the query text. Structural follows prose "
    "when its top-1 member is that same prose holder. The decisive counts are "
    "the after-swap rates."
)
L2_CORPUS_RULE = (
    "Revisions: the 14 primary r8a checkpoint revisions, episode_dir order. "
    "Fork writes: narrow caliber then wide caliber, the R8a _fork_payload and "
    "_write_fork path, 6 + 9 = 15 writes into one store. A second identical "
    "payload is the production idempotent hit, not a second node. Queries: "
    "each narrow failure subgraph. Candidates: retrieval subgraphs on the "
    "other primary revisions. Wide-only revisions are not registered, so a "
    "fork grounded on r8a-wide is not inside a candidate."
)
L2_IDENTITY_RULE = (
    "Full orders are identical when every candidate's 1-based structural rank "
    "equals its 1-based lexical rank. Lexical rank is production "
    "_rank_document_indexes order (the adapter top-3 must match its prefix). "
    "BM25-zero documents, which production drops, share one rank after the "
    "last kept document. The inert sentence is emitted only when this equality "
    "holds on every query. Top-3 Jaccard is reported beside that, and is not "
    "a winner."
)
ATTRIBUTION_RULE = (
    "A disagreement is a candidate whose structural rank differs from its "
    "lexical rank. direction is raised when the structural rank number is "
    "smaller, otherwise lowered. The field lists are the weighted fields that "
    "match the query and those that do not. Versus the lexical occupant of "
    "this candidate's structural rank, exclusive_matches are the fields that "
    "match the query on exactly one of the two signatures. Equal structural "
    "scores are attributed to the frozen tie-break, not to a field."
)

NOT_ANSWERED: tuple[str, ...] = (
    "诊断侧的真实效用（没有裁决、没有修复链，量不了）。",
    "结构通道在**非夹具**数据上的相对收益（L2 只报一致性，不报胜负）。",
    "跨 session／跨 workspace 的泛化（夹具与真切片都在同一 workspace 内构造；ACL 只作为可见性前提）。",
    "模型写的七段／branches 在检索里的作用（#125 的范围）。",
)
FORBIDDEN_PHRASES: tuple[str, ...] = (
    "结构臂更好",
    "结构臂优于",
    "结构臂胜",
    "结构臂赢",
    "structural arm is better",
    "structural arm wins",
)

INERT_SENTENCE = "本语料上结构字段在生产通道里是惰性的"


def _load_production() -> None:
    """Bind local OpenViking and the signature module. Imports stay in here."""
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    for entry in (str(root), str(here)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    global AORange
    global CausalExperiencesStore
    global DIAGNOSTIC_SUBGRAPH_HIT_CAP
    global ForkBranchAdapter
    global InfluenceBlock
    global InfluenceClaim
    global InfluenceEvidenceRef
    global InfluenceViewDraft
    global InfluenceViewRevision
    global OmittedRange
    global PURPOSE_POST_RUN_INDEX
    global RELATION_INFLUENCE
    global RELATION_SKIP
    global RetrievalSubgraph
    global RetrievalSubgraphAdapter
    global SEVEN_SECTION_KEYS
    global SUBGRAPH_TEXT_POLICY_VERSION
    global SegmentRef
    global SourceRef
    global ViewSupersedeIndex
    global SIG
    global _position_inside_subgraph
    global _rank_document_indexes
    global _subgraph_retrieval_text
    import r7b_structural_signature as signature_mod

    from openviking.session.causal_experiences import (
        SEVEN_SECTION_KEYS,
        CausalExperiencesStore,
    )
    from openviking.session.influence_view import (
        PURPOSE_POST_RUN_INDEX,
        RELATION_INFLUENCE,
        RELATION_SKIP,
        AORange,
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
    )
    from openviking.session.retrieval_facade import (
        DIAGNOSTIC_SUBGRAPH_HIT_CAP,
        SUBGRAPH_TEXT_POLICY_VERSION,
        ForkBranchAdapter,
        RetrievalSubgraphAdapter,
        _position_inside_subgraph,
        _rank_document_indexes,
        _subgraph_retrieval_text,
    )

    SIG = signature_mod


_load_production()

_LEXICAL_CALLS = 0


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def _clock() -> datetime:
    return datetime(2026, 9, 30, tzinfo=timezone.utc)


def _acl(_session_id: str) -> tuple[str, ...]:
    return PRINCIPAL


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    if denominator <= 0:
        _fail("rate denominator must be positive")
    return {
        "denominator": denominator,
        "numerator": numerator,
        "rate": f"{numerator / denominator:.4f}",
    }


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _print_policy(n_pairs: int) -> dict[str, Any]:
    """Print every frozen rule. Must run before any discriminability number."""
    se = math.sqrt(0.25 / n_pairs)
    band_low = 0.5 - 2.0 * se
    band_high = 0.5 + 2.0 * se
    print("R7B DISCRIMINABILITY POLICY (frozen before scores)")
    print(f"SCHEMA {SCHEMA}")
    print(f"LABEL_L1 {LABEL_L1}")
    print(f"SUBGRAPH_TEXT_POLICY_VERSION {SUBGRAPH_TEXT_POLICY_VERSION}")
    print(f"DIAGNOSTIC_SUBGRAPH_HIT_CAP {DIAGNOSTIC_SUBGRAPH_HIT_CAP}")
    print(f"WORKSPACE_LABEL {WORKSPACE_LABEL}")
    print(f"NARROW_CODES {NARROW_CODES}")
    print(
        f"AXES {[(item['id'], item['field'], item['positive'], item['negative']) for item in AXES]}"
    )
    print(f"SHUFFLE_SEED {SHUFFLE_SEED}")
    print(f"SHUFFLE_REPEATS {SHUFFLE_REPEATS}")
    print(f"N_PAIRS {n_pairs}")
    print(f"RANDOM_BAND 0.5 ± 2*SE SE={_fmt(se)} LOW={_fmt(band_low)} HIGH={_fmt(band_high)}")
    print(FIXTURE_RULE)
    print(SHUFFLE_RULE)
    print(PROSE_SWAP_RULE)
    print(L2_CORPUS_RULE)
    print(L2_IDENTITY_RULE)
    print(ATTRIBUTION_RULE)
    SIG.print_signature_policy()
    for line in NOT_ANSWERED:
        print(f"NOT_ANSWERED {line}")
    return {
        "axes": [
            {
                "field": item["field"],
                "id": item["id"],
                "negative": item["negative"],
                "positive": item["positive"],
            }
            for item in AXES
        ],
        "attribution_rule": ATTRIBUTION_RULE,
        "cap": DIAGNOSTIC_SUBGRAPH_HIT_CAP,
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "fixture_rule": FIXTURE_RULE,
        "identity_rule": L2_IDENTITY_RULE,
        "l2_corpus_rule": L2_CORPUS_RULE,
        "label_l1": LABEL_L1,
        "n_pairs": n_pairs,
        "narrow_codes": list(NARROW_CODES),
        "prose_swap_rule": PROSE_SWAP_RULE,
        "random_band": {
            "center": "0.500000",
            "high": _fmt(band_high),
            "low": _fmt(band_low),
            "measured_quantity": "mean discriminability over SHUFFLE_REPEATS permutations",
            "se": _fmt(se),
        },
        "rules_printed_before_scores": True,
        "shuffle_repeats": SHUFFLE_REPEATS,
        "shuffle_rule": SHUFFLE_RULE,
        "shuffle_seed": SHUFFLE_SEED,
        "subgraph_text_policy_version": SUBGRAPH_TEXT_POLICY_VERSION,
    }


def _load_r8a() -> Any:
    path = Path(__file__).resolve().parent / "sb_influence_extraction.py"
    spec = importlib.util.spec_from_file_location("sb_influence_extraction_r7b", path)
    if spec is None or spec.loader is None:
        _fail(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_records() -> list[dict[str, Any]]:
    if not CHECKPOINT_DIR.is_dir():
        _fail(f"R8a checkpoints missing: {CHECKPOINT_DIR}")
    paths = sorted(CHECKPOINT_DIR.glob("*.json"))
    if len(paths) != len(EPISODE_CODES):
        _fail(f"expected {len(EPISODE_CODES)} checkpoints, found {len(paths)}")
    records = []
    for code, path in zip(EPISODE_CODES, paths, strict=True):
        if not path.name.startswith(code):
            _fail(f"checkpoint {path.name} does not start with {code}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("episode_dir", "")[:4] != code:
            _fail(f"{path.name}: episode_dir code mismatch")
        records.append(record)
    narrow = tuple(
        record["episode_dir"][:4] for record in records if record.get("in_narrow") is True
    )
    if narrow != NARROW_CODES:
        _fail(f"narrow codes {narrow} != frozen {NARROW_CODES}")
    return records


def _primary(record: dict[str, Any]) -> InfluenceViewRevision:
    return InfluenceViewRevision.from_dict(record["revision"])


def _source_prose(
    revision: InfluenceViewRevision, subgraph: RetrievalSubgraph
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
    blocks = {block.block_id: block for block in revision.blocks}
    claims = {claim.claim_id: claim for claim in revision.claims}
    if len(subgraph.block_ids) < SKELETON_BLOCKS or len(subgraph.claim_ids) < SKELETON_CLAIMS:
        _fail(
            f"{revision.session_id}: subgraph has {len(subgraph.block_ids)} blocks "
            f"and {len(subgraph.claim_ids)} claims; skeleton needs "
            f"{SKELETON_BLOCKS}/{SKELETON_CLAIMS}"
        )
    block_prose = []
    for block_id in subgraph.block_ids[:SKELETON_BLOCKS]:
        block = blocks[block_id]
        block_prose.append((block.summary, block.input, block.output))
    claim_prose = []
    for claim_id in subgraph.claim_ids[:SKELETON_CLAIMS]:
        claim = claims[claim_id]
        if not claim.carried_artifact.strip() or not claim.downstream_effect.strip():
            _fail(f"{revision.session_id}: empty claim prose on {claim_id}")
        claim_prose.append((claim.carried_artifact, claim.downstream_effect))
    return block_prose, claim_prose


def _anchor_index(block_prose: list[tuple[str, str, str]], preferred: int) -> int:
    if block_prose[preferred][0].strip():
        return preferred
    for index in range(1, len(block_prose)):
        if block_prose[index][0].strip():
            return index
    _fail("no skeleton block at index >= 1 has a non-empty summary")
    return preferred


def _pole(axis_id: str, positive: bool, block_prose: list[tuple[str, str, str]]) -> dict[str, Any]:
    if axis_id == "A":
        last = RELATION_INFLUENCE if positive else RELATION_SKIP
        return {
            "anchor_index": 1,
            "omitted": False,
            "overlaps": False,
            "relations": (RELATION_INFLUENCE, RELATION_INFLUENCE, last),
            "write_fork": False,
        }
    if axis_id == "B":
        return {
            "anchor_index": 1 if positive else 3,
            "omitted": False,
            "overlaps": False,
            "relations": (RELATION_INFLUENCE, RELATION_INFLUENCE, RELATION_SKIP),
            "write_fork": False,
        }
    if axis_id == "C":
        return {
            "anchor_index": 1,
            "omitted": positive,
            "overlaps": False,
            "relations": (RELATION_INFLUENCE, RELATION_INFLUENCE, RELATION_SKIP),
            "write_fork": False,
        }
    if axis_id == "D":
        return {
            "anchor_index": _anchor_index(block_prose, 1),
            "omitted": False,
            "overlaps": positive,
            "relations": (RELATION_INFLUENCE, RELATION_INFLUENCE, RELATION_SKIP),
            "write_fork": True,
        }
    _fail(f"unknown axis {axis_id}")
    return {}


def _build_revision(
    session_id: str,
    subgraph_id: str,
    block_prose: list[tuple[str, str, str]],
    claim_prose: list[tuple[str, str]],
    spec: dict[str, Any],
) -> InfluenceViewRevision:
    blocks = []
    for index, (summary, block_input, output) in enumerate(block_prose):
        sequence = index + 1
        ref = SourceRef(
            ao_id=f"ao-{session_id}-{sequence}",
            sequence=sequence,
            participant_id="agent-a",
            segment_id="seg-a",
            session_id=session_id,
            content_hash=f"h-{session_id}-{sequence}",
        )
        blocks.append(
            InfluenceBlock(
                block_id=f"b{index}",
                role="agent_action",
                participant_id="agent-a",
                segment_ref=SegmentRef(segment_id="seg-a", session_id=session_id),
                ao_start_seq=sequence,
                ao_end_seq=sequence,
                summary=summary,
                input=block_input,
                output=output,
                authorship="agent-a",
                granularity_reason="r7b-skeleton",
                source_content_hashes=(ref.content_hash,),
                source_refs=(ref,),
            )
        )
    claims = []
    relations: tuple[str, str, str] = spec["relations"]
    for index, (artifact, effect) in enumerate(claim_prose):
        claims.append(
            InfluenceClaim(
                claim_id=f"c{index}",
                source_block_id=f"b{index}",
                target_block_id=f"b{index + 1}",
                carried_artifact=artifact,
                downstream_effect=effect,
                relation_type=relations[index],
            )
        )
    anchor = blocks[spec["anchor_index"]]
    anchor_ref = anchor.source_refs[0]
    evidence = InfluenceEvidenceRef(
        ao_id=anchor_ref.ao_id,
        source_session_id=session_id,
        source_archive_id=None,
        archive_commit_watermark=None,
        source_sequence=anchor_ref.sequence,
        source_read_snapshot_watermark="wm-r7b",
        evidence_role="hindsight_attribution",
        evidence_content_hash=anchor_ref.content_hash,
        captured_state="committed",
    )
    subgraph = RetrievalSubgraph(
        subgraph_id=subgraph_id,
        block_ids=tuple(block.block_id for block in blocks),
        claim_ids=tuple(claim.claim_id for claim in claims),
        failure_marked=True,
        failure_evidence_refs=(evidence,),
    )
    omitted: tuple[OmittedRange, ...] = ()
    end_seq = 4
    if spec["omitted"]:
        omitted = (OmittedRange(start_seq=5, end_seq=5, reason="gap", segment_id="seg-a"),)
        end_seq = 5
    coverage = CoverageManifest_from(blocks, omitted, end_seq)
    draft = InfluenceViewDraft(
        view_id=f"view-{session_id}",
        purpose="post_run_index",
        session_id=session_id,
        task_run_id=f"task-{session_id}",
        coverage=coverage,
        blocks=blocks,
        claims=claims,
        retrieval_subgraphs=(subgraph,),
    )
    return draft.freeze(
        revision_id=f"rev-{session_id}",
        enforce_checks=False,
        frozen_at=FROZEN_AT,
    )


def CoverageManifest_from(
    blocks: list[InfluenceBlock], omitted: tuple[OmittedRange, ...], end_seq: int
) -> Any:
    from openviking.session.influence_view import CoverageManifest

    refs = tuple(ref for block in blocks for ref in block.source_refs)
    return CoverageManifest(
        total_ao_range=AORange(start_seq=1, end_seq=end_seq),
        inspected_refs=refs,
        included_ranges=tuple(
            AORange(start_seq=block.ao_start_seq, end_seq=block.ao_end_seq, segment_id="seg-a")
            for block in blocks
        ),
        omitted_ranges=omitted,
        snapshot_watermark="wm-r7b",
    )


def _sections() -> dict[str, str]:
    return dict.fromkeys(SEVEN_SECTION_KEYS, "x")


def _write_fork(
    store: CausalExperiencesStore, revision: InfluenceViewRevision, overlaps: bool
) -> Any:
    from openviking.session.subgraph_forks import (
        default_failure_grounding_position,
        write_failure_subgraph_fork,
    )

    subgraph = revision.retrieval_subgraphs[0]
    kind, ref = default_failure_grounding_position(revision, subgraph.subgraph_id)
    anchor_ao = subgraph.failure_evidence_refs[0].ao_id
    sequence = [anchor_ao] if overlaps else ["ao-off"]
    return write_failure_subgraph_fork(
        store,
        revision,
        subgraph.subgraph_id,
        seven_sections=_sections(),
        branches=[
            {
                "ao_sequence": sequence,
                "branch_id": "br",
                "evidence_status": "real",
                "guidance": {"instruction": "g"},
            }
        ],
        position_kind=kind,
        position_ref=ref,
        workspace_id=WORKSPACE_LABEL,
        workspace_labels=PRINCIPAL,
        diagnosis_run_id=f"r7b:{revision.session_id}",
        model_version="r7b-fixture",
        contemporaneous_basis=[
            {
                "ao_id": f"ao-{revision.session_id}-1",
                "archive_commit_watermark": None,
                "captured_state": "committed",
                "evidence_content_hash": f"h-{revision.session_id}-1",
                "evidence_role": "contemporaneous_basis",
                "source_archive_id": None,
                "source_read_snapshot_watermark": "wm-r7b",
                "source_sequence": 1,
                "source_session_id": revision.session_id,
            }
        ],
    )


def _new_store(root: Path) -> CausalExperiencesStore:
    return CausalExperiencesStore(
        root,
        config={"skill_trajectory_mode": "causal"},
        clock=_clock,
    )


def _adapter(
    root: Path, revisions: list[InfluenceViewRevision]
) -> tuple[RetrievalSubgraphAdapter, CausalExperiencesStore]:
    views = ViewSupersedeIndex(root / "views.json")
    for revision in revisions:
        views.register(revision)
    store = _new_store(root / "store")
    forks = ForkBranchAdapter(root / "store", store)
    adapter = RetrievalSubgraphAdapter(views, forks, session_acl=_acl)
    return adapter, store


def _lexical(
    adapter: RetrievalSubgraphAdapter, revision_id: str, subgraph_id: str
) -> tuple[Any, ...]:
    global _LEXICAL_CALLS
    fn = RetrievalSubgraphAdapter.retrieve_historical_subgraphs
    if fn.__module__ != "openviking.session.retrieval_facade":
        _fail(f"lexical callable module is {fn.__module__}")
    _LEXICAL_CALLS += 1
    return fn(adapter, revision_id, subgraph_id, PRINCIPAL)


def _key(session_id: str, subgraph_id: str) -> list[str]:
    return [session_id, subgraph_id]


def _structural_rows(
    query_sig: Any, rows: list[tuple[str, Any, str, str]]
) -> list[tuple[str, int, str, str]]:
    """Return (label, score, session_id, subgraph_id) best-first."""
    scored: list[tuple[int, str, str, str]] = []
    for label, sig, session_id, subgraph_id in rows:
        score = SIG.structural_score(query_sig, sig)
        if score is None:
            continue
        scored.append((score, session_id, subgraph_id, label))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [
        (label, score, session_id, subgraph_id) for score, session_id, subgraph_id, label in scored
    ]


def _arm_metrics(order: list[str], positive: str, negative: str) -> dict[str, Any]:
    pos_rank = order.index(positive) + 1 if positive in order else None
    neg_rank = order.index(negative) + 1 if negative in order else None
    if pos_rank is None:
        ahead = False
        mrr = 0.0
        top1 = 0
        top3 = 0
    else:
        ahead = neg_rank is None or pos_rank < neg_rank
        mrr = 1.0 / pos_rank
        top1 = 1 if pos_rank == 1 else 0
        top3 = 1 if pos_rank <= DIAGNOSTIC_SUBGRAPH_HIT_CAP else 0
    return {
        "discriminated": ahead and top1 == 1,
        "mrr": f"{mrr:.4f}",
        "negative_rank": neg_rank,
        "positive_rank": pos_rank,
        "top1": top1,
        "top3": top3,
    }


def _fields_that_differ(left: Any, right: Any) -> list[str]:
    names = []
    for name, _weight in SIG.FIELD_WEIGHTS:
        if getattr(left, name) != getattr(right, name):
            names.append(name)
    if left.workspace_id != right.workspace_id:
        names.append("workspace_id")
    return names


def _sign(revision: InfluenceViewRevision, forks: tuple[Any, ...] = ()) -> Any:
    return SIG.structural_signature(
        revision,
        revision.retrieval_subgraphs[0],
        workspace_id=WORKSPACE_LABEL,
        forks=forks,
    )


def _prose_digest(revision: InfluenceViewRevision) -> str:
    text = _subgraph_retrieval_text(revision, revision.retrieval_subgraphs[0])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prose_blob(revision: InfluenceViewRevision) -> str:
    return _subgraph_retrieval_text(revision, revision.retrieval_subgraphs[0])


def _run_pair(
    root: Path,
    axis: dict[str, Any],
    code: str,
    block_prose: list[tuple[str, str, str]],
    claim_prose: list[tuple[str, str]],
    *,
    right_prose: tuple[list[tuple[str, str, str]], list[tuple[str, str]]] | None = None,
    tag: str,
) -> dict[str, Any]:
    """Build query/pos/neg, rank both arms, return metrics and signature objects."""
    specs = {
        "query": _pole(axis["id"], True, block_prose),
        "pos": _pole(axis["id"], True, block_prose),
        "neg": _pole(axis["id"], False, block_prose),
    }
    prose = {
        "query": (block_prose, claim_prose),
        "pos": (block_prose, claim_prose),
        "neg": (block_prose, claim_prose),
    }
    if right_prose is not None:
        prose["neg"] = right_prose
    prefix = {"query": "q", "pos": "p", "neg": "n"}
    if tag == "swap":
        prefix = {"query": "sq", "pos": "sl", "neg": "sr"}
    revisions = {}
    for role in ("query", "pos", "neg"):
        session_id = f"{prefix[role]}-{axis['id']}-{code}"
        revisions[role] = _build_revision(
            session_id,
            f"sg-{role}",
            prose[role][0],
            prose[role][1],
            specs[role],
        )
    work = root / f"{tag}-{axis['id']}-{code}"
    work.mkdir(parents=True, exist_ok=True)
    adapter, store = _adapter(work, [revisions["query"], revisions["pos"], revisions["neg"]])
    forks: dict[str, tuple[Any, ...]] = {"query": (), "pos": (), "neg": ()}
    for role, spec in specs.items():
        if spec["write_fork"]:
            written = _write_fork(store, revisions[role], spec["overlaps"])
            forks[role] = (written,)
    signatures = {role: _sign(revisions[role], forks[role]) for role in revisions}
    hits = _lexical(
        adapter,
        revisions["query"].revision_id,
        revisions["query"].retrieval_subgraphs[0].subgraph_id,
    )
    lexical_order = [hit.session_id for hit in hits]
    lexical_bm25 = [
        {"bm25": f"{hit.score_breakdown['bm25']:.8f}", "session_id": hit.session_id} for hit in hits
    ]
    structural = _structural_rows(
        signatures["query"],
        [
            ("pos", signatures["pos"], revisions["pos"].session_id, "sg-pos"),
            ("neg", signatures["neg"], revisions["neg"].session_id, "sg-neg"),
        ],
    )
    struct_labels = [item[0] for item in structural]
    metrics_lex = _arm_metrics(
        lexical_order, revisions["pos"].session_id, revisions["neg"].session_id
    )
    # Structural metrics use labels, then map through the same ahead rule via a synthetic order.
    struct_sessions = [
        revisions["pos"].session_id if label == "pos" else revisions["neg"].session_id
        for label in struct_labels
    ]
    metrics_struct = _arm_metrics(
        struct_sessions, revisions["pos"].session_id, revisions["neg"].session_id
    )
    query_hash = _prose_digest(revisions["query"])
    pos_hash = _prose_digest(revisions["pos"])
    neg_hash = _prose_digest(revisions["neg"])
    return {
        "axis_id": axis["id"],
        "code": code,
        "fields_that_differ": _fields_that_differ(signatures["pos"], signatures["neg"]),
        "forks": forks,
        "lexical_bm25": lexical_bm25,
        "lexical_metrics": metrics_lex,
        "lexical_top1_session": lexical_order[0] if lexical_order else None,
        "neg_hash": neg_hash,
        "neg_session": revisions["neg"].session_id,
        "pos_hash": pos_hash,
        "pos_session": revisions["pos"].session_id,
        "prose_equal_poles": pos_hash == neg_hash,
        "query_hash": query_hash,
        "query_matches_pos_prose": query_hash == pos_hash,
        "revisions": revisions,
        "signatures": signatures,
        "structural_metrics": metrics_struct,
        "structural_order": structural,
        "structural_top1_label": struct_labels[0] if struct_labels else None,
    }


def _public_pair(row: dict[str, Any], axis: dict[str, Any]) -> dict[str, Any]:
    structural = row["structural_order"]
    return {
        "axis": axis["id"],
        "code": row["code"],
        "field": axis["field"],
        "fields_that_differ": row["fields_that_differ"],
        "label": LABEL_L1,
        "lexical": row["lexical_metrics"],
        "lexical_bm25": row["lexical_bm25"],
        "negative": {
            "prose_sha256": row["neg_hash"],
            "session_id": row["neg_session"],
            "signature": row["signatures"]["neg"].to_json_obj(),
            "structural_score": next(item[1] for item in structural if item[0] == "neg"),
        },
        "positive": {
            "prose_sha256": row["pos_hash"],
            "session_id": row["pos_session"],
            "signature": row["signatures"]["pos"].to_json_obj(),
            "structural_score": next(item[1] for item in structural if item[0] == "pos"),
        },
        "prose_equal": row["prose_equal_poles"] and row["query_matches_pos_prose"],
        "query_signature": row["signatures"]["query"].to_json_obj(),
        "structural": row["structural_metrics"],
    }


def _aggregate(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    n = len(rows)
    disc = sum(1 for row in rows if row[arm]["discriminated"])
    top1 = sum(row[arm]["top1"] for row in rows)
    top3 = sum(row[arm]["top3"] for row in rows)
    mrr = sum(float(row[arm]["mrr"]) for row in rows) / n
    return {
        "discriminability": _rate(disc, n),
        "mrr": f"{mrr:.4f}",
        "top1": _rate(top1, n),
        "top3": _rate(top3, n),
    }


def _assert_l1_shape(row: dict[str, Any], axis: dict[str, Any]) -> None:
    field = axis["field"]
    query = row["signatures"]["query"]
    pos = row["signatures"]["pos"]
    neg = row["signatures"]["neg"]
    if getattr(query, field) != axis["positive"] or getattr(pos, field) != axis["positive"]:
        _fail(f"{axis['id']} {row['code']}: positive pole is not {axis['positive']!r}")
    if getattr(neg, field) != axis["negative"]:
        _fail(f"{axis['id']} {row['code']}: negative pole is not {axis['negative']!r}")
    if not row["prose_equal_poles"] or not row["query_matches_pos_prose"]:
        _fail(f"{axis['id']} {row['code']}: controlled-pair prose is not byte-identical")
    pos_score = SIG.structural_score(query, pos)
    neg_score = SIG.structural_score(query, neg)
    if pos_score is None or neg_score is None or pos_score <= neg_score:
        _fail(
            f"{axis['id']} {row['code']}: fixture scores pos={pos_score} neg={neg_score}; "
            "the positive pole does not outscore the negative pole"
        )


def _sources(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = []
    for record in records:
        if record["episode_dir"][:4] not in NARROW_CODES:
            continue
        revision = _primary(record)
        subgraphs = [item for item in revision.retrieval_subgraphs if item.is_failure]
        if len(subgraphs) != 1:
            _fail(f"{record['episode_dir']}: expected one failure subgraph")
        block_prose, claim_prose = _source_prose(revision, subgraphs[0])
        wanted.append(
            {
                "block_prose": block_prose,
                "claim_prose": claim_prose,
                "code": record["episode_dir"][:4],
                "episode_dir": record["episode_dir"],
            }
        )
    if tuple(item["code"] for item in wanted) != NARROW_CODES:
        _fail("source order drifted from NARROW_CODES")
    return wanted


def _isolation(pairs: list[dict[str, Any]], swap_rows: list[dict[str, Any]]) -> dict[str, Any]:
    fn = RetrievalSubgraphAdapter.retrieve_historical_subgraphs
    names = set(fn.__code__.co_names)
    leaks: list[dict[str, str]] = []
    seen: set[str] = set()
    dumps: list[str] = []
    for row in pairs + swap_rows:
        for role in ("query", "pos", "neg"):
            seen |= _tokens(_prose_blob(row["revisions"][role]))
            dumps.append(
                json.dumps(
                    row["signatures"][role].to_json_obj(), ensure_ascii=False, sort_keys=True
                )
            )
    signature_blob = "\n".join(dumps)
    for token in sorted(seen):
        if token in signature_blob:
            leaks.append({"token": token})
    prose_fn = _subgraph_retrieval_text
    passed = (
        not leaks
        and fn.__module__ == "openviking.session.retrieval_facade"
        and "_subgraph_retrieval_text" in names
        and "_rank_document_indexes" in names
        and prose_fn.__module__ == "openviking.session.retrieval_facade"
    )
    return {
        "label": LABEL_L1,
        "leak_count": len(leaks),
        "leaks": leaks[:20],
        "lexical_co_names_include_rank_document_indexes": "_rank_document_indexes" in names,
        "lexical_co_names_include_subgraph_retrieval_text": "_subgraph_retrieval_text" in names,
        "lexical_module": fn.__module__,
        "lexical_qualname": fn.__qualname__,
        "passed": passed,
        "unique_tokens": len(seen),
    }


def _tokens(text: str) -> set[str]:
    found: set[str] = set()
    for part in re.split(r"\s+", text):
        if len(part) >= PROSE_TOKEN_MIN:
            found.add(part)
    for part in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{7,}", text):
        found.add(part)
    for part in re.findall(r"[\u4e00-\u9fff]{8,}", text):
        found.add(part)
    return found


def _shuffle(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    n_pairs = len(pairs)
    se = math.sqrt(0.25 / n_pairs)
    band_low = 0.5 - 2.0 * se
    band_high = 0.5 + 2.0 * se
    rng = random.Random(SHUFFLE_SEED)
    rates: list[float] = []
    for _repeat in range(SHUFFLE_REPEATS):
        hits = 0
        for row in pairs:
            assigned = [row["signatures"]["pos"], row["signatures"]["neg"]]
            rng.shuffle(assigned)
            order = _structural_rows(
                row["signatures"]["query"],
                [
                    ("pos", assigned[0], row["pos_session"], "sg-pos"),
                    ("neg", assigned[1], row["neg_session"], "sg-neg"),
                ],
            )
            if order and order[0][0] == "pos":
                hits += 1
        rates.append(hits / n_pairs)
    mean = sum(rates) / len(rates)
    inside = band_low <= mean <= band_high
    return {
        "band_high": _fmt(band_high),
        "band_low": _fmt(band_low),
        "definition": SHUFFLE_RULE,
        "inside_band": inside,
        "label": LABEL_L1,
        "mean_rate": _fmt(mean),
        "n_pairs": n_pairs,
        "percentile_2_5": _fmt(_percentile(rates, 0.025)),
        "percentile_97_5": _fmt(_percentile(rates, 0.975)),
        "repeat0_rate": _fmt(rates[0]),
        "repeats": SHUFFLE_REPEATS,
        "se": _fmt(se),
        "seed": SHUFFLE_SEED,
    }


def _swap_public(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    def holder(row: dict[str, Any]) -> str | None:
        if row["query_hash"] == row["pos_hash"]:
            return row["pos_session"]
        if row["query_hash"] == row["neg_hash"]:
            return row["neg_session"]
        return None

    before_holder = holder(before)
    after_holder = holder(after)
    return {
        "after_lexical_top1": after["lexical_top1_session"],
        "after_prose_holder": after_holder,
        "after_structural_top1_label": after["structural_top1_label"],
        "axis": before["axis_id"],
        "before_lexical_top1": before["lexical_top1_session"],
        "before_prose_holder": before_holder,
        "before_structural_top1_label": before["structural_top1_label"],
        "code": before["code"],
        "label": LABEL_L1,
        "lexical_follows_prose_after": after["lexical_top1_session"] == after_holder,
        "structural_follows_prose_after": (
            after_holder is not None
            and after["lexical_top1_session"] is not None
            and _structural_session(after) == after_holder
        ),
        "structural_top1_unchanged": before["structural_top1_label"]
        == after["structural_top1_label"],
    }


def _structural_session(row: dict[str, Any]) -> str | None:
    label = row["structural_top1_label"]
    if label == "pos":
        return row["pos_session"]
    if label == "neg":
        return row["neg_session"]
    return None


def _run_l1(records: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    global _LEXICAL_CALLS
    _LEXICAL_CALLS = 0
    sources = _sources(records)
    built: list[dict[str, Any]] = []
    for axis in AXES:
        for source in sources:
            row = _run_pair(
                root,
                axis,
                source["code"],
                source["block_prose"],
                source["claim_prose"],
                tag="l1",
            )
            _assert_l1_shape(row, axis)
            built.append(row)
    public = [_public_pair(row, axis) for row, axis in zip(built, _axis_for(built), strict=True)]
    # public prose_equal already checked. Aggregate in axis order.
    per_axis = []
    for axis in AXES:
        group = [item for item in public if item["axis"] == axis["id"]]
        per_axis.append(
            {
                "axis": axis["id"],
                "field": axis["field"],
                "label": LABEL_L1,
                "lexical": _aggregate(group, "lexical"),
                "n": len(group),
                "negative": axis["negative"],
                "positive": axis["positive"],
                "structural": _aggregate(group, "structural"),
            }
        )
    swap_before = []
    n_sources = len(sources)
    for axis in AXES:
        for index, source in enumerate(sources):
            donor = sources[(index + 1) % n_sources]
            before = _run_pair(
                root,
                axis,
                source["code"],
                source["block_prose"],
                source["claim_prose"],
                right_prose=(donor["block_prose"], donor["claim_prose"]),
                tag="swap-before",
            )
            if before["pos_hash"] == before["neg_hash"]:
                _fail(f"swap donor prose matches source {source['code']}")
            if before["query_hash"] != before["pos_hash"]:
                _fail(f"swap before: query prose is not on the positive member {source['code']}")
            swap_before.append(before)
    swap_after = _swap_after(root, sources)
    for after in swap_after:
        if after["query_hash"] != after["neg_hash"]:
            _fail(f"swap after: query prose is not on the negative member {after['code']}")
        if after["query_hash"] == after["pos_hash"]:
            _fail(f"swap after: positive member still holds the query prose {after['code']}")
    swap_rows = [
        _swap_public(before, after) for before, after in zip(swap_before, swap_after, strict=True)
    ]
    isolation = _isolation(built, swap_before + swap_after)
    shuffle = _shuffle(built)
    lexical_follows = sum(1 for item in swap_rows if item["lexical_follows_prose_after"])
    structural_follows = sum(1 for item in swap_rows if item["structural_follows_prose_after"])
    structural_still = sum(1 for item in swap_rows if item["structural_top1_unchanged"])
    n_swap = len(swap_rows)
    return {
        "axes": per_axis,
        "label": LABEL_L1,
        "lexical_call_count": _LEXICAL_CALLS,
        "pairs": public,
        "probes": {
            "isolation": isolation,
            "prose_swap": {
                "definition": PROSE_SWAP_RULE,
                "label": LABEL_L1,
                "lexical_follows_prose": _rate(lexical_follows, n_swap),
                "n": n_swap,
                "pairs": swap_rows,
                "structural_follows_prose": _rate(structural_follows, n_swap),
                "structural_top1_unchanged": _rate(structural_still, n_swap),
            },
            "shuffle": shuffle,
        },
    }


def _axis_for(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    axes = []
    for row in rows:
        axes.append(next(item for item in AXES if item["id"] == row["axis_id"]))
    return axes


def _swap_after(root: Path, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Query prose stays on source i. Members exchange prose. Structure stays."""
    built = []
    n_sources = len(sources)
    for axis in AXES:
        for index, source in enumerate(sources):
            donor = sources[(index + 1) % n_sources]
            # _run_pair always copies the first prose onto query and pos, and
            # right_prose onto neg. To exchange, pos must receive the donor prose
            # while the query keeps the source prose. _run_pair cannot express
            # that split, so build it here.
            built.append(_run_swapped_after(root, axis, source, donor))
    return built


def _run_swapped_after(
    root: Path,
    axis: dict[str, Any],
    source: dict[str, Any],
    donor: dict[str, Any],
) -> dict[str, Any]:
    """Positive member holds donor prose; negative holds source prose; query holds source."""
    query_spec = _pole(axis["id"], True, source["block_prose"])
    pos_spec = _pole(axis["id"], True, donor["block_prose"])
    neg_spec = _pole(axis["id"], False, source["block_prose"])
    # Anchor index for axis D depends on which prose is non-empty. Both poles
    # must use the source prose's anchor index so the structure, not the donor
    # text, chooses the block. Recompute D from the source prose for all three.
    if axis["id"] == "D":
        query_spec = _pole(axis["id"], True, source["block_prose"])
        pos_spec = _pole(axis["id"], True, source["block_prose"])
        neg_spec = _pole(axis["id"], False, source["block_prose"])
    code = source["code"]
    roles = {
        "query": (f"sq-{axis['id']}-{code}", source, query_spec, "sg-query"),
        "pos": (f"sl-{axis['id']}-{code}", donor, pos_spec, "sg-pos"),
        "neg": (f"sr-{axis['id']}-{code}", source, neg_spec, "sg-neg"),
    }
    revisions = {}
    for role, (session_id, prose_source, spec, subgraph_id) in roles.items():
        revisions[role] = _build_revision(
            session_id,
            subgraph_id,
            prose_source["block_prose"],
            prose_source["claim_prose"],
            spec,
        )
    work = root / f"swap-after-{axis['id']}-{code}"
    work.mkdir(parents=True, exist_ok=True)
    adapter, store = _adapter(work, [revisions["query"], revisions["pos"], revisions["neg"]])
    forks: dict[str, tuple[Any, ...]] = {}
    specs = {"query": query_spec, "pos": pos_spec, "neg": neg_spec}
    for role, spec in specs.items():
        if spec["write_fork"]:
            forks[role] = (_write_fork(store, revisions[role], spec["overlaps"]),)
        else:
            forks[role] = ()
    signatures = {role: _sign(revisions[role], forks[role]) for role in revisions}
    hits = _lexical(
        adapter,
        revisions["query"].revision_id,
        revisions["query"].retrieval_subgraphs[0].subgraph_id,
    )
    lexical_order = [hit.session_id for hit in hits]
    structural = _structural_rows(
        signatures["query"],
        [
            ("pos", signatures["pos"], revisions["pos"].session_id, "sg-pos"),
            ("neg", signatures["neg"], revisions["neg"].session_id, "sg-neg"),
        ],
    )
    return {
        "axis_id": axis["id"],
        "code": code,
        "lexical_top1_session": lexical_order[0] if lexical_order else None,
        "neg_hash": _prose_digest(revisions["neg"]),
        "neg_session": revisions["neg"].session_id,
        "pos_hash": _prose_digest(revisions["pos"]),
        "pos_session": revisions["pos"].session_id,
        "query_hash": _prose_digest(revisions["query"]),
        "revisions": revisions,
        "signatures": signatures,
        "structural_top1_label": structural[0][0] if structural else None,
    }


def _forks_inside(
    forks: list[Any], revision: InfluenceViewRevision, subgraph: RetrievalSubgraph
) -> tuple[Any, ...]:
    chosen = []
    seen: set[str] = set()
    for fork in forks:
        grounding = fork.influence_grounding
        if grounding is None or grounding.view_revision_id != revision.revision_id:
            continue
        if not _position_inside_subgraph(grounding, subgraph):
            continue
        if fork.revision_id in seen:
            continue
        seen.add(fork.revision_id)
        chosen.append(fork)
    chosen.sort(key=lambda item: item.revision_id)
    return tuple(chosen)


def _spearman(left: list[float], right: list[float]) -> str:
    n = len(left)
    if n != len(right) or n < 2:
        return "0.000000"

    def _ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda index: values[index])
        ranks = [0.0] * n
        cursor = 0
        while cursor < n:
            end = cursor
            while end + 1 < n and values[order[end + 1]] == values[order[cursor]]:
                end += 1
            average = (cursor + 1 + end + 1) / 2.0
            for index in range(cursor, end + 1):
                ranks[order[index]] = average
            cursor = end + 1
        return ranks

    ra = _ranks(left)
    rb = _ranks(right)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((a - mean_a) * (b - mean_b) for a, b in zip(ra, rb, strict=True))
    den_a = math.sqrt(sum((a - mean_a) ** 2 for a in ra))
    den_b = math.sqrt(sum((b - mean_b) ** 2 for b in rb))
    if den_a == 0.0 and den_b == 0.0:
        return "1.000000"
    if den_a == 0.0 or den_b == 0.0:
        return "0.000000"
    return _fmt(num / (den_a * den_b))


def _jaccard(left: list[list[str]], right: list[list[str]]) -> str:
    a = {tuple(item) for item in left}
    b = {tuple(item) for item in right}
    if not a and not b:
        return "1.000000"
    return _fmt(len(a & b) / len(a | b))


def _run_l2(records: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    global _LEXICAL_CALLS
    _LEXICAL_CALLS = 0
    r8a = _load_r8a()
    revisions = [_primary(record) for record in records]
    work = root / "l2"
    work.mkdir(parents=True, exist_ok=True)
    views = ViewSupersedeIndex(work / "views.json")
    for revision in revisions:
        views.register(revision)
    store = _new_store(work / "store")
    forks_adapter = ForkBranchAdapter(work / "store", store)
    adapter = RetrievalSubgraphAdapter(views, forks_adapter, session_acl=_acl)
    write_rows = []
    written: list[Any] = []
    seen_ids: set[str] = set()
    for caliber in ("narrow", "wide"):
        for record in records:
            flag = record["in_narrow"] if caliber == "narrow" else record["in_wide"]
            if not flag:
                continue
            payload_rev = record["revision"]
            if caliber == "wide" and record["revision_wide"] is not None:
                payload_rev = record["revision_wide"]
            revision = InfluenceViewRevision.from_dict(payload_rev)
            payload = r8a._fork_payload(record, revision)
            fork = r8a._write_fork(store, revision, record, payload)
            write_rows.append(
                {
                    "caliber": caliber,
                    "episode_dir": record["episode_dir"],
                    "fork_revision_id": fork.revision_id,
                    "view_revision_id": revision.revision_id,
                }
            )
            if fork.revision_id not in seen_ids:
                seen_ids.add(fork.revision_id)
                written.append(fork)
    grounded = 0
    for revision in revisions:
        for subgraph in revision.retrieval_subgraphs:
            grounded += len(_forks_inside(written, revision, subgraph))
    queries = []
    orders_identical = True
    jaccards: list[float] = []
    spearmans: list[str] = []
    for record in records:
        if record["episode_dir"][:4] not in NARROW_CODES:
            continue
        revision = _primary(record)
        subgraph = revision.retrieval_subgraphs[0]
        query_forks = _forks_inside(written, revision, subgraph)
        query_sig = SIG.structural_signature(
            revision, subgraph, workspace_id=WORKSPACE_LABEL, forks=query_forks
        )
        hits = _lexical(adapter, revision.revision_id, subgraph.subgraph_id)
        candidates: list[tuple[InfluenceViewRevision, RetrievalSubgraph]] = []
        documents: list[str] = []
        for other in views.revisions_for_purpose(PURPOSE_POST_RUN_INDEX):
            if other.revision_id == revision.revision_id:
                continue
            for item in other.retrieval_subgraphs:
                candidates.append((other, item))
                documents.append(_subgraph_retrieval_text(other, item))
        query_text = _subgraph_retrieval_text(revision, subgraph)
        ranked = _rank_document_indexes(query_text, documents)
        ranked_keys = [
            [candidates[index][0].session_id, candidates[index][1].subgraph_id]
            for index, _breakdown in ranked
        ]
        hit_keys = [[hit.session_id, hit.subgraph.subgraph_id] for hit in hits]
        if hit_keys != ranked_keys[:DIAGNOSTIC_SUBGRAPH_HIT_CAP]:
            _fail(
                f"{record['episode_dir']}: adapter top-3 {hit_keys} != "
                f"production rank prefix {ranked_keys[:DIAGNOSTIC_SUBGRAPH_HIT_CAP]}"
            )
        kept = {tuple(item) for item in ranked_keys}
        dropped = [
            [other.session_id, item.subgraph_id]
            for other, item in candidates
            if (other.session_id, item.subgraph_id) not in kept
        ]
        dropped.sort()
        lexical_rank: dict[tuple[str, str], int] = {}
        for index, key in enumerate(ranked_keys, start=1):
            lexical_rank[tuple(key)] = index
        bottom = len(ranked_keys) + 1
        for key in dropped:
            lexical_rank[tuple(key)] = bottom
        struct_rows = []
        details = []
        for other, item in candidates:
            forks = _forks_inside(written, other, item)
            cand_sig = SIG.structural_signature(
                other, item, workspace_id=WORKSPACE_LABEL, forks=forks
            )
            score = SIG.structural_score(query_sig, cand_sig)
            key = (other.session_id, item.subgraph_id)
            struct_rows.append((score, other.session_id, item.subgraph_id, cand_sig, key))
            details.append(
                {
                    "key": key,
                    "revision_id": other.revision_id,
                    "score": score,
                    "signature": cand_sig,
                }
            )
        comparable = [row for row in struct_rows if row[0] is not None]
        comparable.sort(key=lambda item: (-int(item[0]), item[1], item[2]))
        structural_rank = {row[4]: index for index, row in enumerate(comparable, start=1)}
        by_key = {item["key"]: item for item in details}
        readout = []
        disagreements = []
        for key, s_rank in structural_rank.items():
            l_rank = lexical_rank[key]
            info = by_key[key]
            row = {
                "lexical_rank": l_rank,
                "session_id": key[0],
                "structural_rank": s_rank,
                "structural_score": info["score"],
                "subgraph_id": key[1],
            }
            readout.append(row)
            if s_rank != l_rank:
                rival_key = next(
                    (
                        item
                        for item, item_rank in lexical_rank.items()
                        if item_rank == s_rank and item != key and item in structural_rank
                    ),
                    None,
                )
                contrast = SIG.field_contrast(query_sig, info["signature"])
                exclusive: list[dict[str, Any]] = []
                scores_tied = False
                if rival_key is not None:
                    rival = by_key[rival_key]
                    scores_tied = info["score"] == rival["score"]
                    for name, weight in SIG.FIELD_WEIGHTS:
                        cand_match = getattr(info["signature"], name) == getattr(query_sig, name)
                        rival_match = getattr(rival["signature"], name) == getattr(query_sig, name)
                        if cand_match != rival_match:
                            exclusive.append(
                                {
                                    "candidate_matches_query": cand_match,
                                    "field": name,
                                    "rival_matches_query": rival_match,
                                    "rival_session_id": rival_key[0],
                                    "rival_subgraph_id": rival_key[1],
                                    "weight": weight,
                                }
                            )
                direction = "raised" if s_rank < l_rank else "lowered"
                note = None
                if scores_tied:
                    note = "structural scores tied; order is (session_id, subgraph_id)"
                disagreements.append(
                    {
                        "direction": direction,
                        "exclusive_matches": exclusive,
                        "fields_differing_from_query": contrast["differed"],
                        "fields_matching_query": contrast["matched"],
                        "lexical_rank": l_rank,
                        "note": note,
                        "session_id": key[0],
                        "structural_rank": s_rank,
                        "structural_score": info["score"],
                        "subgraph_id": key[1],
                    }
                )
        readout.sort(
            key=lambda item: (item["structural_rank"], item["session_id"], item["subgraph_id"])
        )
        if disagreements:
            orders_identical = False
        top_lex = ranked_keys[:DIAGNOSTIC_SUBGRAPH_HIT_CAP]
        top_struct = [[row[1], row[2]] for row in comparable[:DIAGNOSTIC_SUBGRAPH_HIT_CAP]]
        jaccard = _jaccard(top_lex, top_struct)
        jaccards.append(float(jaccard))
        # Spearman over the union, using the rank numbers (ties share a value).
        keys = sorted(structural_rank, key=lambda item: (item[0], item[1]))
        spearman = _spearman(
            [float(lexical_rank[key]) for key in keys],
            [float(structural_rank[key]) for key in keys],
        )
        spearmans.append(spearman)
        queries.append(
            {
                "candidates": readout,
                "disagreements": disagreements,
                "episode_dir": record["episode_dir"],
                "label": "真数据切片，无裁决，不算判别力",
                "lexical_top3": top_lex,
                "query_signature": query_sig.to_json_obj(),
                "spearman": spearman,
                "structural_top3": top_struct,
                "top3_jaccard": jaccard,
            }
        )
    mean_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0.0
    conclusion = (
        INERT_SENTENCE
        if orders_identical
        else (
            "两臂在本语料上的全序存在分歧。分歧点的字段归因见各 query 的 disagreements。"
            "此节不比较胜负。"
        )
    )
    return {
        "conclusion": conclusion,
        "distinct_fork_nodes": len(written),
        "fork_writes": write_rows,
        "forks_grounded_on_primary_revisions": grounded,
        "label": "真数据切片，无裁决，不算判别力",
        "lexical_call_count": _LEXICAL_CALLS,
        "mean_spearman": spearmans,
        "mean_top3_jaccard": _fmt(mean_jaccard),
        "orders_identical": orders_identical,
        "primary_revisions": len(revisions),
        "queries": queries,
    }


def _finding(l2: dict[str, Any]) -> str:
    tail = l2["conclusion"]
    return (
        f"{LABEL_L1} "
        "L1 各轴的判别率、top-1、top-3、MRR 见 l1.axes；洗牌、散文互换、隔离见 l1.probes。"
        f" L2：{tail}"
    )


def _assemble(
    policy: dict[str, Any], l1: dict[str, Any] | None, l2: dict[str, Any] | None
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "finding": _finding(l2) if l2 is not None else LABEL_L1,
        "l1": l1,
        "l2": l2,
        "lexical_arm": {
            "DIAGNOSTIC_SUBGRAPH_HIT_CAP": DIAGNOSTIC_SUBGRAPH_HIT_CAP,
            "SUBGRAPH_TEXT_POLICY_VERSION": SUBGRAPH_TEXT_POLICY_VERSION,
            "callable": (
                "openviking.session.retrieval_facade."
                "RetrievalSubgraphAdapter.retrieve_historical_subgraphs"
            ),
            "call_count": (0 if l1 is None else l1["lexical_call_count"])
            + (0 if l2 is None else l2["lexical_call_count"]),
        },
        "not_answered": list(NOT_ANSWERED),
        "policy": policy,
        "rules_printed_before_scores": True,
        "schema": SCHEMA,
    }
    return report


def _gates(report: dict[str, Any], text: str) -> None:
    lexical = report["lexical_arm"]
    if lexical["SUBGRAPH_TEXT_POLICY_VERSION"] != SUBGRAPH_TEXT_POLICY_VERSION:
        _fail("report missing production SUBGRAPH_TEXT_POLICY_VERSION")
    if lexical["DIAGNOSTIC_SUBGRAPH_HIT_CAP"] != DIAGNOSTIC_SUBGRAPH_HIT_CAP:
        _fail("report missing production DIAGNOSTIC_SUBGRAPH_HIT_CAP")
    if "retrieve_historical_subgraphs" not in lexical["callable"]:
        _fail("lexical arm callable is not the production method")
    l1 = report.get("l1")
    if l1 is not None:
        if l1.get("label") != LABEL_L1:
            _fail("L1 missing fixture label")
        isolation = l1["probes"]["isolation"]
        if not isolation["passed"]:
            _fail(f"isolation failed: {isolation['leaks'][:5]}")
        shuffle = l1["probes"]["shuffle"]
        if not shuffle["inside_band"]:
            _fail(
                f"shuffle mean {shuffle['mean_rate']} outside "
                f"[{shuffle['band_low']}, {shuffle['band_high']}]"
            )
        swap = l1["probes"]["prose_swap"]
        if (
            swap["lexical_follows_prose"]["numerator"]
            != swap["lexical_follows_prose"]["denominator"]
        ):
            _fail("prose swap: lexical arm did not follow the prose on every pair")
        if swap["structural_follows_prose"]["numerator"] != 0:
            _fail("prose swap: structural arm followed the prose")
        if (
            swap["structural_top1_unchanged"]["numerator"]
            != swap["structural_top1_unchanged"]["denominator"]
        ):
            _fail("prose swap: structural top-1 moved")
        ids = [item["axis"] for item in l1["axes"]]
        if ids != [item["id"] for item in AXES]:
            _fail(f"axis order {ids}")
        for axis in l1["axes"]:
            if axis.get("label") != LABEL_L1:
                _fail(f"axis {axis['axis']} missing fixture label")
            for arm in ("lexical", "structural"):
                for key in ("discriminability", "top1", "top3", "mrr"):
                    if key not in axis[arm]:
                        _fail(f"axis {axis['axis']} {arm} missing {key}")
        if LABEL_L1 not in text:
            _fail("serialized report dropped the fixture label")
    for phrase in FORBIDDEN_PHRASES:
        if phrase in text:
            _fail(f"forbidden phrase present: {phrase}")
    for line in NOT_ANSWERED:
        if line not in text:
            _fail(f"missing 不回答 line: {line}")
    if report.get("l2") is not None and "disagreements" not in text and "queries" not in text:
        _fail("L2 readout missing")


def _load_existing(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="R7b diagnosis-side discriminability")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("l1", "l2", "all"), default="all")
    args = parser.parse_args()
    n_pairs = len(AXES) * len(NARROW_CODES)
    policy = _print_policy(n_pairs)
    existing = _load_existing(args.out) if args.resume else None
    want_l1 = args.stage in ("l1", "all")
    want_l2 = args.stage in ("l2", "all")
    l1 = existing.get("l1") if existing and existing.get("l1") else None
    l2 = existing.get("l2") if existing and existing.get("l2") else None
    records = _load_records()
    with tempfile.TemporaryDirectory(prefix="r7b-disc-") as tmp:
        root = Path(tmp)
        if want_l1 and l1 is None:
            print("STAGE l1")
            l1 = _run_l1(records, root)
        if want_l2 and l2 is None:
            print("STAGE l2")
            l2 = _run_l2(records, root)
    report = _assemble(policy, l1, l2)
    text = _dump(report)
    _gates(report, text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"WROTE {args.out}")
    print(f"SHA256 {digest}")


if __name__ == "__main__":
    main()
