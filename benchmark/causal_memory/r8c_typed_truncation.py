#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""R8c: retest the R7b ruler after block_kind is a typed field.

Wiring and representation only. L1 is a fixture: discriminability is not an
effect size. L2 reports distributions and rank consistency. No model is called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

# Frozen before scores. Do not edit these to absorb a miss.
SCHEMA = "r8c-typed-truncation-v1"
LABEL_L1 = "基于夹具，判别力 ≠ 效果量"
V1_CHECKPOINT_DIR = Path("/tmp/sb-influence-r8b/checkpoints")
V2_CHECKPOINT_DIR = Path("/tmp/sb-influence-r8c/checkpoints")
R7B_REPORT = Path("/tmp/r7b-disc/report.json")
R7B_REPORT_SHA256 = "2c77d55f7d53e7a01980615dbacfb13646f4d70482a44b0f89abb8679e9f73fe"
SHUFFLE_SEED = 20261001
BLOCK_KIND_ORDER = ("task", "tool_call", "tool_result", "orchestrator", "conclusion")
BLOCK_KIND_PROFILE_WEIGHT = 2
FIELD_WEIGHTS_V1: tuple[tuple[str, int], ...] = (
    ("boundary", 2),
    ("anchor_kind", 1),
    ("anchor_incidence", 2),
    ("relation_profile", 1),
    ("role_profile", 1),
    ("coverage_crossing", 2),
    ("member_gap", 1),
    ("truncation_lines", 1),
    ("grounding_kind", 1),
    ("evidence_status", 1),
    ("overlaps_anchor", 2),
    ("used_skill_kinds", 1),
)
FIELD_WEIGHTS_V1_PLUS: tuple[tuple[str, int], ...] = FIELD_WEIGHTS_V1 + (
    ("block_kind_profile", BLOCK_KIND_PROFILE_WEIGHT),
)
TIE_BREAK = ("session_id", "subgraph_id")
AXES_V1: tuple[dict[str, Any], ...] = (
    {"id": "A", "field": "boundary", "negative": "call_cut", "positive": "result_cut"},
    {
        "id": "B",
        "field": "anchor_incidence",
        "negative": "skip_only",
        "positive": "influence_both",
    },
    {"id": "C", "field": "coverage_crossing", "negative": False, "positive": True},
    {"id": "D", "field": "overlaps_anchor", "negative": False, "positive": True},
)
AXIS_A_PRIME: dict[str, Any] = {
    "id": "A'",
    "field": "tail_block_kind",
    "negative": "tool_call",
    "positive": "tool_result",
}
AXES: tuple[dict[str, Any], ...] = AXES_V1 + (AXIS_A_PRIME,)
BOUNDARY_V2_TAIL_SHAPE = {
    "call_cut": "cut:toolCall",
    "result_cut": "cut:toolResult",
    "text_end": "text_end",
}
BOUNDARY_V2_RULE = (
    "boundary_v2 reads the tail block's block_kind and nothing else. "
    "The episode tail is the revision block with the greatest (ao_end_seq, block_id). "
    "The subgraph tail is that same maximum over subgraph members. "
    "tool_call -> call_cut. tool_result -> result_cut. unknown -> unknown, and it is reported. "
    "task, orchestrator, and conclusion -> text_end. "
    "Wiring equivalence, read from the v1 checkpoint field tail_shape, not copied by hand: "
    "call_cut = cut:toolCall, result_cut = cut:toolResult, text_end = text_end. "
    "unknown matches nothing. v2's tail_shape copy must equal the v1 field."
)
PROFILE_RULE = (
    "block_kind_profile is the count of subgraph member blocks for each kind in "
    f"BLOCK_KIND_ORDER {BLOCK_KIND_ORDER}. unknown members increment unknown_n and "
    "no weighted count. unknown_n is reported and is not a weighted field. "
    f"The extended score adds {BLOCK_KIND_PROFILE_WEIGHT} when the two profiles are equal "
    "and the v1 score is not None. v1 score is r7b structural_score on FIELD_WEIGHTS_V1."
)
AXIS_A_PRIME_RULE = (
    "Axis A' uses the same 4-block / 3-claim prose skeleton as axes A-D. "
    "Query, positive, and negative receive those same bytes. Relations stay "
    "(influence, influence, skip), anchor index is 1, there is no omitted range, "
    "and no fork is written. The only typed change is the tail block's block_kind: "
    "query and positive are tool_result, negative is tool_call. Every other block "
    "keeps the default unknown, so axes A-D keep the v1 boundary rule. "
    "v1 field scores on an A' pair must be equal. The extended score is what can separate the poles."
)
L1_V1_RULE = (
    "Axes A-D are built by r7b_discriminability._run_pair and checked by "
    "_assert_l1_shape. Fixtures do not pass block_kind, so it stays unknown and "
    "boundary_of stays the v1 rule. Prose is byte-identical inside a pair."
)
LEXICAL_TEXT_RULE = (
    "Fourteen primary revisions in episode_dir order. Each digest is "
    "sha256(utf-8) of the newline join of production _subgraph_retrieval_text "
    "over retrieval_subgraphs in stored order. Zero subgraphs hash the empty "
    "string. v1 checkpoints are /tmp/sb-influence-r8b/checkpoints. "
    "v2 checkpoints are /tmp/sb-influence-r8c/checkpoints. "
    "Subgraph id lists must match. Gate: 14/14 digests equal."
)
RANK_GATE_RULE = (
    "On the v1 checkpoints, recompute L2 structural ranks with "
    "r7b_discriminability._run_l2. Each (episode_dir, session_id, subgraph_id) "
    "structural_rank must equal /tmp/r7b-disc/report.json "
    "l2.queries[*].candidates[*].structural_rank. Gate: 30/30."
)
L2_ABLATION_RULE = (
    "On the v2 checkpoints, the same fork writes and the same tie-break produce "
    "two rank lists. rank_v1 uses FIELD_WEIGHTS_V1. rank_v1_plus uses "
    "FIELD_WEIGHTS_V1_PLUS. rank_delta is rank_v1_plus minus rank_v1. "
    "Both lists are reported. Neither list is a winner. "
    "Lexical top-3 is the production adapter, and it must equal the prefix of "
    "a fresh _rank_document_indexes call."
)
NOT_ANSWERED: tuple[str, ...] = (
    "诊断侧真实效用（仍无裁决、无修复链）。",
    "结构通道在非夹具数据上的**相对收益**（仍然只能报一致性，不能报胜负）。",
    "跨 session／跨 workspace 泛化。",
    "模型写的七段／branches 在检索里的作用（**#125** 的范围）。",
    '`position_kind="cut"` 的语义（本波只把**块级**种类写成类型，没有动失败锚点的位置种类）。',
)
FORBIDDEN_PHRASES: tuple[str, ...] = (
    "结构臂更好",
    "结构臂优于",
    "结构臂胜",
    "结构臂赢",
    "structural arm is better",
    "structural arm wins",
)

R7B: Any = None
SIG: Any = None


def _load() -> None:
    """Bind the frozen R7b modules. Imports stay inside this function."""
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    for entry in (str(root), str(here)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    global R7B
    global SIG
    import r7b_discriminability as r7b_mod
    import r7b_structural_signature as signature_mod

    R7B = r7b_mod
    SIG = signature_mod


_load()


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _assert_frozen_tables() -> None:
    if FIELD_WEIGHTS_V1 != SIG.FIELD_WEIGHTS:
        _fail(f"FIELD_WEIGHTS_V1 != r7b FIELD_WEIGHTS {SIG.FIELD_WEIGHTS}")
    if TIE_BREAK != SIG.TIE_BREAK:
        _fail(f"TIE_BREAK {TIE_BREAK} != {SIG.TIE_BREAK}")
    if SHUFFLE_SEED != R7B.SHUFFLE_SEED:
        _fail(f"SHUFFLE_SEED {SHUFFLE_SEED} != r7b {R7B.SHUFFLE_SEED}")
    if AXES_V1 != R7B.AXES:
        _fail("AXES_V1 != r7b AXES")
    if V1_CHECKPOINT_DIR != R7B.CHECKPOINT_DIR:
        _fail(f"v1 checkpoint dir {V1_CHECKPOINT_DIR} != r7b {R7B.CHECKPOINT_DIR}")
    if SIG.CAP != R7B.DIAGNOSTIC_SUBGRAPH_HIT_CAP:
        _fail("signature cap != production DIAGNOSTIC_SUBGRAPH_HIT_CAP")
    from openviking.session.influence_view import BLOCK_KINDS

    if set(BLOCK_KIND_ORDER) | {"unknown"} != set(BLOCK_KINDS):
        _fail(f"BLOCK_KIND_ORDER does not match BLOCK_KINDS {sorted(BLOCK_KINDS)}")


def _print_policy() -> dict[str, Any]:
    """Print every frozen rule. Callers must do this before any score."""
    _assert_frozen_tables()
    print("R8C TYPED TRUNCATION POLICY (frozen before scores)")
    print(f"SCHEMA {SCHEMA}")
    print(f"LABEL_L1 {LABEL_L1}")
    print(f"SUBGRAPH_TEXT_POLICY_VERSION {R7B.SUBGRAPH_TEXT_POLICY_VERSION}")
    print(f"DIAGNOSTIC_SUBGRAPH_HIT_CAP {R7B.DIAGNOSTIC_SUBGRAPH_HIT_CAP}")
    print(f"FIELD_WEIGHTS_V1 {FIELD_WEIGHTS_V1}")
    print(f"FIELD_WEIGHTS_V1_PLUS {FIELD_WEIGHTS_V1_PLUS}")
    print(f"R7B_FIELD_WEIGHTS {SIG.FIELD_WEIGHTS}")
    print(f"TIE_BREAK {TIE_BREAK}")
    print(f"CAP {SIG.CAP}")
    print(f"SHUFFLE_SEED {SHUFFLE_SEED}")
    print(f"BLOCK_KIND_ORDER {BLOCK_KIND_ORDER}")
    print(f"BLOCK_KIND_PROFILE_WEIGHT {BLOCK_KIND_PROFILE_WEIGHT}")
    print(
        "AXES "
        + str([(item["id"], item["field"], item["positive"], item["negative"]) for item in AXES])
    )
    print(f"BOUNDARY_V2_TAIL_SHAPE {BOUNDARY_V2_TAIL_SHAPE}")
    print(f"R7B_REPORT {R7B_REPORT}")
    print(f"R7B_REPORT_SHA256 {R7B_REPORT_SHA256}")
    print(BOUNDARY_V2_RULE)
    print(PROFILE_RULE)
    print(L1_V1_RULE)
    print(AXIS_A_PRIME_RULE)
    print(LEXICAL_TEXT_RULE)
    print(RANK_GATE_RULE)
    print(L2_ABLATION_RULE)
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
        "block_kind_order": list(BLOCK_KIND_ORDER),
        "block_kind_profile_weight": BLOCK_KIND_PROFILE_WEIGHT,
        "boundary_v2_rule": BOUNDARY_V2_RULE,
        "boundary_v2_tail_shape": BOUNDARY_V2_TAIL_SHAPE,
        "cap": R7B.DIAGNOSTIC_SUBGRAPH_HIT_CAP,
        "field_weights_v1": [list(item) for item in FIELD_WEIGHTS_V1],
        "field_weights_v1_plus": [list(item) for item in FIELD_WEIGHTS_V1_PLUS],
        "l1_v1_rule": L1_V1_RULE,
        "l2_ablation_rule": L2_ABLATION_RULE,
        "label_l1": LABEL_L1,
        "lexical_text_rule": LEXICAL_TEXT_RULE,
        "profile_rule": PROFILE_RULE,
        "r7b_report": str(R7B_REPORT),
        "r7b_report_sha256": R7B_REPORT_SHA256,
        "rank_gate_rule": RANK_GATE_RULE,
        "rules_printed_before_scores": True,
        "shuffle_seed": SHUFFLE_SEED,
        "subgraph_text_policy_version": R7B.SUBGRAPH_TEXT_POLICY_VERSION,
        "tie_break": list(TIE_BREAK),
        "v1_checkpoint_dir": str(V1_CHECKPOINT_DIR),
        "v2_checkpoint_dir": str(V2_CHECKPOINT_DIR),
    }


def _load_records(directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        _fail(f"checkpoints missing: {directory}")
    paths = sorted(directory.glob("*.json"))
    if len(paths) != len(R7B.EPISODE_CODES):
        _fail(f"{directory}: expected {len(R7B.EPISODE_CODES)} checkpoints, found {len(paths)}")
    records = []
    for code, path in zip(R7B.EPISODE_CODES, paths, strict=True):
        if not path.name.startswith(code):
            _fail(f"checkpoint {path.name} does not start with {code}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if str(record.get("episode_dir", ""))[:4] != code:
            _fail(f"{path.name}: episode_dir code mismatch")
        records.append(record)
    return records


def _tail_block(blocks: tuple[Any, ...] | list[Any]) -> Any:
    if not blocks:
        _fail("revision has no blocks")
    return max(blocks, key=lambda block: (block.ao_end_seq, block.block_id))


def _boundary_v2(kind: str) -> str:
    if kind == "tool_call":
        return "call_cut"
    if kind == "tool_result":
        return "result_cut"
    if kind == "unknown":
        return "unknown"
    if kind in {"task", "orchestrator", "conclusion"}:
        return "text_end"
    _fail(f"block_kind {kind!r} is outside the frozen boundary_v2 table")
    return "unknown"


def _profile(revision: Any, subgraph: Any) -> tuple[tuple[int, ...], int]:
    counts = dict.fromkeys(BLOCK_KIND_ORDER, 0)
    by_id = {block.block_id: block for block in revision.blocks}
    unknown_n = 0
    for block_id in subgraph.block_ids:
        block = by_id.get(block_id)
        if block is None:
            continue
        kind = block.block_kind
        if kind == "unknown":
            unknown_n += 1
            continue
        if kind not in counts:
            _fail(f"member block_kind {kind!r} is outside BLOCK_KIND_ORDER")
        counts[kind] += 1
    return tuple(counts[kind] for kind in BLOCK_KIND_ORDER), unknown_n


def _profile_obj(profile: tuple[int, ...], unknown_n: int) -> dict[str, Any]:
    return {
        "counts": {kind: profile[index] for index, kind in enumerate(BLOCK_KIND_ORDER)},
        "unknown_n": unknown_n,
    }


def _score_plus(
    base: int | None, query_profile: tuple[int, ...], profile: tuple[int, ...]
) -> int | None:
    if base is None:
        return None
    if query_profile == profile:
        return base + BLOCK_KIND_PROFILE_WEIGHT
    return base


def _revision_text_digest(revision: Any) -> str:
    parts = [
        R7B._subgraph_retrieval_text(revision, subgraph)
        for subgraph in revision.retrieval_subgraphs
    ]
    return _sha256_text("\n".join(parts))


def _subgraph_ids(revision: Any) -> list[str]:
    return [subgraph.subgraph_id for subgraph in revision.retrieval_subgraphs]


def _gate_lexical_text(
    v1_records: list[dict[str, Any]], v2_records: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = []
    equal = 0
    for left, right in zip(v1_records, v2_records, strict=True):
        if left["episode_dir"] != right["episode_dir"]:
            _fail(f"episode order {left['episode_dir']} != {right['episode_dir']}")
        v1 = R7B.InfluenceViewRevision.from_dict(left["revision"])
        v2 = R7B.InfluenceViewRevision.from_dict(right["revision"])
        ids_v1 = _subgraph_ids(v1)
        ids_v2 = _subgraph_ids(v2)
        if ids_v1 != ids_v2:
            _fail(f"{left['episode_dir']}: subgraph ids {ids_v1} != {ids_v2}")
        digest_v1 = _revision_text_digest(v1)
        digest_v2 = _revision_text_digest(v2)
        matched = digest_v1 == digest_v2
        if matched:
            equal += 1
        rows.append(
            {
                "episode_dir": left["episode_dir"],
                "equal": matched,
                "sha256_v1": digest_v1,
                "sha256_v2": digest_v2,
                "subgraph_ids": ids_v1,
                "subgraph_n": len(ids_v1),
            }
        )
    return {"compared": len(rows), "equal": equal, "rows": rows}


def _gate_ranks(root: Path) -> dict[str, Any]:
    raw = R7B_REPORT.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != R7B_REPORT_SHA256:
        _fail(f"r7b report sha256 {digest} != frozen {R7B_REPORT_SHA256}")
    expected_report = json.loads(raw)
    expected: dict[tuple[str, str, str], int] = {}
    for query in expected_report["l2"]["queries"]:
        for candidate in query["candidates"]:
            key = (query["episode_dir"], candidate["session_id"], candidate["subgraph_id"])
            expected[key] = candidate["structural_rank"]
    if len(expected) != 30:
        _fail(f"r7b report has {len(expected)} structural ranks, expected 30")
    records = R7B._load_records()
    recomputed = R7B._run_l2(records, root / "gate-v1")
    got: dict[tuple[str, str, str], int] = {}
    for query in recomputed["queries"]:
        for candidate in query["candidates"]:
            key = (query["episode_dir"], candidate["session_id"], candidate["subgraph_id"])
            got[key] = candidate["structural_rank"]
    if set(got) != set(expected):
        _fail("recomputed structural-rank keys != r7b report keys")
    equal = sum(1 for key, rank in expected.items() if got[key] == rank)
    rows = [
        {
            "episode_dir": episode,
            "session_id": session_id,
            "structural_rank": rank,
            "subgraph_id": subgraph_id,
        }
        for (episode, session_id, subgraph_id), rank in sorted(expected.items())
    ]
    return {"compared": len(expected), "equal": equal, "rows": rows}


def _episode_kind_counts(record: dict[str, Any]) -> dict[str, int]:
    counts = dict.fromkeys(BLOCK_KIND_ORDER, 0)
    unknown_n = 0
    for block in record["revision"]["blocks"]:
        kind = block.get("block_kind", "unknown")
        if kind == "unknown":
            unknown_n += 1
            continue
        if kind not in counts:
            _fail(f"{record['episode_dir']}: block_kind {kind!r} outside BLOCK_KIND_ORDER")
        counts[kind] += 1
    counts["unknown"] = unknown_n
    return counts


def _gate_boundary(
    v1_records: list[dict[str, Any]], v2_records: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = []
    equal = 0
    unknown_rows = []
    for left, right in zip(v1_records, v2_records, strict=True):
        if left["episode_dir"] != right["episode_dir"]:
            _fail("boundary gate episode order drifted")
        tail_shape = left.get("tail_shape")
        if not isinstance(tail_shape, str):
            _fail(f"{left['episode_dir']}: v1 checkpoint has no tail_shape str")
        if right.get("tail_shape") != tail_shape:
            _fail(
                f"{left['episode_dir']}: v2 tail_shape {right.get('tail_shape')!r} "
                f"!= v1 {tail_shape!r}"
            )
        revision = R7B.InfluenceViewRevision.from_dict(right["revision"])
        tail = _tail_block(revision.blocks)
        boundary = _boundary_v2(tail.block_kind)
        expected = BOUNDARY_V2_TAIL_SHAPE.get(boundary)
        matched = expected == tail_shape
        if matched:
            equal += 1
        row = {
            "block_kind": tail.block_kind,
            "boundary_v2": boundary,
            "episode_dir": left["episode_dir"],
            "match": matched,
            "tail_block_id": tail.block_id,
            "tail_shape": tail_shape,
        }
        rows.append(row)
        if boundary == "unknown":
            unknown_rows.append(row)
    return {
        "compared": len(rows),
        "equal": equal,
        "rows": rows,
        "unknown": unknown_rows,
    }


def _assert_rank_prefix(row: dict[str, Any]) -> None:
    query = row["revisions"]["query"]
    others = [row["revisions"]["pos"], row["revisions"]["neg"]]
    others.sort(key=lambda revision: revision.session_id)
    documents = [
        R7B._subgraph_retrieval_text(revision, revision.retrieval_subgraphs[0])
        for revision in others
    ]
    query_text = R7B._subgraph_retrieval_text(query, query.retrieval_subgraphs[0])
    ranked = R7B._rank_document_indexes(query_text, documents)
    expected = [
        others[index].session_id for index, _breakdown in ranked[: R7B.DIAGNOSTIC_SUBGRAPH_HIT_CAP]
    ]
    actual = [item["session_id"] for item in row["lexical_bm25"]]
    if actual != expected:
        _fail(f"{row['axis_id']} {row['code']}: adapter top {actual} != rank prefix {expected}")


def _with_tail_kind(revision: Any, kind: str) -> Any:
    tail = _tail_block(revision.blocks)
    blocks = [
        replace(block, block_kind=kind) if block.block_id == tail.block_id else block
        for block in revision.blocks
    ]
    draft = R7B.InfluenceViewDraft(
        view_id=revision.view_id,
        purpose=revision.purpose,
        session_id=revision.session_id,
        task_run_id=revision.task_run_id,
        coverage=revision.coverage,
        blocks=blocks,
        claims=list(revision.claims),
        retrieval_subgraphs=list(revision.retrieval_subgraphs),
        critic_rounds=revision.critic_rounds,
    )
    return draft.freeze(
        revision_id=revision.revision_id,
        enforce_checks=False,
        frozen_at=revision.frozen_at,
        construction_mode=revision.construction_mode,
    )


def _run_l1(records: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    R7B._LEXICAL_CALLS = 0
    sources = R7B._sources(records)
    built = []
    for axis in AXES_V1:
        for source in sources:
            row = R7B._run_pair(
                root,
                axis,
                source["code"],
                source["block_prose"],
                source["claim_prose"],
                tag="l1",
            )
            R7B._assert_l1_shape(row, axis)
            _assert_rank_prefix(row)
            for revision in row["revisions"].values():
                for block in revision.blocks:
                    if block.block_kind != "unknown":
                        _fail(f"{axis['id']} {source['code']}: fixture block_kind is not unknown")
            built.append((axis, row))
    per_axis = []
    for axis in AXES_V1:
        group = [R7B._public_pair(row, axis) for item, row in built if item["id"] == axis["id"]]
        per_axis.append(
            {
                "axis": axis["id"],
                "field": axis["field"],
                "label": LABEL_L1,
                "lexical": R7B._aggregate(group, "lexical"),
                "n": len(group),
                "negative": axis["negative"],
                "positive": axis["positive"],
                "structural": R7B._aggregate(group, "structural"),
            }
        )
    prime_rows = []
    spec = {
        "anchor_index": 1,
        "omitted": False,
        "overlaps": False,
        "relations": (R7B.RELATION_INFLUENCE, R7B.RELATION_INFLUENCE, R7B.RELATION_SKIP),
        "write_fork": False,
    }
    for source in sources:
        revisions = {}
        for role, prefix, kind in (
            ("query", "q", "tool_result"),
            ("pos", "p", "tool_result"),
            ("neg", "n", "tool_call"),
        ):
            session_id = f"{prefix}-{AXIS_A_PRIME['id']}-{source['code']}"
            base = R7B._build_revision(
                session_id,
                f"sg-{role}",
                source["block_prose"],
                source["claim_prose"],
                spec,
            )
            revisions[role] = _with_tail_kind(base, kind)
        work = root / f"l1-Ap-{source['code']}"
        work.mkdir(parents=True, exist_ok=True)
        adapter, _store = R7B._adapter(
            work, [revisions["query"], revisions["pos"], revisions["neg"]]
        )
        hits = R7B._lexical(
            adapter,
            revisions["query"].revision_id,
            revisions["query"].retrieval_subgraphs[0].subgraph_id,
        )
        lexical_bm25 = [
            {"bm25": f"{hit.score_breakdown['bm25']:.8f}", "session_id": hit.session_id}
            for hit in hits
        ]
        signatures = {
            role: SIG.structural_signature(
                revisions[role],
                revisions[role].retrieval_subgraphs[0],
                workspace_id=R7B.WORKSPACE_LABEL,
            )
            for role in revisions
        }
        profiles = {
            role: _profile(revisions[role], revisions[role].retrieval_subgraphs[0])
            for role in revisions
        }
        v1_pos = SIG.structural_score(signatures["query"], signatures["pos"])
        v1_neg = SIG.structural_score(signatures["query"], signatures["neg"])
        if v1_pos is None or v1_pos != v1_neg:
            _fail(f"A' {source['code']}: v1 scores are not equal ({v1_pos}, {v1_neg})")
        plus_pos = _score_plus(v1_pos, profiles["query"][0], profiles["pos"][0])
        plus_neg = _score_plus(v1_neg, profiles["query"][0], profiles["neg"][0])
        if plus_pos is None or plus_neg is None or plus_pos <= plus_neg:
            _fail(f"A' {source['code']}: extended scores pos={plus_pos} neg={plus_neg}")
        query_tail = _tail_block(revisions["query"].blocks).block_kind
        pos_tail = _tail_block(revisions["pos"].blocks).block_kind
        neg_tail = _tail_block(revisions["neg"].blocks).block_kind
        if query_tail != "tool_result" or pos_tail != "tool_result" or neg_tail != "tool_call":
            _fail(f"A' {source['code']}: tail kinds {query_tail}, {pos_tail}, {neg_tail}")
        pos_hash = R7B._prose_digest(revisions["pos"])
        neg_hash = R7B._prose_digest(revisions["neg"])
        query_hash = R7B._prose_digest(revisions["query"])
        if not (pos_hash == neg_hash == query_hash):
            _fail(f"A' {source['code']}: prose is not byte-identical")
        row = {
            "axis_id": "A'",
            "code": source["code"],
            "lexical_bm25": lexical_bm25,
            "revisions": revisions,
        }
        _assert_rank_prefix(row)
        lexical_order = [item["session_id"] for item in lexical_bm25]
        struct_order = ["pos", "neg"] if plus_pos > plus_neg else ["neg", "pos"]
        struct_sessions = [
            revisions["pos"].session_id if label == "pos" else revisions["neg"].session_id
            for label in struct_order
        ]
        prime_rows.append(
            {
                "code": source["code"],
                "label": LABEL_L1,
                "lexical": R7B._arm_metrics(
                    lexical_order, revisions["pos"].session_id, revisions["neg"].session_id
                ),
                "negative_profile": _profile_obj(*profiles["neg"]),
                "negative_tail_block_kind": neg_tail,
                "positive_profile": _profile_obj(*profiles["pos"]),
                "positive_tail_block_kind": pos_tail,
                "prose_equal": True,
                "prose_sha256": query_hash,
                "structural": R7B._arm_metrics(
                    struct_sessions, revisions["pos"].session_id, revisions["neg"].session_id
                ),
                "v1_scores_equal": True,
                "v1_structural_score": v1_pos,
            }
        )
    public_prime = [
        {
            "axis": "A'",
            "code": item["code"],
            "label": LABEL_L1,
            "lexical": item["lexical"],
            "structural": item["structural"],
        }
        for item in prime_rows
    ]
    per_axis.append(
        {
            "axis": "A'",
            "field": AXIS_A_PRIME["field"],
            "label": LABEL_L1,
            "lexical": R7B._aggregate(public_prime, "lexical"),
            "n": len(public_prime),
            "negative": AXIS_A_PRIME["negative"],
            "pairs": prime_rows,
            "positive": AXIS_A_PRIME["positive"],
            "structural": R7B._aggregate(public_prime, "structural"),
        }
    )
    return {
        "axes": per_axis,
        "label": LABEL_L1,
        "lexical_call_count": R7B._LEXICAL_CALLS,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _distinct(values: list[Any]) -> int:
    encoded = [json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True) for value in values]
    return len(set(encoded))


def _run_l2(records: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    R7B._LEXICAL_CALLS = 0
    r8a = R7B._load_r8a()
    revisions = [R7B._primary(record) for record in records]
    work = root / "l2-v2"
    work.mkdir(parents=True, exist_ok=True)
    views = R7B.ViewSupersedeIndex(work / "views.json")
    for revision in revisions:
        views.register(revision)
    store = R7B._new_store(work / "store")
    forks_adapter = R7B.ForkBranchAdapter(work / "store", store)
    adapter = R7B.RetrievalSubgraphAdapter(views, forks_adapter, session_acl=R7B._acl)
    written: list[Any] = []
    seen_ids: set[str] = set()
    fork_writes = []
    for caliber in ("narrow", "wide"):
        for record in records:
            flag = record["in_narrow"] if caliber == "narrow" else record["in_wide"]
            if not flag:
                continue
            payload_rev = record["revision"]
            if caliber == "wide" and record["revision_wide"] is not None:
                payload_rev = record["revision_wide"]
            revision = R7B.InfluenceViewRevision.from_dict(payload_rev)
            payload = r8a._fork_payload(record, revision)
            fork = r8a._write_fork(store, revision, record, payload)
            fork_writes.append(
                {
                    "caliber": caliber,
                    "episode_dir": record["episode_dir"],
                    "fork_revision_id": fork.revision_id,
                }
            )
            if fork.revision_id not in seen_ids:
                seen_ids.add(fork.revision_id)
                written.append(fork)
    queries = []
    query_consts: dict[str, list[Any]] = {name: [] for name, _weight in FIELD_WEIGHTS_V1}
    query_consts["block_kind_profile"] = []
    query_consts["boundary_v2"] = []
    rank_deltas = 0
    compared = 0
    for record in records:
        if record["episode_dir"][:4] not in R7B.NARROW_CODES:
            continue
        revision = R7B._primary(record)
        subgraph = revision.retrieval_subgraphs[0]
        query_forks = R7B._forks_inside(written, revision, subgraph)
        query_sig = SIG.structural_signature(
            revision, subgraph, workspace_id=R7B.WORKSPACE_LABEL, forks=query_forks
        )
        query_profile, query_unknown = _profile(revision, subgraph)
        query_boundary_v2 = _boundary_v2(
            _tail_block(
                [block for block in revision.blocks if block.block_id in set(subgraph.block_ids)]
            ).block_kind
        )
        for name, _weight in FIELD_WEIGHTS_V1:
            query_consts[name].append(getattr(query_sig, name))
        query_consts["block_kind_profile"].append(query_profile)
        query_consts["boundary_v2"].append(query_boundary_v2)
        hits = R7B._lexical(adapter, revision.revision_id, subgraph.subgraph_id)
        candidates: list[tuple[Any, Any]] = []
        documents: list[str] = []
        for other in views.revisions_for_purpose(R7B.PURPOSE_POST_RUN_INDEX):
            if other.revision_id == revision.revision_id:
                continue
            for item in other.retrieval_subgraphs:
                candidates.append((other, item))
                documents.append(R7B._subgraph_retrieval_text(other, item))
        query_text = R7B._subgraph_retrieval_text(revision, subgraph)
        ranked = R7B._rank_document_indexes(query_text, documents)
        ranked_keys = [
            [candidates[index][0].session_id, candidates[index][1].subgraph_id]
            for index, _breakdown in ranked
        ]
        hit_keys = [[hit.session_id, hit.subgraph.subgraph_id] for hit in hits]
        prefix = ranked_keys[: R7B.DIAGNOSTIC_SUBGRAPH_HIT_CAP]
        if hit_keys != prefix:
            _fail(f"{record['episode_dir']}: adapter top-3 {hit_keys} != rank prefix {prefix}")
        scored_v1: list[tuple[int, str, str]] = []
        scored_plus: list[tuple[int, str, str]] = []
        details = []
        for other, item in candidates:
            forks = R7B._forks_inside(written, other, item)
            cand_sig = SIG.structural_signature(
                other, item, workspace_id=R7B.WORKSPACE_LABEL, forks=forks
            )
            profile, unknown_n = _profile(other, item)
            members = [block for block in other.blocks if block.block_id in set(item.block_ids)]
            boundary_v2 = _boundary_v2(_tail_block(members).block_kind)
            base = SIG.structural_score(query_sig, cand_sig)
            if base is None:
                continue
            plus = _score_plus(base, query_profile, profile)
            if plus is None:
                continue
            key = (other.session_id, item.subgraph_id)
            scored_v1.append((base, key[0], key[1]))
            scored_plus.append((plus, key[0], key[1]))
            details.append(
                {
                    "boundary": cand_sig.boundary,
                    "boundary_v2": boundary_v2,
                    "block_kind_profile": _profile_obj(profile, unknown_n),
                    "key": key,
                    "score_v1": base,
                    "score_v1_plus": plus,
                }
            )
        rank_v1 = {
            (session_id, subgraph_id): index
            for index, (_score, session_id, subgraph_id) in enumerate(
                sorted(scored_v1, key=lambda item: (-item[0], item[1], item[2])), start=1
            )
        }
        rank_plus = {
            (session_id, subgraph_id): index
            for index, (_score, session_id, subgraph_id) in enumerate(
                sorted(scored_plus, key=lambda item: (-item[0], item[1], item[2])), start=1
            )
        }
        readout = []
        for info in details:
            key = info["key"]
            delta = rank_plus[key] - rank_v1[key]
            compared += 1
            if delta != 0:
                rank_deltas += 1
            readout.append(
                {
                    "block_kind_profile": info["block_kind_profile"],
                    "boundary": info["boundary"],
                    "boundary_v2": info["boundary_v2"],
                    "rank_delta": delta,
                    "rank_v1": rank_v1[key],
                    "rank_v1_plus": rank_plus[key],
                    "score_v1": info["score_v1"],
                    "score_v1_plus": info["score_v1_plus"],
                    "session_id": key[0],
                    "subgraph_id": key[1],
                }
            )
        readout.sort(key=lambda item: (item["session_id"], item["subgraph_id"]))
        top_v1 = [
            [session_id, subgraph_id]
            for session_id, subgraph_id in sorted(rank_v1, key=lambda key: rank_v1[key])[
                : R7B.DIAGNOSTIC_SUBGRAPH_HIT_CAP
            ]
        ]
        top_plus = [
            [session_id, subgraph_id]
            for session_id, subgraph_id in sorted(rank_plus, key=lambda key: rank_plus[key])[
                : R7B.DIAGNOSTIC_SUBGRAPH_HIT_CAP
            ]
        ]
        keys = sorted(rank_v1)
        kept = {tuple(item) for item in ranked_keys}
        lexical_rank = {tuple(key): index for index, key in enumerate(ranked_keys, start=1)}
        bottom = len(ranked_keys) + 1
        for other, item in candidates:
            key = (other.session_id, item.subgraph_id)
            if key not in kept and key in rank_v1:
                lexical_rank[key] = bottom
        spearman_v1 = R7B._spearman(
            [float(lexical_rank[key]) for key in keys],
            [float(rank_v1[key]) for key in keys],
        )
        spearman_plus = R7B._spearman(
            [float(lexical_rank[key]) for key in keys],
            [float(rank_plus[key]) for key in keys],
        )
        queries.append(
            {
                "adapter_top3": hit_keys,
                "candidates": readout,
                "episode_dir": record["episode_dir"],
                "label": "真数据切片，无裁决，不算判别力",
                "lexical_top3": prefix,
                "query_block_kind_profile": _profile_obj(query_profile, query_unknown),
                "query_boundary": query_sig.boundary,
                "query_boundary_v2": query_boundary_v2,
                "rank_prefix_equal": hit_keys == prefix,
                "spearman_v1": spearman_v1,
                "spearman_v1_plus": spearman_plus,
                "structural_top3_v1": top_v1,
                "structural_top3_v1_plus": top_plus,
                "top3_jaccard_v1": R7B._jaccard(prefix, top_v1),
                "top3_jaccard_v1_plus": R7B._jaccard(prefix, top_plus),
            }
        )
    constancy = []
    non_constant = []
    for name in [item[0] for item in FIELD_WEIGHTS_V1] + ["block_kind_profile", "boundary_v2"]:
        count = _distinct(query_consts[name])
        row = {"distinct": count, "field": name, "n": len(query_consts[name])}
        constancy.append(row)
        if count > 1:
            non_constant.append(name)
    return {
        "block_kind_distribution": [
            {
                "block_kind_counts": _episode_kind_counts(record),
                "episode_dir": record["episode_dir"],
            }
            for record in records
        ],
        "candidates_compared": compared,
        "distinct_fork_nodes": len(written),
        "field_constancy": constancy,
        "fork_writes": fork_writes,
        "label": "真数据切片，无裁决，不算判别力",
        "lexical_call_count": R7B._LEXICAL_CALLS,
        "non_constant_fields": non_constant,
        "non_constant_n": len(non_constant),
        "queries": queries,
        "rank_delta_nonzero": rank_deltas,
        "rank_note": "名次差是 v1+ 名次减去 v1 名次。此节只报分布与一致性，不报胜负。",
    }


def _assemble(
    policy: dict[str, Any],
    gates: dict[str, Any],
    l1: dict[str, Any] | None,
    l2: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "finding": (
            f"{LABEL_L1} "
            "L1 各轴读数见 l1.axes。"
            "L2 见 l2.queries 的两套名次、block_kind 分布与 field_constancy。"
            "此报告不比较胜负。"
        ),
        "gates": gates,
        "l1": l1,
        "l2": l2,
        "lexical_arm": {
            "DIAGNOSTIC_SUBGRAPH_HIT_CAP": R7B.DIAGNOSTIC_SUBGRAPH_HIT_CAP,
            "SUBGRAPH_TEXT_POLICY_VERSION": R7B.SUBGRAPH_TEXT_POLICY_VERSION,
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


def _gates(report: dict[str, Any], text: str) -> None:
    lexical = report["lexical_arm"]
    if lexical["SUBGRAPH_TEXT_POLICY_VERSION"] != R7B.SUBGRAPH_TEXT_POLICY_VERSION:
        _fail("report missing production SUBGRAPH_TEXT_POLICY_VERSION")
    if lexical["DIAGNOSTIC_SUBGRAPH_HIT_CAP"] != R7B.DIAGNOSTIC_SUBGRAPH_HIT_CAP:
        _fail("report missing production DIAGNOSTIC_SUBGRAPH_HIT_CAP")
    if "retrieve_historical_subgraphs" not in lexical["callable"]:
        _fail("lexical arm callable is not the production method")
    gates = report["gates"]
    if gates["structural_ranks"]["equal"] != 30 or gates["structural_ranks"]["compared"] != 30:
        _fail(
            "structural rank gate "
            f"{gates['structural_ranks']['equal']}/{gates['structural_ranks']['compared']}"
        )
    if gates["lexical_text"]["equal"] != 14 or gates["lexical_text"]["compared"] != 14:
        _fail(
            f"lexical text gate {gates['lexical_text']['equal']}/{gates['lexical_text']['compared']}"
        )
    if gates["boundary_v2"]["equal"] != 14 or gates["boundary_v2"]["compared"] != 14:
        _fail(
            f"boundary_v2 gate {gates['boundary_v2']['equal']}/{gates['boundary_v2']['compared']}"
        )
    if gates["boundary_v2"]["unknown"]:
        _fail(f"boundary_v2 unknown rows: {gates['boundary_v2']['unknown']}")
    l1 = report.get("l1")
    if l1 is not None:
        if l1.get("label") != LABEL_L1:
            _fail("L1 missing fixture label")
        ids = [item["axis"] for item in l1["axes"]]
        if ids != [item["id"] for item in AXES]:
            _fail(f"axis order {ids}")
        for axis in l1["axes"]:
            if axis.get("label") != LABEL_L1:
                _fail(f"axis {axis['axis']} missing fixture label")
    l2 = report.get("l2")
    if l2 is not None:
        for query in l2["queries"]:
            if query["adapter_top3"] != query["lexical_top3"] or not query["rank_prefix_equal"]:
                _fail(f"{query['episode_dir']}: adapter top-3 is not the rank prefix")
        if l2.get("label") != "真数据切片，无裁决，不算判别力":
            _fail("L2 missing distribution label")
    if report.get("not_answered") != list(NOT_ANSWERED):
        _fail("not_answered drifted from the frozen five lines")
    for line in NOT_ANSWERED:
        if line not in text and line not in report["not_answered"]:
            _fail(f"missing 不回答 line: {line}")
    for phrase in FORBIDDEN_PHRASES:
        if phrase in text:
            _fail(f"forbidden phrase present: {phrase}")


def _load_existing(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="R8c typed truncation retest")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("l1", "l2", "all"), default="all")
    args = parser.parse_args()
    policy = _print_policy()
    existing = _load_existing(args.out) if args.resume else None
    want_l1 = args.stage in ("l1", "all")
    want_l2 = args.stage in ("l2", "all")
    l1 = existing.get("l1") if existing and existing.get("l1") else None
    l2 = existing.get("l2") if existing and existing.get("l2") else None
    v1_records = _load_records(V1_CHECKPOINT_DIR)
    v2_records = _load_records(V2_CHECKPOINT_DIR)
    with tempfile.TemporaryDirectory(prefix="r8c-typed-") as tmp:
        root = Path(tmp)
        print("STAGE gates")
        gates = {
            "boundary_v2": _gate_boundary(v1_records, v2_records),
            "lexical_text": _gate_lexical_text(v1_records, v2_records),
            "structural_ranks": _gate_ranks(root),
        }
        if want_l1 and l1 is None:
            print("STAGE l1")
            l1 = _run_l1(v1_records, root)
        if want_l2 and l2 is None:
            print("STAGE l2")
            l2 = _run_l2(v2_records, root)
    report = _assemble(policy, gates, l1, l2)
    text = _dump(report)
    _gates(report, text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"GATE structural_ranks {gates['structural_ranks']['equal']}/30")
    print(f"GATE lexical_text {gates['lexical_text']['equal']}/14")
    print(f"GATE boundary_v2 {gates['boundary_v2']['equal']}/14")
    print(f"WROTE {args.out}")
    print(f"SHA256 {digest}")


if __name__ == "__main__":
    main()
