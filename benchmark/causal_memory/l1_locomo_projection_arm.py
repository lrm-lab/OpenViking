#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""L1 zero-LLM LoCoMo arm for the projection channel.

Same ingest and same any-hit@{5,10} as ``l1_locomo_hitk.py``, plus a B arm
that actually wires the deterministic fallback view:

    messages -> AOLedger (one segment_id per message)
             -> trajectory_index_hook._page_ao_records
             -> trajectory_index_hook._view_ao_record
             -> build_minimal_fallback_view
             -> ProjectionRegistry.register(acl_labels=("locomo",))
             -> InfluenceProjectionAdapter(views, content_resolver)

P keeps the facts profile (0/0/12). On this branch the fact lane is paused,
so P must stay empty. T is the same token-overlap reference. B uses
``{fork_branch: 0, projection: 4, segment_atom: 0}``.

``l1_locomo_hitk.py`` is the producer of the frozen fact-lane baseline and
is not modified by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from openviking.session.ao_ledger import AOLedger
from openviking.session.bm25 import tokenize as bm25_tokenize
from openviking.session.influence_projection import ProjectionRegistry
from openviking.session.influence_view import (
    PURPOSE_POST_RUN_INDEX,
    InfluenceViewRevision,
    build_minimal_fallback_view,
)
from openviking.session.retrieval_facade import (
    FACT_LANE_PAUSED,
    ForkBranchAdapter,
    HybridRetrievalFacade,
    InfluenceProjectionAdapter,
    SegmentAtomAdapter,
)
from openviking.session.trajectory_index_hook import _page_ao_records, _view_ao_record

LABELS = ["locomo"]
FACTS_PROFILE = {"fork_branch": 0, "projection": 0, "segment_atom": 12}
PROJECTION_PROFILE = {"fork_branch": 0, "projection": 4, "segment_atom": 0}
DISJOINT_LABEL = "not-locomo"
DEFAULT_DATA = "/home/zhoujie22/river2_0/evol_bench/LoCoMo/data/locomo/locomo10.json"

# Full-run numbers this arm must reproduce. Printed with the same .1% format
# as l1_locomo_hitk.py. A mismatch means this script drifted; do not edit the
# frozen baseline to absorb it.
_BASELINE_P_HIT5 = "0.0%"
_BASELINE_P_HIT10 = "0.0%"
_BASELINE_P_ZERO = "100.0%"
_BASELINE_T_HIT5 = "56.0%"
_BASELINE_T_HIT10 = "61.8%"

_STOP = set(
    "the a an to of in on at for with and or is are was were be been this that these those you your "
    "we they it its his her their i not no do does did can could would should must may might will "
    "when where why how what which who whom whose using use used needs need provide provides ensure "
    "from by as into out over under then than them there here all any each every some such only also "
    "did didn't don't about after before during between more most other another same first last next "
    "tell say says said tell me please know like just also very really much many few little".split()
)


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[0-9a-z]+", text.lower().replace("’", "'"))
    return [w for w in words if len(w) > 2 and w not in _STOP]


def _question_keywords(question: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for w in _tokens(question):
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def ingest_conversation(conv: dict, work: Path) -> tuple[str, list[dict]]:
    """One session per conversation; one AO atom per message.

    ``segment_id`` is one message (``session_N:dia_id``) and ``participant_id``
    is that message's speaker. The fallback view then emits one block per
    message: a segment may hold only one participant's contiguous AO run.
    """
    session_id = f"locomo-{conv['sample_id']}"
    ledger = AOLedger(work / session_id, session_id)
    atoms: list[dict] = []
    sessions = sorted(
        ((k, v) for k, v in conv["conversation"].items() if re.fullmatch(r"session_\d+", k)),
        key=lambda kv: int(kv[0].split("_")[1]),
    )
    for sname, msgs in sessions:
        for m in msgs:
            speaker = m["speaker"]
            dia_id = m["dia_id"]
            if not isinstance(speaker, str) or not speaker.strip():
                _fail(f"{session_id} message {dia_id!r} has no speaker")
            if not isinstance(dia_id, str) or not dia_id.strip():
                _fail(f"{session_id} message is missing dia_id")
            segment_id = f"{sname}:{dia_id}"
            rec = ledger.append(
                action={
                    "record_kind": "atom",
                    "acl_labels": LABELS,
                    "speaker": speaker,
                    "text": m["text"],
                    "dia_id": dia_id,
                    "locomo_session": sname,
                    "segment_id": segment_id,
                    "participant_id": speaker,
                },
                observation={"synopsis": m["text"]},
                message_ref={"message_id": dia_id},
            )
            atoms.append(
                {
                    "ao_id": rec.ao_id,
                    "dia_id": dia_id,
                    "text": m["text"],
                    "tokens": _tokens(m["text"]),
                    "segment_id": segment_id,
                    "participant_id": speaker.strip(),
                }
            )
    return session_id, atoms


def _load_view_records(work: Path, session_id: str, atoms: list[dict]) -> list[dict[str, Any]]:
    """Page the ledger the way the commit hook does, then lift each row.

    ``_page_ao_records`` is ``list_ao_ledger(..., include_live=True)`` across
    cursors. Raw rows keep ``segment_id`` / ``participant_id`` on ``action``
    and have no ``content_hash``. ``build_minimal_fallback_view`` requires
    those three fields at the top of each AO record, which is what
    ``_view_ao_record`` produces in production.
    """
    rows = _page_ao_records(work / session_id, session_id)
    records = [_view_ao_record(item, session_id) for item in rows]
    by_ao = {atom["ao_id"]: atom for atom in atoms}
    if len(records) != len(atoms) or set(by_ao) != {row["ao_id"] for row in records}:
        _fail(
            f"{session_id}: paged ledger rows {len(records)} do not match ingested atoms {len(atoms)}"
        )
    segments = [row["segment_id"] for row in records]
    if len(set(segments)) != len(segments):
        _fail(f"{session_id}: segment_id is not unique per message")
    for row in records:
        atom = by_ao[row["ao_id"]]
        if (
            row["segment_id"] != atom["segment_id"]
            or row["participant_id"] != atom["participant_id"]
        ):
            _fail(
                f"{session_id}: ledger did not adopt action segment/participant for {atom['dia_id']}"
            )
    return records


def build_projection(
    work: Path, session_id: str, atoms: list[dict]
) -> tuple[InfluenceViewRevision, ProjectionRegistry]:
    revision = build_minimal_fallback_view(
        {
            "session_id": session_id,
            "task_run_id": f"l1-{session_id}",
            "purpose": PURPOSE_POST_RUN_INDEX,
            # Stable id. The builder otherwise mints a uuid, and that id is
            # part of the card id sitting in the BM25 document. Ties would
            # then depend on the uuid.
            "revision_id": f"post-run:{session_id}:l1",
            "view_id": f"{session_id}:{PURPOSE_POST_RUN_INDEX}",
            "ao_records": _load_view_records(work, session_id, atoms),
        },
        clock=None,
    )
    if revision.construction_mode != "fallback":
        _fail(f"{session_id}: construction_mode={revision.construction_mode!r}, expected fallback")
    if revision.claims:
        _fail(f"{session_id}: post_run_index fallback attached {len(revision.claims)} claims")
    if len(revision.blocks) != len(atoms):
        _fail(f"{session_id}: len(blocks)={len(revision.blocks)} != len(atoms)={len(atoms)}")
    registry = ProjectionRegistry(work / "projections.jsonl")
    cards = registry.register(revision, acl_labels=tuple(LABELS))
    visible = registry.search(principal_labels=tuple(LABELS))
    hidden = registry.search(principal_labels=(DISJOINT_LABEL,))
    if len(cards) != len(revision.blocks) or len(visible) != len(revision.blocks):
        _fail(
            f"{session_id}: registry cards={len(cards)} search(locomo)={len(visible)} "
            f"blocks={len(revision.blocks)}"
        )
    if not visible:
        _fail(f"{session_id}: registry.search(locomo) is empty")
    if hidden:
        _fail(f"{session_id}: disjoint label {DISJOINT_LABEL!r} returned {len(hidden)} cards")
    return revision, registry


def build_facade(
    work: Path,
    session_id: str,
    registry: ProjectionRegistry,
    revision: InfluenceViewRevision,
    ao_text: dict[str, str],
) -> HybridRetrievalFacade:
    def resolve(ao_id: str) -> str:
        text = ao_text.get(ao_id, "")
        return text if isinstance(text, str) else ""

    return HybridRetrievalFacade(
        fork_branch=ForkBranchAdapter(work),
        projection=InfluenceProjectionAdapter(registry, views=[revision], content_resolver=resolve),
        segment_atom=SegmentAtomAdapter(work / session_id, session_id),
    )


def production_ranked_dias(
    facade: HybridRetrievalFacade, keywords: list[str], ao_to_dia: dict[str, str]
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for kw in keywords:
        res = facade.retrieve(kw, principal_labels=LABELS, profile=FACTS_PROFILE)
        for card in res.cards:
            dia = ao_to_dia.get((card.source_pointer or {}).get("ao_id"), "")
            if dia and dia not in seen:
                seen.add(dia)
                ordered.append(dia)
    return ordered


def _blocks_for_card(card: Any, revision: InfluenceViewRevision) -> list[Any]:
    """Block objects behind one projection card.

    A block card is that block. A claim card contributes both endpoint
    blocks, in source then target order. Dias come only from those blocks'
    ``source_refs``, so one block that spans several AOs contributes every
    dia. ``post_run_index`` fallback builds no claims; the claim branch is
    the same lookup if a later view has them.
    """
    entity_id = (card.source_pointer or {}).get("block_or_claim_id") or ""
    if card.kind == "claim_projection":
        claim = next((item for item in revision.claims if item.claim_id == entity_id), None)
        if claim is None:
            return []
        wanted = (claim.source_block_id, claim.target_block_id)
        found = {block.block_id: block for block in revision.blocks}
        return [found[block_id] for block_id in wanted if block_id in found]
    found = {block.block_id: block for block in revision.blocks}
    block = found.get(entity_id)
    return [] if block is None else [block]


def dias_for_card(
    card: Any, revision: InfluenceViewRevision, ao_to_dia: dict[str, str]
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for block in _blocks_for_card(card, revision):
        for ref in block.source_refs:
            dia = ao_to_dia.get(ref.ao_id, "")
            if dia and dia not in seen:
                seen.add(dia)
                ordered.append(dia)
    return ordered


def _keyword_in_text(keyword: str, text: str) -> bool:
    return keyword.casefold() in set(bm25_tokenize(text))


def projection_ranked_dias(
    facade: HybridRetrievalFacade,
    keywords: list[str],
    revision: InfluenceViewRevision,
    ao_to_dia: dict[str, str],
    ao_text: dict[str, str],
    trace: dict[str, Any],
    sample_id: str,
    question: str,
    gold: set[str],
) -> list[str]:
    """Union projection cards across keywords, first-seen dia order.

    Each keyword is its own ``retrieve`` (profile projection=4). The adapter
    BM25-ranks block cards whose document includes the resolved AO text, drops
    zero scores, and keeps four. Default provisional slots are also four, so
    the exploration seat is the fourth BM25 hit and the returned order stays
    BM25 order. Dias inside one card follow ``source_refs``.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for kw in keywords:
        res = facade.retrieve(kw, principal_labels=LABELS, profile=PROJECTION_PROFILE)
        for card in res.cards:
            if not trace.get("example", {}).get("dia_in_gold"):
                _maybe_trace(
                    trace, card, revision, ao_to_dia, ao_text, kw, sample_id, question, gold
                )
            for dia in dias_for_card(card, revision, ao_to_dia):
                if dia not in seen:
                    seen.add(dia)
                    ordered.append(dia)
    return ordered


def _maybe_trace(
    trace: dict[str, Any],
    card: Any,
    revision: InfluenceViewRevision,
    ao_to_dia: dict[str, str],
    ao_text: dict[str, str],
    keyword: str,
    sample_id: str,
    question: str,
    gold: set[str],
) -> None:
    """Keep one card whose keyword occurs in the AO body.

    Prefer a card whose dia is in the question's gold, so the printed example
    is a scored hit and not only a lexical neighbor.
    """
    have_gold = bool(trace.get("example", {}).get("dia_in_gold"))
    if have_gold:
        return
    for block in _blocks_for_card(card, revision):
        for ref in block.source_refs:
            text = ao_text.get(ref.ao_id, "")
            if not isinstance(text, str) or not _keyword_in_text(keyword, text):
                continue
            dia = ao_to_dia.get(ref.ao_id, "")
            if not dia:
                continue
            in_gold = dia in gold
            if trace.get("example") and not in_gold:
                continue
            trace["example"] = {
                "sample_id": sample_id,
                "question": question,
                "gold": sorted(gold),
                "keyword": keyword,
                "card_kind": card.kind,
                "view_revision_id": (card.source_pointer or {}).get("view_revision_id"),
                "block_or_claim_id": (card.source_pointer or {}).get("block_or_claim_id"),
                "block_id": block.block_id,
                "ao_id": ref.ao_id,
                "dia_id": dia,
                "dia_in_gold": in_gold,
                "keyword_in_ao_text": True,
                "text_prefix": text[:180],
            }
            if in_gold:
                return


def token_ranked_dias(atoms: list[dict], q_tokens: list[str]) -> list[str]:
    if not q_tokens:
        return []
    df: Counter = Counter()
    for atom in atoms:
        df.update(set(atom["tokens"]))
    n = len(atoms)
    q_set = set(q_tokens)
    scored = []
    for atom in atoms:
        overlap = q_set & set(atom["tokens"])
        if not overlap:
            continue
        score = sum(math.log(n / df[t]) for t in overlap)
        scored.append((score, atom["sequence"] if "sequence" in atom else 0, atom["dia_id"]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [dia for _, _, dia in scored]


def any_hit(ranked: list[str], gold: set[str], k: int) -> bool:
    return bool(gold & set(ranked[:k]))


def _pct(count: int, total: int) -> str:
    return f"{count / total:.1%}"


def _print_trace(trace: dict[str, Any]) -> None:
    example = trace.get("example")
    print("\n=== chain trace (B card -> block -> AO -> dia) ===")
    if not example:
        print("none")
        return
    for key in (
        "sample_id",
        "question",
        "gold",
        "keyword",
        "card_kind",
        "view_revision_id",
        "block_or_claim_id",
        "block_id",
        "ao_id",
        "dia_id",
        "dia_in_gold",
        "keyword_in_ao_text",
        "text_prefix",
    ):
        print(f"  {key}: {example[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out", default=None, help="write per-question JSON here")
    parser.add_argument("--limit-convs", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    conversations = json.load(open(args.data, encoding="utf-8"))
    if args.limit_convs:
        conversations = conversations[: args.limit_convs]

    print(
        "paging: trajectory_index_hook._page_ao_records "
        "(list_ao_ledger include_live=True); "
        "row shape: trajectory_index_hook._view_ao_record"
    )
    print(
        f"FACT_LANE_PAUSED={FACT_LANE_PAUSED} "
        "(openviking/session/retrieval_facade.py:114); "
        "P profile projection=0 so a wired registry cannot fill P"
    )

    per_question: list[dict[str, Any]] = []
    overall: dict[str, Counter] = {"P": Counter(), "T": Counter(), "B": Counter()}
    trace: dict[str, Any] = {}
    total_q = 0
    for conv in conversations:
        work = Path(tempfile.mkdtemp(prefix=f"l1b-{conv['sample_id']}-"))
        session_id, atoms = ingest_conversation(conv, work)
        revision, registry = build_projection(work, session_id, atoms)
        ao_to_dia = {atom["ao_id"]: atom["dia_id"] for atom in atoms}
        ao_text = {atom["ao_id"]: atom["text"] for atom in atoms}
        facade = build_facade(work, session_id, registry, revision, ao_text)
        for q in conv["qa"]:
            gold = set(q.get("evidence") or [])
            if not gold:
                continue
            total_q += 1
            kws = _question_keywords(q["question"])
            p_ranked = production_ranked_dias(facade, kws, ao_to_dia)
            t_ranked = token_ranked_dias(atoms, _tokens(q["question"]))
            b_ranked = projection_ranked_dias(
                facade,
                kws,
                revision,
                ao_to_dia,
                ao_text,
                trace,
                conv["sample_id"],
                q["question"],
                gold,
            )
            row = {
                "sample_id": conv["sample_id"],
                "category": q.get("category"),
                "question": q["question"],
                "gold": sorted(gold),
                "p_union_size": len(p_ranked),
                "B_union_size": len(b_ranked),
                "P_hit5": any_hit(p_ranked, gold, 5),
                "P_hit10": any_hit(p_ranked, gold, 10),
                "T_hit5": any_hit(t_ranked, gold, 5),
                "T_hit10": any_hit(t_ranked, gold, 10),
                "B_hit5": any_hit(b_ranked, gold, 5),
                "B_hit10": any_hit(b_ranked, gold, 10),
            }
            per_question.append(row)
            for arm in ("P", "T", "B"):
                overall[arm]["hit5"] += row[f"{arm}_hit5"]
                overall[arm]["hit10"] += row[f"{arm}_hit10"]
        print(
            f"[{conv['sample_id']}] atoms={len(atoms)} blocks={len(revision.blocks)} "
            f"mode={revision.construction_mode} cards={len(revision.blocks)} "
            f"qa_with_evidence={sum(1 for r in per_question if r['sample_id'] == conv['sample_id'])}"
        )

    if total_q == 0:
        _fail("no questions with evidence")

    zero_p = sum(1 for row in per_question if row["p_union_size"] == 0)
    zero_b = sum(1 for row in per_question if row["B_union_size"] == 0)
    _print_trace(trace)

    print("\n=== overall (zero-LLM evidence any-hit) ===")
    print(f"questions={total_q}  zero-union(P)={zero_p} ({_pct(zero_p, total_q)})")
    print(f"zero-union(B)={zero_b} ({_pct(zero_b, total_q)})")
    for arm in ("P", "T", "B"):
        print(
            f"{arm}: hit@5={_pct(overall[arm]['hit5'], total_q)}  "
            f"hit@10={_pct(overall[arm]['hit10'], total_q)}"
        )
    print("\n=== per category (P@10 / T@10 / B@10) ===")
    by_cat: dict[Any, Counter] = {}
    for row in per_question:
        bucket = by_cat.setdefault(row["category"], Counter())
        bucket["n"] += 1
        bucket["p10"] += row["P_hit10"]
        bucket["t10"] += row["T_hit10"]
        bucket["b10"] += row["B_hit10"]
    for cat in sorted(by_cat):
        bucket = by_cat[cat]
        print(
            f"cat {cat}: n={bucket['n']}  P@10={_pct(bucket['p10'], bucket['n'])}  "
            f"T@10={_pct(bucket['t10'], bucket['n'])}  B@10={_pct(bucket['b10'], bucket['n'])}"
        )

    if args.out:
        Path(args.out).write_text(
            json.dumps(per_question, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"\nper-question rows -> {args.out}")

    if zero_b == total_q:
        _fail(
            "B arm empty union is 100%. The projection channel returned no card; "
            "this is a wiring failure (labels, content_resolver, or bound view), not a score."
        )
    if "example" not in trace:
        _fail(
            "B arm returned cards, but no card's keyword occurs in the resolved AO text. "
            "Hits are not grounded in the projection chain."
        )
    if not args.limit_convs:
        got = {
            "P_hit5": _pct(overall["P"]["hit5"], total_q),
            "P_hit10": _pct(overall["P"]["hit10"], total_q),
            "P_zero": _pct(zero_p, total_q),
            "T_hit5": _pct(overall["T"]["hit5"], total_q),
            "T_hit10": _pct(overall["T"]["hit10"], total_q),
        }
        expected = {
            "P_hit5": _BASELINE_P_HIT5,
            "P_hit10": _BASELINE_P_HIT10,
            "P_zero": _BASELINE_P_ZERO,
            "T_hit5": _BASELINE_T_HIT5,
            "T_hit10": _BASELINE_T_HIT10,
        }
        if got != expected:
            _fail(f"P/T drifted from the frozen baseline: got {got} expected {expected}")
        print("P/T match the frozen baseline (P 0.0%/0.0%/empty 100.0%, T 56.0%/61.8%).")


if __name__ == "__main__":
    main()
