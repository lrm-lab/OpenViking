#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""R8a: zero-LLM influence-graph and fork extraction on SkillsBench traces.

Deterministic stand-in. No model is called. Task-success generalization is
not measured: the recorded traces have no grader verdict.

Rules below are fixed before any score is computed. They are printed at
startup. Do not edit them to absorb a miss.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import openviking.session.ao_ledger as ao_ledger_mod
from openviking.session.ao_ledger import AOLedger
from openviking.session.bm25 import BM25Index, tokenize
from openviking.session.causal_experiences import (
    SEVEN_SECTION_KEYS,
    CausalExperiencesStore,
)
from openviking.session.extraction_write import freeze_extracted_view
from openviking.session.influence_projection import ProjectionRegistry
from openviking.session.influence_view import (
    MAX_RETRIEVAL_SUBGRAPH_BLOCKS,
    PURPOSE_POST_RUN_INDEX,
    RELATION_INFLUENCE,
    RELATION_SKIP,
    ROLE_CONCLUSION,
    ROLE_TASK,
    AORange,
    CoverageManifest,
    HighSeverityViolation,
    InfluenceBlock,
    InfluenceClaim,
    InfluenceEvidenceRef,
    InfluenceViewRevision,
    OmittedRange,
    RetrievalSubgraph,
    SegmentRef,
    SourceRef,
)
from openviking.session.retrieval_facade import (
    CAUSAL_MEMORY_CAP,
    ForkBranchAdapter,
    HybridRetrievalFacade,
    InfluenceProjectionAdapter,
    SegmentAtomAdapter,
)
from openviking.session.subgraph_forks import (
    default_failure_grounding_position,
    write_failure_subgraph_fork,
)
from openviking.session.trajectory_index_hook import (
    _page_ao_records,
    _stable_snapshot_watermark,
    _view_ao_record,
)

SCHEMA = "r8a-v1"
FROZEN_AT = "2026-09-30T00:00:00.000Z"
FIXED_DT = datetime(2026, 9, 30, tzinfo=timezone.utc)
WORKSPACE_ID = "sb-skillsbench"
WORKSPACE_LABELS = (WORKSPACE_ID,)
ACL_NEGATIVE_LABELS = ("other-team",)
MODEL_VERSION = "r8a-deterministic-stand-in"
GRANULARITY_REASON = (
    "one transcript record = one block; "
    "session/model_change/thinking_level_change records are not blocks"
)
OMITTED_REASON = "record_type_not_a_block"
NON_BLOCK_TYPES = frozenset(("session", "model_change", "thinking_level_change"))
TEXT_CLIP = 400
EFFECT_CLIP = 160
EVENT_TEXT_CLIP = 2000
QUERY_CLIP = 4000
PROMPT_IN_QUERY = 800
COMMAND_IN_QUERY = 180
TEST_PROMPT_CLIP = 1500
K = CAUSAL_MEMORY_CAP
L3_PROFILE = {
    "fork_branch": 4,
    "projection": 8,
    "segment_atom": 0,
    "provisional_slots": 4,
    "rank_mode": "bm25",
}
RETRIEVAL_SEED = "r8a"

SB_DATA = Path("/home/zhoujie22/river2_0/evol_bench/SkillsBench/data/sb-full.jsonl")
SB_EPISODES = Path(
    "/home/zhoujie22/river2_0/evol_bench/SkillsBench/results/"
    "sb-train-wsb-dsv4flash-r1/work-warm-skill-batch/episodes"
)
SB_ATTEMPTS = Path(
    "/home/zhoujie22/river2_0/evol_bench/SkillsBench/results/"
    "sb-train-wsb-dsv4flash-r1/attempts.jsonl"
)

# Census checked against attempts.jsonl and transcript tails before this run.
# A mismatch aborts. It is not relabeled.
EXPECTED: dict[str, tuple[str, str, bool]] = {
    "0001": ("recorded failed", "tool_cut", False),
    "0002": ("recorded success", "text_end", False),
    "0003": ("recorded failed", "tool_cut", False),
    "0004": ("recorded success", "text_end", False),
    "0005": ("recorded success", "text_end", False),
    "0006": ("recorded success", "text_end", False),
    "0007": ("recorded success", "text_end", False),
    "0008": ("unrecorded", "tool_cut", False),
    "0009": ("unrecorded", "text_end", False),
    "0010": ("unrecorded", "tool_cut", False),
    "0011": ("unrecorded", "text_end", False),
    "0012": ("unrecorded", "tool_cut", False),
    "0013": ("unrecorded", "text_end", False),
    "0014": ("unrecorded", "tool_cut", True),
}
NARROW_CODES = frozenset({"0001", "0003", "0008", "0010", "0012", "0014"})
WIDE_EXTRA_CODES = frozenset({"0009", "0011", "0013"})
WIDE_CODES = NARROW_CODES | WIDE_EXTRA_CODES

HIT_DEFINITION = (
    "HIT DEFINITION (frozen before scores). "
    "A tool event is (tool_name, shell_token). shell_token is the first "
    "whitespace token of arguments.command, quotes stripped, basename if it "
    "contains '/', casefolded, then kept as [a-z0-9._+-] or 'empty'. "
    "pattern key = '{tool_name}:{shell_token}' with tool_name casefolded. "
    "Prefix = the first ceil(tool_events/2) events. Second half = the rest. "
    "Denominator = second-half event count. "
    "Query text = task prompt clipped to "
    f"{PROMPT_IN_QUERY} chars, then each prefix command as "
    f"'{{tool}} {{command}}' clipped to {COMMAND_IN_QUERY}, whole query "
    f"clipped to {QUERY_CLIP}. "
    f"k={K}. Structural arm: HybridRetrievalFacade profile={L3_PROFILE} "
    "fact_lane_paused=True seed='r8a', corpus = the other 13 episodes' "
    "frozen projections (no content_resolver) plus that caliber's failure "
    "forks. A card hits an event when the pattern key is on that card: "
    "block cards carry the patterns of that one record; claim cards carry "
    "the source block's tool patterns when relation_type=influence, else "
    "none; fork and branch cards carry every tool pattern inside the "
    "failure subgraph. "
    "Lexical baseline: BM25Index over other episodes' raw tool events "
    f"(command + following toolResult text, each clipped to {EVENT_TEXT_CLIP}). "
    "One document per tool event. Same query, same k. Documents with BM25 "
    "score 0 are dropped, matching the production ranker. Ties keep "
    "(episode_dir, event_index) order. A lexical document hits when its "
    "own pattern key equals the event's pattern key. "
    "(a) an event is a hit if any top-k item matches its pattern key. "
    "(b) cross-family: the matching item's family_id differs from the "
    "query episode's family_id. Same denominator as (a). "
    "(b-nosurf) denominator = second-half events whose shell_token is "
    "absent from bm25.tokenize(query); numerator = those events that are "
    "cross-family hits. This is the no-surface-overlap slice. "
    "(c) 45 test prompts are an interface demo, not a generalization "
    "metric: no second half exists, so no hit rate is computed. "
    "Task-success generalization is unmeasured."
)

SUBGRAPH_RULE = (
    f"SUBGRAPH RULE. At most {MAX_RETRIEVAL_SUBGRAPH_BLOCKS} blocks. "
    "If the episode has at most 12 blocks, take all of them. Otherwise "
    "take the task block plus the last 11 blocks. claim_ids are the view "
    "claims whose two endpoints are both in that set. failure_marked=True "
    "and failure_evidence_refs is the single AO of the chronologically "
    "last block in the set (the truncation neighborhood). position is "
    "default_failure_grounding_position: block:<that block_id>."
)

INFLUENCE_RULE = (
    "INFLUENCE RULE. Claims join adjacent blocks only. "
    "If an omitted AO sits strictly between them, relation_type=skip, "
    "carried_artifact='no-effect', downstream_effect='no observable change', "
    "and both ends cite evidence (production omitted_range_skip_required / "
    "skip_missing_evidence). "
    "Else if the previous block is a tool call and the next is that call's "
    "toolResult: relation_type=influence, carried_artifact=the first tool "
    "name in the call record (run_command / bash / ...), downstream_effect="
    "exit_code=N if the result text contains an exit code, else "
    f"'output:' plus the collapsed result clipped to {EFFECT_CLIP}. "
    "Else if the previous block is a toolResult and the next is a tool "
    "call: relation_type=influence, carried_artifact=the next block's first "
    "tool name, downstream_effect='action:' plus the next command clipped "
    f"to {EFFECT_CLIP}. "
    "Else relation_type=skip with the no-effect pair and no evidence refs. "
    "Every influence edge cites source and target AO evidence taken from "
    "the paged ledger (_view_ao_record content hash). A ref that does not "
    "resolve is a script failure before freeze."
)


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def _clock() -> datetime:
    return FIXED_DT


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit]


def _shell_token(command: str) -> str:
    text = command.strip()
    if not text:
        return "empty"
    token = text.split()[0].strip("'\"")
    if "/" in token:
        token = token.rsplit("/", 1)[-1]
    token = token.casefold()
    cleaned = re.sub(r"[^a-z0-9._+-]", "", token)
    return cleaned or "empty"


def _pattern(tool_name: str, command: str) -> str:
    tool = re.sub(r"[^a-z0-9._+-]", "", tool_name.casefold()) or "unknown_tool"
    return f"{tool}:{_shell_token(command)}"


def _effect(result_text: str) -> str:
    flat = " ".join(result_text.split())
    match = re.search(r"exit[_ ]?code\D{0,12}(-?\d+)", flat, re.IGNORECASE)
    if match:
        return f"exit_code={match.group(1)}"
    if not flat:
        return "output:empty"
    return "output:" + flat[:EFFECT_CLIP]


def _files(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def _rate(hits: int, denominator: int) -> dict[str, Any]:
    if denominator == 0:
        return {"hits": hits, "denominator": 0, "rate": None}
    return {"hits": hits, "denominator": denominator, "rate": f"{hits / denominator:.4f}"}


def _print_policy() -> None:
    print(HIT_DEFINITION)
    print(SUBGRAPH_RULE)
    print(INFLUENCE_RULE)
    print(
        "NOT MEASURED: task-success generalization (no grader verdict on these "
        "14 episodes, and this wave does not touch the paused live stack or "
        "call a model). Model-driven extraction is ledger #125 and is not "
        "this stand-in. The other 28 train tasks and all 45 test tasks have "
        "no recorded trajectory here."
    )
    print(
        f"write path: freeze_extracted_view purpose={PURPOSE_POST_RUN_INDEX} "
        "construction_mode=adaptive. "
        f"granularity_reason={GRANULARITY_REASON}. "
        f"frozen_at={FROZEN_AT}. revision_id=r8a:{{episode_dir}}."
    )


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not isinstance(payload, dict):
                bad += 1
                continue
            rows.append(payload)
    return rows, bad


def _content_parts(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return []
    return [part for part in content if isinstance(part, dict)]


def _tool_calls(parts: list[dict[str, Any]]) -> tuple[list[tuple[str, str]], str]:
    calls: list[tuple[str, str]] = []
    before: list[str] = []
    seen_call = False
    for part in parts:
        kind = part.get("type")
        if kind == "text" and not seen_call:
            text = part.get("text")
            before.append(text if isinstance(text, str) else "")
            continue
        if kind != "toolCall":
            continue
        seen_call = True
        name = part.get("name")
        tool_name = name.strip() if isinstance(name, str) and name.strip() else "unknown_tool"
        arguments = part.get("arguments")
        command = ""
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and isinstance(parsed.get("command"), str):
                command = parsed["command"]
            else:
                command = arguments
        elif isinstance(arguments, dict):
            raw_command = arguments.get("command")
            if isinstance(raw_command, str):
                command = raw_command
        calls.append((tool_name, command))
    return calls, "\n".join(before)


def _text_of(parts: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for part in parts:
        if part.get("type") != "text":
            continue
        text = part.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return "\n".join(chunks)


def _load_catalog() -> dict[str, dict[str, Any]]:
    if not SB_DATA.is_file():
        _fail(f"missing catalog {SB_DATA}")
    by_task: dict[str, dict[str, Any]] = {}
    with SB_DATA.open(encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                _fail("sb-full row missing task_id")
            if task_id in by_task:
                _fail(f"duplicate task_id {task_id}")
            by_task[task_id] = row
    return by_task


def _load_attempts() -> dict[str, dict[str, Any]]:
    if not SB_ATTEMPTS.is_file():
        _fail(f"missing attempts {SB_ATTEMPTS}")
    rows, bad = _read_jsonl(SB_ATTEMPTS)
    if bad:
        _fail(f"attempts.jsonl has {bad} unreadable lines")
    if len(rows) != 7:
        _fail(f"attempts.jsonl has {len(rows)} rows, expected 7")
    by_task: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id in by_task:
            _fail(f"attempts task_id invalid or duplicated: {task_id!r}")
        by_task[task_id] = row
    return by_task


def _episode_dirs() -> list[Path]:
    if not SB_EPISODES.is_dir():
        _fail(f"missing episodes dir {SB_EPISODES}")
    dirs = sorted(path for path in SB_EPISODES.iterdir() if path.is_dir())
    codes = [path.name[:4] for path in dirs]
    if codes != [f"{index:04d}" for index in range(1, 15)]:
        _fail(f"episode dirs are { [path.name for path in dirs] }, expected 0001..0014")
    return dirs


def _transcript_path(episode_dir: Path) -> Path:
    files = sorted(episode_dir.glob("2*.jsonl"))
    if len(files) != 1:
        _fail(f"{episode_dir.name}: expected one 2*.jsonl, found {len(files)}")
    return files[0]


def _confirm_status(task_id: str, attempts: dict[str, dict[str, Any]]) -> tuple[str, str]:
    row = attempts.get(task_id)
    if row is None:
        return "unrecorded", ""
    status = row.get("status")
    error = row.get("error")
    error_text = error if isinstance(error, str) else ""
    if status == "success":
        return "recorded success", error_text
    if status == "failed":
        return "recorded failed", error_text
    _fail(f"{task_id}: attempts status {status!r} is neither success nor failed")
    return "", ""


def _parse_transcript(episode_dir: Path) -> dict[str, Any]:
    rows, bad = _read_jsonl(_transcript_path(episode_dir))
    task_meta = json.loads((episode_dir / "sb-task.json").read_text(encoding="utf-8"))
    task_id = task_meta.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        _fail(f"{episode_dir.name}: sb-task.json missing task_id")
    items: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []
    pending_results: list[str] = []
    first_user_seen = False
    pure_assistant_indexes: list[int] = []
    for row in rows:
        record_type = row.get("type")
        if not isinstance(record_type, str):
            _fail(f"{episode_dir.name}: record missing type")
        if record_type in NON_BLOCK_TYPES:
            items.append(
                {
                    "record_type": record_type,
                    "kind": "omitted",
                    "role_name": "",
                    "text": "",
                    "calls": [],
                    "input_text": "",
                }
            )
            continue
        if record_type != "message":
            _fail(f"{episode_dir.name}: unknown record type {record_type!r}")
        message = row.get("message")
        if not isinstance(message, dict):
            _fail(f"{episode_dir.name}: message record has no message object")
        role = message.get("role")
        role_name = role if isinstance(role, str) else ""
        parts = _content_parts(message)
        calls, input_text = _tool_calls(parts)
        text = _text_of(parts)
        kind = "orchestrator"
        if role_name == "user" and not first_user_seen:
            kind = "task"
            first_user_seen = True
            text = text or input_text
        elif role_name == "toolResult":
            kind = "tool_result"
        elif calls:
            kind = "tool_call"
        elif role_name == "assistant" and text and not calls:
            kind = "assistant_text"
            pure_assistant_indexes.append(len(items))
        items.append(
            {
                "record_type": record_type,
                "kind": kind,
                "role_name": role_name,
                "text": text,
                "calls": calls,
                "input_text": input_text,
            }
        )
        if calls:
            for tool_name, command in calls:
                event_index = len(tool_events)
                tool_events.append(
                    {
                        "index": event_index,
                        "tool": tool_name,
                        "command": command,
                        "token": _shell_token(command),
                        "pattern": _pattern(tool_name, command),
                        "result": "",
                        "item_index": len(items) - 1,
                    }
                )
                pending_results.append(str(event_index))
        elif kind == "tool_result" and pending_results:
            event = tool_events[int(pending_results.pop(0))]
            event["result"] = text
    if not first_user_seen:
        _fail(f"{episode_dir.name}: no user task message")
    last_message = next((item for item in reversed(items) if item["kind"] != "omitted"), None)
    if last_message is None:
        _fail(f"{episode_dir.name}: no message records")
    if last_message["kind"] in {"tool_call", "tool_result"}:
        tail_class = "tool_cut"
        if last_message["kind"] == "tool_call":
            tail_shape = "cut:toolCall"
        else:
            tail_shape = "cut:toolResult"
    elif last_message["kind"] == "assistant_text" or (
        last_message["role_name"] == "assistant" and last_message["kind"] == "orchestrator"
    ):
        tail_class = "text_end"
        tail_shape = "text_end"
    else:
        tail_class = "other"
        tail_shape = f"other:{last_message['role_name']}:{last_message['kind']}"
    conclusion_index = None
    if tail_class == "text_end" and pure_assistant_indexes:
        conclusion_index = pure_assistant_indexes[-1]
        items[conclusion_index]["kind"] = "conclusion"
        for index in pure_assistant_indexes[:-1]:
            items[index]["kind"] = "orchestrator"
    else:
        for index in pure_assistant_indexes:
            items[index]["kind"] = "orchestrator"
    return {
        "task_id": task_id.strip(),
        "items": items,
        "tool_events": tool_events,
        "tail_class": tail_class,
        "tail_shape": tail_shape,
        "bad_json_lines": bad,
        "conclusion": conclusion_index is not None,
    }


def _append_ledger(session_dir: Path, session_id: str, parsed: dict[str, Any], episode_index: int) -> list[dict[str, Any]]:
    if session_dir.exists():
        for child in session_dir.iterdir():
            if child.is_file():
                child.unlink()
    session_dir.mkdir(parents=True, exist_ok=True)
    state = {"n": episode_index * 100_000}
    original = ao_ledger_mod.uuid.uuid4

    def _next() -> uuid.UUID:
        state["n"] += 1
        return uuid.UUID(int=state["n"])

    ao_ledger_mod.uuid.uuid4 = _next
    try:
        ledger = AOLedger(session_dir, session_id, clock=_clock)
        for index, item in enumerate(parsed["items"]):
            if item["kind"] == "task":
                participant_id = "user"
                segment_id = f"{session_id}:task"
            elif item["kind"] == "omitted":
                participant_id = "agent"
                segment_id = f"{session_id}:preamble"
            else:
                participant_id = "agent"
                segment_id = f"{session_id}:agent"
            item["participant_id"] = participant_id
            item["segment_id"] = segment_id
            synopsis = _clip(item["text"] or item["input_text"] or item["record_type"], TEXT_CLIP)
            ledger.append(
                action={
                    "record_kind": "atom",
                    "acl_labels": [WORKSPACE_ID],
                    "segment_id": segment_id,
                    "participant_id": participant_id,
                    "transcript_type": item["record_type"],
                    "block_kind": item["kind"],
                },
                observation={"synopsis": synopsis or item["kind"]},
                skill_invocations=[],
                message_ref={"message_id": f"{session_id}:{index}"},
            )
    finally:
        ao_ledger_mod.uuid.uuid4 = original
    paged = [_view_ao_record(row, session_id) for row in _page_ao_records(session_dir, session_id)]
    paged.sort(key=lambda row: (row["sequence"], row["ao_id"]))
    if len(paged) != len(parsed["items"]):
        _fail(
            f"{session_id}: paged {len(paged)} AO rows, transcript items {len(parsed['items'])}"
        )
    for item, row in zip(parsed["items"], paged, strict=True):
        if row["participant_id"] != item["participant_id"] or row["segment_id"] != item["segment_id"]:
            _fail(f"{session_id}: ledger did not keep segment/participant at seq {row['sequence']}")
        if not row["ao_id"] or not row["content_hash"]:
            _fail(f"{session_id}: AO seq {row['sequence']} missing id or content hash")
        item["ao"] = row
    return paged


def _source_ref(row: dict[str, Any]) -> SourceRef:
    return SourceRef(
        ao_id=row["ao_id"],
        sequence=row["sequence"],
        participant_id=row["participant_id"],
        segment_id=row["segment_id"],
        session_id=row["session_id"],
        content_hash=row["content_hash"],
    )


def _evidence(row: dict[str, Any], watermark: str) -> InfluenceEvidenceRef:
    return InfluenceEvidenceRef(
        ao_id=row["ao_id"],
        source_session_id=row["session_id"],
        source_archive_id=None,
        archive_commit_watermark=None,
        source_sequence=row["sequence"],
        source_read_snapshot_watermark=watermark,
        evidence_role="contemporaneous_basis",
        evidence_content_hash=row["content_hash"],
        captured_state="committed",
    )


def _ledger_index(paged: list[dict[str, Any]]) -> dict[tuple[str, int], str]:
    return {(row["ao_id"], row["sequence"]): row["content_hash"] for row in paged}


def _evidence_resolves(ref: InfluenceEvidenceRef, index: dict[tuple[str, int], str]) -> bool:
    return index.get((ref.ao_id, ref.source_sequence)) == ref.evidence_content_hash


def _build_blocks(
    session_id: str, parsed: dict[str, Any]
) -> tuple[list[InfluenceBlock], list[dict[str, Any]]]:
    blocks: list[InfluenceBlock] = []
    metas: list[dict[str, Any]] = []
    for item in parsed["items"]:
        if item["kind"] == "omitted":
            continue
        row = item["ao"]
        ref = _source_ref(row)
        calls: list[tuple[str, str]] = item["calls"]
        patterns = [_pattern(tool_name, command) for tool_name, command in calls]
        if item["kind"] == "task":
            role = ROLE_TASK
            block_input = ""
            block_output = _clip(item["text"], TEXT_CLIP)
            summary = _clip(item["text"], TEXT_CLIP) or "task"
        elif item["kind"] == "tool_call":
            role = "agent_action"
            commands = " ; ".join(command for _name, command in calls)
            block_input = _clip(item["input_text"], TEXT_CLIP)
            block_output = _clip(commands, TEXT_CLIP)
            summary = (" ".join(f"pattern:{pattern}" for pattern in patterns) + " " + block_output).strip()
            summary = summary or "tool_call"
        elif item["kind"] == "tool_result":
            role = "agent_action"
            block_input = ""
            block_output = _clip(item["text"], TEXT_CLIP)
            summary = ("tool_result " + block_output).strip() or "tool_result"
        elif item["kind"] == "conclusion":
            role = ROLE_CONCLUSION
            block_input = ""
            block_output = _clip(item["text"], TEXT_CLIP)
            summary = block_output or "conclusion"
        else:
            role = "orchestrator"
            block_input = ""
            block_output = _clip(item["text"], TEXT_CLIP)
            summary = block_output or "orchestrator"
        block_id = f"b-{row['sequence']:06d}"
        block = InfluenceBlock(
            block_id=block_id,
            role=role,
            participant_id=row["participant_id"],
            segment_ref=SegmentRef(segment_id=row["segment_id"], session_id=session_id),
            ao_start_seq=row["sequence"],
            ao_end_seq=row["sequence"],
            summary=summary,
            input=block_input,
            output=block_output,
            authorship=row["participant_id"],
            granularity_reason=GRANULARITY_REASON,
            source_content_hashes=(row["content_hash"],),
            source_refs=(ref,),
        )
        blocks.append(block)
        metas.append(
            {
                "block_id": block_id,
                "kind": item["kind"],
                "patterns": patterns,
                "tool_name": calls[0][0] if calls else "",
                "command": calls[0][1] if calls else "",
                "result_text": item["text"] if item["kind"] == "tool_result" else "",
                "sequence": row["sequence"],
                "ao": row,
            }
        )
    if not blocks:
        _fail(f"{session_id}: no blocks")
    return blocks, metas


def _omitted_between(left_seq: int, right_seq: int, omitted_seqs: set[int]) -> bool:
    return any(seq in omitted_seqs for seq in range(left_seq + 1, right_seq))


def _build_claims(
    metas: list[dict[str, Any]],
    omitted_seqs: set[int],
    watermark: str,
    ledger: dict[tuple[str, int], str],
) -> tuple[list[InfluenceClaim], dict[str, list[str]], int]:
    claims: list[InfluenceClaim] = []
    claim_patterns: dict[str, list[str]] = {}
    crossed = 0
    for left, right in zip(metas, metas[1:], strict=False):
        crosses = _omitted_between(left["sequence"], right["sequence"], omitted_seqs)
        if crosses:
            crossed += 1
        need_evidence = False
        patterns: list[str] = []
        if crosses:
            relation = RELATION_SKIP
            artifact = "no-effect"
            effect = "no observable change"
            need_evidence = True
        elif left["kind"] == "tool_call" and right["kind"] == "tool_result":
            relation = RELATION_INFLUENCE
            artifact = left["tool_name"] or "unknown_tool"
            effect = _effect(right["result_text"])
            need_evidence = True
            patterns = list(left["patterns"])
        elif left["kind"] == "tool_result" and right["kind"] == "tool_call":
            relation = RELATION_INFLUENCE
            artifact = right["tool_name"] or "unknown_tool"
            effect = "action:" + _clip(right["command"], EFFECT_CLIP)
            if effect == "action:":
                effect = "action:empty"
            need_evidence = True
            patterns = list(right["patterns"])
        else:
            relation = RELATION_SKIP
            artifact = "no-effect"
            effect = "no observable change"
        source_refs: tuple[InfluenceEvidenceRef, ...] = ()
        target_refs: tuple[InfluenceEvidenceRef, ...] = ()
        if need_evidence:
            source_refs = (_evidence(left["ao"], watermark),)
            target_refs = (_evidence(right["ao"], watermark),)
            for ref in (*source_refs, *target_refs):
                if not _evidence_resolves(ref, ledger):
                    _fail(
                        "evidence does not resolve "
                        f"ao_id={ref.ao_id} sequence={ref.source_sequence}"
                    )
        claim_id = f"c-{left['sequence']:06d}-{right['sequence']:06d}"
        claims.append(
            InfluenceClaim(
                claim_id=claim_id,
                source_block_id=left["block_id"],
                target_block_id=right["block_id"],
                carried_artifact=artifact,
                downstream_effect=effect,
                relation_type=relation,
                source_evidence_refs=source_refs,
                target_evidence_refs=target_refs,
            )
        )
        claim_patterns[claim_id] = patterns
    return claims, claim_patterns, crossed


def _coverage(
    session_id: str,
    paged: list[dict[str, Any]],
    blocks: list[InfluenceBlock],
    metas_by_seq: dict[int, dict[str, Any]],
) -> CoverageManifest:
    del metas_by_seq
    inspected = tuple(_source_ref(row) for row in paged)
    included_seqs = {block.ao_start_seq for block in blocks}
    omitted_rows = [row for row in paged if row["sequence"] not in included_seqs]
    ranges: list[OmittedRange] = []
    if omitted_rows:
        start = omitted_rows[0]["sequence"]
        prev = start
        segment_id = omitted_rows[0]["segment_id"]
        for row in omitted_rows[1:]:
            if row["sequence"] == prev + 1 and row["segment_id"] == segment_id:
                prev = row["sequence"]
                continue
            ranges.append(
                OmittedRange(
                    start_seq=start,
                    end_seq=prev,
                    reason=OMITTED_REASON,
                    segment_id=segment_id,
                )
            )
            start = row["sequence"]
            prev = start
            segment_id = row["segment_id"]
        ranges.append(
            OmittedRange(
                start_seq=start,
                end_seq=prev,
                reason=OMITTED_REASON,
                segment_id=segment_id,
            )
        )
    included = tuple(
        AORange(
            start_seq=block.ao_start_seq,
            end_seq=block.ao_end_seq,
            segment_id=block.segment_ref.segment_id,
        )
        for block in blocks
    )
    watermark = _stable_snapshot_watermark(session_id, paged)
    return CoverageManifest(
        total_ao_range=AORange(start_seq=paged[0]["sequence"], end_seq=paged[-1]["sequence"]),
        inspected_refs=inspected,
        included_ranges=included,
        omitted_ranges=tuple(ranges),
        snapshot_watermark=watermark,
    )


def _select_subgraph_blocks(blocks: list[InfluenceBlock]) -> list[InfluenceBlock]:
    if len(blocks) <= MAX_RETRIEVAL_SUBGRAPH_BLOCKS:
        chosen = list(blocks)
    else:
        task = [block for block in blocks if block.role == ROLE_TASK]
        tail = list(blocks[-11:])
        if task and task[0].block_id not in {block.block_id for block in tail}:
            chosen = [task[0], *tail]
        else:
            chosen = list(blocks[-MAX_RETRIEVAL_SUBGRAPH_BLOCKS:])
    chosen.sort(key=lambda block: (block.ao_start_seq, block.block_id))
    if len(chosen) > MAX_RETRIEVAL_SUBGRAPH_BLOCKS:
        _fail(f"subgraph selection produced {len(chosen)} blocks")
    return chosen


def _make_subgraph(
    blocks: list[InfluenceBlock],
    claims: list[InfluenceClaim],
    metas: list[dict[str, Any]],
    watermark: str,
) -> RetrievalSubgraph:
    chosen = _select_subgraph_blocks(blocks)
    chosen_ids = {block.block_id for block in chosen}
    claim_ids = tuple(
        claim.claim_id
        for claim in claims
        if claim.source_block_id in chosen_ids and claim.target_block_id in chosen_ids
    )
    last = max(chosen, key=lambda block: (block.ao_end_seq, block.block_id))
    meta = next(item for item in metas if item["block_id"] == last.block_id)
    evidence = _evidence(meta["ao"], watermark)
    return RetrievalSubgraph(
        subgraph_id="sg-tail",
        block_ids=tuple(block.block_id for block in chosen),
        claim_ids=claim_ids,
        failure_marked=True,
        failure_evidence_refs=(evidence,),
    )


def _freeze(
    session_id: str,
    task_id: str,
    revision_id: str,
    blocks: list[InfluenceBlock],
    claims: list[InfluenceClaim],
    coverage: CoverageManifest,
    subgraphs: tuple[RetrievalSubgraph, ...],
) -> InfluenceViewRevision:
    try:
        revision = freeze_extracted_view(
            session_id=session_id,
            task_run_id=task_id,
            view_id=f"{session_id}:{PURPOSE_POST_RUN_INDEX}",
            revision_id=revision_id,
            frozen_at=FROZEN_AT,
            blocks=blocks,
            claims=claims,
            coverage=coverage,
            retrieval_subgraphs=subgraphs,
        )
    except HighSeverityViolation as exc:
        print(f"FREEZE REFUSED {session_id} {revision_id}")
        for violation in exc.violations:
            print(json.dumps(violation.to_dict(), ensure_ascii=False, sort_keys=True))
        raise SystemExit(1) from exc
    if revision.construction_mode != "adaptive":
        _fail(f"{session_id}: construction_mode={revision.construction_mode!r}")
    if revision.purpose != PURPOSE_POST_RUN_INDEX:
        _fail(f"{session_id}: purpose={revision.purpose!r}")
    return revision


def _check_structure(
    session_id: str,
    paged: list[dict[str, Any]],
    blocks: list[InfluenceBlock],
    claims: list[InfluenceClaim],
    coverage: CoverageManifest,
    revision: InfluenceViewRevision,
    ledger: dict[tuple[str, int], str],
) -> None:
    ledger_ids = {(row["ao_id"], row["sequence"], row["content_hash"]) for row in paged}
    included: set[int] = set()
    for block in blocks:
        if block.ao_start_seq > block.ao_end_seq:
            _fail(f"{session_id}: {block.block_id} empty span")
        participants = {ref.participant_id for ref in block.source_refs}
        segments = {ref.segment_id for ref in block.source_refs}
        if participants != {block.participant_id} or segments != {block.segment_ref.segment_id}:
            _fail(f"{session_id}: {block.block_id} mixes participant or segment")
        for ref in block.source_refs:
            key = (ref.ao_id, ref.sequence, ref.content_hash)
            if key not in ledger_ids:
                _fail(f"{session_id}: {block.block_id} source ref {ref.ao_id} not in ledger")
            included.add(ref.sequence)
    omitted: set[int] = set()
    for item in coverage.omitted_ranges:
        if not item.reason.strip():
            _fail(f"{session_id}: omitted range missing reason")
        omitted.update(range(item.start_seq, item.end_seq + 1))
    inspected = {ref.sequence for ref in coverage.inspected_refs}
    all_seqs = {row["sequence"] for row in paged}
    if inspected != all_seqs:
        _fail(f"{session_id}: inspected sequences != ledger")
    if included | omitted != all_seqs or included & omitted:
        _fail(f"{session_id}: included/omitted do not partition the ledger")
    for claim in claims:
        if claim.relation_type == RELATION_INFLUENCE:
            if not claim.source_evidence_refs or not claim.target_evidence_refs:
                _fail(f"{session_id}: influence {claim.claim_id} missing evidence")
        for ref in (*claim.source_evidence_refs, *claim.target_evidence_refs):
            if not _evidence_resolves(ref, ledger):
                _fail(f"{session_id}: claim {claim.claim_id} evidence does not resolve")
    block_ids = {block.block_id for block in revision.blocks}
    claim_ids = {claim.claim_id for claim in revision.claims}
    for subgraph in revision.retrieval_subgraphs:
        if len(subgraph.block_ids) > MAX_RETRIEVAL_SUBGRAPH_BLOCKS:
            _fail(f"{session_id}: subgraph exceeds block limit")
        if subgraph.failure_marked != bool(subgraph.failure_evidence_refs):
            _fail(f"{session_id}: failure marker and evidence diverged")
        members = set(subgraph.block_ids)
        if not members <= block_ids:
            _fail(f"{session_id}: subgraph names an unknown block")
        for claim_id in subgraph.claim_ids:
            claim = next((item for item in revision.claims if item.claim_id == claim_id), None)
            if claim is None or claim.claim_id not in claim_ids:
                _fail(f"{session_id}: subgraph names an unknown claim")
            if claim.source_block_id not in members or claim.target_block_id not in members:
                _fail(f"{session_id}: claim {claim_id} endpoint outside subgraph")
        for ref in subgraph.failure_evidence_refs:
            if not _evidence_resolves(ref, ledger):
                _fail(f"{session_id}: failure evidence does not resolve")


def _query_text(prompt: str, events: list[dict[str, Any]]) -> str:
    half = 0 if not events else math.ceil(len(events) / 2)
    lines = [_clip(prompt, PROMPT_IN_QUERY)]
    for event in events[:half]:
        lines.append(f"{event['tool']} {_clip(event['command'], COMMAND_IN_QUERY)}")
    return "\n".join(lines)[:QUERY_CLIP]


def _seven_sections(record: dict[str, Any]) -> dict[str, str]:
    patterns = " ".join(f"pattern:{item}" for item in record["tail_patterns"]) or "no-tool-pattern"
    error = record["attempt_error"].strip() or "no recorded error"
    truncation = (
        f"truncated_json_lines={record['bad_json_lines']}"
        if record["bad_json_lines"]
        else "no truncated json line"
    )
    sections = {
        "session_title": f"{record['episode_dir']} {record['task_id']}",
        "current_state": f"{record['confirm_status']} tail={record['tail_shape']}",
        "task_and_goals": _clip(record["task_prompt"], TEXT_CLIP) or "task",
        "key_facts_and_decisions": patterns,
        "files_and_context": (
            f"tool_calls={record['tool_calls']} blocks={record['block_n']} "
            f"family={record['family']}"
        ),
        "errors_and_corrections": error,
        "open_issues": truncation,
    }
    missing = [key for key in SEVEN_SECTION_KEYS if not sections.get(key, "").strip()]
    if missing:
        _fail(f"{record['episode_dir']}: empty seven-section keys {missing}")
    return {key: sections[key] for key in SEVEN_SECTION_KEYS}


def _extract_episode(
    episode_dir: Path,
    episode_index: int,
    catalog: dict[str, dict[str, Any]],
    attempts: dict[str, dict[str, Any]],
    ledger_root: Path,
) -> dict[str, Any]:
    code = episode_dir.name[:4]
    parsed = _parse_transcript(episode_dir)
    task_id = parsed["task_id"]
    row = catalog.get(task_id)
    if row is None:
        _fail(f"{episode_dir.name}: task_id {task_id} not in sb-full.jsonl")
    confirm, attempt_error = _confirm_status(task_id, attempts)
    expected = EXPECTED[code]
    if (confirm, parsed["tail_class"], parsed["bad_json_lines"] > 0) != expected:
        _fail(
            f"{episode_dir.name}: census mismatch "
            f"got {(confirm, parsed['tail_class'], parsed['bad_json_lines'] > 0)} "
            f"expected {expected}"
        )
    turns = row.get("turns")
    prompt = ""
    if isinstance(turns, list) and turns and isinstance(turns[0], dict):
        raw_prompt = turns[0].get("prompt")
        if isinstance(raw_prompt, str):
            prompt = raw_prompt
    if not prompt:
        task_item = next(item for item in parsed["items"] if item["kind"] == "task")
        prompt = task_item["text"]
    family = row.get("family_id")
    if not isinstance(family, str) or not family:
        _fail(f"{episode_dir.name}: missing family_id")
    session_id = episode_dir.name
    paged = _append_ledger(ledger_root / session_id, session_id, parsed, episode_index)
    ledger = _ledger_index(paged)
    blocks, metas = _build_blocks(session_id, parsed)
    omitted_seqs = {item["ao"]["sequence"] for item in parsed["items"] if item["kind"] == "omitted"}
    coverage = _coverage(session_id, paged, blocks, {item["sequence"]: item for item in metas})
    claims, claim_patterns, crossed = _build_claims(
        metas, omitted_seqs, coverage.snapshot_watermark, ledger
    )
    block_patterns = {item["block_id"]: list(item["patterns"]) for item in metas}
    in_narrow = code in NARROW_CODES
    in_wide = code in WIDE_CODES
    subgraph = (
        _make_subgraph(blocks, claims, metas, coverage.snapshot_watermark) if in_wide else None
    )
    primary_subgraphs: tuple[RetrievalSubgraph, ...] = ()
    if in_narrow and subgraph is not None:
        primary_subgraphs = (subgraph,)
    primary = _freeze(
        session_id,
        task_id,
        f"r8a:{session_id}",
        blocks,
        claims,
        coverage,
        primary_subgraphs,
    )
    _check_structure(session_id, paged, blocks, claims, coverage, primary, ledger)
    wide_revision = None
    if code in WIDE_EXTRA_CODES and subgraph is not None:
        wide_revision = _freeze(
            session_id,
            task_id,
            f"r8a-wide:{session_id}",
            blocks,
            claims,
            coverage,
            (subgraph,),
        )
        _check_structure(session_id, paged, blocks, claims, coverage, wide_revision, ledger)
    influence_n = sum(1 for claim in claims if claim.relation_type == RELATION_INFLUENCE)
    skip_n = sum(1 for claim in claims if claim.relation_type == RELATION_SKIP)
    tail_patterns: list[str] = []
    if subgraph is not None:
        chosen = set(subgraph.block_ids)
        for meta in metas:
            if meta["block_id"] in chosen:
                tail_patterns.extend(meta["patterns"])
    record = {
        "schema": SCHEMA,
        "episode_dir": session_id,
        "episode_id": row.get("episode_id"),
        "task_id": task_id,
        "family": family,
        "confirm_status": confirm,
        "tail_shape": parsed["tail_shape"],
        "tail_class": parsed["tail_class"],
        "bad_json_lines": parsed["bad_json_lines"],
        "json_truncated": parsed["bad_json_lines"] > 0,
        "tool_calls": len(parsed["tool_events"]),
        "tool_calls_missing_command": sum(
            1 for event in parsed["tool_events"] if not event["command"].strip()
        ),
        "block_n": len(blocks),
        "claim_n": len(claims),
        "claim_influence": influence_n,
        "claim_skip": skip_n,
        "omitted_between_claims": crossed,
        "subgraph_narrow": 1 if in_narrow else 0,
        "subgraph_wide": 1 if in_wide else 0,
        "attempt_error": attempt_error,
        "task_prompt": prompt,
        "query_prefix": _query_text(prompt, parsed["tool_events"]),
        "events": [
            {
                "index": event["index"],
                "tool": event["tool"],
                "token": event["token"],
                "pattern": event["pattern"],
                "command": _clip(event["command"], EVENT_TEXT_CLIP),
                "result": _clip(event["result"], EVENT_TEXT_CLIP),
            }
            for event in parsed["tool_events"]
        ],
        "block_patterns": block_patterns,
        "claim_patterns": claim_patterns,
        "tail_patterns": tail_patterns,
        "revision": primary.to_dict(),
        "revision_wide": None if wide_revision is None else wide_revision.to_dict(),
        "in_narrow": in_narrow,
        "in_wide": in_wide,
    }
    return record


def _revision_of(payload: dict[str, Any] | None) -> InfluenceViewRevision | None:
    if payload is None:
        return None
    return InfluenceViewRevision.from_dict(payload)


def _basis_for(revision: InfluenceViewRevision, subgraph: RetrievalSubgraph) -> dict[str, Any]:
    anchor_seq = subgraph.failure_evidence_refs[0].source_sequence
    earlier = [
        block
        for block in revision.blocks
        if block.ao_end_seq < anchor_seq and block.source_refs
    ]
    if not earlier:
        _fail(f"{revision.session_id}: no AO strictly before the failure anchor")
    block = min(earlier, key=lambda item: item.ao_start_seq)
    ref = block.source_refs[0]
    return {
        "ao_id": ref.ao_id,
        "source_session_id": ref.session_id,
        "source_archive_id": None,
        "archive_commit_watermark": None,
        "source_sequence": ref.sequence,
        "source_read_snapshot_watermark": revision.coverage.snapshot_watermark,
        "evidence_role": "contemporaneous_basis",
        "evidence_content_hash": ref.content_hash,
        "captured_state": "committed",
    }


def _fork_payload(record: dict[str, Any], revision: InfluenceViewRevision) -> dict[str, Any]:
    subgraphs = [item for item in revision.retrieval_subgraphs if item.subgraph_id == "sg-tail"]
    if len(subgraphs) != 1:
        _fail(f"{record['episode_dir']}: expected one sg-tail subgraph")
    subgraph = subgraphs[0]
    position_kind, position_ref = default_failure_grounding_position(revision, subgraph.subgraph_id)
    if position_kind != "block":
        _fail(f"{record['episode_dir']}: position_kind={position_kind!r}")
    sections = _seven_sections(record)
    basis = _basis_for(revision, subgraph)
    ao_sequence = []
    by_id = {block.block_id: block for block in revision.blocks}
    for block_id in subgraph.block_ids:
        block = by_id[block_id]
        ao_sequence.append(block.source_refs[0].ao_id)
    last_pattern = record["tail_patterns"][-1] if record["tail_patterns"] else "none"
    return {
        "subgraph_id": subgraph.subgraph_id,
        "position_kind": position_kind,
        "position_ref": position_ref,
        "seven_sections": sections,
        "seven_bytes": sum(len(sections[key].encode("utf-8")) for key in SEVEN_SECTION_KEYS),
        "branches": [
            {
                "branch_id": "br-tail",
                "evidence_status": "real",
                "ao_sequence": ao_sequence,
                "guidance": {"instruction": f"inspect tail pattern {last_pattern}"},
            }
        ],
        "contemporaneous_basis": [basis],
        "block_n": len(subgraph.block_ids),
    }


def _write_fork(
    store: CausalExperiencesStore,
    revision: InfluenceViewRevision,
    record: dict[str, Any],
    payload: dict[str, Any],
) -> Any:
    return write_failure_subgraph_fork(
        store,
        revision,
        payload["subgraph_id"],
        seven_sections=payload["seven_sections"],
        branches=payload["branches"],
        position_kind=payload["position_kind"],
        position_ref=payload["position_ref"],
        workspace_id=WORKSPACE_ID,
        workspace_labels=WORKSPACE_LABELS,
        diagnosis_run_id=f"r8a:{record['episode_dir']}",
        model_version=MODEL_VERSION,
        contemporaneous_basis=payload["contemporaneous_basis"],
    )


def _new_store(root: Path) -> CausalExperiencesStore:
    return CausalExperiencesStore(
        root,
        config={"skill_trajectory_mode": "causal"},
        clock=_clock,
    )


def _run_forks(records: list[dict[str, Any]], work: Path, caliber: str) -> list[dict[str, Any]]:
    wanted = [record for record in records if record["in_narrow" if caliber == "narrow" else "in_wide"]]
    store = _new_store(work / f"forks-{caliber}")
    rows: list[dict[str, Any]] = []
    for record in wanted:
        revision_payload = record["revision"]
        if caliber == "wide" and record["revision_wide"] is not None:
            revision_payload = record["revision_wide"]
        revision = _revision_of(revision_payload)
        if revision is None:
            _fail(f"{record['episode_dir']}: missing revision for {caliber}")
        payload = _fork_payload(record, revision)
        probe = work / "acl-probe" / caliber / record["episode_dir"]
        if probe.exists():
            for path in sorted(probe.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
        probe.mkdir(parents=True, exist_ok=True)
        negative_store = _new_store(probe)
        before_negative = _files(probe)
        try:
            _write_fork_with_labels(
                negative_store,
                revision,
                record,
                payload,
                ACL_NEGATIVE_LABELS,
            )
        except ValueError as exc:
            negative_error = str(exc)
        else:
            _fail(f"{record['episode_dir']}: ACL negative write was accepted")
        after_negative = _files(probe)
        if before_negative != after_negative:
            _fail(
                f"{record['episode_dir']}: ACL negative added files "
                f"{sorted(set(after_negative) - set(before_negative))}"
            )
        if "workspace_labels" not in negative_error:
            _fail(f"{record['episode_dir']}: ACL error did not name workspace_labels: {negative_error}")
        first = _write_fork(store, revision, record, payload)
        readback = _readback(store, first, revision, payload)
        before_second = _files(work / f"forks-{caliber}")
        second = _write_fork(store, revision, record, payload)
        after_second = _files(work / f"forks-{caliber}")
        if first.fork_node_id != second.fork_node_id or first.revision_id != second.revision_id:
            _fail(f"{record['episode_dir']}: fork rewrite changed identity")
        if before_second != after_second:
            _fail(f"{record['episode_dir']}: fork rewrite added files")
        rows.append(
            {
                "episode_dir": record["episode_dir"],
                "caliber": caliber,
                "subgraph_id": payload["subgraph_id"],
                "block_n": payload["block_n"],
                "position_ref": payload["position_ref"],
                "seven_bytes": payload["seven_bytes"],
                "idempotent": True,
                "acl_negative_rejected": True,
                "acl_negative_files_added": 0,
                "readback": True,
                "fork_node_id": first.fork_node_id,
                "fork_revision_id": first.revision_id,
                "view_revision_id": revision.revision_id,
                "content_hash": revision.content_hash,
                "readback_detail": readback,
            }
        )
        print(
            f"FORK {caliber} {record['episode_dir']} blocks={payload['block_n']} "
            f"position={payload['position_ref']} seven_bytes={payload['seven_bytes']} "
            "idempotent=yes acl_negative=rejected"
        )
    return rows


def _write_fork_with_labels(
    store: CausalExperiencesStore,
    revision: InfluenceViewRevision,
    record: dict[str, Any],
    payload: dict[str, Any],
    labels: tuple[str, ...],
) -> Any:
    return write_failure_subgraph_fork(
        store,
        revision,
        payload["subgraph_id"],
        seven_sections=payload["seven_sections"],
        branches=payload["branches"],
        position_kind=payload["position_kind"],
        position_ref=payload["position_ref"],
        workspace_id=WORKSPACE_ID,
        workspace_labels=labels,
        diagnosis_run_id=f"r8a:{record['episode_dir']}",
        model_version=MODEL_VERSION,
        contemporaneous_basis=payload["contemporaneous_basis"],
    )


def _readback(
    store: CausalExperiencesStore,
    written: Any,
    revision: InfluenceViewRevision,
    payload: dict[str, Any],
) -> dict[str, Any]:
    loaded = store.get_fork_node(written.fork_node_id)
    if loaded is None:
        _fail(f"fork {written.fork_node_id} missing on readback")
    latest = loaded["latest_revision"]
    sections = latest.get("seven_sections")
    if sections != payload["seven_sections"]:
        _fail(f"{revision.session_id}: seven sections readback mismatch")
    grounding = latest.get("influence_grounding")
    if not isinstance(grounding, dict):
        _fail(f"{revision.session_id}: readback missing influence_grounding")
    if (
        grounding.get("view_revision_id") != revision.revision_id
        or grounding.get("position_kind") != payload["position_kind"]
        or grounding.get("position_ref") != payload["position_ref"]
        or grounding.get("content_hash") != revision.content_hash
    ):
        _fail(f"{revision.session_id}: grounding readback mismatch {grounding}")
    return {
        "view_revision_id": grounding["view_revision_id"],
        "position_kind": grounding["position_kind"],
        "position_ref": grounding["position_ref"],
        "content_hash": grounding["content_hash"],
    }


def _control_gate(record: dict[str, Any], ledger_root: Path) -> dict[str, Any]:
    revision = _revision_of(record["revision"])
    if revision is None or len(revision.blocks) < 2:
        _fail("control gate needs a frozen episode with two blocks")
    session_id = revision.session_id
    paged = [_view_ao_record(row, session_id) for row in _page_ao_records(ledger_root / session_id, session_id)]
    ledger = _ledger_index(paged)
    fake = InfluenceEvidenceRef(
        ao_id="forged-ao-not-in-ledger",
        source_session_id=session_id,
        source_archive_id=None,
        archive_commit_watermark=None,
        source_sequence=1,
        source_read_snapshot_watermark=revision.coverage.snapshot_watermark,
        evidence_role="contemporaneous_basis",
        evidence_content_hash="forged-hash",
        captured_state="committed",
    )
    if _evidence_resolves(fake, ledger):
        _fail("script resolver accepted a forged ao_id")
    print("CONTROL script resolver rejected forged ao_id forged-ao-not-in-ledger")
    bad_claim = InfluenceClaim(
        claim_id="forged-claim",
        source_block_id="forged-block",
        target_block_id=revision.blocks[-1].block_id,
        carried_artifact="forged",
        downstream_effect="forged",
        relation_type=RELATION_INFLUENCE,
        source_evidence_refs=(fake,),
        target_evidence_refs=(fake,),
    )
    dangling = _freeze_or_codes(
        session_id,
        revision,
        list(revision.claims) + [bad_claim],
        (),
    )
    dangling_codes = list(dangling.get("codes") or [])
    if "dangling_claim" not in dangling_codes:
        _fail(f"forged block endpoint was not rejected, result={dangling}")
    print("CONTROL forged claim rejected " + json.dumps(dangling, ensure_ascii=False, sort_keys=True))
    bad_subgraph = RetrievalSubgraph(
        subgraph_id="sg-forged",
        block_ids=(revision.blocks[0].block_id,),
        claim_ids=("forged-claim-id",),
        failure_marked=False,
    )
    unknown = _freeze_or_codes(
        session_id,
        revision,
        list(revision.claims),
        (bad_subgraph,),
    )
    unknown_codes = list(unknown.get("codes") or [])
    if "subgraph_unknown_member" not in unknown_codes:
        _fail(f"forged claim_id was not rejected, result={unknown}")
    print("CONTROL forged claim_id rejected " + json.dumps(unknown, ensure_ascii=False, sort_keys=True))
    real_source = revision.blocks[0]
    real_target = revision.blocks[1]
    forged_evidence_claim = InfluenceClaim(
        claim_id="c-forged-ao",
        source_block_id=real_source.block_id,
        target_block_id=real_target.block_id,
        carried_artifact="run_command",
        downstream_effect="output:forged",
        relation_type=RELATION_INFLUENCE,
        source_evidence_refs=(fake,),
        target_evidence_refs=(
            InfluenceEvidenceRef(
                ao_id=real_target.source_refs[0].ao_id,
                source_session_id=session_id,
                source_archive_id=None,
                archive_commit_watermark=None,
                source_sequence=real_target.source_refs[0].sequence,
                source_read_snapshot_watermark=revision.coverage.snapshot_watermark,
                evidence_role="contemporaneous_basis",
                evidence_content_hash=real_target.source_refs[0].content_hash,
                captured_state="committed",
            ),
        ),
    )
    ao_result = _freeze_or_codes(
        session_id,
        revision,
        [forged_evidence_claim],
        (),
    )
    print(
        "CONTROL forged evidence ao_id on an otherwise valid claim: "
        + json.dumps(ao_result, ensure_ascii=False, sort_keys=True)
    )
    return {
        "script_resolver_rejects_forged_ao_id": True,
        "forged_claim_endpoint_codes": dangling_codes,
        "forged_claim_endpoint": dangling,
        "forged_claim_id_codes": unknown_codes,
        "forged_claim_id": unknown,
        "forged_evidence_ao_id": ao_result,
        "production_does_not_resolve_evidence_ao_id": ao_result.get("rejected") is False,
    }


def _freeze_or_codes(
    session_id: str,
    revision: InfluenceViewRevision,
    claims: list[InfluenceClaim],
    subgraphs: tuple[RetrievalSubgraph, ...],
) -> dict[str, Any]:
    try:
        freeze_extracted_view(
            session_id=session_id,
            task_run_id=revision.task_run_id,
            view_id=revision.view_id,
            revision_id=f"r8a-control:{session_id}",
            frozen_at=FROZEN_AT,
            blocks=list(revision.blocks),
            claims=claims,
            coverage=revision.coverage,
            retrieval_subgraphs=subgraphs,
        )
    except HighSeverityViolation as exc:
        codes = [item.code for item in exc.violations]
        return {
            "rejected": True,
            "codes": codes,
            "violations": [item.to_dict() for item in exc.violations],
        }
    except (TypeError, ValueError) as exc:
        return {"rejected": True, "codes": [], "error": f"{type(exc).__name__}: {exc}"}
    return {"rejected": False, "codes": []}


def _card_patterns(card: Any, record_by_session: dict[str, dict[str, Any]], fork_owner: dict[str, str]) -> tuple[str, set[str]]:
    pointer = card.source_pointer or {}
    if card.kind in {"block_projection", "claim_projection"}:
        session_id = pointer.get("session_id")
        record = record_by_session.get(session_id)
        if record is None:
            return "", set()
        entity = pointer.get("block_or_claim_id") or ""
        if card.kind == "block_projection":
            patterns = record["block_patterns"].get(entity, [])
        else:
            patterns = record["claim_patterns"].get(entity, [])
        return session_id, set(patterns)
    if card.kind in {"fork", "branch"}:
        owner = fork_owner.get(pointer.get("fork_node_id") or "")
        record = record_by_session.get(owner or "")
        if record is None:
            return owner or "", set()
        return owner or "", set(record["tail_patterns"])
    return "", set()


def _structural_search(
    corpus: list[dict[str, Any]],
    fork_rows: list[dict[str, Any]],
    work: Path,
    tag: str,
) -> Any:
    """One production facade over this corpus. Caller issues every query on it."""
    registry = ProjectionRegistry(work / "l3" / f"projections-{tag}.jsonl", clock=_clock)
    revisions = []
    for record in corpus:
        revision = _revision_of(record["revision"])
        if revision is None:
            _fail(f"{record['episode_dir']}: missing revision")
        registry.register(revision, acl_labels=WORKSPACE_LABELS)
        revisions.append(revision)
    store = _new_store(work / "l3" / f"forks-{tag}")
    fork_owner: dict[str, str] = {}
    by_dir = {row["episode_dir"]: row for row in fork_rows}
    for record in corpus:
        fork_row = by_dir.get(record["episode_dir"])
        if fork_row is None:
            continue
        revision_payload = record["revision"]
        if record["revision_wide"] is not None and fork_row["view_revision_id"].startswith("r8a-wide:"):
            revision_payload = record["revision_wide"]
        revision = _revision_of(revision_payload)
        if revision is None:
            continue
        payload = _fork_payload(record, revision)
        written = _write_fork(store, revision, record, payload)
        fork_owner[written.fork_node_id] = record["episode_dir"]
    facade = HybridRetrievalFacade(
        fork_branch=ForkBranchAdapter(work / "l3" / f"forks-{tag}", store),
        projection=InfluenceProjectionAdapter(registry, views=revisions),
        segment_atom=SegmentAtomAdapter(work / "empty-atom", "empty-atom"),
        seed=RETRIEVAL_SEED,
        fact_lane_paused=True,
    )
    record_by_session = {record["episode_dir"]: record for record in corpus}

    def search(query: str) -> list[dict[str, Any]]:
        result = facade.retrieve(query, principal_labels=WORKSPACE_LABELS, profile=L3_PROFILE)
        cards: list[dict[str, Any]] = []
        for card in result.cards[:K]:
            owner, patterns = _card_patterns(card, record_by_session, fork_owner)
            family = ""
            if owner in record_by_session:
                family = record_by_session[owner]["family"]
            cards.append(
                {
                    "kind": card.kind,
                    "owner": owner,
                    "family": family,
                    "patterns": sorted(patterns),
                    "failure_fork": card.kind in {"fork", "branch"},
                }
            )
        return cards

    return search


def _lexical_hits(query: str, corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: list[str] = []
    meta: list[dict[str, Any]] = []
    for record in corpus:
        for event in record["events"]:
            documents.append(f"{event['command']}\n{event['result']}")
            meta.append(
                {
                    "owner": record["episode_dir"],
                    "family": record["family"],
                    "pattern": event["pattern"],
                    "index": event["index"],
                }
            )
    if not documents:
        return []
    scores = BM25Index(documents).scores(query)
    ranked = [index for index, score in enumerate(scores) if score > 0.0]
    ranked.sort(key=lambda index: (-scores[index], meta[index]["owner"], meta[index]["index"]))
    hits = []
    for index in ranked[:K]:
        item = meta[index]
        hits.append(
            {
                "kind": "lexical_event",
                "owner": item["owner"],
                "family": item["family"],
                "patterns": [item["pattern"]],
                "failure_fork": False,
            }
        )
    return hits


def _score_leave_one_out(
    held: dict[str, Any],
    cards: list[dict[str, Any]],
) -> dict[str, Any]:
    events = held["events"]
    half = 0 if not events else math.ceil(len(events) / 2)
    second = events[half:]
    query_tokens = set(tokenize(held["query_prefix"]))
    struct_hits = 0
    xfamily_hits = 0
    nosurf_hits = 0
    nosurf_denom = 0
    for event in second:
        matched = [card for card in cards if event["pattern"] in card["patterns"]]
        if matched:
            struct_hits += 1
        cross = [card for card in matched if card["family"] and card["family"] != held["family"]]
        if cross:
            xfamily_hits += 1
        if event["token"] not in query_tokens:
            nosurf_denom += 1
            if cross:
                nosurf_hits += 1
    return {
        "half_events": len(second),
        "prefix_events": half,
        "hits": _rate(struct_hits, len(second)),
        "cross_family": _rate(xfamily_hits, len(second)),
        "no_surface_cross_family": _rate(nosurf_hits, nosurf_denom),
    }


def _run_l3(records: list[dict[str, Any]], fork_tables: dict[str, list[dict[str, Any]]], work: Path, tests: list[dict[str, Any]]) -> dict[str, Any]:
    by_caliber: dict[str, Any] = {}
    for caliber, fork_rows in fork_tables.items():
        per_episode = []
        for held in records:
            corpus = [record for record in records if record["episode_dir"] != held["episode_dir"]]
            fork_corpus = [
                row
                for row in fork_rows
                if row["episode_dir"] != held["episode_dir"]
            ]
            tag = f"{caliber}-{held['episode_dir']}"
            structural = _structural_search(corpus, fork_corpus, work, tag)(held["query_prefix"])
            lexical = _lexical_hits(held["query_prefix"], corpus)
            structural_score = _score_leave_one_out(held, structural)
            lexical_score = _score_leave_one_out(held, lexical)
            per_episode.append(
                {
                    "episode_dir": held["episode_dir"],
                    "family": held["family"],
                    "n_second_half": structural_score["half_events"],
                    "structural": structural_score,
                    "lexical": lexical_score,
                }
            )
            print(
                f"L3 {caliber} {held['episode_dir']} "
                f"n={structural_score['half_events']} "
                f"struct={structural_score['hits']['rate']} "
                f"lex={lexical_score['hits']['rate']} "
                f"xfamily_struct={structural_score['cross_family']['rate']} "
                f"xfamily_lex={lexical_score['cross_family']['rate']}"
            )
        micro = _micro(per_episode)
        demo = _test_demo(records, fork_rows, tests, work, caliber)
        by_caliber[caliber] = {
            "n_episodes": len(per_episode),
            "leave_one_out": per_episode,
            "micro": micro,
            "test_prompt_demo": demo,
        }
    return {
        "label": "retrieval-level only; not task-success generalization",
        "k": K,
        "profile": L3_PROFILE,
        "hit_definition": HIT_DEFINITION,
        "calibers": by_caliber,
    }


def _micro(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for arm in ("structural", "lexical"):
        for field, key in (
            ("hits", "hits"),
            ("cross_family", "cross_family"),
            ("no_surface_cross_family", "no_surface"),
        ):
            hits = sum(row[arm][field]["hits"] for row in rows)
            if field == "no_surface_cross_family":
                denominator = sum(row[arm][field]["denominator"] for row in rows)
            else:
                denominator = sum(row[arm][field]["denominator"] for row in rows)
            out[f"{arm}_{key}"] = _rate(hits, denominator)
    return out


def _test_demo(
    records: list[dict[str, Any]],
    fork_rows: list[dict[str, Any]],
    tests: list[dict[str, Any]],
    work: Path,
    caliber: str,
) -> dict[str, Any]:
    structural_families: Counter[str] = Counter()
    lexical_families: Counter[str] = Counter()
    failure_fork_cards = 0
    structural_cards = 0
    lexical_cards = 0
    structural_search = _structural_search(records, fork_rows, work, f"{caliber}-test")
    for test in tests:
        query = _clip(test["prompt"], TEST_PROMPT_CLIP)
        structural = structural_search(query)
        lexical = _lexical_hits(query, records)
        for card in structural:
            structural_cards += 1
            structural_families[card["family"] or "unknown"] += 1
            if card["failure_fork"]:
                failure_fork_cards += 1
        for card in lexical:
            lexical_cards += 1
            lexical_families[card["family"] or "unknown"] += 1
    return {
        "label": "interface demo on unseen task prompts; not a generalization metric",
        "n_prompts": len(tests),
        "k": K,
        "structural_cards": structural_cards,
        "structural_family_counts": dict(sorted(structural_families.items())),
        "structural_failure_fork_cards": failure_fork_cards,
        "lexical_cards": lexical_cards,
        "lexical_family_counts": dict(sorted(lexical_families.items())),
        "lexical_failure_fork_cards": 0,
    }


def _test_prompts(catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for task_id in sorted(catalog):
        row = catalog[task_id]
        if row.get("split") != "test":
            continue
        turns = row.get("turns")
        prompt = ""
        if isinstance(turns, list) and turns and isinstance(turns[0], dict):
            raw = turns[0].get("prompt")
            if isinstance(raw, str):
                prompt = raw
        if not prompt:
            _fail(f"test task {task_id} has no turns[0].prompt")
        family = row.get("family_id")
        if not isinstance(family, str):
            _fail(f"test task {task_id} missing family_id")
        rows.append({"task_id": task_id, "family": family, "prompt": prompt})
    if len(rows) != 45:
        _fail(f"expected 45 test prompts, found {len(rows)}")
    return rows


def _episode_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rows.append(
            {
                "episode_dir": record["episode_dir"],
                "episode_id": record["episode_id"],
                "task_id": record["task_id"],
                "family": record["family"],
                "confirm_status": record["confirm_status"],
                "tail_shape": record["tail_shape"],
                "json_truncated": record["json_truncated"],
                "bad_json_lines": record["bad_json_lines"],
                "tool_calls": record["tool_calls"],
                "block_n": record["block_n"],
                "claim_influence": record["claim_influence"],
                "claim_skip": record["claim_skip"],
                "subgraph_narrow": record["subgraph_narrow"],
                "subgraph_wide": record["subgraph_wide"],
                "omitted_between_claims": record["omitted_between_claims"],
            }
        )
    return rows


def _report(
    records: list[dict[str, Any]],
    forks: dict[str, list[dict[str, Any]]] | None,
    control: dict[str, Any] | None,
    l3: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "hit_definition": HIT_DEFINITION,
        "subgraph_rule": SUBGRAPH_RULE,
        "influence_rule": INFLUENCE_RULE,
        "attempts_path": str(SB_ATTEMPTS),
        "episodes_path": str(SB_EPISODES),
        "not_task_success": True,
        "episodes": _episode_table(records),
        "forks": forks,
        "control": control,
        "l3": l3,
        "unmeasured": [
            "task-success generalization: these 14 episodes have no grader verdict, and the paused live stack was not touched",
            "model-driven extraction: this wave is a deterministic stand-in; the faithful extractor is ledger #125 and needs an authorized model budget",
            "the other 28 train tasks have no trajectory in this run",
            "the 45 test tasks have prompts only; section (c) is an interface demo",
        ],
    }


def _checkpoint_path(work: Path, episode_dir: str) -> Path:
    return work / "checkpoints" / f"{episode_dir}.json"


def _load_checkpoint(work: Path, episode_dir: str) -> dict[str, Any] | None:
    path = _checkpoint_path(work, episode_dir)
    if not path.is_file():
        return None
    payload = _load_json(path)
    if payload.get("schema") != SCHEMA:
        _fail(f"{episode_dir}: checkpoint schema {payload.get('schema')!r}")
    return payload


def _selected_dirs(stage: str, dirs: list[Path]) -> list[Path]:
    if stage == "1":
        return [path for path in dirs if path.name.startswith("0001-")]
    return list(dirs)


def main() -> None:
    parser = argparse.ArgumentParser(description="R8a SkillsBench influence and fork extraction")
    parser.add_argument("--out", type=Path, default=Path("/tmp/sb-influence/report.json"))
    parser.add_argument("--stage", choices=("1", "2", "3", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    _print_policy()
    work = args.out.parent
    work.mkdir(parents=True, exist_ok=True)
    (work / "empty-atom").mkdir(parents=True, exist_ok=True)
    catalog = _load_catalog()
    attempts = _load_attempts()
    dirs = _episode_dirs()
    selected = _selected_dirs(args.stage, dirs)
    records: list[dict[str, Any]] = []
    if args.stage in {"1", "2", "all"}:
        for index, episode_dir in enumerate(dirs):
            if episode_dir not in selected and args.stage != "all":
                existing = _load_checkpoint(work, episode_dir.name) if args.resume else None
                if existing is not None:
                    records.append(existing)
                continue
            existing = _load_checkpoint(work, episode_dir.name) if args.resume else None
            if existing is not None:
                print(f"RESUME {episode_dir.name}")
                records.append(existing)
                continue
            if args.stage == "1" and not episode_dir.name.startswith("0001-"):
                continue
            print(f"STAGE extract {episode_dir.name}")
            record = _extract_episode(
                episode_dir,
                index,
                catalog,
                attempts,
                work / "ledgers",
            )
            _dump(_checkpoint_path(work, episode_dir.name), record)
            records.append(record)
            print(
                f"EPISODE {record['episode_dir']} status={record['confirm_status']} "
                f"tail={record['tail_shape']} tools={record['tool_calls']} "
                f"blocks={record['block_n']} influence={record['claim_influence']} "
                f"skip={record['claim_skip']} "
                f"subgraph_narrow={record['subgraph_narrow']} "
                f"subgraph_wide={record['subgraph_wide']} "
                f"truncated={record['json_truncated']}"
            )
            partial = _report(records, None, None, None)
            _dump(args.out, partial)
    if args.stage == "3":
        records = []
        for episode_dir in dirs:
            existing = _load_checkpoint(work, episode_dir.name)
            if existing is None:
                _fail(f"stage 3 missing checkpoint for {episode_dir.name}")
            records.append(existing)
    records.sort(key=lambda item: item["episode_dir"])
    if args.stage == "1" and len(records) != 1:
        _fail(f"stage 1 produced {len(records)} episodes")
    if args.stage in {"2", "all", "3"} and len(records) != 14:
        _fail(f"expected 14 episode checkpoints, found {len(records)}")
    forks = None
    control = None
    l3 = None
    if args.stage in {"1", "2", "all"}:
        if any(record["episode_dir"].startswith("0001-") for record in records):
            control = _control_gate(
                next(record for record in records if record["episode_dir"].startswith("0001-")),
                work / "ledgers",
            )
        if args.stage == "1":
            forks = {"narrow": _run_forks(records, work, "narrow"), "wide": []}
        else:
            forks = {
                "narrow": _run_forks(records, work, "narrow"),
                "wide": _run_forks(records, work, "wide"),
            }
        _dump(work / "forks-report.json", forks)
        if control is not None:
            _dump(work / "control.json", control)
    if args.stage in {"3", "all"}:
        if forks is None:
            forks = _load_json(work / "forks-report.json")
        if control is None:
            control_path = work / "control.json"
            control = _load_json(control_path) if control_path.is_file() else None
        tests = _test_prompts(catalog)
        print(f"STAGE 3 L3 n_episodes={len(records)} n_test_prompts={len(tests)}")
        l3 = _run_l3(records, forks, work, tests)
    report = _report(records, forks, control, l3)
    _dump(args.out, report)
    print(f"WROTE {args.out}")
    tool_total = sum(record["tool_calls"] for record in records)
    print(f"TOOL_CALLS_SUM {tool_total}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
