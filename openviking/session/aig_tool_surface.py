# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Paper tool surface for one fixed-watermark AO snapshot (ADR-0044).

``T_log`` reads the snapshot and the mutable paper draft. ``T_build`` edits
that draft. Hard invariants fail closed on the call that would break them.
Semantic fidelity and ``Φ(G)`` are not judged here.

Durable state lives at ``<root>/toolloop-runs/<run_id>/``. ``meta.json`` is
created exclusively. Each accepted build verb appends one content-hashed
``toolloop_draft_op`` line to ``draft.jsonl``. Replaying that log rebuilds
the same draft. ``submit`` seals the run read-only.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openviking.session.aig_native_protocol import (
    TOOLLOOP_BUILD_VERBS,
    ProtocolContractError,
    canonical_dumps,
    is_safe_id,
    read_toolloop_draft_op,
    toolloop_draft_op_document,
)
from openviking.session.ao_ledger import AORecord
from openviking.session.append_only import (
    append_jsonl,
    read_json_object,
    read_jsonl_objects,
    write_exclusive,
)
from openviking.session.outcome_signal import (
    OutcomeSignal,
    outcome_fingerprint_body,
    outcome_overview,
)

TOOLLOOP_RUNS_DIRNAME = "toolloop-runs"
META_FILENAME = "meta.json"
DRAFT_LOG_FILENAME = "draft.jsonl"
SEARCH_MATCH_LIMIT = 20
_SNIPPET_LIMIT = 120
_READONLY_FILE = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
_READONLY_DIR = (
    stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
)

T_LOG_VERBS = (
    "get_overview",
    "read_full_log",
    "read_step",
    "read_step_range",
    "list_agents",
    "get_question",
    "search",
    "list_block_kinds",
    "get_story",
    "get_graph",
)
PAPER_KINDS = ("task", "orchestrator", "agent", "conclusion")
AUTHORSHIP_VALUES = ("authored", "relayed", "executed")
_PAPER_KIND_SET = frozenset(PAPER_KINDS)
_AUTHORSHIP_SET = frozenset(AUTHORSHIP_VALUES)
_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "task": ("content", "operative_brief", "step_refs"),
    "orchestrator": ("agent_name", "summary", "authorship", "step_refs"),
    "agent": ("agent_name", "summary", "input", "output", "authorship", "step_refs"),
    "conclusion": ("content", "step_refs"),
}
_OPTIONAL_FIELDS: dict[str, frozenset[str]] = {"agent": frozenset({"defect_recorded"})}
_PARTICIPANT_KEYS = ("participant_id", "participant", "agent_name", "agent")
_SEGMENT_KEYS = ("segment_id", "segment")
_META_BODY_KEYS = (
    "question",
    "question_present",
    "row_count",
    "run_id",
    "snapshot_fingerprint",
    "watermark",
)
_LINK_KIND_ADJACENT = "adjacent"
_LINK_KIND_SKIP = "skip"

TOOL_CATALOG: tuple[tuple[str, str], ...] = (
    ("get_overview", "Compact log metadata: row count, participant, segment, and size."),
    ("read_full_log", "Return every snapshot row joined as text."),
    ("read_step", "Return one snapshot row. Argument index (alias i) is the row handle."),
    ("read_step_range", "Return a closed row interval. Arguments start/end (aliases a/b)."),
    ("list_agents", "List identity, role, and capability. System prompts are omitted."),
    ("get_question", "Return the original user question carried by the snapshot."),
    ("search", "Regex search over row text. At most 20 matches. Optional flags i/m/s."),
    ("list_block_kinds", "Closed paper block vocabulary and required fields."),
    ("get_story", "Render the current draft as Markdown."),
    ("get_graph", "Return the current draft as a JSON object."),
    ("add_block", "Create a task, orchestrator, agent, or conclusion block."),
    (
        "add_inheritance_note",
        "Attach inherited/effect on an adjacent earlier-to-later edge.",
    ),
    ("add_skip_attribution", "Attach inherited/effect on a non-adjacent forward edge."),
    ("modify_block", "Replace fields of an existing block. Kind and id stay fixed."),
    ("delete_block", "Delete a block and every edge that touches it."),
    ("delete_inheritance_note", "Remove an adjacent inheritance note."),
    ("delete_skip_attribution", "Remove a non-adjacent skip attribution."),
    ("mark_defect_recorded", "Record that an agent block missed its instruction."),
    ("submit", "Seal the draft once. Structural Phi is not judged here."),
)


class ToolSurfaceError(ValueError):
    """A tool call was rejected. ``code`` is the stable machine-readable reason."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class SnapshotAgent:
    """One catalog entry. ``system_prompt`` is stored and never returned."""

    identity: str
    role: str
    capability: str
    system_prompt: str = ""


@dataclass(frozen=True)
class AoSnapshot:
    """Fixed AO snapshot. Row handles are positions in ``records``."""

    records: tuple[AORecord, ...]
    watermark: str
    question: str | None = None
    agents: tuple[SnapshotAgent, ...] = ()
    outcome: OutcomeSignal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.watermark, str) or not self.watermark.strip():
            raise ToolSurfaceError("invalid_snapshot")
        if isinstance(self.records, tuple):
            records = self.records
        elif isinstance(self.records, Sequence) and not isinstance(self.records, (str, bytes)):
            records = tuple(self.records)
            object.__setattr__(self, "records", records)
        else:
            raise ToolSurfaceError("invalid_snapshot")
        for record in records:
            if not isinstance(record, AORecord):
                raise ToolSurfaceError("invalid_snapshot")
        if self.question is not None and not isinstance(self.question, str):
            raise ToolSurfaceError("invalid_snapshot")
        if isinstance(self.agents, tuple):
            agents = self.agents
        elif isinstance(self.agents, Sequence) and not isinstance(self.agents, (str, bytes)):
            agents = tuple(self.agents)
            object.__setattr__(self, "agents", agents)
        else:
            raise ToolSurfaceError("invalid_snapshot")
        for agent in agents:
            if not isinstance(agent, SnapshotAgent):
                raise ToolSurfaceError("invalid_snapshot")
            if not isinstance(agent.identity, str) or not agent.identity.strip():
                raise ToolSurfaceError("invalid_snapshot")
            if not isinstance(agent.role, str) or not isinstance(agent.capability, str):
                raise ToolSurfaceError("invalid_snapshot")
            if not isinstance(agent.system_prompt, str):
                raise ToolSurfaceError("invalid_snapshot")
        if self.outcome is not None and not isinstance(self.outcome, OutcomeSignal):
            raise ToolSurfaceError("invalid_snapshot")


@dataclass(frozen=True)
class _Block:
    block_id: str
    kind: str
    fields: dict[str, Any]


@dataclass(frozen=True)
class _Edge:
    source_id: str
    target_id: str
    inherited: str
    effect: str
    link_kind: str


def snapshot_fingerprint(
    records: Sequence[AORecord],
    outcome: OutcomeSignal | None = None,
) -> str:
    """SHA-256 of the row bodies this draft is bound to, plus outcome when set.

    ``outcome=None`` hashes the same row list as before so existing run
    directories stay matched. A different outcome is a different snapshot.
    """
    rows = [
        {
            "action": record.action,
            "ao_id": record.ao_id,
            "captured_state": record.captured_state,
            "created_at": record.created_at,
            "index": index,
            "message_ref": record.message_ref,
            "observation": record.observation,
            "sequence": record.sequence,
            "session_id": record.session_id,
            "skill_invocations": record.skill_invocations,
        }
        for index, record in enumerate(records)
    ]
    body = outcome_fingerprint_body(outcome)
    payload: Any = rows if body is None else {"outcome": body, "rows": rows}
    return hashlib.sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()


def replay_toolloop_draft(root: Path, run_id: str, snapshot: AoSnapshot) -> dict[str, Any]:
    """Rebuild a draft from its log. A second call returns the same graph."""
    return AigToolSurface(snapshot, root=root, run_id=run_id).get_graph()


def _lookup(record: AORecord, keys: Sequence[str]) -> str:
    bags: list[Mapping[str, Any]] = [record.action, record.observation]
    if isinstance(record.message_ref, Mapping):
        bags.append(record.message_ref)
    for bag in bags:
        for key in keys:
            value = bag.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _participant(record: AORecord) -> str:
    return _lookup(record, _PARTICIPANT_KEYS) or "unspecified"


def _segment(record: AORecord) -> str:
    return _lookup(record, _SEGMENT_KEYS) or str(record.sequence)


def _step_text(index: int, record: AORecord) -> str:
    return (
        f"[{index}] participant={_participant(record)} segment={_segment(record)}\n"
        f"action: {canonical_dumps(record.action)}\n"
        f"observation: {canonical_dumps(record.observation)}"
    )


def _question_text(question: str | None) -> str | None:
    if isinstance(question, str) and question.strip():
        return question
    return None


def _exact_args(
    arguments: Mapping[str, Any], allowed: frozenset[str], required: frozenset[str]
) -> None:
    extra = [key for key in arguments if key not in allowed]
    if extra:
        raise ToolSurfaceError("invalid_args")
    missing = [key for key in required if key not in arguments]
    if missing:
        raise ToolSurfaceError("invalid_args")


def _parse_step_refs(value: Any, row_count: int) -> tuple[int, ...]:
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ToolSurfaceError("invalid_step_refs")
    if not value:
        raise ToolSurfaceError("step_refs_empty")
    numbers: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ToolSurfaceError("invalid_step_refs")
        if item < 0 or item >= row_count:
            raise ToolSurfaceError("step_out_of_range")
        numbers.append(item)
    ordered = tuple(sorted(numbers))
    if len(set(ordered)) != len(ordered) or ordered[-1] - ordered[0] + 1 != len(ordered):
        raise ToolSurfaceError("step_refs_not_contiguous")
    return ordered


def _parse_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolSurfaceError("missing_field")
    return value


def _parse_authorship(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolSurfaceError("missing_field")
    if value not in _AUTHORSHIP_SET:
        raise ToolSurfaceError("invalid_authorship")
    return value


def _normalize_fields(kind: str, fields: Any, row_count: int) -> dict[str, Any]:
    if kind not in _PAPER_KIND_SET:
        raise ToolSurfaceError("invalid_kind")
    if not isinstance(fields, Mapping):
        raise ToolSurfaceError("invalid_args")
    if "kind" in fields:
        raise ToolSurfaceError("kind_immutable")
    if "block_id" in fields or "id" in fields:
        raise ToolSurfaceError("id_immutable")
    optional = _OPTIONAL_FIELDS.get(kind, frozenset())
    allowed = set(_REQUIRED_FIELDS[kind]) | set(optional)
    unknown = [key for key in fields if key not in allowed]
    if unknown:
        raise ToolSurfaceError("unknown_field")
    missing = [key for key in _REQUIRED_FIELDS[kind] if key not in fields]
    if missing:
        raise ToolSurfaceError("missing_field")
    normalized: dict[str, Any] = {}
    for key in _REQUIRED_FIELDS[kind]:
        if key == "step_refs":
            normalized[key] = _parse_step_refs(fields[key], row_count)
        elif key == "authorship":
            normalized[key] = _parse_authorship(fields[key])
        else:
            normalized[key] = _parse_text(fields[key])
    if "defect_recorded" in fields:
        normalized["defect_recorded"] = _parse_text(fields["defect_recorded"])
    return normalized


def _fields_json(fields: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in fields.items():
        payload[key] = list(value) if key == "step_refs" else value
    return payload


def _ordered_ids(blocks: Mapping[str, _Block]) -> list[str]:
    ordered = sorted(
        blocks.values(),
        key=lambda block: (block.fields["step_refs"][0], block.block_id),
    )
    return [block.block_id for block in ordered]


def _is_adjacent(blocks: Mapping[str, _Block], source_id: str, target_id: str) -> bool:
    ids = _ordered_ids(blocks)
    return ids.index(target_id) == ids.index(source_id) + 1


def _validate(blocks: Mapping[str, _Block], edges: Sequence[_Edge], row_count: int) -> None:
    tasks = 0
    conclusions = 0
    spans: list[tuple[int, int]] = []
    for block in blocks.values():
        if block.kind not in _PAPER_KIND_SET:
            raise ToolSurfaceError("invalid_kind")
        refs = block.fields.get("step_refs")
        if not isinstance(refs, tuple) or not refs:
            raise ToolSurfaceError("invalid_step_refs")
        if refs[0] < 0 or refs[-1] >= row_count:
            raise ToolSurfaceError("step_out_of_range")
        if block.kind == "task":
            tasks += 1
        elif block.kind == "conclusion":
            conclusions += 1
        spans.append((refs[0], refs[-1]))
    if tasks > 1:
        raise ToolSurfaceError("duplicate_task")
    if conclusions > 1:
        raise ToolSurfaceError("duplicate_conclusion")
    spans.sort()
    for index, (start, end) in enumerate(spans):
        if index and start <= spans[index - 1][1]:
            raise ToolSurfaceError("step_refs_overlap")
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        if edge.source_id not in blocks or edge.target_id not in blocks:
            raise ToolSurfaceError("unknown_block")
        pair = (edge.source_id, edge.target_id)
        if pair in seen:
            raise ToolSurfaceError("duplicate_edge")
        seen.add(pair)
        source_min = blocks[edge.source_id].fields["step_refs"][0]
        target_min = blocks[edge.target_id].fields["step_refs"][0]
        if source_min >= target_min:
            raise ToolSurfaceError("edge_not_forward")
        expected = (
            _LINK_KIND_ADJACENT
            if _is_adjacent(blocks, edge.source_id, edge.target_id)
            else _LINK_KIND_SKIP
        )
        if edge.link_kind != expected:
            raise ToolSurfaceError("link_kind_conflict")


class AigToolSurface:
    """T_log and T_build bound to one snapshot and one durable run directory."""

    def __init__(self, snapshot: AoSnapshot, *, root: Path, run_id: str) -> None:
        if not isinstance(snapshot, AoSnapshot):
            raise ToolSurfaceError("invalid_snapshot")
        self._snapshot = snapshot
        self._records = snapshot.records
        self._watermark = snapshot.watermark
        self._question = _question_text(snapshot.question)
        self._agents = snapshot.agents
        self._fingerprint = snapshot_fingerprint(self._records, snapshot.outcome)
        self._run_dir = _run_directory(root, run_id)
        self._run_id = run_id
        self._meta_path = self._run_dir / META_FILENAME
        self._log_path = self._run_dir / DRAFT_LOG_FILENAME
        self._lock = threading.Lock()
        self._blocks: dict[str, _Block] = {}
        self._edges: list[_Edge] = []
        self._ops: list[dict[str, Any]] = []
        self._prev_hash = ""
        self._submitted = False
        self._refine_open = False
        self._handlers = {
            "get_overview": self._get_overview,
            "read_full_log": self._read_full_log,
            "read_step": self._read_step,
            "read_step_range": self._read_step_range,
            "list_agents": self._list_agents,
            "get_question": self._get_question,
            "search": self._search,
            "list_block_kinds": self._list_block_kinds,
            "get_story": self._get_story,
            "get_graph": self._get_graph,
            "add_block": self._add_block,
            "add_inheritance_note": self._add_inheritance_note,
            "add_skip_attribution": self._add_skip_attribution,
            "modify_block": self._modify_block,
            "delete_block": self._delete_block,
            "delete_inheritance_note": self._delete_inheritance_note,
            "delete_skip_attribution": self._delete_skip_attribution,
            "mark_defect_recorded": self._mark_defect_recorded,
            "submit": self._submit,
        }
        if self._meta_path.exists():
            self._load()
        else:
            self._create()

    @property
    def run_directory(self) -> Path:
        return self._run_dir

    @property
    def draft_log_path(self) -> Path:
        return self._log_path

    @property
    def submitted(self) -> bool:
        with self._lock:
            return self._submitted

    @property
    def operation_count(self) -> int:
        with self._lock:
            return len(self._ops)

    def call(self, verb: str, arguments: Mapping[str, Any] | None = None) -> Any:
        """Run one tool verb. Build failures leave the draft unchanged."""
        with self._lock:
            if not isinstance(verb, str) or verb not in self._handlers:
                raise ToolSurfaceError("unknown_verb")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, Mapping):
                raise ToolSurfaceError("invalid_args")
            copied = dict(arguments)
            if verb in TOOLLOOP_BUILD_VERBS:
                _reject_outcome_mutation(copied)
            return self._handlers[verb](copied)

    def get_overview(self) -> dict[str, Any]:
        """Return row count plus each row's participant, segment, and size."""
        return self.call("get_overview")

    def read_full_log(self) -> str:
        """Return every snapshot row joined as text."""
        return self.call("read_full_log")

    def read_step(self, index: int) -> dict[str, Any]:
        """Return the snapshot row at ``index``."""
        return self.call("read_step", {"index": index})

    def read_step_range(self, start: int, end: int) -> list[dict[str, Any]]:
        """Return the closed interval of snapshot rows."""
        return self.call("read_step_range", {"start": start, "end": end})

    def list_agents(self) -> list[dict[str, str]]:
        """Return identity, role, and capability. System prompts stay hidden."""
        return self.call("list_agents")

    def get_question(self) -> dict[str, str]:
        """Return the snapshot question, or ``question_missing``."""
        return self.call("get_question")

    def search(self, pattern: str, flags: str | int | None = None) -> dict[str, Any]:
        """Regex-search row text. At most 20 matches are returned."""
        arguments: dict[str, Any] = {"pattern": pattern}
        if flags is not None:
            arguments["flags"] = flags
        return self.call("search", arguments)

    def list_block_kinds(self) -> dict[str, Any]:
        """Return the four paper block kinds and their required fields."""
        return self.call("list_block_kinds")

    def get_story(self) -> str:
        """Render the current draft as Markdown."""
        return self.call("get_story")

    def get_graph(self) -> dict[str, Any]:
        """Return the current draft as a JSON object."""
        return self.call("get_graph")

    def add_block(
        self,
        kind: str,
        fields: Mapping[str, Any],
        *,
        block_id: str | None = None,
        from_refs: Sequence[Mapping[str, Any]] | None = None,
        to_refs: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, str]:
        """Create one block. Optional refs attach inherited/effect notes."""
        arguments: dict[str, Any] = {"kind": kind, "fields": fields}
        if block_id is not None:
            arguments["block_id"] = block_id
        if from_refs is not None:
            arguments["from_refs"] = list(from_refs)
        if to_refs is not None:
            arguments["to_refs"] = list(to_refs)
        return self.call("add_block", arguments)

    def add_inheritance_note(
        self, source_id: str, target_id: str, inherited: str, effect: str
    ) -> dict[str, str]:
        """Attach a note on the adjacent edge from an earlier block to a later one."""
        return self.call(
            "add_inheritance_note",
            {"from": source_id, "to": target_id, "inherited": inherited, "effect": effect},
        )

    def add_skip_attribution(
        self, source_id: str, target_id: str, inherited: str, effect: str
    ) -> dict[str, str]:
        """Attach a note on a non-adjacent forward edge."""
        return self.call(
            "add_skip_attribution",
            {"from": source_id, "to": target_id, "inherited": inherited, "effect": effect},
        )

    def modify_block(self, block_id: str, fields: Mapping[str, Any]) -> dict[str, str]:
        """Overwrite fields. Kind and block id cannot change."""
        return self.call("modify_block", {"block_id": block_id, "fields": fields})

    def delete_block(self, block_id: str) -> dict[str, str]:
        """Delete a block and the edges that touch it."""
        return self.call("delete_block", {"block_id": block_id})

    def delete_inheritance_note(self, source_id: str, target_id: str) -> dict[str, str]:
        """Remove an adjacent inheritance note."""
        return self.call("delete_inheritance_note", {"from": source_id, "to": target_id})

    def delete_skip_attribution(self, source_id: str, target_id: str) -> dict[str, str]:
        """Remove a non-adjacent skip attribution."""
        return self.call("delete_skip_attribution", {"from": source_id, "to": target_id})

    def mark_defect_recorded(self, block_id: str, note: str) -> dict[str, str]:
        """Mark an agent block whose output missed the instruction it was given."""
        return self.call("mark_defect_recorded", {"block_id": block_id, "note": note})

    def submit(self) -> dict[str, bool]:
        """Seal the draft. A second call raises ``already_submitted``."""
        return self.call("submit")

    def apply_refine(self, verb: str, arguments: Mapping[str, Any] | None = None) -> Any:
        """Apply one build verb after ``submit``. The sealed builder log stays put.

        Refiner edits live in the caller's append-only trajectory. ``submit``
        is rejected. Hard invariants still fail closed and leave memory unchanged
        when the verb raises before commit.
        """
        with self._lock:
            if not self._submitted:
                raise ToolSurfaceError("not_submitted")
            if verb == "submit":
                raise ToolSurfaceError("submit_forbidden")
            if not isinstance(verb, str) or verb not in TOOLLOOP_BUILD_VERBS:
                raise ToolSurfaceError("unknown_verb")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, Mapping):
                raise ToolSurfaceError("invalid_args")
            copied = dict(arguments)
            _reject_outcome_mutation(copied)
            self._refine_open = True
            try:
                return self._handlers[verb](copied)
            finally:
                self._refine_open = False

    def _create(self) -> None:
        body = self._meta_body()
        payload = dict(body)
        payload["content_hash"] = _meta_digest(body)
        try:
            write_exclusive(self._meta_path, payload)
        except FileExistsError:
            self._load()
            return
        self._prev_hash = payload["content_hash"]

    def _meta_body(self) -> dict[str, Any]:
        return {
            "question": self._question or "",
            "question_present": self._question is not None,
            "row_count": len(self._records),
            "run_id": self._run_id,
            "snapshot_fingerprint": self._fingerprint,
            "watermark": self._watermark,
        }

    def _load(self) -> None:
        payload = read_json_object(self._meta_path, ToolSurfaceError, code="stored_record_invalid")
        if set(payload) != set(_META_BODY_KEYS) | {"content_hash"}:
            raise ToolSurfaceError("stored_record_invalid")
        body = {key: payload[key] for key in _META_BODY_KEYS}
        digest = _meta_digest(body)
        if payload.get("content_hash") != digest:
            raise ToolSurfaceError("stored_record_invalid")
        row_count = body["row_count"]
        question_present = body["question_present"]
        if (
            not isinstance(question_present, bool)
            or isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or not isinstance(body["question"], str)
            or not isinstance(body["run_id"], str)
            or not isinstance(body["watermark"], str)
            or not isinstance(body["snapshot_fingerprint"], str)
        ):
            raise ToolSurfaceError("stored_record_invalid")
        if (
            body["run_id"] != self._run_id
            or body["watermark"] != self._watermark
            or row_count != len(self._records)
            or body["snapshot_fingerprint"] != self._fingerprint
            or question_present != (self._question is not None)
            or (self._question is not None and body["question"] != self._question)
        ):
            raise ToolSurfaceError("snapshot_mismatch")
        self._prev_hash = digest
        rows = read_jsonl_objects(self._log_path, ToolSurfaceError, code="stored_record_invalid")
        blocks: dict[str, _Block] = {}
        edges: list[_Edge] = []
        submitted = False
        for index, row in enumerate(rows):
            try:
                op = read_toolloop_draft_op(row)
            except ProtocolContractError as exc:
                raise ToolSurfaceError("stored_record_invalid") from exc
            if (
                op.seq != index
                or op.run_id != self._run_id
                or op.prev_content_hash != self._prev_hash
            ):
                raise ToolSurfaceError("draft_chain_broken")
            if submitted:
                raise ToolSurfaceError("draft_chain_broken")
            blocks, edges, submitted = self._restore(blocks, edges, op.verb, op.args)
            self._prev_hash = op.content_hash
            self._ops.append(dict(row))
        self._blocks = blocks
        self._edges = edges
        self._submitted = submitted
        if submitted:
            self._seal()

    def _restore(
        self,
        blocks: dict[str, _Block],
        edges: list[_Edge],
        verb: str,
        args: Mapping[str, Any],
    ) -> tuple[dict[str, _Block], list[_Edge], bool]:
        if verb == "add_block":
            return (*self._apply_add(blocks, edges, args), False)
        if verb == "add_inheritance_note":
            return (*self._apply_note(blocks, edges, args, _LINK_KIND_ADJACENT), False)
        if verb == "add_skip_attribution":
            return (*self._apply_note(blocks, edges, args, _LINK_KIND_SKIP), False)
        if verb == "modify_block":
            return (*self._apply_modify(blocks, edges, args), False)
        if verb == "delete_block":
            return (*self._apply_delete_block(blocks, edges, args), False)
        if verb == "delete_inheritance_note":
            return (*self._apply_delete_note(blocks, edges, args, _LINK_KIND_ADJACENT), False)
        if verb == "delete_skip_attribution":
            return (*self._apply_delete_note(blocks, edges, args, _LINK_KIND_SKIP), False)
        if verb == "mark_defect_recorded":
            return (*self._apply_defect(blocks, edges, args), False)
        if verb == "submit":
            if dict(args) != {}:
                raise ToolSurfaceError("stored_record_invalid")
            _validate(blocks, edges, len(self._records))
            return blocks, edges, True
        raise ToolSurfaceError("stored_record_invalid")

    def _get_overview(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        _exact_args(arguments, frozenset(), frozenset())
        rows = []
        for index, record in enumerate(self._records):
            rows.append(
                {
                    "index": index,
                    "participant": _participant(record),
                    "segment": _segment(record),
                    "size": len(_step_text(index, record)),
                }
            )
        payload: dict[str, Any] = {
            "row_count": len(self._records),
            "watermark": self._watermark,
            "rows": rows,
        }
        if self._snapshot.outcome is not None:
            payload["outcome"] = outcome_overview(self._snapshot.outcome)
        return payload

    def _read_full_log(self, arguments: Mapping[str, Any]) -> str:
        _exact_args(arguments, frozenset(), frozenset())
        return "\n\n".join(
            _step_text(index, record) for index, record in enumerate(self._records)
        )

    def _read_step(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        _exact_args(arguments, frozenset({"index", "i"}), frozenset())
        index = _row_index(_one_of(arguments, "index", "i"), len(self._records))
        return {"index": index, "record": self._records[index].to_dict()}

    def _read_step_range(self, arguments: Mapping[str, Any]) -> list[dict[str, Any]]:
        _exact_args(arguments, frozenset({"start", "end", "a", "b"}), frozenset())
        start = _row_index(_one_of(arguments, "start", "a"), len(self._records))
        end = _row_index(_one_of(arguments, "end", "b"), len(self._records))
        if start > end:
            raise ToolSurfaceError("invalid_range")
        return [
            {"index": index, "record": self._records[index].to_dict()}
            for index in range(start, end + 1)
        ]

    def _list_agents(self, arguments: Mapping[str, Any]) -> list[dict[str, str]]:
        _exact_args(arguments, frozenset(), frozenset())
        if self._agents:
            return [
                {
                    "identity": agent.identity,
                    "role": agent.role,
                    "capability": agent.capability,
                }
                for agent in self._agents
            ]
        found: list[dict[str, str]] = []
        seen: set[str] = set()
        for record in self._records:
            identity = _lookup(record, _PARTICIPANT_KEYS)
            if not identity or identity in seen:
                continue
            seen.add(identity)
            found.append(
                {
                    "identity": identity,
                    "role": _lookup(record, ("role",)),
                    "capability": _lookup(record, ("capability",)),
                }
            )
        return found

    def _get_question(self, arguments: Mapping[str, Any]) -> dict[str, str]:
        _exact_args(arguments, frozenset(), frozenset())
        if self._question is None:
            raise ToolSurfaceError("question_missing")
        return {"question": self._question}

    def _search(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        _exact_args(arguments, frozenset({"pattern", "flags"}), frozenset({"pattern"}))
        pattern = arguments["pattern"]
        if not isinstance(pattern, str) or pattern == "":
            raise ToolSurfaceError("invalid_pattern")
        flags = _regex_flags(arguments["flags"]) if "flags" in arguments else 0
        try:
            compiled = re.compile(pattern, flags)
        except re.error as exc:
            raise ToolSurfaceError("invalid_pattern") from exc
        matches: list[dict[str, Any]] = []
        truncated = False
        for index, record in enumerate(self._records):
            for match in compiled.finditer(_step_text(index, record)):
                if len(matches) >= SEARCH_MATCH_LIMIT:
                    truncated = True
                    break
                snippet = match.group(0)
                if len(snippet) > _SNIPPET_LIMIT:
                    snippet = snippet[: _SNIPPET_LIMIT - 3] + "..."
                matches.append({"index": index, "snippet": snippet})
            if truncated:
                break
        return {"matches": matches, "truncated": truncated}

    def _list_block_kinds(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        _exact_args(arguments, frozenset(), frozenset())
        kinds = {}
        for kind in PAPER_KINDS:
            kinds[kind] = {
                "required": list(_REQUIRED_FIELDS[kind]),
                "optional": sorted(_OPTIONAL_FIELDS.get(kind, ())),
            }
        return {"kinds": kinds, "authorship": list(AUTHORSHIP_VALUES)}

    def _get_story(self, arguments: Mapping[str, Any]) -> str:
        _exact_args(arguments, frozenset(), frozenset())
        graph = self._graph()
        lines = ["# Influence graph", ""]
        if not graph["blocks"]:
            lines.append("(no blocks)")
            lines.append("")
        for block in graph["blocks"]:
            lines.append(f"## {block['kind']} `{block['block_id']}`")
            lines.append(f"- steps: {block['step_refs']}")
            for key in block:
                if key in {"block_id", "kind", "step_refs"}:
                    continue
                lines.append(f"- {key}: {block[key]}")
            lines.append("")
        lines.append("## edges")
        if not graph["edges"]:
            lines.append("(none)")
        for edge in graph["edges"]:
            lines.append(
                f"- `{edge['from']}` -> `{edge['to']}` ({edge['link_kind']}): "
                f"inherited={edge['inherited']}; effect={edge['effect']}"
            )
        lines.append("")
        lines.append(f"submitted: {str(graph['submitted']).lower()}")
        return "\n".join(lines)

    def _get_graph(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        _exact_args(arguments, frozenset(), frozenset())
        return self._graph()

    def _graph(self) -> dict[str, Any]:
        blocks = []
        for block_id in _ordered_ids(self._blocks):
            block = self._blocks[block_id]
            payload: dict[str, Any] = {"block_id": block.block_id, "kind": block.kind}
            for key in _REQUIRED_FIELDS[block.kind]:
                value = block.fields[key]
                payload[key] = list(value) if key == "step_refs" else value
            if "defect_recorded" in block.fields:
                payload["defect_recorded"] = block.fields["defect_recorded"]
            blocks.append(payload)
        edges = [
            {
                "from": edge.source_id,
                "to": edge.target_id,
                "inherited": edge.inherited,
                "effect": edge.effect,
                "link_kind": edge.link_kind,
            }
            for edge in sorted(self._edges, key=lambda item: (item.source_id, item.target_id))
        ]
        return {"blocks": blocks, "edges": edges, "submitted": self._submitted}

    def _add_block(self, arguments: Mapping[str, Any]) -> dict[str, str]:
        self._require_open()
        _exact_args(
            arguments,
            frozenset({"kind", "block_id", "fields", "from_refs", "to_refs"}),
            frozenset({"kind", "fields"}),
        )
        stored = self._plan_add(self._blocks, arguments)
        new_blocks, new_edges = self._apply_add(self._blocks, self._edges, stored)
        self._commit(new_blocks, new_edges, "add_block", stored, False)
        return {"block_id": stored["block_id"]}

    def _plan_add(
        self, blocks: Mapping[str, _Block], arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        kind = arguments["kind"]
        if not isinstance(kind, str) or kind not in _PAPER_KIND_SET:
            raise ToolSurfaceError("invalid_kind")
        fields = _normalize_fields(kind, arguments["fields"], len(self._records))
        block_id = arguments.get("block_id", None)
        if block_id is None:
            block_id = self._allocate_id(blocks)
        elif not isinstance(block_id, str) or not is_safe_id(block_id):
            raise ToolSurfaceError("invalid_block_id")
        if block_id in blocks:
            raise ToolSurfaceError("duplicate_block_id")
        staged = dict(blocks)
        staged[block_id] = _Block(block_id, kind, fields)
        links: list[_Edge] = []
        for ref in _optional_refs(arguments, "from_refs"):
            other, inherited, effect = _ref_parts(ref)
            links.append(
                _checked_link(
                    staged, [*self._edges, *links], other, block_id, inherited, effect, None
                )
            )
        for ref in _optional_refs(arguments, "to_refs"):
            other, inherited, effect = _ref_parts(ref)
            links.append(
                _checked_link(
                    staged, [*self._edges, *links], block_id, other, inherited, effect, None
                )
            )
        return {
            "block_id": block_id,
            "kind": kind,
            "fields": _fields_json(fields),
            "links": [_edge_json(edge) for edge in links],
        }

    def _apply_add(
        self,
        blocks: Mapping[str, _Block],
        edges: Sequence[_Edge],
        stored: Mapping[str, Any],
    ) -> tuple[dict[str, _Block], list[_Edge]]:
        kind = stored.get("kind")
        block_id = stored.get("block_id")
        if not isinstance(kind, str) or not isinstance(block_id, str):
            raise ToolSurfaceError("stored_record_invalid")
        if not is_safe_id(block_id):
            raise ToolSurfaceError("invalid_block_id")
        if block_id in blocks:
            raise ToolSurfaceError("duplicate_block_id")
        fields = _normalize_fields(kind, stored.get("fields"), len(self._records))
        staged = dict(blocks)
        staged[block_id] = _Block(block_id, kind, fields)
        raw_links = stored.get("links")
        if not isinstance(raw_links, list):
            raise ToolSurfaceError("stored_record_invalid")
        new_edges = list(edges)
        for link in raw_links:
            if not isinstance(link, Mapping):
                raise ToolSurfaceError("stored_record_invalid")
            expected = link.get("link_kind")
            if expected not in {_LINK_KIND_ADJACENT, _LINK_KIND_SKIP}:
                raise ToolSurfaceError("stored_record_invalid")
            new_edges.append(
                _checked_link(
                    staged,
                    new_edges,
                    link.get("from"),
                    link.get("to"),
                    link.get("inherited"),
                    link.get("effect"),
                    expected,
                )
            )
        _validate(staged, new_edges, len(self._records))
        return staged, new_edges

    def _add_inheritance_note(self, arguments: Mapping[str, Any]) -> dict[str, str]:
        return self._add_note(arguments, "add_inheritance_note", _LINK_KIND_ADJACENT)

    def _add_skip_attribution(self, arguments: Mapping[str, Any]) -> dict[str, str]:
        return self._add_note(arguments, "add_skip_attribution", _LINK_KIND_SKIP)

    def _add_note(
        self, arguments: Mapping[str, Any], verb: str, expected: str
    ) -> dict[str, str]:
        self._require_open()
        _exact_args(
            arguments,
            frozenset({"from", "to", "inherited", "effect"}),
            frozenset({"from", "to", "inherited", "effect"}),
        )
        stored = {
            "from": arguments["from"],
            "to": arguments["to"],
            "inherited": arguments["inherited"],
            "effect": arguments["effect"],
            "link_kind": expected,
        }
        new_blocks, new_edges = self._apply_note(self._blocks, self._edges, stored, expected)
        self._commit(new_blocks, new_edges, verb, stored, False)
        return {"from": stored["from"], "to": stored["to"], "link_kind": expected}

    def _apply_note(
        self,
        blocks: Mapping[str, _Block],
        edges: Sequence[_Edge],
        stored: Mapping[str, Any],
        expected: str,
    ) -> tuple[dict[str, _Block], list[_Edge]]:
        if stored.get("link_kind") != expected:
            raise ToolSurfaceError("stored_record_invalid")
        edge = _checked_link(
            blocks,
            edges,
            stored.get("from"),
            stored.get("to"),
            stored.get("inherited"),
            stored.get("effect"),
            expected,
        )
        new_edges = [*edges, edge]
        copied = dict(blocks)
        _validate(copied, new_edges, len(self._records))
        return copied, new_edges

    def _modify_block(self, arguments: Mapping[str, Any]) -> dict[str, str]:
        self._require_open()
        _exact_args(arguments, frozenset({"block_id", "fields"}), frozenset({"block_id", "fields"}))
        block_id = arguments["block_id"]
        if not isinstance(block_id, str) or block_id not in self._blocks:
            raise ToolSurfaceError("unknown_block")
        patch = arguments["fields"]
        if not isinstance(patch, Mapping) or not patch:
            raise ToolSurfaceError("invalid_args")
        if "kind" in patch:
            raise ToolSurfaceError("kind_immutable")
        if "block_id" in patch or "id" in patch:
            raise ToolSurfaceError("id_immutable")
        block = self._blocks[block_id]
        merged = dict(block.fields)
        merged.update(patch)
        normalized = _normalize_fields(block.kind, merged, len(self._records))
        stored = {"block_id": block_id, "fields": _fields_json(normalized)}
        new_blocks, new_edges = self._apply_modify(self._blocks, self._edges, stored)
        self._commit(new_blocks, new_edges, "modify_block", stored, False)
        return {"block_id": block_id}

    def _apply_modify(
        self,
        blocks: Mapping[str, _Block],
        edges: Sequence[_Edge],
        stored: Mapping[str, Any],
    ) -> tuple[dict[str, _Block], list[_Edge]]:
        block_id = stored.get("block_id")
        if not isinstance(block_id, str) or block_id not in blocks:
            raise ToolSurfaceError("unknown_block")
        block = blocks[block_id]
        normalized = _normalize_fields(block.kind, stored.get("fields"), len(self._records))
        staged = dict(blocks)
        staged[block_id] = _Block(block_id, block.kind, normalized)
        new_edges = list(edges)
        _validate(staged, new_edges, len(self._records))
        return staged, new_edges

    def _delete_block(self, arguments: Mapping[str, Any]) -> dict[str, str]:
        self._require_open()
        _exact_args(arguments, frozenset({"block_id"}), frozenset({"block_id"}))
        block_id = arguments["block_id"]
        stored = {"block_id": block_id}
        new_blocks, new_edges = self._apply_delete_block(self._blocks, self._edges, stored)
        self._commit(new_blocks, new_edges, "delete_block", stored, False)
        return {"block_id": block_id}

    def _apply_delete_block(
        self,
        blocks: Mapping[str, _Block],
        edges: Sequence[_Edge],
        stored: Mapping[str, Any],
    ) -> tuple[dict[str, _Block], list[_Edge]]:
        block_id = stored.get("block_id")
        if not isinstance(block_id, str) or block_id not in blocks:
            raise ToolSurfaceError("unknown_block")
        staged = {key: value for key, value in blocks.items() if key != block_id}
        new_edges = [
            edge
            for edge in edges
            if edge.source_id != block_id and edge.target_id != block_id
        ]
        _validate(staged, new_edges, len(self._records))
        return staged, new_edges

    def _delete_inheritance_note(self, arguments: Mapping[str, Any]) -> dict[str, str]:
        return self._delete_note(arguments, "delete_inheritance_note", _LINK_KIND_ADJACENT)

    def _delete_skip_attribution(self, arguments: Mapping[str, Any]) -> dict[str, str]:
        return self._delete_note(arguments, "delete_skip_attribution", _LINK_KIND_SKIP)

    def _delete_note(
        self, arguments: Mapping[str, Any], verb: str, expected: str
    ) -> dict[str, str]:
        self._require_open()
        _exact_args(arguments, frozenset({"from", "to"}), frozenset({"from", "to"}))
        stored = {"from": arguments["from"], "to": arguments["to"]}
        new_blocks, new_edges = self._apply_delete_note(self._blocks, self._edges, stored, expected)
        self._commit(new_blocks, new_edges, verb, stored, False)
        return {"from": stored["from"], "to": stored["to"]}

    def _apply_delete_note(
        self,
        blocks: Mapping[str, _Block],
        edges: Sequence[_Edge],
        stored: Mapping[str, Any],
        expected: str,
    ) -> tuple[dict[str, _Block], list[_Edge]]:
        source = stored.get("from")
        target = stored.get("to")
        if not isinstance(source, str) or not isinstance(target, str):
            raise ToolSurfaceError("stored_record_invalid")
        matches = [
            edge for edge in edges if edge.source_id == source and edge.target_id == target
        ]
        if not matches:
            raise ToolSurfaceError("edge_missing")
        if matches[0].link_kind != expected:
            raise ToolSurfaceError("edge_kind_mismatch")
        new_edges = [
            edge
            for edge in edges
            if not (edge.source_id == source and edge.target_id == target)
        ]
        copied = dict(blocks)
        _validate(copied, new_edges, len(self._records))
        return copied, new_edges

    def _mark_defect_recorded(self, arguments: Mapping[str, Any]) -> dict[str, str]:
        self._require_open()
        _exact_args(arguments, frozenset({"block_id", "note"}), frozenset({"block_id", "note"}))
        stored = {"block_id": arguments["block_id"], "note": arguments["note"]}
        new_blocks, new_edges = self._apply_defect(self._blocks, self._edges, stored)
        self._commit(new_blocks, new_edges, "mark_defect_recorded", stored, False)
        return {"block_id": stored["block_id"], "defect_recorded": stored["note"]}

    def _apply_defect(
        self,
        blocks: Mapping[str, _Block],
        edges: Sequence[_Edge],
        stored: Mapping[str, Any],
    ) -> tuple[dict[str, _Block], list[_Edge]]:
        block_id = stored.get("block_id")
        note = stored.get("note")
        if not isinstance(block_id, str) or block_id not in blocks:
            raise ToolSurfaceError("unknown_block")
        if not isinstance(note, str) or not note.strip():
            raise ToolSurfaceError("missing_note")
        block = blocks[block_id]
        if block.kind != "agent":
            raise ToolSurfaceError("defect_not_agent")
        fields = dict(block.fields)
        fields["defect_recorded"] = note
        staged = dict(blocks)
        staged[block_id] = _Block(block_id, block.kind, fields)
        new_edges = list(edges)
        _validate(staged, new_edges, len(self._records))
        return staged, new_edges

    def _submit(self, arguments: Mapping[str, Any]) -> dict[str, bool]:
        self._require_open()
        _exact_args(arguments, frozenset(), frozenset())
        _validate(self._blocks, self._edges, len(self._records))
        self._commit(dict(self._blocks), list(self._edges), "submit", {}, True)
        return {"submitted": True}

    def _require_open(self) -> None:
        if self._submitted and not self._refine_open:
            raise ToolSurfaceError("already_submitted")

    def _allocate_id(self, blocks: Mapping[str, _Block]) -> str:
        seq = len(self._ops)
        candidate = f"b-{seq}"
        suffix = 0
        while candidate in blocks or not is_safe_id(candidate):
            suffix += 1
            candidate = f"b-{seq}-{suffix}"
        return candidate

    def _commit(
        self,
        blocks: dict[str, _Block],
        edges: list[_Edge],
        verb: str,
        args: Mapping[str, Any],
        submitted: bool,
    ) -> None:
        if verb not in TOOLLOOP_BUILD_VERBS:
            raise ToolSurfaceError("unknown_verb")
        if self._refine_open:
            if verb == "submit":
                raise ToolSurfaceError("submit_forbidden")
            self._blocks = blocks
            self._edges = edges
            return
        document = toolloop_draft_op_document(
            run_id=self._run_id,
            seq=len(self._ops),
            verb=verb,
            args=args,
            prev_content_hash=self._prev_hash,
        )
        append_jsonl(self._log_path, document)
        self._blocks = blocks
        self._edges = edges
        self._ops.append(document)
        self._prev_hash = str(document["content_hash"])
        self._submitted = submitted
        if submitted:
            self._seal()

    def _seal(self) -> None:
        """Make the run directory read-only. The submit op is the real seal."""
        for path in (self._meta_path, self._log_path):
            if path.is_file():
                try:
                    os.chmod(path, _READONLY_FILE)
                except OSError:
                    continue
        try:
            os.chmod(self._run_dir, _READONLY_DIR)
        except OSError:
            return


def _reject_outcome_mutation(arguments: Mapping[str, Any]) -> None:
    """Outcome is a frozen snapshot fact. No T_build verb may write it."""
    if "outcome" in arguments:
        raise ToolSurfaceError("outcome_readonly")
    fields = arguments.get("fields")
    if isinstance(fields, Mapping) and "outcome" in fields:
        raise ToolSurfaceError("outcome_readonly")


def _run_directory(root: Path, run_id: str) -> Path:
    if not isinstance(run_id, str) or not is_safe_id(run_id):
        raise ToolSurfaceError("invalid_run_id")
    base = (Path(root) / TOOLLOOP_RUNS_DIRNAME).resolve()
    path = (base / run_id).resolve()
    if path.parent != base:
        raise ToolSurfaceError("invalid_run_id")
    return path


def _meta_digest(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_dumps(dict(body)).encode("utf-8")).hexdigest()


def _one_of(arguments: Mapping[str, Any], name: str, alias: str) -> Any:
    has_name = name in arguments
    has_alias = alias in arguments
    if has_name and has_alias and arguments[name] != arguments[alias]:
        raise ToolSurfaceError("invalid_args")
    if has_name:
        return arguments[name]
    if has_alias:
        return arguments[alias]
    raise ToolSurfaceError("invalid_args")


def _row_index(value: Any, row_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolSurfaceError("invalid_args")
    if value < 0 or value >= row_count:
        raise ToolSurfaceError("step_out_of_range")
    return value


def _regex_flags(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        allowed = re.IGNORECASE | re.MULTILINE | re.DOTALL
        if value & ~allowed:
            raise ToolSurfaceError("invalid_flags")
        return value
    if isinstance(value, str):
        flags = 0
        mapping = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}
        for char in value:
            bit = mapping.get(char)
            if bit is None:
                raise ToolSurfaceError("invalid_flags")
            flags |= bit
        return flags
    raise ToolSurfaceError("invalid_flags")


def _optional_refs(arguments: Mapping[str, Any], key: str) -> list[Any]:
    if key not in arguments:
        return []
    value = arguments[key]
    if not isinstance(value, list):
        raise ToolSurfaceError("invalid_args")
    return value


def _ref_parts(value: Any) -> tuple[str, str, str]:
    if not isinstance(value, Mapping):
        raise ToolSurfaceError("invalid_args")
    if set(value) != {"block_id", "inherited", "effect"}:
        raise ToolSurfaceError("invalid_args")
    block_id = value["block_id"]
    if not isinstance(block_id, str):
        raise ToolSurfaceError("invalid_args")
    return block_id, value["inherited"], value["effect"]


def _note_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolSurfaceError("missing_note")
    return value


def _checked_link(
    blocks: Mapping[str, _Block],
    edges: Sequence[_Edge],
    source: Any,
    target: Any,
    inherited: Any,
    effect: Any,
    expected: str | None,
) -> _Edge:
    if not isinstance(source, str) or not isinstance(target, str):
        raise ToolSurfaceError("invalid_args")
    if source not in blocks or target not in blocks:
        raise ToolSurfaceError("unknown_block")
    inherited_text = _note_text(inherited)
    effect_text = _note_text(effect)
    source_min = blocks[source].fields["step_refs"][0]
    target_min = blocks[target].fields["step_refs"][0]
    if source_min >= target_min:
        raise ToolSurfaceError("edge_not_forward")
    if any(edge.source_id == source and edge.target_id == target for edge in edges):
        raise ToolSurfaceError("duplicate_edge")
    kind = _LINK_KIND_ADJACENT if _is_adjacent(blocks, source, target) else _LINK_KIND_SKIP
    if expected == _LINK_KIND_ADJACENT and kind != _LINK_KIND_ADJACENT:
        raise ToolSurfaceError("not_adjacent")
    if expected == _LINK_KIND_SKIP and kind != _LINK_KIND_SKIP:
        raise ToolSurfaceError("not_skip")
    if expected is not None and expected not in {_LINK_KIND_ADJACENT, _LINK_KIND_SKIP}:
        raise ToolSurfaceError("stored_record_invalid")
    return _Edge(source, target, inherited_text, effect_text, expected or kind)


def _edge_json(edge: _Edge) -> dict[str, str]:
    return {
        "from": edge.source_id,
        "to": edge.target_id,
        "inherited": edge.inherited,
        "effect": edge.effect,
        "link_kind": edge.link_kind,
    }
