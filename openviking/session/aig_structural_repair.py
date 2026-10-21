# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Deterministic structural repair of a submitted paper draft.

``Phi(G)`` is purely structural: one task root, one conclusion sink, every
node on a forward path from the root. The builder model regularly misses
exactly these properties (a missing conclusion, a conclusion placed before the
last agent step, blocks left without links), and a small critic/refiner model
then stalls on them. This module closes those gaps with rule-based edits
before the first audit, so the VLM critic/refiner only ever faces semantic
residue.

Every edit goes through ``AigToolSurface.apply_refine``: hard invariants
(``step_refs`` overlap, forward-only edges, participant uniformity) still fail
closed, and a rejected edit leaves the draft unchanged. The function never
raises on a rejected edit; it returns the edits that were applied and the ones
that were refused so the caller can append them to the audit trajectory.

Repairs, in order:

1. Duplicate or misplaced conclusions are removed (a conclusion must be the
   latest block, because every edge points forward in time).
2. A missing conclusion is added over uncovered tail rows when a uniform run of
   them exists, otherwise the latest agent/orchestrator block is rebuilt as the
   conclusion with its recorded content.
3. A missing task root is rebuilt from the earliest block.
4. Blocks are chained in time order wherever a node has no incoming path from
   the root or no outgoing edge. Existing edges are never removed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from openviking.session.aig_tool_surface import AigToolSurface, AoSnapshot, ToolSurfaceError

_TEXT_LIMIT = 240
_FALLBACK_TEXT = "(no recorded content)"


def repair_structure(
    surface: AigToolSurface,
    snapshot: AoSnapshot,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(applied, refused)`` edits. Never raises on a rejected edit."""
    applied: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    if not isinstance(surface, AigToolSurface) or not surface.submitted:
        return applied, refused
    ctx = _Repair(surface, snapshot, applied, refused)
    try:
        ctx.normalize_conclusion()
        ctx.normalize_task()
        ctx.chain_blocks()
    except ToolSurfaceError as exc:  # pragma: no cover - guarded per edit already
        refused.append({"verb": "repair", "code": exc.code})
    return applied, refused


class _Repair:
    def __init__(
        self,
        surface: AigToolSurface,
        snapshot: AoSnapshot,
        applied: list[dict[str, Any]],
        refused: list[dict[str, Any]],
    ) -> None:
        self._surface = surface
        self._snapshot = snapshot
        self._applied = applied
        self._refused = refused
        self._row_count = len(snapshot.records)

    # -- helpers -----------------------------------------------------------

    def _apply(self, verb: str, args: Mapping[str, Any]) -> dict[str, Any] | None:
        try:
            result = self._surface.apply_refine(verb, dict(args))
        except ToolSurfaceError as exc:
            self._refused.append({"verb": verb, "args": _brief(args), "code": exc.code})
            return None
        self._applied.append({"verb": verb, "args": _brief(args)})
        return result if isinstance(result, dict) else {}

    def _graph(self) -> dict[str, Any]:
        return self._surface.get_graph()

    def _blocks(self) -> list[dict[str, Any]]:
        return list(self._graph()["blocks"])

    # -- 1/2: conclusion ---------------------------------------------------

    def normalize_conclusion(self) -> None:
        blocks = self._blocks()
        if not blocks:
            return
        conclusions = [item for item in blocks if item["kind"] == "conclusion"]
        last = blocks[-1]
        for item in conclusions:
            if item["block_id"] != last["block_id"]:
                self._apply("delete_block", {"block_id": item["block_id"]})
        blocks = self._blocks()
        if not blocks:
            return
        if any(item["kind"] == "conclusion" for item in blocks):
            return
        if self._add_tail_conclusion(blocks):
            return
        self._convert_last_to_conclusion(blocks)

    def _add_tail_conclusion(self, blocks: Sequence[Mapping[str, Any]]) -> bool:
        covered = {row for item in blocks for row in item["step_refs"]}
        tail_start = max(covered) + 1
        free = [row for row in range(tail_start, self._row_count)]
        if not free:
            return False
        for start in free:
            rows = list(range(start, self._row_count))
            content = self._rows_text(rows)
            result = self._apply(
                "add_block",
                {"kind": "conclusion", "fields": {"content": content, "step_refs": rows}},
            )
            if result is not None:
                return True
        return False

    def _convert_last_to_conclusion(self, blocks: Sequence[Mapping[str, Any]]) -> None:
        if len(blocks) < 2:
            return
        last = blocks[-1]
        if last["kind"] not in {"agent", "orchestrator"}:
            return
        refs = list(last["step_refs"])
        content = _conclusion_text(last)
        if self._apply("delete_block", {"block_id": last["block_id"]}) is None:
            return
        added = self._apply(
            "add_block",
            {"kind": "conclusion", "fields": {"content": content, "step_refs": refs}},
        )
        if added is None:
            # Restore the removed block so a refused conversion changes nothing.
            self._restore(last)

    def _restore(self, block: Mapping[str, Any]) -> None:
        fields = {
            key: value
            for key, value in block.items()
            if key not in {"block_id", "kind"}
        }
        self._apply(
            "add_block",
            {"kind": block["kind"], "block_id": block["block_id"], "fields": fields},
        )

    # -- 3: task root ------------------------------------------------------

    def normalize_task(self) -> None:
        blocks = self._blocks()
        if not blocks or any(item["kind"] == "task" for item in blocks):
            return
        first = blocks[0]
        if first["kind"] not in {"agent", "orchestrator"} or len(blocks) < 2:
            return
        refs = list(first["step_refs"])
        question = self._snapshot.question if isinstance(self._snapshot.question, str) else ""
        text = _conclusion_text(first)
        content = question.strip() or text
        if self._apply("delete_block", {"block_id": first["block_id"]}) is None:
            return
        added = self._apply(
            "add_block",
            {
                "kind": "task",
                "fields": {
                    "content": content,
                    "operative_brief": _clip(text),
                    "step_refs": refs,
                },
            },
        )
        if added is None:
            self._restore(first)

    # -- 4: edges ----------------------------------------------------------

    def chain_blocks(self) -> None:
        blocks = self._blocks()
        if len(blocks) < 2:
            return
        for index in range(1, len(blocks)):
            blocks = self._blocks()
            graph = self._graph()
            root = blocks[0]["block_id"]
            reached = _reachable(root, graph["edges"])
            current = blocks[index]
            if current["block_id"] in reached or current["kind"] == "task":
                continue
            self._link(blocks[index - 1], current)
        for index in range(len(blocks) - 1):
            blocks = self._blocks()
            graph = self._graph()
            source = blocks[index]
            if source["kind"] == "conclusion":
                continue
            if any(edge["from"] == source["block_id"] for edge in graph["edges"]):
                continue
            self._link(source, blocks[index + 1])

    def _link(self, source: Mapping[str, Any], target: Mapping[str, Any]) -> None:
        if target["kind"] == "task":
            return
        args = {
            "from": source["block_id"],
            "to": target["block_id"],
            "inherited": _clip(_carry_text(source)),
            "effect": _clip(_effect_text(target)),
        }
        if self._apply("add_inheritance_note", args) is not None:
            return
        self._apply("add_skip_attribution", args)

    def _rows_text(self, rows: Sequence[int]) -> str:
        parts: list[str] = []
        for row in rows:
            try:
                step = self._surface.read_step(row)
            except ToolSurfaceError:
                continue
            record = step.get("record", {})
            observation = record.get("observation")
            text = observation.get("text") if isinstance(observation, Mapping) else None
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return _clip(" ".join(parts), 1000) if parts else _FALLBACK_TEXT


def _reachable(root: str, edges: Sequence[Mapping[str, Any]]) -> set[str]:
    graph: dict[str, set[str]] = {}
    for edge in edges:
        graph.setdefault(edge["from"], set()).add(edge["to"])
    seen: set[str] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph.get(node, set()) - seen)
    return seen


def _clip(text: str, limit: int = _TEXT_LIMIT) -> str:
    text = " ".join(str(text).split())
    if not text:
        return _FALLBACK_TEXT
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _first_text(block: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _carry_text(block: Mapping[str, Any]) -> str:
    return _first_text(block, ("output", "summary", "content", "operative_brief")) or (
        f"{block.get('kind', 'block')} {block.get('block_id', '')}"
    )


def _effect_text(block: Mapping[str, Any]) -> str:
    return _first_text(block, ("summary", "content", "input", "operative_brief")) or (
        f"{block.get('kind', 'block')} {block.get('block_id', '')}"
    )


def _conclusion_text(block: Mapping[str, Any]) -> str:
    summary = _first_text(block, ("summary", "content", "operative_brief"))
    output = _first_text(block, ("output",))
    if summary and output and output.strip() != summary.strip():
        return _clip(f"{summary} Output: {output}", 1000)
    return _clip(summary or output, 1000)


def _brief(args: Mapping[str, Any]) -> dict[str, Any]:
    brief: dict[str, Any] = {}
    for key, value in args.items():
        if key == "fields" and isinstance(value, Mapping):
            brief[key] = {
                name: (_clip(item, 80) if isinstance(item, str) else item)
                for name, item in value.items()
            }
        elif isinstance(value, str):
            brief[key] = _clip(value, 80)
        else:
            brief[key] = value
    return brief
