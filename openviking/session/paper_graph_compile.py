# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Deterministic compile of a paper draft into a native-edge revision.

No model is called. ``step_refs`` become a contiguous AO interval,
``agent_name`` becomes ``participant_id``, and ``<inherited, effect>`` becomes
the edge mechanism. Adjacent notes map to ``influence``; skip notes map to
``skip``. ``authorship`` and ``defect_recorded`` are the optional paper fields.
The full paper graph is a sidecar whose content hash is stored on the revision.
Retrieval does not read that sidecar or ``known_violations``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openviking.session.aig_native_protocol import (
    PAPER_AUTHORSHIP_VALUES,
    SCHEMA_VERSION_AIG_NATIVE_V2,
    ProtocolContractError,
    canonical_dumps,
    is_safe_id,
    paper_authorship,
    protocol_content_hash,
)
from openviking.session.aig_tool_surface import AoSnapshot
from openviking.session.append_only import write_exclusive
from openviking.session.ao_ledger import AORecord
from openviking.session.extraction_write import freeze_native_edge_view
from openviking.session.influence_view import (
    CONSTRUCTION_MODE_ADAPTIVE_AGENTIC,
    RELATION_INFLUENCE,
    RELATION_SKIP,
    AORange,
    AigNativeEdge,
    AigNativeViewRevision,
    CoverageManifest,
    HighSeverityViolation,
    InfluenceBlock,
    OmittedRange,
    SegmentRef,
    SourceRef,
)
from openviking.session.paper_graph_phi import (
    CONNECTIVITY_INTERPRETATION,
    CONNECTIVITY_POLICY_VERSION,
    VALIDATOR_PROFILE,
)

SIDECAR_KIND = "paper_graph_sidecar"
AUDIT_DIRNAME = "toolloop-audit"
PAPER_GRANULARITY = "paper_graph_compile"
MECHANISM_MARK = "\n<<effect>>\n"
_KIND_ROLE = {
    "task": "task",
    "orchestrator": "orchestrator",
    "agent": "agent_action",
    "conclusion": "conclusion",
}
_KIND_BLOCK = {
    "task": "task",
    "orchestrator": "orchestrator",
    "agent": "agent",
    "conclusion": "conclusion",
}
_PARTICIPANT_KEYS = ("participant_id", "participant", "agent_name", "agent")
_SEGMENT_KEYS = ("segment_id", "segment")


class PaperCompileError(ValueError):
    """The paper graph could not be frozen. ``code`` is stable."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class PaperCompileResult:
    """Frozen native revision plus the bound paper-graph sidecar."""

    revision: AigNativeViewRevision
    sidecar_path: Path
    sidecar_hash: str
    id_map: dict[str, str]


def join_mechanism(inherited: str, effect: str) -> str:
    """Splice a paper note into the native mechanism string."""
    return f"{inherited}{MECHANISM_MARK}{effect}"


def split_mechanism(mechanism: str) -> tuple[str, str]:
    """Inverse of ``join_mechanism``."""
    if MECHANISM_MARK not in mechanism:
        raise PaperCompileError("mechanism_unsplit")
    inherited, effect = mechanism.split(MECHANISM_MARK, 1)
    return inherited, effect


def compile_paper_graph(
    graph: Mapping[str, Any],
    snapshot: AoSnapshot,
    *,
    root: Path,
    run_id: str,
    session_id: str,
    task_run_id: str,
    view_id: str,
    revision_id: str,
    frozen_at: str,
    known_violations: Sequence[Mapping[str, Any]] = (),
) -> PaperCompileResult:
    """Map one paper draft onto a native revision and write its sidecar."""
    if not isinstance(snapshot, AoSnapshot):
        raise PaperCompileError("invalid_snapshot")
    _require_safe(run_id, "invalid_run_id")
    _require_safe(revision_id, "invalid_revision_id")
    if not snapshot.records:
        raise PaperCompileError("invalid_snapshot")
    for record in snapshot.records:
        if record.session_id != session_id:
            raise PaperCompileError("session_mismatch")
    blocks, edges, id_map = _native_graph(graph, snapshot, session_id)
    coverage = _coverage(snapshot, blocks)
    sidecar_body = {
        "connectivity_interpretation": CONNECTIVITY_INTERPRETATION,
        "connectivity_policy_version": CONNECTIVITY_POLICY_VERSION,
        "edges": list(graph.get("edges") or []),
        "id_map": dict(sorted(id_map.items())),
        "nodes": list(graph.get("blocks") or []),
        "revision_id": revision_id,
        "run_id": run_id,
        "validator_profile": VALIDATOR_PROFILE,
    }
    sidecar_hash = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, SIDECAR_KIND, sidecar_body
    )
    sidecar_path = _sidecar_path(root, run_id, revision_id)
    document = {
        "body": sidecar_body,
        "content_hash": sidecar_hash,
        "object_kind": SIDECAR_KIND,
        "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
    }
    try:
        write_exclusive(sidecar_path, document)
    except FileExistsError as exc:
        raise PaperCompileError("sidecar_exists") from exc
    try:
        revision = freeze_native_edge_view(
            session_id=session_id,
            task_run_id=task_run_id,
            view_id=view_id,
            revision_id=revision_id,
            frozen_at=frozen_at,
            blocks=blocks,
            edges=edges,
            coverage=coverage,
            construction_mode=CONSTRUCTION_MODE_ADAPTIVE_AGENTIC,
            known_violations=known_violations,
            paper_sidecar_hash=sidecar_hash,
        )
    except HighSeverityViolation as exc:
        raise PaperCompileError("freeze_refused") from exc
    except (TypeError, ValueError) as exc:
        raise PaperCompileError("freeze_refused") from exc
    return PaperCompileResult(
        revision=revision,
        sidecar_path=sidecar_path,
        sidecar_hash=sidecar_hash,
        id_map=dict(id_map),
    )


def render_paper_graph(revision: AigNativeViewRevision, snapshot: AoSnapshot) -> dict[str, Any]:
    """Render a frozen revision back into paper nodes and notes."""
    if not isinstance(revision, AigNativeViewRevision) or not isinstance(snapshot, AoSnapshot):
        raise PaperCompileError("invalid_snapshot")
    index_by_ao = {record.ao_id: index for index, record in enumerate(snapshot.records)}
    blocks: list[dict[str, Any]] = []
    for block in sorted(revision.blocks, key=lambda item: (item.ao_start_seq, item.block_id)):
        steps: list[int] = []
        for ref in block.source_refs:
            if ref.ao_id not in index_by_ao:
                raise PaperCompileError("snapshot_mismatch")
            steps.append(index_by_ao[ref.ao_id])
        node: dict[str, Any] = {
            "agent_name": block.participant_id,
            "authorship": block.authorship,
            "block_id": block.block_id,
            "input": block.input,
            "kind": block.block_kind,
            "output": block.output,
            "step_refs": steps,
            "summary": block.summary,
        }
        if block.defect_recorded is not None:
            node["defect_recorded"] = block.defect_recorded
        blocks.append(node)
    edges: list[dict[str, str]] = []
    ordered = sorted(revision.edges, key=lambda item: (item.source_block_id, item.target_block_id))
    for edge in ordered:
        inherited, effect = split_mechanism(edge.mechanism)
        link_kind = "adjacent" if edge.relation_type == RELATION_INFLUENCE else "skip"
        edges.append(
            {
                "effect": effect,
                "from": edge.source_block_id,
                "inherited": inherited,
                "link_kind": link_kind,
                "to": edge.target_block_id,
            }
        )
    return {"blocks": blocks, "edges": edges}


def sidecar_digest(body: Mapping[str, Any]) -> str:
    """The hash stored on the revision. Callers recompute it from the file body."""
    return protocol_content_hash(SCHEMA_VERSION_AIG_NATIVE_V2, SIDECAR_KIND, body)


def _native_graph(
    graph: Mapping[str, Any], snapshot: AoSnapshot, session_id: str
) -> tuple[tuple[InfluenceBlock, ...], tuple[AigNativeEdge, ...], dict[str, str]]:
    raw_blocks = graph.get("blocks")
    raw_edges = graph.get("edges")
    if not isinstance(raw_blocks, list) or not raw_blocks or not isinstance(raw_edges, list):
        raise PaperCompileError("invalid_graph")
    id_map: dict[str, str] = {}
    native_blocks: list[InfluenceBlock] = []
    spans: list[tuple[int, int]] = []
    for index, item in enumerate(raw_blocks):
        if not isinstance(item, Mapping):
            raise PaperCompileError("invalid_graph")
        paper_id = item.get("block_id")
        if not isinstance(paper_id, str) or not paper_id or paper_id in id_map:
            raise PaperCompileError("invalid_graph")
        native_id = paper_id if is_safe_id(paper_id) else _fallback_id(index, id_map)
        id_map[paper_id] = native_id
        block = _block(item, native_id, snapshot, session_id)
        native_blocks.append(block)
        spans.append((block.ao_start_seq, block.ao_end_seq))
    spans.sort()
    for index, (start, end) in enumerate(spans):
        if index and start <= spans[index - 1][1]:
            raise PaperCompileError("step_refs_overlap")
    native_edges: list[AigNativeEdge] = []
    seen: set[tuple[str, str]] = set()
    order = _paper_order(raw_blocks)
    for item in raw_edges:
        if not isinstance(item, Mapping):
            raise PaperCompileError("invalid_graph")
        source = item.get("from")
        target = item.get("to")
        if not isinstance(source, str) or not isinstance(target, str):
            raise PaperCompileError("invalid_graph")
        if source not in id_map or target not in id_map:
            raise PaperCompileError("unknown_block")
        pair = (id_map[source], id_map[target])
        if pair in seen:
            raise PaperCompileError("duplicate_edge")
        seen.add(pair)
        native_edges.append(_edge(item, pair, order, native_blocks))
    return tuple(native_blocks), tuple(native_edges), id_map


def _block(
    item: Mapping[str, Any], native_id: str, snapshot: AoSnapshot, session_id: str
) -> InfluenceBlock:
    kind = item.get("kind")
    if kind not in _KIND_ROLE:
        raise PaperCompileError("invalid_kind")
    refs = _step_refs(item.get("step_refs"), len(snapshot.records))
    records = [snapshot.records[index] for index in refs]
    sequences = [record.sequence for record in records]
    if sequences != list(range(sequences[0], sequences[-1] + 1)):
        raise PaperCompileError("step_refs_not_contiguous")
    participant = _participant_for(kind, item, records)
    segment = _shared(records, _row_segment, "segment_mismatch")
    session = _shared(records, lambda record: record.session_id, "session_mismatch")
    if session != session_id:
        raise PaperCompileError("session_mismatch")
    summary, model_input, output, authorship = _prose(kind, item)
    defect = _defect(kind, item)
    source_refs = tuple(
        SourceRef(
            ao_id=record.ao_id,
            sequence=record.sequence,
            participant_id=participant,
            segment_id=segment,
            session_id=session,
            content_hash=_row_hash(record),
        )
        for record in sorted(records, key=lambda record: record.sequence)
    )
    try:
        return InfluenceBlock(
            block_id=native_id,
            role=_KIND_ROLE[kind],  # type: ignore[arg-type]
            participant_id=participant,
            segment_ref=SegmentRef(segment_id=segment, session_id=session),
            ao_start_seq=sequences[0],
            ao_end_seq=sequences[-1],
            summary=summary,
            input=model_input,
            output=output,
            authorship=authorship,
            granularity_reason=PAPER_GRANULARITY,
            source_content_hashes=tuple(ref.content_hash for ref in source_refs),
            source_refs=source_refs,
            defect_recorded=defect,
            block_kind=_KIND_BLOCK[kind],
        )
    except (TypeError, ValueError) as exc:
        raise PaperCompileError("invalid_block") from exc


def _edge(
    item: Mapping[str, Any],
    pair: tuple[str, str],
    order: list[str],
    blocks: Sequence[InfluenceBlock],
) -> AigNativeEdge:
    inherited = item.get("inherited")
    effect = item.get("effect")
    if not isinstance(inherited, str) or not inherited.strip():
        raise PaperCompileError("missing_note")
    if not isinstance(effect, str) or not effect.strip():
        raise PaperCompileError("missing_note")
    if MECHANISM_MARK in inherited or MECHANISM_MARK in effect:
        raise PaperCompileError("mechanism_mark_reserved")
    link_kind = item.get("link_kind")
    if link_kind is None:
        link_kind = _derived_link(item.get("from"), item.get("to"), order)
    if link_kind == "adjacent":
        relation = RELATION_INFLUENCE
    elif link_kind == "skip":
        relation = RELATION_SKIP
    else:
        raise PaperCompileError("invalid_link_kind")
    by_id = {block.block_id: block for block in blocks}
    source = by_id[pair[0]]
    target = by_id[pair[1]]
    if source.ao_end_seq >= target.ao_start_seq:
        raise PaperCompileError("edge_not_forward")
    edge_id = f"e-{pair[0]}-{pair[1]}"
    if not is_safe_id(edge_id):
        edge_id = "e-" + hashlib.sha256(edge_id.encode("utf-8")).hexdigest()[:16]
    return AigNativeEdge(
        edge_id=edge_id,
        source_block_id=pair[0],
        target_block_id=pair[1],
        relation_type=relation,
        mechanism=join_mechanism(inherited, effect),
    )


def _prose(kind: str, item: Mapping[str, Any]) -> tuple[str, str, str, str]:
    if kind == "agent":
        return (
            _text(item, "summary"),
            _text(item, "input"),
            _text(item, "output"),
            _authorship(item.get("authorship", "")),
        )
    if kind == "orchestrator":
        return (_text(item, "summary"), "", "", _authorship(item.get("authorship", "")))
    if kind == "task":
        return (_text(item, "content"), _text(item, "operative_brief"), "", "")
    return (_text(item, "content"), "", _text(item, "content"), "")


def _authorship(value: Any) -> str:
    try:
        text = paper_authorship(value)
    except ProtocolContractError as exc:
        raise PaperCompileError("invalid_authorship") from exc
    if text and text not in PAPER_AUTHORSHIP_VALUES:
        raise PaperCompileError("invalid_authorship")
    return text


def _defect(kind: str, item: Mapping[str, Any]) -> str | None:
    if "defect_recorded" not in item:
        return None
    if kind != "agent":
        raise PaperCompileError("defect_not_agent")
    note = item.get("defect_recorded")
    if not isinstance(note, str) or not note.strip():
        raise PaperCompileError("invalid_defect_recorded")
    return note


def _participant_for(kind: str, item: Mapping[str, Any], records: Sequence[AORecord]) -> str:
    named = item.get("agent_name")
    row_participant = _shared(records, _row_participant, "participant_mismatch")
    if isinstance(named, str) and named.strip():
        if named != row_participant:
            raise PaperCompileError("participant_mismatch")
        return named
    if kind in {"agent", "orchestrator"}:
        raise PaperCompileError("missing_field")
    if not row_participant:
        raise PaperCompileError("participant_missing")
    return row_participant


def _coverage(snapshot: AoSnapshot, blocks: Sequence[InfluenceBlock]) -> CoverageManifest:
    inspected: list[SourceRef] = []
    segments: set[str] = set()
    for record in snapshot.records:
        segment = _row_segment(record) or "unspecified"
        participant = _row_participant(record) or "unspecified"
        segments.add(segment)
        inspected.append(
            SourceRef(
                ao_id=record.ao_id,
                sequence=record.sequence,
                participant_id=participant,
                segment_id=segment,
                session_id=record.session_id,
                content_hash=_row_hash(record),
            )
        )
    sequences = sorted(record.sequence for record in snapshot.records)
    covered: set[int] = set()
    included: list[AORange] = []
    for block in blocks:
        included.append(
            AORange(
                start_seq=block.ao_start_seq,
                end_seq=block.ao_end_seq,
                segment_id=block.segment_ref.segment_id,
            )
        )
        covered.update(range(block.ao_start_seq, block.ao_end_seq + 1))
    omitted = _omitted(sequences[0], sequences[-1], covered, segments)
    segment_id = next(iter(segments)) if len(segments) == 1 else None
    return CoverageManifest(
        total_ao_range=AORange(
            start_seq=sequences[0], end_seq=sequences[-1], segment_id=segment_id
        ),
        inspected_refs=tuple(inspected),
        included_ranges=tuple(included),
        omitted_ranges=omitted,
        snapshot_watermark=snapshot.watermark,
    )


def _omitted(
    start: int, end: int, covered: set[int], segments: set[str]
) -> tuple[OmittedRange, ...]:
    missing = [seq for seq in range(start, end + 1) if seq not in covered]
    if not missing:
        return ()
    segment_id = next(iter(segments)) if len(segments) == 1 else None
    ranges: list[OmittedRange] = []
    run_start = missing[0]
    previous = missing[0]
    for seq in missing[1:]:
        if seq == previous + 1:
            previous = seq
            continue
        ranges.append(
            OmittedRange(
                start_seq=run_start,
                end_seq=previous,
                reason="not_in_paper_graph",
                segment_id=segment_id,
            )
        )
        run_start = seq
        previous = seq
    ranges.append(
        OmittedRange(
            start_seq=run_start,
            end_seq=previous,
            reason="not_in_paper_graph",
            segment_id=segment_id,
        )
    )
    return tuple(ranges)


def _paper_order(blocks: Sequence[Mapping[str, Any]]) -> list[str]:
    def key(item: Mapping[str, Any]) -> tuple[int, str]:
        refs = item.get("step_refs")
        start = refs[0] if isinstance(refs, list) and refs else 0
        block_id = item.get("block_id")
        return (
            start if isinstance(start, int) else 0,
            block_id if isinstance(block_id, str) else "",
        )

    return [str(item.get("block_id")) for item in sorted(blocks, key=key)]


def _derived_link(source: Any, target: Any, order: Sequence[str]) -> str:
    if source not in order or target not in order:
        raise PaperCompileError("unknown_block")
    if order.index(target) == order.index(source) + 1:
        return "adjacent"
    return "skip"


def _step_refs(value: Any, row_count: int) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise PaperCompileError("invalid_step_refs")
    numbers: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise PaperCompileError("invalid_step_refs")
        if item < 0 or item >= row_count:
            raise PaperCompileError("step_out_of_range")
        numbers.append(item)
    ordered = tuple(sorted(numbers))
    if len(set(ordered)) != len(ordered):
        raise PaperCompileError("step_refs_not_contiguous")
    if ordered[-1] - ordered[0] + 1 != len(ordered):
        raise PaperCompileError("step_refs_not_contiguous")
    return ordered


def _text(item: Mapping[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PaperCompileError("missing_field")
    return value


def _shared(records: Sequence[AORecord], reader: Any, code: str) -> str:
    values = {reader(record) for record in records}
    if len(values) != 1:
        raise PaperCompileError(code)
    value = next(iter(values))
    if not isinstance(value, str) or not value:
        raise PaperCompileError(code)
    return value


def _row_participant(record: AORecord) -> str:
    return _lookup(record, _PARTICIPANT_KEYS)


def _row_segment(record: AORecord) -> str:
    return _lookup(record, _SEGMENT_KEYS)


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


def _row_hash(record: AORecord) -> str:
    return hashlib.sha256(canonical_dumps(record.to_dict()).encode("utf-8")).hexdigest()


def _fallback_id(index: int, taken: Mapping[str, str]) -> str:
    candidate = f"n-{index}"
    suffix = 0
    used = set(taken.values())
    while candidate in used or not is_safe_id(candidate):
        suffix += 1
        candidate = f"n-{index}-{suffix}"
    return candidate


def _require_safe(value: str, code: str) -> None:
    if not isinstance(value, str) or not is_safe_id(value):
        raise PaperCompileError(code)


def _sidecar_path(root: Path, run_id: str, revision_id: str) -> Path:
    base = (Path(root) / AUDIT_DIRNAME).resolve()
    run_dir = (base / run_id).resolve()
    if run_dir.parent != base:
        raise PaperCompileError("invalid_run_id")
    return run_dir / f"{revision_id}.paper-graph.json"
