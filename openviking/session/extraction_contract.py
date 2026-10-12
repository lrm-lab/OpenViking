# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Model extraction contract: ledger projection in, ``ExtractionPayload`` out.

This module does not read config, construct a client, or import ``ov.conf``.
The only model call is ``model_extraction_provider``, and only through a
handle the caller already holds. Parsing is fail-closed: one reconciliation
miss raises ``ExtractionContractError`` and nothing is repaired, partially
accepted, or degraded.

A hit on any frozen limit raises ``ExtractionLimitError``. Values are not
silently clipped to fit.

``parse_native_edge_response`` is the expand-path schema. It emits blocks and
native edges and refuses Claim, status, evidence, and grounding fields. The
v1 parser is unchanged and still requires claims.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from openviking.session.extraction_write import (
    ExtractionPayload,
    ExtractionProvider,
    NativeExtractionPayload,
)
from openviking.session.influence_view import (
    BLOCK_KIND_ROLES,
    BLOCK_KINDS,
    BLOCK_ROLES,
    CLAIM_STATUSES,
    DETERMINISTIC_FALLBACK_REASON,
    MAX_RETRIEVAL_SUBGRAPH_BLOCKS,
    RELATION_INFLUENCE,
    RELATION_SKIP,
    SYNTHETIC_ROOT_SINK_REASON,
    AORange,
    AigNativeEdge,
    CoverageManifest,
    InfluenceBlock,
    InfluenceClaim,
    InfluenceEvidenceRef,
    OmittedRange,
    RetrievalSubgraph,
    SegmentRef,
    SourceRef,
)

# Frozen before any model call or score. Do not edit these to absorb a miss.
EXTRACTION_CONTRACT_VERSION = "extraction-contract-v1"
EXTRACTION_CONTRACT_NATIVE_VERSION = "extraction-contract-aig-native-v2"
EVIDENCE_ROLE = "contemporaneous_basis"
VALUE_CLIP = 4000
MAX_ITEMS = 64
MAX_TOKENS_PER_CALL = 32768
MAX_TOTAL_TOKENS = 2_000_000
MAX_REPAIRS = 1
MAX_INPUT_CHARS = 1_048_576
# Parser-assigned reason for a snapshot sequence outside every declared block.
OMITTED_RANGE_REASON = "record_type_not_a_block"

RELATION_TYPES = frozenset((RELATION_INFLUENCE, RELATION_SKIP))
_RESERVED_GRANULARITY = frozenset((DETERMINISTIC_FALLBACK_REASON, SYNTHETIC_ROOT_SINK_REASON))
_IDENTITY_FIELDS = ("session_id", "view_id", "task_run_id", "revision_id")
_TOP_KEYS = frozenset({"schema_version", "blocks", "claims", "subgraphs", "coverage"})
_BLOCK_KEYS = frozenset(
    {
        "block_id",
        "role",
        "block_kind",
        "participant_id",
        "segment_id",
        "ao_start_seq",
        "ao_end_seq",
        "summary",
        "input",
        "output",
        "authorship",
        "granularity_reason",
    }
)
_CLAIM_KEYS = frozenset(
    {
        "claim_id",
        "source_block_id",
        "target_block_id",
        "carried_artifact",
        "downstream_effect",
        "relation_type",
        "status",
        "source_evidence",
        "target_evidence",
    }
)
_EVIDENCE_KEYS = frozenset({"ao_id", "evidence_content_hash"})
_SUBGRAPH_KEYS = frozenset(
    {"subgraph_id", "block_ids", "claim_ids", "failure_marked", "failure_evidence"}
)
_COVERAGE_KEYS = frozenset(
    {
        "total_ao_start_seq",
        "total_ao_end_seq",
        "inspected_ao_ids",
        "snapshot_watermark",
    }
)
_NATIVE_TOP_KEYS = frozenset({"schema_version", "blocks", "edges", "coverage"})
_EDGE_KEYS = frozenset(
    {
        "edge_id",
        "source_block_id",
        "target_block_id",
        "relation_type",
        "mechanism",
    }
)
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

ExtractionProviderFactory = Callable[[Mapping[str, Any]], ExtractionProvider]
NativeExtractionProvider = Callable[[str, str, str], NativeExtractionPayload]
NativeExtractionProviderFactory = Callable[[Mapping[str, Any]], NativeExtractionProvider]

_ANNOUNCED = False


class ExtractionContractError(ValueError):
    """The response or the snapshot failed reconciliation.

    Callers do not repair the text, keep a prefix, or switch to fallback
    inside this module. Degradation stays the caller's choice.
    """


class ExtractionLimitError(RuntimeError):
    """A frozen size, token, or repair limit was hit. Stop; do not clip."""


def frozen_limits() -> dict[str, Any]:
    """The constants this process will enforce. Selected values stay fixed."""
    return {
        "EXTRACTION_CONTRACT_VERSION": EXTRACTION_CONTRACT_VERSION,
        "EVIDENCE_ROLE": EVIDENCE_ROLE,
        "VALUE_CLIP": VALUE_CLIP,
        "MAX_ITEMS": MAX_ITEMS,
        "MAX_TOKENS_PER_CALL": MAX_TOKENS_PER_CALL,
        "MAX_TOTAL_TOKENS": MAX_TOTAL_TOKENS,
        "MAX_REPAIRS": MAX_REPAIRS,
        "MAX_INPUT_CHARS": MAX_INPUT_CHARS,
        "OMITTED_RANGE_REASON": OMITTED_RANGE_REASON,
    }


def announce_frozen_limits() -> str:
    """Print the frozen constants once, before any model call or score."""
    global _ANNOUNCED
    lines = ["extraction contract limits (frozen before any call or score)"]
    for key, value in frozen_limits().items():
        lines.append(f"{key}={value}")
    text = "\n".join(lines)
    if not _ANNOUNCED:
        print(text, flush=True)
        _ANNOUNCED = True
    return text


def build_model_input(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Project one commit snapshot into the object the model is allowed to see.

    ``ao_records`` and ``ledger_rows`` are zipped by index. ``action`` and
    ``observation`` keep every key. A string longer than ``VALUE_CLIP`` or a
    list longer than ``MAX_ITEMS`` raises ``ExtractionLimitError``. The
    canonical JSON of the result must fit in ``MAX_INPUT_CHARS``.
    """
    if not isinstance(snapshot, Mapping):
        raise ExtractionContractError("snapshot must be a mapping")
    ao_records = snapshot.get("ao_records")
    ledger_rows = snapshot.get("ledger_rows")
    if not isinstance(ao_records, list) or not isinstance(ledger_rows, list):
        raise ExtractionContractError("snapshot requires ao_records and ledger_rows lists")
    if len(ao_records) != len(ledger_rows):
        raise ExtractionContractError(
            f"ao_records ({len(ao_records)}) and ledger_rows ({len(ledger_rows)}) length mismatch"
        )
    if not ao_records:
        raise ExtractionContractError("snapshot has no AO rows")
    watermark = snapshot.get("snapshot_watermark")
    if not isinstance(watermark, str) or not watermark.strip():
        raise ExtractionContractError("snapshot_watermark must be a non-empty str")

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_sequences: set[int] = set()
    for index, (record, ledger) in enumerate(zip(ao_records, ledger_rows, strict=True)):
        if not isinstance(record, Mapping) or not isinstance(ledger, Mapping):
            raise ExtractionContractError(f"row {index} is not a pair of objects")
        ao_id = _require_text(record.get("ao_id"), f"ao_records[{index}].ao_id")
        sequence = _require_int(record.get("sequence"), f"ao_records[{index}].sequence")
        ledger_id = _require_text(ledger.get("ao_id"), f"ledger_rows[{index}].ao_id")
        ledger_sequence = _require_int(ledger.get("sequence"), f"ledger_rows[{index}].sequence")
        if ao_id != ledger_id or sequence != ledger_sequence:
            raise ExtractionContractError(
                f"row {index} ao_id/sequence mismatch: "
                f"projection {(ao_id, sequence)} ledger {(ledger_id, ledger_sequence)}"
            )
        if ao_id in seen_ids:
            raise ExtractionContractError(f"duplicate ao_id {ao_id}")
        if sequence in seen_sequences:
            raise ExtractionContractError(f"duplicate sequence {sequence}")
        seen_ids.add(ao_id)
        seen_sequences.add(sequence)
        captured = ledger.get("captured_state")
        if not isinstance(captured, str) or not captured.strip():
            raise ExtractionContractError(
                f"ledger_rows[{index}].captured_state must be a non-empty str"
            )
        rows.append(
            {
                "ao_id": ao_id,
                "sequence": sequence,
                "participant_id": _require_text(
                    record.get("participant_id"), f"ao_records[{index}].participant_id"
                ),
                "segment_id": _require_text(
                    record.get("segment_id"), f"ao_records[{index}].segment_id"
                ),
                "session_id": _require_text(
                    record.get("session_id"), f"ao_records[{index}].session_id"
                ),
                "content_hash": _require_text(
                    record.get("content_hash"), f"ao_records[{index}].content_hash"
                ),
                "captured_state": captured,
                "action": _project_object(ledger.get("action", {}), f"ledger_rows[{index}].action"),
                "observation": _project_object(
                    ledger.get("observation", {}), f"ledger_rows[{index}].observation"
                ),
            }
        )

    identities = {
        field: value
        for field in _IDENTITY_FIELDS
        if isinstance((value := snapshot.get(field)), str) and value.strip()
    }
    if "session_id" not in identities:
        identities["session_id"] = rows[0]["session_id"]
    model_input = {
        "schema_version": EXTRACTION_CONTRACT_VERSION,
        "snapshot_watermark": watermark,
        "identities": identities,
        "rows": rows,
    }
    encoded = json.dumps(model_input, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded) > MAX_INPUT_CHARS:
        raise ExtractionLimitError(
            f"input limit: model input is {len(encoded)} chars; MAX_INPUT_CHARS={MAX_INPUT_CHARS}"
        )
    return model_input


def render_contract_prompt(model_input: Mapping[str, Any]) -> str:
    """Instruction text plus the value domains imported from the view module."""
    roles = ", ".join(sorted(BLOCK_ROLES))
    kinds = ", ".join(sorted(BLOCK_KINDS))
    relations = ", ".join(sorted(RELATION_TYPES))
    statuses = ", ".join(sorted(CLAIM_STATUSES))
    refinements = ", ".join(
        f"{kind} refines {BLOCK_KIND_ROLES[kind]}" for kind in sorted(BLOCK_KIND_ROLES)
    )
    identities = model_input.get("identities")
    if not isinstance(identities, Mapping):
        identities = {}
    identity_lines = "\n".join(
        f"- {field}={identities[field]}" for field in _IDENTITY_FIELDS if field in identities
    )
    payload = json.dumps(model_input, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "\n".join(
        [
            "You extract one influence view from the AO ledger snapshot below.",
            "Select and name. Do not invent ao_id values or content hashes.",
            "",
            "Rules:",
            "- A block is a contiguous AO interval for one participant_id and one segment_id.",
            "- Blocks do not overlap.",
            "- Cite only ao_id values present in rows.",
            "- evidence_content_hash must be copied exactly from that row's content_hash.",
            "- failure_marked is your judgment. true requires failure_evidence.",
            "- failure_evidence requires failure_marked true. Do not infer the marker from roles.",
            f"- Do not use {DETERMINISTIC_FALLBACK_REASON} or {SYNTHETIC_ROOT_SINK_REASON} "
            "as granularity_reason.",
            "- block_id, claim_id, and subgraph_id must not embed any identity string below.",
            "- Do not write source_refs, source_content_hashes, included_ranges, or omitted_ranges.",
            "- Output one JSON object and no other text.",
            "",
            f"schema_version must be {EXTRACTION_CONTRACT_VERSION}.",
            f"role is one of: {roles}.",
            f"block_kind is one of: {kinds}.",
            f"block_kind compatibility: {refinements}. unknown is allowed with any role.",
            f"relation_type is one of: {relations}.",
            f"status is one of: {statuses}.",
            f"A subgraph has at most {MAX_RETRIEVAL_SUBGRAPH_BLOCKS} block_ids.",
            "",
            "JSON object fields:",
            "- schema_version: string",
            "- blocks: [{block_id, role, block_kind, participant_id, segment_id, "
            "ao_start_seq, ao_end_seq, summary, input, output, authorship, granularity_reason}]",
            "- claims: [{claim_id, source_block_id, target_block_id, carried_artifact, "
            "downstream_effect, relation_type, status, "
            "source_evidence: [{ao_id, evidence_content_hash}], "
            "target_evidence: [{ao_id, evidence_content_hash}]}]",
            "- subgraphs: [{subgraph_id, block_ids, claim_ids, failure_marked, "
            "failure_evidence: [{ao_id, evidence_content_hash}]}]",
            "- coverage: {total_ao_start_seq, total_ao_end_seq, inspected_ao_ids, "
            "snapshot_watermark}",
            "",
            "Identity strings you must not embed:",
            identity_lines or "- (none)",
            "",
            "Ledger snapshot:",
            payload,
        ]
    )


def parse_contract_response(text: str, model_input: Mapping[str, Any]) -> ExtractionPayload:
    """Strip one JSON fence, check the object, reconcile it, and build the payload.

    Derived fields (source refs, evidence provenance, included and omitted
    ranges) are computed from ``model_input``. The model text does not supply
    them. Any miss raises ``ExtractionContractError``.
    """
    if not isinstance(model_input, Mapping):
        raise ExtractionContractError("model_input must be a mapping")
    document = _load_object(_unwrap_json_text(text))
    _exact_keys(document, _TOP_KEYS, "response")
    version = document.get("schema_version")
    if version != EXTRACTION_CONTRACT_VERSION:
        raise ExtractionContractError(
            f"schema_version must be {EXTRACTION_CONTRACT_VERSION!r}, got {version!r}"
        )
    rows = _input_rows(model_input)
    by_id = {row["ao_id"]: row for row in rows}
    by_sequence = {row["sequence"]: row for row in rows}
    identities = _identities(model_input)
    watermark = model_input.get("snapshot_watermark")
    if not isinstance(watermark, str) or not watermark.strip():
        raise ExtractionContractError("model_input.snapshot_watermark must be a non-empty str")

    raw_blocks = _require_list(document.get("blocks"), "blocks")
    raw_claims = _require_list(document.get("claims"), "claims")
    raw_subgraphs = _require_list(document.get("subgraphs"), "subgraphs")
    coverage_doc = document.get("coverage")
    _exact_keys(coverage_doc, _COVERAGE_KEYS, "coverage")
    assert isinstance(coverage_doc, dict)

    if coverage_doc.get("snapshot_watermark") != watermark:
        raise ExtractionContractError("coverage.snapshot_watermark does not match the snapshot")
    inspected_ids = _str_list(coverage_doc.get("inspected_ao_ids"), "coverage.inspected_ao_ids")
    if len(inspected_ids) != len(set(inspected_ids)):
        raise ExtractionContractError("coverage.inspected_ao_ids contains a duplicate")
    for ao_id in inspected_ids:
        if ao_id not in by_id:
            raise ExtractionContractError(f"inspected ao_id {ao_id} is not in the snapshot")
    total_start = _require_int(
        coverage_doc.get("total_ao_start_seq"), "coverage.total_ao_start_seq"
    )
    total_end = _require_int(coverage_doc.get("total_ao_end_seq"), "coverage.total_ao_end_seq")
    sequences = [row["sequence"] for row in rows]
    if total_start != min(sequences) or total_end != max(sequences):
        raise ExtractionContractError(
            "coverage total AO range does not match the snapshot min and max sequence"
        )

    blocks: list[InfluenceBlock] = []
    seen_block_ids: set[str] = set()
    for index, item in enumerate(raw_blocks):
        block = _parse_block(item, index, by_sequence, identities)
        if block.block_id in seen_block_ids:
            raise ExtractionContractError(f"duplicate block_id {block.block_id}")
        seen_block_ids.add(block.block_id)
        blocks.append(block)
    _reject_overlapping_blocks(blocks)

    claims: list[InfluenceClaim] = []
    seen_claim_ids: set[str] = set()
    for index, item in enumerate(raw_claims):
        claim = _parse_claim(item, index, by_id, watermark, identities, seen_block_ids)
        if claim.claim_id in seen_claim_ids:
            raise ExtractionContractError(f"duplicate claim_id {claim.claim_id}")
        seen_claim_ids.add(claim.claim_id)
        claims.append(claim)

    subgraphs: list[RetrievalSubgraph] = []
    seen_subgraph_ids: set[str] = set()
    for index, item in enumerate(raw_subgraphs):
        subgraph = _parse_subgraph(
            item, index, by_id, watermark, identities, seen_block_ids, seen_claim_ids, claims
        )
        if subgraph.subgraph_id in seen_subgraph_ids:
            raise ExtractionContractError(f"duplicate subgraph_id {subgraph.subgraph_id}")
        seen_subgraph_ids.add(subgraph.subgraph_id)
        subgraphs.append(subgraph)

    coverage = _derive_coverage(
        blocks, rows, inspected_ids, by_id, total_start, total_end, watermark
    )
    return ExtractionPayload(
        blocks=blocks,
        claims=claims,
        coverage=coverage,
        retrieval_subgraphs=subgraphs,
    )


def render_native_edge_prompt(model_input: Mapping[str, Any]) -> str:
    """Instruction text for the native-edge schema. Claim lifecycle is absent."""
    roles = ", ".join(sorted(BLOCK_ROLES))
    kinds = ", ".join(sorted(BLOCK_KINDS))
    relations = ", ".join(sorted(RELATION_TYPES))
    refinements = ", ".join(
        f"{kind} refines {BLOCK_KIND_ROLES[kind]}" for kind in sorted(BLOCK_KIND_ROLES)
    )
    identities = model_input.get("identities")
    if not isinstance(identities, Mapping):
        identities = {}
    identity_lines = "\n".join(
        f"- {field}={identities[field]}" for field in _IDENTITY_FIELDS if field in identities
    )
    payload = json.dumps(model_input, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "\n".join(
        [
            "You extract one factual AIG from the AO ledger snapshot below.",
            "Select and name blocks and native edges. Do not invent ao_id values or content hashes.",
            "",
            "Rules:",
            "- A block is a contiguous AO interval for one participant_id and one segment_id.",
            "- Blocks do not overlap.",
            "- Cite only ao_id values present in rows.",
            "- A native edge points from an earlier block to a later block.",
            "- relation_type influence records an inheritance or influence mechanism.",
            "- relation_type skip is the mechanism used when the edge crosses an omitted AO range.",
            "- mechanism is a short note on that edge.",
            "- Do not output a claim, a verdict, evidence refs, grounding, or a retrieval subgraph.",
            f"- Do not use {DETERMINISTIC_FALLBACK_REASON} or {SYNTHETIC_ROOT_SINK_REASON} "
            "as granularity_reason.",
            "- block_id and edge_id must not embed any identity string below.",
            "- Do not write source_refs, source_content_hashes, included_ranges, or omitted_ranges.",
            "- Output one JSON object and no other text.",
            "",
            f"schema_version must be {EXTRACTION_CONTRACT_NATIVE_VERSION}.",
            f"role is one of: {roles}.",
            f"block_kind is one of: {kinds}.",
            f"block_kind compatibility: {refinements}. unknown is allowed with any role.",
            f"relation_type is one of: {relations}.",
            "",
            "JSON object fields:",
            "- schema_version: string",
            "- blocks: [{block_id, role, block_kind, participant_id, segment_id, "
            "ao_start_seq, ao_end_seq, summary, input, output, authorship, granularity_reason}]",
            "- edges: [{edge_id, source_block_id, target_block_id, relation_type, mechanism}]",
            "- coverage: {total_ao_start_seq, total_ao_end_seq, inspected_ao_ids, "
            "snapshot_watermark}",
            "",
            "Identity strings you must not embed:",
            identity_lines or "- (none)",
            "",
            "Ledger snapshot:",
            payload,
        ]
    )


def parse_native_edge_response(text: str, model_input: Mapping[str, Any]) -> NativeExtractionPayload:
    """Parse a native-edge extraction. Claim fields are rejected, not dropped."""
    if not isinstance(model_input, Mapping):
        raise ExtractionContractError("model_input must be a mapping")
    if model_input.get("schema_version") != EXTRACTION_CONTRACT_NATIVE_VERSION:
        raise ExtractionContractError(
            "model_input.schema_version must be "
            f"{EXTRACTION_CONTRACT_NATIVE_VERSION!r}, got {model_input.get('schema_version')!r}"
        )
    document = _load_object(_unwrap_json_text(text))
    _exact_keys(document, _NATIVE_TOP_KEYS, "response")
    version = document.get("schema_version")
    if version != EXTRACTION_CONTRACT_NATIVE_VERSION:
        raise ExtractionContractError(
            f"schema_version must be {EXTRACTION_CONTRACT_NATIVE_VERSION!r}, got {version!r}"
        )
    rows = _input_rows(model_input)
    by_sequence = {row["sequence"]: row for row in rows}
    identities = _identities(model_input)
    watermark = model_input.get("snapshot_watermark")
    if not isinstance(watermark, str) or not watermark.strip():
        raise ExtractionContractError("model_input.snapshot_watermark must be a non-empty str")

    raw_blocks = _require_list(document.get("blocks"), "blocks")
    raw_edges = _require_list(document.get("edges"), "edges")
    if not raw_blocks:
        raise ExtractionContractError("blocks must be a non-empty list")
    coverage_doc = document.get("coverage")
    _exact_keys(coverage_doc, _COVERAGE_KEYS, "coverage")
    assert isinstance(coverage_doc, dict)
    if coverage_doc.get("snapshot_watermark") != watermark:
        raise ExtractionContractError("coverage.snapshot_watermark does not match the snapshot")
    inspected_ids = _str_list(coverage_doc.get("inspected_ao_ids"), "coverage.inspected_ao_ids")
    by_id = {row["ao_id"]: row for row in rows}
    if len(inspected_ids) != len(set(inspected_ids)):
        raise ExtractionContractError("coverage.inspected_ao_ids contains a duplicate")
    for ao_id in inspected_ids:
        if ao_id not in by_id:
            raise ExtractionContractError(f"inspected ao_id {ao_id} is not in the snapshot")
    total_start = _require_int(
        coverage_doc.get("total_ao_start_seq"), "coverage.total_ao_start_seq"
    )
    total_end = _require_int(coverage_doc.get("total_ao_end_seq"), "coverage.total_ao_end_seq")
    sequences = [row["sequence"] for row in rows]
    if total_start != min(sequences) or total_end != max(sequences):
        raise ExtractionContractError(
            "coverage total AO range does not match the snapshot min and max sequence"
        )

    blocks: list[InfluenceBlock] = []
    seen_block_ids: set[str] = set()
    for index, item in enumerate(raw_blocks):
        block = _parse_block(item, index, by_sequence, identities)
        if block.block_id in seen_block_ids:
            raise ExtractionContractError(f"duplicate block_id {block.block_id}")
        seen_block_ids.add(block.block_id)
        blocks.append(block)
    _reject_overlapping_blocks(blocks)

    edges: list[AigNativeEdge] = []
    seen_edge_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for index, item in enumerate(raw_edges):
        edge = _parse_native_edge(item, index, identities, seen_block_ids)
        if edge.edge_id in seen_edge_ids:
            raise ExtractionContractError(f"duplicate edge_id {edge.edge_id}")
        pair = (edge.source_block_id, edge.target_block_id)
        if pair in seen_pairs:
            raise ExtractionContractError(
                f"duplicate native edge {pair[0]}->{pair[1]}"
            )
        seen_edge_ids.add(edge.edge_id)
        seen_pairs.add(pair)
        edges.append(edge)

    coverage = _derive_coverage(
        blocks, rows, inspected_ids, by_id, total_start, total_end, watermark
    )
    return NativeExtractionPayload(blocks=blocks, edges=edges, coverage=coverage)


def _require_model_extraction_bounds(
    vlm_handle: Any,
    *,
    max_repairs: int,
    max_tokens_per_call: int,
    max_total_tokens: int,
) -> None:
    """Shared limit checks for the claim and native-edge model providers."""
    if vlm_handle is None or not hasattr(vlm_handle, "get_completion_async"):
        raise ExtractionLimitError(
            "model extraction needs a provider handle with get_completion_async; none was given"
        )
    if isinstance(max_repairs, bool) or not isinstance(max_repairs, int):
        raise ExtractionLimitError("max_repairs must be an int")
    if max_repairs < 0 or max_repairs > MAX_REPAIRS:
        raise ExtractionLimitError(f"max_repairs must be in 0..{MAX_REPAIRS}, got {max_repairs}")
    _check_cap("max_tokens_per_call", max_tokens_per_call, MAX_TOKENS_PER_CALL)
    _check_cap("max_total_tokens", max_total_tokens, MAX_TOTAL_TOKENS)
    announce_frozen_limits()


def model_native_extraction_provider(
    vlm_handle: Any,
    *,
    max_repairs: int = MAX_REPAIRS,
    max_tokens_per_call: int = MAX_TOKENS_PER_CALL,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
) -> NativeExtractionProviderFactory:
    """Bind a snapshot to a provider that parses the native-edge schema.

    The returned callable yields ``NativeExtractionPayload``, not
    ``ExtractionPayload``. Claim repair text is not used.
    """
    _require_model_extraction_bounds(
        vlm_handle,
        max_repairs=max_repairs,
        max_tokens_per_call=max_tokens_per_call,
        max_total_tokens=max_total_tokens,
    )

    def factory(snapshot: Mapping[str, Any]) -> NativeExtractionProvider:
        model_input = build_model_input(snapshot)
        model_input["schema_version"] = EXTRACTION_CONTRACT_NATIVE_VERSION
        prompt = render_native_edge_prompt(model_input)
        return _SnapshotExtractionProvider(
            vlm_handle,
            model_input=model_input,
            prompt=prompt,
            max_repairs=max_repairs,
            max_tokens_per_call=max_tokens_per_call,
            max_total_tokens=max_total_tokens,
            response_parser=parse_native_edge_response,
        )

    return factory


def model_extraction_provider(
    vlm_handle: Any,
    *,
    max_repairs: int = MAX_REPAIRS,
    max_tokens_per_call: int = MAX_TOKENS_PER_CALL,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
) -> ExtractionProviderFactory:
    """Return a factory that binds each commit snapshot to one provider.

    ``max_repairs``, ``max_tokens_per_call``, and ``max_total_tokens`` may be
    lower than the frozen constants. They may not be higher. Parsing failure
    is the only retried error. A limit error propagates on the first hit.
    """
    _require_model_extraction_bounds(
        vlm_handle,
        max_repairs=max_repairs,
        max_tokens_per_call=max_tokens_per_call,
        max_total_tokens=max_total_tokens,
    )

    def factory(snapshot: Mapping[str, Any]) -> ExtractionProvider:
        model_input = build_model_input(snapshot)
        prompt = render_contract_prompt(model_input)
        return _SnapshotExtractionProvider(
            vlm_handle,
            model_input=model_input,
            prompt=prompt,
            max_repairs=max_repairs,
            max_tokens_per_call=max_tokens_per_call,
            max_total_tokens=max_total_tokens,
        )

    return factory


class _SnapshotExtractionProvider:
    """One commit's provider. Callable as ``(session_id, task_run_id, outcome)``."""

    def __init__(
        self,
        vlm_handle: Any,
        *,
        model_input: Mapping[str, Any],
        prompt: str,
        max_repairs: int,
        max_tokens_per_call: int,
        max_total_tokens: int,
        response_parser: Callable[[str, Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        # ``unavailable`` until a call is actually measured. A zero counter is
        # not a measurement, and the report must not treat it as one.
        self.usage_source = "unavailable"
        self.usage_sources: list[str] = []
        self.finish_reasons: list[str] = []
        self.repairs = 0
        self.calls = 0
        self.truncated = 0
        self._vlm_handle = vlm_handle
        self._model_input = model_input
        self._messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        self._max_repairs = max_repairs
        self._max_tokens_per_call = max_tokens_per_call
        self._max_total_tokens = max_total_tokens
        self._response_parser = (
            parse_contract_response if response_parser is None else response_parser
        )

    def __call__(self, session_id: str, task_run_id: str, outcome_status: str) -> ExtractionPayload:
        del session_id, task_run_id, outcome_status
        while True:
            before = _handle_usage_counts(self._vlm_handle)
            response = _run_completion(self._vlm_handle, self._messages, self._max_tokens_per_call)
            after = _handle_usage_counts(self._vlm_handle)
            self.calls += 1
            self._account(response, before, after)
            text = ""
            try:
                text = _response_text(response)
                return self._response_parser(text, self._model_input)
            except ExtractionContractError as exc:
                if self.calls > self._max_repairs:
                    raise
                self.repairs += 1
                self._messages.append({"role": "assistant", "content": text})
                self._messages.append(
                    {
                        "role": "user",
                        "content": (
                            "ExtractionContractError: "
                            f"{exc}. Output one JSON object and no other text."
                        ),
                    }
                )

    def _account(
        self,
        response: Any,
        before: tuple[int, int, int] | None,
        after: tuple[int, int, int] | None,
    ) -> None:
        """Account one call. Response usage wins. Otherwise use the handle delta.

        The handle delta is a measurement for the books and the report. It
        does not change the frozen caps: the same three checks still run on
        the accounted numbers, and a hit still raises. No reading at all is
        ``unavailable`` — the zero counters are not reported as a cost.
        """
        measured = _measured_usage(response, before, after)
        reason = getattr(response, "finish_reason", None)
        if reason is None and isinstance(response, str):
            reason = "stop"
        reason_text = reason if isinstance(reason, str) else ""
        self.finish_reasons.append(reason_text)
        if measured is None:
            self.usage_sources.append("unavailable")
            self.usage_source = "unavailable"
            if reason_text in {"length", "max_tokens"}:
                self.truncated += 1
                raise ExtractionLimitError(
                    f"per-call output was truncated (finish_reason={reason_text}); "
                    f"MAX_TOKENS_PER_CALL={self._max_tokens_per_call}"
                )
            return
        prompt_tokens, completion_tokens, total, source = measured
        self.usage_sources.append(source)
        self.usage_source = _collapse_usage_source(self.usage_sources)
        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        self.usage["total_tokens"] += total
        if completion_tokens > self._max_tokens_per_call:
            raise ExtractionLimitError(
                f"completion_tokens {completion_tokens} exceed "
                f"MAX_TOKENS_PER_CALL {self._max_tokens_per_call}"
            )
        if reason_text in {"length", "max_tokens"}:
            self.truncated += 1
            raise ExtractionLimitError(
                f"per-call output was truncated (finish_reason={reason_text}); "
                f"MAX_TOKENS_PER_CALL={self._max_tokens_per_call}"
            )
        if self.usage["total_tokens"] > self._max_total_tokens:
            raise ExtractionLimitError(
                f"total tokens {self.usage['total_tokens']} exceed "
                f"MAX_TOTAL_TOKENS {self._max_total_tokens}"
            )


def _check_cap(name: str, value: int, frozen: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > frozen:
        raise ExtractionLimitError(f"{name} must be in 1..{frozen}, got {value!r}")


def _usage_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _collapse_usage_source(sources: list[str]) -> str:
    """One label for the calls so far. Unknown spend is never called a zero."""
    if not sources or "unavailable" in sources:
        return "unavailable"
    if all(item == "handle_tracker" for item in sources):
        return "handle_tracker"
    if all(item == "response" for item in sources):
        return "response"
    return "response"


def _response_usage(response: Any) -> Mapping[str, Any] | None:
    """Usage carried on the response, or None when the response has none."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, Mapping):
        usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return None
    if not any(key in usage for key in _USAGE_KEYS):
        return None
    return usage


def _usage_triple(usage: Mapping[str, Any]) -> tuple[int, int, int]:
    prompt_tokens = _usage_int(usage.get("prompt_tokens"))
    completion_tokens = _usage_int(usage.get("completion_tokens"))
    total_tokens = usage.get("total_tokens")
    total = (
        _usage_int(total_tokens) if total_tokens is not None else prompt_tokens + completion_tokens
    )
    return prompt_tokens, completion_tokens, total


def _coerce_usage_counts(raw: Any) -> tuple[int, int, int] | None:
    """Normalize a flat usage dict or a tracker ``to_dict()`` payload."""
    if not isinstance(raw, Mapping):
        return None
    body: Mapping[str, Any] = raw
    nested = raw.get("total_usage")
    if isinstance(nested, Mapping):
        body = nested
    if not any(key in body for key in _USAGE_KEYS):
        return None
    return _usage_triple(body)


def _read_usage_reader(reader: Any) -> Any:
    try:
        return reader()
    except Exception:
        return None


def _handle_usage_counts(handle: Any) -> tuple[int, int, int] | None:
    """Cumulative prompt/completion/total on the handle, or None if it has none.

    Prefer ``get_token_usage()``. Fall back to ``token_tracker.to_dict()``.
    Neither interface is an error: the caller skips the reading.
    """
    getter = getattr(handle, "get_token_usage", None)
    if callable(getter):
        counts = _coerce_usage_counts(_read_usage_reader(getter))
        if counts is not None:
            return counts
    tracker = getattr(handle, "token_tracker", None)
    to_dict = getattr(tracker, "to_dict", None) if tracker is not None else None
    if not callable(to_dict):
        return None
    return _coerce_usage_counts(_read_usage_reader(to_dict))


def _measured_usage(
    response: Any,
    before: tuple[int, int, int] | None,
    after: tuple[int, int, int] | None,
) -> tuple[int, int, int, str] | None:
    """Response usage, else the handle's before/after delta, else None."""
    response_usage = _response_usage(response)
    if response_usage is not None:
        prompt_tokens, completion_tokens, total = _usage_triple(response_usage)
        return prompt_tokens, completion_tokens, total, "response"
    if before is None or after is None:
        return None
    return (
        after[0] - before[0],
        after[1] - before[1],
        after[2] - before[2],
        "handle_tracker",
    )


def _run_completion(vlm_handle: Any, messages: list[dict[str, str]], max_tokens: int) -> Any:
    coro = vlm_handle.get_completion_async(messages=list(messages), max_tokens=max_tokens)
    if not inspect.isawaitable(coro):
        return coro
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    raise ExtractionContractError("model response has no text content")


def _unwrap_json_text(text: str) -> str:
    if not isinstance(text, str):
        raise ExtractionContractError("model response must be a str")
    stripped = text.strip()
    if stripped.startswith("```"):
        match = _FENCE.fullmatch(stripped)
        if match is None:
            raise ExtractionContractError(
                "json fence must wrap the whole response and nothing else"
            )
        return match.group(1).strip()
    return stripped


def _load_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    try:
        loaded, end = decoder.raw_decode(text)
    except json.JSONDecodeError as exc:
        raise ExtractionContractError(f"response is not JSON: {exc}") from exc
    if text[end:].strip():
        raise ExtractionContractError("extra text around the JSON object")
    if not isinstance(loaded, dict):
        raise ExtractionContractError("response JSON must be an object")
    return loaded


def _exact_keys(value: Any, allowed: frozenset[str], where: str) -> None:
    if not isinstance(value, dict):
        raise ExtractionContractError(f"{where} must be an object")
    keys = set(value)
    missing = sorted(allowed - keys)
    extra = sorted(keys - allowed)
    if missing or extra:
        raise ExtractionContractError(f"{where} keys mismatch missing={missing} extra={extra}")


def _input_rows(model_input: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = model_input.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ExtractionContractError("model_input.rows must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ExtractionContractError(f"model_input.rows[{index}] must be an object")
        normalized.append(dict(row))
    return normalized


def _identities(model_input: Mapping[str, Any]) -> dict[str, str]:
    raw = model_input.get("identities")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ExtractionContractError("model_input.identities must be an object")
    return {
        field: value
        for field in _IDENTITY_FIELDS
        if isinstance((value := raw.get(field)), str) and value
    }


def _parse_block(
    item: Any,
    index: int,
    by_sequence: Mapping[int, Mapping[str, Any]],
    identities: Mapping[str, str],
) -> InfluenceBlock:
    where = f"blocks[{index}]"
    _exact_keys(item, _BLOCK_KEYS, where)
    assert isinstance(item, dict)
    block_id = _require_text(item.get("block_id"), f"{where}.block_id")
    _reject_embedded(block_id, "block_id", identities)
    role = _require_choice(item.get("role"), f"{where}.role", BLOCK_ROLES)
    block_kind = _require_choice(item.get("block_kind"), f"{where}.block_kind", BLOCK_KINDS)
    if block_kind != "unknown" and BLOCK_KIND_ROLES[block_kind] != role:
        refined = BLOCK_KIND_ROLES[block_kind]
        raise ExtractionContractError(
            f"{where} block_kind {block_kind!r} refines role {refined!r}, got {role!r}"
        )
    participant_id = _require_text(item.get("participant_id"), f"{where}.participant_id")
    segment_id = _require_text(item.get("segment_id"), f"{where}.segment_id")
    start = _require_int(item.get("ao_start_seq"), f"{where}.ao_start_seq")
    end = _require_int(item.get("ao_end_seq"), f"{where}.ao_end_seq")
    if start > end:
        raise ExtractionContractError(f"{where} AO range is empty")
    span = _rows_for_span(start, end, by_sequence, where)
    participants = {row["participant_id"] for row in span}
    segments = {row["segment_id"] for row in span}
    if participants != {participant_id}:
        raise ExtractionContractError(f"{where} crosses participant")
    if segments != {segment_id}:
        raise ExtractionContractError(f"{where} crosses segment")
    sessions = {row["session_id"] for row in span}
    if len(sessions) != 1:
        raise ExtractionContractError(f"{where} crosses session")
    reason = _require_text(item.get("granularity_reason"), f"{where}.granularity_reason")
    if reason in _RESERVED_GRANULARITY:
        raise ExtractionContractError(f"{where} granularity_reason uses reserved value {reason!r}")
    refs = tuple(
        SourceRef(
            ao_id=row["ao_id"],
            sequence=row["sequence"],
            participant_id=row["participant_id"],
            segment_id=row["segment_id"],
            session_id=row["session_id"],
            content_hash=row["content_hash"],
        )
        for row in span
    )
    try:
        return InfluenceBlock(
            block_id=block_id,
            role=role,  # type: ignore[arg-type]
            participant_id=participant_id,
            segment_ref=SegmentRef(segment_id=segment_id, session_id=next(iter(sessions))),
            ao_start_seq=start,
            ao_end_seq=end,
            summary=_allow_text(item.get("summary"), f"{where}.summary"),
            input=_allow_text(item.get("input"), f"{where}.input"),
            output=_allow_text(item.get("output"), f"{where}.output"),
            authorship=_allow_text(item.get("authorship"), f"{where}.authorship"),
            granularity_reason=reason,
            source_content_hashes=tuple(ref.content_hash for ref in refs),
            source_refs=refs,
            block_kind=block_kind,
        )
    except (TypeError, ValueError) as exc:
        raise ExtractionContractError(f"{where} {exc}") from exc


def _rows_for_span(
    start: int,
    end: int,
    by_sequence: Mapping[int, Mapping[str, Any]],
    where: str,
) -> list[Mapping[str, Any]]:
    if not by_sequence:
        raise ExtractionContractError(f"{where} snapshot has no sequences")
    lowest = min(by_sequence)
    highest = max(by_sequence)
    if start < lowest or end > highest:
        raise ExtractionContractError(f"{where} AO range {start}-{end} is outside the snapshot")
    span: list[Mapping[str, Any]] = []
    for sequence in range(start, end + 1):
        row = by_sequence.get(sequence)
        if row is None:
            raise ExtractionContractError(
                f"{where} AO range {start}-{end} is not contiguous at sequence {sequence}"
            )
        span.append(row)
    return span


def _reject_overlapping_blocks(blocks: Sequence[InfluenceBlock]) -> None:
    ordered = sorted(
        blocks, key=lambda block: (block.ao_start_seq, block.ao_end_seq, block.block_id)
    )
    previous: InfluenceBlock | None = None
    for block in ordered:
        if previous is not None and block.ao_start_seq <= previous.ao_end_seq:
            raise ExtractionContractError(f"block {block.block_id} overlaps {previous.block_id}")
        previous = block


def _parse_native_edge(
    item: Any,
    index: int,
    identities: Mapping[str, str],
    block_ids: set[str],
) -> AigNativeEdge:
    where = f"edges[{index}]"
    _exact_keys(item, _EDGE_KEYS, where)
    assert isinstance(item, dict)
    edge_id = _require_text(item.get("edge_id"), f"{where}.edge_id")
    _reject_embedded(edge_id, "edge_id", identities)
    source_block_id = _require_text(item.get("source_block_id"), f"{where}.source_block_id")
    target_block_id = _require_text(item.get("target_block_id"), f"{where}.target_block_id")
    if source_block_id not in block_ids or target_block_id not in block_ids:
        raise ExtractionContractError(f"{where} endpoint block is not declared")
    relation = _require_choice(item.get("relation_type"), f"{where}.relation_type", RELATION_TYPES)
    mechanism = _allow_text(item.get("mechanism"), f"{where}.mechanism")
    try:
        return AigNativeEdge(
            edge_id=edge_id,
            source_block_id=source_block_id,
            target_block_id=target_block_id,
            relation_type=relation,
            mechanism=mechanism,
        )
    except (TypeError, ValueError) as exc:
        raise ExtractionContractError(f"{where} {exc}") from exc


def _parse_claim(
    item: Any,
    index: int,
    by_id: Mapping[str, Mapping[str, Any]],
    watermark: str,
    identities: Mapping[str, str],
    block_ids: set[str],
) -> InfluenceClaim:
    where = f"claims[{index}]"
    _exact_keys(item, _CLAIM_KEYS, where)
    assert isinstance(item, dict)
    claim_id = _require_text(item.get("claim_id"), f"{where}.claim_id")
    _reject_embedded(claim_id, "claim_id", identities)
    source_block_id = _require_text(item.get("source_block_id"), f"{where}.source_block_id")
    target_block_id = _require_text(item.get("target_block_id"), f"{where}.target_block_id")
    if source_block_id not in block_ids or target_block_id not in block_ids:
        raise ExtractionContractError(f"{where} endpoint block is not declared")
    relation = _require_choice(item.get("relation_type"), f"{where}.relation_type", RELATION_TYPES)
    status = _require_choice(item.get("status"), f"{where}.status", CLAIM_STATUSES)
    try:
        return InfluenceClaim(
            claim_id=claim_id,
            source_block_id=source_block_id,
            target_block_id=target_block_id,
            carried_artifact=_require_text(
                item.get("carried_artifact"), f"{where}.carried_artifact"
            ),
            downstream_effect=_require_text(
                item.get("downstream_effect"), f"{where}.downstream_effect"
            ),
            relation_type=relation,
            source_evidence_refs=_evidence_refs(
                item.get("source_evidence"), f"{where}.source_evidence", by_id, watermark
            ),
            target_evidence_refs=_evidence_refs(
                item.get("target_evidence"), f"{where}.target_evidence", by_id, watermark
            ),
            status=status,  # type: ignore[arg-type]
            critic_verdict=None,
        )
    except (TypeError, ValueError) as exc:
        raise ExtractionContractError(f"{where} {exc}") from exc


def _parse_subgraph(
    item: Any,
    index: int,
    by_id: Mapping[str, Mapping[str, Any]],
    watermark: str,
    identities: Mapping[str, str],
    block_ids: set[str],
    claim_ids: set[str],
    claims: Sequence[InfluenceClaim],
) -> RetrievalSubgraph:
    where = f"subgraphs[{index}]"
    _exact_keys(item, _SUBGRAPH_KEYS, where)
    assert isinstance(item, dict)
    subgraph_id = _require_text(item.get("subgraph_id"), f"{where}.subgraph_id")
    _reject_embedded(subgraph_id, "subgraph_id", identities)
    members = _str_list(item.get("block_ids"), f"{where}.block_ids")
    cited = _str_list(item.get("claim_ids"), f"{where}.claim_ids")
    if len(members) != len(set(members)):
        raise ExtractionContractError(f"{where} duplicate block id")
    if len(cited) != len(set(cited)):
        raise ExtractionContractError(f"{where} duplicate claim id")
    for block_id in members:
        if block_id not in block_ids:
            raise ExtractionContractError(f"{where} block {block_id} is not declared")
    claims_by_id = {claim.claim_id: claim for claim in claims}
    member_set = set(members)
    for claim_id in cited:
        claim = claims_by_id.get(claim_id)
        if claim is None or claim_id not in claim_ids:
            raise ExtractionContractError(f"{where} claim {claim_id} is not declared")
        if claim.source_block_id not in member_set or claim.target_block_id not in member_set:
            raise ExtractionContractError(
                f"{where} claim {claim_id} endpoint is outside the subgraph"
            )
    if len(members) > MAX_RETRIEVAL_SUBGRAPH_BLOCKS:
        raise ExtractionContractError(
            f"{where} has {len(members)} blocks; limit is {MAX_RETRIEVAL_SUBGRAPH_BLOCKS}"
        )
    failure_marked = item.get("failure_marked")
    if not isinstance(failure_marked, bool):
        raise ExtractionContractError(f"{where}.failure_marked must be a bool")
    evidence = _evidence_refs(
        item.get("failure_evidence"), f"{where}.failure_evidence", by_id, watermark
    )
    if failure_marked and not evidence:
        raise ExtractionContractError(f"{where} failure_marked requires failure_evidence")
    if evidence and not failure_marked:
        raise ExtractionContractError(f"{where} failure_evidence requires failure_marked")
    try:
        return RetrievalSubgraph(
            subgraph_id=subgraph_id,
            block_ids=tuple(members),
            claim_ids=tuple(cited),
            failure_marked=failure_marked,
            failure_evidence_refs=evidence,
        )
    except (TypeError, ValueError) as exc:
        raise ExtractionContractError(f"{where} {exc}") from exc


def _evidence_refs(
    value: Any,
    where: str,
    by_id: Mapping[str, Mapping[str, Any]],
    watermark: str,
) -> tuple[InfluenceEvidenceRef, ...]:
    items = _require_list(value, where)
    refs: list[InfluenceEvidenceRef] = []
    for index, item in enumerate(items):
        slot = f"{where}[{index}]"
        _exact_keys(item, _EVIDENCE_KEYS, slot)
        assert isinstance(item, dict)
        ao_id = _require_text(item.get("ao_id"), f"{slot}.ao_id")
        digest = _require_text(item.get("evidence_content_hash"), f"{slot}.evidence_content_hash")
        row = by_id.get(ao_id)
        if row is None:
            raise ExtractionContractError(f"{slot} ao_id {ao_id} is not in the snapshot")
        if digest != row["content_hash"]:
            raise ExtractionContractError(
                f"{slot} evidence_content_hash does not match the snapshot row"
            )
        refs.append(
            InfluenceEvidenceRef(
                ao_id=ao_id,
                source_session_id=row["session_id"],
                source_archive_id=None,
                archive_commit_watermark=None,
                source_sequence=row["sequence"],
                source_read_snapshot_watermark=watermark,
                evidence_role=EVIDENCE_ROLE,
                evidence_content_hash=digest,
                captured_state=row["captured_state"],
            )
        )
    return tuple(refs)


def _derive_coverage(
    blocks: Sequence[InfluenceBlock],
    rows: Sequence[Mapping[str, Any]],
    inspected_ids: Sequence[str],
    by_id: Mapping[str, Mapping[str, Any]],
    total_start: int,
    total_end: int,
    watermark: str,
) -> CoverageManifest:
    covered: set[int] = set()
    for block in blocks:
        covered.update(range(block.ao_start_seq, block.ao_end_seq + 1))
    included = tuple(
        AORange(
            start_seq=block.ao_start_seq,
            end_seq=block.ao_end_seq,
            segment_id=block.segment_ref.segment_id,
        )
        for block in blocks
    )
    omitted_rows = [
        row
        for row in sorted(rows, key=lambda item: (item["sequence"], item["ao_id"]))
        if row["sequence"] not in covered
    ]
    omitted: list[OmittedRange] = []
    if omitted_rows:
        start = omitted_rows[0]["sequence"]
        prev = start
        segment_id = omitted_rows[0]["segment_id"]
        for row in omitted_rows[1:]:
            if row["sequence"] == prev + 1 and row["segment_id"] == segment_id:
                prev = row["sequence"]
                continue
            omitted.append(
                OmittedRange(
                    start_seq=start,
                    end_seq=prev,
                    reason=OMITTED_RANGE_REASON,
                    segment_id=segment_id,
                )
            )
            start = row["sequence"]
            prev = start
            segment_id = row["segment_id"]
        omitted.append(
            OmittedRange(
                start_seq=start,
                end_seq=prev,
                reason=OMITTED_RANGE_REASON,
                segment_id=segment_id,
            )
        )
    inspected = tuple(
        SourceRef(
            ao_id=by_id[ao_id]["ao_id"],
            sequence=by_id[ao_id]["sequence"],
            participant_id=by_id[ao_id]["participant_id"],
            segment_id=by_id[ao_id]["segment_id"],
            session_id=by_id[ao_id]["session_id"],
            content_hash=by_id[ao_id]["content_hash"],
        )
        for ao_id in inspected_ids
    )
    try:
        return CoverageManifest(
            total_ao_range=AORange(start_seq=total_start, end_seq=total_end),
            inspected_refs=inspected,
            included_ranges=included,
            omitted_ranges=tuple(omitted),
            snapshot_watermark=watermark,
        )
    except (TypeError, ValueError) as exc:
        raise ExtractionContractError(f"coverage {exc}") from exc


def _project_object(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ExtractionContractError(f"{path} must be an object")
    projected = _project_value(value, path)
    if not isinstance(projected, dict):
        raise ExtractionContractError(f"{path} must be an object")
    return projected


def _project_value(value: Any, path: str) -> Any:
    if isinstance(value, str):
        if len(value) > VALUE_CLIP:
            raise ExtractionLimitError(
                f"input limit: string at {path} length {len(value)} exceeds VALUE_CLIP {VALUE_CLIP}"
            )
        return value
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        if len(value) > MAX_ITEMS:
            raise ExtractionLimitError(
                f"input limit: list at {path} length {len(value)} exceeds MAX_ITEMS {MAX_ITEMS}"
            )
        return [_project_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        return {key: _project_value(value[key], f"{path}.{key}") for key in sorted(value)}
    raise ExtractionContractError(f"unsupported value at {path}: {type(value).__name__}")


def _reject_embedded(identifier: str, field: str, identities: Mapping[str, str]) -> None:
    for name, secret in identities.items():
        if secret in identifier:
            raise ExtractionContractError(f"{field} {identifier!r} must not embed {name}")


def _require_list(value: Any, field: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise ExtractionContractError(f"{field} must be a list")
    return value


def _str_list(value: Any, field: str) -> list[str]:
    items = _require_list(value, field)
    texts: list[str] = []
    for index, item in enumerate(items):
        texts.append(_require_text(item, f"{field}[{index}]"))
    return texts


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExtractionContractError(f"{field} must be a non-empty str")
    return value


def _allow_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ExtractionContractError(f"{field} must be a str")
    return value


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExtractionContractError(f"{field} must be an int")
    return value


def _require_choice(value: Any, field: str, choices: frozenset[str]) -> str:
    text = _require_text(value, field)
    if text not in choices:
        raise ExtractionContractError(f"{field} must be one of {sorted(choices)}, got {text!r}")
    return text
