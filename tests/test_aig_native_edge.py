# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""P01: AIG native-edge extraction, draft/revision, and write path.

The legacy Claim extraction path stays in place. These tests cover the
expanded path only: blocks plus native edges, no Claim lifecycle, and a
writer that cannot update a frozen Claim view.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from openviking.session.aig_native_protocol import read_protocol_document
from openviking.session.extraction_contract import (
    EXTRACTION_CONTRACT_NATIVE_VERSION,
    EXTRACTION_CONTRACT_VERSION,
    ExtractionContractError,
    build_model_input,
    model_native_extraction_provider,
    parse_contract_response,
    parse_native_edge_response,
    render_native_edge_prompt,
)
from openviking.session.extraction_write import (
    ExtractionPayload,
    NativeEdgeViewStore,
    NativeEdgeWriteError,
    NativeExtractionPayload,
    audit_legacy_claim_view,
    extraction_index_builder,
    freeze_native_edge_view,
    open_legacy_claim_view_for_audit,
)
from openviking.session.influence_projection import ProjectionRegistry
from openviking.session.influence_view import (
    PURPOSE_FAILURE_DIAGNOSIS,
    PURPOSE_POST_RUN_INDEX,
    AigNativeEdge,
    AigNativeViewDraft,
    AigNativeViewRevision,
    AORange,
    CoverageManifest,
    HighSeverityViolation,
    InfluenceBlock,
    InfluenceClaim,
    InfluenceViewDraft,
    InfluenceViewRevision,
    OmittedRange,
    SegmentRef,
    SourceRef,
    ViewSupersedeIndex,
    compute_content_hash,
)

SESSION = "sess-native"
TASK = "task-native"
VIEW = "view-native"
REVISION = "rev-native"
WATERMARK = "wm-native"
FROZEN = "2026-10-10T00:00:00.000Z"
CLAIM_LIFECYCLE = ("hypothesis", "supported", "contradicted", "insufficient")


def _row(sequence: int, *, participant: str = "agent-a", segment: str = "seg-a") -> dict:
    return {
        "ao_id": f"ao-{sequence}",
        "sequence": sequence,
        "participant_id": participant,
        "segment_id": segment,
        "session_id": SESSION,
        "content_hash": f"hash-{sequence}",
        "captured_state": "committed",
        "action": {},
        "observation": {},
    }


def _snapshot(rows: list[dict]) -> dict:
    return {
        "session_id": SESSION,
        "task_run_id": TASK,
        "view_id": VIEW,
        "revision_id": REVISION,
        "purpose": PURPOSE_POST_RUN_INDEX,
        "snapshot_watermark": WATERMARK,
        "ao_records": [
            {
                "ao_id": row["ao_id"],
                "sequence": row["sequence"],
                "participant_id": row["participant_id"],
                "segment_id": row["segment_id"],
                "session_id": row["session_id"],
                "content_hash": row["content_hash"],
            }
            for row in rows
        ],
        "ledger_rows": [
            {
                "ao_id": row["ao_id"],
                "sequence": row["sequence"],
                "captured_state": row["captured_state"],
                "action": row["action"],
                "observation": row["observation"],
            }
            for row in rows
        ],
    }


def _native_input(snapshot: dict) -> dict:
    model_input = build_model_input(snapshot)
    model_input["schema_version"] = EXTRACTION_CONTRACT_NATIVE_VERSION
    return model_input


def _block_doc(sequence: int, *, participant: str = "agent-a", segment: str = "seg-a") -> dict:
    return {
        "block_id": f"b-{sequence}",
        "role": "agent_action",
        "block_kind": "tool_call",
        "participant_id": participant,
        "segment_id": segment,
        "ao_start_seq": sequence,
        "ao_end_seq": sequence,
        "summary": "step",
        "input": "in",
        "output": "out",
        "authorship": participant,
        "granularity_reason": "stated by the model",
    }


def _native_document(rows: list[dict] | None = None, *, relation_type: str = "influence") -> dict:
    used = rows if rows is not None else [_row(1), _row(2)]
    sequences = [row["sequence"] for row in used]
    edges = []
    if len(sequences) >= 2:
        edges.append(
            {
                "edge_id": "e-1",
                "source_block_id": f"b-{sequences[0]}",
                "target_block_id": f"b-{sequences[-1]}",
                "relation_type": relation_type,
                "mechanism": "later block inherits the earlier result",
            }
        )
    return {
        "schema_version": EXTRACTION_CONTRACT_NATIVE_VERSION,
        "blocks": [_block_doc(sequence) for sequence in sequences],
        "edges": edges,
        "coverage": {
            "total_ao_start_seq": min(sequences),
            "total_ao_end_seq": max(sequences),
            "inspected_ao_ids": [row["ao_id"] for row in used],
            "snapshot_watermark": WATERMARK,
        },
    }


def _ref(
    sequence: int,
    *,
    participant: str = "agent-a",
    segment: str = "seg-a",
    session_id: str = SESSION,
) -> SourceRef:
    return SourceRef(
        ao_id=f"ao-{sequence}",
        sequence=sequence,
        participant_id=participant,
        segment_id=segment,
        session_id=session_id,
        content_hash=f"hash-{sequence}",
    )


def _block(
    block_id: str,
    sequence: int,
    *,
    role: str = "agent_action",
    block_kind: str = "tool_call",
    participant: str = "agent-a",
    segment: str = "seg-a",
    session_id: str = SESSION,
) -> InfluenceBlock:
    source = _ref(sequence, participant=participant, segment=segment, session_id=session_id)
    return InfluenceBlock(
        block_id=block_id,
        role=role,  # type: ignore[arg-type]
        participant_id=participant,
        segment_ref=SegmentRef(segment_id=segment, session_id=session_id),
        ao_start_seq=sequence,
        ao_end_seq=sequence,
        summary=block_id,
        input="in",
        output="out",
        authorship=participant,
        granularity_reason="one tool call",
        source_content_hashes=(source.content_hash,),
        source_refs=(source,),
        block_kind=block_kind,
    )


def _coverage(
    blocks: list[InfluenceBlock],
    *,
    omitted: tuple[OmittedRange, ...] = (),
    total_start: int | None = None,
    total_end: int | None = None,
) -> CoverageManifest:
    refs = tuple(ref for block in blocks for ref in block.source_refs)
    sequences = [ref.sequence for ref in refs]
    return CoverageManifest(
        total_ao_range=AORange(
            start_seq=min(sequences) if total_start is None else total_start,
            end_seq=max(sequences) if total_end is None else total_end,
        ),
        inspected_refs=refs,
        included_ranges=tuple(
            AORange(
                start_seq=block.ao_start_seq,
                end_seq=block.ao_end_seq,
                segment_id=block.segment_ref.segment_id,
            )
            for block in blocks
        ),
        omitted_ranges=omitted,
        snapshot_watermark=WATERMARK,
    )


def _edge(
    edge_id: str = "e-1",
    source: str = "b-1",
    target: str = "b-2",
    *,
    relation_type: str = "influence",
    mechanism: str = "later block inherits the earlier result",
) -> AigNativeEdge:
    return AigNativeEdge(
        edge_id=edge_id,
        source_block_id=source,
        target_block_id=target,
        relation_type=relation_type,
        mechanism=mechanism,
    )


def _freeze_pair(
    *, relation_type: str = "influence", omitted: tuple[OmittedRange, ...] = ()
) -> AigNativeViewRevision:
    blocks = [_block("b-1", 1), _block("b-2", 2 if not omitted else 3)]
    return freeze_native_edge_view(
        session_id=SESSION,
        task_run_id=TASK,
        view_id=VIEW,
        revision_id=REVISION,
        frozen_at=FROZEN,
        blocks=blocks,
        edges=(_edge(target=blocks[1].block_id, relation_type=relation_type),),
        coverage=_coverage(
            blocks,
            omitted=omitted,
            total_end=3 if omitted else None,
        ),
    )


def _claim_revision() -> InfluenceViewRevision:
    blocks = [_block("b-1", 1), _block("b-2", 2)]
    claim = InfluenceClaim(
        claim_id="claim-1",
        source_block_id="b-1",
        target_block_id="b-2",
        carried_artifact="patch",
        downstream_effect="tests passed",
        relation_type="influence",
        status="hypothesis",
    )
    draft = InfluenceViewDraft(
        view_id="view-claim",
        purpose=PURPOSE_POST_RUN_INDEX,
        session_id=SESSION,
        task_run_id=TASK,
        coverage=_coverage(blocks),
        blocks=blocks,
        claims=(claim,),
    )
    return draft.freeze(revision_id="rev-claim", frozen_at=FROZEN)


def test_native_schema_outputs_blocks_and_edges_without_claims() -> None:
    rows = [_row(1), _row(2)]
    payload = parse_native_edge_response(
        json.dumps(_native_document(rows)),
        _native_input(_snapshot(rows)),
    )
    assert isinstance(payload, NativeExtractionPayload)
    assert not isinstance(payload, ExtractionPayload)
    assert not hasattr(payload, "claims")
    assert [block.block_id for block in payload.blocks] == ["b-1", "b-2"]
    assert len(payload.edges) == 1
    edge = payload.edges[0]
    assert isinstance(edge, AigNativeEdge)
    assert not isinstance(edge, InfluenceClaim)
    assert edge.relation_type == "influence"
    assert edge.mechanism == "later block inherits the earlier result"
    assert edge.source_block_id == "b-1"
    assert edge.target_block_id == "b-2"


def test_native_schema_rejects_claim_status_and_grounding_fields() -> None:
    rows = [_row(1), _row(2)]
    model_input = _native_input(_snapshot(rows))
    document = _native_document(rows)
    document["claims"] = []
    with pytest.raises(ExtractionContractError):
        parse_native_edge_response(json.dumps(document), model_input)
    document = _native_document(rows)
    document["edges"][0]["status"] = "hypothesis"
    with pytest.raises(ExtractionContractError):
        parse_native_edge_response(json.dumps(document), model_input)
    document = _native_document(rows)
    document["edges"][0]["source_evidence"] = []
    with pytest.raises(ExtractionContractError):
        parse_native_edge_response(json.dumps(document), model_input)
    document = _native_document(rows)
    document["edges"][0]["carried_artifact"] = "patch"
    with pytest.raises(ExtractionContractError):
        parse_native_edge_response(json.dumps(document), model_input)
    claim_document = {
        "schema_version": EXTRACTION_CONTRACT_VERSION,
        "blocks": document["blocks"],
        "claims": [],
        "subgraphs": [],
        "coverage": document["coverage"],
    }
    with pytest.raises(ExtractionContractError):
        parse_native_edge_response(json.dumps(claim_document), model_input)


def test_v1_parser_rejects_native_schema() -> None:
    rows = [_row(1), _row(2)]
    native = _native_document(rows)
    with pytest.raises(ExtractionContractError):
        parse_contract_response(json.dumps(native), build_model_input(_snapshot(rows)))


def test_native_prompt_has_no_claim_lifecycle_vocabulary() -> None:
    prompt = render_native_edge_prompt(_native_input(_snapshot([_row(1), _row(2)])))
    assert EXTRACTION_CONTRACT_NATIVE_VERSION in prompt
    assert "edges:" in prompt
    assert "influence" in prompt
    assert "skip" in prompt
    for word in CLAIM_LIFECYCLE:
        assert word not in prompt
    assert "cut_grounding" not in prompt
    assert "InfluenceClaim" not in prompt


def test_native_provider_returns_edges_not_claims() -> None:
    rows = [_row(1), _row(2)]
    text = json.dumps(_native_document(rows))

    class _Handle:
        def get_completion_async(self, *, messages, max_tokens):
            del messages, max_tokens
            return text

    factory = model_native_extraction_provider(_Handle())
    payload = factory(_snapshot(rows))(SESSION, TASK, "success")
    assert isinstance(payload, NativeExtractionPayload)
    assert not hasattr(payload, "claims")
    assert payload.edges[0].relation_type == "influence"


def test_edge_records_mechanism_without_claim_lifecycle() -> None:
    edge = _edge()
    assert edge.relation_type == "influence"
    assert edge.mechanism == "later block inherits the earlier result"
    assert not hasattr(edge, "status")
    assert not hasattr(edge, "critic_verdict")
    assert not hasattr(edge, "source_evidence_refs")
    assert not hasattr(edge, "target_evidence_refs")
    with pytest.raises(TypeError):
        AigNativeEdge(
            edge_id="e-status",
            source_block_id="b-1",
            target_block_id="b-2",
            relation_type="influence",
            mechanism="note",
            status="hypothesis",  # type: ignore[call-arg]
        )


def test_skip_edge_crosses_omitted_range_without_evidence() -> None:
    omitted = (OmittedRange(start_seq=2, end_seq=2, reason="record_type_not_a_block"),)
    revision = _freeze_pair(relation_type="skip", omitted=omitted)
    assert isinstance(revision, AigNativeViewRevision)
    assert not isinstance(revision, InfluenceViewRevision)
    assert not hasattr(revision, "claims")
    assert not hasattr(revision, "retrieval_subgraphs")
    assert revision.edges[0].relation_type == "skip"
    assert not hasattr(revision.edges[0], "source_evidence_refs")


def test_influence_edge_crossing_omitted_range_is_structural() -> None:
    omitted = (OmittedRange(start_seq=2, end_seq=2, reason="record_type_not_a_block"),)
    with pytest.raises(HighSeverityViolation) as raised:
        _freeze_pair(relation_type="influence", omitted=omitted)
    assert any(item.code == "omitted_range_skip_required" for item in raised.value.violations)
    assert all(item.code != "skip_missing_evidence" for item in raised.value.violations)


def test_freeze_hash_is_stable_and_distinct_from_claim_hash() -> None:
    first = _freeze_pair()
    second = _freeze_pair()
    assert first.content_hash == second.content_hash
    assert len(first.content_hash) == 64
    claim_hash = compute_content_hash(first.blocks, (), first.coverage)
    assert first.content_hash != claim_hash
    again = AigNativeViewRevision.from_body(first.protocol_body(), content_hash=first.content_hash)
    assert again.protocol_body() == first.protocol_body()
    assert again.edges[0].mechanism == first.edges[0].mechanism


def test_cross_participant_edge_freezes() -> None:
    blocks = [
        _block("b-1", 1, participant="agent-a", segment="seg-a"),
        _block("b-2", 2, participant="agent-b", segment="seg-b"),
    ]
    revision = freeze_native_edge_view(
        session_id=SESSION,
        task_run_id=TASK,
        view_id=VIEW,
        revision_id=REVISION,
        frozen_at=FROZEN,
        blocks=blocks,
        edges=(_edge(),),
        coverage=_coverage(blocks),
    )
    assert revision.edges[0].source_block_id == "b-1"
    assert revision.edges[0].target_block_id == "b-2"
    assert not hasattr(revision, "claims")


def test_dangling_edge_refuses_freeze() -> None:
    blocks = [_block("b-1", 1)]
    with pytest.raises(HighSeverityViolation) as raised:
        freeze_native_edge_view(
            session_id=SESSION,
            task_run_id=TASK,
            view_id=VIEW,
            revision_id=REVISION,
            frozen_at=FROZEN,
            blocks=blocks,
            edges=(_edge(target="missing"),),
            coverage=_coverage(blocks),
        )
    assert any(item.code == "dangling_edge" for item in raised.value.violations)


def test_legacy_claim_view_stays_readable_and_native_writer_cannot_update_it(
    tmp_path: Path,
) -> None:
    claim = _claim_revision()
    legacy_path = tmp_path / "claim-views.json"
    legacy = ViewSupersedeIndex(legacy_path)
    legacy.register(claim)
    before = legacy_path.read_bytes()
    audit = audit_legacy_claim_view(claim)
    assert "read_only" not in audit.to_dict()
    assert audit.revision_id == claim.revision_id
    assert audit.content_hash == claim.content_hash
    assert audit.claim_ids == ("claim-1",)
    assert "edge_ids" not in audit.to_dict()
    opened = ViewSupersedeIndex(legacy_path)
    assert opened.latest(SESSION, PURPOSE_POST_RUN_INDEX) == claim
    audits = open_legacy_claim_view_for_audit(legacy_path)
    assert audits[0].content_hash == claim.content_hash
    assert audits[0].claim_ids == ("claim-1",)
    assert legacy_path.read_bytes() == before

    with pytest.raises(NativeEdgeWriteError, match="read-only"):
        NativeEdgeViewStore(legacy_path)
    assert legacy_path.read_bytes() == before

    native_path = tmp_path / "native-views.json"
    store = NativeEdgeViewStore(native_path, legacy_index=legacy)
    with pytest.raises(NativeEdgeWriteError, match="refuses to update a frozen claim view"):
        store.write(_freeze_pair())
    assert not native_path.exists()
    assert legacy_path.read_bytes() == before
    assert ViewSupersedeIndex(legacy_path).latest(SESSION, PURPOSE_POST_RUN_INDEX) == claim


def test_native_writer_rejects_claim_objects_and_roundtrips_edges(tmp_path: Path) -> None:
    claim = InfluenceClaim(
        claim_id="claim-1",
        source_block_id="b-1",
        target_block_id="b-2",
        carried_artifact="patch",
        downstream_effect="tests passed",
        relation_type="influence",
    )
    blocks = [_block("b-1", 1), _block("b-2", 2)]
    with pytest.raises(NativeEdgeWriteError, match="InfluenceClaim") as claim_error:
        freeze_native_edge_view(
            session_id=SESSION,
            task_run_id=TASK,
            view_id=VIEW,
            revision_id=REVISION,
            frozen_at=FROZEN,
            blocks=blocks,
            edges=(claim,),  # type: ignore[arg-type]
            coverage=_coverage(blocks),
        )
    assert claim_error.value.code == "claim_mapping_forbidden"
    other_blocks = [
        _block("b-1", 1, session_id="sess-other"),
        _block("b-2", 2, session_id="sess-other"),
    ]
    other_session = freeze_native_edge_view(
        session_id="sess-other",
        task_run_id=TASK,
        view_id=VIEW,
        revision_id="rev-other",
        frozen_at=FROZEN,
        blocks=other_blocks,
        edges=(_edge(),),
        coverage=_coverage(other_blocks),
    )
    legacy_path = tmp_path / "claim-views.json"
    ViewSupersedeIndex(legacy_path).register(_claim_revision())
    before = legacy_path.read_bytes()
    store = NativeEdgeViewStore(
        tmp_path / "native-views.json",
        legacy_index=ViewSupersedeIndex(legacy_path),
    )
    store.write(other_session)
    index_document = read_protocol_document(
        json.loads((tmp_path / "native-views.json").read_text(encoding="utf-8"))
    )
    assert index_document.object_kind == "native_edge_index"
    assert index_document.revisions[0].revision.revision_id == "rev-other"
    loaded = NativeEdgeViewStore(tmp_path / "native-views.json").read("rev-other")
    assert loaded.edges[0].edge_id == other_session.edges[0].edge_id
    assert loaded.content_hash == other_session.content_hash
    assert not hasattr(loaded, "claims")
    assert legacy_path.read_bytes() == before
    with pytest.raises(ValueError, match="not a claim view"):
        InfluenceViewRevision.from_dict(loaded.protocol_document())
    with pytest.raises(ValueError, match="claim view"):
        ViewSupersedeIndex(tmp_path / "native-views.json")
    guessed = loaded.protocol_document()
    del guessed["schema_version"]
    guessed["edges"] = guessed["body"]["edges"]
    with pytest.raises(ValueError) as not_native:
        InfluenceViewRevision.from_dict(guessed)
    assert "not a claim view" not in str(not_native.value)


def test_claim_projection_rejects_native_revision(tmp_path: Path) -> None:
    native = _freeze_pair()
    registry = ProjectionRegistry(tmp_path / "cards.json")
    with pytest.raises(ValueError, match="native-edge revision cannot enter the claim projection"):
        registry.register(native)  # type: ignore[arg-type]
    assert registry.search(principal_labels=("team-a",)) == ()
    claim = _claim_revision()
    cards = registry.register(claim, acl_labels=("team-a",))
    assert {card.kind for card in cards} <= {"block", "claim"}
    pointed = {card.source_pointer.block_or_claim_id for card in cards}
    assert native.edges[0].edge_id not in pointed
    assert "claim-1" in pointed
    assert native.edges[0].edge_id not in (tmp_path / "cards.json").read_text(encoding="utf-8")


def test_old_claim_write_signature_is_unchanged() -> None:
    from openviking.session.extraction_write import freeze_extracted_view

    claim_params = inspect.signature(freeze_extracted_view).parameters
    native_params = inspect.signature(freeze_native_edge_view).parameters
    assert "claims" in claim_params
    assert "edges" not in claim_params
    assert "edges" in native_params
    assert "claims" not in native_params
    assert native_params["purpose"].default == PURPOSE_POST_RUN_INDEX


def test_old_index_builder_rejects_native_payload() -> None:
    from openviking.session.legacy_write_gate import LegacyWriteRefused

    rows = [_row(1), _row(2)]
    payload = parse_native_edge_response(
        json.dumps(_native_document(rows)),
        _native_input(_snapshot(rows)),
    )
    refused = extraction_index_builder(lambda *_args: payload, frozen_at=FROZEN)  # type: ignore[arg-type]
    with pytest.raises(LegacyWriteRefused):
        refused(SESSION, TASK, "success")
    builder = extraction_index_builder(
        lambda *_args: payload,  # type: ignore[arg-type]
        frozen_at=FROZEN,
        legacy_audit_opt_in=True,
    )
    with pytest.raises(TypeError):
        builder(SESSION, TASK, "success")


def test_freeze_native_edge_view_accepts_failure_diagnosis() -> None:
    root = _block("root", 1, role="task", block_kind="task")
    mid = _block("mid", 2)
    sink = _block("sink", 3, role="conclusion", block_kind="conclusion")
    blocks = [root, mid, sink]
    revision = freeze_native_edge_view(
        session_id=SESSION,
        task_run_id=TASK,
        view_id=VIEW,
        revision_id=REVISION,
        frozen_at=FROZEN,
        blocks=blocks,
        edges=(
            _edge("e-root", "root", "mid"),
            _edge("e-sink", "mid", "sink"),
        ),
        coverage=_coverage(blocks),
        purpose=PURPOSE_FAILURE_DIAGNOSIS,
    )
    assert revision.purpose == PURPOSE_FAILURE_DIAGNOSIS
    defaulted = _freeze_pair()
    assert defaulted.purpose == PURPOSE_POST_RUN_INDEX
    with pytest.raises(HighSeverityViolation) as raised:
        freeze_native_edge_view(
            session_id=SESSION,
            task_run_id=TASK,
            view_id=VIEW,
            revision_id=REVISION,
            frozen_at=FROZEN,
            blocks=[_block("b-1", 1), _block("b-2", 2)],
            edges=(_edge(),),
            coverage=_coverage([_block("b-1", 1), _block("b-2", 2)]),
            purpose=PURPOSE_FAILURE_DIAGNOSIS,
        )
    assert any(item.code == "missing_task_root" for item in raised.value.violations)


def test_claim_reader_uses_schema_version_not_an_edges_key() -> None:
    claim = _claim_revision().to_dict()
    claim["edges"] = [{"edge_id": "e-extra"}]
    loaded = InfluenceViewRevision.from_dict(claim)
    assert loaded.revision_id == _claim_revision().revision_id
    claim["schema_version"] = EXTRACTION_CONTRACT_NATIVE_VERSION
    with pytest.raises(ValueError, match="not a claim view"):
        InfluenceViewRevision.from_dict(claim)
    claim["schema_version"] = "not-a-known-version"
    with pytest.raises(ValueError, match="claim view"):
        InfluenceViewRevision.from_dict(claim)


def test_failure_diagnosis_uses_edges_as_the_graph() -> None:
    root = _block("root", 1, role="task", block_kind="task")
    mid = _block("mid", 2)
    sink = _block("sink", 3, role="conclusion", block_kind="conclusion")
    blocks = [root, mid, sink]
    draft = AigNativeViewDraft(
        view_id=VIEW,
        purpose=PURPOSE_FAILURE_DIAGNOSIS,
        session_id=SESSION,
        task_run_id=TASK,
        coverage=_coverage(blocks),
        blocks=blocks,
        edges=(
            _edge("e-root", "root", "mid"),
            _edge("e-sink", "mid", "sink"),
        ),
    )
    revision = draft.freeze(revision_id=REVISION, frozen_at=FROZEN)
    assert [edge.edge_id for edge in revision.edges] == ["e-root", "e-sink"]
    disconnected = AigNativeViewDraft(
        view_id=VIEW,
        purpose=PURPOSE_FAILURE_DIAGNOSIS,
        session_id=SESSION,
        task_run_id=TASK,
        coverage=_coverage(blocks),
        blocks=blocks,
        edges=(_edge("e-root", "root", "mid"),),
    )
    with pytest.raises(HighSeverityViolation) as raised:
        disconnected.freeze(revision_id=REVISION, frozen_at=FROZEN)
    assert any(item.code == "root_sink_not_weakly_connected" for item in raised.value.violations)
