# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Write-side freeze of an extraction payload into one adaptive view."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from openviking.session.ao_ledger import AOLedger
from openviking.session.extraction_write import (
    ExtractionPayload,
    degrade_to_fallback_view,
    extraction_index_builder,
    freeze_extracted_view,
)
from openviking.session.influence_view import (
    CONSTRUCTION_MODE_NATIVE_SINGLE_PASS,
    CONSTRUCTION_MODE_FALLBACK,
    DETERMINISTIC_FALLBACK_REASON,
    PURPOSE_POST_RUN_INDEX,
    AORange,
    CoverageManifest,
    HighSeverityViolation,
    InfluenceBlock,
    InfluenceClaim,
    InfluenceEvidenceRef,
    InfluenceViewRevision,
    RetrievalSubgraph,
    SegmentRef,
    SourceRef,
    ViewSupersedeIndex,
)
from openviking.session.trajectory_index import (
    IndexJob,
    TrajectoryIndexService,
    WorkspacePolicy,
)
from openviking.session.trajectory_index_hook import (
    VIEWS_FILENAME,
    maybe_index_session_trajectory,
)
from openviking_cli.utils.config.memory_config import MemoryConfig, TrajectoryIndexConfig

SESSION = "sess-extract-alpha"
TASK = "task-run-extract-alpha"
VIEW = "view-extract-omega"
REVISION = "rev-extract-omega"
FROZEN = "2026-09-29T12:00:00.000Z"
HOOK_SESSION = "session-extract-1"
HOOK_TASK = "task-extract-1"


def _ref(
    sequence: int,
    *,
    session_id: str,
    participant_id: str = "agent-a",
    segment_id: str = "seg-a",
) -> SourceRef:
    return SourceRef(
        ao_id=f"ao-{sequence}",
        sequence=sequence,
        participant_id=participant_id,
        segment_id=segment_id,
        session_id=session_id,
        content_hash=f"hash-{sequence}",
    )


def _block(
    block_id: str,
    sequences: list[int],
    *,
    session_id: str,
    participant_id: str = "agent-a",
    segment_id: str = "seg-a",
    summary: str = "did the step",
) -> InfluenceBlock:
    refs = tuple(
        _ref(
            sequence,
            session_id=session_id,
            participant_id=participant_id,
            segment_id=segment_id,
        )
        for sequence in sequences
    )
    return InfluenceBlock(
        block_id=block_id,
        role="agent_action",
        participant_id=participant_id,
        segment_ref=SegmentRef(segment_id=segment_id, session_id=session_id),
        ao_start_seq=min(sequences),
        ao_end_seq=max(sequences),
        summary=summary,
        input="in",
        output="out",
        authorship=participant_id,
        granularity_reason="one tool call",
        source_content_hashes=tuple(ref.content_hash for ref in refs),
        source_refs=refs,
    )


def _coverage(blocks: list[InfluenceBlock]) -> CoverageManifest:
    refs = tuple(ref for block in blocks for ref in block.source_refs)
    sequences = [ref.sequence for ref in refs]
    return CoverageManifest(
        total_ao_range=AORange(start_seq=min(sequences), end_seq=max(sequences)),
        inspected_refs=refs,
        included_ranges=tuple(
            AORange(
                start_seq=block.ao_start_seq,
                end_seq=block.ao_end_seq,
                segment_id=block.segment_ref.segment_id,
            )
            for block in blocks
        ),
        omitted_ranges=(),
        snapshot_watermark="wm-extract",
    )


def _claim(claim_id: str, source: str, target: str) -> InfluenceClaim:
    return InfluenceClaim(
        claim_id=claim_id,
        source_block_id=source,
        target_block_id=target,
        carried_artifact="patch",
        downstream_effect="tests passed",
        relation_type="influence",
    )


def _evidence(sequence: int, session_id: str) -> InfluenceEvidenceRef:
    return InfluenceEvidenceRef(
        ao_id=f"ao-{sequence}",
        source_session_id=session_id,
        source_archive_id=None,
        archive_commit_watermark=None,
        source_sequence=sequence,
        source_read_snapshot_watermark="wm-extract",
        evidence_role="contemporaneous_basis",
        evidence_content_hash=f"hash-{sequence}",
        captured_state="committed",
    )


def _subgraph(
    subgraph_id: str,
    block_ids: tuple[str, ...],
    claim_ids: tuple[str, ...] = (),
    *,
    failure_marked: bool = False,
    failure_evidence_refs: tuple[InfluenceEvidenceRef, ...] = (),
) -> RetrievalSubgraph:
    return RetrievalSubgraph(
        subgraph_id=subgraph_id,
        block_ids=block_ids,
        claim_ids=claim_ids,
        failure_marked=failure_marked,
        failure_evidence_refs=failure_evidence_refs,
    )


def _happy_parts(
    session_id: str,
) -> tuple[
    list[InfluenceBlock],
    list[InfluenceClaim],
    CoverageManifest,
    tuple[RetrievalSubgraph, ...],
]:
    early = _block("b-script", [1, 2], session_id=session_id, segment_id="seg-a")
    late = _block(
        "b-rate",
        [4, 5],
        session_id=session_id,
        participant_id="agent-b",
        segment_id="seg-b",
    )
    note = _block(
        "b-note",
        [6],
        session_id=session_id,
        participant_id="agent-c",
        segment_id="seg-c",
        summary="noted the rate",
    )
    cutover = _claim("c-cutover", "b-script", "b-rate")
    follow = _claim("c-note", "b-rate", "b-note")
    subgraphs = (
        _subgraph("sg-cutover", ("b-script", "b-rate"), ("c-cutover",)),
        _subgraph(
            "sg-note",
            ("b-rate", "b-note"),
            ("c-note",),
            failure_marked=True,
            failure_evidence_refs=(_evidence(5, session_id),),
        ),
    )
    return [early, late, note], [cutover, follow], _coverage([early, late, note]), subgraphs


def _freeze_kwargs(
    blocks: list[InfluenceBlock],
    claims: list[InfluenceClaim],
    coverage: CoverageManifest,
    subgraphs: tuple[RetrievalSubgraph, ...],
    *,
    session_id: str = SESSION,
    task_run_id: str = TASK,
    view_id: str = VIEW,
    revision_id: str = REVISION,
) -> dict:
    return {
        "session_id": session_id,
        "task_run_id": task_run_id,
        "view_id": view_id,
        "revision_id": revision_id,
        "frozen_at": FROZEN,
        "blocks": blocks,
        "claims": claims,
        "coverage": coverage,
        "retrieval_subgraphs": subgraphs,
        "legacy_audit_opt_in": True,
    }


def _payload_for(session_id: str) -> ExtractionPayload:
    blocks, claims, coverage, subgraphs = _happy_parts(session_id)
    return ExtractionPayload(blocks, claims, coverage, subgraphs)


def _policy() -> WorkspacePolicy:
    return WorkspacePolicy(max_builds_per_window=8, window_seconds=60.0)


def _job(session_id: str = SESSION, task_run_id: str = TASK) -> IndexJob:
    return IndexJob(
        job_id=f"post-run:{session_id}:{task_run_id}",
        session_id=session_id,
        task_run_id=task_run_id,
        priority=2,
        outcome_status="unspecified",
        enqueued_at=FROZEN,
    )


def _service(root: Path, provider) -> TrajectoryIndexService:
    return TrajectoryIndexService(
        root / "trajectory-index.json",
        ViewSupersedeIndex(root / VIEWS_FILENAME),
        builder=extraction_index_builder(
            provider, frozen_at=FROZEN, legacy_audit_opt_in=True
        ),
    )


def _causal_config() -> MemoryConfig:
    return MemoryConfig(
        skill_trajectory_mode="causal",
        trajectory_index=TrajectoryIndexConfig(),
    )


class _DummyFS:
    def __init__(self, root: Path, session_uri: str) -> None:
        self.root = Path(root)
        self.session_uri = session_uri

    def _uri_to_path(self, uri: str, ctx=None) -> str:
        del ctx
        if uri == self.session_uri:
            return str(self.root)
        prefix = self.session_uri + "/"
        if uri.startswith(prefix):
            return str(self.root / uri[len(prefix) :])
        return str(self.root)


def _bound_session(root: Path, session_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        session_id=session_id,
        _session_uri=f"viking://user/default/sessions/{session_id}",
        ctx=None,
        _viking_fs=_DummyFS(root, f"viking://user/default/sessions/{session_id}"),
    )


def _seed_ledger(root: Path, session_id: str) -> None:
    ledger = AOLedger(root, session_id)
    first = ledger.append(
        {"tool": "read_file", "segment_id": "seg-a", "participant_id": "agent-a"},
        {"kind": "text", "summary": ["read_file"]},
    )
    second = ledger.append(
        {"tool": "search", "segment_id": "seg-b", "participant_id": "agent-b"},
        {"kind": "text", "summary": ["search"]},
    )
    assert first is not None and second is not None


def _illegal(case: str, session_id: str):
    early = _block("b-script", [1, 2], session_id=session_id, segment_id="seg-a")
    late = _block(
        "b-rate",
        [4, 5],
        session_id=session_id,
        participant_id="agent-b",
        segment_id="seg-b",
    )
    claim = _claim("c-cutover", "b-script", "b-rate")
    coverage = _coverage([early, late])
    if case == "unknown_block":
        subgraphs = (
            _subgraph("sg-missing", ("b-script", "b-rate", "missing-block"), ("c-cutover",)),
        )
        return [early, late], [claim], coverage, subgraphs, "subgraph_unknown_member"
    if case == "claim_endpoint_outside":
        subgraphs = (_subgraph("sg-half", ("b-script",), ("c-cutover",)),)
        return [early, late], [claim], coverage, subgraphs, "subgraph_claim_endpoint_outside"
    if case == "block_limit":
        blocks = [
            _block(f"b{index}", [index], session_id=session_id, segment_id=f"seg-{index}")
            for index in range(1, 14)
        ]
        subgraphs = (_subgraph("sg-over", tuple(block.block_id for block in blocks)),)
        return blocks, [], _coverage(blocks), subgraphs, "subgraph_block_limit"
    if case == "marker_without_evidence":
        subgraphs = (
            _subgraph(
                "sg-bare",
                ("b-script", "b-rate"),
                ("c-cutover",),
                failure_marked=True,
            ),
        )
        return [early, late], [claim], coverage, subgraphs, "subgraph_failure_evidence"
    if case == "evidence_without_marker":
        subgraphs = (
            _subgraph(
                "sg-extra",
                ("b-script", "b-rate"),
                ("c-cutover",),
                failure_evidence_refs=(_evidence(5, session_id),),
            ),
        )
        return [early, late], [claim], coverage, subgraphs, "subgraph_failure_evidence"
    raise AssertionError(case)


def test_extracted_payload_freezes_adaptive_view_roundtrip() -> None:
    blocks, claims, coverage, subgraphs = _happy_parts(SESSION)
    revision = freeze_extracted_view(**_freeze_kwargs(blocks, claims, coverage, subgraphs))
    assert revision.construction_mode == CONSTRUCTION_MODE_NATIVE_SINGLE_PASS
    assert revision.purpose == PURPOSE_POST_RUN_INDEX
    assert [block.block_id for block in revision.blocks] == ["b-script", "b-rate", "b-note"]
    assert [claim.claim_id for claim in revision.claims] == ["c-cutover", "c-note"]
    by_id = {item.subgraph_id: item for item in revision.retrieval_subgraphs}
    assert set(by_id) == {"sg-cutover", "sg-note"}
    assert by_id["sg-cutover"].is_failure is False
    assert by_id["sg-cutover"].failure_marked is False
    assert by_id["sg-note"].failure_marked is True
    assert by_id["sg-note"].is_failure is True
    assert by_id["sg-note"].failure_evidence_refs[0].ao_id == "ao-5"
    assert InfluenceViewRevision.from_dict(revision.to_dict()) == revision


@pytest.mark.parametrize(
    "case",
    [
        "unknown_block",
        "claim_endpoint_outside",
        "block_limit",
        "marker_without_evidence",
        "evidence_without_marker",
    ],
)
def test_illegal_extraction_raises_and_does_not_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    def _forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("extraction freeze must not degrade to fallback")

    monkeypatch.setattr(
        "openviking.session.extraction_write.build_minimal_fallback_view",
        _forbidden,
    )
    blocks, claims, coverage, subgraphs, code = _illegal(case, SESSION)
    kwargs = _freeze_kwargs(blocks, claims, coverage, subgraphs)
    with pytest.raises(HighSeverityViolation) as caught:
        freeze_extracted_view(**kwargs)
    assert {item.code for item in caught.value.violations} == {code}

    payload = ExtractionPayload(blocks, claims, coverage, subgraphs)

    def _provider(session_id: str, task_run_id: str, outcome_status: str) -> ExtractionPayload:
        assert session_id == SESSION
        assert task_run_id == TASK
        assert outcome_status == "unspecified"
        return payload

    service = _service(tmp_path, _provider)
    service.enqueue(_job())
    with pytest.raises(HighSeverityViolation) as via_service:
        service.process_next(_policy())
    assert {item.code for item in via_service.value.violations} == {code}
    assert not (tmp_path / VIEWS_FILENAME).exists()
    assert not (tmp_path / f"{VIEWS_FILENAME}.tmp").exists()
    history = ViewSupersedeIndex(tmp_path / VIEWS_FILENAME).history(SESSION, PURPOSE_POST_RUN_INDEX)
    assert history == []


def test_caller_degrades_explicitly_after_rejected_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    real = degrade_to_fallback_view

    def _spy(session, *, clock=None):
        calls.append(session)
        return real(session, clock=clock)

    monkeypatch.setattr(
        "tests.test_extraction_write.degrade_to_fallback_view",
        _spy,
    )
    blocks, claims, coverage, subgraphs, code = _illegal("unknown_block", SESSION)
    with pytest.raises(HighSeverityViolation) as caught:
        freeze_extracted_view(**_freeze_kwargs(blocks, claims, coverage, subgraphs))
    assert {item.code for item in caught.value.violations} == {code}
    assert calls == []

    revision = degrade_to_fallback_view(
        {
            "session_id": SESSION,
            "task_run_id": TASK,
            "purpose": PURPOSE_POST_RUN_INDEX,
            "view_id": VIEW,
            "revision_id": REVISION,
            "frozen_at": FROZEN,
            "snapshot_watermark": "wm-fallback",
            "ao_records": [
                {
                    "ao_id": "ao-1",
                    "sequence": 1,
                    "participant_id": "agent-a",
                    "segment_id": "seg-a",
                    "session_id": SESSION,
                    "content_hash": "hash-1",
                }
            ],
        }
    )
    assert len(calls) == 1
    assert revision.construction_mode == CONSTRUCTION_MODE_FALLBACK
    assert revision.retrieval_subgraphs == ()
    assert revision.blocks[0].granularity_reason == DETERMINISTIC_FALLBACK_REASON


@pytest.mark.parametrize(
    ("field", "subgraph_id"),
    [
        ("session_id", f"sg-{SESSION}"),
        ("view_id", f"sg-{VIEW}"),
        ("task_run_id", f"sg-{TASK}"),
        ("revision_id", f"sg-{REVISION}"),
    ],
)
def test_subgraph_id_embedding_identity_is_rejected(
    tmp_path: Path, field: str, subgraph_id: str
) -> None:
    blocks, claims, coverage, subgraphs = _happy_parts(SESSION)
    first = subgraphs[0]
    dirty = RetrievalSubgraph(
        subgraph_id=subgraph_id,
        block_ids=first.block_ids,
        claim_ids=first.claim_ids,
        failure_marked=first.failure_marked,
        failure_evidence_refs=first.failure_evidence_refs,
    )
    kwargs = _freeze_kwargs(blocks, claims, coverage, (dirty, subgraphs[1]))
    with pytest.raises(ValueError, match=f"must not embed {field}") as caught:
        freeze_extracted_view(**kwargs)
    assert type(caught.value) is ValueError

    if field != "session_id":
        return
    payload = ExtractionPayload(blocks, claims, coverage, (dirty, subgraphs[1]))
    service = _service(tmp_path, lambda *_args: payload)
    service.enqueue(_job())
    with pytest.raises(ValueError, match=f"must not embed {field}") as via_service:
        service.process_next(_policy())
    assert type(via_service.value) is ValueError
    assert not (tmp_path / VIEWS_FILENAME).exists()


def test_serialized_extraction_omits_subtask_wording() -> None:
    blocks, claims, coverage, subgraphs = _happy_parts(SESSION)
    revision = freeze_extracted_view(**_freeze_kwargs(blocks, claims, coverage, subgraphs))
    text = json.dumps(revision.to_dict())
    assert "subtask" not in text
    assert "subtask_id" not in text


def test_unconfigured_provider_keeps_ledger_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fallback_calls: list[object] = []
    freeze_calls: list[object] = []
    import openviking.session.extraction_write as write_mod
    import openviking.session.trajectory_index as index_mod

    real_fallback = index_mod.build_minimal_fallback_view
    real_freeze = write_mod.freeze_extracted_view

    def _fallback(session, *, clock=None):
        fallback_calls.append(session)
        return real_fallback(session, clock=clock)

    def _freeze(**kwargs):
        freeze_calls.append(kwargs)
        return real_freeze(**kwargs)

    monkeypatch.setattr(index_mod, "build_minimal_fallback_view", _fallback)
    monkeypatch.setattr(write_mod, "freeze_extracted_view", _freeze)
    _seed_ledger(tmp_path, HOOK_SESSION)
    maybe_index_session_trajectory(
        _bound_session(tmp_path, HOOK_SESSION),
        HOOK_TASK,
        config=_causal_config(),
    )
    revision = ViewSupersedeIndex(tmp_path / VIEWS_FILENAME).latest(
        HOOK_SESSION, PURPOSE_POST_RUN_INDEX
    )
    assert revision is not None
    assert freeze_calls == []
    assert len(fallback_calls) == 1
    assert revision.construction_mode == CONSTRUCTION_MODE_FALLBACK
    assert revision.retrieval_subgraphs == ()
    assert revision.task_run_id == HOOK_TASK
    assert revision.revision_id == f"post-run:{HOOK_SESSION}:{HOOK_TASK}"
    assert [block.segment_ref.segment_id for block in revision.blocks] == ["seg-a", "seg-b"]
    assert all(
        block.granularity_reason == DETERMINISTIC_FALLBACK_REASON for block in revision.blocks
    )
    assert (tmp_path / VIEWS_FILENAME).is_file()


def test_freeze_extracted_view_refuses_unless_legacy_audit_opt_in() -> None:
    from openviking.session.legacy_write_gate import LegacyWriteRefused

    blocks, claims, coverage, subgraphs = _happy_parts(SESSION)
    kwargs = _freeze_kwargs(blocks, claims, coverage, subgraphs)
    kwargs["legacy_audit_opt_in"] = False
    with pytest.raises(LegacyWriteRefused):
        freeze_extracted_view(**kwargs)
    refused = extraction_index_builder(lambda *_args: _payload_for(SESSION), frozen_at=FROZEN)
    with pytest.raises(LegacyWriteRefused):
        refused(SESSION, TASK, "unspecified")
    allowed = freeze_extracted_view(**_freeze_kwargs(blocks, claims, coverage, subgraphs))
    assert allowed.claims


def test_configured_claim_provider_does_not_freeze_a_claim_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import openviking.session.trajectory_index as index_mod

    def _forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("configured provider must not build the ledger fallback")

    monkeypatch.setattr(index_mod, "build_minimal_fallback_view", _forbidden)
    seen: list[tuple[str, str, str]] = []

    def _provider(session_id: str, task_run_id: str, outcome_status: str) -> ExtractionPayload:
        seen.append((session_id, task_run_id, outcome_status))
        return _payload_for(session_id)

    _seed_ledger(tmp_path, HOOK_SESSION)
    maybe_index_session_trajectory(
        _bound_session(tmp_path, HOOK_SESSION),
        HOOK_TASK,
        config=_causal_config(),
        extraction_provider=_provider,
    )
    assert seen == [(HOOK_SESSION, HOOK_TASK, "unspecified")]
    assert not (tmp_path / VIEWS_FILENAME).exists()
    assert not (tmp_path / "native-edges.json").exists()


def test_repeat_extraction_does_not_register_a_second_view(tmp_path: Path) -> None:
    def _provider(session_id: str, task_run_id: str, outcome_status: str) -> ExtractionPayload:
        del task_run_id, outcome_status
        return _payload_for(session_id)

    _seed_ledger(tmp_path, HOOK_SESSION)
    session = _bound_session(tmp_path, HOOK_SESSION)
    config = _causal_config()
    maybe_index_session_trajectory(session, HOOK_TASK, config=config, extraction_provider=_provider)
    maybe_index_session_trajectory(session, HOOK_TASK, config=config, extraction_provider=_provider)
    maybe_index_session_trajectory(
        session, "task-extract-2", config=config, extraction_provider=_provider
    )
    assert not (tmp_path / VIEWS_FILENAME).exists()
    assert not (tmp_path / "native-edges.json").exists()


@pytest.mark.asyncio
async def test_provider_error_does_not_fail_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openviking.message.part import ToolPart
    from openviking.service.task_tracker import set_task_tracker
    from openviking.session.session import Session
    from openviking.session.trajectory_index_hook import maybe_index_session_trajectory as real_hook
    from tests.test_trajectory_index_hook import (
        SESSION_ID,
        SESSION_URI,
        _DummyFS,
        _install_commit_fakes,
        _set_causal,
    )

    warnings: list[tuple[str, BaseException | None]] = []

    def _warning(message: str, *args: object, **kwargs: object) -> None:
        del args, kwargs
        current = sys.exc_info()[1]
        warnings.append((message, current if isinstance(current, BaseException) else None))

    def _raising(session_id: str, task_run_id: str, outcome_status: str) -> ExtractionPayload:
        del session_id, task_run_id, outcome_status
        raise RuntimeError("extraction provider failed")

    def _forward(session, task_id, *, config=None, extraction_provider=None):
        del extraction_provider
        return real_hook(session, task_id, config=config, extraction_provider=_raising)

    _set_causal(monkeypatch, True)
    monkeypatch.setattr("openviking.session.trajectory_index_hook.logger.warning", _warning)
    monkeypatch.setattr("openviking.session.session.maybe_index_session_trajectory", _forward)
    root = tmp_path / "commit"
    root.mkdir()
    storage = _DummyFS(root, SESSION_URI)
    _install_commit_fakes(monkeypatch)
    session = Session(viking_fs=storage, session_id=SESSION_ID, session_uri=SESSION_URI)
    try:
        await session.add_messages_async(
            [
                {
                    "role": "assistant",
                    "parts": [
                        ToolPart(
                            tool_id="t1",
                            tool_name="read_file",
                            tool_input={"path": "a.md"},
                            tool_output="tool body",
                            tool_status="completed",
                        )
                    ],
                }
            ]
        )
        result = await session.commit_async(keep_recent_count=0)
    finally:
        set_task_tracker(None)

    assert result.get("archived") is True
    assert result.get("status") == "accepted"
    assert not (root / VIEWS_FILENAME).exists()
    assert any(
        message == "trajectory index hook failed"
        and isinstance(exc, RuntimeError)
        and "extraction provider failed" in str(exc)
        for message, exc in warnings
    )
