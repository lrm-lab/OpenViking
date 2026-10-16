# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Phase 2 calls the trajectory hook only when an extraction provider is set."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.service.session_service import SessionService
from openviking.service.task_tracker import set_task_tracker
from openviking.session.extraction_write import (
    NATIVE_EDGE_INDEX_FILENAME,
    NativeEdgeViewStore,
    NativeExtractionPayload,
)
from openviking.session.influence_view import (
    CONSTRUCTION_MODE_FALLBACK,
    PURPOSE_POST_RUN_INDEX,
    ViewSupersedeIndex,
)
from openviking.session.session import Session
from openviking.session.trajectory_index_hook import VIEWS_FILENAME
from openviking.storage.queuefs.session_commit_msg import SessionCommitMsg
from openviking_cli.session.user_id import UserIdentifier
from tests.test_extraction_write import ExtractionPayload, _block, _coverage, _payload_for
from tests.test_trajectory_index_hook import (
    SESSION_ID,
    SESSION_URI,
    _DummyFS,
    _install_commit_fakes,
    _set_causal,
)


def _provider(session_id: str, task_run_id: str, outcome_status: str) -> NativeExtractionPayload:
    del task_run_id, outcome_status
    blocks = [_block("b-1", [1], session_id=session_id)]
    return NativeExtractionPayload(blocks, (), _coverage(blocks))


async def _phase2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider=None,
    sink=None,
):
    """Commit (Phase 1 fallback) then resume the queued Phase 2 job."""
    from openviking.message.part import ToolPart

    _set_causal(monkeypatch, True)
    root = tmp_path / "session"
    root.mkdir()
    storage = _DummyFS(root, SESSION_URI)
    _install_commit_fakes(monkeypatch)
    monkeypatch.setattr(
        "openviking.session.session.get_openviking_config",
        lambda: SimpleNamespace(
            memory=SimpleNamespace(
                extraction_enabled=False,
                session_skill_extraction_enabled=False,
            )
        ),
    )

    async def _summary(messages, latest_archive_overview="", checkpoint_requests=None):
        del messages, latest_archive_overview, checkpoint_requests
        return "phase2 summary"

    session = Session(
        viking_fs=storage,
        session_id=SESSION_ID,
        session_uri=SESSION_URI,
        extraction_provider=provider,
        diagnosis_sink=sink,
    )
    monkeypatch.setattr(session, "_generate_archive_summary_async", _summary)
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
        committed = await session.commit_async(keep_recent_count=0)
        phase1 = await session._read_phase1_meta(committed["archive_uri"])
        message = SessionCommitMsg.from_dict(phase1["queue_message"])
        await session.resume_queued_commit(message)
        return session, message, committed
    except Exception:
        set_task_tracker(None)
        raise


def _history(root: Path):
    return ViewSupersedeIndex(root / VIEWS_FILENAME).history(SESSION_ID, PURPOSE_POST_RUN_INDEX)


def _done(session: Session, archive_uri: str) -> bool:
    return f"{archive_uri}/.done" in session._viking_fs.files


@pytest.mark.asyncio
async def test_phase2_without_provider_keeps_the_single_fallback_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []

    def _sink(briefs) -> None:
        calls.append(briefs)

    try:
        session, _message, committed = await _phase2(tmp_path, monkeypatch, sink=_sink)
        history = _history(session._viking_fs.root)
        assert _done(session, committed["archive_uri"])
        assert len(history) == 1
        assert history[0].construction_mode == CONSTRUCTION_MODE_FALLBACK
        assert history[0].revision_id == f"post-run:{SESSION_ID}:{committed['task_id']}"
        assert (
            history[0].revision_id
            == ViewSupersedeIndex(session._viking_fs.root / VIEWS_FILENAME)
            .latest(SESSION_ID, PURPOSE_POST_RUN_INDEX)
            .revision_id
        )
        assert calls == []
    finally:
        set_task_tracker(None)


@pytest.mark.asyncio
async def test_phase2_with_provider_registers_native_beside_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        session, _message, committed = await _phase2(tmp_path, monkeypatch, provider=_provider)
        task_id = committed["task_id"]
        fallback_id = f"post-run:{SESSION_ID}:{task_id}"
        history = _history(session._viking_fs.root)
        native = NativeEdgeViewStore(
            session._viking_fs.root / NATIVE_EDGE_INDEX_FILENAME
        ).revisions()
        assert _done(session, committed["archive_uri"])
        assert [item.revision_id for item in history] == [fallback_id]
        assert history[0].construction_mode == CONSTRUCTION_MODE_FALLBACK
        assert len(native) == 1
        revision = native[0]
        assert revision.revision_id == f"{fallback_id}:native"
        assert revision.content_hash
        assert revision.blocks[0].source_refs[0].sequence == 1
        assert revision.blocks[0].source_refs[0].session_id == SESSION_ID
        assert revision.coverage.inspected_refs == revision.blocks[0].source_refs
    finally:
        set_task_tracker(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["raise", "bad-return"])
async def test_phase2_provider_failure_leaves_fallback_as_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    def _raise(session_id: str, task_run_id: str, outcome_status: str) -> ExtractionPayload:
        del session_id, task_run_id, outcome_status
        raise RuntimeError("extraction provider failed")

    def _bad(session_id: str, task_run_id: str, outcome_status: str):
        del session_id, task_run_id, outcome_status
        return {"not": "a payload"}

    try:
        session, _message, committed = await _phase2(
            tmp_path,
            monkeypatch,
            provider=_raise if failure == "raise" else _bad,
        )
        history = _history(session._viking_fs.root)
        latest = ViewSupersedeIndex(session._viking_fs.root / VIEWS_FILENAME).latest(
            SESSION_ID, PURPOSE_POST_RUN_INDEX
        )
        assert _done(session, committed["archive_uri"])
        assert len(history) == 1
        assert history[0].construction_mode == CONSTRUCTION_MODE_FALLBACK
        assert history[0].revision_id == f"post-run:{SESSION_ID}:{committed['task_id']}"
        assert latest is not None
        assert latest.revision_id == history[0].revision_id
        assert latest.content_hash == history[0].content_hash
    finally:
        set_task_tracker(None)


@pytest.mark.asyncio
async def test_phase2_replay_does_not_register_a_third_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        session, message, committed = await _phase2(tmp_path, monkeypatch, provider=_provider)
        done_uri = f"{committed['archive_uri']}/.done"
        native_path = session._viking_fs.root / NATIVE_EDGE_INDEX_FILENAME
        before = native_path.read_bytes()
        del session._viking_fs.files[done_uri]
        await session.resume_queued_commit(message)
        history = _history(session._viking_fs.root)
        assert _done(session, committed["archive_uri"])
        assert len(history) == 1
        native = NativeEdgeViewStore(native_path).revisions()
        assert len(native) == 1
        assert native[0].revision_id.endswith(":native")
        assert native_path.read_bytes() == before
    finally:
        set_task_tracker(None)


@pytest.mark.asyncio
async def test_phase2_refuses_legacy_claims_without_emitting_diagnosis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[tuple] = []

    def _legacy(session_id: str, task_run_id: str, outcome_status: str) -> ExtractionPayload:
        del task_run_id, outcome_status
        return _payload_for(session_id)

    def _sink(briefs) -> None:
        received.append(tuple(briefs))
        raise RuntimeError("diagnosis sink must not receive rejected claims")

    try:
        session, _message, committed = await _phase2(
            tmp_path, monkeypatch, provider=_legacy, sink=_sink
        )
        assert _done(session, committed["archive_uri"])
        history = _history(session._viking_fs.root)
        assert len(history) == 1
        assert history[0].construction_mode == CONSTRUCTION_MODE_FALLBACK
        native = NativeEdgeViewStore(session._viking_fs.root / NATIVE_EDGE_INDEX_FILENAME)
        assert native.revisions() == ()
        assert received == []
    finally:
        set_task_tracker(None)


def test_session_service_passes_extraction_provider() -> None:
    seen: list[tuple[str, str, str]] = []

    def provider(session_id: str, task_run_id: str, outcome_status: str) -> ExtractionPayload:
        seen.append((session_id, task_run_id, outcome_status))
        return _payload_for(session_id)

    service = SessionService(viking_fs=object())
    assert service._extraction_provider is None
    service.set_extraction_provider(provider)
    ctx = RequestContext(user=UserIdentifier("default", "default"), role=Role.ROOT)
    session = service.session(ctx, session_id="sess-provider")
    assert session._extraction_provider is provider
    service.set_extraction_provider(None)
    cleared = service.session(ctx, session_id="sess-cleared")
    assert cleared._extraction_provider is None
    assert seen == []

    briefs: list[object] = []
    sink = briefs.append
    service.set_diagnosis_sink(sink)
    wired = service.session(ctx, session_id="sess-sink")
    assert wired._diagnosis_sink is sink
    service.set_diagnosis_sink(None)
    unwired = service.session(ctx, session_id="sess-no-sink")
    assert unwired._diagnosis_sink is None
