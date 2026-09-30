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
from openviking.session.influence_projection import REMOVAL_SUPERSEDED, ProjectionRegistry
from openviking.session.influence_view import (
    CONSTRUCTION_MODE_ADAPTIVE,
    CONSTRUCTION_MODE_FALLBACK,
    PURPOSE_POST_RUN_INDEX,
    ViewSupersedeIndex,
)
from openviking.session.retrieval_wiring import build_retrieval_facade
from openviking.session.session import Session
from openviking.session.subgraph_forks import default_failure_grounding_position
from openviking.session.trajectory_index_hook import PROJECTIONS_FILENAME, VIEWS_FILENAME
from openviking.storage.queuefs.session_commit_msg import SessionCommitMsg
from openviking_cli.session.user_id import UserIdentifier
from tests.test_extraction_write import ExtractionPayload, _payload_for
from tests.test_trajectory_index_hook import (
    SESSION_ID,
    SESSION_URI,
    _DummyFS,
    _install_commit_fakes,
    _set_causal,
)

_OWNER_LABEL = "user:default"


def _provider(session_id: str, task_run_id: str, outcome_status: str) -> ExtractionPayload:
    del task_run_id, outcome_status
    return _payload_for(session_id)


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
        assert history[0].revision_id == ViewSupersedeIndex(
            session._viking_fs.root / VIEWS_FILENAME
        ).latest(SESSION_ID, PURPOSE_POST_RUN_INDEX).revision_id
        assert calls == []
    finally:
        set_task_tracker(None)


@pytest.mark.asyncio
async def test_phase2_with_provider_registers_adaptive_beside_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        session, _message, committed = await _phase2(
            tmp_path, monkeypatch, provider=_provider
        )
        task_id = committed["task_id"]
        fallback_id = f"post-run:{SESSION_ID}:{task_id}"
        adaptive_id = f"{fallback_id}:adaptive"
        history = _history(session._viking_fs.root)
        latest = ViewSupersedeIndex(session._viking_fs.root / VIEWS_FILENAME).latest(
            SESSION_ID, PURPOSE_POST_RUN_INDEX
        )
        assert _done(session, committed["archive_uri"])
        assert [item.revision_id for item in history] == [fallback_id, adaptive_id]
        assert [item.construction_mode for item in history] == [
            CONSTRUCTION_MODE_FALLBACK,
            CONSTRUCTION_MODE_ADAPTIVE,
        ]
        assert latest is not None
        assert latest.revision_id == adaptive_id
        assert latest.content_hash != history[0].content_hash

        facade = build_retrieval_facade(
            session=session, ctx=session.ctx, viking_fs=session._viking_fs
        )
        loaded = facade.retrieval_subgraph._revision(adaptive_id)
        assert loaded.revision_id == adaptive_id
        assert loaded.content_hash == latest.content_hash

        registry = ProjectionRegistry(session._viking_fs.root / PROJECTIONS_FILENAME)
        active = registry.search(principal_labels=(_OWNER_LABEL,))
        assert active
        assert {card.source_pointer.view_revision_id for card in active} == {adaptive_id}
        withdrawn = [
            record.card.source_pointer.view_revision_id
            for record in registry.removed_records()
            if record.reason == REMOVAL_SUPERSEDED
        ]
        assert fallback_id in withdrawn
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
        session, message, committed = await _phase2(
            tmp_path, monkeypatch, provider=_provider
        )
        done_uri = f"{committed['archive_uri']}/.done"
        del session._viking_fs.files[done_uri]
        await session.resume_queued_commit(message)
        history = _history(session._viking_fs.root)
        assert _done(session, committed["archive_uri"])
        assert len(history) == 2
        assert history[1].revision_id.endswith(":adaptive")
    finally:
        set_task_tracker(None)


@pytest.mark.asyncio
async def test_phase2_completes_when_diagnosis_sink_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[tuple] = []

    def _sink(briefs) -> None:
        received.append(tuple(briefs))
        raise RuntimeError("diagnosis sink failed")

    try:
        session, _message, committed = await _phase2(
            tmp_path, monkeypatch, provider=_provider, sink=_sink
        )
        latest = ViewSupersedeIndex(session._viking_fs.root / VIEWS_FILENAME).latest(
            SESSION_ID, PURPOSE_POST_RUN_INDEX
        )
        assert _done(session, committed["archive_uri"])
        assert latest is not None
        failures = [item for item in latest.retrieval_subgraphs if item.is_failure]
        assert len(received) == 1
        assert len(received[0]) == len(failures)
        assert len(failures) >= 1
        for brief, subgraph in zip(received[0], failures, strict=True):
            assert brief.view_revision_id == latest.revision_id
            assert brief.subgraph_id == subgraph.subgraph_id
            assert brief.grounding_position == default_failure_grounding_position(
                latest, subgraph.subgraph_id
            )
            assert brief.grounding_position[0] == "block"
            assert brief.grounding_position[1].startswith("block:")
            assert brief.fork_idempotency_key
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
