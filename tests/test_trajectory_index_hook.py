# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Commit hook that freezes a post-run view from the real AO ledger."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.message import TextPart, ToolPart
from openviking.session.ao_ledger import LEDGER_FILENAME, AOLedger
from openviking.session.ao_ledger_reader import MAX_PAGE_LIMIT
from openviking.session.ao_ledger_recorder import ATTRIBUTION_FILENAME
from openviking.session.influence_view import (
    CONSTRUCTION_MODE_FALLBACK,
    DETERMINISTIC_FALLBACK_REASON,
    PURPOSE_POST_RUN_INDEX,
    ViewSupersedeIndex,
)
from openviking.session.session import Session
from openviking.session.trajectory_index import (
    SKIP_BUDGET_EXHAUSTED,
    SKIP_PAUSED,
    SkippedResult,
    TrajectoryIndexService,
)
from openviking.session.trajectory_index_hook import (
    TRAJECTORY_INDEX_FILENAME,
    VIEWS_FILENAME,
    maybe_index_session_trajectory,
)
from openviking_cli.utils.config.memory_config import (
    MemoryConfig,
    TrajectoryIndexConfig,
)

SESSION_ID = "session-1"
SESSION_URI = "viking://user/default/sessions/session-1"


class _DummyFS:
    def __init__(self, root: Path, session_uri: str) -> None:
        self.root = Path(root)
        self.session_uri = session_uri
        self.files: dict[str, str] = {}
        self._async_agfs = AsyncMock()

    def _uri_to_path(self, uri: str, ctx=None) -> str:
        if uri == self.session_uri:
            return str(self.root)
        prefix = self.session_uri + "/"
        if uri.startswith(prefix):
            return str(self.root / uri[len(prefix) :])
        return str(self.root)

    async def read_file(self, uri, ctx=None):
        if uri not in self.files:
            raise FileNotFoundError(uri)
        return self.files[uri]

    async def write_file(self, uri, content, ctx=None, lease_ref=None):
        self.files[uri] = content

    async def append_file(self, uri, content, ctx=None):
        self.files[uri] = self.files.get(uri, "") + content

    async def exists(self, uri, ctx=None):
        return uri in self.files or any(path.startswith(f"{uri}/") for path in self.files)

    async def ls(self, uri, ctx=None):
        prefix = f"{uri}/"
        names = {
            path[len(prefix) :].split("/")[0] for path in self.files if path.startswith(prefix)
        }
        return [{"name": name} for name in sorted(names)]


class _TaskStore:
    def __init__(self):
        self.tasks = {}

    async def create(self, task):
        self.tasks[task.task_id] = task

    async def update(self, task):
        self.tasks[task.task_id] = task

    async def get(self, task_id, *, account_id=None, user_id=None):
        return None

    async def list(self, account_id, *, user_id=None):
        return []

    async def delete(self, task_id, *, account_id, user_id=None):
        self.tasks.pop(task_id, None)


def _causal_config(**index: object) -> MemoryConfig:
    return MemoryConfig(
        skill_trajectory_mode="causal",
        trajectory_index=TrajectoryIndexConfig(**index),
    )


def _bound_session(root: Path, session_id: str = SESSION_ID) -> SimpleNamespace:
    return SimpleNamespace(
        session_id=session_id,
        _session_uri=f"viking://user/default/sessions/{session_id}",
        ctx=None,
        _viking_fs=_DummyFS(root, f"viking://user/default/sessions/{session_id}"),
    )


def _forbid_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbidden(_job: object) -> dict:
        raise AssertionError("synthetic fallback session is not a production path")

    monkeypatch.setattr(
        "openviking.session.trajectory_index._synthetic_fallback_session",
        _forbidden,
    )


def _append_segment(
    ledger: AOLedger,
    *,
    segment_id: str,
    participant_id: str,
    tool: str,
) -> str:
    record = ledger.append(
        {
            "tool": tool,
            "segment_id": segment_id,
            "participant_id": participant_id,
        },
        {"kind": "text", "summary": [tool]},
    )
    assert record is not None
    return record.ao_id


def _revision(root: Path, session_id: str = SESSION_ID):
    return ViewSupersedeIndex(root / VIEWS_FILENAME).latest(session_id, PURPOSE_POST_RUN_INDEX)


def _install_commit_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    from openviking.service.task_tracker import TaskTracker, set_task_tracker

    tracker = TaskTracker(_TaskStore())
    monkeypatch.setattr("openviking.session.session._enabled_memory_types", lambda: set())
    monkeypatch.setattr("openviking.service.task_tracker.get_task_tracker", lambda: tracker)
    monkeypatch.setattr(
        "openviking.storage.queuefs.get_queue_manager",
        lambda: SimpleNamespace(enqueue=AsyncMock()),
    )
    set_task_tracker(tracker)


def _set_causal(monkeypatch: pytest.MonkeyPatch, causal: bool) -> None:
    monkeypatch.setattr(
        "openviking.session.ao_ledger_recorder.is_causal_mode_enabled",
        lambda *_args, **_kwargs: causal,
    )
    monkeypatch.setattr(
        "openviking.session.trajectory_index_hook.is_causal_mode_enabled",
        lambda *_args, **_kwargs: causal,
    )


@pytest.mark.asyncio
async def test_session_commit_seam_causal_vs_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Causal commit enqueues and freezes a view; legacy writes no hook files."""
    from openviking.service.task_tracker import set_task_tracker

    async def _commit_once(*, causal: bool, root: Path) -> dict:
        _set_causal(monkeypatch, causal)
        _forbid_synthetic(monkeypatch)
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
            assert result.get("archived") is True
            return result
        finally:
            set_task_tracker(None)

    causal_root = tmp_path / "causal"
    causal_root.mkdir()
    result = await _commit_once(causal=True, root=causal_root)
    ledger_lines = [
        json.loads(line)
        for line in (causal_root / LEDGER_FILENAME).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(ledger_lines) == 1
    revision = _revision(causal_root)
    assert revision is not None
    assert revision.task_run_id == result["task_id"]
    assert revision.purpose == PURPOSE_POST_RUN_INDEX
    assert revision.construction_mode == CONSTRUCTION_MODE_FALLBACK
    assert len(revision.blocks) == 1
    refs = revision.blocks[0].source_refs
    assert [ref.ao_id for ref in refs] == [ledger_lines[0]["ao_id"]]
    assert refs[0].participant_id != "fallback-agent"
    assert revision.blocks[0].granularity_reason == DETERMINISTIC_FALLBACK_REASON
    queue = TrajectoryIndexService(
        causal_root / TRAJECTORY_INDEX_FILENAME,
        ViewSupersedeIndex(causal_root / "queue-views.json"),
        fallback_session=lambda _job: {},
    )
    assert queue.pending_jobs() == ()

    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    await _commit_once(causal=False, root=legacy_root)
    assert list(legacy_root.rglob("*")) == []
    assert not (legacy_root / LEDGER_FILENAME).exists()
    assert not (legacy_root / "history" / "archive_001" / ATTRIBUTION_FILENAME).exists()
    assert not (legacy_root / VIEWS_FILENAME).exists()
    assert not (legacy_root / TRAJECTORY_INDEX_FILENAME).exists()


def test_real_ledger_segments_are_not_one_synthetic_ao(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_synthetic(monkeypatch)
    ledger = AOLedger(tmp_path, SESSION_ID)
    first = _append_segment(ledger, segment_id="seg-a", participant_id="agent-a", tool="read_file")
    second = _append_segment(ledger, segment_id="seg-a", participant_id="agent-a", tool="edit_file")
    third = _append_segment(ledger, segment_id="seg-b", participant_id="agent-b", tool="search")

    maybe_index_session_trajectory(_bound_session(tmp_path), "task-1", config=_causal_config())

    revision = _revision(tmp_path)
    assert revision is not None
    assert revision.task_run_id == "task-1"
    assert [block.segment_ref.segment_id for block in revision.blocks] == ["seg-a", "seg-b"]
    assert [len(block.source_refs) for block in revision.blocks] == [2, 1]
    assert [ref.ao_id for block in revision.blocks for ref in block.source_refs] == [
        first,
        second,
        third,
    ]
    assert all(
        block.granularity_reason == DETERMINISTIC_FALLBACK_REASON for block in revision.blocks
    )
    assert revision.coverage.snapshot_watermark.startswith("ao-ledger:")
    assert "fallback-agent" not in {
        ref.participant_id for block in revision.blocks for ref in block.source_refs
    }


def test_ledger_pages_past_the_first_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_synthetic(monkeypatch)
    ledger = AOLedger(tmp_path, SESSION_ID)
    expected: list[str] = []
    for index in range(MAX_PAGE_LIMIT + 1):
        record = ledger.append(
            {"tool": "read_file", "path": str(index)},
            {"kind": "text", "summary": [str(index)]},
        )
        assert record is not None
        expected.append(record.ao_id)

    maybe_index_session_trajectory(_bound_session(tmp_path), "task-pages", config=_causal_config())

    revision = _revision(tmp_path)
    assert revision is not None
    found = [ref.ao_id for block in revision.blocks for ref in block.source_refs]
    assert found == expected
    assert len(revision.blocks) == MAX_PAGE_LIMIT + 1


def test_paused_leaves_the_job_queued(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_synthetic(monkeypatch)
    ledger = AOLedger(tmp_path, SESSION_ID)
    _append_segment(ledger, segment_id="seg-a", participant_id="agent-a", tool="read_file")
    seen: list[SkippedResult] = []
    original = TrajectoryIndexService.process_next

    def _spy(self, policy):
        result = original(self, policy)
        seen.append(result)
        return result

    monkeypatch.setattr(TrajectoryIndexService, "process_next", _spy)
    maybe_index_session_trajectory(
        _bound_session(tmp_path),
        "task-1",
        config=_causal_config(paused=True),
    )

    assert len(seen) == 1
    assert isinstance(seen[0], SkippedResult)
    assert seen[0].reason == SKIP_PAUSED
    assert seen[0].projection is None
    assert _revision(tmp_path) is None
    service = TrajectoryIndexService(
        tmp_path / TRAJECTORY_INDEX_FILENAME,
        ViewSupersedeIndex(tmp_path / "reloaded-views.json"),
        fallback_session=lambda _job: {},
    )
    pending = service.pending_jobs()
    assert len(pending) == 1
    job = pending[0]
    assert job.job_id == f"post-run:{SESSION_ID}:task-1"
    assert job.session_id == SESSION_ID
    assert job.task_run_id == "task-1"
    assert job.priority == 2
    assert job.outcome_status == "unspecified"


def test_budget_exhausted_leaves_the_job_queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_synthetic(monkeypatch)
    ledger = AOLedger(tmp_path, SESSION_ID)
    _append_segment(ledger, segment_id="seg-a", participant_id="agent-a", tool="read_file")
    seen: list[SkippedResult] = []
    original = TrajectoryIndexService.process_next

    def _spy(self, policy):
        result = original(self, policy)
        seen.append(result)
        return result

    monkeypatch.setattr(TrajectoryIndexService, "process_next", _spy)
    maybe_index_session_trajectory(
        _bound_session(tmp_path),
        "task-1",
        config=_causal_config(max_builds_per_window=0),
    )

    assert len(seen) == 1
    assert isinstance(seen[0], SkippedResult)
    assert seen[0].reason == SKIP_BUDGET_EXHAUSTED
    assert seen[0].projection is None
    assert _revision(tmp_path) is None
    service = TrajectoryIndexService(
        tmp_path / TRAJECTORY_INDEX_FILENAME,
        ViewSupersedeIndex(tmp_path / "reloaded-views.json"),
        fallback_session=lambda _job: {},
    )
    assert [job.job_id for job in service.pending_jobs()] == [f"post-run:{SESSION_ID}:task-1"]


def test_same_run_repeat_does_not_register_a_second_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_synthetic(monkeypatch)
    ledger = AOLedger(tmp_path, SESSION_ID)
    _append_segment(ledger, segment_id="seg-a", participant_id="agent-a", tool="read_file")
    _append_segment(ledger, segment_id="seg-b", participant_id="agent-b", tool="search")
    session = _bound_session(tmp_path)
    config = _causal_config()

    maybe_index_session_trajectory(session, "task-1", config=config)
    maybe_index_session_trajectory(session, "task-1", config=config)
    maybe_index_session_trajectory(session, "task-2", config=config)

    history = ViewSupersedeIndex(tmp_path / VIEWS_FILENAME).history(
        SESSION_ID, PURPOSE_POST_RUN_INDEX
    )
    assert len(history) == 1
    assert history[0].task_run_id == "task-1"
    assert history[0].revision_id == f"post-run:{SESSION_ID}:task-1"
    assert [block.segment_ref.segment_id for block in history[0].blocks] == ["seg-a", "seg-b"]


@pytest.mark.asyncio
async def test_later_commit_without_new_ao_does_not_add_a_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openviking.service.task_tracker import set_task_tracker

    _set_causal(monkeypatch, True)
    _forbid_synthetic(monkeypatch)
    root = tmp_path / "repeat"
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
        first = await session.commit_async(keep_recent_count=0)
        await session.add_messages_async(
            [{"role": "user", "parts": [TextPart("no new tool exchange")]}]
        )
        second = await session.commit_async(keep_recent_count=0)
    finally:
        set_task_tracker(None)

    assert first.get("archived") is True
    assert second.get("archived") is True
    assert first["task_id"] != second["task_id"]
    history = ViewSupersedeIndex(root / VIEWS_FILENAME).history(SESSION_ID, PURPOSE_POST_RUN_INDEX)
    assert len(history) == 1
    assert history[0].task_run_id == first["task_id"]


@pytest.mark.asyncio
async def test_build_failure_does_not_fail_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openviking.service.task_tracker import set_task_tracker

    warnings: list[tuple[str, BaseException | None]] = []

    def _warning(message: str, *args: object, **kwargs: object) -> None:
        del args, kwargs
        current = sys.exc_info()[1]
        warnings.append((message, current if isinstance(current, BaseException) else None))

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("index build failed")

    _set_causal(monkeypatch, True)
    monkeypatch.setattr("openviking.session.trajectory_index_hook.logger.warning", _warning)
    monkeypatch.setattr("openviking.session.trajectory_index.build_minimal_fallback_view", _boom)
    root = tmp_path / "fail"
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
    assert any(
        message == "trajectory index hook failed"
        and isinstance(exc, RuntimeError)
        and "index build failed" in str(exc)
        for message, exc in warnings
    )


def test_config_defaults_are_unpaused_with_a_positive_budget() -> None:
    section = TrajectoryIndexConfig()
    assert section.paused is False
    assert section.max_builds_per_window == 32
    assert section.window_seconds == 60.0
