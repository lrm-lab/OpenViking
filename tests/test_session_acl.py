# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Session owner labels, projection registry wiring, and the fork write gate."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.session.ao_ledger import AOLedger
from openviking.session.influence_projection import ProjectionRegistry
from openviking.session.influence_view import PURPOSE_POST_RUN_INDEX, ViewSupersedeIndex
from openviking.session.session_acl import (
    session_acl_callback,
    session_owner_label,
    session_stored_owner_label,
)
from openviking.session.trajectory_index_hook import (
    PROJECTIONS_FILENAME,
    VIEWS_FILENAME,
    maybe_index_session_trajectory,
)
from openviking.storage.acl import acl_principals
from openviking_cli.session.user_id import UserIdentifier
from tests.test_trajectory_index_hook import (
    SESSION_ID,
    SESSION_URI,
    _causal_config,
    _DummyFS,
    _forbid_synthetic,
    _install_commit_fakes,
    _set_causal,
)

_SESSION = "sess-acl"


def _session(
    *,
    session_id: str = _SESSION,
    user_id: str | None = None,
    meta_user_id: str | None = None,
    uri: str | None = None,
) -> SimpleNamespace:
    ctx = None
    if user_id is not None:
        ctx = SimpleNamespace(user=SimpleNamespace(user_id=user_id))
    meta = None
    if meta_user_id is not None:
        meta = SimpleNamespace(created_by_user_id=meta_user_id)
    return SimpleNamespace(
        session_id=session_id,
        ctx=ctx,
        _meta=meta,
        _session_uri=uri,
    )


def _append_segment(ledger: AOLedger, tool: str) -> None:
    record = ledger.append(
        {"tool": tool, "segment_id": "seg-a", "participant_id": "agent-a"},
        {"kind": "text", "summary": [tool]},
    )
    assert record is not None


def test_owner_label_prefers_ctx_over_meta_and_uri() -> None:
    session = _session(
        user_id="alice",
        meta_user_id="from-meta",
        uri="viking://user/from-uri/sessions/sess-acl",
    )
    assert session_owner_label(session) == "user:alice"


def test_owner_label_prefers_meta_when_ctx_user_is_empty() -> None:
    session = _session(
        user_id="  ",
        meta_user_id="from-meta",
        uri="viking://user/from-uri/sessions/sess-acl",
    )
    assert session_owner_label(session) == "user:from-meta"


def test_owner_label_uses_uri_user_segment() -> None:
    session = _session(uri="viking://user/from-uri/sessions/sess-acl")
    assert session_owner_label(session) == "user:from-uri"


def test_owner_label_missing_sources_raise() -> None:
    session = _session(uri="viking://agent/skills")
    with pytest.raises(ValueError, match="owner"):
        session_owner_label(session)


def test_stored_owner_ignores_caller_ctx() -> None:
    session = _session(
        user_id="alice",
        meta_user_id="from-meta",
        uri="viking://user/from-uri/sessions/sess-acl",
    )
    assert session_owner_label(session) == "user:alice"
    assert session_stored_owner_label(session) == "user:from-meta"
    uri_only = _session(user_id="alice", uri="viking://user/from-uri/sessions/sess-acl")
    assert session_stored_owner_label(uri_only) == "user:from-uri"
    with pytest.raises(ValueError, match="owner"):
        session_stored_owner_label(_session(user_id="alice"))


def test_callback_rejects_a_different_session_id() -> None:
    session = _session(user_id="alice")
    callback = session_acl_callback(session)
    assert callback(_SESSION) == ("user:alice",)
    with pytest.raises(ValueError, match="refusing"):
        callback("other-session")


def test_owner_label_intersects_reader_principals() -> None:
    user = UserIdentifier("default", "alice")
    ctx = RequestContext(user=user, role=Role.USER, group_ids=("team-a",))
    session = SimpleNamespace(
        session_id=_SESSION,
        ctx=ctx,
        _meta=SimpleNamespace(created_by_user_id="from-meta"),
        _session_uri="viking://user/from-uri/sessions/sess-acl",
    )
    label = session_owner_label(session)
    assert label == "user:alice"
    assert session_acl_callback(session)(_SESSION) == (label,)
    reader = RequestContext(user=user, role=Role.USER, group_ids=("team-a",))
    assert acl_principals(reader).intersection({label})


@pytest.mark.asyncio
async def test_commit_writes_projection_cards_for_the_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openviking.message import ToolPart
    from openviking.service.task_tracker import set_task_tracker
    from openviking.session.session import Session

    _set_causal(monkeypatch, True)
    _forbid_synthetic(monkeypatch)
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
    label = f"user:{session.ctx.user.user_id}"
    assert label == "user:default"
    assert acl_principals(session.ctx).intersection({label})
    registry = ProjectionRegistry(root / PROJECTIONS_FILENAME)
    cards = registry.search(principal_labels=(label,))
    assert cards
    assert {card.acl_labels for card in cards} == {(label,)}
    assert {card.source_pointer.view_revision_id for card in cards} == {
        f"post-run:{SESSION_ID}:{result['task_id']}"
    }
    text = (root / PROJECTIONS_FILENAME).read_text(encoding="utf-8")
    text += (root / VIEWS_FILENAME).read_text(encoding="utf-8")
    assert "subtask" not in text


@pytest.mark.asyncio
async def test_second_commit_withdraws_the_previous_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openviking.message import ToolPart
    from openviking.service.task_tracker import set_task_tracker
    from openviking.session.session import Session

    warnings: list[tuple[str, tuple[object, ...]]] = []

    def _warning(message: str, *args: object, **kwargs: object) -> None:
        del kwargs
        warnings.append((message, args))

    _set_causal(monkeypatch, True)
    _forbid_synthetic(monkeypatch)
    monkeypatch.setattr("openviking.session.trajectory_index_hook.logger.warning", _warning)
    root = tmp_path / "twice"
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
                            tool_output="first body",
                            tool_status="completed",
                        )
                    ],
                }
            ]
        )
        first = await session.commit_async(keep_recent_count=0)
        await session.add_messages_async(
            [
                {
                    "role": "assistant",
                    "parts": [
                        ToolPart(
                            tool_id="t2",
                            tool_name="edit_file",
                            tool_input={"path": "b.md"},
                            tool_output="second body",
                            tool_status="completed",
                        )
                    ],
                }
            ]
        )
        second = await session.commit_async(keep_recent_count=0)
    finally:
        set_task_tracker(None)

    assert first["task_id"] != second["task_id"]
    previous = f"post-run:{SESSION_ID}:{first['task_id']}"
    current = f"post-run:{SESSION_ID}:{second['task_id']}"
    label = f"user:{session.ctx.user.user_id}"
    registry = ProjectionRegistry(root / PROJECTIONS_FILENAME)
    cards = registry.search(principal_labels=(label,))
    assert cards
    assert {card.source_pointer.view_revision_id for card in cards} == {current}
    removed = registry.removed_records()
    assert removed
    assert {record.card.source_pointer.view_revision_id for record in removed} == {previous}
    assert {record.reason for record in removed} == {"superseded"}
    assert any(previous in args for _message, args in warnings)


def test_unresolved_owner_still_writes_the_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[str] = []

    def _warning(message: str, *args: object, **kwargs: object) -> None:
        del args, kwargs
        warnings.append(message)

    _forbid_synthetic(monkeypatch)
    monkeypatch.setattr("openviking.session.trajectory_index_hook.logger.warning", _warning)
    ledger = AOLedger(tmp_path, SESSION_ID)
    _append_segment(ledger, "read_file")
    uri = "viking://agent/not-a-session"
    session = SimpleNamespace(
        session_id=SESSION_ID,
        _session_uri=uri,
        ctx=SimpleNamespace(user=SimpleNamespace(user_id="")),
        _meta=SimpleNamespace(created_by_user_id=""),
        _viking_fs=_DummyFS(tmp_path, uri),
    )

    maybe_index_session_trajectory(session, "task-1", config=_causal_config())

    revision = ViewSupersedeIndex(tmp_path / VIEWS_FILENAME).latest(
        SESSION_ID, PURPOSE_POST_RUN_INDEX
    )
    assert revision is not None
    assert revision.task_run_id == "task-1"
    assert not (tmp_path / PROJECTIONS_FILENAME).exists()
    assert any("projection registry was skipped" in message for message in warnings)
    assert not any("trajectory index hook failed" in message for message in warnings)
