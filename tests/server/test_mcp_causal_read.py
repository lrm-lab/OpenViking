# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Read-side MCP tools over a real causal write.

The facade is built only after ``is_causal_mode_enabled`` passes. Cards keep
the owner label written by the commit hook, and a different caller's
principals do not intersect that label.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import openviking.server.mcp_endpoint as mcp_endpoint
from openviking.core.namespace import canonical_user_root
from openviking.server.dependencies import get_service_or_none, set_service
from openviking.server.identity import RequestContext, Role
from openviking.session.ao_ledger import AOLedger
from openviking.session.trajectory_index_hook import maybe_index_session_trajectory
from openviking_cli.exceptions import FailedPreconditionError, NotFoundError
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.memory_config import MemoryConfig
from openviking_cli.utils.config.open_viking_config import (
    OpenVikingConfig,
    OpenVikingConfigSingleton,
    set_openviking_config,
)
from tests.test_extraction_write import _payload_for

SESSION_ID = "sess-causal-read"
EMPTY_ID = "sess-causal-empty"
TASK_ID = "task-causal-read"
OWNER_ID = "owner"
INTRUDER_ID = "intruder"
AO_SENTINEL = "RAW_AO_BODY_SENTINEL_do_not_leak"

_SEARCH_PARAMETERS = (
    "query",
    "target_uri",
    "session_id",
    "limit",
    "min_score",
    "level",
    "context_type",
    "mode",
    "query_expansion",
    "max_tokens",
    "quotas",
    "purpose",
    "detail",
    "detail_by_category",
    "dedup_turns",
    "exclude_uris",
    "peer_scope",
    "other_peer_penalty",
    "other_peer_penalties",
    "rewrite",
    "rewrite_max_bullets",
    "read_content",
)
_FIND_PARAMETERS = (
    "query",
    "target_uri",
    "limit",
    "min_score",
    "level",
    "context_type",
    "read_content",
)


class _Fs:
    def __init__(self, paths: dict[str, Path]) -> None:
        self.paths = paths
        self.calls: list[str] = []

    def _uri_to_path(self, uri: str, ctx=None) -> str:
        del ctx
        self.calls.append(uri)
        if uri not in self.paths:
            raise FileNotFoundError(uri)
        return str(self.paths[uri])


class _Session:
    def __init__(self, session_id: str, uri: str, fs: _Fs, ctx: RequestContext) -> None:
        self.session_id = session_id
        self._session_uri = uri
        self._viking_fs = fs
        self.ctx = ctx
        self._meta = SimpleNamespace(created_by_user_id=ctx.user.user_id)
        self.loaded = False

    async def load(self) -> None:
        self.loaded = True


class _Sessions:
    def __init__(self, sessions: dict[str, _Session]) -> None:
        self._sessions = sessions
        self.calls: list[str] = []

    def session(self, ctx: RequestContext, session_id: str) -> _Session:
        del ctx
        self.calls.append(session_id)
        found = self._sessions.get(session_id)
        if found is None:
            raise NotFoundError(session_id, "session")
        return found


class _SearchApi:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def is_intent_enabled(self) -> bool:
        return False

    async def search(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(memories=[], resources=[], skills=[])


def _ctx(user_id: str) -> RequestContext:
    return RequestContext(user=UserIdentifier("default", user_id), role=Role.USER)


def _set_mode(mode: str) -> None:
    set_openviking_config(OpenVikingConfig.from_dict({"memory": {"skill_trajectory_mode": mode}}))


def _provider(session_id: str, task_run_id: str, outcome_status: str):
    del task_run_id, outcome_status
    return _payload_for(session_id)


def _write_causal(tmp_path: Path) -> tuple[_Fs, _Session, _Session]:
    session_dir = tmp_path / "session"
    empty_dir = tmp_path / "empty"
    owner_root = tmp_path / "users" / OWNER_ID
    intruder_root = tmp_path / "users" / INTRUDER_ID
    for path in (session_dir, empty_dir, owner_root, intruder_root):
        path.mkdir(parents=True)
    owner = _ctx(OWNER_ID)
    session_uri = f"viking://user/{OWNER_ID}/sessions/{SESSION_ID}"
    empty_uri = f"viking://user/{OWNER_ID}/sessions/{EMPTY_ID}"
    fs = _Fs(
        {
            session_uri: session_dir,
            empty_uri: empty_dir,
            canonical_user_root(owner): owner_root,
            canonical_user_root(_ctx(INTRUDER_ID)): intruder_root,
        }
    )
    written = _Session(SESSION_ID, session_uri, fs, owner)
    empty = _Session(EMPTY_ID, empty_uri, fs, owner)
    AOLedger(session_dir, SESSION_ID).append(
        {"tool": "read_file", "segment_id": "seg-a", "participant_id": "agent-a"},
        {"kind": "text", "summary": ["read_file"], "body": AO_SENTINEL},
    )
    maybe_index_session_trajectory(
        written,
        TASK_ID,
        config=MemoryConfig(skill_trajectory_mode="causal"),
        extraction_provider=_provider,
    )
    assert (session_dir / "views.json").is_file()
    assert (session_dir / "projections.json").is_file()
    return fs, written, empty


@pytest.fixture(autouse=True)
def _restore_process_state():
    previous_config = OpenVikingConfigSingleton._instance
    previous_service = get_service_or_none()
    token = mcp_endpoint._mcp_ctx.set(_ctx(OWNER_ID))
    yield
    mcp_endpoint._mcp_ctx.reset(token)
    set_service(previous_service)
    if previous_config is None:
        OpenVikingConfigSingleton.reset_instance()
    else:
        set_openviking_config(previous_config)


@pytest.fixture
def causal_world(tmp_path: Path):
    fs, written, empty = _write_causal(tmp_path)
    sessions = _Sessions({SESSION_ID: written, EMPTY_ID: empty})
    set_service(SimpleNamespace(sessions=sessions, search=_SearchApi()))
    _set_mode("causal")
    return SimpleNamespace(fs=fs, sessions=sessions, written=written, empty=empty)


def _claim_id(payload: dict) -> str:
    for card in payload["cards"]:
        if card["kind"] == "claim_projection":
            return card["payload_ref"]
    raise AssertionError(f"no claim card in {payload['cards']!r}")


@pytest.mark.asyncio
async def test_causal_mode_off_raises_without_touching_the_facade(monkeypatch, causal_world):
    del causal_world
    _set_mode("legacy")

    def _refuse_facade(**kwargs):
        del kwargs
        raise AssertionError("facade built")

    monkeypatch.setattr(mcp_endpoint, "build_retrieval_facade", _refuse_facade)
    sessions = mcp_endpoint.get_service().sessions
    sessions.calls.clear()

    with pytest.raises(FailedPreconditionError, match="not enabled"):
        await mcp_endpoint.retrieve_causal_cards("patch", SESSION_ID)
    with pytest.raises(FailedPreconditionError, match="not enabled"):
        await mcp_endpoint.expand_causal_neighborhood("any-card", SESSION_ID)

    assert sessions.calls == []


@pytest.mark.asyncio
async def test_missing_session_raises_not_found(causal_world):
    del causal_world
    with pytest.raises(NotFoundError):
        await mcp_endpoint.retrieve_causal_cards("patch", "missing-session")


@pytest.mark.asyncio
async def test_owner_sees_cards_and_another_user_sees_none(causal_world):
    payload = await mcp_endpoint.retrieve_causal_cards("patch", SESSION_ID)

    assert payload["cards"]
    assert any(card["kind"] == "claim_projection" for card in payload["cards"])
    assert {tuple(card["acl_labels"]) for card in payload["cards"]} == {("user:owner",)}
    assert causal_world.written.loaded is True
    assert canonical_user_root(_ctx(OWNER_ID)) in causal_world.fs.calls
    text = json.dumps(payload)
    assert "subtask" not in text
    assert AO_SENTINEL not in text

    token = mcp_endpoint._mcp_ctx.set(_ctx(INTRUDER_ID))
    try:
        hidden = await mcp_endpoint.retrieve_causal_cards("patch", SESSION_ID)
    finally:
        mcp_endpoint._mcp_ctx.reset(token)

    assert hidden["cards"] == []
    assert canonical_user_root(_ctx(INTRUDER_ID)) in causal_world.fs.calls
    assert "subtask" not in json.dumps(hidden)


@pytest.mark.asyncio
async def test_expand_claim_card_and_foreign_card_id(causal_world):
    del causal_world
    payload = await mcp_endpoint.retrieve_causal_cards("patch", SESSION_ID)
    card_id = _claim_id(payload)

    neighborhood = await mcp_endpoint.expand_causal_neighborhood(card_id, SESSION_ID)

    assert neighborhood["entities"]
    assert neighborhood["claim_id"]
    text = json.dumps(neighborhood)
    assert "subtask" not in text
    assert AO_SENTINEL not in text

    # expand_claim_neighborhood does not take principal labels. Another caller
    # who loads this same session still receives the neighborhood.
    token = mcp_endpoint._mcp_ctx.set(_ctx(INTRUDER_ID))
    try:
        shared = await mcp_endpoint.expand_causal_neighborhood(card_id, SESSION_ID)
    finally:
        mcp_endpoint._mcp_ctx.reset(token)
    assert shared == neighborhood

    # A card id that is not in this session's registry raises. An empty session
    # is that case for the owner's card.
    with pytest.raises(ValueError, match="unknown card id"):
        await mcp_endpoint.expand_causal_neighborhood(card_id, EMPTY_ID)
    with pytest.raises(ValueError, match="unknown card id"):
        await mcp_endpoint.expand_causal_neighborhood("claim-card-someone-else", SESSION_ID)


@pytest.mark.asyncio
async def test_session_without_causal_files_returns_empty(causal_world):
    del causal_world
    payload = await mcp_endpoint.retrieve_causal_cards("patch", EMPTY_ID)

    assert payload["cards"] == []
    assert "subtask" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_search_and_find_are_unchanged(monkeypatch):
    def _forbidden(**kwargs):
        del kwargs
        raise AssertionError("causal facade was built from search or find")

    monkeypatch.setattr(mcp_endpoint, "build_retrieval_facade", _forbidden)
    search_api = _SearchApi()
    set_service(SimpleNamespace(search=search_api, sessions=_Sessions({})))

    result = await mcp_endpoint.search(query="anything", limit=3)

    assert result == "No matching context found."
    assert "causal" not in result
    assert search_api.kwargs is not None
    assert search_api.kwargs["query"] == "anything"
    assert search_api.kwargs["limit"] == 3
    assert "causal" not in search_api.kwargs
    assert tuple(inspect.signature(mcp_endpoint.search).parameters) == _SEARCH_PARAMETERS
    assert tuple(inspect.signature(mcp_endpoint.find).parameters) == _FIND_PARAMETERS
    assert "causal" not in inspect.getsource(mcp_endpoint.search)
    assert "causal" not in inspect.getsource(mcp_endpoint.find)
    assert "profile" not in inspect.signature(mcp_endpoint.retrieve_causal_cards).parameters
