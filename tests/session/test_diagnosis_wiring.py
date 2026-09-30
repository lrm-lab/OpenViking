# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Server-side diagnosis reads and fork writes. Not MCP tools."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.session.diagnosis_wiring import (
    latest_post_run_revision,
    post_extraction_diagnosis_briefs,
    retrieve_historical_subgraphs_for_failure,
    write_diagnosis_fork,
)
from openviking.session.influence_view import RetrievalSubgraph, ViewSupersedeIndex
from openviking.session.session_acl import session_owner_label
from openviking.session.trajectory_index_hook import VIEWS_FILENAME
from openviking_cli.session.user_id import UserIdentifier
from tests.test_subgraph_forks import (
    _SESSION,
    _basis,
    _block,
    _evidence,
    _freeze,
    _release_revision,
    _sections,
)

_FORK_BRANCH = [
    {
        "branch_id": "br-cutover",
        "evidence_status": "real",
        "ao_sequence": ["ao-step"],
        "guidance": {"instruction": "切回蓝"},
    }
]


class _FS:
    def __init__(self, session_dir: Path, user_root: Path, session_uri: str) -> None:
        self.session_dir = session_dir
        self.user_root = user_root
        self.session_uri = session_uri

    def _uri_to_path(self, uri: str, ctx=None) -> str:
        del ctx
        if uri == self.session_uri:
            return str(self.session_dir)
        if uri.startswith("viking://user/"):
            return str(self.user_root)
        return ""


def _owner_ctx() -> RequestContext:
    return RequestContext(user=UserIdentifier("default", "default"), role=Role.ROOT)


def _bound(tmp_path: Path, session_id: str = _SESSION):
    ctx = _owner_ctx()
    session_dir = tmp_path / "session"
    user_root = tmp_path / "user"
    session_dir.mkdir()
    user_root.mkdir()
    session_uri = f"viking://user/{ctx.user.user_id}/sessions/{session_id}"
    session = SimpleNamespace(
        session_id=session_id,
        _session_uri=session_uri,
        ctx=ctx,
        _viking_fs=_FS(session_dir, user_root, session_uri),
    )
    return session, session_dir, user_root


def _fork_files(user_root: Path) -> set[Path]:
    root = user_root / "causal-experiences" / "forks"
    if not root.exists():
        return set()
    return set(root.rglob("*.json"))


def test_missing_view_reads_are_empty(tmp_path: Path) -> None:
    session, _session_dir, _user_root = _bound(tmp_path)
    assert latest_post_run_revision(session) is None
    assert post_extraction_diagnosis_briefs(session) == ()


def test_owner_can_retrieve_and_a_disjoint_principal_raises(tmp_path: Path) -> None:
    session, session_dir, _user_root = _bound(tmp_path)
    older = _freeze(
        _SESSION,
        "rev-old",
        [_block(_SESSION, "b-old", 1, "错误率升高")],
        [
            RetrievalSubgraph(
                subgraph_id="sg-old",
                block_ids=("b-old",),
                claim_ids=(),
                failure_marked=False,
            )
        ],
    )
    current = _freeze(
        _SESSION,
        "rev-now",
        [_block(_SESSION, "b-now", 2, "错误率升高")],
        [
            RetrievalSubgraph(
                subgraph_id="sg-now",
                block_ids=("b-now",),
                claim_ids=(),
                failure_marked=True,
                failure_evidence_refs=(_evidence(_SESSION, 2),),
            )
        ],
    )
    index = ViewSupersedeIndex(session_dir / VIEWS_FILENAME)
    index.register(older)
    index.register(current)

    hits = retrieve_historical_subgraphs_for_failure(
        session,
        session.ctx,
        session._viking_fs,
        subgraph_id="sg-now",
    )
    assert [hit.view_revision_id for hit in hits] == ["rev-old"]

    stranger = RequestContext(user=UserIdentifier("default", "stranger"), role=Role.USER)
    with pytest.raises(ValueError, match="not visible to the principal"):
        retrieve_historical_subgraphs_for_failure(
            session,
            stranger,
            session._viking_fs,
            subgraph_id="sg-now",
        )
    with pytest.raises(ValueError, match="not on view revision"):
        retrieve_historical_subgraphs_for_failure(
            session,
            session.ctx,
            session._viking_fs,
            view_revision_id="rev-now",
            subgraph_id="sg-missing",
        )


def test_fork_write_uses_the_owner_label_and_rejects_a_disjoint_label_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openviking.session.causal_experiences.is_causal_mode_enabled",
        lambda *_args, **_kwargs: True,
    )
    session, session_dir, user_root = _bound(tmp_path)
    ViewSupersedeIndex(session_dir / VIEWS_FILENAME).register(_release_revision())
    fields = {
        "subgraph_id": "sg-cutover",
        "seven_sections": _sections("错误率升高就切回蓝"),
        "branches": _FORK_BRANCH,
        "diagnosis_run_id": "diag-cutover",
        "model_version": "diag-model-v1",
        "contemporaneous_basis": [_basis(2)],
    }

    written = write_diagnosis_fork(session, session.ctx, session._viking_fs, **fields)
    assert written.workspace_id == session_owner_label(session)
    assert written.workspace_id == "user:default"
    written_files = _fork_files(user_root)
    assert len(written_files) == 1

    with pytest.raises(ValueError, match="is not in workspace_labels"):
        write_diagnosis_fork(
            session,
            session.ctx,
            session._viking_fs,
            workspace_labels=("user:stranger",),
            **fields,
        )
    assert _fork_files(user_root) == written_files


def test_search_and_find_are_not_diagnosis_entrypoints() -> None:
    import openviking.server.mcp_endpoint as endpoint

    source = Path(endpoint.__file__).read_text(encoding="utf-8")
    assert "diagnosis_wiring" not in source
    for function in (endpoint.find, endpoint.search):
        body = inspect.getsource(function)
        assert "diagnosis_wiring" not in body
        assert "write_diagnosis_fork" not in body
        assert "retrieve_historical_subgraphs_for_failure" not in body
        assert "post_extraction_diagnosis_briefs" not in body
