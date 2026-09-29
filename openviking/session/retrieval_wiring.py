# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Production constructor for the causal retrieval facade.

Write-side commits already drop ``views.json`` and ``projections.json`` in
the session directory. This module is the read-side place that binds those
files, the AO ledger, and the owner-scoped fork store into one
``HybridRetrievalFacade``. Missing files are not an error: a session that
has never been causally indexed still builds a facade and retrieves nothing.
"""

from __future__ import annotations

from typing import Any

from openviking.core.namespace import canonical_user_root
from openviking.session.causal_experiences import CausalExperiencesStore
from openviking.session.influence_projection import ProjectionRegistry
from openviking.session.influence_view import PURPOSES, ViewSupersedeIndex
from openviking.session.retrieval_facade import (
    ForkBranchAdapter,
    HybridRetrievalFacade,
    InfluenceProjectionAdapter,
    RetrievalSubgraphAdapter,
    SegmentAtomAdapter,
)
from openviking.session.session_acl import session_acl_callback
from openviking.session.trajectory_index_hook import (
    PROJECTIONS_FILENAME,
    VIEWS_FILENAME,
    session_directory,
)


def build_retrieval_facade(*, session: Any, ctx: Any, viking_fs: Any) -> HybridRetrievalFacade:
    """Bind one session's on-disk causal files into a retrieval facade.

    The fork store root is the requesting principal's user root
    (``canonical_user_root(ctx)`` mapped through ``viking_fs``), not the
    session directory. Forks are user-scoped; views and projections stay
    session-scoped. ``session_acl_callback(session)`` is shared by the
    projection registry and the subgraph adapter.
    """
    directory = session_directory(session)
    if directory is None:
        raise ValueError("session directory could not be resolved")
    session_id = _session_id(session)
    if viking_fs is None:
        raise ValueError("viking_fs is required to resolve the fork store root")
    user_root_uri = canonical_user_root(ctx)
    user_root = viking_fs._uri_to_path(user_root_uri, ctx=ctx)
    if not user_root:
        raise ValueError("user root could not be resolved")

    session_acl = session_acl_callback(session)
    views = ViewSupersedeIndex(directory / VIEWS_FILENAME)
    revisions = []
    for purpose in sorted(PURPOSES):
        revisions.extend(views.revisions_for_purpose(purpose))
    registry = ProjectionRegistry(
        directory / PROJECTIONS_FILENAME,
        session_acl=session_acl,
    )
    forks = ForkBranchAdapter(
        user_root,
        CausalExperiencesStore(user_root),
        session_dir=directory,
        session_id=session_id,
    )
    return HybridRetrievalFacade(
        forks,
        InfluenceProjectionAdapter(registry, views=revisions),
        SegmentAtomAdapter(directory, session_id),
        retrieval_subgraph=RetrievalSubgraphAdapter(
            views,
            forks,
            session_acl=session_acl,
            session_dirs={session_id: directory},
        ),
    )


def _session_id(session: Any) -> str:
    session_id = getattr(session, "session_id", None)
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id is required")
    return session_id.strip()
