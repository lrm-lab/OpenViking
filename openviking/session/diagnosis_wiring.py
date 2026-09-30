# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Server-side diagnosis calls. These are not agent tools.

After Phase 2 freezes a post-run view, a caller can read which failure
subgraphs still owe a fork, retrieve historical subgraphs for one of them,
and write one ordinary fork. Nothing here calls a model, and nothing here
is exposed on the MCP tool surface.

``workspace_labels`` defaults to the single owner label. Passing a set that
does not contain that owner raises the store's ``ValueError`` and does not
write. ``diagnosis_run_id``, ``model_version``, ``seven_sections``,
``branches``, and ``contemporaneous_basis`` have no producer in this
repository, so this module does not fill them in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openviking.core.namespace import canonical_user_root
from openviking.session.causal_experiences import CausalExperiencesStore, ForkRevision
from openviking.session.influence_view import (
    PURPOSE_POST_RUN_INDEX,
    InfluenceViewRevision,
    ViewSupersedeIndex,
)
from openviking.session.retrieval_facade import DiagnosticSubgraphHit
from openviking.session.retrieval_wiring import build_retrieval_facade
from openviking.session.session_acl import session_owner_label
from openviking.session.subgraph_forks import (
    default_failure_grounding_position,
    failure_subgraph_fork_idempotency_key,
    failure_subgraphs_owed_forks,
    write_failure_subgraph_fork,
)
from openviking.session.trajectory_index_hook import VIEWS_FILENAME, session_directory
from openviking.storage.acl import acl_principals


@dataclass(frozen=True)
class FailureSubgraphBrief:
    """One failure subgraph that still owes an ordinary fork.

    There is one brief per subgraph. Briefs are not grouped and do not
    carry a ``cause_set``.
    """

    view_revision_id: str
    subgraph_id: str
    fork_idempotency_key: str
    grounding_position: tuple[str, str]


def latest_post_run_revision(session: Any) -> InfluenceViewRevision | None:
    """Latest ``post_run_index`` revision, or None when the view file is absent."""
    directory = session_directory(session)
    if directory is None:
        return None
    session_id = getattr(session, "session_id", None)
    if not isinstance(session_id, str) or not session_id.strip():
        return None
    index = ViewSupersedeIndex(directory / VIEWS_FILENAME)
    return index.latest(session_id.strip(), PURPOSE_POST_RUN_INDEX)


def post_extraction_diagnosis_briefs(session: Any) -> tuple[FailureSubgraphBrief, ...]:
    """Read failure subgraphs that owe a fork. Does not call a model or write."""
    revision = latest_post_run_revision(session)
    if revision is None:
        return ()
    briefs: list[FailureSubgraphBrief] = []
    for subgraph in failure_subgraphs_owed_forks(revision):
        briefs.append(
            FailureSubgraphBrief(
                view_revision_id=revision.revision_id,
                subgraph_id=subgraph.subgraph_id,
                fork_idempotency_key=failure_subgraph_fork_idempotency_key(
                    revision.revision_id,
                    subgraph.subgraph_id,
                ),
                grounding_position=default_failure_grounding_position(
                    revision,
                    subgraph.subgraph_id,
                ),
            )
        )
    return tuple(briefs)


def retrieve_historical_subgraphs_for_failure(
    session: Any,
    ctx: Any,
    viking_fs: Any,
    *,
    subgraph_id: str,
    view_revision_id: str | None = None,
) -> tuple[DiagnosticSubgraphHit, ...]:
    """Retrieve historical subgraphs for one failure subgraph on this session.

    ``view_revision_id`` defaults to the latest post-run revision. The
    principal is ``acl_principals(ctx)``. A disjoint principal raises; it
    does not come back as an empty hit list.
    """
    revision_id = view_revision_id
    if revision_id is None:
        latest = latest_post_run_revision(session)
        if latest is None:
            raise ValueError("post-run view revision is not available")
        revision_id = latest.revision_id
    facade = build_retrieval_facade(session=session, ctx=ctx, viking_fs=viking_fs)
    return facade.retrieve_historical_subgraphs(
        revision_id,
        subgraph_id,
        tuple(acl_principals(ctx)),
    )


def write_diagnosis_fork(
    session: Any,
    ctx: Any,
    viking_fs: Any,
    *,
    subgraph_id: str,
    seven_sections: Mapping[str, str],
    branches: Sequence[Mapping[str, Any]],
    diagnosis_run_id: str,
    model_version: str,
    contemporaneous_basis: Sequence[Mapping[str, Any]],
    hindsight_attribution: Sequence[Mapping[str, Any]] | None = None,
    used_skills: Sequence[Mapping[str, Any]] | None = None,
    view_revision_id: str | None = None,
    workspace_labels: Sequence[str] | None = None,
) -> ForkRevision:
    """Write one ordinary fork for one failure subgraph.

    ``workspace_id`` is the session owner label. ``workspace_labels``
    defaults to that single label. The grounding position defaults to the
    block that covers the failure evidence. Caller-supplied diagnosis
    fields are passed through; this function does not invent them.
    """
    revision = _require_revision(session, view_revision_id)
    if viking_fs is None:
        raise ValueError("viking_fs is required to resolve the fork store root")
    user_root_uri = canonical_user_root(ctx)
    user_root = viking_fs._uri_to_path(user_root_uri, ctx=ctx)
    if not user_root:
        raise ValueError("user root could not be resolved")
    workspace_id = session_owner_label(session)
    labels: Sequence[str] = (workspace_id,) if workspace_labels is None else workspace_labels
    position_kind, position_ref = default_failure_grounding_position(revision, subgraph_id)
    store = CausalExperiencesStore(user_root)
    return write_failure_subgraph_fork(
        store,
        revision,
        subgraph_id,
        seven_sections=seven_sections,
        branches=branches,
        position_kind=position_kind,
        position_ref=position_ref,
        workspace_id=workspace_id,
        workspace_labels=labels,
        diagnosis_run_id=diagnosis_run_id,
        model_version=model_version,
        contemporaneous_basis=contemporaneous_basis,
        hindsight_attribution=hindsight_attribution,
        used_skills=used_skills,
    )


def _require_revision(session: Any, view_revision_id: str | None) -> InfluenceViewRevision:
    if view_revision_id is None:
        revision = latest_post_run_revision(session)
        if revision is None:
            raise ValueError("post-run view revision is not available")
        return revision
    directory = session_directory(session)
    if directory is None:
        raise ValueError("session directory could not be resolved")
    session_id = getattr(session, "session_id", None)
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id is required")
    identity = view_revision_id.strip()
    for item in ViewSupersedeIndex(directory / VIEWS_FILENAME).history(
        session_id.strip(),
        PURPOSE_POST_RUN_INDEX,
    ):
        if item.revision_id == identity:
            return item
    raise ValueError(f"view revision {identity!r} is not in the supersede index")
