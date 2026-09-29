# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Post-commit trajectory index hook.

``session.py`` calls :func:`maybe_index_session_trajectory` once, after archive
attribution in Phase 1. Legacy mode is one boolean check and returns. Causal
mode pages the real AO ledger, enqueues one ``IndexJob``, and drives
``process_next``. The service is always given a ``fallback_session`` so the
synthetic one-AO fallback is not the production path. Failures are logged and
swallowed so commit itself still succeeds.

On-disk layout, under the session directory:

* ``views.json`` — ``ViewSupersedeIndex``
* ``trajectory-index.json`` — ``TrajectoryIndexService`` queue snapshot

``task_run_id`` is this commit's ``task_id``. There is no task-run entity.
``priority`` is ``2`` and ``outcome_status`` is ``"unspecified"``: commit
cannot see task success or failure, and the priority enum has no neutral
member. ``2`` is the lowest-urgency legal value; it is not a success verdict.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openviking.session.ao_ledger_reader import MAX_PAGE_LIMIT, list_ao_ledger
from openviking.session.influence_view import PURPOSE_POST_RUN_INDEX, ViewSupersedeIndex
from openviking.session.trajectory_index import (
    PRIORITY_SUCCEEDED,
    IndexJob,
    TrajectoryIndexService,
    WorkspacePolicy,
)
from openviking_cli.utils import get_logger
from openviking_cli.utils.config.memory_config import (
    TrajectoryIndexConfig,
    is_causal_mode_enabled,
    resolve_trajectory_index_config,
)

VIEWS_FILENAME = "views.json"
TRAJECTORY_INDEX_FILENAME = "trajectory-index.json"

# Lowest-urgency legal priority. Not a claim that the task succeeded.
COMMIT_INDEX_PRIORITY = PRIORITY_SUCCEEDED
COMMIT_INDEX_OUTCOME_STATUS = "unspecified"
UNSPECIFIED_PARTICIPANT_ID = "unspecified"

logger = get_logger(__name__)


def _session_dir(session: Any) -> Path | None:
    viking_fs = getattr(session, "_viking_fs", None)
    uri = getattr(session, "_session_uri", None)
    if viking_fs is None or not uri:
        return None
    try:
        raw = viking_fs._uri_to_path(uri, ctx=getattr(session, "ctx", None))
    except Exception:
        return None
    if not raw:
        return None
    return Path(raw)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _text_field(item: Mapping[str, Any], key: str) -> str | None:
    direct = item.get(key)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    action = item.get("action")
    if isinstance(action, dict):
        nested = action.get(key)
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def _ao_content_hash(item: Mapping[str, Any]) -> str:
    """Stable hash of the ledger row. Reader annotations are not part of it."""
    explicit = _text_field(item, "content_hash")
    if explicit:
        return explicit
    body = {
        "ao_id": item.get("ao_id"),
        "session_id": item.get("session_id"),
        "sequence": item.get("sequence"),
        "action": item.get("action"),
        "observation": item.get("observation"),
        "skill_invocations": item.get("skill_invocations"),
        "message_ref": item.get("message_ref"),
        "captured_state": item.get("captured_state"),
        "created_at": item.get("created_at"),
    }
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def _view_ao_record(item: Mapping[str, Any], session_id: str) -> dict[str, Any]:
    """Map one ledger row onto the fallback view's AO shape.

    The ledger schema has no segment or participant. A row that already
    carries ``segment_id`` / ``participant_id`` (top level or on ``action``)
    is passed through. Otherwise the row is its own segment (``ao_id``) and
    the participant is ``"unspecified"``.
    """
    ao_id = _text_field(item, "ao_id")
    if not ao_id:
        raise ValueError("ao ledger row is missing ao_id")
    sequence = item.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise ValueError("ao ledger row sequence must be an int")
    record_session = _text_field(item, "session_id") or session_id
    return {
        "ao_id": ao_id,
        "sequence": sequence,
        "participant_id": _text_field(item, "participant_id") or UNSPECIFIED_PARTICIPANT_ID,
        "segment_id": _text_field(item, "segment_id") or ao_id,
        "session_id": record_session,
        "content_hash": _ao_content_hash(item),
    }


def _page_ao_records(session_dir: Path, session_id: str) -> list[dict[str, Any]]:
    """Page the ledger the same way ``SegmentAtomAdapter._items`` does."""
    items: list[dict[str, Any]] = []
    cursor: str | None = None
    watermark: str | None = None
    while True:
        page = list_ao_ledger(
            session_dir,
            session_id,
            cursor=cursor,
            limit=MAX_PAGE_LIMIT,
            snapshot_watermark=watermark,
            include_live=True,
        )
        items.extend(item for item in page.items if isinstance(item, dict))
        if not page.next_cursor:
            return items
        cursor = page.next_cursor
        watermark = page.snapshot_watermark


def _stable_snapshot_watermark(session_id: str, ao_records: list[Mapping[str, Any]]) -> str:
    """Digest of the paged AO identities.

    ``list_ao_ledger`` watermarks include ``issued_at``, so storing that token
    on the view would change ``content_hash`` on every commit and defeat the
    service's ``(session_id, purpose, content_hash)`` idempotency. Paging
    still uses the reader token; only the stored view field is this digest.
    """
    payload = {
        "session_id": session_id,
        "ao_records": [
            {
                "ao_id": record["ao_id"],
                "sequence": record["sequence"],
                "content_hash": record["content_hash"],
            }
            for record in ao_records
        ],
    }
    digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return f"ao-ledger:{digest}"


def _policy(config: Any) -> WorkspacePolicy:
    try:
        section = resolve_trajectory_index_config(config)
    except FileNotFoundError:
        section = TrajectoryIndexConfig()
    return WorkspacePolicy(
        max_builds_per_window=section.max_builds_per_window,
        window_seconds=section.window_seconds,
        paused=section.paused,
    )


def _index_commit(session: Any, task_id: str, config: Any) -> None:
    if not isinstance(task_id, str) or not task_id.strip():
        return
    task_run_id = task_id.strip()
    session_id = getattr(session, "session_id", None)
    if not isinstance(session_id, str) or not session_id.strip():
        return
    session_id = session_id.strip()
    session_dir = _session_dir(session)
    if session_dir is None:
        return

    ao_records = [
        _view_ao_record(item, session_id) for item in _page_ao_records(session_dir, session_id)
    ]
    if not ao_records:
        return

    watermark = _stable_snapshot_watermark(session_id, ao_records)
    frozen_at = datetime.now(timezone.utc).isoformat()

    def fallback_session(job: IndexJob) -> dict[str, Any]:
        if job.session_id != session_id or job.task_run_id != task_run_id:
            raise ValueError("index job does not match the commit being indexed")
        return {
            "session_id": job.session_id,
            "task_run_id": job.task_run_id,
            "purpose": PURPOSE_POST_RUN_INDEX,
            "view_id": f"{job.session_id}:{PURPOSE_POST_RUN_INDEX}",
            "revision_id": f"post-run:{job.session_id}:{job.task_run_id}",
            "frozen_at": job.enqueued_at,
            "snapshot_watermark": watermark,
            "ao_records": ao_records,
        }

    service = TrajectoryIndexService(
        session_dir / TRAJECTORY_INDEX_FILENAME,
        ViewSupersedeIndex(session_dir / VIEWS_FILENAME),
        fallback_session=fallback_session,
    )
    service.enqueue(
        IndexJob(
            job_id=f"post-run:{session_id}:{task_run_id}",
            session_id=session_id,
            task_run_id=task_run_id,
            priority=COMMIT_INDEX_PRIORITY,
            outcome_status=COMMIT_INDEX_OUTCOME_STATUS,
            enqueued_at=frozen_at,
        )
    )
    service.process_next(_policy(config))


def maybe_index_session_trajectory(session: Any, task_id: str, *, config: Any = None) -> None:
    """Causal-only Phase 1 hook. Legacy is a single boolean check and returns."""
    try:
        if not is_causal_mode_enabled(config, session_id=getattr(session, "session_id", None)):
            return
        _index_commit(session, task_id, config)
    except Exception:
        logger.warning("trajectory index hook failed", exc_info=True)
