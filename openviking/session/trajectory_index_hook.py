# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Post-commit trajectory index hook.

``session.py`` calls :func:`maybe_index_session_trajectory` after archive
attribution in Phase 1, and again from Phase 2 only when an extraction
provider is injected. Legacy mode is one boolean check and returns. Causal
mode pages the real AO ledger, enqueues one ``IndexJob``, and drives
``process_next``. The service is always given a ``fallback_session`` so the
synthetic one-AO fallback is not the production path. Failures are logged and
swallowed so commit itself still succeeds.

``extraction_provider`` defaults to ``None``. The build is then the
deterministic ledger fallback (``construction_mode=fallback``), unchanged
from the commit hook that only knew that path. When a provider is injected,
the build freezes that payload as ``construction_mode=native_single_pass_aig``. The
provider is not read from config and this hook does not call a model.

``extraction_provider_factory`` is the other injection. It receives the
commit snapshot, including ``ledger_rows``, and returns a provider. The
builder stays the three-argument ``(session_id, task_run_id, outcome_status)``
callable. Pass the provider or the factory, not both. Neither is read from
config.

On-disk layout, under the session directory:

* ``views.json`` — ``ViewSupersedeIndex``
* ``trajectory-index.json`` — ``TrajectoryIndexService`` queue snapshot
* ``projections.json`` — ``ProjectionRegistry`` of cards for this session
* ``needs-revalidation.json`` — revision ids notified for revalidation

A session whose owner cannot be resolved still freezes the view and the
index. Projection cards are skipped in that case: visibility fails closed,
and indexing itself is not an access check. The same ``revalidation_callback``
is given to the index service and to ``ProjectionRegistry``. It appends each
notified revision id to ``needs-revalidation.json`` in the session directory:
a JSON array, deduplicated, original order kept. An ACL withdrawal therefore
records the source view revision of each card it removes. When the session
exposes ``causal_experiences.mark_needs_revalidation``, that method is called
with the same revision id and does not rewrite the revision.

``task_run_id`` is this commit's ``task_id``. There is no task-run entity.
When the session exposes ``index_outcome_status`` as ``evaluation``,
``failed``, or ``succeeded``, the job is queued at priority 0, 1, or 2 with
that outcome. Otherwise ``priority`` is ``2`` and ``outcome_status`` is
``"unspecified"``: commit cannot see task success or failure, and the
priority enum has no neutral member. ``2`` is the lowest-urgency legal value;
it is not a success verdict.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openviking.session.ao_ledger_reader import MAX_PAGE_LIMIT, list_ao_ledger
from openviking.session.extraction_contract import ExtractionProviderFactory
from openviking.session.extraction_write import (
    NATIVE_EDGE_INDEX_FILENAME,
    ExtractionPayload,
    ExtractionProvider,
    NativeEdgeViewStore,
    NativeEdgeWriteError,
    NativeExtractionPayload,
    freeze_native_edge_view,
)
from openviking.session.legacy_write_gate import LegacyWriteRefused
from openviking.session.influence_projection import ProjectionRegistry
from openviking.session.influence_view import PURPOSE_POST_RUN_INDEX, ViewSupersedeIndex
from openviking.session.session_acl import session_acl_callback
from openviking.session.trajectory_index import (
    PRIORITY_EVALUATION,
    PRIORITY_FAILED,
    PRIORITY_SUCCEEDED,
    IndexJob,
    IndexedResult,
    Priority,
    SkippedResult,
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
PROJECTIONS_FILENAME = "projections.json"
NEEDS_REVALIDATION_FILENAME = "needs-revalidation.json"

# Lowest-urgency legal priority. Not a claim that the task succeeded.
COMMIT_INDEX_PRIORITY = PRIORITY_SUCCEEDED
COMMIT_INDEX_OUTCOME_STATUS = "unspecified"
_INDEX_OUTCOME_PRIORITY: dict[str, Priority] = {
    "evaluation": PRIORITY_EVALUATION,
    "failed": PRIORITY_FAILED,
    "succeeded": PRIORITY_SUCCEEDED,
}
UNSPECIFIED_PARTICIPANT_ID = "unspecified"

logger = get_logger(__name__)


def session_directory(session: Any) -> Path | None:
    """Return the on-disk directory for ``session``, or None when it cannot be resolved.

    This is the production resolver. Other callers must use it instead of
    parsing ``_session_uri`` again.
    """
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


def _session_dir(session: Any) -> Path | None:
    return session_directory(session)


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


def _commit_index_priority(session: Any) -> tuple[Priority, str]:
    """Map ``index_outcome_status`` onto the queue. Unknown values stay unspecified."""
    raw = getattr(session, "index_outcome_status", None)
    if isinstance(raw, str):
        priority = _INDEX_OUTCOME_PRIORITY.get(raw)
        if priority is not None:
            return priority, raw
    return COMMIT_INDEX_PRIORITY, COMMIT_INDEX_OUTCOME_STATUS


def _read_needs_revalidation(path: Path) -> list[str]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(not isinstance(item, str) for item in payload):
        raise ValueError(f"{path.name} must be a JSON array of revision ids")
    return payload


def _append_needs_revalidation(session_dir: Path, revision_id: str) -> None:
    """Record one notified revision id. Duplicates stay in their original place."""
    if not isinstance(revision_id, str) or not revision_id.strip():
        raise ValueError("revision_id must be a non-empty str")
    path = session_dir / NEEDS_REVALIDATION_FILENAME
    current = _read_needs_revalidation(path)
    if revision_id in current:
        logger.warning(
            "revalidation_callback notified revision %s; already recorded in %s",
            revision_id,
            NEEDS_REVALIDATION_FILENAME,
        )
        return
    current.append(revision_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(current, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    logger.warning(
        "revalidation_callback notified revision %s; appended to %s",
        revision_id,
        NEEDS_REVALIDATION_FILENAME,
    )


def _mark_causal_needs_revalidation(session: Any, revision_id: str) -> None:
    """Flag intervention history when the session store can do that.

    Sessions without ``causal_experiences``, or stores without
    ``mark_needs_revalidation``, are left unchanged.
    """
    experiences = getattr(session, "causal_experiences", None)
    mark = getattr(experiences, "mark_needs_revalidation", None)
    if not callable(mark):
        return
    mark(revision_id)


def _record_needs_revalidation(session: Any, session_dir: Path, revision_id: str) -> None:
    """Persist one notified revision id and flag causal history when present."""
    _append_needs_revalidation(session_dir, revision_id)
    _mark_causal_needs_revalidation(session, revision_id)


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


def _index_commit(
    session: Any,
    task_id: str,
    config: Any,
    extraction_provider: ExtractionProvider | None = None,
    *,
    extraction_provider_factory: ExtractionProviderFactory | None = None,
) -> None:
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

    ledger_rows = _page_ao_records(session_dir, session_id)
    ao_records = [_view_ao_record(item, session_id) for item in ledger_rows]
    if not ao_records:
        return

    watermark = _stable_snapshot_watermark(session_id, ao_records)
    priority, outcome_status = _commit_index_priority(session)
    job = IndexJob(
        job_id=f"post-run:{session_id}:{task_run_id}",
        session_id=session_id,
        task_run_id=task_run_id,
        priority=priority,
        outcome_status=outcome_status,
        enqueued_at=datetime.now(timezone.utc).isoformat(),
    )

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
            "ledger_rows": ledger_rows,
        }

    if extraction_provider is not None and extraction_provider_factory is not None:
        raise ValueError("pass extraction_provider or extraction_provider_factory, not both")
    # Factory binds this commit's snapshot. The builder stays the 3-arg
    # (session_id, task_run_id, outcome_status) callable, so existing builders
    # are not given a new parameter.
    provider = extraction_provider
    if extraction_provider_factory is not None:
        provider = extraction_provider_factory(fallback_session(job))
    if provider is not None:
        _commit_native_extraction(
            provider,
            session_dir=session_dir,
            session_id=session_id,
            task_run_id=task_run_id,
            outcome_status=outcome_status,
            frozen_at=job.enqueued_at,
        )
        return

    def revalidation_callback(revision_id: str) -> None:
        _record_needs_revalidation(session, session_dir, revision_id)

    service = TrajectoryIndexService(
        session_dir / TRAJECTORY_INDEX_FILENAME,
        ViewSupersedeIndex(session_dir / VIEWS_FILENAME),
        fallback_session=fallback_session,
        projection_registry=_projection_registry(session, session_dir, revalidation_callback),
        revalidation_callback=revalidation_callback,
    )
    service.enqueue(job)
    service.process_next(_policy(config))


def _commit_native_extraction(
    provider: ExtractionProvider,
    *,
    session_dir: Path,
    session_id: str,
    task_run_id: str,
    outcome_status: str,
    frozen_at: str,
) -> None:
    """Freeze a native-edge revision. Claim payloads are refused, not mapped.

    A second commit of the same session and task keeps the first revision.
    Claim payloads are refused here. The legacy claim index builder is not used.
    """
    payload = provider(session_id, task_run_id, outcome_status)
    if isinstance(payload, ExtractionPayload):
        raise LegacyWriteRefused()
    if not isinstance(payload, NativeExtractionPayload):
        raise TypeError("extraction provider must return a NativeExtractionPayload")
    revision = freeze_native_edge_view(
        session_id=session_id,
        task_run_id=task_run_id,
        view_id=f"{session_id}:{PURPOSE_POST_RUN_INDEX}",
        revision_id=f"post-run:{session_id}:{task_run_id}:native",
        frozen_at=frozen_at,
        blocks=payload.blocks,
        edges=payload.edges,
        coverage=payload.coverage,
    )
    store = NativeEdgeViewStore(session_dir / NATIVE_EDGE_INDEX_FILENAME)
    try:
        store.write(revision)
    except NativeEdgeWriteError as exc:
        if exc.code != "duplicate_revision":
            raise
    logger.info(
        "phase2 native-edge index revision_id=%s session_id=%s",
        revision.revision_id,
        session_id,
    )


def _projection_registry(
    session: Any,
    session_dir: Path,
    revalidation_callback: Callable[[str], None],
) -> ProjectionRegistry | None:
    """Build this commit's registry, or skip it when the owner is unknown.

    Indexing still runs. The missing registry is the visibility failure, not
    a reason to drop the frozen view. ``revalidation_callback`` is the same
    function the index service uses, so an ACL withdrawal writes
    ``needs-revalidation.json`` too.
    """
    try:
        session_acl = session_acl_callback(session)
    except ValueError:
        logger.warning(
            "session owner label could not be resolved; projection registry was skipped",
            exc_info=True,
        )
        return None
    return ProjectionRegistry(
        session_dir / PROJECTIONS_FILENAME,
        session_acl=session_acl,
        revalidation_callback=revalidation_callback,
    )


def _log_phase2_index_result(result: IndexedResult | SkippedResult) -> None:
    """Record the Phase 2 build outcome. A skip used to be silent."""
    if isinstance(result, SkippedResult):
        logger.warning(
            "phase2 trajectory index skipped reason=%s job_id=%s",
            result.reason,
            result.job_id,
        )
        return
    logger.info(
        "phase2 trajectory index indexed revision_id=%s construction_mode=%s job_id=%s",
        result.revision.revision_id,
        result.revision.construction_mode,
        result.job.job_id,
    )


def maybe_index_session_trajectory(
    session: Any,
    task_id: str,
    *,
    config: Any = None,
    extraction_provider: ExtractionProvider | None = None,
    extraction_provider_factory: ExtractionProviderFactory | None = None,
) -> None:
    """Causal-only Phase 1 hook. Legacy is a single boolean check and returns.

    ``extraction_provider`` defaults to ``None``, which keeps the deterministic
    ledger fallback. A provider is called only from the index builder, as
    ``(session_id, task_run_id, outcome_status)``, and must return an
    ``ExtractionPayload``. ``extraction_provider_factory`` receives the commit
    snapshot and returns that provider. Both default to ``None``.
    """
    try:
        if not is_causal_mode_enabled(config, session_id=getattr(session, "session_id", None)):
            return
        _index_commit(
            session,
            task_id,
            config,
            extraction_provider=extraction_provider,
            extraction_provider_factory=extraction_provider_factory,
        )
    except Exception:
        logger.warning("trajectory index hook failed", exc_info=True)
