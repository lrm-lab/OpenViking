"""Durable, snapshot-pinned explicit diagnosis using the production four roles.

This entry never commits a live session or dispatches remediation. A request
has one immutable ledger snapshot and one bounded construction attempt.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openviking.session.aig_native_protocol import is_safe_id
from openviking.session.aig_production_pipeline import run_production_aig
from openviking.session.aig_tool_surface import AoSnapshot
from openviking.session.aig_vlm_roles import providers_for_vlm
from openviking.session.ao_ledger import AORecord
from openviking.session.append_only import write_exclusive
from openviking.session.causal_mode import is_session_causal
from openviking.session.trajectory_index_hook import (
    _page_ao_records,
    _stable_snapshot_watermark,
    _view_ao_record,
    session_directory,
)

TIMEOUT_SECONDS = 600
MAX_ACTIVE_DIAGNOSES = 3
MAX_RECORDS = 128
MAX_SNAPSHOT_BYTES = 262144
_tasks: set[asyncio.Task] = set()


CONTROL_PARTICIPANT = "openviking:diagnosis-control"


def _capture_request_boundary(
    rows: list[dict[str, Any]], session_id: str, request_id: str, question: str
) -> list[dict[str, Any]]:
    """Capture this actual API admission, separately from Task-authored evidence.

    The event lives in the immutable per-request snapshot, not the source
    conversation ledger. It records acceptance for analysis, never a Task
    execution, verdict, or remediation. Retries reuse the saved snapshot.
    """
    event = AORecord(
        ao_id=str(uuid.uuid4()),
        session_id=session_id,
        sequence=max(row["sequence"] for row in rows) + 1,
        action={
            "participant_id": CONTROL_PARTICIPANT,
            "segment_id": "diagnosis-control",
            "tool_name": "diagnosis.request",
            "request_id": request_id,
            "question": question,
        },
        observation={
            "status": "accepted",
            "text": "Diagnosis API request accepted for bounded analysis. Task outcome unknown; no repair performed.",
        },
        skill_invocations=[],
        message_ref=None,
        created_at=datetime.now(timezone.utc).isoformat(),
        captured_state="live",
    )
    return [*rows, event.to_dict()]


def _save(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _public(state: dict[str, Any]) -> dict[str, Any]:
    result = {key: state[key] for key in ("request_id", "status", "watermark")}
    for key in ("reason", "report", "report_uri", "content_hash", "revision_id"):
        if key in state:
            result[key] = state[key]
    return result


def read_diagnosis(session: Any, request_id: str) -> dict[str, Any] | None:
    if not is_safe_id(request_id) or len(request_id) > 128:
        raise ValueError("invalid_diagnosis_request_id")
    root = session_directory(session)
    if root is None:
        raise ValueError("session_directory_missing")
    path = root / "explicit-diagnoses" / f"{request_id}.json"
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if state["status"] == "running" and time.time() > state["deadline"]:
        return {**_public(state), "status": "failed", "reason": "diagnosis_timeout"}
    return _public(state)


async def request_diagnosis(session: Any, request_id: str, question: str) -> dict[str, Any]:
    if not is_safe_id(request_id) or len(request_id) > 128:
        raise ValueError("invalid_diagnosis_request_id")
    if not isinstance(question, str) or not question.strip() or len(question) > 8192:
        raise ValueError("invalid_diagnosis_question")
    root = session_directory(session)
    if root is None or not is_session_causal(session):
        raise ValueError("causal_session_required")
    path = root / "explicit-diagnoses" / f"{request_id}.json"
    question_hash = hashlib.sha256(question.encode()).hexdigest()
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["question_hash"] != question_hash:
            raise ValueError("diagnosis_request_conflict")
        if state["status"] == "running" and time.time() > state["deadline"]:
            # Do not rerun an interrupted/expired attempt or overwrite its file
            # from a polling worker. An identical request remains terminal.
            return {**_public(state), "status": "failed", "reason": "diagnosis_timeout"}
        return _public(state)
    if len(_tasks) >= MAX_ACTIVE_DIAGNOSES:
        raise ValueError("diagnosis_capacity_exhausted")
    rows = _page_ao_records(root, session.session_id)
    if not rows:
        raise ValueError("empty_ledger")
    if len(rows) > MAX_RECORDS or len(json.dumps(rows).encode()) > MAX_SNAPSHOT_BYTES:
        raise ValueError("diagnosis_snapshot_budget_exceeded")
    source_watermark = _stable_snapshot_watermark(
        session.session_id, [_view_ao_record(row, session.session_id) for row in rows]
    )
    rows = _capture_request_boundary(rows, session.session_id, request_id, question)
    if len(rows) > MAX_RECORDS or len(json.dumps(rows).encode()) > MAX_SNAPSHOT_BYTES:
        raise ValueError("diagnosis_snapshot_budget_exceeded")
    watermark = _stable_snapshot_watermark(
        session.session_id, [_view_ao_record(row, session.session_id) for row in rows]
    )
    snapshot = AoSnapshot(
        records=tuple(AORecord.from_dict(row) for row in rows),
        watermark=watermark,
        question=question,
        outcome=None,
    )
    state = {
        "request_id": request_id,
        "question_hash": question_hash,
        "status": "running",
        "watermark": watermark,
        "deadline": time.time() + TIMEOUT_SECONDS,
        "snapshot": rows,
        "source_watermark": source_watermark,
        "request_boundary_ao_id": rows[-1]["ao_id"],
    }
    try:
        write_exclusive(path, state)
    except FileExistsError:
        return await request_diagnosis(session, request_id, question)
    task = asyncio.create_task(_run(session, root, path, state, snapshot))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return _public(state)


async def _run(session: Any, root: Path, path: Path, state: dict, snapshot: AoSnapshot) -> None:
    try:

        async def construct():
            model = await session._get_vlm_config()
            if model is None:
                raise ValueError("diagnosis_model_unavailable")
            return await asyncio.to_thread(
                run_production_aig,
                snapshot,
                root=root,
                task_run_id=f"diagnosis-{state['request_id']}",
                providers=providers_for_vlm(model),
            )

        result = await asyncio.wait_for(construct(), timeout=TIMEOUT_SECONDS)
        if result.status != "frozen" or result.report_path is None:
            state.update(status="failed", reason=result.reason)
        else:
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            attribution = report.get("report", {})
            refs = attribution.get("decisive_ao_refs", [])
            if (
                not attribution.get("responsible_agent")
                or attribution["responsible_agent"] == CONTROL_PARTICIPANT
                or not refs
                or len(snapshot.records) - 1 in refs
            ):
                raise ValueError("diagnosis_source_attribution_missing")
            report["evidence_scope"] = "session AO snapshot plus separately attributed diagnosis API admission; no verified task outcome supplied"
            content = json.dumps(report, ensure_ascii=False, sort_keys=True)
            uri = f"{session._session_uri}/diagnoses/{state['request_id']}/report.json"
            await session._viking_fs.write_file(uri, content, ctx=session.ctx)
            state.update(
                status="completed",
                report=report,
                report_uri=uri,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                revision_id=result.revision_id,
            )
    except asyncio.TimeoutError:
        state.update(status="failed", reason="diagnosis_timeout")
    except Exception:
        # Provider errors can contain credentials. Never return their text.
        state.update(status="failed", reason="diagnosis_execution_failed")
    _save(path, state)
