# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Attach an official outcome to an already committed session and rebuild its AIG.

A verdict can arrive after the commit (judge outage, rescoring) or the commit-time
four-role construction can fall back. This entry replays the commit-time four roles
(and the card hook) on the session's immutable AO ledger with the supplied evaluator
results. It never edits the ledger or an existing revision: the rebuild runs under its
own ``task_run_id`` (deterministic in the evaluator results, so a repeat is a no-op) and
the result is recorded under ``adjudications/``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openviking.session.aig_commit_wiring import run_commit_four_roles
from openviking.session.append_only import write_exclusive
from openviking.session.causal_mode import is_session_causal
from openviking.session.outcome_contract import external_evidence_from_session
from openviking.session.trajectory_index_hook import session_directory

_locks: dict[str, asyncio.Lock] = {}


def adjudication_run_id(evaluator_results: list[dict[str, Any]]) -> str:
    blob = json.dumps(evaluator_results, ensure_ascii=False, sort_keys=True)
    return "adj-" + hashlib.sha256(blob.encode()).hexdigest()[:24]


def _fresh_run_id(root: Path, base_id: str) -> str:
    """A fallback leaves its toolloop run dir (with a submitted draft) behind; reusing the same
    run id would make the builder see ``already_submitted`` and rebuild nothing. Retry under a
    new attempt-suffixed id instead."""
    runs = root / "toolloop-runs"
    if not (runs / base_id).exists():
        return base_id
    n = 2
    while (runs / f"{base_id}-r{n}").exists():
        n += 1
    return f"{base_id}-r{n}"


def _public(record: dict[str, Any]) -> dict[str, Any]:
    keys = ("task_run_id", "status", "reason", "revision_id", "card_ids", "signal", "reward")
    return {key: record[key] for key in keys if key in record}


async def adjudicate_session(
    session: Any,
    evaluator_results: list[dict[str, Any]],
    *,
    aig_question: str | None = None,
) -> dict[str, Any]:
    if not isinstance(evaluator_results, list) or not evaluator_results:
        raise ValueError("evaluator_results_required")
    evidence = external_evidence_from_session(evaluator_results=evaluator_results)
    if not evidence or len({item.signal for item in evidence}) != 1:
        raise ValueError("ambiguous_outcome")
    if aig_question is not None and (not isinstance(aig_question, str) or not aig_question.strip()):
        raise ValueError("aig_question must be a non-empty string")
    root = session_directory(session)
    if root is None or not is_session_causal(session):
        raise ValueError("causal_session_required")
    base_id = adjudication_run_id(evaluator_results)
    path = Path(root) / "adjudications" / f"{base_id}.json"
    lock = _locks.setdefault(str(path), asyncio.Lock())
    async with lock:
        if path.exists():  # same evaluator results already adjudicated: terminal, idempotent
            return {**_public(json.loads(path.read_text(encoding="utf-8"))), "duplicate": True}
        model = await session._get_vlm_config()
        if model is None:
            raise ValueError("adjudication_model_unavailable")
        run_id = _fresh_run_id(Path(root), base_id)
        session.outcome_evaluator_results = list(evaluator_results)
        if aig_question is not None:
            session.aig_question = aig_question.strip()
        result = await asyncio.to_thread(run_commit_four_roles, session, run_id, model)
        reward = next(
            (r["reward"] for r in evaluator_results if isinstance(r.get("reward"), (int, float))),
            None,
        )
        record = {
            "task_run_id": run_id,
            "adjudication_id": base_id,
            "status": result.status,
            "reason": result.reason,
            "revision_id": result.revision_id,
            "card_ids": list(result.card_ids),
            "card_trace": result.card_trace,
            "signal": evidence[0].signal,
            "reward": reward,
            "evaluator_results": evaluator_results,
            "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if result.status == "frozen":
            path.parent.mkdir(parents=True, exist_ok=True)
            write_exclusive(path, record)
        # A fallback is not recorded: the same call can be retried.
        return _public(record)
