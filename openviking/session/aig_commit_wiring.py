# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Commit snapshot and verified failure signal for four-role AIG."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openviking.session.aig_production_pipeline import ProductionAigResult, run_production_aig
from openviking.session.aig_tool_surface import AoSnapshot
from openviking.session.aig_vlm_roles import providers_for_vlm
from openviking.session.ao_ledger import AORecord
from openviking.session.causal_mode import is_session_causal
from openviking.session.outcome_contract import external_evidence_from_session
from openviking.session.outcome_signal import OutcomeSignal
from openviking.session.trajectory_index_hook import (
    _page_ao_records,
    _stable_snapshot_watermark,
    _view_ao_record,
    session_directory,
)


def commit_outcome_signal(session: Any) -> OutcomeSignal | None:
    """Use explicit evaluator results; index priority and telemetry are insufficient."""
    raw = getattr(session, "outcome_evaluator_results", None)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        return None
    evidence = external_evidence_from_session(evaluator_results=raw)
    if not evidence or len({item.signal for item in evidence}) != 1:
        return None
    return OutcomeSignal(
        evidence[0].signal,
        "verifier",
        {
            "evidence_refs": [item.evidence_ref for item in evidence],
        },
    )


def commit_failure_signal(session: Any) -> OutcomeSignal | None:
    signal = commit_outcome_signal(session)
    return signal if signal is not None and signal.signal == "fail" else None


def _question(session: Any, records: tuple[AORecord, ...]) -> str | None:
    explicit = getattr(session, "aig_question", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if not records:
        return None
    for field in ("question", "content", "text", "prompt"):
        value = records[0].action.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def run_commit_four_roles(session: Any, task_id: str, vlm_handle: Any) -> ProductionAigResult:
    """Run the account model on a fixed AO ledger page after Phase 2 extraction."""
    session_id = getattr(session, "session_id", None)
    if not isinstance(session_id, str) or not is_session_causal(session):
        return ProductionAigResult("skipped", "causal_mode_disabled")
    signal = commit_outcome_signal(session)
    root = session_directory(session)
    if root is None:
        return ProductionAigResult("fallback", "session_directory_missing")
    rows = _page_ao_records(Path(root), session_id)
    if not rows:
        return ProductionAigResult("fallback", "empty_ledger")
    records = tuple(AORecord.from_dict(row) for row in rows)
    question = _question(session, records)
    if question is None:
        return ProductionAigResult("fallback", "question_missing")
    view_rows = [_view_ao_record(row, session_id) for row in rows]
    snapshot = AoSnapshot(
        records=records,
        watermark=_stable_snapshot_watermark(session_id, view_rows),
        question=question,
        outcome=signal,
    )
    return run_production_aig(
        snapshot,
        root=Path(root),
        task_run_id=task_id,
        providers=providers_for_vlm(vlm_handle),
    )
