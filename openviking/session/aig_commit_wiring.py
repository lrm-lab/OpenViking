# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Commit snapshot and verified failure signal for four-role AIG."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openviking.session.aig_production_pipeline import ProductionAigResult, run_production_aig
from openviking.session.aig_tool_surface import AoSnapshot
from openviking.session.aig_vlm_roles import providers_for_vlm
from openviking.session.ao_ledger import AORecord
from openviking.session.causal_mode import is_session_causal
from openviking.session.outcome_contract import external_evidence_from_session
from openviking.session.outcome_signal import OUTCOME_FEEDBACK_LIMIT, OutcomeSignal
from openviking.session.session_acl import session_owner_label
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
    raw_signal: dict[str, Any] = {"evidence_refs": [item.evidence_ref for item in evidence]}
    rewards = [
        item["reward"]
        for item in raw
        if isinstance(item, dict) and isinstance(item.get("reward"), (int, float))
    ]
    if rewards and len(rewards) == len(raw):
        raw_signal["reward"] = float(min(rewards))  # the weakest evaluator decides
    reports = [
        item["feedback"].strip()
        for item in raw
        if isinstance(item, dict) and isinstance(item.get("feedback"), str) and item["feedback"].strip()
    ]
    if reports:
        raw_signal["feedback"] = "\n\n".join(reports)[:OUTCOME_FEEDBACK_LIMIT]
    return OutcomeSignal(evidence[0].signal, "verifier", raw_signal)


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


def user_root_for_session(session: Any) -> Path | None:
    """On-disk user root that holds the overlay and card stores, or None."""
    from openviking.core.namespace import canonical_user_root

    viking_fs = getattr(session, "_viking_fs", None)
    ctx = getattr(session, "ctx", None)
    if viking_fs is None or ctx is None:
        return None
    try:
        raw = viking_fs._uri_to_path(canonical_user_root(ctx), ctx=ctx)
    except Exception:
        return None
    return Path(raw) if raw else None


def card_hook_for_session(session: Any, vlm_handle: Any) -> Any:
    """Post-freeze card hook, or None. Off unless the session opts in (ADR-0046)."""
    if getattr(session, "experience_cards_enabled", False) is not True:
        return None
    root = user_root_for_session(session)
    if root is None:
        return None
    try:
        label = session_owner_label(session)
    except ValueError:
        return None
    from functools import partial

    from openviking.session.aig_vlm_roles import VLMCardProposer
    from openviking.session.card_proposer import run_card_proposer

    def hook(revision: Any, snapshot: AoSnapshot, read: Any) -> Any:
        provider = VLMCardProposer(vlm_handle, time.monotonic() + 60.0)
        return partial(run_card_proposer, card_root=root, owner_label=label)(
            revision, snapshot, read, provider
        )

    return hook


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
        card_hook=card_hook_for_session(session, vlm_handle),
    )
