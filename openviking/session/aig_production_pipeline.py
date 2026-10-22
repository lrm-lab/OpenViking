# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Fail-closed, commit-facing four-role AIG construction."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openviking.session.aig_native_protocol import is_safe_id
from openviking.session.aig_tool_surface import AigToolSurface, AoSnapshot
from openviking.session.aig_toolloop_builder import ToolloopProvider, run_toolloop
from openviking.session.aig_toolloop_critic import CriticProvider
from openviking.session.aig_toolloop_orchestrator import run_critic_refiner
from openviking.session.aig_toolloop_reader import ReaderProvider, run_reader
from openviking.session.aig_toolloop_refiner import RefinerProvider
from openviking.session.extraction_write import NATIVE_EDGE_INDEX_FILENAME, NativeEdgeViewStore
from openviking.session.outcome_signal import select_attribution_mode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AigRoleProviders:
    builder: ToolloopProvider
    critic: CriticProvider
    refiner: RefinerProvider
    reader: ReaderProvider


@dataclass(frozen=True)
class ProductionAigResult:
    status: str
    reason: str
    revision_id: str | None = None
    reader_report_hash: str | None = None
    report_path: Path | None = None
    card_ids: tuple[str, ...] = ()


def _failed_provider(providers: AigRoleProviders) -> bool:
    return any(
        bool(getattr(provider, "failures", ()))
        for provider in (providers.builder, providers.critic, providers.refiner, providers.reader)
    )


def run_production_aig(
    snapshot: AoSnapshot,
    *,
    root: Path,
    task_run_id: str,
    providers: AigRoleProviders,
    frozen_at: str | None = None,
    card_hook: Callable[[Any, AoSnapshot, Any], Any] | None = None,
) -> ProductionAigResult:
    """Publish only after Builder, Critic, Refiner and Reader all finish.

    The Phase 1 fallback remains authoritative whenever construction fails.
    The role audit and draft can remain on disk for diagnosis, but they are
    never registered as a native view until Reader returns a usable report.
    """
    if not isinstance(snapshot, AoSnapshot) or not snapshot.records:
        return ProductionAigResult("fallback", "empty_snapshot")
    if not isinstance(task_run_id, str) or not is_safe_id(task_run_id):
        return ProductionAigResult("fallback", "invalid_task_run_id")
    if len({row.session_id for row in snapshot.records}) != 1:
        return ProductionAigResult("fallback", "mixed_sessions")
    mode = select_attribution_mode(snapshot.outcome) or "neutral"
    question = snapshot.question
    if not isinstance(question, str) or not question.strip():
        return ProductionAigResult("fallback", "question_missing")
    root = Path(root)
    frozen_at = frozen_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        store = NativeEdgeViewStore(root / NATIVE_EDGE_INDEX_FILENAME)
        report_path = root / "aig-paper-audit" / task_run_id / "reader-report.json"
        for existing in store.revisions():
            if existing.task_run_id == task_run_id:
                if report_path.is_file():
                    body = json.loads(report_path.read_text(encoding="utf-8"))
                    if body.get("revision_id") == existing.revision_id and body.get(
                        "reader_report_hash"
                    ):
                        return ProductionAigResult(
                            "frozen",
                            "already_frozen",
                            existing.revision_id,
                            body["reader_report_hash"],
                            report_path,
                        )
                return ProductionAigResult("fallback", "native_revision_without_reader_report")
        built = run_toolloop(
            snapshot, question, providers.builder, root=root, run_id=task_run_id, mode=mode
        )
        if built.status != "submitted" or _failed_provider(providers):
            return ProductionAigResult("fallback", "builder_incomplete")
        surface = AigToolSurface(snapshot, root=root, run_id=task_run_id)
        binder = getattr(providers.critic, "bind_view", None)
        if callable(binder):
            from openviking.session.aig_toolloop_critic import CriticView

            binder(CriticView(surface))
        cycle = run_critic_refiner(
            surface,
            snapshot,
            critic=providers.critic,
            refiner=providers.refiner,
            root=root,
            run_id=task_run_id,
            frozen_at=frozen_at,
            mode=mode,
        )
        if cycle.revision is None or _failed_provider(providers):
            return ProductionAigResult("fallback", "critic_refiner_incomplete")
        if cycle.run.phase == "BUILD":
            cycle.run.submit_draft(cycle.run.session_for("builder"), surface.get_graph())
        read = run_reader(
            cycle.revision,
            snapshot,
            providers.reader,
            root=root,
            run_id=task_run_id,
            construction_run=cycle.run,
            sidecar_path=cycle.sidecar_path,
            mode=mode,
        )
        if (
            read.budget_exhausted
            or not read.report.decisive_ao_refs
            or read.report.unresolved_areas
            or _failed_provider(providers)
        ):
            return ProductionAigResult("fallback", "reader_incomplete")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "task_run_id": task_run_id,
            "revision_id": cycle.revision.revision_id,
            "reader_report_hash": read.reader_report_hash,
            "report": read.report.to_dict(),
            "anchor_candidates": list(read.anchor_candidates),
        }
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, report_path)
        store.write(cycle.revision)
        card_ids: tuple[str, ...] = ()
        if card_hook is not None:
            # ADR-0046: optional post-freeze card proposals. Never alters the result.
            try:
                outcome = card_hook(cycle.revision, snapshot, read)
                card_ids = tuple(getattr(outcome, "card_ids", ()))
            except Exception as exc:
                logger.warning("card hook failed: %s", type(exc).__name__)
        return ProductionAigResult(
            "frozen",
            "complete",
            cycle.revision.revision_id,
            read.reader_report_hash,
            report_path,
            card_ids,
        )
    except Exception as exc:
        logger.warning(
            "four-role AIG construction fell back: %s", type(exc).__name__, exc_info=True
        )
        return ProductionAigResult("fallback", getattr(exc, "code", type(exc).__name__))
