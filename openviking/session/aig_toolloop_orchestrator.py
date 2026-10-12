# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Critic/refiner loop after a paper-graph submit (ADR-0044).

An empty violation set compiles immediately and does not enter the loop.
Otherwise at most ``MAX_REVISION_ROUNDS`` critique/refine rounds run on
``AigConstructionRun``. High-severity residue becomes a deterministic
fallback and is not frozen. Low-severity residue is frozen with
``known_violations``; retrieval does not read that field. Every audit and
edit round is appended to the trajectory log.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openviking.session.aig_native_protocol import canonical_dumps, is_safe_id
from openviking.session.aig_role_chain import MAX_REVISION_ROUNDS, AigConstructionRun
from openviking.session.aig_tool_surface import AigToolSurface, AoSnapshot
from openviking.session.aig_toolloop_builder import CONSTRUCTION_FALLBACK
from openviking.session.aig_toolloop_critic import CriticProvider, audit_semantics
from openviking.session.aig_toolloop_refiner import (
    RefinerProvider,
    RefinerSession,
    run_refiner_round,
)
from openviking.session.append_only import append_jsonl
from openviking.session.influence_view import (
    CONSTRUCTION_MODE_ADAPTIVE_AGENTIC,
    AigNativeViewRevision,
)
from openviking.session.paper_graph_compile import AUDIT_DIRNAME, compile_paper_graph
from openviking.session.paper_graph_phi import PaperViolation, evaluate_phi

_FALLBACK_VIEW = "build_minimal_fallback_view"


class OrchestratorError(ValueError):
    """The loop could not start. ``code`` is stable."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass
class ToolloopCycleResult:
    """Frozen paper graph, or a fallback marker when high violations remain."""

    status: str
    construction_mode: str
    entered_loop: bool
    revision_rounds: int
    revision: AigNativeViewRevision | None
    sidecar_path: Path | None
    sidecar_hash: str | None
    known_violations: tuple[dict[str, Any], ...]
    fallback_marker: dict[str, Any] | None
    trajectory_path: Path
    run: AigConstructionRun


def run_critic_refiner(
    surface: AigToolSurface,
    snapshot: AoSnapshot,
    *,
    critic: CriticProvider,
    refiner: RefinerProvider,
    root: Path,
    run_id: str,
    frozen_at: str,
    construction_run: AigConstructionRun | None = None,
    mode: str = "fail",
) -> ToolloopCycleResult:
    """Audit the submitted draft, refine up to three rounds, then compile or fall back."""
    if not isinstance(surface, AigToolSurface) or not surface.submitted:
        raise OrchestratorError("not_submitted")
    if not isinstance(snapshot, AoSnapshot):
        raise OrchestratorError("invalid_snapshot")
    if not is_safe_id(run_id):
        raise OrchestratorError("invalid_run_id")
    if not isinstance(frozen_at, str) or not frozen_at.strip():
        raise OrchestratorError("invalid_frozen_at")
    overview = surface.get_overview()
    row_count = overview.get("row_count")
    watermark = overview.get("watermark")
    if row_count != len(snapshot.records) or watermark != snapshot.watermark:
        raise OrchestratorError("snapshot_mismatch")
    run = construction_run or AigConstructionRun(run_id)
    if not isinstance(run, AigConstructionRun) or run.run_id != run_id:
        raise OrchestratorError("run_mismatch")
    trajectory = _Trajectory(_trajectory_path(root, run_id), run_id)
    violations = _audit(surface, critic, trajectory, 0, mode=mode)
    if not violations:
        return _freeze(
            surface,
            snapshot,
            root=root,
            run_id=run_id,
            frozen_at=frozen_at,
            known=(),
            trajectory=trajectory,
            run=run,
            entered_loop=False,
            rounds=0,
        )
    run.submit_draft(run.session_for("builder"), surface.get_graph())
    rounds = 0
    for round_index in range(1, MAX_REVISION_ROUNDS + 1):
        rounds = round_index
        run.critique(run.session_for("critic"), [item.to_dict() for item in violations])
        session = RefinerSession(surface, violations)
        refined = run_refiner_round(session, refiner, mode=mode)
        run.refine(run.session_for("refiner"), surface.get_graph())
        trajectory.append(
            "refine_round",
            round_index,
            {
                "edits": [dict(item) for item in refined.edits],
                "unresolvable": [dict(item) for item in refined.unresolvable],
            },
        )
        declared = {item["violation_id"] for item in refined.unresolvable}
        current_ids = {item.violation_id for item in violations}
        if current_ids and current_ids <= declared and not refined.edits:
            break
        violations = _audit(surface, critic, trajectory, round_index, mode=mode)
        if not violations:
            break
    if not violations:
        return _freeze(
            surface,
            snapshot,
            root=root,
            run_id=run_id,
            frozen_at=frozen_at,
            known=(),
            trajectory=trajectory,
            run=run,
            entered_loop=True,
            rounds=rounds,
        )
    if any(item.severity == "high" for item in violations):
        run.fail()
        marker = _fallback_marker(run_id, snapshot.watermark, violations)
        trajectory.append(
            "decision",
            rounds,
            {
                "construction_mode": CONSTRUCTION_FALLBACK,
                "known_violations": [item.to_dict() for item in violations],
                "reason": "high_severity_violations",
                "status": "fallback",
            },
        )
        return ToolloopCycleResult(
            status="fallback",
            construction_mode=CONSTRUCTION_FALLBACK,
            entered_loop=True,
            revision_rounds=rounds,
            revision=None,
            sidecar_path=None,
            sidecar_hash=None,
            known_violations=tuple(item.to_dict() for item in violations),
            fallback_marker=marker,
            trajectory_path=trajectory.path,
            run=run,
        )
    return _freeze(
        surface,
        snapshot,
        root=root,
        run_id=run_id,
        frozen_at=frozen_at,
        known=tuple(item.to_dict() for item in violations),
        trajectory=trajectory,
        run=run,
        entered_loop=True,
        rounds=rounds,
    )


def _audit(
    surface: AigToolSurface,
    critic: CriticProvider,
    trajectory: _Trajectory,
    round_index: int,
    *,
    mode: str = "fail",
) -> tuple[PaperViolation, ...]:
    report = evaluate_phi(surface.get_graph())
    semantic = audit_semantics(surface, report.violations, critic, mode=mode)
    trajectory.append(
        "phi",
        round_index,
        {
            "policy": report.policy_dict(),
            "violations": [item.to_dict() for item in report.violations],
        },
    )
    trajectory.append(
        "critic",
        round_index,
        {"violations": [item.to_dict() for item in semantic]},
    )
    return tuple(sorted([*report.violations, *semantic], key=lambda item: item.violation_id))


def _freeze(
    surface: AigToolSurface,
    snapshot: AoSnapshot,
    *,
    root: Path,
    run_id: str,
    frozen_at: str,
    known: Sequence[Mapping[str, Any]],
    trajectory: _Trajectory,
    run: AigConstructionRun,
    entered_loop: bool,
    rounds: int,
) -> ToolloopCycleResult:
    session_id = snapshot.records[0].session_id
    compiled = compile_paper_graph(
        surface.get_graph(),
        snapshot,
        root=root,
        run_id=run_id,
        session_id=session_id,
        task_run_id=run_id,
        view_id=f"view-{run_id}",
        revision_id=f"rev-{run_id}",
        frozen_at=frozen_at,
        known_violations=known,
    )
    trajectory.append(
        "decision",
        rounds,
        {
            "construction_mode": CONSTRUCTION_MODE_ADAPTIVE_AGENTIC,
            "known_violations": [dict(item) for item in known],
            "sidecar_hash": compiled.sidecar_hash,
            "status": "frozen",
        },
    )
    return ToolloopCycleResult(
        status="frozen",
        construction_mode=CONSTRUCTION_MODE_ADAPTIVE_AGENTIC,
        entered_loop=entered_loop,
        revision_rounds=rounds,
        revision=compiled.revision,
        sidecar_path=compiled.sidecar_path,
        sidecar_hash=compiled.sidecar_hash,
        known_violations=tuple(dict(item) for item in known),
        fallback_marker=None,
        trajectory_path=trajectory.path,
        run=run,
    )


def _fallback_marker(
    run_id: str, watermark: str, violations: Sequence[PaperViolation]
) -> dict[str, Any]:
    return {
        "accepted_as_aig": False,
        "construction_mode": CONSTRUCTION_FALLBACK,
        "deterministic_view": _FALLBACK_VIEW,
        "reason": "high_severity_violations",
        "run_id": run_id,
        "violation_ids": [item.violation_id for item in violations],
        "watermark": watermark,
    }


def _trajectory_path(root: Path, run_id: str) -> Path:
    base = (Path(root) / AUDIT_DIRNAME).resolve()
    run_dir = (base / run_id).resolve()
    if run_dir.parent != base:
        raise OrchestratorError("invalid_run_id")
    return run_dir / "trajectory.jsonl"


class _Trajectory:
    """Append-only hash chain of audits and edits."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self._prev = ""
        self._seq = 0

    def append(self, record_kind: str, round_index: int, body: Mapping[str, Any]) -> None:
        payload: dict[str, Any] = {
            "body": dict(body),
            "prev_content_hash": self._prev,
            "record_kind": record_kind,
            "round": round_index,
            "run_id": self.run_id,
            "seq": self._seq,
        }
        digest = hashlib.sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()
        payload["content_hash"] = digest
        append_jsonl(self.path, payload)
        self._prev = digest
        self._seq += 1
