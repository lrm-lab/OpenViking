#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ADR-0045 Wave E2 four-domain verdict smoke driver.

Synthetic mode is zero-LLM and is what the unit tests run. Real mode reads
verdict JSONL plus exported AORecord JSONL. A missing API key is fail-closed
unless ``--dry-run`` or an injected builder is supplied.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openviking.session.aig_role_chain import AigContractError  # noqa: E402
from openviking.session.aig_tool_surface import AigToolSurface, AoSnapshot  # noqa: E402
from openviking.session.aig_toolloop_builder import (  # noqa: E402
    FakeToolloopProvider,
    ToolAuditRecord,
    ToolloopProvider,
    run_toolloop,
)
from openviking.session.aig_toolloop_critic import FakeCriticProvider  # noqa: E402
from openviking.session.aig_toolloop_orchestrator import run_critic_refiner  # noqa: E402
from openviking.session.aig_toolloop_reader import (  # noqa: E402
    DeterministicReader,
    ReaderError,
    parse_reader_report,
    run_reader,
)
from openviking.session.aig_toolloop_refiner import FakeRefinerProvider  # noqa: E402
from openviking.session.ao_ledger import AORecord  # noqa: E402
from openviking.session.outcome_signal import (  # noqa: E402
    normalize_verdict,
    select_attribution_mode,
)
from openviking.session.paper_graph_phi import evaluate_phi  # noqa: E402

from benchmark.causal_memory.ao_record_export import (  # noqa: E402
    DEFAULT_EXPORT_PATH,
    group_records,
    read_ao_jsonl,
)
from benchmark.causal_memory.lenovo_gemma_provider import (  # noqa: E402
    API_KEY_ENV,
    LenovoGemmaError,
    LenovoGemmaProvider,
)

SCHEMA = "aig0045-verdict-smoke-v1"
FROZEN_AT = "2026-10-12T00:00:00.000Z"
DEFAULT_MAX_CALLS = 64
DEFAULT_MAX_SECONDS = 1800.0
DEFAULT_REPORT = Path("benchmark/causal_memory/out/aig0045_smoke_report.json")
SUBMIT_RATE_THRESHOLD = 0.80
ERROR_NOTE_MARKERS = ("出错", "错误")
FAIL_EFFECT = "复用后下游出错"
SUCCESS_EFFECT = "复用促成最终正确结果"

ROW_KEYS = (
    "run_id",
    "domain",
    "verdict",
    "mode",
    "submit",
    "fallback_reason",
    "phi_violations_after_refine",
    "reader_schema_ok",
    "decisive_refs_in_snapshot",
    "budget_exhausted",
)


@dataclass(frozen=True)
class MaterialSpec:
    """One smoke case: AO rows plus a sidecar/attempt-shaped verdict."""

    run_id: str
    domain_kind: str
    style: str
    question: str
    records: tuple[AORecord, ...]
    reward: Any
    graded: Any
    error: Any
    detail: Mapping[str, Any] | None = None


def synthetic_materials() -> list[MaterialSpec]:
    """Six snapshots (3 code + 3 doc) covering four domains and three verdicts.

    The 4×3 cartesian is 12 cells; these six hit every domain and every
    verdict class. The remaining cells stay in ``tests/test_outcome_signal.py``.
    """
    return [
        _code_spec(
            "syn-lcb-pass",
            "lcb",
            reward=1.0,
            graded=True,
            error=None,
            question="return the sum of two integers",
            work="wrote add(a, b)",
            close="returned 3",
        ),
        _code_spec(
            "syn-swe-fail",
            "swe",
            reward=0.0,
            graded=True,
            error="No patch generated",
            question="fix the off-by-one in the parser",
            work="opened the file and stopped",
            close="no patch",
        ),
        _code_spec(
            "syn-lcb-nosignal",
            "lcb",
            reward=0.0,
            graded=False,
            error="timeout",
            question="compute factorial",
            work="started a solution",
            close="grader never returned",
        ),
        _doc_spec(
            "syn-browsecomp-pass",
            "browsecomp",
            reward=1.0,
            graded=True,
            error=None,
            question="who published the 2024 guidance note?",
            work="opened the agency page",
            close="named the agency",
        ),
        _doc_spec(
            "syn-gdpval-fail",
            "gdpval",
            reward=0.3,
            graded=True,
            error=None,
            question="draft the one-page brief",
            work="summarized two sources",
            close="missed a required section",
        ),
        _doc_spec(
            "syn-browsecomp-nosignal",
            "browsecomp",
            reward=0.0,
            graded=True,
            error="judge parse_error",
            question="when did the notice take effect?",
            work="quoted a date from a blog",
            close="judge output was unparseable",
        ),
    ]


def legal_builder_script(mode: str) -> list[tuple[str, dict[str, Any]]]:
    """A Φ-legal three-block graph. Success notes avoid 出错/错误."""
    effect = SUCCESS_EFFECT if mode == "success" else FAIL_EFFECT
    return [
        ("get_overview", {}),
        (
            "add_block",
            {
                "block_id": "task-1",
                "fields": {
                    "content": "the question",
                    "operative_brief": "start here",
                    "step_refs": [0],
                },
                "kind": "task",
            },
        ),
        (
            "add_block",
            {
                "block_id": "agent-1",
                "fields": {
                    "agent_name": "worker",
                    "authorship": "authored",
                    "input": "do the task",
                    "output": f"worked; {effect}",
                    "step_refs": [1],
                    "summary": "did the work",
                },
                "kind": "agent",
            },
        ),
        (
            "add_block",
            {
                "block_id": "end-1",
                "fields": {"content": f"final; {effect}", "step_refs": [2]},
                "kind": "conclusion",
            },
        ),
        (
            "add_inheritance_note",
            {
                "effect": effect,
                "from": "task-1",
                "inherited": "the question",
                "to": "agent-1",
            },
        ),
        (
            "add_inheritance_note",
            {
                "effect": effect,
                "from": "agent-1",
                "inherited": "the work",
                "to": "end-1",
            },
        ),
        ("submit", {}),
    ]


def load_real_materials(verdicts_path: Path, sessions_dir: Path) -> list[MaterialSpec]:
    """Join verdict JSONL rows to exported AORecord JSONL under ``sessions_dir``."""
    by_session = _load_session_groups(sessions_dir)
    materials: list[MaterialSpec] = []
    with Path(verdicts_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, Mapping):
                continue
            run_id = str(row.get("run_id") or "")
            domain = str(row.get("domain_kind") or "")
            if not run_id or not domain:
                continue
            records = _match_records(row, by_session)
            if not records:
                continue
            materials.append(
                MaterialSpec(
                    run_id=_safe_run_id(run_id),
                    domain_kind=domain,
                    style="real",
                    question=_question_from_records(records) or run_id,
                    records=tuple(records),
                    reward=row.get("reward"),
                    graded=row.get("graded"),
                    error=row.get("error"),
                    detail=row.get("detail") if isinstance(row.get("detail"), Mapping) else None,
                )
            )
    return materials


def run_smoke(
    materials: Sequence[MaterialSpec],
    *,
    root: Path,
    max_calls: int = DEFAULT_MAX_CALLS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    dry_run: bool = False,
    builder: ToolloopProvider | None = None,
    builder_factory: Callable[[str], ToolloopProvider] | None = None,
) -> dict[str, Any]:
    """Run every material. ``builder_factory(mode)`` is used when set."""
    rows: list[dict[str, Any]] = []
    provider_audit: list[dict[str, Any]] = []
    for spec in materials:
        row, extra = _run_one(
            spec,
            root=root,
            max_calls=max_calls,
            max_seconds=max_seconds,
            dry_run=dry_run,
            builder=builder,
            builder_factory=builder_factory,
        )
        rows.append(row)
        provider_audit.extend(extra)
    gates = score_gates(rows)
    return {
        "gates": gates,
        "materials": rows,
        "provider_audit": provider_audit,
        "schema": SCHEMA,
        "settings": {
            "dry_run": dry_run,
            "max_calls": max_calls,
            "max_seconds": max_seconds,
        },
    }


def score_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Five ADR-0045 lines. Budget exhaustion is recorded, not a fail line."""
    triggered = [row for row in rows if row.get("verdict") != "no-signal"]
    submitted = [row for row in triggered if row.get("submit") is True]
    submit_rate = (len(submitted) / len(triggered)) if triggered else 1.0
    phi_total = 0
    phi_scored = 0
    reader_ok = 0
    reader_scored = 0
    for row in triggered:
        phi = row.get("phi_violations_after_refine")
        if isinstance(phi, int):
            phi_total += phi
            phi_scored += 1
        if row.get("reader_schema_ok") is True and row.get("decisive_refs_in_snapshot") is True:
            reader_ok += 1
        if row.get("submit") is True:
            reader_scored += 1
    reader_rate = (reader_ok / reader_scored) if reader_scored else 1.0
    distinguish = _distinguish_success(rows)
    exhausted = sum(1 for row in triggered if row.get("budget_exhausted") is True)
    exhaust_rate = (exhausted / len(triggered)) if triggered else 0.0
    return {
        "budget_exhausted_rate": {
            "pass": True,
            "rate": exhaust_rate,
            "record_only": True,
        },
        "dual_mode_distinguishable": distinguish,
        "phi_violations_zero": {
            "count": phi_total,
            "pass": phi_total == 0 and (phi_scored > 0 or not triggered),
            "scored": phi_scored,
        },
        "reader_schema_and_refs": {
            "pass": reader_rate >= SUBMIT_RATE_THRESHOLD,
            "rate": reader_rate,
            "scored": reader_scored,
        },
        "submit_rate": {
            "pass": submit_rate >= SUBMIT_RATE_THRESHOLD,
            "rate": submit_rate,
            "scored": len(triggered),
            "threshold": SUBMIT_RATE_THRESHOLD,
        },
    }


def write_report(report: Mapping[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--verdicts", type=Path, help="Verdict JSONL for --material real.")
    parser.add_argument(
        "--sessions",
        type=Path,
        help="Directory of AORecord JSONL (export default "
        f"{DEFAULT_EXPORT_PATH}).",
    )
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / DEFAULT_REPORT)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument(
        "--builder",
        choices=("fake", "gemma"),
        default=None,
        help="Override the builder. Default: fake on synthetic, gemma on real.",
    )
    args = parser.parse_args(argv)
    if args.material == "synthetic":
        materials = synthetic_materials()
    else:
        if args.verdicts is None or args.sessions is None:
            print("--material real requires --verdicts and --sessions", file=sys.stderr)
            return 2
        materials = load_real_materials(args.verdicts, args.sessions)
    builder_choice = args.builder or ("fake" if args.material == "synthetic" else "gemma")
    work = args.work_root or Path(tempfile.mkdtemp(prefix="aig0045-"))
    work.mkdir(parents=True, exist_ok=True)
    factory: Callable[[str], ToolloopProvider] | None = None
    shared: ToolloopProvider | None = None
    if args.dry_run:
        factory = None
    elif builder_choice == "fake":
        factory = _fake_builder
    else:
        try:
            shared = LenovoGemmaProvider(
                audit_path=work / "provider_audit.jsonl",
            )
        except LenovoGemmaError as exc:
            print(f"gemma provider failed ({exc.code}); export {API_KEY_ENV}", file=sys.stderr)
            return 2
    report = run_smoke(
        materials,
        root=work,
        max_calls=args.max_calls,
        max_seconds=args.max_seconds,
        dry_run=args.dry_run,
        builder=shared,
        builder_factory=factory,
    )
    write_report(report, args.out)
    if shared is not None:
        shared.flush_audit(work / "provider_audit.jsonl")
    print(json.dumps({"gates": report["gates"], "out": str(args.out)}, ensure_ascii=False, indent=2))
    return 0


def _run_one(
    spec: MaterialSpec,
    *,
    root: Path,
    max_calls: int,
    max_seconds: float,
    dry_run: bool,
    builder: ToolloopProvider | None,
    builder_factory: Callable[[str], ToolloopProvider] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outcome = normalize_verdict(
        spec.domain_kind,
        reward=spec.reward,
        graded=spec.graded,
        error=spec.error,
        detail=spec.detail,
    )
    verdict = "no-signal" if outcome is None else outcome.signal
    mode = select_attribution_mode(outcome)
    snapshot = AoSnapshot(
        records=spec.records,
        watermark=f"wm-{spec.run_id}",
        question=spec.question,
        outcome=outcome,
    )
    row: dict[str, Any] = {
        "budget_exhausted": False,
        "decisive_refs_in_snapshot": None,
        "domain": spec.domain_kind,
        "fallback_reason": None,
        "mode": mode,
        "phi_violations_after_refine": None,
        "reader_schema_ok": None,
        "run_id": spec.run_id,
        "submit": False,
        "triggered": mode is not None,
        "verdict": verdict,
    }
    extra: list[dict[str, Any]] = []
    if mode is None:
        row["fallback_reason"] = "not_triggered"
        return row, extra
    if dry_run:
        return row, extra
    provider = builder_factory(mode) if builder_factory is not None else builder
    if provider is None:
        row["fallback_reason"] = "missing_provider"
        return row, extra
    prior_stats = len(getattr(provider, "stats", ()) or ())
    case_root = Path(root) / spec.run_id
    case_root.mkdir(parents=True, exist_ok=True)
    loop = run_toolloop(
        snapshot,
        spec.question,
        provider,
        root=case_root,
        run_id=spec.run_id,
        max_calls=max_calls,
        max_seconds=max_seconds,
        mode=mode,
    )
    extra.extend(_provider_stats(provider, skip=prior_stats))
    row["toolloop_audit"] = [_audit_dict(item) for item in loop.audit]
    if loop.status != "submitted" or loop.draft is None:
        row["budget_exhausted"] = loop.status == "budget_exhausted"
        marker = loop.fallback_marker or {}
        row["fallback_reason"] = marker.get("reason") or loop.status
        return row, extra
    row["submit"] = True
    surface = AigToolSurface(snapshot, root=case_root, run_id=spec.run_id)
    cycle = run_critic_refiner(
        surface,
        snapshot,
        critic=FakeCriticProvider(),
        refiner=FakeRefinerProvider(),
        root=case_root,
        run_id=spec.run_id,
        frozen_at=FROZEN_AT,
        mode=mode,
    )
    graph = surface.get_graph()
    row["phi_violations_after_refine"] = len(evaluate_phi(graph).violations)
    if cycle.status != "frozen" or cycle.revision is None:
        row["fallback_reason"] = (cycle.fallback_marker or {}).get("reason") or cycle.status
        return row, extra
    try:
        cycle.run.submit_draft(cycle.run.session_for("builder"), surface.get_graph())
    except AigContractError:
        pass
    navigation = run_reader(
        cycle.revision,
        snapshot,
        DeterministicReader(),
        root=case_root,
        run_id=spec.run_id,
        construction_run=cycle.run,
        sidecar_path=cycle.sidecar_path,
        mode=mode,
    )
    row["budget_exhausted"] = navigation.budget_exhausted
    try:
        parse_reader_report(navigation.report.to_dict(), navigation.graph)
        row["reader_schema_ok"] = True
    except ReaderError:
        row["reader_schema_ok"] = False
    refs = list(navigation.report.decisive_ao_refs)
    row["decisive_refs_in_snapshot"] = bool(refs) and all(
        0 <= item < len(snapshot.records) for item in refs
    )
    row["effect_notes"] = [
        edge.get("effect")
        for edge in navigation.graph.get("edges", [])
        if isinstance(edge, Mapping)
    ]
    return row, extra


def _distinguish_success(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    success = [
        row
        for row in rows
        if row.get("mode") == "success" and row.get("submit") is True
    ]
    if not success:
        return {
            "note": "no submitted success materials to sample",
            "pass": True,
            "sampled": 0,
            "sampled_fraction": 1.0,
        }
    sample = list(success)
    dirty = 0
    inspected = 0
    for row in sample:
        notes = row.get("effect_notes")
        if not isinstance(notes, list):
            continue
        inspected += 1
        if any(
            isinstance(note, str) and any(marker in note for marker in ERROR_NOTE_MARKERS)
            for note in notes
        ):
            dirty += 1
    fraction = 1.0
    return {
        "dirty": dirty,
        "markers": list(ERROR_NOTE_MARKERS),
        "note": "sampled every submitted success row (100% ≥ 10%)",
        "pass": dirty == 0 and inspected == len(success),
        "sampled": inspected,
        "sampled_fraction": fraction,
    }


def _load_session_groups(sessions_dir: Path) -> dict[str, list[AORecord]]:
    sessions_dir = Path(sessions_dir)
    grouped: dict[str, list[AORecord]] = {}
    if sessions_dir.is_file():
        return group_records(read_ao_jsonl(sessions_dir))
    if not sessions_dir.is_dir():
        return grouped
    for path in sorted(sessions_dir.glob("*.jsonl")):
        records = read_ao_jsonl(path)
        if not records:
            continue
        grouped[path.stem] = records
        for session_id, rows in group_records(records).items():
            grouped.setdefault(session_id, rows)
    return grouped


def _match_records(
    verdict: Mapping[str, Any], grouped: Mapping[str, list[AORecord]]
) -> list[AORecord]:
    run_id = str(verdict.get("run_id") or "")
    session_id = str(verdict.get("session_id") or "")
    for key in (run_id, session_id, _safe_run_id(run_id)):
        if key and key in grouped:
            return list(grouped[key])
    return []


def _question_from_records(records: Sequence[AORecord]) -> str:
    first = records[0]
    text = first.action.get("text") if isinstance(first.action, Mapping) else None
    if isinstance(text, str) and text.strip():
        return text.strip()
    return first.session_id


def _safe_run_id(run_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._:-" else "-" for ch in run_id)
    if cleaned and cleaned[0].isalnum():
        return cleaned[:128]
    return ("r-" + cleaned)[:128]


def _fake_builder(mode: str) -> ToolloopProvider:
    return FakeToolloopProvider(legal_builder_script(mode))


def _provider_stats(provider: Any, *, skip: int = 0) -> list[dict[str, Any]]:
    stats = getattr(provider, "stats", None)
    if not stats:
        return []
    rows: list[dict[str, Any]] = []
    for item in list(stats)[skip:]:
        if hasattr(item, "verb"):
            rows.append(
                {"elapsed_ms": item.elapsed_ms, "status": item.status, "verb": item.verb}
            )
    return rows


def _audit_dict(item: ToolAuditRecord) -> dict[str, Any]:
    return asdict(item)


def _code_spec(
    run_id: str,
    domain: str,
    *,
    reward: Any,
    graded: Any,
    error: Any,
    question: str,
    work: str,
    close: str,
) -> MaterialSpec:
    mode = _preview_mode(domain, reward=reward, graded=graded, error=error)
    effect = SUCCESS_EFFECT if mode == "success" else FAIL_EFFECT
    return MaterialSpec(
        run_id=run_id,
        domain_kind=domain,
        style="code",
        question=question,
        records=_three_rows(
            run_id,
            ("worker", question),
            ("worker", f"{work}; {effect}"),
            ("worker", f"{close}; {effect}"),
        ),
        reward=reward,
        graded=graded,
        error=error,
    )


def _doc_spec(
    run_id: str,
    domain: str,
    *,
    reward: Any,
    graded: Any,
    error: Any,
    question: str,
    work: str,
    close: str,
) -> MaterialSpec:
    mode = _preview_mode(domain, reward=reward, graded=graded, error=error)
    effect = SUCCESS_EFFECT if mode == "success" else FAIL_EFFECT
    return MaterialSpec(
        run_id=run_id,
        domain_kind=domain,
        style="doc",
        question=question,
        records=_three_rows(
            run_id,
            ("worker", question),
            ("worker", f"{work}; {effect}"),
            ("worker", f"{close}; {effect}"),
        ),
        reward=reward,
        graded=graded,
        error=error,
    )


def _preview_mode(domain: str, *, reward: Any, graded: Any, error: Any) -> str | None:
    return select_attribution_mode(
        normalize_verdict(domain, reward=reward, graded=graded, error=error)
    )


def _three_rows(
    session_id: str, *turns: tuple[str, str]
) -> tuple[AORecord, ...]:
    rows: list[AORecord] = []
    for index, (participant, text) in enumerate(turns):
        rows.append(
            AORecord(
                ao_id=f"{session_id}-{index}",
                session_id=session_id,
                sequence=index + 1,
                action={
                    "participant_id": participant,
                    "segment_id": "seg",
                    "text": text,
                },
                observation={"text": text},
                skill_invocations=[],
                message_ref=None,
                created_at=FROZEN_AT,
                captured_state="live",
            )
        )
    return tuple(rows)


if __name__ == "__main__":
    raise SystemExit(main())
