#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ADR-0045 Wave E2 four-domain verdict smoke driver.

Synthetic mode defaults to zero-LLM and is what the unit tests run. Real mode reads
verdict JSONL plus exported AORecord JSONL. A missing API key is fail-closed
unless ``--dry-run`` or an injected builder is supplied.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.causal_memory.aig0045_gemma_roles import (  # noqa: E402
    GemmaCriticProvider,
    GemmaReaderProvider,
    GemmaRefinerProvider,
)
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
from benchmark.causal_memory.model_usage import read_usage  # noqa: E402
from openviking.session.aig_role_chain import AigContractError  # noqa: E402
from openviking.session.aig_tool_surface import AigToolSurface, AoSnapshot  # noqa: E402
from openviking.session.aig_toolloop_builder import (  # noqa: E402
    FakeToolloopProvider,
    ToolAuditRecord,
    ToolloopProvider,
    run_toolloop,
)
from openviking.session.aig_toolloop_critic import CriticView, FakeCriticProvider  # noqa: E402
from openviking.session.aig_toolloop_orchestrator import run_critic_refiner  # noqa: E402
from openviking.session.aig_toolloop_reader import (  # noqa: E402
    DeterministicReader,
    ReaderError,
    parse_reader_report,
    run_reader,
)
from openviking.session.aig_toolloop_refiner import FakeRefinerProvider  # noqa: E402
from openviking.session.ao_ledger import AORecord  # noqa: E402
from openviking.session.append_only import write_exclusive  # noqa: E402
from openviking.session.outcome_signal import (  # noqa: E402
    normalize_verdict,
    select_attribution_mode,
)
from openviking.session.paper_graph_phi import evaluate_phi  # noqa: E402

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


def load_real_materials(
    verdicts_path: Path, sessions_dir: Path, manifest_path: Path | None = None
) -> list[MaterialSpec]:
    """Join verdict JSONL rows to exported AORecord JSONL under ``sessions_dir``."""
    by_session = _load_session_groups(sessions_dir)
    original_questions = {}
    if manifest_path is not None:
        for line in Path(manifest_path).read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                original_questions[str(entry["episode_id"])] = entry["turns"][0]["prompt"]
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
            if len({record.ao_id for record in records}) != len(records):
                raise ValueError("material_duplicate_ao_id")
            if any(
                right.sequence != left.sequence + 1
                for left, right in zip(records, records[1:], strict=False)
            ):
                raise ValueError("material_ao_out_of_order")
            materials.append(
                MaterialSpec(
                    run_id=_safe_run_id(run_id),
                    domain_kind=domain,
                    style="real",
                    question=str(
                        original_questions.get(str(row.get("episode_id")))
                        or row.get("question")
                        or (row.get("detail") or {}).get("original_question")
                        or _question_from_records(records)
                        or run_id
                    ),
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
    critic_factory: Callable[[str], Any] | None = None,
    refiner_factory: Callable[[str], Any] | None = None,
    reader_factory: Callable[[str], Any] | None = None,
    jobs: int = 1,
    attempts: int = 1,
    attempt_policy: str = "fixed",
) -> dict[str, Any]:
    """Run every material. Role factories receive the attribution mode.

    New roles default to fake/fake/deterministic. Each configured role factory
    creates a fresh provider per material attempt; dry-run invokes no factory.

    ``jobs > 1`` runs materials on a thread pool. Rows keep input order and
    one material's exception becomes a ``driver_exception`` row instead of
    aborting the batch. A shared ``builder`` would then be called from
    several threads, so the CLI passes a per-material factory instead.

    ``attempts > 1`` runs each material N times (paper-style test protocol)
    and folds attempts into one material row: ``submit`` is any-attempt
    submit, quality fields come from the first submitting attempt, and the
    per-attempt outcomes ride along under ``attempts``.
    """
    if attempt_policy not in {"fixed", "until-verified"}:
        raise ValueError("invalid_attempt_policy")
    attempts = max(1, int(attempts))
    specs: list[MaterialSpec] = []
    bases: dict[str, str] = {}
    for spec in materials:
        for index in range(1, attempts + 1):
            run_id = spec.run_id if attempts == 1 else f"{spec.run_id}-a{index}"
            specs.append(replace(spec, run_id=run_id))
            bases[run_id] = spec.run_id
    options = {
        "root": root,
        "max_calls": max_calls,
        "max_seconds": max_seconds,
        "dry_run": dry_run,
        "builder": builder,
        "builder_factory": builder_factory,
        "role_factories": {
            "critic": critic_factory,
            "refiner": refiner_factory,
            "reader": reader_factory,
        },
    }
    if attempt_policy == "fixed":
        outcomes = _run_all(specs, jobs=jobs, **options)
    else:
        groups = [specs[index : index + attempts] for index in range(0, len(specs), attempts)]
        if jobs <= 1:
            outcomes = [
                outcome for group in groups for outcome in _run_until_verified(group, **options)
            ]
        else:
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                futures = [pool.submit(_run_until_verified, group, **options) for group in groups]
                outcomes = [outcome for future in futures for outcome in future.result()]
    rows: list[dict[str, Any]] = []
    provider_audit: list[dict[str, Any]] = []
    if attempts == 1:
        for row, extra in outcomes:
            rows.append(row)
            provider_audit.extend(extra)
    else:
        groups: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]]]]] = {}
        for outcome in outcomes:
            base = bases[outcome[0]["run_id"]]
            groups.setdefault(base, []).append(outcome)
        for spec in materials:
            attempt_rows = groups[spec.run_id]
            submitting = [row for row, _extra in attempt_rows if row.get("submit") is True]
            clean = [row for row, _extra in attempt_rows if row.get("reader_verified") is True]
            source = (clean or submitting or [attempt_rows[-1][0]])[0]
            best = dict(source)
            best["run_id"] = spec.run_id
            best["submit"] = bool(submitting)
            best["attempts"] = [
                {
                    "fallback_reason": row.get("fallback_reason"),
                    "phi_violations_after_refine": row.get("phi_violations_after_refine"),
                    "reader_schema_ok": row.get("reader_schema_ok"),
                    "run_id": row["run_id"],
                    "submit": row.get("submit") is True,
                    "builder_submitted": row.get("builder_submitted", False),
                    "graph_frozen": row.get("graph_frozen", False),
                    "reader_verified": row.get("reader_verified", False),
                    "status": row.get("status"),
                    **(
                        {
                            key: row[key]
                            for key in ("reader_report", "reader_report_hash", "reader_report_path")
                            if key in row
                        }
                    ),
                    **(
                        {"provider_failures": row["provider_failures"]}
                        if row.get("provider_failures")
                        else {}
                    ),
                }
                for row, _extra in attempt_rows
            ]
            if not submitting and best.get("triggered"):
                best["fallback_reason"] = f"all {len(attempt_rows)} attempts failed to submit"
            rows.append(best)
            for _row, extra in attempt_rows:
                provider_audit.extend(extra)
    gates = score_gates(rows)
    return {
        "gates": gates,
        "materials": rows,
        "attempt_summary": {
            "scheduled": len(specs),
            "executed": len(outcomes),
            "triggered": sum(bool(row.get("triggered")) for row, _ in outcomes),
            "builder_submitted": sum(bool(row.get("builder_submitted")) for row, _ in outcomes),
            "graph_frozen": sum(bool(row.get("graph_frozen")) for row, _ in outcomes),
            "reader_verified": sum(bool(row.get("reader_verified")) for row, _ in outcomes),
            "skipped_no_signal": sum(
                row.get("status") == "skipped_no_signal" for row, _ in outcomes
            ),
        },
        "provider_audit": provider_audit,
        "schema": SCHEMA,
        "settings": {
            "attempts": attempts,
            "attempt_policy": attempt_policy,
            "dry_run": dry_run,
            "jobs": jobs,
            "max_calls": max_calls,
            "max_seconds": max_seconds,
        },
    }


def _run_until_verified(
    specs: Sequence[MaterialSpec], **options: Any
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    outcomes = []
    for spec in specs:
        outcome = _run_one_safe(spec, **options)
        outcomes.append(outcome)
        if outcome[0].get("reader_verified") or not outcome[0].get("triggered"):
            break
    return outcomes


def _run_all(
    materials: Sequence[MaterialSpec],
    *,
    root: Path,
    max_calls: int,
    max_seconds: float,
    dry_run: bool,
    builder: ToolloopProvider | None,
    builder_factory: Callable[[str], ToolloopProvider] | None,
    role_factories: Mapping[str, Callable[[str], Any] | None],
    jobs: int,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    if jobs <= 1 or len(materials) <= 1:
        return [
            _run_one_safe(
                spec,
                root=root,
                max_calls=max_calls,
                max_seconds=max_seconds,
                dry_run=dry_run,
                builder=builder,
                builder_factory=builder_factory,
                role_factories=role_factories,
            )
            for spec in materials
        ]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(
                _run_one_safe,
                spec,
                root=root,
                max_calls=max_calls,
                max_seconds=max_seconds,
                dry_run=dry_run,
                builder=builder,
                builder_factory=builder_factory,
                role_factories=role_factories,
            )
            for spec in materials
        ]
        return [future.result() for future in futures]


def _run_one_safe(spec: MaterialSpec, **kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``_run_one`` plus batch isolation: one crash must not kill the others."""
    try:
        row, audit = _run_one(spec, **kwargs)
        row.setdefault("builder_submitted", row.get("submit") is True)
        row.setdefault("graph_frozen", False)
        row["reader_verified"] = (
            row.get("graph_frozen") is True
            and row.get("reader_schema_ok") is True
            and row.get("decisive_refs_in_snapshot") is True
            and not row.get("provider_failures")
        )
        row["status"] = (
            "skipped_no_signal"
            if not row.get("triggered")
            else "reader_verified"
            if row["reader_verified"]
            else "graph_frozen"
            if row["graph_frozen"]
            else "builder_submitted"
            if row["builder_submitted"]
            else "failed"
        )
        return row, audit
    except Exception as exc:  # noqa: BLE001 - recorded as a failed material
        code = getattr(exc, "code", None)
        suffix = f":{code}" if isinstance(code, str) and code else ""
        row: dict[str, Any] = {
            "budget_exhausted": False,
            "decisive_refs_in_snapshot": None,
            "domain": spec.domain_kind,
            "fallback_reason": f"driver_exception:{type(exc).__name__}{suffix}",
            "builder_submitted": False,
            "graph_frozen": False,
            "reader_verified": False,
            "status": "failed",
            "mode": None,
            "phi_violations_after_refine": None,
            "reader_schema_ok": None,
            "run_id": spec.run_id,
            "submit": False,
            "triggered": True,
            "verdict": "driver_exception",
        }
        return row, []


def score_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Five ADR-0045 lines. Budget exhaustion is recorded, not a fail line."""
    triggered = [row for row in rows if row.get("verdict") != "no-signal"]
    provider_failed = [row for row in triggered if row.get("provider_failures")]
    submitted = [
        row for row in triggered if row.get("submit") is True and not row.get("provider_failures")
    ]
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
        if (
            row.get("reader_schema_ok") is True
            and row.get("decisive_refs_in_snapshot") is True
            and not row.get("provider_failures")
        ):
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
            "pass": phi_total == 0 and (phi_scored > 0 or not triggered) and not provider_failed,
            "scored": phi_scored,
        },
        "reader_schema_and_refs": {
            "pass": reader_rate >= SUBMIT_RATE_THRESHOLD and not provider_failed,
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
        "--manifest",
        type=Path,
        help="Original episode manifest, preferred over historical AO text.",
    )
    parser.add_argument(
        "--sessions",
        type=Path,
        help=f"Directory of AORecord JSONL (export default {DEFAULT_EXPORT_PATH}).",
    )
    parser.add_argument(
        "--attempt-policy",
        choices=("fixed", "until-verified"),
        default="fixed",
        help="Fixed N-run evaluation, or debug mode stopping at first complete verification.",
    )
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Material-level parallelism. With gemma and jobs>1 each material "
        "gets its own provider instance; HTTP usage is appended to the work-root ledger.",
    )
    parser.add_argument(
        "--url", default=None, help="Chat completions endpoint, including a shared local LLM pool."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the gemma provider model (e.g. gpt-oss-120b cross-backbone probe).",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Runs per material (paper-style test protocol). >1 folds attempts "
        "into one row: submit = any attempt submitted; quality fields from "
        "the first submitting attempt; per-attempt outcomes under 'attempts'.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--critic", choices=("fake", "gemma"), default="fake")
    parser.add_argument("--refiner", choices=("fake", "gemma"), default="fake")
    parser.add_argument("--reader", choices=("deterministic", "gemma"), default="deterministic")
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
        materials = load_real_materials(args.verdicts, args.sessions, args.manifest)
    builder_choice = args.builder or ("fake" if args.material == "synthetic" else "gemma")
    work = args.work_root or Path(tempfile.mkdtemp(prefix="aig0045-"))
    work.mkdir(parents=True, exist_ok=True)
    factory: Callable[[str], ToolloopProvider] | None = None
    shared: ToolloopProvider | None = None
    if args.dry_run:
        factory = None
    elif builder_choice == "fake":
        factory = _fake_builder
    elif args.jobs > 1:
        try:
            LenovoGemmaProvider(
                **_model_kwargs(args.model, args.url)
            )  # fail fast; instance discarded
        except LenovoGemmaError as exc:
            print(f"gemma provider failed ({exc.code}); export {API_KEY_ENV}", file=sys.stderr)
            return 2
        factory = _parallel_gemma_factory(
            args.model, url=args.url, usage_path=work / "model-usage.jsonl"
        )
    else:
        try:
            shared = LenovoGemmaProvider(
                audit_path=work / "provider_audit.jsonl",
                usage_path=work / "model-usage.jsonl",
                **_model_kwargs(args.model, args.url),
            )
        except LenovoGemmaError as exc:
            print(f"gemma provider failed ({exc.code}); export {API_KEY_ENV}", file=sys.stderr)
            return 2
    role_factories: dict[str, Callable[[str], Any]] = {}
    if not args.dry_run:
        for role, provider_type in (
            ("critic", GemmaCriticProvider),
            ("refiner", GemmaRefinerProvider),
            ("reader", GemmaReaderProvider),
        ):
            if getattr(args, role) == "gemma":
                try:
                    provider_type(
                        **_model_kwargs(args.model, args.url)
                    )  # Validate credentials, zero requests.
                except LenovoGemmaError as exc:
                    print(
                        f"{role} gemma provider failed ({exc.code}); export {API_KEY_ENV}",
                        file=sys.stderr,
                    )
                    return 2
                role_factories[role] = _gemma_role_factory(
                    provider_type, args.model, url=args.url, usage_path=work / "model-usage.jsonl"
                )
    report = run_smoke(
        materials,
        root=work,
        max_calls=args.max_calls,
        max_seconds=args.max_seconds,
        dry_run=args.dry_run,
        builder=shared,
        builder_factory=factory,
        critic_factory=role_factories.get("critic"),
        refiner_factory=role_factories.get("refiner"),
        reader_factory=role_factories.get("reader"),
        jobs=max(1, args.jobs),
        attempts=max(1, args.attempts),
        attempt_policy=args.attempt_policy,
    )
    report["settings"].update(
        {
            "builder": builder_choice,
            "critic": args.critic,
            "refiner": args.refiner,
            "reader": args.reader,
        }
    )
    report["model_usage_log"] = str(work / "model-usage.jsonl")
    report["token_usage"] = read_usage(work / "model-usage.jsonl")
    write_report(report, args.out)
    if shared is not None:
        shared.flush_audit(work / "provider_audit.jsonl")
    print(
        json.dumps({"gates": report["gates"], "out": str(args.out)}, ensure_ascii=False, indent=2)
    )
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
    role_factories: Mapping[str, Callable[[str], Any] | None],
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
    _bind_audit_context(provider, role="builder", run_id=spec.run_id)
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
    extra.extend(_provider_stats(provider, skip=prior_stats, role="builder", run_id=spec.run_id))
    row["toolloop_audit"] = [_audit_dict(item) for item in loop.audit]
    if loop.status != "submitted" or loop.draft is None:
        row["budget_exhausted"] = loop.status == "budget_exhausted"
        marker = loop.fallback_marker or {}
        row["fallback_reason"] = marker.get("reason") or loop.status
        return row, extra
    row["submit"] = True
    row["builder_submitted"] = True
    surface = AigToolSurface(snapshot, root=case_root, run_id=spec.run_id)
    critic = (
        role_factories["critic"](mode) if role_factories.get("critic") else FakeCriticProvider()
    )
    refiner = (
        role_factories["refiner"](mode) if role_factories.get("refiner") else FakeRefinerProvider()
    )
    _bind_audit_context(critic, role="critic", run_id=spec.run_id)
    _bind_audit_context(refiner, role="refiner", run_id=spec.run_id)
    if isinstance(critic, GemmaCriticProvider):
        critic.bind_view(CriticView(surface))
    try:
        cycle = run_critic_refiner(
            surface,
            snapshot,
            critic=critic,
            refiner=refiner,
            root=case_root,
            run_id=spec.run_id,
            frozen_at=FROZEN_AT,
            mode=mode,
        )
    except Exception as exc:  # Record role failure for both serial and parallel batches.
        row["fallback_reason"] = f"critic_refiner_failed:{getattr(exc, 'code', type(exc).__name__)}"
        return row, extra
    finally:
        _collect_role_audit(critic, "critic", spec.run_id, row, extra)
        _collect_role_audit(refiner, "refiner", spec.run_id, row, extra)
    if any(item.get("role") == "critic" for item in row.get("provider_failures", [])):
        row["submit"] = False
        row["fallback_reason"] = "critic_provider_failed"
        return row, extra
    graph = surface.get_graph()
    violations = evaluate_phi(graph).violations
    row["phi_violations_after_refine"] = len(violations)
    row["phi_violation_ids"] = [item.violation_id for item in violations]
    if cycle.status != "frozen" or cycle.revision is None:
        row["fallback_reason"] = (cycle.fallback_marker or {}).get("reason") or cycle.status
        return row, extra
    row["graph_frozen"] = True
    try:
        cycle.run.submit_draft(cycle.run.session_for("builder"), surface.get_graph())
    except AigContractError:
        pass
    reader = (
        role_factories["reader"](mode) if role_factories.get("reader") else DeterministicReader()
    )
    _bind_audit_context(reader, role="reader", run_id=spec.run_id)
    try:
        navigation = run_reader(
            cycle.revision,
            snapshot,
            reader,
            root=case_root,
            run_id=spec.run_id,
            construction_run=cycle.run,
            sidecar_path=cycle.sidecar_path,
            mode=mode,
        )
    except Exception as exc:
        row["reader_schema_ok"] = False
        row["fallback_reason"] = f"reader_failed:{getattr(exc, 'code', type(exc).__name__)}"
        return row, extra
    finally:
        _collect_role_audit(reader, "reader", spec.run_id, row, extra)
    row["budget_exhausted"] = navigation.budget_exhausted
    row["reader_report"] = navigation.report.to_dict()
    row["reader_report_hash"] = navigation.reader_report_hash
    report_path = case_root / "reader-report.json"
    write_exclusive(
        report_path,
        {
            "run_id": spec.run_id,
            "report": navigation.report.to_dict(),
            "reader_report_hash": navigation.reader_report_hash,
            "anchor_candidates": list(navigation.anchor_candidates),
            "reads": list(navigation.reads),
            "budget_exhausted": navigation.budget_exhausted,
            "sidecar_path": str(cycle.sidecar_path),
        },
    )
    row["reader_report_path"] = str(report_path)
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
    success = [row for row in rows if row.get("mode") == "success" and row.get("submit") is True]
    inspectable = [row for row in success if isinstance(row.get("effect_notes"), list)]
    excluded_without_notes = len(success) - len(inspectable)
    exclude_note = "submitted-but-unfrozen rows (no effect_notes) are excluded from inspection"
    if not inspectable:
        return {
            "dirty": 0,
            "excluded_without_notes": excluded_without_notes,
            "markers": list(ERROR_NOTE_MARKERS),
            "note": (
                "no submitted success materials to sample"
                if not success
                else f"no frozen success rows with effect_notes; {exclude_note}"
            ),
            "pass": True,
            "sampled": 0,
            "sampled_fraction": 1.0,
        }
    dirty = 0
    inspected = 0
    for row in inspectable:
        notes = row.get("effect_notes")
        inspected += 1
        if any(
            isinstance(note, str) and any(marker in note for marker in ERROR_NOTE_MARKERS)
            for note in notes
        ):
            dirty += 1
    return {
        "dirty": dirty,
        "excluded_without_notes": excluded_without_notes,
        "markers": list(ERROR_NOTE_MARKERS),
        "note": f"sampled every frozen success row with effect_notes (100% ≥ 10%); {exclude_note}",
        "pass": dirty == 0 and inspected == len(inspectable),
        "sampled": inspected,
        "sampled_fraction": 1.0,
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
    # Historical exports put the original channel message in a tool observation,
    # while their first user action is merely an inbox notification.
    for record in records:
        observation = record.observation
        text = observation.get("text") if isinstance(observation, Mapping) else None
        if isinstance(text, str):
            match = re.search(
                r"(?ms)^\[target=[^\]]*\btype=human\]\s+[^:\n]+:\s*(.*?)(?=^\[target=|\Z)", text
            )
            if match and match[1].strip():
                return match[1].strip()
    for record in records:
        text = record.action.get("text") if isinstance(record.action, Mapping) else None
        if (
            isinstance(text, str)
            and text.strip()
            and record.action.get("role", "user") == "user"
            and not text.startswith("[CoForge inbox notice:")
            and text.strip() not in {"Start.", "No work."}
        ):
            return text.strip()
    return records[0].session_id


def _safe_run_id(run_id: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._:-" else "-" for ch in run_id)
    if cleaned and cleaned[0].isalnum():
        return cleaned[:128]
    return ("r-" + cleaned)[:128]


def _fake_builder(mode: str) -> ToolloopProvider:
    return FakeToolloopProvider(legal_builder_script(mode))


def _parallel_gemma_factory(
    model: str | None = None,
    *,
    url: str | None = None,
    usage_path: Path | None = None,
) -> Callable[[str], ToolloopProvider]:
    """One provider per material with a shared append-only HTTP usage ledger."""

    def factory(mode: str) -> ToolloopProvider:
        return LenovoGemmaProvider(usage_path=usage_path, **_model_kwargs(model, url))

    return factory


def _model_kwargs(model: str | None, url: str | None = None) -> dict[str, str]:
    return {k: v for k, v in {"model": model, "url": url}.items() if v}


def _gemma_role_factory(
    provider_type: type,
    model: str | None,
    *,
    url: str | None = None,
    usage_path: Path | None = None,
) -> Callable[[str], Any]:
    """Fresh provider for every material/attempt, even with jobs=1."""

    def factory(mode: str) -> Any:
        kwargs: dict[str, Any] = _model_kwargs(model, url)
        if usage_path is not None:
            kwargs["usage_path"] = usage_path
        return provider_type(**kwargs)

    return factory


def _bind_audit_context(provider: Any, *, role: str, run_id: str) -> None:
    bind = getattr(provider, "bind_audit_context", None)
    if bind is not None:
        bind(role=role, run_id=run_id)


def _collect_role_audit(
    provider: Any, role: str, run_id: str, row: dict[str, Any], extra: list[dict[str, Any]]
) -> None:
    extra.extend(_provider_stats(provider, role=role, run_id=run_id))
    failures = getattr(provider, "failures", ())
    if failures:
        row.setdefault("provider_failures", []).extend({"role": role, **item} for item in failures)


def _provider_stats(
    provider: Any, *, skip: int = 0, role: str = "builder", run_id: str = ""
) -> list[dict[str, Any]]:
    stats = getattr(provider, "stats", None)
    if not stats:
        return []
    rows: list[dict[str, Any]] = []
    for item in list(stats)[skip:]:
        if hasattr(item, "verb"):
            rows.append(
                {
                    "elapsed_ms": item.elapsed_ms,
                    "status": item.status,
                    "verb": item.verb,
                    "role": role,
                    "run_id": run_id,
                }
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


def _three_rows(session_id: str, *turns: tuple[str, str]) -> tuple[AORecord, ...]:
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
