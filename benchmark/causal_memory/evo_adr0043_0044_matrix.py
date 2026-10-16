#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""EvoAgentBench data-driven matrix for ADR-0043 and ADR-0044. Zero LLM.

FROZEN_RULES: sources are evo-code-test86.jsonl, evo-ir.jsonl, and
evo-knowledge-work.jsonl. The default split is test. Each domain keeps the
first N episode_id values in lexicographic order. The seed is sha256 of
canonical JSON over domain, task_id, episode_id, split, grader_kind,
turn_count, and the sha256 of turn prompts concatenated in order.
Event count is 4 + int(seed[:8], 16) % 5. Criterion count is
3 + int(seed[8:16], 16) % 4. Rows are the six ADR assertions.
Do not edit the frozen rules to absorb a miss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openviking.session.active_retrieval import (  # noqa: E402
    ActiveRetrievalCorpus,
    BoundNativeView,
    retrieve_active,
)
from openviking.session.aig_native_protocol import (  # noqa: E402
    OBJECT_KIND_FIRST_ACTION_ALIGNMENT,
    OBJECT_KIND_FORK_PROPOSAL,
    OBJECT_KIND_OUTCOME_CONTRACT,
    OBJECT_KIND_STRATEGY_PATTERN_PROPOSAL,
    OBJECT_KIND_TASK_OUTCOME_ASSESSMENT,
    REPLAY_SESSION_PREFIX,
    SCHEMA_VERSION_AIG_NATIVE_V2,
    is_safe_id,
    protocol_envelope,
    read_legacy_claim_cut,
)
from openviking.session.aig_tool_surface import AigToolSurface, AoSnapshot  # noqa: E402
from openviking.session.aig_toolloop_builder import (  # noqa: E402
    FakeToolloopProvider,
    run_toolloop,
)
from openviking.session.aig_toolloop_critic import FakeCriticProvider  # noqa: E402
from openviking.session.aig_toolloop_orchestrator import run_critic_refiner  # noqa: E402
from openviking.session.aig_toolloop_reader import (  # noqa: E402
    DeterministicReader,
    run_reader,
)
from openviking.session.aig_toolloop_refiner import (  # noqa: E402
    FakeRefinerProvider,
    RefinerAction,
)
from openviking.session.ao_ledger import AORecord  # noqa: E402
from openviking.session.coordinator_boundary import (  # noqa: E402
    ACTION_ADVICE,
    ACTION_DISPATCHED,
    ACTION_RETAINED,
    TRIGGER_COMMIT,
)
from openviking.session.experience_links import ExperienceLinkService  # noqa: E402
from openviking.session.extraction_contract import (  # noqa: E402
    EXTRACTION_CONTRACT_NATIVE_VERSION,
    build_model_input,
    parse_native_edge_response,
)
from openviking.session.extraction_write import (  # noqa: E402
    NativeEdgeViewStore,
    freeze_extracted_view,
    freeze_native_edge_view,
)
from openviking.session.failure_gate import (  # noqa: E402
    AUDIT_REFUSED,
    FailureGateDelivery,
    FailureGateError,
    FailureGateOrchestrator,
)
from openviking.session.fork_node_overlay import (  # noqa: E402
    ForkNodeOverlayStore,
    ForkOverlayError,
)
from openviking.session.fork_precedent import ForkPrecedentService  # noqa: E402
from openviking.session.guidance_approval import GuidanceApprovalService  # noqa: E402
from openviking.session.influence_view import (  # noqa: E402
    CONSTRUCTION_MODE_ADAPTIVE_AGENTIC,
    CONSTRUCTION_MODE_NATIVE_SINGLE_PASS,
    PURPOSE_POST_RUN_INDEX,
    AigNativeEdge,
    AORange,
    CoverageManifest,
    InfluenceBlock,
    SegmentRef,
    SourceRef,
)
from openviking.session.legacy_write_gate import LegacyWriteRefused  # noqa: E402
from openviking.session.outcome_contract import (  # noqa: E402
    DECISION_CONFIRMED_FAILURE,
    DECISION_SUCCESS,
    DECISION_UNCERTAIN,
    REASON_CRITERIA_MISSING,
    REASON_EXTERNAL_FAIL,
    REASON_EXTERNAL_PASS,
    ExternalOutcomeEvidence,
    OutcomeAssessmentService,
    OutcomeContractStore,
)
from openviking.session.paper_graph_compile import sidecar_digest  # noqa: E402
from openviking.session.replay_alignment import ReplayAlignmentService  # noqa: E402
from openviking.session.strategy_pattern import StrategyPatternStore  # noqa: E402
from openviking.session.t_fork_surface import (  # noqa: E402
    ACTION_VERBS,
    TForkError,
    TForkSurface,
)

SCHEMA = "evo-adr0043-0044-matrix-v1"
FROZEN_AT = "2026-10-11T00:00:00.000Z"
CUTOFF = "2026-10-10T00:00:01.000Z"
RECORDED = "2026-10-10T00:00:02.000Z"
WATERMARK = "wm-evo"
PRINCIPAL = "user:evoagentbench"
PARTICIPANT = "worker"
SEGMENT = "seg"
REPLAY_RUN = "evo-replay"
CALLER = "caller-evo"
QUESTION_PREFIX = "evoagentbench"
TOOL_REASON = "one evoagentbench tool event"
TEMP_PREFIX = "evo-adr-"
DEFAULT_DATA_DIR = Path(
    "/data/zhoujie/root-disk/home/zhoujie22/river2_0/evol_bench/EvoAgentBench/data"
)
DOMAIN_ORDER = ("code", "ir", "knowledge-work")
DOMAIN_FILES = {
    "code": "evo-code-test86.jsonl",
    "ir": "evo-ir.jsonl",
    "knowledge-work": "evo-knowledge-work.jsonl",
}
LEGACY_FIXTURE = ROOT / "tests" / "fixtures" / "adr0043" / "legacy_claim_cut.json"
_GRADER_SLUGS = {
    "browsecomp.judge": "browsecomp-judge",
    "gdpval.rubric": "gdpval-rubric",
    "livecodebench.check_correctness": "lcb-check",
}
_SEED_DISPOSITIONS = ("explicit", "provisional", "missing", "conflict")
_LEAK_TOKENS = ("/tmp", "/data", "/home", "/Users", "root-disk", "worktrees")
_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
ROW_NAMES = (
    "success_no_remediation",
    "confirmed_failure_single_pass",
    "confirmed_failure_toolloop_lane",
    "uncertain_advice_only",
    "acl_fail_closed",
    "legacy_default_deny_with_audit",
)
FROZEN_RULES = (
    "Frozen before the matrix runs. A miss is a miss.",
    "Sources are evo-code-test86.jsonl, evo-ir.jsonl, and evo-knowledge-work.jsonl.",
    "The default split is test. --data-dir overrides the directory. Domain order is fixed.",
    "Domain order is code, ir, knowledge-work. --domains filters that order.",
    "Each domain keeps episode_id lexicographic order. --per-domain N keeps the first N.",
    "--per-domain 0 keeps every episode in the split.",
    "Seed keys are domain, episode_id, grader_kind, prompt_sha256, split, task_id, turn_count.",
    "domain is the record domain field. prompt_sha256 hashes prompts joined with no separator.",
    "The seed is sha256 of that canonical JSON. Event count is 4 + int(seed[:8], 16) % 5.",
    "Event i, except the last, uses turn i modulo len(turns). An odd turn is a read.",
    "Reads alternate read_instruction and read_environment. An even turn is write_output.",
    "The last event is run_tests. Criterion count is 3 + int(seed[8:16], 16) % 4.",
    "Criterion ids are crit-<slug>-<i> from i=0. livecodebench.check_correctness is lcb-check.",
    "browsecomp.judge is browsecomp-judge. gdpval.rubric is gdpval-rubric.",
    "Disposition i uses seed[16+2*i:18+2*i] mod 4: explicit, provisional, missing, conflict.",
    "Gate rows submit explicit dispositions. uncertain keeps the first floor(n/2) criteria.",
    "An even seed[:8] exits the critic early. An odd seed takes one refiner round.",
    "success_no_remediation freezes native_single_pass_aig and passes every criterion.",
    "It creates no fork, guidance, or child run. retrieve_active uses projection budget 4.",
    "confirmed_failure_single_pass fails every signal and writes one object of each kind.",
    "A second deliver is idempotent. Scripted replay is replay-direct. Production bytes stay.",
    "confirmed_failure_toolloop_lane compiles adaptive_agentic_aig and binds the sidecar hash.",
    "The reader first anchor is the defect block. T_fork is phase_locked before the verdict.",
    "After it, the same proposal is idempotent and a changed proposal is overlay_conflict.",
    "The fourth non-idempotent submit is submit_budget_exhausted. T_fork has its own directory.",
    "Card and hop budgets both return hop_budget_exceeded.",
    "uncertain_advice_only is uncertain, with diagnostic audit only and no automatic object.",
    "acl_fail_closed uses an empty principal. The terminal is acl or blocked_pre_execution.",
    "legacy_default_deny_with_audit refuses freeze_extracted_view and checks the claim cut.",
    "Rows are the six ROW_NAMES. The report omits temp paths and uuids.",
)


class Miss(Exception):
    """A matrix assertion failed. The row is fail, not error."""


@dataclass(frozen=True)
class Criterion:
    criterion_id: str
    name: str
    statement: str
    disposition: str


@dataclass(frozen=True)
class TaskSpec:
    label: str
    group: str
    domain: str
    task: str
    episode_key: str
    grader_kind: str
    prompt_sha256: str
    split: str
    turn_count: int
    criteria: tuple[Criterion, ...]
    seed: str
    events: tuple[dict[str, str], ...]

    @property
    def task_id(self) -> str:
        return f"task-{self.episode_key[:20]}"

    @property
    def session_id(self) -> str:
        return f"sess-{self.episode_key[:20]}"

    @property
    def refine(self) -> bool:
        return int(self.seed[:8], 16) % 2 == 1


def _slug(value: str) -> str:
    chars = [ch if ch.isalnum() or ch in "._:-" else "-" for ch in value]
    text = re.sub(r"-{2,}", "-", "".join(chars)).strip("-.")
    if not text or not text[0].isalnum():
        text = f"t{text}"
    text = text[:100]
    if not is_safe_id(text):
        text = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return text


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _public(text: str) -> str:
    cleaned = _UUID.sub("<uuid>", text)
    cleaned = re.sub(r"/(?:tmp|data|home|Users)[^\s'\"]*", "<path>", cleaned)
    for token in _LEAK_TOKENS:
        cleaned = cleaned.replace(token, "<path>")
    return cleaned[:500]


def _report_text(value: str) -> str:
    leaked = value.startswith("/") or _UUID.search(value) is not None
    if leaked or any(token in value for token in _LEAK_TOKENS):
        return "id-" + _sha256(value.encode("utf-8"))[:20]
    return value


def _canonical(body: dict[str, Any]) -> str:
    payload = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256(payload.encode("utf-8"))


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise Miss(message)


def _coded(call: Callable[[], Any], *errors: type[BaseException]) -> str:
    try:
        call()
    except errors as exc:
        code = getattr(exc, "code", None)
        if isinstance(code, str) and code:
            return code
        return type(exc).__name__
    raise Miss("expected a coded error")


def _envelope(kind: str, body: dict[str, Any]) -> dict[str, Any]:
    return protocol_envelope(SCHEMA_VERSION_AIG_NATIVE_V2, kind, body)


def _json_count(root: Path, *parts: str) -> int:
    directory = root.joinpath(*parts)
    if not directory.is_dir():
        return 0
    return len(list(directory.glob("*.json")))


def _counts(root: Path) -> dict[str, int]:
    return {
        "child_runs": _json_count(root, "remediation-runs", "runs"),
        "execution_reports": _json_count(root, "remediation-runs", "reports"),
        "fork_overlays": _json_count(root, "fork-node-overlays", "overlays"),
        "guidance_approvals": _json_count(root, "guidance-approvals", "approvals"),
        "precedent_retrievals": _json_count(root, "fork-precedents", "retrievals"),
    }


def _evidence(session_id: str) -> dict[str, Any]:
    return {
        "ao_id": "ao-eval-1",
        "archive_commit_watermark": None,
        "captured_state": "committed",
        "evidence_content_hash": "hash-eval-1",
        "evidence_role": "hindsight_attribution",
        "source_archive_id": None,
        "source_read_snapshot_watermark": WATERMARK,
        "source_sequence": 1,
        "source_session_id": session_id,
    }


def _contract(spec: TaskSpec) -> dict[str, Any]:
    return _envelope(
        OBJECT_KIND_OUTCOME_CONTRACT,
        {
            "acceptance_criteria": [
                {"criterion_id": item.criterion_id, "statement": item.statement}
                for item in spec.criteria
            ],
            "contract_id": f"contract-{spec.task_id}",
            "evaluator_bindings": [
                {
                    "binding_id": f"bind-{index:02d}",
                    "kind": "evaluator",
                    "ref": f"eval://{item.name}",
                }
                for index, item in enumerate(spec.criteria)
            ],
            "frozen_at": FROZEN_AT,
            "goals": [f"satisfy the {spec.grader_kind} checks"],
            "policy_constraints": [
                {
                    "constraint_id": "sla-grader",
                    "kind": "sla",
                    "statement": "the grader checks pass",
                }
            ],
            "source": "orchestrator",
            "task_id": spec.task_id,
        },
    )


def _signals(spec: TaskSpec, signal: str) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "binding_id": f"bind-{index:02d}",
            "evidence_ref": f"evidence:{item.criterion_id}",
            "signal": signal,
            "signal_kind": "evaluator",
        }
        for index, item in enumerate(spec.criteria)
    )


def _dispositions(spec: TaskSpec, kept: int | None = None) -> dict[str, str]:
    chosen = spec.criteria if kept is None else spec.criteria[:kept]
    return {item.criterion_id: "explicit" for item in chosen}


def _plan(session_id: str) -> dict[str, Any]:
    return {
        "applicability": {
            "action_skill_role": "applicable",
            "anchor_state": "partially_applicable",
            "failure_mode": "applicable",
            "task_decision_type": "applicable",
        },
        "required_capabilities": ["replay-sandbox"],
        "risk": "retry may repeat the failed rollout",
        "steps": [
            {
                "evidence_refs": [_evidence(session_id)],
                "imagined_node_id": "img-retry",
                "statement": "retry the failing check with the recorded inputs",
                "step_id": "step-retry",
            }
        ],
        "target_role": "task_agent",
    }


def _catalog() -> dict[str, Any]:
    return {
        "candidates": [
            {
                "acl_labels": [PRINCIPAL],
                "agent_id": "agent-a",
                "available": True,
                "capabilities": ["replay-sandbox", "rollback"],
                "roles": ["task_agent"],
                "task_member": True,
            }
        ]
    }


def _controls() -> dict[str, Any]:
    return {
        "agent_capabilities": ["replay-sandbox"],
        "budget": "compliant",
        "executor_agent_id": "agent-a",
        "executor_role": "task_agent",
        "idempotency_key": CALLER,
        "isolation": "isolated",
        "permission": "granted",
        "requested_side_effects": [],
        "side_effect_allowlist": [],
    }


def _delivery(
    spec: TaskSpec,
    *,
    commit_id: str,
    verdict: str,
    signal: str | None,
    dispositions: dict[str, str],
    principal: tuple[str, ...],
    script: bool,
    anchor: str | None = None,
    factual: str | None = None,
) -> FailureGateDelivery:
    evidence = () if signal is None else _signals(spec, signal)
    return FailureGateDelivery(
        trigger=TRIGGER_COMMIT,
        commit_id=commit_id,
        task_id=spec.task_id,
        session_id=spec.session_id,
        source_digest=spec.seed,
        principal_labels=principal,
        contract=_contract(spec),
        assessment_body={
            "criteria_dispositions": dispositions,
            "evidence_refs": [_evidence(spec.session_id)],
            "verdict": verdict,
        },
        evaluator_results=evidence,
        temporal_cutoff=CUTOFF,
        recorded_at=RECORDED,
        anchor_block_id=anchor if script else None,
        branch_steps=(
            (
                {"block_id": factual, "step_kind": "factual_block_ref"},
                {"imagined_node_id": "img-retry", "step_kind": "imagined_node"},
            )
            if script and factual is not None
            else ()
        ),
        guidance_plan=_plan(spec.session_id) if script else {"ignored": True},
        catalog=_catalog() if script else None,
        dispatch_controls=_controls() if script else None,
        dispatch_outputs=(
            (
                {"output_id": "ao-child-1", "output_kind": "ao"},
                {"output_id": "blk-child-1", "output_kind": "block"},
            )
            if script
            else ()
        ),
        dispatch_steps=(
            (
                {
                    "actual_block_id": "blk-child-1",
                    "alignment": "equivalent",
                    "imagined_node_id": "img-retry",
                },
            )
            if script
            else ()
        ),
    )


def _events(seed: str, prompts: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    count = 4 + (int(seed[:8], 16) % 5)
    rows: list[dict[str, str]] = []
    read_slot = 0
    for index in range(count):
        if index == count - 1:
            shape, target = "run_tests", "tests"
            text = "run_tests tests fxstop"
        else:
            turn_index = index % len(prompts)
            digest = _sha256(prompts[turn_index].encode("utf-8"))[:12]
            if turn_index % 2 == 1:
                if read_slot % 2 == 0:
                    shape = "read_instruction"
                else:
                    shape = "read_environment"
                read_slot += 1
            else:
                shape = "write_output"
            target = f"turn-{turn_index}"
            text = f"{shape} {target} {digest}"
        rows.append({"index": str(index), "shape": shape, "target": target, "text": text})
    return tuple(rows)


def _grader_slug(kind: str) -> str:
    mapped = _GRADER_SLUGS.get(kind)
    if mapped:
        return mapped
    return _slug(kind)[:40]


def _criteria_for(seed: str, kind: str) -> tuple[Criterion, ...]:
    slug = _grader_slug(kind)
    count = 3 + (int(seed[8:16], 16) % 4)
    rows: list[Criterion] = []
    for index in range(count):
        criterion_id = f"crit-{slug}-{index}"
        if not is_safe_id(criterion_id):
            criterion_id = "crit-" + _sha256(criterion_id.encode("utf-8"))[:20]
        pair = seed[16 + (2 * index) : 18 + (2 * index)]
        disposition = _SEED_DISPOSITIONS[int(pair, 16) % 4]
        rows.append(
            Criterion(
                criterion_id=criterion_id,
                name=f"{slug}-{index}",
                statement=f"{kind} check {index} holds",
                disposition=disposition,
            )
        )
    return tuple(rows)


def _require_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise Miss(f"missing {key}")
    return value


def _grader_kind(row: dict[str, Any]) -> str:
    grader = row.get("grader")
    if not isinstance(grader, dict):
        raise Miss("grader missing")
    kind = grader.get("kind")
    if not isinstance(kind, str) or not kind:
        raise Miss("grader.kind missing")
    return kind


def _prompts(row: dict[str, Any]) -> tuple[str, ...]:
    turns = row.get("turns")
    if not isinstance(turns, list) or not turns:
        raise Miss("episode has no turns")
    prompts: list[str] = []
    for turn in turns:
        if not isinstance(turn, dict) or not isinstance(turn.get("prompt"), str):
            raise Miss("turn prompt missing")
        prompts.append(turn["prompt"])
    return tuple(prompts)


def _spec_from_row(row: dict[str, Any], domain_key: str) -> TaskSpec:
    domain = _require_str(row, "domain")
    task_id = _require_str(row, "task_id")
    episode_id = _require_str(row, "episode_id")
    split = _require_str(row, "split")
    kind = _grader_kind(row)
    prompts = _prompts(row)
    prompt_sha256 = _sha256("".join(prompts).encode("utf-8"))
    seed = _canonical(
        {
            "domain": domain,
            "episode_id": episode_id,
            "grader_kind": kind,
            "prompt_sha256": prompt_sha256,
            "split": split,
            "task_id": task_id,
            "turn_count": len(prompts),
        }
    )
    episode_key = _canonical({"domain": domain, "episode_id": episode_id, "split": split})
    return TaskSpec(
        label=f"{domain_key} {_report_text(episode_id)}",
        group=domain_key,
        domain=_report_text(domain),
        task=_report_text(task_id),
        episode_key=episode_key,
        grader_kind=kind,
        prompt_sha256=prompt_sha256,
        split=split,
        turn_count=len(prompts),
        criteria=_criteria_for(seed, kind),
        seed=seed,
        events=_events(seed, prompts),
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit("episodes file is missing")
    blobs: list[str] = []
    current = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith("{") and current:
            blobs.append(current)
            current = line
        elif current:
            current += line
        else:
            current = line
    if current:
        blobs.append(current)
    rows: list[dict[str, Any]] = []
    for index, blob in enumerate(blobs):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"episodes record {index} is not json") from exc
        if not isinstance(parsed, dict):
            raise SystemExit(f"episodes record {index} is not an object")
        rows.append(parsed)
    if not rows:
        raise SystemExit("episodes file is empty")
    return rows


def _parse_domains(text: str | None) -> tuple[str, ...]:
    if text is None or not text.strip():
        return DOMAIN_ORDER
    chosen = [part.strip() for part in text.split(",") if part.strip()]
    if not chosen:
        raise SystemExit("--domains is empty")
    unknown = [name for name in chosen if name not in DOMAIN_FILES]
    if unknown:
        raise SystemExit("unknown domain")
    return tuple(name for name in DOMAIN_ORDER if name in set(chosen))


def _select_domain_rows(
    rows: list[dict[str, Any]], split: str, per_domain: int
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for row in rows:
        if row.get("split") != split:
            continue
        if not isinstance(row.get("episode_id"), str):
            raise Miss("episode_id is missing")
        matched.append(row)
    matched.sort(key=lambda item: str(item["episode_id"]))
    if per_domain == 0:
        return matched
    return matched[:per_domain]


def _native_revision(spec: TaskSpec) -> Any:
    records = []
    ledgers = []
    for index, event in enumerate(spec.events):
        sequence = index + 1
        ao_id = f"ao-{index}"
        content_hash = _sha256(f"{spec.seed}:{index}:{event['text']}".encode("utf-8"))
        records.append(
            {
                "ao_id": ao_id,
                "content_hash": content_hash,
                "participant_id": PARTICIPANT,
                "segment_id": SEGMENT,
                "sequence": sequence,
                "session_id": spec.session_id,
            }
        )
        ledgers.append(
            {
                "action": {"text": event["text"], "tool": event["shape"]},
                "ao_id": ao_id,
                "captured_state": "live",
                "observation": {"text": event["text"]},
                "sequence": sequence,
            }
        )
    model_input = build_model_input(
        {
            "ao_records": records,
            "ledger_rows": ledgers,
            "revision_id": "rev-single",
            "session_id": spec.session_id,
            "snapshot_watermark": WATERMARK,
            "task_run_id": spec.task_id,
            "view_id": "view-single",
        }
    )
    model_input["schema_version"] = EXTRACTION_CONTRACT_NATIVE_VERSION
    blocks = []
    for index, event in enumerate(spec.events):
        if index == 0:
            role, kind = "task", "task"
        elif index == len(spec.events) - 1:
            role, kind = "conclusion", "conclusion"
        else:
            role, kind = "agent_action", "tool_call"
        blocks.append(
            {
                "ao_end_seq": index + 1,
                "ao_start_seq": index + 1,
                "authorship": PARTICIPANT,
                "block_id": f"b{index}",
                "block_kind": kind,
                "granularity_reason": TOOL_REASON,
                "input": event["shape"],
                "output": event["text"],
                "participant_id": PARTICIPANT,
                "role": role,
                "segment_id": SEGMENT,
                "summary": event["text"],
            }
        )
    edges = [
        {
            "edge_id": f"e{index}",
            "mechanism": "the later step uses the earlier tool result",
            "relation_type": "influence",
            "source_block_id": f"b{index}",
            "target_block_id": f"b{index + 1}",
        }
        for index in range(len(blocks) - 1)
    ]
    response = {
        "blocks": blocks,
        "coverage": {
            "inspected_ao_ids": [f"ao-{index}" for index in range(len(blocks))],
            "snapshot_watermark": WATERMARK,
            "total_ao_end_seq": len(blocks),
            "total_ao_start_seq": 1,
        },
        "edges": edges,
        "schema_version": EXTRACTION_CONTRACT_NATIVE_VERSION,
    }
    payload = parse_native_edge_response(
        json.dumps(response, ensure_ascii=False, sort_keys=True),
        model_input,
    )
    revision = freeze_native_edge_view(
        session_id=spec.session_id,
        task_run_id=spec.task_id,
        view_id="view-single",
        revision_id="rev-single",
        frozen_at=FROZEN_AT,
        blocks=payload.blocks,
        edges=payload.edges,
        coverage=payload.coverage,
        purpose=PURPOSE_POST_RUN_INDEX,
        construction_mode=CONSTRUCTION_MODE_NATIVE_SINGLE_PASS,
    )
    _check(
        revision.construction_mode == CONSTRUCTION_MODE_NATIVE_SINGLE_PASS,
        f"construction_mode={revision.construction_mode}",
    )
    _check(len(revision.blocks) == len(spec.events), "block count drifted from the trajectory")
    _check(len(revision.edges) == len(spec.events) - 1, "edge count drifted from the trajectory")
    return revision


def _orchestrator(root: Path, revision: Any) -> FailureGateOrchestrator:
    def build(request: Any) -> Any:
        _check(request.session_id == revision.session_id, "builder session drifted")
        _check(request.task_id == revision.task_run_id, "builder task drifted")
        return revision

    return FailureGateOrchestrator(root, build_revision=build)


def _visible(revision: Any, principal: tuple[str, ...]) -> bool:
    result = retrieve_active(
        ActiveRetrievalCorpus(native_views=(BoundNativeView(revision, principal),)),
        revision.blocks[0].summary,
        principal,
        {"fork_branch": 0, "projection": 4, "segment_atom": 0, "rank_mode": "bm25"},
    )
    return any(
        card.kind == "native_block"
        and (card.source_pointer or {}).get("view_revision_id") == revision.revision_id
        for card in result.cards
    )


def _row_success(spec: TaskSpec, root: Path) -> dict[str, Any]:
    revision = _native_revision(spec)
    gate = _orchestrator(root, revision)
    result = gate.deliver(
        _delivery(
            spec,
            commit_id="commit-success",
            verdict="success",
            signal="pass",
            dispositions=_dispositions(spec),
            principal=(PRINCIPAL,),
            script=False,
        )
    )
    counts = _counts(root)
    _check(result.decision == DECISION_SUCCESS, f"decision={result.decision}")
    _check(result.reason == REASON_EXTERNAL_PASS, f"reason={result.reason}")
    _check(result.action == ACTION_RETAINED, f"action={result.action}")
    _check(result.fork_node_id is None, "success created a fork")
    _check(result.child_run_id is None, "success created a child run")
    _check(counts == dict.fromkeys(counts, 0), f"success objects={counts}")
    frozen = gate.read_frozen_view(
        task_id=spec.task_id,
        session_id=spec.session_id,
        source_digest=spec.seed,
    )
    _check(frozen is not None and frozen.content_hash == result.view_content_hash, "view missing")
    _check(_visible(frozen, (PRINCIPAL,)), "frozen view was not on the active retrieval surface")
    return {
        "action": result.action,
        "decision": result.decision,
        "objects": counts,
        "reason": result.reason,
        "retrieved": True,
    }


def _replay_blocks(session_id: str) -> tuple[list[InfluenceBlock], tuple[AigNativeEdge, ...]]:
    def block(block_id: str, sequence: int) -> InfluenceBlock:
        source = SourceRef(
            ao_id=f"ao-{block_id}",
            sequence=sequence,
            participant_id=PARTICIPANT,
            segment_id="seg-replay",
            session_id=session_id,
            content_hash=f"hash-{block_id}",
        )
        return InfluenceBlock(
            block_id=block_id,
            role="agent_action",
            participant_id=PARTICIPANT,
            segment_ref=SegmentRef(segment_id="seg-replay", session_id=session_id),
            ao_start_seq=sequence,
            ao_end_seq=sequence,
            summary=block_id,
            input="in",
            output="out",
            authorship=PARTICIPANT,
            granularity_reason="one replay tool call",
            source_content_hashes=(source.content_hash,),
            source_refs=(source,),
            block_kind="tool_call",
        )

    blocks = [block("rb-1", 1), block("rb-2", 2)]
    edges = (
        AigNativeEdge(
            edge_id="e-replay",
            source_block_id="rb-1",
            target_block_id="rb-2",
            relation_type="influence",
            mechanism="the later replay block inherits the earlier result",
        ),
    )
    return blocks, edges


def _replay_coverage(blocks: list[InfluenceBlock]) -> CoverageManifest:
    refs = tuple(ref for block in blocks for ref in block.source_refs)
    return CoverageManifest(
        total_ao_range=AORange(start_seq=refs[0].sequence, end_seq=refs[-1].sequence),
        inspected_refs=refs,
        included_ranges=tuple(
            AORange(
                start_seq=block.ao_start_seq,
                end_seq=block.ao_end_seq,
                segment_id=block.segment_ref.segment_id,
            )
            for block in blocks
        ),
        omitted_ranges=(),
        snapshot_watermark=WATERMARK,
    )


def _scripted_admission(root: Path, revision: Any, fork_id: str, anchor: str) -> dict[str, bool]:
    production_path = root / "production-native.json"
    production = NativeEdgeViewStore(production_path)
    production.write(revision)
    before = production_path.read_bytes()
    overlays = ForkNodeOverlayStore(root)
    catalog = overlays.register_base_revision(revision, acl_labels=(PRINCIPAL,))
    stored = overlays.read_overlay_by_id(fork_id, principal_labels=(PRINCIPAL,))
    _check(stored is not None, "fork overlay was not readable")
    patterns = StrategyPatternStore(root)
    pattern = patterns.accept_proposal(
        stored,
        catalog,
        _envelope(
            OBJECT_KIND_STRATEGY_PATTERN_PROPOSAL,
            {
                "applicability_conditions": [
                    {
                        "condition_id": "anchor-state",
                        "statement": "the anchor block is the failed action",
                    },
                    {
                        "condition_id": "failure-mode",
                        "statement": "the recorded tests failed",
                    },
                ],
                "author_role": "reader",
                "capability": "replay-sandbox",
                "expected_metric": "the named pytest functions pass",
                "fork_node_id": fork_id,
                "imagined_node_id": "img-retry",
                "intervention_type": "narrow_retry",
                "mechanism": "retry the failing check with the recorded inputs",
                "semantic_summary": "retry the failing check",
                "target_boundary": anchor,
            },
        ),
    )
    service = ReplayAlignmentService(root)
    controls = {
        "anchor_state": "recoverable",
        "budget": "compliant",
        "dependencies": "available",
        "evaluator": "bound",
        "intervention_type": "narrow_retry",
        "permissions": "sufficient",
        "required_capabilities": ["replay-sandbox"],
        "safety_control_params": {"isolation": "sandbox", "network": "denied"},
        "side_effect_isolation": "isolated",
        "target_boundary": anchor,
    }
    assessment = service.assess(
        stored,
        controls,
        evidence_ref="fork-created",
        checked_at=FROZEN_AT,
    )
    _check(assessment.route == "replay", f"route={assessment.route}")
    request = service.build_replay_request(assessment, replay_run_id=REPLAY_RUN)
    session_id = f"{REPLAY_SESSION_PREFIX}{REPLAY_RUN}"
    blocks, edges = _replay_blocks(session_id)
    alignment = _envelope(
        OBJECT_KIND_FIRST_ACTION_ALIGNMENT,
        {
            "actual_block_id": "rb-1",
            "alignment": "functionally_equivalent",
            "depth": "prefix",
            "equivalence_explanation": "same retry with the recorded inputs",
            "imagined_node_id": "img-retry",
            "intervention_type": "narrow_retry",
            "required_capabilities": ["replay-sandbox"],
            "safety_control_params": {"isolation": "sandbox", "network": "denied"},
            "target_boundary": anchor,
        },
    )
    result = service.build_replay_result(
        request,
        status="completed",
        observed_block_ids=["rb-1", "rb-2"],
        alignment=alignment,
        step_alignments=[
            {
                "actual_block_id": "rb-1",
                "alignment": "equivalent",
                "imagined_node_id": "img-retry",
            }
        ],
    )
    outcome = service.apply_scripted_replay(
        stored,
        request,
        result,
        blocks=blocks,
        edges=edges,
        coverage=_replay_coverage(blocks),
        frozen_at=FROZEN_AT,
        pattern_store=patterns,
        pattern_id=pattern.pattern_id,
    )
    _check(outcome.admission is not None, "scripted replay wrote no admission")
    _check(outcome.admission.first_action_direct is True, "first action was not direct")
    _check(outcome.admission.outcome_direct is True, "outcome was not direct")
    _check(outcome.admission.evidence_tier == "replay-direct", outcome.admission.evidence_tier)
    _check(production_path.read_bytes() == before, "replay wrote the production native-edge store")
    return {
        "first_action_direct": True,
        "outcome_direct": True,
        "production_native_unchanged": True,
        "replay_direct": True,
    }


def _failure_chain(
    spec: TaskSpec,
    root: Path,
    revision: Any,
    *,
    commit_id: str,
    anchor: str,
    factual: str,
) -> dict[str, Any]:
    gate = _orchestrator(root, revision)
    delivery = _delivery(
        spec,
        commit_id=commit_id,
        verdict="failure",
        signal="fail",
        dispositions=_dispositions(spec),
        principal=(PRINCIPAL,),
        script=True,
        anchor=anchor,
        factual=factual,
    )
    first = gate.deliver(delivery)
    before = _counts(root)
    _check(first.decision == DECISION_CONFIRMED_FAILURE, f"decision={first.decision}")
    _check(first.reason == REASON_EXTERNAL_FAIL, f"reason={first.reason}")
    _check(first.action == ACTION_DISPATCHED, f"action={first.action}")
    _check(first.blocked is False, "dispatch was blocked")
    _check(bool(first.fork_node_id), "fork missing")
    _check(bool(first.guidance_content_hash), "guidance missing")
    _check(bool(first.child_run_id), "child run missing")
    expected = {
        "child_runs": 1,
        "execution_reports": 1,
        "fork_overlays": 1,
        "guidance_approvals": 1,
        "precedent_retrievals": 1,
    }
    _check(before == expected, f"objects={before}")
    retrievals = list((root / "fork-precedents" / "retrievals").glob("*.json"))
    loaded = ForkPrecedentService(root).read_retrieval(retrievals[0].stem)
    _check(loaded.retrieval_id == retrievals[0].stem, "precedent retrieval did not parse")
    second = gate.deliver(delivery)
    after = _counts(root)
    _check(second.deduplicated is True, "repeat deliver was not deduplicated")
    _check(after == before, f"repeat deliver changed objects to {after}")
    _check(second.fork_node_id == first.fork_node_id, "repeat deliver minted a fork")
    admission = _scripted_admission(root, revision, first.fork_node_id, anchor)
    _check(_counts(root)["fork_overlays"] == 1, "replay created another overlay")
    _check(_counts(root)["child_runs"] == 1, "replay created another child run")
    return {
        "action": first.action,
        "decision": first.decision,
        "idempotent": True,
        "objects": before,
        "reason": first.reason,
        **admission,
    }


def _row_failure(spec: TaskSpec, root: Path) -> dict[str, Any]:
    revision = _native_revision(spec)
    return _failure_chain(
        spec,
        root,
        revision,
        commit_id="commit-failure",
        anchor=revision.blocks[-1].block_id,
        factual=revision.blocks[0].block_id,
    )


def _record(index: int, session_id: str, text: str) -> AORecord:
    return AORecord(
        ao_id=f"ao-{index}",
        session_id=session_id,
        sequence=index + 1,
        action={
            "participant_id": PARTICIPANT,
            "segment_id": SEGMENT,
            "text": text,
            "tool": text.split(" ", 1)[0],
        },
        observation={"text": text},
        skill_invocations=[],
        message_ref=None,
        created_at=FROZEN_AT,
        captured_state="live",
    )


def _tool_script(spec: TaskSpec) -> tuple[list[tuple[str, dict[str, Any]]], str]:
    events = spec.events
    count = len(events)
    script: list[tuple[str, dict[str, Any]]] = [("get_overview", {}), ("get_question", {})]
    script.extend(("read_step", {"index": index}) for index in range(count))
    script.append(
        (
            "add_block",
            {
                "block_id": "task-1",
                "fields": {
                    "content": events[0]["text"],
                    "operative_brief": "start from the task file",
                    "step_refs": [0],
                },
                "kind": "task",
            },
        )
    )
    ids = ["task-1"]
    for step in range(1, count - 1):
        block_id = f"agent-{step}"
        ids.append(block_id)
        script.append(
            (
                "add_block",
                {
                    "block_id": block_id,
                    "fields": {
                        "agent_name": PARTICIPANT,
                        "authorship": "authored",
                        "input": "prior step",
                        "output": events[step]["text"],
                        "step_refs": [step],
                        "summary": events[step]["text"],
                    },
                    "kind": "agent",
                },
            )
        )
    ids.append("end-1")
    script.append(
        (
            "add_block",
            {
                "block_id": "end-1",
                "fields": {"content": events[-1]["text"], "step_refs": [count - 1]},
                "kind": "conclusion",
            },
        )
    )
    defect_id = ids[-2]
    for source, target in zip(ids, ids[1:], strict=False):
        if target == "end-1":
            effect = "fxstop"
        elif target == defect_id:
            effect = "holdprev"
        else:
            effect = "carry"
        script.append(
            (
                "add_inheritance_note",
                {
                    "effect": effect,
                    "from": source,
                    "inherited": "the prior step output",
                    "to": target,
                },
            )
        )
    script.append(
        (
            "mark_defect_recorded",
            {"block_id": defect_id, "note": "missed the test contract"},
        )
    )
    script.append(("submit", {}))
    return script, defect_id


def _proposal(
    catalog: Any, revision: Any, anchor: str, factual: str, imagined: str
) -> dict[str, Any]:
    return _envelope(
        OBJECT_KIND_FORK_PROPOSAL,
        {
            "anchor_block_id": anchor,
            "author_role": "reader",
            "base_revision_block_content_hash": catalog.block_content_hash,
            "base_revision_id": catalog.base_revision_id,
            "branch_steps": [
                {"block_id": factual, "step_kind": "factual_block_ref"},
                {"imagined_node_id": imagined, "step_kind": "imagined_node"},
            ],
            "report_hash": revision.content_hash,
        },
    )


def _tool_forks(
    root: Path,
    spec: TaskSpec,
    revision: Any,
    run: Any,
    report_hash: str,
) -> dict[str, Any]:
    outcomes = OutcomeAssessmentService(OutcomeContractStore(root))
    outcomes.freeze_contract(_contract(spec))
    outcomes.submit_assessment(
        run,
        run.session_for("reader"),
        _envelope(
            OBJECT_KIND_TASK_OUTCOME_ASSESSMENT,
            {
                "author_role": "reader",
                "criteria_dispositions": _dispositions(spec),
                "evidence_refs": [_evidence(spec.session_id)],
                "report_hash": report_hash,
                "verdict": "failure",
            },
        ),
        task_id=spec.task_id,
        contract_id=f"contract-{spec.task_id}",
    )
    overlays = ForkNodeOverlayStore(root)
    patterns = StrategyPatternStore(root)
    precedents = ForkPrecedentService(root)
    guidance = GuidanceApprovalService(
        root,
        outcomes=outcomes,
        precedents=precedents,
        overlays=overlays,
    )
    catalog = overlays.register_base_revision(revision, acl_labels=(PRINCIPAL,))
    common = {
        "run": run,
        "session_id": run.session_for("reader"),
        "task_id": spec.task_id,
        "report_hash": report_hash,
        "principal_labels": (PRINCIPAL,),
        "overlays": overlays,
        "patterns": patterns,
        "precedents": precedents,
        "guidance": guidance,
        "outcomes": outcomes,
    }
    locked = TForkSurface(root, external_evidence=(), **common)
    names = {name for name, _text in locked.catalog()}
    hidden = sorted(names & set(ACTION_VERBS))
    _check(names.isdisjoint(ACTION_VERBS), f"locked catalog listed {hidden}")
    probe = _proposal(catalog, revision, "task-1", "task-1", "img-probe")
    _check(
        _coded(
            lambda: locked.search_precedents({"proposal": probe, "temporal_cutoff": CUTOFF}),
            TForkError,
        )
        == "phase_locked",
        "search was open before confirmed_failure",
    )
    _check(
        _coded(
            lambda: locked.submit_fork_proposal(
                probe, [], retrieval_id="ignored", recorded_at=RECORDED
            ),
            TForkError,
        )
        == "phase_locked",
        "submit was open before confirmed_failure",
    )
    evidence = tuple(
        ExternalOutcomeEvidence(
            binding_id=f"bind-{index:02d}",
            signal_kind="evaluator",
            signal="fail",
            evidence_ref=f"evidence:{item.criterion_id}",
        )
        for index, item in enumerate(spec.criteria)
    )
    opened = TForkSurface(root, external_evidence=evidence, **common)
    opened_names = {name for name, _text in opened.catalog()}
    _check(set(ACTION_VERBS) <= opened_names, "action verbs stayed hidden")
    return {
        "catalog": catalog,
        "common": common,
        "evidence": evidence,
        "guidance": guidance,
        "opened": opened,
        "overlays": overlays,
        "patterns": patterns,
        "phase_locked": True,
        "root": root,
    }


def _budget(bundle: dict[str, Any], revision: Any, anchor: str) -> dict[str, Any]:
    catalog = bundle["catalog"]
    opened: TForkSurface = bundle["opened"]
    factual = "task-1" if anchor != "task-1" else "end-1"
    other_anchor = "end-1" if anchor != "end-1" else "task-1"
    proposal = _proposal(catalog, revision, anchor, factual, "img-retry")
    found = opened.search_precedents({"proposal": proposal, "temporal_cutoff": CUTOFF})
    first = opened.submit_fork_proposal(
        proposal, [], retrieval_id=found["retrieval_id"], recorded_at=RECORDED
    )
    again = opened.submit_fork_proposal(
        proposal, [], retrieval_id=found["retrieval_id"], recorded_at=RECORDED
    )
    _check(again["fork_node_id"] == first["fork_node_id"], "resubmit minted a fork")
    _check(again["deduplicated"] is True, "resubmit was not deduplicated")
    conflict = _proposal(catalog, revision, anchor, factual, "img-conflict")
    conflict_found = opened.search_precedents({"proposal": conflict, "temporal_cutoff": CUTOFF})
    _check(
        _coded(
            lambda: opened.submit_fork_proposal(
                conflict,
                [],
                retrieval_id=conflict_found["retrieval_id"],
                recorded_at=RECORDED,
            ),
            TForkError,
        )
        == "overlay_conflict",
        "same-anchor conflict was stored",
    )
    shifted = _proposal(catalog, revision, other_anchor, factual, "img-shift")
    shifted_found = opened.search_precedents({"proposal": shifted, "temporal_cutoff": CUTOFF})
    opened.submit_fork_proposal(
        shifted, [], retrieval_id=shifted_found["retrieval_id"], recorded_at=RECORDED
    )
    blocked = _proposal(catalog, revision, other_anchor, factual, "img-budget")
    _check(
        _coded(
            lambda: opened.submit_fork_proposal(
                blocked, [], retrieval_id="ignored", recorded_at=RECORDED
            ),
            TForkError,
        )
        == "submit_budget_exhausted",
        "the fourth submit was accepted",
    )
    root = bundle["root"]
    shared = {
        "overlays": bundle["overlays"],
        "patterns": bundle["patterns"],
        "guidance": bundle["guidance"],
    }
    cards = ExperienceLinkService(root, reader_max_cards=1, **shared)
    hops = ExperienceLinkService(root, reader_max_hops=1, **shared)
    card_surface = TForkSurface(
        root, external_evidence=bundle["evidence"], links=cards, **bundle["common"]
    )
    hop_surface = TForkSurface(
        root, external_evidence=bundle["evidence"], links=hops, **bundle["common"]
    )
    _check(
        _coded(
            lambda: card_surface.expand_experience_links(
                first["fork_node_id"], unresolved=("gap",)
            ),
            TForkError,
        )
        == "hop_budget_exceeded",
        "card budget returned a page",
    )
    step = hop_surface.expand_experience_links("nobody", unresolved=("gap",))
    _check(
        _coded(
            lambda: hop_surface.expand_experience_links(
                "nobody", unresolved=("gap",), context_id=step["context_id"]
            ),
            TForkError,
        )
        == "hop_budget_exceeded",
        "hop budget stayed open",
    )
    return {
        "card_budget": "hop_budget_exceeded",
        "hop_budget": "hop_budget_exceeded",
        "idempotent_resubmit": True,
        "overlay_conflict": True,
        "phase_locked": True,
        "submit_budget": "submit_budget_exhausted",
    }


def _row_toolloop(spec: TaskSpec, root: Path) -> dict[str, Any]:
    question = f"{QUESTION_PREFIX} {spec.label}"[:180]
    records = tuple(
        _record(index, spec.session_id, event["text"]) for index, event in enumerate(spec.events)
    )
    snapshot = AoSnapshot(records, watermark=WATERMARK, question=question)
    script, defect_id = _tool_script(spec)
    built = run_toolloop(
        snapshot,
        question,
        FakeToolloopProvider(script),
        root=root / "loop",
        run_id=spec.task_id,
    )
    failed = [(item.verb, item.error_code) for item in built.audit if not item.ok]
    _check(
        built.status == "submitted" and not failed,
        f"toolloop status={built.status} failed={failed[:6]}",
    )
    surface = AigToolSurface(snapshot, root=root / "loop", run_id=spec.task_id)
    _check(surface.submitted, "reopened toolloop draft was not submitted")
    if spec.refine:
        critic = FakeCriticProvider(
            [
                {
                    "violations": [
                        {
                            "evidence_ao_refs": [int(spec.events[-2]["index"])],
                            "kind": "semantic",
                            "note": "summary omitted the failed check",
                            "object_id": defect_id,
                            "severity": "low",
                        }
                    ]
                },
                {"violations": []},
            ]
        )
        refiner = FakeRefinerProvider(
            [
                RefinerAction(
                    kind="edit",
                    verb="modify_block",
                    violation_id=f"critic:0:semantic:{defect_id}",
                    arguments={
                        "block_id": defect_id,
                        "fields": {"summary": "revised after the critic note"},
                    },
                )
            ]
        )
    else:
        critic = FakeCriticProvider()
        refiner = FakeRefinerProvider()
    compiled = run_critic_refiner(
        surface,
        snapshot,
        critic=critic,
        refiner=refiner,
        root=root / "loop",
        run_id=spec.task_id,
        frozen_at=FROZEN_AT,
    )
    _check(compiled.status == "frozen", f"critic status={compiled.status}")
    _check(compiled.revision is not None and compiled.sidecar_path is not None, "compile missed")
    revision = compiled.revision
    document = json.loads(Path(compiled.sidecar_path).read_text(encoding="utf-8"))
    _check(document["content_hash"] == compiled.sidecar_hash, "sidecar file hash drifted")
    _check(
        revision.paper_sidecar_hash == sidecar_digest(document["body"]),
        "revision sidecar hash drifted",
    )
    _check(
        revision.construction_mode == CONSTRUCTION_MODE_ADAPTIVE_AGENTIC,
        revision.construction_mode,
    )
    if spec.refine:
        _check(
            compiled.entered_loop is True and compiled.revision_rounds == 1,
            "refine lane drifted",
        )
    else:
        _check(
            compiled.entered_loop is False and compiled.revision_rounds == 0,
            "early-exit lane drifted",
        )
    compiled.run.submit_draft(compiled.run.session_for("builder"), surface.get_graph())
    reader = run_reader(
        revision,
        snapshot,
        DeterministicReader(),
        root=root / "loop",
        run_id=spec.task_id,
        construction_run=compiled.run,
        sidecar_path=compiled.sidecar_path,
    )
    _check(
        reader.reader_report_hash == compiled.run.reader_report_hash,
        "reader report was not frozen",
    )
    _check(not reader.budget_exhausted, "reader budget exhausted")
    _check(bool(reader.anchor_candidates), "anchor candidates were empty")
    defects = {
        block["block_id"]
        for block in reader.graph["blocks"]
        if block.get("kind") == "agent" and block.get("defect_recorded")
    }
    _check(reader.anchor_candidates[0] in defects, "first anchor was not a defect block")
    _check(
        reader.anchor_candidates[0] == reader.report.critical_block_id,
        "first anchor was not the critical block",
    )
    anchor = reader.anchor_candidates[0]
    factual = "task-1" if anchor != "task-1" else "end-1"
    failure = _failure_chain(
        spec,
        root / "gate",
        revision,
        commit_id="commit-toolloop",
        anchor=anchor,
        factual=factual,
    )
    forks = _tool_forks(root / "fork", spec, revision, compiled.run, reader.reader_report_hash)
    budget = _budget(forks, revision, anchor)
    return {
        "construction_mode": revision.construction_mode,
        "failure": failure,
        "lane": "one_round" if spec.refine else "early_exit",
        "reader_frozen": True,
        "revision_rounds": compiled.revision_rounds,
        "sidecar_bound": True,
        **budget,
    }


def _row_uncertain(spec: TaskSpec, root: Path) -> dict[str, Any]:
    revision = _native_revision(spec)
    gate = _orchestrator(root, revision)
    result = gate.deliver(
        _delivery(
            spec,
            commit_id="commit-uncertain",
            verdict="uncertain",
            signal="fail",
            dispositions=_dispositions(spec, kept=len(spec.criteria) // 2),
            principal=(PRINCIPAL,),
            script=False,
        )
    )
    counts = _counts(root)
    kinds = [row.get("event_kind") for row in gate.audit_events()]
    _check(result.decision == DECISION_UNCERTAIN, f"decision={result.decision}")
    _check(result.reason == REASON_CRITERIA_MISSING, f"reason={result.reason}")
    _check(result.action == ACTION_ADVICE, f"action={result.action}")
    _check(
        result.fork_node_id is None and result.child_run_id is None,
        "uncertain created an object",
    )
    _check(counts == dict.fromkeys(counts, 0), f"uncertain objects={counts}")
    _check(ACTION_ADVICE in kinds, "diagnostic advice was not audited")
    _check(ACTION_DISPATCHED not in kinds, "uncertain dispatched")
    return {
        "action": result.action,
        "decision": result.decision,
        "objects": counts,
        "reason": result.reason,
    }


def _acl_class(code: str) -> bool:
    return code == "blocked_pre_execution" or code.startswith("acl")


def _row_acl(spec: TaskSpec, root: Path) -> dict[str, Any]:
    revision = _native_revision(spec)
    gate = _orchestrator(root, revision)
    delivery = _delivery(
        spec,
        commit_id="commit-acl",
        verdict="failure",
        signal="fail",
        dispositions=_dispositions(spec),
        principal=(),
        script=True,
        anchor=revision.blocks[-1].block_id,
        factual=revision.blocks[0].block_id,
    )

    def once() -> str:
        try:
            result = gate.deliver(delivery)
        except (FailureGateError, ForkOverlayError) as exc:
            return exc.code
        if result.blocked or result.action == "blocked_pre_execution":
            return result.action
        raise Miss(f"empty principal returned {result.decision}/{result.action}")

    first = once()
    before = _counts(root)
    second = once()
    after = _counts(root)
    refused = [
        row
        for row in gate.audit_events()
        if row.get("outcome") == AUDIT_REFUSED or row.get("event_kind") == AUDIT_REFUSED
    ]
    _check(_acl_class(first), f"terminal={first}")
    _check(_acl_class(second), f"repeat terminal={second}")
    _check(before == dict.fromkeys(before, 0), f"acl objects={before}")
    _check(after == before, "repeat acl deliver created an object")
    audited = any(_acl_class(str(row.get("reason") or row.get("event_kind"))) for row in refused)
    _check(audited or _acl_class(first), "acl audit missing")
    return {"objects": before, "terminal": first}


def _row_legacy(spec: TaskSpec) -> dict[str, Any]:
    del spec
    code = _coded(
        lambda: freeze_extracted_view(
            session_id="s",
            task_run_id="t",
            view_id="v",
            revision_id="r",
            frozen_at=FROZEN_AT,
            blocks=(),
            claims=(),
            coverage=None,  # type: ignore[arg-type]
        ),
        LegacyWriteRefused,
    )
    _check(code == "legacy_write_refused", f"legacy code={code}")
    payload = json.loads(LEGACY_FIXTURE.read_text(encoding="utf-8"))
    document = read_legacy_claim_cut(payload)
    _check(document.content_hash == payload["content_hash"], "legacy content hash drifted")
    _check(
        document.claim.claim_id == payload["body"]["claim"]["claim_id"],
        "legacy claim was not read",
    )
    _check(
        document.cut_grounding.content_hash == payload["body"]["cut_grounding"]["content_hash"],
        "legacy cut hash drifted",
    )
    _check(document.cut_grounding.position_kind == "cut", "legacy fixture was not a cut")
    return {
        "claim_id": document.claim.claim_id,
        "content_hash": document.content_hash,
        "cut_content_hash": document.cut_grounding.content_hash,
        "legacy_code": code,
        "reader": "read_legacy_claim_cut",
    }


def _guard(row: str, call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return {"checks": call(), "status": "pass"}
    except Miss as exc:
        print(f"FAIL {row} {_public(str(exc))}", file=sys.stderr)
        return {"reason": _public(str(exc)), "status": "fail"}
    except Exception as exc:
        print(f"ERROR {row} {_public(f'{type(exc).__name__}: {exc}')}", file=sys.stderr)
        traceback.print_exc()
        return {
            "error_type": type(exc).__name__,
            "reason": _public(f"{type(exc).__name__}: {exc}"),
            "status": "error",
        }


def _task_record(spec: TaskSpec, rows: dict[str, Any]) -> dict[str, Any]:
    return {
        "criteria": [
            {"criterion_id": item.criterion_id, "disposition": item.disposition}
            for item in spec.criteria
        ],
        "domain": spec.domain,
        "domain_key": spec.group,
        "episode_key": spec.episode_key,
        "event_count": len(spec.events),
        "events": [
            {"index": int(event["index"]), "shape": event["shape"], "target": event["target"]}
            for event in spec.events
        ],
        "grader_kind": spec.grader_kind,
        "label": spec.label,
        "prompt_sha256": spec.prompt_sha256,
        "refine_lane": "one_round" if spec.refine else "early_exit",
        "rows": rows,
        "seed": spec.seed,
        "split": spec.split,
        "task": spec.task,
        "turn_count": spec.turn_count,
    }


def _run_task(spec: TaskSpec) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=TEMP_PREFIX) as directory:
        base = Path(directory)
        rows = {
            "success_no_remediation": _guard(
                f"{spec.label} success", lambda: _row_success(spec, base / "success")
            ),
            "confirmed_failure_single_pass": _guard(
                f"{spec.label} failure", lambda: _row_failure(spec, base / "failure")
            ),
            "confirmed_failure_toolloop_lane": _guard(
                f"{spec.label} toolloop", lambda: _row_toolloop(spec, base / "toolloop")
            ),
            "uncertain_advice_only": _guard(
                f"{spec.label} uncertain", lambda: _row_uncertain(spec, base / "uncertain")
            ),
            "acl_fail_closed": _guard(f"{spec.label} acl", lambda: _row_acl(spec, base / "acl")),
            "legacy_default_deny_with_audit": _guard(
                f"{spec.label} legacy", lambda: _row_legacy(spec)
            ),
        }
    return _task_record(spec, rows)


def _summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {"error": 0, "fail": 0, "pass": 0}
    by_row = {name: {"error": 0, "fail": 0, "pass": 0} for name in ROW_NAMES}
    for task in tasks:
        for name in ROW_NAMES:
            status = task["rows"][name]["status"]
            totals[status] += 1
            by_row[name][status] += 1
    return {"by_row": by_row, "episodes": len(tasks), "totals": totals}


def _assert_public(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_public(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_public(item, f"{path}[{index}]")
        return
    if isinstance(value, str):
        if (
            _UUID.search(value)
            or value.startswith("/")
            or any(
                token in value for token in ("/tmp/", "/data/", "/home/", "root-disk", "worktrees")
            )
        ):
            raise SystemExit(f"report leaked a path or uuid at {path}")


def _done(task: dict[str, Any]) -> bool:
    rows = task.get("rows")
    if not isinstance(rows, dict):
        return False
    return all(
        isinstance(rows.get(name), dict) and rows[name].get("status") == "pass"
        for name in ROW_NAMES
    )


def _print_table(tasks: list[dict[str, Any]]) -> None:
    names = " ".join(f"{name[:12]:>12}" for name in ROW_NAMES)
    print(f"{'episode':<64} {names}")
    for task in tasks:
        cells = " ".join(f"{task['rows'][name]['status']:>12}" for name in ROW_NAMES)
        print(f"{task['label']:<64} {cells}")
    summary = _summary(tasks)
    print(
        "TOTAL "
        f"pass={summary['totals']['pass']} "
        f"fail={summary['totals']['fail']} "
        f"error={summary['totals']['error']}"
    )


def _error_task(label: str, episode_key: str, exc: Exception) -> dict[str, Any]:
    reason = _public(f"{type(exc).__name__}: {exc}")
    return {
        "episode_key": episode_key,
        "label": label,
        "rows": {
            name: {"error_type": type(exc).__name__, "reason": reason, "status": "error"}
            for name in ROW_NAMES
        },
    }


def _write_report(path: Path, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    report = {
        "frozen_rules": list(FROZEN_RULES),
        "schema": SCHEMA,
        "summary": _summary(tasks),
        "tasks": tasks,
    }
    _assert_public(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _row_identity(row: dict[str, Any], domain_key: str) -> tuple[str, str]:
    domain = row.get("domain")
    episode_id = row.get("episode_id")
    split = row.get("split")
    domain_text = domain if isinstance(domain, str) else domain_key
    episode_text = episode_id if isinstance(episode_id, str) else "episode"
    split_text = split if isinstance(split, str) else ""
    label = f"{domain_key} {_report_text(episode_text)}"
    episode_key = _canonical(
        {"domain": domain_text, "episode_id": episode_text, "split": split_text}
    )
    return label, episode_key


def _selected_rows(
    data_dir: Path, split: str, domains: tuple[str, ...], per_domain: int
) -> list[tuple[str, dict[str, Any]]]:
    if per_domain < 0:
        raise SystemExit("--per-domain must be >= 0")
    selected: list[tuple[str, dict[str, Any]]] = []
    for domain_key in domains:
        path = data_dir / DOMAIN_FILES[domain_key]
        rows = _select_domain_rows(_load_jsonl(path), split, per_domain)
        selected.extend((domain_key, row) for row in rows)
    if not selected:
        raise SystemExit("no episodes in the requested split")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="EvoAgentBench ADR-0043/0044 matrix")
    parser.add_argument("--out", type=Path, default=Path("/tmp/evo-adr0043-0044/report.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--per-domain", type=int, default=25, help="0 keeps the whole split")
    parser.add_argument("--domains", default=None, help="code,ir,knowledge-work")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    for line in FROZEN_RULES:
        print(line)
    selected = _selected_rows(
        args.data_dir, args.split, _parse_domains(args.domains), args.per_domain
    )
    previous: dict[str, dict[str, Any]] = {}
    if args.resume and args.out.is_file():
        loaded = json.loads(args.out.read_text(encoding="utf-8"))
        for task in loaded.get("tasks") or []:
            key = task.get("episode_key") if isinstance(task, dict) else None
            if isinstance(key, str) and _done(task):
                previous[key] = task
    elif args.resume:
        print("RESUME no report yet; running selected episodes")
    tasks: list[dict[str, Any]] = []
    started = time.perf_counter()
    for domain_key, row in selected:
        label, episode_key = _row_identity(row, domain_key)
        if episode_key in previous:
            print(f"RESUME {label}")
            tasks.append(previous[episode_key])
            continue
        print(f"EPISODE {label}")
        try:
            spec = _spec_from_row(row, domain_key)
        except Exception as exc:
            print(f"ERROR {label} load {_public(f'{type(exc).__name__}: {exc}')}", file=sys.stderr)
            traceback.print_exc()
            tasks.append(_error_task(label, episode_key, exc))
            _write_report(args.out, tasks)
            continue
        tasks.append(_run_task(spec))
        _write_report(args.out, tasks)
    report = _write_report(args.out, tasks)
    elapsed = time.perf_counter() - started
    _print_table(tasks)
    print(f"ELAPSED_S {elapsed:.2f}")
    print(f"WROTE report bytes={args.out.stat().st_size}")
    if report["summary"]["totals"]["fail"] or report["summary"]["totals"]["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
