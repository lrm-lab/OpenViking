# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Reader navigation for a frozen paper graph (ADR-0044 ``read_verify``).

The graph the reader walks is ``render_paper_graph`` of the frozen native
revision. Anchor ids are those native block ids. When the revision carries
``paper_sidecar_hash``, the sidecar file is loaded only to check that hash.
Sidecar prose is not a second graph.

The read-only verbs are ``get_story``, ``get_graph``, and the ``T_log``
subset ``read_step``, ``read_step_range``, ``get_overview``, ``search``,
``get_question``. Build verbs are rejected.

``DeterministicReader`` is the zero-LLM stand-in for the transfer rule. It
starts at the conclusion (else a ``defect_recorded`` node). Incoming edges
are tried from the latest source step to the earliest. An edge transfers
only when its effect is a substring of the downstream node's output and of
a ``read_step`` payload for one of that node's ``step_refs``. Conclusion
content is stored as output by the compiler. No confirming edge stops the
walk. A provider may instead return its own strict JSON report. The server
checks that report names a real block and cites that block's step refs, then
freezes it through ``AigConstructionRun.read_verify`` and ``freeze_report``.
The reader report hash is that existing chain; this module does not mint
another one.

A call budget that ends before a report still freezes a report whose
``unresolved_areas`` is ``["budget_exhausted"]``. ``anchor_candidates`` lists
the critical block, then every other ``defect_recorded`` agent block. It does
not open a fork or call the failure gate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from openviking.session.aig_native_protocol import TOOLLOOP_BUILD_VERBS, canonical_dumps, is_safe_id
from openviking.session.aig_role_chain import AigConstructionRun, AigContractError
from openviking.session.aig_tool_surface import (
    TOOL_CATALOG,
    AigToolSurface,
    AoSnapshot,
    ToolSurfaceError,
)
from openviking.session.ao_ledger import AORecord
from openviking.session.influence_view import AigNativeViewRevision
from openviking.session.outcome_signal import closed_attribution_mode
from openviking.session.paper_graph_compile import (
    SIDECAR_KIND,
    PaperCompileError,
    render_paper_graph,
    sidecar_digest,
)

READER_SYSTEM_PROMPT = (
    "输入：冻结的影响图（get_story/get_graph）+ 只读日志工具。\n流程：从疑似出错节点（如 conclusion 或 defect_recorded 节点）出发，"
    '沿入向继承边逐条回溯。对每条边：读 effect 注记 → 用 read_step/read_step_range 查原始 AO 行 → 仅当日志确认"当前节点的输出确实呈现该边 '
    'effect 所述下游错误"时，才把责任转移到边的源节点；否则停在此节点。\n终止后：用所停节点的 step_refs 解析出责任 agent 与决定性 AO；关键块 = 责任所在节点。'
    "\n输出（严格 JSON）：{responsible_agent, decisive_ao_refs, reason, critical_block_id, "
    "unresolved_areas: [..]}。图是冻结的，你没有任何编辑工具。"
)

READER_SUCCESS_PROMPT = (
    "输入：冻结的影响图（get_story/get_graph）+ 只读日志工具。\n流程：从 conclusion 出发，沿入向继承边逐条回溯，寻找关键贡献节点。"
    '对每条边：读 effect 注记 → 用 read_step/read_step_range 查原始 AO 行 → 仅当日志确认"当前节点的输出确实呈现该边 '
    'effect 所述的下游成功贡献"时，才把责任转移到边的源节点；否则停在此节点。\n终止后：用所停节点的 step_refs 解析出归因对象与决定性 AO；关键块 = 归因所在节点。'
    "\n输出（严格 JSON）：{responsible_agent, decisive_ao_refs, reason, critical_block_id, "
    "unresolved_areas: [..]}。图是冻结的，你没有任何编辑工具。"
)

READER_OUTPUT_SCHEMA = (
    "{responsible_agent, decisive_ao_refs, reason, critical_block_id, unresolved_areas: [..]}"
)


def reader_prompt_for_mode(mode: str | None) -> str:
    """Fail-closed: None or an unknown mode keeps the fail narrative."""
    if closed_attribution_mode(mode) == "success":
        return READER_SUCCESS_PROMPT
    return READER_SYSTEM_PROMPT


READER_VERBS = frozenset(
    {
        "get_graph",
        "get_overview",
        "get_question",
        "get_story",
        "read_step",
        "read_step_range",
        "search",
    }
)
DEFAULT_READER_CALLS = 16
_REPORT_KEYS = frozenset(
    {
        "critical_block_id",
        "decisive_ao_refs",
        "reason",
        "responsible_agent",
        "unresolved_areas",
    }
)
_FROZEN_TEXT = {
    "get_graph": "Return the frozen native revision rendered as a paper graph.",
    "get_story": "Render the frozen native revision as a paper-graph story.",
}


class ReaderError(ValueError):
    """The reader refused a call or a report. ``code`` is stable."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ReaderReport:
    """Strict reader attribution. This object is what ``read_verify`` freezes."""

    responsible_agent: str
    decisive_ao_refs: tuple[int, ...]
    reason: str
    critical_block_id: str
    unresolved_areas: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "critical_block_id": self.critical_block_id,
            "decisive_ao_refs": list(self.decisive_ao_refs),
            "reason": self.reason,
            "responsible_agent": self.responsible_agent,
            "unresolved_areas": list(self.unresolved_areas),
        }


@dataclass(frozen=True)
class ReaderInvocation:
    """One provider decision: a read-only tool call, or the final JSON report."""

    kind: str
    verb: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"tool", "report"}:
            raise ReaderError("invalid_provider_result")
        object.__setattr__(self, "arguments", dict(self.arguments))
        if self.report is not None and not isinstance(self.report, Mapping):
            raise ReaderError("invalid_provider_result")
        if isinstance(self.report, Mapping):
            object.__setattr__(self, "report", dict(self.report))


@dataclass(frozen=True)
class ReaderTurn:
    """Prompt, read-only tools, frozen graph, and tool history."""

    system_prompt: str
    tools: tuple[tuple[str, str], ...]
    graph: dict[str, Any]
    history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ReaderNavigationResult:
    """Frozen reader report plus the anchor candidates derived from it."""

    report: ReaderReport
    reader_report_hash: str
    anchor_candidates: tuple[str, ...]
    tool_calls: int
    budget_exhausted: bool
    graph: dict[str, Any]
    reads: tuple[dict[str, Any], ...] = ()


class ReaderProvider(Protocol):
    """Return the next tool call or the final report. A fake must not call a model."""

    def __call__(self, turn: ReaderTurn) -> ReaderInvocation:
        """Choose a read, or emit the strict JSON report."""


class FakeReaderProvider:
    """Scripted reader. An exhausted script repeats ``get_graph``."""

    def __init__(self, script: Sequence[ReaderInvocation] = ()) -> None:
        self._script = list(script)
        self._cursor = 0
        self.turns: list[ReaderTurn] = []

    def __call__(self, turn: ReaderTurn) -> ReaderInvocation:
        self.turns.append(turn)
        if self._cursor >= len(self._script):
            return ReaderInvocation("tool", "get_graph", {})
        item = self._script[self._cursor]
        self._cursor += 1
        if not isinstance(item, ReaderInvocation):
            raise ReaderError("invalid_provider_result")
        return item


class DeterministicReader:
    """Walk incoming edges with the transfer rule, using the read-only tools."""

    def __init__(self) -> None:
        self.turns: list[ReaderTurn] = []

    def __call__(self, turn: ReaderTurn) -> ReaderInvocation:
        self.turns.append(turn)
        graph = _history_graph(turn.history)
        if graph is None:
            return ReaderInvocation("tool", "get_graph", {})
        texts = _history_steps(turn.history)
        missing, report = _navigate(graph, texts)
        if missing is not None:
            return ReaderInvocation("tool", "read_step", {"index": missing})
        if report is None:
            raise ReaderError("invalid_graph")
        return ReaderInvocation("report", report=report.to_dict())


def reader_catalog() -> tuple[tuple[str, str], ...]:
    """Read-only verbs. No ``T_build`` entry is included."""
    rows = []
    for name, text in TOOL_CATALOG:
        if name in READER_VERBS:
            rows.append((name, _FROZEN_TEXT.get(name, text)))
    return tuple(rows)


def step_log_text(index: int, record: AORecord) -> str:
    """Canonical text of one ``read_step`` payload. Effect checks use this string."""
    return canonical_dumps({"index": index, "record": record.to_dict()})


def load_frozen_paper_graph(
    revision: AigNativeViewRevision,
    snapshot: AoSnapshot,
    *,
    sidecar_path: Path | None = None,
) -> dict[str, Any]:
    """Render the native revision and, when a sidecar hash is set, verify it."""
    if not isinstance(revision, AigNativeViewRevision) or not isinstance(snapshot, AoSnapshot):
        raise ReaderError("invalid_revision")
    try:
        rendered = render_paper_graph(revision, snapshot)
    except PaperCompileError as exc:
        raise ReaderError(exc.code) from exc
    if revision.paper_sidecar_hash:
        if sidecar_path is None:
            raise ReaderError("sidecar_required")
        _verify_sidecar(Path(sidecar_path), revision.paper_sidecar_hash)
    return {"blocks": rendered["blocks"], "edges": rendered["edges"], "submitted": True}


def transfer_responsibility(graph: Mapping[str, Any], snapshot: AoSnapshot) -> ReaderReport:
    """Apply the transfer rule using every snapshot row as a ``read_step`` payload."""
    if not isinstance(snapshot, AoSnapshot):
        raise ReaderError("invalid_snapshot")
    texts = {index: step_log_text(index, record) for index, record in enumerate(snapshot.records)}
    _missing, report = _navigate(graph, texts)
    if report is None:
        raise ReaderError("invalid_graph")
    return report


def anchor_candidates(
    graph: Mapping[str, Any], report: ReaderReport | Mapping[str, Any]
) -> tuple[str, ...]:
    """Critical block, then other defect-recorded blocks, in block-id order.

    Callers pick ``FailureGateDelivery.anchor_block_id`` from this tuple.
    Nothing here writes a fork or starts remediation.
    """
    if isinstance(report, ReaderReport):
        critical = report.critical_block_id
    elif isinstance(report, Mapping) and isinstance(report.get("critical_block_id"), str):
        critical = report["critical_block_id"]
    else:
        raise ReaderError("invalid_reader_output")
    blocks = _block_map(graph)
    defects = [
        block_id
        for block_id, block in blocks.items()
        if block.get("kind") == "agent"
        and isinstance(block.get("defect_recorded"), str)
        and block.get("defect_recorded")
    ]
    ordered: list[str] = []
    if critical and critical in blocks:
        ordered.append(critical)
    for block_id in sorted(defects):
        if block_id not in ordered:
            ordered.append(block_id)
    return tuple(ordered)


def parse_reader_report(
    payload: Mapping[str, Any] | str | None, graph: Mapping[str, Any]
) -> ReaderReport:
    """Parse the strict JSON object and bind it to one frozen block."""
    loaded = _load_report(payload)
    if not isinstance(loaded, Mapping) or set(loaded) != _REPORT_KEYS:
        raise ReaderError("invalid_reader_output")
    critical = loaded.get("critical_block_id")
    agent = loaded.get("responsible_agent")
    reason = loaded.get("reason")
    refs = loaded.get("decisive_ao_refs")
    areas = loaded.get("unresolved_areas")
    blocks = _block_map(graph)
    if not isinstance(critical, str) or critical not in blocks:
        raise ReaderError("invalid_reader_output")
    if not isinstance(agent, str) or not isinstance(reason, str) or not reason.strip():
        raise ReaderError("invalid_reader_output")
    if isinstance(refs, (str, bytes)) or not isinstance(refs, list):
        raise ReaderError("invalid_reader_output")
    if isinstance(areas, (str, bytes)) or not isinstance(areas, list):
        raise ReaderError("invalid_reader_output")
    if any(not isinstance(item, str) for item in areas):
        raise ReaderError("invalid_reader_output")
    numbers: list[int] = []
    for item in refs:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ReaderError("invalid_reader_output")
        numbers.append(item)
    block = blocks[critical]
    expected = _step_refs(block)
    named = block.get("agent_name")
    if not isinstance(named, str):
        named = ""
    if agent != named or numbers != expected:
        raise ReaderError("invalid_reader_output")
    return ReaderReport(
        responsible_agent=agent,
        decisive_ao_refs=tuple(numbers),
        reason=reason.strip(),
        critical_block_id=critical,
        unresolved_areas=tuple(areas),
    )


class ReaderView:
    """Frozen graph plus the snapshot's ``T_log`` subset."""

    def __init__(
        self, snapshot: AoSnapshot, graph: Mapping[str, Any], *, root: Path, run_id: str
    ) -> None:
        if not isinstance(snapshot, AoSnapshot) or not is_safe_id(run_id):
            raise ReaderError("invalid_snapshot")
        log_id = "reader-" + hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:20]
        self._graph = {
            "blocks": list(graph["blocks"]),
            "edges": list(graph["edges"]),
            "submitted": True,
        }
        self._log = AigToolSurface(snapshot, root=root, run_id=log_id)

    def catalog(self) -> tuple[tuple[str, str], ...]:
        return reader_catalog()

    def call(self, verb: str, arguments: Mapping[str, Any] | None = None) -> Any:
        if not isinstance(verb, str):
            raise ReaderError("unknown_verb")
        if verb in TOOLLOOP_BUILD_VERBS:
            raise ReaderError("build_verb_forbidden")
        if verb not in READER_VERBS:
            raise ReaderError("verb_not_in_role")
        args = {} if arguments is None else arguments
        if not isinstance(args, Mapping):
            raise ReaderError("invalid_args")
        if verb in {"get_story", "get_graph"}:
            if dict(args):
                raise ReaderError("invalid_args")
            return self._story() if verb == "get_story" else dict(self._graph)
        try:
            return self._log.call(verb, args)
        except ToolSurfaceError as exc:
            raise ReaderError(exc.code) from exc

    def _story(self) -> str:
        lines = ["# Influence graph", ""]
        for block in self._graph["blocks"]:
            lines.append(f"## {block['kind']} `{block['block_id']}`")
            lines.append(f"- steps: {block['step_refs']}")
            for key in block:
                if key in {"block_id", "kind", "step_refs"}:
                    continue
                lines.append(f"- {key}: {block[key]}")
            lines.append("")
        lines.append("## edges")
        if not self._graph["edges"]:
            lines.append("(none)")
        for edge in self._graph["edges"]:
            lines.append(
                f"- `{edge['from']}` -> `{edge['to']}` ({edge.get('link_kind', 'inherited')}): "
                f"inherited={edge['inherited']}; effect={edge['effect']}"
            )
        lines.append("")
        lines.append("submitted: true")
        return "\n".join(lines)


def run_reader(
    revision: AigNativeViewRevision,
    snapshot: AoSnapshot,
    provider: ReaderProvider,
    *,
    root: Path,
    run_id: str,
    construction_run: AigConstructionRun,
    sidecar_path: Path | None = None,
    session_id: str | None = None,
    max_calls: int = DEFAULT_READER_CALLS,
    mode: str = "fail",
) -> ReaderNavigationResult:
    """Navigate, then freeze the report on the existing reader chain.

    ``construction_run`` must already have an open builder draft. This function
    calls ``read_verify`` and ``freeze_report`` and returns that report hash.
    """
    if not isinstance(construction_run, AigConstructionRun):
        raise ReaderError("invalid_run")
    if not callable(provider):
        raise ReaderError("invalid_provider")
    if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
        raise ReaderError("invalid_budget")
    graph = load_frozen_paper_graph(revision, snapshot, sidecar_path=sidecar_path)
    view = ReaderView(snapshot, graph, root=root, run_id=run_id)
    history: list[dict[str, Any]] = []
    tool_calls = 0
    report: ReaderReport | None = None
    exhausted = False
    while tool_calls < max_calls:
        turn = ReaderTurn(
            system_prompt=reader_prompt_for_mode(mode),
            tools=reader_catalog(),
            graph=graph,
            history=tuple(history),
        )
        try:
            invocation = provider(turn)
        except ReaderError:
            raise
        except Exception as exc:
            raise ReaderError("invalid_provider_result") from exc
        if not isinstance(invocation, ReaderInvocation):
            raise ReaderError("invalid_provider_result")
        if invocation.kind == "report":
            report = parse_reader_report(invocation.report, graph)
            break
        if invocation.verb in TOOLLOOP_BUILD_VERBS:
            raise ReaderError("build_verb_forbidden")
        if invocation.verb not in READER_VERBS:
            raise ReaderError("verb_not_in_role")
        tool_calls += 1
        try:
            result = view.call(invocation.verb, invocation.arguments)
        except ReaderError as exc:
            history.append(
                {
                    "arguments": dict(invocation.arguments),
                    "error_code": exc.code,
                    "ok": False,
                    "verb": invocation.verb,
                }
            )
            continue
        history.append(
            {
                "arguments": dict(invocation.arguments),
                "error_code": None,
                "ok": True,
                "result": result,
                "verb": invocation.verb,
            }
        )
    else:
        exhausted = True
    if report is None:
        exhausted = True
        report = ReaderReport(
            responsible_agent="",
            decisive_ao_refs=(),
            reason="budget_exhausted",
            critical_block_id="",
            unresolved_areas=("budget_exhausted",),
        )
    session = construction_run.session_for("reader") if session_id is None else session_id
    body = report.to_dict()
    try:
        digest = construction_run.read_verify(session, body)
        frozen = construction_run.freeze_report(session)
    except AigContractError as exc:
        raise ReaderError(exc.code) from exc
    if frozen != digest:
        raise ReaderError("report_hash_mismatch")
    return ReaderNavigationResult(
        report=report,
        reader_report_hash=digest,
        anchor_candidates=anchor_candidates(graph, report),
        tool_calls=tool_calls,
        budget_exhausted=exhausted,
        graph=graph,
        reads=tuple(history),
    )


def _navigate(
    graph: Mapping[str, Any], texts: Mapping[int, str]
) -> tuple[int | None, ReaderReport | None]:
    blocks = _block_map(graph)
    edges = _edge_list(graph)
    current = _start_id(blocks)
    seen: set[str] = set()
    while current not in seen:
        seen.add(current)
        block = blocks[current]
        for index in _step_refs(block):
            if index not in texts:
                return index, None
        moved = False
        for edge in _incoming(current, edges, blocks):
            if _effect_confirmed(edge.get("effect"), block, texts):
                current = str(edge["from"])
                moved = True
                break
        if not moved:
            return None, _stopped(block, current, ())
    return None, _stopped(blocks[current], current, ("cycle",))


def _effect_confirmed(effect: object, block: Mapping[str, Any], texts: Mapping[int, str]) -> bool:
    if not isinstance(effect, str) or not effect:
        return False
    output = block.get("output")
    if not isinstance(output, str) or effect not in output:
        content = block.get("content")
        if not isinstance(content, str) or effect not in content:
            return False
    for index in _step_refs(block):
        if effect in texts.get(index, ""):
            return True
    return False


def _stopped(block: Mapping[str, Any], block_id: str, unresolved: tuple[str, ...]) -> ReaderReport:
    agent = block.get("agent_name")
    if not isinstance(agent, str):
        agent = ""
    return ReaderReport(
        responsible_agent=agent,
        decisive_ao_refs=tuple(_step_refs(block)),
        reason=f"responsibility stopped at {block_id}",
        critical_block_id=block_id,
        unresolved_areas=unresolved,
    )


def _start_id(blocks: Mapping[str, Mapping[str, Any]]) -> str:
    conclusions = [item for item in blocks.values() if item.get("kind") == "conclusion"]
    defects = [
        item
        for item in blocks.values()
        if isinstance(item.get("defect_recorded"), str) and item.get("defect_recorded")
    ]
    pool = conclusions or defects or list(blocks.values())

    def key(item: Mapping[str, Any]) -> tuple[int, str]:
        refs = _step_refs(item)
        last = max(refs) if refs else 0
        block_id = item.get("block_id")
        return (last, block_id if isinstance(block_id, str) else "")

    return str(max(pool, key=key)["block_id"])


def _incoming(
    block_id: str,
    edges: Sequence[Mapping[str, Any]],
    blocks: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    rows = [edge for edge in edges if edge.get("to") == block_id and edge.get("from") in blocks]

    def key(edge: Mapping[str, Any]) -> tuple[int, str]:
        refs = _step_refs(blocks[str(edge["from"])])
        last = max(refs) if refs else 0
        return (last, str(edge["from"]))

    return sorted(rows, key=key, reverse=True)


def _block_map(graph: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = graph.get("blocks")
    if not isinstance(raw, list) or not raw:
        raise ReaderError("invalid_graph")
    found: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ReaderError("invalid_graph")
        block_id = item.get("block_id")
        if not isinstance(block_id, str) or not block_id or block_id in found:
            raise ReaderError("invalid_graph")
        found[block_id] = item
    return found


def _edge_list(graph: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = graph.get("edges")
    if not isinstance(raw, list):
        raise ReaderError("invalid_graph")
    edges: list[Mapping[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ReaderError("invalid_graph")
        if not isinstance(item.get("from"), str) or not isinstance(item.get("to"), str):
            raise ReaderError("invalid_graph")
        effect = item.get("effect")
        inherited = item.get("inherited")
        if not isinstance(effect, str) or not isinstance(inherited, str):
            raise ReaderError("invalid_graph")
        edges.append(item)
    return edges


def _step_refs(block: Mapping[str, Any]) -> list[int]:
    raw = block.get("step_refs")
    if not isinstance(raw, list):
        return []
    refs: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            return []
        refs.append(item)
    return refs


def _history_graph(history: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    found: dict[str, Any] | None = None
    for item in history:
        if (
            item.get("verb") == "get_graph"
            and item.get("ok") is True
            and isinstance(item.get("result"), Mapping)
        ):
            found = dict(item["result"])
    return found


def _history_steps(history: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    texts: dict[int, str] = {}
    for item in history:
        if item.get("verb") != "read_step" or item.get("ok") is not True:
            continue
        result = item.get("result")
        if not isinstance(result, Mapping):
            continue
        index = result.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        texts[index] = canonical_dumps(dict(result))
    return texts


def _load_report(payload: Mapping[str, Any] | str | None) -> Any:
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ReaderError("invalid_reader_output") from exc
    if isinstance(payload, Mapping):
        return payload
    raise ReaderError("invalid_reader_output")


def _verify_sidecar(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ReaderError("sidecar_mismatch")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReaderError("sidecar_mismatch") from exc
    if not isinstance(document, Mapping):
        raise ReaderError("sidecar_mismatch")
    body = document.get("body")
    if (
        document.get("object_kind") != SIDECAR_KIND
        or document.get("content_hash") != expected
        or not isinstance(body, Mapping)
        or sidecar_digest(body) != expected
    ):
        raise ReaderError("sidecar_mismatch")
