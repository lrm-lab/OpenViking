# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Drive one Builder tool loop over a fixed AO snapshot (ADR-0044).

The provider sees ``T_log`` and ``T_build`` and returns one structured call
at a time. Hard-invariant rejections come back as error codes so the
provider can retry. ``submit`` seals an ``adaptive_agentic_aig`` draft.
Call budget or wall-clock budget returns ``construction_mode=fallback`` and
does not present that partial draft as an AIG. This module does not call a
model and does not build the deterministic minimal view; the marker names
``build_minimal_fallback_view`` for the caller.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from openviking.session.aig_native_protocol import canonical_dumps
from openviking.session.aig_tool_surface import (
    TOOL_CATALOG,
    AigToolSurface,
    AoSnapshot,
    ToolSurfaceError,
)
from openviking.session.influence_view import CONSTRUCTION_MODE_FALLBACK
from openviking.session.outcome_signal import closed_attribution_mode

CONSTRUCTION_ADAPTIVE_AGENTIC = "adaptive_agentic_aig"
CONSTRUCTION_FALLBACK = CONSTRUCTION_MODE_FALLBACK
DEFAULT_MAX_CALLS = 64
DEFAULT_MAX_SECONDS = 120.0
_FALLBACK_VIEW = "build_minimal_fallback_view"
REPEAT_CALL_CODE = "repeat_call"
REPEAT_LOOP_REASON = "repeat_loop"
_REPEAT_TRIP = 3
_REPEAT_STOP = 5
_REPEAT_MESSAGE = "identical call repeated; change arguments or proceed"

BUILDER_SYSTEM_PROMPT = (
    "你从空图构建一条失败轨迹的自适应影响图。你有一组检查工具（T_log：概览/读整段/读区间/读单步/正则搜索/列智能体/看问题/看当前图）与一组构建工具（T_build）。日志步骤即 AO "
    '行（行动-观察对）。\n流程：先用 get_overview/get_question 建立全局认识 → 交替"读日志（read_full_segment/read_step_range/read_step/search）'
    '—建图（add_block/add_inheritance_note/add_skip_attribution）"，直到你认为图完整、忠实、可导航。\n规则：\n- '
    "读策略：get_overview 返回交互 DAG（节点=某个智能体在两次群聊交换之间的 AO 区间 segment，附行号范围、规模、所用工具；边=local_continuation/emit/deliver，事件=群聊消息）。这是只读导航地图，不是要照抄的分块方案：一个 segment 可以拆成多个块，块内所有行必须同一参与者、同一 segment；需要逐行表时传 include_rows=true。先用 read_full_segment 读取 segment（参数取 DAG 节点的 segment 值）；返回最多 20,000 字符，只在完整 AO 边界截断；若有截断提示，从提示的 next AO index 用 read_step_range 继续读取未返回的部分，"
    "不要逐行 read_step 穷举扫描；只关心局部时用 read_step_range 取大区间；read_step 仅用于核验个别疑点行；已读过的内容不重复读；为建图、修补与 submit 留足调用预算。\n- "
    "只把日志里真实存在的步骤合入节点（step_refs 必须落在快照内；服务端逐调用校验）。\n- 四类节点：task=用户问题（唯一根）；orchestrator=调度回合；"
    "agent=工作回合（唯一带 input/output/authorship 的类型）；conclusion=最终答案（唯一汇）。\n- 边只能从较早节点指向较晚节点。相邻继承用 "
    'add_inheritance_note；跨接归因用 add_skip_attribution。边应形成从 task 到 conclusion 的链，每个中间节点都要在链上；不要从 task 直接连到 conclusion 而绕过中间节点。inherited 写"后一节点复用了前一节点的什么"，effect '
    '写"它在下游如何出错"——没有失效相关复用就不要加边。\n- 节点不重叠（每个行号只属于一个块；行号冲突时先用 modify_block 调整已有块的 step_refs）；summary 是抽象不是抄录；authorship 如实（authored/relayed/executed'
    "）。\n- 发现某节点输出未满足下发的 instruction_or_question 却被提交下游，用 mark_defect_recorded 标记。\n停止条件：submit 前自检三件事——有且仅有一个 task 根；有且仅有一个 conclusion 汇；每个非 task 节点都有指向它的入边（沿边可从 task 根到达它），每个非 conclusion 节点都有出边（沿边可达 conclusion）。三者齐备才调用 "
    "submit（一次性，不可撤销）。预算耗尽由服务端截断。"
)

BUILDER_SUCCESS_PROMPT = (
    "你从空图构建一条成功轨迹的自适应影响图。你有一组检查工具（T_log：概览/读整段/读区间/读单步/正则搜索/列智能体/看问题/看当前图）与一组构建工具（T_build）。日志步骤即 AO "
    '行（行动-观察对）。\n流程：先用 get_overview/get_question 建立全局认识 → 交替"读日志（read_full_segment/read_step_range/read_step/search）'
    '—建图（add_block/add_inheritance_note/add_skip_attribution）"，直到你认为图完整、忠实、可导航。\n规则：\n- '
    "读策略：get_overview 返回交互 DAG（节点=某个智能体在两次群聊交换之间的 AO 区间 segment，附行号范围、规模、所用工具；边=local_continuation/emit/deliver，事件=群聊消息）。这是只读导航地图，不是要照抄的分块方案：一个 segment 可以拆成多个块，块内所有行必须同一参与者、同一 segment；需要逐行表时传 include_rows=true。先用 read_full_segment 读取 segment（参数取 DAG 节点的 segment 值）；返回最多 20,000 字符，只在完整 AO 边界截断；若有截断提示，从提示的 next AO index 用 read_step_range 继续读取未返回的部分，"
    "不要逐行 read_step 穷举扫描；只关心局部时用 read_step_range 取大区间；read_step 仅用于核验个别疑点行；已读过的内容不重复读；为建图、修补与 submit 留足调用预算。\n- "
    "只把日志里真实存在的步骤合入节点（step_refs 必须落在快照内；服务端逐调用校验）。\n- 四类节点：task=用户问题（唯一根）；orchestrator=调度回合；"
    "agent=工作回合（唯一带 input/output/authorship 的类型）；conclusion=最终答案（唯一汇）。\n- 边只能从较早节点指向较晚节点。相邻继承用 "
    'add_inheritance_note；跨接归因用 add_skip_attribution。边应形成从 task 到 conclusion 的链，每个中间节点都要在链上；不要从 task 直接连到 conclusion 而绕过中间节点。inherited 写"后一节点复用了前一节点的什么"，effect '
    '写"这次复用如何促成最终正确结果"——没有促成正确结果的复用就不要加边。\n- 节点不重叠（每个行号只属于一个块；行号冲突时先用 modify_block 调整已有块的 step_refs）；summary 是抽象不是抄录；authorship 如实（authored/relayed/executed'
    "）。\n- 发现某节点输出未满足下发的 instruction_or_question 却被提交下游，用 mark_defect_recorded 标记。成功轨迹仍可有未达标记录。"
    "\n停止条件：submit 前自检三件事——有且仅有一个 task 根；有且仅有一个 conclusion 汇；每个非 task 节点都有指向它的入边（沿边可从 task 根到达它），每个非 conclusion 节点都有出边（沿边可达 conclusion）。三者齐备才调用 submit（一次性，不可撤销）。预算耗尽由服务端截断。"
)

BUILDER_NEUTRAL_PROMPT = BUILDER_SUCCESS_PROMPT.replace(
    "一条成功轨迹", "一条结果尚未判定的轨迹"
).replace(
    "这次复用如何促成最终正确结果", "这次复用如何影响下游状态或结果"
).replace(
    "没有促成正确结果的复用就不要加边", "没有日志证实的复用就不要加边"
).replace(
    "成功轨迹仍可有未达标记录", "任务结果未知也应如实标记未达标记录"
)


def _base_builder_prompt_for_mode(mode: str | None) -> str:
    """Fail-closed: None or an unknown mode keeps the fail narrative."""
    if closed_attribution_mode(mode) == "success":
        return BUILDER_SUCCESS_PROMPT
    if mode == "neutral":
        return BUILDER_NEUTRAL_PROMPT
    return BUILDER_SYSTEM_PROMPT


BUILDER_JUDGE_REPORT_GUARD = '\n评审报告（get_overview 的 outcome.feedback，若存在）：只用来决定先读哪里、在哪里拆分；块字段只能写日志内容，不得抄录或转述报告；报告提到而日志里找不到的内容不要写入块。get_overview 的交互 DAG 是只读导航，不是要照抄的分块方案。'


def builder_prompt_for_mode(mode: str | None) -> str:
    """Role prompt for ``mode`` plus the judge-report guardrails (cycle 007 D-007)."""
    return _base_builder_prompt_for_mode(mode) + BUILDER_JUDGE_REPORT_GUARD


class ToolloopError(ValueError):
    """The tool loop could not start. ``code`` is stable."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ToolInvocation:
    """One provider decision: a verb plus JSON arguments."""

    verb: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.verb, str) or not self.verb:
            raise ToolloopError("invalid_provider_result")
        if not isinstance(self.arguments, Mapping):
            raise ToolloopError("invalid_provider_result")
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True)
class ToolloopTurn:
    """What the provider sees on one round: prompt, tool list, and history."""

    system_prompt: str
    tools: tuple[tuple[str, str], ...]
    history: tuple[dict[str, Any], ...]
    draft: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolAuditRecord:
    """One executed tool call. Arguments are a short canonical summary."""

    verb: str
    arguments_summary: str
    ok: bool
    error_code: str | None
    result_summary: str
    elapsed_ms: int
    repeat_count: int = 0


@dataclass(frozen=True)
class ToolloopResult:
    """Submitted paper draft, or a fallback marker when the budget runs out."""

    status: str
    construction_mode: str
    draft: dict[str, Any] | None
    audit: tuple[ToolAuditRecord, ...]
    fallback_marker: dict[str, Any] | None


class ToolloopProvider(Protocol):
    """Return the next tool call. A fake provider must not call a model."""

    def __call__(self, turn: ToolloopTurn) -> ToolInvocation:
        """Choose the next verb from the turn history."""


class FakeToolloopProvider:
    """Scripted provider for tests. An exhausted script repeats ``get_overview``."""

    def __init__(self, script: Sequence[tuple[str, Mapping[str, Any]]] = ()) -> None:
        self._script: list[tuple[str, dict[str, Any]]] = []
        for verb, arguments in script:
            if not isinstance(verb, str) or not isinstance(arguments, Mapping):
                raise ToolloopError("invalid_provider_result")
            self._script.append((verb, dict(arguments)))
        self._cursor = 0
        self.turns: list[ToolloopTurn] = []

    def __call__(self, turn: ToolloopTurn) -> ToolInvocation:
        self.turns.append(turn)
        if self._cursor >= len(self._script):
            return ToolInvocation("get_overview", {})
        verb, arguments = self._script[self._cursor]
        self._cursor += 1
        return ToolInvocation(verb, arguments)


def run_toolloop(
    snapshot: AoSnapshot,
    question: str,
    provider: ToolloopProvider,
    *,
    root: Path,
    run_id: str,
    max_calls: int = DEFAULT_MAX_CALLS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    clock: Callable[[], float] | None = None,
    mode: str = "fail",
) -> ToolloopResult:
    """Run inspect/build turns until ``submit``, budget, or a repeat loop."""
    if not isinstance(question, str) or not question.strip():
        raise ToolloopError("question_missing")
    if not callable(provider):
        raise ToolloopError("invalid_provider")
    if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
        raise ToolloopError("invalid_budget")
    if (
        isinstance(max_seconds, bool)
        or not isinstance(max_seconds, (int, float))
        or not max_seconds > 0
    ):
        raise ToolloopError("invalid_budget")
    now = clock or time.monotonic
    sample = now()
    if isinstance(sample, bool) or not isinstance(sample, (int, float)):
        raise ToolloopError("invalid_budget")
    bound = AoSnapshot(
        records=snapshot.records,
        watermark=snapshot.watermark,
        question=question,
        agents=snapshot.agents,
        outcome=snapshot.outcome,
    )
    surface = AigToolSurface(bound, root=root, run_id=run_id)
    history: list[dict[str, Any]] = []
    audit: list[ToolAuditRecord] = []
    started = now()
    system_prompt = builder_prompt_for_mode(mode)
    last_call_key: tuple[str, str] | None = None
    repeat_count = 0
    edge_rejections: dict[str, int] = {}
    stagnant_calls = 0
    seen_information: set[str] = set()
    while True:
        if stagnant_calls >= 6:
            return _fallback(run_id, bound.watermark, audit, reason="no_progress")
        if len(audit) >= max_calls or now() - started >= max_seconds:
            return _fallback(run_id, bound.watermark, audit)
        turn = ToolloopTurn(
            system_prompt=system_prompt,
            tools=TOOL_CATALOG,
            history=tuple(_json_copy(item) for item in history),
            draft=surface.get_graph(),
        )
        invocation = provider(turn)
        stagnant_calls += 1
        if not isinstance(invocation, ToolInvocation):
            raise ToolloopError("invalid_provider_result")
        try:
            arguments = _json_copy(invocation.arguments)
        except (TypeError, ValueError):
            arguments = {}
            _record(
                audit,
                history,
                invocation.verb,
                arguments,
                ok=False,
                error_code="invalid_args",
                result=None,
                elapsed_ms=0,
            )
            continue
        if not isinstance(arguments, dict):
            arguments = {}
        call_key = _repeat_key(invocation.verb, arguments)
        if call_key == last_call_key:
            repeat_count += 1
        else:
            last_call_key = call_key
            repeat_count = 1
        if repeat_count >= _REPEAT_TRIP:
            _record(
                audit,
                history,
                invocation.verb,
                arguments,
                ok=False,
                error_code=REPEAT_CALL_CODE,
                result={
                    "code": REPEAT_CALL_CODE,
                    "message": (
                        f"identical call repeated ({repeat_count}/{_REPEAT_STOP}); "
                        "take a different action — the run terminates at "
                        f"{_REPEAT_STOP} identical calls"
                    ),
                },
                elapsed_ms=0,
                repeat_count=repeat_count,
            )
            if repeat_count >= _REPEAT_STOP:
                return _fallback(run_id, bound.watermark, audit, reason=REPEAT_LOOP_REASON)
            continue
        call_started = now()
        before_graph = surface.get_graph()
        try:
            result = surface.call(invocation.verb, arguments)
        except ToolSurfaceError as exc:
            elapsed = _elapsed_ms(call_started, now())
            detail = getattr(exc, "detail", None)
            if exc.code == "edge_not_forward" and isinstance(detail, dict):
                pair_key = json.dumps([detail["source"], detail["target"]], sort_keys=True)
                edge_rejections[pair_key] = edge_rejections.get(pair_key, 0) + 1
                detail = {**detail, "same_pair_rejections": edge_rejections[pair_key]}
                if edge_rejections[pair_key] > 1:
                    detail["progress_hint"] = (
                        "Changing inherited/effect text has not repaired this pair's "
                        "time order. Recheck the AO anchors before trying this edge again."
                    )
            _record(
                audit,
                history,
                invocation.verb,
                arguments,
                ok=False,
                error_code=exc.code,
                result=detail,
                elapsed_ms=elapsed,
                repeat_count=repeat_count,
            )
            continue
        if surface.get_graph() != before_graph:
            stagnant_calls = 0
        elif invocation.verb in {
            "get_overview",
            "get_question",
            "read_step",
            "read_step_range",
            "read_full_segment",
            "search",
            "list_agents",
        }:
            information = json.dumps({"verb": invocation.verb, "result": result}, sort_keys=True)
            if information not in seen_information:
                seen_information.add(information)
                stagnant_calls = 0
        elapsed = _elapsed_ms(call_started, now())
        _record(
            audit,
            history,
            invocation.verb,
            arguments,
            ok=True,
            error_code=None,
            result=result,
            elapsed_ms=elapsed,
            repeat_count=repeat_count,
        )
        if invocation.verb == "submit":
            return ToolloopResult(
                status="submitted",
                construction_mode=CONSTRUCTION_ADAPTIVE_AGENTIC,
                draft=surface.get_graph(),
                audit=tuple(audit),
                fallback_marker=None,
            )


def _fallback(
    run_id: str,
    watermark: str,
    audit: Sequence[ToolAuditRecord],
    *,
    reason: str = "budget_exhausted",
) -> ToolloopResult:
    marker: dict[str, Any] = {
        "accepted_as_aig": False,
        "calls_used": len(audit),
        "construction_mode": CONSTRUCTION_FALLBACK,
        "deterministic_view": _FALLBACK_VIEW,
        "reason": reason,
        "run_id": run_id,
        "watermark": watermark,
    }
    if reason == REPEAT_LOOP_REASON:
        marker["repeat_count"] = audit[-1].repeat_count if audit else 0
    return ToolloopResult(
        status=reason,
        construction_mode=CONSTRUCTION_FALLBACK,
        draft=None,
        audit=tuple(audit),
        fallback_marker=marker,
    )


def _record(
    audit: list[ToolAuditRecord],
    history: list[dict[str, Any]],
    verb: str,
    arguments: Mapping[str, Any],
    *,
    ok: bool,
    error_code: str | None,
    result: Any,
    elapsed_ms: int,
    repeat_count: int = 0,
) -> None:
    summary = error_code if error_code is not None else _result_summary(verb, result)
    audit.append(
        ToolAuditRecord(
            verb=verb,
            arguments_summary=_arguments_summary(arguments),
            ok=ok,
            error_code=error_code,
            result_summary=summary or "ok",
            elapsed_ms=elapsed_ms,
            repeat_count=repeat_count,
        )
    )
    history.append(
        {
            "verb": verb,
            "arguments": dict(arguments),
            "ok": ok,
            "error_code": error_code,
            "result": result,
        }
    )


def _repeat_key(verb: str, arguments: Mapping[str, Any]) -> tuple[str, str]:
    try:
        return (verb, canonical_dumps(dict(arguments)))
    except (TypeError, ValueError):
        return (verb, "{}")


def _arguments_summary(arguments: Mapping[str, Any]) -> str:
    try:
        text = canonical_dumps(dict(arguments))
    except (TypeError, ValueError):
        return "{}"
    if len(text) > 180:
        return text[:177] + "..."
    return text


def _result_summary(verb: str, result: Any) -> str:
    if verb == "search" and isinstance(result, Mapping):
        matches = result.get("matches")
        count = len(matches) if isinstance(matches, list) else 0
        return f"matches={count}"
    if verb == "get_overview" and isinstance(result, Mapping):
        return f"rows={result.get('row_count')}"
    if verb == "add_block" and isinstance(result, Mapping):
        return f"block_id={result.get('block_id')}"
    if verb == "submit":
        return "submitted"
    if isinstance(result, str):
        return f"chars={len(result)}"
    if isinstance(result, list):
        return f"items={len(result)}"
    if isinstance(result, Mapping):
        return f"keys={len(result)}"
    return "ok"


def _elapsed_ms(started: float, finished: float) -> int:
    elapsed = int((finished - started) * 1000)
    if elapsed < 0:
        return 0
    return elapsed


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_dumps(value))
