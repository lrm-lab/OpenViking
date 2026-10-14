# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Minimal paper-graph edits bound to one violation id (ADR-0044).

The refiner holds every build verb except ``submit``, plus ``get_graph``,
``read_step``, and ``read_step_range``. Each edit names the violation it
addresses. The server rejects an edit whose touched objects fall outside
that violation. A semantic or granularity violation may only ``modify_block``
the named node. A graph-scoped structural violation may ``add_block``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from openviking.session.aig_native_protocol import TOOLLOOP_BUILD_VERBS
from openviking.session.aig_tool_surface import TOOL_CATALOG, AigToolSurface, ToolSurfaceError
from openviking.session.outcome_signal import closed_attribution_mode
from openviking.session.paper_graph_phi import PaperViolation

REFINER_SYSTEM_PROMPT = """输入：当前图 + 违规清单。你持构建工具（除 submit 外全部动词）与只读工具。
规则：
- 每次编辑必须声明它解决的 violation_id；只动该违规涉及的对象（服务端校验）。
- 最小化编辑：能改字段就不删块，能删注记就不删块。
- 结构违规（缺根/缺汇/不连通）允许 add_block 补齐——这是"解决违规"，不是重构。
- 确实无法解决的违规，输出 {violation_id, action: "unresolvable", reason}，不要盲改。
停止条件：所有违规解决或声明 unresolvable 后交回；轮数由服务端计（≤3）。"""

REFINER_SUCCESS_PROMPT = """输入：当前图 + 违规清单。你持构建工具（除 submit 外全部动词）与只读工具。本轮在成功轨迹上修订，判据仍由违规清单驱动。
规则：
- 每次编辑必须声明它解决的 violation_id；只动该违规涉及的对象（服务端校验）。
- 最小化编辑：能改字段就不删块，能删注记就不删块。
- 结构违规（缺根/缺汇/不连通）允许 add_block 补齐——这是"解决违规"，不是重构。
- 确实无法解决的违规，输出 {violation_id, action: "unresolvable", reason}，不要盲改。
停止条件：所有违规解决或声明 unresolvable 后交回；轮数由服务端计（≤3）。"""


def refiner_prompt_for_mode(mode: str | None) -> str:
    """Fail-closed: None or an unknown mode keeps the fail narrative."""
    if closed_attribution_mode(mode) == "success":
        return REFINER_SUCCESS_PROMPT
    return REFINER_SYSTEM_PROMPT


REFINER_READ_VERBS = frozenset({"get_graph", "read_step", "read_step_range"})
REFINER_EDIT_VERBS = frozenset(TOOLLOOP_BUILD_VERBS - {"submit"})
_ENDPOINT_VERBS = frozenset(
    {
        "add_inheritance_note",
        "delete_inheritance_note",
        "add_skip_attribution",
        "delete_skip_attribution",
    }
)
_BLOCK_VERBS = frozenset({"modify_block", "delete_block", "mark_defect_recorded"})
DEFAULT_REFINER_CALLS = 8


class RefinerError(ValueError):
    """A refiner action was rejected. ``code`` is stable."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class RefinerAction:
    """One provider decision: an edit, an unresolvable mark, or done."""

    kind: str
    verb: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    violation_id: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"edit", "unresolvable", "done"}:
            raise RefinerError("invalid_provider_result")
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True)
class RefinerTurn:
    """What the refiner sees on one call inside a revision round."""

    system_prompt: str
    tools: tuple[tuple[str, str], ...]
    draft: dict[str, Any]
    violations: tuple[dict[str, Any], ...]
    history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RefinerRound:
    """Edits and unresolvable marks from one revision round."""

    edits: tuple[dict[str, Any], ...]
    unresolvable: tuple[dict[str, Any], ...]
    draft: dict[str, Any]
    stop_reason: str | None = None


class RefinerProvider(Protocol):
    """Return the next refiner action. A fake provider must not call a model."""

    def __call__(self, turn: RefinerTurn) -> RefinerAction:
        """Choose an edit, an unresolvable mark, or done."""


class FakeRefinerProvider:
    """Scripted refiner. An exhausted script returns done."""

    def __init__(self, script: Sequence[RefinerAction] = ()) -> None:
        self._script = list(script)
        self._cursor = 0
        self.turns: list[RefinerTurn] = []

    def __call__(self, turn: RefinerTurn) -> RefinerAction:
        self.turns.append(turn)
        if self._cursor >= len(self._script):
            return RefinerAction(kind="done")
        action = self._script[self._cursor]
        self._cursor += 1
        if not isinstance(action, RefinerAction):
            raise RefinerError("invalid_provider_result")
        return action


def refiner_catalog() -> tuple[tuple[str, str], ...]:
    """Build verbs except ``submit``, plus the read-only subset."""
    allowed = REFINER_READ_VERBS | REFINER_EDIT_VERBS
    return tuple(item for item in TOOL_CATALOG if item[0] in allowed)


class RefinerSession:
    """Enforce violation scope, then reuse the surface verb implementations."""

    def __init__(self, surface: AigToolSurface, violations: Sequence[PaperViolation]) -> None:
        if not isinstance(surface, AigToolSurface):
            raise RefinerError("invalid_surface")
        self._surface = surface
        self._by_id = {item.violation_id: item for item in violations}
        self._unresolvable: list[dict[str, str]] = []

    def catalog(self) -> tuple[tuple[str, str], ...]:
        return refiner_catalog()

    def call(
        self,
        verb: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        violation_id: str | None = None,
    ) -> Any:
        if verb == "submit":
            raise RefinerError("submit_forbidden")
        if verb in REFINER_READ_VERBS:
            try:
                return self._surface.call(verb, arguments)
            except ToolSurfaceError as exc:
                raise RefinerError(exc.code) from exc
        if verb not in REFINER_EDIT_VERBS:
            raise RefinerError("verb_not_in_role")
        chosen, args = _take_violation_id(arguments, violation_id)
        self._check_scope(chosen, verb, args)
        if verb == "modify_block":
            block = next(
                (
                    item
                    for item in self._surface.get_graph()["blocks"]
                    if item["block_id"] == args.get("block_id")
                ),
                None,
            )
            fields = args.get("fields")
            if (
                block is not None
                and isinstance(fields, Mapping)
                and all(key in block and block[key] == value for key, value in fields.items())
            ):
                raise RefinerError("no_effect_edit")
        try:
            return self._surface.apply_refine(verb, args)
        except ToolSurfaceError as exc:
            raise RefinerError(exc.code) from exc

    def declare_unresolvable(self, violation_id: str, reason: str) -> dict[str, str]:
        if not isinstance(violation_id, str) or violation_id not in self._by_id:
            raise RefinerError("unknown_violation")
        if not isinstance(reason, str) or not reason.strip():
            raise RefinerError("invalid_unresolvable")
        record = {"action": "unresolvable", "reason": reason, "violation_id": violation_id}
        self._unresolvable.append(record)
        return dict(record)

    def _check_scope(self, violation_id: str, verb: str, arguments: Mapping[str, Any]) -> None:
        violation = self._by_id.get(violation_id)
        if violation is None:
            raise RefinerError("unknown_violation")
        graph_ids = _graph_ids(self._surface.get_graph())
        touched = _touched(verb, arguments)
        if violation.kind in {"semantic", "granularity"}:
            if verb != "modify_block" or not touched <= set(violation.involved_ids):
                raise RefinerError("edit_out_of_violation_scope")
            return
        if verb == "add_block":
            if violation.scope != "graph":
                raise RefinerError("edit_out_of_violation_scope")
            return
        if violation.scope == "graph":
            if touched and not touched <= graph_ids:
                raise RefinerError("edit_out_of_violation_scope")
            return
        if not touched or not touched <= set(violation.involved_ids):
            raise RefinerError("edit_out_of_violation_scope")


def run_refiner_round(
    session: RefinerSession,
    provider: RefinerProvider,
    *,
    max_calls: int = DEFAULT_REFINER_CALLS,
    mode: str = "fail",
) -> RefinerRound:
    """Call the provider until it is done, or the inner call budget runs out."""
    if not isinstance(session, RefinerSession) or not callable(provider):
        raise RefinerError("invalid_provider")
    if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
        raise RefinerError("invalid_budget")
    history: list[dict[str, Any]] = []
    edits: list[dict[str, Any]] = []
    declared: list[dict[str, str]] = []
    failures: dict[str, int] = {}
    seen_reads: set[str] = set()
    stop_reason = None
    for _ in range(max_calls):
        turn = RefinerTurn(
            system_prompt=refiner_prompt_for_mode(mode),
            tools=session.catalog(),
            draft=session._surface.get_graph(),
            violations=tuple(item.turn_dict() for item in session._by_id.values()),
            history=tuple(history),
        )
        action = provider(turn)
        if not isinstance(action, RefinerAction):
            raise RefinerError("invalid_provider_result")
        if action.kind == "done":
            break
        if action.kind == "unresolvable":
            try:
                record = session.declare_unresolvable(action.violation_id, action.reason)
            except RefinerError as exc:
                history.append({"action": "unresolvable", "error_code": exc.code, "ok": False})
                continue
            declared.append(record)
            history.append(
                {"action": "unresolvable", "ok": True, "violation_id": action.violation_id}
            )
            continue
        if action.kind != "edit":
            raise RefinerError("invalid_provider_result")
        if action.verb in REFINER_READ_VERBS:
            from openviking.session.aig_native_protocol import canonical_dumps

            read_key = canonical_dumps({"verb": action.verb, "arguments": action.arguments})
            if read_key in seen_reads:
                stop_reason = "no_progress"
                break
            seen_reads.add(read_key)
        try:
            result = session.call(action.verb, action.arguments, violation_id=action.violation_id)
        except RefinerError as exc:
            history.append(
                {
                    "error_code": exc.code,
                    "ok": False,
                    "verb": action.verb,
                    "arguments": dict(action.arguments),
                    "violation_id": action.violation_id,
                }
            )
            failures[action.violation_id] = failures.get(action.violation_id, 0) + 1
            if failures[action.violation_id] >= 2:
                stop_reason = "no_progress"
                break
            continue
        edit = {
            "arguments": dict(action.arguments),
            "result": result,
            "verb": action.verb,
            "violation_id": action.violation_id,
        }
        if action.verb in REFINER_EDIT_VERBS:
            edits.append(edit)
            failures.clear()
            seen_reads.clear()
        history.append({"ok": True, **edit})
    if stop_reason:
        declared_ids = {item["violation_id"] for item in declared}
        for violation_id in session._by_id:
            if violation_id not in declared_ids:
                declared.append(session.declare_unresolvable(violation_id, stop_reason))
    return RefinerRound(
        edits=tuple(edits),
        unresolvable=tuple(declared),
        draft=session._surface.get_graph(),
        stop_reason=stop_reason,
    )


def _take_violation_id(
    arguments: Mapping[str, Any] | None, explicit: str | None
) -> tuple[str, dict[str, Any]]:
    args = dict(arguments or {})
    embedded = args.pop("violation_id", None)
    if explicit and embedded and explicit != embedded:
        raise RefinerError("invalid_args")
    chosen = explicit or embedded
    if not isinstance(chosen, str) or not chosen:
        raise RefinerError("violation_id_missing")
    return chosen, args


def _graph_ids(graph: Mapping[str, Any]) -> set[str]:
    found: set[str] = set()
    for block in graph.get("blocks", ()):
        if isinstance(block, Mapping) and isinstance(block.get("block_id"), str):
            found.add(block["block_id"])
    return found


def _touched(verb: str, arguments: Mapping[str, Any]) -> frozenset[str]:
    if verb == "add_block":
        return frozenset()
    if verb in _BLOCK_VERBS:
        block_id = arguments.get("block_id")
        if not isinstance(block_id, str) or not block_id:
            raise RefinerError("invalid_args")
        return frozenset({block_id})
    if verb in _ENDPOINT_VERBS:
        source = arguments.get("from")
        target = arguments.get("to")
        if not isinstance(source, str) or not isinstance(target, str):
            raise RefinerError("invalid_args")
        return frozenset({source, target})
    raise RefinerError("unknown_verb")
