# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""One semantic audit of a submitted paper draft (ADR-0044).

The server has already run ``Phi(G)``. This call does not recheck structure.
The provider sees a read-only tool list and returns strict JSON. Build verbs
are rejected before they reach the draft.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from openviking.session.aig_native_protocol import TOOLLOOP_BUILD_VERBS
from openviking.session.aig_tool_surface import TOOL_CATALOG, AigToolSurface, ToolSurfaceError
from openviking.session.outcome_signal import closed_attribution_mode
from openviking.session.paper_graph_phi import PaperViolation

CRITIC_SYSTEM_PROMPT = (
    "输入：候选图（get_graph 可再取）+ 服务端 Φ(G) 结构违规清单（已程序化算出，你不必复查结构，只补充语义判断）。\n你的任务（对照日志，用你的只读工具核验）：\n1) "
    "每个节点的 summary/input/output 是否忠实于其 step_refs 引用的 AO 行原文；\n2) 每条〈inherited, "
    "effect〉注记是否成立（下游错误确实呈现、复用关系确实存在）；\n3) 粒度是否合理（节点边界是否割裂/混合了不同行动）。\n输出（严格 JSON）：violations: "
    "[{object_id, kind: semantic|granularity, severity: high|low, evidence_ao_refs: [..], note}]；"
    "无违规输出空数组。\n你没有编辑工具。不要臆造日志里不存在的内容。"
)

CRITIC_SUCCESS_PROMPT = (
    "输入：候选图（get_graph 可再取）+ 服务端 Φ(G) 结构违规清单（已程序化算出，你不必复查结构，只补充语义判断）。\n你的任务（对照日志，用你的只读工具核验）：\n1) "
    "每个节点的 summary/input/output 是否忠实于其 step_refs 引用的 AO 行原文；\n2) 每条〈inherited, "
    "effect〉注记是否成立（下游成功确实源于该复用、复用关系确实存在）；\n3) 粒度是否合理（节点边界是否割裂/混合了不同行动）。\n输出（严格 JSON）：violations: "
    "[{object_id, kind: semantic|granularity, severity: high|low, evidence_ao_refs: [..], note}]；"
    "无违规输出空数组。\n你没有编辑工具。不要臆造日志里不存在的内容。"
)

CRITIC_NEUTRAL_PROMPT = CRITIC_SUCCESS_PROMPT.replace(
    "下游成功确实源于该复用", "注记中的下游影响确有日志依据"
)


def critic_prompt_for_mode(mode: str | None) -> str:
    """Fail-closed: None or an unknown mode keeps the fail narrative."""
    if closed_attribution_mode(mode) == "success":
        return CRITIC_SUCCESS_PROMPT
    if mode == "neutral":
        return CRITIC_NEUTRAL_PROMPT
    return CRITIC_SYSTEM_PROMPT


CRITIC_VERBS = frozenset(
    {
        "get_graph",
        "get_story",
        "get_overview",
        "get_question",
        "read_step",
        "read_step_range",
        "search",
    }
)
_SEMANTIC_KINDS = frozenset({"semantic", "granularity"})
_SEVERITIES = frozenset({"high", "low"})
_ITEM_KEYS = frozenset({"object_id", "kind", "severity", "evidence_ao_refs", "note"})


class CriticError(ValueError):
    """The semantic audit was rejected. ``code`` is stable."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class CriticTurn:
    """One critic call: prompt, read-only tools, draft, and the Phi list."""

    system_prompt: str
    tools: tuple[tuple[str, str], ...]
    draft: dict[str, Any]
    structural_violations: tuple[dict[str, Any], ...]


class CriticProvider(Protocol):
    """Return the strict JSON audit. A fake provider must not call a model."""

    def __call__(self, turn: CriticTurn) -> Mapping[str, Any] | str:
        """Produce ``{"violations": [...]}`` or a bare list."""


class FakeCriticProvider:
    """Scripted critic. After the script ends, the last payload is repeated."""

    def __init__(self, script: Sequence[Mapping[str, Any] | str] = ()) -> None:
        self._script = list(script)
        self._cursor = 0
        self.turns: list[CriticTurn] = []

    def __call__(self, turn: CriticTurn) -> Mapping[str, Any] | str:
        self.turns.append(turn)
        if not self._script:
            return {"violations": []}
        index = min(self._cursor, len(self._script) - 1)
        self._cursor += 1
        return self._script[index]


def critic_catalog() -> tuple[tuple[str, str], ...]:
    """Read-only verbs. No ``T_build`` entry is included."""
    return tuple(item for item in TOOL_CATALOG if item[0] in CRITIC_VERBS)


class CriticView:
    """Read-only facade. A build verb is rejected before the draft is touched."""

    def __init__(self, surface: AigToolSurface) -> None:
        if not isinstance(surface, AigToolSurface):
            raise CriticError("invalid_surface")
        self._surface = surface

    def catalog(self) -> tuple[tuple[str, str], ...]:
        return critic_catalog()

    def call(self, verb: str, arguments: Mapping[str, Any] | None = None) -> Any:
        if not isinstance(verb, str):
            raise CriticError("unknown_verb")
        if verb in TOOLLOOP_BUILD_VERBS:
            raise CriticError("build_verb_forbidden")
        if verb not in CRITIC_VERBS:
            raise CriticError("verb_not_in_role")
        try:
            return self._surface.call(verb, arguments)
        except ToolSurfaceError as exc:
            raise CriticError(exc.code) from exc


def audit_semantics(
    surface: AigToolSurface,
    structural: Sequence[PaperViolation],
    provider: CriticProvider,
    *,
    mode: str = "fail",
) -> tuple[PaperViolation, ...]:
    """One provider call. The returned tuple is the semantic violation set."""
    if not isinstance(surface, AigToolSurface):
        raise CriticError("invalid_surface")
    if not callable(provider):
        raise CriticError("invalid_provider")
    draft = surface.get_graph()
    overview = surface.get_overview()
    row_count = overview.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int):
        raise CriticError("invalid_surface")
    block_ids = set()
    for block in draft.get("blocks", ()):
        if isinstance(block, Mapping) and isinstance(block.get("block_id"), str):
            block_ids.add(block["block_id"])
    turn = CriticTurn(
        system_prompt=critic_prompt_for_mode(mode),
        tools=critic_catalog(),
        draft=draft,
        structural_violations=tuple(item.to_dict() for item in structural),
    )
    try:
        raw = provider(turn)
    except CriticError:
        raise
    except Exception as exc:
        raise CriticError("invalid_critic_output") from exc
    return parse_critic_violations(raw, block_ids=block_ids, row_count=row_count)


def parse_critic_violations(
    payload: Mapping[str, Any] | str | list[Any],
    *,
    block_ids: set[str],
    row_count: int,
) -> tuple[PaperViolation, ...]:
    """Parse the strict JSON object. An empty list is a clean audit."""
    loaded = _load(payload)
    if isinstance(loaded, list):
        items = loaded
    elif isinstance(loaded, Mapping) and set(loaded) == {"violations"}:
        items = loaded.get("violations")
    else:
        raise CriticError("invalid_critic_output")
    if not isinstance(items, list):
        raise CriticError("invalid_critic_output")
    violations: list[PaperViolation] = []
    for index, item in enumerate(items):
        violations.append(_one(index, item, block_ids, row_count))
    return tuple(violations)


def _load(payload: Mapping[str, Any] | str | list[Any]) -> Any:
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise CriticError("invalid_critic_output") from exc
    if isinstance(payload, (Mapping, list)):
        return payload
    raise CriticError("invalid_critic_output")


def _one(index: int, item: Any, block_ids: set[str], row_count: int) -> PaperViolation:
    if not isinstance(item, Mapping) or set(item) != _ITEM_KEYS:
        raise CriticError("invalid_critic_output")
    object_id = item.get("object_id")
    kind = item.get("kind")
    severity = item.get("severity")
    refs = item.get("evidence_ao_refs")
    note = item.get("note")
    if not isinstance(object_id, str) or not object_id or object_id not in block_ids:
        raise CriticError("invalid_critic_output")
    if kind not in _SEMANTIC_KINDS or severity not in _SEVERITIES:
        raise CriticError("invalid_critic_output")
    if not isinstance(note, str) or not note.strip():
        raise CriticError("invalid_critic_output")
    if isinstance(refs, (str, bytes)) or not isinstance(refs, list):
        raise CriticError("invalid_critic_output")
    numbers: list[int] = []
    for ref in refs:
        if isinstance(ref, bool) or not isinstance(ref, int) or ref < 0 or ref >= row_count:
            raise CriticError("invalid_critic_output")
        numbers.append(ref)
    return PaperViolation(
        violation_id=f"critic:{index}:{kind}:{object_id}",
        object_id=object_id,
        kind=kind,
        severity=severity,
        evidence=note,
        evidence_ao_refs=tuple(numbers),
        note=note,
        scope="objects",
        involved_ids=(object_id,),
    )
