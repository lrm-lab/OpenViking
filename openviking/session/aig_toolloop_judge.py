# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Judge: score one candidate action from the four roles' frozen products.

The Judge reads the frozen paper graph built by Builder/Refiner and checked by
Critic/Reader, plus a subset of the read-only ``T_log`` verbs, and returns one
strict JSON verdict for the action under review. It has no edit tools. It reuses
``ReaderView`` for the graph and snapshot reads and adds no new storage.

Verdict contract (matches the AReaL external-candidate contract):

* ``status: "scored"`` carries a finite ``score`` in ``[-1, 1]`` and a boolean
  ``used``. ``used`` false requires ``score == 0``. Out-of-range scores are
  rejected, never clipped. A genuine negative score is a valid verdict.
* ``status: "unscored"`` carries a reason and no score. Insufficient evidence is
  unscored. It is never mapped to a numeric zero.
* ``evidence_block_ids`` must name blocks of the frozen graph and
  ``decisive_ao_refs`` must be raw AO rows the Judge actually read.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from openviking.session.aig_native_protocol import TOOLLOOP_BUILD_VERBS
from openviking.session.aig_tool_surface import TOOL_CATALOG, AoSnapshot
from openviking.session.aig_toolloop_reader import (
    ReaderError,
    ReaderView,
    _read_refs,
)

ROLES = ("builder", "critic", "refiner", "reader")
# A subset of the Reader's read verbs: no question lookup, no build verbs.
JUDGE_VERBS = frozenset(
    {"get_graph", "get_overview", "get_story", "read_step", "read_step_range", "search"}
)
DEFAULT_JUDGE_CALLS = 12
JUDGE_SYSTEM_PROMPT = (
    "输入：冻结的影响图（get_story/get_graph）、四角色（Builder/Critic/Refiner/Reader）"
    "的产物摘要，以及只读日志工具。\n"
    "任务：只评估这一条被审查的候选动作。先在图里找到该动作对应的块，再用 "
    "read_step/read_step_range 查原始 AO 行，确认它在分支里的实际贡献。\n"
    "评分：score 在 [-1, 1]；明确未被使用或没有影响则 used=false 且 score=0；"
    "真实的负面贡献给负分，不要丢弃；证据不足时返回 unscored，不要填 0。\n"
    '输出（严格 JSON）：{"status":"scored","score":..,"used":..,'
    '"evidence_block_ids":[..],"decisive_ao_refs":[..],"reason":".."} 或 '
    '{"status":"unscored","reason":".."}。你没有任何编辑工具。'
)
_SCORED_KEYS = frozenset(
    {"status", "score", "used", "evidence_block_ids", "decisive_ao_refs", "reason"}
)
_UNSCORED_KEYS = frozenset({"status", "reason"})


class JudgeError(ValueError):
    """The Judge refused a call or a verdict. ``code`` is stable and carries no text."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class JudgeVerdict:
    """Strict verdict. ``score`` is ``None`` exactly when ``status`` is unscored."""

    status: str
    reason: str
    score: float | None = None
    used: bool | None = None
    evidence_block_ids: tuple[str, ...] = ()
    decisive_ao_refs: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        if self.status == "unscored":
            return {"status": "unscored", "reason": self.reason}
        return {
            "status": "scored",
            "score": self.score,
            "used": self.used,
            "evidence_block_ids": list(self.evidence_block_ids),
            "decisive_ao_refs": list(self.decisive_ao_refs),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class JudgeInvocation:
    """One provider decision: a read-only tool call, or the final JSON verdict."""

    kind: str
    verb: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    verdict: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"tool", "verdict"}:
            raise JudgeError("invalid_provider_result")
        object.__setattr__(self, "arguments", dict(self.arguments))
        if self.verdict is not None:
            if not isinstance(self.verdict, Mapping):
                raise JudgeError("invalid_provider_result")
            object.__setattr__(self, "verdict", dict(self.verdict))


@dataclass(frozen=True)
class JudgeTurn:
    """Prompt, read-only tools, frozen graph, role products and tool history."""

    system_prompt: str
    tools: tuple[tuple[str, str], ...]
    graph: dict[str, Any]
    role_products: dict[str, Any]
    history: tuple[dict[str, Any], ...]
    read_ao_refs: tuple[int, ...] = ()
    remaining_calls: int = 0


@dataclass(frozen=True)
class JudgeResult:
    verdict: JudgeVerdict
    tool_calls: int
    read_ao_refs: tuple[int, ...]
    reads: tuple[dict[str, Any], ...] = ()


class JudgeProvider(Protocol):
    """Return the next tool call or the final verdict. A fake must not call a model."""

    def __call__(self, turn: JudgeTurn) -> JudgeInvocation:
        """Choose a read, or emit the strict JSON verdict."""


class FakeJudgeProvider:
    """Scripted Judge. An exhausted script repeats ``get_graph``."""

    def __init__(self, script: Sequence[JudgeInvocation] = ()) -> None:
        self._script = list(script)
        self._cursor = 0
        self.turns: list[JudgeTurn] = []

    def __call__(self, turn: JudgeTurn) -> JudgeInvocation:
        self.turns.append(turn)
        if self._cursor >= len(self._script):
            return JudgeInvocation("tool", "get_graph", {})
        item = self._script[self._cursor]
        self._cursor += 1
        if not isinstance(item, JudgeInvocation):
            raise JudgeError("invalid_provider_result")
        return item


def judge_catalog() -> tuple[tuple[str, str], ...]:
    """Read-only verbs the Judge may use. No ``T_build`` entry is included."""
    return tuple((name, text) for name, text in TOOL_CATALOG if name in JUDGE_VERBS)


def parse_judge_verdict(
    payload: Mapping[str, Any] | None,
    graph: Mapping[str, Any],
    read_refs: set[int],
) -> JudgeVerdict:
    """Validate the strict verdict. Nothing is clipped, coerced or defaulted."""
    if not isinstance(payload, Mapping):
        raise JudgeError("invalid_verdict")
    status = payload.get("status")
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise JudgeError("invalid_verdict")
    if status == "unscored":
        if set(payload) != _UNSCORED_KEYS:
            raise JudgeError("invalid_verdict")
        return JudgeVerdict("unscored", reason.strip())
    if status != "scored" or set(payload) != _SCORED_KEYS:
        raise JudgeError("invalid_verdict")
    score, used = payload.get("score"), payload.get("used")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise JudgeError("invalid_score")
    if not math.isfinite(score) or not -1 <= score <= 1:
        raise JudgeError("invalid_score")
    if type(used) is not bool or (used is False and score != 0):
        raise JudgeError("invalid_score")
    blocks = {
        block.get("block_id") for block in graph.get("blocks", []) if isinstance(block, Mapping)
    }
    ids = payload.get("evidence_block_ids")
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(item, str) or item not in blocks for item in ids)
    ):
        raise JudgeError("invalid_evidence")
    refs = payload.get("decisive_ao_refs")
    if (
        not isinstance(refs, list)
        or not refs
        or any(type(item) is not int for item in refs)
        or not set(refs) <= read_refs
    ):
        raise JudgeError("unread_decisive_refs")
    return JudgeVerdict(
        "scored",
        reason.strip(),
        score=float(score),
        used=used,
        evidence_block_ids=tuple(dict.fromkeys(ids)),
        decisive_ao_refs=tuple(dict.fromkeys(refs)),
    )


def run_judge(
    snapshot: AoSnapshot,
    graph: Mapping[str, Any],
    provider: JudgeProvider,
    *,
    root: Path,
    run_id: str,
    role_products: Mapping[str, Any],
    max_calls: int = DEFAULT_JUDGE_CALLS,
) -> JudgeResult:
    """Run the Judge tool loop over a frozen graph and return a strict verdict.

    ``role_products`` must hold a product for each of the four roles; the Judge is
    refused otherwise, so a candidate whose pipeline did not finish is never scored.
    """
    if not callable(provider):
        raise JudgeError("invalid_provider")
    if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
        raise JudgeError("invalid_budget")
    if not isinstance(role_products, Mapping) or set(role_products) != set(ROLES):
        raise JudgeError("roles_incomplete")
    try:
        view = ReaderView(snapshot, graph, root=root, run_id="judge:" + run_id)
    except ReaderError as exc:
        raise JudgeError(exc.code) from exc
    frozen_graph = {
        "blocks": list(graph["blocks"]),
        "edges": list(graph["edges"]),
        "submitted": True,
    }
    products = dict(role_products)
    history: list[dict[str, Any]] = []
    read_refs: set[int] = set()
    calls = 0
    while True:
        turn = JudgeTurn(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            tools=judge_catalog(),
            graph=frozen_graph,
            role_products=products,
            history=tuple(history),
            read_ao_refs=tuple(sorted(read_refs)),
            remaining_calls=max_calls - calls,
        )
        try:
            invocation = provider(turn)
        except JudgeError:
            raise
        except Exception as exc:
            raise JudgeError("invalid_provider_result") from exc
        if not isinstance(invocation, JudgeInvocation):
            raise JudgeError("invalid_provider_result")
        if invocation.kind == "verdict":
            verdict = parse_judge_verdict(invocation.verdict, frozen_graph, read_refs)
            return JudgeResult(verdict, calls, tuple(sorted(read_refs)), tuple(history))
        if invocation.verb in TOOLLOOP_BUILD_VERBS:
            raise JudgeError("build_verb_forbidden")
        if invocation.verb not in JUDGE_VERBS:
            raise JudgeError("verb_not_in_role")
        if calls >= max_calls:
            raise JudgeError("budget_exhausted")
        calls += 1
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
        read_refs.update(_read_refs(invocation.verb, result))
        history.append(
            {
                "arguments": dict(invocation.arguments),
                "error_code": None,
                "ok": True,
                "result": result,
                "verb": invocation.verb,
            }
        )


def to_external_result(verdict: JudgeVerdict, graph: Mapping[str, Any]) -> dict[str, Any]:
    """Map a scored verdict to the AReaL ``judge_score`` fields.

    An unscored verdict raises: it must fail the candidate, never become a number.
    The caller adds the identity fields (``action_id``, ``interaction_id`` ...).
    """
    if verdict.status != "scored" or verdict.score is None or verdict.used is None:
        raise JudgeError("unscored")
    del graph
    refs = [f"block:{item}" for item in verdict.evidence_block_ids]
    refs += [f"ao:{item}" for item in verdict.decisive_ao_refs]
    return {
        "status": "scored",
        "score": verdict.score,
        "used": verdict.used,
        "evidence_refs": refs,
        "roles_completed": list(ROLES),
    }


__all__ = [
    "DEFAULT_JUDGE_CALLS",
    "JUDGE_SYSTEM_PROMPT",
    "JUDGE_VERBS",
    "ROLES",
    "FakeJudgeProvider",
    "JudgeError",
    "JudgeInvocation",
    "JudgeProvider",
    "JudgeResult",
    "JudgeTurn",
    "JudgeVerdict",
    "judge_catalog",
    "parse_judge_verdict",
    "run_judge",
    "to_external_result",
]
