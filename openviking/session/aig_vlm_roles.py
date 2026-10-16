# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Account VLM adapters for the four commit-time AIG roles.

The model proposes one JSON action per call. Server role surfaces validate
and execute every action; malformed output aborts construction.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from typing import Any

from openviking.session.aig_production_pipeline import AigRoleProviders
from openviking.session.aig_toolloop_builder import ToolInvocation, ToolloopTurn
from openviking.session.aig_toolloop_critic import CriticTurn, CriticView, parse_critic_violations
from openviking.session.aig_toolloop_reader import ReaderInvocation, ReaderTurn
from openviking.session.aig_toolloop_refiner import RefinerAction, RefinerTurn


def _json_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError("model_response_missing")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate_model_key")
            result[key] = item
        return result

    loaded = json.loads(
        value,
        object_pairs_hook=unique_pairs,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite_model_number")),
    )
    if not isinstance(loaded, dict):
        raise ValueError("model_response_not_object")
    return loaded


class _RoleModel:
    def __init__(self, handle: Any, deadline: float) -> None:
        self._handle = handle
        self._deadline = deadline

    def _ask(self, prompt: str, state: Mapping[str, Any]) -> dict[str, Any]:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("aig_role_deadline_exceeded")

        async def complete():
            return await asyncio.wait_for(
                self._handle.get_completion_async(
                    messages=[
                        {
                            "role": "system",
                            "content": prompt
                            + "\nReturn exactly one JSON object, without markdown.",
                        },
                        {
                            "role": "user",
                            "content": json.dumps(state, ensure_ascii=False, default=str),
                        },
                    ],
                    thinking=False,
                ),
                timeout=min(45.0, remaining),
            )

        response = asyncio.run(complete())
        content = response if isinstance(response, str) else getattr(response, "content", None)
        return _json_object(content)


class VLMBuilder(_RoleModel):
    def __call__(self, turn: ToolloopTurn) -> ToolInvocation:
        result = self._ask(
            turn.system_prompt + '\nOutput {"verb":"...","arguments":{...}}.',
            {
                "tools": turn.tools,
                "history": turn.history,
            },
        )
        if set(result) != {"verb", "arguments"} or not isinstance(result["arguments"], dict):
            raise ValueError("invalid_builder_action")
        if result["verb"] not in {name for name, _ in turn.tools}:
            raise ValueError("builder_verb_forbidden")
        return ToolInvocation(result["verb"], result["arguments"])


class VLMCritic(_RoleModel):
    def __init__(self, handle: Any, deadline: float) -> None:
        super().__init__(handle, deadline)
        self._view: CriticView | None = None

    def bind_view(self, view: CriticView) -> None:
        self._view = view

    def __call__(self, turn: CriticTurn) -> dict[str, Any]:
        if self._view is None:
            raise ValueError("critic_view_missing")
        history: list[dict[str, Any]] = []
        for _ in range(9):
            result = self._ask(
                turn.system_prompt + '\nRead with {"verb":"read_step","arguments":{"index":0}} '
                'or finish with {"violations":[...]}. Never infer a clean audit from a tool error.',
                {
                    "tools": turn.tools,
                    "draft": turn.draft,
                    "structural_violations": turn.structural_violations,
                    "history": history,
                },
            )
            if set(result) == {"violations"}:
                overview = self._view.call("get_overview")
                block_ids = {block["block_id"] for block in turn.draft.get("blocks", [])}
                parse_critic_violations(
                    result, block_ids=block_ids, row_count=overview["row_count"]
                )
                return result
            if set(result) != {"verb", "arguments"} or not isinstance(result["arguments"], dict):
                raise ValueError("invalid_critic_action")
            if result["verb"] not in {name for name, _ in turn.tools}:
                raise ValueError("critic_verb_forbidden")
            value = self._view.call(result["verb"], result["arguments"])
            history.append(
                {"verb": result["verb"], "arguments": result["arguments"], "result": value}
            )
        raise ValueError("critic_read_budget_exhausted")


class VLMRefiner(_RoleModel):
    def __call__(self, turn: RefinerTurn) -> RefinerAction:
        result = self._ask(
            turn.system_prompt + '\nOutput one of {"action":"done"}, '
            '{"action":"unresolvable","violation_id":"...","reason":"..."}, '
            'or {"action":"edit","verb":"...","arguments":{...},"violation_id":"..."}.',
            {
                "tools": turn.tools,
                "draft": turn.draft,
                "violations": turn.violations,
                "history": turn.history,
            },
        )
        action = result.get("action")
        if action == "done" and set(result) == {"action"}:
            return RefinerAction("done")
        if action == "unresolvable" and set(result) == {"action", "violation_id", "reason"}:
            return RefinerAction(
                "unresolvable", violation_id=result["violation_id"], reason=result["reason"]
            )
        if action == "edit" and set(result) == {"action", "verb", "arguments", "violation_id"}:
            if result["verb"] not in {name for name, _ in turn.tools}:
                raise ValueError("refiner_verb_forbidden")
            return RefinerAction(
                "edit", result["verb"], result["arguments"], result["violation_id"]
            )
        raise ValueError("invalid_refiner_action")


class VLMReader(_RoleModel):
    def __call__(self, turn: ReaderTurn) -> ReaderInvocation:
        result = self._ask(
            turn.system_prompt + '\nRead with {"verb":"...","arguments":{...}} '
            "or return the exact five-field report JSON.",
            {"tools": turn.tools, "frozen_graph": turn.graph, "history": turn.history},
        )
        if set(result) == {"verb", "arguments"} and isinstance(result["arguments"], dict):
            if result["verb"] not in {name for name, _ in turn.tools}:
                raise ValueError("reader_verb_forbidden")
            return ReaderInvocation("tool", result["verb"], result["arguments"])
        return ReaderInvocation("report", report=result)


def providers_for_vlm(handle: Any) -> AigRoleProviders:
    """Create separate role states on the request's account-scoped model handle."""
    deadline = time.monotonic() + 150.0
    return AigRoleProviders(
        VLMBuilder(handle, deadline),
        VLMCritic(handle, deadline),
        VLMRefiner(handle, deadline),
        VLMReader(handle, deadline),
    )
