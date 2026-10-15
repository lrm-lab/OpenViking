# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Gemma role adapters. All output is checked before it reaches a role loop.

The inherited gateway transport, timeout, network retry and JSONL audit are
shared with the builder. Each completion (including output retries) has one
stat; terminal degradations are also available through ``failures``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from benchmark.causal_memory.lenovo_gemma_provider import (
    LenovoGemmaError,
    LenovoGemmaProvider,
    _first_choice,
    messages_from_turn,
    openai_tools_from_catalog,
    parse_tool_invocation,
)
from openviking.session.aig_tool_surface import TOOL_SCHEMAS
from openviking.session.aig_toolloop_builder import ToolInvocation, ToolloopTurn
from openviking.session.aig_toolloop_critic import (
    CriticError,
    CriticTurn,
    CriticView,
    critic_catalog,
    parse_critic_violations,
)
from openviking.session.aig_toolloop_reader import (
    ReaderError,
    ReaderInvocation,
    ReaderTurn,
    parse_reader_report,
    reader_catalog,
)
from openviking.session.aig_toolloop_refiner import (
    REFINER_EDIT_VERBS,
    RefinerAction,
    RefinerTurn,
    _graph_ids,
    _touched,
    refiner_catalog,
)

DEFAULT_OUTPUT_RETRIES = 2
DEFAULT_CRITIC_READS = 8


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _strict_json(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(_value: str) -> Any:
        raise ValueError("non-finite JSON number")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _message(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    choice = _first_choice(payload)
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise LenovoGemmaError("invalid_provider_result")
    calls = message.get("tool_calls")
    if calls is not None and not isinstance(calls, list):
        raise LenovoGemmaError("invalid_provider_result")
    if calls and (len(calls) != 1 or message.get("content")):
        raise LenovoGemmaError("ambiguous_provider_result")
    if choice.get("finish_reason") not in {None, "stop", "tool_calls"}:
        raise LenovoGemmaError("invalid_provider_result")
    if not calls and choice.get("finish_reason") not in {None, "stop"}:
        raise LenovoGemmaError("invalid_provider_result")
    return message


def _validate_schema(value: Any, schema: Mapping[str, Any]) -> None:
    """Validate the closed object/string/integer/array schemas in TOOL_SCHEMAS."""
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            raise LenovoGemmaError("invalid_tool_arguments")
        properties = schema.get("properties", {})
        if not set(schema.get("required", ())) <= set(value):
            raise LenovoGemmaError("invalid_tool_arguments")
        if schema.get("additionalProperties") is False and not set(value) <= set(properties):
            raise LenovoGemmaError("invalid_tool_arguments")
        for key, item in value.items():
            if key in properties:
                _validate_schema(item, properties[key])
    elif kind == "array":
        if not isinstance(value, list):
            raise LenovoGemmaError("invalid_tool_arguments")
        for item in value:
            _validate_schema(item, schema.get("items", {}))
    elif kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise LenovoGemmaError("invalid_tool_arguments")
    elif kind == "string" and not isinstance(value, str):
        raise LenovoGemmaError("invalid_tool_arguments")
    if "enum" in schema and value not in schema["enum"]:
        raise LenovoGemmaError("invalid_tool_arguments")


def _tool(payload: Mapping[str, Any], catalog: tuple[tuple[str, str], ...]) -> ToolInvocation:
    message = _message(payload)
    if not message.get("tool_calls"):
        raise LenovoGemmaError("invalid_provider_result")
    call = message["tool_calls"][0]
    if not isinstance(call, Mapping) or not isinstance(call.get("function"), Mapping):
        raise LenovoGemmaError("invalid_provider_result")
    if call.get("type") not in {None, "function"}:
        raise LenovoGemmaError("invalid_provider_result")
    # Use the builder parser, but reject duplicate keys/non-finite values first.
    raw = call["function"].get("arguments")
    if isinstance(raw, str):
        _strict_json(raw)
    invocation = parse_tool_invocation(payload)
    if invocation.verb not in {name for name, _ in catalog}:
        raise LenovoGemmaError("verb_not_in_role")
    return invocation


def role_request_body(
    turn: CriticTurn | RefinerTurn | ReaderTurn,
    *,
    role: str,
    model: str,
) -> dict[str, Any]:
    """Mode prompt, closed role catalog, current graph and completed history."""
    catalogs = {"critic": critic_catalog, "refiner": refiner_catalog, "reader": reader_catalog}
    catalog = tuple(item for item in catalogs[role]() if item[0] in dict(turn.tools))
    examples = {
        "critic": 'Final JSON only, no extra keys: {"violations": []} when clean; '
        'otherwise {"violations": [{"object_id":"b1","kind":"semantic",'
        '"severity":"high","evidence_ao_refs":[1],"note":"AO contradicts summary"}]}. '
        "object_id must name a block in draft.blocks (never an edge); "
        "kind is semantic or granularity; severity is high or low; "
        "evidence_ao_refs are integer snapshot indices; note is nonempty text. "
        "Do not copy structural violations into this output. Otherwise call one read-only tool.",
        "refiner": "Call one tool; edits require violation_id in arguments. Example: "
        '{"block_id":"b1","fields":{"summary":"corrected"},"violation_id":"v1"}. '
        'Or JSON {"action":"unresolvable","violation_id":"v1","reason":"missing evidence"}; '
        'finish with {"action":"done"}.',
        "reader": "Call one read-only tool or final JSON, for example "
        '{"responsible_agent":"worker","decisive_ao_refs":[1],"reason":"AO confirms effect",'
        '"critical_block_id":"b1","unresolved_areas":[]}. Exactly these five keys. '
        "Choose critical_block_id from frozen_graph.blocks based on verified evidence. "
        "Copy that block's entire step_refs list in the same order into decisive_ao_refs "
        "(not a subset, not indices from other blocks). Copy its agent_name exactly into "
        'responsible_agent; if agent_name is absent, use the empty string "". '
        "reason must be nonempty text; unresolved_areas must be an array of strings. "
        "The example IDs are placeholders, not evidence.",
    }
    prompt = (
        turn.system_prompt
        + "\nOnly use this role catalog. One tool or JSON per response; no markdown.\n"
        + examples[role]
    )
    tools = openai_tools_from_catalog(catalog)
    if role == "refiner":
        for tool in tools:
            if tool["function"]["name"] in REFINER_EDIT_VERBS:
                schema = tool["function"]["parameters"]
                schema["properties"]["violation_id"] = {"type": "string"}
                schema["required"].append("violation_id")
    state = {"catalog": catalog}
    if role == "reader":
        state["frozen_graph"] = turn.graph
        if turn.remaining_calls is not None:
            state["reader_progress"] = {
                "read_ao_refs": list(turn.read_ao_refs),
                "remaining_calls": turn.remaining_calls,
                "stop_reason": turn.stop_reason,
            }
            prompt += "\nDecisive AO refs must have been read through read_step/read_step_range."
            if turn.remaining_calls <= 3 or turn.final_only:
                prompt += "\nReading budget nearly exhausted. Form a report from verified evidence now; list uncertainty in unresolved_areas. Do not guess unread evidence."
            if turn.final_only:
                prompt += '\nNo more tools. Return the final report now. If evidence is insufficient, return {"responsible_agent":"","decisive_ao_refs":[],"reason":"insufficient verified evidence","critical_block_id":"","unresolved_areas":["describe the missing verification"]}.'
    else:
        state["draft"] = turn.draft
        state["structural_violations" if role == "critic" else "violations"] = (
            turn.structural_violations if role == "critic" else turn.violations
        )
    history = getattr(turn, "history", ())
    exchanges = []
    for item in history:
        item = dict(item)
        if (
            role == "refiner"
            and item.get("verb") in REFINER_EDIT_VERBS
            and item.get("violation_id")
        ):
            item["arguments"] = {**item.get("arguments", {}), "violation_id": item["violation_id"]}
        if isinstance(item.get("arguments"), Mapping) and isinstance(item.get("verb"), str):
            exchanges.append(item)
    messages = messages_from_turn(ToolloopTurn(prompt, catalog, tuple(exchanges)))
    # Only non-tool outcomes need a separate representation; tool results are
    # already present in messages and must not be serialized twice.
    outcomes = [
        item
        for item in history
        if not (isinstance(item.get("arguments"), Mapping) and isinstance(item.get("verb"), str))
    ]
    if outcomes:
        state["history"] = outcomes
    if role == "refiner":
        state["edit_constraints"] = {
            item["violation_id"]: {
                "allowed_verbs": ["modify_block"]
                if item["kind"] in {"semantic", "granularity"}
                else sorted(REFINER_EDIT_VERBS),
                "scope": item.get("scope"),
                "involved_ids": item.get("involved_ids", []),
            }
            for item in turn.violations
        }
    messages.append({"role": "user", "content": _dump(state)})
    return {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "none" if role == "reader" and turn.final_only else "auto",
    }


class _GemmaRoleProvider(LenovoGemmaProvider):
    fallback: str

    def __init__(
        self,
        *,
        output_retries: int = DEFAULT_OUTPUT_RETRIES,
        **kwargs: Any,
    ) -> None:
        if (
            isinstance(output_retries, bool)
            or not isinstance(output_retries, int)
            or output_retries < 0
        ):
            raise ValueError("output_retries must be a nonnegative integer")
        super().__init__(**kwargs)
        self._output_retries = output_retries
        self._failures: list[dict[str, str]] = []

    @property
    def failures(self) -> tuple[dict[str, str], ...]:
        return tuple(dict(item) for item in self._failures)

    def _validated(self, body: dict[str, Any], parse: Callable[..., Any]) -> Any:
        for attempt in range(self._output_retries + 1):
            started = self._clock()
            try:
                payload = self._complete(body)
                value, verb = parse(payload)
            except Exception as exc:  # Output/transport failures are local to this role call.
                code = getattr(exc, "code", "invalid_provider_result")
                self._record("", started, code)
                if (
                    code in {"network_error", "http_error", "missing_api_key"}
                    or attempt == self._output_retries
                ):
                    self._failures.append({"code": code, "fallback": self.fallback})
                    return None
                body["messages"].append(
                    {
                        "role": "user",
                        "content": f"Previous response rejected ({code}). Follow the catalog and exact JSON schema. Retry.",
                    }
                )
            else:
                self._record(verb, started, "ok")
                return value
        return None


class GemmaCriticProvider(_GemmaRoleProvider):
    """Single audit with a bounded internal read loop; failed audits return empty."""

    fallback = "empty_violations"

    def __init__(
        self,
        *,
        view: CriticView | None = None,
        max_reads: int = DEFAULT_CRITIC_READS,
        **kwargs: Any,
    ) -> None:
        if isinstance(max_reads, bool) or not isinstance(max_reads, int) or max_reads < 0:
            raise ValueError("max_reads must be a nonnegative integer")
        super().__init__(**kwargs)
        self._view = view
        self._max_reads = max_reads

    def bind_view(self, view: CriticView) -> None:
        self._view = view

    def __call__(self, turn: CriticTurn) -> Mapping[str, Any] | str:
        body = role_request_body(
            turn,
            role="critic",
            model=self._model,
        )
        if self._view is not None:
            row_count = self._view.call("get_overview")["row_count"]
        else:
            # Standalone use can validate only AO handles present in the draft.
            refs = [
                ref for block in turn.draft.get("blocks", []) for ref in block.get("step_refs", [])
            ]
            row_count = max(refs, default=-1) + 1
        body["messages"].append({"role": "user", "content": _dump({"row_count": row_count})})
        block_ids = {block["block_id"] for block in turn.draft.get("blocks", [])}

        def parse(payload: Mapping[str, Any]) -> tuple[Any, str]:
            message = _message(payload)
            if message.get("tool_calls"):
                if read_index == self._max_reads:
                    raise LenovoGemmaError("critic_read_budget_exhausted")
                invocation = _tool(
                    payload, tuple((tool["function"]["name"], "") for tool in body["tools"])
                )
                _validate_schema(invocation.arguments, TOOL_SCHEMAS[invocation.verb])
                if self._view is None:
                    raise LenovoGemmaError("critic_view_missing")
                return invocation, invocation.verb
            content = message.get("content")
            # Preserve raw text on JSON failure so the protocol parser owns the error.
            try:
                loaded = _strict_json(content)
            except (TypeError, ValueError):
                loaded = content
                if isinstance(content, str):
                    # Reject duplicate keys/nonfinite JSON even if json.loads accepts them.
                    try:
                        json.loads(content)
                    except ValueError:
                        pass
                    else:
                        raise CriticError("invalid_critic_output")
            parse_critic_violations(loaded, block_ids=block_ids, row_count=row_count)
            return {
                "violations": loaded if isinstance(loaded, list) else loaded["violations"]
            }, "violations"

        for read_index in range(self._max_reads + 1):
            if read_index == self._max_reads:
                body["tool_choice"] = "none"
                body["messages"].append(
                    {
                        "role": "user",
                        "content": 'Read budget exhausted. Return final {"violations": [...]} now.',
                    }
                )
            value = self._validated(body, parse)
            if value is None:
                return {"violations": []}
            if isinstance(value, Mapping):
                return value
            try:
                result = self._view.call(value.verb, value.arguments)
                record = {
                    "verb": value.verb,
                    "arguments": value.arguments,
                    "ok": True,
                    "result": result,
                }
            except CriticError as exc:
                record = {
                    "verb": value.verb,
                    "arguments": value.arguments,
                    "ok": False,
                    "error_code": exc.code,
                }
            exchange = messages_from_turn(ToolloopTurn("", turn.tools, (record,)))[1:]
            call_id = f"critic-read-{read_index}"
            exchange[0]["tool_calls"][0]["id"] = call_id
            exchange[1]["tool_call_id"] = call_id
            body["messages"].extend(exchange)
        return {"violations": []}


class GemmaRefinerProvider(_GemmaRoleProvider):
    fallback = "done"

    def __call__(self, turn: RefinerTurn) -> RefinerAction:
        body = role_request_body(
            turn,
            role="refiner",
            model=self._model,
        )
        by_id = {item["violation_id"]: item for item in turn.violations}

        def parse(payload: Mapping[str, Any]) -> tuple[RefinerAction, str]:
            message = _message(payload)
            if not message.get("tool_calls"):
                content = message.get("content")
                if content is None or (isinstance(content, str) and not content.strip()):
                    return RefinerAction("done"), "done"
                loaded = _strict_json(content)
                if loaded == {"action": "done"}:
                    return RefinerAction("done"), "done"
                if not isinstance(loaded, dict) or set(loaded) != {
                    "action",
                    "violation_id",
                    "reason",
                }:
                    raise LenovoGemmaError("invalid_refiner_output")
                if (
                    loaded["action"] != "unresolvable"
                    or loaded["violation_id"] not in by_id
                    or not isinstance(loaded["reason"], str)
                    or not loaded["reason"].strip()
                ):
                    raise LenovoGemmaError("invalid_refiner_output")
                return RefinerAction(
                    "unresolvable", violation_id=loaded["violation_id"], reason=loaded["reason"]
                ), "unresolvable"
            invocation = _tool(
                payload, tuple((tool["function"]["name"], "") for tool in body["tools"])
            )
            args = dict(invocation.arguments)
            has_violation_id = "violation_id" in args
            violation_id = args.pop("violation_id", "")
            if invocation.verb in REFINER_EDIT_VERBS:
                if not isinstance(violation_id, str) or violation_id not in by_id:
                    raise LenovoGemmaError("unknown_violation")
                violation = by_id[violation_id]
                touched = _touched(invocation.verb, args)
                if violation["kind"] in {"semantic", "granularity"} and (
                    invocation.verb != "modify_block"
                    or args.get("block_id") not in violation.get("involved_ids", [])
                ):
                    raise LenovoGemmaError("edit_out_of_violation_scope")
                if violation["kind"] not in {"semantic", "granularity"}:
                    if invocation.verb == "add_block":
                        valid_scope = violation.get("scope") == "graph"
                    elif violation.get("scope") == "graph":
                        valid_scope = touched <= _graph_ids(turn.draft)
                    else:
                        valid_scope = bool(touched) and touched <= set(
                            violation.get("involved_ids", [])
                        )
                    if not valid_scope:
                        raise LenovoGemmaError("edit_out_of_violation_scope")
            elif has_violation_id:
                raise LenovoGemmaError("invalid_tool_arguments")
            _validate_schema(args, TOOL_SCHEMAS[invocation.verb])
            # RefinerSession enforces all structural scope and graph constraints.
            return RefinerAction("edit", invocation.verb, args, violation_id), invocation.verb

        result = self._validated(body, parse)
        return result if result is not None else RefinerAction("done")


class GemmaReaderProvider(_GemmaRoleProvider):
    fallback = "reader_failure"

    def __call__(self, turn: ReaderTurn) -> ReaderInvocation:
        body = role_request_body(turn, role="reader", model=self._model)

        def parse(payload: Mapping[str, Any]) -> tuple[ReaderInvocation, str]:
            message = _message(payload)
            if message.get("tool_calls"):
                invocation = _tool(
                    payload, tuple((tool["function"]["name"], "") for tool in body["tools"])
                )
                _validate_schema(invocation.arguments, TOOL_SCHEMAS[invocation.verb])
                return ReaderInvocation(
                    "tool", invocation.verb, invocation.arguments
                ), invocation.verb
            loaded = _strict_json(message.get("content"))
            report = parse_reader_report(
                loaded,
                turn.graph,
                allow_unresolved=turn.remaining_calls is not None,
            )
            if turn.remaining_calls is not None and not set(report.decisive_ao_refs) <= set(
                turn.read_ao_refs
            ):
                raise ReaderError("unread_decisive_refs")
            return ReaderInvocation("report", report=report.to_dict()), "report"

        result = self._validated(body, parse)
        if result is None:
            raise ReaderError("gemma_reader_failed")
        return result
