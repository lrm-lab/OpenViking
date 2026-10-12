# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Lenovo gateway Gemma adapter for the builder ToolloopProvider (ADR-0045).

Zero third-party HTTP stack: requests go through stdlib urllib. The API key is
read from ``LENOVO_GATEWAY_API_KEY`` and is never logged or stored on ``stats``.
"""

from __future__ import annotations

import atexit
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openviking.session.aig_native_protocol import canonical_dumps
from openviking.session.aig_tool_surface import TOOL_CATALOG
from openviking.session.aig_toolloop_builder import ToolInvocation, ToolloopError, ToolloopTurn

GATEWAY_URL = "https://modelfactory.lenovo.com/llm/v1/chat/completions"
DEFAULT_MODEL = "gemma-4-31B-it"
API_KEY_ENV = "LENOVO_GATEWAY_API_KEY"
DEFAULT_TIMEOUT_SECONDS = 120.0
RETRY_BACKOFF_SECONDS = 0.25
MAX_NETWORK_RETRIES = 1

# Known argument names copied from AigToolSurface._exact_args. A bare
# ``{"type": "object"}`` would let the model invent keys that the server
# rejects as invalid_args; listing the closed set keeps the schema aligned
# with the call that actually runs.
TOOL_ARGUMENT_NAMES: Mapping[str, tuple[str, ...]] = {
    "get_overview": (),
    "read_full_log": (),
    "read_step": ("index", "i"),
    "read_step_range": ("start", "end", "a", "b"),
    "list_agents": (),
    "get_question": (),
    "search": ("pattern", "flags"),
    "list_block_kinds": (),
    "get_story": (),
    "get_graph": (),
    "add_block": ("kind", "block_id", "fields", "from_refs", "to_refs"),
    "add_inheritance_note": ("from", "to", "inherited", "effect"),
    "add_skip_attribution": ("from", "to", "inherited", "effect"),
    "modify_block": ("block_id", "fields"),
    "delete_block": ("block_id",),
    "delete_inheritance_note": ("from", "to"),
    "delete_skip_attribution": ("from", "to"),
    "mark_defect_recorded": ("block_id", "note"),
    "submit": (),
}

_OBJECT_KEYS = frozenset({"fields"})
_ARRAY_KEYS = frozenset({"from_refs", "to_refs"})
_INTEGER_KEYS = frozenset({"index", "i", "start", "end", "a", "b"})


class LenovoGemmaError(ToolloopError):
    """Adapter-level failure. ``code`` is stable and never carries the key."""


@dataclass(frozen=True)
class ProviderCallStat:
    """One model-side call. ``verb`` is empty when no tool call was parsed."""

    verb: str
    elapsed_ms: int
    status: str


def openai_tools_from_catalog(
    catalog: Sequence[tuple[str, str]] = TOOL_CATALOG,
) -> list[dict[str, Any]]:
    """Convert ``TOOL_CATALOG`` into OpenAI function tools."""
    tools: list[dict[str, Any]] = []
    for verb, description in catalog:
        names = TOOL_ARGUMENT_NAMES.get(verb, ())
        properties: dict[str, Any] = {}
        for name in names:
            if name in _OBJECT_KEYS:
                schema: dict[str, Any] = {"type": "object"}
            elif name in _ARRAY_KEYS:
                schema = {"type": "array"}
            elif name in _INTEGER_KEYS:
                schema = {"type": "integer"}
            else:
                schema = {"type": "string"}
            properties[name] = schema
        parameters: dict[str, Any] = {"type": "object", "additionalProperties": False}
        if properties:
            parameters["properties"] = properties
        else:
            parameters["properties"] = {}
        extra = f" Known arguments: {', '.join(names)}." if names else " No arguments."
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": verb,
                    "description": description + extra,
                    "parameters": parameters,
                },
            }
        )
    return tools


def messages_from_turn(turn: ToolloopTurn) -> list[dict[str, Any]]:
    """Map a builder turn onto chat messages.

    History rows are completed server tool exchanges, not raw model messages:
    each row becomes an assistant ``tool_calls`` message plus a ``tool`` result.
    """
    messages: list[dict[str, Any]] = [{"role": "system", "content": turn.system_prompt}]
    for index, item in enumerate(turn.history):
        if not isinstance(item, Mapping):
            continue
        verb = item.get("verb")
        if not isinstance(verb, str) or not verb:
            continue
        arguments = item.get("arguments")
        if not isinstance(arguments, Mapping):
            arguments = {}
        call_id = f"call-{index}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": verb,
                            "arguments": _dump_arguments(arguments),
                        },
                    }
                ],
            }
        )
        payload = {
            "error_code": item.get("error_code"),
            "ok": item.get("ok"),
            "result": item.get("result"),
        }
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": _dump_json(payload),
            }
        )
    return messages


def request_body_from_turn(
    turn: ToolloopTurn, *, model: str = DEFAULT_MODEL
) -> dict[str, Any]:
    """JSON body for ``/chat/completions``. No secrets."""
    return {
        "messages": messages_from_turn(turn),
        "model": model,
        "tool_choice": "required",
        "tools": openai_tools_from_catalog(turn.tools),
    }


def parse_tool_invocation(payload: Mapping[str, Any]) -> ToolInvocation:
    """Parse one completion. Bad argument JSON is an auditable error, not a crash."""
    choice = _first_choice(payload)
    message = choice.get("message")
    if not isinstance(message, Mapping):
        message = {}
    tool_calls = message.get("tool_calls")
    finish = choice.get("finish_reason")
    if isinstance(tool_calls, list) and tool_calls:
        return _invocation_from_tool_call(tool_calls[0])
    content = message.get("content")
    if finish == "tool_calls":
        raise LenovoGemmaError("invalid_provider_result")
    if isinstance(content, str) and content.strip():
        raise LenovoGemmaError("invalid_provider_result")
    raise LenovoGemmaError("invalid_provider_result")


class LenovoGemmaProvider:
    """True-LLM ``ToolloopProvider``. Tests monkeypatch ``urlopen``, never a model."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        url: str = GATEWAY_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        urlopen: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
        audit_path: Path | str | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
        if not isinstance(key, str) or not key.strip():
            raise LenovoGemmaError("missing_api_key")
        self._api_key = key.strip()
        self._model = model
        self._url = url
        self._timeout = float(timeout_seconds)
        self._urlopen = urlopen or urllib.request.urlopen
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self._stats: list[ProviderCallStat] = []
        self._audit_path = Path(audit_path) if audit_path is not None else None
        if self._audit_path is not None:
            atexit.register(self.flush_audit)

    @property
    def stats(self) -> tuple[ProviderCallStat, ...]:
        return tuple(self._stats)

    def flush_audit(self, path: Path | str | None = None) -> Path | None:
        """Write model-side stats as JSONL. Used as the process-exit hook."""
        target = Path(path) if path is not None else self._audit_path
        if target is None:
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(
                {"elapsed_ms": item.elapsed_ms, "status": item.status, "verb": item.verb},
                ensure_ascii=False,
            )
            for item in self._stats
        ]
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return target

    def __call__(self, turn: ToolloopTurn) -> ToolInvocation:
        if not isinstance(turn, ToolloopTurn):
            raise LenovoGemmaError("invalid_provider_result")
        started = self._clock()
        try:
            payload = self._complete(request_body_from_turn(turn, model=self._model))
            invocation = parse_tool_invocation(payload)
        except LenovoGemmaError as exc:
            verb = _verb_hint(payload) if "payload" in locals() else ""
            self._record(verb, started, exc.code)
            if exc.code == "invalid_arguments_json":
                return ToolInvocation(verb or "unknown", {"_error": "invalid_arguments_json"})
            if exc.code == "invalid_provider_result":
                return ToolInvocation("invalid_provider_result", {})
            raise
        except Exception:
            self._record("", started, "provider_error")
            raise
        self._record(invocation.verb, started, "ok")
        return invocation

    def _complete(self, body: Mapping[str, Any]) -> dict[str, Any]:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        attempts = 1 + MAX_NETWORK_RETRIES
        for attempt in range(attempts):
            request = urllib.request.Request(
                self._url,
                data=raw,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with self._urlopen(request, timeout=self._timeout) as response:
                    status = int(getattr(response, "status", 200) or 200)
                    blob = response.read()
                if status >= 500:
                    raise _RetryableHttp(status)
                if status >= 400:
                    raise LenovoGemmaError("http_error")
                try:
                    loaded = json.loads(blob.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise LenovoGemmaError("invalid_provider_result") from exc
                if not isinstance(loaded, dict):
                    raise LenovoGemmaError("invalid_provider_result")
                return loaded
            except _RetryableHttp as exc:
                last_error = exc
            except urllib.error.HTTPError as exc:
                if exc.code >= 500:
                    last_error = exc
                else:
                    raise LenovoGemmaError("http_error") from exc
            except urllib.error.URLError as exc:
                last_error = exc
            except TimeoutError as exc:
                last_error = exc
            if attempt + 1 < attempts:
                self._sleep(RETRY_BACKOFF_SECONDS)
                continue
            if last_error is not None:
                raise LenovoGemmaError("network_error") from last_error
        raise LenovoGemmaError("network_error")

    def _record(self, verb: str, started: float, status: str) -> None:
        elapsed = int((self._clock() - started) * 1000)
        if elapsed < 0:
            elapsed = 0
        self._stats.append(ProviderCallStat(verb=verb, elapsed_ms=elapsed, status=status))


def _invocation_from_tool_call(call: Any) -> ToolInvocation:
    if not isinstance(call, Mapping):
        raise LenovoGemmaError("invalid_provider_result")
    function = call.get("function")
    if not isinstance(function, Mapping):
        function = call
    verb = function.get("name")
    if not isinstance(verb, str) or not verb:
        raise LenovoGemmaError("invalid_provider_result")
    raw_args = function.get("arguments", {})
    if isinstance(raw_args, Mapping):
        arguments = dict(raw_args)
    elif isinstance(raw_args, str):
        if not raw_args.strip():
            arguments = {}
        else:
            try:
                loaded = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise LenovoGemmaError("invalid_arguments_json") from exc
            if not isinstance(loaded, Mapping):
                raise LenovoGemmaError("invalid_arguments_json")
            arguments = dict(loaded)
    else:
        raise LenovoGemmaError("invalid_arguments_json")
    return ToolInvocation(verb, arguments)


def _first_choice(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LenovoGemmaError("invalid_provider_result")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise LenovoGemmaError("invalid_provider_result")
    return choice


def _verb_hint(payload: Mapping[str, Any]) -> str:
    try:
        call = _first_choice(payload).get("message", {})
        if isinstance(call, Mapping):
            tools = call.get("tool_calls")
            if isinstance(tools, list) and tools and isinstance(tools[0], Mapping):
                function = tools[0].get("function", tools[0])
                if isinstance(function, Mapping):
                    name = function.get("name")
                    if isinstance(name, str) and name:
                        return name
    except LenovoGemmaError:
        return "unknown"
    return "unknown"


def _dump_arguments(arguments: Mapping[str, Any]) -> str:
    try:
        return canonical_dumps(dict(arguments))
    except (TypeError, ValueError):
        return "{}"


def _dump_json(value: Any) -> str:
    try:
        return canonical_dumps(value)
    except (TypeError, ValueError):
        return json.dumps(value, ensure_ascii=False, default=str)


class _RetryableHttp(Exception):
    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(status)
