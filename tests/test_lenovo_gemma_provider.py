# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ADR-0045 Wave E2: Lenovo Gemma ToolloopProvider, zero LLM."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from benchmark.causal_memory.lenovo_gemma_provider import (
    API_KEY_ENV,
    DEFAULT_MODEL,
    GATEWAY_URL,
    LenovoGemmaError,
    LenovoGemmaProvider,
    openai_tools_from_catalog,
    parse_tool_invocation,
    request_body_from_turn,
)
from openviking.session.aig_tool_surface import TOOL_CATALOG, TOOL_SCHEMAS
from openviking.session.aig_toolloop_builder import ToolloopTurn


def _turn() -> ToolloopTurn:
    return ToolloopTurn(
        system_prompt="build the graph",
        tools=TOOL_CATALOG,
        history=(
            {
                "arguments": {"index": 0},
                "error_code": None,
                "ok": True,
                "result": {"index": 0},
                "verb": "read_step",
            },
        ),
    )


def _response(payload: dict[str, Any], status: int = 200) -> _FakeHttp:
    blob = json.dumps(payload).encode("utf-8")
    return _FakeHttp(blob, status)


class _FakeHttp:
    def __init__(self, blob: bytes, status: int) -> None:
        self._blob = blob
        self.status = status

    def read(self) -> bytes:
        return self._blob

    def __enter__(self) -> _FakeHttp:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_request_body_maps_system_tools_and_history() -> None:
    body = request_body_from_turn(_turn())
    assert body["model"] == DEFAULT_MODEL
    assert body["messages"][0] == {"role": "system", "content": "build the graph"}
    assert body["messages"][1]["role"] == "assistant"
    assert body["messages"][1]["tool_calls"][0]["function"]["name"] == "read_step"
    assert json.loads(body["messages"][1]["tool_calls"][0]["function"]["arguments"]) == {
        "index": 0
    }
    assert body["messages"][2]["role"] == "tool"
    assert body["messages"][2]["tool_call_id"] == "call-0"
    names = [item["function"]["name"] for item in body["tools"]]
    assert names == [verb for verb, _text in TOOL_CATALOG]
    add_block = next(item for item in body["tools"] if item["function"]["name"] == "add_block")
    props = add_block["function"]["parameters"]["properties"]
    assert set(props) == set(TOOL_SCHEMAS["add_block"]["properties"])
    assert add_block["function"]["parameters"]["additionalProperties"] is False
    assert "properties" in props["fields"]


def test_messages_from_turn_cover_catalog_verbs() -> None:
    tools = openai_tools_from_catalog()
    assert len(tools) == len(TOOL_CATALOG)
    search = next(item for item in tools if item["function"]["name"] == "search")
    assert "pattern" in search["function"]["parameters"]["properties"]
    submit = next(item for item in tools if item["function"]["name"] == "submit")
    assert submit["function"]["parameters"]["properties"] == {}


def test_parse_tool_calls_into_invocation() -> None:
    invocation = parse_tool_invocation(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": '{"index": 1}',
                                    "name": "read_step",
                                }
                            }
                        ]
                    },
                }
            ]
        }
    )
    assert invocation.verb == "read_step"
    assert invocation.arguments == {"index": 1}


def test_bad_arguments_json_is_auditable_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "test-key")
    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {"function": {"arguments": "{not-json", "name": "add_block"}}
                    ]
                },
            }
        ]
    }

    def urlopen(_request: object, timeout: float = 0) -> _FakeHttp:
        del timeout
        return _response(payload)

    provider = LenovoGemmaProvider(urlopen=urlopen, sleep=lambda _s: None)
    invocation = provider(_turn())
    assert invocation.verb == "add_block"
    assert invocation.arguments["_error"] == "invalid_arguments_json"
    assert provider.stats[-1].status == "invalid_arguments_json"


def test_content_without_tool_calls_is_audited_and_loop_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(API_KEY_ENV, "test-key")
    payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "I am done", "tool_calls": []},
            }
        ]
    }

    def urlopen(_request: object, timeout: float = 0) -> _FakeHttp:
        del timeout
        return _response(payload)

    provider = LenovoGemmaProvider(urlopen=urlopen, sleep=lambda _s: None)
    invocation = provider(_turn())
    assert invocation.verb == "invalid_provider_result"
    assert provider.stats[-1].status == "invalid_provider_result"


def test_retries_once_on_5xx_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "test-key")
    calls = {"n": 0}
    sleeps: list[float] = []
    ok = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {"function": {"arguments": "{}", "name": "get_overview"}}
                    ]
                },
            }
        ]
    }

    def urlopen(request: object, timeout: float = 0) -> _FakeHttp:
        del timeout
        assert isinstance(request, object)
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                GATEWAY_URL, 502, "bad gateway", None, io.BytesIO(b"")
            )
        return _response(ok)

    provider = LenovoGemmaProvider(urlopen=urlopen, sleep=sleeps.append)
    invocation = provider(_turn())
    assert invocation.verb == "get_overview"
    assert calls["n"] == 2
    assert sleeps == [0.25]


def test_missing_key_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(LenovoGemmaError) as caught:
        LenovoGemmaProvider()
    assert caught.value.code == "missing_api_key"


def test_http_request_uses_gateway_and_does_not_echo_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(API_KEY_ENV, "test-key")
    captured: dict[str, Any] = {}
    ok = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [{"function": {"arguments": "{}", "name": "submit"}}]
                },
            }
        ]
    }

    def urlopen(request: Any, timeout: float = 0) -> _FakeHttp:
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = {key.title(): value for key, value in request.header_items()}
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _response(ok)

    provider = LenovoGemmaProvider(urlopen=urlopen, sleep=lambda _s: None)
    provider(_turn())
    assert captured["url"] == GATEWAY_URL
    assert captured["timeout"] == 120.0
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert "test-key" not in json.dumps(captured["body"])
    assert captured["body"]["model"] == DEFAULT_MODEL


def test_flush_audit_writes_stats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "test-key")
    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [{"function": {"arguments": "{}", "name": "submit"}}]
                },
            }
        ]
    }
    provider = LenovoGemmaProvider(
        urlopen=lambda _r, timeout=0: _response(payload),
        sleep=lambda _s: None,
        audit_path=tmp_path / "audit.jsonl",
    )
    provider(_turn())
    path = provider.flush_audit()
    assert path is not None
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["verb"] == "submit"
    assert json.loads(lines[0])["status"] == "ok"
