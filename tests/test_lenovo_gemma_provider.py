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

from benchmark.causal_memory import lenovo_gemma_provider as provider_module
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
    assert json.loads(body["messages"][1]["tool_calls"][0]["function"]["arguments"]) == {"index": 0}
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
                    "tool_calls": [{"function": {"arguments": "{not-json", "name": "add_block"}}]
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
                    "tool_calls": [{"function": {"arguments": "{}", "name": "get_overview"}}]
                },
            }
        ]
    }

    def urlopen(request: object, timeout: float = 0) -> _FakeHttp:
        del timeout
        assert isinstance(request, object)
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(GATEWAY_URL, 502, "bad gateway", None, io.BytesIO(b""))
        return _response(ok)

    provider = LenovoGemmaProvider(urlopen=urlopen, sleep=sleeps.append)
    invocation = provider(_turn())
    assert invocation.verb == "get_overview"
    assert calls["n"] == 2
    assert len(sleeps) == 1
    assert 15 <= sleeps[0] <= 20


def test_retry_backoff_delay_is_exponential_with_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_module.random, "uniform", lambda _a, b: b)
    assert provider_module.retry_backoff_delay(0) == pytest.approx(20)
    assert provider_module.retry_backoff_delay(1) == pytest.approx(35)
    assert provider_module.retry_backoff_delay(2) == pytest.approx(65)
    assert provider_module.retry_backoff_delay(-3) == pytest.approx(20)


def test_exhausts_retries_on_persistent_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "test-key")
    calls = {"n": 0}
    sleeps: list[float] = []

    def urlopen(request: object, timeout: float = 0) -> _FakeHttp:
        del timeout
        assert isinstance(request, object)
        calls["n"] += 1
        raise urllib.error.HTTPError(GATEWAY_URL, 502, "bad gateway", None, io.BytesIO(b""))

    provider = LenovoGemmaProvider(urlopen=urlopen, sleep=sleeps.append)
    with pytest.raises(LenovoGemmaError) as caught:
        provider(_turn())
    assert caught.value.code == "network_error"
    assert calls["n"] == 1 + provider_module.MAX_NETWORK_RETRIES
    assert len(sleeps) == provider_module.MAX_NETWORK_RETRIES
    assert sleeps[0] < sleeps[1] < sleeps[2]


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
                "message": {"tool_calls": [{"function": {"arguments": "{}", "name": "submit"}}]},
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
    assert captured["timeout"] == provider_module.DEFAULT_TIMEOUT_SECONDS
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert "test-key" not in json.dumps(captured["body"])
    assert captured["body"]["model"] == DEFAULT_MODEL


def test_flush_audit_writes_stats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "test-key")
    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {"tool_calls": [{"function": {"arguments": "{}", "name": "submit"}}]},
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


@pytest.mark.parametrize(
    "failure",
    [
        429,
        408,
        ConnectionResetError(),
        __import__("http.client", fromlist=["IncompleteRead"]).IncompleteRead(b"partial"),
    ],
)
def test_transient_transport_failure_stays_inside_request_retry(failure):
    calls = []
    sleeps = []

    def transport(request, **kwargs):
        calls.append(request)
        if len(calls) == 1:
            if isinstance(failure, int):
                raise urllib.error.HTTPError(
                    GATEWAY_URL, failure, "temporary", {"Retry-After": "3"}, io.BytesIO()
                )
            raise failure
        return _response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "get_overview", "arguments": "{}"}}
                            ]
                        }
                    }
                ]
            }
        )

    instance = LenovoGemmaProvider(api_key="test-key", urlopen=transport, sleep=sleeps.append)
    assert instance(_turn()).verb == "get_overview"
    assert len(calls) == 2 and len(instance.stats) == 1
    assert sleeps[0] >= (3 if isinstance(failure, int) else 0.25)


def test_pooled_request_client_waits_longer_than_pool_budget(monkeypatch):
    monkeypatch.setenv("LENOVO_GATEWAY_TIMEOUT_SECONDS", "960")
    seen = []

    def transport(request, *, timeout):
        seen.append(timeout)
        return _response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "get_overview", "arguments": "{}"}}
                            ]
                        }
                    }
                ]
            }
        )

    instance = LenovoGemmaProvider(api_key="test-key", urlopen=transport)
    assert instance(_turn()).verb == "get_overview"
    assert seen == [960]


def test_builder_compacts_only_consumed_evidence_and_old_build_actions() -> None:
    from benchmark.causal_memory.lenovo_gemma_provider import request_body_from_turn
    from openviking.session.aig_toolloop_builder import ToolloopTurn

    history = [
        {
            "verb": "read_step_range",
            "arguments": {"start": 0, "end": 1},
            "ok": True,
            "result": [
                {"index": 0, "record": {"action": {"text": "ALREADY_COVERED_EVIDENCE"}}},
                {"index": 1, "record": {"action": {"text": "STILL_REQUIRED_EVIDENCE"}}},
            ],
        },
        *[
            {
                "verb": "add_block",
                "arguments": {"kind": "agent", "fields": {"summary": f"OLD_OPERATION_{i}"}},
                "ok": True,
                "result": {"block_id": f"b{i}"},
            }
            for i in range(10)
        ],
    ]
    draft = {"blocks": [{"block_id": "b0", "step_refs": [0]}], "edges": []}
    turn = ToolloopTurn("instructions", (), tuple(history), draft=draft)
    body = request_body_from_turn(turn)
    serialized = json.dumps(body)
    assert "ALREADY_COVERED_EVIDENCE" not in serialized
    assert "STILL_REQUIRED_EVIDENCE" in serialized
    assert "OLD_OPERATION_0" not in serialized
    assert "OLD_OPERATION_9" in serialized
    assert json.loads(body["messages"][-1]["content"])["draft"] == draft


def test_builder_keeps_freshly_reread_evidence_even_when_covered() -> None:
    from benchmark.causal_memory.lenovo_gemma_provider import request_body_from_turn
    from openviking.session.aig_toolloop_builder import ToolloopTurn

    history = (
        {
            "verb": "read_step",
            "arguments": {"index": 0},
            "ok": True,
            "result": {"index": 0, "record": {"observation": {"text": "FRESH_VERIFICATION"}}},
        },
    )
    body = request_body_from_turn(
        ToolloopTurn(
            "instructions",
            (),
            history,
            draft={"blocks": [{"block_id": "b0", "step_refs": [0]}], "edges": []},
        )
    )
    assert "FRESH_VERIFICATION" in json.dumps(body)
