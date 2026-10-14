# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Gemma four-role wiring, using only injected HTTP responses (zero LLM)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from benchmark.causal_memory import aig0045_verdict_smoke as smoke
from benchmark.causal_memory.aig0045_gemma_roles import (
    GemmaCriticProvider,
    GemmaReaderProvider,
    GemmaRefinerProvider,
    role_request_body,
)
from benchmark.causal_memory import lenovo_gemma_provider as lenovo
from benchmark.causal_memory.lenovo_gemma_provider import (
    API_KEY_ENV,
    DEFAULT_MODEL,
    LenovoGemmaProvider,
)
from openviking.session.aig_tool_surface import AigToolSurface, AoSnapshot
from openviking.session.aig_toolloop_builder import FakeToolloopProvider
from openviking.session.aig_toolloop_critic import (
    CriticTurn,
    CriticView,
    critic_catalog,
    critic_prompt_for_mode,
)
from openviking.session.aig_toolloop_reader import (
    ReaderError,
    ReaderTurn,
    reader_catalog,
    reader_prompt_for_mode,
)
from openviking.session.aig_toolloop_refiner import (
    RefinerTurn,
    refiner_catalog,
    refiner_prompt_for_mode,
)


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    def blocked(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Real network is forbidden in role tests")

    monkeypatch.setattr(urllib.request, "urlopen", blocked)


class FakeHttp:
    status = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()

    def __enter__(self) -> FakeHttp:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class Transport:
    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: Any, **kwargs: Any) -> FakeHttp:
        self.bodies.append(json.loads(request.data))
        response = self.responses[min(len(self.bodies) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return FakeHttp(response)


def content(value: Any) -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": value if isinstance(value, str) else json.dumps(value)},
            }
        ]
    }


def tool(name: str, arguments: Any) -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments
                                if isinstance(arguments, str)
                                else json.dumps(arguments),
                            },
                        }
                    ]
                },
            }
        ]
    }


def graph() -> dict[str, Any]:
    return {
        "blocks": [
            {
                "block_id": "b1",
                "kind": "agent",
                "agent_name": "worker",
                "step_refs": [1],
                "summary": "action",
                "input": "input",
                "output": "result",
            }
        ],
        "edges": [],
    }


def violation() -> dict[str, Any]:
    return {
        "violation_id": "v1",
        "kind": "semantic",
        "scope": "objects",
        "involved_ids": ["b1"],
        "object_id": "b1",
        "severity": "high",
        "evidence": "incorrect summary",
    }


def turn(role: str, mode: str = "success") -> Any:
    history = (
        {"verb": "read_step", "arguments": {"index": 1}, "ok": True, "result": "AO evidence"},
    )
    if role == "critic":
        return CriticTurn(
            critic_prompt_for_mode(mode), critic_catalog(), graph(), ({"kind": "structural"},)
        )
    if role == "refiner":
        return RefinerTurn(
            refiner_prompt_for_mode(mode), refiner_catalog(), graph(), (violation(),), history
        )
    return ReaderTurn(reader_prompt_for_mode(mode), reader_catalog(), graph(), history)


def report(**updates: Any) -> dict[str, Any]:
    value = {
        "responsible_agent": "worker",
        "decisive_ao_refs": [1],
        "reason": "AO confirms effect",
        "critical_block_id": "b1",
        "unresolved_areas": [],
    }
    return {**value, **updates}


def provider(provider_type: type, transport: Transport, **kwargs: Any) -> Any:
    return provider_type(api_key="unit-test-key", urlopen=transport, sleep=lambda _: None, **kwargs)


@pytest.mark.parametrize("role", ["critic", "refiner", "reader"])
@pytest.mark.parametrize("mode", ["fail", "success"])
def test_role_messages_catalog_mode_and_context(role: str, mode: str) -> None:
    current = turn(role, mode)
    body = role_request_body(current, role=role, model=DEFAULT_MODEL)
    assert body["model"] == DEFAULT_MODEL
    assert current.system_prompt in body["messages"][0]["content"]
    assert "Only use this role catalog" in body["messages"][0]["content"]
    assert body["tool_choice"] == "auto"
    assert [item["function"]["name"] for item in body["tools"]] == [
        name for name, _ in current.tools
    ]
    state = json.loads(body["messages"][-1]["content"])
    assert state["catalog"] == [list(item) for item in current.tools]
    assert state["frozen_graph" if role == "reader" else "draft"] == graph()
    if role != "critic":
        assert state["history"][0]["result"] == "AO evidence"
        assert any(
            item["role"] == "tool" and "AO evidence" in item["content"] for item in body["messages"]
        )
    else:
        assert state["structural_violations"] == [{"kind": "structural"}]
    if role == "reader":
        descriptions = {
            item["function"]["name"]: item["function"]["description"] for item in body["tools"]
        }
        assert "frozen native revision" in descriptions["get_graph"]
    if role == "refiner":
        schemas = {
            item["function"]["name"]: item["function"]["parameters"] for item in body["tools"]
        }
        assert "violation_id" in schemas["modify_block"]["required"]
        assert "violation_id" not in schemas["read_step"]["properties"]


@pytest.mark.parametrize("bare_list", [False, True])
def test_critic_valid_violations(bare_list: bool) -> None:
    items = [
        {
            "object_id": "b1",
            "kind": "semantic",
            "severity": "high",
            "evidence_ao_refs": [1],
            "note": "AO contradicts summary",
        }
    ]
    transport = Transport(content(items if bare_list else {"violations": items}))
    current = provider(GemmaCriticProvider, transport)
    assert current(turn("critic")) == {"violations": items}
    assert len(current.stats) == 1
    assert current.stats[0].status == "ok"
    assert current.failures == ()


def test_critic_prompt_explains_nonempty_output_contract() -> None:
    body = role_request_body(turn("critic"), role="critic", model=DEFAULT_MODEL)
    prompt = body["messages"][0]["content"]
    assert '"evidence_ao_refs"' in prompt
    assert '"note"' in prompt
    assert "object_id must name a block" in prompt
    assert "no extra keys" in prompt


def test_reader_prompt_explains_exact_frozen_binding() -> None:
    body = role_request_body(turn("reader"), role="reader", model=DEFAULT_MODEL)
    prompt = body["messages"][0]["content"]
    assert "entire step_refs list in the same order" in prompt
    assert "agent_name is absent" in prompt
    assert "not a subset" in prompt


def test_critic_internal_read_exchange_and_actual_row_count(tmp_path: Path) -> None:
    spec = smoke.synthetic_materials()[0]
    snapshot = AoSnapshot(spec.records, watermark="wm", question=spec.question)
    surface = AigToolSurface(snapshot, root=tmp_path, run_id="critic-test")
    transport = Transport(
        tool("read_step", {"index": 2}),
        content(
            {
                "violations": [
                    {
                        "object_id": "b1",
                        "kind": "semantic",
                        "severity": "high",
                        "evidence_ao_refs": [2],
                        "note": "verified row 2",
                    }
                ]
            }
        ),
    )
    current = provider(GemmaCriticProvider, transport, view=CriticView(surface))
    assert current(turn("critic"))["violations"][0]["evidence_ao_refs"] == [2]
    assert len(transport.bodies) == 2
    exchange = transport.bodies[1]["messages"][-2:]
    assert exchange[0]["tool_calls"][0]["id"] == exchange[1]["tool_call_id"]
    assert json.loads(exchange[1]["content"])["ok"] is True
    assert [item.verb for item in current.stats] == ["read_step", "violations"]


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        '{"violations":[],"violations":[]}',
        {"violations": [], "extra": True},
        {"violations": [{"object_id": "unknown"}]},
    ],
)
def test_critic_invalid_retries_then_empty(bad: Any) -> None:
    transport = Transport(content(bad))
    current = provider(GemmaCriticProvider, transport)
    assert current(turn("critic")) == {"violations": []}
    assert len(transport.bodies) == len(current.stats) == 3
    assert all(item.status == "invalid_critic_output" for item in current.stats)
    assert current.failures[-1]["fallback"] == "empty_violations"
    assert "Previous response rejected" in transport.bodies[1]["messages"][-1]["content"]


@pytest.mark.parametrize(
    "response,kind",
    [
        (
            tool(
                "modify_block",
                {"violation_id": "v1", "block_id": "b1", "fields": {"summary": "corrected"}},
            ),
            "edit",
        ),
        (tool("read_step", {"index": 1}), "edit"),
        (
            content({"action": "unresolvable", "violation_id": "v1", "reason": "missing evidence"}),
            "unresolvable",
        ),
        (content({"action": "done"}), "done"),
        (content(""), "done"),
    ],
)
def test_refiner_valid_actions(response: dict[str, Any], kind: str) -> None:
    current = provider(GemmaRefinerProvider, Transport(response))
    action = current(turn("refiner"))
    assert action.kind == kind
    if action.verb == "modify_block":
        assert action.violation_id == "v1"
        assert action.arguments == {"block_id": "b1", "fields": {"summary": "corrected"}}
    assert current.stats[0].status == "ok"


@pytest.mark.parametrize(
    "response",
    [
        tool("submit", {}),
        tool("modify_block", {"block_id": "b1", "fields": {"summary": "bad"}}),
        tool(
            "modify_block", {"violation_id": "v1", "block_id": "b2", "fields": {"summary": "bad"}}
        ),
        tool("modify_block", {"violation_id": "unknown", "block_id": "b1", "fields": {}}),
        tool("read_step", {"index": True}),
        tool("read_step", {"index": 1, "extra": 1}),
        tool("read_step", "{bad json"),
        content({"action": "unresolvable", "violation_id": "unknown", "reason": "missing"}),
        content("arbitrary prose"),
    ],
)
def test_refiner_invalid_retries_then_done(response: dict[str, Any]) -> None:
    transport = Transport(response)
    current = provider(GemmaRefinerProvider, transport)
    assert current(turn("refiner")).kind == "done"
    assert len(transport.bodies) == len(current.stats) == 3
    assert all(item.status != "ok" for item in current.stats)
    assert current.failures[-1]["fallback"] == "done"


def test_reader_legal_tool_then_report_and_audit(tmp_path: Path) -> None:
    transport = Transport(tool("read_step", {"index": 1}), content(report()))
    current = provider(GemmaReaderProvider, transport, audit_path=tmp_path / "stats.jsonl")
    invocation = current(turn("reader"))
    assert invocation.kind == "tool" and invocation.arguments == {"index": 1}
    assert current(turn("reader")).report == report()
    path = current.flush_audit()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [item["verb"] for item in rows] == ["read_step", "report"]
    assert all(item["status"] == "ok" and item["elapsed_ms"] >= 0 for item in rows)
    assert "unit-test-key" not in path.read_text()


@pytest.mark.parametrize(
    "bad",
    [
        report(responsible_agent="invented"),
        report(decisive_ao_refs=[0]),
        report(decisive_ao_refs=[True]),
        report(critical_block_id="unknown"),
        report(extra="unexpected"),
        report(reason=""),
        report(unresolved_areas="bad"),
        "```json\n{}\n```",
        '{"critical_block_id":"b1","reason":NaN}',
    ],
)
def test_reader_report_strict_failure_and_bounded_retry(bad: Any) -> None:
    transport = Transport(content(bad))
    current = provider(GemmaReaderProvider, transport)
    with pytest.raises(ReaderError, match="gemma_reader_failed"):
        current(turn("reader"))
    assert len(transport.bodies) == len(current.stats) == 3
    assert all(item.status != "ok" for item in current.stats)
    assert current.failures[-1]["fallback"] == "reader_failure"


@pytest.mark.parametrize(
    "response",
    [
        tool("delete_block", {"block_id": "b1"}),
        tool("read_step", {"index": "1"}),
        tool("read_step", '{"index":1,"index":2}'),
        {"choices": []},
    ],
)
def test_reader_illegal_tools_retry_and_recover(response: dict[str, Any]) -> None:
    current = provider(GemmaReaderProvider, Transport(response, content(report())))
    assert current(turn("reader")).report == report()
    assert len(current.stats) == 2
    assert current.stats[0].status != "ok" and current.stats[1].status == "ok"
    assert not current.failures


def test_multiple_tools_rejected() -> None:
    response = tool("read_step", {"index": 1})
    response["choices"][0]["message"]["tool_calls"] *= 2
    current = provider(GemmaReaderProvider, Transport(response), output_retries=0)
    with pytest.raises(ReaderError):
        current(turn("reader"))
    assert current.stats[0].status == "ambiguous_provider_result"


@pytest.mark.parametrize(
    "provider_type,role",
    [
        (GemmaCriticProvider, "critic"),
        (GemmaRefinerProvider, "refiner"),
        (GemmaReaderProvider, "reader"),
    ],
)
def test_transport_failure_is_audited_and_bounded(provider_type: type, role: str) -> None:
    transport = Transport(urllib.error.URLError("offline"))
    current = provider(provider_type, transport)
    if role == "reader":
        with pytest.raises(ReaderError):
            current(turn(role))
    else:
        current(turn(role))
    assert len(current.stats) == 1
    # Transport owns its retry budget; output correction never repeats failures.
    attempts = 1 + lenovo.MAX_NETWORK_RETRIES
    assert len(transport.bodies) == attempts
    assert all(item.status == "network_error" for item in current.stats)
    assert all("Previous response rejected" not in json.dumps(message)
               for body in transport.bodies for message in body["messages"])


@pytest.mark.parametrize("builder", ["fake", "gemma"])
def test_cli_all_gemma_dry_run_does_not_construct_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, builder: str
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("dry-run must not construct a Gemma provider")

    for name in (
        "LenovoGemmaProvider",
        "GemmaCriticProvider",
        "GemmaRefinerProvider",
        "GemmaReaderProvider",
    ):
        monkeypatch.setattr(smoke, name, forbidden)
    out = tmp_path / "report.json"
    assert (
        smoke.main(
            [
                "--material",
                "synthetic",
                "--builder",
                builder,
                "--critic",
                "gemma",
                "--refiner",
                "gemma",
                "--reader",
                "gemma",
                "--dry-run",
                "--jobs",
                "3",
                "--out",
                str(out),
                "--work-root",
                str(tmp_path / "work"),
            ]
        )
        == 0
    )
    result = json.loads(out.read_text())
    assert result["provider_audit"] == []
    assert (
        result["settings"]["critic"]
        == result["settings"]["refiner"]
        == result["settings"]["reader"]
        == "gemma"
    )
    assert all(row["submit"] is False for row in result["materials"])


@pytest.mark.parametrize("jobs", [1, 3])
def test_smoke_role_failures_preserve_audit_and_survive_batch(tmp_path: Path, jobs: int) -> None:
    instances: list[Any] = []

    def critic_factory(mode: str) -> Any:
        # Semantic violations force a refiner call; repeatedly invalid refiner
        # output degrades to done. The critic then returns a clean audit.
        bad = [
            {
                "object_id": "agent-1",
                "kind": "semantic",
                "severity": "high",
                "evidence_ao_refs": [1],
                "note": "bad summary",
            }
        ]
        current = provider(
            GemmaCriticProvider,
            Transport(content({"violations": bad}), content({"violations": []})),
        )
        instances.append(current)
        return current

    def refiner_factory(mode: str) -> Any:
        current = provider(GemmaRefinerProvider, Transport(tool("submit", {})))
        instances.append(current)
        return current

    def reader_factory(mode: str) -> Any:
        current = provider(
            GemmaReaderProvider, Transport(content(report(critical_block_id="invented")))
        )
        instances.append(current)
        return current

    result = smoke.run_smoke(
        smoke.synthetic_materials(),
        root=tmp_path,
        jobs=jobs,
        builder_factory=lambda mode: FakeToolloopProvider(smoke.legal_builder_script(mode)),
        critic_factory=critic_factory,
        refiner_factory=refiner_factory,
        reader_factory=reader_factory,
    )
    triggered = [row for row in result["materials"] if row["triggered"]]
    assert len(instances) == len(triggered) * 3
    assert len({id(item) for item in instances}) == len(instances)
    assert all(row["submit"] and row["reader_schema_ok"] is False for row in triggered)
    assert all(row["fallback_reason"] == "reader_failed:gemma_reader_failed" for row in triggered)
    assert all(
        {item["role"] for item in row["provider_failures"]} == {"refiner", "reader"}
        for row in triggered
    )
    audit = result["provider_audit"]
    assert {item["role"] for item in audit} == {"critic", "refiner", "reader"}
    assert len(audit) == len(triggered) * 8  # critic 2 + refiner 3 + reader 3
    assert {item["run_id"] for item in audit} == {row["run_id"] for row in triggered}
    assert result["gates"]["reader_schema_and_refs"]["pass"] is False


def test_cli_defaults_and_role_key_validation(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    assert smoke.main(["--out", str(out), "--work-root", str(tmp_path / "work")]) == 0
    settings = json.loads(out.read_text())["settings"]
    assert settings["critic"] == settings["refiner"] == "fake"
    assert settings["reader"] == "deterministic"
    assert smoke.main(["--critic", "gemma", "--out", str(out)]) == 2


def test_role_factory_model_and_instance_isolation() -> None:
    factory = smoke._gemma_role_factory(lambda **kwargs: kwargs, None)
    first, second = factory("fail"), factory("success")
    assert first == second == {} and first is not second
    assert smoke._gemma_role_factory(lambda **kwargs: kwargs, "custom")("fail") == {
        "model": "custom"
    }


def test_critic_read_budget_is_bounded(tmp_path: Path) -> None:
    spec = smoke.synthetic_materials()[0]
    snapshot = AoSnapshot(spec.records, watermark="wm", question=spec.question)
    surface = AigToolSurface(snapshot, root=tmp_path, run_id="critic-budget")
    transport = Transport(tool("get_graph", {}))
    current = provider(GemmaCriticProvider, transport, view=CriticView(surface), max_reads=1)
    assert current(turn("critic")) == {"violations": []}
    assert len(transport.bodies) == 4  # One read followed by three final-output attempts.
    assert transport.bodies[-1]["tool_choice"] == "none"
    assert current.stats[0].status == "ok"
    assert current.failures[-1]["code"] == "critic_read_budget_exhausted"


@pytest.mark.parametrize(
    "updates",
    [
        {"object_id": "missing"},
        {"evidence_ao_refs": [True]},
        {"evidence_ao_refs": [2]},
        {"severity": "medium"},
        {"note": " "},
    ],
)
def test_critic_semantic_schema_rejects_invalid_evidence(updates: dict[str, Any]) -> None:
    item = {
        "object_id": "b1",
        "kind": "semantic",
        "severity": "high",
        "evidence_ao_refs": [1],
        "note": "evidence",
    }
    current = provider(
        GemmaCriticProvider,
        Transport(content({"violations": [{**item, **updates}]})),
        output_retries=0,
    )
    assert current(turn("critic")) == {"violations": []}
    assert current.stats[0].status == "invalid_critic_output"


@pytest.mark.parametrize(
    "scope,block_id,accepted",
    [
        ("objects", "b1", True),
        ("objects", "b2", False),
        ("graph", "b1", True),
        ("graph", "b2", False),
    ],
)
def test_refiner_structural_scope_checked_before_protocol(
    scope: str, block_id: str, accepted: bool
) -> None:
    current_turn = turn("refiner")
    current_turn.violations[0].update({"kind": "structural", "scope": scope})
    current = provider(
        GemmaRefinerProvider,
        Transport(tool("delete_block", {"block_id": block_id, "violation_id": "v1"})),
        output_retries=0,
    )
    action = current(current_turn)
    assert (action.kind == "edit") is accepted
    assert (current.stats[0].status == "ok") is accepted


@pytest.mark.parametrize(
    "response",
    [
        tool("read_step", {"index": 1, "violation_id": ""}),
        tool("modify_block", {"violation_id": "v1", "block_id": "b1", "fields": {"extra": "bad"}}),
        {"choices": [{"finish_reason": "length", "message": {"content": ""}}]},
    ],
)
def test_refiner_invalid_arguments_or_truncation_are_failures(response: dict[str, Any]) -> None:
    current = provider(GemmaRefinerProvider, Transport(response), output_retries=0)
    assert current(turn("refiner")).kind == "done"
    assert current.stats[0].status != "ok"
    assert current.failures


@pytest.mark.parametrize("jobs", [1, 3])
def test_cli_four_gemma_roles_full_chain_fake_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: int
) -> None:
    created: dict[str, list[Any]] = {
        role: [] for role in ("builder", "critic", "refiner", "reader")
    }

    class BuilderTransport(Transport):
        def __call__(self, request: Any, **kwargs: Any) -> FakeHttp:
            body = json.loads(request.data)
            self.bodies.append(body)
            # The real HTTP builder is stateless: a fresh attempt starts with
            # an empty tool history even when the provider object is shared.
            index = (len(body["messages"]) - 1) // 2
            return FakeHttp(self.responses[index])

    def builder_type(**kwargs: Any) -> Any:
        # CLI jobs>1 factory supplies no mode; this test has a fail material.
        script = smoke.legal_builder_script("fail")
        current = provider(
            LenovoGemmaProvider,
            BuilderTransport(*(tool(verb, arguments) for verb, arguments in script)),
            **kwargs,
        )
        created["builder"].append(current)
        return current

    class CriticConstructor(GemmaCriticProvider):
        def __init__(self, **kwargs: Any) -> None:
            items = [
                {
                    "object_id": "agent-1",
                    "kind": "semantic",
                    "severity": "high",
                    "evidence_ao_refs": [1],
                    "note": "correct summary",
                }
            ]
            super().__init__(
                api_key="unit-test-key",
                urlopen=Transport(content({"violations": items}), content({"violations": []})),
                sleep=lambda _: None,
                **kwargs,
            )
            created["critic"].append(self)

    def refiner_type(**kwargs: Any) -> Any:
        current = provider(
            GemmaRefinerProvider,
            Transport(
                tool(
                    "modify_block",
                    {
                        "violation_id": "critic:0:semantic:agent-1",
                        "block_id": "agent-1",
                        "fields": {"summary": "AO verified action"},
                    },
                ),
                content({"action": "done"}),
            ),
            **kwargs,
        )
        created["refiner"].append(current)
        return current

    def reader_type(**kwargs: Any) -> Any:
        current = provider(
            GemmaReaderProvider,
            Transport(
                tool("read_step", {"index": 1}), content(report(critical_block_id="agent-1"))
            ),
            **kwargs,
        )
        created["reader"].append(current)
        return current

    monkeypatch.setattr(smoke, "LenovoGemmaProvider", builder_type)
    monkeypatch.setattr(smoke, "GemmaRefinerProvider", refiner_type)
    monkeypatch.setattr(smoke, "GemmaReaderProvider", reader_type)
    monkeypatch.setattr(smoke, "GemmaCriticProvider", CriticConstructor)
    materials = [spec for spec in smoke.synthetic_materials() if spec.run_id == "syn-swe-fail"]
    monkeypatch.setattr(smoke, "synthetic_materials", lambda: materials)
    out = tmp_path / "report.json"
    assert (
        smoke.main(
            [
                "--material",
                "synthetic",
                "--builder",
                "gemma",
                "--critic",
                "gemma",
                "--refiner",
                "gemma",
                "--reader",
                "gemma",
                "--url",
                "http://127.0.0.1:18740/llm/v1/chat/completions",
                "--jobs",
                str(jobs),
                "--attempts",
                "2",
                "--out",
                str(out),
                "--work-root",
                str(tmp_path / "work"),
            ]
        )
        == 0
    )
    result = json.loads(out.read_text())
    row = result["materials"][0]
    assert row["submit"] is True and row["reader_schema_ok"] is True
    assert row["phi_violations_after_refine"] == 0
    assert row["decisive_refs_in_snapshot"] is True
    assert "provider_failures" not in row
    assert {item["role"] for item in result["provider_audit"]} == set(created)
    assert len({item["run_id"] for item in result["provider_audit"]}) == 2
    usage = result["token_usage"]
    assert set(usage["by_role"]) == set(created)
    assert len(usage["by_run"]) == 2
    assert usage["overall"]["http_attempts_started"] > 0
    assert usage["overall"]["incomplete_http_attempts"] == 0
    # Fake responses omit usage: the summary must explicitly report that gap.
    assert usage["overall"]["usage_complete"] is False
    ledger = Path(result["model_usage_log"])
    assert ledger.is_file() and "unit-test-key" not in ledger.read_text()
    for instances in created.values():
        assert all(item._url == "http://127.0.0.1:18740/llm/v1/chat/completions" for item in instances)
    for role, instances in created.items():
        if jobs == 1 and role == "builder":
            assert len(instances) == 1 and instances[0].stats
            continue
        assert len(instances) == 3  # Fail-fast instance plus two material attempts.
        assert not instances[0].stats
        assert all(item.stats for item in instances[1:])


def test_attempt_folding_keeps_critic_degradation_auditable(tmp_path: Path) -> None:
    materials = [spec for spec in smoke.synthetic_materials() if spec.run_id == "syn-swe-fail"]
    result = smoke.run_smoke(
        materials,
        root=tmp_path,
        attempts=2,
        jobs=2,
        builder_factory=lambda mode: FakeToolloopProvider(smoke.legal_builder_script(mode)),
        critic_factory=lambda mode: provider(GemmaCriticProvider, Transport(content("not JSON"))),
    )
    row = result["materials"][0]
    assert row["submit"] is False and row["reader_schema_ok"] is None
    assert row["fallback_reason"] == "all 2 attempts failed to submit"
    assert result["gates"]["submit_rate"]["pass"] is False
    for attempt in row["attempts"]:
        assert attempt["fallback_reason"] == "critic_provider_failed"
        assert attempt["provider_failures"] == [
            {"role": "critic", "code": "invalid_critic_output", "fallback": "empty_violations"}
        ]
    assert len(result["provider_audit"]) == 6
    assert all(item["status"] == "invalid_critic_output" for item in result["provider_audit"])


@pytest.mark.parametrize("bad_field", ["type", "finish_reason"])
def test_reader_rejects_invalid_tool_type_or_truncated_completion(bad_field: str) -> None:
    response = tool("get_graph", {})
    if bad_field == "type":
        response["choices"][0]["message"]["tool_calls"][0]["type"] = "unknown"
    else:
        response["choices"][0]["finish_reason"] = "length"
    current = provider(GemmaReaderProvider, Transport(response), output_retries=0)
    with pytest.raises(ReaderError, match="gemma_reader_failed"):
        current(turn("reader"))
    assert current.stats[0].status == "invalid_provider_result"


@pytest.mark.parametrize("status", [424, 401])
def test_exhausted_transport_is_not_fed_back_as_model_output_error(status):
    import io
    import urllib.error
    requests = []
    def transport(request, **kwargs):
        requests.append(json.loads(request.data))
        raise urllib.error.HTTPError("http://pool/llm/v1/chat/completions", status, "failed", {}, io.BytesIO())
    current = provider(GemmaCriticProvider, transport)
    current(turn("critic"))
    assert len(requests) == 1
    assert current.failures[-1]["code"] == "http_error"
    assert all("Previous response rejected" not in item["content"] for item in requests[0]["messages"])
