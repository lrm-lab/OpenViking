# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Unit tests for ADR-0045 real-material pipeline (no HTTP)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from benchmark.causal_memory.aig0045_real_material import (
    cmd_export,
    cmd_reverify,
    transcript_to_records,
)
from benchmark.causal_memory.ao_record_export import read_ao_jsonl
from openviking.session.ao_ledger import AORecord

SESSION_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
_RECORDS_LINE = re.compile(r": (\d+) records ->")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _session_header(session_id: str) -> dict[str, Any]:
    return {
        "type": "session",
        "version": 3,
        "id": session_id,
        "timestamp": "2026-01-01T00:00:00Z",
    }


def _message(msg_id: str, role: str, content: list[dict[str, Any]], ts: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": msg_id,
        "timestamp": ts,
        "message": {"role": role, "content": content},
    }


def _write_session(path: Path, session_id: str, messages: list[dict[str, Any]]) -> Path:
    return _write_jsonl(path, [_session_header(session_id), *messages])


def _user_text(msg_id: str, text: str, ts: str) -> dict[str, Any]:
    return _message(msg_id, "user", [{"type": "text", "text": text}], ts)


def _load_ao_jsonl(path: Path) -> list[AORecord]:
    """Read an AO jsonl written by ``AORecord.to_dict``.

    ``transcript_to_records`` stores ``message_ref=""``, ``observation=None``,
    and ``captured_state=None``. ``AORecord.from_dict`` (used by
    ``read_ao_jsonl``) cannot ingest those defaults, so coerce them first.
    """
    coerced: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if payload.get("message_ref") == "":
                payload["message_ref"] = None
            if payload.get("observation") is None:
                payload["observation"] = {}
            if not payload.get("captured_state"):
                payload["captured_state"] = "live"
            coerced.append(payload)
    scratch = path.with_name(path.name + ".from_dict.jsonl")
    _write_jsonl(scratch, coerced)
    return read_ao_jsonl(scratch)


def _assert_to_dict_roundtrip(records: list[AORecord], path: Path) -> list[AORecord]:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    raw = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert raw == [record.to_dict() for record in records]
    loaded = _load_ao_jsonl(path)
    assert len(loaded) == len(records)
    for original, back in zip(records, loaded, strict=True):
        assert isinstance(back, AORecord)
        assert back.ao_id == original.ao_id
        assert back.session_id == original.session_id
        assert back.sequence == original.sequence
        assert back.action == original.action
        if original.observation is not None:
            assert back.observation == original.observation
    return loaded


def _export_args(
    tmp_path: Path, attempts: Path, manifest: Path, verdicts: Path
) -> argparse.Namespace:
    return argparse.Namespace(
        attempts=attempts,
        manifest=manifest,
        verdicts=verdicts,
        sessions_out=tmp_path / "sessions-out",
    )


def _assert_compile_preflight(records: list[AORecord], *, participant: str, segment: str) -> None:
    """Minimal check compile_paper_graph/_shared would run before AigToolSurface."""
    assert records
    participant_ids = [record.action.get("participant_id") for record in records]
    segment_ids = [record.action.get("segment_id") for record in records]
    assert all(isinstance(value, str) and value for value in participant_ids)
    assert set(participant_ids) == {participant}
    assert all(isinstance(value, str) and value for value in segment_ids)
    assert set(segment_ids) == {segment}


def _assert_exported_sessions(sessions_out: Path, stdout: str, *, participant: str) -> None:
    printed = [int(match) for match in _RECORDS_LINE.findall(stdout)]
    files = sorted(sessions_out.glob("*.jsonl"))
    assert files
    assert len(files) == len(printed)
    for path, count in zip(files, printed, strict=True):
        rows = _load_ao_jsonl(path)
        assert len(rows) == count
        assert [row.sequence for row in rows] == list(range(1, count + 1))
        _assert_compile_preflight(rows, participant=participant, segment=rows[0].session_id)


def test_transcript_to_records_pairs_tools_skips_thinking_and_roundtrips(tmp_path: Path) -> None:
    transcript = tmp_path / ".builtin-sessions" / f"{SESSION_ID}.jsonl"
    _write_session(
        transcript,
        SESSION_ID,
        [
            _user_text("m0", "please list files", "2026-01-01T00:00:01Z"),
            _message(
                "m1",
                "assistant",
                [
                    {"type": "thinking", "text": "I should run ls, not a real action"},
                    {
                        "type": "toolCall",
                        "id": "call-bash",
                        "name": "bash",
                        "arguments": {"command": "ls"},
                    },
                ],
                "2026-01-01T00:00:02Z",
            ),
            {
                "type": "message",
                "id": "m2",
                "timestamp": "2026-01-01T00:00:03Z",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call-bash",
                    "toolName": "bash",
                    "content": [{"type": "text", "text": "out"}],
                },
            },
            _message(
                "m3",
                "assistant",
                [{"type": "text", "text": "listed the directory"}],
                "2026-01-01T00:00:04Z",
            ),
            _message(
                "m4",
                "assistant",
                [
                    {
                        "type": "toolCall",
                        "id": "call-grep",
                        "name": "grep",
                        "arguments": {"pattern": "foo"},
                    }
                ],
                "2026-01-01T00:00:05Z",
            ),
            {
                "type": "message",
                "id": "m5",
                "timestamp": "2026-01-01T00:00:06Z",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call-grep",
                    "toolName": "grep",
                    "content": [{"type": "text", "text": "hits"}],
                },
            },
            {
                "type": "message",
                "id": "m6",
                "timestamp": "2026-01-01T00:00:07Z",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "no-pending",
                    "toolName": "bash",
                    "content": [{"type": "text", "text": "orphan-out"}],
                },
            },
            _message(
                "m7",
                "assistant",
                [{"type": "toolCall", "name": "read", "arguments": {"path": "/tmp/x"}}],
                "2026-01-01T00:00:08Z",
            ),
        ],
    )

    records = transcript_to_records(transcript)
    sequences = [record.sequence for record in records]
    assert sequences == list(range(1, len(records) + 1))
    assert len(set(sequences)) == len(sequences)
    assert {record.session_id for record in records} == {SESSION_ID}
    _assert_compile_preflight(records, participant="daemon-agent", segment=SESSION_ID)
    assert all(
        not (record.action.get("text") or "").startswith("I should run ls") for record in records
    )

    paired = [record for record in records if record.action.get("tool") in {"bash", "grep"}]
    assert [record.action["tool"] for record in paired] == ["bash", "grep"]
    assert paired[0].action["args"] == {"command": "ls"}
    assert paired[0].observation == {"text": "out"}
    assert paired[1].action["args"] == {"pattern": "foo"}
    assert paired[1].observation == {"text": "hits"}

    orphans = [record for record in records if record.action.get("tool") == "tool_result_orphan"]
    assert len(orphans) == 1
    assert orphans[0].observation == {"text": "orphan-out"}

    pending = [record for record in records if record.action.get("tool") == "read"]
    assert len(pending) == 1
    assert pending[0].action["args"] == {"path": "/tmp/x"}
    assert pending[0].observation is None

    texts = [record.action.get("text") for record in records if "text" in record.action]
    assert texts == ["please list files", "listed the directory"]

    _assert_to_dict_roundtrip(records, tmp_path / "ao-roundtrip.jsonl")


def test_cmd_export_success_zip(tmp_path: Path, capsys: Any) -> None:
    task = tmp_path / "agents" / "09879895-aaaa-4bbb-8ccc-dddddddddddd"
    sessions = task / ".builtin-sessions"
    prompt_one = "PROMPT_ONE unique alpha question for zip"
    prompt_two = "PROMPT_TWO unique beta question for zip"
    sid_a = "11111111-1111-1111-1111-111111111111"
    sid_b = "22222222-2222-2222-2222-222222222222"
    warm = [
        _user_text("u1", prompt_one, "2026-01-01T00:00:01Z"),
        _user_text("u2", prompt_two, "2026-01-01T00:00:02Z"),
    ]
    _write_session(sessions / f"{sid_a}.jsonl", sid_a, warm)
    _write_session(sessions / f"{sid_b}.jsonl", sid_b, warm)

    attempts = _write_jsonl(
        tmp_path / "attempts.jsonl",
        [
            {"run_id": "run-z", "episode_id": "ep-1", "status": "success", "work_dir": str(task)},
            {"run_id": "run-z", "episode_id": "ep-2", "status": "success", "work_dir": str(task)},
        ],
    )
    manifest = _write_jsonl(
        tmp_path / "manifest.jsonl",
        [
            {"episode_id": "ep-1", "turns": [{"prompt": prompt_one}]},
            {"episode_id": "ep-2", "turns": [{"prompt": prompt_two}]},
        ],
    )
    verdicts = _write_jsonl(
        tmp_path / "verdicts.jsonl",
        [
            {"run_id": "run-z", "episode_id": "ep-1", "session_id": None},
            {"run_id": "run-z", "episode_id": "ep-2", "session_id": None},
        ],
    )

    rc = cmd_export(_export_args(tmp_path, attempts, manifest, verdicts))
    captured = capsys.readouterr()
    assert rc == 0
    assert "task-zip" in captured.out
    rewritten = [
        json.loads(line)
        for line in verdicts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_episode = {row["episode_id"]: row["session_id"] for row in rewritten}
    assert by_episode == {"ep-1": sid_a, "ep-2": sid_b}
    _assert_exported_sessions(tmp_path / "sessions-out", captured.out, participant="task-agent")


def test_cmd_export_task_seq(tmp_path: Path, capsys: Any) -> None:
    task = tmp_path / "agents" / "09879895-aaaa-4bbb-8ccc-dddddddddddd"
    sessions = task / ".builtin-sessions"
    prompt_alpha = "PROMPT_ALPHA unique for sequential claim"
    prompt_beta = "PROMPT_BETA unique for sequential claim"
    sid_a = "11111111-1111-1111-1111-111111111111"
    sid_b = "22222222-2222-2222-2222-222222222222"
    _write_session(
        sessions / f"{sid_a}.jsonl", sid_a, [_user_text("u1", prompt_beta, "2026-01-01T00:00:01Z")]
    )
    _write_session(
        sessions / f"{sid_b}.jsonl", sid_b, [_user_text("u1", prompt_alpha, "2026-01-01T00:00:01Z")]
    )

    attempts = _write_jsonl(
        tmp_path / "attempts.jsonl",
        [
            {"run_id": "run-s", "episode_id": "ep-1", "status": "failed", "work_dir": str(task)},
            {"run_id": "run-s", "episode_id": "ep-2", "status": "failed", "work_dir": str(task)},
        ],
    )
    manifest = _write_jsonl(
        tmp_path / "manifest.jsonl",
        [
            {"episode_id": "ep-1", "turns": [{"prompt": prompt_alpha}]},
            {"episode_id": "ep-2", "turns": [{"prompt": prompt_beta}]},
        ],
    )
    verdicts = _write_jsonl(
        tmp_path / "verdicts.jsonl",
        [
            {"run_id": "run-s", "episode_id": "ep-1", "session_id": None},
            {"run_id": "run-s", "episode_id": "ep-2", "session_id": None},
        ],
    )

    rc = cmd_export(_export_args(tmp_path, attempts, manifest, verdicts))
    captured = capsys.readouterr()
    assert rc == 0
    assert "task-seq" in captured.out
    rewritten = [
        json.loads(line)
        for line in verdicts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_episode = {row["episode_id"]: row["session_id"] for row in rewritten}
    assert by_episode == {"ep-1": sid_b, "ep-2": sid_a}
    _assert_exported_sessions(tmp_path / "sessions-out", captured.out, participant="task-agent")


def test_cmd_export_memory_seq(tmp_path: Path, capsys: Any) -> None:
    agents = tmp_path / "agents"
    task = agents / "09879895-aaaa-4bbb-8ccc-dddddddddddd"
    task.mkdir(parents=True)
    memory = agents / "a4a9195c-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    prompt = "PROMPT_MEMORY unique fallback question"
    sid_mem = "33333333-3333-3333-3333-333333333333"
    _write_session(
        memory / ".builtin-sessions" / f"{sid_mem}.jsonl",
        sid_mem,
        [_user_text("u1", prompt, "2026-01-01T00:00:01Z")],
    )

    attempts = _write_jsonl(
        tmp_path / "attempts.jsonl",
        [{"run_id": "run-m", "episode_id": "ep-1", "status": "failed", "work_dir": str(task)}],
    )
    manifest = _write_jsonl(
        tmp_path / "manifest.jsonl",
        [{"episode_id": "ep-1", "turns": [{"prompt": prompt}]}],
    )
    verdicts = _write_jsonl(
        tmp_path / "verdicts.jsonl",
        [{"run_id": "run-m", "episode_id": "ep-1", "session_id": None}],
    )

    rc = cmd_export(_export_args(tmp_path, attempts, manifest, verdicts))
    captured = capsys.readouterr()
    assert rc == 0
    assert "memory-seq" in captured.out
    rewritten = [
        json.loads(line)
        for line in verdicts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rewritten[0]["session_id"] == sid_mem
    _assert_exported_sessions(tmp_path / "sessions-out", captured.out, participant="memory-agent")


def test_transcript_to_records_participant_passthrough(tmp_path: Path) -> None:
    transcript = tmp_path / f"{SESSION_ID}.jsonl"
    _write_session(
        transcript,
        SESSION_ID,
        [
            _user_text("m0", "please list files", "2026-01-01T00:00:01Z"),
            _message(
                "m1",
                "assistant",
                [
                    {
                        "type": "toolCall",
                        "id": "call-bash",
                        "name": "bash",
                        "arguments": {"command": "ls"},
                    }
                ],
                "2026-01-01T00:00:02Z",
            ),
            {
                "type": "message",
                "id": "m2",
                "timestamp": "2026-01-01T00:00:03Z",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call-bash",
                    "toolName": "bash",
                    "content": [{"type": "text", "text": "out"}],
                },
            },
        ],
    )

    for participant in ("agent-alpha", "agent-beta"):
        records = transcript_to_records(transcript, participant=participant)
        assert records
        assert {record.action["participant_id"] for record in records} == {participant}
        assert {record.action["segment_id"] for record in records} == {SESSION_ID}


def test_transcript_to_records_compile_preflight_ids(tmp_path: Path) -> None:
    """Converter output must satisfy the shared-id check compile_paper_graph runs."""
    transcript = tmp_path / f"{SESSION_ID}.jsonl"
    _write_session(
        transcript,
        SESSION_ID,
        [
            _user_text("m0", "please list files", "2026-01-01T00:00:01Z"),
            _message(
                "m1",
                "assistant",
                [
                    {"type": "thinking", "text": "skip me"},
                    {
                        "type": "toolCall",
                        "id": "call-bash",
                        "name": "bash",
                        "arguments": {"command": "ls"},
                    },
                ],
                "2026-01-01T00:00:02Z",
            ),
            {
                "type": "message",
                "id": "m2",
                "timestamp": "2026-01-01T00:00:03Z",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call-bash",
                    "toolName": "bash",
                    "content": [{"type": "text", "text": "out"}],
                },
            },
            _message(
                "m3",
                "assistant",
                [{"type": "text", "text": "listed the directory"}],
                "2026-01-01T00:00:04Z",
            ),
            {
                "type": "message",
                "id": "m4",
                "timestamp": "2026-01-01T00:00:05Z",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "no-pending",
                    "toolName": "bash",
                    "content": [{"type": "text", "text": "orphan-out"}],
                },
            },
        ],
    )

    records = transcript_to_records(transcript, participant="daemon-uuid-1")
    _assert_compile_preflight(records, participant="daemon-uuid-1", segment=SESSION_ID)
    assert all(record.action.get("tool") != "thinking" for record in records)
    assert any(record.action.get("tool") == "tool_result_orphan" for record in records)


def test_cmd_reverify_routes_detail_and_skips_unknown(tmp_path: Path, monkeypatch: Any) -> None:
    import benchmark.causal_memory.aig0045_real_material as real_material

    captured: list[tuple[str, dict[str, Any]]] = []

    def _fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        captured.append((url, payload))
        if payload["episode_id"] == "ir-1":
            return {
                "reward": 1,
                "graded": True,
                "error": None,
                "method": "browse-judge",
                "predicted": "yes",
                "evaluation_score": 0.91,
                "judge": "gemma",
                "noise": "drop-me",
            }
        return {
            "reward": 0,
            "graded": True,
            "error": None,
            "files_found": ["report.md"],
            "evaluation_score": 0.4,
            "judge": "glm",
            "noise": "drop-me",
        }

    monkeypatch.setattr(real_material, "_post_json", _fake_post)

    attempts = _write_jsonl(
        tmp_path / "attempts.jsonl",
        [
            {
                "run_id": "run-r",
                "episode_id": "ir-1",
                "domain": "information_retrieval",
                "final_output": "the answer is 42",
                "work_dir": "/should-not-use",
            },
            {
                "run_id": "run-r",
                "episode_id": "kw-1",
                "domain": "knowledge_work",
                "final_output": "should-not-send",
                "work_dir": "/ws/kw",
            },
            {
                "run_id": "run-r",
                "episode_id": "other-1",
                "domain": "code_generation",
                "final_output": "ignored",
                "work_dir": "/ws/other",
            },
        ],
    )
    out = tmp_path / "verdicts.jsonl"
    rc = cmd_reverify(
        argparse.Namespace(
            attempts=attempts,
            out=out,
            ir_port=9101,
            kw_port=9102,
            timeout=9.0,
        )
    )
    assert rc == 0
    assert len(captured) == 2

    ir_url, ir_payload = captured[0]
    assert urlparse(ir_url).port == 9101
    assert ir_payload == {"episode_id": "ir-1", "agent_response": "the answer is 42"}

    kw_url, kw_payload = captured[1]
    assert urlparse(kw_url).port == 9102
    assert kw_payload == {"episode_id": "kw-1", "workspace_dir": "/ws/kw"}

    rows = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert [row["episode_id"] for row in rows] == ["ir-1", "kw-1"]
    assert rows[0]["domain_kind"] == "browsecomp"
    assert rows[0]["detail"] == {
        "method": "browse-judge",
        "predicted": "yes",
        "evaluation_score": 0.91,
        "judge": "gemma",
    }
    assert "noise" not in rows[0]["detail"]
    assert rows[1]["domain_kind"] == "gdpval"
    assert rows[1]["detail"] == {
        "files_found": ["report.md"],
        "evaluation_score": 0.4,
        "judge": "glm",
    }
    assert "noise" not in rows[1]["detail"]


def test_interleaved_notices_and_parallel_tools_keep_action_order_and_ids(tmp_path: Path) -> None:
    transcript = tmp_path / f"{SESSION_ID}.jsonl"
    _write_session(
        transcript,
        SESSION_ID,
        [
            _user_text("u0", "original task", "2026-01-01T00:00:01Z"),
            _message(
                "a1",
                "assistant",
                [
                    {
                        "type": "toolCall",
                        "id": "call-a",
                        "name": "bash",
                        "arguments": {"command": "first"},
                    },
                    {
                        "type": "toolCall",
                        "id": "call-b",
                        "name": "bash",
                        "arguments": {"command": "second"},
                    },
                ],
                "2026-01-01T00:00:02Z",
            ),
            _user_text("u1", "inbox notice", "2026-01-01T00:00:03Z"),
            {
                "type": "message",
                "id": "r-b",
                "timestamp": "2026-01-01T00:00:04Z",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call-b",
                    "content": [{"type": "text", "text": "second result"}],
                },
            },
            {
                "type": "message",
                "id": "r-a",
                "timestamp": "2026-01-01T00:00:05Z",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call-a",
                    "content": [{"type": "text", "text": "first result"}],
                },
            },
        ],
    )
    rows = transcript_to_records(transcript, participant="task-agent")
    assert [r.sequence for r in rows] == [1, 2, 3, 4]
    assert [r.action.get("text", r.action.get("args", {}).get("command")) for r in rows] == [
        "original task",
        "first",
        "second",
        "inbox notice",
    ]
    assert rows[1].observation["text"] == "first result"
    assert rows[2].observation["text"] == "second result"
    assert rows[1].action["tool_call_id"] == "call-a"
    assert rows[2].action["tool_call_id"] == "call-b"


def test_explicit_unknown_tool_result_does_not_attach_to_other_call(tmp_path: Path) -> None:
    transcript = tmp_path / f"{SESSION_ID}.jsonl"
    _write_session(
        transcript,
        SESSION_ID,
        [
            _message(
                "a1",
                "assistant",
                [
                    {
                        "type": "toolCall",
                        "id": "known",
                        "name": "bash",
                        "arguments": {"command": "work"},
                    },
                ],
                "2026-01-01T00:00:01Z",
            ),
            {
                "type": "message",
                "id": "r1",
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "different",
                    "content": [{"type": "text", "text": "unrelated"}],
                },
            },
        ],
    )
    rows = transcript_to_records(transcript)
    assert rows[0].observation is None
    assert rows[1].action["tool"] == "tool_result_orphan"
    assert rows[1].observation["text"] == "unrelated"


def test_unknown_explicit_id_cannot_consume_an_idless_legacy_call(tmp_path: Path) -> None:
    transcript = tmp_path / f"{SESSION_ID}.jsonl"
    _write_session(
        transcript,
        SESSION_ID,
        [
            _message(
                "a",
                "assistant",
                [{"type": "toolCall", "name": "bash", "arguments": {}}],
                "2026-01-01T00:00:01Z",
            ),
            {
                "type": "message",
                "id": "r",
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "unknown",
                    "content": [{"type": "text", "text": "unrelated"}],
                },
            },
        ],
    )
    rows = transcript_to_records(transcript)
    assert rows[0].observation is None
    assert rows[1].action["tool"] == "tool_result_orphan"


def test_idless_result_with_mixed_candidates_remains_orphan(tmp_path: Path) -> None:
    transcript = tmp_path / f"{SESSION_ID}.jsonl"
    _write_session(
        transcript,
        SESSION_ID,
        [
            _message(
                "a",
                "assistant",
                [
                    {"type": "toolCall", "name": "bash", "arguments": {}},
                    {"type": "toolCall", "id": "known", "name": "read", "arguments": {}},
                ],
                "2026-01-01T00:00:01Z",
            ),
            {
                "type": "message",
                "id": "r",
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "role": "toolResult",
                    "content": [{"type": "text", "text": "ambiguous"}],
                },
            },
        ],
    )
    rows = transcript_to_records(transcript)
    assert rows[0].observation is None
    assert rows[1].observation is None
    assert rows[2].action["tool"] == "tool_result_orphan"
