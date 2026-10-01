# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Contract parser, ledger-snapshot seam, and a fake extraction factory.

Contract and pipeline checks. They do not measure retrieval value and they
do not compare model extraction with deterministic extraction.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.service.session_service import SessionService
from openviking.service.task_tracker import set_task_tracker
from openviking.session.extraction_contract import (
    EXTRACTION_CONTRACT_VERSION,
    MAX_ITEMS,
    MAX_TOKENS_PER_CALL,
    VALUE_CLIP,
    ExtractionContractError,
    ExtractionLimitError,
    build_model_input,
    model_extraction_provider,
    parse_contract_response,
    render_contract_prompt,
)
from openviking.session.extraction_write import freeze_extracted_view
from openviking.session.influence_view import (
    DETERMINISTIC_FALLBACK_REASON,
    PURPOSE_POST_RUN_INDEX,
    SYNTHETIC_ROOT_SINK_REASON,
    ViewSupersedeIndex,
    build_minimal_fallback_view,
)
from openviking.session.retrieval_wiring import build_retrieval_facade
from openviking.session.session import Session
from openviking.storage.queuefs.session_commit_msg import SessionCommitMsg
from openviking_cli.session.user_id import UserIdentifier
from tests.test_extraction_write import _payload_for
from tests.test_trajectory_index_hook import (
    SESSION_ID,
    SESSION_URI,
    _DummyFS,
    _install_commit_fakes,
    _set_causal,
)

SESSION = "sess-contract"
VIEW = "view-contract"
TASK = "task-contract"
REVISION = "rev-contract"
WATERMARK = "ao-ledger:contract-test"
FROZEN = "2026-10-01T00:00:00.000Z"
_CLOCK = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _row(
    sequence: int,
    *,
    participant: str = "agent-a",
    segment: str = "seg-a",
    action: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ao_id": f"ao-{sequence}",
        "sequence": sequence,
        "participant_id": participant,
        "segment_id": segment,
        "session_id": SESSION,
        "content_hash": f"hash-{sequence}",
        "captured_state": "committed",
        "action": {} if action is None else action,
        "observation": {} if observation is None else observation,
    }


def _snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "session_id": SESSION,
        "task_run_id": TASK,
        "view_id": VIEW,
        "revision_id": REVISION,
        "purpose": PURPOSE_POST_RUN_INDEX,
        "snapshot_watermark": WATERMARK,
        "ao_records": [
            {
                "ao_id": row["ao_id"],
                "sequence": row["sequence"],
                "participant_id": row["participant_id"],
                "segment_id": row["segment_id"],
                "session_id": row["session_id"],
                "content_hash": row["content_hash"],
            }
            for row in rows
        ],
        "ledger_rows": [
            {
                "ao_id": row["ao_id"],
                "sequence": row["sequence"],
                "captured_state": row["captured_state"],
                "action": row["action"],
                "observation": row["observation"],
            }
            for row in rows
        ],
    }


def _block(sequence: int, *, role: str = "agent_action", kind: str = "tool_call") -> dict[str, Any]:
    return {
        "block_id": f"b-{sequence}",
        "role": role,
        "block_kind": kind,
        "participant_id": "agent-a",
        "segment_id": "seg-a",
        "ao_start_seq": sequence,
        "ao_end_seq": sequence,
        "summary": "step",
        "input": "in",
        "output": "out",
        "authorship": "agent-a",
        "granularity_reason": "stated by the model",
    }


def _evidence(sequence: int) -> dict[str, str]:
    return {"ao_id": f"ao-{sequence}", "evidence_content_hash": f"hash-{sequence}"}


def _happy_document(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    used = rows if rows is not None else [_row(1), _row(2)]
    sequences = [row["sequence"] for row in used]
    return {
        "schema_version": EXTRACTION_CONTRACT_VERSION,
        "blocks": [_block(sequence) for sequence in sequences],
        "claims": [
            {
                "claim_id": "c-1",
                "source_block_id": f"b-{sequences[0]}",
                "target_block_id": f"b-{sequences[1]}",
                "carried_artifact": "patch",
                "downstream_effect": "next step",
                "relation_type": "influence",
                "status": "hypothesis",
                "source_evidence": [_evidence(sequences[0])],
                "target_evidence": [_evidence(sequences[1])],
            }
        ]
        if len(sequences) >= 2
        else [],
        "subgraphs": [
            {
                "subgraph_id": "sg-1",
                "block_ids": [f"b-{sequence}" for sequence in sequences],
                "claim_ids": ["c-1"] if len(sequences) >= 2 else [],
                "failure_marked": True,
                "failure_evidence": [_evidence(sequences[-1])],
            }
        ],
        "coverage": {
            "total_ao_start_seq": min(sequences),
            "total_ao_end_seq": max(sequences),
            "inspected_ao_ids": [row["ao_id"] for row in used],
            "snapshot_watermark": WATERMARK,
        },
    }


def _happy() -> tuple[dict[str, Any], str]:
    rows = [_row(1), _row(2)]
    return _snapshot(rows), json.dumps(_happy_document(rows))


def _loads_case(snapshot: dict[str, Any], document: dict[str, Any]) -> tuple[dict[str, Any], str]:
    return snapshot, json.dumps(document)


def _case_unknown_ao() -> tuple[dict[str, Any], str]:
    snapshot, _text = _happy()
    document = _happy_document()
    document["claims"][0]["source_evidence"][0]["ao_id"] = "ao-missing"
    return _loads_case(snapshot, document)


def _case_bad_hash() -> tuple[dict[str, Any], str]:
    snapshot, _text = _happy()
    document = _happy_document()
    document["claims"][0]["source_evidence"][0]["evidence_content_hash"] = "hash-wrong"
    return _loads_case(snapshot, document)


def _case_bad_watermark() -> tuple[dict[str, Any], str]:
    snapshot, _text = _happy()
    document = _happy_document()
    document["coverage"]["snapshot_watermark"] = "ao-ledger:other"
    return _loads_case(snapshot, document)


def _case_foreign_inspected() -> tuple[dict[str, Any], str]:
    snapshot, _text = _happy()
    document = _happy_document()
    document["coverage"]["inspected_ao_ids"].append("ao-foreign")
    return _loads_case(snapshot, document)


def _case_range_outside() -> tuple[dict[str, Any], str]:
    snapshot, _text = _happy()
    document = _happy_document()
    document["blocks"][0]["ao_end_seq"] = 9
    return _loads_case(snapshot, document)


def _case_cross_participant() -> tuple[dict[str, Any], str]:
    rows = [_row(1, participant="agent-a"), _row(2, participant="agent-b")]
    document = _happy_document(rows)
    document["blocks"] = [
        {
            **_block(1),
            "ao_end_seq": 2,
            "participant_id": "agent-a",
        }
    ]
    document["claims"] = []
    document["subgraphs"] = [
        {
            "subgraph_id": "sg-1",
            "block_ids": ["b-1"],
            "claim_ids": [],
            "failure_marked": False,
            "failure_evidence": [],
        }
    ]
    return _loads_case(_snapshot(rows), document)


def _case_cross_segment() -> tuple[dict[str, Any], str]:
    rows = [_row(1, segment="seg-a"), _row(2, segment="seg-b")]
    document = _happy_document(rows)
    document["blocks"] = [{**_block(1), "ao_end_seq": 2, "segment_id": "seg-a"}]
    document["claims"] = []
    document["subgraphs"] = [
        {
            "subgraph_id": "sg-1",
            "block_ids": ["b-1"],
            "claim_ids": [],
            "failure_marked": False,
            "failure_evidence": [],
        }
    ]
    return _loads_case(_snapshot(rows), document)


def _case_not_contiguous() -> tuple[dict[str, Any], str]:
    rows = [_row(1), _row(3)]
    document = _happy_document(rows)
    document["blocks"] = [{**_block(1), "ao_end_seq": 3}]
    document["claims"] = []
    document["subgraphs"] = [
        {
            "subgraph_id": "sg-1",
            "block_ids": ["b-1"],
            "claim_ids": [],
            "failure_marked": False,
            "failure_evidence": [],
        }
    ]
    document["coverage"]["inspected_ao_ids"] = ["ao-1", "ao-3"]
    document["coverage"]["total_ao_end_seq"] = 3
    return _loads_case(_snapshot(rows), document)


def _case_overlap() -> tuple[dict[str, Any], str]:
    rows = [_row(1), _row(2), _row(3)]
    document = _happy_document(rows)
    document["blocks"] = [
        {**_block(1), "ao_end_seq": 2},
        {**_block(2), "ao_start_seq": 2, "ao_end_seq": 3, "block_id": "b-2"},
    ]
    document["claims"] = []
    document["subgraphs"] = []
    document["coverage"]["total_ao_end_seq"] = 3
    document["coverage"]["inspected_ao_ids"] = ["ao-1", "ao-2", "ao-3"]
    return _loads_case(_snapshot(rows), document)


def _case_incompatible_kind() -> tuple[dict[str, Any], str]:
    snapshot, _text = _happy()
    document = _happy_document()
    document["blocks"][0]["role"] = "task"
    document["blocks"][0]["block_kind"] = "tool_call"
    return _loads_case(snapshot, document)


def _case_unknown_claim_endpoint() -> tuple[dict[str, Any], str]:
    snapshot, _text = _happy()
    document = _happy_document()
    document["claims"][0]["target_block_id"] = "b-missing"
    return _loads_case(snapshot, document)


def _case_unknown_subgraph_block() -> tuple[dict[str, Any], str]:
    snapshot, _text = _happy()
    document = _happy_document()
    document["subgraphs"][0]["block_ids"].append("b-missing")
    return _loads_case(snapshot, document)


def _case_unknown_subgraph_claim() -> tuple[dict[str, Any], str]:
    snapshot, _text = _happy()
    document = _happy_document()
    document["subgraphs"][0]["claim_ids"].append("c-missing")
    return _loads_case(snapshot, document)


def _case_failure_without_evidence() -> tuple[dict[str, Any], str]:
    snapshot, _text = _happy()
    document = _happy_document()
    document["subgraphs"][0]["failure_evidence"] = []
    return _loads_case(snapshot, document)


def _case_evidence_without_failure() -> tuple[dict[str, Any], str]:
    snapshot, _text = _happy()
    document = _happy_document()
    document["subgraphs"][0]["failure_marked"] = False
    return _loads_case(snapshot, document)


def _case_too_many_members() -> tuple[dict[str, Any], str]:
    rows = [_row(sequence) for sequence in range(1, 14)]
    document = _happy_document([rows[0], rows[1]])
    document["blocks"] = [_block(sequence) for sequence in range(1, 14)]
    document["claims"] = []
    document["subgraphs"] = [
        {
            "subgraph_id": "sg-1",
            "block_ids": [f"b-{sequence}" for sequence in range(1, 14)],
            "claim_ids": [],
            "failure_marked": False,
            "failure_evidence": [],
        }
    ]
    document["coverage"] = {
        "total_ao_start_seq": 1,
        "total_ao_end_seq": 13,
        "inspected_ao_ids": [f"ao-{sequence}" for sequence in range(1, 14)],
        "snapshot_watermark": WATERMARK,
    }
    return _loads_case(_snapshot(rows), document)


def _case_duplicate_block_id() -> tuple[dict[str, Any], str]:
    snapshot, _text = _happy()
    document = _happy_document()
    document["blocks"][1]["block_id"] = "b-1"
    document["claims"] = []
    document["subgraphs"] = []
    return _loads_case(snapshot, document)


def _case_embed(secret: str) -> tuple[dict[str, Any], str]:
    snapshot, _text = _happy()
    document = _happy_document()
    document["blocks"][0]["block_id"] = f"id-{secret}"
    document["claims"][0]["source_block_id"] = f"id-{secret}"
    document["subgraphs"][0]["block_ids"][0] = f"id-{secret}"
    return _loads_case(snapshot, document)


def _case_reserved(reason: str) -> tuple[dict[str, Any], str]:
    snapshot, _text = _happy()
    document = _happy_document()
    document["blocks"][0]["granularity_reason"] = reason
    return _loads_case(snapshot, document)


def _case_extra_text() -> tuple[dict[str, Any], str]:
    snapshot, text = _happy()
    return snapshot, "here is the object\n" + text


def _case_fence_and_extra() -> tuple[dict[str, Any], str]:
    snapshot, text = _happy()
    return snapshot, f"```json\n{text}\n```\nthanks"


def _case_not_object() -> tuple[dict[str, Any], str]:
    snapshot, _text = _happy()
    return snapshot, "[1, 2, 3]"


def _case_extra_evidence_field() -> tuple[dict[str, Any], str]:
    snapshot, _text = _happy()
    document = _happy_document()
    document["claims"][0]["source_evidence"][0]["source_sequence"] = 99
    return _loads_case(snapshot, document)


_REJECTIONS = [
    ("unknown_ao", _case_unknown_ao),
    ("bad_hash", _case_bad_hash),
    ("bad_watermark", _case_bad_watermark),
    ("foreign_inspected", _case_foreign_inspected),
    ("range_outside", _case_range_outside),
    ("cross_participant", _case_cross_participant),
    ("cross_segment", _case_cross_segment),
    ("not_contiguous", _case_not_contiguous),
    ("overlap", _case_overlap),
    ("incompatible_kind", _case_incompatible_kind),
    ("unknown_claim_endpoint", _case_unknown_claim_endpoint),
    ("unknown_subgraph_block", _case_unknown_subgraph_block),
    ("unknown_subgraph_claim", _case_unknown_subgraph_claim),
    ("failure_without_evidence", _case_failure_without_evidence),
    ("evidence_without_failure", _case_evidence_without_failure),
    ("too_many_members", _case_too_many_members),
    ("duplicate_block_id", _case_duplicate_block_id),
    ("embed_session", lambda: _case_embed(SESSION)),
    ("embed_view", lambda: _case_embed(VIEW)),
    ("embed_task", lambda: _case_embed(TASK)),
    ("embed_revision", lambda: _case_embed(REVISION)),
    ("reserved_fallback", lambda: _case_reserved(DETERMINISTIC_FALLBACK_REASON)),
    ("reserved_sink", lambda: _case_reserved(SYNTHETIC_ROOT_SINK_REASON)),
    ("extra_text", _case_extra_text),
    ("fence_and_extra", _case_fence_and_extra),
    ("not_object", _case_not_object),
    ("extra_evidence_sequence", _case_extra_evidence_field),
]


def _seed(tmp_path: Path, snapshot: dict[str, Any]) -> tuple[Path, bytes, str]:
    session = {
        "session_id": snapshot["session_id"],
        "task_run_id": snapshot["task_run_id"],
        "purpose": PURPOSE_POST_RUN_INDEX,
        "view_id": snapshot["view_id"],
        "revision_id": snapshot["revision_id"],
        "snapshot_watermark": snapshot["snapshot_watermark"],
        "ao_records": snapshot["ao_records"],
    }
    revision = build_minimal_fallback_view(session, clock=lambda: _CLOCK)
    path = tmp_path / "views.json"
    ViewSupersedeIndex(path).register(revision)
    return path, path.read_bytes(), revision.revision_id


def _assert_store_unchanged(path: Path, blob: bytes, revision_id: str, session_id: str) -> None:
    assert path.read_bytes() == blob
    index = ViewSupersedeIndex(path)
    history = index.history(session_id, PURPOSE_POST_RUN_INDEX)
    latest = index.latest(session_id, PURPOSE_POST_RUN_INDEX)
    assert len(history) == 1
    assert latest is not None
    assert latest.revision_id == revision_id
    assert latest.construction_mode == "fallback"
    assert history[0].revision_id == revision_id


def _freeze_hash(snapshot: dict[str, Any], text: str) -> str:
    payload = parse_contract_response(text, build_model_input(snapshot))
    revision = freeze_extracted_view(
        session_id=SESSION,
        task_run_id=TASK,
        view_id=VIEW,
        revision_id=REVISION,
        frozen_at=FROZEN,
        blocks=payload.blocks,
        claims=payload.claims,
        coverage=payload.coverage,
        retrieval_subgraphs=payload.retrieval_subgraphs,
    )
    return revision.content_hash


def test_compliant_json_freezes_with_a_stable_content_hash() -> None:
    snapshot, text = _happy()
    first = _freeze_hash(snapshot, text)
    second = _freeze_hash(snapshot, text)
    fenced = _freeze_hash(snapshot, f"```json\n{text}\n```")
    assert first == second == fenced
    assert len(first) == 64


def test_prompt_imports_production_vocabularies() -> None:
    snapshot, _text = _happy()
    prompt = render_contract_prompt(build_model_input(snapshot))
    assert "agent_action" in prompt
    assert "tool_call" in prompt
    assert "influence" in prompt
    assert "hypothesis" in prompt
    assert DETERMINISTIC_FALLBACK_REASON in prompt
    assert "one JSON object" in prompt
    assert SESSION in prompt


@pytest.mark.parametrize(("name", "builder"), _REJECTIONS, ids=[item[0] for item in _REJECTIONS])
def test_rejection_writes_nothing(name: str, builder, tmp_path: Path) -> None:
    del name
    snapshot, text = builder()
    path, blob, revision_id = _seed(tmp_path, _happy()[0])
    with pytest.raises(ExtractionContractError):
        parse_contract_response(text, build_model_input(snapshot))
    _assert_store_unchanged(path, blob, revision_id, SESSION)


def test_unchanged_payload_passes_and_one_field_is_refused(tmp_path: Path) -> None:
    snapshot, text = _happy()
    path, blob, revision_id = _seed(tmp_path, snapshot)
    model_input = build_model_input(snapshot)
    assert _freeze_hash(snapshot, text)
    _assert_store_unchanged(path, blob, revision_id, SESSION)

    forged_id = json.loads(text)
    forged_id["claims"][0]["target_evidence"][0]["ao_id"] = "ao-forged"
    with pytest.raises(ExtractionContractError):
        parse_contract_response(json.dumps(forged_id), model_input)
    _assert_store_unchanged(path, blob, revision_id, SESSION)

    forged_hash = json.loads(text)
    forged_hash["claims"][0]["target_evidence"][0]["evidence_content_hash"] = "deadbeef"
    with pytest.raises(ExtractionContractError):
        parse_contract_response(json.dumps(forged_hash), model_input)
    _assert_store_unchanged(path, blob, revision_id, SESSION)


def test_sequence_mismatch_between_projection_and_ledger_is_refused(tmp_path: Path) -> None:
    snapshot, _text = _happy()
    path, blob, revision_id = _seed(tmp_path, snapshot)
    snapshot["ledger_rows"][0]["sequence"] = 9
    with pytest.raises(ExtractionContractError, match="sequence mismatch"):
        build_model_input(snapshot)
    _assert_store_unchanged(path, blob, revision_id, SESSION)


def test_projection_keeps_r8a_and_production_shapes() -> None:
    r8a_action = {
        "record_kind": "atom",
        "acl_labels": ["sb-skillsbench"],
        "segment_id": "seg-a",
        "participant_id": "agent-a",
        "transcript_type": "toolCall",
        "block_kind": "tool_call",
    }
    production_action = {"tool": "read", "tool_id": "t1", "arguments": {"path": "a.md"}}
    production_observation = {
        "kind": "text",
        "title": "note",
        "summary": ["line"],
        "structure": ["body"],
        "notable_items": ["item"],
        "sample": "excerpt",
    }
    rows = [
        _row(1, action=r8a_action, observation={"synopsis": "clipped"}),
        _row(2, action=production_action, observation=production_observation),
    ]
    model_input = build_model_input(_snapshot(rows))
    assert list(model_input["rows"][0]["action"]) == sorted(r8a_action)
    assert model_input["rows"][0]["observation"] == {"synopsis": "clipped"}
    assert list(model_input["rows"][1]["action"]) == sorted(production_action)
    assert model_input["rows"][1]["observation"]["summary"] == ["line"]


def test_value_and_list_limits_raise_instead_of_clipping() -> None:
    rows = [_row(1), _row(2)]
    rows[0]["observation"] = {"synopsis": "x" * (VALUE_CLIP + 1)}
    with pytest.raises(ExtractionLimitError, match="VALUE_CLIP"):
        build_model_input(_snapshot(rows))
    rows[0]["observation"] = {"synopsis": "x" * VALUE_CLIP}
    rows[0]["action"] = {"items": ["a"] * (MAX_ITEMS + 1)}
    with pytest.raises(ExtractionLimitError, match="MAX_ITEMS"):
        build_model_input(_snapshot(rows))


def test_ledger_rows_do_not_change_fallback_content_hash() -> None:
    snapshot, _text = _happy()
    bare = {
        "session_id": snapshot["session_id"],
        "task_run_id": snapshot["task_run_id"],
        "purpose": PURPOSE_POST_RUN_INDEX,
        "view_id": snapshot["view_id"],
        "revision_id": snapshot["revision_id"],
        "snapshot_watermark": snapshot["snapshot_watermark"],
        "ao_records": snapshot["ao_records"],
    }
    with_rows = dict(bare)
    with_rows["ledger_rows"] = snapshot["ledger_rows"]
    without = build_minimal_fallback_view(bare, clock=lambda: _CLOCK)
    with_ledger = build_minimal_fallback_view(with_rows, clock=lambda: _CLOCK)
    assert without.content_hash == with_ledger.content_hash


class _StubVLM:
    def __init__(
        self,
        texts: list[str],
        *,
        usage: dict[str, int] | None = None,
        finish_reason: str = "stop",
        error: BaseException | None = None,
    ) -> None:
        self.texts = texts
        self.usage = usage or {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
        self.finish_reason = finish_reason
        self.error = error
        self.calls = 0
        self.max_tokens_seen: list[int] = []

    async def get_completion_async(self, messages: list[dict[str, str]], max_tokens: int) -> Any:
        del messages
        self.calls += 1
        self.max_tokens_seen.append(max_tokens)
        if self.error is not None:
            raise self.error
        index = min(self.calls - 1, len(self.texts) - 1)
        return SimpleNamespace(
            content=self.texts[index],
            usage=dict(self.usage),
            finish_reason=self.finish_reason,
        )


def test_stub_handle_parses_accumulates_usage_and_repairs_once() -> None:
    snapshot, text = _happy()
    stub = _StubVLM(["not json", text])
    provider = model_extraction_provider(stub)(snapshot)
    payload = provider(SESSION, TASK, "unspecified")
    assert payload.blocks
    assert stub.calls == 2
    assert provider.repairs == 1
    assert provider.usage["total_tokens"] == 16
    assert stub.max_tokens_seen == [MAX_TOKENS_PER_CALL, MAX_TOKENS_PER_CALL]


def test_stub_handle_does_not_retry_a_non_parse_error() -> None:
    snapshot, _text = _happy()
    stub = _StubVLM(["{}"], error=RuntimeError("transport down"))
    provider = model_extraction_provider(stub)(snapshot)
    with pytest.raises(RuntimeError, match="transport down"):
        provider(SESSION, TASK, "unspecified")
    assert stub.calls == 1
    assert provider.repairs == 0


def test_token_cap_raises_and_a_missing_handle_fails() -> None:
    snapshot, text = _happy()
    stub = _StubVLM([text], usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 50})
    provider = model_extraction_provider(stub, max_total_tokens=1)(snapshot)
    with pytest.raises(ExtractionLimitError, match="MAX_TOTAL_TOKENS"):
        provider(SESSION, TASK, "unspecified")
    assert provider.usage["total_tokens"] == 50
    assert stub.calls == 1
    with pytest.raises(ExtractionLimitError, match="none was given"):
        model_extraction_provider(None)


def test_truncated_finish_reason_is_not_retried() -> None:
    snapshot, text = _happy()
    stub = _StubVLM([text, text], finish_reason="length")
    provider = model_extraction_provider(stub)(snapshot)
    with pytest.raises(ExtractionLimitError, match="truncated"):
        provider(SESSION, TASK, "unspecified")
    assert stub.calls == 1
    assert provider.truncated == 1


def test_input_limit_does_not_call_the_model() -> None:
    snapshot, _text = _happy()
    snapshot["ledger_rows"][0]["observation"] = {"synopsis": "y" * (VALUE_CLIP + 1)}
    stub = _StubVLM(["{}"])
    with pytest.raises(ExtractionLimitError, match="input limit"):
        model_extraction_provider(stub)(snapshot)
    assert stub.calls == 0


async def _commit_with_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, factory):
    from openviking.message.part import ToolPart

    _set_causal(monkeypatch, True)
    root = tmp_path / "session"
    root.mkdir()
    storage = _DummyFS(root, SESSION_URI)
    _install_commit_fakes(monkeypatch)
    monkeypatch.setattr(
        "openviking.session.session.get_openviking_config",
        lambda: SimpleNamespace(
            memory=SimpleNamespace(
                extraction_enabled=False,
                session_skill_extraction_enabled=False,
            )
        ),
    )

    async def _summary(messages, latest_archive_overview="", checkpoint_requests=None):
        del messages, latest_archive_overview, checkpoint_requests
        return "phase2 summary"

    service = SessionService(viking_fs=storage)
    service.set_extraction_provider_factory(factory)
    ctx = RequestContext(user=UserIdentifier("default", "default"), role=Role.ROOT)
    session = service.session(ctx, session_id=SESSION_ID, session_uri=SESSION_URI)
    assert isinstance(session, Session)
    assert session._extraction_provider_factory is factory
    monkeypatch.setattr(session, "_generate_archive_summary_async", _summary)
    await session.add_messages_async(
        [
            {
                "role": "assistant",
                "parts": [
                    ToolPart(
                        tool_id="t1",
                        tool_name="read_file",
                        tool_input={"path": "a.md"},
                        tool_output="tool body",
                        tool_status="completed",
                    )
                ],
            }
        ]
    )
    committed = await session.commit_async(keep_recent_count=0)
    phase1 = await session._read_phase1_meta(committed["archive_uri"])
    message = SessionCommitMsg.from_dict(phase1["queue_message"])
    await session.resume_queued_commit(message)
    return session, committed


@pytest.mark.asyncio
async def test_factory_lands_adaptive_and_keeps_fallback_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []

    def factory(snapshot):
        seen.append(dict(snapshot))

        def provider(session_id: str, task_run_id: str, outcome_status: str):
            del task_run_id, outcome_status
            return _payload_for(session_id)

        return provider

    try:
        session, committed = await _commit_with_factory(tmp_path, monkeypatch, factory)
        task_id = committed["task_id"]
        fallback_id = f"post-run:{SESSION_ID}:{task_id}"
        adaptive_id = f"{fallback_id}:adaptive"
        index = ViewSupersedeIndex(session._viking_fs.root / "views.json")
        history = index.history(SESSION_ID, PURPOSE_POST_RUN_INDEX)
        latest = index.latest(SESSION_ID, PURPOSE_POST_RUN_INDEX)
        assert [item.revision_id for item in history] == [fallback_id, adaptive_id]
        assert latest is not None
        assert latest.revision_id == adaptive_id
        assert latest.construction_mode == "adaptive"
        assert history[0].construction_mode == "fallback"
        facade = build_retrieval_facade(
            session=session, ctx=session.ctx, viking_fs=session._viking_fs
        )
        loaded_adaptive = facade.retrieval_subgraph._revision(adaptive_id)
        loaded_fallback = facade.retrieval_subgraph._revision(fallback_id)
        assert loaded_adaptive.revision_id == adaptive_id
        assert loaded_fallback.revision_id == fallback_id
        assert seen and "ledger_rows" in seen[0] and "ao_records" in seen[0]
        assert len(seen[0]["ledger_rows"]) == len(seen[0]["ao_records"])
    finally:
        set_task_tracker(None)


@pytest.mark.asyncio
async def test_factory_contract_error_leaves_fallback_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def factory(snapshot):
        del snapshot
        raise ExtractionContractError("refused")

    try:
        session, committed = await _commit_with_factory(tmp_path, monkeypatch, factory)
        task_id = committed["task_id"]
        fallback_id = f"post-run:{SESSION_ID}:{task_id}"
        index = ViewSupersedeIndex(session._viking_fs.root / "views.json")
        history = index.history(SESSION_ID, PURPOSE_POST_RUN_INDEX)
        latest = index.latest(SESSION_ID, PURPOSE_POST_RUN_INDEX)
        assert len(history) == 1
        assert history[0].revision_id == fallback_id
        assert history[0].construction_mode == "fallback"
        assert latest is not None
        assert latest.revision_id == fallback_id
        facade = build_retrieval_facade(
            session=session, ctx=session.ctx, viking_fs=session._viking_fs
        )
        loaded = facade.retrieval_subgraph._revision(fallback_id)
        assert loaded.revision_id == fallback_id
    finally:
        set_task_tracker(None)
