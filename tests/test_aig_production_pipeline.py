"""Commit-facing four-role AIG pipeline keeps incomplete runs out of the native store."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from openviking.message import Message
from openviking.message.part import TextPart
from openviking.session.active_retrieval import corpus_for_session, retrieve_active
from openviking.session.aig_commit_wiring import commit_failure_signal, run_commit_four_roles
from openviking.session.aig_production_pipeline import AigRoleProviders, run_production_aig
from openviking.session.aig_tool_surface import AoSnapshot
from openviking.session.aig_toolloop_builder import FakeToolloopProvider
from openviking.session.aig_toolloop_critic import FakeCriticProvider
from openviking.session.aig_toolloop_reader import DeterministicReader
from openviking.session.aig_toolloop_refiner import FakeRefinerProvider
from openviking.session.ao_ledger import AORecord
from openviking.session.ao_ledger_recorder import AOLedgerRecorder
from openviking.session.causal_mode import is_session_causal
from openviking.session.extraction_write import NATIVE_EDGE_INDEX_FILENAME, NativeEdgeViewStore
from openviking.session.outcome_signal import OutcomeSignal
from openviking.session.session import SessionMeta
from openviking.storage.queuefs.session_commit_msg import SessionCommitMsg


def _snapshot() -> AoSnapshot:
    texts = ("repair the parser", "the wrong branch returned zero", "wrong output: zero")
    records = tuple(
        AORecord(
            ao_id=f"ao-{index}",
            session_id="session-a",
            sequence=index + 1,
            action={"participant_id": "worker", "segment_id": "segment-a", "text": text},
            observation={"text": text},
            skill_invocations=[],
            message_ref=None,
            created_at="2026-10-14T00:00:00Z",
            captured_state="live",
        )
        for index, text in enumerate(texts)
    )
    return AoSnapshot(
        records=records,
        watermark="wm-a",
        question="repair the parser",
        outcome=OutcomeSignal(signal="fail", source="model_inferred", raw={"status": "failed"}),
    )


def _script() -> list[tuple[str, dict]]:
    return [
        (
            "add_block",
            {
                "block_id": "task",
                "kind": "task",
                "fields": {
                    "content": "repair the parser",
                    "operative_brief": "repair",
                    "step_refs": [0],
                },
            },
        ),
        (
            "add_block",
            {
                "block_id": "worker",
                "kind": "agent",
                "fields": {
                    "agent_name": "worker",
                    "authorship": "executed",
                    "summary": "ran wrong branch",
                    "input": "repair",
                    "output": "the wrong branch returned zero",
                    "step_refs": [1],
                },
            },
        ),
        (
            "add_block",
            {
                "block_id": "end",
                "kind": "conclusion",
                "fields": {"content": "wrong output: zero", "step_refs": [2]},
            },
        ),
        (
            "add_inheritance_note",
            {
                "from": "task",
                "to": "worker",
                "inherited": "repair",
                "effect": "the wrong branch returned zero",
            },
        ),
        (
            "add_inheritance_note",
            {"from": "worker", "to": "end", "inherited": "zero", "effect": "wrong output: zero"},
        ),
        ("submit", {}),
    ]


def test_four_roles_freeze_native_view_and_reader_report(tmp_path: Path) -> None:
    result = run_production_aig(
        _snapshot(),
        root=tmp_path,
        task_run_id="task-a",
        frozen_at="2026-10-14T00:00:00Z",
        providers=AigRoleProviders(
            builder=FakeToolloopProvider(_script()),
            critic=FakeCriticProvider(),
            refiner=FakeRefinerProvider(),
            reader=DeterministicReader(),
        ),
    )
    assert result.status == "frozen"
    assert result.reader_report_hash
    revisions = NativeEdgeViewStore(tmp_path / NATIVE_EDGE_INDEX_FILENAME).revisions()
    assert len(revisions) == 1
    assert revisions[0].task_run_id == "task-a"
    assert result.report_path is not None and result.report_path.is_file()
    replayed = run_production_aig(
        _snapshot(),
        root=tmp_path,
        task_run_id="task-a",
        providers=AigRoleProviders(
            builder=FakeToolloopProvider(),
            critic=FakeCriticProvider(),
            refiner=FakeRefinerProvider(),
            reader=DeterministicReader(),
        ),
    )
    assert replayed.status == "frozen" and replayed.reason == "already_frozen"


def test_failed_critic_never_publishes_a_native_view(tmp_path: Path) -> None:
    def failed_critic(_turn):
        raise RuntimeError("model unavailable")

    result = run_production_aig(
        _snapshot(),
        root=tmp_path,
        task_run_id="task-a",
        frozen_at="2026-10-14T00:00:00Z",
        providers=AigRoleProviders(
            builder=FakeToolloopProvider(_script()),
            critic=failed_critic,
            refiner=FakeRefinerProvider(),
            reader=DeterministicReader(),
        ),
    )
    assert result.status == "fallback"
    assert not (tmp_path / NATIVE_EDGE_INDEX_FILENAME).exists()


def test_commit_without_verdict_still_freezes_searchable_aig(tmp_path: Path) -> None:
    result = run_production_aig(
        replace(_snapshot(), outcome=None),
        root=tmp_path,
        task_run_id="task-neutral",
        providers=AigRoleProviders(
            builder=FakeToolloopProvider(_script()),
            critic=FakeCriticProvider(),
            refiner=FakeRefinerProvider(),
            reader=DeterministicReader(),
        ),
    )
    assert result.status == "frozen"
    assert (
        NativeEdgeViewStore(tmp_path / NATIVE_EDGE_INDEX_FILENAME).revisions()[0].task_run_id
        == "task-neutral"
    )
    corpus = corpus_for_session(tmp_path, tmp_path, owner_label="owner-a")
    hits = retrieve_active(corpus, "parser", ("owner-a",))
    assert any(card.kind == "native_block" for card in hits.cards)


def test_public_channel_messages_enter_ao_snapshot_once(tmp_path: Path) -> None:
    recorder = AOLedgerRecorder(tmp_path, "session-a")
    message = Message(
        id="message-a",
        role="assistant",
        parts=[TextPart(text="agent reported a parser error")],
        peer_id="coforge__worker",
        source_message_ids=["channel-message-a"],
    )
    assert len(recorder.record_messages([message])) == 1
    assert recorder.record_messages([message]) == []
    row = recorder.ledger.records()[0]
    assert row.action["participant_id"] == "coforge__worker"
    assert row.observation["text"] == "agent reported a parser error"
    assert row.message_ref["source_message_ids"] == ["channel-message-a"]


def test_session_causal_mode_is_frozen_in_metadata() -> None:
    meta = SessionMeta.from_dict(SessionMeta(skill_trajectory_mode="causal").to_dict())
    assert is_session_causal(SimpleNamespace(_meta=meta), {"skill_trajectory_mode": "legacy"})


def test_commit_wiring_builds_without_evaluator_signal(tmp_path: Path, monkeypatch) -> None:
    rows = [record.to_dict() for record in _snapshot().records]
    monkeypatch.setattr(
        "openviking.session.aig_commit_wiring.session_directory", lambda _session: tmp_path
    )
    monkeypatch.setattr(
        "openviking.session.aig_commit_wiring._page_ao_records", lambda *_args: rows
    )
    monkeypatch.setattr(
        "openviking.session.aig_commit_wiring.providers_for_vlm",
        lambda _handle: AigRoleProviders(
            FakeToolloopProvider(_script()),
            FakeCriticProvider(),
            FakeRefinerProvider(),
            DeterministicReader(),
        ),
    )
    session = SimpleNamespace(
        session_id="session-a",
        _meta=SessionMeta(skill_trajectory_mode="causal"),
        outcome_evaluator_results=[],
        aig_question="repair the parser",
    )
    result = run_commit_four_roles(session, "task-a", object())
    assert result.status == "frozen"
    assert len(NativeEdgeViewStore(tmp_path / NATIVE_EDGE_INDEX_FILENAME).revisions()) == 1


def test_commit_signal_requires_explicit_unanimous_evaluator_failure() -> None:
    signal = {
        "binding_id": "binding-1",
        "signal_kind": "evaluator",
        "signal": "fail",
        "evidence_ref": "verifier:1",
    }
    assert commit_failure_signal(SimpleNamespace(index_outcome_status="failed")) is None
    assert (
        commit_failure_signal(SimpleNamespace(outcome_evaluator_results=[signal])).signal == "fail"
    )
    assert (
        commit_failure_signal(
            SimpleNamespace(
                outcome_evaluator_results=[
                    signal,
                    {**signal, "binding_id": "binding-2", "signal": "pass"},
                ]
            )
        )
        is None
    )


def test_queued_commit_preserves_verifier_result_and_question() -> None:
    signal = {
        "binding_id": "binding-1",
        "signal_kind": "test",
        "signal": "fail",
        "evidence_ref": "test:1",
    }
    message = SessionCommitMsg(
        task_id="task-a",
        session_id="session-a",
        session_uri="viking://x",
        archive_uri="viking://x/a",
        user={},
        outcome_evaluator_results=[signal],
        aig_question="repair parser",
    )
    restored = SessionCommitMsg.from_dict(message.to_dict())
    assert restored.outcome_evaluator_results == [signal]
    assert restored.aig_question == "repair parser"
