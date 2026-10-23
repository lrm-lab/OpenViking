"""Outcome adjudication: attach a late verdict to a committed session and rebuild its AIG."""

import asyncio
from types import SimpleNamespace

import pytest

from openviking.session import outcome_adjudication as adj
from openviking.session.aig_production_pipeline import ProductionAigResult

RESULT = {
    "binding_id": "eval:ep-1",
    "signal_kind": "evaluator",
    "signal": "pass",
    "evidence_ref": "sidecar-verify:ep-1",
    "reward": 0.8,
    "feedback": "solid",
}


class _Session(SimpleNamespace):
    async def _get_vlm_config(self):
        return object()


def _session(tmp_path):
    return _Session(session_id="s1", outcome_evaluator_results=[], aig_question=None, root=tmp_path)


def _patch(monkeypatch, tmp_path, calls, result):
    monkeypatch.setattr(adj, "session_directory", lambda session: tmp_path)
    monkeypatch.setattr(adj, "is_session_causal", lambda session: True)

    def fake(session, task_id, model):
        calls.append((task_id, list(session.outcome_evaluator_results), session.aig_question))
        return result

    monkeypatch.setattr(adj, "run_commit_four_roles", fake)


def test_adjudication_rebuilds_with_the_outcome_and_is_idempotent(monkeypatch, tmp_path):
    calls = []
    _patch(monkeypatch, tmp_path, calls, ProductionAigResult("frozen", "complete", "rev-1", "h", None, ("card-1",)))
    session = _session(tmp_path)
    first = asyncio.run(adj.adjudicate_session(session, [RESULT], aig_question="q?"))
    assert first["status"] == "frozen" and first["card_ids"] == ["card-1"]
    assert first["signal"] == "pass" and first["reward"] == 0.8
    task_id, results, question = calls[0]
    assert task_id == adj.adjudication_run_id([RESULT]) and results == [RESULT] and question == "q?"
    again = asyncio.run(adj.adjudicate_session(session, [RESULT]))
    assert again["duplicate"] is True and len(calls) == 1
    other = dict(RESULT, signal="fail", reward=0.2)
    asyncio.run(adj.adjudicate_session(session, [other]))
    assert len(calls) == 2 and calls[1][0] != task_id


def test_fallback_is_not_recorded_so_it_can_be_retried(monkeypatch, tmp_path):
    calls = []
    _patch(monkeypatch, tmp_path, calls, ProductionAigResult("fallback", "invalid_critic_output"))
    session = _session(tmp_path)
    assert asyncio.run(adj.adjudicate_session(session, [RESULT]))["status"] == "fallback"
    asyncio.run(adj.adjudicate_session(session, [RESULT]))
    assert len(calls) == 2


@pytest.mark.parametrize(
    "results",
    [[], [dict(RESULT, signal="pass"), dict(RESULT, binding_id="eval:2", signal="fail")], [{"signal": "pass"}]],
)
def test_bad_or_ambiguous_results_are_refused(monkeypatch, tmp_path, results):
    _patch(monkeypatch, tmp_path, [], ProductionAigResult("frozen", "complete"))
    with pytest.raises(ValueError):
        asyncio.run(adj.adjudicate_session(_session(tmp_path), results))
