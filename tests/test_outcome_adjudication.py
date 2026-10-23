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


def test_fresh_run_id_skips_leftover_run_dir(tmp_path):
    from openviking.session.outcome_adjudication import _fresh_run_id

    assert _fresh_run_id(tmp_path, "adj-x") == "adj-x"
    (tmp_path / "toolloop-runs" / "adj-x").mkdir(parents=True)
    assert _fresh_run_id(tmp_path, "adj-x") == "adj-x-r2"
    (tmp_path / "toolloop-runs" / "adj-x-r2").mkdir()
    assert _fresh_run_id(tmp_path, "adj-x") == "adj-x-r3"


def test_critic_bad_object_id_is_sent_back_and_retried():
    import asyncio
    import json
    import time
    from types import SimpleNamespace

    from openviking.session.aig_vlm_roles import VLMCritic

    bad = {"violations": [{"object_id": ["a", "b"], "kind": "granularity", "severity": "low",
                           "evidence_ao_refs": [0], "note": "overlap"}]}
    good = {"violations": [{"object_id": "a", "kind": "granularity", "severity": "low",
                            "evidence_ao_refs": [0], "note": "overlap with b"}]}
    replies = [json.dumps(bad), json.dumps(good)]
    seen = []

    class Handle:
        async def get_completion_async(self, **kw):
            seen.append(kw["messages"])
            await asyncio.sleep(0)
            return replies.pop(0)

    view = SimpleNamespace(call=lambda verb, args=None: {"row_count": 3})
    critic = VLMCritic(Handle(), time.monotonic() + 30)
    critic.bind_view(view)
    turn = SimpleNamespace(system_prompt="p", tools=(), structural_violations=(),
                           draft={"blocks": [{"block_id": "a"}, {"block_id": "b"}]})
    assert critic(turn) == good
    assert len(seen) == 2 and "Rejected output" in seen[1][-1]["content"]
    schema = critic._output_schema({"tools": [], "draft": turn.draft})
    item = schema["anyOf"][-1]["properties"]["violations"]["items"]
    assert item["properties"]["object_id"]["enum"] == ["a", "b"]
