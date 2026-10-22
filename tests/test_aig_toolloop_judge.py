# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Judge: strict verdicts over the four roles' frozen graph. No model is called."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from openviking.session.aig_toolloop_judge import (
    JUDGE_VERBS,
    ROLES,
    FakeJudgeProvider,
    JudgeError,
    JudgeInvocation,
    judge_catalog,
    run_judge,
    to_external_result,
)
from openviking.session.aig_toolloop_reader import READER_VERBS, load_frozen_paper_graph
from tests.test_aig_toolloop_reader import _compiled

_RUNS = itertools.count()
PRODUCTS = {role: {"evidence": f"{role} done"} for role in ROLES}


def _setup(tmp_path: Path):
    # Each call builds its own run directory; surfaces are created exclusively.
    tmp_path = tmp_path / f"run{next(_RUNS)}"
    tmp_path.mkdir()
    _surface, snapshot, compiled = _compiled(tmp_path)
    graph = load_frozen_paper_graph(compiled.revision, snapshot, sidecar_path=compiled.sidecar_path)
    return snapshot, graph


def _read(index: int) -> JudgeInvocation:
    return JudgeInvocation("tool", "read_step", {"index": index})


def _verdict(**fields) -> JudgeInvocation:
    body = {
        "status": "scored",
        "score": 0.4,
        "used": True,
        "evidence_block_ids": ["agent-b"],
        "decisive_ao_refs": [1],
        "reason": "the script was reused",
    }
    body.update(fields)
    return JudgeInvocation("verdict", verdict=body)


def _run(tmp_path, *script, products=None, max_calls=12):
    snapshot, graph = _setup(tmp_path)
    provider = FakeJudgeProvider(script)
    result = run_judge(
        snapshot,
        graph,
        provider,
        root=tmp_path / "judge",
        run_id="run-judge",
        role_products=PRODUCTS if products is None else products,
        max_calls=max_calls,
    )
    return result, provider


def test_judge_scores_after_reading_the_decisive_row(tmp_path):
    result, provider = _run(tmp_path, JudgeInvocation("tool", "get_graph"), _read(1), _verdict())
    assert result.verdict.score == 0.4 and result.verdict.used is True
    assert result.read_ao_refs == (1,) and result.tool_calls == 2
    # The Judge is shown all four role products on every turn.
    assert set(provider.turns[0].role_products) == set(ROLES)


def test_judge_verbs_are_a_read_only_subset_of_the_reader(tmp_path):
    assert JUDGE_VERBS < READER_VERBS
    assert {name for name, _ in judge_catalog()} == JUDGE_VERBS
    assert "get_question" not in JUDGE_VERBS


def test_build_and_non_judge_verbs_are_refused(tmp_path):
    with pytest.raises(JudgeError) as caught:
        _run(tmp_path, JudgeInvocation("tool", "add_block", {}))
    assert caught.value.code == "build_verb_forbidden"
    with pytest.raises(JudgeError) as caught:
        _run(tmp_path, JudgeInvocation("tool", "get_question", {}))
    assert caught.value.code == "verb_not_in_role"


def test_verified_non_use_is_neutral_zero(tmp_path):
    result, _ = _run(tmp_path, _read(1), _verdict(score=0, used=False))
    assert result.verdict.score == 0 and result.verdict.used is False
    with pytest.raises(JudgeError) as caught:
        _run(tmp_path, _read(1), _verdict(score=0.3, used=False))
    assert caught.value.code == "invalid_score"


def test_a_real_negative_score_is_kept(tmp_path):
    result, _ = _run(tmp_path, _read(1), _verdict(score=-1))
    assert result.verdict.score == -1.0


@pytest.mark.parametrize("bad", [1.5, -1.01, float("nan"), float("inf"), True, "0.4", None])
def test_out_of_range_or_non_numeric_scores_are_rejected_not_clipped(tmp_path, bad):
    with pytest.raises(JudgeError) as caught:
        _run(tmp_path, _read(1), _verdict(score=bad))
    assert caught.value.code == "invalid_score"


def test_unscored_is_not_a_number_and_cannot_become_an_external_result(tmp_path):
    result, _ = _run(
        tmp_path,
        JudgeInvocation("verdict", verdict={"status": "unscored", "reason": "no evidence"}),
    )
    assert result.verdict.status == "unscored" and result.verdict.score is None
    with pytest.raises(JudgeError) as caught:
        to_external_result(result.verdict, {})
    assert caught.value.code == "unscored"


def test_unscored_with_a_score_is_invalid(tmp_path):
    with pytest.raises(JudgeError) as caught:
        _run(
            tmp_path,
            JudgeInvocation("verdict", verdict={"status": "unscored", "reason": "x", "score": 0}),
        )
    assert caught.value.code == "invalid_verdict"


def test_cited_rows_must_have_been_read(tmp_path):
    with pytest.raises(JudgeError) as caught:
        _run(tmp_path, _verdict(decisive_ao_refs=[1]))
    assert caught.value.code == "unread_decisive_refs"
    with pytest.raises(JudgeError) as caught:
        _run(tmp_path, JudgeInvocation("tool", "search", {"pattern": "script"}), _verdict())
    assert caught.value.code == "unread_decisive_refs"


def test_cited_blocks_must_exist_in_the_graph(tmp_path):
    with pytest.raises(JudgeError) as caught:
        _run(tmp_path, _read(1), _verdict(evidence_block_ids=["ghost"]))
    assert caught.value.code == "invalid_evidence"


def test_judge_is_refused_unless_all_four_roles_produced_output(tmp_path):
    partial = {role: PRODUCTS[role] for role in ROLES[:3]}
    with pytest.raises(JudgeError) as caught:
        _run(tmp_path, _read(1), _verdict(), products=partial)
    assert caught.value.code == "roles_incomplete"


def test_tool_budget_is_bounded(tmp_path):
    with pytest.raises(JudgeError) as caught:
        _run(tmp_path, _read(0), _read(1), _read(2), max_calls=2)
    assert caught.value.code == "budget_exhausted"


def test_a_bad_tool_call_is_recorded_and_the_loop_continues(tmp_path):
    result, _ = _run(tmp_path, _read(99), _read(1), _verdict())
    assert result.reads[0]["ok"] is False and result.reads[0]["error_code"]
    assert result.verdict.score == 0.4


def test_scored_verdict_maps_to_the_areal_contract_fields(tmp_path):
    result, _ = _run(tmp_path, _read(1), _verdict())
    mapped = to_external_result(result.verdict, {})
    assert mapped == {
        "status": "scored",
        "score": 0.4,
        "used": True,
        "evidence_refs": ["block:agent-b", "ao:1"],
        "roles_completed": ["builder", "critic", "refiner", "reader"],
    }
