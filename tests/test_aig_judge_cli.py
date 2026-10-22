# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Judge CLI: JSON in, strict verdict out. The chat endpoint is a scripted fake."""

from __future__ import annotations

import io
import itertools
import json

from openviking.session.aig_judge_cli import main
from openviking.session.aig_toolloop_judge import ROLES
from openviking.session.aig_toolloop_reader import load_frozen_paper_graph
from tests.test_aig_toolloop_reader import _compiled

_RUNS = itertools.count()
ENV = {"JUDGE_API_KEY": "k-test", "JUDGE_BASE_URL": "http://gw/v1", "JUDGE_MODEL": "m"}
VERDICT = {
    "status": "scored",
    "score": 0.4,
    "used": True,
    "evidence_block_ids": ["agent-b"],
    "decisive_ao_refs": [1],
    "reason": "reused",
}


def _request(tmp_path):
    tmp_path = tmp_path / f"run{next(_RUNS)}"
    tmp_path.mkdir()
    _surface, snapshot, compiled = _compiled(tmp_path)
    graph = load_frozen_paper_graph(compiled.revision, snapshot, sidecar_path=compiled.sidecar_path)
    return {
        "ao_records": [record.to_dict() for record in snapshot.records],
        "watermark": snapshot.watermark,
        "question": snapshot.question,
        "graph": {"blocks": graph["blocks"], "edges": graph["edges"]},
        "role_products": {role: {"evidence": role} for role in ROLES},
        "response_text": "candidate action text",
    }


class _Chat:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, request, timeout):
        self.seen.append(json.loads(request.data.decode()))
        content = self.replies.pop(0)

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        body = {"choices": [{"message": {"content": content}}]}
        return Response(json.dumps(body).encode())


def _run(request, chat, env=ENV):
    out, err = io.StringIO(), io.StringIO()
    code = main(
        stdin=io.StringIO(json.dumps(request)),
        stdout=out,
        stderr=err,
        environ=env,
        urlopen=chat,
    )
    return code, out.getvalue(), err.getvalue()


def test_scored_verdict_after_reading_the_cited_row(tmp_path):
    chat = _Chat('thinking {"tool": "read_step", "args": {"index": 1}}', json.dumps(VERDICT))
    code, out, err = _run(_request(tmp_path), chat)
    assert (code, err) == (0, "")
    result = json.loads(out)
    assert result["status"] == "scored" and result["score"] == 0.4
    assert result["roles_completed"] == list(ROLES)
    assert "ao:1" in result["evidence_refs"] and "block:agent-b" in result["evidence_refs"]
    assert "k-test" not in out and "candidate action text" in json.dumps(chat.seen[0])


def test_citing_an_unread_row_fails_closed(tmp_path):
    code, out, err = _run(_request(tmp_path), _Chat(json.dumps(VERDICT)))
    assert code == 2 and out == "" and err.strip() != ""


def test_unscored_is_exit_three_and_no_number(tmp_path):
    chat = _Chat(json.dumps({"status": "unscored", "reason": "not enough evidence"}))
    code, out, err = _run(_request(tmp_path), chat)
    assert (code, out, err.strip()) == (3, "", "unscored")


def test_missing_key_and_missing_roles_fail_closed(tmp_path):
    assert _run(_request(tmp_path), _Chat(), env={})[0] == 2
    request = _request(tmp_path)
    request["role_products"].pop("reader")
    code, out, err = _run(request, _Chat())
    assert (code, out, err.strip()) == (2, "", "roles_incomplete")


def test_provider_failure_never_leaks_key_or_body(tmp_path):
    def boom(request, timeout):
        raise OSError("k-test leaked")

    code, out, err = _run(_request(tmp_path), boom)
    assert code == 2 and out == "" and "k-test" not in err
