# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Bundle CLI: four roles finish, then the Judge input is written; else nothing."""

from __future__ import annotations

import io
import json

from openviking.session.aig_bundle_cli import main
from openviking.session.aig_judge_cli import judge_request
from openviking.session.aig_production_pipeline import AigRoleProviders
from openviking.session.aig_toolloop_builder import FakeToolloopProvider
from openviking.session.aig_toolloop_critic import FakeCriticProvider
from openviking.session.aig_toolloop_judge import ROLES, FakeJudgeProvider, JudgeInvocation
from openviking.session.aig_toolloop_reader import DeterministicReader
from openviking.session.aig_toolloop_refiner import FakeRefinerProvider
from tests.test_aig_production_pipeline import _script, _snapshot


def _request(tmp_path, **extra):
    snap = _snapshot()
    body = {
        "ao_records": [row.to_dict() for row in snap.records],
        "watermark": snap.watermark,
        "question": snap.question,
        "task_run_id": "cand-1",
        "out": str(tmp_path / "bundle.json"),
    }
    body.update(extra)
    return body


def _providers(builder=None):
    return AigRoleProviders(
        builder=builder or FakeToolloopProvider(_script()),
        critic=FakeCriticProvider(),
        refiner=FakeRefinerProvider(),
        reader=DeterministicReader(),
    )


def _run(request, providers):
    out, err = io.StringIO(), io.StringIO()
    code = main(stdin=io.StringIO(json.dumps(request)), stdout=out, stderr=err, providers=providers)
    return code, out.getvalue(), err.getvalue()


def test_bundle_has_graph_and_all_four_role_products_and_is_judgeable(tmp_path):
    request = _request(tmp_path)
    code, out, err = _run(request, _providers())
    assert (code, err) == (0, "")
    assert json.loads(out) == {"judge_bundle": request["out"]}
    bundle = json.loads((tmp_path / "bundle.json").read_text())
    assert set(bundle["role_products"]) == set(ROLES)
    assert bundle["graph"]["blocks"] and bundle["graph"]["edges"]
    (tmp_path / "judge").mkdir()
    judged = judge_request(
        dict(bundle, response_text="candidate"),
        FakeJudgeProvider(
            [
                JudgeInvocation("tool", "read_step", {"index": 1}),
                JudgeInvocation(
                    "verdict",
                    verdict={
                        "status": "scored",
                        "score": -0.3,
                        "used": True,
                        "evidence_block_ids": ["worker"],
                        "decisive_ao_refs": [1],
                        "reason": "wrong branch",
                    },
                ),
            ]
        ),
        root=tmp_path / "judge",
    )
    assert judged["score"] == -0.3 and judged["roles_completed"] == list(ROLES)


def test_incomplete_builder_writes_no_bundle(tmp_path):
    code, out, err = _run(_request(tmp_path), _providers(FakeToolloopProvider([])))
    assert code == 2 and out == ""
    assert err.startswith("roles_incomplete_")
    assert not (tmp_path / "bundle.json").exists()


def test_missing_key_and_bad_request_fail_closed(tmp_path):
    out, err = io.StringIO(), io.StringIO()
    code = main(
        stdin=io.StringIO(json.dumps(_request(tmp_path))), stdout=out, stderr=err, environ={}
    )
    assert (code, out.getvalue(), err.getvalue().strip()) == (2, "", "missing_api_key")
    assert _run({"out": str(tmp_path / "x")}, _providers())[0] == 2
