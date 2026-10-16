import asyncio
import json
from types import SimpleNamespace

import pytest

from openviking.session import explicit_diagnosis as module
from openviking.session.aig_production_pipeline import AigRoleProviders
from openviking.session.aig_toolloop_builder import FakeToolloopProvider
from openviking.session.aig_toolloop_critic import FakeCriticProvider
from openviking.session.aig_toolloop_reader import DeterministicReader
from openviking.session.aig_toolloop_refiner import FakeRefinerProvider
from tests.test_aig_production_pipeline import _script, _snapshot


def session(tmp_path):
    (tmp_path / "ao-ledger.jsonl").write_text(
        "".join(json.dumps(row.to_dict()) + "\n" for row in _snapshot().records)
    )
    writes = []

    async def write(uri, content, **kwargs):
        writes.append((uri, json.loads(content)))

    async def model():
        return object()

    return SimpleNamespace(
        session_id="session-a",
        _meta=SimpleNamespace(skill_trajectory_mode="causal"),
        _session_uri="viking://user/worker/sessions/session-a",
        ctx=None,
        _viking_fs=SimpleNamespace(_uri_to_path=lambda *a, **k: str(tmp_path), write_file=write),
        _get_vlm_config=model,
        writes=writes,
    )


@pytest.mark.asyncio
async def test_explicit_request_runs_real_four_role_core_and_replays_without_reconstruction(
    tmp_path, monkeypatch
):
    owner = session(tmp_path)
    calls = []

    def providers(model):
        calls.append(model)
        return AigRoleProviders(
            FakeToolloopProvider(_script()),
            FakeCriticProvider(),
            FakeRefinerProvider(),
            DeterministicReader(),
        )

    monkeypatch.setattr(module, "providers_for_vlm", providers)
    started = await module.request_diagnosis(owner, "request-a", "repair the parser")
    assert started["status"] == "running"
    same = await module.request_diagnosis(owner, "request-a", "repair the parser")
    assert same["watermark"] == started["watermark"]
    await asyncio.gather(*list(module._tasks))
    result = module.read_diagnosis(owner, "request-a")
    assert result["status"] == "completed"
    assert result["report"]["report"]["decisive_ao_refs"]
    assert result["report_uri"].endswith("/diagnoses/request-a/report.json")
    assert "snapshot" not in result
    assert len(owner.writes) == 1
    assert (await module.request_diagnosis(owner, "request-a", "repair the parser")) == result
    assert len(calls) == 1
    with pytest.raises(ValueError, match="conflict"):
        await module.request_diagnosis(owner, "request-a", "different question")


@pytest.mark.asyncio
async def test_incomplete_critic_reports_failure_without_publishing_report(tmp_path, monkeypatch):
    owner = session(tmp_path)
    critic = FakeCriticProvider()
    critic.failures = ["provider failed"]
    monkeypatch.setattr(
        module,
        "providers_for_vlm",
        lambda _: AigRoleProviders(
            FakeToolloopProvider(_script()), critic, FakeRefinerProvider(), DeterministicReader()
        ),
    )
    await module.request_diagnosis(owner, "request-failed", "repair the parser")
    await asyncio.gather(*list(module._tasks))
    assert module.read_diagnosis(owner, "request-failed")["status"] == "failed"
    assert owner.writes == []


@pytest.mark.asyncio
async def test_expired_or_interrupted_attempt_is_not_restarted(tmp_path, monkeypatch):
    owner = session(tmp_path)
    gate = asyncio.Event()

    async def blocked():
        await gate.wait()

    owner._get_vlm_config = blocked
    await module.request_diagnosis(owner, "interrupted", "repair the parser")
    pending = list(module._tasks)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    monkeypatch.setattr(module.time, "time", lambda: 99999999999)
    assert module.read_diagnosis(owner, "interrupted")["reason"] == "diagnosis_timeout"
    assert (await module.request_diagnosis(owner, "interrupted", "repair the parser"))[
        "status"
    ] == "failed"
    assert not module._tasks


@pytest.mark.asyncio
async def test_legacy_and_path_traversal_requests_are_rejected_before_start(tmp_path):
    owner = session(tmp_path)
    with pytest.raises(ValueError, match="request_id"):
        await module.request_diagnosis(owner, "../outside", "repair")
    owner._meta.skill_trajectory_mode = "legacy"
    with pytest.raises(ValueError, match="causal_session_required"):
        await module.request_diagnosis(owner, "valid", "repair")
    assert not (tmp_path / "explicit-diagnoses").exists()


@pytest.mark.asyncio
async def test_model_concurrency_is_bounded_before_a_fourth_snapshot_is_created(tmp_path):
    owner = session(tmp_path)
    gate = asyncio.Event()

    async def blocked():
        await gate.wait()

    owner._get_vlm_config = blocked
    try:
        for index in range(3):
            await module.request_diagnosis(owner, f"request-{index}", "repair the parser")
        with pytest.raises(ValueError, match="capacity_exhausted"):
            await module.request_diagnosis(owner, "request-fourth", "repair the parser")
        assert not (tmp_path / "explicit-diagnoses" / "request-fourth.json").exists()
    finally:
        pending = list(module._tasks)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
