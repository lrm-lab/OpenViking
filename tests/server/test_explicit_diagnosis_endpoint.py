import json
import time
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from openviking.server.auth import get_session_request_context
from openviking.server.routers import sessions as routes
from openviking.session import explicit_diagnosis as jobs
from openviking.session.aig_production_pipeline import AigRoleProviders
from openviking.session.aig_toolloop_builder import FakeToolloopProvider
from openviking.session.aig_toolloop_critic import FakeCriticProvider
from openviking.session.aig_toolloop_reader import DeterministicReader
from openviking.session.aig_toolloop_refiner import FakeRefinerProvider
from tests.server.test_coordinator_endpoint import _ctx, _session
from tests.test_aig_production_pipeline import _script, _snapshot


def test_http_diagnosis_uses_real_four_roles_and_keeps_snapshot_private(tmp_path, monkeypatch):
    owner = _session(tmp_path, session_id="session-a")
    owner._meta = SimpleNamespace(skill_trajectory_mode="causal", created_by_user_id="carol")
    (tmp_path / "ao-ledger.jsonl").write_text(
        "".join(json.dumps(row.to_dict()) + "\n" for row in _snapshot().records)
    )

    async def load():
        pass

    async def model():
        return object()

    async def write(*args, **kwargs):
        pass

    owner.load = load
    owner._get_vlm_config = model
    owner._viking_fs.write_file = write
    monkeypatch.setattr(
        routes,
        "get_service",
        lambda: SimpleNamespace(sessions=SimpleNamespace(session=lambda *a, **k: owner)),
    )
    monkeypatch.setattr(
        jobs,
        "providers_for_vlm",
        lambda _: AigRoleProviders(
            FakeToolloopProvider(_script()),
            FakeCriticProvider(),
            FakeRefinerProvider(),
            DeterministicReader(),
        ),
    )
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_session_request_context] = lambda: _ctx()
    with TestClient(app) as client:
        body = {"request_id": "request-a", "question": "repair the parser"}
        path = "/api/v1/sessions/session-a/diagnosis"
        assert client.get(path + "/request-a").status_code == 404
        response = client.post(path, json=body)
        assert response.status_code == 200
        assert response.json()["result"]["status"] == "running"
        for _ in range(100):
            state = client.get(path + "/request-a").json()["result"]
            if state["status"] != "running":
                break
            time.sleep(0.01)
        assert state["status"] == "completed"
        assert "snapshot" not in state
        assert state["report"]["report"]["decisive_ao_refs"]
        assert client.post(path, json=body).json()["result"] == state


def test_http_diagnosis_rejects_other_session_owner_before_model_call(tmp_path, monkeypatch):
    owner = _session(tmp_path, owner="someone-else")

    async def load():
        pass

    owner.load = load
    monkeypatch.setattr(
        routes,
        "get_service",
        lambda: SimpleNamespace(sessions=SimpleNamespace(session=lambda *a, **k: owner)),
    )
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_session_request_context] = lambda: _ctx()
    # Surface the domain refusal, rather than a fake successful response.
    with TestClient(app, raise_server_exceptions=True) as client:
        import pytest

        from openviking_cli.exceptions import OpenVikingError

        with pytest.raises(OpenVikingError, match="not visible"):
            client.post(
                "/api/v1/sessions/session-a/diagnosis", json={"request_id": "r", "question": "q"}
            )
    assert not (tmp_path / "explicit-diagnoses").exists()
