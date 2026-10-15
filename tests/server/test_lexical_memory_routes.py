from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from openviking.server.auth import get_request_context
from openviking.server.identity import RequestContext, Role
from openviking.server.routers import search
from openviking_cli.session.user_id import UserIdentifier


@pytest.mark.asyncio
async def test_bm25_find_does_not_enter_the_embedding_search_path(tmp_path, monkeypatch):
    from openviking.service.lexical_memory_service import LexicalMemoryService

    class Files:
        async def stat(self, uri, **kwargs):
            return {"isDir": False}

        async def read_file(self, uri, **kwargs):
            return "Lumen Observatory opened in 2041."

    def forbidden(*args, **kwargs):
        raise AssertionError("embedding search was called")

    lexical = LexicalMemoryService(tmp_path, lambda: Files())
    ctx = RequestContext(UserIdentifier("account-a", "memory-agent"), Role.USER)
    lexical.index(ctx).upsert(
        "viking://resources/docs/opening.md", "Lumen Observatory opened in 2041."
    )
    service = SimpleNamespace(lexical=lexical, search=SimpleNamespace(find=forbidden))
    monkeypatch.setattr(search, "get_service", lambda: service)
    app = FastAPI()
    app.include_router(search.router)
    app.dependency_overrides[get_request_context] = lambda: ctx
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/search/find", json={"query": "Lumen", "retrieval_mode": "bm25"}
        )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["retrieval_mode"] == "bm25"
    assert result["resources"][0]["uri"] == "viking://resources/docs/opening.md"
    assert result["chain_ref"]


@pytest.mark.asyncio
async def test_causal_publication_checks_stored_owner_before_reading_corpus(monkeypatch):
    from openviking.server.routers import lexical_memory
    from openviking_cli.exceptions import PermissionDeniedError

    class Session:
        _meta = SimpleNamespace(created_by_user_id="bob")
        _session_uri = "viking://user/bob/sessions/session-a"

        async def load(self):
            return None

    service = SimpleNamespace(sessions=SimpleNamespace(session=lambda *args: Session()))
    monkeypatch.setattr(lexical_memory, "get_service", lambda: service)
    ctx = RequestContext(UserIdentifier("account-a", "alice"), Role.ADMIN)
    with pytest.raises(PermissionDeniedError, match="stored session owner"):
        await lexical_memory.publish_causal(
            lexical_memory.PublishCausalRequest(session_id="session-a"), ctx
        )


@pytest.mark.asyncio
async def test_lexical_find_filters_files_that_changed_since_indexing(tmp_path):
    from openviking.service.lexical_memory_service import LexicalMemoryService

    class Files:
        async def stat(self, uri, **kwargs):
            return {"isDir": False}

        async def read_file(self, uri, **kwargs):
            return "The original document was replaced."

    service = LexicalMemoryService(tmp_path, lambda: Files())
    ctx = RequestContext(UserIdentifier("account-a", "alice"), Role.USER)
    service.index(ctx).upsert("viking://resources/docs/a.md", "Lumen secret text")
    result = await service.find("Lumen", ctx)
    assert result["resources"] == []
