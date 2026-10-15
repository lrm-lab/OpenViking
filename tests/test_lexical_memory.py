from pathlib import Path

from openviking.service.lexical_memory import LexicalMemoryIndex


def test_bm25_documents_persist_and_accounts_are_isolated(tmp_path: Path):
    index = LexicalMemoryIndex(tmp_path, "workspace-a")
    index.upsert("viking://resources/docs/a.md", "The Lumen Observatory opened in 2041.")
    index.upsert("viking://resources/docs/b.md", "An unrelated coastal weather report.")
    hits = LexicalMemoryIndex(tmp_path, "workspace-a").find("Lumen observatory", limit=5)
    assert [hit["uri"] for hit in hits] == ["viking://resources/docs/a.md"]
    assert hits[0]["retrieval_mode"] == "bm25"
    assert "2041" in hits[0]["excerpt"]
    assert LexicalMemoryIndex(tmp_path, "workspace-b").find("Lumen", limit=5) == []


async def _allow(_uri: str) -> bool:
    return True


async def test_memory_follows_two_explicit_hops_without_requerying_or_embedding(tmp_path: Path):
    from openviking.service.lexical_retrieval import LexicalMemoryRetrieval

    index = LexicalMemoryIndex(tmp_path, "workspace-a")
    fork = "viking://resources/causal/fork.md"
    block = "viking://resources/causal/block.md"
    edge = "viking://resources/causal/edge.md"
    index.upsert(
        fork,
        "Observatory retrieval alternative",
        {"kind": "fork_overlay", "links": {"anchor": [block]}},
    )
    index.upsert(
        block,
        "Source selection checkpoint",
        {"kind": "native_block", "links": {"outgoing": [edge]}},
    )
    index.upsert(
        edge,
        "Earlier source selection influenced the final answer",
        {"kind": "native_edge", "links": {"source": [block]}},
    )
    retrieval = LexicalMemoryRetrieval(index)
    initial = await retrieval.find("Observatory", principal="memory-a", can_read=_allow)
    first = await retrieval.expand(
        initial["chain_ref"],
        fork,
        "anchor",
        principal="memory-a",
        can_read=_allow,
        operation_id="hop-1",
    )
    second = await retrieval.expand(
        initial["chain_ref"],
        block,
        "outgoing",
        principal="memory-a",
        can_read=_allow,
        operation_id="hop-2",
    )
    assert [hit["uri"] for hit in first["resources"]] == [block]
    assert [hit["uri"] for hit in second["resources"]] == [edge]
    assert second["step"] == 2
    assert second["retrieval_mode"] == "bm25"


async def test_expansion_rejects_unserved_anchors_other_principals_and_hidden_targets(
    tmp_path: Path,
):
    import pytest

    from openviking.service.lexical_retrieval import LexicalMemoryRetrieval

    index = LexicalMemoryIndex(tmp_path, "workspace-a")
    anchor = "viking://resources/causal/anchor.md"
    hidden = "viking://resources/causal/hidden.md"
    index.upsert(
        anchor, "Searchable checkpoint", {"kind": "native_block", "links": {"outgoing": [hidden]}}
    )
    index.upsert(hidden, "Restricted evidence", {"kind": "native_edge"})
    retrieval = LexicalMemoryRetrieval(index)
    initial = await retrieval.find("Searchable", principal="memory-a", can_read=_allow)
    with pytest.raises(ValueError, match="anchor_not_served"):
        await retrieval.expand(
            initial["chain_ref"],
            hidden,
            "outgoing",
            principal="memory-a",
            can_read=_allow,
            operation_id="forged",
        )
    with pytest.raises(ValueError, match="chain_unavailable"):
        await retrieval.expand(
            initial["chain_ref"],
            anchor,
            "outgoing",
            principal="memory-b",
            can_read=_allow,
            operation_id="other-actor",
        )

    async def visible(uri: str) -> bool:
        return uri != hidden

    result = await retrieval.expand(
        initial["chain_ref"],
        anchor,
        "outgoing",
        principal="memory-a",
        can_read=visible,
        operation_id="denied-target",
    )
    assert result["resources"] == []
    assert result["exhausted"] is True


async def test_expansion_retries_are_idempotent_and_recheck_revoked_access(tmp_path: Path):
    import asyncio
    import json

    import pytest

    from openviking.service.lexical_retrieval import LexicalMemoryRetrieval

    index = LexicalMemoryIndex(tmp_path, "a")
    anchor = "viking://resources/causal/a.md"
    target = "viking://resources/causal/b.md"
    index.upsert(
        anchor,
        "opening checkpoint",
        {"kind": "native_block", "links": {"outgoing": [target]}, "acl_labels": ["secret-label"]},
    )
    index.upsert(target, "recorded dependency", {"kind": "native_edge"})
    retrieval = LexicalMemoryRetrieval(index)
    initial = await retrieval.find("opening", principal="memory", can_read=_allow)
    assert target not in json.dumps(initial)
    assert "secret-label" not in json.dumps(initial)

    async def hop():
        return await retrieval.expand(
            initial["chain_ref"],
            anchor,
            "outgoing",
            principal="memory",
            can_read=_allow,
            operation_id="same",
        )

    responses = await asyncio.gather(hop(), hop())
    assert {response["step"] for response in responses} == {1}
    assert sum(response["duplicate"] for response in responses) == 1

    async def revoked(uri):
        return uri != target

    with pytest.raises(ValueError, match="replayed_target_unavailable"):
        await retrieval.expand(
            initial["chain_ref"],
            anchor,
            "outgoing",
            principal="memory",
            can_read=revoked,
            operation_id="same",
        )


async def test_frozen_native_fork_publication_can_be_searched_and_traversed(tmp_path):
    from openviking.server.identity import RequestContext, Role
    from openviking.service.lexical_memory_service import LexicalMemoryService
    from openviking.session.active_retrieval import ActiveRetrievalCorpus, BoundNativeView
    from openviking.session.fork_node_overlay import ForkNodeOverlayStore
    from openviking_cli.session.user_id import UserIdentifier
    from tests.test_fork_node_overlay import _native_revision, _proposal

    revision = _native_revision(summary="Observatory retrieval checkpoint")
    overlays = ForkNodeOverlayStore(tmp_path / "overlays")
    labels = ("user:alice",)
    catalog = overlays.register_base_revision(revision, acl_labels=labels)
    overlays.submit_proposal(
        _proposal(catalog, document_hash=revision.content_hash), principal_labels=labels
    )
    corpus = ActiveRetrievalCorpus(
        native_views=(BoundNativeView(revision, labels),), overlay_store=overlays
    )

    class Files:
        def __init__(self):
            self.text = {}

        async def write_file(self, uri, text, **kwargs):
            self.text[uri] = text

        async def stat(self, uri, **kwargs):
            return {"isDir": False}

        async def read_file(self, uri, **kwargs):
            return self.text[uri]

    files = Files()
    service = LexicalMemoryService(tmp_path / "index", lambda: files)
    ctx = RequestContext(UserIdentifier("account", "alice"), Role.ADMIN)
    published = await service.publish_corpus(ctx, corpus, "user:alice")
    assert published["resources"] == 6
    initial = await service.find("Counterfactual branch", ctx)
    fork = next(hit for hit in initial["resources"] if hit["metadata"]["kind"] == "fork_overlay")
    # Start with only the fork served, so its anchor is a new first hop.
    initial = await service.find("Counterfactual branch", ctx, target_uri=fork["uri"])
    first = await service.expand(
        ctx,
        chain_ref=initial["chain_ref"],
        uri=fork["uri"],
        relation="anchor",
        operation_id="anchor",
    )
    block = first["resources"][0]
    second = await service.expand(
        ctx,
        chain_ref=initial["chain_ref"],
        uri=block["uri"],
        relation="incoming",
        operation_id="edge",
    )
    assert second["resources"][0]["metadata"]["kind"] == "native_edge"
    assert second["step"] == 2
    from openviking.service.causal_memory_projection import causal_memory_projection

    assert causal_memory_projection(corpus, "user:bob") == []


async def test_account_deletion_removes_lexical_text_and_chain_state(tmp_path):
    import pytest

    from openviking.server.identity import RequestContext, Role
    from openviking.service.lexical_memory_service import LexicalMemoryService
    from openviking_cli.session.user_id import UserIdentifier

    service = LexicalMemoryService(tmp_path, lambda: None)
    ctx = RequestContext(UserIdentifier("account", "alice"), Role.ADMIN)
    retrieval = service._retrieval(ctx)
    retrieval.index.upsert("viking://resources/a.md", "Private observatory")
    initial = await retrieval.find("observatory", principal="account:alice", can_read=_allow)
    await service.delete_account("account")
    assert not list(tmp_path.glob("*.sqlite3*"))
    assert service.index(ctx).counts() == {}
    with pytest.raises(ValueError, match="chain_unavailable"):
        await service.expand(
            ctx,
            chain_ref=initial["chain_ref"],
            uri="viking://resources/a.md",
            relation="outgoing",
            operation_id="old",
        )


async def test_concurrent_imports_keep_file_and_index_at_the_same_version(tmp_path):
    import asyncio

    from openviking.server.identity import RequestContext, Role
    from openviking.service.lexical_memory_service import LexicalMemoryService
    from openviking_cli.session.user_id import UserIdentifier

    first_written, release_first = asyncio.Event(), asyncio.Event()

    class Files:
        text = ""

        async def write_file(self, uri, text, **kwargs):
            self.text = text
            if text == "Observatory first":
                first_written.set()
                await release_first.wait()

        async def stat(self, uri, **kwargs):
            return {"isDir": False}

        async def read_file(self, uri, **kwargs):
            return self.text

    files = Files()
    service = LexicalMemoryService(tmp_path, lambda: files)
    ctx = RequestContext(UserIdentifier("account", "alice"), Role.ADMIN)
    uri = "viking://resources/a.md"
    first = asyncio.create_task(
        service.put_resource(ctx, uri=uri, text="Observatory first", kind="document")
    )
    await first_written.wait()
    second = asyncio.create_task(
        service.put_resource(ctx, uri=uri, text="Observatory second", kind="document")
    )
    await asyncio.sleep(0)
    release_first.set()
    await asyncio.gather(first, second)
    assert files.text == "Observatory second"
    assert (await service.find("second", ctx))["resources"][0]["uri"] == uri


def test_graph_identity_includes_owner_and_session_even_for_same_revision_ids():
    from dataclasses import replace

    from openviking.service.causal_memory_projection import causal_memory_projection
    from openviking.session.active_retrieval import ActiveRetrievalCorpus, BoundNativeView
    from openviking.session.extraction_write import freeze_native_edge_view
    from tests.test_fork_node_overlay import _coverage, _native_revision

    first = _native_revision()
    blocks = tuple(
        replace(
            block,
            segment_ref=replace(block.segment_ref, session_id="another-session"),
            source_refs=tuple(
                replace(ref, session_id="another-session") for ref in block.source_refs
            ),
        )
        for block in first.blocks
    )
    second = freeze_native_edge_view(
        session_id="another-session",
        task_run_id=first.task_run_id,
        view_id=first.view_id,
        revision_id=first.revision_id,
        frozen_at=first.frozen_at,
        blocks=blocks,
        edges=first.edges,
        coverage=_coverage(blocks),
    )
    corpus = ActiveRetrievalCorpus(
        native_views=(
            BoundNativeView(first, ("user:alice", "user:bob")),
            BoundNativeView(second, ("user:alice",)),
        )
    )
    rows = causal_memory_projection(corpus, "user:alice")
    assert len(rows) == len({row[0] for row in rows}) == 6
    bob = causal_memory_projection(corpus, "user:bob")
    assert {row[0] for row in rows}.isdisjoint(row[0] for row in bob)
