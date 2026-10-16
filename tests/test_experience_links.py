# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Derived experience edges: direction, status, ACL, replay, and both budgets."""

from __future__ import annotations

from pathlib import Path

from openviking.session.aig_native_protocol import guidance_execution_report_document
from openviking.session.append_only import write_exclusive
from openviking.session.experience_links import (
    DEFAULT_MEMORY_MAX_CARDS,
    DEFAULT_MEMORY_MAX_HOPS,
    DEFAULT_READER_MAX_CARDS,
    DEFAULT_READER_MAX_HOPS,
    LINK_TYPES,
    ExperienceLinkError,
    ExperienceLinkService,
)
from openviking.session.fork_node_overlay import ForkNodeOverlayStore
from openviking.session.remediation_dispatch import REMEDIATION_RUN_DIRNAME
from openviking.session.strategy_pattern import StrategyPatternStore
from tests.test_fork_node_overlay import _native_revision, _proposal
from tests.test_guidance_approval import _candidate, _catalog, _propose, _world
from tests.test_strategy_pattern import _evaluation, _proposal_payload

_ALICE = "user:alice"
_BOB = "user:bob"


def _code(call) -> str:
    try:
        call()
    except ExperienceLinkError as exc:
        return exc.code
    raise AssertionError("expected a coded error")


def _service(tmp_path: Path, **limits: int) -> tuple[ExperienceLinkService, object]:
    world = _world(tmp_path)
    links = ExperienceLinkService(
        tmp_path,
        overlays=world.overlays,
        patterns=world.patterns,
        guidance=world.service,
        **limits,
    )
    return links, world


def _types(result: dict, link_type: str) -> list[dict]:
    return [edge for edge in result["edges"] if edge["link_type"] == link_type]


def test_closed_link_set_and_default_budgets() -> None:
    assert LINK_TYPES == (
        "anchors",
        "factual_ref",
        "supports",
        "refutes",
        "instance_of",
        "has_member",
        "guides",
        "executed_as",
    )
    assert (DEFAULT_READER_MAX_HOPS, DEFAULT_READER_MAX_CARDS) == (8, 32)
    assert (DEFAULT_MEMORY_MAX_HOPS, DEFAULT_MEMORY_MAX_CARDS) == (4, 16)


def test_anchor_and_factual_directions(tmp_path: Path) -> None:
    links, world = _service(tmp_path)
    fork_id = world.fork_node_id
    outgoing = links.expand(fork_id, principal_labels=("user:carol",), unresolved=("gap",))
    anchors = _types(outgoing, "anchors")
    factual = _types(outgoing, "factual_ref")
    assert len(anchors) == 1
    assert anchors[0]["direction"] == "out"
    assert anchors[0]["source_id"] == fork_id
    assert anchors[0]["target_id"] == "b-2"
    assert anchors[0]["peer"]["kind"] == "block"
    assert anchors[0]["peer"]["acl_visible"] is True
    assert factual[0]["direction"] == "out"
    assert factual[0]["target_id"] == "b-1"
    incoming = links.expand(
        "b-2",
        link_types=["anchors"],
        principal_labels=("user:carol",),
        context_id=outgoing["context_id"],
        unresolved=("gap",),
    )
    assert incoming["edges"][0]["direction"] == "in"
    assert incoming["edges"][0]["source_id"] == fork_id
    audit = (tmp_path / "experience-link-audit" / "hops.jsonl").read_text(encoding="utf-8")
    assert "gap" in audit


def test_invalidated_evaluation_becomes_refutes_and_retired_keeps_its_flag(tmp_path: Path) -> None:
    links, world = _service(tmp_path)
    overlay = world.stored.stored
    pattern = world.patterns.accept_proposal(
        overlay, world.base, _proposal_payload(overlay.overlay, "img-retry")
    )
    world.patterns.append_evaluation(
        _evaluation(
            "imagined_node",
            "img-retry",
            "eval-active",
            "evidence:active",
            "analogical",
            "active",
        )
    )
    supported = links.expand(
        "img-retry",
        link_types=["supports", "refutes"],
        principal_labels=("user:carol",),
        unresolved=(),
    )
    assert [edge["link_type"] for edge in supported["edges"]] == ["supports"]
    assert supported["edges"][0]["direction"] == "in"
    world.patterns.append_evaluation(
        _evaluation(
            "imagined_node",
            "img-retry",
            "eval-invalid",
            "evidence:invalid",
            "analogical",
            "invalidated",
        )
    )
    refuted = links.expand(
        "img-retry",
        link_types=["supports", "refutes"],
        principal_labels=("user:carol",),
        context_id=supported["context_id"],
        unresolved=(),
    )
    assert [edge["link_type"] for edge in refuted["edges"]] == ["refutes"]
    assert refuted["edges"][0]["status"] == "invalidated"
    member = links.expand(
        "img-retry",
        link_types=["instance_of"],
        principal_labels=("user:carol",),
        unresolved=(),
    )
    assert member["edges"][0]["direction"] == "out"
    assert member["edges"][0]["target_id"] == pattern.pattern_id
    assert member["edges"][0]["peer"]["kind"] == "strategy_pattern"
    reverse = links.expand(
        pattern.pattern_id,
        link_types=["has_member"],
        principal_labels=("user:carol",),
        unresolved=(),
    )
    assert reverse["edges"][0]["direction"] == "out"
    assert reverse["edges"][0]["target_id"] == "img-retry"


def test_retired_is_visible_to_reader_and_hidden_from_memory(tmp_path: Path) -> None:
    links, world = _service(tmp_path)
    overlay = world.stored.stored
    world.patterns.accept_proposal(
        overlay, world.base, _proposal_payload(overlay.overlay, "img-retry")
    )
    world.patterns.append_evaluation(
        _evaluation(
            "imagined_node",
            "img-retry",
            "eval-retired",
            "evidence:retired",
            "analogical",
            "retired",
        )
    )
    reader = links.expand(
        "img-retry",
        link_types=["supports"],
        principal_labels=("user:carol",),
        profile="reader",
        unresolved=("old",),
    )
    assert reader["edges"][0]["status"] == "retired"
    memory = links.expand(
        "img-retry",
        link_types=["supports"],
        principal_labels=("user:carol",),
        profile="memory",
        unresolved=("old",),
    )
    assert memory["edges"] == []
    assert memory["profile"] == "memory"


def test_acl_hides_the_other_principal_and_replay_is_unreachable(tmp_path: Path) -> None:
    links, world = _service(tmp_path)
    hidden = links.expand(world.fork_node_id, principal_labels=(_BOB,), unresolved=())
    assert hidden["edges"] == []
    assert _code(
        lambda: links.expand("replay-rev:hidden", principal_labels=("user:carol",), unresolved=())
    ) == ("replay_provenance_excluded")
    replay = _native_revision("replay-rev:hidden")
    catalog = world.overlays.register_base_revision(replay, acl_labels=("user:carol",))
    payload = _proposal(catalog, document_hash=replay.content_hash)
    stored = world.overlays.submit_proposal(payload, principal_labels=("user:carol",))
    again = links.expand(
        stored.stored.fork_node_id,
        link_types=["anchors"],
        principal_labels=("user:carol",),
        unresolved=(),
    )
    assert again["edges"] == []
    world.patterns.freeze_branch(world.stored.stored, world.base)
    world.patterns.append_evaluation(
        _evaluation(
            "imagined_node",
            "img-retry",
            "eval-replay",
            "replay-rev:evidence",
            "analogical",
            "active",
        )
    )
    plain = links.expand(
        "img-retry",
        link_types=["supports", "refutes"],
        principal_labels=("user:carol",),
        unresolved=(),
    )
    assert plain["edges"] == []


def test_guides_and_executed_as(tmp_path: Path) -> None:
    links, world = _service(tmp_path)
    proposed = _propose(world)
    approved = world.service.approve(
        proposed.proposal.content_hash,
        _catalog(world, [_candidate()]),
        principal_labels=("user:carol",),
    )
    guided = links.expand(
        approved.guidance.guidance_id,
        link_types=["guides"],
        principal_labels=("user:carol",),
        unresolved=(),
    )
    assert guided["edges"][0]["direction"] == "out"
    assert guided["edges"][0]["target_id"] == world.fork_node_id
    assert guided["edges"][0]["peer"]["kind"] == "fork_node"
    document = guidance_execution_report_document(
        guidance_id=approved.guidance.guidance_id,
        source_fork_id=world.fork_node_id,
        parent_session_id="sess-parent",
        steps=[
            {
                "actual_block_id": "b-1",
                "alignment": "equivalent",
                "imagined_node_id": "img-retry",
            }
        ],
    )
    write_exclusive(
        tmp_path / REMEDIATION_RUN_DIRNAME / "reports" / f"{document['content_hash']}.json",
        document,
    )
    executed = links.expand(
        "img-retry",
        link_types=["executed_as"],
        principal_labels=("user:carol",),
        unresolved=("unresolved-step",),
    )
    assert executed["edges"][0]["direction"] == "in"
    assert executed["edges"][0]["source_id"] == document["content_hash"]
    assert executed["edges"][0]["peer"]["kind"] == "execution_report"
    replay_doc = guidance_execution_report_document(
        guidance_id=approved.guidance.guidance_id,
        source_fork_id=world.fork_node_id,
        parent_session_id="replay-session:child",
        steps=[
            {
                "actual_block_id": "b-1",
                "alignment": "equivalent",
                "imagined_node_id": "img-retry",
            }
        ],
    )
    write_exclusive(
        tmp_path / REMEDIATION_RUN_DIRNAME / "reports" / f"{replay_doc['content_hash']}.json",
        replay_doc,
    )
    still = links.expand(
        "img-retry",
        link_types=["executed_as"],
        principal_labels=("user:carol",),
        unresolved=(),
    )
    assert [edge["source_id"] for edge in still["edges"]] == [document["content_hash"]]


def test_hop_and_card_budgets_fail_closed(tmp_path: Path) -> None:
    links, world = _service(tmp_path, reader_max_hops=2, reader_max_cards=1)
    first = links.expand("missing-subject", principal_labels=("user:carol",), unresolved=())
    second = links.expand(
        "missing-subject",
        principal_labels=("user:carol",),
        context_id=first["context_id"],
        unresolved=(),
    )
    assert second["hops"] == 2
    assert (
        _code(
            lambda: links.expand(
                "missing-subject",
                principal_labels=("user:carol",),
                context_id=first["context_id"],
                unresolved=(),
            )
        )
        == "hop_budget_exceeded"
    )
    cards, _world_cards = _service(tmp_path / "cards", reader_max_cards=1)
    assert (
        _code(
            lambda: cards.expand(
                _world_cards.fork_node_id,
                principal_labels=("user:carol",),
                unresolved=(),
            )
        )
        == "hop_budget_exceeded"
    )
    assert _code(
        lambda: links.expand(
            "x",
            link_types=["not-a-link"],
            principal_labels=("user:carol",),
            unresolved=(),
        )
    ) == ("unknown_link_type")


def test_reader_profile_allows_eight_hops_then_refuses(tmp_path: Path) -> None:
    overlays = ForkNodeOverlayStore(tmp_path)
    patterns = StrategyPatternStore(tmp_path)
    links = ExperienceLinkService(tmp_path, overlays=overlays, patterns=patterns)
    context_id = None
    for _index in range(DEFAULT_READER_MAX_HOPS):
        result = links.expand(
            "nobody",
            principal_labels=(_ALICE,),
            context_id=context_id,
            unresolved=(),
        )
        context_id = result["context_id"]
    assert result["hops"] == DEFAULT_READER_MAX_HOPS
    assert (
        _code(
            lambda: links.expand(
                "nobody",
                principal_labels=(_ALICE,),
                context_id=context_id,
                unresolved=(),
            )
        )
        == "hop_budget_exceeded"
    )
