# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Critic semantic audit: strict JSON, empty set, and a read-only tool list."""

from __future__ import annotations

from pathlib import Path

import pytest

from openviking.session.aig_native_protocol import TOOLLOOP_BUILD_VERBS
from openviking.session.aig_tool_surface import AigToolSurface, AoSnapshot
from openviking.session.aig_toolloop_critic import (
    CRITIC_SYSTEM_PROMPT,
    CRITIC_VERBS,
    CriticError,
    CriticView,
    FakeCriticProvider,
    audit_semantics,
    critic_catalog,
)
from openviking.session.ao_ledger import AORecord


def _record(index: int) -> AORecord:
    return AORecord(
        ao_id=f"ao-{index}",
        session_id="sess-critic",
        sequence=index + 1,
        action={"participant_id": "worker", "segment_id": "seg", "text": f"row {index}"},
        observation={"text": f"saw {index}"},
        skill_invocations=[],
        message_ref=None,
        created_at="2026-10-11T00:00:00.000Z",
        captured_state="live",
    )


def _surface(tmp_path: Path) -> AigToolSurface:
    snapshot = AoSnapshot((_record(0), _record(1)), watermark="wm-critic", question="q")
    surface = AigToolSurface(snapshot, root=tmp_path, run_id="run-critic")
    surface.add_block(
        "agent",
        {
            "agent_name": "worker",
            "summary": "did the work",
            "input": "do the task",
            "output": "a draft",
            "authorship": "authored",
            "step_refs": [0],
        },
        block_id="agent-1",
    )
    surface.add_block("conclusion", {"content": "the answer", "step_refs": [1]}, block_id="end-1")
    return surface


def _item(object_id: str = "agent-1") -> dict[str, object]:
    return {
        "object_id": object_id,
        "kind": "semantic",
        "severity": "high",
        "evidence_ao_refs": [0],
        "note": "summary contradicts the row",
    }


def test_fake_provider_json_is_parsed(tmp_path: Path) -> None:
    surface = _surface(tmp_path)
    provider = FakeCriticProvider([{"violations": [_item()]}])
    found = audit_semantics(surface, (), provider)
    assert len(found) == 1
    assert found[0].violation_id == "critic:0:semantic:agent-1"
    assert found[0].kind == "semantic"
    assert found[0].severity == "high"
    assert found[0].evidence_ao_refs == (0,)
    assert found[0].involved_ids == ("agent-1",)
    assert found[0].scope == "objects"
    assert provider.turns[0].system_prompt == CRITIC_SYSTEM_PROMPT
    assert "step_refs" in CRITIC_SYSTEM_PROMPT


def test_empty_violation_array_is_a_clean_audit(tmp_path: Path) -> None:
    surface = _surface(tmp_path)
    provider = FakeCriticProvider(["[]"])
    assert audit_semantics(surface, (), provider) == ()


def test_catalog_hides_build_verbs(tmp_path: Path) -> None:
    surface = _surface(tmp_path)
    names = {name for name, _description in critic_catalog()}
    assert names == CRITIC_VERBS
    assert names.isdisjoint(TOOLLOOP_BUILD_VERBS)
    view = CriticView(surface)
    with pytest.raises(CriticError) as caught:
        view.call("add_block", {"kind": "task", "fields": {}})
    assert caught.value.code == "build_verb_forbidden"
    provider = FakeCriticProvider([[]])
    audit_semantics(surface, (), provider)
    assert {name for name, _description in provider.turns[0].tools} == names


def test_malformed_critic_output_is_rejected(tmp_path: Path) -> None:
    surface = _surface(tmp_path)
    provider = FakeCriticProvider(["{"])
    with pytest.raises(CriticError) as caught:
        audit_semantics(surface, (), provider)
    assert caught.value.code == "invalid_critic_output"
