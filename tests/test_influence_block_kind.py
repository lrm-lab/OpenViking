# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""``InfluenceBlock.block_kind`` refines ``role`` and stays compatible with old payloads."""

from __future__ import annotations

import dataclasses
import json

import pytest

from openviking.session.influence_view import (
    BLOCK_KIND_ROLES,
    BLOCK_KINDS,
    BLOCK_ROLES,
    InfluenceBlock,
    SegmentRef,
    SourceRef,
)

SESSION = "sess-kind"


def _ref(sequence: int = 1) -> SourceRef:
    return SourceRef(
        ao_id=f"ao-{sequence}",
        sequence=sequence,
        participant_id="agent-a",
        segment_id="seg-a",
        session_id=SESSION,
        content_hash=f"hash-{sequence}",
    )


def _block(role: str = "agent_action", block_kind: str | None = None) -> InfluenceBlock:
    ref = _ref()
    kwargs: dict[str, object] = {
        "block_id": "b1",
        "role": role,
        "participant_id": "agent-a",
        "segment_ref": SegmentRef(segment_id="seg-a", session_id=SESSION),
        "ao_start_seq": 1,
        "ao_end_seq": 1,
        "summary": "s",
        "input": "",
        "output": "",
        "authorship": "agent-a",
        "granularity_reason": "one record",
        "source_content_hashes": (ref.content_hash,),
        "source_refs": (ref,),
    }
    if block_kind is not None:
        kwargs["block_kind"] = block_kind
    return InfluenceBlock(**kwargs)  # type: ignore[arg-type]


def _payload(role: str = "agent_action") -> dict[str, object]:
    block = _block(role)
    payload = block.to_dict()
    payload.pop("block_kind")
    return payload


def test_block_kind_roles_cover_block_roles_and_omit_unknown() -> None:
    assert set(BLOCK_KIND_ROLES.values()) <= set(BLOCK_ROLES)
    assert "unknown" not in BLOCK_KIND_ROLES
    assert set(BLOCK_KINDS) == set(BLOCK_KIND_ROLES) | {"unknown"}
    for kind, role in BLOCK_KIND_ROLES.items():
        assert kind in BLOCK_KINDS
        assert role in BLOCK_ROLES


def test_block_kind_is_the_last_field_and_defaults_to_unknown() -> None:
    names = [field.name for field in dataclasses.fields(InfluenceBlock)]
    assert names[-1] == "block_kind"
    assert InfluenceBlock.__dataclass_fields__["block_kind"].default == "unknown"
    block = _block()
    assert block.block_kind == "unknown"
    block.validate()


def test_block_kind_roundtrip_and_legacy_payload_without_the_key() -> None:
    typed = _block("agent_action", "tool_call")
    restored = InfluenceBlock.from_dict(json.loads(json.dumps(typed.to_dict())))
    assert restored == typed
    assert restored.block_kind == "tool_call"
    assert restored.to_dict()["block_kind"] == "tool_call"

    legacy = _payload("task")
    assert "block_kind" not in legacy
    legacy["surplus_key"] = "ignored"
    loaded = InfluenceBlock.from_dict(legacy)
    assert loaded.block_kind == "unknown"
    assert loaded.role == "task"
    loaded.validate()
    assert "surplus_key" not in loaded.to_dict()


def test_block_kind_conflict_with_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="refines role"):
        _block("task", "tool_call")
    with pytest.raises(ValueError, match="refines role"):
        _block("agent_action", "conclusion")
    with pytest.raises(ValueError, match="block_kind"):
        _block("agent_action", "assistant_text")

    for kind, role in BLOCK_KIND_ROLES.items():
        block = _block(role, kind)
        assert block.block_kind == kind
        assert block.role == role

    for role in sorted(BLOCK_ROLES):
        block = _block(role, "unknown")
        assert block.block_kind == "unknown"
