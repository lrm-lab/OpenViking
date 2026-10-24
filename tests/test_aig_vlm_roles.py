# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Account VLM role adapters: constrained JSON output and critic diagnostics."""

from __future__ import annotations

import logging
import time
from typing import Any

import pytest

from openviking.session.aig_toolloop_critic import CriticTurn, critic_catalog
from openviking.session.aig_vlm_roles import VLMCritic


class _Handle:
    def __init__(self, model: str, replies: list[str]) -> None:
        self.model = model
        self.replies = list(replies)
        self.options: list[dict[str, Any]] = []

    async def get_completion_async(self, messages: Any, thinking: bool = False, **options: Any) -> str:
        self.options.append(options)
        return self.replies.pop(0)


class _View:
    def call(self, verb: str, arguments: Any = None) -> Any:
        assert verb == "get_overview"
        return {"row_count": 3}


def _critic(model: str, replies: list[str]) -> tuple[VLMCritic, _Handle, CriticTurn]:
    handle = _Handle(model, replies)
    critic = VLMCritic(handle, time.monotonic() + 60)
    critic.bind_view(_View())  # type: ignore[arg-type]
    turn = CriticTurn(
        system_prompt="audit",
        tools=critic_catalog(),
        draft={"blocks": [{"block_id": "b-1"}]},
        structural_violations=(),
    )
    return critic, handle, turn


@pytest.mark.parametrize("model", ["google/gemma-4-31b-it", "qwen/qwen3.5-27b", "gemma-4-31b-it"])
def test_verified_models_get_strict_json_schema(model: str) -> None:
    critic, handle, turn = _critic(model, ['{"violations": []}'])
    assert critic(turn) == {"violations": []}
    fmt = handle.options[0]["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True


def test_unverified_model_keeps_unconstrained_parsing() -> None:
    critic, handle, turn = _critic("some/other-model", ['{"violations": []}'])
    critic(turn)
    assert "response_format" not in handle.options[0]


def test_invalid_critic_action_logs_the_raw_output_and_still_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw = '{"violations": [], "verb": "read_step", "note": "unexpected-shape-marker"}'
    critic, _handle, turn = _critic("some/other-model", [raw])
    with caplog.at_level(logging.WARNING, logger="openviking.session.aig_vlm_roles"):
        with pytest.raises(ValueError, match="invalid_critic_action"):
            critic(turn)
    assert any("unexpected-shape-marker" in record.getMessage() for record in caplog.records)
    assert any(a.get("result") == "invalid_action" for a in critic.attempts)
