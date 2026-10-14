# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Session-frozen causal mode with the server default for older sessions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from openviking_cli.utils.config.memory_config import is_causal_mode_enabled


def is_session_causal(
    session: Any,
    config: Any = None,
    *,
    mode_resolver: Callable[..., bool] = is_causal_mode_enabled,
) -> bool:
    mode = getattr(getattr(session, "_meta", None), "skill_trajectory_mode", None)
    if mode == "causal":
        return True
    if mode == "legacy":
        return False
    if config is None:
        return mode_resolver(session_id=getattr(session, "session_id", None))
    return mode_resolver(config, session_id=getattr(session, "session_id", None))
