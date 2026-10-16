# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Default refusal for legacy Claim/Cut writers (ADR-0043 P11).

Production call sites do not pass the opt-in. The legacy writers stay
available as an explicit audit tool.
"""

from __future__ import annotations


class LegacyWriteRefused(ValueError):
    """A legacy Claim or Cut write ran without ``legacy_audit_opt_in=True``."""

    code = "legacy_write_refused"

    def __init__(self) -> None:
        super().__init__("legacy Claim/Cut write is refused unless legacy_audit_opt_in=True")


def require_legacy_audit_opt_in(opt_in: bool) -> None:
    """Refuse unless the caller set the explicit legacy-audit opt-in."""
    if opt_in is not True:
        raise LegacyWriteRefused()
