# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Exclusive create and append-only JSON lines for session stores.

Callers keep their own lock, idempotency rule, and error type.
``write_exclusive`` raises ``FileExistsError`` when the path is already there.
Both writes fsync the file. The bytes are canonical JSON plus a newline.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from openviking.session.aig_native_protocol import canonical_dumps


def canonical_json_line(payload: Mapping[str, Any]) -> bytes:
    """One canonical JSON object and a trailing newline."""
    return (canonical_dumps(dict(payload)) + "\n").encode("utf-8")


def write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Create ``path`` with canonical JSON. Existing paths raise ``FileExistsError``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json_line(payload)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, text)
        os.fsync(fd)
    finally:
        os.close(fd)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append one canonical JSON line and fsync the index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json_line(row)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json_object(
    path: Path,
    error_cls: Callable[[str], BaseException],
    *,
    code: str = "stored_record_invalid",
) -> dict[str, Any]:
    """Read one JSON object. ``error_cls(code)`` is the caller's refusal."""
    if not path.is_file():
        raise error_cls(code)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise error_cls(code) from exc
    if not isinstance(payload, dict):
        raise error_cls(code)
    return payload


def read_jsonl_objects(
    path: Path,
    error_cls: Callable[[str], BaseException],
    *,
    code: str = "stored_record_invalid",
) -> list[dict[str, Any]]:
    """Read an append-only index. A missing file is an empty ledger."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise error_cls(code) from exc
        if not isinstance(row, dict):
            raise error_cls(code)
        rows.append(row)
    return rows


def write_new_json(
    path: Path,
    payload: Mapping[str, Any],
    error_cls: Callable[[str], BaseException],
    *,
    code: str = "stored_record_invalid",
) -> None:
    """Create ``path``. An existing path becomes ``error_cls(code)``."""
    try:
        write_exclusive(path, payload)
    except FileExistsError as exc:
        raise error_cls(code) from exc
