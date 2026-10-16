# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Read AORecord rows from an OV session tree (ADR-0045 Wave E2).

Probe of the 1933 prototype (container ``coforge-openviking-prototype-1``,
``/data/viking/evol-389e96deae16/user/eval-admin/sessions``, 136
``coforge-quiet-*`` dirs) found **zero** ``ao-ledger.jsonl`` files under
``/data/viking``. Causal mode is unset in that config, so the live hook in
``maybe_record_session_tool_exchanges`` never wrote a ledger.

Nearest evidence, in order:
1. ``<session>/ao-ledger.jsonl`` when present (canonical ``AORecord.to_dict``).
2. ``<session>/history/archive_*/messages.jsonl`` plus live ``messages.jsonl``.
   Tool parts are replayed through ``AOLedgerRecorder.record_messages`` in a
   temp dir; remaining text turns become derived AO rows. This is a derivation,
   not a native ledger.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from openviking.message.message import Message
from openviking.session.ao_ledger import LEDGER_FILENAME, AORecord
from openviking.session.ao_ledger_recorder import HISTORY_DIRNAME, AOLedgerRecorder

DERIVED_STATE = "derived_from_archive"
DEFAULT_EXPORT_PATH = Path("benchmark/causal_memory/out/ao_records.jsonl")
PROBED_SESSIONS_ROOT = "/data/viking/evol-389e96deae16/user/eval-admin/sessions"


def iter_session_dirs(root: Path) -> Iterator[Path]:
    """Yield session directories. A single session dir is yielded as itself."""
    root = Path(root)
    if _looks_like_session(root):
        yield root
        return
    if not root.is_dir():
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and _looks_like_session(child):
            yield child


def load_session_records(session_dir: Path) -> tuple[list[AORecord], str]:
    """Load AO rows from one session. ``source`` is ``ledger`` or ``derived``."""
    session_dir = Path(session_dir)
    session_id = _session_id(session_dir)
    ledger = session_dir / LEDGER_FILENAME
    if ledger.is_file():
        records = _read_ledger(ledger)
        if records:
            return records, "ledger"
    derived = _derive_from_messages(session_dir, session_id)
    return derived, "derived"


def export_sessions(root: Path, out_path: Path) -> dict[str, Any]:
    """Write one AORecord JSONL and a small manifest."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    sessions = 0
    sources = {"ledger": 0, "derived": 0, "empty": 0}
    with out_path.open("w", encoding="utf-8") as handle:
        for session_dir in iter_session_dirs(root):
            sessions += 1
            records, source = load_session_records(session_dir)
            if not records:
                sources["empty"] += 1
                continue
            sources[source] = sources.get(source, 0) + 1
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                written += 1
    return {
        "empty_sessions": sources["empty"],
        "out": str(out_path),
        "records": written,
        "sessions": sessions,
        "source_derived_sessions": sources["derived"],
        "source_ledger_sessions": sources["ledger"],
    }


def group_records(records: Iterable[AORecord]) -> dict[str, list[AORecord]]:
    grouped: dict[str, list[AORecord]] = {}
    for record in records:
        grouped.setdefault(record.session_id, []).append(record)
    for rows in grouped.values():
        rows.sort(key=lambda item: item.sequence)
    return grouped


def read_ao_jsonl(path: Path) -> list[AORecord]:
    records: list[AORecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict):
                records.append(AORecord.from_dict(payload))
    return records


def _looks_like_session(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (
        (path / LEDGER_FILENAME).is_file()
        or (path / "messages.jsonl").exists()
        or (path / ".meta.json").is_file()
        or (path / HISTORY_DIRNAME).is_dir()
    )


def _session_id(session_dir: Path) -> str:
    meta = session_dir / ".meta.json"
    if meta.is_file():
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            found = payload.get("session_id")
            if isinstance(found, str) and found.strip():
                return found.strip()
    return session_dir.name


def _read_ledger(path: Path) -> list[AORecord]:
    records: list[AORecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    records.append(AORecord.from_dict(payload))
            except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                continue
    records.sort(key=lambda item: item.sequence)
    return records


def _derive_from_messages(session_dir: Path, session_id: str) -> list[AORecord]:
    messages = list(_iter_session_messages(session_dir))
    if not messages:
        return []
    tool_rows = _records_from_tool_parts(session_id, messages)
    if tool_rows:
        return tool_rows
    return _records_from_text_turns(session_id, messages)


def _iter_session_messages(session_dir: Path) -> Iterator[Message]:
    history = session_dir / HISTORY_DIRNAME
    if history.is_dir():
        for archive in sorted(history.iterdir()):
            path = archive / "messages.jsonl"
            if path.is_file():
                yield from _read_messages(path)
    live = session_dir / "messages.jsonl"
    if live.is_file() and live.stat().st_size > 0:
        yield from _read_messages(live)


def _read_messages(path: Path) -> Iterator[Message]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            try:
                yield Message.from_dict(payload)
            except (KeyError, TypeError, ValueError):
                continue


def _records_from_tool_parts(session_id: str, messages: list[Message]) -> list[AORecord]:
    with tempfile.TemporaryDirectory(prefix="ao-export-") as tmp:
        recorder = AOLedgerRecorder(Path(tmp), session_id)
        recorded = recorder.record_messages(messages, lookup_messages=messages)
        return list(recorded)


def _records_from_text_turns(session_id: str, messages: list[Message]) -> list[AORecord]:
    records: list[AORecord] = []
    for index, message in enumerate(messages, start=1):
        text = message.content if isinstance(message.content, str) else ""
        participant = message.peer_id or message.role
        created = message.created_at or ""
        if not isinstance(created, str):
            created = str(created)
        ao_id = hashlib.sha256(f"{session_id}:{index}:{message.id}".encode("utf-8")).hexdigest()[
            :32
        ]
        records.append(
            AORecord(
                ao_id=ao_id,
                session_id=session_id,
                sequence=index,
                action={
                    "participant_id": participant or message.role,
                    "role": message.role,
                    "segment_id": f"msg-{index}",
                    "text": text,
                },
                observation={"text": text},
                skill_invocations=[],
                message_ref={"message_id": message.id, "source": "archive_messages"},
                created_at=created or "1970-01-01T00:00:00.000Z",
                captured_state=DERIVED_STATE,
            )
        )
    return records
