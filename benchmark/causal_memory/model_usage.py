"""Crash-visible, credential-free accounting for benchmark model requests."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOKEN_FIELDS = (
    "input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens"
)


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def normalized_usage(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = (payload or {}).get("usage")
    raw = raw if isinstance(raw, Mapping) else {}
    prompt = raw.get("prompt_tokens_details") or raw.get("input_tokens_details")
    completion = raw.get("completion_tokens_details") or raw.get("output_tokens_details")
    prompt = prompt if isinstance(prompt, Mapping) else {}
    completion = completion if isinstance(completion, Mapping) else {}
    usage = {
        "input_tokens": _count(raw.get("prompt_tokens", raw.get("input_tokens"))),
        "output_tokens": _count(raw.get("completion_tokens", raw.get("output_tokens"))),
        "total_tokens": _count(raw.get("total_tokens")),
        "cached_input_tokens": _count(prompt.get("cached_tokens")),
        "reasoning_tokens": _count(completion.get("reasoning_tokens")),
    }
    if usage["total_tokens"] is None and all(
        usage[k] is not None for k in ("input_tokens", "output_tokens")
    ):
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    usage["reported"] = any(usage[k] is not None for k in TOKEN_FIELDS)
    return usage


class ModelUsageRecorder:
    """Append before/after each HTTP attempt; never persist prompts or headers."""

    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path is not None else None
        self.role = "builder"
        self.run_id = ""
        self.completion_id = ""

    def bind(self, *, role: str, run_id: str) -> None:
        self.role, self.run_id = role, run_id

    def append(self, event: str, **fields: Any) -> None:
        if self.path is None:
            return
        record = {
            "event": event,
            "utc": datetime.now(timezone.utc).isoformat(),
            "role": self.role,
            "run_id": self.run_id,
            "completion_id": self.completion_id,
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        blob = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            if os.write(fd, blob) != len(blob):
                raise OSError("incomplete_model_usage_write")
        finally:
            os.close(fd)

    def begin_completion(self) -> None:
        self.completion_id = uuid.uuid4().hex

    def started(self, *, model: str, network_attempt: int) -> tuple[str, float]:
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        self.append(
            "request_started", request_id=request_id, model=model,
            network_attempt=network_attempt,
        )
        return request_id, started

    def completed(
        self, *, request_id: str, started: float, model: str, network_attempt: int,
        status: str, http_status: int | None, payload: Mapping[str, Any] | None,
    ) -> None:
        self.append(
            "request_completed", request_id=request_id, model=model,
            network_attempt=network_attempt, status=status, http_status=http_status,
            elapsed_ms=max(0, round((time.monotonic() - started) * 1000)),
            usage=normalized_usage(payload),
        )


def summarize_usage(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)

    def aggregate(group: list[Mapping[str, Any]]) -> dict[str, Any]:
        started = {r["request_id"] for r in group if r.get("event") == "request_started"}
        finished = {
            r["request_id"]: r for r in group if r.get("event") == "request_completed"
        }
        done = list(finished.values())
        missing = sum(
            any((r.get("usage") or {}).get(k) is None for k in TOKEN_FIELDS[:3])
            for r in done
        )
        return {
            "http_attempts_started": len(started),
            "http_attempts_completed": len(done),
            "incomplete_http_attempts": len(started - finished.keys()),
            "logical_completions": len({r.get("completion_id") for r in group
                                        if r.get("event") == "request_started"}),
            "network_retry_attempts": sum(r.get("network_attempt", 0) > 0 for r in group
                                          if r.get("event") == "request_started"),
            "http_status_counts": dict(Counter(str(r.get("http_status")) for r in done)),
            "request_status_counts": dict(Counter(r.get("status") for r in done)),
            "validation_status_counts": dict(Counter(r.get("status") for r in group
                                                      if r.get("event") == "completion_validated")),
            "completed_attempts_without_full_usage": missing,
            "usage_complete": not missing and not (started - finished.keys()),
            **{"known_" + k: sum((r.get("usage") or {}).get(k) or 0 for r in done)
               for k in TOKEN_FIELDS},
        }

    roles = sorted({r.get("role", "") for r in rows})
    runs = sorted({r.get("run_id", "") for r in rows})
    return {
        "overall": aggregate(rows),
        "by_role": {role: aggregate([r for r in rows if r.get("role") == role])
                    for role in roles},
        "by_run": {run: {
            "overall": aggregate([r for r in rows if r.get("run_id") == run]),
            "by_role": {role: aggregate([r for r in rows if r.get("run_id") == run
                                         and r.get("role") == role]) for role in roles},
        } for run in runs},
    }


def read_usage(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                # A killed process may leave a final partial line.
                continue
    return summarize_usage(rows)
