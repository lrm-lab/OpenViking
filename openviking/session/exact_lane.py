# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Current-log exact lane (Q140–Q142, Q149–Q152).

A legal exact clause searches caller-supplied field slices. Literal matching
always runs in this module. Regex uses a RE2-compatible engine when one is
provided or importable as ``re2``; the host ``re`` module is never used.
ACL, withdrawal, and scope filtering happen before the public hit limit.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

MAX_SPANS = 8
DEFAULT_EXACT_LIMIT = 20
CLOSED_FIELDS = ("action", "observation", "message")
FIELD_RANK = {name: index for index, name in enumerate(CLOSED_FIELDS)}
REPRESENTATION = {
    "action": "canonical_action",
    "observation": "raw_observation",
    "message": "channel_message",
}
OFFSET_UNIT = "unicode_code_point"
FRAGMENT_POLICY = "match_text"


class ExactInputError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def parse_exact_clause(raw: Any) -> dict[str, Any]:
    """Return the canonical clause or raise ``ExactInputError``."""
    if not isinstance(raw, Mapping):
        raise ExactInputError("invalid_exact_clause")
    allowed = {"expression", "syntax", "fields", "caseSensitive"}
    if set(raw) - allowed:
        raise ExactInputError("invalid_exact_clause")
    expression = raw.get("expression")
    syntax = raw.get("syntax")
    if not isinstance(expression, str) or expression == "":
        raise ExactInputError("invalid_exact_clause")
    if syntax not in ("literal", "regex"):
        raise ExactInputError("invalid_exact_clause")
    if "fields" in raw and raw["fields"] is not None:
        fields = raw["fields"]
        if (
            isinstance(fields, (str, bytes))
            or not isinstance(fields, Sequence)
            or not fields
            or len(set(fields)) != len(fields)
            or any(field not in FIELD_RANK for field in fields)
        ):
            raise ExactInputError("invalid_exact_clause")
        ordered = tuple(field for field in CLOSED_FIELDS if field in fields)
    else:
        ordered = CLOSED_FIELDS
    if "caseSensitive" in raw and raw["caseSensitive"] is not None:
        case_sensitive = raw["caseSensitive"]
        if type(case_sensitive) is not bool:
            raise ExactInputError("invalid_exact_clause")
    else:
        case_sensitive = True
    return {
        "expression": expression,
        "syntax": syntax,
        "fields": ordered,
        "caseSensitive": case_sensitive,
    }


def resolve_regex_engine(engine: Any | None) -> Any | None:
    """Return an explicit engine, or import ``re2``. ``None`` means unavailable."""
    if engine is not None:
        return engine
    try:
        import re2
    except ImportError:
        return None
    return re2


def compile_regex(clause: Mapping[str, Any], engine: Any) -> Any:
    flags = 0 if clause["caseSensitive"] else engine.IGNORECASE
    try:
        return engine.compile(clause["expression"], flags)
    except Exception as exc:
        raise ExactInputError("invalid_exact_clause") from exc


def execute_exact_lane(
    clause: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]] | None,
    *,
    limit: int,
    snapshot_ref: str,
    compiled_pattern: Any | None = None,
    dependency_unavailable: bool = False,
) -> dict[str, Any]:
    """Filter, match, coalesce, then apply ``limit``.

    ``dependency_unavailable`` publishes no hits. A missing record sequence
    is an internal failure, also with no hits. ``clause`` is canonicalized
    with the same parser as the retrieval entry.
    """
    clause = parse_exact_clause(clause)
    if dependency_unavailable:
        return _lane("terminated", "none", "dependency_unavailable", [], False)
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        return _lane("terminated", "none", "internal_error", [], False)
    try:
        refs = _collect(clause, records, snapshot_ref, compiled_pattern)
    except Exception:
        return _lane("terminated", "none", "internal_error", [], False)
    ordered = sorted(refs, key=_order_key)
    for ref in ordered:
        ref.pop("_sessionOrder", None)
    truncated = len(ordered) > limit
    published = ordered[:limit]
    state = "hits" if published else "empty"
    return _lane("completed", state, None, published, truncated)


def _lane(
    execution: str,
    result: str,
    reason: str | None,
    refs: list[dict[str, Any]],
    truncated: bool,
) -> dict[str, Any]:
    return {
        "currentEvidenceRefs": refs,
        "exactLaneAudit": {
            "executionState": execution,
            "resultState": result,
            "reasonCode": reason,
            "returnedCount": len(refs),
            "truncated": truncated,
            "featureVersions": {
                "offsetUnit": OFFSET_UNIT,
                "maxSpans": str(MAX_SPANS),
                "fragment": FRAGMENT_POLICY,
            },
            "nativeAuditRefs": [],
        },
    }


def _collect(
    clause: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    snapshot_ref: str,
    compiled_pattern: Any | None,
) -> list[dict[str, Any]]:
    session_order: dict[str, int] = {}
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not _authorized(record):
            continue
        field = record.get("matchedField")
        text = record.get("text")
        session = record.get("sourceSessionRef")
        ao_id = record.get("aoId")
        sequence = record.get("sequence")
        if (
            field not in clause["fields"]
            or not isinstance(text, str)
            or not isinstance(session, str)
            or not session
            or not isinstance(ao_id, str)
            or not ao_id
            or type(sequence) is not int
        ):
            continue
        spans = _spans(text, clause, compiled_pattern)
        if not spans:
            continue
        if session not in session_order:
            session_order[session] = len(session_order)
        identity = (session, ao_id, field)
        bucket = grouped.get(identity)
        if bucket is None:
            grouped[identity] = {
                "text": text,
                "spans": list(spans),
                "sequence": sequence,
                "session": session,
                "ao_id": ao_id,
                "field": field,
                "order": session_order[session],
            }
            continue
        if bucket["text"] != text:
            continue
        bucket["spans"].extend(spans)
    refs: list[dict[str, Any]] = []
    for bucket in grouped.values():
        text = bucket["text"]
        unique = sorted(set(bucket["spans"]))
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        listed = unique[:MAX_SPANS]
        refs.append(
            {
                "kind": "current_log_match",
                "sourceSessionRef": bucket["session"],
                "aoId": bucket["ao_id"],
                "sequence": bucket["sequence"],
                "matchedField": bucket["field"],
                "representation": REPRESENTATION[bucket["field"]],
                "contentHash": digest,
                "snapshotRef": snapshot_ref,
                "evidenceRef": f"evidence:{bucket['ao_id']}@{digest[:16]}",
                "spans": [
                    {"start": start, "end": end, "fragment": text[start:end]}
                    for start, end in listed
                ],
                "totalOccurrences": len(unique),
                "spansTruncated": len(unique) > MAX_SPANS,
                "_sessionOrder": bucket["order"],
            }
        )
    return refs


def _authorized(record: Mapping[str, Any]) -> bool:
    return (
        record.get("visible") is True
        and record.get("inScope") is True
        and record.get("withdrawn") is False
    )


def _spans(
    text: str,
    clause: Mapping[str, Any],
    compiled_pattern: Any | None,
) -> list[tuple[int, int]]:
    if clause["syntax"] == "literal":
        return _literal_spans(text, clause["expression"], clause["caseSensitive"])
    if compiled_pattern is None:
        raise RuntimeError("regex pattern is not compiled")
    return [(match.start(), match.end()) for match in compiled_pattern.finditer(text)]


def _literal_spans(text: str, expression: str, case_sensitive: bool) -> list[tuple[int, int]]:
    if case_sensitive:
        haystack = text
        needle = expression
    else:
        haystack = text.casefold()
        needle = expression.casefold()
        if len(haystack) != len(text):
            return _casefold_spans(text, expression)
    spans: list[tuple[int, int]] = []
    start = 0
    step = len(needle) or 1
    while True:
        found = haystack.find(needle, start)
        if found < 0:
            return spans
        spans.append((found, found + len(needle)))
        start = found + step


def _casefold_spans(text: str, expression: str) -> list[tuple[int, int]]:
    needle = expression.casefold()
    spans: list[tuple[int, int]] = []
    for start in range(len(text)):
        for end in range(start + 1, len(text) + 1):
            if text[start:end].casefold() == needle:
                spans.append((start, end))
                break
    return spans


def _order_key(ref: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        ref["_sessionOrder"],
        ref["sequence"],
        FIELD_RANK[ref["matchedField"]],
        ref["spans"][0]["start"],
        ref["sourceSessionRef"],
        ref["aoId"],
    )
