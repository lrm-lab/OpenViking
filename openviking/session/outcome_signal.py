# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Verifier verdict → frozen OutcomeSignal (ADR-0045). Zero LLM.

``normalize_verdict`` is a pure function: the same sidecar/attempt fields
always recompute the same signal. Judge-side misses (parse failure, verifier
timeout/exception, ``graded=false``) are no-signal. Agent-side misses (no
patch, no extracted code) are fail. Pass thresholds are policy, not protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from openviking.session.aig_native_protocol import canonical_dumps

DOMAIN_KINDS = frozenset({"lcb", "swe", "browsecomp", "gdpval"})
OUTCOME_SIGNALS = frozenset({"pass", "fail"})
OUTCOME_SOURCES = frozenset({"verifier", "model_inferred"})
ATTRIBUTION_MODE_FAIL = "fail"
ATTRIBUTION_MODE_SUCCESS = "success"
ATTRIBUTION_MODES = frozenset({ATTRIBUTION_MODE_FAIL, ATTRIBUTION_MODE_SUCCESS})
OUTCOME_SOURCE_VERIFIER = "verifier"
OUTCOME_SOURCE_MODEL_INFERRED = "model_inferred"
OUTCOME_DIGEST_LIMIT = 200

PASS_REWARD_THRESHOLDS: Mapping[str, float] = {
    "browsecomp": 1.0,
    "gdpval": 0.5,
    "lcb": 1.0,
    "swe": 1.0,
}

# Judge/verifier never finished. Substring match against the caller-supplied
# error text. The driver copies these from sidecar/attempt rows.
NO_SIGNAL_ERROR_PATTERNS: Mapping[str, frozenset[str]] = {
    "browsecomp": frozenset({"parse_error", "timeout", "verifier_error", "verifier_exception"}),
    "gdpval": frozenset(
        {
            "fallback_5.0",
            "parse_error",
            "score_parse_failed",
            "timeout",
            "verifier_error",
            "verifier_exception",
        }
    ),
    "lcb": frozenset({"timeout", "verifier_error", "verifier_exception"}),
    "swe": frozenset({"timeout", "verifier_error", "verifier_exception"}),
}

# Agent delivered nothing the grader can score. These stay fail even at
# reward=0.0; they are the highest-signal real failures.
FAIL_ERROR_PATTERNS: Mapping[str, frozenset[str]] = {
    "browsecomp": frozenset(),
    "gdpval": frozenset(),
    "lcb": frozenset({"no_code_extracted"}),
    "swe": frozenset({"No patch generated", "Patch apply failed"}),
}

# Truthy detail flags (or ``fallback_score`` == 5 / 5.0) also mean no-signal.
NO_SIGNAL_DETAIL_FLAGS: Mapping[str, frozenset[str]] = {
    "browsecomp": frozenset({"parse_error", "verifier_exception", "verifier_timeout"}),
    "gdpval": frozenset(
        {
            "fallback_score",
            "gdpval_fallback",
            "parse_error",
            "score_parse_failed",
            "verifier_exception",
            "verifier_timeout",
        }
    ),
    "lcb": frozenset({"verifier_exception", "verifier_timeout"}),
    "swe": frozenset({"verifier_exception", "verifier_timeout"}),
}


class OutcomeSignalError(ValueError):
    """An outcome value was rejected. ``code`` is stable."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class OutcomeSignal:
    """Closed-set pass/fail plus who asserted it, with opaque grader detail."""

    signal: str
    source: str
    raw: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.signal not in OUTCOME_SIGNALS:
            raise OutcomeSignalError("invalid_signal")
        if self.source not in OUTCOME_SOURCES:
            raise OutcomeSignalError("invalid_source")
        if isinstance(self.raw, (str, bytes)) or not isinstance(self.raw, Mapping):
            raise OutcomeSignalError("invalid_raw")
        object.__setattr__(self, "raw", MappingProxyType(dict(self.raw)))


def closed_attribution_mode(mode: str | None) -> str:
    """Fail-closed: only ``success`` selects the success narrative."""
    if mode == ATTRIBUTION_MODE_SUCCESS:
        return ATTRIBUTION_MODE_SUCCESS
    return ATTRIBUTION_MODE_FAIL


def select_attribution_mode(outcome: OutcomeSignal | None) -> str | None:
    """Pick fail/success, or None when the four roles must not run.

    verifier+fail → fail; verifier+pass → success; model_inferred+fail →
    fail; model_inferred+pass → None (inferred success must not trigger
    success attribution); None → None.
    """
    if outcome is None:
        return None
    if not isinstance(outcome, OutcomeSignal):
        raise OutcomeSignalError("invalid_outcome")
    if outcome.signal == "fail":
        return ATTRIBUTION_MODE_FAIL
    if outcome.source == OUTCOME_SOURCE_VERIFIER:
        return ATTRIBUTION_MODE_SUCCESS
    return None


def normalize_verdict(
    domain_kind: str,
    *,
    reward: Any,
    graded: Any,
    error: Any,
    detail: Mapping[str, Any] | None = None,
) -> OutcomeSignal | None:
    """Map one domain verdict onto pass/fail, or None when there is no signal."""
    if domain_kind not in DOMAIN_KINDS:
        raise OutcomeSignalError("invalid_domain_kind")
    raw = _raw_payload(domain_kind, reward=reward, graded=graded, error=error, detail=detail)
    if graded is not True:
        return None
    if _is_no_signal(domain_kind, error=error, detail=detail):
        return None
    if _error_matches(error, FAIL_ERROR_PATTERNS[domain_kind]):
        return OutcomeSignal(signal="fail", source=OUTCOME_SOURCE_VERIFIER, raw=raw)
    score = _as_reward(reward)
    if score is None:
        return None
    signal = "pass" if score >= PASS_REWARD_THRESHOLDS[domain_kind] else "fail"
    return OutcomeSignal(signal=signal, source=OUTCOME_SOURCE_VERIFIER, raw=raw)


def outcome_overview(outcome: OutcomeSignal) -> dict[str, str]:
    """Compact read-only summary for ``get_overview``."""
    if not isinstance(outcome, OutcomeSignal):
        raise OutcomeSignalError("invalid_outcome")
    return {
        "digest": outcome_digest(outcome.raw),
        "signal": outcome.signal,
        "source": outcome.source,
    }


def outcome_digest(raw: Mapping[str, Any]) -> str:
    """One-line canonical summary of ``raw``, at most 200 characters."""
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Mapping):
        raise OutcomeSignalError("invalid_raw")
    try:
        text = canonical_dumps(_jsonable(raw))
    except (TypeError, ValueError):
        text = str(dict(raw))
    text = text.replace("\n", " ").replace("\r", " ")
    if len(text) > OUTCOME_DIGEST_LIMIT:
        return text[: OUTCOME_DIGEST_LIMIT - 3] + "..."
    return text


def outcome_fingerprint_body(outcome: OutcomeSignal | None) -> dict[str, Any] | None:
    """Canonical outcome payload for ``snapshot_fingerprint``. None is omitted."""
    if outcome is None:
        return None
    if not isinstance(outcome, OutcomeSignal):
        raise OutcomeSignalError("invalid_outcome")
    return {
        "raw": _jsonable(outcome.raw),
        "signal": outcome.signal,
        "source": outcome.source,
    }


def _raw_payload(
    domain_kind: str,
    *,
    reward: Any,
    graded: Any,
    error: Any,
    detail: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if isinstance(detail, Mapping) and not isinstance(detail, (str, bytes)):
        payload.update(dict(detail))
    payload["domain_kind"] = domain_kind
    payload["graded"] = graded
    payload["reward"] = reward
    if error is not None:
        payload["error"] = error
    return payload


def _is_no_signal(domain_kind: str, *, error: Any, detail: Mapping[str, Any] | None) -> bool:
    if _error_matches(error, NO_SIGNAL_ERROR_PATTERNS[domain_kind]):
        return True
    return _detail_says_no_signal(domain_kind, detail)


def _detail_says_no_signal(domain_kind: str, detail: Mapping[str, Any] | None) -> bool:
    if not isinstance(detail, Mapping) or isinstance(detail, (str, bytes)):
        return False
    for key in NO_SIGNAL_DETAIL_FLAGS[domain_kind]:
        if key not in detail:
            continue
        value = detail[key]
        if key == "fallback_score":
            if value == 5 or value == 5.0:
                return True
            continue
        if value:
            return True
    return False


def _error_matches(error: Any, patterns: frozenset[str]) -> bool:
    if error is None:
        return False
    text = error if isinstance(error, str) else str(error)
    if not text:
        return False
    return any(pattern in text for pattern in patterns)


def _as_reward(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping) and not isinstance(value, (str, bytes)):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
