# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ADR-0045 Wave E1: verdict normalization and attribution-mode selection."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from openviking.session.outcome_signal import (
    ATTRIBUTION_MODE_FAIL,
    ATTRIBUTION_MODE_SUCCESS,
    DOMAIN_KINDS,
    FAIL_ERROR_PATTERNS,
    NO_SIGNAL_DETAIL_FLAGS,
    NO_SIGNAL_ERROR_PATTERNS,
    PASS_REWARD_THRESHOLDS,
    OutcomeSignal,
    OutcomeSignalError,
    closed_attribution_mode,
    normalize_verdict,
    select_attribution_mode,
)

_DOMAINS = ("lcb", "swe", "browsecomp", "gdpval")


def _signal(
    domain: str,
    *,
    reward: object,
    graded: object = True,
    error: object = None,
    detail: dict[str, object] | None = None,
) -> OutcomeSignal | None:
    return normalize_verdict(domain, reward=reward, graded=graded, error=error, detail=detail)


@pytest.mark.parametrize("domain", _DOMAINS)
def test_clean_pass_per_domain(domain: str) -> None:
    reward = 0.8 if domain == "gdpval" else 1.0
    found = _signal(domain, reward=reward)
    assert found is not None
    assert found.signal == "pass"
    assert found.source == "verifier"
    assert found.raw["reward"] == reward
    assert found.raw["domain_kind"] == domain


@pytest.mark.parametrize("domain", _DOMAINS)
def test_clean_fail_per_domain(domain: str) -> None:
    reward = 0.3 if domain == "gdpval" else 0.0
    found = _signal(domain, reward=reward)
    assert found is not None
    assert found.signal == "fail"
    assert found.source == "verifier"


def test_gdpval_half_is_pass_and_just_under_is_fail() -> None:
    half = _signal("gdpval", reward=0.50)
    under = _signal("gdpval", reward=0.49)
    assert half is not None and half.signal == "pass"
    assert under is not None and under.signal == "fail"
    assert PASS_REWARD_THRESHOLDS["gdpval"] == 0.5


def test_browsecomp_parse_error_is_no_signal() -> None:
    assert _signal("browsecomp", reward=0.0, error="judge parse_error") is None
    assert _signal("browsecomp", reward=0.0, detail={"parse_error": True}) is None
    assert "parse_error" in NO_SIGNAL_ERROR_PATTERNS["browsecomp"]


def test_gdpval_fallback_five_is_no_signal() -> None:
    assert _signal("gdpval", reward=0.5, error="fallback_5.0") is None
    assert _signal("gdpval", reward=0.5, error="score_parse_failed") is None
    assert _signal("gdpval", reward=0.5, detail={"gdpval_fallback": True}) is None
    assert _signal("gdpval", reward=0.5, detail={"fallback_score": 5.0}) is None
    assert "fallback_5.0" in NO_SIGNAL_ERROR_PATTERNS["gdpval"]
    assert "fallback_score" in NO_SIGNAL_DETAIL_FLAGS["gdpval"]


def test_swe_no_patch_is_fail_even_at_zero_reward() -> None:
    found = _signal("swe", reward=0.0, error="No patch generated")
    assert found is not None
    assert found.signal == "fail"
    assert "No patch generated" in FAIL_ERROR_PATTERNS["swe"]


def test_swe_patch_apply_failed_is_fail() -> None:
    found = _signal("swe", reward=0.0, error="Patch apply failed: hunk 1")
    assert found is not None
    assert found.signal == "fail"
    assert "Patch apply failed" in FAIL_ERROR_PATTERNS["swe"]


def test_lcb_no_code_extracted_is_fail() -> None:
    found = _signal("lcb", reward=0.0, error="no_code_extracted")
    assert found is not None
    assert found.signal == "fail"
    assert "no_code_extracted" in FAIL_ERROR_PATTERNS["lcb"]


@pytest.mark.parametrize("domain", _DOMAINS)
def test_graded_false_is_no_signal(domain: str) -> None:
    reward = 1.0 if domain != "gdpval" else 0.8
    assert _signal(domain, reward=reward, graded=False) is None
    assert _signal(domain, reward=reward, graded=None) is None


def test_verifier_timeout_and_exception_are_no_signal() -> None:
    assert _signal("lcb", reward=0.0, error="verifier timeout after 30s") is None
    assert _signal("swe", reward=0.0, error="verifier_exception: boom") is None
    assert _signal("browsecomp", reward=0.0, detail={"verifier_timeout": True}) is None


def test_tables_cover_the_four_domains() -> None:
    assert set(DOMAIN_KINDS) == set(_DOMAINS)
    assert set(NO_SIGNAL_ERROR_PATTERNS) == set(_DOMAINS)
    assert set(FAIL_ERROR_PATTERNS) == set(_DOMAINS)
    assert set(NO_SIGNAL_DETAIL_FLAGS) == set(_DOMAINS)
    assert set(PASS_REWARD_THRESHOLDS) == set(_DOMAINS)


def test_invalid_domain_is_rejected() -> None:
    with pytest.raises(OutcomeSignalError) as caught:
        normalize_verdict("unknown", reward=1.0, graded=True, error=None)
    assert caught.value.code == "invalid_domain_kind"


def test_outcome_signal_closed_sets() -> None:
    with pytest.raises(OutcomeSignalError) as signal:
        OutcomeSignal(signal="ok", source="verifier", raw={})
    assert signal.value.code == "invalid_signal"
    with pytest.raises(OutcomeSignalError) as source:
        OutcomeSignal(signal="pass", source="guess", raw={})
    assert source.value.code == "invalid_source"
    with pytest.raises(OutcomeSignalError) as raw:
        OutcomeSignal(signal="pass", source="verifier", raw=["x"])  # type: ignore[arg-type]
    assert raw.value.code == "invalid_raw"
    frozen = OutcomeSignal(signal="fail", source="model_inferred", raw={"k": 1})
    assert isinstance(frozen.raw, MappingProxyType)
    assert frozen.raw["k"] == 1


def test_select_attribution_mode_four_rules_and_nones() -> None:
    verifier_fail = OutcomeSignal(signal="fail", source="verifier", raw={})
    verifier_pass = OutcomeSignal(signal="pass", source="verifier", raw={})
    inferred_fail = OutcomeSignal(signal="fail", source="model_inferred", raw={})
    inferred_pass = OutcomeSignal(signal="pass", source="model_inferred", raw={})
    assert select_attribution_mode(verifier_fail) == ATTRIBUTION_MODE_FAIL
    assert select_attribution_mode(verifier_pass) == ATTRIBUTION_MODE_SUCCESS
    assert select_attribution_mode(inferred_fail) == ATTRIBUTION_MODE_FAIL
    assert select_attribution_mode(inferred_pass) is None
    assert select_attribution_mode(None) is None


def test_closed_attribution_mode_is_fail_closed() -> None:
    assert closed_attribution_mode("success") == "success"
    assert closed_attribution_mode("fail") == "fail"
    assert closed_attribution_mode(None) == "fail"
    assert closed_attribution_mode("SUCCESS") == "fail"
    assert closed_attribution_mode("bogus") == "fail"
