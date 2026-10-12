# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Run-scoped AIG role chain: sessions, phase gate, round cap, fallback, evidence."""

from __future__ import annotations

import copy

import pytest

from openviking.session.aig_role_chain import (
    AigConstructionRun,
    AigContractError,
    content_hash,
)

_ROLES = ("builder", "critic", "refiner", "reader")
_DRAFT = {"nodes": ["a"], "meta": {"k": 1}}
_VIOLATIONS = [{"code": "gap", "at": "n1"}]
_REVISION = {"nodes": ["a"], "edges": []}
_REPORT = {"verdict": "supported", "detail": {"b": 1, "a": ["x", "y"]}}


def _sessions(run: AigConstructionRun) -> dict[str, str]:
    return {role: run.session_for(role) for role in _ROLES}


def _assert_code(action, code: str) -> None:
    with pytest.raises(AigContractError) as exc_info:
        action()
    assert exc_info.value.code == code


def _chain_report(run_id: str, report: dict) -> str:
    run = AigConstructionRun(run_id)
    sessions = _sessions(run)
    run.submit_draft(sessions["builder"], copy.deepcopy(_DRAFT))
    run.critique(sessions["critic"], copy.deepcopy(_VIOLATIONS))
    run.refine(sessions["refiner"], copy.deepcopy(_REVISION))
    return run.read_verify(sessions["reader"], copy.deepcopy(report))


def test_default_mode_failure_is_provisional_fallback_not_ig():
    fresh = AigConstructionRun("fresh")
    assert fresh.construction_mode == "native_single_pass_aig"
    assert fresh.construction_mode != "fixed_partition_single_pass_ig"

    built = AigConstructionRun("built")
    built_sessions = _sessions(built)
    built.submit_draft(built_sessions["builder"], {"graph": []})
    failed = built.fail()
    assert failed == {"constructionMode": "fallback", "trust": "provisional"}
    assert built.construction_mode == "fallback"
    assert built.trust == "provisional"
    assert built.construction_mode != "fixed_partition_single_pass_ig"
    failed["constructionMode"] = "fixed_partition_single_pass_ig"
    failed["trust"] = "frozen"
    assert built.construction_mode == "fallback"
    assert built.trust == "provisional"
    assert fresh.construction_mode == "native_single_pass_aig"

    fresh_failed = fresh.fail()
    assert fresh_failed == {"constructionMode": "fallback", "trust": "provisional"}
    assert fresh.construction_mode == "fallback"
    assert fresh.trust == "provisional"
    assert fresh.construction_mode != "fixed_partition_single_pass_ig"

    done = AigConstructionRun("done")
    done_sessions = _sessions(done)
    done.submit_draft(done_sessions["builder"], {"graph": []})
    done.read_verify(done_sessions["reader"], {"verdict": "ok"})
    done.freeze_report(done_sessions["reader"])
    assert done.trust == "frozen"
    assert done.construction_mode == "native_single_pass_aig"
    downgraded = done.fail()
    assert downgraded == {"constructionMode": "fallback", "trust": "provisional"}
    assert done.construction_mode == "fallback"
    assert done.trust == "provisional"
    assert done.construction_mode != "fixed_partition_single_pass_ig"
    _assert_code(
        lambda: done.propose_fork(done_sessions["reader"], {"fork": {"bridge": "late"}}),
        "fallback_is_provisional",
    )
    assert done.construction_mode == "fallback"
    assert done.trust == "provisional"


def test_role_sessions_are_run_scoped_and_disjoint():
    first = AigConstructionRun("run-1")
    second = AigConstructionRun("run-2")
    first_ids = _sessions(first)
    second_ids = _sessions(second)

    assert len(set(first_ids.values())) == 4
    assert len(set(second_ids.values())) == 4
    assert all(isinstance(session_id, str) and session_id for session_id in first_ids.values())
    assert set(first_ids.values()).isdisjoint(second_ids.values())
    for role, session_id in first_ids.items():
        assert session_id == first.session_for(role)
        assert session_id != second.session_for(role)
    assert first.session_for("diagnosis") == first_ids["reader"]
    assert second.session_for("diagnosis") == second_ids["reader"]
    assert first.session_for("diagnosis") != second.session_for("diagnosis")


def test_diagnosis_shares_reader_session_and_memory_is_refused():
    run = AigConstructionRun("run-mem")
    reader = run.session_for("reader")
    assert run.session_for("diagnosis") == reader
    assert reader not in {run.session_for(role) for role in ("builder", "critic", "refiner")}
    _assert_code(
        lambda: run.submit_draft(run.session_for("diagnosis"), {"graph": []}),
        "session_role_mismatch",
    )

    known_ids = [run.session_for(role) for role in _ROLES]
    with pytest.raises(AigContractError) as exc_info:
        run.session_for("memory")
    assert exc_info.value.code == "memory_session_not_shared"
    for session_id in known_ids:
        assert session_id not in str(exc_info.value)
        assert session_id not in repr(exc_info.value)


def test_wrong_role_session_cannot_handoff():
    run = AigConstructionRun("run-role")
    other = AigConstructionRun("run-other")
    own = _sessions(run)
    foreign = _sessions(other)

    for role in ("critic", "refiner", "reader"):
        _assert_code(
            lambda role=role: run.submit_draft(own[role], {"graph": []}),
            "session_role_mismatch",
        )
    _assert_code(
        lambda: run.submit_draft(foreign["builder"], {"graph": []}),
        "session_role_mismatch",
    )
    assert run.submit_draft(own["builder"], {"graph": []}) == content_hash({"graph": []})

    for role in ("builder", "refiner", "reader"):
        _assert_code(
            lambda role=role: run.critique(own[role], [{"code": "x"}]),
            "session_role_mismatch",
        )
    _assert_code(
        lambda: run.critique(foreign["critic"], [{"code": "x"}]),
        "session_role_mismatch",
    )
    assert run.critique(own["critic"], [{"code": "x"}]) == 1

    for role in ("builder", "critic", "reader"):
        _assert_code(
            lambda role=role: run.refine(own[role], {"graph": ["n"]}),
            "session_role_mismatch",
        )
    _assert_code(
        lambda: run.refine(foreign["refiner"], {"graph": ["n"]}),
        "session_role_mismatch",
    )
    assert run.refine(own["refiner"], {"graph": ["n"]}) == content_hash({"graph": ["n"]})

    for role in ("builder", "critic", "refiner"):
        _assert_code(
            lambda role=role: run.read_verify(own[role], {"verdict": "ok"}),
            "session_role_mismatch",
        )
    _assert_code(
        lambda: run.read_verify(foreign["reader"], {"verdict": "ok"}),
        "session_role_mismatch",
    )
    digest = run.read_verify(own["reader"], {"verdict": "ok"})

    for role in ("builder", "critic", "refiner"):
        _assert_code(
            lambda role=role: run.propose_fork(own[role], {"fork": {"bridge": "b"}}),
            "session_role_mismatch",
        )
    _assert_code(
        lambda: run.propose_fork(foreign["reader"], {"fork": {"bridge": "b"}}),
        "session_role_mismatch",
    )
    _assert_code(
        lambda: run.propose_fork(own["reader"], {"fork": {"bridge": "b"}}),
        "report_not_frozen",
    )
    assert run.freeze_report(own["reader"]) == digest
    for role in ("builder", "critic", "refiner"):
        _assert_code(
            lambda role=role: run.propose_fork(own[role], {"fork": {"bridge": "b"}}),
            "session_role_mismatch",
        )
    proposed = run.propose_fork(own["reader"], {"fork": {"bridge": "b"}})
    assert proposed["readerReportHash"] == digest


def test_critic_and_refiner_stop_at_three_rounds():
    def build(run_id: str, *, extra_critic: bool = False, extra_refine: bool = False) -> str:
        run = AigConstructionRun(run_id)
        sessions = _sessions(run)
        run.submit_draft(sessions["builder"], {"v": 0})
        for index in range(3):
            assert run.critique(sessions["critic"], [{"round": index}]) == index + 1
        if extra_critic:
            _assert_code(
                lambda: run.critique(sessions["critic"], [{"round": 99}]),
                "critic_round_limit",
            )
        for index in range(3):
            revised = {"v": index + 1}
            assert run.refine(sessions["refiner"], revised) == content_hash(revised)
        if extra_refine:
            _assert_code(
                lambda: run.refine(sessions["refiner"], {"v": 99}),
                "refiner_round_limit",
            )
        assert run.session_for("critic") == sessions["critic"]
        assert run.session_for("refiner") == sessions["refiner"]
        return run.read_verify(sessions["reader"], {"verdict": "ok"})

    baseline = build("base")
    assert build("extra-critic", extra_critic=True) == baseline
    assert build("extra-refiner", extra_refine=True) == baseline
    assert build("extra-both", extra_critic=True, extra_refine=True) == baseline


def test_read_verify_phase_gate_and_fork_after_freeze():
    run = AigConstructionRun("run-gate")
    sessions = _sessions(run)
    _assert_code(
        lambda: run.propose_fork(sessions["reader"], {"fork": {"bridge": "early"}}),
        "report_not_frozen",
    )

    run.submit_draft(sessions["builder"], {"graph": []})
    run.critique(sessions["critic"], [{"code": "gap"}])
    run.refine(sessions["refiner"], {"graph": ["cause"]})
    _assert_code(
        lambda: run.read_verify(
            sessions["reader"],
            {"verdict": "supported", "fork": {"bridge": "early"}},
        ),
        "read_verify_cannot_propose_fork",
    )
    assert run.reader_report_hash is None
    revised = {"graph": ["cause"]}
    assert run.refine(sessions["refiner"], revised) == content_hash(revised)

    _assert_code(
        lambda: run.propose_fork(sessions["reader"], {"fork": {"bridge": "early"}}),
        "report_not_frozen",
    )
    report_hash = run.read_verify(sessions["reader"], {"verdict": "supported", "evidence": ["e1"]})
    assert run.phase == "READ_VERIFY"
    assert report_hash == run.read_verify(
        sessions["reader"],
        {"evidence": ["e1"], "verdict": "supported"},
    )
    _assert_code(
        lambda: run.read_verify(
            sessions["reader"],
            {"verdict": "supported", "fork": {"bridge": "sneak"}},
        ),
        "read_verify_cannot_propose_fork",
    )
    assert run.reader_report_hash == report_hash
    _assert_code(
        lambda: run.propose_fork(sessions["reader"], {"fork": {"bridge": "early"}}),
        "report_not_frozen",
    )
    _assert_code(
        lambda: run.submit_draft(sessions["builder"], {"graph": ["tamper"]}),
        "frozen_view",
    )
    _assert_code(
        lambda: run.critique(sessions["critic"], [{"code": "late"}]),
        "frozen_view",
    )
    _assert_code(
        lambda: run.refine(sessions["refiner"], {"graph": ["tamper"]}),
        "frozen_view",
    )

    assert run.freeze_report(sessions["reader"]) == report_hash
    assert run.phase == "SYNTHESIZE_PROPOSE"
    _assert_code(
        lambda: run.read_verify(sessions["reader"], {"verdict": "rewritten"}),
        "frozen_view",
    )
    assert run.reader_report_hash == report_hash

    proposed = run.propose_fork(
        run.session_for("diagnosis"),
        {"fork": {"bridge": "cut-blue"}, "readerReportHash": "client-supplied"},
    )
    assert proposed["readerReportHash"] == report_hash
    assert proposed["fork"] == {"bridge": "cut-blue"}
    assert run.reader_report_hash == report_hash
    assert run.construction_mode == "native_single_pass_aig"
    assert run.construction_mode != "fixed_partition_single_pass_ig"

    proposed["fork"]["bridge"] = "mutated"
    proposed["readerReportHash"] = "deadbeef"
    _assert_code(
        lambda: run.submit_draft(sessions["builder"], {"graph": ["after-fork"]}),
        "frozen_view",
    )
    _assert_code(
        lambda: run.critique(sessions["critic"], [{"code": "after-fork"}]),
        "frozen_view",
    )
    _assert_code(
        lambda: run.refine(sessions["refiner"], {"graph": ["after-fork"]}),
        "frozen_view",
    )
    _assert_code(
        lambda: run.read_verify(sessions["reader"], {"verdict": "rewritten"}),
        "frozen_view",
    )
    assert run.reader_report_hash == report_hash
    again = run.propose_fork(sessions["reader"], {"fork": {"bridge": "cut-blue"}})
    assert again == {"fork": {"bridge": "cut-blue"}, "readerReportHash": report_hash}


def test_reader_report_content_hash_is_stable():
    swapped = {"detail": {"a": ["x", "y"], "b": 1}, "verdict": "supported"}
    first = _chain_report("run-h1", _REPORT)
    second = _chain_report("run-h2", swapped)
    assert first == second
    assert len(first) == 64
    assert all(char in "0123456789abcdef" for char in first)

    changed = {"verdict": "unsupported", "detail": {"b": 1, "a": ["x", "y"]}}
    assert _chain_report("run-h3", changed) != first
    reordered = {"verdict": "supported", "detail": {"b": 1, "a": ["y", "x"]}}
    assert _chain_report("run-h4", reordered) != first

    run = AigConstructionRun("run-h5")
    sessions = _sessions(run)
    mutable_draft = copy.deepcopy(_DRAFT)
    run.submit_draft(sessions["builder"], mutable_draft)
    mutable_draft["nodes"].append("tamper")
    mutable_draft["meta"]["k"] = 2
    violations = copy.deepcopy(_VIOLATIONS)
    run.critique(sessions["critic"], violations)
    violations[0]["code"] = "tampered"
    mutable_revision = copy.deepcopy(_REVISION)
    run.refine(sessions["refiner"], mutable_revision)
    mutable_revision["edges"].append("tamper")
    mutable_report = {"verdict": "supported", "detail": {"n": 1}}
    digest = run.read_verify(sessions["reader"], mutable_report)
    assert digest == run.read_verify(
        sessions["reader"],
        {"detail": {"n": 1}, "verdict": "supported"},
    )
    mutable_report["detail"]["n"] = 5
    assert run.reader_report_hash == digest
    assert digest == _chain_report("run-h6", {"verdict": "supported", "detail": {"n": 1}})


def _open_read_verify(run_id: str) -> tuple[AigConstructionRun, dict[str, str], str]:
    run = AigConstructionRun(run_id)
    sessions = _sessions(run)
    run.submit_draft(sessions["builder"], {"graph": []})
    run.critique(sessions["critic"], [{"code": "gap"}])
    run.refine(sessions["refiner"], {"graph": ["n"]})
    report_hash = run.read_verify(sessions["reader"], {"verdict": "ok"})
    assert run.phase == "READ_VERIFY"
    assert run.evidence_tier is None
    return run, sessions, report_hash


def test_evidence_exposure_climbs_one_tier_and_never_returns_raw_ao():
    run, sessions, report_hash = _open_read_verify("run-ev")
    other, other_sessions, _other_hash = _open_read_verify("run-ev-other")
    reader = sessions["reader"]
    diagnosis = run.session_for("diagnosis")
    assert diagnosis == reader

    for skipped in ("summary", "synopsis", "selected_raw"):
        _assert_code(lambda skipped=skipped: run.escalate_evidence(reader, skipped), "evidence_tier_skip")
        assert run.evidence_tier is None
    _assert_code(lambda: run.escalate_evidence(diagnosis, "selected_raw"), "evidence_tier_skip")
    _assert_code(lambda: run.escalate_evidence(reader, "raw-ao-body"), "invalid_evidence_tier")
    _assert_code(lambda: run.escalate_evidence(reader, "selected raw evidence"), "invalid_evidence_tier")
    secret = "ao-secret-body-should-not-leak"
    with pytest.raises(AigContractError) as exc_info:
        run.escalate_evidence(reader, secret)
    assert exc_info.value.code == "invalid_evidence_tier"
    assert secret not in str(exc_info.value)
    assert secret not in repr(exc_info.value)
    assert run.evidence_tier is None
    assert other.evidence_tier is None

    assert run.escalate_evidence(reader, "projection") == "projection"
    assert run.evidence_tier == "projection"
    _assert_code(lambda: run.escalate_evidence(reader, "projection"), "evidence_tier_skip")
    _assert_code(lambda: run.escalate_evidence(diagnosis, "synopsis"), "evidence_tier_skip")
    _assert_code(lambda: run.escalate_evidence(reader, "selected_raw"), "evidence_tier_skip")
    assert run.evidence_tier == "projection"
    assert other.evidence_tier is None

    _assert_code(
        lambda: other.escalate_evidence(reader, "summary"),
        "session_role_mismatch",
    )
    assert other.escalate_evidence(other_sessions["reader"], "projection") == "projection"
    assert run.evidence_tier == "projection"
    assert other.evidence_tier == "projection"

    assert run.escalate_evidence(diagnosis, "summary") == "summary"
    _assert_code(lambda: run.escalate_evidence(reader, "selected_raw"), "evidence_tier_skip")
    assert run.evidence_tier == "summary"
    assert run.escalate_evidence(reader, "synopsis") == "synopsis"
    _assert_code(lambda: run.escalate_evidence(diagnosis, "summary"), "evidence_tier_skip")
    granted = run.escalate_evidence(diagnosis, "selected_raw")
    assert granted == "selected_raw"
    assert isinstance(granted, str)
    assert run.evidence_tier == "selected_raw"
    assert granted == run.evidence_tier
    _assert_code(lambda: run.escalate_evidence(reader, "selected_raw"), "evidence_tier_skip")
    _assert_code(lambda: run.escalate_evidence(reader, "projection"), "evidence_tier_skip")
    assert run.evidence_tier == "selected_raw"
    assert run.reader_report_hash == report_hash
    assert run.phase == "READ_VERIFY"
    assert run.trust == "draft"
    assert run.construction_mode == "native_single_pass_aig"
    assert "ao" not in granted
    for value in vars(run).values():
        if isinstance(value, str):
            assert secret not in value
            assert "ao-secret" not in value


def test_evidence_escalation_only_reader_during_read_verify():
    early = AigConstructionRun("run-ev-early")
    early_sessions = _sessions(early)
    _assert_code(
        lambda: early.escalate_evidence(early_sessions["reader"], "projection"),
        "evidence_not_in_read_verify",
    )
    _assert_code(
        lambda: early.escalate_evidence(early_sessions["builder"], "projection"),
        "session_role_mismatch",
    )
    assert early.evidence_tier is None
    early.submit_draft(early_sessions["builder"], {"graph": []})
    _assert_code(
        lambda: early.escalate_evidence(early.session_for("diagnosis"), "projection"),
        "evidence_not_in_read_verify",
    )
    _assert_code(
        lambda: early.read_verify(
            early_sessions["reader"],
            {"verdict": "supported", "fork": {"bridge": "early"}},
        ),
        "read_verify_cannot_propose_fork",
    )
    _assert_code(
        lambda: early.escalate_evidence(early_sessions["reader"], "projection"),
        "evidence_not_in_read_verify",
    )
    assert early.phase != "READ_VERIFY"
    assert early.evidence_tier is None

    run, sessions, report_hash = _open_read_verify("run-ev-gate")
    for role in ("builder", "critic", "refiner"):
        _assert_code(
            lambda role=role: run.escalate_evidence(sessions[role], "projection"),
            "session_role_mismatch",
        )
    _assert_code(lambda: run.session_for("memory"), "memory_session_not_shared")
    _assert_code(lambda: run.escalate_evidence("memory", "projection"), "session_role_mismatch")
    foreign = AigConstructionRun("run-ev-foreign")
    _assert_code(
        lambda: run.escalate_evidence(foreign.session_for("diagnosis"), "projection"),
        "session_role_mismatch",
    )
    assert run.evidence_tier is None
    assert run.escalate_evidence(sessions["reader"], "projection") == "projection"

    run.read_verify(sessions["reader"], {"verdict": "still-ok"})
    assert run.evidence_tier == "projection"
    assert run.reader_report_hash != report_hash
    _assert_code(
        lambda: run.propose_fork(sessions["reader"], {"fork": {"bridge": "during-verify"}}),
        "report_not_frozen",
    )

    assert run.freeze_report(sessions["reader"]) == run.reader_report_hash
    _assert_code(
        lambda: run.escalate_evidence(run.session_for("diagnosis"), "summary"),
        "frozen_view",
    )
    _assert_code(
        lambda: run.escalate_evidence(sessions["builder"], "summary"),
        "session_role_mismatch",
    )
    assert run.evidence_tier == "projection"
    assert run.phase == "SYNTHESIZE_PROPOSE"

    downgraded = run.fail()
    assert downgraded == {"constructionMode": "fallback", "trust": "provisional"}
    _assert_code(
        lambda: run.escalate_evidence(sessions["reader"], "summary"),
        "fallback_is_provisional",
    )
    assert run.evidence_tier == "projection"
    assert run.construction_mode == "fallback"

    provisional = AigConstructionRun("run-ev-fallback")
    provisional_sessions = _sessions(provisional)
    provisional.submit_draft(provisional_sessions["builder"], {"graph": []})
    provisional.read_verify(provisional_sessions["reader"], {"verdict": "ok"})
    assert provisional.escalate_evidence(provisional_sessions["reader"], "projection") == "projection"
    assert provisional.escalate_evidence(provisional.session_for("diagnosis"), "summary") == "summary"
    assert provisional.fail() == {"constructionMode": "fallback", "trust": "provisional"}
    _assert_code(
        lambda: provisional.escalate_evidence(provisional_sessions["reader"], "synopsis"),
        "fallback_is_provisional",
    )
    _assert_code(
        lambda: provisional.escalate_evidence(provisional_sessions["critic"], "synopsis"),
        "session_role_mismatch",
    )
    assert provisional.evidence_tier == "summary"
    assert provisional.phase == "FALLBACK"


def test_outcome_write_is_reader_only_and_critic_violation_is_not_failure():
    run = AigConstructionRun("run-outcome-auth")
    sessions = _sessions(run)
    for role in ("builder", "critic", "refiner"):
        _assert_code(
            lambda role=role: run.authorize_outcome_write(sessions[role]),
            "outcome_write_forbidden",
        )
    _assert_code(
        lambda: run.authorize_outcome_write(sessions["reader"]),
        "report_not_frozen",
    )

    run.submit_draft(sessions["builder"], {"graph": []})
    assert run.critique(sessions["critic"], [{"code": "gap", "severity": "high"}]) == 1
    assert run.trust == "draft"
    assert run.phase == "CRITIQUE"
    assert run.reader_report_hash is None
    _assert_code(
        lambda: run.authorize_outcome_write(sessions["critic"]),
        "outcome_write_forbidden",
    )
    _assert_code(
        lambda: run.authorize_outcome_write(sessions["reader"]),
        "report_not_frozen",
    )

    run.refine(sessions["refiner"], {"graph": ["n"]})
    report_hash = run.read_verify(sessions["reader"], {"verdict": "supported"})
    _assert_code(
        lambda: run.authorize_outcome_write(sessions["reader"]),
        "report_not_frozen",
    )
    assert run.freeze_report(sessions["reader"]) == report_hash
    assert run.phase == "SYNTHESIZE_PROPOSE"
    assert run.trust == "frozen"
    assert run.authorize_outcome_write(sessions["reader"]) == report_hash
    assert run.authorize_outcome_write(run.session_for("diagnosis")) == report_hash
    assert run.phase == "SYNTHESIZE_PROPOSE"
    assert run.trust == "frozen"
    assert run.reader_report_hash == report_hash
    for role in ("builder", "critic", "refiner"):
        _assert_code(
            lambda role=role: run.authorize_outcome_write(sessions[role]),
            "outcome_write_forbidden",
        )

    foreign = AigConstructionRun("run-outcome-other")
    _assert_code(
        lambda: run.authorize_outcome_write(foreign.session_for("reader")),
        "session_role_mismatch",
    )
    assert run.fail() == {"constructionMode": "fallback", "trust": "provisional"}
    _assert_code(
        lambda: run.authorize_outcome_write(sessions["reader"]),
        "fallback_is_provisional",
    )
    _assert_code(
        lambda: run.authorize_outcome_write(sessions["critic"]),
        "outcome_write_forbidden",
    )
    assert run.reader_report_hash == report_hash
    assert run.construction_mode == "fallback"
