# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ADR-0045 Wave E2: synthetic verdict smoke, zero LLM."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.causal_memory.ao_record_export import export_sessions, load_session_records
from benchmark.causal_memory.aig0045_verdict_smoke import (
    ROW_KEYS,
    SCHEMA,
    legal_builder_script,
    load_real_materials,
    run_smoke,
    score_gates,
    synthetic_materials,
    write_report,
)
from openviking.session.aig_toolloop_builder import FakeToolloopProvider
from openviking.session.ao_ledger import AORecord
from openviking.session.outcome_signal import normalize_verdict, select_attribution_mode


def test_synthetic_covers_styles_domains_and_verdict_classes() -> None:
    materials = synthetic_materials()
    assert len(materials) == 6
    assert sum(item.style == "code" for item in materials) == 3
    assert sum(item.style == "doc" for item in materials) == 3
    domains = {item.domain_kind for item in materials}
    assert domains == {"lcb", "swe", "browsecomp", "gdpval"}
    verdicts = set()
    modes = []
    for item in materials:
        outcome = normalize_verdict(
            item.domain_kind,
            reward=item.reward,
            graded=item.graded,
            error=item.error,
            detail=item.detail,
        )
        verdicts.add("no-signal" if outcome is None else outcome.signal)
        modes.append(select_attribution_mode(outcome))
    assert verdicts == {"pass", "fail", "no-signal"}
    assert None in modes
    assert "fail" in modes
    assert "success" in modes


def test_no_signal_does_not_trigger_the_four_roles(tmp_path: Path) -> None:
    materials = [item for item in synthetic_materials() if "nosignal" in item.run_id]
    assert materials
    report = run_smoke(
        materials,
        root=tmp_path,
        builder_factory=lambda mode: FakeToolloopProvider(legal_builder_script(mode)),
    )
    assert report["materials"]
    for row in report["materials"]:
        assert row["verdict"] == "no-signal"
        assert row["mode"] is None
        assert row["submit"] is False
        assert row["fallback_reason"] == "not_triggered"
        assert row["triggered"] is False
        assert row["phi_violations_after_refine"] is None


def test_synthetic_full_chain_with_fake_provider(tmp_path: Path) -> None:
    report = run_smoke(
        synthetic_materials(),
        root=tmp_path,
        builder_factory=lambda mode: FakeToolloopProvider(legal_builder_script(mode)),
    )
    assert report["schema"] == SCHEMA
    rows = report["materials"]
    assert len(rows) == 6
    for row in rows:
        for key in ROW_KEYS:
            assert key in row
    triggered = [row for row in rows if row["verdict"] != "no-signal"]
    silent = [row for row in rows if row["verdict"] == "no-signal"]
    assert silent
    assert all(row["submit"] is True for row in triggered)
    assert all(row["phi_violations_after_refine"] == 0 for row in triggered)
    assert all(row["reader_schema_ok"] is True for row in triggered)
    assert all(row["decisive_refs_in_snapshot"] is True for row in triggered)
    gates = report["gates"]
    assert gates["submit_rate"]["pass"] is True
    assert gates["phi_violations_zero"]["pass"] is True
    assert gates["reader_schema_and_refs"]["pass"] is True
    assert gates["dual_mode_distinguishable"]["pass"] is True
    assert gates["dual_mode_distinguishable"]["sampled_fraction"] >= 0.10
    assert gates["budget_exhausted_rate"]["record_only"] is True
    path = write_report(report, tmp_path / "aig0045_smoke_report.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["schema"] == SCHEMA
    assert [item["run_id"] for item in loaded["materials"]] == [item["run_id"] for item in rows]


def test_dry_run_skips_the_model_and_still_records_no_signal(tmp_path: Path) -> None:
    report = run_smoke(synthetic_materials(), root=tmp_path, dry_run=True)
    assert report["settings"]["dry_run"] is True
    by_id = {row["run_id"]: row for row in report["materials"]}
    assert by_id["syn-lcb-pass"]["mode"] == "success"
    assert by_id["syn-swe-fail"]["mode"] == "fail"
    assert by_id["syn-lcb-nosignal"]["fallback_reason"] == "not_triggered"
    assert all(row["submit"] is False for row in report["materials"])
    assert all(row.get("toolloop_audit") is None for row in report["materials"])


def test_score_gates_schema_and_fail_lines() -> None:
    passing = score_gates(
        [
            {
                "budget_exhausted": False,
                "decisive_refs_in_snapshot": True,
                "mode": "success",
                "phi_violations_after_refine": 0,
                "reader_schema_ok": True,
                "submit": True,
                "verdict": "pass",
                "effect_notes": ["复用促成最终正确结果"],
            },
            {
                "budget_exhausted": False,
                "decisive_refs_in_snapshot": True,
                "mode": "fail",
                "phi_violations_after_refine": 0,
                "reader_schema_ok": True,
                "submit": True,
                "verdict": "fail",
                "effect_notes": ["复用后下游出错"],
            },
            {
                "budget_exhausted": False,
                "decisive_refs_in_snapshot": None,
                "mode": None,
                "phi_violations_after_refine": None,
                "reader_schema_ok": None,
                "submit": False,
                "verdict": "no-signal",
            },
        ]
    )
    assert passing["submit_rate"]["pass"] is True
    assert passing["phi_violations_zero"]["pass"] is True
    dirty = score_gates(
        [
            {
                "budget_exhausted": False,
                "decisive_refs_in_snapshot": True,
                "effect_notes": ["这次复用导致错误"],
                "mode": "success",
                "phi_violations_after_refine": 0,
                "reader_schema_ok": True,
                "submit": True,
                "verdict": "pass",
            }
        ]
    )
    assert dirty["dual_mode_distinguishable"]["pass"] is False
    low_submit = score_gates(
        [
            {
                "budget_exhausted": True,
                "decisive_refs_in_snapshot": None,
                "mode": "fail",
                "phi_violations_after_refine": None,
                "reader_schema_ok": None,
                "submit": False,
                "verdict": "fail",
            }
        ]
    )
    assert low_submit["submit_rate"]["pass"] is False
    assert low_submit["budget_exhausted_rate"]["rate"] == 1.0


def test_score_gates_dual_mode_excludes_success_without_notes() -> None:
    """Submitted-but-unfrozen success rows (no effect_notes) are not the denominator."""
    clean_notes = ["复用促成最终正确结果"]
    excluded = score_gates(
        [
            {
                "budget_exhausted": False,
                "decisive_refs_in_snapshot": None,
                "mode": "success",
                "phi_violations_after_refine": 1,
                "reader_schema_ok": None,
                "submit": True,
                "verdict": "pass",
            },
            {
                "budget_exhausted": False,
                "decisive_refs_in_snapshot": True,
                "effect_notes": list(clean_notes),
                "mode": "success",
                "phi_violations_after_refine": 0,
                "reader_schema_ok": True,
                "submit": True,
                "verdict": "pass",
            },
            {
                "budget_exhausted": False,
                "decisive_refs_in_snapshot": True,
                "effect_notes": list(clean_notes),
                "mode": "success",
                "phi_violations_after_refine": 0,
                "reader_schema_ok": True,
                "submit": True,
                "verdict": "pass",
            },
        ]
    )
    distinguish = excluded["dual_mode_distinguishable"]
    assert distinguish["pass"] is True
    assert distinguish["excluded_without_notes"] == 1
    assert distinguish["sampled"] == 2
    assert distinguish["dirty"] == 0

    marked = score_gates(
        [
            {
                "budget_exhausted": False,
                "decisive_refs_in_snapshot": True,
                "effect_notes": ["这次复用导致错误"],
                "mode": "success",
                "phi_violations_after_refine": 0,
                "reader_schema_ok": True,
                "submit": True,
                "verdict": "pass",
            },
            {
                "budget_exhausted": False,
                "decisive_refs_in_snapshot": True,
                "effect_notes": list(clean_notes),
                "mode": "success",
                "phi_violations_after_refine": 0,
                "reader_schema_ok": True,
                "submit": True,
                "verdict": "pass",
            },
        ]
    )
    assert marked["dual_mode_distinguishable"]["pass"] is False
    assert marked["dual_mode_distinguishable"]["excluded_without_notes"] == 0
    assert marked["dual_mode_distinguishable"]["dirty"] == 1


def test_real_materials_join_exported_jsonl(tmp_path: Path) -> None:
    record = AORecord(
        ao_id="ao-real-0",
        session_id="run-real-1",
        sequence=1,
        action={"participant_id": "worker", "segment_id": "seg", "text": "hello"},
        observation={"text": "hello"},
        skill_invocations=[],
        message_ref=None,
        created_at="2026-10-12T00:00:00.000Z",
        captured_state="live",
    )
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "run-real-1.jsonl").write_text(
        json.dumps(record.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text(
        json.dumps(
            {
                "run_id": "run-real-1",
                "domain_kind": "lcb",
                "reward": 1.0,
                "graded": True,
                "error": None,
                "detail": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    found = load_real_materials(verdicts, sessions)
    assert len(found) == 1
    assert found[0].domain_kind == "lcb"
    assert found[0].records[0].ao_id == "ao-real-0"
    report = run_smoke(found, root=tmp_path / "work", dry_run=True)
    assert report["materials"][0]["mode"] == "success"
    assert report["materials"][0]["submit"] is False


def test_export_derives_text_turns_when_ledger_is_absent(tmp_path: Path) -> None:
    session = tmp_path / "sess-derived"
    session.mkdir()
    (session / "messages.jsonl").write_text(
        json.dumps(
            {
                "created_at": "2026-10-12T00:00:00.000Z",
                "id": "msg-1",
                "parts": [{"text": "please add 1 and 2", "type": "text"}],
                "role": "user",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records, source = load_session_records(session)
    assert source == "derived"
    assert len(records) == 1
    assert records[0].action["text"] == "please add 1 and 2"
    summary = export_sessions(tmp_path, tmp_path / "ao_records.jsonl")
    assert summary["records"] == 1
    assert summary["source_derived_sessions"] == 1
    assert summary["source_ledger_sessions"] == 0


def test_jobs_parallel_rows_match_serial(tmp_path: Path) -> None:
    materials = synthetic_materials()
    factory = lambda mode: FakeToolloopProvider(legal_builder_script(mode))  # noqa: E731
    serial = run_smoke(materials, root=tmp_path / "serial", builder_factory=factory)
    parallel = run_smoke(
        materials, root=tmp_path / "parallel", builder_factory=factory, jobs=4
    )

    def strip_timing(rows):
        rows = json.loads(json.dumps(rows))
        for row in rows:
            for entry in row.get("toolloop_audit") or []:
                entry.pop("elapsed_ms", None)
        return rows

    assert serial["settings"]["jobs"] == 1
    assert parallel["settings"]["jobs"] == 4
    assert strip_timing(parallel["materials"]) == strip_timing(serial["materials"])
    assert parallel["provider_audit"] == serial["provider_audit"]
    assert parallel["gates"] == serial["gates"]


def test_parallel_batch_isolates_driver_exceptions(tmp_path: Path) -> None:
    def factory(mode: str):
        if mode == "success":
            raise RuntimeError("boom")
        return FakeToolloopProvider(legal_builder_script(mode))

    report = run_smoke(synthetic_materials(), root=tmp_path, builder_factory=factory, jobs=3)
    rows = {row["run_id"]: row for row in report["materials"]}
    crashed = [row for row in rows.values() if row["verdict"] == "driver_exception"]
    assert crashed, "success-mode materials must crash the factory"
    for row in crashed:
        assert row["submit"] is False
        assert row["triggered"] is True
        assert row["fallback_reason"] == "driver_exception:RuntimeError"
    survivors = [row for row in rows.values() if row["verdict"] != "driver_exception"]
    assert any(row["submit"] is True for row in survivors), (
        "fail-mode materials must still run to submit"
    )


def test_cli_jobs_flag_records_setting(tmp_path: Path) -> None:
    from benchmark.causal_memory.aig0045_verdict_smoke import main

    out = tmp_path / "report.json"
    code = main(["--material", "synthetic", "--dry-run", "--jobs", "3", "--out", str(out)])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["settings"]["jobs"] == 3


def test_attempts_folds_any_submit_and_reports_detail(tmp_path: Path) -> None:
    import json

    from benchmark.causal_memory.aig0045_verdict_smoke import legal_builder_script
    from openviking.session.aig_toolloop_builder import FakeToolloopProvider

    seen = {"success": 0}

    def factory(mode: str):
        if mode == "success":
            seen["success"] += 1
            if seen["success"] == 1:
                raise RuntimeError("first-attempt daze")
        return FakeToolloopProvider(legal_builder_script(mode))

    report = run_smoke(
        synthetic_materials(), root=tmp_path, builder_factory=factory, attempts=2, jobs=2
    )
    assert report["settings"]["attempts"] == 2
    rows = {row["run_id"]: row for row in report["materials"]}
    success_rows = [row for row in rows.values() if row.get("mode") == "success"]
    assert success_rows
    crashed_attempts = 0
    for row in success_rows:
        assert row["submit"] is True, "one flaky attempt must not fail the material"
        assert len(row["attempts"]) == 2
        crashed_attempts += sum(
            1 for item in row["attempts"] if item["submit"] is False
        )
    assert crashed_attempts >= 1
    for row in rows.values():
        assert len(row["attempts"]) == 2
        for key in ("run_id", "submit", "verdict"):
            assert key in row
    assert json.dumps(report["materials"], ensure_ascii=False)
