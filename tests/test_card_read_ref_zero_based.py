"""Frozen SWE/IR adjudication identity fixtures; no model or storage writes."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from openviking.session.card_proposer import read_ao_ids

CASES = [json.loads(path.read_text()) for path in sorted((Path(__file__).parent / "fixtures/card-read-ref-adjudications").glob("*.json"))]


def objects(case):
    return (NS(blocks=[NS(source_refs=[NS(**r) for r in b["source_refs"]]) for b in case["blocks"]]),
            NS(records=[NS(**r) for r in case["records"]]))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["domain"] + ":" + c["adjudication_id"])
def test_real_adjudication_read_identity(case, monkeypatch):
    revision, snapshot = objects(case)
    refs = case["raw_read_lower_bound"]
    members = {r.ao_id for b in revision.blocks for r in b.source_refs}
    actually_read = {snapshot.records[i].ao_id for i in refs}
    monkeypatch.setenv("OPENVIKING_CARD_READ_REF_ZERO_BASED", "1")
    actual = read_ao_ids(revision, refs, snapshot=snapshot)
    assert set(actual) == actually_read & members
    assert not set(actual) - actually_read  # no unread AO can become registered evidence
    monkeypatch.delenv("OPENVIKING_CARD_READ_REF_ZERO_BASED")
    expected_legacy = tuple(sorted({r.ao_id for b in revision.blocks for r in b.source_refs if r.sequence in set(refs)}))
    assert read_ao_ids(revision, refs, snapshot=snapshot) == expected_legacy
    assert read_ao_ids(revision, refs) == expected_legacy
    monkeypatch.setenv("OPENVIKING_CARD_READ_REF_ZERO_BASED", "0")
    assert read_ao_ids(revision, refs, snapshot=snapshot) == expected_legacy


def test_all_confirmed_rejection_counterexamples_present():
    assert len([c for c in CASES if c["domain"] == "swe"]) == 87
    assert len([c for c in CASES if c["domain"] == "ir"]) == 6  # fifth ao_not_read plus one ao_not_in_block, all read_aos=0
    for domain, count in [("swe", 32), ("ir", 5)]:
        assert sum(c["domain"] == domain and any(r[0] == "ao_not_read" for r in c["rejected"]) for c in CASES) == count


def test_sparse_reordered_sequences_use_positions_then_block_membership(monkeypatch):
    monkeypatch.setenv("OPENVIKING_CARD_READ_REF_ZERO_BASED", "1")
    snapshot = NS(records=[NS(ao_id="a", sequence=70), NS(ao_id="b", sequence=3), NS(ao_id="outside", sequence=42)])
    revision = NS(blocks=[NS(source_refs=[NS(ao_id="a", sequence=70), NS(ao_id="b", sequence=3)])])
    assert read_ao_ids(revision, [0, 0, 2], snapshot=snapshot) == ("a",)
    assert read_ao_ids(revision, [1], snapshot=snapshot) == ("b",)
    assert read_ao_ids(revision, [], snapshot=snapshot) == ()


@pytest.mark.parametrize("bad", [-1, 3, True, False, 1.0, "1", None])
def test_illegal_index_fails_explicitly(bad, monkeypatch):
    monkeypatch.setenv("OPENVIKING_CARD_READ_REF_ZERO_BASED", "1")
    revision, snapshot = objects({"blocks": [], "records": [{"ao_id": str(i), "sequence": i + 1} for i in range(3)]})
    with pytest.raises(ValueError, match="invalid_raw_read_index"):
        read_ao_ids(revision, [bad], snapshot=snapshot)


def test_enabled_requires_snapshot(monkeypatch):
    monkeypatch.setenv("OPENVIKING_CARD_READ_REF_ZERO_BASED", "1")
    with pytest.raises(ValueError, match="raw_read_snapshot_required"):
        read_ao_ids(NS(blocks=[]), [0])


def test_invalid_index_stops_proposer_before_any_storage_or_provider(tmp_path, monkeypatch):
    from dataclasses import replace
    from openviking.session.card_proposer import run_card_proposer
    from openviking.session.outcome_signal import OutcomeSignal
    from tests.test_aig_production_pipeline import _snapshot

    monkeypatch.setenv("OPENVIKING_CARD_READ_REF_ZERO_BASED", "1")
    snapshot = replace(_snapshot(), outcome=OutcomeSignal("pass", "verifier", {"reward": 1.0}))
    root = tmp_path / "must-not-create"

    def provider(_):
        raise AssertionError("invalid evidence reached provider")

    with pytest.raises(ValueError, match="invalid_raw_read_index"):
        run_card_proposer(NS(blocks=[]), snapshot, NS(read_ao_refs=(-1,)), provider, card_root=root, owner_label="test")
    assert not root.exists()


def test_previous_row_cannot_be_registered_as_read(monkeypatch):
    snapshot = NS(records=[NS(ao_id="unread", sequence=1), NS(ao_id="read", sequence=2)])
    revision = NS(blocks=[NS(source_refs=list(snapshot.records))])
    monkeypatch.setenv("OPENVIKING_CARD_READ_REF_ZERO_BASED", "1")
    assert read_ao_ids(revision, [1], snapshot=snapshot) == ("read",)
    monkeypatch.delenv("OPENVIKING_CARD_READ_REF_ZERO_BASED")
    assert read_ao_ids(revision, [1], snapshot=snapshot) == ("unread",)
