# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""P05: BranchStepRef, server-derived StrategyPattern, and the evidence ledger.

Pattern identity is the digest of the versioned controlled fields. A semantic
neighbor is a different pattern. BranchEvaluation rows are append-only;
analogical evidence is never rewritten as direct evidence. Retired rows leave
the precedent candidate list and stay in the audit log.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from openviking.session.aig_native_protocol import (
    OBJECT_KIND_BRANCH_EVALUATION,
    OBJECT_KIND_FORK_NODE_OVERLAY,
    OBJECT_KIND_STRATEGY_PATTERN_IDENTITY,
    PATTERN_IDENTITY_VERSION,
    REPLAY_REVISION_PREFIX,
    SCHEMA_VERSION_AIG_NATIVE_V2,
    STEP_KIND_IMAGINED,
    BaseRevisionCatalog,
    BranchStepRef,
    ForkNodeOverlay,
    GuidanceStepAlignment,
    ProtocolContractError,
    direct_evidence_admission_document,
    read_first_action_alignment,
    strategy_pattern_document,
    strategy_pattern_id,
)
from openviking.session.fork_node_overlay import (
    FORK_NODE_OVERLAY_DIRNAME,
    LegacyForkGrounding,
)
from openviking.session.strategy_pattern import (
    STRATEGY_PATTERN_DIRNAME,
    FrozenBranch,
    PrecedentCandidate,
    StrategyPatternError,
    StrategyPatternStore,
)
from tests.test_fork_node_overlay import _native_revision, _proposal
from tests.test_fork_node_overlay import _store as _overlay_store

_DIRECT_TIERS = ("production-direct", "replay-direct")


def _spec_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _spec_hash(object_kind: str, body: dict) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
        "object_kind": object_kind,
        "body": body,
    }
    return hashlib.sha256(_spec_dumps(payload).encode("utf-8")).hexdigest()


def _document(object_kind: str, body: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
        "object_kind": object_kind,
        "body": body,
        "content_hash": _spec_hash(object_kind, body),
    }


def _catalog(
    revision: str = "rev-1",
    blocks: tuple[str, ...] = ("b-1", "b-2"),
    block_hash: str = "61b024e1db7c82bc1a3e521938ab76d1da6707c1318bd0ef8dcb09b51645c436",
) -> BaseRevisionCatalog:
    return BaseRevisionCatalog(
        base_revision_id=revision,
        block_content_hash=block_hash,
        block_ids=blocks,
    )


def _overlay(
    catalog: BaseRevisionCatalog,
    fork_node_id: str = "fork-a",
    imagined: tuple[str, ...] = ("img-retry",),
    *,
    factual: tuple[str, ...] = ("b-1",),
    anchor: str = "b-2",
) -> ForkNodeOverlay:
    steps: list[BranchStepRef] = [
        BranchStepRef(step_kind="factual_block_ref", block_id=block_id) for block_id in factual
    ]
    steps.extend(
        BranchStepRef(step_kind="imagined_node", imagined_node_id=node_id) for node_id in imagined
    )
    return ForkNodeOverlay(
        schema_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        object_kind=OBJECT_KIND_FORK_NODE_OVERLAY,
        content_hash="overlay-hash",
        fork_node_id=fork_node_id,
        base_revision_id=catalog.base_revision_id,
        base_revision_block_content_hash=catalog.block_content_hash,
        anchor_block_id=anchor,
        branch_steps=tuple(steps),
    )


def _conditions() -> list[dict[str, str]]:
    return [
        {"condition_id": "anchor-state", "statement": "anchor block is the failed action"},
        {"condition_id": "failure-mode", "statement": "timeout on the first call"},
    ]


def _core(**overrides: object) -> dict:
    core: dict = {
        "applicability_conditions": _conditions(),
        "capability": "replay-sandbox",
        "expected_metric": "error_rate<0.01",
        "intervention_type": "narrow_retry",
        "mechanism": "retry the call with a narrower timeout",
        "target_boundary": "blk-action",
    }
    core.update(overrides)
    return core


def _spec_pattern_id(core: dict) -> str:
    conditions = sorted(core["applicability_conditions"], key=lambda item: item["condition_id"])
    identity = {
        "applicability_conditions": conditions,
        "capability": core["capability"],
        "expected_metric": core["expected_metric"],
        "identity_version": "strategy-pattern-identity-v1",
        "intervention_type": core["intervention_type"],
        "mechanism": core["mechanism"],
        "target_boundary": core["target_boundary"],
    }
    return _spec_hash(OBJECT_KIND_STRATEGY_PATTERN_IDENTITY, identity)


def _proposal_payload(
    overlay: ForkNodeOverlay,
    imagined_node_id: str,
    core: dict | None = None,
    summary: str | None = "a narrower retry after timeout",
) -> dict:
    fields = _core() if core is None else core
    body: dict = {
        "applicability_conditions": fields["applicability_conditions"],
        "author_role": "reader",
        "capability": fields["capability"],
        "expected_metric": fields["expected_metric"],
        "fork_node_id": overlay.fork_node_id,
        "imagined_node_id": imagined_node_id,
        "intervention_type": fields["intervention_type"],
        "mechanism": fields["mechanism"],
        "target_boundary": fields["target_boundary"],
    }
    if summary is not None:
        body["semantic_summary"] = summary
    return _document("strategy_pattern_proposal", body)


def _evaluation(
    subject_kind: str,
    subject_id: str,
    evaluation_id: str,
    evidence_ref: str,
    tier: str,
    status: str = "active",
) -> dict:
    return _document(
        OBJECT_KIND_BRANCH_EVALUATION,
        {
            "evaluation_id": evaluation_id,
            "evidence_ref": evidence_ref,
            "evidence_tier": tier,
            "status": status,
            "subject_id": subject_id,
            "subject_kind": subject_kind,
        },
    )


def _direct_admission(imagined_node_id: str, *, outcome: bool = False) -> dict:
    """A reader-valid replay-direct admission for one imagined node."""
    block_id = "rb-admitted"
    alignment = read_first_action_alignment(
        _document(
            "first_action_alignment",
            {
                "actual_block_id": block_id,
                "alignment": "equivalent",
                "depth": "prefix" if outcome else "first_action",
                "imagined_node_id": imagined_node_id,
                "intervention_type": "narrow_retry",
                "required_capabilities": ["replay-sandbox"],
                "safety_control_params": {"isolation": "sandbox"},
                "target_boundary": "b-2",
            },
        )
    )
    steps = ()
    if outcome:
        steps = (GuidanceStepAlignment(imagined_node_id, "equivalent", block_id),)
    return direct_evidence_admission_document(
        fork_node_id="fork-admission",
        base_revision_id="rev-admission",
        replay_run_id="run-admission",
        replay_revision_id=REPLAY_REVISION_PREFIX + ("ab" * 32),
        alignment=alignment,
        expected_intervention_type="narrow_retry",
        expected_target_boundary="b-2",
        expected_required_capabilities=("replay-sandbox",),
        expected_safety_control_params={"isolation": "sandbox"},
        first_imagined_node=True,
        route="replay",
        step_alignments=steps,
        branch_steps=(
            BranchStepRef(step_kind=STEP_KIND_IMAGINED, imagined_node_id=imagined_node_id),
        ),
        replay_block_ids=(block_id,),
    )


def _store(tmp_path: Path) -> StrategyPatternStore:
    return StrategyPatternStore(tmp_path)


def _index(tmp_path: Path) -> list[dict]:
    path = tmp_path / STRATEGY_PATTERN_DIRNAME / "index.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _pattern_files(tmp_path: Path) -> list[Path]:
    directory = tmp_path / STRATEGY_PATTERN_DIRNAME / "patterns"
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def _evaluation_files(tmp_path: Path) -> list[Path]:
    directory = tmp_path / STRATEGY_PATTERN_DIRNAME / "evaluations"
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.glob("*.json"))


def _imagined_files(tmp_path: Path) -> list[Path]:
    directory = tmp_path / STRATEGY_PATTERN_DIRNAME / "imagined"
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def _code(call) -> str:
    with pytest.raises(StrategyPatternError) as exc:
        call()
    return exc.value.code


def test_factual_block_ref_is_limited_to_the_same_base_revision(tmp_path: Path) -> None:
    store = _store(tmp_path)
    catalog = _catalog()
    overlay = _overlay(catalog)
    other_revision = _catalog(revision="rev-other")
    with pytest.raises(ProtocolContractError, match="base_revision_id"):
        store.freeze_branch(overlay, other_revision)

    foreign = _overlay(catalog, factual=("b-missing",), anchor="b-2")
    lying = _catalog(blocks=("b-2",))
    with pytest.raises(ProtocolContractError, match="factual_block_ref"):
        store.freeze_branch(foreign, lying)

    collision = _overlay(catalog, imagined=("b-1",))
    with pytest.raises(ProtocolContractError, match="branch-local"):
        store.freeze_branch(collision, catalog)

    duplicate = _overlay(catalog, imagined=("img-retry", "img-retry"))
    assert _code(lambda: store.freeze_branch(duplicate, catalog)) == "imagined_node_conflict"
    assert not (tmp_path / STRATEGY_PATTERN_DIRNAME).exists()


def test_imagined_node_stays_counterfactual_when_evidence_is_appended(tmp_path: Path) -> None:
    revision = _native_revision()
    overlay_store = _overlay_store(tmp_path)
    catalog = overlay_store.register_base_revision(revision, acl_labels=("user:alice",))
    stored = overlay_store.submit_proposal(
        _proposal(catalog, document_hash=revision.content_hash),
        principal_labels=("user:alice",),
    ).stored
    overlay_path = tmp_path / FORK_NODE_OVERLAY_DIRNAME / "overlays" / f"{stored.fork_node_id}.json"
    catalog_path = next((tmp_path / FORK_NODE_OVERLAY_DIRNAME / "base-revisions").glob("*.json"))
    overlay_before = overlay_path.read_bytes()
    catalog_before = catalog_path.read_bytes()

    store = _store(tmp_path)
    frozen = store.freeze_branch(stored, catalog)
    assert isinstance(frozen, FrozenBranch)
    assert frozen.factual_block_ids == ("b-1",)
    assert frozen.imagined_nodes[0].imagined_node_id == "img-retry"
    assert frozen.imagined_nodes[0].counterfactual is True
    assert frozen.imagined_nodes[0].branch_local is True
    assert frozen.imagined_nodes[0].imagined_node_id not in catalog.block_ids
    assert frozen.factual_block_ids[0] in catalog.block_ids
    imagined_path = _imagined_files(tmp_path)[0]
    imagined_body = json.loads(imagined_path.read_text(encoding="utf-8"))
    assert set(imagined_body) == {
        "base_revision_id",
        "branch_local",
        "counterfactual",
        "fork_node_id",
        "imagined_node_id",
        "step_kind",
    }
    assert imagined_body["step_kind"] == "imagined_node"
    assert imagined_body["counterfactual"] is True
    assert imagined_body["branch_local"] is True
    imagined_before = imagined_path.read_bytes()
    index_before = (tmp_path / STRATEGY_PATTERN_DIRNAME / "index.jsonl").read_text(encoding="utf-8")
    assert store.freeze_branch(stored, catalog) == frozen
    assert (tmp_path / STRATEGY_PATTERN_DIRNAME / "index.jsonl").read_text(encoding="utf-8") == (
        index_before
    )

    payload = _proposal_payload(stored.overlay, "img-retry")
    pattern = store.accept_proposal(stored, catalog, payload)
    assert pattern.pattern_id == _spec_pattern_id(_core())
    store.append_similar_fork_evidence(
        evaluation_id="eval-analog-1",
        subject_kind="imagined_node",
        subject_id="img-retry",
        evidence_ref="evidence:similar-fork",
        estimated_confidence=0.25,
    )
    assert overlay_path.read_bytes() == overlay_before
    assert catalog_path.read_bytes() == catalog_before
    assert imagined_path.read_bytes() == imagined_before
    assert "img-retry" not in catalog_path.read_text(encoding="utf-8")
    recorded = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert recorded["block_ids"] == ["b-1", "b-2"]


def test_exact_controlled_fields_accumulate_on_one_pattern(tmp_path: Path) -> None:
    store = _store(tmp_path)
    catalog = _catalog()
    first = _overlay(catalog, "fork-a", ("img-a",))
    second = _overlay(catalog, "fork-b", ("img-b",))
    ordered = _overlay(catalog, "fork-c", ("img-c",))
    core = _core()
    reversed_core = _core(applicability_conditions=list(reversed(core["applicability_conditions"])))
    pattern = store.accept_proposal(
        first, catalog, _proposal_payload(first, "img-a", summary="wording one")
    )
    assert pattern.pattern_id == _spec_pattern_id(core)
    assert pattern.pattern_id == strategy_pattern_id(
        intervention_type=core["intervention_type"],
        target_boundary=core["target_boundary"],
        mechanism=core["mechanism"],
        expected_metric=core["expected_metric"],
        capability=core["capability"],
        applicability_conditions=core["applicability_conditions"],
    )
    assert PATTERN_IDENTITY_VERSION == "strategy-pattern-identity-v1"
    pattern_path = _pattern_files(tmp_path)[0]
    pattern_bytes = pattern_path.read_bytes()
    document = json.loads(pattern_bytes)
    assert set(document["body"]) == {
        "applicability_conditions",
        "capability",
        "expected_metric",
        "identity_version",
        "intervention_type",
        "mechanism",
        "pattern_id",
        "target_boundary",
    }
    assert "semantic_summary" not in document["body"]

    again = store.accept_proposal(
        second, catalog, _proposal_payload(second, "img-b", summary="wording two")
    )
    reordered = store.accept_proposal(
        ordered,
        catalog,
        _proposal_payload(ordered, "img-c", reversed_core, summary="same fields, other order"),
    )
    assert again.pattern_id == pattern.pattern_id == reordered.pattern_id
    assert pattern_path.read_bytes() == pattern_bytes
    assert len(_pattern_files(tmp_path)) == 1
    summaries = [item.semantic_summary for item in store.memberships(pattern.pattern_id)]
    assert summaries == ["wording one", "wording two", "same fields, other order"]
    assert {item.fork_node_id for item in store.memberships(pattern.pattern_id)} == {
        "fork-a",
        "fork-b",
        "fork-c",
    }


def test_semantic_neighbor_is_not_merged(tmp_path: Path) -> None:
    store = _store(tmp_path)
    catalog = _catalog()
    exact = _overlay(catalog, "fork-exact", ("img-exact",))
    neighbor = _overlay(catalog, "fork-near", ("img-near",))
    exact_core = _core()
    near_core = _core(mechanism="retry the request with a narrower timeout")
    exact_pattern = store.accept_proposal(
        exact, catalog, _proposal_payload(exact, "img-exact", exact_core)
    )
    near_pattern = store.accept_proposal(
        neighbor,
        catalog,
        _proposal_payload(neighbor, "img-near", near_core, summary="almost the same retry"),
    )
    assert exact_pattern.pattern_id != near_pattern.pattern_id
    assert len(_pattern_files(tmp_path)) == 2
    assert [item.pattern_id for item in store.list_patterns()] == sorted(
        [exact_pattern.pattern_id, near_pattern.pattern_id]
    )
    assert store.find_exact_pattern(**exact_core) == exact_pattern
    assert store.find_exact_pattern(**near_core) == near_pattern
    missed = _core(mechanism="retry the invocation with a narrower timeout")
    before = _pattern_files(tmp_path)
    assert store.find_exact_pattern(**missed) is None
    assert _pattern_files(tmp_path) == before
    assert not hasattr(store, "merge_patterns")
    assert not hasattr(store, "find_similar")

    same_node = _proposal_payload(exact, "img-exact", near_core, summary="retarget this node")
    index_before = (tmp_path / STRATEGY_PATTERN_DIRNAME / "index.jsonl").read_bytes()
    assert _code(lambda: store.accept_proposal(exact, catalog, same_node)) == (
        "pattern_binding_conflict"
    )
    assert _pattern_files(tmp_path) == before
    assert (tmp_path / STRATEGY_PATTERN_DIRNAME / "index.jsonl").read_bytes() == index_before


def test_reader_cannot_choose_the_pattern_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    catalog = _catalog()
    overlay = _overlay(catalog)
    payload = _proposal_payload(overlay, "img-retry")
    payload["body"] = {**payload["body"], "pattern_id": "a" * 64}
    payload["content_hash"] = _spec_hash("strategy_pattern_proposal", payload["body"])
    with pytest.raises(ProtocolContractError, match="unexpected field"):
        store.accept_proposal(overlay, catalog, payload)
    assert _pattern_files(tmp_path) == []
    assert "pattern_id" not in inspect.signature(strategy_pattern_document).parameters
    assert "pattern_id" not in inspect.signature(strategy_pattern_id).parameters


def test_analogical_evidence_cannot_be_promoted_to_direct(tmp_path: Path) -> None:
    store = _store(tmp_path)
    catalog = _catalog()
    overlay = _overlay(catalog)
    store.freeze_branch(overlay, catalog)
    for tier in _DIRECT_TIERS:
        before = _evaluation_files(tmp_path)
        assert (
            _code(
                lambda tier=tier: store.append_similar_fork_evidence(
                    evaluation_id=f"eval-promote-{tier}",
                    subject_kind="imagined_node",
                    subject_id="img-retry",
                    evidence_ref="evidence:similar",
                    estimated_confidence=0.9,
                    evidence_tier=tier,
                )
            )
            == "analogical_promotion_forbidden"
        )
        assert _evaluation_files(tmp_path) == before
    assert (
        _code(
            lambda: store.append_similar_fork_evidence(
                evaluation_id="eval-observe",
                subject_kind="imagined_node",
                subject_id="img-retry",
                evidence_ref="evidence:observe",
                estimated_confidence=0.5,
                evidence_tier="observational",
            )
        )
        == "analogical_tier_required"
    )

    stored = store.append_similar_fork_evidence(
        evaluation_id="eval-analog-1",
        subject_kind="imagined_node",
        subject_id="img-retry",
        evidence_ref="evidence:similar",
        estimated_confidence=0.25,
    )
    assert stored.evidence_tier == "analogical"
    assert store.estimated_confidence("imagined_node", "img-retry") == 0.25
    assert store.has_direct_evidence("imagined_node", "img-retry") is False
    evaluation_path = _evaluation_files(tmp_path)[0]
    evaluation_bytes = evaluation_path.read_bytes()
    document = json.loads(evaluation_bytes)
    assert document["object_kind"] == "branch_evaluation"
    assert document["body"]["evidence_tier"] == "analogical"
    assert "estimated_confidence" not in document["body"]
    assert "estimated_confidence" not in document

    promoted = _evaluation(
        "imagined_node",
        "img-retry",
        "eval-direct-same-ref",
        "evidence:similar",
        "replay-direct",
    )
    assert (
        _code(lambda: store.append_evaluation(promoted, admission=_direct_admission("img-retry")))
        == "analogical_promotion_forbidden"
    )
    assert evaluation_path.read_bytes() == evaluation_bytes
    assert len(_evaluation_files(tmp_path)) == 1
    assert store.has_direct_evidence("imagined_node", "img-retry") is False
    assert store.estimated_confidence("imagined_node", "img-retry") == 0.25

    rewritten = _evaluation(
        "imagined_node",
        "img-retry",
        "eval-analog-1",
        "evidence:similar",
        "replay-direct",
    )
    assert _code(lambda: store.append_evaluation(rewritten)) == "evaluation_conflict"
    assert evaluation_path.read_bytes() == evaluation_bytes


def test_later_direct_record_does_not_rewrite_analogical_evidence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    catalog = _catalog()
    overlay = _overlay(catalog)
    pattern = store.accept_proposal(overlay, catalog, _proposal_payload(overlay, "img-retry"))
    store.append_similar_fork_evidence(
        evaluation_id="eval-analog-1",
        subject_kind="imagined_node",
        subject_id="img-retry",
        evidence_ref="evidence:similar",
        estimated_confidence=0.25,
    )
    analogical_path = _evaluation_files(tmp_path)[0]
    analogical_bytes = analogical_path.read_bytes()
    bare = _evaluation(
        "imagined_node",
        "img-retry",
        "eval-bare-direct",
        "evidence:bare",
        "replay-direct",
    )
    assert _code(lambda: store.append_evaluation(bare)) == "direct_admission_required"
    mismatched = _evaluation(
        "imagined_node",
        "img-retry",
        "eval-prod-mismatch",
        "evidence:production-run",
        "production-direct",
    )
    assert (
        _code(lambda: store.append_evaluation(mismatched, admission=_direct_admission("img-retry")))
        == "direct_admission_mismatch"
    )
    wrong_node = _evaluation(
        "imagined_node",
        "img-retry",
        "eval-wrong-node",
        "evidence:wrong-node",
        "replay-direct",
    )
    assert (
        _code(lambda: store.append_evaluation(wrong_node, admission=_direct_admission("img-other")))
        == "direct_admission_mismatch"
    )
    direct = store.append_evaluation(
        _evaluation(
            "imagined_node",
            "img-retry",
            "eval-prod-1",
            "evidence:production-run",
            "replay-direct",
        ),
        admission=_direct_admission("img-retry"),
    )
    assert direct.evidence_tier == "replay-direct"
    assert json.loads(analogical_bytes)["body"]["evidence_tier"] == "analogical"
    assert analogical_path.read_bytes() == analogical_bytes
    assert store.estimated_confidence("imagined_node", "img-retry") == 0.25
    assert store.has_direct_evidence("imagined_node", "img-retry") is True
    assert store.has_direct_evidence("strategy_pattern", pattern.pattern_id) is False
    observed = store.append_evaluation(
        _evaluation(
            "strategy_pattern",
            pattern.pattern_id,
            "eval-model-1",
            "evidence:model",
            "model-only",
        )
    )
    assert observed.evidence_tier == "model-only"
    assert store.has_direct_evidence("strategy_pattern", pattern.pattern_id) is False
    assert (
        _code(
            lambda: store.append_evaluation(
                _evaluation(
                    "strategy_pattern",
                    pattern.pattern_id,
                    "eval-outcome-missing",
                    "evidence:outcome-missing",
                    "replay-direct",
                ),
                admission=_direct_admission("img-retry"),
            )
        )
        == "direct_admission_mismatch"
    )
    replay = store.append_evaluation(
        _evaluation(
            "strategy_pattern",
            pattern.pattern_id,
            "eval-replay-1",
            "evidence:replay-run",
            "replay-direct",
        ),
        admission=_direct_admission("img-retry", outcome=True),
    )
    assert replay.evidence_tier == "replay-direct"
    assert store.has_direct_evidence("strategy_pattern", pattern.pattern_id) is True
    observational = store.append_evaluation(
        _evaluation(
            "imagined_node",
            "img-retry",
            "eval-obs-1",
            "evidence:watch",
            "observational",
        )
    )
    assert observational.evidence_tier == "observational"
    assert store.estimated_confidence("imagined_node", "img-retry") == 0.25


def test_confidence_adjustment_is_a_new_analogical_append(tmp_path: Path) -> None:
    store = _store(tmp_path)
    catalog = _catalog()
    overlay = _overlay(catalog)
    store.freeze_branch(overlay, catalog)
    store.append_similar_fork_evidence(
        evaluation_id="eval-analog-1",
        subject_kind="imagined_node",
        subject_id="img-retry",
        evidence_ref="evidence:similar-1",
        estimated_confidence=0.25,
    )
    assert (
        _code(
            lambda: store.append_similar_fork_evidence(
                evaluation_id="eval-analog-1",
                subject_kind="imagined_node",
                subject_id="img-retry",
                evidence_ref="evidence:similar-1",
                estimated_confidence=0.5,
            )
        )
        == "evaluation_conflict"
    )
    assert store.estimated_confidence("imagined_node", "img-retry") == 0.25
    updated = store.append_similar_fork_evidence(
        evaluation_id="eval-analog-2",
        subject_kind="imagined_node",
        subject_id="img-retry",
        evidence_ref="evidence:similar-2",
        estimated_confidence=0.5,
    )
    assert updated.evidence_tier == "analogical"
    assert store.estimated_confidence("imagined_node", "img-retry") == 0.5
    assert store.has_direct_evidence("imagined_node", "img-retry") is False
    assert [item.evidence_tier for item in store.audit_evaluations()] == [
        "analogical",
        "analogical",
    ]
    for confidence in (0.0, 1.0):
        store.append_similar_fork_evidence(
            evaluation_id=f"eval-bound-{confidence}",
            subject_kind="imagined_node",
            subject_id="img-retry",
            evidence_ref=f"evidence:bound-{confidence}",
            estimated_confidence=confidence,
        )
    before = _evaluation_files(tmp_path)
    for bad in (1.1, -0.01, True, "0.25"):
        assert (
            _code(
                lambda bad=bad: store.append_similar_fork_evidence(
                    evaluation_id="eval-bad-confidence",
                    subject_kind="imagined_node",
                    subject_id="img-retry",
                    evidence_ref="evidence:bad",
                    estimated_confidence=bad,
                )
            )
            == "invalid_confidence"
        )
    assert _evaluation_files(tmp_path) == before


def test_retired_evaluation_leaves_precedent_candidates_and_stays_auditable(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    catalog = _catalog()
    nodes = ("img-active", "img-challenged", "img-invalidated", "img-retired", "img-superseded")
    overlay = _overlay(catalog, imagined=nodes)
    store.freeze_branch(overlay, catalog)
    statuses = {
        "img-active": "active",
        "img-challenged": "challenged",
        "img-invalidated": "invalidated",
        "img-retired": "retired",
        "img-superseded": "superseded",
    }
    for node_id, status in statuses.items():
        if node_id == "img-retired":
            store.append_evaluation(
                _evaluation(
                    "imagined_node",
                    node_id,
                    "eval-img-retired-earlier",
                    "evidence:img-retired-earlier",
                    "observational",
                    "active",
                )
            )
        store.append_evaluation(
            _evaluation(
                "imagined_node",
                node_id,
                f"eval-{node_id}",
                f"evidence:{node_id}",
                "model-only",
                status,
            )
        )
    candidates = store.precedent_candidates()
    assert all(isinstance(item, PrecedentCandidate) for item in candidates)
    assert {item.subject_id for item in candidates} == {
        "img-active",
        "img-challenged",
        "img-invalidated",
        "img-superseded",
    }
    assert "img-retired" not in {item.subject_id for item in candidates}
    ordered = sorted(item.subject_id for item in candidates)
    assert [item.subject_id for item in candidates] == ordered
    assert "rank" not in inspect.signature(store.precedent_candidates).parameters
    assert "principal_labels" not in inspect.signature(store.precedent_candidates).parameters
    audited = store.audit_evaluations()
    retired = store.read_evaluation("eval-img-retired")
    assert retired is not None
    assert retired.status == "retired"
    assert retired.evaluation_id in {item.evaluation_id for item in audited}
    assert store.read_evaluation("eval-img-retired-earlier") is not None
    assert any(item.status == "retired" for item in audited)


def test_unknown_subject_and_legacy_fork_do_not_enter_the_ledger(tmp_path: Path) -> None:
    store = _store(tmp_path)
    payload = _evaluation(
        "imagined_node", "img-missing", "eval-missing", "evidence:missing", "model-only"
    )
    assert _code(lambda: store.append_evaluation(payload)) == "subject_unknown"
    assert not (tmp_path / STRATEGY_PATTERN_DIRNAME).exists()

    legacy = LegacyForkGrounding(
        fork_node_id="legacy-1",
        revision_id="legacy-rev",
        position_kind="cut",
        position_ref="cut:a:b",
        view_revision_id="rev-legacy",
        content_hash="hash-legacy",
        semantic_summary_snapshot="old cut grounding",
        grounded_at="2026-10-10T00:00:00.000Z",
        workspace_id="ws-legacy",
    )
    catalog = _catalog()
    with pytest.raises(StrategyPatternError) as exc:
        store.accept_proposal(legacy, catalog, _proposal_payload(_overlay(catalog), "img-retry"))
    assert exc.value.code == "legacy_fork_excluded"
    assert not (tmp_path / STRATEGY_PATTERN_DIRNAME).exists()
    source = inspect.getsource(__import__("openviking.session.strategy_pattern", fromlist=["x"]))
    assert "InfluenceClaim(" not in source
    assert "GuidanceDocument(" not in source


def test_append_evaluation_rejects_an_unknown_status_without_writing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    catalog = _catalog()
    overlay = _overlay(catalog)
    store.freeze_branch(overlay, catalog)
    payload = _evaluation(
        "imagined_node", "img-retry", "eval-deleted", "evidence:deleted", "analogical", "deleted"
    )
    with pytest.raises(ProtocolContractError, match="status"):
        store.append_evaluation(payload)
    assert _evaluation_files(tmp_path) == []


def test_fork_mismatch_and_unknown_imagined_node_do_not_create_a_pattern(tmp_path: Path) -> None:
    store = _store(tmp_path)
    catalog = _catalog()
    overlay = _overlay(catalog)
    mismatched = _proposal_payload(overlay, "img-retry")
    mismatched["body"] = {**mismatched["body"], "fork_node_id": "fork-other"}
    mismatched["content_hash"] = _spec_hash("strategy_pattern_proposal", mismatched["body"])
    assert _code(lambda: store.accept_proposal(overlay, catalog, mismatched)) == "fork_mismatch"
    missing = _proposal_payload(overlay, "img-other")
    assert _code(lambda: store.accept_proposal(overlay, catalog, missing)) == (
        "imagined_node_not_on_branch"
    )
    assert _pattern_files(tmp_path) == []
    assert _imagined_files(tmp_path) == []
