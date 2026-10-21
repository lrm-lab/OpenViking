# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Post-commit failure gate (ADR-0043 P10).

``FailureGateOrchestrator.deliver`` is the shared pipeline for a finished
commit and for ``@memory diagnose``. It reuses a fresh frozen AIG view,
asks the Reader for an outcome assessment, and lets OutcomeEvidencePolicy
decide. Only ``confirmed_failure`` continues through ``ForkPrecedentService``,
``GuidanceApprovalService``, and ``RemediationDispatchService``.

The Coforge worker is not called. P12 connects that worker. This module
does not route group chat and does not open a network connection.

Commit-path wiring (``failure_gate_wiring``) sets ``auto_remediate=False``.
Confirmed failure then records the outcome and stops: no Fork, Guidance, or
child run. Diagnose and an explicit remediation script keep the fail-closed
dispatch gates.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from openviking.session.aig_native_protocol import (
    OBJECT_KIND_AGENT_CANDIDATE_CATALOG,
    OBJECT_KIND_FORK_PROPOSAL,
    OBJECT_KIND_GUIDED_COUNTERFACTUAL_BRANCH,
    OBJECT_KIND_TASK_OUTCOME_ASSESSMENT,
    SCHEMA_VERSION_AIG_NATIVE_V2,
    AigNativeViewRevision,
    ProtocolContractError,
    guidance_execution_report_document,
    is_safe_id,
    is_sha256_hex,
    protocol_content_hash,
    protocol_envelope,
    read_native_edge_revision,
)
from openviking.session.aig_role_chain import (
    AigConstructionRun,
    AigContractError,
    FrozenRunSnapshot,
)
from openviking.session.append_only import (
    append_jsonl,
    read_json_object,
    read_jsonl_objects,
    write_exclusive,
    write_new_json,
)
from openviking.session.coordinator_boundary import (
    ACTION_ADVICE,
    ACTION_BLOCKED,
    ACTION_DISPATCHED,
    ACTION_RETAINED,
    TRIGGER_COMMIT,
    TRIGGER_DIAGNOSE,
    CoordinatorDiagnoseRequest,
    coordinator_delivery_binding_hash,
    coordinator_failure_gate_event_document,
    read_coordinator_diagnose_request,
)
from openviking.session.fork_node_overlay import (
    ForkNodeOverlayStore,
    ForkOverlayError,
    ForkWriteAuthorizer,
)
from openviking.session.fork_precedent import ForkPrecedentService
from openviking.session.guidance_approval import GuidanceApprovalError, GuidanceApprovalService
from openviking.session.outcome_contract import (
    DECISION_CONFIRMED_FAILURE,
    DECISION_SUCCESS,
    DECISION_UNCERTAIN,
    OUTCOME_EVIDENCE_POLICY_VERSION,
    ExternalOutcomeEvidence,
    OutcomeAdjudication,
    OutcomeAssessmentService,
    OutcomeContractStore,
    external_evidence_from_session,
)
from openviking.session.remediation_dispatch import RemediationDispatchService
from openviking.session.strategy_pattern import StrategyPatternStore

AIG_CONSTRUCTION_PIPELINE_VERSION = "aig-construction-v1"
FRESHNESS_RECORD_VERSION = "aig-view-freshness-v1"
FAILURE_GATE_DIRNAME = "failure-gate"
AUDIT_VIEW_CONSTRUCTED = "aig_view_constructed"
AUDIT_VIEW_REUSED = "aig_view_reused"
AUDIT_ADJUDICATED = "outcome_adjudicated"
AUDIT_DIAGNOSE_REQUESTED = "diagnose_requested"
AUDIT_REPLAYED = "delivery_replayed"
AUDIT_REFUSED = "refused"

_TRIGGERS = frozenset({TRIGGER_COMMIT, TRIGGER_DIAGNOSE})
_PLAN_KEYS = frozenset({"applicability", "required_capabilities", "risk", "steps", "target_role"})


class FailureGateError(ValueError):
    """A post-commit delivery was refused."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class FrozenViewBuilder(Protocol):
    """Build one native AIG revision. The orchestrator freezes the role chain."""

    def __call__(self, request: ConstructionRequest) -> AigNativeViewRevision: ...


@dataclass(frozen=True)
class ConstructionRequest:
    """Inputs of one construction. The pipeline version is part of freshness."""

    pipeline_version: str
    session_id: str
    task_id: str
    source_digest: str


@dataclass(frozen=True)
class FailureGateDelivery:
    """One commit or diagnose delivery. The remediation script is optional.

    ``auto_remediate`` defaults to True. Commit-path wiring sets it False so
    a confirmed failure records the outcome and does not dispatch.
    """

    trigger: str
    commit_id: str
    task_id: str
    session_id: str
    source_digest: str
    principal_labels: tuple[str, ...]
    contract: Mapping[str, Any]
    assessment_body: Mapping[str, Any]
    evaluator_results: tuple[Mapping[str, Any], ...]
    temporal_cutoff: str
    recorded_at: str
    anchor_block_id: str | None = None
    branch_steps: tuple[Mapping[str, Any], ...] = ()
    guidance_plan: Mapping[str, Any] | None = None
    catalog: Mapping[str, Any] | None = None
    dispatch_controls: Mapping[str, Any] | None = None
    dispatch_outputs: tuple[Mapping[str, Any], ...] = ()
    dispatch_steps: tuple[Mapping[str, Any], ...] = ()
    dispositions: tuple[Mapping[str, Any], ...] = ()
    diagnose_request_hash: str | None = None
    auto_remediate: bool = True
    # Commit-path fork authoring: on confirmed failure persist the Reader's
    # fork overlay (anchor + branch steps) and stop. No guidance approval, no
    # child run. ``auto_remediate`` stays the only dispatch door.
    fork_only: bool = False


@dataclass(frozen=True)
class FailureGateResult:
    """Outcome of one delivery. ``deduplicated`` means the commit was already finished."""

    orchestration_run_id: str
    trigger: str
    decision: str
    reason: str
    policy_version: str
    action: str
    reused_view: bool
    deduplicated: bool
    view_content_hash: str
    reader_report_hash: str
    fork_node_id: str | None
    guidance_content_hash: str | None
    child_run_id: str | None
    blocked: bool
    audit_kinds: tuple[str, ...]
    coordinator_event: dict[str, Any] | None


@dataclass(frozen=True)
class _Binding:
    revision: AigNativeViewRevision
    run: AigConstructionRun
    reader_session_id: str
    reader_report_hash: str
    reused: bool


def freshness_key(*, session_id: str, task_id: str, source_digest: str) -> str:
    """Digest of the versioned freshness inputs. The view file is named by this."""
    return protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        "aig_view_freshness",
        {
            "freshness_version": FRESHNESS_RECORD_VERSION,
            "pipeline_version": AIG_CONSTRUCTION_PIPELINE_VERSION,
            "session_id": session_id,
            "source_digest": source_digest,
            "task_id": task_id,
        },
    )


def orchestration_run_id(*, commit_id: str, trigger: str, fingerprint: str) -> str:
    """One version of a commit delivery.

    The id is the commit, the trigger, the input fingerprint, and the
    adjudication policy version. The same inputs replay. A different trigger
    or fingerprint is a new version; the previous version stays on disk.
    """
    return protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        "failure_gate_run",
        {
            "commit_id": commit_id,
            "fingerprint": fingerprint,
            "policy_version": OUTCOME_EVIDENCE_POLICY_VERSION,
            "trigger": trigger,
        },
    )


def view_is_fresh(
    record: Mapping[str, Any],
    *,
    session_id: str,
    task_id: str,
    source_digest: str,
) -> bool:
    """True when the stored view matches this pipeline version and commit source.

    A record is fresh only when its freshness version, pipeline version,
    source digest, session, task, and ``trust`` are the current frozen
    values, and the embedded native-edge document still hashes to
    ``view_content_hash``. Anything else is not fresh.
    """
    if not isinstance(record, Mapping):
        return False
    document = record.get("view")
    if not isinstance(document, Mapping):
        return False
    try:
        parsed = read_native_edge_revision(document)
    except (ProtocolContractError, TypeError, ValueError):
        return False
    return (
        record.get("freshness_version") == FRESHNESS_RECORD_VERSION
        and record.get("pipeline_version") == AIG_CONSTRUCTION_PIPELINE_VERSION
        and record.get("source_digest") == source_digest
        and record.get("session_id") == session_id
        and record.get("task_id") == task_id
        and record.get("trust") == "frozen"
        and record.get("view_content_hash") == parsed.content_hash
        and parsed.revision.session_id == session_id
        and parsed.revision.task_run_id == task_id
    )


def bind_role_chain_authorizer(
    run: AigConstructionRun,
    *,
    reader_report_hash: str,
    view_content_hash: str,
) -> ForkWriteAuthorizer:
    """Adapt ``authorize_outcome_write`` to the overlay's document hash.

    The role chain decides who may write and whether the Reader report is
    frozen. The overlay then requires the hook to return the frozen view's
    document hash, which is the proposal ``report_hash``. A session the
    role chain refuses never reaches that return.
    """

    def authorize(session_id: str) -> str:
        granted = run.authorize_outcome_write(session_id)
        if granted != reader_report_hash:
            raise ForkOverlayError("report_hash_mismatch")
        return view_content_hash

    return authorize


class FailureGateOrchestrator:
    """Run the post-commit gate against the existing server stores."""

    def __init__(
        self,
        root: str | Path,
        build_revision: FrozenViewBuilder,
        *,
        overlay_root: str | Path | None = None,
    ) -> None:
        if not callable(build_revision):
            raise FailureGateError("builder_required")
        self._root = Path(root)
        self._build_revision = build_revision
        self._dir = self._root / FAILURE_GATE_DIRNAME
        self._audit_path = self._dir / "audit.jsonl"
        self._outcomes = OutcomeAssessmentService(OutcomeContractStore(self._root))
        # Retrieval and publish read forks from the user root, not the session.
        self._overlays = ForkNodeOverlayStore(
            Path(overlay_root) if overlay_root is not None else self._root
        )
        self._patterns = StrategyPatternStore(self._root)
        self._precedents = ForkPrecedentService(self._root)
        self._guidance = GuidanceApprovalService(
            self._root,
            outcomes=self._outcomes,
            precedents=self._precedents,
            overlays=self._overlays,
        )
        self._dispatch = RemediationDispatchService(self._root, guidance=self._guidance)
        self._lock = threading.Lock()

    def deliver(self, delivery: FailureGateDelivery) -> FailureGateResult:
        """Run commit or diagnose. A finished identical commit is returned as-is."""
        _validate_delivery(delivery)
        with self._lock:
            fingerprint = _fingerprint(delivery)
            run_id = orchestration_run_id(
                commit_id=delivery.commit_id,
                trigger=delivery.trigger,
                fingerprint=fingerprint,
            )
            existing = self._read_run(run_id)
            if existing is not None:
                if (
                    existing.get("fingerprint") != fingerprint
                    or existing.get("status") != "complete"
                ):
                    raise FailureGateError("delivery_conflict")
                self._require_fresh_view(delivery)
                return self._replay(delivery, existing, run_id)
            try:
                result = self._execute(delivery, run_id)
            except Exception as exc:
                self._audit_refusal(delivery, run_id, _exception_code(exc))
                raise
            self._write_run(run_id, fingerprint, result)
            return result

    def diagnose(
        self,
        request_document: Mapping[str, Any],
        delivery: FailureGateDelivery,
    ) -> FailureGateResult:
        """Accept one coordinator diagnose request and run ``deliver``."""
        parsed = read_coordinator_diagnose_request(request_document)
        if not isinstance(delivery, FailureGateDelivery) or delivery.trigger != TRIGGER_DIAGNOSE:
            raise FailureGateError("diagnose_trigger_required")
        if not _same_request(parsed, delivery):
            raise FailureGateError("diagnose_mismatch")
        return self.deliver(replace(delivery, diagnose_request_hash=parsed.content_hash))

    def read_frozen_view(
        self,
        *,
        task_id: str,
        session_id: str,
        source_digest: str,
    ) -> AigNativeViewRevision | None:
        """Return the searchable frozen AIG, or None when this source has none."""
        path = self._view_path(
            freshness_key(session_id=session_id, task_id=task_id, source_digest=source_digest)
        )
        if not path.is_file():
            return None
        record = read_json_object(path, FailureGateError)
        if not view_is_fresh(
            record, session_id=session_id, task_id=task_id, source_digest=source_digest
        ):
            raise FailureGateError("stored_record_invalid")
        return read_native_edge_revision(record["view"]).revision

    def audit_events(self) -> tuple[dict[str, Any], ...]:
        """Audit rows in append order."""
        return tuple(read_jsonl_objects(self._audit_path, FailureGateError))

    def _execute(self, delivery: FailureGateDelivery, run_id: str) -> FailureGateResult:
        kinds: list[str] = []
        if delivery.trigger == TRIGGER_DIAGNOSE:
            self._audit(kinds, AUDIT_DIAGNOSE_REQUESTED, delivery, run_id)
        binding = self._ensure_view(delivery)
        self._audit(
            kinds,
            AUDIT_VIEW_REUSED if binding.reused else AUDIT_VIEW_CONSTRUCTED,
            delivery,
            run_id,
        )
        contract = self._outcomes.freeze_contract(delivery.contract)
        if contract.task_id != delivery.task_id:
            raise FailureGateError("task_mismatch")
        assessment_body = dict(delivery.assessment_body)
        assessment_body["author_role"] = "reader"
        assessment_body["report_hash"] = binding.reader_report_hash
        self._outcomes.submit_assessment(
            binding.run,
            binding.reader_session_id,
            protocol_envelope(
                SCHEMA_VERSION_AIG_NATIVE_V2,
                OBJECT_KIND_TASK_OUTCOME_ASSESSMENT,
                assessment_body,
            ),
            task_id=contract.task_id,
            contract_id=contract.contract_id,
        )
        evidence = external_evidence_from_session(evaluator_results=delivery.evaluator_results)
        adjudication = self._outcomes.adjudicate(
            task_id=delivery.task_id,
            report_hash=binding.reader_report_hash,
            external_evidence=evidence,
        )
        self._audit(
            kinds,
            AUDIT_ADJUDICATED,
            delivery,
            run_id,
            decision=adjudication.decision,
            reason=adjudication.reason,
        )
        if adjudication.decision == DECISION_CONFIRMED_FAILURE:
            if delivery.auto_remediate:
                return self._remediate(delivery, run_id, binding, adjudication, evidence, kinds)
            fork_node_id = None
            if delivery.fork_only:
                fork_node_id = self._commit_fork(delivery, binding).fork_node_id
            action = ACTION_RETAINED
            self._audit(
                kinds,
                action,
                delivery,
                run_id,
                decision=adjudication.decision,
                reason=adjudication.reason,
            )
            return self._finish(
                delivery,
                run_id,
                binding,
                adjudication,
                action=action,
                blocked=False,
                fork_node_id=fork_node_id,
                guidance_content_hash=None,
                child_run_id=None,
                kinds=kinds,
            )
        if adjudication.decision == DECISION_SUCCESS:
            action = ACTION_RETAINED
        elif adjudication.decision == DECISION_UNCERTAIN:
            action = ACTION_ADVICE
        else:
            raise FailureGateError("decision_invalid")
        self._audit(
            kinds,
            action,
            delivery,
            run_id,
            decision=adjudication.decision,
            reason=adjudication.reason,
        )
        return self._finish(
            delivery,
            run_id,
            binding,
            adjudication,
            action=action,
            blocked=False,
            fork_node_id=None,
            guidance_content_hash=None,
            child_run_id=None,
            kinds=kinds,
        )

    def _remediate(
        self,
        delivery: FailureGateDelivery,
        run_id: str,
        binding: _Binding,
        adjudication: OutcomeAdjudication,
        evidence: Sequence[ExternalOutcomeEvidence],
        kinds: list[str],
    ) -> FailureGateResult:
        if not _script_ready(delivery):
            raise FailureGateError("remediation_script_required")
        catalog, commit, retrieval = self._commit_fork_with_catalog(delivery, binding)
        plan = delivery.guidance_plan
        catalog_body = delivery.catalog
        if not isinstance(plan, Mapping) or not isinstance(catalog_body, Mapping):
            raise FailureGateError("remediation_script_required")
        proposal_document = protocol_envelope(
            SCHEMA_VERSION_AIG_NATIVE_V2,
            OBJECT_KIND_GUIDED_COUNTERFACTUAL_BRANCH,
            {
                "applicability": dict(plan["applicability"]),
                "author_role": "reader",
                "base_revision_id": catalog.base_revision_id,
                "report_hash": binding.reader_report_hash,
                "required_capabilities": list(plan["required_capabilities"]),
                "retrieval_id": retrieval.retrieval_id,
                "risk": plan["risk"],
                "source_fork_id": commit.stored.fork_node_id,
                "steps": [dict(step) for step in plan["steps"]],
                "target_role": plan["target_role"],
                "task_id": delivery.task_id,
            },
        )
        try:
            self._guidance.propose(
                binding.run,
                binding.reader_session_id,
                proposal_document,
                principal_labels=delivery.principal_labels,
                external_evidence=evidence,
            )
        except GuidanceApprovalError as exc:
            if exc.code != "adjudication_conflict":
                raise
        approved = self._guidance.approve(
            str(proposal_document["content_hash"]),
            protocol_envelope(
                SCHEMA_VERSION_AIG_NATIVE_V2,
                OBJECT_KIND_AGENT_CANDIDATE_CATALOG,
                {
                    "candidates": list(catalog_body["candidates"]),
                    "task_id": delivery.task_id,
                },
            ),
            principal_labels=delivery.principal_labels,
        )
        self._patterns.freeze_branch(commit.stored, catalog)
        controls = dict(delivery.dispatch_controls or {})
        # Same guidance keeps one dispatch fingerprint across orchestration
        # versions. A new guidance hash opens a new run; the same fingerprint
        # reloads the stored run.
        controls["idempotency_key"] = approved.approval.guidance_content_hash
        report = guidance_execution_report_document(
            guidance_id=approved.approval.guidance_id,
            source_fork_id=approved.approval.source_fork_id,
            parent_session_id=binding.reader_session_id,
            steps=[dict(step) for step in delivery.dispatch_steps],
        )
        outcome = self._dispatch.dispatch_scripted(
            approval_content_hash=approved.approval.content_hash,
            principal_labels=delivery.principal_labels,
            parent_session_id=binding.reader_session_id,
            controls=controls,
            outputs=[dict(item) for item in delivery.dispatch_outputs],
            report=report,
            pattern_store=self._patterns,
            orchestration_run_id=run_id,
        )
        action = ACTION_BLOCKED if outcome.blocked else ACTION_DISPATCHED
        child_run_id = None if outcome.child_run is None else outcome.child_run.child_run_id
        self._audit(
            kinds,
            action,
            delivery,
            run_id,
            decision=adjudication.decision,
            reason=adjudication.reason,
        )
        return self._finish(
            delivery,
            run_id,
            binding,
            adjudication,
            action=action,
            blocked=outcome.blocked,
            fork_node_id=commit.stored.fork_node_id,
            guidance_content_hash=approved.approval.guidance_content_hash,
            child_run_id=child_run_id,
            kinds=kinds,
        )

    def _commit_fork(self, delivery: FailureGateDelivery, binding: _Binding) -> Any:
        """Persist the Reader's fork overlay and return the stored overlay."""
        if not isinstance(delivery.anchor_block_id, str) or not delivery.branch_steps:
            raise FailureGateError("fork_script_required")
        _catalog, commit, _retrieval = self._commit_fork_with_catalog(delivery, binding)
        return commit.stored

    def _commit_fork_with_catalog(
        self, delivery: FailureGateDelivery, binding: _Binding
    ) -> tuple[Any, Any, Any]:
        catalog = self._overlays.register_base_revision(
            binding.revision, acl_labels=delivery.principal_labels
        )
        if delivery.anchor_block_id not in catalog.block_ids:
            raise FailureGateError("anchor_missing")
        fork_payload = protocol_envelope(
            SCHEMA_VERSION_AIG_NATIVE_V2,
            OBJECT_KIND_FORK_PROPOSAL,
            {
                "anchor_block_id": delivery.anchor_block_id,
                "author_role": "reader",
                "base_revision_block_content_hash": catalog.block_content_hash,
                "base_revision_id": catalog.base_revision_id,
                "branch_steps": [dict(step) for step in delivery.branch_steps],
                "report_hash": binding.revision.content_hash,
            },
        )
        retrieval = self._precedents.retrieve(
            self._overlays,
            self._patterns,
            fork_payload,
            principal_labels=delivery.principal_labels,
            session_id=binding.reader_session_id,
            temporal_cutoff=delivery.temporal_cutoff,
        )
        authorizer = bind_role_chain_authorizer(
            binding.run,
            reader_report_hash=binding.reader_report_hash,
            view_content_hash=binding.revision.content_hash,
        )
        commit = self._precedents.submit_proposal(
            self._overlays,
            fork_payload,
            retrieval_id=retrieval.retrieval_id,
            dispositions=[dict(item) for item in delivery.dispositions],
            principal_labels=delivery.principal_labels,
            session_id=binding.reader_session_id,
            recorded_at=delivery.recorded_at,
            authorize=authorizer,
        )
        return catalog, commit, retrieval

    def _finish(
        self,
        delivery: FailureGateDelivery,
        run_id: str,
        binding: _Binding,
        adjudication: OutcomeAdjudication,
        *,
        action: str,
        blocked: bool,
        fork_node_id: str | None,
        guidance_content_hash: str | None,
        child_run_id: str | None,
        kinds: list[str],
    ) -> FailureGateResult:
        return FailureGateResult(
            orchestration_run_id=run_id,
            trigger=delivery.trigger,
            decision=adjudication.decision,
            reason=adjudication.reason,
            policy_version=adjudication.policy_version,
            action=action,
            reused_view=binding.reused,
            deduplicated=False,
            view_content_hash=binding.revision.content_hash,
            reader_report_hash=binding.reader_report_hash,
            fork_node_id=fork_node_id,
            guidance_content_hash=guidance_content_hash,
            child_run_id=child_run_id,
            blocked=blocked,
            audit_kinds=tuple(kinds),
            coordinator_event=self._event(
                delivery,
                run_id,
                action=action,
                decision=adjudication.decision,
                reason=adjudication.reason,
            ),
        )

    def _replay(
        self,
        delivery: FailureGateDelivery,
        existing: Mapping[str, Any],
        run_id: str,
    ) -> FailureGateResult:
        kinds: list[str] = []
        if delivery.trigger == TRIGGER_DIAGNOSE:
            self._audit(kinds, AUDIT_DIAGNOSE_REQUESTED, delivery, run_id)
        self._audit(kinds, AUDIT_REPLAYED, delivery, run_id)
        stored = existing.get("result")
        if not isinstance(stored, Mapping):
            raise FailureGateError("stored_record_invalid")
        action = _stored_text(stored, "action")
        decision = _stored_text(stored, "decision")
        reason = _stored_text(stored, "reason")
        return FailureGateResult(
            orchestration_run_id=run_id,
            trigger=delivery.trigger,
            decision=decision,
            reason=reason,
            policy_version=_stored_text(stored, "policy_version"),
            action=action,
            reused_view=True,
            deduplicated=True,
            view_content_hash=_stored_text(stored, "view_content_hash"),
            reader_report_hash=_stored_text(stored, "reader_report_hash"),
            fork_node_id=_stored_optional(stored, "fork_node_id"),
            guidance_content_hash=_stored_optional(stored, "guidance_content_hash"),
            child_run_id=_stored_optional(stored, "child_run_id"),
            blocked=stored.get("blocked") is True,
            audit_kinds=tuple(kinds),
            coordinator_event=self._event(
                delivery, run_id, action=action, decision=decision, reason=reason
            ),
        )

    def _require_fresh_view(self, delivery: FailureGateDelivery) -> None:
        """Refuse a finished commit whose frozen view no longer matches freshness."""
        path = self._view_path(
            freshness_key(
                session_id=delivery.session_id,
                task_id=delivery.task_id,
                source_digest=delivery.source_digest,
            )
        )
        if not path.is_file():
            raise FailureGateError("stored_record_invalid")
        record = read_json_object(path, FailureGateError)
        if not view_is_fresh(
            record,
            session_id=delivery.session_id,
            task_id=delivery.task_id,
            source_digest=delivery.source_digest,
        ):
            raise FailureGateError("stored_record_invalid")

    def _ensure_view(self, delivery: FailureGateDelivery) -> _Binding:
        key = freshness_key(
            session_id=delivery.session_id,
            task_id=delivery.task_id,
            source_digest=delivery.source_digest,
        )
        path = self._view_path(key)
        if path.is_file():
            record = read_json_object(path, FailureGateError)
            if not view_is_fresh(
                record,
                session_id=delivery.session_id,
                task_id=delivery.task_id,
                source_digest=delivery.source_digest,
            ):
                raise FailureGateError("stored_record_invalid")
            return self._restore_binding(record)
        request = ConstructionRequest(
            pipeline_version=AIG_CONSTRUCTION_PIPELINE_VERSION,
            session_id=delivery.session_id,
            task_id=delivery.task_id,
            source_digest=delivery.source_digest,
        )
        revision = self._build_revision(request)
        if not isinstance(revision, AigNativeViewRevision):
            raise FailureGateError("view_required")
        if revision.session_id != delivery.session_id or revision.task_run_id != delivery.task_id:
            raise FailureGateError("view_identity_mismatch")
        run, reader_session_id, report_hash = _freeze_role_chain(revision, "aig-" + key[:32])
        snapshot = run.frozen_snapshot()
        record = {
            "draft_hash": snapshot.draft_hash,
            "freshness_version": FRESHNESS_RECORD_VERSION,
            "pipeline_version": AIG_CONSTRUCTION_PIPELINE_VERSION,
            "reader_report_hash": report_hash,
            "run_id": snapshot.run_id,
            "session_id": delivery.session_id,
            "sessions": dict(snapshot.sessions),
            "source_digest": delivery.source_digest,
            "task_id": delivery.task_id,
            "trust": "frozen",
            "view": revision.protocol_document(),
            "view_content_hash": revision.content_hash,
        }
        try:
            write_exclusive(path, record)
        except FileExistsError as exc:
            stored = read_json_object(path, FailureGateError)
            if not view_is_fresh(
                stored,
                session_id=delivery.session_id,
                task_id=delivery.task_id,
                source_digest=delivery.source_digest,
            ):
                raise FailureGateError("stored_record_invalid") from exc
            return self._restore_binding(stored)
        return _Binding(
            revision=revision,
            run=run,
            reader_session_id=reader_session_id,
            reader_report_hash=report_hash,
            reused=False,
        )

    def _restore_binding(self, record: Mapping[str, Any]) -> _Binding:
        try:
            document = read_native_edge_revision(record["view"])
            snapshot_sessions = record.get("sessions")
            if not isinstance(snapshot_sessions, Mapping):
                raise FailureGateError("stored_record_invalid")
            run = AigConstructionRun.restore_frozen(
                FrozenRunSnapshot(
                    run_id=_stored_text(record, "run_id"),
                    sessions=snapshot_sessions,
                    report_hash=_stored_text(record, "reader_report_hash"),
                    draft_hash=_stored_text(record, "draft_hash"),
                )
            )
        except FailureGateError:
            raise
        except (AigContractError, ProtocolContractError, KeyError, TypeError, ValueError) as exc:
            raise FailureGateError("stored_record_invalid") from exc
        reader_session_id = run.session_for("reader")
        if run.authorize_outcome_write(reader_session_id) != record.get("reader_report_hash"):
            raise FailureGateError("stored_record_invalid")
        return _Binding(
            revision=document.revision,
            run=run,
            reader_session_id=reader_session_id,
            reader_report_hash=_stored_text(record, "reader_report_hash"),
            reused=True,
        )

    def _event(
        self,
        delivery: FailureGateDelivery,
        run_id: str,
        *,
        action: str,
        decision: str,
        reason: str,
    ) -> dict[str, Any]:
        request_hash = delivery.diagnose_request_hash
        if request_hash is None:
            request_hash = coordinator_delivery_binding_hash(
                commit_id=delivery.commit_id,
                source_digest=delivery.source_digest,
                task_id=delivery.task_id,
                trigger=delivery.trigger,
            )
        return coordinator_failure_gate_event_document(
            action=action,
            commit_id=delivery.commit_id,
            decision=decision,
            orchestration_run_id=run_id,
            reason=reason,
            request_content_hash=request_hash,
            task_id=delivery.task_id,
            trigger=delivery.trigger,
        )

    def _audit(
        self,
        kinds: list[str],
        kind: str,
        delivery: FailureGateDelivery,
        run_id: str,
        **extra: str,
    ) -> None:
        kinds.append(kind)
        row: dict[str, Any] = {
            "commit_id": delivery.commit_id,
            "event_kind": kind,
            "orchestration_run_id": run_id,
            "task_id": delivery.task_id,
            "trigger": delivery.trigger,
        }
        row.update(extra)
        append_jsonl(self._audit_path, row)

    def _audit_refusal(self, delivery: FailureGateDelivery, run_id: str, reason: str) -> None:
        """Append one terminal refusal. The same version and reason is not repeated."""
        for row in read_jsonl_objects(self._audit_path, FailureGateError):
            if (
                row.get("orchestration_run_id") == run_id
                and row.get("outcome") == AUDIT_REFUSED
                and row.get("reason") == reason
            ):
                return
        append_jsonl(
            self._audit_path,
            {
                "commit_id": delivery.commit_id,
                "event_kind": AUDIT_REFUSED,
                "orchestration_run_id": run_id,
                "outcome": AUDIT_REFUSED,
                "reason": reason,
                "task_id": delivery.task_id,
                "trigger": delivery.trigger,
            },
        )

    def _write_run(self, run_id: str, fingerprint: str, result: FailureGateResult) -> None:
        write_new_json(
            self._run_path(run_id),
            {
                "fingerprint": fingerprint,
                "orchestration_run_id": run_id,
                "result": {
                    "action": result.action,
                    "blocked": result.blocked,
                    "child_run_id": result.child_run_id,
                    "decision": result.decision,
                    "fork_node_id": result.fork_node_id,
                    "guidance_content_hash": result.guidance_content_hash,
                    "policy_version": result.policy_version,
                    "reader_report_hash": result.reader_report_hash,
                    "reason": result.reason,
                    "view_content_hash": result.view_content_hash,
                },
                "status": "complete",
            },
            FailureGateError,
        )

    def _read_run(self, run_id: str) -> dict[str, Any] | None:
        path = self._run_path(run_id)
        if not path.is_file():
            return None
        return read_json_object(path, FailureGateError)

    def _view_path(self, key: str) -> Path:
        return self._dir / "views" / f"{key}.json"

    def _run_path(self, run_id: str) -> Path:
        return self._dir / "runs" / f"{run_id}.json"


def _freeze_role_chain(
    revision: AigNativeViewRevision, run_id: str
) -> tuple[AigConstructionRun, str, str]:
    run = AigConstructionRun(run_id)
    sessions = {role: run.session_for(role) for role in ("builder", "critic", "refiner", "reader")}
    run.submit_draft(
        sessions["builder"],
        {
            "revision_id": revision.revision_id,
            "view_content_hash": revision.content_hash,
        },
    )
    run.critique(sessions["critic"], [])
    report_hash = run.read_verify(
        sessions["reader"],
        {"view_content_hash": revision.content_hash},
    )
    if run.freeze_report(sessions["reader"]) != report_hash or run.trust != "frozen":
        raise FailureGateError("view_not_frozen")
    return run, sessions["reader"], report_hash


def _script_ready(delivery: FailureGateDelivery) -> bool:
    plan = delivery.guidance_plan
    catalog = delivery.catalog
    return (
        isinstance(delivery.anchor_block_id, str)
        and bool(delivery.branch_steps)
        and isinstance(plan, Mapping)
        and set(plan) == _PLAN_KEYS
        and isinstance(catalog, Mapping)
        and set(catalog) == {"candidates"}
        and isinstance(delivery.dispatch_controls, Mapping)
        and bool(delivery.dispatch_outputs)
        and bool(delivery.dispatch_steps)
    )


def _validate_delivery(delivery: FailureGateDelivery) -> None:
    if not isinstance(delivery, FailureGateDelivery):
        raise FailureGateError("delivery_required")
    if delivery.trigger not in _TRIGGERS:
        raise FailureGateError("trigger_invalid")
    if (
        not is_safe_id(delivery.commit_id)
        or not is_safe_id(delivery.task_id)
        or not is_safe_id(delivery.session_id)
    ):
        raise FailureGateError("invalid_id")
    if not is_sha256_hex(delivery.source_digest):
        raise FailureGateError("source_digest_invalid")


def _same_request(parsed: CoordinatorDiagnoseRequest, delivery: FailureGateDelivery) -> bool:
    return (
        parsed.commit_id == delivery.commit_id
        and parsed.task_id == delivery.task_id
        and parsed.session_id == delivery.session_id
        and parsed.source_digest == delivery.source_digest
        and frozenset(parsed.principal_labels) == frozenset(delivery.principal_labels)
    )


def _fingerprint(delivery: FailureGateDelivery) -> str:
    controls = delivery.dispatch_controls
    if isinstance(controls, Mapping):
        controls = {key: value for key, value in controls.items() if key != "idempotency_key"}
    body = {
        "anchor_block_id": delivery.anchor_block_id,
        "assessment_body": dict(delivery.assessment_body),
        "auto_remediate": delivery.auto_remediate,
        "branch_steps": [dict(step) for step in delivery.branch_steps],
        "catalog": None if delivery.catalog is None else dict(delivery.catalog),
        "commit_id": delivery.commit_id,
        "contract": dict(delivery.contract),
        "dispatch_controls": controls,
        "dispatch_outputs": [dict(item) for item in delivery.dispatch_outputs],
        "dispatch_steps": [dict(step) for step in delivery.dispatch_steps],
        "dispositions": [dict(item) for item in delivery.dispositions],
        "evaluator_results": [dict(item) for item in delivery.evaluator_results],
        "guidance_plan": None if delivery.guidance_plan is None else dict(delivery.guidance_plan),
        "pipeline_version": AIG_CONSTRUCTION_PIPELINE_VERSION,
        "policy_version": OUTCOME_EVIDENCE_POLICY_VERSION,
        "principal_labels": list(delivery.principal_labels),
        "recorded_at": delivery.recorded_at,
        "session_id": delivery.session_id,
        "source_digest": delivery.source_digest,
        "task_id": delivery.task_id,
        "temporal_cutoff": delivery.temporal_cutoff,
    }
    if delivery.fork_only:
        body["fork_only"] = True
    return protocol_content_hash(SCHEMA_VERSION_AIG_NATIVE_V2, "failure_gate_fingerprint", body)


def _exception_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    return "store_failed"


def _stored_text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise FailureGateError("stored_record_invalid")
    return value


def _stored_optional(record: Mapping[str, Any], field: str) -> str | None:
    value = record.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise FailureGateError("stored_record_invalid")
    return value
