# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Reader guidance approval and target selection (ADR-0043 P08).

A Reader proposes a GuidedCounterfactualBranch only for a frozen report whose
OutcomeEvidencePolicy decision is confirmed failure, and only by citing a
precedent snapshot from ``ForkPrecedentService``. The server freezes a
GuidanceDocument projection and the selected agent into one approval. A new
candidate catalog is a new approval version. Memory publishes that snapshot
for a principal, with a caller-supplied timestamp, and cannot add an action
or rewrite the target. The publication record key is the approval content
hash together with that principal.

This module does not dispatch, check a budget, or open a child remediation
run. It does not consume GuidanceExecutionReport.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openviking.session.aig_native_protocol import (
    GUIDANCE_REJECTION_REASON,
    SCHEMA_VERSION_AIG_NATIVE_V2,
    AgentCandidateCatalog,
    GuidanceApproval,
    GuidanceDocument,
    GuidancePublication,
    GuidanceRejection,
    GuidedCounterfactualBranch,
    OutcomeEvidenceSignal,
    guidance_approval_document,
    guidance_document_from_approval,
    guidance_publication_document,
    guidance_rejection_document,
    is_sha256_hex,
    protocol_content_hash,
    protocol_envelope,
    read_agent_candidate_catalog,
    read_guidance_approval,
    read_guidance_document,
    read_guidance_publication,
    read_guidance_rejection,
    read_guided_counterfactual_branch,
)
from openviking.session.aig_role_chain import AigConstructionRun
from openviking.session.append_only import append_jsonl, read_json_object, write_new_json
from openviking.session.fork_node_overlay import (
    ForkNodeOverlayStore,
    ForkOverlayError,
    require_label_set,
)
from openviking.session.fork_precedent import (
    ForkPrecedentError,
    ForkPrecedentService,
    PrecedentRetrievalRecord,
)
from openviking.session.outcome_contract import (
    DECISION_CONFIRMED_FAILURE,
    ExternalOutcomeEvidence,
    OutcomeAssessmentService,
)

GUIDANCE_APPROVAL_DIRNAME = "guidance-approvals"
_ADJUDICATION_KIND = "guidance_adjudication"
_PUBLISHED_AT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


class GuidanceApprovalError(ValueError):
    """A guidance proposal, approval, or publication was refused."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class GuidanceProposalCommit:
    """Stored Reader proposal plus the server adjudication."""

    proposal: GuidedCounterfactualBranch
    decision: str
    reason: str
    policy_version: str
    deduplicated: bool


@dataclass(frozen=True)
class GuidanceApprovalCommit:
    """Frozen approval and the GuidanceDocument projected from it."""

    approval: GuidanceApproval
    guidance: GuidanceDocument
    deduplicated: bool


@dataclass(frozen=True)
class GuidanceApprovalBasis:
    """The session and principal frozen with an approval. Not a dispatch."""

    session_id: str
    principal_labels: tuple[str, ...]
    task_id: str


@dataclass(frozen=True)
class GuidancePublishCommit:
    """The publication snapshot Memory is allowed to render."""

    publication: GuidancePublication
    envelope: dict[str, Any]
    deduplicated: bool


@dataclass(frozen=True)
class _Adjudication:
    proposal_content_hash: str
    policy_version: str
    decision: str
    reason: str
    report_hash: str
    task_id: str
    session_id: str
    principal_labels: tuple[str, ...]
    outcome_evidence: tuple[OutcomeEvidenceSignal, ...]


def select_target_agent(
    proposal: GuidedCounterfactualBranch,
    catalog: AgentCandidateCatalog,
    principal_labels: frozenset[str],
) -> str:
    """Pick one agent. Every gate is required; nothing is ranked or relaxed."""
    if not isinstance(principal_labels, frozenset) or not principal_labels:
        raise GuidanceApprovalError("acl_denied")
    required = set(proposal.required_capabilities)
    eligible = [
        candidate.agent_id
        for candidate in catalog.candidates
        if proposal.target_role in candidate.roles
        and required <= set(candidate.capabilities)
        and principal_labels.intersection(candidate.acl_labels)
        and candidate.task_member
        and candidate.available
    ]
    if not eligible:
        raise GuidanceApprovalError("no_eligible_target")
    return min(eligible)


class GuidanceApprovalService:
    """Append-only guidance store under ``guidance-approvals``."""

    def __init__(
        self,
        root: str | Path,
        *,
        outcomes: OutcomeAssessmentService,
        precedents: ForkPrecedentService,
        overlays: ForkNodeOverlayStore,
    ) -> None:
        if not isinstance(outcomes, OutcomeAssessmentService):
            raise GuidanceApprovalError("store_required")
        if not isinstance(precedents, ForkPrecedentService):
            raise GuidanceApprovalError("store_required")
        if not isinstance(overlays, ForkNodeOverlayStore):
            raise GuidanceApprovalError("store_required")
        self._root = Path(root)
        self._dir = self._root / GUIDANCE_APPROVAL_DIRNAME
        self._index_path = self._dir / "index.jsonl"
        self._outcomes = outcomes
        self._precedents = precedents
        self._overlays = overlays
        self._lock = threading.Lock()

    def propose(
        self,
        run: AigConstructionRun,
        session_id: str,
        payload: Mapping[str, Any],
        *,
        principal_labels: Sequence[str],
        external_evidence: Sequence[ExternalOutcomeEvidence] = (),
    ) -> GuidanceProposalCommit:
        """Store a Reader proposal. Success, uncertain, and missing retrievals are refused."""
        if not isinstance(run, AigConstructionRun):
            raise GuidanceApprovalError("proposal_required")
        granted = run.authorize_outcome_write(session_id)
        if not isinstance(payload, Mapping):
            raise GuidanceApprovalError("proposal_required")
        proposal = read_guided_counterfactual_branch(payload)
        if proposal.report_hash != granted:
            raise GuidanceApprovalError("report_hash_mismatch")
        principal = require_label_set(principal_labels, code="acl_denied")
        record = self._cited_retrieval(proposal.retrieval_id)
        if (
            record.session_id != session_id
            or frozenset(record.principal_labels) != principal
            or record.base_revision_id != proposal.base_revision_id
        ):
            raise GuidanceApprovalError("retrieval_mismatch")
        self._require_source_fork(proposal, record.proposal_content_hash, principal)
        signals = _canonical_signals(external_evidence)
        if not signals:
            raise GuidanceApprovalError("outcome_not_confirmed_failure")
        adjudication = self._outcomes.adjudicate(
            task_id=proposal.task_id,
            report_hash=proposal.report_hash,
            external_evidence=external_evidence,
        )
        if adjudication.decision != DECISION_CONFIRMED_FAILURE:
            raise GuidanceApprovalError("outcome_not_confirmed_failure")
        frozen = _Adjudication(
            proposal_content_hash=proposal.content_hash,
            policy_version=adjudication.policy_version,
            decision=adjudication.decision,
            reason=adjudication.reason,
            report_hash=proposal.report_hash,
            task_id=proposal.task_id,
            session_id=session_id,
            principal_labels=tuple(sorted(principal)),
            outcome_evidence=signals,
        )
        with self._lock:
            existing = self._load_proposal(proposal.content_hash)
            if existing is not None:
                if existing != proposal:
                    raise GuidanceApprovalError("stored_record_invalid")
                if self._load_adjudication(proposal.content_hash) != frozen:
                    raise GuidanceApprovalError("adjudication_conflict")
                return _proposal_commit(existing, frozen, deduplicated=True)
            body = payload.get("body")
            if not isinstance(body, dict):
                raise GuidanceApprovalError("proposal_required")
            self._write_new(
                self._proposal_path(proposal.content_hash),
                protocol_envelope(proposal.schema_version, proposal.object_kind, body),
            )
            self._write_new(
                self._adjudication_path(proposal.content_hash),
                _adjudication_envelope(frozen),
            )
            self._append_index(
                {
                    "content_hash": proposal.content_hash,
                    "record_kind": proposal.object_kind,
                    "task_id": proposal.task_id,
                }
            )
            return _proposal_commit(proposal, frozen, deduplicated=False)

    def approve(
        self,
        proposal_content_hash: str,
        catalog_payload: Mapping[str, Any],
        *,
        principal_labels: Sequence[str],
    ) -> GuidanceApprovalCommit:
        """Select one agent and freeze the approval. A new catalog is a new version."""
        if not is_sha256_hex(proposal_content_hash):
            raise GuidanceApprovalError("proposal_required")
        if not isinstance(catalog_payload, Mapping):
            raise GuidanceApprovalError("catalog_required")
        catalog = read_agent_candidate_catalog(catalog_payload)
        principal = require_label_set(principal_labels, code="acl_denied")
        proposal = self._load_proposal(proposal_content_hash)
        if proposal is None:
            raise GuidanceApprovalError("proposal_required")
        if self._load_rejection(proposal_content_hash) is not None:
            raise GuidanceApprovalError("proposal_rejected")
        basis = self._load_adjudication(proposal_content_hash)
        if basis.decision != DECISION_CONFIRMED_FAILURE:
            raise GuidanceApprovalError("outcome_not_confirmed_failure")
        if catalog.task_id != proposal.task_id:
            raise GuidanceApprovalError("catalog_mismatch")
        if frozenset(basis.principal_labels) != principal:
            raise GuidanceApprovalError("retrieval_mismatch")
        record = self._cited_retrieval(proposal.retrieval_id)
        if (
            record.session_id != basis.session_id
            or frozenset(record.principal_labels) != principal
            or record.base_revision_id != proposal.base_revision_id
        ):
            raise GuidanceApprovalError("retrieval_mismatch")
        self._require_source_fork(proposal, record.proposal_content_hash, principal)
        selected = select_target_agent(proposal, catalog, principal)
        envelope = guidance_approval_document(
            proposal,
            candidate_catalog_hash=catalog.content_hash,
            selected_agent_id=selected,
            policy_version=basis.policy_version,
            decision=basis.decision,
            reason=basis.reason,
            outcome_evidence=[_signal_record(item) for item in basis.outcome_evidence],
        )
        approval = read_guidance_approval(envelope)
        guidance_envelope = guidance_document_from_approval(approval)
        guidance = read_guidance_document(guidance_envelope)
        with self._lock:
            if self._load_rejection(proposal_content_hash) is not None:
                raise GuidanceApprovalError("proposal_rejected")
            existing = self._find_approval(proposal_content_hash, catalog.content_hash)
            if existing is not None:
                if existing != approval:
                    raise GuidanceApprovalError("approval_conflict")
                return GuidanceApprovalCommit(
                    approval=existing, guidance=guidance, deduplicated=True
                )
            catalog_body = catalog_payload.get("body")
            if not isinstance(catalog_body, dict):
                raise GuidanceApprovalError("catalog_required")
            self._store_if_absent(
                self._catalog_path(catalog.content_hash),
                protocol_envelope(catalog.schema_version, catalog.object_kind, catalog_body),
                read_agent_candidate_catalog,
            )
            self._write_new(self._approval_path(approval.content_hash), envelope)
            self._write_new(self._guidance_path(guidance.content_hash), guidance_envelope)
            self._append_index(
                {
                    "candidate_catalog_hash": catalog.content_hash,
                    "content_hash": approval.content_hash,
                    "guidance_id": approval.guidance_id,
                    "proposal_content_hash": proposal.content_hash,
                    "record_kind": approval.object_kind,
                    "selected_agent_id": approval.selected_agent_id,
                }
            )
            return GuidanceApprovalCommit(approval=approval, guidance=guidance, deduplicated=False)

    def publish(
        self,
        approval_content_hash: str,
        *,
        principal_labels: Sequence[str],
        published_at: str,
        render: Mapping[str, Any] | None = None,
    ) -> GuidancePublishCommit:
        """Publish one frozen approval for one principal.

        The publication record key is the approval content hash plus the
        principal. ``published_at`` is stored as given; this method does not
        read a clock. A render that differs from that record is refused.
        """
        if not is_sha256_hex(approval_content_hash):
            raise GuidanceApprovalError("approval_required")
        try:
            publisher = tuple(
                sorted(require_label_set(principal_labels, code="publisher_required"))
            )
        except ForkOverlayError as exc:
            raise GuidanceApprovalError(exc.code) from exc
        if not isinstance(published_at, str) or _PUBLISHED_AT.fullmatch(published_at) is None:
            raise GuidanceApprovalError("published_at_required")
        approval = self.read_approval(approval_content_hash)
        if approval is None:
            raise GuidanceApprovalError("approval_required")
        if self._load_rejection(approval.proposal_content_hash) is not None:
            raise GuidanceApprovalError("proposal_rejected")
        envelope = guidance_publication_document(
            approval, publisher_labels=publisher, published_at=published_at
        )
        publication = read_guidance_publication(envelope)
        if render is not None:
            if not isinstance(render, Mapping):
                raise GuidanceApprovalError("publication_mismatch")
            delta = _publication_delta(publication, read_guidance_publication(render))
            if delta is not None:
                raise GuidanceApprovalError(delta)
        record_id = _publication_record_id(approval.content_hash, publisher)
        with self._lock:
            existing = self._load_publication(record_id)
            if existing is not None:
                if existing != publication:
                    raise GuidanceApprovalError("publication_mismatch")
                return GuidancePublishCommit(
                    publication=existing, envelope=envelope, deduplicated=True
                )
            self._write_new(self._publication_path(record_id), envelope)
            self._append_index(
                {
                    "approval_content_hash": approval.content_hash,
                    "content_hash": publication.content_hash,
                    "publication_record_id": record_id,
                    "published_at": published_at,
                    "publisher_labels": list(publisher),
                    "record_kind": publication.object_kind,
                }
            )
            return GuidancePublishCommit(
                publication=publication, envelope=envelope, deduplicated=False
            )

    def reject(self, proposal_content_hash: str) -> GuidanceRejection:
        """Reject a stored proposal. An approval already frozen stays frozen."""
        if not is_sha256_hex(proposal_content_hash):
            raise GuidanceApprovalError("proposal_required")
        if self._load_proposal(proposal_content_hash) is None:
            raise GuidanceApprovalError("proposal_required")
        if self._has_approval(proposal_content_hash):
            raise GuidanceApprovalError("approval_frozen")
        existing = self._load_rejection(proposal_content_hash)
        if existing is not None:
            return existing
        envelope = guidance_rejection_document(proposal_content_hash)
        rejection = read_guidance_rejection(envelope)
        if rejection.reason != GUIDANCE_REJECTION_REASON:
            raise GuidanceApprovalError("stored_record_invalid")
        with self._lock:
            if self._has_approval(proposal_content_hash):
                raise GuidanceApprovalError("approval_frozen")
            stored = self._load_rejection(proposal_content_hash)
            if stored is not None:
                return stored
            self._write_new(self._rejection_path(proposal_content_hash), envelope)
            self._append_index(
                {
                    "content_hash": rejection.content_hash,
                    "proposal_content_hash": proposal_content_hash,
                    "record_kind": rejection.object_kind,
                }
            )
            return rejection

    def read_basis(self, approval_content_hash: str) -> GuidanceApprovalBasis:
        """Return the frozen session and principal. This does not dispatch."""
        approval = self.read_approval(approval_content_hash)
        if approval is None:
            raise GuidanceApprovalError("approval_required")
        basis = self._load_adjudication(approval.proposal_content_hash)
        return GuidanceApprovalBasis(
            session_id=basis.session_id,
            principal_labels=basis.principal_labels,
            task_id=basis.task_id,
        )

    def read_approval(self, content_hash: str) -> GuidanceApproval | None:
        """Return one approval after checking it still matches its proposal and catalog."""
        if not is_sha256_hex(content_hash):
            return None
        path = self._approval_path(content_hash)
        if not path.is_file():
            return None
        approval = read_guidance_approval(
            read_json_object(path, GuidanceApprovalError, code="stored_record_invalid")
        )
        if approval.content_hash != content_hash:
            raise GuidanceApprovalError("stored_record_invalid")
        proposal = self._load_proposal(approval.proposal_content_hash)
        basis = self._load_adjudication(approval.proposal_content_hash)
        catalog = self._load_catalog(approval.candidate_catalog_hash)
        guidance = self._load_guidance(approval.guidance_content_hash)
        if (
            proposal is None
            or catalog is None
            or guidance is None
            or approval.steps != proposal.steps
            or approval.capability_limits != proposal.required_capabilities
            or approval.retrieval_id != proposal.retrieval_id
            or approval.report_hash != proposal.report_hash
            or approval.source_fork_id != proposal.source_fork_id
            or approval.base_revision_id != proposal.base_revision_id
            or approval.target_role != proposal.target_role
            or approval.decision != basis.decision
            or approval.reason != basis.reason
            or approval.policy_version != basis.policy_version
            or approval.outcome_evidence != basis.outcome_evidence
            or guidance.content_hash != approval.guidance_content_hash
        ):
            raise GuidanceApprovalError("stored_record_invalid")
        selected = select_target_agent(proposal, catalog, frozenset(basis.principal_labels))
        if selected != approval.selected_agent_id:
            raise GuidanceApprovalError("stored_record_invalid")
        return approval

    def read_publication(
        self, approval_content_hash: str, *, principal_labels: Sequence[str]
    ) -> GuidancePublication | None:
        """Return one principal's publication of an approval, if Memory published it."""
        if not is_sha256_hex(approval_content_hash):
            return None
        try:
            publisher = tuple(
                sorted(require_label_set(principal_labels, code="publisher_required"))
            )
        except ForkOverlayError as exc:
            raise GuidanceApprovalError(exc.code) from exc
        approval = self.read_approval(approval_content_hash)
        if approval is None:
            return None
        publication = self._load_publication(
            _publication_record_id(approval.content_hash, publisher)
        )
        if publication is None:
            return None
        expected = read_guidance_publication(
            guidance_publication_document(
                approval,
                publisher_labels=publisher,
                published_at=publication.published_at,
            )
        )
        if publication != expected or publication.publisher_labels != publisher:
            raise GuidanceApprovalError("stored_record_invalid")
        return publication

    def _cited_retrieval(self, retrieval_id: str) -> PrecedentRetrievalRecord:
        try:
            return self._precedents.read_retrieval(retrieval_id)
        except ForkPrecedentError as exc:
            if exc.code == "retrieval_not_found":
                raise GuidanceApprovalError("retrieval_required") from exc
            raise

    def _require_source_fork(
        self,
        proposal: GuidedCounterfactualBranch,
        fork_proposal_hash: str,
        principal: frozenset[str],
    ) -> None:
        stored = self._overlays.read_overlay_by_id(
            proposal.source_fork_id, principal_labels=tuple(sorted(principal))
        )
        if stored is None:
            raise GuidanceApprovalError("source_fork_unknown")
        if (
            stored.proposal_content_hash != fork_proposal_hash
            or stored.overlay.base_revision_id != proposal.base_revision_id
        ):
            raise GuidanceApprovalError("retrieval_mismatch")

    def _find_approval(
        self, proposal_content_hash: str, catalog_hash: str
    ) -> GuidanceApproval | None:
        directory = self._dir / "approvals"
        if not directory.is_dir():
            return None
        found: GuidanceApproval | None = None
        for path in sorted(directory.glob("*.json")):
            approval = read_guidance_approval(
                read_json_object(path, GuidanceApprovalError, code="stored_record_invalid")
            )
            if (
                approval.proposal_content_hash != proposal_content_hash
                or approval.candidate_catalog_hash != catalog_hash
            ):
                continue
            if found is not None and found != approval:
                raise GuidanceApprovalError("approval_conflict")
            found = approval
        return found

    def _has_approval(self, proposal_content_hash: str) -> bool:
        directory = self._dir / "approvals"
        if not directory.is_dir():
            return False
        for path in directory.glob("*.json"):
            payload = read_json_object(path, GuidanceApprovalError, code="stored_record_invalid")
            body = payload.get("body")
            if (
                isinstance(body, dict)
                and body.get("proposal_content_hash") == proposal_content_hash
            ):
                return True
        return False

    def _load_proposal(self, content_hash: str) -> GuidedCounterfactualBranch | None:
        return _load_hashed(
            self._proposal_path(content_hash),
            content_hash,
            read_guided_counterfactual_branch,
        )

    def _load_adjudication(self, proposal_content_hash: str) -> _Adjudication:
        path = self._adjudication_path(proposal_content_hash)
        if not path.is_file():
            raise GuidanceApprovalError("stored_record_invalid")
        payload = read_json_object(path, GuidanceApprovalError, code="stored_record_invalid")
        body = payload.get("body")
        if (
            payload.get("schema_version") != SCHEMA_VERSION_AIG_NATIVE_V2
            or payload.get("object_kind") != _ADJUDICATION_KIND
            or not isinstance(body, dict)
        ):
            raise GuidanceApprovalError("stored_record_invalid")
        digest = protocol_content_hash(SCHEMA_VERSION_AIG_NATIVE_V2, _ADJUDICATION_KIND, body)
        if payload.get("content_hash") != digest:
            raise GuidanceApprovalError("stored_record_invalid")
        evidence = body.get("outcome_evidence")
        if not isinstance(evidence, list):
            raise GuidanceApprovalError("stored_record_invalid")
        labels = body.get("principal_labels")
        if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
            raise GuidanceApprovalError("stored_record_invalid")
        signals = tuple(
            OutcomeEvidenceSignal(
                binding_id=str(item["binding_id"]),
                signal_kind=str(item["signal_kind"]),
                signal=str(item["signal"]),
                evidence_ref=str(item["evidence_ref"]),
            )
            for item in evidence
            if isinstance(item, dict)
        )
        if len(signals) != len(evidence):
            raise GuidanceApprovalError("stored_record_invalid")
        return _Adjudication(
            proposal_content_hash=_text(body, "proposal_content_hash"),
            policy_version=_text(body, "policy_version"),
            decision=_text(body, "decision"),
            reason=_text(body, "reason"),
            report_hash=_text(body, "report_hash"),
            task_id=_text(body, "task_id"),
            session_id=_text(body, "session_id"),
            principal_labels=tuple(labels),
            outcome_evidence=signals,
        )

    def _load_catalog(self, content_hash: str) -> AgentCandidateCatalog | None:
        return _load_hashed(
            self._catalog_path(content_hash), content_hash, read_agent_candidate_catalog
        )

    def _load_guidance(self, content_hash: str) -> GuidanceDocument | None:
        return _load_hashed(self._guidance_path(content_hash), content_hash, read_guidance_document)

    def _load_publication(self, record_id: str) -> GuidancePublication | None:
        path = self._publication_path(record_id)
        if not path.is_file():
            return None
        return read_guidance_publication(
            read_json_object(path, GuidanceApprovalError, code="stored_record_invalid")
        )

    def _load_rejection(self, proposal_content_hash: str) -> GuidanceRejection | None:
        path = self._rejection_path(proposal_content_hash)
        if not path.is_file():
            return None
        return read_guidance_rejection(
            read_json_object(path, GuidanceApprovalError, code="stored_record_invalid")
        )

    def _store_if_absent(
        self,
        path: Path,
        payload: Mapping[str, Any],
        reader: Callable[[Mapping[str, Any]], AgentCandidateCatalog],
    ) -> None:
        if path.is_file():
            existing = reader(
                read_json_object(path, GuidanceApprovalError, code="stored_record_invalid")
            )
            loaded = reader(payload)
            if existing != loaded:
                raise GuidanceApprovalError("stored_record_invalid")
            return
        self._write_new(path, payload)

    def _proposal_path(self, content_hash: str) -> Path:
        return self._dir / "proposals" / f"{content_hash}.json"

    def _adjudication_path(self, proposal_content_hash: str) -> Path:
        return self._dir / "adjudications" / f"{proposal_content_hash}.json"

    def _catalog_path(self, content_hash: str) -> Path:
        return self._dir / "catalogs" / f"{content_hash}.json"

    def _approval_path(self, content_hash: str) -> Path:
        return self._dir / "approvals" / f"{content_hash}.json"

    def _guidance_path(self, content_hash: str) -> Path:
        return self._dir / "guidance" / f"{content_hash}.json"

    def _publication_path(self, record_id: str) -> Path:
        return self._dir / "publications" / f"{record_id}.json"

    def _rejection_path(self, proposal_content_hash: str) -> Path:
        return self._dir / "rejections" / f"{proposal_content_hash}.json"

    def _append_index(self, row: Mapping[str, Any]) -> None:
        append_jsonl(self._index_path, row)

    def _write_new(self, path: Path, payload: Mapping[str, Any]) -> None:
        write_new_json(path, payload, GuidanceApprovalError, code="stored_record_invalid")


def read_stored_guidance_publications(root: str | Path) -> tuple[GuidancePublication, ...]:
    """Read publication files in path order. This does not rewrite them.

    An empty directory is an empty tuple. A corrupt file fails closed.
    """
    directory = Path(root) / GUIDANCE_APPROVAL_DIRNAME / "publications"
    if not directory.is_dir():
        return ()
    found: list[GuidancePublication] = []
    for path in sorted(directory.glob("*.json")):
        found.append(
            read_guidance_publication(
                read_json_object(path, GuidanceApprovalError, code="stored_record_invalid")
            )
        )
    return tuple(found)


def _proposal_commit(
    proposal: GuidedCounterfactualBranch, basis: _Adjudication, *, deduplicated: bool
) -> GuidanceProposalCommit:
    return GuidanceProposalCommit(
        proposal=proposal,
        decision=basis.decision,
        reason=basis.reason,
        policy_version=basis.policy_version,
        deduplicated=deduplicated,
    )


def _canonical_signals(
    evidence: Sequence[ExternalOutcomeEvidence],
) -> tuple[OutcomeEvidenceSignal, ...]:
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise GuidanceApprovalError("outcome_evidence_invalid")
    items: list[OutcomeEvidenceSignal] = []
    for item in evidence:
        if not isinstance(item, ExternalOutcomeEvidence):
            raise GuidanceApprovalError("outcome_evidence_invalid")
        items.append(
            OutcomeEvidenceSignal(
                binding_id=item.binding_id,
                signal_kind=item.signal_kind,
                signal=item.signal,
                evidence_ref=item.evidence_ref,
            )
        )
    items.sort(
        key=lambda signal: (
            signal.binding_id,
            signal.signal_kind,
            signal.signal,
            signal.evidence_ref,
        )
    )
    unique: list[OutcomeEvidenceSignal] = []
    seen: set[tuple[str, str, str, str]] = set()
    for signal in items:
        key = (signal.binding_id, signal.signal_kind, signal.signal, signal.evidence_ref)
        if key in seen:
            continue
        seen.add(key)
        unique.append(signal)
    return tuple(unique)


def _signal_record(signal: OutcomeEvidenceSignal) -> dict[str, str]:
    return {
        "binding_id": signal.binding_id,
        "evidence_ref": signal.evidence_ref,
        "signal": signal.signal,
        "signal_kind": signal.signal_kind,
    }


def _adjudication_envelope(basis: _Adjudication) -> dict[str, Any]:
    body = {
        "decision": basis.decision,
        "outcome_evidence": [_signal_record(item) for item in basis.outcome_evidence],
        "policy_version": basis.policy_version,
        "principal_labels": list(basis.principal_labels),
        "proposal_content_hash": basis.proposal_content_hash,
        "reason": basis.reason,
        "report_hash": basis.report_hash,
        "session_id": basis.session_id,
        "task_id": basis.task_id,
    }
    return protocol_envelope(SCHEMA_VERSION_AIG_NATIVE_V2, _ADJUDICATION_KIND, body)


def _publication_record_id(approval_content_hash: str, publisher_labels: Sequence[str]) -> str:
    """Idempotency key for one approval published by one principal."""
    return protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        "guidance_publication_record",
        {
            "approval_content_hash": approval_content_hash,
            "publisher_labels": list(publisher_labels),
        },
    )


def _publication_delta(expected: GuidancePublication, submitted: GuidancePublication) -> str | None:
    if (
        submitted.selected_agent_id != expected.selected_agent_id
        or submitted.target_role != expected.target_role
    ):
        return "target_rewrite_forbidden"
    if submitted.steps != expected.steps:
        return "action_not_in_proposal"
    if (
        submitted.evidence_refs != expected.evidence_refs
        or submitted.capability_limits != expected.capability_limits
    ):
        return "unevidenced_action"
    if (
        submitted.decision != expected.decision
        or submitted.reason != expected.reason
        or submitted.policy_version != expected.policy_version
        or submitted.outcome_evidence != expected.outcome_evidence
        or submitted.report_hash != expected.report_hash
    ):
        return "basis_rewrite_forbidden"
    if submitted != expected:
        return "publication_mismatch"
    return None


def _load_hashed(path: Path, content_hash: str, reader):
    if not path.is_file():
        return None
    loaded = reader(read_json_object(path, GuidanceApprovalError, code="stored_record_invalid"))
    if loaded.content_hash != content_hash:
        raise GuidanceApprovalError("stored_record_invalid")
    return loaded


def _text(body: Mapping[str, Any], field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str) or not value:
        raise GuidanceApprovalError("stored_record_invalid")
    return value
