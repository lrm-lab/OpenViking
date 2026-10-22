# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Reader action tools T_fork (ADR-0044).

Four verbs. ``search_precedents`` calls ``ForkPrecedentService.retrieve``.
``submit_fork_proposal`` calls ``ForkPrecedentService.submit_proposal`` with
the dispositions embedded in the call. ``submit_guidance_plan`` calls P08
``propose`` then ``approve``, so the server still freezes the guidance
document and selects the target. ``expand_experience_links`` is the read-only
hop in ``experience_links``.

The three action verbs stay hidden until ``OutcomeEvidencePolicy`` returns
``confirmed_failure``. Anything else, including a missing assessment, is
``phase_locked``. ``expand_experience_links`` is available once the
construction run is in ``READ_VERIFY`` or ``SYNTHESIZE_PROPOSE``, and the
call must carry ``unresolved``, which is written to the audit.

Fork proposal ``report_hash`` stays the registered revision document hash
(P04). ``authorize_outcome_write`` returns the reader report hash, so this
surface does not pass that authorizer into ``submit_proposal``; doing so
would reject every proposal. Guidance plans still go through ``propose``,
which checks the reader report hash itself.

Submit attempts have three gates. The same proposal content hash returns the
stored overlay and does not spend budget. A different proposal on an anchor
that already has an overlay is ``overlay_conflict``. More than
``DEFAULT_SUBMIT_BUDGET`` non-idempotent attempts fail closed with
``submit_budget_exhausted``. Every attempt, including a refusal, is one
audit row. Principal labels and the reader session are bound on the surface;
the tool arguments cannot replace them.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openviking.session.aig_native_protocol import (
    OBJECT_KIND_FORK_PROPOSAL,
    SCHEMA_VERSION_AIG_NATIVE_V2,
    ProtocolContractError,
    canonical_dumps,
    protocol_content_hash,
)
from openviking.session.aig_role_chain import AigConstructionRun
from openviking.session.append_only import append_jsonl
from openviking.session.experience_links import (
    PROFILE_READER,
    ExperienceLinkError,
    ExperienceLinkService,
)
from openviking.session.experience_cards import (
    CARD_KIND_FORK,
    CARD_KIND_SKILL,
    ExperienceCardError,
    ExperienceCardStore,
)
from openviking.session.fork_node_overlay import ForkNodeOverlayStore, ForkOverlayError
from openviking.session.fork_precedent import (
    ForkPrecedent,
    ForkPrecedentError,
    ForkPrecedentService,
)
from openviking.session.guidance_approval import GuidanceApprovalError, GuidanceApprovalService
from openviking.session.outcome_contract import (
    DECISION_CONFIRMED_FAILURE,
    DECISION_SUCCESS,
    ExternalOutcomeEvidence,
    OutcomeAssessmentService,
    OutcomeContractError,
)
from openviking.session.strategy_pattern import StrategyPatternStore

T_FORK_AUDIT_DIRNAME = "t-fork-audit"
DEFAULT_SUBMIT_BUDGET = 3
ACTION_VERBS = ("search_precedents", "submit_fork_proposal", "submit_guidance_plan")
EXPAND_VERB = "expand_experience_links"
# ADR-0046: offered only when a card store is bound. Not part of ACTION_VERBS,
# so surfaces without a card store keep their closed list unchanged.
CARD_VERB = "submit_experience_card"
DEFAULT_CARD_BUDGET = 6
_CARD_KEYS = frozenset({"proposal"})
_EXPAND_PHASES = frozenset({"READ_VERIFY", "SYNTHESIZE_PROPOSE"})
_ACTION_TEXT = {
    "search_precedents": "Retrieve tiered precedents for one fork proposal.",
    "submit_fork_proposal": "Submit one fork proposal and its precedent dispositions.",
    "submit_guidance_plan": "Propose guidance for an accepted fork and freeze the selected target.",
    "expand_experience_links": "Read one hop of derived experience-network edges.",
    "submit_experience_card": (
        "Propose one skill_card (after a confirmed pass) or one fork card "
        "(after a confirmed failure). The server checks grounding and writes."
    ),
}
_SEARCH_KEYS = frozenset({"proposal", "temporal_cutoff"})
_SUBMIT_KEYS = frozenset({"dispositions", "proposal", "recorded_at", "retrieval_id"})
_GUIDANCE_KEYS = frozenset({"catalog", "proposal"})
_EXPAND_KEYS = frozenset({"context_id", "link_types", "profile", "subject_id", "unresolved"})


_CARD_DECISIONS = frozenset({DECISION_SUCCESS, DECISION_CONFIRMED_FAILURE})


class TForkError(ValueError):
    """A T_fork call was refused. ``code`` is stable."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class TForkSurface:
    """Closed T_fork tool list for one frozen reader session."""

    def __init__(
        self,
        root: str | Path,
        *,
        run: AigConstructionRun,
        session_id: str,
        task_id: str,
        report_hash: str,
        principal_labels: Sequence[str],
        overlays: ForkNodeOverlayStore,
        patterns: StrategyPatternStore,
        precedents: ForkPrecedentService,
        guidance: GuidanceApprovalService,
        outcomes: OutcomeAssessmentService,
        external_evidence: Sequence[ExternalOutcomeEvidence] = (),
        links: ExperienceLinkService | None = None,
        submit_budget: int = DEFAULT_SUBMIT_BUDGET,
        cards: ExperienceCardStore | None = None,
        task_score: float | None = None,
        block_aos: Mapping[str, Sequence[str]] | None = None,
        read_aos: Sequence[str] | None = None,
        ao_participants: Mapping[str, str] | None = None,
        card_budget: int = DEFAULT_CARD_BUDGET,
    ) -> None:
        if not isinstance(run, AigConstructionRun):
            raise TForkError("invalid_run")
        if not isinstance(session_id, str) or not session_id:
            raise TForkError("session_required")
        if not isinstance(task_id, str) or not task_id:
            raise TForkError("task_required")
        if not isinstance(report_hash, str) or not report_hash:
            raise TForkError("report_required")
        if (
            isinstance(submit_budget, bool)
            or not isinstance(submit_budget, int)
            or submit_budget < 1
        ):
            raise TForkError("invalid_budget")
        for store, expected in (
            (overlays, ForkNodeOverlayStore),
            (patterns, StrategyPatternStore),
            (precedents, ForkPrecedentService),
            (guidance, GuidanceApprovalService),
            (outcomes, OutcomeAssessmentService),
        ):
            if not isinstance(store, expected):
                raise TForkError("store_required")
        self._root = Path(root)
        self._run = run
        self._session_id = session_id
        self._task_id = task_id
        self._report_hash = report_hash
        self._principal = tuple(principal_labels)
        self._overlays = overlays
        self._patterns = patterns
        self._precedents = precedents
        self._guidance = guidance
        self._outcomes = outcomes
        self._evidence = tuple(external_evidence)
        self._links = links or ExperienceLinkService(
            root, overlays=overlays, patterns=patterns, guidance=guidance
        )
        self._budget = submit_budget
        if cards is not None and not isinstance(cards, ExperienceCardStore):
            raise TForkError("store_required")
        self._cards = cards
        self._task_score = task_score
        self._block_aos = block_aos
        self._ao_participants = ao_participants
        self._read_aos = read_aos
        self._card_budget = card_budget
        self._card_attempts = 0
        self._card_results: dict[str, dict[str, Any]] = {}
        self._attempts = 0
        self._accepted: dict[str, dict[str, Any]] = {}
        self._fork_ids: set[str] = set()
        self._contexts: dict[str, str] = {}
        self._prev = ""
        self._audit_seq = 0
        self._audit_path = self._root / T_FORK_AUDIT_DIRNAME / "attempts.jsonl"
        self._lock = threading.RLock()
        self._disposition_keys: dict[str, str] = {}

    def catalog(self) -> tuple[tuple[str, str], ...]:
        """Verbs the caller may see. Action verbs appear only after confirmed failure."""
        names: list[str] = []
        if self._expand_open():
            names.append(EXPAND_VERB)
        if self._actions_open():
            names.extend(ACTION_VERBS)
        if self._cards is not None and (self._decision() in _CARD_DECISIONS):
            names.append(CARD_VERB)
        return tuple((name, _ACTION_TEXT[name]) for name in names)

    def call(self, verb: str, arguments: Mapping[str, Any] | None = None) -> Any:
        if not isinstance(verb, str):
            raise TForkError("unknown_verb")
        args = {} if arguments is None else arguments
        if not isinstance(args, Mapping):
            raise TForkError("invalid_args")
        if verb in ACTION_VERBS:
            self._require_actions(verb)
            if verb == "search_precedents":
                return self._search(args)
            if verb == "submit_fork_proposal":
                return self._submit(args)
            return self._guidance_plan(args)
        if verb == EXPAND_VERB:
            self._require_expand()
            return self._expand(args)
        if verb == CARD_VERB and self._cards is not None:
            return self._submit_card(args)
        raise TForkError("unknown_verb")

    def search_precedents(self, query: Mapping[str, Any]) -> dict[str, Any]:
        """Tiered precedents. ACL, self-citation, and replay exclusion stay in P06."""
        return self.call("search_precedents", query)

    def submit_fork_proposal(
        self, proposal: Mapping[str, Any], dispositions: Sequence[Mapping[str, str]], **fields: Any
    ) -> dict[str, Any]:
        "Submit ``proposal`` with dispositions. ``retrieval_id`` and ``recorded_at`` are required."
        payload = {
            "dispositions": list(dispositions),
            "proposal": proposal,
            "recorded_at": fields.get("recorded_at"),
            "retrieval_id": fields.get("retrieval_id"),
        }
        return self.call("submit_fork_proposal", payload)

    def submit_guidance_plan(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        """Intake a plan that cites an accepted fork, then freeze the selected target."""
        return self.call("submit_guidance_plan", plan)

    def expand_experience_links(
        self,
        subject_id: str,
        link_types: Sequence[str] | None = None,
        *,
        unresolved: Sequence[str],
        profile: str = PROFILE_READER,
        context_id: str | None = None,
    ) -> dict[str, Any]:
        """One hop. ``unresolved`` is recorded even when the hop returns no cards."""
        payload: dict[str, Any] = {
            "profile": profile,
            "subject_id": subject_id,
            "unresolved": list(unresolved),
        }
        if link_types is not None:
            payload["link_types"] = list(link_types)
        if context_id is not None:
            payload["context_id"] = context_id
        return self.call(EXPAND_VERB, payload)

    def _search(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        _exact(arguments, _SEARCH_KEYS)
        proposal = arguments.get("proposal")
        cutoff = arguments.get("temporal_cutoff")
        if not isinstance(proposal, Mapping) or not isinstance(cutoff, str):
            raise TForkError("invalid_args")
        try:
            record = self._precedents.retrieve(
                self._overlays,
                self._patterns,
                proposal,
                principal_labels=self._principal,
                session_id=self._session_id,
                temporal_cutoff=cutoff,
            )
        except (ForkPrecedentError, ForkOverlayError, ProtocolContractError) as exc:
            self._audit("search_precedents", ok=False, error=_code_of(exc), detail="")
            raise TForkError(_code_of(exc)) from exc
        self._audit("search_precedents", ok=True, error="", detail=record.retrieval_id)
        return {
            "anchor_block_id": record.anchor_block_id,
            "exploratory": [_hit_dict(hit) for hit in record.exploratory],
            "high_confidence": [_hit_dict(hit) for hit in record.high_confidence],
            "negative_retired": [_hit_dict(hit) for hit in record.negative_retired],
            "retrieval_id": record.retrieval_id,
        }

    def _submit(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        _exact(arguments, _SUBMIT_KEYS)
        proposal = arguments.get("proposal")
        dispositions = arguments.get("dispositions")
        retrieval_id = arguments.get("retrieval_id")
        recorded_at = arguments.get("recorded_at")
        if (
            not isinstance(proposal, Mapping)
            or isinstance(dispositions, (str, bytes))
            or not isinstance(dispositions, Sequence)
        ):
            raise TForkError("invalid_args")
        if not isinstance(retrieval_id, str) or not isinstance(recorded_at, str):
            raise TForkError("invalid_args")
        digest = _proposal_hash(proposal)
        disposition_key = _disposition_key(dispositions)
        with self._lock:
            if digest is not None and self._disposition_keys.get(digest) == disposition_key:
                stored = dict(self._accepted[digest])
                stored["deduplicated"] = True
                self._audit(
                    "submit_fork_proposal", ok=True, error="", detail=stored["fork_node_id"]
                )
                return stored
            if self._attempts >= self._budget:
                self._audit(
                    "submit_fork_proposal",
                    ok=False,
                    error="submit_budget_exhausted",
                    detail=digest or "",
                )
                raise TForkError("submit_budget_exhausted")
            self._attempts += 1
        try:
            commit = self._precedents.submit_proposal(
                self._overlays,
                proposal,
                retrieval_id=retrieval_id,
                dispositions=dispositions,
                principal_labels=self._principal,
                session_id=self._session_id,
                recorded_at=recorded_at,
            )
        except (ForkPrecedentError, ForkOverlayError, ProtocolContractError) as exc:
            self._audit("submit_fork_proposal", ok=False, error=_code_of(exc), detail=digest or "")
            raise TForkError(_code_of(exc)) from exc
        result = {
            "deduplicated": commit.deduplicated,
            "fork_node_id": commit.stored.fork_node_id,
            "proposal_content_hash": commit.stored.proposal_content_hash,
            "retrieval_id": commit.retrieval_id,
        }
        with self._lock:
            key = digest or commit.stored.proposal_content_hash
            self._accepted[key] = dict(result)
            if disposition_key is not None:
                self._disposition_keys[key] = disposition_key
            self._fork_ids.add(commit.stored.fork_node_id)
            self._audit(
                "submit_fork_proposal", ok=True, error="", detail=commit.stored.fork_node_id
            )
        return result

    def _guidance_plan(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        _exact(arguments, _GUIDANCE_KEYS)
        proposal = arguments.get("proposal")
        catalog = arguments.get("catalog")
        if not isinstance(proposal, Mapping) or not isinstance(catalog, Mapping):
            raise TForkError("invalid_args")
        body = proposal.get("body")
        source = body.get("source_fork_id") if isinstance(body, Mapping) else None
        if not isinstance(source, str) or source not in self._fork_ids:
            self._audit("submit_guidance_plan", ok=False, error="fork_required_first", detail="")
            raise TForkError("fork_required_first")
        try:
            proposed = self._guidance.propose(
                self._run,
                self._session_id,
                proposal,
                principal_labels=self._principal,
                external_evidence=self._evidence,
            )
            approved = self._guidance.approve(
                proposed.proposal.content_hash,
                catalog,
                principal_labels=self._principal,
            )
        except (
            GuidanceApprovalError,
            ForkPrecedentError,
            ForkOverlayError,
            OutcomeContractError,
        ) as exc:
            self._audit("submit_guidance_plan", ok=False, error=_code_of(exc), detail=source)
            raise TForkError(_code_of(exc)) from exc
        result = {
            "approval_content_hash": approved.approval.content_hash,
            "deduplicated": proposed.deduplicated and approved.deduplicated,
            "guidance_id": approved.guidance.guidance_id,
            "selected_agent_id": approved.approval.selected_agent_id,
            "source_fork_id": approved.approval.source_fork_id,
        }
        self._audit("submit_guidance_plan", ok=True, error="", detail=result["guidance_id"])
        return result

    def _expand(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if "unresolved" not in arguments:
            raise TForkError("unresolved_required")
        _exact(arguments, _EXPAND_KEYS, optional={"context_id", "link_types", "profile"})
        unresolved = arguments.get("unresolved")
        subject_id = arguments.get("subject_id")
        if not isinstance(subject_id, str) or not isinstance(unresolved, list):
            raise TForkError("unresolved_required")
        if any(not isinstance(item, str) for item in unresolved):
            raise TForkError("unresolved_required")
        profile = arguments.get("profile", PROFILE_READER)
        if not isinstance(profile, str):
            raise TForkError("unknown_profile")
        context_id = arguments.get("context_id", self._contexts.get(profile))
        if context_id is not None and not isinstance(context_id, str):
            raise TForkError("context_not_found")
        link_types = arguments.get("link_types") if "link_types" in arguments else None
        try:
            result = self._links.expand(
                subject_id,
                link_types=link_types,
                principal_labels=self._principal,
                profile=profile,
                context_id=context_id,
                unresolved=unresolved,
            )
        except ExperienceLinkError as exc:
            self._audit(
                EXPAND_VERB,
                ok=False,
                error=exc.code,
                detail=subject_id,
                unresolved=unresolved,
            )
            raise TForkError(exc.code) from exc
        self._contexts[profile] = result["context_id"]
        self._audit(EXPAND_VERB, ok=True, error="", detail=subject_id, unresolved=unresolved)
        return result

    def _decision(self) -> str | None:
        try:
            return self._outcomes.adjudicate(
                task_id=self._task_id,
                report_hash=self._report_hash,
                external_evidence=self._evidence,
            ).decision
        except OutcomeContractError:
            return None

    def _submit_card(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        _exact(arguments, _CARD_KEYS)
        proposal = arguments.get("proposal")
        if not isinstance(proposal, Mapping):
            raise TForkError("invalid_args")
        kind = proposal.get("kind")
        decision = self._decision()
        if kind == CARD_KIND_SKILL and decision != DECISION_SUCCESS:
            self._audit(CARD_VERB, ok=False, error="phase_locked", detail="skill_card")
            raise TForkError("phase_locked")
        if kind == CARD_KIND_FORK and decision != DECISION_CONFIRMED_FAILURE:
            self._audit(CARD_VERB, ok=False, error="phase_locked", detail="fork")
            raise TForkError("phase_locked")
        if kind not in (CARD_KIND_SKILL, CARD_KIND_FORK):
            raise TForkError("invalid_args")
        # The score and author are bound by the server; the Reader cannot set them.
        bound = dict(proposal)
        bound["author_role"] = "reader"
        if kind == CARD_KIND_SKILL:
            bound["task_score"] = self._task_score
        else:
            bound.pop("task_score", None)
        key = canonical_dumps(bound)
        with self._lock:
            if key in self._card_results:
                stored = dict(self._card_results[key])
                stored["deduplicated"] = True
                self._audit(CARD_VERB, ok=True, error="", detail=stored["card_id"])
                return stored
            if self._card_attempts >= self._card_budget:
                self._audit(CARD_VERB, ok=False, error="submit_budget_exhausted", detail="")
                raise TForkError("submit_budget_exhausted")
            self._card_attempts += 1
        assert self._cards is not None
        try:
            card, duplicate = self._cards.submit(
                bound,
                principal_labels=self._principal,
                block_aos=self._block_aos,
                read_aos=self._read_aos,
                ao_participants=self._ao_participants,
            )
        except ExperienceCardError as exc:
            self._audit(CARD_VERB, ok=False, error=exc.code, detail=str(kind))
            raise TForkError(exc.code) from exc
        result = {"card_id": card.card_id, "kind": card.kind, "deduplicated": duplicate}
        with self._lock:
            self._card_results[key] = dict(result)
        self._audit(CARD_VERB, ok=True, error="", detail=card.card_id)
        return result

    def _actions_open(self) -> bool:
        try:
            adjudication = self._outcomes.adjudicate(
                task_id=self._task_id,
                report_hash=self._report_hash,
                external_evidence=self._evidence,
            )
        except OutcomeContractError:
            return False
        return adjudication.decision == DECISION_CONFIRMED_FAILURE

    def _expand_open(self) -> bool:
        return self._run.phase in _EXPAND_PHASES

    def _require_actions(self, verb: str) -> None:
        if self._actions_open():
            return
        self._audit(verb, ok=False, error="phase_locked", detail="")
        raise TForkError("phase_locked")

    def _require_expand(self) -> None:
        if self._expand_open():
            return
        self._audit(EXPAND_VERB, ok=False, error="phase_locked", detail="")
        raise TForkError("phase_locked")

    def _audit(
        self,
        verb: str,
        *,
        ok: bool,
        error: str,
        detail: str,
        unresolved: Sequence[str] = (),
    ) -> None:
        body = {
            "detail": detail,
            "error_code": error,
            "ok": ok,
            "session_id": self._session_id,
            "unresolved": list(unresolved),
            "verb": verb,
        }
        with self._lock:
            payload: dict[str, Any] = {
                "body": body,
                "prev_content_hash": self._prev,
                "record_kind": "t_fork_attempt",
                "seq": self._audit_seq,
            }
            digest = hashlib.sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()
            payload["content_hash"] = digest
            append_jsonl(self._audit_path, payload)
            self._prev = digest
            self._audit_seq += 1


def _exact(
    arguments: Mapping[str, Any],
    allowed: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    keys = set(arguments)
    if not keys <= allowed or not (allowed - optional) <= keys:
        raise TForkError("invalid_args")


def _disposition_key(dispositions: Sequence[Any]) -> str | None:
    rows: list[dict[str, Any]] = []
    for item in dispositions:
        if not isinstance(item, Mapping):
            return None
        rows.append(
            {
                "disposition": item.get("disposition"),
                "precedent_id": item.get("precedent_id"),
                "reason": item.get("reason"),
            }
        )
    rows.sort(key=lambda item: str(item["precedent_id"]))
    return canonical_dumps(rows)


def _proposal_hash(payload: Mapping[str, Any]) -> str | None:
    body = payload.get("body")
    stated = payload.get("content_hash")
    if payload.get("object_kind") != OBJECT_KIND_FORK_PROPOSAL:
        return None
    if payload.get("schema_version") != SCHEMA_VERSION_AIG_NATIVE_V2:
        return None
    if not isinstance(body, dict) or not isinstance(stated, str):
        return None
    expected = protocol_content_hash(SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_FORK_PROPOSAL, body)
    if stated != expected:
        return None
    return stated


def _code_of(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    return "proposal_rejected"


def _hit_dict(hit: ForkPrecedent) -> dict[str, Any]:
    return {
        "base_revision_id": hit.base_revision_id,
        "candidate": hit.candidate,
        "disposition_required": hit.disposition_required,
        "evidence_tier": hit.evidence_tier,
        "fork_node_id": hit.fork_node_id,
        "precedent_id": hit.precedent_id,
        "rank": hit.rank,
        "record_kind": hit.record_kind,
        "recorded_at": hit.recorded_at,
        "relevance": hit.relevance,
        "session_id": hit.session_id,
        "source": hit.source,
        "status": hit.status,
        "subject_id": hit.subject_id,
        "subject_kind": hit.subject_kind,
        "tier": hit.tier,
    }
