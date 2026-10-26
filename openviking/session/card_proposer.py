# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Post-freeze experience-card proposals from a finished four-role run (ADR-0046).

After Builder, Critic, Refiner and Reader finish, a proposer provider may
suggest cards over the frozen revision. The server decides:

* ``skill_card`` is allowed only on a verifier pass whose reward is at least
  ``SKILL_CARD_MIN_SCORE``; ``task_score`` and ``author_role`` are written
  from the outcome signal, never from the proposal;
* ``fork`` is allowed only on a verifier fail. The proposal names the block
  where the run went wrong (``anchor_block_id``); the server writes the
  ForkNode overlay for it (one imagined node per card step) and then the card
  that points at that overlay. ``violates`` / ``diagnosis_report_id`` are not
  taken from the proposal;
* the revision is registered with the overlay store (ACL = owner label), and
  ``report_hash`` is bound to that revision's document hash;
* AO references are checked against the blocks' source refs and against the
  AOs the Reader actually read, and a card must cite at least one such AO:
  an experience that cannot point at the trajectory is not stored.

Failures never change the pipeline result.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openviking.session.aig_native_protocol import (
    OBJECT_KIND_FORK_PROPOSAL,
    SCHEMA_VERSION_AIG_NATIVE_V2,
    ProtocolContractError,
    protocol_envelope,
)
from openviking.session.aig_toolloop_reader import ReaderNavigationResult
from openviking.session.aig_tool_surface import AoSnapshot, _participant
from openviking.session.experience_cards import (
    CARD_KIND_FORK,
    CARD_KIND_SKILL,
    EXPERIENCE_CARD_PROPOSER_PROMPT,
    SKILL_CARD_MIN_SCORE,
    ExperienceCardError,
    ExperienceCardStore,
    read_card_proposal,
)
from openviking.session.fork_node_overlay import (
    ForkNodeOverlayStore,
    ForkOverlayError,
    fork_node_stable_id,
)
from openviking.session.influence_view import AigNativeViewRevision
from openviking.session.outcome_signal import OUTCOME_SOURCE_VERIFIER, OutcomeSignal

logger = logging.getLogger(__name__)
MAX_PROPOSALS = 3


@dataclass(frozen=True)
class ProposerTurn:
    """What the provider sees: prompt, frozen graph, and the facts it may rely on."""

    system_prompt: str
    graph: dict[str, Any]
    base_revision_id: str
    base_revision_block_content_hash: str
    report_hash: str
    block_aos: dict[str, tuple[str, ...]]
    read_aos: tuple[str, ...]
    outcome_digest: str
    allowed_kinds: tuple[str, ...]
    # Judge report text on a failed run. It points at what to look for; it is not a source.
    outcome_feedback: str = ""


# A provider returns proposal dicts. It may return none.
ProposerProvider = Callable[[ProposerTurn], Sequence[Mapping[str, Any]]]


@dataclass(frozen=True)
class CardRunResult:
    status: str  # skipped | proposed
    reason: str
    card_ids: tuple[str, ...] = ()
    rejected: tuple[tuple[str, str], ...] = ()  # (error code, kind)
    trace: dict[str, Any] | None = field(default=None, compare=False)


def outcome_task_score(outcome: OutcomeSignal | None) -> float | None:
    """Reward from a verifier pass; anything else is not a card-eligible outcome."""
    if outcome is None or outcome.signal != "pass" or outcome.source != OUTCOME_SOURCE_VERIFIER:
        return None
    reward = outcome.raw.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        return None
    return float(reward) if float(reward) >= SKILL_CARD_MIN_SCORE else None


def outcome_is_fork_eligible(outcome: OutcomeSignal | None) -> bool:
    """A verifier-confirmed failure. Model-inferred failures never write a fork card."""
    return (
        outcome is not None
        and outcome.signal == "fail"
        and outcome.source == OUTCOME_SOURCE_VERIFIER
    )


def _outcome_reward(outcome: OutcomeSignal) -> float | None:
    reward = outcome.raw.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        return None
    return float(reward)


def block_ao_index(revision: AigNativeViewRevision) -> dict[str, tuple[str, ...]]:
    return {b.block_id: tuple(ref.ao_id for ref in b.source_refs) for b in revision.blocks}


def read_ao_ids(
    revision: AigNativeViewRevision,
    read_sequences: Sequence[int],
    *,
    snapshot: AoSnapshot | None = None,
) -> tuple[str, ...]:
    """Map raw Reader positions by snapshot identity; preserve legacy matching by default."""
    if os.environ.get("OPENVIKING_CARD_READ_REF_ZERO_BASED") == "1":
        if snapshot is None:
            raise ValueError("raw_read_snapshot_required")
        read_ids: set[str] = set()
        for index in read_sequences:
            if type(index) is not int or not 0 <= index < len(snapshot.records):
                raise ValueError("invalid_raw_read_index")
            read_ids.add(snapshot.records[index].ao_id)
        block_ids = {ref.ao_id for block in revision.blocks for ref in block.source_refs}
        return tuple(sorted(read_ids & block_ids))
    wanted = set(read_sequences)
    found = {
        ref.ao_id
        for block in revision.blocks
        for ref in block.source_refs
        if ref.sequence in wanted
    }
    return tuple(sorted(found))


def run_card_proposer(
    revision: AigNativeViewRevision,
    snapshot: AoSnapshot,
    read: ReaderNavigationResult,
    provider: ProposerProvider,
    *,
    card_root: str | Path,
    owner_label: str,
) -> CardRunResult:
    score = outcome_task_score(snapshot.outcome)
    if score is not None:
        kind_allowed = CARD_KIND_SKILL
    elif outcome_is_fork_eligible(snapshot.outcome):
        kind_allowed = CARD_KIND_FORK
    else:
        return CardRunResult("skipped", "outcome_not_card_eligible", trace={"verdict": "outcome_not_card_eligible"})
    outcome = snapshot.outcome
    assert outcome is not None
    reward = _outcome_reward(outcome)
    digest = (
        f"pass reward={score}"
        if kind_allowed == CARD_KIND_SKILL
        else "fail" + (f" reward={reward}" if reward is not None else "")
    )
    feedback = outcome.raw.get("feedback")
    # Validate all raw positions before registering a revision or invoking a provider.
    read_aos = (
        read_ao_ids(revision, read.read_ao_refs, snapshot=snapshot)
        if os.environ.get("OPENVIKING_CARD_READ_REF_ZERO_BASED") == "1" else None
    )
    overlays = ForkNodeOverlayStore(card_root)
    labels = [owner_label]
    try:
        catalog = overlays.register_base_revision(revision, acl_labels=labels)
        document_hash = overlays.registered_document_hash(
            catalog.base_revision_id, principal_labels=labels
        )
    except ForkOverlayError as exc:
        return CardRunResult("skipped", exc.code, trace={"verdict": exc.code, "outcome": digest})
    store = ExperienceCardStore(card_root, overlays)
    block_aos = block_ao_index(revision)
    if read_aos is None:
        read_aos = read_ao_ids(revision, read.read_ao_refs)
    # Server-bound identity of each cited AO's actor (cycle 006 D-004).
    ao_participants = {record.ao_id: _participant(record) for record in snapshot.records}
    turn = ProposerTurn(
        system_prompt=EXPERIENCE_CARD_PROPOSER_PROMPT,
        graph=read.graph,
        base_revision_id=catalog.base_revision_id,
        base_revision_block_content_hash=catalog.block_content_hash,
        report_hash=document_hash,
        block_aos={k: v for k, v in block_aos.items()},
        read_aos=read_aos,
        outcome_digest=digest,
        allowed_kinds=(kind_allowed,),
        outcome_feedback=feedback.strip() if isinstance(feedback, str) else "",
    )
    attempts = lambda: list(getattr(provider, "attempts", ()))  # noqa: E731
    trace: dict[str, Any] = {
        "allowed_kinds": list(turn.allowed_kinds), "outcome": digest, "read_aos": len(read_aos),
    }
    try:
        proposals = list(provider(turn))
    except Exception as exc:  # a provider fault never changes the pipeline result
        logger.warning("card proposer failed: %s", type(exc).__name__)
        done = attempts()
        verdict = (
            "provider_timeout" if isinstance(exc, TimeoutError)
            else "retries_exhausted" if done and all(a["result"] == "rejected" for a in done)
            else "provider_error"
        )
        trace.update(verdict=verdict, attempts=done, error=f"{type(exc).__name__}: {exc}"[:300])
        return CardRunResult("skipped", "provider_error", trace=trace)
    accepted: list[str] = []
    rejected: list[tuple[str, str]] = []
    for proposal in proposals[:MAX_PROPOSALS]:
        kind = str(proposal.get("kind")) if isinstance(proposal, Mapping) else "?"
        if kind != kind_allowed:
            rejected.append(("kind_not_allowed", kind))
            continue
        if not has_trajectory_evidence(proposal):
            rejected.append(("no_trajectory_evidence", kind))
            continue
        bound = dict(proposal)
        # Identity of the frozen revision is bound by the server, not copied by the model.
        bound["base_revision_id"] = catalog.base_revision_id
        bound["base_revision_block_content_hash"] = catalog.block_content_hash
        bound["report_hash"] = document_hash
        bound["author_role"] = "reader"
        try:
            if kind == CARD_KIND_SKILL:
                bound["task_score"] = score
            else:
                bound = _bind_fork(
                    bound, overlays, labels, catalog, document_hash,
                    block_aos, read_aos, ao_participants,
                )
            card, _dup = store.submit(
                bound,
                principal_labels=labels,
                block_aos=block_aos,
                read_aos=read_aos,
                ao_participants=ao_participants,
            )
        except (ExperienceCardError, ForkOverlayError, ProtocolContractError) as exc:
            rejected.append((getattr(exc, "code", type(exc).__name__), kind))
            continue
        accepted.append(card.card_id)
    trace.update(
        verdict="model_returned_empty" if not proposals else "proposed" if accepted else "proposals_all_rejected",
        attempts=attempts(), proposals_returned=len(proposals), accepted=len(accepted), rejected=[list(r) for r in rejected],
    )
    return CardRunResult("proposed", "complete", tuple(accepted), tuple(rejected), trace)


def has_trajectory_evidence(proposal: Any) -> bool:
    """At least one step cites an AO. Whether the citation is real is the store's check."""
    steps = proposal.get("steps") if isinstance(proposal, Mapping) else None
    if not isinstance(steps, list):
        return False
    return any(
        isinstance(step, Mapping) and isinstance(step.get("ao_refs"), list) and step["ao_refs"]
        for step in steps
    )


def _bind_fork(
    proposal: dict[str, Any],
    overlays: ForkNodeOverlayStore,
    labels: Sequence[str],
    catalog: Any,
    document_hash: str,
    block_aos: Mapping[str, Sequence[str]],
    read_aos: Sequence[str],
    ao_participants: Mapping[str, str],
) -> dict[str, Any]:
    """Write the ForkNode overlay for ``anchor_block_id`` and point the card at it.

    The branch is one imagined node per card step, named by the server. The model does not
    choose ids, so they cannot collide with a factual block, and ``violates`` /
    ``diagnosis_report_id`` / ``fork_node_id`` from the model are dropped.
    """
    anchor = proposal.pop("anchor_block_id", None)
    for key in ("fork_node_id", "violates", "diagnosis_report_id", "task_score"):
        proposal.pop(key, None)
    if not isinstance(anchor, str) or not anchor:
        raise ExperienceCardError("fork_anchor_missing", "a fork proposal names anchor_block_id")
    steps = proposal.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ExperienceCardError("shape", "steps must be a non-empty list")
    ids = [s.get("step_id") for s in steps if isinstance(s, Mapping)]
    if len(ids) != len(steps) or any(not isinstance(i, str) or not i for i in ids):
        raise ExperienceCardError("shape", "every step needs a step_id")
    body = {
        "anchor_block_id": anchor,
        "author_role": "reader",
        "base_revision_block_content_hash": catalog.block_content_hash,
        "base_revision_id": catalog.base_revision_id,
        "branch_steps": [
            {"imagined_node_id": f"img-{step_id}", "step_kind": "imagined_node"} for step_id in ids
        ],
        "report_hash": document_hash,
    }
    gamma = proposal.get("gamma")
    if isinstance(gamma, Mapping):
        anchors = list(gamma.get("anchors") or [])
        if anchor not in anchors:
            anchors.append(anchor)
        proposal["gamma"] = {**gamma, "anchors": anchors}
    # The id is a function of (revision, anchor, block hash), so the card can be validated
    # before anything is written: a bad card must not leave an overlay behind.
    proposal["fork_node_id"] = fork_node_stable_id(
        catalog.base_revision_id, anchor, catalog.block_content_hash
    )
    read_card_proposal(
        proposal, catalog, block_aos=block_aos, read_aos=read_aos, ao_participants=ao_participants
    )
    envelope = protocol_envelope(SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_FORK_PROPOSAL, body)
    commit = overlays.submit_proposal(envelope, principal_labels=labels)
    if commit.stored.fork_node_id != proposal["fork_node_id"]:
        raise ExperienceCardError("fork_overlay_missing", "overlay id differs from the stable id")
    return proposal
