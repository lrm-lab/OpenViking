# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Post-freeze skill_card proposals from a finished four-role run (ADR-0046).

After Builder, Critic, Refiner and Reader finish, a proposer provider may
suggest ``skill_card`` proposals over the frozen revision. The server decides:

* a card is allowed only on a verifier pass whose reward is at least
  ``SKILL_CARD_MIN_SCORE``; ``task_score`` and ``author_role`` are written
  from the outcome signal, never from the proposal;
* the revision is registered with the overlay store (ACL = owner label), and
  ``report_hash`` is bound to that revision's document hash;
* AO references are checked against the blocks' source refs and against the
  AOs the Reader actually read.

Fork cards are not produced here: a fork needs a stored ForkNode overlay,
which this run does not create. Failures never change the pipeline result.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openviking.session.aig_toolloop_reader import ReaderNavigationResult
from openviking.session.aig_tool_surface import AoSnapshot
from openviking.session.experience_cards import (
    CARD_KIND_SKILL,
    EXPERIENCE_CARD_PROPOSER_PROMPT,
    SKILL_CARD_MIN_SCORE,
    ExperienceCardError,
    ExperienceCardStore,
)
from openviking.session.fork_node_overlay import ForkNodeOverlayStore, ForkOverlayError
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


# A provider returns proposal dicts. It may return none.
ProposerProvider = Callable[[ProposerTurn], Sequence[Mapping[str, Any]]]


@dataclass(frozen=True)
class CardRunResult:
    status: str  # skipped | proposed
    reason: str
    card_ids: tuple[str, ...] = ()
    rejected: tuple[tuple[str, str], ...] = ()  # (error code, kind)


def outcome_task_score(outcome: OutcomeSignal | None) -> float | None:
    """Reward from a verifier pass; anything else is not a card-eligible outcome."""
    if outcome is None or outcome.signal != "pass" or outcome.source != OUTCOME_SOURCE_VERIFIER:
        return None
    reward = outcome.raw.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        return None
    return float(reward) if float(reward) >= SKILL_CARD_MIN_SCORE else None


def block_ao_index(revision: AigNativeViewRevision) -> dict[str, tuple[str, ...]]:
    return {b.block_id: tuple(ref.ao_id for ref in b.source_refs) for b in revision.blocks}


def read_ao_ids(revision: AigNativeViewRevision, read_sequences: Sequence[int]) -> tuple[str, ...]:
    """AO ids whose sequence the Reader read (Reader refs are AO sequences)."""
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
    if score is None:
        return CardRunResult("skipped", "outcome_not_card_eligible")
    overlays = ForkNodeOverlayStore(card_root)
    labels = [owner_label]
    try:
        catalog = overlays.register_base_revision(revision, acl_labels=labels)
        document_hash = overlays.registered_document_hash(
            catalog.base_revision_id, principal_labels=labels
        )
    except ForkOverlayError as exc:
        return CardRunResult("skipped", exc.code)
    store = ExperienceCardStore(card_root, overlays)
    block_aos = block_ao_index(revision)
    read_aos = read_ao_ids(revision, read.read_ao_refs)
    turn = ProposerTurn(
        system_prompt=EXPERIENCE_CARD_PROPOSER_PROMPT,
        graph=read.graph,
        base_revision_id=catalog.base_revision_id,
        base_revision_block_content_hash=catalog.block_content_hash,
        report_hash=document_hash,
        block_aos={k: v for k, v in block_aos.items()},
        read_aos=read_aos,
        outcome_digest=f"pass reward={score}",
        allowed_kinds=(CARD_KIND_SKILL,),
    )
    try:
        proposals = list(provider(turn))
    except Exception as exc:  # a provider fault never changes the pipeline result
        logger.warning("card proposer failed: %s", type(exc).__name__)
        return CardRunResult("skipped", "provider_error")
    accepted: list[str] = []
    rejected: list[tuple[str, str]] = []
    for proposal in proposals[:MAX_PROPOSALS]:
        kind = str(proposal.get("kind")) if isinstance(proposal, Mapping) else "?"
        if kind != CARD_KIND_SKILL:
            rejected.append(("kind_not_allowed", kind))
            continue
        bound = dict(proposal)
        bound["author_role"] = "reader"
        bound["task_score"] = score
        try:
            card, _dup = store.submit(
                bound, principal_labels=labels, block_aos=block_aos, read_aos=read_aos
            )
        except ExperienceCardError as exc:
            rejected.append((exc.code, kind))
            continue
        accepted.append(card.card_id)
    return CardRunResult("proposed", "complete", tuple(accepted), tuple(rejected))
