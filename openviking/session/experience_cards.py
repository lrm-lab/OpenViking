# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Experience cards: ``skill_card`` and ``fork`` (ADR-0046, Q237-Q254).

A card is an immutable overlay on one frozen factual AIG revision. Two kinds
share one schema:

* ``skill_card`` (Method): anchors a critical path of factual blocks from a
  passing run. Its steps are typed ``produce`` or ``verify``.
* ``fork`` (Guard): a sidecar for an existing ForkNode overlay. It adds the
  trigger ``gamma``, the boundary, and an optional ``violates`` link to one
  ``(skill_card, verify step)``. The overlay document itself is not rewritten,
  so its content hash and idempotency keys stay valid.

Grounding: block anchors are mandatory and must exist in the registered base
revision. ``ao_refs`` on a step are optional. When present, the caller-supplied
block-to-AO index must contain them. Cards are written only by the server
after this validation; the Reader only proposes.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openviking.session.aig_native_protocol import (
    SCHEMA_VERSION_AIG_NATIVE_V2,
    BaseRevisionCatalog,
    canonical_dumps,
)
from openviking.session.append_only import append_jsonl, write_exclusive
from openviking.session.bm25 import BM25Index, tokenize
from openviking.session.fork_node_overlay import (
    ForkNodeOverlayStore,
    ForkOverlayError,
    require_label_set,
)

CARD_KIND_SKILL = "skill_card"
CARD_KIND_FORK = "fork"
CARD_KINDS = frozenset({CARD_KIND_SKILL, CARD_KIND_FORK})
STEP_PRODUCE = "produce"
STEP_VERIFY = "verify"
STEP_KINDS = frozenset({STEP_PRODUCE, STEP_VERIFY})

DELIVERABLE_KINDS = ("pdf", "docx", "xlsx", "pptx", "av", "text")
FAILURE_KINDS = (
    "no_artifact",
    "placeholder",
    "constraint",
    "missing_section",
    "incorrect",
    "format",
    "domain",
)
# A skill card is proposed only from runs scoring at least this (Q243).
SKILL_CARD_MIN_SCORE = 0.6
CARDS_DIRNAME = "experience-cards"
OBJECT_KIND_EXPERIENCE_CARD = "experience_card"
_READER_ROLE = "reader"
_BOOST_OCCUPATION = 0.3
_BOOST_ISSUE = 0.3


class ExperienceCardError(ValueError):
    """A card proposal, write, or read was refused."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class CardGamma:
    """Trigger: frozen slots plus grounded abstract text."""

    deliverable: str
    abstract_text: str
    anchors: tuple[str, ...]
    occupation: str | None = None
    issue_kind: str | None = None


@dataclass(frozen=True)
class CardStep:
    step_id: str
    step_kind: str
    statement: str
    block_id: str
    ao_refs: tuple[str, ...] = ()
    issue_kind: str | None = None  # verify: the issue checked
    check_how: str | None = None  # verify only
    violation_sign: str | None = None  # verify only
    recovery: bool = False  # produce: an error -> repair fragment


@dataclass(frozen=True)
class ViolatesRef:
    card_id: str
    step_id: str


@dataclass(frozen=True)
class ExperienceCard:
    card_id: str
    kind: str
    base_revision_id: str
    base_revision_block_content_hash: str
    report_hash: str
    author_role: str
    gamma: CardGamma
    boundary: str
    steps: tuple[CardStep, ...]
    task_score: float | None = None
    fork_node_id: str | None = None
    violates: ViolatesRef | None = None
    diagnosis_report_id: str | None = None

    @property
    def anchor_block_ids(self) -> tuple[str, ...]:
        return self.gamma.anchors


@dataclass(frozen=True)
class CardHit:
    card: ExperienceCard
    score: float


@dataclass(frozen=True)
class CardFindResult:
    hits: tuple[CardHit, ...]
    relaxed: bool  # the deliverable filter was dropped because it matched nothing


# --------------------------------------------------------------------------
# Parsing and validation
# --------------------------------------------------------------------------

def read_card_proposal(
    payload: Mapping[str, Any],
    catalog: BaseRevisionCatalog,
    *,
    block_aos: Mapping[str, Sequence[str]] | None = None,
    read_aos: Sequence[str] | None = None,
) -> ExperienceCard:
    """Validate a Reader proposal against the registered revision.

    ``block_aos`` maps block id to the AO ids that block contains. It is
    required only when some step carries ``ao_refs``. ``read_aos``, when
    given, is the set of AO ids the Reader actually read.
    """
    if not isinstance(catalog, BaseRevisionCatalog):
        raise ExperienceCardError("catalog_required")
    if not isinstance(payload, Mapping):
        raise ExperienceCardError("shape", "proposal must be a dict")
    kind = _choice(payload.get("kind"), "kind", CARD_KINDS)
    if payload.get("author_role") != _READER_ROLE:
        raise ExperienceCardError("author_role", "author_role must be reader")
    if payload.get("base_revision_id") != catalog.base_revision_id:
        raise ExperienceCardError("base_revision_mismatch")
    if payload.get("base_revision_block_content_hash") != catalog.block_content_hash:
        raise ExperienceCardError("block_hash_mismatch")
    members = frozenset(catalog.block_ids)

    gamma = _parse_gamma(payload.get("gamma"), members)
    boundary = _text(payload.get("boundary"), "boundary")
    steps = _parse_steps(payload.get("steps"), members, block_aos, read_aos)
    score = _score(payload.get("task_score"))

    fork_node_id = payload.get("fork_node_id")
    violates = _parse_violates(payload.get("violates"))
    report_id = payload.get("diagnosis_report_id")
    if kind == CARD_KIND_SKILL:
        _check_skill(steps, score, fork_node_id, violates, report_id)
    else:
        if not isinstance(fork_node_id, str) or not fork_node_id:
            raise ExperienceCardError("fork_node_required", "a fork card names its overlay")
        if report_id is not None and (not isinstance(report_id, str) or not report_id):
            raise ExperienceCardError("shape", "diagnosis_report_id must be a non-empty str")
        if any(step.step_kind == STEP_VERIFY for step in steps):
            raise ExperienceCardError("fork_step_kind", "fork steps are not verify steps")
    report_hash = _text(payload.get("report_hash"), "report_hash")
    body = _card_body(
        kind, catalog, report_hash, gamma, boundary, steps, score,
        fork_node_id if kind == CARD_KIND_FORK else None,
        violates, report_id if kind == CARD_KIND_FORK else None,
    )
    return _card_from_body(body)


def _check_skill(
    steps: tuple[CardStep, ...],
    score: float | None,
    fork_node_id: Any,
    violates: ViolatesRef | None,
    report_id: Any,
) -> None:
    if fork_node_id is not None or violates is not None or report_id is not None:
        raise ExperienceCardError("skill_card_fields", "fork-only fields on a skill_card")
    if score is None or score < SKILL_CARD_MIN_SCORE:
        raise ExperienceCardError(
            "score_gate", f"skill_card needs task_score >= {SKILL_CARD_MIN_SCORE}"
        )
    if not any(s.step_kind == STEP_VERIFY or s.recovery for s in steps):
        raise ExperienceCardError(
            "no_verify_or_recovery", "skill_card needs a verify step or a recovery fragment"
        )


def _parse_gamma(value: Any, members: frozenset[str]) -> CardGamma:
    if not isinstance(value, Mapping):
        raise ExperienceCardError("shape", "gamma must be a dict")
    deliverable = _choice(value.get("deliverable"), "gamma.deliverable", frozenset(DELIVERABLE_KINDS))
    issue = value.get("issue_kind")
    if issue is not None:
        issue = _choice(issue, "gamma.issue_kind", frozenset(FAILURE_KINDS))
    occupation = value.get("occupation")
    if occupation is not None:
        occupation = _text(occupation, "gamma.occupation")
    anchors = value.get("anchors")
    if not isinstance(anchors, list) or not anchors:
        raise ExperienceCardError("shape", "gamma.anchors must be a non-empty list")
    for anchor in anchors:
        if anchor not in members:
            raise ExperienceCardError("anchor_unknown", f"gamma anchor {anchor!r} is not a Block")
    return CardGamma(
        deliverable=deliverable,
        abstract_text=_text(value.get("abstract_text"), "gamma.abstract_text"),
        anchors=tuple(dict.fromkeys(anchors)),
        occupation=occupation,
        issue_kind=issue,
    )


def _parse_steps(
    value: Any,
    members: frozenset[str],
    block_aos: Mapping[str, Sequence[str]] | None,
    read_aos: Sequence[str] | None,
) -> tuple[CardStep, ...]:
    if not isinstance(value, list) or not value:
        raise ExperienceCardError("shape", "steps must be a non-empty list")
    read = frozenset(read_aos) if read_aos is not None else None
    steps: list[CardStep] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        label = f"steps[{index}]"
        if not isinstance(item, Mapping):
            raise ExperienceCardError("shape", f"{label} must be a dict")
        step_id = _text(item.get("step_id"), f"{label}.step_id")
        if step_id in seen:
            raise ExperienceCardError("shape", f"duplicate step_id {step_id!r}")
        seen.add(step_id)
        kind = _choice(item.get("step_kind"), f"{label}.step_kind", STEP_KINDS)
        block_id = _text(item.get("block_id"), f"{label}.block_id")
        if block_id not in members:
            raise ExperienceCardError("anchor_unknown", f"{label} block {block_id!r} is not a Block")
        aos = item.get("ao_refs", [])
        if not isinstance(aos, list) or any(not isinstance(a, str) or not a for a in aos):
            raise ExperienceCardError("shape", f"{label}.ao_refs must be a list of str")
        if aos:
            if block_aos is None:
                raise ExperienceCardError("ao_index_required", f"{label} cites AOs without an index")
            contained = frozenset(block_aos.get(block_id, ()))
            for ao in aos:
                if ao not in contained:
                    raise ExperienceCardError(
                        "ao_not_in_block", f"{label} AO {ao!r} is not in block {block_id!r}"
                    )
                if read is not None and ao not in read:
                    raise ExperienceCardError("ao_not_read", f"{label} AO {ao!r} was not read")
        issue = item.get("issue_kind")
        if issue is not None:
            issue = _choice(issue, f"{label}.issue_kind", frozenset(FAILURE_KINDS))
        check_how = item.get("check_how")
        sign = item.get("violation_sign")
        if kind == STEP_VERIFY:
            check_how = _text(check_how, f"{label}.check_how")
            sign = _text(sign, f"{label}.violation_sign")
            if issue is None:
                raise ExperienceCardError("shape", f"{label} verify step needs issue_kind")
        elif check_how is not None or sign is not None:
            raise ExperienceCardError("shape", f"{label} produce step has verify-only fields")
        steps.append(
            CardStep(
                step_id=step_id,
                step_kind=kind,
                statement=_text(item.get("statement"), f"{label}.statement"),
                block_id=block_id,
                ao_refs=tuple(aos),
                issue_kind=issue,
                check_how=check_how,
                violation_sign=sign,
                recovery=bool(item.get("recovery", False)),
            )
        )
    return tuple(steps)


def _parse_violates(value: Any) -> ViolatesRef | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ExperienceCardError("shape", "violates must be a dict")
    return ViolatesRef(
        card_id=_text(value.get("card_id"), "violates.card_id"),
        step_id=_text(value.get("step_id"), "violates.step_id"),
    )


def _score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ExperienceCardError("shape", "task_score must be a number in [0, 1]")
    return float(value)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperienceCardError("shape", f"{field} must be a non-empty str")
    return value


def _choice(value: Any, field: str, choices: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ExperienceCardError("shape", f"{field} must be one of {sorted(choices)}, got {value!r}")
    return value


# --------------------------------------------------------------------------
# Canonical body and identity
# --------------------------------------------------------------------------

def _card_body(
    kind: str,
    catalog: BaseRevisionCatalog,
    report_hash: str,
    gamma: CardGamma,
    boundary: str,
    steps: tuple[CardStep, ...],
    score: float | None,
    fork_node_id: str | None,
    violates: ViolatesRef | None,
    report_id: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "author_role": _READER_ROLE,
        "base_revision_block_content_hash": catalog.block_content_hash,
        "base_revision_id": catalog.base_revision_id,
        "boundary": boundary,
        "gamma": {
            "abstract_text": gamma.abstract_text,
            "anchors": list(gamma.anchors),
            "deliverable": gamma.deliverable,
        },
        "kind": kind,
        "report_hash": report_hash,
        "steps": [_step_dict(step) for step in steps],
    }
    if gamma.occupation is not None:
        body["gamma"]["occupation"] = gamma.occupation
    if gamma.issue_kind is not None:
        body["gamma"]["issue_kind"] = gamma.issue_kind
    if score is not None:
        body["task_score"] = score
    if fork_node_id is not None:
        body["fork_node_id"] = fork_node_id
    if violates is not None:
        body["violates"] = {"card_id": violates.card_id, "step_id": violates.step_id}
    if report_id is not None:
        body["diagnosis_report_id"] = report_id
    return body


def _step_dict(step: CardStep) -> dict[str, Any]:
    out: dict[str, Any] = {
        "block_id": step.block_id,
        "statement": step.statement,
        "step_id": step.step_id,
        "step_kind": step.step_kind,
    }
    if step.ao_refs:
        out["ao_refs"] = list(step.ao_refs)
    if step.issue_kind is not None:
        out["issue_kind"] = step.issue_kind
    if step.check_how is not None:
        out["check_how"] = step.check_how
    if step.violation_sign is not None:
        out["violation_sign"] = step.violation_sign
    if step.recovery:
        out["recovery"] = True
    return out


def card_id_for(body: Mapping[str, Any]) -> str:
    """Content-addressed id. The same proposal always maps to the same card."""
    payload = {
        "body": body,
        "object_kind": OBJECT_KIND_EXPERIENCE_CARD,
        "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
    }
    return hashlib.sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()


def _card_from_body(body: Mapping[str, Any]) -> ExperienceCard:
    gamma = body["gamma"]
    violates = body.get("violates")
    return ExperienceCard(
        card_id=card_id_for(body),
        kind=body["kind"],
        base_revision_id=body["base_revision_id"],
        base_revision_block_content_hash=body["base_revision_block_content_hash"],
        report_hash=body["report_hash"],
        author_role=body["author_role"],
        gamma=CardGamma(
            deliverable=gamma["deliverable"],
            abstract_text=gamma["abstract_text"],
            anchors=tuple(gamma["anchors"]),
            occupation=gamma.get("occupation"),
            issue_kind=gamma.get("issue_kind"),
        ),
        boundary=body["boundary"],
        steps=tuple(
            CardStep(
                step_id=s["step_id"],
                step_kind=s["step_kind"],
                statement=s["statement"],
                block_id=s["block_id"],
                ao_refs=tuple(s.get("ao_refs", ())),
                issue_kind=s.get("issue_kind"),
                check_how=s.get("check_how"),
                violation_sign=s.get("violation_sign"),
                recovery=bool(s.get("recovery", False)),
            )
            for s in body["steps"]
        ),
        task_score=body.get("task_score"),
        fork_node_id=body.get("fork_node_id"),
        violates=ViolatesRef(violates["card_id"], violates["step_id"]) if violates else None,
        diagnosis_report_id=body.get("diagnosis_report_id"),
    )


def card_to_dict(card: ExperienceCard) -> dict[str, Any]:
    """Public JSON view, including the id and the derived verify summary."""
    catalog = BaseRevisionCatalog(
        base_revision_id=card.base_revision_id,
        block_content_hash=card.base_revision_block_content_hash,
        block_ids=(),
    )
    body = _card_body(
        card.kind, catalog, card.report_hash, card.gamma, card.boundary, card.steps,
        card.task_score, card.fork_node_id, card.violates, card.diagnosis_report_id,
    )
    return {"card_id": card.card_id, **body}


# --------------------------------------------------------------------------
# Store and retrieval
# --------------------------------------------------------------------------

class ExperienceCardStore:
    """Append-only card store under ``<root>/experience-cards/``.

    Visibility is the ACL of the registered base revision in the overlay
    store; this store keeps no labels of its own.
    """

    def __init__(self, root: str | Path, overlay_store: ForkNodeOverlayStore) -> None:
        if not isinstance(overlay_store, ForkNodeOverlayStore):
            raise ExperienceCardError("store_required")
        self._dir = Path(root) / CARDS_DIRNAME
        self._overlays = overlay_store
        self._lock = threading.Lock()

    def submit(
        self,
        payload: Mapping[str, Any],
        *,
        principal_labels: Sequence[str],
        block_aos: Mapping[str, Sequence[str]] | None = None,
        read_aos: Sequence[str] | None = None,
    ) -> tuple[ExperienceCard, bool]:
        """Validate and write one card. Returns ``(card, deduplicated)``."""
        stated = payload.get("base_revision_id") if isinstance(payload, Mapping) else None
        if not isinstance(stated, str) or not stated:
            raise ExperienceCardError("base_revision_missing")
        principal = require_label_set(principal_labels, code="acl_denied")
        try:
            catalog = self._overlays.registered_catalog(stated, principal_labels=sorted(principal))
        except ForkOverlayError as exc:
            raise ExperienceCardError(exc.code) from exc
        card = read_card_proposal(payload, catalog, block_aos=block_aos, read_aos=read_aos)
        try:
            document_hash = self._overlays.registered_document_hash(
                stated, principal_labels=sorted(principal)
            )
        except ForkOverlayError as exc:
            raise ExperienceCardError(exc.code) from exc
        if card.report_hash != document_hash:
            raise ExperienceCardError("report_hash_mismatch")
        with self._lock:
            self._check_links(card, principal)
            path = self._path(card.card_id)
            if path.is_file():
                return card, True
            write_exclusive(path, card_to_dict(card))
            append_jsonl(
                self._dir / "index.jsonl",
                {"card_id": card.card_id, "kind": card.kind,
                 "base_revision_id": card.base_revision_id},
            )
        return card, False

    def read(self, card_id: str, *, principal_labels: Sequence[str]) -> ExperienceCard | None:
        principal = require_label_set(principal_labels, code="acl_denied")
        card = self._load(card_id)
        if card is None or not self._visible(card, principal):
            return None
        return card

    def visible_cards(self, *, principal_labels: Sequence[str]) -> tuple[ExperienceCard, ...]:
        principal = require_label_set(principal_labels, code="acl_denied")
        directory = self._dir / "cards"
        if not directory.is_dir():
            return ()
        acl_cache: dict[str, bool] = {}
        found: list[ExperienceCard] = []
        for path in sorted(directory.glob("*.json")):
            card = self._load(path.stem)
            if card is None:
                continue
            ok = acl_cache.get(card.base_revision_id)
            if ok is None:
                ok = self._visible(card, principal)
                acl_cache[card.base_revision_id] = ok
            if ok:
                found.append(card)
        return tuple(found)

    def forks_violating(
        self, card_id: str, step_id: str, *, principal_labels: Sequence[str]
    ) -> tuple[ExperienceCard, ...]:
        """Reverse relation: forks that name this verify step as violated."""
        return tuple(
            c for c in self.visible_cards(principal_labels=principal_labels)
            if c.kind == CARD_KIND_FORK and c.violates is not None
            and c.violates.card_id == card_id and c.violates.step_id == step_id
        )

    def repair_group(
        self, diagnosis_report_id: str, *, principal_labels: Sequence[str]
    ) -> tuple[ExperienceCard, ...]:
        """Forks sharing one diagnosis report (a multi-agent repair group)."""
        return tuple(
            c for c in self.visible_cards(principal_labels=principal_labels)
            if c.kind == CARD_KIND_FORK and c.diagnosis_report_id == diagnosis_report_id
        )

    def find(
        self,
        *,
        principal_labels: Sequence[str],
        query: str = "",
        kind: str | None = None,
        deliverable: str | None = None,
        occupation: str | None = None,
        issue_kind: str | None = None,
        limit: int = 8,
    ) -> CardFindResult:
        """Hard-filter deliverable (relaxed if empty), soft-boost the rest, BM25 on gamma."""
        if kind is not None and kind not in CARD_KINDS:
            raise ExperienceCardError("shape", f"kind must be one of {sorted(CARD_KINDS)}")
        if deliverable is not None and deliverable not in DELIVERABLE_KINDS:
            raise ExperienceCardError("shape", f"deliverable must be one of {list(DELIVERABLE_KINDS)}")
        if issue_kind is not None and issue_kind not in FAILURE_KINDS:
            raise ExperienceCardError("shape", f"issue_kind must be one of {list(FAILURE_KINDS)}")
        pool = [
            c for c in self.visible_cards(principal_labels=principal_labels)
            if kind is None or c.kind == kind
        ]
        relaxed = False
        scoped = pool
        if deliverable is not None:
            scoped = [c for c in pool if c.gamma.deliverable == deliverable]
            if not scoped and pool:
                scoped, relaxed = pool, True
        return CardFindResult(
            hits=_rank(scoped, query, occupation, issue_kind, max(1, limit)), relaxed=relaxed
        )

    # internals ----------------------------------------------------------

    def _check_links(self, card: ExperienceCard, principal: frozenset[str]) -> None:
        if card.kind == CARD_KIND_FORK:
            try:
                stored = self._overlays.read_overlay_by_id(
                    card.fork_node_id or "", principal_labels=sorted(principal)
                )
            except ForkOverlayError as exc:
                raise ExperienceCardError(exc.code) from exc
            if stored is None:
                raise ExperienceCardError("fork_overlay_missing", card.fork_node_id or "")
            if stored.overlay.anchor_block_id not in card.gamma.anchors:
                raise ExperienceCardError(
                    "fork_anchor_mismatch", "gamma anchors must include the overlay anchor block"
                )
        if card.violates is not None:
            target = self._load(card.violates.card_id)
            if target is None or target.kind != CARD_KIND_SKILL:
                raise ExperienceCardError("violates_unknown", card.violates.card_id)
            if target.base_revision_id != card.base_revision_id and not self._visible(target, principal):
                raise ExperienceCardError("violates_unknown", card.violates.card_id)
            if not any(
                s.step_id == card.violates.step_id and s.step_kind == STEP_VERIFY
                for s in target.steps
            ):
                raise ExperienceCardError("violates_step_unknown", card.violates.step_id)

    def _visible(self, card: ExperienceCard, principal: frozenset[str]) -> bool:
        try:
            self._overlays.registered_catalog(card.base_revision_id, principal_labels=sorted(principal))
        except ForkOverlayError:
            return False
        return True

    def _path(self, card_id: str) -> Path:
        return self._dir / "cards" / f"{card_id}.json"

    def _load(self, card_id: str) -> ExperienceCard | None:
        import json

        if not isinstance(card_id, str) or len(card_id) != 64 or not all(
            ch in "0123456789abcdef" for ch in card_id
        ):
            return None
        path = self._path(card_id)
        if not path.is_file():
            return None
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ExperienceCardError("stored_record_invalid") from exc
        body = {k: v for k, v in stored.items() if k != "card_id"}
        card = _card_from_body(body)
        if card.card_id != card_id or stored.get("card_id") != card_id:
            raise ExperienceCardError("stored_record_invalid")
        return card


def _gamma_text(card: ExperienceCard) -> str:
    return card.gamma.abstract_text


def _card_issues(card: ExperienceCard) -> set[str]:
    issues = {s.issue_kind for s in card.steps if s.issue_kind}
    if card.gamma.issue_kind:
        issues.add(card.gamma.issue_kind)
    return issues


def _rank(
    cards: Sequence[ExperienceCard],
    query: str,
    occupation: str | None,
    issue_kind: str | None,
    limit: int,
) -> tuple[CardHit, ...]:
    if not cards:
        return ()
    raw = [0.0] * len(cards)
    if tokenize(query):
        raw = list(BM25Index([_gamma_text(c) for c in cards]).scores(query))
    peak = max(raw) if raw and max(raw) > 0 else 1.0
    occ = occupation.casefold() if occupation else None
    scored: list[CardHit] = []
    for card, bm in zip(cards, raw, strict=True):
        score = bm / peak
        if occ and card.gamma.occupation and card.gamma.occupation.casefold() == occ:
            score += _BOOST_OCCUPATION
        if issue_kind and issue_kind in _card_issues(card):
            score += _BOOST_ISSUE
        scored.append(CardHit(card=card, score=round(score, 6)))
    scored.sort(key=lambda h: (-h.score, h.card.card_id))
    return tuple(scored[:limit])


def _prompt() -> str:
    return (
        "经验卡提案（submit_experience_card，参数 {proposal}）。服务端核对后才写入；你只提案。\n"
        "何时提：确认通过的任务只提 skill_card；确认失败的任务只提 fork。结果未判定时两者都不提。\n"
        "skill_card：沿成功路径选一串事实块作为 steps，每步 step_kind 为 produce 或 verify。"
        "至少含一个 verify 步骤（写明 check_how 怎么查、violation_sign 违反表现、issue_kind 检查的问题）"
        "或一段出错→修复片段（produce 步骤标 recovery=true）。不要提不含检查或修复的纯流水账。\n"
        "fork：经 fork_node_id 指向已写入的 ForkNode；可带 violates={card_id, step_id}（必须指向已有 skill_card 的 verify 步骤）"
        "和 diagnosis_report_id（同一事故涉及的多个 fork 共用）。fork 的步骤不能是 verify。\n"
        f"gamma（触发条件）：deliverable 取 {'/'.join(DELIVERABLE_KINDS)} 之一；"
        f"issue_kind 取 {'/'.join(FAILURE_KINDS)} 之一；occupation 可选；"
        "abstract_text 用不含具体实体名的抽象句描述情境，anchors 列出支撑该情境的块。\n"
        "boundary（适用边界）必填：说明什么情况下不要用这张卡。\n"
        "接地：每一步 block_id 必须是冻结图中真实存在的块；ao_refs 可选，写则必须是该块内且你读过的 AO。"
        "不确定就省略 ao_refs，不要猜。task_score、author_role 由服务端写入，你填的会被覆盖。"
    )


EXPERIENCE_CARD_PROPOSER_PROMPT = _prompt()
