# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ForkNode overlay store and read-only legacy fork grounding (ADR-0043 P04).

A Reader ``ForkProposal`` is checked again on the server and then stored as a
``ForkNode`` overlay. The overlay hangs off one frozen base AIG revision at
one ``anchor_block_id``. It is not an AO-derived Block and it does not rewrite
that revision.

Identity
--------
Stable identity and revision identity are both SHA-256 digests of the same
content-addressed context: ``base_revision_id``, ``anchor_block_id``, and the
P00 ``block_content_hash`` (canonical Block records sorted by ``block_id``).
The two digests use different ``identity_kind`` tags, so they are not equal.
A second proposal for that same context conflicts instead of creating another
overlay. Replaying the same proposal returns the stored overlay.

The idempotency key adds the proposal content hash to that context. The same
key is not written twice.

Legacy
------
Edge and cut forks that already live in ``CausalExperiencesStore`` can be
displayed with their original ``InfluenceGrounding``. That display is not an
overlay. It is refused by the v2 writer and by the pattern / guided-branch
consumer gate. Implicit migration is refused. Explicit one-fork migration lives in
``legacy_fork_migration`` and is not called from this module. Nothing
here creates an ``InfluenceClaim`` or cut grounding.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openviking.session.aig_native_protocol import (
    OBJECT_KIND_FORK_NODE_OVERLAY,
    OBJECT_KIND_FORK_PROPOSAL,
    SCHEMA_VERSION_AIG_NATIVE_V2,
    BaseRevisionCatalog,
    BranchStepRef,
    ForkNodeOverlay,
    ForkProposal,
    ProtocolContractError,
    base_revision_catalog_from_view,
    block_content_hash,
    canonical_dumps,
    protocol_content_hash,
    read_fork_node_overlay,
    read_fork_proposal,
)
from openviking.session.append_only import append_jsonl, write_exclusive
from openviking.session.causal_experiences import (
    CausalExperiencesStore,
    ForkDraft,
    ForkNodeRevision,
    InfluenceGrounding,
)
from openviking.session.influence_view import (
    AigNativeViewRevision,
    InfluenceViewRevision,
)

# ``authorize(session_id) -> report_hash``, same shape as
# ``AigConstructionRun.authorize_outcome_write``. P10 passes that method.
ForkWriteAuthorizer = Callable[[str], str]

FORK_NODE_OVERLAY_DIRNAME = "fork-node-overlays"
_V2_CONSUMERS = frozenset({"pattern", "replay", "guided_branch", "v2_writer", "v2_retrieval"})
_HEX_ID = re.compile(r"^[0-9a-f]{64}$")
_STABLE_KIND = "fork_node_stable"
_REVISION_KIND = "fork_node_revision"
_UNREGISTERED_CATALOG = BaseRevisionCatalog(
    base_revision_id="unregistered-base-revision",
    block_content_hash="0" * 64,
    block_ids=("unregistered-block",),
)


class ForkOverlayError(ValueError):
    """A fork overlay write or legacy gate was refused."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class StoredForkOverlay:
    """One durable overlay. ``revision_id`` is not a field of the v2 document."""

    fork_node_id: str
    revision_id: str
    idempotency_key: str
    proposal_content_hash: str
    overlay: ForkNodeOverlay


@dataclass(frozen=True)
class ForkOverlayCommit:
    """Result of ``submit_proposal``. A replay sets ``deduplicated``."""

    stored: StoredForkOverlay
    deduplicated: bool


@dataclass(frozen=True)
class LegacyForkGrounding:
    """Read-only view of an old fork's grounding. Not a v2 overlay."""

    fork_node_id: str
    revision_id: str
    position_kind: str
    position_ref: str
    view_revision_id: str
    content_hash: str
    semantic_summary_snapshot: str
    grounded_at: str
    workspace_id: str


@dataclass(frozen=True)
class ForkDualRead:
    """One fork, either a v2 overlay or a legacy grounding display."""

    kind: str
    overlay: StoredForkOverlay | None = None
    legacy: LegacyForkGrounding | None = None


def fork_node_stable_id(
    base_revision_id: str, anchor_block_id: str, block_content_hash: str
) -> str:
    """Stable ForkNode id bound to the base revision and block content hash."""
    return _bound_id(_STABLE_KIND, base_revision_id, anchor_block_id, block_content_hash)


def fork_node_revision_id(
    base_revision_id: str, anchor_block_id: str, block_content_hash: str
) -> str:
    """Revision id bound to the same context as the stable id."""
    return _bound_id(_REVISION_KIND, base_revision_id, anchor_block_id, block_content_hash)


def fork_proposal_idempotency_key(proposal: ForkProposal) -> str:
    """Canonical key for one proposal. The same key must not create a second overlay."""
    if not isinstance(proposal, ForkProposal):
        raise ForkOverlayError("revision_required")
    return canonical_dumps(
        {
            "anchor_block_id": proposal.anchor_block_id,
            "base_revision_block_content_hash": proposal.base_revision_block_content_hash,
            "base_revision_id": proposal.base_revision_id,
            "op": "fork_node_overlay",
            "proposal_content_hash": proposal.content_hash,
            "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
        }
    )


def catalog_for_base_revision(
    revision: InfluenceViewRevision | AigNativeViewRevision,
) -> BaseRevisionCatalog:
    """Block membership and P00 block digest of a frozen factual revision.

    Claims, native edges, and coverage are not part of the digest. Passing
    anything other than a frozen revision is refused.
    """
    if isinstance(revision, InfluenceViewRevision):
        return base_revision_catalog_from_view(revision)
    if isinstance(revision, AigNativeViewRevision):
        if not revision.blocks:
            raise ForkOverlayError("revision_required")
        return BaseRevisionCatalog(
            base_revision_id=revision.revision_id,
            block_content_hash=block_content_hash(revision.blocks),
            block_ids=tuple(block.block_id for block in revision.blocks),
        )
    raise ForkOverlayError("revision_required")


def read_legacy_fork_grounding(
    store: CausalExperiencesStore,
    fork_node_id: str,
    *,
    principal_labels: Sequence[str],
) -> LegacyForkGrounding | None:
    """Show the latest stored grounding. Does not write an overlay."""
    if not isinstance(store, CausalExperiencesStore):
        raise ForkOverlayError("store_required")
    principal = require_label_set(principal_labels, code="acl_denied")
    if not isinstance(fork_node_id, str) or not fork_node_id:
        return None
    node = store.get_fork_node(fork_node_id)
    if node is None:
        return None
    latest = node.get("latest_revision")
    if not isinstance(latest, Mapping):
        raise ForkOverlayError("legacy_grounding_missing")
    workspace_id = latest.get("workspace_id")
    if not isinstance(workspace_id, str) or workspace_id not in principal:
        raise ForkOverlayError("acl_denied")
    raw = latest.get("influence_grounding")
    if not isinstance(raw, dict):
        raise ForkOverlayError("legacy_grounding_missing")
    try:
        grounding = InfluenceGrounding.from_dict(raw)
    except (TypeError, ValueError) as exc:
        raise ForkOverlayError("legacy_grounding_missing") from exc
    return LegacyForkGrounding(
        fork_node_id=str(latest.get("fork_node_id") or ""),
        revision_id=str(latest.get("revision_id") or ""),
        position_kind=grounding.position_kind,
        position_ref=grounding.position_ref,
        view_revision_id=grounding.view_revision_id,
        content_hash=grounding.content_hash,
        semantic_summary_snapshot=grounding.semantic_summary_snapshot,
        grounded_at=grounding.grounded_at,
        workspace_id=workspace_id,
    )


def guard_v2_fork_consumer(subject: Any, consumer: str) -> None:
    """Refuse legacy forks as inputs of pattern, guided branch, or the v2 writer.

    Expected consumers are P05 (StrategyPattern), P06 (precedent retrieval),
    P07 (replay alignment), and P11 (``v2_retrieval``).
    A stored overlay is
    eligible for those consumers. This function does not build a pattern, a
    guided branch, or another fork.
    """
    if consumer not in _V2_CONSUMERS:
        raise ForkOverlayError("unknown_v2_consumer")
    if isinstance(subject, StoredForkOverlay):
        subject = subject.overlay
    if (
        isinstance(subject, ForkNodeOverlay)
        and subject.object_kind == OBJECT_KIND_FORK_NODE_OVERLAY
    ):
        return None
    raise ForkOverlayError("legacy_fork_excluded")


def fork_overlay_for(subject: Any, consumer: str, error_cls: type[Exception]) -> ForkNodeOverlay:
    """Return a v2 overlay, mapping guard failures into ``error_cls``."""
    try:
        guard_v2_fork_consumer(subject, consumer)
    except ForkOverlayError as exc:
        raise error_cls(exc.code) from exc
    if isinstance(subject, StoredForkOverlay):
        return subject.overlay
    if isinstance(subject, ForkNodeOverlay):
        return subject
    raise error_cls("overlay_required")


def require_label_set(value: Any, *, code: str) -> frozenset[str]:
    """Non-empty set of exact label strings. Empty, padded, or scalar input fails."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ForkOverlayError(code)
    if len(value) == 0:
        raise ForkOverlayError(code)
    labels: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item or item.strip() != item:
            raise ForkOverlayError(code)
        labels.append(item)
    return frozenset(labels)


def refuse_legacy_migration(grounding: LegacyForkGrounding) -> None:
    """Legacy grounding is not turned into an overlay."""
    del grounding
    raise ForkOverlayError("legacy_migration_forbidden")


def dual_read_fork(
    fork_node_id: str,
    *,
    overlay_store: ForkNodeOverlayStore,
    legacy_store: CausalExperiencesStore,
    principal_labels: Sequence[str],
) -> ForkDualRead | None:
    """Read one overlay or, failing that, one legacy grounding.

    A legacy hit is never written back as an overlay. P05, P06, and P11 are
    the expected consumers: they can show legacy grounding without migrating it.
    """
    if not isinstance(overlay_store, ForkNodeOverlayStore):
        raise ForkOverlayError("store_required")
    if not isinstance(legacy_store, CausalExperiencesStore):
        raise ForkOverlayError("store_required")
    overlay = overlay_store.read_overlay_by_id(fork_node_id, principal_labels=principal_labels)
    if overlay is not None:
        return ForkDualRead(kind="overlay", overlay=overlay, legacy=None)
    legacy = read_legacy_fork_grounding(
        legacy_store, fork_node_id, principal_labels=principal_labels
    )
    if legacy is None:
        return None
    return ForkDualRead(kind="legacy", overlay=None, legacy=legacy)


class ForkNodeOverlayStore:
    """Append-only overlay store under ``<root>/fork-node-overlays/``."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._dir = self._root / FORK_NODE_OVERLAY_DIRNAME
        self._index_path = self._dir / "index.jsonl"
        self._lock = threading.Lock()

    def overlay_root(self) -> Path:
        """Root the store was opened on. This does not list or rewrite records."""
        return self._root

    def register_base_revision(
        self,
        revision: InfluenceViewRevision | AigNativeViewRevision,
        *,
        acl_labels: Sequence[str],
    ) -> BaseRevisionCatalog:
        """Record block membership for one frozen revision. Does not copy its blocks."""
        catalog = catalog_for_base_revision(revision)
        document_hash = revision.content_hash
        if not isinstance(document_hash, str) or not document_hash:
            raise ForkOverlayError("revision_required")
        labels = require_label_set(acl_labels, code="acl_labels_required")
        with self._lock:
            existing = self._load_base(catalog.base_revision_id)
            if existing is not None:
                stored_catalog, stored_labels, stored_hash = existing
                if stored_catalog != catalog or stored_hash != document_hash:
                    raise ForkOverlayError("base_revision_conflict")
                if stored_labels != labels:
                    raise ForkOverlayError("acl_conflict")
                return stored_catalog
            self._write_new(
                self._base_path(catalog.base_revision_id),
                _base_record(catalog, labels, document_hash),
            )
            self._append_index(
                {
                    "base_revision_id": catalog.base_revision_id,
                    "block_content_hash": catalog.block_content_hash,
                    "record_kind": "base_revision",
                }
            )
            return catalog

    def submit_proposal(
        self,
        payload: Mapping[str, Any] | LegacyForkGrounding | ForkNodeRevision | ForkDraft,
        *,
        principal_labels: Sequence[str],
        authorize: ForkWriteAuthorizer | None = None,
        session_id: str | None = None,
    ) -> ForkOverlayCommit:
        """Validate a Reader proposal against the registered revision and store it.

        Shape, hash, anchor membership, and branch checks are ``read_fork_proposal``.
        ACL, revision existence, and the idempotency key are checked here.
        ``proposal.report_hash`` must equal the registered revision's document
        content hash (native revision or claim view). Anything else is
        ``report_hash_mismatch``.

        ``authorize`` is optional until P10 wires ``authorize_outcome_write``.
        When it is omitted, ``author_role`` stays a declared protocol field and
        this method does not consult the role chain. When it is provided, it is
        called as ``authorize(session_id)`` and must return ``proposal.report_hash``
        before any write, including an idempotent replay. A raised exception
        refuses the write.

        This method is the persistence primitive. It does not retrieve
        precedents. Reader submissions go through
        ``ForkPrecedentService.submit_proposal``, which refuses the write
        until a server retrieval record exists.
        """
        if isinstance(payload, (LegacyForkGrounding, ForkNodeRevision, ForkDraft)):
            raise ForkOverlayError("legacy_fork_excluded")
        principal = require_label_set(principal_labels, code="acl_denied")
        with self._lock:
            stated = _stated_base_revision_id(payload)
            if stated is None:
                read_fork_proposal(payload, _UNREGISTERED_CATALOG)
                raise ForkOverlayError("base_revision_missing")
            loaded = self._load_base(stated)
            if loaded is None:
                raise ForkOverlayError("base_revision_missing")
            catalog, acl, document_hash = loaded
            if not principal.intersection(acl):
                raise ForkOverlayError("acl_denied")
            proposal = read_fork_proposal(payload, catalog)
            self._require_bound_report(
                proposal,
                document_hash,
                authorize=authorize,
                session_id=session_id,
            )
            return self._persist(proposal, catalog)

    def _require_bound_report(
        self,
        proposal: ForkProposal,
        document_hash: str,
        *,
        authorize: ForkWriteAuthorizer | None,
        session_id: str | None,
    ) -> None:
        if proposal.report_hash != document_hash:
            raise ForkOverlayError("report_hash_mismatch")
        if authorize is None:
            return
        if not isinstance(session_id, str) or not session_id:
            raise ForkOverlayError("session_required")
        granted = authorize(session_id)
        if granted != proposal.report_hash:
            raise ForkOverlayError("report_hash_mismatch")

    def read_overlay(
        self,
        base_revision_id: str,
        anchor_block_id: str,
        *,
        principal_labels: Sequence[str],
    ) -> StoredForkOverlay | None:
        """Return the overlay at this anchor, or None when the anchor has none."""
        principal = require_label_set(principal_labels, code="acl_denied")
        with self._lock:
            loaded = self._load_base(base_revision_id)
            if loaded is None:
                raise ForkOverlayError("base_revision_missing")
            catalog, acl, _document_hash = loaded
            if not principal.intersection(acl):
                raise ForkOverlayError("acl_denied")
            return self._find_overlay(catalog, anchor_block_id)

    def read_overlay_by_id(
        self,
        fork_node_id: str,
        *,
        principal_labels: Sequence[str],
    ) -> StoredForkOverlay | None:
        """Return one overlay by its stable id. Unknown ids are None."""
        principal = require_label_set(principal_labels, code="acl_denied")
        path = self._overlay_path(fork_node_id)
        if path is None or not path.is_file():
            return None
        with self._lock:
            payload = _read_json_file(path)
            document = payload.get("document")
            body = document.get("body") if isinstance(document, Mapping) else None
            revision_id = body.get("base_revision_id") if isinstance(body, Mapping) else None
            if not isinstance(revision_id, str):
                raise ForkOverlayError("stored_record_invalid")
            loaded = self._load_base(revision_id)
            if loaded is None:
                raise ForkOverlayError("stored_record_invalid")
            catalog, acl, _document_hash = loaded
            if not principal.intersection(acl):
                raise ForkOverlayError("acl_denied")
            return self._parse_overlay(payload, catalog, path)

    def registered_catalog(
        self,
        base_revision_id: str,
        *,
        principal_labels: Sequence[str],
    ) -> BaseRevisionCatalog:
        """Return the catalog when ``principal_labels`` may read this revision."""
        principal = require_label_set(principal_labels, code="acl_denied")
        with self._lock:
            loaded = self._load_base(base_revision_id)
            if loaded is None:
                raise ForkOverlayError("base_revision_missing")
            catalog, acl, _document_hash = loaded
            if not principal.intersection(acl):
                raise ForkOverlayError("acl_denied")
            return catalog

    def registered_document_hash(
        self,
        base_revision_id: str,
        *,
        principal_labels: Sequence[str],
    ) -> str:
        """Document content hash of a registered revision, for report binding."""
        principal = require_label_set(principal_labels, code="acl_denied")
        with self._lock:
            loaded = self._load_base(base_revision_id)
            if loaded is None:
                raise ForkOverlayError("base_revision_missing")
            _catalog, acl, document_hash = loaded
            if not principal.intersection(acl):
                raise ForkOverlayError("acl_denied")
            return document_hash

    def visible_overlays(self, *, principal_labels: Sequence[str]) -> tuple[StoredForkOverlay, ...]:
        """Overlays whose registered base-revision ACL intersects the principal.

        A revision the principal cannot see is omitted. An empty or malformed
        principal is ``acl_denied``. Order is the stable fork id.
        """
        principal = require_label_set(principal_labels, code="acl_denied")
        with self._lock:
            directory = self._dir / "overlays"
            if not directory.is_dir():
                return ()
            found: list[StoredForkOverlay] = []
            for path in sorted(directory.glob("*.json")):
                payload = _read_json_file(path)
                document = payload.get("document")
                body = document.get("body") if isinstance(document, Mapping) else None
                revision_id = body.get("base_revision_id") if isinstance(body, Mapping) else None
                if not isinstance(revision_id, str):
                    raise ForkOverlayError("stored_record_invalid")
                loaded = self._load_base(revision_id)
                if loaded is None:
                    raise ForkOverlayError("stored_record_invalid")
                catalog, acl, _document_hash = loaded
                if not principal.intersection(acl):
                    continue
                found.append(self._parse_overlay(payload, catalog, path))
            found.sort(key=lambda item: item.fork_node_id)
            return tuple(found)

    def _persist(self, proposal: ForkProposal, catalog: BaseRevisionCatalog) -> ForkOverlayCommit:
        key = fork_proposal_idempotency_key(proposal)
        recorded = self._recorded_idempotency(key)
        if recorded is not None:
            stored = self._commit_from_idempotency(recorded, proposal, catalog)
            return ForkOverlayCommit(stored=stored, deduplicated=True)
        existing = self._find_overlay(catalog, proposal.anchor_block_id)
        if existing is not None:
            if existing.proposal_content_hash == proposal.content_hash:
                return ForkOverlayCommit(stored=existing, deduplicated=True)
            raise ForkOverlayError("overlay_conflict")
        stable = fork_node_stable_id(
            proposal.base_revision_id,
            proposal.anchor_block_id,
            proposal.base_revision_block_content_hash,
        )
        revision_id = fork_node_revision_id(
            proposal.base_revision_id,
            proposal.anchor_block_id,
            proposal.base_revision_block_content_hash,
        )
        document = _overlay_document(proposal, stable)
        read_fork_node_overlay(document, catalog)
        wrapper = {
            "document": document,
            "idempotency_key": key,
            "proposal_content_hash": proposal.content_hash,
            "revision_id": revision_id,
        }
        path = self._overlay_path(stable)
        if path is None:
            raise ForkOverlayError("stored_record_invalid")
        self._write_new(path, wrapper)
        self._append_index(
            {
                "anchor_block_id": proposal.anchor_block_id,
                "base_revision_block_content_hash": proposal.base_revision_block_content_hash,
                "base_revision_id": proposal.base_revision_id,
                "fork_node_id": stable,
                "proposal_content_hash": proposal.content_hash,
                "record_kind": "fork_node_overlay",
                "revision_id": revision_id,
            }
        )
        self._write_new(
            self._idempotency_path(key),
            {
                "fork_node_id": stable,
                "idempotency_key": key,
                "proposal_content_hash": proposal.content_hash,
                "revision_id": revision_id,
            },
        )
        return ForkOverlayCommit(
            stored=self._parse_overlay(wrapper, catalog, path),
            deduplicated=False,
        )

    def _recorded_idempotency(self, key: str) -> dict[str, Any] | None:
        path = self._idempotency_path(key)
        if not path.is_file():
            return None
        payload = _read_json_file(path)
        if payload.get("idempotency_key") != key:
            raise ForkOverlayError("overlay_conflict")
        return payload

    def _commit_from_idempotency(
        self,
        recorded: Mapping[str, Any],
        proposal: ForkProposal,
        catalog: BaseRevisionCatalog,
    ) -> StoredForkOverlay:
        if recorded.get("proposal_content_hash") != proposal.content_hash:
            raise ForkOverlayError("overlay_conflict")
        fork_node_id = recorded.get("fork_node_id")
        path = self._overlay_path(fork_node_id) if isinstance(fork_node_id, str) else None
        if path is None or not path.is_file():
            raise ForkOverlayError("stored_record_invalid")
        stored = self._parse_overlay(_read_json_file(path), catalog, path)
        same_hash = stored.proposal_content_hash == proposal.content_hash
        same_key = stored.idempotency_key == recorded.get("idempotency_key")
        if not same_hash or not same_key:
            raise ForkOverlayError("overlay_conflict")
        return stored

    def _find_overlay(
        self, catalog: BaseRevisionCatalog, anchor_block_id: str
    ) -> StoredForkOverlay | None:
        directory = self._dir / "overlays"
        if not directory.is_dir():
            return None
        found: StoredForkOverlay | None = None
        for path in sorted(directory.glob("*.json")):
            payload = _read_json_file(path)
            document = payload.get("document")
            body = document.get("body") if isinstance(document, Mapping) else None
            if not isinstance(body, Mapping):
                raise ForkOverlayError("stored_record_invalid")
            if body.get("base_revision_id") != catalog.base_revision_id:
                continue
            if body.get("anchor_block_id") != anchor_block_id:
                continue
            if body.get("base_revision_block_content_hash") != catalog.block_content_hash:
                continue
            loaded = self._parse_overlay(payload, catalog, path)
            if found is not None:
                raise ForkOverlayError("overlay_conflict")
            found = loaded
        return found

    def _parse_overlay(
        self,
        payload: Mapping[str, Any],
        catalog: BaseRevisionCatalog,
        path: Path,
    ) -> StoredForkOverlay:
        document = payload.get("document")
        if not isinstance(document, Mapping):
            raise ForkOverlayError("stored_record_invalid")
        try:
            overlay = read_fork_node_overlay(document, catalog)
        except ProtocolContractError as exc:
            raise ForkOverlayError("stored_record_invalid") from exc
        expected_stable = fork_node_stable_id(
            overlay.base_revision_id,
            overlay.anchor_block_id,
            overlay.base_revision_block_content_hash,
        )
        expected_revision = fork_node_revision_id(
            overlay.base_revision_id,
            overlay.anchor_block_id,
            overlay.base_revision_block_content_hash,
        )
        proposal_hash = payload.get("proposal_content_hash")
        key = payload.get("idempotency_key")
        revision_id = payload.get("revision_id")
        if (
            overlay.fork_node_id != expected_stable
            or revision_id != expected_revision
            or path.stem != expected_stable
            or not isinstance(proposal_hash, str)
            or not proposal_hash
            or key != _idempotency_key_for(overlay, proposal_hash)
        ):
            raise ForkOverlayError("stored_record_invalid")
        return StoredForkOverlay(
            fork_node_id=expected_stable,
            revision_id=expected_revision,
            idempotency_key=key,
            proposal_content_hash=proposal_hash,
            overlay=overlay,
        )

    def _load_base(
        self, base_revision_id: str
    ) -> tuple[BaseRevisionCatalog, frozenset[str], str] | None:
        path = self._base_path(base_revision_id)
        if not path.is_file():
            return None
        payload = _read_json_file(path)
        expected = {
            "acl_labels",
            "base_revision_id",
            "block_content_hash",
            "block_ids",
            "document_content_hash",
        }
        if set(payload) != expected or payload.get("base_revision_id") != base_revision_id:
            raise ForkOverlayError("stored_record_invalid")
        block_ids = payload.get("block_ids")
        labels = payload.get("acl_labels")
        block_hash = payload.get("block_content_hash")
        document_hash = payload.get("document_content_hash")
        if (
            not isinstance(block_ids, list)
            or not block_ids
            or not isinstance(block_hash, str)
            or not block_hash
            or not isinstance(document_hash, str)
            or not document_hash
            or not isinstance(labels, list)
            or not labels
        ):
            raise ForkOverlayError("stored_record_invalid")
        catalog = BaseRevisionCatalog(
            base_revision_id=base_revision_id,
            block_content_hash=block_hash,
            block_ids=tuple(str(item) for item in block_ids),
        )
        return catalog, frozenset(str(item) for item in labels), document_hash

    def _base_path(self, base_revision_id: str) -> Path:
        digest = hashlib.sha256(base_revision_id.encode("utf-8")).hexdigest()
        return self._dir / "base-revisions" / f"{digest}.json"

    def _overlay_path(self, fork_node_id: str) -> Path | None:
        if not isinstance(fork_node_id, str) or _HEX_ID.fullmatch(fork_node_id) is None:
            return None
        return self._dir / "overlays" / f"{fork_node_id}.json"

    def _idempotency_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._dir / ".idempotency" / f"{digest}.json"

    def _write_new(self, path: Path, payload: Mapping[str, Any]) -> None:
        try:
            write_exclusive(path, payload)
        except FileExistsError as exc:
            raise ForkOverlayError("stored_record_invalid") from exc

    def _append_index(self, row: Mapping[str, Any]) -> None:
        append_jsonl(self._index_path, row)


def _bound_id(
    kind: str, base_revision_id: str, anchor_block_id: str, block_content_hash: str
) -> str:
    _require_context(base_revision_id, anchor_block_id, block_content_hash)
    payload = {
        "anchor_block_id": anchor_block_id,
        "base_revision_block_content_hash": block_content_hash,
        "base_revision_id": base_revision_id,
        "identity_kind": kind,
        "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
    }
    return hashlib.sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()


def _require_context(base_revision_id: str, anchor_block_id: str, block_content_hash: str) -> None:
    for value in (base_revision_id, anchor_block_id, block_content_hash):
        if not isinstance(value, str) or not value:
            raise ForkOverlayError("revision_required")


def _base_record(
    catalog: BaseRevisionCatalog, labels: frozenset[str], document_content_hash: str
) -> dict[str, Any]:
    return {
        "acl_labels": sorted(labels),
        "base_revision_id": catalog.base_revision_id,
        "block_content_hash": catalog.block_content_hash,
        "block_ids": list(catalog.block_ids),
        "document_content_hash": document_content_hash,
    }


def _overlay_document(proposal: ForkProposal, fork_node_id: str) -> dict[str, Any]:
    body = {
        "anchor_block_id": proposal.anchor_block_id,
        "base_revision_block_content_hash": proposal.base_revision_block_content_hash,
        "base_revision_id": proposal.base_revision_id,
        "branch_steps": [_step_body(step) for step in proposal.branch_steps],
        "fork_node_id": fork_node_id,
    }
    return {
        "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
        "object_kind": OBJECT_KIND_FORK_NODE_OVERLAY,
        "content_hash": protocol_content_hash(
            SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_FORK_NODE_OVERLAY, body
        ),
        "body": body,
    }


def _step_body(step: BranchStepRef) -> dict[str, str]:
    if step.step_kind == "factual_block_ref":
        if not isinstance(step.block_id, str):
            raise ForkOverlayError("stored_record_invalid")
        return {"block_id": step.block_id, "step_kind": step.step_kind}
    if not isinstance(step.imagined_node_id, str):
        raise ForkOverlayError("stored_record_invalid")
    return {"imagined_node_id": step.imagined_node_id, "step_kind": step.step_kind}


def _idempotency_key_for(overlay: ForkNodeOverlay, proposal_hash: str) -> str:
    return canonical_dumps(
        {
            "anchor_block_id": overlay.anchor_block_id,
            "base_revision_block_content_hash": overlay.base_revision_block_content_hash,
            "base_revision_id": overlay.base_revision_id,
            "op": "fork_node_overlay",
            "proposal_content_hash": proposal_hash,
            "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
        }
    )


def _stated_base_revision_id(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        raise ProtocolContractError("protocol document must be a dict")
    if payload.get("object_kind") != OBJECT_KIND_FORK_PROPOSAL:
        return None
    body = payload.get("body")
    if not isinstance(body, Mapping):
        return None
    revision_id = body.get("base_revision_id")
    if isinstance(revision_id, str) and revision_id:
        return revision_id
    return None


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForkOverlayError("stored_record_invalid") from exc
    if not isinstance(payload, dict):
        raise ForkOverlayError("stored_record_invalid")
    return payload
