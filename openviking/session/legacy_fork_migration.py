# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Explicit migration of one legacy fork into a new ForkNode overlay.

The caller names one fork id and a migration reason, and supplies the new
Reader proposal. P04 validates that proposal. The legacy fork is not
rewritten. Claim and cut records are not copied into the overlay. Nothing
in this module scans the fork store, and nothing else calls this function.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openviking.session.aig_native_protocol import (
    SCHEMA_VERSION_AIG_NATIVE_V2,
    protocol_content_hash,
    protocol_envelope,
)
from openviking.session.append_only import read_json_object, write_new_json
from openviking.session.causal_experiences import CausalExperiencesStore
from openviking.session.fork_node_overlay import (
    FORK_NODE_OVERLAY_DIRNAME,
    ForkNodeOverlayStore,
    LegacyForkGrounding,
    StoredForkOverlay,
    read_legacy_fork_grounding,
)

_MIGRATION_KIND = "legacy_fork_migration"
_KEY_KIND = "legacy_fork_migration_key"
_SNAPSHOT_FIELDS = (
    "fork_node_id",
    "revision_id",
    "position_kind",
    "position_ref",
    "view_revision_id",
    "content_hash",
    "semantic_summary_snapshot",
    "grounded_at",
    "workspace_id",
)


class LegacyMigrationError(ValueError):
    """One explicit migration was refused."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class LegacyForkMigration:
    """Audit record for one legacy fork. The overlay is a new object."""

    schema_version: str
    object_kind: str
    content_hash: str
    legacy_fork_id: str
    migration_reason: str
    grounding_snapshot: dict[str, str]
    proposal_content_hash: str
    overlay_fork_node_id: str
    overlay_content_hash: str


@dataclass(frozen=True)
class LegacyForkMigrationCommit:
    """Result of one explicit migration. A replay sets ``deduplicated``."""

    migration: LegacyForkMigration
    overlay: StoredForkOverlay
    deduplicated: bool


def migrate_legacy_fork_to_overlay(
    *,
    legacy_store: CausalExperiencesStore,
    overlay_store: ForkNodeOverlayStore,
    fork_node_id: str,
    migration_reason: str,
    proposal: Mapping[str, Any],
    principal_labels: Sequence[str],
) -> LegacyForkMigrationCommit:
    """Create one new overlay for one legacy fork and record why.

    ``proposal`` is the new Reader proposal. It is not derived from the
    legacy cut or claim. The same fork, reason, proposal, and grounding
    snapshot return the stored migration. A different reason or proposal
    conflicts and does not write another overlay.
    """
    if not isinstance(legacy_store, CausalExperiencesStore):
        raise LegacyMigrationError("store_required")
    if not isinstance(overlay_store, ForkNodeOverlayStore):
        raise LegacyMigrationError("store_required")
    if not isinstance(fork_node_id, str) or not fork_node_id:
        raise LegacyMigrationError("legacy_fork_missing")
    reason = _reason(migration_reason)
    if isinstance(proposal, (str, bytes)) or not isinstance(proposal, Mapping):
        raise LegacyMigrationError("proposal_required")
    proposal_hash = _proposal_hash(proposal)
    grounding = read_legacy_fork_grounding(
        legacy_store, fork_node_id, principal_labels=principal_labels
    )
    if grounding is None:
        raise LegacyMigrationError("legacy_fork_missing")
    snapshot = _snapshot(grounding)
    path = _record_path(overlay_store, fork_node_id)
    existing = _read_record(path)
    if existing is not None:
        if not _same_migration(existing, fork_node_id, reason, proposal_hash, snapshot):
            raise LegacyMigrationError("migration_conflict")
        commit = overlay_store.submit_proposal(proposal, principal_labels=principal_labels)
        if (
            commit.stored.fork_node_id != existing.overlay_fork_node_id
            or commit.stored.overlay.content_hash != existing.overlay_content_hash
            or commit.stored.proposal_content_hash != proposal_hash
        ):
            raise LegacyMigrationError("migration_conflict")
        return LegacyForkMigrationCommit(
            migration=existing, overlay=commit.stored, deduplicated=True
        )
    commit = overlay_store.submit_proposal(proposal, principal_labels=principal_labels)
    if commit.stored.proposal_content_hash != proposal_hash:
        raise LegacyMigrationError("migration_conflict")
    migration = _migration(
        fork_node_id=fork_node_id,
        reason=reason,
        snapshot=snapshot,
        proposal_hash=proposal_hash,
        overlay=commit.stored,
    )
    write_new_json(path, _document(migration), LegacyMigrationError)
    return LegacyForkMigrationCommit(
        migration=migration, overlay=commit.stored, deduplicated=False
    )


def _reason(value: Any) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise LegacyMigrationError("migration_reason_required")
    return value


def _proposal_hash(payload: Mapping[str, Any]) -> str:
    schema = payload.get("schema_version")
    kind = payload.get("object_kind")
    body = payload.get("body")
    if not isinstance(schema, str) or not isinstance(kind, str) or not isinstance(body, Mapping):
        raise LegacyMigrationError("proposal_required")
    try:
        expected = protocol_content_hash(schema, kind, body)
    except (TypeError, ValueError) as exc:
        raise LegacyMigrationError("proposal_required") from exc
    if payload.get("content_hash") != expected:
        raise LegacyMigrationError("proposal_required")
    return expected


def _snapshot(grounding: LegacyForkGrounding) -> dict[str, str]:
    raw = {
        "fork_node_id": grounding.fork_node_id,
        "revision_id": grounding.revision_id,
        "position_kind": grounding.position_kind,
        "position_ref": grounding.position_ref,
        "view_revision_id": grounding.view_revision_id,
        "content_hash": grounding.content_hash,
        "semantic_summary_snapshot": grounding.semantic_summary_snapshot,
        "grounded_at": grounding.grounded_at,
        "workspace_id": grounding.workspace_id,
    }
    return {field: raw[field] for field in _SNAPSHOT_FIELDS}


def _record_path(store: ForkNodeOverlayStore, fork_node_id: str) -> Path:
    digest = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        _KEY_KIND,
        {"legacy_fork_id": fork_node_id},
    )
    return store.overlay_root() / FORK_NODE_OVERLAY_DIRNAME / "migrations" / f"{digest}.json"


def _read_record(path: Path) -> LegacyForkMigration | None:
    if not path.is_file():
        return None
    payload = read_json_object(path, LegacyMigrationError)
    return _parse(payload)


def _parse(payload: Mapping[str, Any]) -> LegacyForkMigration:
    if (
        payload.get("schema_version") != SCHEMA_VERSION_AIG_NATIVE_V2
        or payload.get("object_kind") != _MIGRATION_KIND
    ):
        raise LegacyMigrationError("stored_record_invalid")
    body = payload.get("body")
    if not isinstance(body, Mapping):
        raise LegacyMigrationError("stored_record_invalid")
    try:
        expected = protocol_content_hash(SCHEMA_VERSION_AIG_NATIVE_V2, _MIGRATION_KIND, body)
    except (TypeError, ValueError) as exc:
        raise LegacyMigrationError("stored_record_invalid") from exc
    if payload.get("content_hash") != expected:
        raise LegacyMigrationError("stored_record_invalid")
    snapshot = body.get("grounding_snapshot")
    if not isinstance(snapshot, Mapping) or set(snapshot) != set(_SNAPSHOT_FIELDS):
        raise LegacyMigrationError("stored_record_invalid")
    text = {field: snapshot.get(field) for field in _SNAPSHOT_FIELDS}
    if any(not isinstance(value, str) for value in text.values()):
        raise LegacyMigrationError("stored_record_invalid")
    try:
        return LegacyForkMigration(
            schema_version=SCHEMA_VERSION_AIG_NATIVE_V2,
            object_kind=_MIGRATION_KIND,
            content_hash=expected,
            legacy_fork_id=_text(body, "legacy_fork_id"),
            migration_reason=_text(body, "migration_reason"),
            grounding_snapshot={field: str(text[field]) for field in _SNAPSHOT_FIELDS},
            proposal_content_hash=_text(body, "proposal_content_hash"),
            overlay_fork_node_id=_text(body, "overlay_fork_node_id"),
            overlay_content_hash=_text(body, "overlay_content_hash"),
        )
    except LegacyMigrationError:
        raise
    except (TypeError, ValueError) as exc:
        raise LegacyMigrationError("stored_record_invalid") from exc


def _text(body: Mapping[str, Any], field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str) or not value:
        raise LegacyMigrationError("stored_record_invalid")
    return value


def _same_migration(
    existing: LegacyForkMigration,
    fork_node_id: str,
    reason: str,
    proposal_hash: str,
    snapshot: Mapping[str, str],
) -> bool:
    return (
        existing.legacy_fork_id == fork_node_id
        and existing.migration_reason == reason
        and existing.proposal_content_hash == proposal_hash
        and existing.grounding_snapshot == dict(snapshot)
    )


def _migration(
    *,
    fork_node_id: str,
    reason: str,
    snapshot: dict[str, str],
    proposal_hash: str,
    overlay: StoredForkOverlay,
) -> LegacyForkMigration:
    body = _body(
        fork_node_id=fork_node_id,
        reason=reason,
        snapshot=snapshot,
        proposal_hash=proposal_hash,
        overlay=overlay,
    )
    document = protocol_envelope(SCHEMA_VERSION_AIG_NATIVE_V2, _MIGRATION_KIND, body)
    parsed = _parse(document)
    return parsed


def _document(migration: LegacyForkMigration) -> dict[str, Any]:
    return protocol_envelope(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        _MIGRATION_KIND,
        _body(
            fork_node_id=migration.legacy_fork_id,
            reason=migration.migration_reason,
            snapshot=migration.grounding_snapshot,
            proposal_hash=migration.proposal_content_hash,
            overlay_fork_node_id=migration.overlay_fork_node_id,
            overlay_content_hash=migration.overlay_content_hash,
        ),
    )


def _body(
    *,
    fork_node_id: str,
    reason: str,
    snapshot: Mapping[str, str],
    proposal_hash: str,
    overlay: StoredForkOverlay | None = None,
    overlay_fork_node_id: str | None = None,
    overlay_content_hash: str | None = None,
) -> dict[str, Any]:
    if overlay is not None:
        overlay_fork_node_id = overlay.fork_node_id
        overlay_content_hash = overlay.overlay.content_hash
    return {
        "grounding_snapshot": {field: snapshot[field] for field in _SNAPSHOT_FIELDS},
        "legacy_fork_id": fork_node_id,
        "migration_reason": reason,
        "overlay_content_hash": overlay_content_hash,
        "overlay_fork_node_id": overlay_fork_node_id,
        "proposal_content_hash": proposal_hash,
    }
