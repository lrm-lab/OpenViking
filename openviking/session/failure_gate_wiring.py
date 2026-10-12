# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Commit-path adapter for ``FailureGateOrchestrator.deliver`` (ADR-0043 P10).

Trigger point
-------------
The gate runs from ``Session._run_post_extraction_causal_seam``, after Phase 2
has called ``maybe_index_session_trajectory`` when a provider is present.
That is the first durable moment the frozen native revision exists. Phase 1
only archives and enqueues; hanging the gate there would block the
path-lock and run before the native extraction is on disk.

``FrozenViewBuilder`` reads ``NativeEdgeViewStore`` (and, when present, the
``ViewSupersedeIndex`` pointer). It does not rebuild a view.

Skip paths
----------
Causal mode off, or a session with no native AIG revision for this task,
returns immediately. No orchestrator, no audit file.

Failure semantics
-----------------
``auto_remediate=False``. Confirmed failure records the outcome and stops
(P10 commit boundary: no automatic dispatch). Diagnose still owns the
scripted remediation path. Any exception from ``deliver`` is audited by the
orchestrator and swallowed here so commit / Phase 2 still finish.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openviking.session.aig_native_protocol import (
    OBJECT_KIND_OUTCOME_CONTRACT,
    SCHEMA_VERSION_AIG_NATIVE_V2,
    is_safe_id,
    protocol_envelope,
)
from openviking.session.coordinator_boundary import TRIGGER_COMMIT
from openviking.session.extraction_write import NATIVE_EDGE_INDEX_FILENAME, NativeEdgeViewStore
from openviking.session.failure_gate import (
    ConstructionRequest,
    FailureGateDelivery,
    FailureGateOrchestrator,
    FailureGateResult,
    FrozenViewBuilder,
)
from openviking.session.influence_view import (
    CONSTRUCTION_MODE_ADAPTIVE,
    CONSTRUCTION_MODE_ADAPTIVE_AGENTIC,
    CONSTRUCTION_MODE_NATIVE_SINGLE_PASS,
    PURPOSE_POST_RUN_INDEX,
    AigNativeViewRevision,
    ViewSupersedeIndex,
)
from openviking.session.outcome_contract import OutcomeContract, OutcomeContractStore
from openviking.session.session_acl import session_owner_label
from openviking.session.trajectory_index_hook import VIEWS_FILENAME, session_directory
from openviking.utils.time_utils import format_iso8601, parse_iso_datetime
from openviking_cli.utils import get_logger
from openviking_cli.utils.config.memory_config import is_causal_mode_enabled

_AIG_MODES = frozenset(
    {
        CONSTRUCTION_MODE_ADAPTIVE,
        CONSTRUCTION_MODE_ADAPTIVE_AGENTIC,
        CONSTRUCTION_MODE_NATIVE_SINGLE_PASS,
    }
)
_DEFAULT_CRITERION = "crit-commit"

logger = get_logger(__name__)


def commit_id_for_archive(archive_uri: str, task_id: str) -> str:
    """Stable commit id: archive directory name, else the Phase 2 task id."""
    name = archive_uri.rstrip("/").rsplit("/", 1)[-1] if archive_uri else ""
    if is_safe_id(name):
        return name
    if is_safe_id(task_id):
        return task_id
    return "commit-unknown"


def latest_native_aig_revision(
    session_dir: Path,
    session_id: str,
    task_id: str,
) -> AigNativeViewRevision | None:
    """Frozen native revision for this commit, or None when AIG never ran.

    Prefers ``post-run:{session}:{task}:native`` from the trajectory hook,
    then a ``ViewSupersedeIndex`` pointer that already names a native
    revision, then the last native write for this session and task.
    Fallback ledger views are ignored.
    """
    native_path = session_dir / NATIVE_EDGE_INDEX_FILENAME
    if not native_path.is_file():
        return None
    store = NativeEdgeViewStore(native_path)
    named = f"post-run:{session_id}:{task_id}:native"
    try:
        revision = store.read(named)
    except KeyError:
        revision = None
    if revision is None:
        views_path = session_dir / VIEWS_FILENAME
        if views_path.is_file():
            latest = ViewSupersedeIndex(views_path).latest(session_id, PURPOSE_POST_RUN_INDEX)
            if latest is not None:
                try:
                    revision = store.read(latest.revision_id)
                except KeyError:
                    revision = None
    if revision is None:
        matches = [
            item
            for item in store.revisions()
            if item.session_id == session_id and item.task_run_id == task_id
        ]
        revision = matches[-1] if matches else None
    if revision is None or revision.construction_mode not in _AIG_MODES:
        return None
    return revision


def session_frozen_view_builder(session_dir: Path) -> FrozenViewBuilder:
    """Return a builder that reads the already-frozen native revision."""

    directory = Path(session_dir)

    def build(request: ConstructionRequest) -> AigNativeViewRevision:
        revision = latest_native_aig_revision(directory, request.session_id, request.task_id)
        if revision is None:
            raise ValueError("view_required")
        return revision

    return build


def maybe_deliver_commit_failure_gate(
    session: Any,
    *,
    task_id: str,
    commit_id: str,
    config: Any = None,
    build_revision: FrozenViewBuilder | None = None,
) -> FailureGateResult | None:
    """Deliver one finished commit. Exceptions never propagate to the caller."""
    session_id = getattr(session, "session_id", None)
    if not isinstance(session_id, str) or not session_id.strip():
        return None
    session_id = session_id.strip()
    try:
        causal = is_causal_mode_enabled(config, session_id=session_id)
    except FileNotFoundError:
        return None
    if not causal:
        return None
    if not isinstance(task_id, str) or not is_safe_id(task_id):
        return None
    if not isinstance(commit_id, str) or not is_safe_id(commit_id):
        return None
    session_dir = session_directory(session)
    if session_dir is None:
        return None
    try:
        revision = latest_native_aig_revision(session_dir, session_id, task_id)
        if revision is None:
            return None
        builder = build_revision or session_frozen_view_builder(session_dir)
        delivery = build_commit_delivery(
            session,
            revision,
            task_id=task_id,
            commit_id=commit_id,
            session_dir=session_dir,
        )
        return FailureGateOrchestrator(session_dir, builder).deliver(delivery)
    except Exception:
        logger.warning("commit failure gate refused", exc_info=True)
        return None


def build_commit_delivery(
    session: Any,
    revision: AigNativeViewRevision,
    *,
    task_id: str,
    commit_id: str,
    session_dir: Path,
) -> FailureGateDelivery:
    """Build one commit delivery from the frozen view and session products."""
    stamp = _stamp(revision.frozen_at)
    contract = _contract_document(session, session_dir, task_id, stamp)
    evaluator_results = _evaluator_results(session)
    assessment = getattr(session, "outcome_assessment_body", None)
    if not isinstance(assessment, Mapping):
        assessment = _derived_assessment(contract, evaluator_results, revision)
    return FailureGateDelivery(
        trigger=TRIGGER_COMMIT,
        commit_id=commit_id,
        task_id=task_id,
        session_id=revision.session_id,
        source_digest=revision.content_hash,
        principal_labels=_principal_labels(session),
        contract=contract,
        assessment_body=dict(assessment),
        evaluator_results=evaluator_results,
        temporal_cutoff=stamp,
        recorded_at=stamp,
        auto_remediate=False,
    )


def _principal_labels(session: Any) -> tuple[str, ...]:
    try:
        return (session_owner_label(session),)
    except ValueError:
        return ()


def _evaluator_results(session: Any) -> tuple[Mapping[str, Any], ...]:
    raw = getattr(session, "outcome_evaluator_results", None)
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return ()
    return tuple(dict(item) for item in raw if isinstance(item, Mapping))


def _contract_document(
    session: Any,
    session_dir: Path,
    task_id: str,
    stamp: str,
) -> dict[str, Any]:
    raw = getattr(session, "outcome_contract", None)
    if isinstance(raw, Mapping):
        return _as_contract_envelope(raw, task_id)
    stored = OutcomeContractStore(session_dir).read_task_contract(task_id)
    if stored is not None:
        return _envelope_from_stored(stored)
    return protocol_envelope(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        OBJECT_KIND_OUTCOME_CONTRACT,
        {
            "acceptance_criteria": [
                {
                    "criterion_id": _DEFAULT_CRITERION,
                    "statement": "the committed task meets its frozen goal",
                }
            ],
            "contract_id": f"contract-{task_id}",
            "evaluator_bindings": [],
            "frozen_at": stamp,
            "goals": ["complete the committed task"],
            "policy_constraints": [],
            "source": "orchestrator",
            "task_id": task_id,
        },
    )


def _as_contract_envelope(raw: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    body = raw.get("body")
    if isinstance(body, Mapping) and raw.get("object_kind") == OBJECT_KIND_OUTCOME_CONTRACT:
        payload = dict(body)
        payload["task_id"] = task_id
        return protocol_envelope(
            SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_OUTCOME_CONTRACT, payload
        )
    body = dict(raw)
    body["task_id"] = task_id
    return protocol_envelope(SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_OUTCOME_CONTRACT, body)


def _envelope_from_stored(document: OutcomeContract) -> dict[str, Any]:
    return protocol_envelope(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        OBJECT_KIND_OUTCOME_CONTRACT,
        {
            "acceptance_criteria": [
                {"criterion_id": item.criterion_id, "statement": item.statement}
                for item in document.acceptance_criteria
            ],
            "contract_id": document.contract_id,
            "evaluator_bindings": [
                {"binding_id": item.binding_id, "kind": item.kind, "ref": item.ref}
                for item in document.evaluator_bindings
            ],
            "frozen_at": document.frozen_at,
            "goals": list(document.goals),
            "policy_constraints": [
                {
                    "constraint_id": item.constraint_id,
                    "kind": item.kind,
                    "statement": item.statement,
                }
                for item in document.policy_constraints
            ],
            "source": document.source,
            "task_id": document.task_id,
        },
    )


def _derived_assessment(
    contract: Mapping[str, Any],
    evaluator_results: Sequence[Mapping[str, Any]],
    revision: AigNativeViewRevision,
) -> dict[str, Any]:
    body = contract.get("body") if isinstance(contract.get("body"), Mapping) else contract
    criteria = body.get("acceptance_criteria") if isinstance(body, Mapping) else None
    ids = [
        item["criterion_id"]
        for item in criteria or ()
        if isinstance(item, Mapping) and isinstance(item.get("criterion_id"), str)
    ]
    signals = [item.get("signal") for item in evaluator_results]
    unanimous = bool(signals) and all(item == signals[0] for item in signals)
    if unanimous and signals[0] in {"pass", "fail"} and ids:
        return {
            "criteria_dispositions": {key: "explicit" for key in ids},
            "evidence_refs": [_evidence_from_revision(revision)],
            "verdict": "success" if signals[0] == "pass" else "failure",
        }
    return {
        "criteria_dispositions": {key: "provisional" for key in ids},
        "evidence_refs": [],
        "verdict": "uncertain",
    }


def _evidence_from_revision(revision: AigNativeViewRevision) -> dict[str, Any]:
    block = revision.blocks[0]
    ref = block.source_refs[0]
    return {
        "ao_id": ref.ao_id,
        "archive_commit_watermark": None,
        "captured_state": "committed",
        "evidence_content_hash": ref.content_hash,
        "evidence_role": "hindsight_attribution",
        "source_archive_id": None,
        "source_read_snapshot_watermark": revision.coverage.snapshot_watermark,
        "source_sequence": ref.sequence,
        "source_session_id": ref.session_id,
    }


def _stamp(value: str) -> str:
    return format_iso8601(parse_iso_datetime(value))
