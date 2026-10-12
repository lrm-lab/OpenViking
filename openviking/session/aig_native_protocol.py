# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ADR-0043 versioned protocol: legacy Claim/Cut fixtures and AIG-native v2 DTOs.

This module is the cross-repository contract for schema version, content
hashes, and the minimum v2 transport shapes. It does not dispatch remediation,
write a fork store, or call a model.

Canonical content hash
----------------------
The hashed input is exactly::

    {"schema_version": <version>, "object_kind": <kind>, "body": <body>}

serialized with ``json.dumps(..., ensure_ascii=False, sort_keys=True,
separators=(",", ":"))`` and digested with SHA-256 hex. ``body`` is the
domain payload. The envelope ``content_hash`` is not part of ``body``.
``schema_version`` and ``object_kind`` are inside the hashed input, so a
legacy Claim/Cut document and a v2 document cannot share a digest.

Legacy ``legacy_claim_cut`` bodies keep an ``InfluenceClaim`` and a cut
``InfluenceGrounding``. v2 writes reject those Claim/Cut fields. A v2 fork
anchors one Block in the same base revision; an imagined node is branch-local
and is never a factual Block.

Base revision catalog
---------------------
``BaseRevisionCatalog.block_content_hash`` is the SHA-256 of canonical JSON
for that revision's Block records, sorted by ``block_id``. Claims, coverage,
and retrieval subgraphs are not inputs. It is not
``InfluenceViewRevision.content_hash``.

Guidance target role
--------------------
``GuidanceDocument.target_role`` names the agent that may carry out frozen
guidance. It is not an Influence View ``BlockRole`` (``task``,
``orchestrator``, ``agent_action``, ``conclusion``) and not an AIG
construction role from ``aig_role_chain`` (``builder``, ``critic``,
``refiner``, ``reader``; diagnosis shares the reader session). Memory
publishes the document and Reader proposes it; neither executes it. The
closed set is ``TARGET_AGENT_ROLES`` (``task_agent``, the acting agent
already distinguished from those construction roles).

Native-edge draft and revision documents use ``aig-native-v2``. Their bodies
are blocks plus edges. Claim, status, evidence, and cut grounding fields are
rejected. These documents are not members of the eight-document fork fixture
set.

An outcome contract uses ``aig-native-v2`` and is also outside that set. It
freezes the task goal, acceptance criteria, evaluator/test/acceptance
bindings, and policy/SLA constraints. This reader does not decide success
or start remediation.

A strategy-pattern proposal and the server-generated counterfactual pattern
also use ``aig-native-v2`` and sit outside that set. ``pattern_id`` is the
digest of the versioned controlled fields only: intervention type, target
boundary, mechanism, expected metric, capability, and applicability
conditions. The proposal's semantic summary is not an input. This module
does not merge neighbors, rank precedents, or append the evidence ledger.
``strategy_pattern_identity`` tags that digest. It is not a document kind.

Replay transport
----------------
``replay_request`` and ``replay_result`` are the versioned boundary with the
Coforge sandbox replay worker. The worker is the implementer. This module
does not call it, open a socket, or start a model. P09 and P12 attach a real
worker. A completed ``replay_result`` carries ``step_alignments``. Each
record uses the same fields as ``GuidanceStepAlignment`` (imagined node,
alignment, actual block). A self-reported prefix or full-branch depth does
not admit an outcome; ``judge_direct_admission`` checks those records
against the branch and the Replay AIG blocks. A replay eligibility
assessment, a Replay AIG envelope, and a direct evidence admission sit
outside the eight-document fork fixture as well. Identity tags
(``replay_request_identity`` and the other ``*_identity`` kinds) are hash
inputs, not readable documents.

Guidance approval
-----------------
``guided_counterfactual_branch`` is the Reader proposal.
``agent_candidate_catalog`` is one version of the candidate directory.
``guidance_approval`` freezes the GuidanceDocument projection, the selected
agent, the catalog hash, and the outcome basis. ``guidance_publication`` is
the Memory render of that approval. It records the publisher principal and the
caller-supplied ``published_at``; storage does not read a clock. The publication
record key is the approval content hash together with that principal.
``guidance_rejection`` records a rejected
proposal. ``guidance_approval_identity`` is a hash tag, not a document. These
readers do not dispatch a remediation run or consume a GuidanceExecutionReport.

Remediation dispatch
--------------------
``remediation_dispatch_request`` and ``remediation_dispatch_confirmation``
are the versioned boundary with the Coforge worker. The worker is the real
executor. This module does not call it, open a socket, or start a model.
P12 connects the worker. Until then a scripted entry may seal the same
documents locally. ``blocked_pre_execution`` is the audit record when any
dispatch gate fails. ``production_direct_admission`` is the server gate that
lets a verified child-run block enter the evidence ledger as
production-direct. These documents sit outside the eight-document fork
fixture.

Tool-loop draft operations
--------------------------
``toolloop_draft_op`` is one append-only builder edit (ADR-0044). The body
is the verb, its arguments, the run id, the sequence, and the previous
operation hash. Replaying the log rebuilds the mutable paper draft. This
reader does not call a model or judge whether the prose is faithful.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from openviking.session.causal_experiences import (
    ApplicabilityCheck,
    EvidenceRef,
    InfluenceGrounding,
)
from openviking.session.influence_view import (
    NATIVE_EDGE_INDEX_KIND,
    AigNativeViewDraft,
    AigNativeViewRevision,
    InfluenceClaim,
    InfluenceViewRevision,
    validate_native_view,
)

SCHEMA_VERSION_LEGACY_CLAIM_CUT = "legacy-claim-cut-v1"
SCHEMA_VERSION_AIG_NATIVE_V2 = "aig-native-v2"

OBJECT_KIND_LEGACY_CLAIM_CUT = "legacy_claim_cut"
OBJECT_KIND_V2_FIXTURE_SET = "v2_fixture_set"
OBJECT_KIND_FORK_PROPOSAL = "fork_proposal"
OBJECT_KIND_FORK_NODE_OVERLAY = "fork_node_overlay"
OBJECT_KIND_TASK_OUTCOME_ASSESSMENT = "task_outcome_assessment"
OBJECT_KIND_REPLAY_ELIGIBILITY = "replay_eligibility"
OBJECT_KIND_GUIDANCE_DOCUMENT = "guidance_document"
OBJECT_KIND_GUIDANCE_EXECUTION_REPORT = "guidance_execution_report"
OBJECT_KIND_BRANCH_EVALUATION = "branch_evaluation"
OBJECT_KIND_FIRST_ACTION_ALIGNMENT = "first_action_alignment"
OBJECT_KIND_NATIVE_EDGE_DRAFT = "native_edge_draft"
OBJECT_KIND_NATIVE_EDGE_REVISION = "native_edge_revision"
OBJECT_KIND_NATIVE_EDGE_INDEX = NATIVE_EDGE_INDEX_KIND
OBJECT_KIND_OUTCOME_CONTRACT = "outcome_contract"
OBJECT_KIND_STRATEGY_PATTERN_PROPOSAL = "strategy_pattern_proposal"
OBJECT_KIND_STRATEGY_PATTERN = "strategy_pattern"
# Hash tag for the controlled-field digest. Not a readable document kind.
OBJECT_KIND_STRATEGY_PATTERN_IDENTITY = "strategy_pattern_identity"
PATTERN_IDENTITY_VERSION = "strategy-pattern-identity-v1"
OBJECT_KIND_REPLAY_ELIGIBILITY_ASSESSMENT = "replay_eligibility_assessment"
OBJECT_KIND_REPLAY_REQUEST = "replay_request"
OBJECT_KIND_REPLAY_RESULT = "replay_result"
OBJECT_KIND_REPLAY_AIG = "replay_aig"
OBJECT_KIND_DIRECT_EVIDENCE_ADMISSION = "direct_evidence_admission"
# Hash tags. Not readable document kinds.
OBJECT_KIND_REPLAY_ELIGIBILITY_IDENTITY = "replay_eligibility_identity"
OBJECT_KIND_REPLAY_REQUEST_IDENTITY = "replay_request_identity"
OBJECT_KIND_REPLAY_AIG_IDENTITY = "replay_aig_identity"
OBJECT_KIND_DIRECT_EVIDENCE_ADMISSION_IDENTITY = "direct_evidence_admission_identity"
OBJECT_KIND_GUIDED_COUNTERFACTUAL_BRANCH = "guided_counterfactual_branch"
OBJECT_KIND_AGENT_CANDIDATE_CATALOG = "agent_candidate_catalog"
OBJECT_KIND_GUIDANCE_APPROVAL = "guidance_approval"
OBJECT_KIND_GUIDANCE_PUBLICATION = "guidance_publication"
OBJECT_KIND_GUIDANCE_REJECTION = "guidance_rejection"
# Hash tag for proposal + catalog + selected agent. Not a readable document.
OBJECT_KIND_GUIDANCE_APPROVAL_IDENTITY = "guidance_approval_identity"
GUIDANCE_REJECTION_REASON = "guidance_rejected"
OBJECT_KIND_DISPATCH_GATE_DECISION = "dispatch_gate_decision"
OBJECT_KIND_BLOCKED_PRE_EXECUTION = "blocked_pre_execution"
OBJECT_KIND_REMEDIATION_DISPATCH_REQUEST = "remediation_dispatch_request"
OBJECT_KIND_REMEDIATION_DISPATCH_CONFIRMATION = "remediation_dispatch_confirmation"
OBJECT_KIND_CHILD_REMEDIATION_RUN = "child_remediation_run"
OBJECT_KIND_CHILD_RUN_OUTPUT = "child_run_output"
OBJECT_KIND_GUIDANCE_ADJUSTMENT = "guidance_adjustment"
OBJECT_KIND_REMEDIATION_CONSTRAINT = "remediation_constraint"
OBJECT_KIND_PRODUCTION_DIRECT_ADMISSION = "production_direct_admission"
# Hash tags. Not readable document kinds.
OBJECT_KIND_DISPATCH_GATE_IDENTITY = "dispatch_gate_identity"
OBJECT_KIND_REMEDIATION_DISPATCH_REQUEST_IDENTITY = "remediation_dispatch_request_identity"
OBJECT_KIND_REMEDIATION_DISPATCH_CONFIRMATION_IDENTITY = (
    "remediation_dispatch_confirmation_identity"
)
OBJECT_KIND_CHILD_REMEDIATION_RUN_IDENTITY = "child_remediation_run_identity"
OBJECT_KIND_GUIDANCE_ADJUSTMENT_IDENTITY = "guidance_adjustment_identity"
OBJECT_KIND_REMEDIATION_CONSTRAINT_IDENTITY = "remediation_constraint_identity"
OBJECT_KIND_PRODUCTION_DIRECT_ADMISSION_IDENTITY = "production_direct_admission_identity"
# One append-only edit of a paper-schema tool-loop draft. Not a fork fixture.
OBJECT_KIND_TOOLLOOP_DRAFT_OP = "toolloop_draft_op"
TOOLLOOP_BUILD_VERBS = frozenset(
    {
        "add_block",
        "add_inheritance_note",
        "add_skip_attribution",
        "modify_block",
        "delete_block",
        "delete_inheritance_note",
        "delete_skip_attribution",
        "mark_defect_recorded",
        "submit",
    }
)

REPLAY_PROVENANCE = "replay"
REPLAY_REVISION_PREFIX = "replay-rev:"
REPLAY_SESSION_PREFIX = "replay-session:"
REPLAY_TASK_RUN_PREFIX = "replay-run:"
REPLAY_VIEW_PREFIX = "replay-view:"
ROUTE_REPLAY = "replay"
ROUTE_ANALOGICAL = "analogical"

_V2_DOCUMENT_KINDS = (
    OBJECT_KIND_FORK_PROPOSAL,
    OBJECT_KIND_FORK_NODE_OVERLAY,
    OBJECT_KIND_TASK_OUTCOME_ASSESSMENT,
    OBJECT_KIND_REPLAY_ELIGIBILITY,
    OBJECT_KIND_GUIDANCE_DOCUMENT,
    OBJECT_KIND_GUIDANCE_EXECUTION_REPORT,
    OBJECT_KIND_BRANCH_EVALUATION,
    OBJECT_KIND_FIRST_ACTION_ALIGNMENT,
)
_FORK_OBJECT_KINDS = frozenset(
    {OBJECT_KIND_FORK_PROPOSAL, OBJECT_KIND_FORK_NODE_OVERLAY}
)
_TOP_LEVEL_DOCUMENT_KINDS = frozenset(
    {
        OBJECT_KIND_LEGACY_CLAIM_CUT,
        OBJECT_KIND_V2_FIXTURE_SET,
        OBJECT_KIND_NATIVE_EDGE_DRAFT,
        OBJECT_KIND_NATIVE_EDGE_REVISION,
        OBJECT_KIND_NATIVE_EDGE_INDEX,
        OBJECT_KIND_OUTCOME_CONTRACT,
        OBJECT_KIND_STRATEGY_PATTERN_PROPOSAL,
        OBJECT_KIND_STRATEGY_PATTERN,
        OBJECT_KIND_REPLAY_ELIGIBILITY_ASSESSMENT,
        OBJECT_KIND_REPLAY_REQUEST,
        OBJECT_KIND_REPLAY_RESULT,
        OBJECT_KIND_REPLAY_AIG,
        OBJECT_KIND_DIRECT_EVIDENCE_ADMISSION,
        OBJECT_KIND_GUIDED_COUNTERFACTUAL_BRANCH,
        OBJECT_KIND_AGENT_CANDIDATE_CATALOG,
        OBJECT_KIND_GUIDANCE_APPROVAL,
        OBJECT_KIND_GUIDANCE_PUBLICATION,
        OBJECT_KIND_GUIDANCE_REJECTION,
        OBJECT_KIND_DISPATCH_GATE_DECISION,
        OBJECT_KIND_BLOCKED_PRE_EXECUTION,
        OBJECT_KIND_REMEDIATION_DISPATCH_REQUEST,
        OBJECT_KIND_REMEDIATION_DISPATCH_CONFIRMATION,
        OBJECT_KIND_CHILD_REMEDIATION_RUN,
        OBJECT_KIND_CHILD_RUN_OUTPUT,
        OBJECT_KIND_GUIDANCE_ADJUSTMENT,
        OBJECT_KIND_REMEDIATION_CONSTRAINT,
        OBJECT_KIND_PRODUCTION_DIRECT_ADMISSION,
        OBJECT_KIND_TOOLLOOP_DRAFT_OP,
    }
)
_KIND_SCHEMA_VERSION = {
    OBJECT_KIND_LEGACY_CLAIM_CUT: SCHEMA_VERSION_LEGACY_CLAIM_CUT,
    OBJECT_KIND_V2_FIXTURE_SET: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_NATIVE_EDGE_DRAFT: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_NATIVE_EDGE_REVISION: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_NATIVE_EDGE_INDEX: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_OUTCOME_CONTRACT: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_STRATEGY_PATTERN_PROPOSAL: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_STRATEGY_PATTERN: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_REPLAY_ELIGIBILITY_ASSESSMENT: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_REPLAY_REQUEST: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_REPLAY_RESULT: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_REPLAY_AIG: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_DIRECT_EVIDENCE_ADMISSION: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_GUIDED_COUNTERFACTUAL_BRANCH: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_AGENT_CANDIDATE_CATALOG: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_GUIDANCE_APPROVAL: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_GUIDANCE_PUBLICATION: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_GUIDANCE_REJECTION: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_DISPATCH_GATE_DECISION: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_BLOCKED_PRE_EXECUTION: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_REMEDIATION_DISPATCH_REQUEST: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_REMEDIATION_DISPATCH_CONFIRMATION: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_CHILD_REMEDIATION_RUN: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_CHILD_RUN_OUTPUT: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_GUIDANCE_ADJUSTMENT: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_REMEDIATION_CONSTRAINT: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_PRODUCTION_DIRECT_ADMISSION: SCHEMA_VERSION_AIG_NATIVE_V2,
    OBJECT_KIND_TOOLLOOP_DRAFT_OP: SCHEMA_VERSION_AIG_NATIVE_V2,
    **{kind: SCHEMA_VERSION_AIG_NATIVE_V2 for kind in _V2_DOCUMENT_KINDS},
}
_ENVELOPE_KEYS = frozenset({"schema_version", "object_kind", "content_hash", "body"})

V2_CLAIM_CUT_FIELDS = frozenset(
    {
        "claim_id",
        "claim_ids",
        "claims",
        "cut",
        "cut_id",
        "cuts",
        "influence_claim",
        "influence_claims",
        "influence_grounding",
        "position_kind",
        "position_ref",
    }
)
_CLAIM_KEYS = frozenset(
    {
        "claim_id",
        "source_block_id",
        "target_block_id",
        "carried_artifact",
        "downstream_effect",
        "relation_type",
        "source_evidence_refs",
        "target_evidence_refs",
        "status",
        "critic_verdict",
    }
)
_GROUNDING_KEYS = frozenset(
    {
        "view_revision_id",
        "position_kind",
        "position_ref",
        "semantic_summary_snapshot",
        "content_hash",
        "grounded_at",
    }
)
_EVIDENCE_REF_KEYS = frozenset(
    {
        "ao_id",
        "source_session_id",
        "source_archive_id",
        "archive_commit_watermark",
        "source_sequence",
        "source_read_snapshot_watermark",
        "evidence_role",
        "evidence_content_hash",
        "captured_state",
    }
)
_APPLICABILITY_KEYS = frozenset(
    {
        "task_decision_type",
        "anchor_state",
        "failure_mode",
        "action_skill_role",
    }
)
_CATALOG_KEYS = frozenset({"base_revision_id", "block_content_hash", "block_ids"})
_FORK_COMMON_KEYS = frozenset(
    {
        "base_revision_id",
        "base_revision_block_content_hash",
        "anchor_block_id",
        "branch_steps",
    }
)
_BODY_KEYS: dict[str, frozenset[str]] = {
    OBJECT_KIND_LEGACY_CLAIM_CUT: frozenset({"claim", "cut_grounding"}),
    OBJECT_KIND_V2_FIXTURE_SET: frozenset({"base_revision", "documents"}),
    OBJECT_KIND_FORK_PROPOSAL: _FORK_COMMON_KEYS | frozenset({"report_hash", "author_role"}),
    OBJECT_KIND_FORK_NODE_OVERLAY: _FORK_COMMON_KEYS | frozenset({"fork_node_id"}),
    OBJECT_KIND_TASK_OUTCOME_ASSESSMENT: frozenset(
        {
            "report_hash",
            "author_role",
            "verdict",
            "criteria_dispositions",
            "evidence_refs",
        }
    ),
    OBJECT_KIND_REPLAY_ELIGIBILITY: frozenset(
        {
            "fork_node_id",
            "base_revision_id",
            "anchor_block_id",
            "eligibility",
            "checked_at",
        }
    ),
    OBJECT_KIND_GUIDANCE_DOCUMENT: frozenset(
        {
            "guidance_id",
            "source_fork_id",
            "base_revision_id",
            "target_role",
            "capability_limits",
            "report_hash",
            "applicability",
            "risk",
            "evidence_refs",
        }
    ),
    OBJECT_KIND_GUIDANCE_EXECUTION_REPORT: frozenset(
        {"guidance_id", "source_fork_id", "parent_session_id", "steps"}
    ),
    OBJECT_KIND_BRANCH_EVALUATION: frozenset(
        {
            "evaluation_id",
            "subject_kind",
            "subject_id",
            "evidence_tier",
            "evidence_ref",
        }
    ),
    OBJECT_KIND_FIRST_ACTION_ALIGNMENT: frozenset(
        {
            "imagined_node_id",
            "actual_block_id",
            "alignment",
            "depth",
            "intervention_type",
            "target_boundary",
            "required_capabilities",
            "safety_control_params",
        }
    ),
    OBJECT_KIND_NATIVE_EDGE_DRAFT: frozenset(
        {
            "view_id",
            "purpose",
            "session_id",
            "task_run_id",
            "blocks",
            "edges",
            "coverage",
        }
    ),
    OBJECT_KIND_NATIVE_EDGE_REVISION: frozenset(
        {
            "revision_id",
            "view_id",
            "purpose",
            "session_id",
            "task_run_id",
            "blocks",
            "edges",
            "coverage",
            "construction_mode",
            "frozen_at",
        }
    ),
    OBJECT_KIND_NATIVE_EDGE_INDEX: frozenset({"revisions"}),
    OBJECT_KIND_OUTCOME_CONTRACT: frozenset(
        {
            "contract_id",
            "task_id",
            "source",
            "goals",
            "acceptance_criteria",
            "evaluator_bindings",
            "policy_constraints",
            "frozen_at",
        }
    ),
    OBJECT_KIND_STRATEGY_PATTERN_PROPOSAL: frozenset(
        {
            "author_role",
            "fork_node_id",
            "imagined_node_id",
            "intervention_type",
            "target_boundary",
            "mechanism",
            "expected_metric",
            "capability",
            "applicability_conditions",
        }
    ),
    OBJECT_KIND_STRATEGY_PATTERN: frozenset(
        {
            "pattern_id",
            "identity_version",
            "intervention_type",
            "target_boundary",
            "mechanism",
            "expected_metric",
            "capability",
            "applicability_conditions",
        }
    ),
    OBJECT_KIND_REPLAY_ELIGIBILITY_ASSESSMENT: frozenset(
        {
            "assessment_id",
            "fork_node_id",
            "base_revision_id",
            "anchor_block_id",
            "eligibility",
            "route",
            "reasons",
            "evidence_ref",
            "checked_at",
            "intervention_type",
            "target_boundary",
            "required_capabilities",
            "safety_control_params",
            "anchor_state",
            "dependencies",
            "permissions",
            "side_effect_isolation",
            "budget",
            "evaluator",
        }
    ),
    OBJECT_KIND_REPLAY_REQUEST: frozenset(
        {
            "request_id",
            "assessment_id",
            "fork_node_id",
            "base_revision_id",
            "anchor_block_id",
            "replay_run_id",
            "intervention_type",
            "target_boundary",
            "required_capabilities",
            "safety_control_params",
            "eligibility",
        }
    ),
    OBJECT_KIND_REPLAY_RESULT: frozenset(
        {
            "request_id",
            "replay_run_id",
            "fork_node_id",
            "base_revision_id",
            "status",
            "observed_block_ids",
            "first_action_alignment",
            "step_alignments",
        }
    ),
    OBJECT_KIND_REPLAY_AIG: frozenset(
        {
            "provenance",
            "base_revision_id",
            "replay_run_id",
            "source_fork_id",
            "revision_id",
            "native_revision",
        }
    ),
    OBJECT_KIND_DIRECT_EVIDENCE_ADMISSION: frozenset(
        {
            "admission_id",
            "fork_node_id",
            "base_revision_id",
            "replay_run_id",
            "replay_revision_id",
            "imagined_node_id",
            "actual_block_id",
            "alignment",
            "depth",
            "intervention_type",
            "target_boundary",
            "required_capabilities",
            "safety_control_params",
            "expected_intervention_type",
            "expected_target_boundary",
            "expected_required_capabilities",
            "expected_safety_control_params",
            "equivalence_explanation",
            "first_imagined_node",
            "route",
            "alignment_content_hash",
            "step_alignments",
            "branch_imagined_ids",
            "replay_block_ids",
            "first_action_direct",
            "outcome_direct",
            "evidence_tier",
            "reasons",
        }
    ),
    OBJECT_KIND_GUIDED_COUNTERFACTUAL_BRANCH: frozenset(
        {
            "author_role",
            "report_hash",
            "task_id",
            "source_fork_id",
            "base_revision_id",
            "retrieval_id",
            "target_role",
            "required_capabilities",
            "applicability",
            "risk",
            "steps",
        }
    ),
    OBJECT_KIND_AGENT_CANDIDATE_CATALOG: frozenset({"task_id", "candidates"}),
    OBJECT_KIND_GUIDANCE_APPROVAL: frozenset(
        {
            "guidance_id",
            "proposal_content_hash",
            "retrieval_id",
            "candidate_catalog_hash",
            "selected_agent_id",
            "source_fork_id",
            "base_revision_id",
            "target_role",
            "capability_limits",
            "report_hash",
            "applicability",
            "risk",
            "evidence_refs",
            "steps",
            "policy_version",
            "decision",
            "reason",
            "outcome_evidence",
            "guidance_content_hash",
        }
    ),
    OBJECT_KIND_GUIDANCE_PUBLICATION: frozenset(
        {
            "approval_content_hash",
            "guidance_content_hash",
            "guidance_id",
            "selected_agent_id",
            "target_role",
            "capability_limits",
            "report_hash",
            "source_fork_id",
            "base_revision_id",
            "steps",
            "evidence_refs",
            "policy_version",
            "decision",
            "reason",
            "outcome_evidence",
            "risk",
            "applicability",
            "publisher_labels",
            "published_at",
        }
    ),
    OBJECT_KIND_GUIDANCE_REJECTION: frozenset({"proposal_content_hash", "reason"}),
    OBJECT_KIND_TOOLLOOP_DRAFT_OP: frozenset(
        {"run_id", "seq", "verb", "args", "prev_content_hash"}
    ),
    OBJECT_KIND_DISPATCH_GATE_DECISION: frozenset(
        {
            "guidance_id",
            "guidance_content_hash",
            "parent_session_id",
            "bound_session_id",
            "source_fork_id",
            "source_guidance_id",
            "selected_agent_id",
            "executor_role",
            "executor_agent_id",
            "principal_labels",
            "bound_principal_labels",
            "permission",
            "budget",
            "isolation",
            "side_effect_allowlist",
            "requested_side_effects",
            "agent_capabilities",
            "capability_limits",
            "idempotency_state",
            "idempotency_key",
            "decision_id",
            "gates",
            "outcome",
            "failed_gate",
            "reasons",
        }
    ),
    OBJECT_KIND_BLOCKED_PRE_EXECUTION: frozenset(
        {
            "guidance_id",
            "guidance_content_hash",
            "parent_session_id",
            "bound_session_id",
            "source_fork_id",
            "source_guidance_id",
            "selected_agent_id",
            "executor_role",
            "executor_agent_id",
            "principal_labels",
            "bound_principal_labels",
            "permission",
            "budget",
            "isolation",
            "side_effect_allowlist",
            "requested_side_effects",
            "agent_capabilities",
            "capability_limits",
            "idempotency_state",
            "idempotency_key",
            "decision_id",
            "gates",
            "outcome",
            "failed_gate",
            "reasons",
        }
    ),
    OBJECT_KIND_REMEDIATION_DISPATCH_REQUEST: frozenset(
        {
            "request_id",
            "decision_id",
            "guidance_id",
            "guidance_content_hash",
            "parent_session_id",
            "source_fork_id",
            "source_guidance_id",
            "selected_agent_id",
            "target_role",
            "capability_limits",
            "idempotency_key",
            "requested_side_effects",
        }
    ),
    OBJECT_KIND_REMEDIATION_DISPATCH_CONFIRMATION: frozenset(
        {
            "confirmation_id",
            "request_id",
            "child_run_id",
            "status",
            "executor",
        }
    ),
    OBJECT_KIND_CHILD_REMEDIATION_RUN: frozenset(
        {
            "child_run_id",
            "parent_session_id",
            "source_fork_id",
            "source_guidance_id",
            "guidance_content_hash",
            "request_id",
            "selected_agent_id",
            "child_session_id",
            "output_ids",
        }
    ),
    OBJECT_KIND_CHILD_RUN_OUTPUT: frozenset(
        {
            "output_kind",
            "output_id",
            "child_run_id",
            "child_session_id",
            "parent_session_id",
            "source_fork_id",
            "source_guidance_id",
        }
    ),
    OBJECT_KIND_GUIDANCE_ADJUSTMENT: frozenset(
        {
            "adjustment_id",
            "guidance_content_hash",
            "guidance_id",
            "author_agent_id",
            "bound_agent_id",
            "parent_session_id",
            "target_role",
            "bound_target_role",
            "capability_limits",
            "bound_capability_limits",
            "bound_imagined_node_ids",
            "revised_steps",
        }
    ),
    OBJECT_KIND_REMEDIATION_CONSTRAINT: frozenset(
        {
            "constraint_id",
            "imagined_node_id",
            "alignment",
            "applicability",
            "statement",
            "source_fork_id",
            "source_guidance_id",
            "parent_session_id",
            "child_run_id",
            "report_content_hash",
        }
    ),
    OBJECT_KIND_PRODUCTION_DIRECT_ADMISSION: frozenset(
        {
            "admission_id",
            "evidence_tier",
            "imagined_node_id",
            "actual_block_id",
            "alignment",
            "source_fork_id",
            "source_guidance_id",
            "parent_session_id",
            "child_run_id",
            "report_content_hash",
            "child_block_ids",
            "reasons",
        }
    ),
}
_NATIVE_EDGE_LIFECYCLE_FIELDS = frozenset(
    {
        "status",
        "critic_verdict",
        "carried_artifact",
        "downstream_effect",
        "source_evidence",
        "target_evidence",
        "source_evidence_refs",
        "target_evidence_refs",
        "grounding",
        "projection",
        "retrieval_subgraphs",
        "failure_marked",
        "failure_evidence",
        "subgraphs",
        "subgraph_id",
    }
)
_NATIVE_BLOCK_KEYS = frozenset(
    {
        "block_id",
        "role",
        "participant_id",
        "segment_ref",
        "ao_start_seq",
        "ao_end_seq",
        "summary",
        "input",
        "output",
        "authorship",
        "granularity_reason",
        "source_content_hashes",
        "source_refs",
        "block_kind",
    }
)
_NATIVE_EDGE_KEYS = frozenset(
    {
        "edge_id",
        "source_block_id",
        "target_block_id",
        "relation_type",
        "mechanism",
    }
)
_NATIVE_COVERAGE_KEYS = frozenset(
    {
        "total_ao_range",
        "inspected_refs",
        "included_ranges",
        "omitted_ranges",
        "snapshot_watermark",
    }
)
_NATIVE_RANGE_KEYS = frozenset({"start_seq", "end_seq", "segment_id"})
_NATIVE_OMITTED_KEYS = frozenset({"start_seq", "end_seq", "reason", "segment_id"})
_NATIVE_SOURCE_KEYS = frozenset(
    {
        "ao_id",
        "sequence",
        "participant_id",
        "segment_id",
        "session_id",
        "content_hash",
    }
)
_NATIVE_SEGMENT_KEYS = frozenset({"segment_id", "session_id"})
# Paper tool-loop fields. ``authorship`` is already required; a paper ``agent``
# block must use the three-value enum or leave it empty. ``defect_recorded``
# is omitted when absent.
PAPER_AUTHORSHIP_VALUES = frozenset({"authored", "relayed", "executed"})
_NATIVE_BLOCK_OPTIONAL_KEYS = frozenset({"defect_recorded"})
_DEFECT_RECORDED_KEYS = frozenset({"note"})
_OPTIONAL_BODY_KEYS: dict[str, frozenset[str]] = {
    OBJECT_KIND_BRANCH_EVALUATION: frozenset({"status"}),
    OBJECT_KIND_FIRST_ACTION_ALIGNMENT: frozenset({"equivalence_explanation"}),
    OBJECT_KIND_STRATEGY_PATTERN_PROPOSAL: frozenset({"semantic_summary"}),
    OBJECT_KIND_NATIVE_EDGE_REVISION: frozenset({"known_violations", "paper_sidecar_hash"}),
}
_APPLICABILITY_CONDITION_KEYS = frozenset({"condition_id", "statement"})
_PATTERN_IDENTITY_VERSIONS = frozenset({PATTERN_IDENTITY_VERSION})

_ACCEPTANCE_CRITERION_KEYS = frozenset({"criterion_id", "statement"})
_EVALUATOR_BINDING_KEYS = frozenset({"binding_id", "kind", "ref"})
_POLICY_CONSTRAINT_KEYS = frozenset({"constraint_id", "kind", "statement"})
_CONTRACT_SOURCES = frozenset({"user", "orchestrator", "task_system"})
_BINDING_KINDS = frozenset({"evaluator", "test", "acceptance"})
_CONSTRAINT_KINDS = frozenset({"policy", "sla"})
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_VERDICTS = frozenset({"success", "failure", "uncertain"})
_CRITERIA_DISPOSITIONS = frozenset({"explicit", "provisional", "missing", "conflict"})
_NON_EXPLICIT_CRITERIA = frozenset({"provisional", "missing", "conflict"})
_ELIGIBILITY_VALUES = frozenset(
    {"replayable", "conditionally_replayable", "non_replayable"}
)
EVIDENCE_TIER_PRODUCTION_DIRECT = "production-direct"
EVIDENCE_TIER_REPLAY_DIRECT = "replay-direct"
EVIDENCE_TIER_ANALOGICAL = "analogical"
EVIDENCE_TIER_OBSERVATIONAL = "observational"
EVIDENCE_TIER_MODEL_ONLY = "model-only"
_EVIDENCE_TIERS = frozenset(
    {
        EVIDENCE_TIER_PRODUCTION_DIRECT,
        EVIDENCE_TIER_REPLAY_DIRECT,
        EVIDENCE_TIER_ANALOGICAL,
        EVIDENCE_TIER_OBSERVATIONAL,
        EVIDENCE_TIER_MODEL_ONLY,
    }
)
_EVALUATION_STATUSES = frozenset(
    {"active", "challenged", "invalidated", "retired", "superseded"}
)
_EVALUATION_STATUS_ACTIVE = "active"
_SUBJECT_KINDS = frozenset({"imagined_node", "strategy_pattern"})
ALIGNMENT_EQUIVALENT = "equivalent"
ALIGNMENT_FUNCTIONALLY_EQUIVALENT = "functionally_equivalent"
ALIGNMENT_DIVERGENT = "divergent"
ALIGNMENT_NOT_EXECUTED = "not_executed"
_ALIGNMENTS = frozenset(
    {
        ALIGNMENT_EQUIVALENT,
        ALIGNMENT_FUNCTIONALLY_EQUIVALENT,
        ALIGNMENT_DIVERGENT,
        ALIGNMENT_NOT_EXECUTED,
    }
)
_ALIGNMENT_DEPTHS = frozenset({"first_action", "prefix", "full_branch"})
_REPLAY_ROUTES = frozenset({ROUTE_REPLAY, ROUTE_ANALOGICAL})
_REQUEST_ELIGIBILITY = frozenset({"replayable", "conditionally_replayable"})
_REPLAY_RESULT_STATUSES = frozenset({"completed", "failed", "blocked"})
_ADMISSION_TIERS = frozenset({EVIDENCE_TIER_REPLAY_DIRECT, "none"})
_ANCHOR_STATES = frozenset({"recoverable", "unrecoverable"})
_DEPENDENCY_STATES = frozenset({"available", "missing"})
_PERMISSION_STATES = frozenset({"sufficient", "insufficient"})
_ISOLATION_STATES = frozenset({"isolated", "unisolated"})
_BUDGET_STATES = frozenset({"compliant", "exceeded", "unbounded"})
_EVALUATOR_STATES = frozenset({"bound", "unbound"})
_DIRECT_ALIGNMENTS = frozenset(
    {ALIGNMENT_EQUIVALENT, ALIGNMENT_FUNCTIONALLY_EQUIVALENT}
)
_OUTCOME_DEPTHS = frozenset({"prefix", "full_branch"})
_HARD_ELIGIBILITY_REASONS = frozenset(
    {
        "anchor_unrecoverable",
        "capabilities_missing",
        "dependencies_missing",
        "isolation_not_sandbox",
        "permissions_insufficient",
        "side_effects_unisolated",
    }
)
_REPLAY_CONTROL_KEYS = frozenset(
    {
        "anchor_state",
        "budget",
        "dependencies",
        "evaluator",
        "intervention_type",
        "permissions",
        "required_capabilities",
        "safety_control_params",
        "side_effect_isolation",
        "target_boundary",
    }
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_UTC_MILLIS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
_STEP_KIND_FACTUAL = "factual_block_ref"
_STEP_KIND_IMAGINED = "imagined_node"
_READER_ROLE = "reader"
# Acting agent that may carry out guidance. See the module docstring.
TARGET_AGENT_ROLES = frozenset({"task_agent"})

# Closed vocabularies. Other modules import these instead of restating them.
EVIDENCE_TIERS = _EVIDENCE_TIERS
EVALUATION_STATUSES = _EVALUATION_STATUSES
EVALUATION_STATUS_ACTIVE = _EVALUATION_STATUS_ACTIVE
STEP_KIND_FACTUAL = _STEP_KIND_FACTUAL
STEP_KIND_IMAGINED = _STEP_KIND_IMAGINED
DIRECT_EVIDENCE_TIERS = frozenset(
    {EVIDENCE_TIER_PRODUCTION_DIRECT, EVIDENCE_TIER_REPLAY_DIRECT}
)
EXPLORATORY_EVIDENCE_TIERS = frozenset(
    {
        EVIDENCE_TIER_ANALOGICAL,
        EVIDENCE_TIER_OBSERVATIONAL,
        EVIDENCE_TIER_MODEL_ONLY,
    }
)
NEGATIVE_EVALUATION_STATUSES = frozenset(
    {"challenged", "invalidated", "retired", "superseded"}
)
OUTCOME_ALIGNMENT_DEPTHS = _OUTCOME_DEPTHS
ALIGNMENTS = _ALIGNMENTS
DIRECT_ALIGNMENTS = _DIRECT_ALIGNMENTS
BUDGET_STATES = _BUDGET_STATES
ISOLATION_STATES = _ISOLATION_STATES
DISPATCH_GATE_ORDER = (
    "acl",
    "permission",
    "budget",
    "security",
    "capability",
    "idempotency",
)
NON_EXECUTOR_ROLES = frozenset({"builder", "critic", "refiner", "reader", "memory"})
_DISPATCH_PERMISSIONS = frozenset({"granted", "denied"})
_IDEMPOTENCY_STATES = frozenset({"clear", "conflict"})
_DISPATCH_OUTCOMES = frozenset({"admitted", "blocked_pre_execution"})
_DISPATCH_EXECUTORS = frozenset({"scripted"})
_DISPATCH_CONFIRMATION_STATUSES = frozenset({"accepted"})
_CHILD_OUTPUT_KINDS = frozenset({"ao", "block"})
_CONSTRAINT_ALIGNMENTS = frozenset({ALIGNMENT_DIVERGENT, ALIGNMENT_NOT_EXECUTED})
_GATE_PASS_REASONS = {
    "acl": "acl_matched",
    "permission": "permission_granted",
    "budget": "budget_compliant",
    "security": "side_effects_isolated",
    "capability": "capability_covered",
    "idempotency": "idempotency_clear",
}
_DISPATCH_CONTROL_KEYS = frozenset(
    {
        "agent_capabilities",
        "bound_principal_labels",
        "bound_session_id",
        "budget",
        "capability_limits",
        "executor_agent_id",
        "executor_role",
        "guidance_content_hash",
        "guidance_id",
        "idempotency_key",
        "idempotency_state",
        "isolation",
        "parent_session_id",
        "permission",
        "principal_labels",
        "requested_side_effects",
        "selected_agent_id",
        "side_effect_allowlist",
        "source_fork_id",
        "source_guidance_id",
    }
)
_GATE_DERIVED_KEYS = frozenset({"decision_id", "failed_gate", "gates", "outcome", "reasons"})
_GATE_RESULT_KEYS = frozenset({"gate", "passed", "reason"})
_REVISED_STEP_KEYS = frozenset({"imagined_node_id", "statement"})


class SchemaVersion(str, Enum):
    """Closed protocol versions. Unknown values are rejected, never aliased."""

    LEGACY_CLAIM_CUT_V1 = SCHEMA_VERSION_LEGACY_CLAIM_CUT
    AIG_NATIVE_V2 = SCHEMA_VERSION_AIG_NATIVE_V2


SCHEMA_VERSIONS = frozenset(item.value for item in SchemaVersion)


class ProtocolContractError(ValueError):
    """A versioned protocol document was rejected."""


class UnsupportedSchemaVersion(ProtocolContractError):
    """``schema_version`` is absent or not a known version.

    The reader does not guess, downgrade, or map the payload onto another version.
    """


def canonical_dumps(value: Any) -> str:
    """Canonical JSON used by the protocol content hash."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash_input(
    schema_version: str, object_kind: str, body: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the exact object that ``protocol_content_hash`` digests."""
    return {
        "schema_version": schema_version,
        "object_kind": object_kind,
        "body": body,
    }


def protocol_content_hash(
    schema_version: str, object_kind: str, body: Mapping[str, Any]
) -> str:
    """SHA-256 hex of the version-tagged canonical body."""
    encoded = canonical_dumps(canonical_hash_input(schema_version, object_kind, body))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def is_safe_id(value: str) -> bool:
    """True when ``value`` matches the protocol id grammar."""
    return isinstance(value, str) and _SAFE_ID.fullmatch(value) is not None


def is_sha256_hex(value: object) -> bool:
    """True when ``value`` is a lowercase SHA-256 hex digest."""
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def protocol_envelope(
    schema_version: str, object_kind: str, body: Mapping[str, Any]
) -> dict[str, Any]:
    """Versioned envelope. The content hash is computed, not copied from the caller."""
    payload = dict(body)
    return {
        "schema_version": schema_version,
        "object_kind": object_kind,
        "content_hash": protocol_content_hash(schema_version, object_kind, payload),
        "body": payload,
    }


def block_content_hash(blocks: Sequence[Any]) -> str:
    """SHA-256 of canonical Block records sorted by ``block_id``.

    Each record is ``InfluenceBlock.to_dict()``. Claims, coverage, edges, and
    retrieval subgraphs are not part of the hashed list. This is the only
    block-digest implementation.
    """
    records = [block.to_dict() for block in blocks]
    records.sort(key=lambda record: record["block_id"])
    return hashlib.sha256(canonical_dumps(records).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BranchStepRef:
    """One branch step: a same-revision Block, or a branch-local imagined node."""

    step_kind: str
    block_id: str | None = None
    imagined_node_id: str | None = None


@dataclass(frozen=True)
class BaseRevisionCatalog:
    """Block membership and block-only digest of one frozen factual AIG revision.

    ``block_content_hash`` covers Block records sorted by ``block_id``.
    Claim ids are not members. Fork grounding may name only these block ids.
    """

    base_revision_id: str
    block_content_hash: str
    block_ids: tuple[str, ...]


@dataclass(frozen=True)
class LegacyClaimCutDocument:
    """Read-only legacy Claim plus cut grounding. Not a v2 fork overlay."""

    schema_version: str
    object_kind: str
    content_hash: str
    claim: InfluenceClaim
    cut_grounding: InfluenceGrounding


@dataclass(frozen=True)
class ForkProposal:
    """Reader fork proposal anchored to one Block on a frozen base revision."""

    schema_version: str
    object_kind: str
    content_hash: str
    base_revision_id: str
    base_revision_block_content_hash: str
    anchor_block_id: str
    report_hash: str
    author_role: str
    branch_steps: tuple[BranchStepRef, ...]


@dataclass(frozen=True)
class ForkNodeOverlay:
    """ForkNode overlay. It is not an AO-derived factual Block."""

    schema_version: str
    object_kind: str
    content_hash: str
    fork_node_id: str
    base_revision_id: str
    base_revision_block_content_hash: str
    anchor_block_id: str
    branch_steps: tuple[BranchStepRef, ...]


@dataclass(frozen=True)
class CriterionDisposition:
    """One acceptance criterion's disposition. The key is the contract id."""

    criterion_id: str
    disposition: str


@dataclass(frozen=True)
class TaskOutcomeAssessment:
    """Reader verdict. Each contract criterion has its own disposition.

    A non-explicit disposition, a missing criterion, or a key set that does
    not match the contract stays uncertain. The policy, not this record,
    compares the keys with an OutcomeContract.
    """

    schema_version: str
    object_kind: str
    content_hash: str
    report_hash: str
    author_role: str
    verdict: str
    criterion_dispositions: tuple[CriterionDisposition, ...]
    evidence_refs: tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class AcceptanceCriterion:
    """One frozen acceptance criterion. The statement is not rewritable."""

    criterion_id: str
    statement: str


@dataclass(frozen=True)
class EvaluatorBinding:
    """Evaluator, test, or acceptance check bound when the task starts."""

    binding_id: str
    kind: str
    ref: str


@dataclass(frozen=True)
class PolicyConstraint:
    """Frozen policy or SLA constraint. The text is not an outcome signal."""

    constraint_id: str
    kind: str
    statement: str


@dataclass(frozen=True)
class OutcomeContract:
    """Task-start contract. Reading it does not adjudicate or dispatch."""

    schema_version: str
    object_kind: str
    content_hash: str
    contract_id: str
    task_id: str
    source: str
    goals: tuple[str, ...]
    acceptance_criteria: tuple[AcceptanceCriterion, ...]
    evaluator_bindings: tuple[EvaluatorBinding, ...]
    policy_constraints: tuple[PolicyConstraint, ...]
    frozen_at: str


@dataclass(frozen=True)
class ApplicabilityCondition:
    """One controlled applicability condition. The statement is matched exactly."""

    condition_id: str
    statement: str


@dataclass(frozen=True)
class StrategyPatternProposal:
    """Reader input. ``semantic_summary`` is not part of pattern identity."""

    schema_version: str
    object_kind: str
    content_hash: str
    author_role: str
    fork_node_id: str
    imagined_node_id: str
    intervention_type: str
    target_boundary: str
    mechanism: str
    expected_metric: str
    capability: str
    applicability_conditions: tuple[ApplicabilityCondition, ...]
    semantic_summary: str | None = None


@dataclass(frozen=True)
class CounterfactualStrategyPattern:
    """Server-generated pattern. ``pattern_id`` is the controlled-field digest."""

    schema_version: str
    object_kind: str
    content_hash: str
    pattern_id: str
    identity_version: str
    intervention_type: str
    target_boundary: str
    mechanism: str
    expected_metric: str
    capability: str
    applicability_conditions: tuple[ApplicabilityCondition, ...]


@dataclass(frozen=True)
class ReplayEligibility:
    """Re-evaluable replay status. This value does not start a replay."""

    schema_version: str
    object_kind: str
    content_hash: str
    fork_node_id: str
    base_revision_id: str
    anchor_block_id: str
    eligibility: str
    checked_at: str


@dataclass(frozen=True)
class GuidanceDocument:
    """Frozen, Reader-approved guidance. Publishing it does not dispatch an agent."""

    schema_version: str
    object_kind: str
    content_hash: str
    guidance_id: str
    source_fork_id: str
    base_revision_id: str
    target_role: str
    capability_limits: tuple[str, ...]
    report_hash: str
    applicability: ApplicabilityCheck
    risk: str
    evidence_refs: tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class GuidanceStepAlignment:
    """One imagined node mapped to an actual Block, or marked not executed."""

    imagined_node_id: str
    alignment: str
    actual_block_id: str | None


@dataclass(frozen=True)
class GuidanceExecutionReport:
    """Structured fulfillment record. It does not mutate the source run."""

    schema_version: str
    object_kind: str
    content_hash: str
    guidance_id: str
    source_fork_id: str
    parent_session_id: str
    steps: tuple[GuidanceStepAlignment, ...]


@dataclass(frozen=True)
class GuidanceActionStep:
    """One proposed action. It carries the evidence that supports it."""

    step_id: str
    imagined_node_id: str
    statement: str
    evidence_refs: tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class OutcomeEvidenceSignal:
    """One evaluator signal frozen into an approval. Not a telemetry summary."""

    binding_id: str
    signal_kind: str
    signal: str
    evidence_ref: str


@dataclass(frozen=True)
class GuidedCounterfactualBranch:
    """Reader proposal. It does not select an agent or freeze guidance."""

    schema_version: str
    object_kind: str
    content_hash: str
    author_role: str
    report_hash: str
    task_id: str
    source_fork_id: str
    base_revision_id: str
    retrieval_id: str
    target_role: str
    required_capabilities: tuple[str, ...]
    applicability: ApplicabilityCheck
    risk: str
    steps: tuple[GuidanceActionStep, ...]


@dataclass(frozen=True)
class AgentCandidate:
    """One directory row. Eligibility is not a score on this row."""

    agent_id: str
    roles: tuple[str, ...]
    capabilities: tuple[str, ...]
    acl_labels: tuple[str, ...]
    task_member: bool
    available: bool


@dataclass(frozen=True)
class AgentCandidateCatalog:
    """One version of the candidate directory. A later version is a new hash."""

    schema_version: str
    object_kind: str
    content_hash: str
    task_id: str
    candidates: tuple[AgentCandidate, ...]


@dataclass(frozen=True)
class GuidanceApproval:
    """Server freeze of a GuidanceDocument projection plus the selected agent."""

    schema_version: str
    object_kind: str
    content_hash: str
    guidance_id: str
    proposal_content_hash: str
    retrieval_id: str
    candidate_catalog_hash: str
    selected_agent_id: str
    source_fork_id: str
    base_revision_id: str
    target_role: str
    capability_limits: tuple[str, ...]
    report_hash: str
    applicability: ApplicabilityCheck
    risk: str
    evidence_refs: tuple[EvidenceRef, ...]
    steps: tuple[GuidanceActionStep, ...]
    policy_version: str
    decision: str
    reason: str
    outcome_evidence: tuple[OutcomeEvidenceSignal, ...]
    guidance_content_hash: str


@dataclass(frozen=True)
class GuidancePublication:
    """Memory render of one approval. It does not start a remediation run."""

    schema_version: str
    object_kind: str
    content_hash: str
    approval_content_hash: str
    guidance_content_hash: str
    guidance_id: str
    selected_agent_id: str
    target_role: str
    capability_limits: tuple[str, ...]
    report_hash: str
    source_fork_id: str
    base_revision_id: str
    steps: tuple[GuidanceActionStep, ...]
    evidence_refs: tuple[EvidenceRef, ...]
    policy_version: str
    decision: str
    reason: str
    outcome_evidence: tuple[OutcomeEvidenceSignal, ...]
    risk: str
    applicability: ApplicabilityCheck
    publisher_labels: tuple[str, ...]
    published_at: str


@dataclass(frozen=True)
class ToolloopDraftOp:
    """One durable tool-loop edit. Replaying the log rebuilds the draft."""

    schema_version: str
    object_kind: str
    content_hash: str
    run_id: str
    seq: int
    verb: str
    args: dict[str, Any]
    prev_content_hash: str


@dataclass(frozen=True)
class GuidanceRejection:
    """A proposal that must not be approved or published."""

    schema_version: str
    object_kind: str
    content_hash: str
    proposal_content_hash: str
    reason: str


@dataclass(frozen=True)
class DispatchGateResult:
    """One server gate. ``passed`` is derived, not chosen by the caller."""

    gate: str
    passed: bool
    reason: str


@dataclass(frozen=True)
class DispatchGateJudgment:
    """Derived dispatch verdict. Submitters do not set ``outcome``."""

    gates: tuple[DispatchGateResult, ...]
    outcome: str
    failed_gate: str | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class DispatchGateDecision:
    """Versioned gate audit. A failure uses object kind ``blocked_pre_execution``."""

    schema_version: str
    object_kind: str
    content_hash: str
    decision_id: str
    guidance_id: str
    guidance_content_hash: str
    parent_session_id: str
    bound_session_id: str
    source_fork_id: str
    source_guidance_id: str
    selected_agent_id: str
    executor_role: str
    executor_agent_id: str
    principal_labels: tuple[str, ...]
    bound_principal_labels: tuple[str, ...]
    permission: str
    budget: str
    isolation: str
    side_effect_allowlist: tuple[str, ...]
    requested_side_effects: tuple[str, ...]
    agent_capabilities: tuple[str, ...]
    capability_limits: tuple[str, ...]
    idempotency_state: str
    idempotency_key: str
    gates: tuple[DispatchGateResult, ...]
    outcome: str
    failed_gate: str | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RemediationDispatchRequest:
    """Worker input. The Coforge worker is the executor; P12 connects it."""

    schema_version: str
    object_kind: str
    content_hash: str
    request_id: str
    decision_id: str
    guidance_id: str
    guidance_content_hash: str
    parent_session_id: str
    source_fork_id: str
    source_guidance_id: str
    selected_agent_id: str
    target_role: str
    capability_limits: tuple[str, ...]
    idempotency_key: str
    requested_side_effects: tuple[str, ...]


@dataclass(frozen=True)
class RemediationDispatchConfirmation:
    """Boundary acknowledgement. ``scripted`` is the only executor until P12."""

    schema_version: str
    object_kind: str
    content_hash: str
    confirmation_id: str
    request_id: str
    child_run_id: str
    status: str
    executor: str


@dataclass(frozen=True)
class ChildRemediationRun:
    """Linked child run. It does not append to the parent failure run."""

    schema_version: str
    object_kind: str
    content_hash: str
    child_run_id: str
    parent_session_id: str
    source_fork_id: str
    source_guidance_id: str
    guidance_content_hash: str
    request_id: str
    selected_agent_id: str
    child_session_id: str
    output_ids: tuple[str, ...]


@dataclass(frozen=True)
class ChildRunOutput:
    """One AO or Block from the child run. Provenance fields are mandatory."""

    schema_version: str
    object_kind: str
    content_hash: str
    output_kind: str
    output_id: str
    child_run_id: str
    child_session_id: str
    parent_session_id: str
    source_fork_id: str
    source_guidance_id: str


@dataclass(frozen=True)
class GuidanceRevisionStep:
    """One adjusted statement. It does not replace the frozen guidance step."""

    imagined_node_id: str
    statement: str


@dataclass(frozen=True)
class GuidanceAdjustment:
    """Versioned appendix. The frozen GuidanceDocument stays unchanged."""

    schema_version: str
    object_kind: str
    content_hash: str
    adjustment_id: str
    guidance_content_hash: str
    guidance_id: str
    author_agent_id: str
    bound_agent_id: str
    parent_session_id: str
    target_role: str
    bound_target_role: str
    capability_limits: tuple[str, ...]
    bound_capability_limits: tuple[str, ...]
    bound_imagined_node_ids: tuple[str, ...]
    revised_steps: tuple[GuidanceRevisionStep, ...]


@dataclass(frozen=True)
class RemediationConstraint:
    """Applicability note for a step that is not production-direct evidence."""

    schema_version: str
    object_kind: str
    content_hash: str
    constraint_id: str
    imagined_node_id: str
    alignment: str
    applicability: ApplicabilityCheck
    statement: str
    source_fork_id: str
    source_guidance_id: str
    parent_session_id: str
    child_run_id: str
    report_content_hash: str


@dataclass(frozen=True)
class ProductionDirectAdmission:
    """Server admission of one aligned child-run block as production-direct."""

    schema_version: str
    object_kind: str
    content_hash: str
    admission_id: str
    evidence_tier: str
    imagined_node_id: str
    actual_block_id: str
    alignment: str
    source_fork_id: str
    source_guidance_id: str
    parent_session_id: str
    child_run_id: str
    report_content_hash: str
    child_block_ids: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class BranchEvaluation:
    """Append-only evaluation. Analogical is stored as analogical.

    A new append is recorded as ``active`` when ``status`` is omitted.
    Counterevidence is a later append: challenged, invalidated, retired, or
    superseded. This record is not deleted here, and precedent retrieval is
    not filtered here. ``evidence_ref`` points at the evidence source.
    """

    schema_version: str
    object_kind: str
    content_hash: str
    evaluation_id: str
    subject_kind: str
    subject_id: str
    evidence_tier: str
    evidence_ref: str
    status: str = _EVALUATION_STATUS_ACTIVE


@dataclass(frozen=True)
class FirstActionAlignment:
    """Submitter's first-action observation. It does not append evidence.

    ``alignment``, ``depth``, and the audited alignment fields are stored as
    submitted. Prefix or full-branch depth is an observation. It does not
    grant outcome evidence until ``judge_direct_admission`` verifies
    ``step_alignments`` against the branch and the Replay AIG.
    """

    schema_version: str
    object_kind: str
    content_hash: str
    imagined_node_id: str
    actual_block_id: str | None
    alignment: str
    depth: str
    intervention_type: str
    target_boundary: str
    required_capabilities: tuple[str, ...]
    safety_control_params: dict[str, str]
    equivalence_explanation: str | None = None


@dataclass(frozen=True)
class ReplayControlDecision:
    """Server reading of one recoverability snapshot. Callers do not pick eligibility."""

    eligibility: str
    route: str
    reasons: tuple[str, ...]
    intervention_type: str
    target_boundary: str
    required_capabilities: tuple[str, ...]
    safety_control_params: dict[str, str]
    anchor_state: str
    dependencies: str
    permissions: str
    side_effect_isolation: str
    budget: str
    evaluator: str


@dataclass(frozen=True)
class DirectAdmissionDecision:
    """Server gate for replay-direct evidence. Submitters do not set these flags."""

    first_action_direct: bool
    outcome_direct: bool
    evidence_tier: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ReplayEligibilityAssessment:
    """One append-only eligibility judgment. A later judgment does not replace it."""

    schema_version: str
    object_kind: str
    content_hash: str
    assessment_id: str
    fork_node_id: str
    base_revision_id: str
    anchor_block_id: str
    eligibility: str
    route: str
    reasons: tuple[str, ...]
    evidence_ref: str
    checked_at: str
    intervention_type: str
    target_boundary: str
    required_capabilities: tuple[str, ...]
    safety_control_params: dict[str, str]
    anchor_state: str
    dependencies: str
    permissions: str
    side_effect_isolation: str
    budget: str
    evaluator: str


@dataclass(frozen=True)
class ReplayRequest:
    """Worker input. ``non_replayable`` is not a request the worker can run."""

    schema_version: str
    object_kind: str
    content_hash: str
    request_id: str
    assessment_id: str
    fork_node_id: str
    base_revision_id: str
    anchor_block_id: str
    replay_run_id: str
    intervention_type: str
    target_boundary: str
    required_capabilities: tuple[str, ...]
    safety_control_params: dict[str, str]
    eligibility: str


@dataclass(frozen=True)
class ReplayResult:
    """Scripted or worker output. Reading it does not freeze an AIG."""

    schema_version: str
    object_kind: str
    content_hash: str
    request_id: str
    replay_run_id: str
    fork_node_id: str
    base_revision_id: str
    status: str
    observed_block_ids: tuple[str, ...]
    first_action_alignment: FirstActionAlignment | None
    step_alignments: tuple[GuidanceStepAlignment, ...]


@dataclass(frozen=True)
class ReplayAigDocument:
    """Frozen replay revision plus the provenance that keeps it out of production."""

    schema_version: str
    object_kind: str
    content_hash: str
    provenance: str
    base_revision_id: str
    replay_run_id: str
    source_fork_id: str
    revision_id: str
    revision: AigNativeViewRevision


@dataclass(frozen=True)
class DirectEvidenceAdmission:
    """Audit record of a server admission. The explanation is stored with it."""

    schema_version: str
    object_kind: str
    content_hash: str
    admission_id: str
    fork_node_id: str
    base_revision_id: str
    replay_run_id: str
    replay_revision_id: str
    imagined_node_id: str
    actual_block_id: str | None
    alignment: str
    depth: str
    intervention_type: str
    target_boundary: str
    required_capabilities: tuple[str, ...]
    safety_control_params: dict[str, str]
    expected_intervention_type: str
    expected_target_boundary: str
    expected_required_capabilities: tuple[str, ...]
    expected_safety_control_params: dict[str, str]
    equivalence_explanation: str | None
    first_imagined_node: bool
    route: str
    alignment_content_hash: str
    step_alignments: tuple[GuidanceStepAlignment, ...]
    branch_imagined_ids: tuple[str, ...]
    replay_block_ids: tuple[str, ...]
    first_action_direct: bool
    outcome_direct: bool
    evidence_tier: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class V2FixtureSet:
    """One golden v2 set: catalog plus the eight transport documents."""

    schema_version: str
    object_kind: str
    content_hash: str
    base_revision: BaseRevisionCatalog
    fork_proposal: ForkProposal
    fork_node_overlay: ForkNodeOverlay
    task_outcome_assessment: TaskOutcomeAssessment
    replay_eligibility: ReplayEligibility
    guidance_document: GuidanceDocument
    guidance_execution_report: GuidanceExecutionReport
    branch_evaluation: BranchEvaluation
    first_action_alignment: FirstActionAlignment


@dataclass(frozen=True)
class NativeEdgeDraftDocument:
    """Native-edge draft. Edges are not claims and have no lifecycle fields."""

    schema_version: str
    object_kind: str
    content_hash: str
    draft: AigNativeViewDraft


@dataclass(frozen=True)
class NativeEdgeRevisionDocument:
    """Frozen native-edge revision. It is not a claim view."""

    schema_version: str
    object_kind: str
    content_hash: str
    revision: AigNativeViewRevision


@dataclass(frozen=True)
class NativeEdgeIndexDocument:
    """Append-only index of frozen native-edge revisions.

    This is the store envelope written by ``NativeEdgeViewStore``. It is not
    one of the eight fork-fixture documents.
    """

    schema_version: str
    object_kind: str
    content_hash: str
    revisions: tuple[NativeEdgeRevisionDocument, ...]


def base_revision_catalog_from_view(revision: InfluenceViewRevision) -> BaseRevisionCatalog:
    """Block ids and a block-only digest of a frozen revision.

    The digest is not ``revision.content_hash``. Claims and coverage stay
    out of it, so the catalog does not change when only claims change.
    """
    if not isinstance(revision, InfluenceViewRevision):
        raise ProtocolContractError("base revision must be an InfluenceViewRevision")
    return BaseRevisionCatalog(
        base_revision_id=revision.revision_id,
        block_content_hash=block_content_hash(revision.blocks),
        block_ids=tuple(block.block_id for block in revision.blocks),
    )


def validate_v2_fork_grounding(
    document: ForkProposal | ForkNodeOverlay, catalog: BaseRevisionCatalog
) -> None:
    """Reject fork grounding that is not a Block in ``catalog``'s revision."""
    if not isinstance(catalog, BaseRevisionCatalog):
        raise ProtocolContractError("v2 fork grounding requires a base revision catalog")
    if document.base_revision_id != catalog.base_revision_id:
        raise ProtocolContractError(
            f"base_revision_id {document.base_revision_id!r} does not match "
            f"base revision {catalog.base_revision_id!r}"
        )
    if document.base_revision_block_content_hash != catalog.block_content_hash:
        raise ProtocolContractError(
            "base_revision block_content_hash does not match "
            f"base revision {catalog.base_revision_id!r}"
        )
    if document.anchor_block_id not in catalog.block_ids:
        raise ProtocolContractError(
            f"anchor_block_id {document.anchor_block_id!r} is not a Block "
            f"in base revision {catalog.base_revision_id!r}"
        )
    for step in document.branch_steps:
        if step.step_kind == _STEP_KIND_FACTUAL:
            if step.block_id not in catalog.block_ids:
                raise ProtocolContractError(
                    f"factual_block_ref {step.block_id!r} is not a Block "
                    f"in base revision {catalog.base_revision_id!r}"
                )
        elif step.imagined_node_id in catalog.block_ids:
            raise ProtocolContractError(
                f"imagined_node {step.imagined_node_id!r} is not branch-local; "
                "it collides with a factual Block"
            )


def read_protocol_document(
    payload: Mapping[str, Any],
) -> (
    LegacyClaimCutDocument
    | V2FixtureSet
    | NativeEdgeDraftDocument
    | NativeEdgeRevisionDocument
    | NativeEdgeIndexDocument
    | OutcomeContract
    | StrategyPatternProposal
    | CounterfactualStrategyPattern
    | ReplayEligibilityAssessment
    | ReplayRequest
    | ReplayResult
    | ReplayAigDocument
    | DirectEvidenceAdmission
    | GuidedCounterfactualBranch
    | AgentCandidateCatalog
    | GuidanceApproval
    | GuidancePublication
    | GuidanceRejection
    | DispatchGateDecision
    | RemediationDispatchRequest
    | RemediationDispatchConfirmation
    | ChildRemediationRun
    | ChildRunOutput
    | GuidanceAdjustment
    | RemediationConstraint
    | ProductionDirectAdmission
    | ToolloopDraftOp
):
    """Parse a golden fixture. Unknown versions fail closed."""
    if not isinstance(payload, Mapping):
        raise ProtocolContractError("protocol document must be a dict")
    version = _schema_version_of(payload)
    kind = payload.get("object_kind")
    if version == SCHEMA_VERSION_AIG_NATIVE_V2 and kind in _FORK_OBJECT_KINDS:
        raise ProtocolContractError(
            f"object_kind {kind!r} requires a base revision catalog; "
            "refusing to guess grounding"
        )
    reader = _KIND_READERS.get(kind) if isinstance(kind, str) else None
    if (
        reader is None
        or not isinstance(kind, str)
        or kind not in _TOP_LEVEL_DOCUMENT_KINDS
        or _KIND_SCHEMA_VERSION.get(kind) != version
    ):
        raise ProtocolContractError(
            f"schema_version {version!r} does not accept object_kind {kind!r}"
        )
    return reader(payload)


def read_legacy_claim_cut(payload: Mapping[str, Any]) -> LegacyClaimCutDocument:
    """Parse a legacy Claim/Cut fixture. v2 payloads are refused, not mapped."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_LEGACY_CLAIM_CUT,
        expected_kind=OBJECT_KIND_LEGACY_CLAIM_CUT,
        reader_name="legacy reader",
        reject_claim_cut=False,
    )
    claim = _parse_claim(body["claim"])
    grounding = _parse_cut_grounding(body["cut_grounding"])
    return LegacyClaimCutDocument(
        schema_version=version,
        object_kind=OBJECT_KIND_LEGACY_CLAIM_CUT,
        content_hash=digest,
        claim=claim,
        cut_grounding=grounding,
    )


def read_v2_fixture_set(payload: Mapping[str, Any]) -> V2FixtureSet:
    """Parse the golden v2 set and ground its fork documents on that set's blocks.

    Execution-report steps must name imagined nodes defined on the overlay.
    Factual block refs fail closed unless they belong to the set's catalog.
    """
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_V2_FIXTURE_SET,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    catalog = _parse_catalog(body["base_revision"])
    documents = body["documents"]
    if not isinstance(documents, dict):
        raise ProtocolContractError("documents must be a dict")
    _require_exact_keys(documents, frozenset(_V2_DOCUMENT_KINDS), "documents")
    members = {
        kind: _read_v2_member(documents[kind], kind, catalog) for kind in _V2_DOCUMENT_KINDS
    }
    proposal = members[OBJECT_KIND_FORK_PROPOSAL]
    overlay = members[OBJECT_KIND_FORK_NODE_OVERLAY]
    replay = members[OBJECT_KIND_REPLAY_ELIGIBILITY]
    guidance = members[OBJECT_KIND_GUIDANCE_DOCUMENT]
    report = members[OBJECT_KIND_GUIDANCE_EXECUTION_REPORT]
    _require_same_base(replay.base_revision_id, catalog, "replay_eligibility.base_revision_id")
    if replay.anchor_block_id not in catalog.block_ids:
        raise ProtocolContractError(
            f"anchor_block_id {replay.anchor_block_id!r} is not a Block "
            f"in base revision {catalog.base_revision_id!r}"
        )
    _require_same_base(guidance.base_revision_id, catalog, "guidance_document.base_revision_id")
    if guidance.source_fork_id != overlay.fork_node_id:
        raise ProtocolContractError(
            "guidance_document.source_fork_id does not match fork_node_overlay.fork_node_id"
        )
    if report.source_fork_id != overlay.fork_node_id:
        raise ProtocolContractError(
            "guidance_execution_report.source_fork_id does not match "
            "fork_node_overlay.fork_node_id"
        )
    if report.guidance_id != guidance.guidance_id:
        raise ProtocolContractError(
            "guidance_execution_report.guidance_id does not match guidance_document.guidance_id"
        )
    if proposal.anchor_block_id != overlay.anchor_block_id:
        raise ProtocolContractError(
            "fork_proposal.anchor_block_id does not match fork_node_overlay.anchor_block_id"
        )
    _require_defined_imagined_nodes(overlay, report)
    return V2FixtureSet(
        schema_version=version,
        object_kind=OBJECT_KIND_V2_FIXTURE_SET,
        content_hash=digest,
        base_revision=catalog,
        fork_proposal=proposal,
        fork_node_overlay=overlay,
        task_outcome_assessment=members[OBJECT_KIND_TASK_OUTCOME_ASSESSMENT],
        replay_eligibility=replay,
        guidance_document=guidance,
        guidance_execution_report=report,
        branch_evaluation=members[OBJECT_KIND_BRANCH_EVALUATION],
        first_action_alignment=members[OBJECT_KIND_FIRST_ACTION_ALIGNMENT],
    )


def read_fork_proposal(
    payload: Mapping[str, Any], catalog: BaseRevisionCatalog
) -> ForkProposal:
    """Parse a v2 ForkProposal and require its anchor Block in ``catalog``."""
    _require_catalog(catalog)
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_FORK_PROPOSAL,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    base_revision_id, block_hash, anchor_block_id, branch_steps = _parse_fork_common(body)
    proposal = ForkProposal(
        schema_version=version,
        object_kind=OBJECT_KIND_FORK_PROPOSAL,
        content_hash=digest,
        base_revision_id=base_revision_id,
        base_revision_block_content_hash=block_hash,
        anchor_block_id=anchor_block_id,
        report_hash=_require_str(body.get("report_hash"), "report_hash"),
        author_role=_require_reader_author(body.get("author_role"), "ForkProposal"),
        branch_steps=branch_steps,
    )
    validate_v2_fork_grounding(proposal, catalog)
    return proposal


def read_fork_node_overlay(
    payload: Mapping[str, Any], catalog: BaseRevisionCatalog
) -> ForkNodeOverlay:
    """Parse a v2 ForkNode overlay and require its anchor Block in ``catalog``."""
    _require_catalog(catalog)
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_FORK_NODE_OVERLAY,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    base_revision_id, block_hash, anchor_block_id, branch_steps = _parse_fork_common(body)
    overlay = ForkNodeOverlay(
        schema_version=version,
        object_kind=OBJECT_KIND_FORK_NODE_OVERLAY,
        content_hash=digest,
        fork_node_id=_require_str(body.get("fork_node_id"), "fork_node_id"),
        base_revision_id=base_revision_id,
        base_revision_block_content_hash=block_hash,
        anchor_block_id=anchor_block_id,
        branch_steps=branch_steps,
    )
    validate_v2_fork_grounding(overlay, catalog)
    return overlay


def read_task_outcome_assessment(payload: Mapping[str, Any]) -> TaskOutcomeAssessment:
    """Parse a Reader TaskOutcomeAssessment. Other roles are rejected."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_TASK_OUTCOME_ASSESSMENT,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    author = _require_reader_author(body.get("author_role"), "TaskOutcomeAssessment")
    verdict = _require_choice(body.get("verdict"), "verdict", _VERDICTS)
    dispositions = _parse_criterion_dispositions(body.get("criteria_dispositions"))
    evidence_refs = _parse_evidence_refs(body.get("evidence_refs"))
    non_explicit = [
        item for item in dispositions if item.disposition in _NON_EXPLICIT_CRITERIA
    ]
    if not dispositions or non_explicit:
        shown = non_explicit[0].disposition if non_explicit else "missing"
        if verdict != "uncertain":
            raise ProtocolContractError(
                f"criteria_disposition {shown!r} must be uncertain"
            )
    elif not evidence_refs:
        raise ProtocolContractError("explicit criteria require evidence_refs")
    return TaskOutcomeAssessment(
        schema_version=version,
        object_kind=OBJECT_KIND_TASK_OUTCOME_ASSESSMENT,
        content_hash=digest,
        report_hash=_require_str(body.get("report_hash"), "report_hash"),
        author_role=author,
        verdict=verdict,
        criterion_dispositions=dispositions,
        evidence_refs=evidence_refs,
    )


def read_outcome_contract(payload: Mapping[str, Any]) -> OutcomeContract:
    """Parse a frozen OutcomeContract. It does not adjudicate or dispatch."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_OUTCOME_CONTRACT,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    return OutcomeContract(
        schema_version=version,
        object_kind=OBJECT_KIND_OUTCOME_CONTRACT,
        content_hash=digest,
        contract_id=_require_safe_id(body.get("contract_id"), "contract_id"),
        task_id=_require_safe_id(body.get("task_id"), "task_id"),
        source=_require_choice(body.get("source"), "source", _CONTRACT_SOURCES),
        goals=_parse_nonempty_str_list(body.get("goals"), "goals"),
        acceptance_criteria=_parse_acceptance_criteria(body.get("acceptance_criteria")),
        evaluator_bindings=_parse_evaluator_bindings(body.get("evaluator_bindings")),
        policy_constraints=_parse_policy_constraints(body.get("policy_constraints")),
        frozen_at=_require_str(body.get("frozen_at"), "frozen_at"),
    )


def strategy_pattern_identity_body(
    *,
    intervention_type: str,
    target_boundary: str,
    mechanism: str,
    expected_metric: str,
    capability: str,
    applicability_conditions: Sequence[ApplicabilityCondition | Mapping[str, str]],
) -> dict[str, Any]:
    """Canonical controlled fields. Order and semantic summary are not inputs."""
    return {
        "applicability_conditions": _condition_records(applicability_conditions),
        "capability": _require_str(capability, "capability"),
        "expected_metric": _require_str(expected_metric, "expected_metric"),
        "identity_version": PATTERN_IDENTITY_VERSION,
        "intervention_type": _require_str(intervention_type, "intervention_type"),
        "mechanism": _require_str(mechanism, "mechanism"),
        "target_boundary": _require_str(target_boundary, "target_boundary"),
    }


def strategy_pattern_id(
    *,
    intervention_type: str,
    target_boundary: str,
    mechanism: str,
    expected_metric: str,
    capability: str,
    applicability_conditions: Sequence[ApplicabilityCondition | Mapping[str, str]],
) -> str:
    """Digest of the versioned controlled fields. Callers do not supply an id."""
    return protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        OBJECT_KIND_STRATEGY_PATTERN_IDENTITY,
        strategy_pattern_identity_body(
            intervention_type=intervention_type,
            target_boundary=target_boundary,
            mechanism=mechanism,
            expected_metric=expected_metric,
            capability=capability,
            applicability_conditions=applicability_conditions,
        ),
    )


def strategy_pattern_document(proposal: StrategyPatternProposal) -> dict[str, Any]:
    """Build the server pattern envelope from a Reader proposal.

    The proposal's semantic summary, fork id, and imagined-node id are not
    copied. ``pattern_id`` is derived here.
    """
    if not isinstance(proposal, StrategyPatternProposal):
        raise ProtocolContractError("strategy_pattern_proposal required")
    identity = strategy_pattern_identity_body(
        intervention_type=proposal.intervention_type,
        target_boundary=proposal.target_boundary,
        mechanism=proposal.mechanism,
        expected_metric=proposal.expected_metric,
        capability=proposal.capability,
        applicability_conditions=proposal.applicability_conditions,
    )
    pattern_id = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_STRATEGY_PATTERN_IDENTITY, identity
    )
    body = {**identity, "pattern_id": pattern_id}
    return {
        "schema_version": SCHEMA_VERSION_AIG_NATIVE_V2,
        "object_kind": OBJECT_KIND_STRATEGY_PATTERN,
        "content_hash": protocol_content_hash(
            SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_STRATEGY_PATTERN, body
        ),
        "body": body,
    }


def read_strategy_pattern_proposal(payload: Mapping[str, Any]) -> StrategyPatternProposal:
    """Parse a Reader proposal. A client-supplied pattern id is rejected."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_STRATEGY_PATTERN_PROPOSAL,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    return StrategyPatternProposal(
        schema_version=version,
        object_kind=OBJECT_KIND_STRATEGY_PATTERN_PROPOSAL,
        content_hash=digest,
        author_role=_require_reader_author(body.get("author_role"), "StrategyPatternProposal"),
        fork_node_id=_require_str(body.get("fork_node_id"), "fork_node_id"),
        imagined_node_id=_require_str(body.get("imagined_node_id"), "imagined_node_id"),
        intervention_type=_require_str(body.get("intervention_type"), "intervention_type"),
        target_boundary=_require_str(body.get("target_boundary"), "target_boundary"),
        mechanism=_require_str(body.get("mechanism"), "mechanism"),
        expected_metric=_require_str(body.get("expected_metric"), "expected_metric"),
        capability=_require_str(body.get("capability"), "capability"),
        applicability_conditions=_parse_applicability_conditions(
            body.get("applicability_conditions")
        ),
        semantic_summary=_optional_semantic_summary(body),
    )


def read_strategy_pattern(payload: Mapping[str, Any]) -> CounterfactualStrategyPattern:
    """Parse a server pattern and require ``pattern_id`` to match its fields."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_STRATEGY_PATTERN,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    identity_version = _require_choice(
        body.get("identity_version"), "identity_version", _PATTERN_IDENTITY_VERSIONS
    )
    conditions = _parse_applicability_conditions(body.get("applicability_conditions"))
    pattern_id = _require_str(body.get("pattern_id"), "pattern_id")
    expected = strategy_pattern_id(
        intervention_type=_require_str(body.get("intervention_type"), "intervention_type"),
        target_boundary=_require_str(body.get("target_boundary"), "target_boundary"),
        mechanism=_require_str(body.get("mechanism"), "mechanism"),
        expected_metric=_require_str(body.get("expected_metric"), "expected_metric"),
        capability=_require_str(body.get("capability"), "capability"),
        applicability_conditions=conditions,
    )
    if pattern_id != expected:
        raise ProtocolContractError(
            f"pattern_id {pattern_id!r} does not match controlled identity fields"
        )
    return CounterfactualStrategyPattern(
        schema_version=version,
        object_kind=OBJECT_KIND_STRATEGY_PATTERN,
        content_hash=digest,
        pattern_id=pattern_id,
        identity_version=identity_version,
        intervention_type=_require_str(body.get("intervention_type"), "intervention_type"),
        target_boundary=_require_str(body.get("target_boundary"), "target_boundary"),
        mechanism=_require_str(body.get("mechanism"), "mechanism"),
        expected_metric=_require_str(body.get("expected_metric"), "expected_metric"),
        capability=_require_str(body.get("capability"), "capability"),
        applicability_conditions=conditions,
    )


def read_replay_eligibility(payload: Mapping[str, Any]) -> ReplayEligibility:
    """Parse a replay eligibility judgment. It does not start a replay run."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_REPLAY_ELIGIBILITY,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    return ReplayEligibility(
        schema_version=version,
        object_kind=OBJECT_KIND_REPLAY_ELIGIBILITY,
        content_hash=digest,
        fork_node_id=_require_str(body.get("fork_node_id"), "fork_node_id"),
        base_revision_id=_require_str(body.get("base_revision_id"), "base_revision_id"),
        anchor_block_id=_require_str(body.get("anchor_block_id"), "anchor_block_id"),
        eligibility=_require_choice(body.get("eligibility"), "eligibility", _ELIGIBILITY_VALUES),
        checked_at=_require_str(body.get("checked_at"), "checked_at"),
    )


def read_guidance_document(payload: Mapping[str, Any]) -> GuidanceDocument:
    """Parse a frozen GuidanceDocument. It does not select or dispatch a target."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_GUIDANCE_DOCUMENT,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    parsed_applicability = _parse_applicability(body.get("applicability"))
    return GuidanceDocument(
        schema_version=version,
        object_kind=OBJECT_KIND_GUIDANCE_DOCUMENT,
        content_hash=digest,
        guidance_id=_require_str(body.get("guidance_id"), "guidance_id"),
        source_fork_id=_require_str(body.get("source_fork_id"), "source_fork_id"),
        base_revision_id=_require_str(body.get("base_revision_id"), "base_revision_id"),
        target_role=_require_choice(body.get("target_role"), "target_role", TARGET_AGENT_ROLES),
        capability_limits=_parse_str_list(body.get("capability_limits"), "capability_limits"),
        report_hash=_require_str(body.get("report_hash"), "report_hash"),
        applicability=parsed_applicability,
        risk=_require_str(body.get("risk"), "risk"),
        evidence_refs=_parse_evidence_refs(body.get("evidence_refs")),
    )


def read_guidance_execution_report(payload: Mapping[str, Any]) -> GuidanceExecutionReport:
    """Parse a GuidanceExecutionReport. It does not write production evidence."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_GUIDANCE_EXECUTION_REPORT,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    return GuidanceExecutionReport(
        schema_version=version,
        object_kind=OBJECT_KIND_GUIDANCE_EXECUTION_REPORT,
        content_hash=digest,
        guidance_id=_require_str(body.get("guidance_id"), "guidance_id"),
        source_fork_id=_require_str(body.get("source_fork_id"), "source_fork_id"),
        parent_session_id=_require_str(body.get("parent_session_id"), "parent_session_id"),
        steps=_parse_execution_steps(body.get("steps")),
    )


def read_branch_evaluation(payload: Mapping[str, Any]) -> BranchEvaluation:
    """Parse one append-only evaluation. Unknown tiers and statuses are rejected."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_BRANCH_EVALUATION,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    tier = _require_choice(body.get("evidence_tier"), "evidence_tier", _EVIDENCE_TIERS)
    return BranchEvaluation(
        schema_version=version,
        object_kind=OBJECT_KIND_BRANCH_EVALUATION,
        content_hash=digest,
        evaluation_id=_require_str(body.get("evaluation_id"), "evaluation_id"),
        subject_kind=_require_choice(body.get("subject_kind"), "subject_kind", _SUBJECT_KINDS),
        subject_id=_require_str(body.get("subject_id"), "subject_id"),
        evidence_tier=tier,
        evidence_ref=_require_str(body.get("evidence_ref"), "evidence_ref"),
        status=_evaluation_status(body),
    )


def read_native_edge_draft(payload: Mapping[str, Any]) -> NativeEdgeDraftDocument:
    """Parse a native-edge draft. Claim and cut fields are refused."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_NATIVE_EDGE_DRAFT,
        reader_name="native-edge reader",
        reject_claim_cut=True,
    )
    _reject_native_lifecycle_fields(payload)
    _check_native_body_shape(
        body,
        paper_strict=(body.get("construction_mode") == "adaptive_agentic_aig"),
    )
    try:
        draft = AigNativeViewDraft.from_body(body)
    except (TypeError, ValueError) as exc:
        raise ProtocolContractError(str(exc)) from exc
    _reject_native_structure(draft)
    return NativeEdgeDraftDocument(
        schema_version=version,
        object_kind=OBJECT_KIND_NATIVE_EDGE_DRAFT,
        content_hash=digest,
        draft=draft,
    )


def read_native_edge_revision(payload: Mapping[str, Any]) -> NativeEdgeRevisionDocument:
    """Parse a frozen native-edge revision. It does not project claims."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_NATIVE_EDGE_REVISION,
        reader_name="native-edge reader",
        reject_claim_cut=True,
    )
    _reject_native_lifecycle_fields(payload)
    _check_native_body_shape(
        body,
        paper_strict=(body.get("construction_mode") == "adaptive_agentic_aig"),
    )
    try:
        revision = AigNativeViewRevision.from_body(body, content_hash=digest)
    except (TypeError, ValueError) as exc:
        raise ProtocolContractError(str(exc)) from exc
    _reject_native_structure(revision)
    return NativeEdgeRevisionDocument(
        schema_version=version,
        object_kind=OBJECT_KIND_NATIVE_EDGE_REVISION,
        content_hash=digest,
        revision=revision,
    )


def read_native_edge_index(payload: Mapping[str, Any]) -> NativeEdgeIndexDocument:
    """Parse a native-edge index. Each revision is a native-edge document."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_NATIVE_EDGE_INDEX,
        reader_name="native-edge reader",
        reject_claim_cut=True,
    )
    raw = body.get("revisions")
    if not isinstance(raw, list):
        raise ProtocolContractError("revisions must be a list")
    revisions = tuple(read_native_edge_revision(item) for item in raw)
    return NativeEdgeIndexDocument(
        schema_version=version,
        object_kind=OBJECT_KIND_NATIVE_EDGE_INDEX,
        content_hash=digest,
        revisions=revisions,
    )


def read_first_action_alignment(payload: Mapping[str, Any]) -> FirstActionAlignment:
    """Parse a first-action alignment observation.

    ``alignment`` and ``depth`` are the submitter's record, including prefix
    depth. This reader does not admit direct evidence.
    ``judge_direct_admission`` is the server gate.
    """
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_FIRST_ACTION_ALIGNMENT,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    alignment = _require_choice(body.get("alignment"), "alignment", _ALIGNMENTS)
    observed = body.get("actual_block_id")
    if alignment == ALIGNMENT_NOT_EXECUTED:
        if observed is not None:
            raise ProtocolContractError(
                "not_executed alignment requires actual_block_id to be null"
            )
        actual_block_id = None
    else:
        actual_block_id = _require_str(observed, "actual_block_id")
    return FirstActionAlignment(
        schema_version=version,
        object_kind=OBJECT_KIND_FIRST_ACTION_ALIGNMENT,
        content_hash=digest,
        imagined_node_id=_require_str(body.get("imagined_node_id"), "imagined_node_id"),
        actual_block_id=actual_block_id,
        alignment=alignment,
        depth=_require_choice(body.get("depth"), "depth", _ALIGNMENT_DEPTHS),
        intervention_type=_require_str(body.get("intervention_type"), "intervention_type"),
        target_boundary=_require_str(body.get("target_boundary"), "target_boundary"),
        required_capabilities=_parse_str_list(
            body.get("required_capabilities"), "required_capabilities"
        ),
        safety_control_params=_parse_str_map(
            body.get("safety_control_params"), "safety_control_params"
        ),
        equivalence_explanation=_optional_explanation(body),
    )


def is_replay_provenance_identity(
    *,
    revision_id: str,
    session_id: str,
    task_run_id: str,
    view_id: str | None = None,
) -> bool:
    """True when any identity field uses the reserved replay prefix."""
    if (
        isinstance(revision_id, str)
        and revision_id.startswith(REPLAY_REVISION_PREFIX)
    ):
        return True
    if isinstance(session_id, str) and session_id.startswith(REPLAY_SESSION_PREFIX):
        return True
    if isinstance(task_run_id, str) and task_run_id.startswith(REPLAY_TASK_RUN_PREFIX):
        return True
    return isinstance(view_id, str) and view_id.startswith(REPLAY_VIEW_PREFIX)


def is_replay_revision_id(revision_id: str) -> bool:
    """True when a revision id was minted for a Replay AIG."""
    return isinstance(revision_id, str) and revision_id.startswith(REPLAY_REVISION_PREFIX)


def is_replay_session_id(session_id: str) -> bool:
    """True when a session id belongs to a replay run."""
    return isinstance(session_id, str) and session_id.startswith(REPLAY_SESSION_PREFIX)


def replay_aig_revision_id(
    *, base_revision_id: str, replay_run_id: str, source_fork_id: str
) -> str:
    """Content-addressed revision id. The prefix keeps it out of production ids."""
    digest = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        OBJECT_KIND_REPLAY_AIG_IDENTITY,
        {
            "base_revision_id": _require_str(base_revision_id, "base_revision_id"),
            "provenance": REPLAY_PROVENANCE,
            "replay_run_id": _require_safe_id(replay_run_id, "replay_run_id"),
            "source_fork_id": _require_str(source_fork_id, "source_fork_id"),
        },
    )
    return REPLAY_REVISION_PREFIX + digest


def inspect_replay_controls(controls: Mapping[str, Any]) -> ReplayControlDecision:
    """Judge replayability from controlled fields. The caller does not pass a verdict."""
    if not isinstance(controls, Mapping):
        raise ProtocolContractError("replay controls must be a dict")
    _require_exact_keys(controls, _REPLAY_CONTROL_KEYS, "replay_controls")
    intervention_type = _require_str(controls.get("intervention_type"), "intervention_type")
    target_boundary = _require_str(controls.get("target_boundary"), "target_boundary")
    capabilities = _parse_str_list(
        controls.get("required_capabilities"), "required_capabilities"
    )
    safety = _parse_str_map(controls.get("safety_control_params"), "safety_control_params")
    anchor_state = _require_choice(controls.get("anchor_state"), "anchor_state", _ANCHOR_STATES)
    dependencies = _require_choice(
        controls.get("dependencies"), "dependencies", _DEPENDENCY_STATES
    )
    permissions = _require_choice(controls.get("permissions"), "permissions", _PERMISSION_STATES)
    isolation = _require_choice(
        controls.get("side_effect_isolation"), "side_effect_isolation", _ISOLATION_STATES
    )
    budget = _require_choice(controls.get("budget"), "budget", _BUDGET_STATES)
    evaluator = _require_choice(controls.get("evaluator"), "evaluator", _EVALUATOR_STATES)
    failures: list[str] = []
    if anchor_state != "recoverable":
        failures.append("anchor_unrecoverable")
    if not capabilities:
        failures.append("capabilities_missing")
    if dependencies != "available":
        failures.append("dependencies_missing")
    if safety.get("isolation") != "sandbox":
        failures.append("isolation_not_sandbox")
    if permissions != "sufficient":
        failures.append("permissions_insufficient")
    if isolation != "isolated":
        failures.append("side_effects_unisolated")
    if budget != "compliant":
        failures.append("budget_not_compliant")
    if evaluator == "unbound":
        failures.append("evaluator_unbound")
    if any(reason in _HARD_ELIGIBILITY_REASONS for reason in failures):
        eligibility = "non_replayable"
        route = ROUTE_ANALOGICAL
        reasons = tuple(sorted(failures))
    elif failures:
        eligibility = "conditionally_replayable"
        route = ROUTE_REPLAY
        reasons = tuple(sorted(failures))
    else:
        eligibility = "replayable"
        route = ROUTE_REPLAY
        reasons = ("controls_recoverable",)
    return ReplayControlDecision(
        eligibility=eligibility,
        route=route,
        reasons=reasons,
        intervention_type=intervention_type,
        target_boundary=target_boundary,
        required_capabilities=capabilities,
        safety_control_params=dict(safety),
        anchor_state=anchor_state,
        dependencies=dependencies,
        permissions=permissions,
        side_effect_isolation=isolation,
        budget=budget,
        evaluator=evaluator,
    )


def derive_replay_eligibility(
    controls: Mapping[str, Any],
) -> tuple[str, str, tuple[str, ...]]:
    """Return ``(eligibility, route, reasons)`` for one controlled snapshot."""
    decision = inspect_replay_controls(controls)
    return decision.eligibility, decision.route, decision.reasons


def verify_outcome_alignment(
    *,
    depth: str,
    branch_steps: Sequence[BranchStepRef],
    step_alignments: Sequence[GuidanceStepAlignment],
    replay_block_ids: Sequence[str],
) -> bool:
    """True when step records cover the claimed prefix or the whole branch.

    Imagined steps are taken in branch order. ``prefix`` accepts a non-empty
    leading prefix of those steps, including the whole branch. ``full_branch``
    accepts only the complete list. Each record must name the step in that
    position, use ``equivalent`` or ``functionally_equivalent``, and map
    ``actual_block_id`` to a Block id in ``replay_block_ids``.
    """
    if depth not in _OUTCOME_DEPTHS:
        return False
    imagined = [
        step.imagined_node_id
        for step in branch_steps
        if step.step_kind == STEP_KIND_IMAGINED and step.imagined_node_id
    ]
    if not imagined or not step_alignments:
        return False
    if depth == "full_branch":
        required = imagined
    else:
        if len(step_alignments) > len(imagined):
            return False
        required = imagined[: len(step_alignments)]
    if len(step_alignments) != len(required):
        return False
    blocks = set(replay_block_ids)
    for expected_id, record in zip(required, step_alignments):
        if not isinstance(record, GuidanceStepAlignment):
            return False
        if record.imagined_node_id != expected_id:
            return False
        if record.alignment not in _DIRECT_ALIGNMENTS:
            return False
        if record.actual_block_id is None or record.actual_block_id not in blocks:
            return False
    return True


def judge_direct_admission(
    *,
    alignment: str,
    depth: str,
    intervention_type: str,
    target_boundary: str,
    required_capabilities: Sequence[str],
    safety_control_params: Mapping[str, str],
    expected_intervention_type: str,
    expected_target_boundary: str,
    expected_required_capabilities: Sequence[str],
    expected_safety_control_params: Mapping[str, str],
    equivalence_explanation: str | None,
    route: str,
    first_imagined_node: bool,
    step_alignments: Sequence[GuidanceStepAlignment] = (),
    branch_steps: Sequence[BranchStepRef] = (),
    replay_block_ids: Sequence[str] = (),
) -> DirectAdmissionDecision:
    """Admit replay-direct evidence. A first-action depth never admits an outcome.

    Prefix and full-branch depths admit an outcome only when
    ``verify_outcome_alignment`` accepts the step records. The depth string
    alone does not.
    """
    if route != ROUTE_REPLAY:
        return DirectAdmissionDecision(False, False, "none", ("analogical_route",))
    if not first_imagined_node:
        return DirectAdmissionDecision(False, False, "none", ("not_first_action",))
    submitted_safety = dict(safety_control_params)
    expected_safety = dict(expected_safety_control_params)
    fields_match = (
        intervention_type == expected_intervention_type
        and target_boundary == expected_target_boundary
        and tuple(required_capabilities) == tuple(expected_required_capabilities)
        and submitted_safety == expected_safety
    )
    if not fields_match:
        return DirectAdmissionDecision(False, False, "none", ("audited_field_mismatch",))
    if alignment not in _DIRECT_ALIGNMENTS:
        return DirectAdmissionDecision(False, False, "none", ("alignment_not_direct",))
    if alignment == ALIGNMENT_FUNCTIONALLY_EQUIVALENT and not equivalence_explanation:
        return DirectAdmissionDecision(False, False, "none", ("explanation_required",))
    if depth == "first_action":
        return DirectAdmissionDecision(
            True,
            False,
            EVIDENCE_TIER_REPLAY_DIRECT,
            ("first_action_direct", "first_action_does_not_verify_outcome"),
        )
    if depth in _OUTCOME_DEPTHS:
        if verify_outcome_alignment(
            depth=depth,
            branch_steps=branch_steps,
            step_alignments=step_alignments,
            replay_block_ids=replay_block_ids,
        ):
            return DirectAdmissionDecision(
                True,
                True,
                EVIDENCE_TIER_REPLAY_DIRECT,
                ("first_action_direct", "outcome_aligned"),
            )
        return DirectAdmissionDecision(
            True,
            False,
            EVIDENCE_TIER_REPLAY_DIRECT,
            ("first_action_direct", "outcome_alignment_unverified"),
        )
    return DirectAdmissionDecision(False, False, "none", ("alignment_depth_rejected",))


def _envelope(object_kind: str, body: Mapping[str, Any]) -> dict[str, Any]:
    return protocol_envelope(SCHEMA_VERSION_AIG_NATIVE_V2, object_kind, body)


def _require_hex(value: Any, field: str) -> str:
    text = _require_str(value, field)
    if not is_sha256_hex(text):
        raise ProtocolContractError(f"{field} must be a sha256 hex digest")
    return text


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolContractError(f"{field} must be a bool")
    return value


def replay_eligibility_assessment_document(
    *,
    fork_node_id: str,
    base_revision_id: str,
    anchor_block_id: str,
    evidence_ref: str,
    checked_at: str,
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one judgment. ``eligibility`` is derived, not copied from ``controls``."""
    decision = inspect_replay_controls(controls)
    identity: dict[str, Any] = {
        "anchor_block_id": _require_str(anchor_block_id, "anchor_block_id"),
        "anchor_state": decision.anchor_state,
        "base_revision_id": _require_str(base_revision_id, "base_revision_id"),
        "budget": decision.budget,
        "dependencies": decision.dependencies,
        "eligibility": decision.eligibility,
        "evaluator": decision.evaluator,
        "evidence_ref": _require_str(evidence_ref, "evidence_ref"),
        "fork_node_id": _require_str(fork_node_id, "fork_node_id"),
        "intervention_type": decision.intervention_type,
        "permissions": decision.permissions,
        "reasons": list(decision.reasons),
        "required_capabilities": list(decision.required_capabilities),
        "route": decision.route,
        "safety_control_params": dict(decision.safety_control_params),
        "side_effect_isolation": decision.side_effect_isolation,
        "target_boundary": decision.target_boundary,
    }
    assessment_id = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_REPLAY_ELIGIBILITY_IDENTITY, identity
    )
    body = {
        **identity,
        "assessment_id": assessment_id,
        "checked_at": _require_str(checked_at, "checked_at"),
    }
    return _envelope(OBJECT_KIND_REPLAY_ELIGIBILITY_ASSESSMENT, body)


def replay_request_document(
    assessment: ReplayEligibilityAssessment, *, replay_run_id: str
) -> dict[str, Any]:
    """Build a worker request. ``non_replayable`` cannot be requested."""
    if not isinstance(assessment, ReplayEligibilityAssessment):
        raise ProtocolContractError("replay_eligibility_assessment required")
    if assessment.eligibility not in _REQUEST_ELIGIBILITY:
        raise ProtocolContractError(
            f"eligibility must be one of {sorted(_REQUEST_ELIGIBILITY)}, "
            f"got {assessment.eligibility!r}"
        )
    identity = {
        "anchor_block_id": assessment.anchor_block_id,
        "assessment_id": assessment.assessment_id,
        "base_revision_id": assessment.base_revision_id,
        "eligibility": assessment.eligibility,
        "fork_node_id": assessment.fork_node_id,
        "intervention_type": assessment.intervention_type,
        "replay_run_id": _require_safe_id(replay_run_id, "replay_run_id"),
        "required_capabilities": list(assessment.required_capabilities),
        "safety_control_params": dict(assessment.safety_control_params),
        "target_boundary": assessment.target_boundary,
    }
    body = {
        **identity,
        "request_id": protocol_content_hash(
            SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_REPLAY_REQUEST_IDENTITY, identity
        ),
    }
    return _envelope(OBJECT_KIND_REPLAY_REQUEST, body)


def replay_result_document(
    request: ReplayRequest,
    *,
    status: str,
    observed_block_ids: Sequence[str] = (),
    alignment: Mapping[str, Any] | None = None,
    step_alignments: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a result document and fail closed unless the reader accepts it.

    The Coforge sandbox replay worker is the implementer of a real result.
    P09 and P12 attach that worker. This builder only seals a local document.
    ``step_alignments`` uses the ``GuidanceStepAlignment`` field names.
    """
    if not isinstance(request, ReplayRequest):
        raise ProtocolContractError("replay_request required")
    if alignment is not None and not isinstance(alignment, Mapping):
        raise ProtocolContractError("first_action_alignment must be a dict or null")
    body: dict[str, Any] = {
        "base_revision_id": request.base_revision_id,
        "first_action_alignment": None if alignment is None else dict(alignment),
        "fork_node_id": request.fork_node_id,
        "observed_block_ids": list(observed_block_ids),
        "replay_run_id": request.replay_run_id,
        "request_id": request.request_id,
        "status": status,
        "step_alignments": _copy_step_alignment_mappings(step_alignments),
    }
    document = _envelope(OBJECT_KIND_REPLAY_RESULT, body)
    read_replay_result(document)
    return document


def replay_aig_document(
    *,
    base_revision_id: str,
    replay_run_id: str,
    source_fork_id: str,
    revision: AigNativeViewRevision,
) -> dict[str, Any]:
    """Seal a frozen native revision with replay provenance."""
    if not isinstance(revision, AigNativeViewRevision):
        raise ProtocolContractError("replay AIG requires an AigNativeViewRevision")
    revision_id = replay_aig_revision_id(
        base_revision_id=base_revision_id,
        replay_run_id=replay_run_id,
        source_fork_id=source_fork_id,
    )
    if revision.revision_id != revision_id:
        raise ProtocolContractError("revision_id does not match replay provenance")
    body = {
        "base_revision_id": _require_str(base_revision_id, "base_revision_id"),
        "native_revision": revision.protocol_document(),
        "provenance": REPLAY_PROVENANCE,
        "replay_run_id": _require_safe_id(replay_run_id, "replay_run_id"),
        "revision_id": revision_id,
        "source_fork_id": _require_str(source_fork_id, "source_fork_id"),
    }
    document = _envelope(OBJECT_KIND_REPLAY_AIG, body)
    read_replay_aig(document)
    return document


def direct_evidence_admission_document(
    *,
    fork_node_id: str,
    base_revision_id: str,
    replay_run_id: str,
    replay_revision_id: str,
    alignment: FirstActionAlignment,
    expected_intervention_type: str,
    expected_target_boundary: str,
    expected_required_capabilities: Sequence[str],
    expected_safety_control_params: Mapping[str, str],
    first_imagined_node: bool,
    route: str,
    step_alignments: Sequence[GuidanceStepAlignment] = (),
    branch_steps: Sequence[BranchStepRef] = (),
    replay_block_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Record the server decision and the Reader's equivalence explanation.

    Outcome flags are recomputed from ``step_alignments``, the branch's
    imagined steps, and ``replay_block_ids``. A depth claim is not enough.
    """
    if not isinstance(alignment, FirstActionAlignment):
        raise ProtocolContractError("first_action_alignment required")
    sealed_steps = _seal_step_alignments(step_alignments)
    imagined_ids = _imagined_ids_from_branch(branch_steps)
    sealed_blocks = _canonical_block_ids(replay_block_ids)
    decision = judge_direct_admission(
        alignment=alignment.alignment,
        depth=alignment.depth,
        intervention_type=alignment.intervention_type,
        target_boundary=alignment.target_boundary,
        required_capabilities=alignment.required_capabilities,
        safety_control_params=alignment.safety_control_params,
        expected_intervention_type=expected_intervention_type,
        expected_target_boundary=expected_target_boundary,
        expected_required_capabilities=expected_required_capabilities,
        expected_safety_control_params=expected_safety_control_params,
        equivalence_explanation=alignment.equivalence_explanation,
        route=route,
        first_imagined_node=first_imagined_node,
        step_alignments=sealed_steps,
        branch_steps=tuple(
            BranchStepRef(step_kind=STEP_KIND_IMAGINED, imagined_node_id=node_id)
            for node_id in imagined_ids
        ),
        replay_block_ids=tuple(sealed_blocks),
    )
    identity: dict[str, Any] = {
        "actual_block_id": alignment.actual_block_id,
        "alignment": alignment.alignment,
        "alignment_content_hash": alignment.content_hash,
        "base_revision_id": _require_str(base_revision_id, "base_revision_id"),
        "branch_imagined_ids": imagined_ids,
        "depth": alignment.depth,
        "equivalence_explanation": alignment.equivalence_explanation,
        "expected_intervention_type": _require_str(
            expected_intervention_type, "expected_intervention_type"
        ),
        "expected_required_capabilities": list(expected_required_capabilities),
        "expected_safety_control_params": dict(expected_safety_control_params),
        "expected_target_boundary": _require_str(
            expected_target_boundary, "expected_target_boundary"
        ),
        "first_imagined_node": bool(first_imagined_node),
        "fork_node_id": _require_str(fork_node_id, "fork_node_id"),
        "imagined_node_id": alignment.imagined_node_id,
        "intervention_type": alignment.intervention_type,
        "replay_block_ids": sealed_blocks,
        "replay_revision_id": _require_str(replay_revision_id, "replay_revision_id"),
        "replay_run_id": _require_safe_id(replay_run_id, "replay_run_id"),
        "required_capabilities": list(alignment.required_capabilities),
        "route": route,
        "safety_control_params": dict(alignment.safety_control_params),
        "step_alignments": _public_step_alignments(sealed_steps),
        "target_boundary": alignment.target_boundary,
    }
    body = {
        **identity,
        "admission_id": protocol_content_hash(
            SCHEMA_VERSION_AIG_NATIVE_V2,
            OBJECT_KIND_DIRECT_EVIDENCE_ADMISSION_IDENTITY,
            identity,
        ),
        "evidence_tier": decision.evidence_tier,
        "first_action_direct": decision.first_action_direct,
        "outcome_direct": decision.outcome_direct,
        "reasons": list(decision.reasons),
    }
    document = _envelope(OBJECT_KIND_DIRECT_EVIDENCE_ADMISSION, body)
    read_direct_evidence_admission(document)
    return document


def read_replay_eligibility_assessment(
    payload: Mapping[str, Any],
) -> ReplayEligibilityAssessment:
    """Parse one judgment. The verdict must match the controlled fields."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_REPLAY_ELIGIBILITY_ASSESSMENT,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    decision = inspect_replay_controls(
        {key: body[key] for key in _REPLAY_CONTROL_KEYS}
    )
    if body.get("eligibility") != decision.eligibility or body.get("route") != decision.route:
        raise ProtocolContractError("eligibility does not match controlled fields")
    if body.get("reasons") != list(decision.reasons):
        raise ProtocolContractError("reasons do not match controlled fields")
    identity = {key: body[key] for key in body if key not in {"assessment_id", "checked_at"}}
    assessment_id = _require_hex(body.get("assessment_id"), "assessment_id")
    expected = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_REPLAY_ELIGIBILITY_IDENTITY, identity
    )
    if assessment_id != expected:
        raise ProtocolContractError(
            f"assessment_id {assessment_id!r} does not match controlled fields"
        )
    return ReplayEligibilityAssessment(
        schema_version=version,
        object_kind=OBJECT_KIND_REPLAY_ELIGIBILITY_ASSESSMENT,
        content_hash=digest,
        assessment_id=assessment_id,
        fork_node_id=_require_str(body.get("fork_node_id"), "fork_node_id"),
        base_revision_id=_require_str(body.get("base_revision_id"), "base_revision_id"),
        anchor_block_id=_require_str(body.get("anchor_block_id"), "anchor_block_id"),
        eligibility=decision.eligibility,
        route=decision.route,
        reasons=decision.reasons,
        evidence_ref=_require_str(body.get("evidence_ref"), "evidence_ref"),
        checked_at=_require_str(body.get("checked_at"), "checked_at"),
        intervention_type=decision.intervention_type,
        target_boundary=decision.target_boundary,
        required_capabilities=decision.required_capabilities,
        safety_control_params=dict(decision.safety_control_params),
        anchor_state=decision.anchor_state,
        dependencies=decision.dependencies,
        permissions=decision.permissions,
        side_effect_isolation=decision.side_effect_isolation,
        budget=decision.budget,
        evaluator=decision.evaluator,
    )


def read_replay_request(payload: Mapping[str, Any]) -> ReplayRequest:
    """Parse a replay request. The Coforge sandbox replay worker executes it.

    This reader does not start that worker. P09 and P12 attach the real worker.
    ``non_replayable`` is rejected here so a request cannot smuggle that route.
    """
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_REPLAY_REQUEST,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    eligibility = _require_choice(body.get("eligibility"), "eligibility", _REQUEST_ELIGIBILITY)
    identity = {key: body[key] for key in body if key != "request_id"}
    request_id = _require_hex(body.get("request_id"), "request_id")
    expected = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_REPLAY_REQUEST_IDENTITY, identity
    )
    if request_id != expected:
        raise ProtocolContractError("request_id does not match the request body")
    return ReplayRequest(
        schema_version=version,
        object_kind=OBJECT_KIND_REPLAY_REQUEST,
        content_hash=digest,
        request_id=request_id,
        assessment_id=_require_hex(body.get("assessment_id"), "assessment_id"),
        fork_node_id=_require_str(body.get("fork_node_id"), "fork_node_id"),
        base_revision_id=_require_str(body.get("base_revision_id"), "base_revision_id"),
        anchor_block_id=_require_str(body.get("anchor_block_id"), "anchor_block_id"),
        replay_run_id=_require_safe_id(body.get("replay_run_id"), "replay_run_id"),
        intervention_type=_require_str(body.get("intervention_type"), "intervention_type"),
        target_boundary=_require_str(body.get("target_boundary"), "target_boundary"),
        required_capabilities=_parse_str_list(
            body.get("required_capabilities"), "required_capabilities"
        ),
        safety_control_params=_parse_str_map(
            body.get("safety_control_params"), "safety_control_params"
        ),
        eligibility=eligibility,
    )


def read_replay_result(payload: Mapping[str, Any]) -> ReplayResult:
    """Parse a replay result. A completed result carries the alignment observation.

    The Coforge sandbox replay worker is the implementer. This reader does not
    call it. P09 and P12 attach the real worker.
    """
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_REPLAY_RESULT,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    status = _require_choice(body.get("status"), "status", _REPLAY_RESULT_STATUSES)
    raw_ids = body.get("observed_block_ids")
    ids = _parse_str_list(raw_ids, "observed_block_ids")
    if len(ids) != len(set(ids)):
        raise ProtocolContractError("observed_block_ids must be unique")
    raw_alignment = body.get("first_action_alignment")
    step_alignments = _parse_step_alignments(
        body.get("step_alignments"), "step_alignments", allow_empty=True
    )
    _reject_duplicate_step_ids(step_alignments, "step_alignments")
    if status == "completed":
        if not ids:
            raise ProtocolContractError("observed_block_ids must be a non-empty list")
        if not isinstance(raw_alignment, dict):
            raise ProtocolContractError(
                "completed replay result requires first_action_alignment"
            )
        alignment = read_first_action_alignment(raw_alignment)
        if alignment.actual_block_id is not None and alignment.actual_block_id not in ids:
            raise ProtocolContractError("actual_block_id is not an observed block")
        for index, step in enumerate(step_alignments):
            if step.actual_block_id is not None and step.actual_block_id not in ids:
                raise ProtocolContractError(
                    f"step_alignments[{index}].actual_block_id is not an observed block"
                )
    else:
        if ids:
            raise ProtocolContractError("observed_block_ids must be empty")
        if raw_alignment is not None:
            raise ProtocolContractError(
                "unfinished replay result requires first_action_alignment to be null"
            )
        if step_alignments:
            raise ProtocolContractError(
                "unfinished replay result requires step_alignments to be empty"
            )
        alignment = None
    return ReplayResult(
        schema_version=version,
        object_kind=OBJECT_KIND_REPLAY_RESULT,
        content_hash=digest,
        request_id=_require_hex(body.get("request_id"), "request_id"),
        replay_run_id=_require_safe_id(body.get("replay_run_id"), "replay_run_id"),
        fork_node_id=_require_str(body.get("fork_node_id"), "fork_node_id"),
        base_revision_id=_require_str(body.get("base_revision_id"), "base_revision_id"),
        status=status,
        observed_block_ids=tuple(ids),
        first_action_alignment=alignment,
        step_alignments=step_alignments,
    )


def read_replay_aig(payload: Mapping[str, Any]) -> ReplayAigDocument:
    """Parse a Replay AIG. Provenance is ``replay`` and the revision id is bound to it."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_REPLAY_AIG,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    provenance = body.get("provenance")
    if provenance != REPLAY_PROVENANCE:
        raise ProtocolContractError(
            f"provenance must be {REPLAY_PROVENANCE!r}, got {provenance!r}"
        )
    base_revision_id = _require_str(body.get("base_revision_id"), "base_revision_id")
    replay_run_id = _require_safe_id(body.get("replay_run_id"), "replay_run_id")
    source_fork_id = _require_str(body.get("source_fork_id"), "source_fork_id")
    revision_id = _require_str(body.get("revision_id"), "revision_id")
    expected_id = replay_aig_revision_id(
        base_revision_id=base_revision_id,
        replay_run_id=replay_run_id,
        source_fork_id=source_fork_id,
    )
    if revision_id != expected_id:
        raise ProtocolContractError("revision_id does not match replay provenance")
    if is_replay_revision_id(base_revision_id):
        raise ProtocolContractError("base_revision_id must be a production revision")
    native = body.get("native_revision")
    if not isinstance(native, Mapping):
        raise ProtocolContractError("native_revision must be a dict")
    parsed = read_native_edge_revision(native)
    revision = parsed.revision
    if revision.revision_id != revision_id:
        raise ProtocolContractError("native revision_id does not match replay provenance")
    if revision.session_id != f"{REPLAY_SESSION_PREFIX}{replay_run_id}":
        raise ProtocolContractError("replay session_id does not match replay_run_id")
    if revision.task_run_id != f"{REPLAY_TASK_RUN_PREFIX}{replay_run_id}":
        raise ProtocolContractError("replay task_run_id does not match replay_run_id")
    if revision.view_id != f"{REPLAY_VIEW_PREFIX}{replay_run_id}":
        raise ProtocolContractError("replay view_id does not match replay_run_id")
    return ReplayAigDocument(
        schema_version=version,
        object_kind=OBJECT_KIND_REPLAY_AIG,
        content_hash=digest,
        provenance=REPLAY_PROVENANCE,
        base_revision_id=base_revision_id,
        replay_run_id=replay_run_id,
        source_fork_id=source_fork_id,
        revision_id=revision_id,
        revision=revision,
    )


_ADMISSION_DERIVED_KEYS = frozenset(
    {
        "admission_id",
        "evidence_tier",
        "first_action_direct",
        "outcome_direct",
        "reasons",
    }
)


def read_direct_evidence_admission(payload: Mapping[str, Any]) -> DirectEvidenceAdmission:
    """Parse an admission. Flags that disagree with the server rule are rejected."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_DIRECT_EVIDENCE_ADMISSION,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    alignment = _require_choice(body.get("alignment"), "alignment", _ALIGNMENTS)
    observed = body.get("actual_block_id")
    if alignment == ALIGNMENT_NOT_EXECUTED:
        if observed is not None:
            raise ProtocolContractError(
                "not_executed alignment requires actual_block_id to be null"
            )
        actual_block_id = None
    else:
        actual_block_id = _require_str(observed, "actual_block_id")
    explanation = body.get("equivalence_explanation")
    if explanation is not None:
        explanation = _require_str(explanation, "equivalence_explanation")
    identity = {key: body[key] for key in body if key not in _ADMISSION_DERIVED_KEYS}
    admission_id = _require_hex(body.get("admission_id"), "admission_id")
    expected_id = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        OBJECT_KIND_DIRECT_EVIDENCE_ADMISSION_IDENTITY,
        identity,
    )
    if admission_id != expected_id:
        raise ProtocolContractError("admission_id does not match the admission body")
    route = _require_choice(body.get("route"), "route", _REPLAY_ROUTES)
    first_node = _require_bool(body.get("first_imagined_node"), "first_imagined_node")
    step_alignments = _parse_step_alignments(
        body.get("step_alignments"), "step_alignments", allow_empty=True
    )
    _reject_duplicate_step_ids(step_alignments, "step_alignments")
    branch_imagined_ids = _parse_str_list(
        body.get("branch_imagined_ids"), "branch_imagined_ids"
    )
    if len(branch_imagined_ids) != len(set(branch_imagined_ids)):
        raise ProtocolContractError("branch_imagined_ids must be unique")
    replay_block_ids = _parse_str_list(body.get("replay_block_ids"), "replay_block_ids")
    if list(replay_block_ids) != sorted(set(replay_block_ids)):
        raise ProtocolContractError("replay_block_ids must be sorted unique ids")
    depth = _require_choice(body.get("depth"), "depth", _ALIGNMENT_DEPTHS)
    decision = judge_direct_admission(
        alignment=alignment,
        depth=depth,
        intervention_type=_require_str(body.get("intervention_type"), "intervention_type"),
        target_boundary=_require_str(body.get("target_boundary"), "target_boundary"),
        required_capabilities=_parse_str_list(
            body.get("required_capabilities"), "required_capabilities"
        ),
        safety_control_params=_parse_str_map(
            body.get("safety_control_params"), "safety_control_params"
        ),
        expected_intervention_type=_require_str(
            body.get("expected_intervention_type"), "expected_intervention_type"
        ),
        expected_target_boundary=_require_str(
            body.get("expected_target_boundary"), "expected_target_boundary"
        ),
        expected_required_capabilities=_parse_str_list(
            body.get("expected_required_capabilities"), "expected_required_capabilities"
        ),
        expected_safety_control_params=_parse_str_map(
            body.get("expected_safety_control_params"), "expected_safety_control_params"
        ),
        equivalence_explanation=explanation,
        route=route,
        first_imagined_node=first_node,
        step_alignments=step_alignments,
        branch_steps=tuple(
            BranchStepRef(step_kind=STEP_KIND_IMAGINED, imagined_node_id=node_id)
            for node_id in branch_imagined_ids
        ),
        replay_block_ids=replay_block_ids,
    )
    if (
        body.get("first_action_direct") is not decision.first_action_direct
        or body.get("outcome_direct") is not decision.outcome_direct
        or body.get("evidence_tier") != decision.evidence_tier
        or body.get("reasons") != list(decision.reasons)
    ):
        raise ProtocolContractError("admission flags do not match the server rule")
    replay_revision_id = _require_str(body.get("replay_revision_id"), "replay_revision_id")
    if not is_replay_revision_id(replay_revision_id):
        raise ProtocolContractError("replay_revision_id must carry replay provenance")
    return DirectEvidenceAdmission(
        schema_version=version,
        object_kind=OBJECT_KIND_DIRECT_EVIDENCE_ADMISSION,
        content_hash=digest,
        admission_id=admission_id,
        fork_node_id=_require_str(body.get("fork_node_id"), "fork_node_id"),
        base_revision_id=_require_str(body.get("base_revision_id"), "base_revision_id"),
        replay_run_id=_require_safe_id(body.get("replay_run_id"), "replay_run_id"),
        replay_revision_id=replay_revision_id,
        imagined_node_id=_require_str(body.get("imagined_node_id"), "imagined_node_id"),
        actual_block_id=actual_block_id,
        alignment=alignment,
        depth=_require_choice(body.get("depth"), "depth", _ALIGNMENT_DEPTHS),
        intervention_type=_require_str(body.get("intervention_type"), "intervention_type"),
        target_boundary=_require_str(body.get("target_boundary"), "target_boundary"),
        required_capabilities=_parse_str_list(
            body.get("required_capabilities"), "required_capabilities"
        ),
        safety_control_params=_parse_str_map(
            body.get("safety_control_params"), "safety_control_params"
        ),
        expected_intervention_type=_require_str(
            body.get("expected_intervention_type"), "expected_intervention_type"
        ),
        expected_target_boundary=_require_str(
            body.get("expected_target_boundary"), "expected_target_boundary"
        ),
        expected_required_capabilities=_parse_str_list(
            body.get("expected_required_capabilities"), "expected_required_capabilities"
        ),
        expected_safety_control_params=_parse_str_map(
            body.get("expected_safety_control_params"), "expected_safety_control_params"
        ),
        equivalence_explanation=explanation,
        first_imagined_node=first_node,
        route=route,
        alignment_content_hash=_require_hex(
            body.get("alignment_content_hash"), "alignment_content_hash"
        ),
        step_alignments=step_alignments,
        branch_imagined_ids=branch_imagined_ids,
        replay_block_ids=replay_block_ids,
        first_action_direct=decision.first_action_direct,
        outcome_direct=decision.outcome_direct,
        evidence_tier=_require_choice(body.get("evidence_tier"), "evidence_tier", _ADMISSION_TIERS),
        reasons=decision.reasons,
    )


_GUIDANCE_STEP_KEYS = frozenset(
    {"evidence_refs", "imagined_node_id", "statement", "step_id"}
)
_CANDIDATE_KEYS = frozenset(
    {"acl_labels", "agent_id", "available", "capabilities", "roles", "task_member"}
)
_OUTCOME_EVIDENCE_KEYS = frozenset(
    {"binding_id", "evidence_ref", "signal", "signal_kind"}
)
_REJECTION_REASONS = frozenset({GUIDANCE_REJECTION_REASON})


def _parse_applicability(value: Any) -> ApplicabilityCheck:
    if not isinstance(value, dict):
        raise ProtocolContractError("applicability must be a dict")
    _require_exact_keys(value, _APPLICABILITY_KEYS, "applicability")
    try:
        return ApplicabilityCheck(
            task_decision_type=value["task_decision_type"],
            anchor_state=value["anchor_state"],
            failure_mode=value["failure_mode"],
            action_skill_role=value["action_skill_role"],
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolContractError(str(exc)) from exc


def _parse_unique_strs(value: Any, field: str, *, allow_empty: bool) -> tuple[str, ...]:
    items = _parse_str_list(value, field)
    if not allow_empty and not items:
        raise ProtocolContractError(f"{field} must be a non-empty list")
    if len(items) != len(set(items)):
        raise ProtocolContractError(f"duplicate {field}")
    return items


def _parse_exact_labels(value: Any, field: str) -> tuple[str, ...]:
    labels = _parse_unique_strs(value, field, allow_empty=True)
    for label in labels:
        if label.strip() != label:
            raise ProtocolContractError(f"{field} must be exact labels")
    return labels


def _parse_publisher_labels(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProtocolContractError("publisher_labels must be a list")
    labels = _parse_unique_strs(list(value), "publisher_labels", allow_empty=False)
    for label in labels:
        if label.strip() != label:
            raise ProtocolContractError("publisher_labels must be exact labels")
    if list(labels) != sorted(labels):
        raise ProtocolContractError("publisher_labels must be sorted")
    return labels


def _require_published_at(value: Any) -> str:
    text = _require_str(value, "published_at")
    if _UTC_MILLIS.fullmatch(text) is None:
        raise ProtocolContractError("published_at must be a UTC millisecond timestamp")
    return text


def _parse_target_roles(value: Any, field: str) -> tuple[str, ...]:
    roles = _parse_unique_strs(value, field, allow_empty=False)
    for role in roles:
        _require_choice(role, field, TARGET_AGENT_ROLES)
    return roles


def _parse_guidance_steps(value: Any) -> tuple[GuidanceActionStep, ...]:
    if not isinstance(value, list) or not value:
        raise ProtocolContractError("steps must be a non-empty list")
    steps: list[GuidanceActionStep] = []
    for index, item in enumerate(value):
        label = f"steps[{index}]"
        if not isinstance(item, dict):
            raise ProtocolContractError(f"{label} must be a dict")
        _require_exact_keys(item, _GUIDANCE_STEP_KEYS, label)
        refs = item.get("evidence_refs")
        if not isinstance(refs, list) or not refs:
            raise ProtocolContractError(f"{label}.evidence_refs must be a non-empty list")
        steps.append(
            GuidanceActionStep(
                step_id=_require_safe_id(item.get("step_id"), f"{label}.step_id"),
                imagined_node_id=_require_safe_id(
                    item.get("imagined_node_id"), f"{label}.imagined_node_id"
                ),
                statement=_require_str(item.get("statement"), f"{label}.statement"),
                evidence_refs=_parse_evidence_refs(refs),
            )
        )
    identities = [step.step_id for step in steps]
    if len(identities) != len(set(identities)):
        raise ProtocolContractError("duplicate step_id")
    return tuple(steps)


def _step_record(step: GuidanceActionStep) -> dict[str, Any]:
    return {
        "evidence_refs": [ref.to_dict() for ref in step.evidence_refs],
        "imagined_node_id": step.imagined_node_id,
        "statement": step.statement,
        "step_id": step.step_id,
    }


def _union_evidence(steps: Sequence[GuidanceActionStep]) -> tuple[EvidenceRef, ...]:
    found: list[EvidenceRef] = []
    seen: set[str] = set()
    for step in steps:
        for ref in step.evidence_refs:
            key = canonical_dumps(ref.to_dict())
            if key in seen:
                continue
            seen.add(key)
            found.append(ref)
    return tuple(found)


def _parse_outcome_evidence(value: Any) -> tuple[OutcomeEvidenceSignal, ...]:
    from openviking.session.outcome_contract import OUTCOME_SIGNAL_KINDS, OUTCOME_SIGNALS

    if not isinstance(value, list) or not value:
        raise ProtocolContractError("outcome_evidence must be a non-empty list")
    items: list[OutcomeEvidenceSignal] = []
    for index, item in enumerate(value):
        label = f"outcome_evidence[{index}]"
        if not isinstance(item, dict):
            raise ProtocolContractError(f"{label} must be a dict")
        _require_exact_keys(item, _OUTCOME_EVIDENCE_KEYS, label)
        items.append(
            OutcomeEvidenceSignal(
                binding_id=_require_str(item.get("binding_id"), f"{label}.binding_id"),
                signal_kind=_require_choice(
                    item.get("signal_kind"), f"{label}.signal_kind", OUTCOME_SIGNAL_KINDS
                ),
                signal=_require_choice(item.get("signal"), f"{label}.signal", OUTCOME_SIGNALS),
                evidence_ref=_require_str(item.get("evidence_ref"), f"{label}.evidence_ref"),
            )
        )
    keys = [(item.binding_id, item.signal_kind, item.signal, item.evidence_ref) for item in items]
    if keys != sorted(keys):
        raise ProtocolContractError("outcome_evidence must be sorted")
    if len(keys) != len(set(keys)):
        raise ProtocolContractError("duplicate outcome_evidence")
    return tuple(items)


def _outcome_record(item: OutcomeEvidenceSignal) -> dict[str, str]:
    return {
        "binding_id": item.binding_id,
        "evidence_ref": item.evidence_ref,
        "signal": item.signal,
        "signal_kind": item.signal_kind,
    }


def _require_confirmed_failure(policy_version: str, decision: str) -> None:
    from openviking.session.outcome_contract import (
        DECISION_CONFIRMED_FAILURE,
        OUTCOME_EVIDENCE_POLICY_VERSION,
    )

    if policy_version != OUTCOME_EVIDENCE_POLICY_VERSION or decision != DECISION_CONFIRMED_FAILURE:
        raise ProtocolContractError(
            f"decision must be {DECISION_CONFIRMED_FAILURE!r} "
            f"under {OUTCOME_EVIDENCE_POLICY_VERSION}"
        )


def _parse_candidates(value: Any) -> tuple[AgentCandidate, ...]:
    if not isinstance(value, list):
        raise ProtocolContractError("candidates must be a list")
    rows: list[AgentCandidate] = []
    for index, item in enumerate(value):
        label = f"candidates[{index}]"
        if not isinstance(item, dict):
            raise ProtocolContractError(f"{label} must be a dict")
        _require_exact_keys(item, _CANDIDATE_KEYS, label)
        rows.append(
            AgentCandidate(
                agent_id=_require_safe_id(item.get("agent_id"), f"{label}.agent_id"),
                roles=_parse_target_roles(item.get("roles"), f"{label}.roles"),
                capabilities=_parse_unique_strs(
                    item.get("capabilities"), f"{label}.capabilities", allow_empty=True
                ),
                acl_labels=_parse_exact_labels(item.get("acl_labels"), f"{label}.acl_labels"),
                task_member=_require_bool(item.get("task_member"), f"{label}.task_member"),
                available=_require_bool(item.get("available"), f"{label}.available"),
            )
        )
    identities = [row.agent_id for row in rows]
    if identities != sorted(identities):
        raise ProtocolContractError("candidates must be sorted by agent_id")
    if len(identities) != len(set(identities)):
        raise ProtocolContractError("duplicate agent_id")
    return tuple(rows)


def guidance_approval_id(
    proposal_content_hash: str, candidate_catalog_hash: str, selected_agent_id: str
) -> str:
    """Digest of the versioned selection inputs. Callers do not supply this id."""
    body = {
        "candidate_catalog_hash": _require_hex(
            candidate_catalog_hash, "candidate_catalog_hash"
        ),
        "proposal_content_hash": _require_hex(proposal_content_hash, "proposal_content_hash"),
        "selected_agent_id": _require_safe_id(selected_agent_id, "selected_agent_id"),
    }
    return protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_GUIDANCE_APPROVAL_IDENTITY, body
    )


def _projected_guidance_body(
    *,
    guidance_id: str,
    source_fork_id: str,
    base_revision_id: str,
    target_role: str,
    capability_limits: Sequence[str],
    report_hash: str,
    applicability: ApplicabilityCheck,
    risk: str,
    evidence_refs: Sequence[EvidenceRef],
) -> dict[str, Any]:
    return {
        "applicability": applicability.to_dict(),
        "base_revision_id": base_revision_id,
        "capability_limits": list(capability_limits),
        "evidence_refs": [ref.to_dict() for ref in evidence_refs],
        "guidance_id": guidance_id,
        "report_hash": report_hash,
        "risk": risk,
        "source_fork_id": source_fork_id,
        "target_role": target_role,
    }


def _require_step_evidence(
    steps: Sequence[GuidanceActionStep], evidence_refs: Sequence[EvidenceRef]
) -> tuple[EvidenceRef, ...]:
    evidence = _union_evidence(steps)
    if evidence != tuple(evidence_refs):
        raise ProtocolContractError("evidence_refs must be the evidenced steps")
    return evidence


def read_guided_counterfactual_branch(payload: Mapping[str, Any]) -> GuidedCounterfactualBranch:
    """Parse a Reader proposal. It does not select an agent."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_GUIDED_COUNTERFACTUAL_BRANCH,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    steps = _parse_guidance_steps(body.get("steps"))
    if any(not step.evidence_refs for step in steps):
        raise ProtocolContractError("steps evidence_refs must be a non-empty list")
    return GuidedCounterfactualBranch(
        schema_version=version,
        object_kind=OBJECT_KIND_GUIDED_COUNTERFACTUAL_BRANCH,
        content_hash=digest,
        author_role=_require_reader_author(body.get("author_role"), "GuidedCounterfactualBranch"),
        report_hash=_require_hex(body.get("report_hash"), "report_hash"),
        task_id=_require_safe_id(body.get("task_id"), "task_id"),
        source_fork_id=_require_safe_id(body.get("source_fork_id"), "source_fork_id"),
        base_revision_id=_require_safe_id(body.get("base_revision_id"), "base_revision_id"),
        retrieval_id=_require_hex(body.get("retrieval_id"), "retrieval_id"),
        target_role=_require_choice(body.get("target_role"), "target_role", TARGET_AGENT_ROLES),
        required_capabilities=_parse_unique_strs(
            body.get("required_capabilities"), "required_capabilities", allow_empty=True
        ),
        applicability=_parse_applicability(body.get("applicability")),
        risk=_require_str(body.get("risk"), "risk"),
        steps=steps,
    )


def read_agent_candidate_catalog(payload: Mapping[str, Any]) -> AgentCandidateCatalog:
    """Parse one candidate-directory version. It does not select an agent."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_AGENT_CANDIDATE_CATALOG,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    return AgentCandidateCatalog(
        schema_version=version,
        object_kind=OBJECT_KIND_AGENT_CANDIDATE_CATALOG,
        content_hash=digest,
        task_id=_require_safe_id(body.get("task_id"), "task_id"),
        candidates=_parse_candidates(body.get("candidates")),
    )


def guidance_approval_document(
    proposal: GuidedCounterfactualBranch,
    *,
    candidate_catalog_hash: str,
    selected_agent_id: str,
    policy_version: str,
    decision: str,
    reason: str,
    outcome_evidence: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Freeze approval fields from the proposal. There is no step argument."""
    if not isinstance(proposal, GuidedCounterfactualBranch):
        raise ProtocolContractError("guided_counterfactual_branch required")
    _require_confirmed_failure(policy_version, decision)
    if isinstance(outcome_evidence, (str, bytes)) or not isinstance(outcome_evidence, Sequence):
        raise ProtocolContractError("outcome_evidence must be a non-empty list")
    evidence = _union_evidence(proposal.steps)
    if not evidence:
        raise ProtocolContractError("steps evidence_refs must be a non-empty list")
    agent_id = _require_safe_id(selected_agent_id, "selected_agent_id")
    catalog_hash = _require_hex(candidate_catalog_hash, "candidate_catalog_hash")
    guidance_id = guidance_approval_id(proposal.content_hash, catalog_hash, agent_id)
    projected = _projected_guidance_body(
        guidance_id=guidance_id,
        source_fork_id=proposal.source_fork_id,
        base_revision_id=proposal.base_revision_id,
        target_role=proposal.target_role,
        capability_limits=proposal.required_capabilities,
        report_hash=proposal.report_hash,
        applicability=proposal.applicability,
        risk=proposal.risk,
        evidence_refs=evidence,
    )
    signals = _parse_outcome_evidence(list(outcome_evidence))
    body = {
        **projected,
        "candidate_catalog_hash": catalog_hash,
        "decision": decision,
        "guidance_content_hash": protocol_content_hash(
            SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_GUIDANCE_DOCUMENT, projected
        ),
        "outcome_evidence": [_outcome_record(item) for item in signals],
        "policy_version": policy_version,
        "proposal_content_hash": proposal.content_hash,
        "reason": _require_str(reason, "reason"),
        "retrieval_id": proposal.retrieval_id,
        "selected_agent_id": agent_id,
        "steps": [_step_record(step) for step in proposal.steps],
    }
    return _envelope(OBJECT_KIND_GUIDANCE_APPROVAL, body)


def read_guidance_approval(payload: Mapping[str, Any]) -> GuidanceApproval:
    """Parse a frozen approval. It does not dispatch the selected agent."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_GUIDANCE_APPROVAL,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    _require_confirmed_failure(
        _require_str(body.get("policy_version"), "policy_version"),
        _require_str(body.get("decision"), "decision"),
    )
    steps = _parse_guidance_steps(body.get("steps"))
    evidence_refs = _parse_evidence_refs(body.get("evidence_refs"))
    _require_step_evidence(steps, evidence_refs)
    applicability = _parse_applicability(body.get("applicability"))
    capability_limits = _parse_unique_strs(
        body.get("capability_limits"), "capability_limits", allow_empty=True
    )
    guidance_id = _require_str(body.get("guidance_id"), "guidance_id")
    proposal_hash = _require_hex(body.get("proposal_content_hash"), "proposal_content_hash")
    catalog_hash = _require_hex(body.get("candidate_catalog_hash"), "candidate_catalog_hash")
    selected_agent_id = _require_safe_id(body.get("selected_agent_id"), "selected_agent_id")
    source_fork_id = _require_safe_id(body.get("source_fork_id"), "source_fork_id")
    base_revision_id = _require_safe_id(body.get("base_revision_id"), "base_revision_id")
    target_role = _require_choice(body.get("target_role"), "target_role", TARGET_AGENT_ROLES)
    report_hash = _require_hex(body.get("report_hash"), "report_hash")
    risk = _require_str(body.get("risk"), "risk")
    expected_id = guidance_approval_id(proposal_hash, catalog_hash, selected_agent_id)
    if guidance_id != expected_id:
        raise ProtocolContractError("guidance_id does not match approval identity")
    projected = _projected_guidance_body(
        guidance_id=guidance_id,
        source_fork_id=source_fork_id,
        base_revision_id=base_revision_id,
        target_role=target_role,
        capability_limits=capability_limits,
        report_hash=report_hash,
        applicability=applicability,
        risk=risk,
        evidence_refs=evidence_refs,
    )
    guidance_content_hash = _require_hex(
        body.get("guidance_content_hash"), "guidance_content_hash"
    )
    expected_guidance = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_GUIDANCE_DOCUMENT, projected
    )
    if guidance_content_hash != expected_guidance:
        raise ProtocolContractError(
            "guidance_content_hash does not match the guidance projection"
        )
    return GuidanceApproval(
        schema_version=version,
        object_kind=OBJECT_KIND_GUIDANCE_APPROVAL,
        content_hash=digest,
        guidance_id=guidance_id,
        proposal_content_hash=proposal_hash,
        retrieval_id=_require_hex(body.get("retrieval_id"), "retrieval_id"),
        candidate_catalog_hash=catalog_hash,
        selected_agent_id=selected_agent_id,
        source_fork_id=source_fork_id,
        base_revision_id=base_revision_id,
        target_role=target_role,
        capability_limits=capability_limits,
        report_hash=report_hash,
        applicability=applicability,
        risk=risk,
        evidence_refs=evidence_refs,
        steps=steps,
        policy_version=_require_str(body.get("policy_version"), "policy_version"),
        decision=_require_str(body.get("decision"), "decision"),
        reason=_require_str(body.get("reason"), "reason"),
        outcome_evidence=_parse_outcome_evidence(body.get("outcome_evidence")),
        guidance_content_hash=guidance_content_hash,
    )


def guidance_document_from_approval(approval: GuidanceApproval) -> dict[str, Any]:
    """Project the P00 GuidanceDocument. The selected agent stays on the approval."""
    if not isinstance(approval, GuidanceApproval):
        raise ProtocolContractError("guidance_approval required")
    projected = _projected_guidance_body(
        guidance_id=approval.guidance_id,
        source_fork_id=approval.source_fork_id,
        base_revision_id=approval.base_revision_id,
        target_role=approval.target_role,
        capability_limits=approval.capability_limits,
        report_hash=approval.report_hash,
        applicability=approval.applicability,
        risk=approval.risk,
        evidence_refs=approval.evidence_refs,
    )
    document = _envelope(OBJECT_KIND_GUIDANCE_DOCUMENT, projected)
    if document["content_hash"] != approval.guidance_content_hash:
        raise ProtocolContractError(
            "guidance_content_hash does not match the guidance projection"
        )
    return document


def guidance_publication_document(
    approval: GuidanceApproval,
    *,
    publisher_labels: Sequence[str],
    published_at: str,
) -> dict[str, Any]:
    """Render an approval. The render copies steps and the selected agent.

    ``publisher_labels`` and ``published_at`` are caller-supplied. This function
    does not read a clock. The same approval and principal are one publication
    record; ``published_at`` is stored on that record.
    """
    if not isinstance(approval, GuidanceApproval):
        raise ProtocolContractError("guidance_approval required")
    labels = _parse_publisher_labels(publisher_labels)
    body = {
        "applicability": approval.applicability.to_dict(),
        "approval_content_hash": approval.content_hash,
        "base_revision_id": approval.base_revision_id,
        "capability_limits": list(approval.capability_limits),
        "decision": approval.decision,
        "evidence_refs": [ref.to_dict() for ref in approval.evidence_refs],
        "guidance_content_hash": approval.guidance_content_hash,
        "guidance_id": approval.guidance_id,
        "outcome_evidence": [_outcome_record(item) for item in approval.outcome_evidence],
        "policy_version": approval.policy_version,
        "published_at": _require_published_at(published_at),
        "publisher_labels": list(labels),
        "reason": approval.reason,
        "report_hash": approval.report_hash,
        "risk": approval.risk,
        "selected_agent_id": approval.selected_agent_id,
        "source_fork_id": approval.source_fork_id,
        "steps": [_step_record(step) for step in approval.steps],
        "target_role": approval.target_role,
    }
    return _envelope(OBJECT_KIND_GUIDANCE_PUBLICATION, body)


def read_guidance_publication(payload: Mapping[str, Any]) -> GuidancePublication:
    """Parse a publication snapshot. It does not rewrite the approval."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_GUIDANCE_PUBLICATION,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    _require_confirmed_failure(
        _require_str(body.get("policy_version"), "policy_version"),
        _require_str(body.get("decision"), "decision"),
    )
    steps = _parse_guidance_steps(body.get("steps"))
    evidence_refs = _parse_evidence_refs(body.get("evidence_refs"))
    _require_step_evidence(steps, evidence_refs)
    applicability = _parse_applicability(body.get("applicability"))
    capability_limits = _parse_unique_strs(
        body.get("capability_limits"), "capability_limits", allow_empty=True
    )
    guidance_id = _require_str(body.get("guidance_id"), "guidance_id")
    source_fork_id = _require_safe_id(body.get("source_fork_id"), "source_fork_id")
    base_revision_id = _require_safe_id(body.get("base_revision_id"), "base_revision_id")
    target_role = _require_choice(body.get("target_role"), "target_role", TARGET_AGENT_ROLES)
    report_hash = _require_hex(body.get("report_hash"), "report_hash")
    risk = _require_str(body.get("risk"), "risk")
    projected = _projected_guidance_body(
        guidance_id=guidance_id,
        source_fork_id=source_fork_id,
        base_revision_id=base_revision_id,
        target_role=target_role,
        capability_limits=capability_limits,
        report_hash=report_hash,
        applicability=applicability,
        risk=risk,
        evidence_refs=evidence_refs,
    )
    guidance_content_hash = _require_hex(
        body.get("guidance_content_hash"), "guidance_content_hash"
    )
    if guidance_content_hash != protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_GUIDANCE_DOCUMENT, projected
    ):
        raise ProtocolContractError(
            "guidance_content_hash does not match the guidance projection"
        )
    return GuidancePublication(
        schema_version=version,
        object_kind=OBJECT_KIND_GUIDANCE_PUBLICATION,
        content_hash=digest,
        approval_content_hash=_require_hex(
            body.get("approval_content_hash"), "approval_content_hash"
        ),
        guidance_content_hash=guidance_content_hash,
        guidance_id=guidance_id,
        selected_agent_id=_require_safe_id(body.get("selected_agent_id"), "selected_agent_id"),
        target_role=target_role,
        capability_limits=capability_limits,
        report_hash=report_hash,
        source_fork_id=source_fork_id,
        base_revision_id=base_revision_id,
        steps=steps,
        evidence_refs=evidence_refs,
        policy_version=_require_str(body.get("policy_version"), "policy_version"),
        decision=_require_str(body.get("decision"), "decision"),
        reason=_require_str(body.get("reason"), "reason"),
        outcome_evidence=_parse_outcome_evidence(body.get("outcome_evidence")),
        risk=risk,
        applicability=applicability,
        publisher_labels=_parse_publisher_labels(body.get("publisher_labels")),
        published_at=_require_published_at(body.get("published_at")),
    )


def guidance_rejection_document(proposal_content_hash: str) -> dict[str, Any]:
    """Build the rejection record. The reason is the closed rejection code."""
    body = {
        "proposal_content_hash": _require_hex(proposal_content_hash, "proposal_content_hash"),
        "reason": GUIDANCE_REJECTION_REASON,
    }
    return _envelope(OBJECT_KIND_GUIDANCE_REJECTION, body)


def read_guidance_rejection(payload: Mapping[str, Any]) -> GuidanceRejection:
    """Parse a rejection. It does not approve the proposal."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_GUIDANCE_REJECTION,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    return GuidanceRejection(
        schema_version=version,
        object_kind=OBJECT_KIND_GUIDANCE_REJECTION,
        content_hash=digest,
        proposal_content_hash=_require_hex(
            body.get("proposal_content_hash"), "proposal_content_hash"
        ),
        reason=_require_choice(body.get("reason"), "reason", _REJECTION_REASONS),
    )


def _sorted_id_list(value: Any, field: str, *, allow_empty: bool) -> tuple[str, ...]:
    items = _parse_unique_strs(value, field, allow_empty=allow_empty)
    if list(items) != sorted(items):
        raise ProtocolContractError(f"{field} must be sorted")
    return items


def judge_dispatch_gates(controls: Mapping[str, Any]) -> DispatchGateJudgment:
    """Derive every dispatch gate. The caller does not choose the outcome."""
    if not isinstance(controls, Mapping):
        raise ProtocolContractError("dispatch controls must be a dict")
    _require_exact_keys(controls, _DISPATCH_CONTROL_KEYS, "dispatch controls")
    guidance_id = _require_safe_id(controls.get("guidance_id"), "guidance_id")
    source_guidance_id = _require_safe_id(
        controls.get("source_guidance_id"), "source_guidance_id"
    )
    if source_guidance_id != guidance_id:
        raise ProtocolContractError("source_guidance_id must equal guidance_id")
    _require_hex(controls.get("guidance_content_hash"), "guidance_content_hash")
    _require_safe_id(controls.get("source_fork_id"), "source_fork_id")
    parent_session_id = _require_safe_id(
        controls.get("parent_session_id"), "parent_session_id"
    )
    bound_session_id = _require_safe_id(controls.get("bound_session_id"), "bound_session_id")
    principal_labels = _sorted_id_list(
        controls.get("principal_labels"), "principal_labels", allow_empty=True
    )
    bound_principal_labels = _sorted_id_list(
        controls.get("bound_principal_labels"),
        "bound_principal_labels",
        allow_empty=True,
    )
    permission = _require_choice(
        controls.get("permission"), "permission", _DISPATCH_PERMISSIONS
    )
    executor_role = _require_str(controls.get("executor_role"), "executor_role")
    executor_agent_id = _require_safe_id(
        controls.get("executor_agent_id"), "executor_agent_id"
    )
    selected_agent_id = _require_safe_id(
        controls.get("selected_agent_id"), "selected_agent_id"
    )
    budget = _require_choice(controls.get("budget"), "budget", _BUDGET_STATES)
    isolation = _require_choice(controls.get("isolation"), "isolation", _ISOLATION_STATES)
    allowlist = _sorted_id_list(
        controls.get("side_effect_allowlist"), "side_effect_allowlist", allow_empty=True
    )
    requested = _sorted_id_list(
        controls.get("requested_side_effects"), "requested_side_effects", allow_empty=True
    )
    capability_limits = _sorted_id_list(
        controls.get("capability_limits"), "capability_limits", allow_empty=True
    )
    agent_capabilities = _sorted_id_list(
        controls.get("agent_capabilities"), "agent_capabilities", allow_empty=True
    )
    idempotency_state = _require_choice(
        controls.get("idempotency_state"), "idempotency_state", _IDEMPOTENCY_STATES
    )
    _require_safe_id(controls.get("idempotency_key"), "idempotency_key")
    failures: list[tuple[str, str]] = []
    if not principal_labels or principal_labels != bound_principal_labels:
        failures.append(("acl", "acl_denied"))
    if parent_session_id != bound_session_id:
        failures.append(("permission", "parent_session_mismatch"))
    if permission != "granted":
        failures.append(("permission", "permission_denied"))
    if executor_role not in TARGET_AGENT_ROLES or executor_role in NON_EXECUTOR_ROLES:
        failures.append(("permission", "executor_not_target"))
    if executor_agent_id != selected_agent_id:
        failures.append(("permission", "executor_not_selected"))
    if budget != "compliant":
        failures.append(("budget", "budget_not_compliant"))
    if isolation != "isolated":
        failures.append(("security", "side_effects_unisolated"))
    if not set(requested).issubset(allowlist):
        failures.append(("security", "side_effect_not_allowlisted"))
    elif requested:
        failures.append(("security", "scripted_side_effect_forbidden"))
    if not set(capability_limits).issubset(agent_capabilities):
        failures.append(("capability", "capability_missing"))
    if idempotency_state != "clear":
        failures.append(("idempotency", "idempotency_conflict"))
    grouped: dict[str, list[str]] = {}
    for gate, code in failures:
        grouped.setdefault(gate, []).append(code)
    gates = tuple(
        DispatchGateResult(
            gate=gate,
            passed=gate not in grouped,
            reason=grouped[gate][0] if gate in grouped else _GATE_PASS_REASONS[gate],
        )
        for gate in DISPATCH_GATE_ORDER
    )
    if failures:
        return DispatchGateJudgment(
            gates=gates,
            outcome="blocked_pre_execution",
            failed_gate=failures[0][0],
            reasons=tuple(f"{gate}:{code}" for gate, code in failures),
        )
    return DispatchGateJudgment(
        gates=gates,
        outcome="admitted",
        failed_gate=None,
        reasons=("gates_passed",),
    )


def _parse_gate_results(value: Any) -> tuple[DispatchGateResult, ...]:
    if not isinstance(value, list):
        raise ProtocolContractError("gates must be a list")
    parsed: list[DispatchGateResult] = []
    for index, item in enumerate(value):
        label = f"gates[{index}]"
        if not isinstance(item, dict):
            raise ProtocolContractError(f"{label} must be a dict")
        _require_exact_keys(item, _GATE_RESULT_KEYS, label)
        parsed.append(
            DispatchGateResult(
                gate=_require_choice(item.get("gate"), f"{label}.gate", frozenset(DISPATCH_GATE_ORDER)),
                passed=_require_bool(item.get("passed"), f"{label}.passed"),
                reason=_require_str(item.get("reason"), f"{label}.reason"),
            )
        )
    return tuple(parsed)


def _read_dispatch_gate(payload: Mapping[str, Any], expected_kind: str) -> DispatchGateDecision:
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=expected_kind,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    judgment = judge_dispatch_gates(
        {key: body[key] for key in _DISPATCH_CONTROL_KEYS}
    )
    identity = {key: body[key] for key in body if key not in _GATE_DERIVED_KEYS}
    decision_id = _require_hex(body.get("decision_id"), "decision_id")
    expected_id = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_DISPATCH_GATE_IDENTITY, identity
    )
    if decision_id != expected_id:
        raise ProtocolContractError("decision_id does not match the gate body")
    if _parse_gate_results(body.get("gates")) != judgment.gates:
        raise ProtocolContractError("gates do not match the server rule")
    failed = body.get("failed_gate")
    if judgment.failed_gate is None:
        if failed is not None:
            raise ProtocolContractError("admitted dispatch requires failed_gate to be null")
    elif failed != judgment.failed_gate:
        raise ProtocolContractError("failed_gate does not match the server rule")
    if (
        body.get("outcome") != judgment.outcome
        or body.get("reasons") != list(judgment.reasons)
    ):
        raise ProtocolContractError("outcome does not match the server rule")
    expected_outcome = (
        "blocked_pre_execution"
        if expected_kind == OBJECT_KIND_BLOCKED_PRE_EXECUTION
        else "admitted"
    )
    if judgment.outcome != expected_outcome:
        raise ProtocolContractError("object_kind does not match the gate outcome")
    return DispatchGateDecision(
        schema_version=version,
        object_kind=expected_kind,
        content_hash=digest,
        decision_id=decision_id,
        guidance_id=_require_safe_id(body.get("guidance_id"), "guidance_id"),
        guidance_content_hash=_require_hex(
            body.get("guidance_content_hash"), "guidance_content_hash"
        ),
        parent_session_id=_require_safe_id(body.get("parent_session_id"), "parent_session_id"),
        bound_session_id=_require_safe_id(body.get("bound_session_id"), "bound_session_id"),
        source_fork_id=_require_safe_id(body.get("source_fork_id"), "source_fork_id"),
        source_guidance_id=_require_safe_id(
            body.get("source_guidance_id"), "source_guidance_id"
        ),
        selected_agent_id=_require_safe_id(body.get("selected_agent_id"), "selected_agent_id"),
        executor_role=_require_str(body.get("executor_role"), "executor_role"),
        executor_agent_id=_require_safe_id(body.get("executor_agent_id"), "executor_agent_id"),
        principal_labels=_sorted_id_list(
            body.get("principal_labels"), "principal_labels", allow_empty=True
        ),
        bound_principal_labels=_sorted_id_list(
            body.get("bound_principal_labels"), "bound_principal_labels", allow_empty=True
        ),
        permission=_require_choice(body.get("permission"), "permission", _DISPATCH_PERMISSIONS),
        budget=_require_choice(body.get("budget"), "budget", _BUDGET_STATES),
        isolation=_require_choice(body.get("isolation"), "isolation", _ISOLATION_STATES),
        side_effect_allowlist=_sorted_id_list(
            body.get("side_effect_allowlist"), "side_effect_allowlist", allow_empty=True
        ),
        requested_side_effects=_sorted_id_list(
            body.get("requested_side_effects"), "requested_side_effects", allow_empty=True
        ),
        agent_capabilities=_sorted_id_list(
            body.get("agent_capabilities"), "agent_capabilities", allow_empty=True
        ),
        capability_limits=_sorted_id_list(
            body.get("capability_limits"), "capability_limits", allow_empty=True
        ),
        idempotency_state=_require_choice(
            body.get("idempotency_state"), "idempotency_state", _IDEMPOTENCY_STATES
        ),
        idempotency_key=_require_safe_id(body.get("idempotency_key"), "idempotency_key"),
        gates=judgment.gates,
        outcome=judgment.outcome,
        failed_gate=judgment.failed_gate,
        reasons=judgment.reasons,
    )


def read_dispatch_gate_decision(payload: Mapping[str, Any]) -> DispatchGateDecision:
    """Parse an admitted gate record. A blocked record is a different kind."""
    return _read_dispatch_gate(payload, OBJECT_KIND_DISPATCH_GATE_DECISION)


def read_blocked_pre_execution(payload: Mapping[str, Any]) -> DispatchGateDecision:
    """Parse a blocked_pre_execution audit record. It does not create a run."""
    return _read_dispatch_gate(payload, OBJECT_KIND_BLOCKED_PRE_EXECUTION)


def dispatch_gate_document(controls: Mapping[str, Any]) -> dict[str, Any]:
    """Build the gate audit. ``outcome`` is derived from the controlled fields."""
    if not isinstance(controls, Mapping):
        raise ProtocolContractError("dispatch controls must be a dict")
    _require_exact_keys(controls, _DISPATCH_CONTROL_KEYS, "dispatch controls")
    judgment = judge_dispatch_gates(controls)
    identity = {key: controls[key] for key in sorted(_DISPATCH_CONTROL_KEYS)}
    body: dict[str, Any] = dict(identity)
    body["decision_id"] = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_DISPATCH_GATE_IDENTITY, identity
    )
    body["failed_gate"] = judgment.failed_gate
    body["gates"] = [
        {"gate": item.gate, "passed": item.passed, "reason": item.reason}
        for item in judgment.gates
    ]
    body["outcome"] = judgment.outcome
    body["reasons"] = list(judgment.reasons)
    kind = (
        OBJECT_KIND_BLOCKED_PRE_EXECUTION
        if judgment.outcome == "blocked_pre_execution"
        else OBJECT_KIND_DISPATCH_GATE_DECISION
    )
    document = _envelope(kind, body)
    _read_dispatch_gate(document, kind)
    return document


def remediation_dispatch_request_document(
    decision: DispatchGateDecision, *, target_role: str
) -> dict[str, Any]:
    """Seal the dispatch request.

    The Coforge worker is the real executor of this request. P12 connects
    that worker. This builder does not call the worker or start a model.
    """
    if not isinstance(decision, DispatchGateDecision):
        raise ProtocolContractError("dispatch_gate_decision required")
    if (
        decision.object_kind != OBJECT_KIND_DISPATCH_GATE_DECISION
        or decision.outcome != "admitted"
        or decision.failed_gate is not None
    ):
        raise ProtocolContractError("dispatch_not_admitted")
    role = _require_choice(target_role, "target_role", TARGET_AGENT_ROLES)
    identity = {
        "capability_limits": list(decision.capability_limits),
        "decision_id": decision.decision_id,
        "guidance_content_hash": decision.guidance_content_hash,
        "guidance_id": decision.guidance_id,
        "idempotency_key": decision.idempotency_key,
        "parent_session_id": decision.parent_session_id,
        "requested_side_effects": list(decision.requested_side_effects),
        "selected_agent_id": decision.selected_agent_id,
        "source_fork_id": decision.source_fork_id,
        "source_guidance_id": decision.source_guidance_id,
        "target_role": role,
    }
    body = {
        **identity,
        "request_id": protocol_content_hash(
            SCHEMA_VERSION_AIG_NATIVE_V2,
            OBJECT_KIND_REMEDIATION_DISPATCH_REQUEST_IDENTITY,
            identity,
        ),
    }
    document = _envelope(OBJECT_KIND_REMEDIATION_DISPATCH_REQUEST, body)
    read_remediation_dispatch_request(document)
    return document


def read_remediation_dispatch_request(payload: Mapping[str, Any]) -> RemediationDispatchRequest:
    """Parse a dispatch request. Reading it does not start the Coforge worker."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_REMEDIATION_DISPATCH_REQUEST,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    identity = {key: body[key] for key in body if key != "request_id"}
    request_id = _require_hex(body.get("request_id"), "request_id")
    expected = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        OBJECT_KIND_REMEDIATION_DISPATCH_REQUEST_IDENTITY,
        identity,
    )
    if request_id != expected:
        raise ProtocolContractError("request_id does not match the request body")
    guidance_id = _require_safe_id(body.get("guidance_id"), "guidance_id")
    source_guidance_id = _require_safe_id(body.get("source_guidance_id"), "source_guidance_id")
    if source_guidance_id != guidance_id:
        raise ProtocolContractError("source_guidance_id must equal guidance_id")
    return RemediationDispatchRequest(
        schema_version=version,
        object_kind=OBJECT_KIND_REMEDIATION_DISPATCH_REQUEST,
        content_hash=digest,
        request_id=request_id,
        decision_id=_require_hex(body.get("decision_id"), "decision_id"),
        guidance_id=guidance_id,
        guidance_content_hash=_require_hex(
            body.get("guidance_content_hash"), "guidance_content_hash"
        ),
        parent_session_id=_require_safe_id(body.get("parent_session_id"), "parent_session_id"),
        source_fork_id=_require_safe_id(body.get("source_fork_id"), "source_fork_id"),
        source_guidance_id=source_guidance_id,
        selected_agent_id=_require_safe_id(body.get("selected_agent_id"), "selected_agent_id"),
        target_role=_require_choice(body.get("target_role"), "target_role", TARGET_AGENT_ROLES),
        capability_limits=_sorted_id_list(
            body.get("capability_limits"), "capability_limits", allow_empty=True
        ),
        idempotency_key=_require_safe_id(body.get("idempotency_key"), "idempotency_key"),
        requested_side_effects=_sorted_id_list(
            body.get("requested_side_effects"), "requested_side_effects", allow_empty=True
        ),
    )


def remediation_dispatch_confirmation_document(
    *, request_id: str, child_run_id: str
) -> dict[str, Any]:
    """Seal a scripted confirmation.

    The Coforge worker is the real executor. P12 connects it. This builder
    accepts only the scripted stand-in and does not call the worker.
    """
    identity = {
        "child_run_id": _require_safe_id(child_run_id, "child_run_id"),
        "executor": "scripted",
        "request_id": _require_hex(request_id, "request_id"),
        "status": "accepted",
    }
    body = {
        **identity,
        "confirmation_id": protocol_content_hash(
            SCHEMA_VERSION_AIG_NATIVE_V2,
            OBJECT_KIND_REMEDIATION_DISPATCH_CONFIRMATION_IDENTITY,
            identity,
        ),
    }
    document = _envelope(OBJECT_KIND_REMEDIATION_DISPATCH_CONFIRMATION, body)
    read_remediation_dispatch_confirmation(document)
    return document


def read_remediation_dispatch_confirmation(
    payload: Mapping[str, Any],
) -> RemediationDispatchConfirmation:
    """Parse a confirmation. An unknown executor, including a premature worker, fails closed."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_REMEDIATION_DISPATCH_CONFIRMATION,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    identity = {key: body[key] for key in body if key != "confirmation_id"}
    confirmation_id = _require_hex(body.get("confirmation_id"), "confirmation_id")
    expected = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        OBJECT_KIND_REMEDIATION_DISPATCH_CONFIRMATION_IDENTITY,
        identity,
    )
    if confirmation_id != expected:
        raise ProtocolContractError("confirmation_id does not match the confirmation body")
    return RemediationDispatchConfirmation(
        schema_version=version,
        object_kind=OBJECT_KIND_REMEDIATION_DISPATCH_CONFIRMATION,
        content_hash=digest,
        confirmation_id=confirmation_id,
        request_id=_require_hex(body.get("request_id"), "request_id"),
        child_run_id=_require_safe_id(body.get("child_run_id"), "child_run_id"),
        status=_require_choice(
            body.get("status"), "status", _DISPATCH_CONFIRMATION_STATUSES
        ),
        executor=_require_choice(body.get("executor"), "executor", _DISPATCH_EXECUTORS),
    )


def child_remediation_run_document(
    *,
    parent_session_id: str,
    source_fork_id: str,
    source_guidance_id: str,
    guidance_content_hash: str,
    request_id: str,
    selected_agent_id: str,
    child_session_id: str,
    output_ids: Sequence[str],
) -> dict[str, Any]:
    """Seal a child run. ``child_session_id`` must not be the parent session."""
    parent = _require_safe_id(parent_session_id, "parent_session_id")
    child_session = _require_safe_id(child_session_id, "child_session_id")
    if child_session == parent:
        raise ProtocolContractError("child_session_id must not be the parent session")
    identity = {
        "child_session_id": child_session,
        "guidance_content_hash": _require_hex(guidance_content_hash, "guidance_content_hash"),
        "output_ids": list(_sorted_id_list(list(output_ids), "output_ids", allow_empty=False)),
        "parent_session_id": parent,
        "request_id": _require_hex(request_id, "request_id"),
        "selected_agent_id": _require_safe_id(selected_agent_id, "selected_agent_id"),
        "source_fork_id": _require_safe_id(source_fork_id, "source_fork_id"),
        "source_guidance_id": _require_safe_id(source_guidance_id, "source_guidance_id"),
    }
    body = {
        **identity,
        "child_run_id": protocol_content_hash(
            SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_CHILD_REMEDIATION_RUN_IDENTITY, identity
        ),
    }
    document = _envelope(OBJECT_KIND_CHILD_REMEDIATION_RUN, body)
    read_child_remediation_run(document)
    return document


def read_child_remediation_run(payload: Mapping[str, Any]) -> ChildRemediationRun:
    """Parse a child run. It does not open the parent failure run."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_CHILD_REMEDIATION_RUN,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    identity = {key: body[key] for key in body if key != "child_run_id"}
    child_run_id = _require_hex(body.get("child_run_id"), "child_run_id")
    expected = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_CHILD_REMEDIATION_RUN_IDENTITY, identity
    )
    if child_run_id != expected:
        raise ProtocolContractError("child_run_id does not match the run body")
    parent = _require_safe_id(body.get("parent_session_id"), "parent_session_id")
    child_session = _require_safe_id(body.get("child_session_id"), "child_session_id")
    if child_session == parent:
        raise ProtocolContractError("child_session_id must not be the parent session")
    return ChildRemediationRun(
        schema_version=version,
        object_kind=OBJECT_KIND_CHILD_REMEDIATION_RUN,
        content_hash=digest,
        child_run_id=child_run_id,
        parent_session_id=parent,
        source_fork_id=_require_safe_id(body.get("source_fork_id"), "source_fork_id"),
        source_guidance_id=_require_safe_id(body.get("source_guidance_id"), "source_guidance_id"),
        guidance_content_hash=_require_hex(
            body.get("guidance_content_hash"), "guidance_content_hash"
        ),
        request_id=_require_hex(body.get("request_id"), "request_id"),
        selected_agent_id=_require_safe_id(body.get("selected_agent_id"), "selected_agent_id"),
        child_session_id=child_session,
        output_ids=_sorted_id_list(body.get("output_ids"), "output_ids", allow_empty=False),
    )


def child_run_output_document(
    *,
    output_kind: str,
    output_id: str,
    child_run_id: str,
    child_session_id: str,
    parent_session_id: str,
    source_fork_id: str,
    source_guidance_id: str,
) -> dict[str, Any]:
    """Seal one child AO or Block. Provenance is required and not optional."""
    parent = _require_safe_id(parent_session_id, "parent_session_id")
    child_session = _require_safe_id(child_session_id, "child_session_id")
    if child_session == parent:
        raise ProtocolContractError("child_session_id must not be the parent session")
    body = {
        "child_run_id": _require_safe_id(child_run_id, "child_run_id"),
        "child_session_id": child_session,
        "output_id": _require_safe_id(output_id, "output_id"),
        "output_kind": _require_choice(output_kind, "output_kind", _CHILD_OUTPUT_KINDS),
        "parent_session_id": parent,
        "source_fork_id": _require_safe_id(source_fork_id, "source_fork_id"),
        "source_guidance_id": _require_safe_id(source_guidance_id, "source_guidance_id"),
    }
    document = _envelope(OBJECT_KIND_CHILD_RUN_OUTPUT, body)
    read_child_run_output(document)
    return document


def read_child_run_output(payload: Mapping[str, Any]) -> ChildRunOutput:
    """Parse a child output. Missing provenance fails closed."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_CHILD_RUN_OUTPUT,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    parent = _require_safe_id(body.get("parent_session_id"), "parent_session_id")
    child_session = _require_safe_id(body.get("child_session_id"), "child_session_id")
    if child_session == parent:
        raise ProtocolContractError("child_session_id must not be the parent session")
    return ChildRunOutput(
        schema_version=version,
        object_kind=OBJECT_KIND_CHILD_RUN_OUTPUT,
        content_hash=digest,
        output_kind=_require_choice(body.get("output_kind"), "output_kind", _CHILD_OUTPUT_KINDS),
        output_id=_require_safe_id(body.get("output_id"), "output_id"),
        child_run_id=_require_safe_id(body.get("child_run_id"), "child_run_id"),
        child_session_id=child_session,
        parent_session_id=parent,
        source_fork_id=_require_safe_id(body.get("source_fork_id"), "source_fork_id"),
        source_guidance_id=_require_safe_id(body.get("source_guidance_id"), "source_guidance_id"),
    )


def _parse_revised_steps(value: Any) -> tuple[GuidanceRevisionStep, ...]:
    if not isinstance(value, list) or not value:
        raise ProtocolContractError("revised_steps must be a non-empty list")
    steps: list[GuidanceRevisionStep] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        label = f"revised_steps[{index}]"
        if not isinstance(item, dict):
            raise ProtocolContractError(f"{label} must be a dict")
        _require_exact_keys(item, _REVISED_STEP_KEYS, label)
        imagined = _require_safe_id(item.get("imagined_node_id"), f"{label}.imagined_node_id")
        if imagined in seen:
            raise ProtocolContractError("revised_steps imagined_node_id must be unique")
        seen.add(imagined)
        steps.append(
            GuidanceRevisionStep(
                imagined_node_id=imagined,
                statement=_require_str(item.get("statement"), f"{label}.statement"),
            )
        )
    return tuple(steps)


def guidance_adjustment_document(
    *,
    guidance_content_hash: str,
    guidance_id: str,
    author_agent_id: str,
    bound_agent_id: str,
    parent_session_id: str,
    target_role: str,
    bound_target_role: str,
    capability_limits: Sequence[str],
    bound_capability_limits: Sequence[str],
    bound_imagined_node_ids: Sequence[str],
    revised_steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal an appendix. It does not rewrite the frozen guidance document."""
    identity = {
        "author_agent_id": _require_safe_id(author_agent_id, "author_agent_id"),
        "bound_agent_id": _require_safe_id(bound_agent_id, "bound_agent_id"),
        "bound_capability_limits": list(
            _sorted_id_list(
                list(bound_capability_limits), "bound_capability_limits", allow_empty=True
            )
        ),
        "bound_imagined_node_ids": list(
            _sorted_id_list(
                list(bound_imagined_node_ids), "bound_imagined_node_ids", allow_empty=False
            )
        ),
        "bound_target_role": _require_choice(
            bound_target_role, "bound_target_role", TARGET_AGENT_ROLES
        ),
        "capability_limits": list(
            _sorted_id_list(list(capability_limits), "capability_limits", allow_empty=True)
        ),
        "guidance_content_hash": _require_hex(guidance_content_hash, "guidance_content_hash"),
        "guidance_id": _require_safe_id(guidance_id, "guidance_id"),
        "parent_session_id": _require_safe_id(parent_session_id, "parent_session_id"),
        "revised_steps": [
            {
                "imagined_node_id": step.imagined_node_id,
                "statement": step.statement,
            }
            for step in _parse_revised_steps(list(revised_steps))
        ],
        "target_role": _require_choice(target_role, "target_role", TARGET_AGENT_ROLES),
    }
    body = {
        **identity,
        "adjustment_id": protocol_content_hash(
            SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_GUIDANCE_ADJUSTMENT_IDENTITY, identity
        ),
    }
    document = _envelope(OBJECT_KIND_GUIDANCE_ADJUSTMENT, body)
    read_guidance_adjustment(document)
    return document


def read_guidance_adjustment(payload: Mapping[str, Any]) -> GuidanceAdjustment:
    """Parse an adjustment appendix. Wider permissions are rejected."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_GUIDANCE_ADJUSTMENT,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    identity = {key: body[key] for key in body if key != "adjustment_id"}
    adjustment_id = _require_hex(body.get("adjustment_id"), "adjustment_id")
    expected = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_GUIDANCE_ADJUSTMENT_IDENTITY, identity
    )
    if adjustment_id != expected:
        raise ProtocolContractError("adjustment_id does not match the adjustment body")
    author = _require_safe_id(body.get("author_agent_id"), "author_agent_id")
    bound_agent = _require_safe_id(body.get("bound_agent_id"), "bound_agent_id")
    if author != bound_agent:
        raise ProtocolContractError("author_agent_id exceeds the frozen permission")
    target_role = _require_choice(body.get("target_role"), "target_role", TARGET_AGENT_ROLES)
    bound_role = _require_choice(
        body.get("bound_target_role"), "bound_target_role", TARGET_AGENT_ROLES
    )
    if target_role != bound_role:
        raise ProtocolContractError("target_role exceeds the frozen permission")
    limits = _sorted_id_list(body.get("capability_limits"), "capability_limits", allow_empty=True)
    bound_limits = _sorted_id_list(
        body.get("bound_capability_limits"), "bound_capability_limits", allow_empty=True
    )
    if not set(limits).issubset(bound_limits):
        raise ProtocolContractError("capability_limits exceeds the frozen permission")
    bound_nodes = _sorted_id_list(
        body.get("bound_imagined_node_ids"), "bound_imagined_node_ids", allow_empty=False
    )
    revised = _parse_revised_steps(body.get("revised_steps"))
    if any(step.imagined_node_id not in set(bound_nodes) for step in revised):
        raise ProtocolContractError("revised_steps exceeds the frozen permission")
    return GuidanceAdjustment(
        schema_version=version,
        object_kind=OBJECT_KIND_GUIDANCE_ADJUSTMENT,
        content_hash=digest,
        adjustment_id=adjustment_id,
        guidance_content_hash=_require_hex(
            body.get("guidance_content_hash"), "guidance_content_hash"
        ),
        guidance_id=_require_safe_id(body.get("guidance_id"), "guidance_id"),
        author_agent_id=author,
        bound_agent_id=bound_agent,
        parent_session_id=_require_safe_id(body.get("parent_session_id"), "parent_session_id"),
        target_role=target_role,
        bound_target_role=bound_role,
        capability_limits=limits,
        bound_capability_limits=bound_limits,
        bound_imagined_node_ids=bound_nodes,
        revised_steps=revised,
    )


def remediation_constraint_document(
    *,
    imagined_node_id: str,
    alignment: str,
    applicability: Mapping[str, Any],
    statement: str,
    source_fork_id: str,
    source_guidance_id: str,
    parent_session_id: str,
    child_run_id: str,
    report_content_hash: str,
) -> dict[str, Any]:
    """Record a constraint. This document is not a direct refutation."""
    parsed = _parse_applicability(applicability)
    identity = {
        "alignment": _require_choice(alignment, "alignment", _CONSTRAINT_ALIGNMENTS),
        "applicability": parsed.to_dict(),
        "child_run_id": _require_safe_id(child_run_id, "child_run_id"),
        "imagined_node_id": _require_safe_id(imagined_node_id, "imagined_node_id"),
        "parent_session_id": _require_safe_id(parent_session_id, "parent_session_id"),
        "report_content_hash": _require_hex(report_content_hash, "report_content_hash"),
        "source_fork_id": _require_safe_id(source_fork_id, "source_fork_id"),
        "source_guidance_id": _require_safe_id(source_guidance_id, "source_guidance_id"),
        "statement": _require_str(statement, "statement"),
    }
    body = {
        **identity,
        "constraint_id": protocol_content_hash(
            SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_REMEDIATION_CONSTRAINT_IDENTITY, identity
        ),
    }
    document = _envelope(OBJECT_KIND_REMEDIATION_CONSTRAINT, body)
    read_remediation_constraint(document)
    return document


def read_remediation_constraint(payload: Mapping[str, Any]) -> RemediationConstraint:
    """Parse a constraint. Direct-evidence fields are not part of this record."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_REMEDIATION_CONSTRAINT,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    identity = {key: body[key] for key in body if key != "constraint_id"}
    constraint_id = _require_hex(body.get("constraint_id"), "constraint_id")
    expected = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2, OBJECT_KIND_REMEDIATION_CONSTRAINT_IDENTITY, identity
    )
    if constraint_id != expected:
        raise ProtocolContractError("constraint_id does not match the constraint body")
    return RemediationConstraint(
        schema_version=version,
        object_kind=OBJECT_KIND_REMEDIATION_CONSTRAINT,
        content_hash=digest,
        constraint_id=constraint_id,
        imagined_node_id=_require_safe_id(body.get("imagined_node_id"), "imagined_node_id"),
        alignment=_require_choice(body.get("alignment"), "alignment", _CONSTRAINT_ALIGNMENTS),
        applicability=_parse_applicability(body.get("applicability")),
        statement=_require_str(body.get("statement"), "statement"),
        source_fork_id=_require_safe_id(body.get("source_fork_id"), "source_fork_id"),
        source_guidance_id=_require_safe_id(body.get("source_guidance_id"), "source_guidance_id"),
        parent_session_id=_require_safe_id(body.get("parent_session_id"), "parent_session_id"),
        child_run_id=_require_safe_id(body.get("child_run_id"), "child_run_id"),
        report_content_hash=_require_hex(body.get("report_content_hash"), "report_content_hash"),
    )


def production_direct_admission_document(
    *,
    imagined_node_id: str,
    actual_block_id: str,
    alignment: str,
    source_fork_id: str,
    source_guidance_id: str,
    parent_session_id: str,
    child_run_id: str,
    report_content_hash: str,
    child_block_ids: Sequence[str],
) -> dict[str, Any]:
    """Admit one aligned child block. Divergent steps cannot use this document."""
    chosen = _require_choice(alignment, "alignment", _DIRECT_ALIGNMENTS)
    blocks = _sorted_id_list(list(child_block_ids), "child_block_ids", allow_empty=False)
    actual = _require_safe_id(actual_block_id, "actual_block_id")
    if actual not in blocks:
        raise ProtocolContractError("actual_block_id is not a child block")
    identity = {
        "actual_block_id": actual,
        "alignment": chosen,
        "child_block_ids": list(blocks),
        "child_run_id": _require_safe_id(child_run_id, "child_run_id"),
        "imagined_node_id": _require_safe_id(imagined_node_id, "imagined_node_id"),
        "parent_session_id": _require_safe_id(parent_session_id, "parent_session_id"),
        "report_content_hash": _require_hex(report_content_hash, "report_content_hash"),
        "source_fork_id": _require_safe_id(source_fork_id, "source_fork_id"),
        "source_guidance_id": _require_safe_id(source_guidance_id, "source_guidance_id"),
    }
    body = {
        **identity,
        "admission_id": protocol_content_hash(
            SCHEMA_VERSION_AIG_NATIVE_V2,
            OBJECT_KIND_PRODUCTION_DIRECT_ADMISSION_IDENTITY,
            identity,
        ),
        "evidence_tier": EVIDENCE_TIER_PRODUCTION_DIRECT,
        "reasons": ["production_direct_admitted"],
    }
    document = _envelope(OBJECT_KIND_PRODUCTION_DIRECT_ADMISSION, body)
    read_production_direct_admission(document)
    return document


def read_production_direct_admission(payload: Mapping[str, Any]) -> ProductionDirectAdmission:
    """Parse a production-direct admission. The tier is not caller-selected."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_PRODUCTION_DIRECT_ADMISSION,
        reader_name="v2 reader",
        reject_claim_cut=True,
    )
    identity = {
        key: body[key]
        for key in body
        if key not in {"admission_id", "evidence_tier", "reasons"}
    }
    admission_id = _require_hex(body.get("admission_id"), "admission_id")
    expected = protocol_content_hash(
        SCHEMA_VERSION_AIG_NATIVE_V2,
        OBJECT_KIND_PRODUCTION_DIRECT_ADMISSION_IDENTITY,
        identity,
    )
    if admission_id != expected:
        raise ProtocolContractError("admission_id does not match the admission body")
    alignment = _require_choice(body.get("alignment"), "alignment", _DIRECT_ALIGNMENTS)
    blocks = _sorted_id_list(body.get("child_block_ids"), "child_block_ids", allow_empty=False)
    actual = _require_safe_id(body.get("actual_block_id"), "actual_block_id")
    if actual not in blocks:
        raise ProtocolContractError("actual_block_id is not a child block")
    if body.get("evidence_tier") != EVIDENCE_TIER_PRODUCTION_DIRECT or body.get("reasons") != [
        "production_direct_admitted"
    ]:
        raise ProtocolContractError("admission flags do not match the server rule")
    return ProductionDirectAdmission(
        schema_version=version,
        object_kind=OBJECT_KIND_PRODUCTION_DIRECT_ADMISSION,
        content_hash=digest,
        admission_id=admission_id,
        evidence_tier=EVIDENCE_TIER_PRODUCTION_DIRECT,
        imagined_node_id=_require_safe_id(body.get("imagined_node_id"), "imagined_node_id"),
        actual_block_id=actual,
        alignment=alignment,
        source_fork_id=_require_safe_id(body.get("source_fork_id"), "source_fork_id"),
        source_guidance_id=_require_safe_id(body.get("source_guidance_id"), "source_guidance_id"),
        parent_session_id=_require_safe_id(body.get("parent_session_id"), "parent_session_id"),
        child_run_id=_require_safe_id(body.get("child_run_id"), "child_run_id"),
        report_content_hash=_require_hex(body.get("report_content_hash"), "report_content_hash"),
        child_block_ids=blocks,
        reasons=("production_direct_admitted",),
    )


def guidance_execution_report_document(
    *,
    guidance_id: str,
    source_fork_id: str,
    parent_session_id: str,
    steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal a target-agent report. It does not write production evidence."""
    body = {
        "guidance_id": _require_str(guidance_id, "guidance_id"),
        "parent_session_id": _require_str(parent_session_id, "parent_session_id"),
        "source_fork_id": _require_str(source_fork_id, "source_fork_id"),
        "steps": _copy_step_alignment_mappings(steps),
    }
    document = _envelope(OBJECT_KIND_GUIDANCE_EXECUTION_REPORT, body)
    read_guidance_execution_report(document)
    return document


def _toolloop_json_value(value: Any, field: str) -> Any:
    """Copy JSON data. Tuples become lists so the hash matches the log."""
    if value is None or isinstance(value, str) or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        raise ProtocolContractError(f"{field} must not contain a float")
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ProtocolContractError(f"{field} keys must be non-empty str")
            copied[key] = _toolloop_json_value(item, field)
        return copied
    if isinstance(value, (list, tuple)):
        return [_toolloop_json_value(item, field) for item in value]
    raise ProtocolContractError(f"{field} must be JSON data")


def toolloop_draft_op_document(
    *,
    run_id: str,
    seq: int,
    verb: str,
    args: Mapping[str, Any],
    prev_content_hash: str,
) -> dict[str, Any]:
    """Seal one draft edit. The content hash is computed, not copied."""
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        raise ProtocolContractError("seq must be a non-negative int")
    if not isinstance(args, Mapping):
        raise ProtocolContractError("args must be a dict")
    body = {
        "args": _toolloop_json_value(dict(args), "args"),
        "prev_content_hash": _require_hex(prev_content_hash, "prev_content_hash"),
        "run_id": _require_safe_id(run_id, "run_id"),
        "seq": seq,
        "verb": _require_choice(verb, "verb", TOOLLOOP_BUILD_VERBS),
    }
    document = _envelope(OBJECT_KIND_TOOLLOOP_DRAFT_OP, body)
    read_toolloop_draft_op(document)
    return document


def read_toolloop_draft_op(payload: Mapping[str, Any]) -> ToolloopDraftOp:
    """Parse one draft edit. Unknown verbs are rejected."""
    version, body, digest = _open_envelope(
        payload,
        expected_version=SCHEMA_VERSION_AIG_NATIVE_V2,
        expected_kind=OBJECT_KIND_TOOLLOOP_DRAFT_OP,
        reader_name="tool-loop reader",
        reject_claim_cut=True,
    )
    seq = body.get("seq")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        raise ProtocolContractError("seq must be a non-negative int")
    args = body.get("args")
    if not isinstance(args, dict):
        raise ProtocolContractError("args must be a dict")
    return ToolloopDraftOp(
        schema_version=version,
        object_kind=OBJECT_KIND_TOOLLOOP_DRAFT_OP,
        content_hash=digest,
        run_id=_require_safe_id(body.get("run_id"), "run_id"),
        seq=seq,
        verb=_require_choice(body.get("verb"), "verb", TOOLLOOP_BUILD_VERBS),
        args=_toolloop_json_value(args, "args"),
        prev_content_hash=_require_hex(body.get("prev_content_hash"), "prev_content_hash"),
    )


_KIND_READERS: dict[str, Callable[..., Any]] = {
    OBJECT_KIND_LEGACY_CLAIM_CUT: read_legacy_claim_cut,
    OBJECT_KIND_V2_FIXTURE_SET: read_v2_fixture_set,
    OBJECT_KIND_FORK_PROPOSAL: read_fork_proposal,
    OBJECT_KIND_FORK_NODE_OVERLAY: read_fork_node_overlay,
    OBJECT_KIND_TASK_OUTCOME_ASSESSMENT: read_task_outcome_assessment,
    OBJECT_KIND_REPLAY_ELIGIBILITY: read_replay_eligibility,
    OBJECT_KIND_GUIDANCE_DOCUMENT: read_guidance_document,
    OBJECT_KIND_GUIDANCE_EXECUTION_REPORT: read_guidance_execution_report,
    OBJECT_KIND_BRANCH_EVALUATION: read_branch_evaluation,
    OBJECT_KIND_FIRST_ACTION_ALIGNMENT: read_first_action_alignment,
    OBJECT_KIND_NATIVE_EDGE_DRAFT: read_native_edge_draft,
    OBJECT_KIND_NATIVE_EDGE_REVISION: read_native_edge_revision,
    OBJECT_KIND_NATIVE_EDGE_INDEX: read_native_edge_index,
    OBJECT_KIND_OUTCOME_CONTRACT: read_outcome_contract,
    OBJECT_KIND_STRATEGY_PATTERN_PROPOSAL: read_strategy_pattern_proposal,
    OBJECT_KIND_STRATEGY_PATTERN: read_strategy_pattern,
    OBJECT_KIND_REPLAY_ELIGIBILITY_ASSESSMENT: read_replay_eligibility_assessment,
    OBJECT_KIND_REPLAY_REQUEST: read_replay_request,
    OBJECT_KIND_REPLAY_RESULT: read_replay_result,
    OBJECT_KIND_REPLAY_AIG: read_replay_aig,
    OBJECT_KIND_DIRECT_EVIDENCE_ADMISSION: read_direct_evidence_admission,
    OBJECT_KIND_GUIDED_COUNTERFACTUAL_BRANCH: read_guided_counterfactual_branch,
    OBJECT_KIND_AGENT_CANDIDATE_CATALOG: read_agent_candidate_catalog,
    OBJECT_KIND_GUIDANCE_APPROVAL: read_guidance_approval,
    OBJECT_KIND_GUIDANCE_PUBLICATION: read_guidance_publication,
    OBJECT_KIND_GUIDANCE_REJECTION: read_guidance_rejection,
    OBJECT_KIND_DISPATCH_GATE_DECISION: read_dispatch_gate_decision,
    OBJECT_KIND_BLOCKED_PRE_EXECUTION: read_blocked_pre_execution,
    OBJECT_KIND_REMEDIATION_DISPATCH_REQUEST: read_remediation_dispatch_request,
    OBJECT_KIND_REMEDIATION_DISPATCH_CONFIRMATION: read_remediation_dispatch_confirmation,
    OBJECT_KIND_CHILD_REMEDIATION_RUN: read_child_remediation_run,
    OBJECT_KIND_CHILD_RUN_OUTPUT: read_child_run_output,
    OBJECT_KIND_GUIDANCE_ADJUSTMENT: read_guidance_adjustment,
    OBJECT_KIND_REMEDIATION_CONSTRAINT: read_remediation_constraint,
    OBJECT_KIND_PRODUCTION_DIRECT_ADMISSION: read_production_direct_admission,
    OBJECT_KIND_TOOLLOOP_DRAFT_OP: read_toolloop_draft_op,
}


def _read_v2_member(
    payload: Mapping[str, Any], kind: str, catalog: BaseRevisionCatalog
) -> Any:
    reader = _KIND_READERS.get(kind)
    if reader is None or kind not in _V2_DOCUMENT_KINDS:
        raise ProtocolContractError(f"object_kind must be a v2 document, got {kind!r}")
    if kind in _FORK_OBJECT_KINDS:
        return reader(payload, catalog)
    return reader(payload)


def _open_envelope(
    payload: Mapping[str, Any],
    *,
    expected_version: str,
    expected_kind: str,
    reader_name: str,
    reject_claim_cut: bool,
) -> tuple[str, dict[str, Any], str]:
    if not isinstance(payload, Mapping):
        raise ProtocolContractError("protocol document must be a dict")
    version = _schema_version_of(payload)
    if version != expected_version:
        raise ProtocolContractError(f"{reader_name} refuses schema_version {version!r}")
    if reject_claim_cut:
        _reject_claim_cut_fields(payload)
    _require_exact_keys(payload, _ENVELOPE_KEYS, "protocol document")
    kind = payload.get("object_kind")
    if kind != expected_kind:
        raise ProtocolContractError(f"object_kind must be {expected_kind!r}, got {kind!r}")
    body = payload.get("body")
    if not isinstance(body, dict):
        raise ProtocolContractError("body must be a dict")
    _require_exact_keys(
        body,
        _BODY_KEYS[expected_kind],
        expected_kind,
        optional=_OPTIONAL_BODY_KEYS.get(expected_kind, frozenset()),
    )
    stored = payload.get("content_hash")
    digest = protocol_content_hash(version, expected_kind, body)
    if stored != digest:
        raise ProtocolContractError(
            f"content_hash mismatch: stored {stored!r} != computed {digest!r}"
        )
    return version, body, digest


def _schema_version_of(payload: Mapping[str, Any]) -> str:
    if "schema_version" not in payload:
        return _parse_schema_version(None)
    return _parse_schema_version(payload.get("schema_version"))


def _parse_schema_version(value: Any) -> str:
    if value is None:
        raise UnsupportedSchemaVersion(
            "unsupported schema_version missing; refusing to guess or downgrade"
        )
    if not isinstance(value, str) or value not in SCHEMA_VERSIONS:
        raise UnsupportedSchemaVersion(
            f"unsupported schema_version {value!r}; refusing to guess or downgrade"
        )
    return value


def _reject_native_lifecycle_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _NATIVE_EDGE_LIFECYCLE_FIELDS:
                raise ProtocolContractError(
                    f"native edge rejects claim lifecycle field {key!r}"
                )
            _reject_native_lifecycle_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_native_lifecycle_fields(item)


def paper_authorship(value: Any) -> str:
    """Paper authorship. Blank means the node did not carry the field."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ProtocolContractError("invalid_authorship")
    if value == "":
        return ""
    if value not in PAPER_AUTHORSHIP_VALUES:
        raise ProtocolContractError("invalid_authorship")
    return value


def _exact_native_dict(
    payload: Any,
    allowed: frozenset[str],
    label: str,
    *,
    optional: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProtocolContractError(f"{label} must be a dict")
    _require_exact_keys(payload, allowed, label, optional=optional)
    return payload


def _check_paper_block_fields(
    record: Mapping[str, Any], label: str, *, paper_strict: bool
) -> None:
    """Optional paper fields on a native block.

    ``defect_recorded``, when present, must be an object with a non-empty
    ``note`` and requires ``block_kind`` ``agent``. That constraint applies
    on every native-edge document: the field only arrives from the paper path.

    The authorship enum applies only when ``paper_strict`` is true. An
    ``agent`` block must then use ``authored``, ``relayed``, ``executed``, or
    an empty string. Single-pass revisions keep free-string authorship.
    """
    if "defect_recorded" in record:
        defect = record["defect_recorded"]
        if not isinstance(defect, dict):
            raise ProtocolContractError(f"{label}.defect_recorded must be an object")
        _require_exact_keys(defect, _DEFECT_RECORDED_KEYS, f"{label}.defect_recorded")
        note = defect.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ProtocolContractError(f"{label}.defect_recorded.note must be a non-empty str")
        if record.get("block_kind") != "agent":
            raise ProtocolContractError("defect_recorded requires block_kind agent")
    if paper_strict and record.get("block_kind") == "agent":
        authorship = record.get("authorship", "")
        if authorship not in PAPER_AUTHORSHIP_VALUES and authorship != "":
            raise ProtocolContractError("invalid_authorship")


def _check_native_body_shape(body: Mapping[str, Any], *, paper_strict: bool = False) -> None:
    blocks = body.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ProtocolContractError("blocks must be a non-empty list")
    for index, block in enumerate(blocks):
        record = _exact_native_dict(
            block,
            _NATIVE_BLOCK_KEYS,
            f"blocks[{index}]",
            optional=_NATIVE_BLOCK_OPTIONAL_KEYS,
        )
        _check_paper_block_fields(record, f"blocks[{index}]", paper_strict=paper_strict)
        _exact_native_dict(record.get("segment_ref"), _NATIVE_SEGMENT_KEYS, f"blocks[{index}].segment_ref")
        refs = record.get("source_refs")
        hashes = record.get("source_content_hashes")
        if not isinstance(refs, list) or not refs:
            raise ProtocolContractError(f"blocks[{index}].source_refs must be a non-empty list")
        if not isinstance(hashes, list):
            raise ProtocolContractError(
                f"blocks[{index}].source_content_hashes must be a list"
            )
        for ref_index, ref in enumerate(refs):
            _exact_native_dict(
                ref, _NATIVE_SOURCE_KEYS, f"blocks[{index}].source_refs[{ref_index}]"
            )
    edges = body.get("edges")
    if not isinstance(edges, list):
        raise ProtocolContractError("edges must be a list")
    for index, edge in enumerate(edges):
        _exact_native_dict(edge, _NATIVE_EDGE_KEYS, f"edges[{index}]")
    coverage = _exact_native_dict(body.get("coverage"), _NATIVE_COVERAGE_KEYS, "coverage")
    _exact_native_dict(coverage.get("total_ao_range"), _NATIVE_RANGE_KEYS, "coverage.total_ao_range")
    inspected = coverage.get("inspected_refs")
    if not isinstance(inspected, list):
        raise ProtocolContractError("coverage.inspected_refs must be a list")
    for index, ref in enumerate(inspected):
        _exact_native_dict(ref, _NATIVE_SOURCE_KEYS, f"coverage.inspected_refs[{index}]")
    included = coverage.get("included_ranges")
    if not isinstance(included, list):
        raise ProtocolContractError("coverage.included_ranges must be a list")
    for index, item in enumerate(included):
        _exact_native_dict(item, _NATIVE_RANGE_KEYS, f"coverage.included_ranges[{index}]")
    omitted = coverage.get("omitted_ranges")
    if not isinstance(omitted, list):
        raise ProtocolContractError("coverage.omitted_ranges must be a list")
    for index, item in enumerate(omitted):
        _exact_native_dict(item, _NATIVE_OMITTED_KEYS, f"coverage.omitted_ranges[{index}]")


def _reject_native_structure(view: AigNativeViewDraft | AigNativeViewRevision) -> None:
    high = [item for item in validate_native_view(view) if item.severity == "high"]
    if high:
        codes = ", ".join(item.code for item in high)
        raise ProtocolContractError(f"native-edge document failed structural checks: {codes}")


def _reject_claim_cut_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in V2_CLAIM_CUT_FIELDS:
                raise ProtocolContractError(f"v2 write rejects Claim/Cut field {key!r}")
            _reject_claim_cut_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_claim_cut_fields(item)


def _require_exact_keys(
    payload: Mapping[str, Any],
    allowed: frozenset[str],
    label: str,
    *,
    optional: frozenset[str] = frozenset(),
) -> None:
    permitted = allowed | optional
    unexpected = [key for key in payload if key not in permitted]
    if unexpected:
        raise ProtocolContractError(f"unexpected field {unexpected[0]!r}")
    missing = [key for key in sorted(allowed) if key not in payload]
    if missing:
        raise ProtocolContractError(f"{label} missing field {missing[0]!r}")


def _require_safe_id(value: Any, field: str) -> str:
    text = _require_str(value, field)
    if not is_safe_id(text):
        raise ProtocolContractError(f"{field} must be a safe id, got {text!r}")
    return text


def _parse_criterion_dispositions(value: Any) -> tuple[CriterionDisposition, ...]:
    """Map criterion id to a closed disposition. Keys are contract criterion ids."""
    if not isinstance(value, dict):
        raise ProtocolContractError(
            "criteria_dispositions must be an object keyed by criterion id"
        )
    items: list[CriterionDisposition] = []
    for key in sorted(value):
        if not is_safe_id(key):
            raise ProtocolContractError(
                f"criteria_dispositions key {key!r} must be a criterion id"
            )
        disposition = _require_choice(
            value[key], f"criteria_dispositions.{key}", _CRITERIA_DISPOSITIONS
        )
        items.append(CriterionDisposition(criterion_id=key, disposition=disposition))
    return tuple(items)


def _parse_nonempty_str_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or isinstance(value, (str, bytes)) or not value:
        raise ProtocolContractError(f"{field} must be a non-empty list")
    return _parse_str_list(value, field)


def _parse_acceptance_criteria(value: Any) -> tuple[AcceptanceCriterion, ...]:
    records = _parse_id_records(
        value,
        "acceptance_criteria",
        _ACCEPTANCE_CRITERION_KEYS,
        "criterion_id",
    )
    return tuple(
        AcceptanceCriterion(
            criterion_id=record["criterion_id"],
            statement=record["statement"],
        )
        for record in records
    )


def _parse_evaluator_bindings(value: Any) -> tuple[EvaluatorBinding, ...]:
    records = _parse_id_records(
        value,
        "evaluator_bindings",
        _EVALUATOR_BINDING_KEYS,
        "binding_id",
    )
    return tuple(
        EvaluatorBinding(
            binding_id=record["binding_id"],
            kind=_require_choice(record.get("kind"), "evaluator_bindings.kind", _BINDING_KINDS),
            ref=record["ref"],
        )
        for record in records
    )


def _parse_policy_constraints(value: Any) -> tuple[PolicyConstraint, ...]:
    records = _parse_id_records(
        value,
        "policy_constraints",
        _POLICY_CONSTRAINT_KEYS,
        "constraint_id",
    )
    return tuple(
        PolicyConstraint(
            constraint_id=record["constraint_id"],
            kind=_require_choice(record.get("kind"), "policy_constraints.kind", _CONSTRAINT_KINDS),
            statement=record["statement"],
        )
        for record in records
    )


def _parse_id_records(
    value: Any,
    field: str,
    keys: frozenset[str],
    id_field: str,
) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or isinstance(value, (str, bytes)):
        raise ProtocolContractError(f"{field} must be a list")
    seen: set[str] = set()
    records: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ProtocolContractError(f"{field}[{index}] must be a dict")
        _require_exact_keys(item, keys, f"{field}[{index}]")
        identity = _require_safe_id(item.get(id_field), f"{field}[{index}].{id_field}")
        if identity in seen:
            raise ProtocolContractError(f"duplicate {id_field} {identity!r}")
        seen.add(identity)
        parsed = {
            key: _require_str(item.get(key), f"{field}[{index}].{key}")
            for key in keys
            if key != id_field
        }
        parsed[id_field] = identity
        records.append(parsed)
    return tuple(records)


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolContractError(f"{field} must be a non-empty str")
    return value


def _require_choice(value: Any, field: str, choices: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ProtocolContractError(f"{field} must be one of {sorted(choices)}, got {value!r}")
    return value


def _require_reader_author(value: Any, document: str) -> str:
    author = _require_str(value, "author_role")
    if author != _READER_ROLE:
        raise ProtocolContractError(
            f"{document} author_role must be reader, got {author!r}"
        )
    return author


def _require_catalog(catalog: BaseRevisionCatalog) -> None:
    if not isinstance(catalog, BaseRevisionCatalog):
        raise ProtocolContractError("v2 fork grounding requires a base revision catalog")


def _require_same_base(revision_id: str, catalog: BaseRevisionCatalog, field: str) -> None:
    if revision_id != catalog.base_revision_id:
        raise ProtocolContractError(
            f"{field} {revision_id!r} does not match base revision {catalog.base_revision_id!r}"
        )


def _require_defined_imagined_nodes(
    overlay: ForkNodeOverlay, report: GuidanceExecutionReport
) -> None:
    defined = {
        step.imagined_node_id
        for step in overlay.branch_steps
        if step.step_kind == _STEP_KIND_IMAGINED and step.imagined_node_id is not None
    }
    for step in report.steps:
        if step.imagined_node_id not in defined:
            raise ProtocolContractError(
                "guidance_execution_report imagined_node_id "
                f"{step.imagined_node_id!r} is not defined on fork_node_overlay"
            )


def _evaluation_status(body: Mapping[str, Any]) -> str:
    if "status" not in body:
        return _EVALUATION_STATUS_ACTIVE
    return _require_choice(body.get("status"), "status", _EVALUATION_STATUSES)


def _optional_explanation(body: Mapping[str, Any]) -> str | None:
    if "equivalence_explanation" not in body or body.get("equivalence_explanation") is None:
        return None
    return _require_str(body.get("equivalence_explanation"), "equivalence_explanation")


def _optional_semantic_summary(body: Mapping[str, Any]) -> str | None:
    if "semantic_summary" not in body:
        return None
    return _require_str(body.get("semantic_summary"), "semantic_summary")


def _parse_applicability_conditions(value: Any) -> tuple[ApplicabilityCondition, ...]:
    records = _parse_id_records(
        value,
        "applicability_conditions",
        _APPLICABILITY_CONDITION_KEYS,
        "condition_id",
    )
    if not records:
        raise ProtocolContractError("applicability_conditions must be a non-empty list")
    return tuple(
        ApplicabilityCondition(
            condition_id=record["condition_id"],
            statement=record["statement"],
        )
        for record in records
    )


def _condition_records(
    value: Sequence[ApplicabilityCondition | Mapping[str, str]],
) -> list[dict[str, str]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProtocolContractError("applicability_conditions must be a non-empty list")
    raw: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, ApplicabilityCondition):
            raw.append({"condition_id": item.condition_id, "statement": item.statement})
        elif isinstance(item, Mapping):
            raw.append(dict(item))
        else:
            raise ProtocolContractError(f"applicability_conditions[{index}] must be a dict")
    parsed = _parse_applicability_conditions(raw)
    ordered = sorted(parsed, key=lambda condition: condition.condition_id)
    return [
        {"condition_id": condition.condition_id, "statement": condition.statement}
        for condition in ordered
    ]


def _parse_claim(value: Any) -> InfluenceClaim:
    if not isinstance(value, dict):
        raise ProtocolContractError("claim must be a dict")
    _require_exact_keys(value, _CLAIM_KEYS, "claim")
    if not isinstance(value.get("source_evidence_refs"), list):
        raise ProtocolContractError("claim.source_evidence_refs must be a list")
    if not isinstance(value.get("target_evidence_refs"), list):
        raise ProtocolContractError("claim.target_evidence_refs must be a list")
    try:
        return InfluenceClaim.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolContractError(str(exc)) from exc


def _parse_cut_grounding(value: Any) -> InfluenceGrounding:
    if not isinstance(value, dict):
        raise ProtocolContractError("cut_grounding must be a dict")
    _require_exact_keys(value, _GROUNDING_KEYS, "cut_grounding")
    try:
        grounding = InfluenceGrounding.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolContractError(str(exc)) from exc
    if grounding.position_kind != "cut":
        raise ProtocolContractError(
            "legacy_claim_cut cut_grounding position_kind must be 'cut', "
            f"got {grounding.position_kind!r}"
        )
    return grounding


def _parse_catalog(value: Any) -> BaseRevisionCatalog:
    if not isinstance(value, dict):
        raise ProtocolContractError("base_revision must be a dict")
    _require_exact_keys(value, _CATALOG_KEYS, "base_revision")
    block_ids = value.get("block_ids")
    if not isinstance(block_ids, list) or isinstance(block_ids, (str, bytes)):
        raise ProtocolContractError("block_ids must be a list")
    if not block_ids:
        raise ProtocolContractError("block_ids must be a non-empty list")
    seen: set[str] = set()
    ordered: list[str] = []
    for index, item in enumerate(block_ids):
        block_id = _require_str(item, f"block_ids[{index}]")
        if block_id in seen:
            raise ProtocolContractError(f"duplicate block_id {block_id!r}")
        seen.add(block_id)
        ordered.append(block_id)
    return BaseRevisionCatalog(
        base_revision_id=_require_str(value.get("base_revision_id"), "base_revision_id"),
        block_content_hash=_require_str(
            value.get("block_content_hash"), "base_revision.block_content_hash"
        ),
        block_ids=tuple(ordered),
    )


def _parse_fork_common(
    body: Mapping[str, Any],
) -> tuple[str, str, str, tuple[BranchStepRef, ...]]:
    """Parse the fork anchor fields shared by proposal and overlay."""
    return (
        _require_str(body.get("base_revision_id"), "base_revision_id"),
        _require_str(
            body.get("base_revision_block_content_hash"),
            "base_revision_block_content_hash",
        ),
        _require_str(body.get("anchor_block_id"), "anchor_block_id"),
        _parse_branch_steps(body.get("branch_steps")),
    )


def _parse_branch_steps(value: Any) -> tuple[BranchStepRef, ...]:
    if not isinstance(value, list) or not value:
        raise ProtocolContractError("branch_steps must be a non-empty list")
    steps: list[BranchStepRef] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ProtocolContractError(f"branch_steps[{index}] must be a dict")
        kind = item.get("step_kind")
        label = f"branch_steps[{index}]"
        if kind == _STEP_KIND_FACTUAL:
            _require_exact_keys(item, frozenset({"step_kind", "block_id"}), label)
            steps.append(
                BranchStepRef(
                    step_kind=kind,
                    block_id=_require_str(item.get("block_id"), f"{label}.block_id"),
                )
            )
        elif kind == _STEP_KIND_IMAGINED:
            _require_exact_keys(item, frozenset({"step_kind", "imagined_node_id"}), label)
            steps.append(
                BranchStepRef(
                    step_kind=kind,
                    imagined_node_id=_require_str(
                        item.get("imagined_node_id"), f"{label}.imagined_node_id"
                    ),
                )
            )
        else:
            raise ProtocolContractError(
                f"{label}.step_kind must be 'factual_block_ref' or 'imagined_node', got {kind!r}"
            )
    return tuple(steps)


def _parse_evidence_refs(value: Any) -> tuple[EvidenceRef, ...]:
    if not isinstance(value, list):
        raise ProtocolContractError("evidence_refs must be a list")
    refs: list[EvidenceRef] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ProtocolContractError(f"evidence_refs[{index}] must be a dict")
        _require_exact_keys(item, _EVIDENCE_REF_KEYS, f"evidence_refs[{index}]")
        try:
            refs.append(EvidenceRef.from_dict(item))
        except (TypeError, ValueError) as exc:
            raise ProtocolContractError(str(exc)) from exc
    return tuple(refs)


def _parse_str_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ProtocolContractError(f"{field} must be a list")
    items: list[str] = []
    for index, item in enumerate(value):
        items.append(_require_str(item, f"{field}[{index}]"))
    return tuple(items)


def _parse_str_map(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ProtocolContractError(f"{field} must be a dict")
    parsed: dict[str, str] = {}
    for key, item in value.items():
        name = _require_str(key, f"{field} key")
        parsed[name] = _require_str(item, f"{field}.{name}")
    return parsed


_STEP_ALIGNMENT_KEYS = frozenset({"imagined_node_id", "alignment", "actual_block_id"})


def _parse_one_step_alignment(item: Any, label: str) -> GuidanceStepAlignment:
    if not isinstance(item, dict):
        raise ProtocolContractError(f"{label} must be a dict")
    _require_exact_keys(item, _STEP_ALIGNMENT_KEYS, label)
    alignment = _require_choice(item.get("alignment"), f"{label}.alignment", _ALIGNMENTS)
    actual = item.get("actual_block_id")
    if alignment == ALIGNMENT_NOT_EXECUTED:
        if actual is not None:
            raise ProtocolContractError(
                "not_executed alignment requires actual_block_id to be null"
            )
        actual_block_id = None
    else:
        actual_block_id = _require_str(actual, f"{label}.actual_block_id")
    return GuidanceStepAlignment(
        imagined_node_id=_require_str(item.get("imagined_node_id"), f"{label}.imagined_node_id"),
        alignment=alignment,
        actual_block_id=actual_block_id,
    )


def _parse_step_alignments(
    value: Any, field: str, *, allow_empty: bool
) -> tuple[GuidanceStepAlignment, ...]:
    if not isinstance(value, list):
        raise ProtocolContractError(f"{field} must be a list")
    if not value and not allow_empty:
        raise ProtocolContractError(f"{field} must be a non-empty list")
    return tuple(
        _parse_one_step_alignment(item, f"{field}[{index}]") for index, item in enumerate(value)
    )


def _parse_execution_steps(value: Any) -> tuple[GuidanceStepAlignment, ...]:
    return _parse_step_alignments(value, "steps", allow_empty=False)


def _reject_duplicate_step_ids(
    steps: Sequence[GuidanceStepAlignment], field: str
) -> None:
    seen: set[str] = set()
    for step in steps:
        if step.imagined_node_id in seen:
            raise ProtocolContractError(f"{field} imagined_node_id must be unique")
        seen.add(step.imagined_node_id)


def _copy_step_alignment_mappings(
    value: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProtocolContractError("step_alignments must be a list")
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ProtocolContractError("step_alignments must be a list of dicts")
        rows.append(dict(item))
    return rows


def _seal_step_alignments(
    steps: Sequence[GuidanceStepAlignment],
) -> tuple[GuidanceStepAlignment, ...]:
    sealed: list[GuidanceStepAlignment] = []
    for step in steps:
        if not isinstance(step, GuidanceStepAlignment):
            raise ProtocolContractError("step alignment must be a GuidanceStepAlignment")
        sealed.append(step)
    return tuple(sealed)


def _public_step_alignments(
    steps: Sequence[GuidanceStepAlignment],
) -> list[dict[str, Any]]:
    return [
        {
            "actual_block_id": step.actual_block_id,
            "alignment": step.alignment,
            "imagined_node_id": step.imagined_node_id,
        }
        for step in steps
    ]


def _imagined_ids_from_branch(branch_steps: Sequence[BranchStepRef]) -> list[str]:
    ids: list[str] = []
    for step in branch_steps:
        if not isinstance(step, BranchStepRef):
            raise ProtocolContractError("branch step must be a BranchStepRef")
        if step.step_kind == STEP_KIND_IMAGINED and step.imagined_node_id:
            ids.append(step.imagined_node_id)
    return ids


def _canonical_block_ids(values: Sequence[str]) -> list[str]:
    ids = [_require_str(item, "replay_block_ids") for item in values]
    return sorted(set(ids))
