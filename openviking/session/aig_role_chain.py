# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Run-scoped AIG Builder / Critic / Refiner / Reader chain.

Roles do not share sessions. Reader and Diagnosis share one session and a
phase gate. A failed run stays on the deterministic fallback and does not
switch to the fixed-partition IG builder. Evidence exposure starts closed
and climbs projection, summary, synopsis, then selected_raw. selected_raw
only records that the upgrade was authorized; it does not carry raw AO text.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any
from uuid import uuid4

MAX_REVISION_ROUNDS = 3
CONSTRUCTION_ADAPTIVE = "adaptive_agentic_aig"
CONSTRUCTION_FALLBACK = "fallback"
CONSTRUCTION_IG = "fixed_partition_single_pass_ig"

_BUILD_ROLES = frozenset({"builder", "critic", "refiner", "reader"})
_FORK_KEY = "fork"
_EVIDENCE_LADDER = ("projection", "summary", "synopsis", "selected_raw")


class AigContractError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(body: Any) -> str:
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


class AigConstructionRun:
    def __init__(self, run_id: str) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise AigContractError("invalid_run")
        self.run_id = run_id.strip()
        self._construction_mode = CONSTRUCTION_ADAPTIVE
        self.trust = "draft"
        self._sessions = {role: f"{self.run_id}:{role}:{uuid4().hex}" for role in _BUILD_ROLES}
        self._rounds = {"critic": 0, "refiner": 0}
        self._draft_hash: str | None = None
        self._violations: list[Any] = []
        self._report_hash: str | None = None
        self._report_frozen = False
        self._view_frozen = False
        self._phase = "BUILD"
        self._fork: dict[str, Any] | None = None
        self._fallback = False
        self._evidence_tier: str | None = None

    @property
    def construction_mode(self) -> str:
        return self._construction_mode

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def reader_report_hash(self) -> str | None:
        return self._report_hash

    @property
    def evidence_tier(self) -> str | None:
        return self._evidence_tier

    def session_for(self, role: str) -> str:
        if role == "memory":
            raise AigContractError("memory_session_not_shared")
        if role == "diagnosis":
            role = "reader"
        if role not in self._sessions:
            raise AigContractError("unknown_role")
        return self._sessions[role]

    def submit_draft(self, session_id: str, draft: Any) -> str:
        self._require_session(session_id, "builder")
        self._require_mutable_view()
        self._draft_hash = content_hash(draft)
        self._phase = "CRITIQUE"
        return self._draft_hash

    def critique(self, session_id: str, violations: Any) -> int:
        self._require_session(session_id, "critic")
        self._require_open_draft()
        self._require_mutable_view()
        next_round = self._rounds["critic"] + 1
        if next_round > MAX_REVISION_ROUNDS:
            raise AigContractError("critic_round_limit")
        if not isinstance(violations, (list, tuple)):
            raise AigContractError("invalid_violations")
        self._rounds["critic"] = next_round
        self._violations = copy.deepcopy(list(violations))
        return next_round

    def refine(self, session_id: str, draft: Any) -> str:
        self._require_session(session_id, "refiner")
        self._require_open_draft()
        self._require_mutable_view()
        next_round = self._rounds["refiner"] + 1
        if next_round > MAX_REVISION_ROUNDS:
            raise AigContractError("refiner_round_limit")
        self._rounds["refiner"] = next_round
        self._draft_hash = content_hash(draft)
        return self._draft_hash

    def read_verify(self, session_id: str, report: Any) -> str:
        self._require_session(session_id, "reader")
        self._require_open_draft()
        if self._report_frozen:
            raise AigContractError("frozen_view")
        if isinstance(report, dict) and _FORK_KEY in report:
            raise AigContractError("read_verify_cannot_propose_fork")
        self._phase = "READ_VERIFY"
        self._report_hash = content_hash(
            {
                "draftHash": self._draft_hash,
                "violations": copy.deepcopy(self._violations),
                "report": copy.deepcopy(report),
            }
        )
        self._view_frozen = True
        return self._report_hash

    def escalate_evidence(self, session_id: str, tier: str) -> str:
        self._require_session(session_id, "reader")
        if self._fallback:
            raise AigContractError("fallback_is_provisional")
        if self._report_frozen:
            raise AigContractError("frozen_view")
        if self._phase != "READ_VERIFY":
            raise AigContractError("evidence_not_in_read_verify")
        if not isinstance(tier, str) or tier not in _EVIDENCE_LADDER:
            raise AigContractError("invalid_evidence_tier")
        current = -1 if self._evidence_tier is None else _EVIDENCE_LADDER.index(self._evidence_tier)
        requested = _EVIDENCE_LADDER.index(tier)
        if requested != current + 1:
            raise AigContractError("evidence_tier_skip")
        granted = _EVIDENCE_LADDER[requested]
        self._evidence_tier = granted
        return granted

    def freeze_report(self, session_id: str) -> str:
        self._require_session(session_id, "reader")
        if self._fallback:
            raise AigContractError("fallback_is_provisional")
        if self._report_hash is None:
            raise AigContractError("no_report")
        self._report_frozen = True
        self._view_frozen = True
        self._phase = "SYNTHESIZE_PROPOSE"
        self.trust = "frozen"
        return self._report_hash

    def propose_fork(self, session_id: str, fork_draft: dict[str, Any]) -> dict[str, Any]:
        self._require_session(session_id, "reader")
        if self._fallback:
            raise AigContractError("fallback_is_provisional")
        if self._phase == "READ_VERIFY" or not self._report_frozen:
            raise AigContractError("report_not_frozen")
        if not isinstance(fork_draft, dict) or _FORK_KEY not in fork_draft:
            raise AigContractError("invalid_fork")
        self._fork = {
            "fork": copy.deepcopy(fork_draft[_FORK_KEY]),
            "readerReportHash": self._report_hash,
        }
        return copy.deepcopy(self._fork)

    def fail(self) -> dict[str, str]:
        self._fallback = True
        self._construction_mode = CONSTRUCTION_FALLBACK
        self.trust = "provisional"
        self._phase = "FALLBACK"
        if self._construction_mode == CONSTRUCTION_IG:
            raise AigContractError("ig_fallback_forbidden")
        return {"constructionMode": self._construction_mode, "trust": self.trust}

    def _require_open_draft(self) -> None:
        if self._draft_hash is None or self._fallback:
            raise AigContractError("no_draft")

    def _require_mutable_view(self) -> None:
        if self._view_frozen or self._report_frozen or self._fallback:
            raise AigContractError("frozen_view")

    def _require_session(self, session_id: str, role: str) -> None:
        if session_id != self._sessions[role]:
            raise AigContractError("session_role_mismatch")
