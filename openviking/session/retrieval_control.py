# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Retrieval control tree (Q143-A, Q144-B, Q145-A, Q146-B).

A parent lease is split, before dispatch, into frozen child leases and
child deadlines. A child failure ends only that lane. Parent cancel, the
parent deadline, and shutdown end every lane that is still running.
Completed siblings keep their public results. A terminated lane publishes
nothing. Memory has no exact control surface.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from openviking.session.memory_role_retrieval import EXACT_CAPABILITY_ABSENT, MEMORY_ROLE

CONTROL_POLICY_VERSION = (
    "memory-control-budget/Q143-A"
    "+memory-deadline/Q144-B"
    "+memory-cancellation/Q145-A"
)
COMPLETENESS_POLICY_VERSION = "memory-partial-hint/Q146-B"
EXACT_LANE = "current_log_exact"
DISCOVERY_LANE = "memory_discovery"
PARENT_CANCEL_CAUSES = frozenset(
    {"client_abort", "orchestrator_cancel", "shutdown", "revocation"}
)


class ControlError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ControlPolicy:
    """Server-owned windows and meters. Callers do not author these."""

    exact_window_ms: int
    discovery_window_ms: int
    exact_budget: int
    discovery_budget: int


@dataclass
class _Lane:
    name: str
    lease_ref: str
    deadline_ms: int
    budget: int
    execution: str = "running"
    reason: str | None = None
    public: list[str] | None = None
    withheld: list[str] | None = None
    secondary: list[str] | None = None

    def __post_init__(self) -> None:
        self.public = []
        self.withheld = []
        self.secondary = []


class RetrievalControl:
    def __init__(
        self,
        *,
        role: str,
        parent_lease_ref: str,
        parent_deadline_ms: int,
        now_ms: int,
        dispatched: frozenset[str],
        policy: ControlPolicy,
    ) -> None:
        if role == MEMORY_ROLE and EXACT_LANE in dispatched:
            raise ControlError(EXACT_CAPABILITY_ABSENT)
        unknown = dispatched - {EXACT_LANE, DISCOVERY_LANE}
        if not dispatched or unknown:
            raise ControlError("invalid_dispatch")
        self.role = role
        self.parent_lease_ref = parent_lease_ref
        self.parent_deadline_ms = parent_deadline_ms
        self.now_ms = now_ms
        self._parent_reason: str | None = None
        self.lanes = {
            name: _open_lane(parent_lease_ref, name, parent_deadline_ms, now_ms, policy)
            for name in sorted(dispatched)
        }
        self.expansion = _expansion_binding(parent_lease_ref, parent_deadline_ms, now_ms)

    def remaining_ms(self, lane: str, now_ms: int) -> int:
        self._advance(now_ms)
        state = self._lane(lane)
        if state.execution != "running":
            return 0
        return max(0, self._effective(state) - self.now_ms)

    def charge(self, lane: str, amount: int, now_ms: int) -> None:
        self.checkpoint(lane, now_ms)
        state = self._lane(lane)
        if state.execution != "running":
            return
        if type(amount) is not int or amount < 1:
            raise ControlError("invalid_charge")
        if amount > state.budget:
            self._terminate(state, "budget_exhausted")
            return
        state.budget -= amount
        if state.budget == 0:
            self._terminate(state, "budget_exhausted")

    def stage(self, lane: str, hit_id: str) -> None:
        state = self._lane(lane)
        if state.execution == "running":
            state.public.append(hit_id)

    def complete(self, lane: str) -> None:
        state = self._lane(lane)
        if state.execution != "running":
            return
        state.execution = "completed"
        state.reason = None

    def child_signal(self, lane: str, cause: str, now_ms: int) -> None:
        self._advance(now_ms)
        if cause not in {"deadline_exceeded", "budget_exhausted", "dependency_unavailable"}:
            raise ControlError("invalid_cause")
        self._terminate(self._lane(lane), cause)

    def parent_signal(self, cause: str, now_ms: int) -> None:
        self._advance(now_ms)
        if cause in PARENT_CANCEL_CAUSES:
            reason = "cancelled"
        elif cause == "parent_deadline":
            reason = "deadline_exceeded"
        else:
            raise ControlError("invalid_cause")
        if self._parent_reason is None:
            self._parent_reason = reason
        for state in self.lanes.values():
            self._terminate(state, reason)

    def checkpoint(self, lane: str, now_ms: int) -> None:
        self._advance(now_ms)
        state = self._lane(lane)
        if self._parent_reason is not None:
            self._terminate(state, self._parent_reason)
            return
        if self.now_ms >= self.parent_deadline_ms:
            self.parent_signal("parent_deadline", self.now_ms)
            return
        if state.execution == "running" and self.now_ms >= self._effective(state):
            self._terminate(state, "deadline_exceeded")

    def is_running(self, lane: str) -> bool:
        return self._lane(lane).execution == "running"

    def public_view(self) -> dict[str, object]:
        lanes: dict[str, dict[str, object]] = {}
        for name, state in self.lanes.items():
            published = list(state.public) if state.execution == "completed" else []
            if state.execution == "terminated":
                result = "none"
                returned = 0
            elif published:
                result = "hits"
                returned = len(published)
            else:
                result = "empty"
                returned = 0
            lanes[name] = {
                "executionState": state.execution,
                "resultState": result,
                "reasonCode": state.reason,
                "returnedCount": returned,
                "truncated": False,
            }
        view: dict[str, object] = {
            "policyVersions": {
                "control": CONTROL_POLICY_VERSION,
                "completeness": COMPLETENESS_POLICY_VERSION,
            },
            "completeness": completeness(lanes),
            "lanes": lanes,
            "results": {
                name: list(state.public)
                for name, state in self.lanes.items()
                if state.execution == "completed"
            },
        }
        if self.role == MEMORY_ROLE:
            view["lanes"] = {DISCOVERY_LANE: lanes[DISCOVERY_LANE]}
            view["results"] = {DISCOVERY_LANE: view["results"].get(DISCOVERY_LANE, [])}
        return view

    def native_audit(self, lane: str) -> dict[str, object]:
        state = self._lane(lane)
        return {
            "leaseRef": state.lease_ref,
            "secondarySignals": list(state.secondary),
            "withheldHitIds": list(state.withheld),
        }

    def _lane(self, lane: str) -> _Lane:
        try:
            return self.lanes[lane]
        except KeyError as exc:
            raise ControlError("lane_not_dispatched") from exc

    def _advance(self, now_ms: int) -> None:
        if type(now_ms) is not int:
            raise ControlError("invalid_clock")
        self.now_ms = max(self.now_ms, now_ms)

    def _effective(self, state: _Lane) -> int:
        return min(self.parent_deadline_ms, state.deadline_ms)

    def _terminate(self, state: _Lane, reason: str) -> None:
        if state.execution == "completed":
            return
        if state.execution == "terminated":
            if reason not in state.secondary and reason != state.reason:
                state.secondary.append(reason)
            return
        state.execution = "terminated"
        state.reason = reason
        state.withheld.extend(state.public)
        state.public.clear()


def completeness(lanes: dict[str, dict[str, object]]) -> str:
    dispatched = [
        audit
        for audit in lanes.values()
        if audit.get("executionState") != "not_dispatched"
    ]
    if not dispatched:
        return "unavailable"
    states = [audit["executionState"] for audit in dispatched]
    if all(state == "completed" for state in states):
        return "complete"
    if any(state == "completed" for state in states) and any(
        state == "terminated" for state in states
    ):
        return "partial"
    if all(state == "terminated" for state in states):
        return "unavailable"
    return "partial"


def _open_lane(
    parent_lease_ref: str,
    name: str,
    parent_deadline_ms: int,
    now_ms: int,
    policy: ControlPolicy,
) -> _Lane:
    window = policy.exact_window_ms if name == EXACT_LANE else policy.discovery_window_ms
    budget = policy.exact_budget if name == EXACT_LANE else policy.discovery_budget
    if type(window) is not int or window < 1 or type(budget) is not int or budget < 1:
        raise ControlError("invalid_policy")
    deadline = min(parent_deadline_ms, now_ms + window)
    digest = hashlib.sha256(f"{parent_lease_ref}|{CONTROL_POLICY_VERSION}|{name}".encode()).hexdigest()
    return _Lane(name=name, lease_ref=f"child-{digest[:16]}", deadline_ms=deadline, budget=budget)


def _expansion_binding(parent_lease_ref: str, parent_deadline_ms: int, now_ms: int) -> dict[str, object]:
    digest = hashlib.sha256(
        f"{parent_lease_ref}|expansion|{CONTROL_POLICY_VERSION}".encode()
    ).hexdigest()
    return {
        "snapshot_ref": f"child-snap-{digest[:16]}",
        "budget_lease_ref": f"child-budget-{digest[:16]}",
        "deadline_ref": f"child-deadline-{digest[:16]}",
        "cancel_ref": f"child-cancel-{digest[:16]}",
        "deadline_ms": parent_deadline_ms,
        "opened_ms": now_ms,
        "cancelled": False,
    }
