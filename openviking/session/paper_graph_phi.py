# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Paper-exact ``Phi(G)`` for a submitted paper-schema draft (ADR-0044).

The paper requires connectivity, acyclicity, one task root, and one conclusion
sink, and it does not define the directed connectivity predicate. This module
records that reading as ``paper-exact-connectivity-v1``: every node is
reachable from the task root along inheritance edges. Passage is structural
only. It does not judge summary fidelity or whether an effect note is true.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

VALIDATOR_PROFILE = "paper_exact"
CONNECTIVITY_POLICY_VERSION = "paper-exact-connectivity-v1"
CONNECTIVITY_INTERPRETATION = (
    "The paper requires connectivity and does not define the directed graph "
    "predicate. paper-exact-connectivity-v1 reads connectivity as directed "
    "reachability from the task root: every node is reachable from that root "
    "along inheritance edges. ADR-0018 requires this reading to be versioned "
    "instead of claimed as the paper's own wording. Structural passage is not "
    "semantic or causal validation."
)
_PAPER_KINDS = frozenset({"task", "orchestrator", "agent", "conclusion"})
_KINDS = frozenset({"structural", "semantic", "granularity"})
_SEVERITIES = frozenset({"high", "low"})
_SCOPES = frozenset({"graph", "objects"})


class PhiError(ValueError):
    """The draft is not a paper graph. ``code`` is stable."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class PaperViolation:
    """One critic-loop violation. Structural rows come from ``Phi(G)``."""

    violation_id: str
    object_id: str
    kind: str
    severity: str
    evidence: str
    evidence_ao_refs: tuple[int, ...] = ()
    note: str = ""
    scope: str = "objects"
    involved_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.violation_id, str) or not self.violation_id:
            raise PhiError("invalid_violation")
        if not isinstance(self.object_id, str) or not self.object_id:
            raise PhiError("invalid_violation")
        if self.kind not in _KINDS or self.severity not in _SEVERITIES:
            raise PhiError("invalid_violation")
        if not isinstance(self.evidence, str) or not self.evidence:
            raise PhiError("invalid_violation")
        if self.scope not in _SCOPES:
            raise PhiError("invalid_violation")
        if any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in self.evidence_ao_refs
        ):
            raise PhiError("invalid_violation")
        if not isinstance(self.note, str):
            raise PhiError("invalid_violation")
        if any(not isinstance(item, str) or not item for item in self.involved_ids):
            raise PhiError("invalid_violation")

    def to_dict(self) -> dict[str, Any]:
        """Audit summary. Scope stays on the server and is not stored."""
        payload: dict[str, Any] = {
            "evidence": self.evidence,
            "kind": self.kind,
            "object_id": self.object_id,
            "severity": self.severity,
            "violation_id": self.violation_id,
        }
        if self.evidence_ao_refs:
            payload["evidence_ao_refs"] = list(self.evidence_ao_refs)
        if self.note:
            payload["note"] = self.note
        return payload

    def turn_dict(self) -> dict[str, Any]:
        """What a refiner is shown, including the server scope annotation."""
        payload = self.to_dict()
        payload["involved_ids"] = list(self.involved_ids)
        payload["scope"] = self.scope
        return payload


@dataclass(frozen=True)
class PhiReport:
    """``Phi(G)`` plus the versioned connectivity reading."""

    profile: str
    connectivity_policy_version: str
    connectivity_interpretation: str
    violations: tuple[PaperViolation, ...]

    def policy_dict(self) -> dict[str, str]:
        return {
            "connectivity_interpretation": self.connectivity_interpretation,
            "connectivity_policy_version": self.connectivity_policy_version,
            "profile": self.profile,
        }


def evaluate_phi(graph: Mapping[str, Any]) -> PhiReport:
    """Run paper-exact ``Phi(G)``. An empty violation tuple is a pass."""
    nodes, edges = _graph(graph)
    findings: list[PaperViolation] = []
    by_kind: dict[str, list[str]] = {kind: [] for kind in _PAPER_KINDS}
    incoming: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for source, target in edges:
        outgoing[source].append(target)
        incoming[target].append(source)
    for node_id, kind in nodes.items():
        by_kind[kind].append(node_id)
    tasks = sorted(by_kind["task"])
    conclusions = sorted(by_kind["conclusion"])
    if len(tasks) == 0:
        findings.append(
            _structural(
                "phi:missing_task_root",
                "graph",
                "paper_exact requires exactly one task node with no incoming edge",
                scope="graph",
            )
        )
    elif len(tasks) > 1:
        findings.append(
            _structural(
                "phi:duplicate_task_root",
                "graph",
                "paper_exact requires exactly one task root",
                scope="objects",
                involved=tuple(tasks),
            )
        )
    else:
        task_id = tasks[0]
        sources = tuple(sorted(set(incoming[task_id])))
        if sources:
            findings.append(
                _structural(
                    f"phi:task_has_incoming:{task_id}",
                    task_id,
                    "the task node has an incoming edge so it is not a root",
                    scope="objects",
                    involved=(task_id, *sources),
                )
            )
    if len(conclusions) == 0:
        findings.append(
            _structural(
                "phi:missing_conclusion_sink",
                "graph",
                "paper_exact requires exactly one conclusion node with no outgoing edge",
                scope="graph",
            )
        )
    elif len(conclusions) > 1:
        findings.append(
            _structural(
                "phi:duplicate_conclusion_sink",
                "graph",
                "paper_exact requires exactly one conclusion sink",
                scope="objects",
                involved=tuple(conclusions),
            )
        )
    else:
        conclusion_id = conclusions[0]
        targets = tuple(sorted(set(outgoing[conclusion_id])))
        if targets:
            findings.append(
                _structural(
                    f"phi:conclusion_has_outgoing:{conclusion_id}",
                    conclusion_id,
                    "the conclusion node has an outgoing edge so it is not a sink",
                    scope="objects",
                    involved=(conclusion_id, *targets),
                )
            )
    task_ids = set(tasks)
    conclusion_ids = set(conclusions)
    for node_id in sorted(nodes):
        if node_id not in task_ids and not incoming[node_id]:
            findings.append(
                _structural(
                    f"phi:extra_root:{node_id}",
                    node_id,
                    "a non-task node has no incoming edge",
                    scope="objects",
                    involved=(node_id,),
                )
            )
        if node_id not in conclusion_ids and not outgoing[node_id]:
            findings.append(
                _structural(
                    f"phi:extra_sink:{node_id}",
                    node_id,
                    "a non-conclusion node has no outgoing edge",
                    scope="objects",
                    involved=(node_id,),
                )
            )
    for source, target in _cycles(nodes, edges):
        findings.append(
            _structural(
                f"phi:cycle:{source}->{target}",
                f"{source}->{target}",
                "inheritance edges contain a directed cycle",
                scope="objects",
                involved=(source, target),
            )
        )
    if len(tasks) == 1:
        reached = _reachable(tasks[0], edges)
        for node_id in sorted(set(nodes) - reached):
            findings.append(
                _structural(
                    f"phi:unreachable:{node_id}",
                    node_id,
                    "the node is not reachable from the task root along inheritance edges",
                    scope="graph",
                    involved=(node_id,),
                )
            )
    findings.sort(key=lambda item: item.violation_id)
    return PhiReport(
        profile=VALIDATOR_PROFILE,
        connectivity_policy_version=CONNECTIVITY_POLICY_VERSION,
        connectivity_interpretation=CONNECTIVITY_INTERPRETATION,
        violations=tuple(findings),
    )


def _structural(
    violation_id: str,
    object_id: str,
    evidence: str,
    *,
    scope: str,
    involved: Sequence[str] = (),
) -> PaperViolation:
    return PaperViolation(
        violation_id=violation_id,
        object_id=object_id,
        kind="structural",
        severity="high",
        evidence=evidence,
        scope=scope,
        involved_ids=tuple(involved),
    )


def _graph(graph: Mapping[str, Any]) -> tuple[dict[str, str], tuple[tuple[str, str], ...]]:
    if not isinstance(graph, Mapping):
        raise PhiError("invalid_graph")
    raw_blocks = graph.get("blocks")
    raw_edges = graph.get("edges")
    if not isinstance(raw_blocks, list) or not isinstance(raw_edges, list):
        raise PhiError("invalid_graph")
    nodes: dict[str, str] = {}
    for item in raw_blocks:
        if not isinstance(item, Mapping):
            raise PhiError("invalid_graph")
        block_id = item.get("block_id")
        kind = item.get("kind")
        if not isinstance(block_id, str) or not block_id or block_id in nodes:
            raise PhiError("invalid_graph")
        if kind not in _PAPER_KINDS:
            raise PhiError("invalid_graph")
        nodes[block_id] = kind
    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_edges:
        if not isinstance(item, Mapping):
            raise PhiError("invalid_graph")
        source = item.get("from")
        target = item.get("to")
        if not isinstance(source, str) or not isinstance(target, str):
            raise PhiError("invalid_graph")
        if source not in nodes or target not in nodes:
            raise PhiError("invalid_graph")
        pair = (source, target)
        if pair in seen:
            raise PhiError("invalid_graph")
        seen.add(pair)
        edges.append(pair)
    return nodes, tuple(edges)


def _reachable(root: str, edges: Sequence[tuple[str, str]]) -> set[str]:
    graph: dict[str, set[str]] = {}
    for source, target in edges:
        graph.setdefault(source, set()).add(target)
    seen: set[str] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph.get(node, set()) - seen)
    return seen


def _cycles(
    nodes: Mapping[str, str], edges: Sequence[tuple[str, str]]
) -> tuple[tuple[str, str], ...]:
    graph: dict[str, list[str]] = {}
    for source, target in edges:
        graph.setdefault(source, []).append(target)
    color = {node_id: 0 for node_id in nodes}
    found: list[tuple[str, str]] = []

    def walk(node: str) -> None:
        color[node] = 1
        for nxt in graph.get(node, ()):
            if color[nxt] == 1:
                found.append((node, nxt))
            elif color[nxt] == 0:
                walk(nxt)
        color[node] = 2

    for node_id in sorted(nodes):
        if color[node_id] == 0:
            walk(node_id)
    unique = sorted(set(found))
    return tuple(unique)
