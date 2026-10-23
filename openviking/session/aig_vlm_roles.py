# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Account VLM adapters for the four commit-time AIG roles.

The model proposes one JSON action per call. Server role surfaces validate
and execute every action. Rejected output gets bounded feedback within the
shared deadline; exhausted retries abort construction without repairing it.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from openviking.session.aig_production_pipeline import AigRoleProviders
from openviking.session.aig_tool_surface import TOOL_SCHEMAS, _REQUIRED_FIELDS, _OPTIONAL_FIELDS
from openviking.session.aig_toolloop_builder import ToolInvocation, ToolloopTurn
from openviking.session.aig_toolloop_critic import CriticTurn, CriticView, parse_critic_violations
from openviking.session.aig_toolloop_reader import READER_NEUTRAL_PROMPT, ReaderInvocation, ReaderTurn, parse_reader_report
from openviking.session.aig_toolloop_refiner import RefinerAction, RefinerTurn

logger = logging.getLogger(__name__)


def _json_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError("model_response_missing")

    fenced = re.fullmatch(
        r"```json[ \t]*\r?\n(.*?)\r?\n```", value.strip(), re.DOTALL | re.IGNORECASE
    )
    if fenced is not None:
        value = fenced.group(1)

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate_model_key")
            result[key] = item
        return result

    loaded = json.loads(
        value,
        object_pairs_hook=unique_pairs,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite_model_number")),
    )
    if not isinstance(loaded, dict):
        raise ValueError("model_response_not_object")
    return loaded


class _RoleModel:
    def __init__(self, handle: Any, deadline: float) -> None:
        self._handle = handle
        self._deadline = deadline

    def _output_schema(self, state: Mapping[str, Any]) -> dict[str, Any]:
        def obj(properties: dict[str, Any]) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            }

        tools = state.get("tools", [])
        variants = [
            obj(
                {
                    "verb": {"type": "string", "enum": [tool["name"]]},
                    "arguments": copy.deepcopy(tool["parameters"]),
                }
            )
            for tool in tools
        ]
        # add_block has kind-dependent fields. The flat tool schema allows
        # partial edits; it is not a complete creation contract.
        for variant in variants:
            if variant["properties"]["verb"]["enum"] != ["add_block"]:
                continue
            base = variant["properties"]["arguments"]
            branches = []
            for kind, required in _REQUIRED_FIELDS.items():
                branch = copy.deepcopy(base)
                branch["properties"]["kind"] = {"type": "string", "enum": [kind]}
                fields = branch["properties"]["fields"]
                allowed = set(required) | set(_OPTIONAL_FIELDS.get(kind, ()))
                fields["properties"] = {
                    key: value for key, value in fields["properties"].items() if key in allowed
                }
                fields["required"] = list(required)
                for value in fields["properties"].values():
                    if value.get("type") == "string":
                        value["minLength"] = 1
                branches.append(branch)
            if isinstance(self, VLMBuilder):
                rows = next((
                    entry["result"]["rows"] for entry in reversed(state.get("history", ()))
                    if entry.get("verb") == "get_overview" and entry.get("ok")
                    and isinstance(entry.get("result"), dict)
                    and isinstance(entry["result"].get("rows"), list)
                ), None)
                if rows is not None:
                    blocks = (state.get("draft") or {}).get("blocks", [])
                    occupied = {
                        ref for b in blocks if b.get("step_refs")
                        for ref in range(min(b["step_refs"]), max(b["step_refs"]) + 1)
                    }
                    existing_kinds = {b.get("kind") for b in blocks}
                    groups: dict[tuple[str, str], list[int]] = {}
                    for row in rows:
                        if row["index"] not in occupied:
                            groups.setdefault((row["participant"], row["segment"]), []).append(row["index"])
                    has_boundary = any(r["participant"] == "openviking:diagnosis-control" for r in rows)
                    bound_branches = []
                    for branch in branches:
                        kind = branch["properties"]["kind"]["enum"][0]
                        if kind in {"task", "conclusion"} and kind in existing_kinds:
                            continue
                        for (participant, _segment), refs in groups.items():
                            boundary = participant == "openviking:diagnosis-control"
                            if has_boundary and ((kind == "conclusion") != boundary):
                                continue
                            bound = copy.deepcopy(branch)
                            fields = bound["properties"]["fields"]["properties"]
                            fields["step_refs"].update({
                                "minItems": 1, "maxItems": len(refs),
                                "items": {"type": "integer", "enum": refs},
                            })
                            if "agent_name" in fields:
                                fields["agent_name"]["enum"] = [participant]
                            bound_branches.append(bound)
                    branches = bound_branches
            variant["properties"]["arguments"] = {"anyOf": branches}
        # No legal unoccupied rows means add_block is unavailable. The model
        # must link, revise, inspect or submit; the server still validates it.
        variants = [v for v in variants if v["properties"]["arguments"].get("anyOf") != []]
        # Empty or invented row references remain invalid at both boundaries.
        for variant in variants:
            args = variant["properties"]["arguments"]
            for branch in args.get("anyOf", [args]):
                fields = branch.get("properties", {}).get("fields", {})
                refs = fields.get("properties", {}).get("step_refs")
                if refs is not None:
                    refs["minItems"] = 1
                    refs["items"]["minimum"] = 0
        if isinstance(self, VLMCritic):
            block_ids = [
                b["block_id"] for b in (state.get("draft") or {}).get("blocks", ())
                if isinstance(b, dict) and isinstance(b.get("block_id"), str)
            ]
            item: dict[str, Any] = {"type": "object"}
            if block_ids:
                # One violation names exactly one existing block; others go in ``note``.
                item = obj({
                    "object_id": {"type": "string", "enum": block_ids},
                    "kind": {"type": "string", "enum": ["semantic", "granularity"]},
                    "severity": {"type": "string", "enum": ["high", "low"]},
                    "evidence_ao_refs": {"type": "array", "items": {"type": "integer", "minimum": 0}},
                    "note": {"type": "string", "minLength": 1},
                })
            variants.append(obj({"violations": {"type": "array", "items": item}}))
        elif isinstance(self, VLMRefiner):
            variants = [
                obj(
                    {
                        "action": {"enum": ["edit"], "type": "string"},
                        **variant["properties"],
                        "violation_id": {"type": "string"},
                    }
                )
                for variant in variants
            ]
            variants.extend(
                [
                    obj({"action": {"enum": ["done"], "type": "string"}}),
                    obj(
                        {
                            "action": {"enum": ["unresolvable"], "type": "string"},
                            "violation_id": {"type": "string"},
                            "reason": {"type": "string"},
                        }
                    ),
                ]
            )
        elif isinstance(self, VLMReader):
            # Bind each report branch to one real named work block. This is
            # validation data, never an inferred attribution or repaired output.
            graph = state.get("frozen_graph", {})
            for block in graph.get("blocks", []):
                name = block.get("agent_name")
                refs = block.get("step_refs")
                if (
                    (state.get("neutral_state_report") and block.get("kind") not in {"agent", "orchestrator"})
                    or not isinstance(name, str) or not name or not refs
                    or name == "openviking:diagnosis-control"
                ):
                    continue
                variants.append(obj({
                    "critical_block_id": {"type": "string", "enum": [block["block_id"]]},
                    "responsible_agent": {"type": "string", "enum": [name]},
                    "reason": {"type": "string", "minLength": 1},
                    "decisive_ao_refs": {"type": "array", "enum": [list(refs)]},
                    # A usable production report must have fully verified
                    # author/ref binding. Remaining task work belongs in reason;
                    # uncertain attribution must choose the abstention branch.
                    "unresolved_areas": {"type": "array", "maxItems": 0, "items": {"type": "string"}},
                }))
            # A provider may explicitly abstain. Production rejects unresolved
            # reports, including this branch, instead of publishing a diagnosis.
            variants.append(obj({
                "critical_block_id": {"type": "string", "enum": [""]},
                "responsible_agent": {"type": "string", "enum": [""]},
                "reason": {"type": "string", "minLength": 1},
                "decisive_ao_refs": {"type": "array", "maxItems": 0, "items": {"type": "integer"}},
                "unresolved_areas": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
            }))
        return {"type": "object", "anyOf": variants} if variants else {"type": "object"}

    def _ask(
        self,
        prompt: str,
        state: Mapping[str, Any],
        validate: Callable[[dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": prompt + "\nReturn exactly one JSON object, without markdown.",
            },
            {"role": "user", "content": json.dumps(state, ensure_ascii=False, default=str)},
        ]
        # The existing shared deadline includes every retry; no budget extension.
        for attempt in range(3):
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("aig_role_deadline_exceeded")

            async def complete(timeout_seconds: float):
                options: dict[str, Any] = {}
                # These model/provider paths are verified by real constrained
                # requests. Other handles retain strict parsing and bounded
                # error feedback.
                if getattr(self._handle, "model", None) in {
                    "google/gemma-4-31b-it",
                    "qwen/qwen3.5-27b",
                }:
                    options["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "aig_role_output",
                            "strict": True,
                            "schema": self._output_schema(state),
                        },
                    }
                return await asyncio.wait_for(
                    self._handle.get_completion_async(
                        messages=copy.deepcopy(messages), thinking=False, **options
                    ),
                    timeout=min(240.0, timeout_seconds),
                )

            try:
                response = asyncio.run(complete(remaining))
            except TimeoutError:
                # A single slow request (asyncio.wait_for) must not kill the
                # whole run; retry while the shared deadline allows it.
                if attempt == 2 or self._deadline - time.monotonic() <= 0:
                    raise TimeoutError("aig_role_request_timeout") from None
                continue
            content = response if isinstance(response, str) else getattr(response, "content", None)
            try:
                result = _json_object(content)
                if validate is not None:
                    validate(result)
                return result
            except ValueError as exc:
                logger.warning(
                    "aig role output rejected (attempt %d/3): %s: %s | raw=%s",
                    attempt + 1, type(exc).__name__, exc, (content or "")[:1500],
                )
                if attempt == 2:
                    raise
                # Send the rejected output verbatim. Only the model may propose a
                # replacement; no repair, inferred refs, or attribution is inserted.
                messages.append({"role": "assistant", "content": content or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Rejected output: {type(exc).__name__}: {exc}. "
                            "Return a new valid JSON object using the supplied contract. "
                            "Do not invent evidence, step_refs, or attribution."
                        ),
                    }
                )
        raise ValueError("aig_role_retry_exhausted")


def _tool_contracts(tools: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    return [
        {"name": name, "description": description, "parameters": TOOL_SCHEMAS[name]}
        for name, description in tools
    ]


class VLMBuilder(_RoleModel):
    def __call__(self, turn: ToolloopTurn) -> ToolInvocation:
        result = self._ask(
            turn.system_prompt + '\nOutput {"verb":"...","arguments":{...}}.',
            {
                "tools": _tool_contracts(turn.tools),
                "history": turn.history,
                "draft": turn.draft,
                "edit_contract": "modify_block cannot change kind. To reclassify a mistaken block, delete it and recreate with the correct kind and nonempty real refs; recreate removed edges from evidence. Preserve row authorship exactly.",
            },
        )
        if set(result) != {"verb", "arguments"} or not isinstance(result["arguments"], dict):
            raise ValueError("invalid_builder_action")
        if result["verb"] not in {name for name, _ in turn.tools}:
            raise ValueError("builder_verb_forbidden")
        return ToolInvocation(result["verb"], result["arguments"])


class VLMCritic(_RoleModel):
    def __init__(self, handle: Any, deadline: float) -> None:
        super().__init__(handle, deadline)
        self._view: CriticView | None = None

    def bind_view(self, view: CriticView) -> None:
        self._view = view

    def __call__(self, turn: CriticTurn) -> dict[str, Any]:
        if self._view is None:
            raise ValueError("critic_view_missing")
        history: list[dict[str, Any]] = []
        block_ids = {block["block_id"] for block in turn.draft.get("blocks", [])}
        row_count = self._view.call("get_overview")["row_count"]

        def validate(output: dict[str, Any]) -> None:
            # A malformed audit is sent back to the model with the exact reason (bounded
            # by _ask's retries) instead of failing the whole four-role construction.
            if set(output) == {"violations"}:
                try:
                    parse_critic_violations(output, block_ids=block_ids, row_count=row_count)
                except ValueError as exc:
                    raise ValueError(
                        f"{exc}. Each violation must be exactly "
                        "{object_id, kind, severity, evidence_ao_refs, note}; object_id is ONE "
                        f"existing block_id from {sorted(block_ids)} (a string, never a list or "
                        "a joined string; mention other blocks in note); kind is semantic or "
                        "granularity; severity is high or low; evidence_ao_refs are integer "
                        f"row indexes in [0, {row_count - 1}]; note is non-empty."
                    ) from exc

        for _ in range(9):
            result = self._ask(
                turn.system_prompt + '\nRead with {"verb":"read_step","arguments":{"index":0}} '
                'or finish with {"violations":[...]}. Never infer a clean audit from a tool error.',
                {
                    "tools": _tool_contracts(turn.tools),
                    "draft": turn.draft,
                    "structural_violations": turn.structural_violations,
                    "history": history,
                },
                validate=validate,
            )
            if set(result) == {"violations"}:
                parse_critic_violations(result, block_ids=block_ids, row_count=row_count)
                return result
            if set(result) != {"verb", "arguments"} or not isinstance(result["arguments"], dict):
                raise ValueError("invalid_critic_action")
            if result["verb"] not in {name for name, _ in turn.tools}:
                raise ValueError("critic_verb_forbidden")
            value = self._view.call(result["verb"], result["arguments"])
            history.append(
                {"verb": result["verb"], "arguments": result["arguments"], "result": value}
            )
        raise ValueError("critic_read_budget_exhausted")


class VLMRefiner(_RoleModel):
    def __call__(self, turn: RefinerTurn) -> RefinerAction:
        result = self._ask(
            turn.system_prompt + '\nOutput one of {"action":"done"}, '
            '{"action":"unresolvable","violation_id":"...","reason":"..."}, '
            'or {"action":"edit","verb":"...","arguments":{...},"violation_id":"..."}.',
            {
                "tools": _tool_contracts(turn.tools),
                "draft": turn.draft,
                "violations": turn.violations,
                "history": turn.history,
            },
        )
        action = result.get("action")
        if action == "done" and set(result) == {"action"}:
            return RefinerAction("done")
        if action == "unresolvable" and set(result) == {"action", "violation_id", "reason"}:
            return RefinerAction(
                "unresolvable", violation_id=result["violation_id"], reason=result["reason"]
            )
        if action == "edit" and set(result) == {"action", "verb", "arguments", "violation_id"}:
            if result["verb"] not in {name for name, _ in turn.tools}:
                raise ValueError("refiner_verb_forbidden")
            return RefinerAction(
                "edit", result["verb"], result["arguments"], result["violation_id"]
            )
        raise ValueError("invalid_refiner_action")


class VLMReader(_RoleModel):
    def __call__(self, turn: ReaderTurn) -> ReaderInvocation:
        unresolved_attempts = 0

        def validate(result: dict[str, Any]) -> None:
            if set(result) == {"verb", "arguments"}:
                if result["verb"] not in {name for name, _ in turn.tools}:
                    raise ValueError("reader_verb_forbidden")
                if not isinstance(result["arguments"], dict):
                    raise ValueError("invalid_reader_arguments")
                return
            report = parse_reader_report(result, turn.graph, allow_unresolved=True)
            if report.critical_block_id and turn.system_prompt == READER_NEUTRAL_PROMPT:
                block = next(b for b in turn.graph["blocks"] if b["block_id"] == report.critical_block_id)
                if block.get("kind") not in {"agent", "orchestrator"} or report.responsible_agent == "openviking:diagnosis-control":
                    raise ValueError("neutral_report_requires_observed_work_block")
            if not set(report.decisive_ao_refs) <= set(turn.read_ao_refs):
                raise ValueError("unread_decisive_refs: read the cited AO rows before reporting")
            if report.unresolved_areas:
                nonlocal unresolved_attempts
                unresolved_attempts += 1
                if unresolved_attempts < 3:
                    raise ValueError(
                        "reader_unresolved: production cannot accept this report. Recheck the bounded question and source refs. "
                        "You are the final Reader producing the current diagnosis, not awaiting a future Memory diagnosis. "
                        "Do not invent private causes. If the related author or source refs truly cannot be verified, "
                        "retain unresolved_areas and abstain; the report will remain rejected."
                    )

        result = self._ask(
            turn.system_prompt + '\nRead with {"verb":"...","arguments":{...}} '
            "or return the exact five-field report JSON.",
            {
                "tools": _tool_contracts(turn.tools),
                "frozen_graph": turn.graph,
                "neutral_state_report": turn.system_prompt == READER_NEUTRAL_PROMPT,
                "history": turn.history,
                "read_ao_refs": turn.read_ao_refs,
                "remaining_calls": turn.remaining_calls,
                "final_only": turn.final_only,
                "report_contract": "critical_block_id must exist; responsible_agent must equal that block.agent_name; decisive_ao_refs must equal that block.step_refs, and must already be read. No negative or invented refs.",
            },
            validate=validate,
        )
        if set(result) == {"verb", "arguments"} and isinstance(result["arguments"], dict):
            if result["verb"] not in {name for name, _ in turn.tools}:
                raise ValueError("reader_verb_forbidden")
            return ReaderInvocation("tool", result["verb"], result["arguments"])
        return ReaderInvocation("report", report=result)


class VLMCardProposer(_RoleModel):
    """One call: the frozen graph and facts in, ``{"proposals": [...]}`` out."""

    def __call__(self, turn: Any) -> list[dict[str, Any]]:
        def validate(result: dict[str, Any]) -> None:
            proposals = result.get("proposals")
            if set(result) != {"proposals"} or not isinstance(proposals, list):
                raise ValueError("invalid_card_proposals")
            for item in proposals:
                if not isinstance(item, dict):
                    raise ValueError("invalid_card_proposals")
                _check_card_evidence(item, turn.block_aos, turn.read_aos)

        result = self._ask(
            turn.system_prompt + '\nOutput {"proposals":[<proposal>, ...]}; an empty list is fine.',
            {
                "frozen_graph": turn.graph,
                "base_revision_id": turn.base_revision_id,
                "base_revision_block_content_hash": turn.base_revision_block_content_hash,
                "report_hash": turn.report_hash,
                "block_aos": turn.block_aos,
                "read_aos": turn.read_aos,
                "outcome": turn.outcome_digest,
                "outcome_feedback": getattr(turn, "outcome_feedback", ""),
                "allowed_kinds": turn.allowed_kinds,
            },
            validate=validate,
        )
        proposals = result.get("proposals")
        if set(result) != {"proposals"} or not isinstance(proposals, list):
            raise ValueError("invalid_card_proposals")
        if any(not isinstance(item, dict) for item in proposals):
            raise ValueError("invalid_card_proposals")
        return proposals


def _check_card_evidence(
    proposal: Mapping[str, Any],
    block_aos: Mapping[str, Sequence[str]],
    read_aos: Sequence[str],
) -> None:
    """Send an uncited or mis-cited card back to the model. Nothing is repaired or inferred.

    A card must rest on the trajectory: at least one step cites an AO that sits in the step's
    block and that the Reader read. With nothing readable there is nothing to cite, so the
    check does not apply (the server drops such a card on its own).
    """
    steps = proposal.get("steps")
    if not isinstance(steps, list) or not read_aos:
        return
    read = set(read_aos)
    cited = 0
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        refs = step.get("ao_refs") or []
        if not isinstance(refs, list):
            raise ValueError(f"steps[{index}].ao_refs must be a list of AO ids")
        block_id = step.get("block_id")
        usable = [ao for ao in block_aos.get(block_id, ()) if ao in read] if isinstance(block_id, str) else []
        for ao in refs:
            if ao not in usable:
                raise ValueError(
                    f"steps[{index}].ao_refs has {ao!r}, which is not an AO you read in block "
                    f"{step.get('block_id')!r}; AOs you may cite there: {usable}"
                )
        cited += bool(refs)
    if not cited:
        raise ValueError(
            "no step cites an AO. Every card must rest on this trajectory: add ao_refs "
            "(AO ids from read_aos that are inside the step's block) to the steps the "
            "trajectory supports, and drop steps it does not support. "
            "Return an empty proposals list if nothing in this trajectory is worth keeping."
        )


def providers_for_vlm(handle: Any) -> AigRoleProviders:
    """Create separate role states on the request's account-scoped model handle."""
    deadline = time.monotonic() + 900.0
    return AigRoleProviders(
        VLMBuilder(handle, deadline),
        VLMCritic(handle, deadline),
        VLMRefiner(handle, deadline),
        VLMReader(handle, deadline),
    )
