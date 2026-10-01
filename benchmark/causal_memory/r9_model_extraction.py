#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Round-trip the frozen canvas through the extraction contract.

Contract and pipeline test. It does not answer retrieval value and it does
not answer whether a model extraction is better. ``--mode fake`` calls no
model. ``--mode model`` stops unless the caller passes a provider handle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Frozen before scores. Do not edit these to absorb a miss.
LABEL = "契约与管道测试，不回答检索价值／不回答模型是否更好"
SCHEMA = "r9-extraction-contract-roundtrip-v1"
FROZEN_AT = "2026-10-01T00:00:00.000Z"
CHECKPOINTS = Path("/tmp/sb-influence-r8c/checkpoints")
COMPARE_FIELDS = ("blocks", "claims", "coverage", "retrieval_subgraphs")


def _bind() -> Any:
    """Import production and the R8a helpers after the repo root is on the path."""
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from benchmark.causal_memory import sb_influence_extraction as sb
    from openviking.session.extraction_contract import (
        EXTRACTION_CONTRACT_VERSION,
        announce_frozen_limits,
        build_model_input,
        frozen_limits,
        model_extraction_provider,
        parse_contract_response,
    )
    from openviking.session.extraction_write import freeze_extracted_view
    from openviking.session.influence_view import ViewSupersedeIndex
    from openviking.session.trajectory_index_hook import (
        _page_ao_records,
        _stable_snapshot_watermark,
        _view_ao_record,
    )

    return {
        "announce_frozen_limits": announce_frozen_limits,
        "build_model_input": build_model_input,
        "frozen_limits": frozen_limits,
        "model_extraction_provider": model_extraction_provider,
        "parse_contract_response": parse_contract_response,
        "freeze_extracted_view": freeze_extracted_view,
        "ViewSupersedeIndex": ViewSupersedeIndex,
        "page_ao_records": _page_ao_records,
        "snapshot_watermark": _stable_snapshot_watermark,
        "view_ao_record": _view_ao_record,
        "sb": sb,
        "schema_version": EXTRACTION_CONTRACT_VERSION,
    }


def encode_contract(revision: dict[str, Any], schema_version: str) -> dict[str, Any]:
    """Canvas revision to the model-written contract object.

    Derived fields stay out of this object. The production parser fills them.
    """
    coverage = revision["coverage"]
    return {
        "schema_version": schema_version,
        "blocks": [_encode_block(block) for block in revision["blocks"]],
        "claims": [_encode_claim(claim) for claim in revision["claims"]],
        "subgraphs": [_encode_subgraph(item) for item in revision["retrieval_subgraphs"]],
        "coverage": {
            "total_ao_start_seq": coverage["total_ao_range"]["start_seq"],
            "total_ao_end_seq": coverage["total_ao_range"]["end_seq"],
            "inspected_ao_ids": [ref["ao_id"] for ref in coverage["inspected_refs"]],
            "snapshot_watermark": coverage["snapshot_watermark"],
        },
    }


def snapshot_from_revision(revision: dict[str, Any]) -> dict[str, Any]:
    """Identity snapshot the parser reconciles against. Captured state is the canvas value."""
    ao_records: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    for ref in revision["coverage"]["inspected_refs"]:
        ao_records.append(
            {
                "ao_id": ref["ao_id"],
                "sequence": ref["sequence"],
                "participant_id": ref["participant_id"],
                "segment_id": ref["segment_id"],
                "session_id": ref["session_id"],
                "content_hash": ref["content_hash"],
            }
        )
        ledger_rows.append(
            {
                "ao_id": ref["ao_id"],
                "sequence": ref["sequence"],
                "captured_state": "committed",
                "action": {},
                "observation": {},
            }
        )
    return {
        "session_id": revision["session_id"],
        "task_run_id": revision["task_run_id"],
        "view_id": revision["view_id"],
        "revision_id": revision["revision_id"],
        "purpose": revision["purpose"],
        "snapshot_watermark": revision["coverage"]["snapshot_watermark"],
        "ao_records": ao_records,
        "ledger_rows": ledger_rows,
    }


def semantic_diff(canvas: dict[str, Any], frozen: dict[str, Any]) -> list[dict[str, Any]]:
    """Differences on the four payload sections. Envelope fields are not compared."""
    diffs: list[dict[str, Any]] = []
    for field in COMPARE_FIELDS:
        _walk(field, canvas.get(field), frozen.get(field), diffs)
    return diffs


def round_trip_episode(api: dict[str, Any], revision: dict[str, Any]) -> dict[str, Any]:
    """Encode, parse with the production parser, freeze, and diff against the canvas."""
    snapshot = snapshot_from_revision(revision)
    model_input = api["build_model_input"](snapshot)
    text = json.dumps(
        encode_contract(revision, api["schema_version"]),
        ensure_ascii=False,
        sort_keys=True,
    )
    payload = api["parse_contract_response"](text, model_input)
    frozen = api["freeze_extracted_view"](
        session_id=revision["session_id"],
        task_run_id=revision["task_run_id"],
        view_id=revision["view_id"],
        revision_id=revision["revision_id"],
        frozen_at=FROZEN_AT,
        blocks=payload.blocks,
        claims=payload.claims,
        coverage=payload.coverage,
        retrieval_subgraphs=payload.retrieval_subgraphs,
    )
    frozen_dict = frozen.to_dict()
    diffs = semantic_diff(revision, frozen_dict)
    return {
        "equal": not diffs,
        "diff_count": len(diffs),
        "diffs": diffs,
        "frozen": frozen,
    }


def episode_snapshot(
    api: dict[str, Any], episode_dir: Path, episode_index: int, work: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write one episode into a real ledger and return the commit snapshot plus the parse."""
    sb = api["sb"]
    parsed = sb._parse_transcript(episode_dir)
    session_id = episode_dir.name
    task_id = parsed["task_id"]
    session_dir = work / "ledgers" / session_id
    sb._append_ledger(session_dir, session_id, parsed, episode_index)
    raw = api["page_ao_records"](session_dir, session_id)
    raw.sort(key=lambda row: (row["sequence"], row["ao_id"]))
    ao_records = [api["view_ao_record"](item, session_id) for item in raw]
    watermark = api["snapshot_watermark"](session_id, ao_records)
    snapshot = {
        "session_id": session_id,
        "task_run_id": task_id,
        "view_id": f"{session_id}:post_run_index",
        "revision_id": f"r9:{episode_dir.name[:4]}",
        "purpose": "post_run_index",
        "snapshot_watermark": watermark,
        "ao_records": ao_records,
        "ledger_rows": raw,
    }
    return snapshot, parsed


def run_model_episode(
    api: dict[str, Any],
    vlm_handle: Any,
    episode_dir: Path,
    episode_index: int,
    work: Path,
    *,
    max_total_tokens: int | None = None,
    max_tokens_per_call: int | None = None,
) -> dict[str, Any]:
    """One episode through the real ledger, the provider, and a frozen adaptive revision."""
    snapshot, parsed = episode_snapshot(api, episode_dir, episode_index, work)
    session_id = snapshot["session_id"]
    task_id = snapshot["task_run_id"]
    kwargs: dict[str, Any] = {}
    if max_total_tokens is not None:
        kwargs["max_total_tokens"] = max_total_tokens
    if max_tokens_per_call is not None:
        kwargs["max_tokens_per_call"] = max_tokens_per_call
    factory = api["model_extraction_provider"](vlm_handle, **kwargs)
    provider = factory(snapshot)
    payload = provider(session_id, task_id, "unspecified")
    revision = api["freeze_extracted_view"](
        session_id=session_id,
        task_run_id=task_id,
        view_id=snapshot["view_id"],
        revision_id=f"post-run:{session_id}:{task_id}:adaptive",
        frozen_at=FROZEN_AT,
        blocks=payload.blocks,
        claims=payload.claims,
        coverage=payload.coverage,
        retrieval_subgraphs=payload.retrieval_subgraphs,
    )
    index = api["ViewSupersedeIndex"](work / "views.json")
    index.register(revision)
    canvas = _checkpoint_revision(episode_dir.name)
    ledger_cut = parsed.get("tail_class") == "tool_cut"
    marked = any(item.failure_marked for item in payload.retrieval_subgraphs)
    return {
        "episode_id": episode_dir.name,
        "compliant": True,
        "repairs": provider.repairs,
        "calls": provider.calls,
        "usage": dict(provider.usage),
        "finish_reasons": list(provider.finish_reasons),
        "truncated": provider.truncated,
        "revision_id": revision.revision_id,
        "construction_mode": revision.construction_mode,
        "structure_counts": _counts(payload, canvas),
        "ledger_cut": ledger_cut,
        "failure_marked": marked,
        "marker_matches_ledger_cut": marked == ledger_cut,
    }


def main(
    argv: list[str] | None = None, *, vlm_handle: Any = None, max_total_tokens: int | None = None
) -> int:
    parser = argparse.ArgumentParser(description="R9 extraction contract round-trip")
    parser.add_argument("--mode", choices=("fake", "model"), default="fake")
    parser.add_argument("--out", type=Path, default=Path("/tmp/r9-contract/report.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--episode", default="")
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINTS)
    args = parser.parse_args(argv)
    api = _bind()
    api["announce_frozen_limits"]()
    print(f"label={LABEL}", flush=True)
    print(f"frozen_at={FROZEN_AT}", flush=True)
    print(f"mode={args.mode}", flush=True)
    if args.mode == "model" and vlm_handle is None:
        print(
            "mode=model needs an extraction provider handle. "
            "None was given, so this run stops. It does not fall back to mode=fake.",
            file=sys.stderr,
        )
        return 2
    if not args.checkpoints.is_dir():
        print(f"missing checkpoints {args.checkpoints}", file=sys.stderr)
        return 1
    paths = _checkpoint_paths(args.checkpoints, args.episode)
    if args.mode == "fake" and not args.episode and len(paths) != 14:
        print(f"expected 14 checkpoints, found {len(paths)}", file=sys.stderr)
        return 1
    work = args.out.parent
    work.mkdir(parents=True, exist_ok=True)
    previous = _load_previous(args.out) if args.resume else {}
    episodes: list[dict[str, Any]] = []
    failed = False
    if args.mode == "fake":
        for path in paths:
            episode_id = path.name
            cached = previous.get(episode_id)
            if cached is not None:
                print(f"RESUME {episode_id}", flush=True)
                episodes.append(cached)
                failed = failed or not cached.get("equal", False)
                continue
            record = _run_fake_file(api, path)
            episodes.append(record)
            _print_fake(record)
            failed = failed or not record["equal"]
    else:
        episode_dirs = api["sb"]._episode_dirs()
        for index, episode_dir in enumerate(episode_dirs):
            if args.episode and not episode_dir.name.startswith(args.episode):
                continue
            cached = previous.get(episode_dir.name)
            if cached is not None:
                print(f"RESUME {episode_dir.name}", flush=True)
                episodes.append(cached)
                continue
            record = run_model_episode(
                api,
                vlm_handle,
                episode_dir,
                index,
                work,
                max_total_tokens=max_total_tokens,
            )
            episodes.append(record)
            print(
                f"{episode_dir.name} compliant={record['compliant']} "
                f"tokens={record['usage']['total_tokens']} repairs={record['repairs']}",
                flush=True,
            )
    report = _report(args.mode, api["frozen_limits"](), episodes)
    _write_report(args.out, report)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"sha256 {digest}", flush=True)
    print(f"wrote {args.out}", flush=True)
    if args.mode == "fake":
        equal = sum(1 for item in episodes if item["equal"])
        print(f"round_trip {equal}/{len(episodes)}", flush=True)
        return 1 if failed else 0
    return 0


def _encode_block(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_id": block["block_id"],
        "role": block["role"],
        "block_kind": block["block_kind"],
        "participant_id": block["participant_id"],
        "segment_id": block["segment_ref"]["segment_id"],
        "ao_start_seq": block["ao_start_seq"],
        "ao_end_seq": block["ao_end_seq"],
        "summary": block["summary"],
        "input": block["input"],
        "output": block["output"],
        "authorship": block["authorship"],
        "granularity_reason": block["granularity_reason"],
    }


def _encode_evidence(ref: dict[str, Any]) -> dict[str, str]:
    return {"ao_id": ref["ao_id"], "evidence_content_hash": ref["evidence_content_hash"]}


def _encode_claim(claim: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_id": claim["claim_id"],
        "source_block_id": claim["source_block_id"],
        "target_block_id": claim["target_block_id"],
        "carried_artifact": claim["carried_artifact"],
        "downstream_effect": claim["downstream_effect"],
        "relation_type": claim["relation_type"],
        "status": claim["status"],
        "source_evidence": [_encode_evidence(item) for item in claim["source_evidence_refs"]],
        "target_evidence": [_encode_evidence(item) for item in claim["target_evidence_refs"]],
    }


def _encode_subgraph(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "subgraph_id": item["subgraph_id"],
        "block_ids": list(item["block_ids"]),
        "claim_ids": list(item["claim_ids"]),
        "failure_marked": item["failure_marked"],
        "failure_evidence": [_encode_evidence(ref) for ref in item["failure_evidence_refs"]],
    }


def _walk(path: str, left: Any, right: Any, diffs: list[dict[str, Any]]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                diffs.append(
                    {"path": f"{path}.{key}", "canvas": left.get(key), "parsed": right.get(key)}
                )
                continue
            _walk(f"{path}.{key}", left[key], right[key], diffs)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            diffs.append({"path": path, "canvas_len": len(left), "parsed_len": len(right)})
        for index, (item, other) in enumerate(zip(left, right, strict=False)):
            _walk(f"{path}[{index}]", item, other, diffs)
        return
    if left != right:
        diffs.append({"path": path, "canvas": left, "parsed": right})


def _run_fake_file(api: dict[str, Any], path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    revision = payload["revision"]
    try:
        result = round_trip_episode(api, revision)
    except Exception as exc:
        return {
            "episode_id": path.name,
            "equal": False,
            "diff_count": 1,
            "diffs": [
                {"path": "round_trip", "canvas": "frozen", "parsed": f"{type(exc).__name__}: {exc}"}
            ],
        }
    return {
        "episode_id": path.name,
        "equal": result["equal"],
        "diff_count": result["diff_count"],
        "diffs": result["diffs"],
    }


def _print_fake(record: dict[str, Any]) -> None:
    if record["equal"]:
        print(f"{record['episode_id']} equal", flush=True)
        return
    print(f"{record['episode_id']} diffs={record['diff_count']}", flush=True)
    for item in record["diffs"]:
        print(f"  {item['path']}", flush=True)


def _counts(payload: Any, canvas: dict[str, Any] | None) -> dict[str, int]:
    claims = list(payload.claims)
    canvas_claims = canvas["claims"] if canvas else []
    return {
        "blocks": len(payload.blocks),
        "claims": len(claims),
        "influence_claims": sum(1 for item in claims if item.relation_type == "influence"),
        "skip_claims": sum(1 for item in claims if item.relation_type == "skip"),
        "subgraphs": len(payload.retrieval_subgraphs),
        "canvas_blocks": len(canvas["blocks"]) if canvas else 0,
        "canvas_claims": len(canvas_claims),
        "canvas_subgraphs": len(canvas["retrieval_subgraphs"]) if canvas else 0,
    }


def _checkpoint_revision(episode_name: str) -> dict[str, Any] | None:
    matches = sorted(CHECKPOINTS.glob(f"{episode_name}.json"))
    if not matches:
        matches = sorted(CHECKPOINTS.glob(f"{episode_name[:4]}*.json"))
    if not matches:
        return None
    return json.loads(matches[0].read_text(encoding="utf-8"))["revision"]


def _checkpoint_paths(root: Path, episode: str) -> list[Path]:
    paths = sorted(path for path in root.glob("*.json") if path.is_file())
    if not episode:
        return paths
    return [path for path in paths if path.name.startswith(episode)]


def _load_previous(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        return {}
    return {
        item["episode_id"]: item
        for item in episodes
        if isinstance(item, dict) and isinstance(item.get("episode_id"), str)
    }


def _report(mode: str, limits: dict[str, Any], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(episodes, key=lambda item: item["episode_id"])
    summary: dict[str, Any] = {"episodes": len(ordered)}
    if mode == "fake":
        summary["equal"] = sum(1 for item in ordered if item.get("equal"))
        summary["diff_rows"] = sum(item.get("diff_count", 0) for item in ordered)
    else:
        summary["compliant"] = sum(1 for item in ordered if item.get("compliant"))
        summary["repairs"] = sum(item.get("repairs", 0) for item in ordered)
        summary["total_tokens"] = sum(
            item.get("usage", {}).get("total_tokens", 0) for item in ordered
        )
        summary["truncated"] = sum(item.get("truncated", 0) for item in ordered)
        decided = [item for item in ordered if "marker_matches_ledger_cut" in item]
        summary["marker_matches_ledger_cut"] = sum(
            1 for item in decided if item["marker_matches_ledger_cut"]
        )
        summary["marker_compared"] = len(decided)
    return {
        "label": LABEL,
        "schema": SCHEMA,
        "mode": mode,
        "frozen_at": FROZEN_AT,
        "frozen_limits": limits,
        "episodes": ordered,
        "summary": summary,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
