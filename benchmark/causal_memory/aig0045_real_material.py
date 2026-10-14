#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ADR-0045 real-material pipeline (eval-direct, ADR §verdict 来源与模型配置).

Three subcommands over the evolbench-collab runner output:

  reverify  attempts.jsonl -> verdicts.jsonl. Each attempt row is re-graded
            through the verify sidecar with the explicit payload the inline
            runner call cannot supply (final_output / workspace_dir).
  export    Daemon session transcripts (``<work_dir>/.builtin-sessions``) ->
            per-session AORecord jsonl plus the episode<->session join
            (prompt substring match), filling session_id into verdicts.jsonl.
            The daemon transcript is the task agent's full operation log
            (tool calls, results, narration); the OV drain segment only
            carries the channel-visible summary, so the daemon side is the
            material source.
  sessions  Diagnostic: list session files under a work_dir with match info.

Judge routing is per-domain: information_retrieval -> --ir-sidecar
(gemma-4-31B-it on the lenovo gateway; OpenRouter qwen was credit-blocked),
knowledge_work -> --kw-sidecar (GLM-5.3-Flash),
software_engineering -> --swe-sidecar (pytest in container, up to 30 min).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openviking.session.ao_ledger import AORecord  # noqa: E402

DOMAIN_KIND = {
    "information_retrieval": "browsecomp",
    "knowledge_work": "gdpval",
    "software_engineering": "swe",
}


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_attempts(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cmd_reverify(args: argparse.Namespace) -> int:
    attempts = _read_attempts(args.attempts)
    verdicts: list[dict] = []
    for row in attempts:
        domain = str(row.get("domain") or "")
        kind = DOMAIN_KIND.get(domain)
        if kind is None:
            print(f"skip {row.get('episode_id')}: domain {domain!r} not routed", file=sys.stderr)
            continue
        if kind == "browsecomp":
            port = args.ir_port
        elif kind == "swe":
            port = getattr(args, "swe_port", 8737)
        else:
            port = args.kw_port
        url = f"http://127.0.0.1:{port}/verify"
        payload: dict = {"episode_id": str(row["episode_id"])}
        if kind == "browsecomp":
            payload["agent_response"] = row.get("final_output") or ""
        else:
            payload["workspace_dir"] = row.get("work_dir") or ""
        timeout = max(args.timeout, 1900.0) if kind == "swe" else args.timeout
        verdict = _post_json(url, payload, timeout=timeout)
        detail = {
            key: verdict[key]
            for key in (
                "method",
                "predicted",
                "files_found",
                "evaluation_score",
                "judge",
                "resolved",
                "returncode",
            )
            if key in verdict
        }
        verdicts.append(
            {
                "run_id": row.get("run_id"),
                "episode_id": row.get("episode_id"),
                "session_id": None,
                "domain_kind": kind,
                "reward": verdict.get("reward"),
                "graded": verdict.get("graded"),
                "error": verdict.get("error"),
                "detail": detail,
                "attempt_status": row.get("status"),
            }
        )
        print(
            f"{kind} {row.get('episode_id')}: reward={verdict.get('reward')} "
            f"graded={verdict.get('graded')} err={verdict.get('error')}"
        )
    with Path(args.out).open("w", encoding="utf-8") as handle:
        for row in verdicts:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(verdicts)} verdicts -> {args.out}")
    return 0


# --- daemon transcript -> AORecord -------------------------------------------


def _iter_transcript_messages(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "message":
                rows.append(payload)
    return rows


def transcript_to_records(path: Path, *, participant: str | None = None) -> list[AORecord]:
    """Keep actions in native order and attach results by their tool-call identity."""
    with path.open("r", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
    session_id = str(header.get("id") or path.stem)
    participant_id = participant or "daemon-agent"
    records: list[AORecord] = []
    pending: dict[str, AORecord] = {}
    legacy_pending: list[AORecord] = []
    timestamp = ""
    native_message_id = ""

    def record(action: dict, observation: dict | None = None) -> AORecord:
        sequence = len(records) + 1
        row = AORecord(
            ao_id=f"{session_id}:{sequence}",
            session_id=session_id,
            sequence=sequence,
            action={
                **action,
                "participant_id": participant_id,
                "segment_id": session_id,
                "native_message_id": native_message_id,
            },
            observation=observation,
            skill_invocations=[],
            message_ref="",
            created_at=timestamp,
            captured_state=None,
        )
        records.append(row)
        return row

    def attach_result(call_id: str, body: str) -> None:
        row = pending.pop(call_id, None) if call_id else None
        if (
            row is None
            and not call_id
            and len(legacy_pending) + len(pending) == 1
            and legacy_pending
        ):
            row = legacy_pending.pop()
        elif row is None and not call_id and len(legacy_pending) + len(pending) == 1 and pending:
            row = pending.pop(next(iter(pending)))
        if row is not None:
            records[row.sequence - 1] = replace(row, observation={"text": body[:4000]})
        else:
            record(
                {"tool": "tool_result_orphan", "role": "toolResult", "tool_call_id": call_id},
                {"text": body[:4000]},
            )

    for payload in _iter_transcript_messages(path):
        message = payload.get("message") or {}
        role = message.get("role")
        content = message.get("content")
        timestamp = str(payload.get("timestamp") or "")
        native_message_id = str(payload.get("id") or "")
        parts = (
            content
            if isinstance(content, list)
            else ([{"type": "text", "text": content}] if isinstance(content, str) else [])
        )
        if role == "toolResult":
            body = "\n".join(
                str(part.get("text") or "")
                for part in parts
                if isinstance(part, dict) and part.get("type") == "text"
            )
            attach_result(str(message.get("toolCallId") or ""), body)
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "toolCall":
                call_id = str(part.get("id") or "")
                if call_id and call_id in pending:
                    raise ValueError("duplicate_tool_call_id")
                row = record(
                    {
                        "tool": str(part.get("name") or ""),
                        "args": part.get("arguments"),
                        "role": role,
                        **({"tool_call_id": call_id} if call_id else {}),
                    }
                )
                if call_id:
                    pending[call_id] = row
                else:
                    legacy_pending.append(row)
            elif kind == "toolResult":
                body = part.get("value")
                if not isinstance(body, str):
                    body = json.dumps(body, ensure_ascii=False)
                attach_result(str(part.get("toolCallId") or part.get("id") or ""), body)
            elif kind == "text":
                text = str(part.get("text") or "")
                if text.strip():
                    record({"text": text[:6000], "role": role})
    return records


def _session_blob(path: Path) -> str:
    chunks = []
    for payload in _iter_transcript_messages(path):
        for part in payload.get("message", {}).get("content") or []:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(str(part.get("text") or ""))
    return "\n".join(chunks)


def cmd_export(args: argparse.Namespace) -> int:
    manifest = [
        json.loads(line) for line in Path(args.manifest).open("r", encoding="utf-8") if line.strip()
    ]
    questions = {str(row["episode_id"]): row["turns"][0]["prompt"] for row in manifest}
    prompts = {episode: question[:300] for episode, question in questions.items()}
    verdicts_path = Path(args.verdicts)
    verdicts = (
        [json.loads(line) for line in verdicts_path.open("r", encoding="utf-8") if line.strip()]
        if verdicts_path.exists()
        else []
    )
    by_episode = {str(row.get("episode_id")): row for row in verdicts}
    sessions_out = Path(args.sessions_out)
    sessions_out.mkdir(parents=True, exist_ok=True)

    attempts = _read_attempts(args.attempts)
    # episodes in execution order per workspace (attempt rows append in order)
    work_dirs: dict[str, list[str]] = {}
    status_of: dict[str, str] = {}
    for row in attempts:
        if row.get("work_dir"):
            work_dirs.setdefault(str(row["work_dir"]), []).append(str(row["episode_id"]))
            status_of[str(row["episode_id"])] = str(row.get("status") or "")

    matches: dict[str, list[tuple[str, int]]] = {}
    exported = 0

    def _export_session(path: Path, claim: str, agent_hint: str, participant: str) -> None:
        nonlocal exported
        records = transcript_to_records(path, participant=participant)
        if not records:
            return
        session_id = records[0].session_id
        out_path = sessions_out / f"{session_id}.jsonl"
        if not out_path.exists():
            with out_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            exported += 1
        matches.setdefault(claim, []).append((session_id, len(records)))
        print(f"  {path.name[:44]}: {len(records)} records -> {claim} ({agent_hint})")

    for work_dir, episode_order in work_dirs.items():
        agent_root = Path(work_dir).parent
        task_sessions = sorted((Path(work_dir) / ".builtin-sessions").glob("*.jsonl"))
        successes = [ep for ep in episode_order if status_of.get(ep) == "success"]
        if successes and len(successes) == len(task_sessions):
            # Success means the task agent engaged, and engagement creates a
            # session; both sequences are chronological, so zip 1:1. Content
            # matching cannot discriminate here: the task agent re-reads the
            # full channel history, so every session contains every prompt.
            for path, claim in zip(task_sessions, successes, strict=True):
                _export_session(path, claim, "task-zip", "task-agent")
            unassigned = [ep for ep in episode_order if ep not in successes]
        else:
            # Sequential claim: earliest unassigned episode whose prompt is in
            # the session blob. Warm-channel history only ADDS older prompts,
            # so the earliest unassigned hit is the session's own episode.
            unassigned = list(episode_order)
            for path in task_sessions:
                claim = next(
                    (
                        ep
                        for ep in unassigned
                        if prompts.get(ep, "") and prompts[ep] in _session_blob(path)
                    ),
                    None,
                )
                if claim is None:
                    continue
                _export_session(path, claim, "task-seq", "task-agent")
                unassigned.remove(claim)
        if unassigned:
            # Memory-agent fallback: episodes where the task agent never woke
            # still have the memory agent's session as their only trajectory.
            for sibling in sorted(agent_root.iterdir()):
                if not sibling.is_dir() or sibling.name == Path(work_dir).name:
                    continue
                mem_root = sibling / ".builtin-sessions"
                if not mem_root.is_dir():
                    continue
                for path in sorted(mem_root.glob("*.jsonl")):
                    claim = next(
                        (
                            ep
                            for ep in unassigned
                            if prompts.get(ep, "") and prompts[ep] in _session_blob(path)
                        ),
                        None,
                    )
                    if claim is None:
                        continue
                    _export_session(path, claim, "memory-seq", "memory-agent")
                    unassigned.remove(claim)
        if unassigned:
            print(f"  {work_dir}: no session for episodes {unassigned}")
    print(f"exported {exported} sessions -> {sessions_out}")

    joined = 0
    for episode_id, candidates in matches.items():
        session_id, count = max(candidates, key=lambda item: item[1])
        row = by_episode.get(episode_id)
        if row is None:
            print(f"  episode {episode_id} matched session {session_id} but has no verdict row")
            continue
        row["session_id"] = session_id
        row["question"] = questions[episode_id]
        row.setdefault("detail", {})["session_candidates"] = [
            {"session_id": sid, "records": n} for sid, n in sorted(candidates)
        ]
        joined += 1
        print(f"  episode {episode_id} <- session {session_id} ({count} records)")
    with verdicts_path.open("w", encoding="utf-8") as handle:
        for row in verdicts:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    missing = [str(row.get("episode_id")) for row in verdicts if not row.get("session_id")]
    print(f"joined {joined}/{len(verdicts)} verdict rows; missing session_id: {missing}")
    return 0 if joined > 0 else 1


def cmd_sessions(args: argparse.Namespace) -> int:
    for work_dir in sorted(
        {str(row["work_dir"]) for row in _read_attempts(args.attempts) if row.get("work_dir")}
    ):
        print(f"== {work_dir}")
        for path in sorted((Path(work_dir) / ".builtin-sessions").glob("*.jsonl")):
            print(f"  {path.name} ({path.stat().st_size} bytes)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    reverify = sub.add_parser("reverify", help="attempts -> verdicts via sidecars")
    reverify.add_argument("--attempts", type=Path, required=True)
    reverify.add_argument("--out", type=Path, required=True)
    reverify.add_argument("--ir-port", type=int, default=8735)
    reverify.add_argument("--kw-port", type=int, default=8736)
    reverify.add_argument("--swe-port", type=int, default=8737)
    reverify.add_argument("--timeout", type=float, default=240.0)
    reverify.set_defaults(func=cmd_reverify)

    export = sub.add_parser("export", help="daemon transcripts -> AO jsonl + join")
    export.add_argument("--attempts", type=Path, required=True)
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--verdicts", type=Path, required=True)
    export.add_argument("--sessions-out", type=Path, required=True)
    export.set_defaults(func=cmd_export)

    sessions = sub.add_parser("sessions", help="list session files per work_dir")
    sessions.add_argument("--attempts", type=Path, required=True)
    sessions.set_defaults(func=cmd_sessions)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
