# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Judge entry point for an external host: one JSON request in, one verdict out.

An external trainer runs this as a subprocess (``python -m
openviking.session.aig_judge_cli``) so it does not link against OpenViking code.

stdin is one JSON object::

    {"ao_records": [AORecord dicts], "watermark": "...", "question": "...",
     "graph": {"blocks": [...], "edges": [...]},
     "role_products": {"builder": .., "critic": .., "refiner": .., "reader": ..},
     "response_text": "the candidate action under review", "max_calls": 12}

stdout is the ``judge_score`` fields (``status/score/used/evidence_refs/
roles_completed``). Failures exit non-zero with only a stable code on stderr and
nothing on stdout; an unscored verdict is exit 3 and is never a number. The model is
any OpenAI-compatible chat endpoint; the key is read only from the environment.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from openviking.session.aig_tool_surface import AoSnapshot
from openviking.session.aig_toolloop_judge import (
    DEFAULT_JUDGE_CALLS,
    JUDGE_SYSTEM_PROMPT,
    JudgeError,
    JudgeInvocation,
    JudgeProvider,
    JudgeTurn,
    run_judge,
    to_external_result,
)
from openviking.session.ao_ledger import AORecord

API_KEY_ENV = "JUDGE_API_KEY"
FALLBACK_KEY_ENV = "LENOVO_GATEWAY_API_KEY"
URL_ENV = "JUDGE_BASE_URL"
FALLBACK_URL_ENV = "LENOVO_GATEWAY_URL"
MODEL_ENV = "JUDGE_MODEL"
EXIT_FAILED = 2
EXIT_UNSCORED = 3
_MAX_HISTORY_CHARS = 20000


class ChatJudgeProvider:
    """Drive the Judge loop with a chat model that answers in JSON text.

    The model replies with ``{"tool": verb, "args": {...}}`` or a verdict object.
    ``urlopen`` is injectable so tests never reach a network.
    """

    def __init__(
        self,
        response_text: str,
        *,
        key: str,
        base_url: str,
        model: str,
        urlopen: Callable[..., Any] | None = None,
        timeout: float = 240.0,
        max_tokens: int = 4096,
    ) -> None:
        self._response_text = response_text
        self._key = key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._urlopen = urlopen or urllib.request.urlopen
        self._timeout = timeout
        self._max_tokens = max_tokens

    def __call__(self, turn: JudgeTurn) -> JudgeInvocation:
        text = self._complete(self._messages(turn))
        try:
            payload = _first_json_object(text)
        except ValueError as exc:
            raise JudgeError("invalid_provider_result") from exc
        if "tool" in payload:
            args = payload.get("args", {})
            if not isinstance(payload["tool"], str) or not isinstance(args, dict):
                raise JudgeError("invalid_provider_result")
            return JudgeInvocation("tool", payload["tool"], args)
        return JudgeInvocation("verdict", verdict=payload)

    def _messages(self, turn: JudgeTurn) -> list[dict[str, str]]:
        tools = "\n".join(f"- {name}: {doc}" for name, doc in turn.tools)
        history = json.dumps(list(turn.history), ensure_ascii=False, sort_keys=True)
        if len(history) > _MAX_HISTORY_CHARS:
            history = history[-_MAX_HISTORY_CHARS:]
        user = (
            "候选动作（被审查的文本）：\n" + self._response_text + "\n\n"
            "四角色产物：\n"
            + json.dumps(turn.role_products, ensure_ascii=False, sort_keys=True)
            + "\n\n冻结图：\n"
            + json.dumps(turn.graph, ensure_ascii=False, sort_keys=True)
            + '\n\n可用只读工具（每次回复一个 JSON：{"tool":名,"args":{..}} 或最终裁决）：\n'
            + tools
            + f"\n\n剩余工具调用次数：{turn.remaining_calls}\n已读 AO 行：{list(turn.read_ao_refs)}\n"
            + "工具历史：\n"
            + history
        )
        return [
            {"role": "system", "content": turn.system_prompt or JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def _complete(self, messages: list[dict[str, str]]) -> str:
        body = json.dumps(
            {"model": self._model, "messages": messages, "max_tokens": self._max_tokens}
        ).encode()
        request = urllib.request.Request(
            self._url,
            data=body,
            method="POST",
            headers={
                "Authorization": "Bearer " + self._key,
                "Content-Type": "application/json",
            },
        )
        try:
            with self._urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode())
            content = payload["choices"][0]["message"]["content"]
        except Exception as exc:  # never carry the body or the key in the error
            raise JudgeError("provider_unavailable") from exc
        if not isinstance(content, str) or not content.strip():
            raise JudgeError("invalid_provider_result")
        return content


def _first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _end = decoder.raw_decode(text[start:])
        except ValueError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    raise ValueError("no_json")


def _snapshot(request: Mapping[str, Any]) -> AoSnapshot:
    rows = request.get("ao_records")
    if not isinstance(rows, list) or not rows:
        raise JudgeError("invalid_request")
    try:
        records = tuple(AORecord.from_dict(row) for row in rows)
        return AoSnapshot(
            records,
            watermark=str(request["watermark"]),
            question=request.get("question"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise JudgeError("invalid_request") from exc


def judge_request(
    request: Mapping[str, Any],
    provider: JudgeProvider,
    *,
    root: Path,
    run_id: str = "external",
) -> dict[str, Any]:
    """Run one Judge and return the external result. Raises ``JudgeError``."""
    graph = request.get("graph")
    products = request.get("role_products")
    if not isinstance(graph, Mapping) or "blocks" not in graph or "edges" not in graph:
        raise JudgeError("invalid_request")
    if not isinstance(products, Mapping):
        raise JudgeError("invalid_request")
    max_calls = request.get("max_calls", DEFAULT_JUDGE_CALLS)
    result = run_judge(
        _snapshot(request),
        graph,
        provider,
        root=root,
        run_id=run_id,
        role_products=products,
        max_calls=max_calls,
    )
    return to_external_result(result.verdict, graph)


def main(
    argv: list[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
    environ: Mapping[str, str] | None = None,
    urlopen: Callable[..., Any] | None = None,
) -> int:
    del argv
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    env = os.environ if environ is None else environ
    key = env.get(API_KEY_ENV) or env.get(FALLBACK_KEY_ENV) or ""
    if not key:
        stderr.write("missing_api_key\n")
        return EXIT_FAILED
    base_url = env.get(URL_ENV) or env.get(FALLBACK_URL_ENV) or ""
    model = env.get(MODEL_ENV) or env.get("LENOVO_GLM_MODEL") or ""
    if not base_url or not model:
        stderr.write("judge_not_configured\n")
        return EXIT_FAILED
    try:
        request = json.loads(stdin.read())
        if not isinstance(request, dict):
            raise JudgeError("invalid_request")
        text = request.get("response_text")
        if not isinstance(text, str) or not text:
            raise JudgeError("invalid_request")
        provider = ChatJudgeProvider(text, key=key, base_url=base_url, model=model, urlopen=urlopen)
        with tempfile.TemporaryDirectory(prefix="aig-judge-") as tmp:
            result = judge_request(request, provider, root=Path(tmp))
    except JudgeError as exc:
        stderr.write(exc.code + "\n")
        return EXIT_UNSCORED if exc.code == "unscored" else EXIT_FAILED
    except (ValueError, TypeError):
        stderr.write("invalid_request\n")
        return EXIT_FAILED
    stdout.write(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
