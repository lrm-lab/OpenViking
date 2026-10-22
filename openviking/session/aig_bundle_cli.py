# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Build the Judge's evidence bundle from one candidate branch's AO rows.

An external host runs ``python -m openviking.session.aig_bundle_cli`` as a
subprocess. stdin is one JSON object::

    {"ao_records": [AORecord dicts], "watermark": "...", "question": "...",
     "task_run_id": "safe-id", "out": "/path/bundle.json"}

Builder, Critic, Refiner and Reader (``run_production_aig``) must all finish. The
frozen graph and the four roles' products are then written to ``out`` in the shape
``aig_judge_cli`` reads, and the last stdout line is ``{"judge_bundle": out}``. Any
incomplete role, a fallback, or a model failure exits non-zero with only a stable
code on stderr and no bundle, so a half-built branch is never scored.

The role model is any OpenAI-compatible chat endpoint (key from the environment).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from openviking.session.aig_production_pipeline import AigRoleProviders, run_production_aig
from openviking.session.aig_tool_surface import AoSnapshot
from openviking.session.aig_toolloop_judge import ROLES
from openviking.session.aig_toolloop_reader import load_frozen_paper_graph
from openviking.session.aig_vlm_roles import VLMBuilder, VLMCritic, VLMReader, VLMRefiner
from openviking.session.ao_ledger import AORecord
from openviking.session.extraction_write import NATIVE_EDGE_INDEX_FILENAME, NativeEdgeViewStore
from openviking.session.paper_graph_compile import AUDIT_DIRNAME

EXIT_FAILED = 2
DEFAULT_DEADLINE_SECONDS = 1200.0


class ChatHandle:
    """Minimal ``get_completion_async`` over an OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        key: str,
        base_url: str,
        model: str,
        urlopen: Callable[..., Any] | None = None,
        max_tokens: int = 8192,
    ) -> None:
        self._key = key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._urlopen = urlopen or urllib.request.urlopen
        self._max_tokens = max_tokens

    async def get_completion_async(self, *, messages: list[dict[str, str]], **_: Any) -> str:
        return await asyncio.to_thread(self._complete, messages)

    def _complete(self, messages: list[dict[str, str]]) -> str:
        last: Exception | None = None
        for attempt in range(3):
            try:
                return self._complete_once(messages)
            except Exception as exc:  # transport errors and empty content are transient
                last = exc
                time.sleep(1.0 + attempt)
        assert last is not None
        raise last

    def _complete_once(self, messages: list[dict[str, str]]) -> str:
        body = json.dumps(
            {"model": self._model, "messages": messages, "max_tokens": self._max_tokens}
        ).encode()
        request = urllib.request.Request(
            self._url,
            data=body,
            method="POST",
            headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"},
        )
        with self._urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode())
        content = payload["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("model_response_missing")
        return _first_object_text(content)


def _first_object_text(content: str) -> str:
    """The first JSON object in ``content``, verbatim, so strict role parsing still applies.

    Chat models often wrap the object in prose or fences, or add a second object.
    Role parsers stay strict (duplicate keys, non-finite numbers) on what they get.
    """
    decoder = json.JSONDecoder()
    start = content.find("{")
    while start != -1:
        try:
            value, end = decoder.raw_decode(content[start:])
        except ValueError:
            start = content.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return content[start : start + end]
        start = content.find("{", start + 1)
    return content


_ARGUMENT_FORMAT = (
    "\nArgument formats (use exactly these keys, no others): "
    'add_block {"block_id":"<id>","kind":"task|orchestrator|agent|conclusion",'
    '"fields":{<the required fields of that kind, including step_refs>}}; '
    'add_inheritance_note {"from":"<earlier block id>","to":"<later block id>",'
    '"inherited":"<text>","effect":"<text>"}; submit {}. '
    'All block content lives inside "fields"; step_refs are AO row indices. '
    "You are done only after you call submit {}; do that right after the last note, "
    "at most one defect mark. Never stop without submit. "
    "If a tool call is rejected, read the error code in the history and fix the arguments."
)


_CRITIC_FORMAT = (
    '\nFinish with {"violations":[...]} and no other keys. Each violation has exactly '
    '{"object_id":"<block id>","kind":"semantic|granularity","severity":"high|low",'
    '"evidence_ao_refs":[<int AO row indices>],"note":"<why>"}. '
    'Return {"violations":[]} when the graph is clean. Read steps with '
    '{"verb":"read_step","arguments":{"index":<int>}} before judging.'
)
_READER_FORMAT = (
    "\nThe final report has exactly these keys: responsible_agent (string), "
    "decisive_ao_refs (list of int AO row indices you read), reason (string), "
    "critical_block_id (a block id of the graph), unresolved_areas (list of strings, "
    "empty when everything is resolved). Read the decisive rows before reporting."
)


class _WithFormat:
    """Role mixin: the tool catalog does not spell out argument shapes, so say them."""

    FORMAT = _ARGUMENT_FORMAT

    def _ask(self, prompt: str, state: Mapping[str, Any]) -> dict[str, Any]:
        return super()._ask(prompt + self.FORMAT, state)  # type: ignore[misc]

    def __call__(self, turn: Any) -> Any:
        """Re-ask when the model breaks the strict reply shape; the turn is unchanged."""
        last: Exception | None = None
        for _ in range(3):
            try:
                return super().__call__(turn)  # type: ignore[misc]
            except (ValueError, TimeoutError) as exc:
                last = exc
        assert last is not None
        raise last


class _Builder(_WithFormat, VLMBuilder):
    pass


class _Critic(_WithFormat, VLMCritic):
    FORMAT = _CRITIC_FORMAT


class _Refiner(_WithFormat, VLMRefiner):
    pass


class _Reader(_WithFormat, VLMReader):
    FORMAT = _READER_FORMAT


def providers_for(handle: Any, deadline_seconds: float) -> AigRoleProviders:
    deadline = time.monotonic() + deadline_seconds
    return AigRoleProviders(
        _Builder(handle, deadline),
        _Critic(handle, deadline),
        _Refiner(handle, deadline),
        _Reader(handle, deadline),
    )


class BundleError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _snapshot(request: Mapping[str, Any]) -> AoSnapshot:
    rows = request.get("ao_records")
    if not isinstance(rows, list) or not rows:
        raise BundleError("invalid_request")
    try:
        return AoSnapshot(
            tuple(AORecord.from_dict(row) for row in rows),
            watermark=str(request["watermark"]),
            question=request.get("question"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleError("invalid_request") from exc


def _audit_products(root: Path, run_id: str) -> dict[str, int]:
    path = root / AUDIT_DIRNAME / run_id / "trajectory.jsonl"
    kinds: dict[str, int] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                kind = str(json.loads(line).get("record_kind", "unknown"))
                kinds[kind] = kinds.get(kind, 0) + 1
    return kinds


def build_bundle(
    request: Mapping[str, Any], providers: AigRoleProviders, *, root: Path
) -> dict[str, Any]:
    """Run the four roles and return the bundle. Raises ``BundleError``."""
    snapshot = _snapshot(request)
    run_id = request.get("task_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise BundleError("invalid_request")
    result = run_production_aig(snapshot, root=root, task_run_id=run_id, providers=providers)
    if result.status != "frozen" or result.report_path is None:
        raise BundleError(f"roles_incomplete_{result.reason}")
    revision = next(
        (
            item
            for item in NativeEdgeViewStore(root / NATIVE_EDGE_INDEX_FILENAME).revisions()
            if item.task_run_id == run_id
        ),
        None,
    )
    if revision is None:
        raise BundleError("revision_missing")
    sidecar = root / AUDIT_DIRNAME / run_id / f"{revision.revision_id}.paper-graph.json"
    graph = load_frozen_paper_graph(revision, snapshot, sidecar_path=sidecar)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    audit = _audit_products(root, run_id)
    products = {
        "builder": {"blocks": len(graph["blocks"]), "edges": len(graph["edges"])},
        "critic": {"audit_records": audit},
        "refiner": {"audit_records": audit, "revision_id": revision.revision_id},
        "reader": report["report"],
    }
    if set(products) != set(ROLES):
        raise BundleError("roles_incomplete")
    return {
        "ao_records": [row.to_dict() for row in snapshot.records],
        "watermark": snapshot.watermark,
        "question": snapshot.question,
        "graph": {"blocks": graph["blocks"], "edges": graph["edges"]},
        "role_products": products,
    }


def main(
    argv: list[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
    environ: Mapping[str, str] | None = None,
    urlopen: Callable[..., Any] | None = None,
    providers: AigRoleProviders | None = None,
) -> int:
    del argv
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    env = os.environ if environ is None else environ
    try:
        request = json.loads(stdin.read())
        if not isinstance(request, dict):
            raise BundleError("invalid_request")
        out = request.get("out")
        if not isinstance(out, str) or not out:
            raise BundleError("invalid_request")
        providers_factory: Callable[[], AigRoleProviders] | None = None
        if providers is None:
            key = env.get("BUNDLE_API_KEY") or env.get("LENOVO_GATEWAY_API_KEY") or ""
            if not key:
                raise BundleError("missing_api_key")
            base_url = env.get("BUNDLE_BASE_URL") or env.get("LENOVO_GATEWAY_URL") or ""
            model = env.get("BUNDLE_MODEL") or ""
            if not base_url or not model:
                raise BundleError("bundle_not_configured")
            handle = ChatHandle(key=key, base_url=base_url, model=model, urlopen=urlopen)
            deadline = float(env.get("BUNDLE_DEADLINE_SECONDS") or DEFAULT_DEADLINE_SECONDS)
            providers_factory = lambda: providers_for(handle, deadline)  # noqa: E731
            providers = providers_factory()
        attempts = max(1, int(env.get("BUNDLE_ATTEMPTS") or 3))
        bundle = None
        for attempt in range(attempts):
            if attempt:
                providers = providers_factory() if providers_factory else providers
            try:
                with tempfile.TemporaryDirectory(prefix="aig-bundle-") as tmp:
                    bundle = build_bundle(request, providers, root=Path(tmp))
                break
            except BundleError as exc:
                if attempt + 1 >= attempts or exc.code == "invalid_request":
                    raise
        assert bundle is not None
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(bundle, ensure_ascii=False, sort_keys=True), "utf-8")
        os.replace(temporary, target)
    except BundleError as exc:
        stderr.write(exc.code + "\n")
        return EXIT_FAILED
    except Exception:  # codes only: model bodies and keys never reach stderr
        stderr.write("bundle_failed\n")
        return EXIT_FAILED
    stdout.write(json.dumps({"judge_bundle": out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
