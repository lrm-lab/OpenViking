#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Export OV session trees to AORecord JSONL (ADR-0045 Wave E2).

Default output (frozen in README-aig0045.md):
    benchmark/causal_memory/out/ao_records.jsonl

The 1933 prototype stores drained sessions at
    /data/viking/evol-389e96deae16/user/eval-admin/sessions
inside container ``coforge-openviking-prototype-1``. Copy that tree to the
host first; this script only reads a local directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.causal_memory.ao_record_export import (  # noqa: E402
    DEFAULT_EXPORT_PATH,
    PROBED_SESSIONS_ROOT,
    export_sessions,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions-root",
        type=Path,
        required=True,
        help=f"Host copy of a session tree (container path was {PROBED_SESSIONS_ROOT}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / DEFAULT_EXPORT_PATH,
        help=f"AORecord JSONL path (default {DEFAULT_EXPORT_PATH}).",
    )
    args = parser.parse_args(argv)
    if not args.sessions_root.exists():
        print(f"sessions-root missing: {args.sessions_root}", file=sys.stderr)
        return 2
    summary = export_sessions(args.sessions_root, args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
