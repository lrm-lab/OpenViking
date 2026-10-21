# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Persistent account-scoped BM25 index. This module has no model dependency."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from openviking.session.bm25 import tokenize

_QUOTED_SEGMENT = re.compile(r'"([^"]*)"')
# Bounded excerpt search: per-term position cap and candidate-window cap keep the
# scan linear even for pathological documents or queries.
_EXCERPT_TERM_POSITION_CAP = 50
_EXCERPT_QUERY_TERMS_CAP = 12
_EXCERPT_WINDOW = 1600

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  rowid INTEGER PRIMARY KEY, uri TEXT NOT NULL UNIQUE, content TEXT NOT NULL,
  tokens TEXT NOT NULL, content_hash TEXT NOT NULL, metadata TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS terms USING fts5(tokens, content=documents, content_rowid=rowid);
CREATE TRIGGER IF NOT EXISTS document_insert AFTER INSERT ON documents BEGIN
  INSERT INTO terms(rowid,tokens) VALUES(new.rowid,new.tokens);
END;
CREATE TRIGGER IF NOT EXISTS document_delete AFTER DELETE ON documents BEGIN
  INSERT INTO terms(terms,rowid,tokens) VALUES('delete',old.rowid,old.tokens);
END;
CREATE TRIGGER IF NOT EXISTS document_update AFTER UPDATE ON documents BEGIN
  INSERT INTO terms(terms,rowid,tokens) VALUES('delete',old.rowid,old.tokens);
  INSERT INTO terms(rowid,tokens) VALUES(new.rowid,new.tokens);
END;
CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), generation INTEGER NOT NULL);
INSERT OR IGNORE INTO state VALUES(1,0);
"""


def _match_expressions(query: str) -> tuple[str | None, str | None]:
    """Build (strict, relaxed) FTS5 MATCH expressions from one query string.

    A multi-word quoted segment carries adjacency intent and becomes an FTS5
    phrase clause that is required (AND). A single quoted word is stylistic
    emphasis, not a hard constraint, so it stays in the OR pool. The relaxed
    expression is the legacy pure-OR expression over every token of the query
    and runs only when the strict form matches nothing, so recall never drops
    below legacy. ``None`` means the query has no tokens at all.
    """
    required: list[str] = []
    for segment in _QUOTED_SEGMENT.findall(query):
        words = list(dict.fromkeys(tokenize(segment)))
        if len(words) > 1:
            required.append('"' + " ".join(words) + '"')
    all_words = list(dict.fromkeys(tokenize(query)))
    if not all_words:
        return None, None
    relaxed = " OR ".join('"' + word + '"' for word in all_words)
    if not required:
        return relaxed, relaxed
    # The OR pool keeps every query token (phrase words included): the phrase
    # clause subsumes them for matching, but their presence in the expression
    # measurably improves bm25 ranking of phrase-matching documents.
    strict = " AND ".join(required) + " AND (" + relaxed + ")"
    return strict, relaxed


class LexicalMemoryIndex:
    """SQLite FTS5 BM25 over explicit resource text, never a vector fallback."""

    def __init__(self, root: Path, account_id: str) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / (hashlib.sha256(account_id.encode()).hexdigest() + ".sqlite3")
        with self._connection() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            with connection:
                yield connection
        finally:
            connection.close()

    @property
    def generation(self) -> int:
        with self._connection() as connection:
            return connection.execute("SELECT generation FROM state WHERE id=1").fetchone()[0]

    def upsert(self, uri: str, content: str, metadata: dict[str, Any] | None = None) -> str:
        return self.upsert_many([(uri, content, metadata or {})])[uri]

    def upsert_many(self, resources: list[tuple[str, str, dict[str, Any]]]) -> dict[str, str]:
        rows = []
        for uri, content, metadata in resources:
            if not uri.startswith("viking://resources/") or not content.strip():
                raise ValueError(
                    "lexical resources require a shared resource URI and nonempty text"
                )
            digest = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
            serialized = json.dumps(metadata, sort_keys=True, ensure_ascii=False)
            rows.append((uri, content, " ".join(tokenize(content)), digest, serialized))
        changed = False
        with self._connection() as connection:
            for row in rows:
                previous = connection.execute(
                    "SELECT content_hash,metadata FROM documents WHERE uri=?", (row[0],)
                ).fetchone()
                if (
                    previous
                    and previous["content_hash"] == row[3]
                    and previous["metadata"] == row[4]
                ):
                    continue
                connection.execute(
                    "INSERT INTO documents(uri,content,tokens,content_hash,metadata) VALUES(?,?,?,?,?) "
                    "ON CONFLICT(uri) DO UPDATE SET content=excluded.content,tokens=excluded.tokens,"
                    "content_hash=excluded.content_hash,metadata=excluded.metadata",
                    row,
                )
                changed = True
            if changed:
                connection.execute("UPDATE state SET generation=generation+1 WHERE id=1")
        return {row[0]: row[3] for row in rows}

    def counts(self) -> dict[str, int]:
        with self._connection() as connection:
            return {
                row[0]: row[1]
                for row in connection.execute(
                    "SELECT json_extract(metadata,'$.kind'),COUNT(*) FROM documents GROUP BY json_extract(metadata,'$.kind')"
                )
            }

    def get(self, uri: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM documents WHERE uri=?", (uri,)).fetchone()
        return self._hit(row, "", 0.0) if row else None

    def find(
        self,
        query: str,
        *,
        target_uri: str = "viking://resources",
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 50 or offset < 0:
            raise ValueError("invalid lexical result window")
        strict, relaxed = _match_expressions(query)
        prefix = target_uri.rstrip("/") + "/"
        with self._connection() as connection:
            # Keep ranking and materialization on one snapshot. Sorting full
            # documents here copies gigabytes for broad queries before LIMIT.
            connection.execute("BEGIN")
            def run(expression: str) -> list[sqlite3.Row]:
                return connection.execute(
                    "SELECT documents.rowid,documents.uri,bm25(terms) AS rank FROM terms JOIN documents ON terms.rowid=documents.rowid "
                    "WHERE terms MATCH ? AND (uri=? OR substr(uri,1,?)=?) "
                    "ORDER BY rank ASC,uri ASC LIMIT ? OFFSET ?",
                    (expression, target_uri, len(prefix), prefix, limit, offset),
                ).fetchall()
            ranked = run(strict)
            if strict is None:
                ranked = []
            elif not ranked and relaxed is not None and relaxed != strict:
                # Quoted constraints that match nothing degrade to bag-of-words
                # recall instead of returning an empty page.
                ranked = run(relaxed)
            return [
                self._hit(
                    connection.execute(
                        "SELECT * FROM documents WHERE rowid=?", (candidate["rowid"],)
                    ).fetchone(),
                    query,
                    -candidate["rank"],
                )
                for candidate in ranked
            ]

    @staticmethod
    def _hit(row: sqlite3.Row, query: str, score: float) -> dict[str, Any]:
        text = row["content"]
        terms = list(dict.fromkeys(tokenize(query)))[:_EXCERPT_QUERY_TERMS_CAP]
        positions_by_term: dict[str, list[int]] = {}
        lowered = text.casefold()
        for term in terms:
            hits: list[int] = []
            position = lowered.find(term)
            while position >= 0 and len(hits) < _EXCERPT_TERM_POSITION_CAP:
                hits.append(position)
                position = lowered.find(term, position + len(term))
            if hits:
                positions_by_term[term] = hits
        start = 0
        if positions_by_term:
            # Densest-window excerpt: prefer the window covering the most
            # distinct query terms, then the most total occurrences, then the
            # earliest window. First-occurrence excerpts routinely missed every
            # constraint but one on multi-hop queries.
            best: tuple[int, int, int] | None = None  # (distinct, total, -start)
            candidates = sorted({p for hits in positions_by_term.values() for p in hits})
            for anchor in candidates:
                window_start = max(0, anchor - 200)
                window_end = window_start + _EXCERPT_WINDOW
                distinct = sum(
                    1 for hits in positions_by_term.values()
                    if any(window_start <= p < window_end for p in hits)
                )
                total = sum(
                    1 for hits in positions_by_term.values()
                    for p in hits if window_start <= p < window_end
                )
                key = (distinct, total, -window_start)
                if best is None or key > best:
                    best = key
            if best is not None:
                start = -best[2]
        return {
            "uri": row["uri"],
            "content_hash": row["content_hash"],
            "level": 2,
            "excerpt": text[start : start + _EXCERPT_WINDOW],
            "score": score,
            "retrieval_mode": "bm25",
            "metadata": json.loads(row["metadata"]),
        }
