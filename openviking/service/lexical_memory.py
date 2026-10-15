# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Persistent account-scoped BM25 index. This module has no model dependency."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from openviking.session.bm25 import tokenize

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
        words = list(dict.fromkeys(tokenize(query)))
        if not words:
            return []
        if not 1 <= limit <= 50 or offset < 0:
            raise ValueError("invalid lexical result window")
        expression = " OR ".join('"' + word.replace('"', '""') + '"' for word in words)
        prefix = target_uri.rstrip("/") + "/"
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT documents.*,bm25(terms) AS rank FROM terms JOIN documents ON terms.rowid=documents.rowid "
                "WHERE terms MATCH ? AND (uri=? OR substr(uri,1,?)=?) "
                "ORDER BY rank ASC,uri ASC LIMIT ? OFFSET ?",
                (expression, target_uri, len(prefix), prefix, limit, offset),
            ).fetchall()
        return [self._hit(row, query, -row["rank"]) for row in rows]

    @staticmethod
    def _hit(row: sqlite3.Row, query: str, score: float) -> dict[str, Any]:
        text = row["content"]
        positions = [text.casefold().find(term) for term in tokenize(query)]
        positions = [position for position in positions if position >= 0]
        start = max(0, min(positions, default=0) - 200)
        return {
            "uri": row["uri"],
            "content_hash": row["content_hash"],
            "level": 2,
            "excerpt": text[start : start + 1600],
            "score": score,
            "retrieval_mode": "bm25",
            "metadata": json.loads(row["metadata"]),
        }
