# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Okapi BM25 for causal-memory initial ranking.

No third-party ranker. The IDF is the Lucene variant, which stays non-negative:

    IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score  = sum_t IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))

An empty query scores every document 0.0. Callers keep that tie in write order.
A non-empty query scores 0.0 when the document shares no token with the query.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Case-folded word tokens. Punctuation separates tokens; it is not matched."""
    if not text:
        return []
    return _TOKEN.findall(text.casefold())


class BM25Index:
    """In-memory BM25 over a corpus. Document ``i`` is ``corpus[i]``."""

    def __init__(
        self,
        corpus: Sequence[str],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        if isinstance(k1, bool) or not isinstance(k1, (int, float)) or k1 < 0:
            raise ValueError("k1 must be a number >= 0")
        if isinstance(b, bool) or not isinstance(b, (int, float)) or not 0.0 <= float(b) <= 1.0:
            raise ValueError("b must be a number in [0, 1]")
        self.k1 = float(k1)
        self.b = float(b)
        self._docs: list[Counter[str]] = []
        self._lengths: list[int] = []
        document_frequency: Counter[str] = Counter()
        for text in corpus:
            counts = Counter(tokenize(text))
            self._docs.append(counts)
            length = sum(counts.values())
            self._lengths.append(length)
            document_frequency.update(counts.keys())
        self._document_frequency = document_frequency
        self.n = len(self._docs)
        total_length = sum(self._lengths)
        self.avgdl = (total_length / self.n) if self.n else 0.0

    def idf(self, term: str) -> float:
        frequency = self._document_frequency.get(term, 0)
        if frequency == 0 or self.n == 0:
            return 0.0
        return math.log(1.0 + (self.n - frequency + 0.5) / (frequency + 0.5))

    def scores(self, query: str) -> list[float]:
        """One score per corpus document, in corpus order.

        An empty query returns 0.0 for every document. Those ties are not
        "no shared terms": callers retain every document and keep write order.
        """
        if self.n == 0:
            return []
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            return [0.0] * self.n
        idf = {term: self.idf(term) for term in terms}
        k1 = self.k1
        b = self.b
        avgdl = self.avgdl
        scored: list[float] = []
        for counts, length in zip(self._docs, self._lengths, strict=True):
            if avgdl == 0.0:
                norm = 0.0
            else:
                norm = 1.0 - b + b * (length / avgdl)
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + k1 * norm
                if denom == 0.0:
                    continue
                score += idf[term] * (tf * (k1 + 1.0)) / denom
            scored.append(score)
        return scored
