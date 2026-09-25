"""FAQ-style knowledge base with lightweight keyword retrieval.

No vector database: entries are scored by weighted token overlap, which is
enough for a small FAQ and keeps the retrieval logic easy to read and test.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_KB_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge_base.json"

TITLE_WEIGHT = 3.0
KEYWORD_WEIGHT = 2.0
CONTENT_WEIGHT = 1.0

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {"a", "an", "the", "is", "are", "to", "of", "and", "or", "in", "on", "for",
     "my", "i", "it", "do", "how", "not", "with", "can", "cannot", "why"}
)


def tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


@dataclass(frozen=True)
class KBEntry:
    id: str
    title: str
    category: str
    keywords: tuple[str, ...]
    content: str


@dataclass(frozen=True)
class SearchResult:
    entry: KBEntry
    score: float


class KnowledgeBase:
    def __init__(self, entries: list[KBEntry]):
        self._entries = entries
        self._index = [
            (
                entry,
                tokenize(entry.title),
                tokenize(" ".join(entry.keywords)),
                tokenize(entry.content),
            )
            for entry in entries
        ]

    @classmethod
    def from_file(cls, path: Path | str = DEFAULT_KB_PATH) -> KnowledgeBase:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = [
            KBEntry(
                id=item["id"],
                title=item["title"],
                category=item["category"],
                keywords=tuple(item.get("keywords", [])),
                content=item["content"],
            )
            for item in raw
        ]
        return cls(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        results = []
        for entry, title_t, keyword_t, content_t in self._index:
            score = (
                TITLE_WEIGHT * len(query_tokens & title_t)
                + KEYWORD_WEIGHT * len(query_tokens & keyword_t)
                + CONTENT_WEIGHT * len(query_tokens & content_t)
            )
            if score > 0:
                results.append(SearchResult(entry=entry, score=score))

        results.sort(key=lambda r: (-r.score, r.entry.id))
        return results[:top_k]
