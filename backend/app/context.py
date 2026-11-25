"""Context modes and phrase-pack retrieval.

A phrase pack (``data/context_packs/<mode>.json``) holds common sentences for a
situation, written in German, English and Spanish, with draft gloss sequences
for DGS, ASL and LSE. Retrieval is a small BM25 over all three spoken versions,
so it works whatever language the hearing person uses. Swap in a vector store
(Qdrant) by implementing ``search`` with the same signature.

The gloss sequences shipped here are drafts and must be reviewed by Deaf
signers of each language before real use.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[\wäöüßñáéíóúü]+", re.IGNORECASE)


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


@dataclass
class Phrase:
    id: str
    text: dict[str, str]     # spoken lang -> sentence
    gloss: dict[str, list]   # sign lang -> gloss sequence


class PhrasePack:
    def __init__(self, mode: str, title: dict[str, str], phrases: list[Phrase], vocabulary: dict[str, list]):
        self.mode, self.title, self.phrases, self.vocabulary = mode, title, phrases, vocabulary
        self._docs = [tokens(" ".join(p.text.values())) for p in phrases]
        self._df = Counter(t for d in self._docs for t in set(d))
        self._avg = sum(len(d) for d in self._docs) / max(1, len(self._docs))

    @classmethod
    def load(cls, path: Path) -> "PhrasePack":
        d = json.loads(path.read_text(encoding="utf-8"))
        phrases = [Phrase(p["id"], p["text"], p.get("gloss", {})) for p in d["phrases"]]
        return cls(d["mode"], d["title"], phrases, d.get("vocabulary", {}))

    def search(self, query: str, k: int = 4) -> list[tuple[Phrase, float]]:
        q = tokens(query)
        n = len(self._docs)
        scored = []
        for phrase, doc in zip(self.phrases, self._docs):
            tf = Counter(doc)
            s = 0.0
            for t in q:
                if t in tf:
                    idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                    s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(doc) / self._avg))
            if s > 0:
                scored.append((phrase, s))
        return sorted(scored, key=lambda x: -x[1])[:k]


class ContextPacks:
    def __init__(self, root: Path):
        self.packs = {p.mode: p for p in (PhrasePack.load(f) for f in sorted(root.glob("*.json")))}

    def get(self, mode: str | None) -> PhrasePack | None:
        return self.packs.get(mode or "")

    def modes(self) -> list[dict]:
        return [{"id": p.mode, "title": p.title, "phrases": len(p.phrases)} for p in self.packs.values()]
