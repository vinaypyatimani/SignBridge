"""The sign library: verified reference trajectories the avatar plays.

Built by ``ml/trisign_ml/build_sign_library.py`` into ``data/sign_library/<lang>.json``.
Fingerspelling letters are stored as glosses named ``FS:<letter>`` when available.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SignLibrary:
    lang: str
    fps: float = 20.0
    signs: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, lang: str) -> "SignLibrary":
        if not path.exists():
            return cls(lang)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(lang, data.get("fps", 20.0), data.get("signs", {}))

    @property
    def glosses(self) -> list[str]:
        return sorted(g for g in self.signs if not g.startswith("FS:"))

    def has(self, gloss: str) -> bool:
        return gloss in self.signs

    def frames(self, gloss: str) -> list | None:
        entry = self.signs.get(gloss)
        return entry["frames"] if entry else None

    def can_spell(self, letter: str) -> bool:
        return f"FS:{letter}" in self.signs

    def spell_plan(self, word: str) -> list[str]:
        """Letters to fingerspell, uppercased, accents kept where the alphabet has them (Ñ, Ä...)."""
        out = []
        for ch in word.upper():
            if ch.isalnum():
                if not self.can_spell(ch):
                    base = unicodedata.normalize("NFD", ch)[0]
                    ch = base if self.can_spell(base) else ch
                out.append(ch)
        return out


class Libraries:
    def __init__(self, root: Path, langs: list[str]):
        self.root = root
        self.by_lang = {l: SignLibrary.load(root / f"{l}.json", l) for l in langs}

    def __getitem__(self, lang: str) -> SignLibrary:
        return self.by_lang[lang]
