"""Gloss <-> sentence conversion.

``LLMConverter`` uses Claude (or any LangChain chat model) with structured output.
``RuleConverter`` is a deterministic fallback used when no API key is set and in
tests: it picks the top candidate, joins glosses and matches phrase packs. It is
good enough to exercise the whole pipeline, not for real conversations.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

from pydantic import BaseModel, Field

from .context import Phrase
from .languages import GRAMMAR_NOTES, SIGN_LANGUAGES, SPOKEN_LANGUAGES


class SignToTextResult(BaseModel):
    glosses: list[str] = Field(description="the gloss chosen for each sign, in signing order")
    text: str = Field(description="a fluent sentence in the target spoken language")
    confidence: float = Field(ge=0, le=1, description="how sure you are the sentence is what was signed")
    alternatives: list[str] = Field(default_factory=list,
                                    description="up to 2 other plausible sentences when unsure, else empty")


class TextToSignResult(BaseModel):
    glosses: list[str] = Field(description="gloss sequence; FS:WORD for fingerspelled words")
    simplified: str = Field(description="the plain-language sentence the glosses express, in the input language")


class Converter(Protocol):
    def sign_to_text(self, sign_lang: str, spoken_lang: str, candidates: list[list[tuple[str, float]]],
                     context: list[Phrase], history: list[str]) -> SignToTextResult: ...

    def text_to_sign(self, sign_lang: str, spoken_lang: str, text: str, vocabulary: list[str],
                     context: list[Phrase], history: list[str]) -> TextToSignResult: ...

    def summarize(self, lang: str, turns: list[str]) -> str: ...


def _examples(context: list[Phrase], sign_lang: str, spoken_lang: str) -> str:
    rows = [f"- {' '.join(p.gloss[sign_lang])}  =>  {p.text[spoken_lang]}" for p in context if sign_lang in p.gloss]
    return "\n".join(rows) or "(none)"


class LLMConverter:
    def __init__(self, llm):
        self.llm = llm
        self._s2t = llm.with_structured_output(SignToTextResult, method="json_schema")
        self._t2s = llm.with_structured_output(TextToSignResult, method="json_schema")

    def sign_to_text(self, sign_lang, spoken_lang, candidates, context, history):
        sl, tl = SIGN_LANGUAGES[sign_lang], SPOKEN_LANGUAGES[spoken_lang]["name"]
        signs = "\n".join(f"{i + 1}. " + ", ".join(f"{g} ({p:.2f})" for g, p in c) for i, c in enumerate(candidates))
        system = (
            f"You turn {sl['name']} ({sl['short']}) recognized by a camera into {tl}. "
            "The recognizer gives up to five candidate glosses per sign with probabilities. "
            "Pick one candidate per sign so the whole utterance makes sense in context, then write it as one "
            f"natural {tl} sentence. Sign languages have their own grammar: {GRAMMAR_NOTES[sign_lang]} "
            "Never add facts that were not signed. Keep numbers, doses and times exactly as signed. "
            "If the candidates do not give a clear meaning, set confidence below 0.6 and offer alternatives."
        )
        user = (f"Recent conversation:\n{chr(10).join(history[-6:]) or '(start)'}\n\n"
                f"Typical phrases in this setting:\n{_examples(context, sign_lang, spoken_lang)}\n\n"
                f"Signs, in order:\n{signs}")
        return self._s2t.invoke([("system", system), ("user", user)])

    def text_to_sign(self, sign_lang, spoken_lang, text, vocabulary, context, history):
        sl = SIGN_LANGUAGES[sign_lang]
        system = (
            f"You translate {SPOKEN_LANGUAGES[spoken_lang]['name']} into a {sl['name']} ({sl['short']}) gloss "
            f"sequence that an avatar will sign. {GRAMMAR_NOTES[sign_lang]} First simplify the sentence into short, "
            "plain clauses. Use ONLY glosses from the vocabulary list. Write names, numbers and words with no "
            "matching gloss as FS:WORD (fingerspelling) in the spelling of the original word. Do not drop "
            "medical numbers, doses or times."
        )
        user = (f"Vocabulary ({len(vocabulary)} glosses): {', '.join(vocabulary) or '(empty: fingerspell everything)'}\n\n"
                f"Examples from this setting:\n{_examples(context, sign_lang, spoken_lang)}\n\n"
                f"Recent conversation:\n{chr(10).join(history[-6:]) or '(start)'}\n\nSentence: {text}")
        return self._t2s.invoke([("system", system), ("user", user)])

    def summarize(self, lang, turns):
        name = SPOKEN_LANGUAGES[lang]["name"]
        msg = self.llm.invoke([
            ("system", f"Summarize this conversation in plain {name} for the Deaf participant to keep. "
                       "Use short bullet points. List any instructions, doses, dates and next steps exactly. "
                       "Do not add anything that was not said."),
            ("user", "\n".join(turns) or "(no turns)"),
        ])
        return msg.content if isinstance(msg.content, str) else json.dumps(msg.content)


_WORD = re.compile(r"[\wÄÖÜäöüßÑñÁÉÍÓÚáéíóú'-]+")


class RuleConverter:
    """Deterministic stand-in for the LLM."""

    def sign_to_text(self, sign_lang, spoken_lang, candidates, context, history):
        glosses = [c[0][0] for c in candidates if c]
        conf = min((c[0][1] for c in candidates if c), default=0.0)
        for p in context:  # exact phrase-pack match gives a fluent sentence
            if p.gloss.get(sign_lang) == glosses:
                return SignToTextResult(glosses=glosses, text=p.text[spoken_lang], confidence=conf)
        words = " ".join(g.replace("FS:", "").replace("-", " ").lower() for g in glosses)
        text = (words[:1].upper() + words[1:] + ".") if words else ""
        return SignToTextResult(glosses=glosses, text=text, confidence=conf)

    def text_to_sign(self, sign_lang, spoken_lang, text, vocabulary, context, history):
        norm = text.strip().lower().rstrip(".!?¿¡")
        for p in context:
            if p.text.get(spoken_lang, "").lower().strip(".!?¿¡ ") == norm.strip("¿¡ ") and sign_lang in p.gloss:
                return TextToSignResult(glosses=p.gloss[sign_lang], simplified=text)
        vocab = set(vocabulary)
        glosses = []
        for w in _WORD.findall(text):
            g = w.upper()
            glosses.append(g if g in vocab else f"FS:{g}")
        return TextToSignResult(glosses=glosses, simplified=text)

    def summarize(self, lang, turns):
        return "\n".join(f"- {t}" for t in turns) or "- (nothing was said)"


def llm_provider(settings) -> str | None:
    if settings.openai_api_key:
        return "openai"
    if settings.anthropic_api_key:
        return "claude"
    return None


def make_converter(settings) -> Converter:
    provider = llm_provider(settings)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return LLMConverter(ChatOpenAI(model=settings.llm_model or "gpt-4o-mini",
                                       api_key=settings.openai_api_key, temperature=0))
    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        return LLMConverter(ChatAnthropic(model=settings.llm_model or "claude-sonnet-5-5",
                                          api_key=settings.anthropic_api_key, temperature=0))
    return RuleConverter()
