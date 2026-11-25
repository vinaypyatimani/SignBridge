"""The conversation agent (LangGraph).

Two routes share one graph and one memory per session (thread):

    sign_to_text:  retrieve context -> choose glosses + write sentence -> confidence check
    text_to_sign:  retrieve context -> simplify + gloss -> validate against the sign library

The agent never invents a sign: any gloss that is not in the verified library is
fingerspelled (if the alphabet is recorded) or shown as text beside the avatar.
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .config import Settings
from .context import ContextPacks
from .converter import Converter
from .languages import SIGN_LANGUAGES, check
from .sign_library import Libraries


class State(TypedDict, total=False):
    direction: Literal["sign_to_text", "text_to_sign"]
    sign_lang: str
    spoken_lang: str
    mode: str | None
    candidates: list[list[list]]   # per sign: [[gloss, prob], ...]
    text: str
    context_ids: list[str]
    result: dict
    history: Annotated[list[str], operator.add]


class SignAgent:
    def __init__(self, settings: Settings, converter: Converter, packs: ContextPacks, libraries: Libraries):
        self.settings, self.converter, self.packs, self.libs = settings, converter, packs, libraries
        self._phrases = {(m, p.id): p for m, pack in packs.packs.items() for p in pack.phrases}

        @tool
        def phrase_search(query: str, mode: str) -> list[str]:
            """Find phrase-pack entries for the active context mode that match the query."""
            pack = packs.get(mode)
            return [p.id for p, _ in pack.search(query)] if pack else []

        @tool
        def sign_library_lookup(gloss: str, sign_lang: str) -> dict:
            """Return the verified reference trajectory for a gloss, or a fingerspelling plan."""
            lib = libraries[sign_lang]
            word = gloss.removeprefix("FS:")
            if not gloss.startswith("FS:") and lib.has(gloss):
                return {"type": "sign", "gloss": gloss, "frames": lib.frames(gloss)}
            letters = lib.spell_plan(word)
            if gloss.startswith("FS:") or letters:
                return {"type": "spell", "gloss": gloss, "letters": [
                    {"letter": c, "frames": lib.frames(f"FS:{c}")} for c in letters]}
            return {"type": "text", "gloss": gloss, "text": word}

        self.phrase_search, self.sign_library_lookup = phrase_search, sign_library_lookup
        self.graph = self._build().compile(checkpointer=MemorySaver())

    # ---------------------------------------------------------------- nodes
    def _context(self, state: State) -> list:
        return [self._phrases[(state.get("mode"), i)] for i in state.get("context_ids", [])
                if (state.get("mode"), i) in self._phrases]

    def retrieve(self, state: State) -> dict:
        if state["direction"] == "sign_to_text":
            query = " ".join(c[0][0] for c in state["candidates"] if c).replace("-", " ")
        else:
            query = state["text"]
        mode = state.get("mode") or ""
        ids = self.phrase_search.invoke({"query": query, "mode": mode}) if mode else []
        if state["direction"] == "sign_to_text" and mode:
            # Gloss words rarely match spoken text, so also include the whole (small) pack.
            pack = self.packs.get(mode)
            ids = list(dict.fromkeys(ids + [p.id for p in pack.phrases][:12]))
        return {"context_ids": ids}

    def sign_to_text(self, state: State) -> dict:
        cands = [[(g, float(p)) for g, p in c] for c in state["candidates"]]
        res = self.converter.sign_to_text(state["sign_lang"], state["spoken_lang"], cands,
                                          self._context(state), state.get("history", []))
        low = [i for i, c in enumerate(cands) if not c or c[0][1] < self.settings.confirm_threshold]
        confirm = None
        if low or res.confidence < 0.6:
            confirm = {"uncertain_signs": [{"index": i, "options": [g for g, _ in cands[i][:3]]} for i in low],
                       "alternatives": res.alternatives[:2]}
        result = {"type": "caption", "text": res.text, "glosses": res.glosses,
                  "confidence": round(res.confidence, 3), "confirm": confirm}
        if confirm:
            return {"result": result}
        short = SIGN_LANGUAGES[state["sign_lang"]]["short"]
        return {"result": result, "history": [f"Signer ({short}): {res.text}"]}

    def text_to_sign(self, state: State) -> dict:
        lib = self.libs[state["sign_lang"]]
        pack = self.packs.get(state.get("mode"))
        vocab = sorted(set(lib.glosses))
        res = self.converter.text_to_sign(state["sign_lang"], state["spoken_lang"], state["text"], vocab,
                                          self._context(state), state.get("history", []))
        items = [self.sign_library_lookup.invoke({"gloss": g, "sign_lang": state["sign_lang"]})
                 for g in res.glosses]
        missing = [i["gloss"] for i in items if i["type"] != "sign" and not i["gloss"].startswith("FS:")]
        result = {"type": "sign_sequence", "text": state["text"], "simplified": res.simplified,
                  "glosses": res.glosses, "items": items, "fps": lib.fps,
                  "not_in_library": missing,
                  "pack_has_glosses": bool(pack and set(res.glosses) & set(pack.vocabulary.get(state["sign_lang"], [])))}
        return {"result": result, "history": [f"Speaker ({state['spoken_lang']}): {state['text']}"]}

    def _build(self) -> StateGraph:
        g = StateGraph(State)
        g.add_node("retrieve", self.retrieve)
        g.add_node("sign_to_text", self.sign_to_text)
        g.add_node("text_to_sign", self.text_to_sign)
        g.add_edge(START, "retrieve")
        g.add_conditional_edges("retrieve", lambda s: s["direction"], ["sign_to_text", "text_to_sign"])
        g.add_edge("sign_to_text", END)
        g.add_edge("text_to_sign", END)
        return g

    # ------------------------------------------------------------- public API
    def run(self, session_id: str, **inputs) -> dict:
        check(inputs["sign_lang"], inputs["spoken_lang"])
        cfg = {"configurable": {"thread_id": session_id}, "run_name": inputs["direction"]}
        out = self.graph.invoke(inputs, cfg)
        return out["result"]

    def history(self, session_id: str) -> list[str]:
        snap = self.graph.get_state({"configurable": {"thread_id": session_id}})
        return list(snap.values.get("history", [])) if snap and snap.values else []

    def summary(self, session_id: str, lang: str) -> str:
        return self.converter.summarize(lang, self.history(session_id))
