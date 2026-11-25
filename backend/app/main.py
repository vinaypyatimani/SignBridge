"""HTTP + WebSocket API.

WebSocket ``/ws/{session_id}`` messages (JSON):

client -> server
    {"type": "signs", "sign_lang": "ase", "spoken_lang": "en", "mode": "doctor",
     "signs": [{"candidates": [["HEADACHE", 0.82], ["HEAD", 0.09], ...]}, ...]}
    {"type": "text", "text": "Where does it hurt?", "spoken_lang": "en", "sign_lang": "ase", "mode": "doctor"}

server -> client
    {"type": "caption", "text": ..., "glosses": [...], "confidence": 0.9, "confirm": null | {...}}
    {"type": "sign_sequence", "text": ..., "items": [{"type": "sign"|"spell"|"text", ...}], "fps": 20}
    {"type": "error", "message": ...}

Video never reaches this server: the browser sends only recognized glosses.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .agent import SignAgent
from .config import Settings
from .context import ContextPacks
from .converter import llm_provider, make_converter
from .languages import SIGN_LANGUAGES, SPOKEN_LANGUAGES
from .sign_library import Libraries
from .speech import SpeechUnavailable, transcribe

log = logging.getLogger("trisign")


class SummaryRequest(BaseModel):
    lang: str = "en"


def create_app(settings: Settings | None = None, converter=None) -> FastAPI:
    settings = settings or Settings()
    packs = ContextPacks(settings.data_dir / "context_packs")
    libs = Libraries(settings.data_dir / "sign_library", list(SIGN_LANGUAGES))
    agent = SignAgent(settings, converter or make_converter(settings), packs, libs)

    app = FastAPI(title="TriSign API", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"],
                       allow_headers=["*"])
    app.state.agent = agent

    @app.get("/api/health")
    def health():
        return {"ok": True, "llm": llm_provider(settings) or "rule-based fallback",
                "library": {l: len(libs[l].glosses) for l in SIGN_LANGUAGES}}

    @app.get("/api/languages")
    def languages():
        return {"sign": SIGN_LANGUAGES, "spoken": SPOKEN_LANGUAGES}

    @app.get("/api/modes")
    def modes():
        return packs.modes()

    @app.get("/api/modes/{mode}/phrases")
    def phrases(mode: str, lang: str = "en"):
        pack = packs.get(mode)
        if not pack:
            raise HTTPException(404, f"unknown mode {mode}")
        return [{"id": p.id, "text": p.text.get(lang), "gloss": p.gloss} for p in pack.phrases]

    @app.post("/api/sessions/{session_id}/summary")
    def summary(session_id: str, req: SummaryRequest):
        if req.lang not in SPOKEN_LANGUAGES:
            raise HTTPException(400, "lang must be en, de or es")
        return {"summary": agent.summary(session_id, req.lang), "turns": agent.history(session_id)}

    @app.post("/api/transcribe")
    async def stt(audio: UploadFile = File(...), lang: str = Form("en")):
        if lang not in SPOKEN_LANGUAGES:
            raise HTTPException(400, "lang must be en, de or es")
        try:
            return transcribe(await audio.read(), lang, settings.whisper_model, settings.whisper_device)
        except SpeechUnavailable as e:
            raise HTTPException(501, str(e))

    @app.websocket("/ws/{session_id}")
    async def ws(socket: WebSocket, session_id: str):
        await socket.accept()
        try:
            while True:
                msg = await socket.receive_json()
                try:
                    common = {"sign_lang": msg["sign_lang"], "spoken_lang": msg["spoken_lang"],
                              "mode": msg.get("mode")}
                    if msg.get("type") == "signs":
                        cands = [s["candidates"] for s in msg["signs"]]
                        out = await asyncio.to_thread(agent.run, session_id, direction="sign_to_text",
                                                    candidates=cands, **common)
                    elif msg.get("type") == "text":
                        out = await asyncio.to_thread(agent.run, session_id, direction="text_to_sign",
                                                    text=msg["text"], **common)
                    else:
                        out = {"type": "error", "message": f"unknown message type {msg.get('type')!r}"}
                except (KeyError, ValueError, TypeError) as e:
                    out = {"type": "error", "message": f"bad message: {e}"}
                except Exception as e:  # LLM or network failure: tell the client, keep the socket
                    log.exception("agent failed")
                    out = {"type": "error", "message": f"processing failed: {type(e).__name__}"}
                await socket.send_json(out)
        except WebSocketDisconnect:
            pass

    return app


app = create_app()
