"""Speech-to-text with faster-whisper (optional dependency).

Text-to-speech runs in the browser (Web Speech API), which has German, English
and Spanish voices on all major platforms, so the backend only does ASR.
"""
from __future__ import annotations

import io
from functools import lru_cache


class SpeechUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _model(name: str, device: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # pragma: no cover - depends on optional install
        raise SpeechUnavailable("install faster-whisper (pip install -e '.[speech]') to enable speech input") from e
    compute = "int8" if device in ("cpu", "auto") else "float16"
    return WhisperModel(name, device=device, compute_type=compute)


def transcribe(audio: bytes, lang: str, model_name: str, device: str) -> dict:
    model = _model(model_name, device)
    segments, info = model.transcribe(io.BytesIO(audio), language=lang, vad_filter=True, beam_size=5)
    text = " ".join(s.text.strip() for s in segments).strip()
    return {"text": text, "language": info.language, "duration": round(info.duration, 2)}
