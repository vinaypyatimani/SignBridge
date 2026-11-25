from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None)
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY") or None)
    # Empty means the default for whichever provider has a key (see converter.make_converter).
    llm_model: str | None = field(default_factory=lambda: os.getenv("TRISIGN_LLM_MODEL") or None)
    whisper_model: str = field(default_factory=lambda: os.getenv("TRISIGN_WHISPER_MODEL", "large-v3"))
    whisper_device: str = field(default_factory=lambda: os.getenv("TRISIGN_WHISPER_DEVICE", "auto"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("TRISIGN_DATA_DIR", ROOT / "data")))
    cors_origins: list[str] = field(default_factory=lambda: os.getenv(
        "TRISIGN_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","))
    # Below this top-1 probability for any sign, the signer is asked to confirm before captions show.
    confirm_threshold: float = field(default_factory=lambda: float(os.getenv("TRISIGN_CONFIRM_THRESHOLD", "0.6")))
