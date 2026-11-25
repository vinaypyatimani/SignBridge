import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.converter import RuleConverter
from app.main import create_app

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def client(tmp_path):
    # A tiny library: two ASL signs and the letters A, N.
    lib = tmp_path / "sign_library"
    lib.mkdir()
    frame = [[0.0, 0.0]] * 104
    signs = {g: {"n_clips": 1, "frames": [frame] * 4} for g in ["HURT", "WHERE", "FS:A", "FS:N"]}
    (lib / "ase.json").write_text(json.dumps({"lang": "ase", "fps": 20, "signs": signs}), encoding="utf-8")
    shutil.copytree(DATA / "context_packs", tmp_path / "context_packs")  # symlinks need admin rights on Windows
    settings = Settings(anthropic_api_key=None, openai_api_key=None, data_dir=tmp_path)
    return TestClient(create_app(settings, converter=RuleConverter()))


def test_health_and_modes(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["library"]["ase"] == 2
    modes = {m["id"] for m in client.get("/api/modes").json()}
    assert modes == {"doctor", "pharmacy", "office"}


def test_phrase_packs_cover_all_languages():
    for f in (DATA / "context_packs").glob("*.json"):
        pack = json.loads(f.read_text(encoding="utf-8"))
        for p in pack["phrases"]:
            assert set(p["text"]) == {"en", "de", "es"}, (f.name, p["id"])
            assert set(p["gloss"]) == {"ase", "gsg", "ssp"}, (f.name, p["id"])


def test_sign_to_text_uses_phrase_pack(client):
    with client.websocket_connect("/ws/s1") as ws:
        ws.send_json({"type": "signs", "sign_lang": "gsg", "spoken_lang": "es", "mode": "doctor",
                      "signs": [{"candidates": [["SCHMERZ", 0.9], ["KOPF", 0.05]]},
                                {"candidates": [["WO", 0.8]]}]})
        out = ws.receive_json()
    # DGS signer -> Spanish caption (cross-language).
    assert out["type"] == "caption" and out["text"] == "¿Dónde le duele?" and out["confirm"] is None


def test_low_confidence_asks_to_confirm_and_skips_memory(client):
    with client.websocket_connect("/ws/s2") as ws:
        ws.send_json({"type": "signs", "sign_lang": "ase", "spoken_lang": "en", "mode": "doctor",
                      "signs": [{"candidates": [["FEVER", 0.4], ["SICK", 0.35], ["HOT", 0.1]]}]})
        out = ws.receive_json()
    assert out["confirm"]["uncertain_signs"][0]["options"] == ["FEVER", "SICK", "HOT"]
    assert client.post("/api/sessions/s2/summary", json={"lang": "en"}).json()["turns"] == []


def test_text_to_sign_validates_against_library(client):
    with client.websocket_connect("/ws/s3") as ws:
        ws.send_json({"type": "text", "text": "Where does it hurt?", "spoken_lang": "en",
                      "sign_lang": "ase", "mode": "doctor"})
        out = ws.receive_json()
        assert [i["type"] for i in out["items"]] == ["sign", "sign"]
        ws.send_json({"type": "text", "text": "Anna", "spoken_lang": "en", "sign_lang": "ase"})
        out = ws.receive_json()
    item = out["items"][0]
    assert item["type"] == "spell" and [l["letter"] for l in item["letters"]] == list("ANNA")
    assert item["letters"][0]["frames"] is not None


def test_memory_and_summary(client):
    with client.websocket_connect("/ws/s4") as ws:
        ws.send_json({"type": "signs", "sign_lang": "ase", "spoken_lang": "en", "mode": "doctor",
                      "signs": [{"candidates": [["ME", 0.9]]}, {"candidates": [["HEADACHE", 0.9]]}]})
        ws.receive_json()
        ws.send_json({"type": "text", "text": "Since yesterday?", "spoken_lang": "en", "sign_lang": "ase",
                      "mode": "doctor"})
        ws.receive_json()
    turns = client.post("/api/sessions/s4/summary", json={"lang": "en"}).json()["turns"]
    assert turns == ["Signer (ASL): I have a headache.", "Speaker (en): Since yesterday?"]


def test_bad_messages_keep_socket_open(client):
    with client.websocket_connect("/ws/s5") as ws:
        ws.send_json({"type": "signs", "sign_lang": "xxx", "spoken_lang": "en", "signs": []})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "nope", "sign_lang": "ase", "spoken_lang": "en"})
        assert ws.receive_json()["type"] == "error"
