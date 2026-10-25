import array
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from vis_podcaster import audio, qwen
from vis_podcaster.qwen import CLONE_MODEL, ENGINE, MODEL, PRESETS, QwenSpeech

STUB = r"""
import json, os, sys, time, wave
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if req.get("voice") == "slow":
        time.sleep(1)
    with open(os.environ["STUB_LOG"], "a") as log:
        log.write(line)
    if req.get("voice") == "broken":
        print(json.dumps({"error": "ValueError: bad voice"}), flush=True)
        continue
    with wave.open(req["out"], "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(b"\x00\x10" * 2400)
    print(json.dumps({"ok": True, "seed": req["seed"], "language": req["language"]}), flush=True)
"""


class Base:
    def __init__(self):
        self.calls = []

    def voices(self, engine=None):
        self.calls.append(("voices", engine))
        return ["vis-voice"]

    def speak(self, text, *, voice=None, engine=None, seed=None, timeout=600):
        self.calls.append(("speak", engine, voice))
        return "vis-speech"

    def transcribe(self, data, *, engine=None, timeout=600):
        self.calls.append(("transcribe", engine))
        return SimpleNamespace(text="heard words")

    def add_voice(self, *args, **kwargs):
        return "added"


def test_other_engines_go_to_vis():
    base = Base()
    s = QwenSpeech(lambda: base, python="/missing/python")
    assert s.voices("vibevoice-local") == ["vis-voice"]
    assert s.speak("hi", voice="emma", engine="vibevoice-local") == "vis-speech"
    assert s.transcribe(b"", engine="parakeet-local").text == "heard words"
    assert s.add_voice(b"", "me") == "added"
    assert ("speak", "vibevoice-local", "emma") in base.calls


def test_missing_environment_lists_no_voices():
    s = QwenSpeech(Base(), python="/missing/python")
    assert not s.is_available
    with pytest.raises(RuntimeError, match="mlx-audio"):
        s.voices(ENGINE)


def test_worker_serves_takes_and_reports_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(qwen, "WORKER", STUB)
    monkeypatch.setenv("STUB_LOG", str(tmp_path / "requests.jsonl"))
    s = QwenSpeech(Base(), python=sys.executable)
    try:
        assert {v.id for v in s.voices(ENGINE)} >= {"serena", "aiden"}
        first = s.speak("Hello there.", voice="serena", engine=ENGINE, seed=7)
        second = s.speak("Again.", voice="aiden", engine=ENGINE)
        assert first.engine == ENGINE and audio.decode(first.content).seconds == pytest.approx(0.1)
        assert second.voice == "aiden"
        with pytest.raises(RuntimeError, match="bad voice"):
            s.speak("x", voice="broken", engine=ENGINE)
    finally:
        s.close()


def test_takes_at_the_same_time_never_wait_for_a_busy_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(qwen, "WORKER", STUB)
    monkeypatch.setenv("STUB_LOG", str(tmp_path / "requests.jsonl"))
    s = QwenSpeech(Base(), python=sys.executable)
    try:
        s.speak("Warm up.", voice="serena", engine=ENGINE)
        start = time.monotonic()
        with ThreadPoolExecutor(3) as pool:
            takes = list(pool.map(lambda t: s.speak(t, voice="slow", engine=ENGINE), ["One.", "Two.", "Three."]))
        assert time.monotonic() - start < 2.5
        assert all(t.engine == ENGINE for t in takes) and len(s._workers) == 3
        s.speak("Again.", voice="serena", engine=ENGINE)
        assert len(s._workers) == 3
    finally:
        s.close()
    assert not s._workers and not s._idle


def tone(seconds):
    n = int(audio.RATE * seconds)
    pcm = array.array("h", (int(6000 * math.sin(2 * math.pi * 200 * i / audio.RATE)) for i in range(n)))
    return audio.encode_wav(audio.Clip(audio.RATE, pcm))


def test_bundled_voices_have_a_clip_a_transcript_and_a_license():
    entries = json.loads((PRESETS / "voices.json").read_text(encoding="utf-8"))
    assert len(entries) >= 9 and len({e["id"] for e in entries}) == len(entries)
    for e in entries:
        clip = audio.decode(PRESETS / f"{e['id']}.wav")
        assert 3 <= clip.seconds <= 30, e["id"]
        assert e["transcript"] and e["language"] and e["license"] in ("CC0-1.0", "CC-BY-4.0") and e["source"]


def test_cloned_voices_use_the_base_model_with_their_clip(monkeypatch, tmp_path):
    monkeypatch.setattr(qwen, "WORKER", STUB)
    log = tmp_path / "requests.jsonl"
    monkeypatch.setenv("STUB_LOG", str(log))
    s = QwenSpeech(Base(), python=sys.executable, library=tmp_path / "library")
    try:
        assert {"stuart", "andrea", "serena"} <= {v.id for v in s.voices(ENGINE)}
        s.speak("Bonjour.", voice="estelle", engine=ENGINE, language="fr")
        s.speak("Hola.", voice="serena", engine=ENGINE, language="es")
        s.speak("Cześć.", voice="peter", engine=ENGINE, language="pl")
    finally:
        s.close()
    clone, builtin, polish = (json.loads(line) for line in log.read_text().splitlines())
    assert clone["model"] == CLONE_MODEL and clone["ref_audio"].endswith("estelle.wav") and clone["ref_text"]
    assert clone["language"] == "french" and "voice" not in clone
    assert builtin["model"] == MODEL and builtin["voice"] == "serena" and builtin["language"] == "spanish"
    assert polish["language"] == "auto"


def test_added_voices_live_in_the_library(tmp_path):
    s = QwenSpeech(Base(), python=sys.executable, library=tmp_path)
    v = s.add_voice(tone(4), "My Voice", language="en", engine=ENGINE)
    assert (v.id, v.is_imported) == ("my-voice", True) and s.is_added("my-voice")
    assert (tmp_path / "my-voice.wav").exists()
    assert json.loads((tmp_path / "voices.json").read_text())[0]["transcript"] == "heard words"
    assert s.update_voice("my-voice", engine=ENGINE, name="Me", transcript="hello").label == "Me"
    assert next(x for x in s.voices(ENGINE) if x.id == "my-voice").label == "Me"
    with pytest.raises(ValueError, match="3 to 30"):
        s.add_voice(tone(1), "short", engine=ENGINE)
    with pytest.raises(ValueError, match="bundled"):
        s.add_voice(tone(4), "Stuart", engine=ENGINE)
    assert s.remove_voice("my-voice", engine=ENGINE)
    assert not s.remove_voice("my-voice", engine=ENGINE) and not (tmp_path / "my-voice.wav").exists()
