import sys

import pytest

from vis_podcaster import audio, qwen
from vis_podcaster.qwen import ENGINE, QwenSpeech

STUB = r"""
import json, sys, wave
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if req["voice"] == "broken":
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
        return "text"

    def add_voice(self, *args, **kwargs):
        return "added"


def test_other_engines_go_to_vis():
    base = Base()
    s = QwenSpeech(lambda: base, python="/missing/python")
    assert s.voices("vibevoice-local") == ["vis-voice"]
    assert s.speak("hi", voice="emma", engine="vibevoice-local") == "vis-speech"
    assert s.transcribe(b"", engine="parakeet-local") == "text"
    assert s.add_voice(b"", "me") == "added"
    assert ("speak", "vibevoice-local", "emma") in base.calls


def test_missing_environment_lists_no_voices():
    s = QwenSpeech(Base(), python="/missing/python")
    assert not s.is_available
    with pytest.raises(RuntimeError, match="mlx-audio"):
        s.voices(ENGINE)


def test_worker_serves_takes_and_reports_errors(monkeypatch):
    monkeypatch.setattr(qwen, "WORKER", STUB)
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
