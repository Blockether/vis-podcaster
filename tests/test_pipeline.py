"""The pipeline against a fake speech API: retakes, cache, timeline, mastering and gates."""

import array
import json
import math
import re
import shutil
from dataclasses import dataclass

import pytest

from vis_podcaster import Gates, Podcaster, ScriptError, audio

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

SCRIPT = {
    "title": "Walls",
    "seed": 7,
    "hosts": [{"id": "a", "name": "Ann", "voice": "emma"}, {"id": "b", "name": "Ben", "voice": "carter"}],
    "critical_facts": ["sticky rice"],
    "sources": ["test"],
    "lines": [
        {"speaker": "a", "text": "What held the bricks together?"},
        {"speaker": "b", "text": "Builders mixed sticky rice into the lime mortar."},
        {"speaker": "a", "text": "That is wonderful.", "pause": 0.5},
    ],
}


@dataclass
class V:
    id: str
    label: str
    language: str = "en"
    is_imported: bool = False


@dataclass
class A:
    content: bytes


@dataclass
class T:
    text: str


class FakeSpeech:
    """Speaks a tone of 0.3 s per word. The first take of `bad` texts is misheard."""

    def __init__(self, bad=()):
        self.bad, self.last, self.spoken, self.judged = set(bad), "", [], 0

    def voices(self, engine=None):
        return {"vibevoice-local": [V("emma", "Emma"), V("carter", "Carter")]}.get(engine, [])

    def speak(self, text, *, voice=None, engine=None, seed=None, timeout=600):
        self.spoken.append((text, voice, seed))
        n = int(audio.RATE * 0.3 * len(text.split()))
        tone = array.array("h", (int(8000 * math.sin(2 * math.pi * 220 * i / audio.RATE)) for i in range(n)))
        lead = array.array("h", bytes(2 * int(0.2 * audio.RATE)))
        self.last = "garbled words" if text in self.bad and sum(t == text for t, *_ in self.spoken) == 1 else text
        return A(audio.encode_wav(audio.Clip(audio.RATE, lead + tone + lead)))

    def transcribe(self, data, *, engine=None, timeout=600):
        if engine == "vibevoice-asr-local":
            self.judged += 1
            return T(" ".join(line["text"] for line in SCRIPT["lines"]))
        return T(self.last)


def test_render_retakes_misheard_lines_and_caches_takes(tmp_path):
    speech = FakeSpeech(bad={SCRIPT["lines"][1]["text"]})
    pc = Podcaster(speech)
    steps = []
    r = pc.render(SCRIPT, tmp_path, progress=lambda done, total, msg: steps.append((done, total)))
    assert [t.attempts for t in r.takes] == [1, 2, 1]
    assert r.retakes == 1 and r.unverified == ()
    assert len({seed for *_, seed in speech.spoken}) == 4
    assert steps[:3] == [(1, 3), (2, 3), (3, 3)]
    first, second, third = r.takes
    assert second.start == pytest.approx(first.end + 0.35, abs=0.001)
    assert third.start == pytest.approx(second.end + 0.35 + 0.5, abs=0.001)
    assert "<v Ben>Builders mixed" in (tmp_path / "episode.vtt").read_text()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["voices"] == {"a": "vibevoice-local:emma", "b": "vibevoice-local:carter"}
    assert abs(r.quality.lufs + 16) <= 0.3 and r.quality.true_peak <= -1.0
    assert r.quality.wer == 0.0 and r.quality.facts_heard == 1

    again = pc.render(SCRIPT, tmp_path)
    assert all(t.is_cached for t in again.takes) and len(speech.spoken) == 4
    # Another retake policy reuses the cached takes instead of speaking every line again.
    stricter = pc.render(SCRIPT, tmp_path, max_takes=5, max_wer=0.0)
    assert all(t.is_cached for t in stricter.takes) and len(speech.spoken) == 4
    assert [t.attempts for t in stricter.takes] == [1, 2, 1]


def test_render_keeps_the_best_take_when_no_take_verifies(tmp_path):
    speech = FakeSpeech()
    speech.transcribe = lambda data, engine=None, timeout=600: T("nothing like it")
    r = Podcaster(speech).render(SCRIPT, tmp_path, max_takes=2, gates=Gates(max_wer=1.0, min_wpm=0))
    assert r.unverified == (0, 1, 2) and all(t.attempts == 2 for t in r.takes)
    assert not r.quality.is_passed
    # The summary says why each line failed, not only which line.
    assert re.search(r"lines\[1\] \(WER 100%, \d\.\d words/s\)", str(r))


def test_missing_voices_stop_the_render_with_the_alternatives(tmp_path):
    bad = {**SCRIPT, "hosts": [{"id": "a", "voice": "zed"}, {"id": "b", "voice": "carter"}]}
    pc = Podcaster(FakeSpeech())
    assert not pc.check(bad).is_ready
    with pytest.raises(ScriptError, match="vibevoice-local:emma"):
        pc.render(bad, tmp_path)


def test_a_voice_can_be_named_in_the_engine_voice_form_that_errors_list():
    named = {**SCRIPT, "hosts": [{"id": "a", "voice": "vibevoice-local:emma"}, {"id": "b", "voice": "carter"}]}
    assert Podcaster(FakeSpeech()).check(named).is_ready


def test_evaluate_reports_failed_gates_for_any_audio(tmp_path):
    clip = audio.Clip(audio.RATE, array.array("h", bytes(2 * audio.RATE * 5)))
    wav = tmp_path / "silence.wav"
    wav.write_bytes(audio.encode_wav(clip))
    q = Podcaster(FakeSpeech()).evaluate(wav, "a completely different reference text")
    assert q.wer > 0.5 and not q.is_passed and "PASSED" not in str(q)


def test_master_hits_the_target(tmp_path):
    n = audio.RATE * 4
    loud = array.array("h", (int(30000 * math.sin(2 * math.pi * 300 * i / audio.RATE)) for i in range(n)))
    src = tmp_path / "in.wav"
    src.write_bytes(audio.encode_wav(audio.Clip(audio.RATE, loud)))
    result = audio.master(src, tmp_path / "out.mp3", lufs=-19)
    assert abs(result.lufs + 19) <= 0.2 and result.true_peak <= -1.5
