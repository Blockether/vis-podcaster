"""The pipeline against a fake speech API: retakes, cache, timeline, mastering and gates."""

import array
import json
import math
import re
import shutil
import threading
from collections import Counter
from dataclasses import dataclass

import pytest

from vis_podcaster import Gates, Podcaster, ScriptError, audio
from vis_podcaster.pipeline import GAP_JITTER, QUESTION_GAP, TURN_GAP, _gap
from vis_podcaster.script import load_episode

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
    """Speaks a tone of 0.3 s per word. The first take of `bad` texts is misheard.

    The tone amplitude names the take, so ASR hears the right take while renders overlap lines.
    """

    def __init__(self, bad=()):
        self.bad, self.heard, self.spoken, self.judged = set(bad), [], [], 0

    def voices(self, engine=None):
        return {"vibevoice-local": [V("emma", "Emma"), V("carter", "Carter")]}.get(engine, [])

    def speak(self, text, *, voice=None, engine=None, seed=None, timeout=600):
        self.spoken.append((text, voice, seed))
        is_first = sum(t == text for t, *_ in self.spoken) == 1
        self.heard.append("garbled words" if text in self.bad and is_first else text)
        amp = 3000 + 20 * (len(self.heard) - 1)
        n = int(audio.RATE * 0.3 * len(text.split()))
        tone = array.array("h", (int(amp * math.sin(2 * math.pi * 220 * i / audio.RATE)) for i in range(n)))
        lead = array.array("h", bytes(2 * int(0.2 * audio.RATE)))
        return A(audio.encode_wav(audio.Clip(audio.RATE, lead + tone + lead)))

    def transcribe(self, data, *, engine=None, timeout=600):
        if engine == "vibevoice-asr-local":
            self.judged += 1
            return T(" ".join(line["text"] for line in SCRIPT["lines"]))
        peak = max(map(abs, audio.decode(data).samples), default=0)
        return T(self.heard[round((peak - 3000) / 20)])


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
    episode = load_episode(SCRIPT)
    assert second.start == pytest.approx(first.end + _gap(episode, 1), abs=0.001)
    assert third.start == pytest.approx(second.end + _gap(episode, 2) + 0.5, abs=0.001)
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


class MlxSpeech(FakeSpeech):
    """Adds a `qwen3-tts-mlx` engine with a library of added voices."""

    is_available = True

    def __init__(self, bad=()):
        super().__init__(bad)
        self.languages, self.calls, self.threads, self.speakers = [], [], set(), set()

    def voices(self, engine=None):
        return [V("stuart", "Stuart")] if engine == "qwen3-tts-mlx" else super().voices(engine)

    def speak(self, text, *, language=None, **kwargs):
        if kwargs.get("engine") == "qwen3-tts-mlx":
            self.speakers.add(threading.current_thread())
        self.languages.append(language)
        return super().speak(text, **kwargs)

    def transcribe(self, data, **kwargs):
        # Every Vis call stays on the render thread.
        self.threads.add(threading.current_thread())
        return super().transcribe(data, **kwargs)

    def add_voice(self, recording, name, **kwargs):
        self.calls.append(("add", kwargs["engine"]))
        return V("me", name, kwargs["language"], True)

    def is_added(self, voice_id):
        return voice_id == "me"

    def update_voice(self, voice_id, *, engine=None, **changes):
        self.calls.append(("update", engine))
        return V(voice_id, changes.get("name") or voice_id, "en", True)

    def remove_voice(self, voice_id, *, engine=None):
        self.calls.append(("remove", engine))
        return True


def test_turn_gaps_follow_questions_faster_and_vary():
    lines = [{"speaker": "ab"[i % 2], "text": "Why?" if i % 2 else "Because."} for i in range(20)]
    episode = load_episode({**SCRIPT, "critical_facts": [], "lines": lines})
    after_question = [_gap(episode, i) for i in range(2, 20, 2)]
    after_statement = [_gap(episode, i) for i in range(1, 20, 2)]
    assert all(abs(g - QUESTION_GAP) <= GAP_JITTER for g in after_question)
    assert all(abs(g - TURN_GAP) <= GAP_JITTER for g in after_statement)
    assert len(set(after_statement)) > 1 and [_gap(episode, i) for i in range(20)] == [
        _gap(episode, i) for i in range(20)
    ]


def test_mlx_retakes_use_the_one_speech_thread_of_the_render(tmp_path):
    # A retake on the render thread used to start a second MLX worker, which loaded the model again.
    speech = MlxSpeech(bad={"That is wonderful."})
    script = {**SCRIPT, "hosts": [{**h, "voice": "stuart"} for h in SCRIPT["hosts"]]}
    script["hosts"][1]["voice"] = "stuart"
    r = Podcaster(speech).render(script, tmp_path, gates=Gates(max_wer=1.0, min_wpm=0))
    assert r.takes[2].attempts == 2
    assert len(speech.speakers) == 1 and threading.current_thread() not in speech.speakers


def test_a_take_shorter_than_a_second_is_still_transcribed(tmp_path):
    # Vis refuses audio under one second, and a one-word line crashed the render of a French episode.
    speech = FakeSpeech()
    hear = speech.transcribe

    def transcribe(data, **kwargs):
        if audio.decode(data).seconds < 1.0:
            raise RuntimeError("Voice recording too short - try again")
        return hear(data, **kwargs)

    speech.transcribe = transcribe
    pc = Podcaster(speech)
    _, rec = pc._take(
        load_episode({**SCRIPT, "critical_facts": [], "lines": [{"speaker": "a", "text": "Hi"}]}),
        0,
        "vibevoice-local",
        "emma",
        tmp_path,
        1,
        0.1,
    )
    assert rec["heard"] == "Hi" and rec["wer"] == 0


def test_a_take_that_only_the_first_transcriber_mishears_needs_no_retake(tmp_path):
    # Parakeet heard French lines of the tour-eiffel episode as English; VibeVoice heard them right.
    speech = FakeSpeech(bad={"Bonjour tout le monde"})
    hear = speech.transcribe
    speech.transcribe = lambda data, *, engine=None, **k: (
        T("Bonjour tout le monde") if engine == "vibevoice-asr-local" else hear(data, engine=engine, **k)
    )
    line = {"speaker": "a", "text": "Bonjour tout le monde"}
    episode = load_episode({**SCRIPT, "language": "fr", "critical_facts": [], "lines": [line]})
    _, rec = Podcaster(speech)._take(episode, 0, "vibevoice-local", "emma", tmp_path, 3, 0.1)
    assert rec["attempts"] == 1 and rec["wer"] == 0 and rec["heard"] == "Bonjour tout le monde"
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["judged"] == "Bonjour tout le monde"


def test_mlx_takes_get_the_episode_language_and_added_voices_go_to_its_library(tmp_path):
    speech = MlxSpeech()
    pc = Podcaster(speech)
    script = {**SCRIPT, "language": "fr", "hosts": [{**SCRIPT["hosts"][0], "voice": "stuart"}, SCRIPT["hosts"][1]]}
    r = pc.render(script, tmp_path)
    assert [t.engine for t in r.takes] == ["qwen3-tts-mlx", "vibevoice-local", "qwen3-tts-mlx"]
    # Prefetch threads speak in any order.
    assert Counter(speech.languages) == Counter(["fr", None, "fr"])
    assert speech.threads == {threading.current_thread()}
    assert pc.add_voice(b"", "Me").engine == "qwen3-tts-mlx"
    assert pc.update_voice("me", name="Myself").engine == "qwen3-tts-mlx"
    assert pc.remove_voice("emma-clone")
    assert speech.calls == [("add", "qwen3-tts-mlx"), ("update", "qwen3-tts-mlx"), ("remove", "pocket-tts-local")]


def test_split_cuts_long_audio_in_its_pauses():
    second = audio.RATE
    tone = array.array("h", (int(8000 * math.sin(2 * math.pi * 220 * i / second)) for i in range(second)))
    pcm = array.array("h")
    for k in range(25):
        pcm += tone * 2 + array.array("h", bytes(2 * (second // 2 if k % 5 == 4 else second // 10)))
    clip = audio.Clip(audio.RATE, pcm)
    parts = audio.split(clip, 20.0, search=5.0)
    assert sum(len(p.samples) for p in parts) == len(pcm) and len(parts) == 4
    for p in parts[:-1]:
        assert p.seconds <= 20.0 and max(map(abs, p.samples[-second // 20 :])) == 0
