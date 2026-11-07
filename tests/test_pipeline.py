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
from vis_podcaster.pipeline import (
    GAP_JITTER,
    JUDGE_MIN_PART_SECONDS,
    JUDGE_PART_SECONDS,
    QUESTION_GAP,
    TURN_GAP,
    _gap,
    _legacy_seed,
    _seed,
    _sha,
)
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
        self.judge_seconds, self.misheard = [], {}

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
        clip = audio.decode(data)
        if engine == "vibevoice-asr-local":
            # The judge hears each take in the audio, by its tone amplitude, like ASR; `misheard` changes the mix.
            self.judged += 1
            self.judge_seconds.append(clip.seconds)
            text = " ".join(self.heard[k] for k in takes_in(clip))
            for said, heard in self.misheard.items():
                text = text.replace(said, heard)
            return T(text)
        peak = max(map(abs, clip.samples), default=0)
        return T(self.heard[round((peak - 3000) / 20)])


def takes_in(clip):
    """Return the take numbers in a mix in order. Each take is one tone with its own amplitude."""
    n = int(clip.rate * 0.02)
    peaks = [max(map(abs, clip.samples[i : i + n]), default=0) for i in range(0, len(clip.samples), n)]
    found, top = [], 0
    for peak in [*peaks, 0]:
        if peak > 1000:
            top = max(top, peak)
        elif top:
            found.append(round((top - 3000) / 20))
            top = 0
    return found


def long_script(n=24):
    """Return a script of `n` different lines of eight words, about 2.4 s each."""
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    words += ["golf", "hotel", "india", "juliet", "kilo", "lima"]
    lines = [
        {"speaker": "ab"[i % 2], "text": f"Line {i} has " + " ".join(words[(i + k) % 12] for k in range(6)) + "."}
        for i in range(n)
    ]
    return {**SCRIPT, "critical_facts": [], "lines": lines}


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


def test_the_judge_hears_whole_lines_and_reuses_unchanged_parts(tmp_path):
    speech = FakeSpeech()
    pc = Podcaster(speech)
    script = long_script()
    first = pc.render(script, tmp_path)
    parts = first.quality.parts
    assert len(parts) >= 2 and speech.judged == len(parts)
    # The parts cover every line once, end between lines and stay under the part limit.
    assert [p.first_line for p in parts] == [0] + [p.last_line + 1 for p in parts[:-1]]
    assert parts[-1].last_line == len(script["lines"]) - 1
    assert all(p.end - p.start <= JUDGE_PART_SECONDS for p in parts)
    assert first.quality.wer == 0 and all(p.errors == 0 for p in parts)
    edited = [dict(line) for line in script["lines"]]
    edited[20]["text"] = "Line twenty is new and quite different now."
    again = pc.render({**script, "lines": edited}, tmp_path)
    fresh = [p for p in again.quality.parts if not p.is_cached]
    assert speech.judged == len(parts) + len(fresh) and 1 <= len(fresh) <= 2
    assert all(p.is_cached for p in again.quality.parts if p.last_line < 20 and p.first_line <= 20 - 10)
    assert again.changed == (20,) and again.review and (tmp_path / "changes.mp3").exists()
    # A review holds only the changed line and one second before it.
    assert audio.decode(again.review).seconds < 6


def test_an_inserted_line_keeps_the_takes_and_gaps_of_the_other_lines(tmp_path):
    speech = FakeSpeech()
    pc = Podcaster(speech)
    script = long_script(8)
    pc.render(script, tmp_path)
    before = load_episode(script)
    spoken = len(speech.spoken)
    lines = [*script["lines"][:3], {"speaker": "b", "text": "An added line in the middle."}, *script["lines"][3:]]
    r = pc.render({**script, "lines": lines}, tmp_path)
    after = load_episode({**script, "lines": lines})
    assert len(speech.spoken) == spoken + 1 and r.changed == (3,)
    assert [_gap(after, i) for i in (1, 2, 6, 7, 8)] == [_gap(before, i) for i in (1, 2, 5, 6, 7)]


def test_takes_cached_by_line_position_are_still_used(tmp_path):
    speech = FakeSpeech(bad={SCRIPT["lines"][1]["text"]})
    pc = Podcaster(speech)
    pc.render(SCRIPT, tmp_path)
    episode, takes = load_episode(SCRIPT), tmp_path / "takes"
    # Version 0.1 keyed each take by a seed from the line position: move every take to that key.
    for i, line in enumerate(episode.lines):
        voice = {"a": "emma", "b": "carter"}[line.speaker]
        for attempt in range(2):
            new = _sha("vibevoice-local", voice, line.text, _seed(episode, i, attempt), "parakeet-local")
            old = _sha("vibevoice-local", voice, line.text, _legacy_seed(episode, i, attempt), "parakeet-local")
            for suffix in (".wav", ".json"):
                if (takes / f"{new}{suffix}").exists():
                    (takes / f"{new}{suffix}").rename(takes / f"{old}{suffix}")
    spoken = len(speech.spoken)
    r = pc.render(SCRIPT, tmp_path, force=True)
    assert len(speech.spoken) == spoken and all(t.is_cached for t in r.takes)
    assert [t.attempts for t in r.takes] == [1, 2, 1]


def test_a_fact_that_only_the_take_check_heard_counts_and_a_lost_fact_says_where(tmp_path):
    speech = FakeSpeech()
    speech.misheard = {"sticky rice": "sticky ice"}
    r = Podcaster(speech).render(SCRIPT, tmp_path)
    (fact,) = r.quality.facts
    assert r.quality.facts_heard == 1 and fact.heard_by == ("take:parakeet-local",) and fact.line == 1
    assert "Heard only by a second engine" in str(r.quality)
    speech.heard = ["sticky ice" if "sticky" in h else h for h in speech.heard]
    # The raw mix keeps the tone amplitudes that name the takes; mastering changes them.
    lost = Podcaster(speech).evaluate(tmp_path / "episode-raw.wav", SCRIPT)
    assert lost.missing_facts == ("sticky rice",) and not lost.is_passed
    assert '- "sticky rice" (lines[1])' in str(lost)


def test_a_fact_next_to_another_number_is_heard(tmp_path):
    # The year and the count are two numbers: the comma and the digits show where the year ends.
    line = {"speaker": "b", "text": "In 1990, twenty people died, and in 2008, 40 percent of farms failed."}
    script = {**SCRIPT, "critical_facts": ["twenty people", "40 percent of farms"], "lines": [SCRIPT["lines"][0], line]}
    r = Podcaster(FakeSpeech()).render(script, tmp_path)
    assert r.quality.facts_heard == 2 and all(f.heard_by[0].startswith("mix:") for f in r.quality.facts)


def test_an_unchanged_render_returns_at_once_and_force_renders_again(tmp_path):
    speech = FakeSpeech()
    pc = Podcaster(speech)
    first = pc.render(SCRIPT, tmp_path)
    judged, spoken = speech.judged, len(speech.spoken)
    again = pc.render(SCRIPT, tmp_path)
    assert again.is_skipped and (speech.judged, len(speech.spoken)) == (judged, spoken)
    assert again.quality == first.quality and again.audio == first.audio
    forced = pc.render(SCRIPT, tmp_path, force=True)
    assert not forced.is_skipped and forced.changed == () and not (tmp_path / "changes.mp3").exists()
    # A forced render speaks nothing again, but the judge hears every part again.
    assert len(speech.spoken) == spoken and speech.judged == judged + len(forced.quality.parts)
    assert not any(p.is_cached for p in forced.quality.parts)


def test_a_wrong_or_empty_judge_result_is_not_kept(tmp_path):
    speech = FakeSpeech()
    real = speech.transcribe
    # An empty transcript, as from a failed judge call, is not cached.
    speech.transcribe = lambda data, *, engine=None, timeout=600: (
        T("") if engine == "vibevoice-asr-local" else real(data, engine=engine, timeout=timeout)
    )
    pc = Podcaster(speech)
    assert pc.render(SCRIPT, tmp_path).quality.wer == 1
    del speech.transcribe
    speech.misheard = {"wonderful": "wander full"}
    # The next render judges again, without `force`.
    again = pc.render(SCRIPT, tmp_path)
    assert not again.is_skipped and again.quality.wer > 0 and not any(p.is_cached for p in again.quality.parts)
    # A wrong transcript stays in the cache until a forced render judges again.
    speech.misheard = {}
    assert pc.render(SCRIPT, tmp_path, force=True).quality.wer == 0


def test_a_render_without_its_takes_speaks_them_again(tmp_path):
    speech = FakeSpeech()
    pc = Podcaster(speech)
    pc.render(SCRIPT, tmp_path)
    spoken = len(speech.spoken)
    shutil.rmtree(tmp_path / "takes")
    r = pc.render(SCRIPT, tmp_path)
    assert not r.is_skipped and len(speech.spoken) == spoken + len(SCRIPT["lines"])


def test_a_new_pronunciation_is_a_change_and_a_skipped_render_still_tells_it(tmp_path):
    speech = FakeSpeech()
    pc = Podcaster(speech)
    pc.render(SCRIPT, tmp_path)
    script = {**SCRIPT, "pronunciations": {"mortar": "more tar"}}
    r = pc.render(script, tmp_path)
    assert r.changed == (1,) and r.review and (tmp_path / "changes.mp3").exists()
    again = pc.render(script, tmp_path)
    assert again.is_skipped and again.changed == (1,) and again.review == r.review
    assert "Changed in the last render" in str(again)


def test_a_pause_before_the_first_line_moves_its_timeline_too(tmp_path):
    lines = [{**SCRIPT["lines"][0], "pause": 1.5}, *SCRIPT["lines"][1:]]
    r = Podcaster(FakeSpeech()).render({**SCRIPT, "lines": lines}, tmp_path)
    mix = audio.decode(tmp_path / "episode-raw.wav")
    voiced = next(i for i, x in enumerate(mix.samples) if abs(x) > 1000) / mix.rate
    # Each take starts with 60 ms of the room that trimming keeps.
    assert abs(voiced - r.takes[0].start) < 0.1
    assert "00:00:01.8" in (tmp_path / "episode.vtt").read_text()


def test_an_edit_judges_again_only_the_parts_near_it(tmp_path):
    speech = FakeSpeech()
    pc = Podcaster(speech)
    script = long_script(36)
    first = pc.render(script, tmp_path)
    assert len(first.quality.parts) >= 3
    edited = [dict(line) for line in script["lines"]]
    # The same count of words gives a take of the same length. Cuts that counted from the start of
    # each part moved in all later parts after this edit.
    edited[15]["text"] = "Line 15 has kilo lima alpha bravo charlie delta."
    judged = speech.judged
    again = pc.render({**script, "lines": edited}, tmp_path)
    fresh = [p for p in again.quality.parts if not p.is_cached]
    assert speech.judged - judged == len(fresh) <= 2 and any(p.first_line <= 15 <= p.last_line for p in fresh)
    near = (again.takes[15].start - JUDGE_MIN_PART_SECONDS - 1, again.takes[15].end + JUDGE_MIN_PART_SECONDS + 1)
    assert all(p.is_cached for p in again.quality.parts if p.end < near[0] or p.start > near[1])
