"""The production pipeline: check, render with ASR verification and retakes, master and evaluate."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from vis_podcaster import audio
from vis_podcaster.qwen import ENGINE as QWEN_ENGINE
from vis_podcaster.script import Episode, Issue, ScriptError, check_script, estimated_minutes, load_episode
from vis_podcaster.text import fact_heard, proper_names, tokens, word_count, word_errors

# The fastest engine comes first. A host without `engine` gets the first engine that has its voice.
ENGINES = (QWEN_ENGINE, "qwen3-tts-local", "vibevoice-local", "pocket-tts-local", "piper-local")
# Vis clones voices with this engine when the MLX environment is missing.
CLONING_ENGINE = "pocket-tts-local"
ASR_ENGINE = "parakeet-local"
# The judge differs from the verifier, so repairs do not overfit one recognizer.
JUDGE_ENGINE = "vibevoice-asr-local"
# The judge needs 0.4 s per second of audio up to two minutes, and 0.57 s for five minutes.
JUDGE_PART_SECONDS = 60.0
# Vis refuses to transcribe audio shorter than one second.
MIN_HEARD_SECONDS = 1.05
TURN_GAP = 0.4
# An answer follows a question faster than a new turn follows a statement.
QUESTION_GAP = 0.25
# Each turn gap varies by up to this many seconds, so the hosts do not alternate like a metronome.
GAP_JITTER = 0.08
SAME_SPEAKER_GAP = 0.25
LEAD_IN = 0.3
TAIL = 0.8
MIN_WORDS_PER_SECOND = 1.2
# Pocket TTS speaks short lines at almost 5 words/s; the WER check catches skipped words.
MAX_WORDS_PER_SECOND = 5.5


class SpeechApi(Protocol):
    """The part of `blockether.vis.extension.speech` that the pipeline uses."""

    def voices(self, engine: str | None = None) -> list[Any]: ...
    def speak(
        self,
        text: str,
        *,
        voice: str | None = None,
        engine: str | None = None,
        seed: int | None = None,
        timeout: float = 600,
    ) -> Any: ...
    def transcribe(self, audio: bytes, *, engine: str | None = None, timeout: float = 600) -> Any: ...


@dataclass(frozen=True)
class Gates:
    """Pass thresholds for an episode. The defaults suit a spoken-word podcast."""

    max_wer: float = 0.05
    min_fact_recall: float = 1.0
    target_lufs: float = -16.0
    lufs_tolerance: float = 1.0
    max_true_peak: float = -1.0
    min_wpm: float = 120.0
    max_wpm: float = 190.0
    max_pause: float = 2.0
    max_clipped: int = 0


@dataclass(frozen=True)
class VoiceChoice:
    """A voice that a host can use."""

    id: str
    label: str
    engine: str
    language: str | None
    is_imported: bool


@dataclass(frozen=True)
class ScriptCheck:
    """The result of `check`: issues, size and the voice that each host gets."""

    is_ready: bool
    issues: tuple[Issue, ...]
    words: int
    minutes: float
    voices: dict[str, str]

    @property
    def errors(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == "error")

    def __str__(self) -> str:
        head = f"{'Ready' if self.is_ready else 'Not ready'}: {self.words} words, about {self.minutes:g} min"
        voices = ", ".join(f"{h} = {v}" for h, v in self.voices.items())
        rows = [f"- {i.severity} {i.where}: {i.message}" for i in self.issues]
        return "\n".join([head, *([f"Voices: {voices}"] if voices else []), *rows])


@dataclass(frozen=True)
class Take:
    """The take that the episode uses for one line, with its verification."""

    line: int
    speaker: str
    text: str
    engine: str
    voice: str
    seed: int
    attempts: int
    wer: float
    heard: str
    start: float
    end: float
    is_verified: bool
    is_cached: bool
    pace: float = 0.0

    @property
    def problem(self) -> str:
        """Why the verifier did not accept the take, or an empty string."""
        if self.is_verified:
            return ""
        return f"WER {self.wer:.0%}, {self.pace:.1f} words/s"


@dataclass(frozen=True)
class Quality:
    """Measured quality of finished audio against its script, and the gates that failed."""

    wer: float
    reference_words: int
    facts_heard: int
    facts_total: int
    missing_facts: tuple[str, ...]
    lufs: float
    true_peak: float
    lra: float
    seconds: float
    words_per_minute: float
    longest_pause: float
    long_pauses: int
    clipped_samples: int
    failures: tuple[str, ...]
    transcript: str = field(repr=False, default="")

    @property
    def is_passed(self) -> bool:
        return not self.failures

    @property
    def fact_recall(self) -> float:
        return self.facts_heard / self.facts_total if self.facts_total else 1.0

    def __str__(self) -> str:
        verdict = "PASSED" if self.is_passed else "FAILED: " + "; ".join(self.failures)
        return (
            f"{verdict}\nWER {self.wer:.1%} of {self.reference_words} words · facts {self.facts_heard}/"
            f"{self.facts_total} · {self.lufs:.1f} LUFS · peak {self.true_peak:.1f} dBTP · LRA {self.lra:.1f}\n"
            f"{self.seconds / 60:.1f} min · {self.words_per_minute:.0f} words/min · longest pause "
            f"{self.longest_pause:.1f} s · clipped {self.clipped_samples}"
            + (f"\nMissing facts: {', '.join(self.missing_facts)}" if self.missing_facts else "")
        )


@dataclass(frozen=True)
class Render:
    """A rendered episode: files, takes and the measured quality."""

    audio: str
    transcript: str
    manifest: str
    takes: tuple[Take, ...]
    retakes: int
    unverified: tuple[int, ...]
    render_seconds: float
    quality: Quality
    attachment_id: str | None = None

    def __str__(self) -> str:
        cached = sum(1 for t in self.takes if t.is_cached)
        return (
            f"{self.audio}\n{len(self.takes)} lines ({cached} cached), {self.retakes} retakes, "
            f"unverified: {', '.join(f'lines[{t.line}] ({t.problem})' for t in self.takes if not t.is_verified) or 'none'}, "
            f"rendered in {self.render_seconds / 60:.1f} min\n{self.quality}"
        )


def _sha(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _seed(episode: Episode, index: int, attempt: int) -> int:
    return (episode.seed + 7919 * index + 104729 * attempt) % (2**31 - 1)


def _gap(episode: Episode, index: int) -> float:
    """Return the silence before line `index`, without its own `pause`."""
    if index == 0:
        return LEAD_IN
    before = episode.lines[index - 1]
    if before.speaker == episode.lines[index].speaker:
        return SAME_SPEAKER_GAP
    base = QUESTION_GAP if before.text.rstrip().endswith("?") else TURN_GAP
    # The same seed gives the same timing, so a render with cached takes keeps its timeline.
    jitter = (_seed(episode, index, 0) % 1001 / 500 - 1) * GAP_JITTER
    return round(base + jitter, 3)


def _vtt_time(t: float) -> str:
    h, rest = divmod(t, 3600)
    m, s = divmod(rest, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


class Podcaster:
    """Produce two-host podcasts with Vis speech engines.

    `speech` is `blockether.vis.extension.speech` inside Vis, or `GatewayClient(...).speech`
    from the SDK. A callable that returns it is also accepted, so the host binds it per call.
    """

    def __init__(
        self,
        speech: SpeechApi | Callable[[], SpeechApi],
        *,
        asr_engine: str = ASR_ENGINE,
        judge_engine: str = JUDGE_ENGINE,
    ):
        self._speech = speech
        self._asr_engine = asr_engine
        self._judge_engine = judge_engine

    @property
    def _api(self) -> SpeechApi:
        s = self._speech
        return s() if callable(s) and not hasattr(s, "speak") else s

    def voices(self, engine: str | None = None, *, language: str | None = "en") -> list[VoiceChoice]:
        """List the voices that hosts can use, by engine. `language` None keeps every language.

        An engine that fails is left out. Raises RuntimeError when every engine fails.
        """
        out, failures = [], []
        names = (engine,) if engine else ENGINES
        for name in names:
            try:
                found = self._api.voices(name)
            except Exception as e:  # noqa: BLE001 - an engine that is not installed has no voices
                failures.append(f"{name}: {e}")
                continue
            for v in found:
                lang = getattr(v, "language", None)
                if language and lang and not lang.lower().startswith(language.lower()):
                    continue
                out.append(VoiceChoice(v.id, v.label, name, lang, bool(getattr(v, "is_imported", False))))
        if len(failures) == len(names):
            raise RuntimeError("no speech engine answered: " + "; ".join(failures))
        return out

    def add_voice(
        self, recording: str | Path | bytes, name: str, *, transcript: str | None = None, language: str = "en"
    ) -> VoiceChoice:
        """Clone a voice from 3 to 30 seconds of one clean speaker.

        The voice goes to the `qwen3-tts-mlx` library when MLX is available, else to `pocket-tts-local`.
        `transcript` is what the clip says and improves the clone. A voice with the same
        name is replaced. Use the returned `id` as a host `voice`.
        """
        engine = QWEN_ENGINE if getattr(self._api, "is_available", False) else CLONING_ENGINE
        v = self._api.add_voice(recording, name, language=language, transcript=transcript, engine=engine)
        return VoiceChoice(v.id, v.label, engine, v.language, True)

    def _owner(self, voice_id: str) -> str:
        """Return the engine that stores the added voice `voice_id`."""
        is_added = getattr(self._api, "is_added", None)
        return QWEN_ENGINE if is_added and is_added(voice_id) else CLONING_ENGINE

    def update_voice(
        self, voice_id: str, *, name: str | None = None, language: str | None = None, transcript: str | None = None
    ) -> VoiceChoice:
        """Change the label, language or clip transcript of a cloned voice. The id stays the same.

        None keeps a value and "" removes it. Raises GatewayError 404 for an unknown voice.
        """
        given = {"name": name, "language": language, "transcript": transcript}
        changes = {k: (v or None) for k, v in given.items() if v is not None}
        engine = self._owner(voice_id)
        v = self._api.update_voice(voice_id, engine=engine, **changes)
        return VoiceChoice(v.id, v.label, engine, v.language, True)

    def remove_voice(self, voice_id: str) -> bool:
        """Delete a cloned voice. Return False when it does not exist."""
        return self._api.remove_voice(voice_id, engine=self._owner(voice_id))

    def _resolve(self, episode: Episode) -> tuple[dict[str, tuple[str, str]], list[Issue]]:
        """Map each host to (engine, voice id), and report the hosts without a voice."""
        catalog = self.voices(language=None)
        chosen, issues = {}, []
        for i, host in enumerate(episode.hosts):
            want, engine = host.voice.lower(), host.engine
            # Accept the `engine:voice` form that the error below lists.
            prefix, _, rest = want.partition(":")
            if rest and engine in (None, prefix) and any(v.engine == prefix for v in catalog):
                want, engine = rest, prefix
            match = next(
                (v for v in catalog if (engine in (None, v.engine)) and want in (v.id.lower(), v.label.lower())),
                None,
            )
            if match is None:
                names = ", ".join(sorted({f"{v.engine}:{v.id}" for v in catalog if host.engine in (None, v.engine)}))
                issues.append(
                    Issue("error", f"hosts[{i}]", f"voice {host.voice!r} is not available; use one of {names}")
                )
            else:
                chosen[host.id] = (match.engine, match.id)
        return chosen, issues

    def check(self, script: str | Path | dict[str, Any] | Episode) -> ScriptCheck:
        """Validate a script and its voices without speaking. Read every issue before you render."""
        if isinstance(script, str | Path) and not str(script).lstrip().startswith("{"):
            script = json.loads(Path(script).expanduser().read_text(encoding="utf-8"))
        elif isinstance(script, str):
            script = json.loads(script)
        episode, issues = check_script(script)
        if episode is None:
            return ScriptCheck(False, issues, 0, 0.0, {})
        chosen, voice_issues = self._resolve(episode)
        all_issues = (*issues, *voice_issues)
        return ScriptCheck(
            not any(i.severity == "error" for i in all_issues),
            all_issues,
            word_count(episode.spoken(episode.text)),
            round(estimated_minutes(episode), 2),
            {h: f"{e}:{v}" for h, (e, v) in chosen.items()},
        )

    def _speak(self, episode: Episode, engine: str, voice: str, spoken: str, seed: int) -> bytes:
        # Only the MLX engine takes a language; the Vis engines know the language of each voice.
        extra = {"language": episode.language} if engine == QWEN_ENGINE else {}
        return self._api.speak(spoken, voice=voice, engine=engine, seed=seed, **extra).content

    def _take(
        self,
        episode: Episode,
        index: int,
        engine: str,
        voice: str,
        cache: Path,
        max_takes: int,
        max_wer: float,
        first: Future | None = None,
        speaker: ThreadPoolExecutor | None = None,
    ) -> tuple[audio.Clip, dict]:
        """Speak one line until ASR confirms it, and return the best clip with its record.

        Each take is cached by its engine, voice, text, seed and verifier, not by the retake policy.
        A render with another `max_takes` or `max_wer` therefore reuses every take that it already has.
        `first` is the audio of the first take when a background thread already speaks it.
        `speaker` is that thread: retakes run on it too, so one render keeps one model worker.
        """
        line = episode.lines[index]
        spoken = episode.spoken(line.text)
        best = None
        is_cached = True
        words = word_count(spoken)
        names = proper_names(line.text) | proper_names(spoken)
        for attempt in range(max_takes):
            seed = _seed(episode, index, attempt)
            key = _sha(engine, voice, spoken, seed, self._asr_engine)
            wav, meta = cache / f"{key}.wav", cache / f"{key}.json"
            if wav.exists() and meta.exists():
                clip, saved = audio.decode(wav), json.loads(meta.read_text())
            else:
                is_cached = False
                is_ahead = attempt == 0 and first is not None
                if is_ahead:
                    content = first.result()
                elif speaker is not None and engine == QWEN_ENGINE:
                    # The render's one speech thread: a retake never starts a second model worker.
                    content = speaker.submit(self._speak, episode, engine, voice, spoken, seed).result()
                else:
                    content = self._speak(episode, engine, voice, spoken, seed)
                clip = audio.trim(audio.decode(content))
                saved = {"seed": seed, "heard": self._hear(clip, self._asr_engine)}
                wav.write_bytes(audio.encode_wav(clip))
                meta.write_text(json.dumps(saved))
            heard = saved["heard"]
            wer = min(word_errors(spoken, heard, names=names).wer, word_errors(line.text, heard, names=names).wer)
            if wer > max_wer and self._judge_engine not in (None, self._asr_engine):
                # Parakeet can hear a French line as English. A take fails only when the judge mishears it too.
                if "judged" not in saved:
                    saved["judged"] = self._hear(clip, self._judge_engine)
                    meta.write_text(json.dumps(saved))
                judged = min(
                    word_errors(spoken, saved["judged"], names=names).wer,
                    word_errors(line.text, saved["judged"], names=names).wer,
                )
                if judged < wer:
                    wer, heard = judged, saved["judged"]
            rate = words / clip.seconds if clip.seconds else 0.0
            ok = wer <= max_wer and MIN_WORDS_PER_SECOND <= rate <= MAX_WORDS_PER_SECOND
            score = (not ok, wer)
            if best is None or score < best[0]:
                record = {"seed": seed, "wer": round(wer, 4), "pace": round(rate, 2), "heard": heard, "is_verified": ok}
                best = (score, clip, record)
            if ok:
                break
        _, clip, record = best
        return clip, {**record, "attempts": attempt + 1, "is_cached": is_cached}

    def _hear(self, clip: audio.Clip, engine: str) -> str:
        """Return what `engine` hears in one take. A take under a second gets silence after it, as Vis needs."""
        padded = audio.pad(clip, MIN_HEARD_SECONDS)
        return self._api.transcribe(audio.encode_wav(padded), engine=engine).text

    def render(
        self,
        script: str | Path | dict[str, Any] | Episode,
        out_dir: str | Path,
        *,
        max_takes: int = 3,
        max_wer: float = 0.08,
        gates: Gates | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> Render:
        """Speak every line, verify it with ASR, retake failed lines, then master and evaluate.

        Each line gets up to `max_takes` seeds until its WER is at most `max_wer` and its pace is
        plausible; else the best take stays and the line is listed in `unverified`. Takes are cached
        in `out_dir/takes`, so a second render speaks only changed lines. Writes `episode.mp3`,
        `episode.vtt` with speaker tags and `manifest.json`. `progress(done, total, message)` hears each
        line. Raises ScriptError for an invalid script or a missing voice.
        """
        started = time.monotonic()
        episode = load_episode(script)
        chosen, voice_issues = self._resolve(episode)
        if voice_issues:
            raise ScriptError(tuple(voice_issues))
        out = Path(out_dir).expanduser()
        cache = out / "takes"
        cache.mkdir(parents=True, exist_ok=True)
        parts, takes, clock = [], [], LEAD_IN

        def ahead(i: int) -> Future | None:
            """Speak the first take of line `i` in the background when the MLX worker speaks it.

            ASR then checks one line while the GPU speaks the next. Vis calls stay on this thread.
            """
            if i >= len(episode.lines):
                return None
            engine, voice = chosen[episode.lines[i].speaker]
            spoken, seed = episode.spoken(episode.lines[i].text), _seed(episode, i, 0)
            if engine != QWEN_ENGINE or (cache / f"{_sha(engine, voice, spoken, seed, self._asr_engine)}.wav").exists():
                return None
            return pool.submit(self._speak, episode, engine, voice, spoken, seed)

        results = []
        with ThreadPoolExecutor(1, thread_name_prefix="podcast-speech") as pool:
            following = ahead(0)
            try:
                for i, line in enumerate(episode.lines):
                    current, following = following, ahead(i + 1)
                    clip, rec = self._take(
                        episode, i, *chosen[line.speaker], cache, max_takes, max_wer, first=current, speaker=pool
                    )
                    results.append((clip, rec))
                    if progress:
                        state = "cached" if rec["is_cached"] else f"{rec['attempts']} take(s), WER {rec['wer']:.0%}"
                        progress(
                            i + 1, len(episode.lines), f"Line {i + 1}/{len(episode.lines)} ({line.speaker}): {state}"
                        )
            except BaseException:
                pool.shutdown(cancel_futures=True)
                raise
        for i, line in enumerate(episode.lines):
            engine, voice = chosen[line.speaker]
            clip, rec = results[i]
            gap = _gap(episode, i) + line.pause
            clock += 0 if i == 0 else gap
            takes.append(
                Take(
                    i,
                    line.speaker,
                    line.text,
                    engine,
                    voice,
                    rec["seed"],
                    rec["attempts"],
                    rec["wer"],
                    rec["heard"],
                    round(clock, 3),
                    round(clock + clip.seconds, 3),
                    rec["is_verified"],
                    rec["is_cached"],
                    rec["pace"],
                )
            )
            parts.append((gap, clip))
            clock += clip.seconds
        joined = audio.join(parts + [(TAIL, audio.Clip(audio.RATE, audio.array.array("h")))])
        raw = out / "episode-raw.wav"
        raw.write_bytes(audio.encode_wav(joined))
        final = out / "episode.mp3"
        g = gates or Gates()
        if progress:
            progress(len(episode.lines), len(episode.lines), "Mastering and judging the episode")
        with ThreadPoolExecutor(1, thread_name_prefix="podcast-master") as pool:
            # FFmpeg masters while the judge hears the raw mix: mastering changes only the loudness.
            mastered = pool.submit(audio.master, raw, final, lufs=g.target_lufs)
            heard = self._judge(audio.decode(raw, rate=16000))
            mastered.result()
        vtt = out / "episode.vtt"
        names = {h.id: h.name for h in episode.hosts}
        vtt.write_text(
            "WEBVTT\n\n"
            + "\n".join(
                f"{t.line + 1}\n{_vtt_time(t.start)} --> {_vtt_time(t.end)}\n<v {names[t.speaker]}>{t.text}\n"
                for t in takes
            ),
            encoding="utf-8",
        )
        quality = self._evaluate(final, episode, g, heard=heard)
        elapsed = time.monotonic() - started
        manifest = out / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "title": episode.title,
                    "script": episode.to_dict(),
                    "script_sha": _sha(episode.to_dict()),
                    "voices": {h: f"{e}:{v}" for h, (e, v) in chosen.items()},
                    "asr_engine": self._asr_engine,
                    "judge_engine": self._judge_engine,
                    "max_takes": max_takes,
                    "max_wer": max_wer,
                    "render_seconds": round(elapsed, 1),
                    "takes": [asdict(t) for t in takes],
                    "quality": {**asdict(quality), "is_passed": quality.is_passed},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return Render(
            str(final),
            str(vtt),
            str(manifest),
            tuple(takes),
            sum(t.attempts - 1 for t in takes if not t.is_cached),
            tuple(t.line for t in takes if not t.is_verified),
            round(elapsed, 1),
            quality,
        )

    def evaluate(
        self,
        audio_file: str | Path,
        script: str | Path | dict[str, Any] | Episode,
        *,
        gates: Gates | None = None,
        timeout: float = 3600,
    ) -> Quality:
        """Measure finished audio from any producer against its script, and list the failed gates.

        `script` is an episode or the plain reference text. The judge engine transcribes the file in
        parts of at most one minute, cut in pauses: its cost grows faster than the length. WER takes
        the better of the written and the spoken form. Facts allow ASR spelling variants of names.
        """
        return self._evaluate(audio_file, script, gates or Gates(), timeout=timeout)

    def _judge(self, clip: audio.Clip, timeout: float = 3600) -> str:
        """Return what the judge engine hears in a clip, part by part."""
        return " ".join(
            self._api.transcribe(audio.encode_wav(part), engine=self._judge_engine, timeout=timeout).text.strip()
            for part in audio.split(clip, JUDGE_PART_SECONDS)
        )

    def _evaluate(
        self,
        audio_file: str | Path,
        script: str | Path | dict[str, Any] | Episode,
        g: Gates,
        *,
        timeout: float = 3600,
        heard: str | None = None,
    ) -> Quality:
        """Measure `audio_file` like `evaluate`. `heard` is the judge transcript when the caller has it."""
        if isinstance(script, str) and not script.lstrip().startswith("{") and not Path(script).expanduser().exists():
            reference, spoken, facts = script, script, ()
        else:
            episode = load_episode(script)
            reference, spoken = episode.text, episode.spoken(episode.text)
            facts = tuple((f, episode.spoken(f)) for f in episode.critical_facts)
        clip = audio.decode(audio_file, rate=16000)
        if heard is None:
            heard = self._judge(clip, timeout)
        errors = min(word_errors(reference, heard), word_errors(spoken, heard), key=lambda e: e.wer)
        heard_tokens = tokens(heard)
        missing = tuple(
            f
            for f, said in facts
            if not (fact_heard(f, heard_tokens, fuzzy=True) or fact_heard(said, heard_tokens, fuzzy=True))
        )
        loud = audio.measure(audio_file)
        pauses = audio.silences(clip, min_seconds=1.0)
        seconds = clip.seconds
        wpm = word_count(reference) / (seconds / 60) if seconds else 0.0
        longest = max((p[1] for p in pauses), default=0.0)
        clipped = audio.clipped_samples(audio.decode(audio_file))
        # The report and the failure message show the same rounded value.
        wer = round(errors.wer, 4)
        checks = [
            (wer <= g.max_wer, f"WER {wer:.1%} is above {g.max_wer:.0%}"),
            (
                (len(facts) - len(missing)) / len(facts) >= g.min_fact_recall if facts else True,
                f"{len(missing)} critical fact(s) not heard",
            ),
            (
                abs(loud.lufs - g.target_lufs) <= g.lufs_tolerance,
                f"loudness {loud.lufs:.1f} LUFS is outside {g.target_lufs:g}±{g.lufs_tolerance:g}",
            ),
            (loud.true_peak <= g.max_true_peak, f"true peak {loud.true_peak:.1f} dBTP is above {g.max_true_peak:g}"),
            (g.min_wpm <= wpm <= g.max_wpm, f"pace {wpm:.0f} words/min is outside {g.min_wpm:g}-{g.max_wpm:g}"),
            (longest <= g.max_pause, f"a pause of {longest:.1f} s is longer than {g.max_pause:g} s"),
            (clipped <= g.max_clipped, f"{clipped} clipped samples"),
        ]
        return Quality(
            wer,
            errors.reference_words,
            len(facts) - len(missing),
            len(facts),
            missing,
            loud.lufs,
            loud.true_peak,
            loud.lra,
            round(seconds, 2),
            round(wpm, 1),
            round(longest, 2),
            sum(1 for p in pauses if p[1] > 1.5),
            clipped,
            tuple(msg for ok, msg in checks if not ok),
            heard,
        )
