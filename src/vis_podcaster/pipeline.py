"""The production pipeline: check, render with ASR verification and retakes, master and evaluate."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from vis_podcaster import audio
from vis_podcaster.qwen import ENGINE as QWEN_ENGINE
from vis_podcaster.script import Episode, Issue, ScriptError, check_script, estimated_minutes, load_episode
from vis_podcaster.text import WordErrors, fact_heard, proper_names, word_count, word_errors

# The fastest engine comes first. A host without `engine` gets the first engine that has its voice.
ENGINES = (QWEN_ENGINE, "qwen3-tts-local", "vibevoice-local", "pocket-tts-local", "piper-local")
# Vis clones voices with this engine when the MLX environment is missing.
CLONING_ENGINE = "pocket-tts-local"
ASR_ENGINE = "parakeet-local"
# The judge differs from the verifier, so repairs do not overfit one recognizer.
JUDGE_ENGINE = "vibevoice-asr-local"
# The judge needs 0.4 s per second of audio up to two minutes, and 0.57 s for five minutes.
JUDGE_PART_SECONDS = 60.0
# A judge part holds whole lines. It ends after a line whose text hash is higher than that of each
# line that ends this many seconds or less from it. The cuts move with the content, so an edit changes
# only the parts near it, and most parts last 20 to 60 seconds.
JUDGE_MIN_PART_SECONDS = 20.0
# Silence before and after each judge part.
JUDGE_PAD = 0.2
# Each changed line in `changes.mp3` starts this many seconds early and ends this many seconds late.
REVIEW_LEAD = 1.0
REVIEW_TAIL = 0.3
REVIEW_GAP = 1.0
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
    # The engine that wrote `heard`: the ASR engine, or the judge when it heard the take better.
    heard_by: str = ""
    # The cache key: the take is `takes/<key>.wav`, and a new key means new audio.
    key: str = ""

    @property
    def problem(self) -> str:
        """Why the verifier did not accept the take, or an empty string."""
        if self.is_verified:
            return ""
        return f"WER {self.wer:.0%}, {self.pace:.1f} words/s"


@dataclass(frozen=True)
class JudgePart:
    """One part of the mix that the judge engine heard. A part holds whole lines."""

    first_line: int
    last_line: int
    start: float
    end: float
    heard: str
    reference_words: int = 0
    errors: int = 0
    wer: float = 0.0
    is_cached: bool = False


@dataclass(frozen=True)
class FactCheck:
    """One critical fact: the engines that heard it and where the script says it.

    `heard_by` names each source as `mix:<judge>` (the full mix), `take:<engine>` (the take check
    of the line) or `file:<engine>` (a second engine on the whole file).
    """

    fact: str
    heard_by: tuple[str, ...]
    line: int | None = None
    start: float | None = None
    end: float | None = None
    judge_heard: str = ""
    take_heard: str = ""

    @property
    def is_heard(self) -> bool:
        return bool(self.heard_by)

    def __str__(self) -> str:
        where = f"lines[{self.line}]" if self.line is not None else "no line says it"
        if self.start is not None and self.end is not None:
            where += f", {_clock(self.start)}-{_clock(self.end)}"
        if self.is_heard:
            return f'"{self.fact}" ({where}): heard by {", ".join(self.heard_by)}'
        judge = f'; the judge heard "{self.judge_heard}"' if self.judge_heard else ""
        take = f'; the take check heard "{self.take_heard}"' if self.take_heard else ""
        return f'"{self.fact}" ({where}){judge}{take}'


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
    facts: tuple[FactCheck, ...] = ()
    parts: tuple[JudgePart, ...] = field(repr=False, default=())

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
            + "".join(f"\n- {f}" for f in self.facts if not f.is_heard)
            + "".join(
                f"\nHeard only by a second engine: {f}"
                for f in self.facts
                if f.is_heard and not any(s.startswith("mix:") for s in f.heard_by)
            )
            + "".join(
                f"\nJudge part lines[{p.first_line}-{p.last_line}] ({_clock(p.start)}-{_clock(p.end)}): "
                f'WER {p.wer:.0%}, heard "{p.heard[:160]}"'
                for p in sorted(self.parts, key=lambda p: -p.wer)[:3]
                if p.errors
            )
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
    # The attachment of `review`, when Vis attached it.
    review_attachment_id: str | None = None
    # The lines whose audio differs from the earlier render in the same directory.
    changed: tuple[int, ...] = ()
    # `changes.mp3`: only the changed lines, each with one second before it.
    review: str | None = None
    # True when the script, voices, settings and code did not change, so nothing was rendered again.
    is_skipped: bool = False

    def __str__(self) -> str:
        cached = sum(1 for t in self.takes if t.is_cached)
        head = "Unchanged since the last render: nothing was rendered again.\n" if self.is_skipped else ""
        label = "Changed in the last render" if self.is_skipped else "Changed lines"
        listen = f"; listen: {self.review}" if self.review else ""
        changes = f"{label}: {', '.join(map(str, self.changed))}{listen}\n" if self.changed else ""
        return (
            f"{self.audio}\n{head}{changes}{len(self.takes)} lines ({cached} cached), {self.retakes} retakes, "
            f"unverified: {', '.join(f'lines[{t.line}] ({t.problem})' for t in self.takes if not t.is_verified) or 'none'}, "
            f"rendered in {self.render_seconds / 60:.1f} min\n{self.quality}"
        )


def _sha(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _seed(episode: Episode, index: int, attempt: int) -> int:
    """Return the speech seed of one take of line `index`.

    The seed comes from the speaker, the text and the count of earlier lines with the same speaker
    and text, not from the position. An inserted or removed line therefore keeps the other takes.
    """
    line = episode.lines[index]
    same = sum(1 for other in episode.lines[:index] if (other.speaker, other.text) == (line.speaker, line.text))
    return (episode.seed + int(_sha(line.speaker, line.text, same), 16) + 104729 * attempt) % (2**31 - 1)


def _legacy_seed(episode: Episode, index: int, attempt: int) -> int:
    """Return the seed that version 0.1 gave a take by line position, so its cached takes stay in use."""
    return (episode.seed + 7919 * index + 104729 * attempt) % (2**31 - 1)


def _gap(episode: Episode, index: int) -> float:
    """Return the silence before line `index`, without its own `pause`."""
    if index == 0:
        return LEAD_IN
    before, line = episode.lines[index - 1], episode.lines[index]
    if before.speaker == line.speaker:
        return SAME_SPEAKER_GAP
    base = QUESTION_GAP if before.text.rstrip().endswith("?") else TURN_GAP
    # The two lines give the jitter, so a render keeps its timeline and an edit moves only its own gaps.
    # A pair that the script repeats counts its earlier copies, so repeats still vary.
    pair = (before.speaker, before.text, line.speaker, line.text)
    lines = episode.lines
    same = sum(
        1 for j in range(1, index) if (lines[j - 1].speaker, lines[j - 1].text, lines[j].speaker, lines[j].text) == pair
    )
    jitter = (int(_sha(episode.seed, *pair, same), 16) % 1001 / 500 - 1) * GAP_JITTER
    return round(base + jitter, 3)


def _clock(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _read_json(path: Path) -> dict[str, Any]:
    """Return the JSON object in `path`, or an empty dict when the file is missing or broken."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_atomic(path: Path, data: str | bytes) -> None:
    """Write a whole file or nothing, so an interrupted render never leaves half a file."""
    tmp = path.with_name(f".{path.name}.tmp")
    if isinstance(data, bytes):
        tmp.write_bytes(data)
    else:
        tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


@lru_cache(maxsize=1)
def _code_sha() -> str:
    """Hash the code of this package, so a render that other code made is not reused."""
    files = sorted(Path(__file__).parent.glob("*.py"))
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()[:20]


def _judge_spans(takes: Sequence[Take]) -> list[tuple[int, int]]:
    """Group lines into judge parts: (first, last) line pairs that fit the judge limit with their pads.

    A part ends after a line whose text hash is higher than that of each line that ends at most
    JUDGE_MIN_PART_SECONDS from it, but not in the first or the last JUDGE_MIN_PART_SECONDS. A cut
    thus depends only on the lines near it, and an edit changes only the parts near it. A part that
    is too long is cut into parts of about the same length; a line longer than the limit is a part alone.
    """
    if not takes:
        return []
    window = JUDGE_MIN_PART_SECONDS
    rank = [int(_sha(t.speaker, t.text), 16) for t in takes]
    begin, finish = takes[0].start, takes[-1].end
    cuts = []
    for i, take in enumerate(takes[:-1]):
        if take.end - begin < window or finish - take.end < window:
            continue
        j = i - 1
        while j >= 0 and take.end - takes[j].end <= window and rank[j] < rank[i]:
            j -= 1
        k = i + 1
        while k < len(takes) and takes[k].end - take.end <= window and rank[k] < rank[i]:
            k += 1
        # An equal line nearby, such as a second "Yes.", stops the cut on both lines.
        if (j < 0 or take.end - takes[j].end > window) and (k == len(takes) or takes[k].end - take.end > window):
            cuts.append(i)
    spans, first = [], 0
    for last in [*cuts, len(takes) - 1]:
        spans += _fit_part(takes, first, last)
        first = last + 1
    return spans


def _fit_part(takes: Sequence[Take], first: int, last: int) -> list[tuple[int, int]]:
    """Cut lines `first` to `last` into parts of about the same length that fit the judge limit with their pads."""
    limit = JUDGE_PART_SECONDS - 2 * JUDGE_PAD
    length = takes[last].end - takes[first].start
    if length <= limit or first == last:
        return [(first, last)]
    target = length / -(-length // limit)
    spans, start = [], first
    for i in range(first, last):
        if takes[i + 1].end - takes[start].start > limit or takes[i].end - takes[start].start >= target:
            spans.append((start, i))
            start = i + 1
    spans.append((start, last))
    return spans


def _changed(earlier: dict[str, Any], takes: Sequence[Take]) -> tuple[int, ...]:
    """Return the lines whose audio is not in the earlier render of the same directory."""
    if not earlier.get("takes"):
        return ()
    # The cache key covers the spoken text too, so a new pronunciation is a change.
    # A version 0.1 manifest has no keys; its takes compare by their text, voice and seed.
    keyed = all(t.get("key") for t in earlier["takes"])

    def ident(t: dict[str, Any]) -> Any:
        return t["key"] if keyed else (t["speaker"], t["text"], t["engine"], t["voice"], t["seed"])

    before = Counter(ident(t) for t in earlier["takes"])
    out = []
    for t in takes:
        key = ident(asdict(t))
        if before[key]:
            before[key] -= 1
        else:
            out.append(t.line)
    return tuple(out)


def _restore(data: dict[str, Any], final: Path, vtt: Path, manifest: Path) -> Render:
    """Rebuild the result of an earlier, unchanged render from its manifest."""
    # Nothing was spoken again, so every take comes from the cache.
    takes = tuple(Take(**{**t, "is_cached": True}) for t in data["takes"])
    q = {k: v for k, v in data["quality"].items() if k != "is_passed"}
    quality = Quality(
        **{
            **q,
            "missing_facts": tuple(q["missing_facts"]),
            "failures": tuple(q["failures"]),
            "facts": tuple(FactCheck(**{**f, "heard_by": tuple(f["heard_by"])}) for f in q.get("facts", ())),
            "parts": tuple(JudgePart(**p) for p in q.get("parts", ())),
        }
    )
    unverified = tuple(t.line for t in takes if not t.is_verified)
    review = data.get("review")
    return Render(
        str(final),
        str(vtt),
        str(manifest),
        takes,
        0,
        unverified,
        0.0,
        quality,
        changed=tuple(data.get("changed") or ()),
        review=review if review and Path(review).exists() else None,
        is_skipped=True,
    )


def _errors(text: str, episode: Episode, heard: str, lang: str) -> WordErrors:
    """Return the errors of the better of two forms: the written text, where a heard spoken form
    counts as written, or the spoken text."""
    written = word_errors(text, heard, language=lang, pronunciations=episode.pronunciations)
    spoken = word_errors(episode.spoken(text), heard, language=lang)
    return min(written, spoken, key=lambda e: e.wer)


def _has_fact(fact: str, heard: str, lang: str, names: frozenset[str] = frozenset()) -> bool:
    return fact_heard(fact, heard, fuzzy=True, language=lang, names=names)


def _script_names(episode: Episode | None) -> frozenset[str]:
    """Return the names of a script in their written and their spoken form; see `proper_names`."""
    return proper_names(episode.text) | proper_names(episode.spoken(episode.text)) if episode else frozenset()


def _excerpt(episode: Episode, part: JudgePart, line: int) -> str:
    """Return the words of a judge part near the place of `line`, found by word position."""
    counts = [word_count(episode.lines[i].text) for i in range(part.first_line, part.last_line + 1)]
    before = sum(counts[: line - part.first_line])
    words = part.heard.split()
    scale = len(words) / max(sum(counts), 1)
    start = max(int(before * scale) - 3, 0)
    end = int((before + counts[line - part.first_line]) * scale) + 3
    return " ".join(words[start:end])


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
            key = self._take_key(cache, engine, voice, spoken, seed, _legacy_seed(episode, index, attempt))
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
                # The WAV comes first: a JSON file means that its WAV is complete.
                _write_atomic(wav, audio.encode_wav(clip))
                _write_atomic(meta, json.dumps(saved))
            heard, heard_by = saved["heard"], self._asr_engine
            lang = episode.language
            wer = min(
                word_errors(spoken, heard, names=names, language=lang).wer,
                word_errors(line.text, heard, names=names, language=lang).wer,
            )
            if wer > max_wer and self._judge_engine not in (None, self._asr_engine):
                # Parakeet can hear a French line as English. A take fails only when the judge mishears it too.
                if "judged" not in saved:
                    saved["judged"] = self._hear(clip, self._judge_engine)
                    _write_atomic(meta, json.dumps(saved))
                judged = min(
                    word_errors(spoken, saved["judged"], names=names, language=lang).wer,
                    word_errors(line.text, saved["judged"], names=names, language=lang).wer,
                )
                if judged < wer:
                    wer, heard, heard_by = judged, saved["judged"], self._judge_engine
            rate = words / clip.seconds if clip.seconds else 0.0
            ok = wer <= max_wer and MIN_WORDS_PER_SECOND <= rate <= MAX_WORDS_PER_SECOND
            score = (not ok, wer)
            if best is None or score < best[0]:
                record = {
                    "seed": saved.get("seed", seed),
                    "wer": round(wer, 4),
                    "pace": round(rate, 2),
                    "heard": heard,
                    "heard_by": heard_by,
                    "is_verified": ok,
                    "key": key,
                }
                best = (score, clip, record)
            if ok:
                break
        _, clip, record = best
        return clip, {**record, "attempts": attempt + 1, "is_cached": is_cached}

    def _take_key(self, cache: Path, engine: str, voice: str, spoken: str, seed: int, legacy: int) -> str:
        """Return the cache key of a take. A take that version 0.1 cached under its position seed gets the new key."""
        key = _sha(engine, voice, spoken, seed, self._asr_engine)
        old = _sha(engine, voice, spoken, legacy, self._asr_engine)
        new_wav, old_wav = cache / f"{key}.wav", cache / f"{old}.wav"
        if not new_wav.exists() and old_wav.exists() and (cache / f"{old}.json").exists():
            # The WAV comes first: a JSON file means that its WAV is complete.
            for suffix in (".wav", ".json"):
                tmp = cache / f".{key}{suffix}.tmp"
                tmp.write_bytes((cache / f"{old}{suffix}").read_bytes())
                os.replace(tmp, cache / f"{key}{suffix}")
        return key

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
        force: bool = False,
    ) -> Render:
        """Speak every line, verify it with ASR, retake failed lines, then master and evaluate.

        Each line gets up to `max_takes` seeds until its WER is at most `max_wer` and its pace is
        plausible; else the best take stays and the line is listed in `unverified`. Takes are cached
        in `out_dir/takes` and judge parts in `out_dir/judge`, so a second render speaks and judges
        only changed lines. A render with the same script, voices, settings and code, whose takes are
        all in the cache, returns the earlier result at once. `force=True` renders again from the
        cached takes and lets the judge hear every part again. Writes `episode.mp3`, `episode.vtt` with
        speaker tags and `manifest.json`. When the directory has an earlier render, `changes.mp3`
        holds only the lines whose audio changed. `progress(done, total, message)` hears each line.
        Raises ScriptError for an invalid script or a missing voice.
        """
        started = time.monotonic()
        episode = load_episode(script)
        chosen, voice_issues = self._resolve(episode)
        if voice_issues:
            raise ScriptError(tuple(voice_issues))
        out = Path(out_dir).expanduser()
        cache = out / "takes"
        cache.mkdir(parents=True, exist_ok=True)
        g = gates or Gates()
        manifest, final, vtt = out / "manifest.json", out / "episode.mp3", out / "episode.vtt"
        voices = {h: f"{e}:{v}" for h, (e, v) in chosen.items()}
        settings = (voices, self._asr_engine, self._judge_engine, max_takes, max_wer, asdict(g), _code_sha())
        fingerprint = _sha(episode.to_dict(), *settings)
        earlier = _read_json(manifest)
        if (
            not force
            and earlier.get("fingerprint") == fingerprint
            and final.exists()
            and vtt.exists()
            and all((cache / f"{t.get('key')}.wav").exists() for t in earlier.get("takes", ()))
        ):
            return _restore(earlier, final, vtt, manifest)
        if earlier:
            # An interrupted render must not look finished: only a complete render writes a fingerprint.
            _write_atomic(manifest, json.dumps({**earlier, "fingerprint": None}, indent=2))
        parts, takes, clock = [], [], LEAD_IN

        def ahead(i: int) -> Future | None:
            """Speak the first take of line `i` in the background when the MLX worker speaks it.

            ASR then checks one line while the GPU speaks the next. Vis calls stay on this thread.
            """
            if i >= len(episode.lines):
                return None
            engine, voice = chosen[episode.lines[i].speaker]
            spoken, seed = episode.spoken(episode.lines[i].text), _seed(episode, i, 0)
            key = self._take_key(cache, engine, voice, spoken, seed, _legacy_seed(episode, i, 0))
            if engine != QWEN_ENGINE or (cache / f"{key}.wav").exists():
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
            # The silence before the first line is the lead-in, already in the clock, and its pause.
            clock += line.pause if i == 0 else gap
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
                    rec["heard_by"],
                    rec["key"],
                )
            )
            parts.append((gap, clip))
            clock += clip.seconds
        joined = audio.join(parts + [(TAIL, audio.Clip(audio.RATE, audio.array.array("h")))])
        raw = out / "episode-raw.wav"
        _write_atomic(raw, audio.encode_wav(joined))
        if progress:
            progress(len(episode.lines), len(episode.lines), "Mastering and judging the episode")
        with ThreadPoolExecutor(1, thread_name_prefix="podcast-master") as pool:
            # FFmpeg masters while the judge hears the raw mix: mastering changes only the loudness.
            mastered = pool.submit(audio.master, raw, final, lufs=g.target_lufs)
            heard_parts = self._judge_lines(episode, parts, takes, out / "judge", force=force)
            mastered.result()
        changed = _changed(earlier, takes)
        review = self._review(joined, parts, changed, out / "changes.mp3", g) if earlier.get("takes") else None
        names = {h.id: h.name for h in episode.hosts}
        _write_atomic(
            vtt,
            "WEBVTT\n\n"
            + "\n".join(
                f"{t.line + 1}\n{_vtt_time(t.start)} --> {_vtt_time(t.end)}\n<v {names[t.speaker]}>{t.text}\n"
                for t in takes
            ),
        )
        quality = self._evaluate(final, episode, g, parts=heard_parts, takes=takes)
        # A judge part with no words is a failed judge call, so the next render must judge again.
        is_judged = all(part["heard"] for part in heard_parts)
        elapsed = time.monotonic() - started
        _write_atomic(
            manifest,
            json.dumps(
                {
                    "title": episode.title,
                    "script": episode.to_dict(),
                    "script_sha": _sha(episode.to_dict()),
                    "fingerprint": fingerprint if is_judged else None,
                    "voices": voices,
                    "asr_engine": self._asr_engine,
                    "judge_engine": self._judge_engine,
                    "max_takes": max_takes,
                    "max_wer": max_wer,
                    "render_seconds": round(elapsed, 1),
                    "changed": list(changed),
                    "review": review,
                    "takes": [asdict(t) for t in takes],
                    "quality": {**asdict(quality), "is_passed": quality.is_passed},
                },
                indent=2,
            ),
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
            changed=changed,
            review=review,
        )

    def _judge_lines(
        self,
        episode: Episode,
        parts: list[tuple[float, audio.Clip]],
        takes: list[Take],
        cache: Path,
        *,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """Let the judge hear the mix in parts of whole lines, and reuse each part that it heard before.

        A part is built from its takes and the gaps between them, so it does not depend on the lines
        around it, and the cache key is its audio and the judge engine. `force` judges every part
        again. An empty result is not cached. Returns one row per part.
        """
        cache.mkdir(parents=True, exist_ok=True)
        engine = self._judge_engine
        empty = audio.Clip(audio.RATE, audio.array.array("h"))
        rows = []
        for first, last in _judge_spans(takes):
            pieces = [(JUDGE_PAD, parts[first][1]), *parts[first + 1 : last + 1], (JUDGE_PAD, empty)]
            clip = audio.pad(audio.join(pieces), MIN_HEARD_SECONDS)
            wav = audio.encode_wav(clip)
            path = cache / f"{hashlib.sha256(str(engine).encode() + wav).hexdigest()[:20]}.json"
            saved = {} if force else _read_json(path)
            is_cached = "heard" in saved
            if not is_cached:
                # `_judge` splits a part that one very long line makes longer than the judge limit.
                saved = {"engine": engine, "heard": self._judge(clip, engine=engine)}
                if saved["heard"]:
                    _write_atomic(path, json.dumps(saved))
            rows.append(
                {
                    "first_line": first,
                    "last_line": last,
                    "start": takes[first].start,
                    "end": takes[last].end,
                    "heard": saved["heard"],
                    "is_cached": is_cached,
                }
            )
        return rows

    def _review(
        self,
        joined: audio.Clip,
        parts: list[tuple[float, audio.Clip]],
        changed: tuple[int, ...],
        target: Path,
        g: Gates,
    ) -> str | None:
        """Write the changed lines, each with one second before it, to `target`; remove an old file when none changed."""
        if not changed:
            target.unlink(missing_ok=True)
            return None
        bounds, at = [], 0
        for gap, clip in parts:
            at += int(gap * joined.rate)
            bounds.append((at, at + len(clip.samples)))
            at += len(clip.samples)
        pieces, run = [], [changed[0]]
        for line in [*changed[1:], None]:
            if line is not None and line == run[-1] + 1:
                run.append(line)
                continue
            start = max(0, bounds[run[0]][0] - int(REVIEW_LEAD * joined.rate))
            end = min(len(joined.samples), bounds[run[-1]][1] + int(REVIEW_TAIL * joined.rate))
            pieces.append((REVIEW_GAP if pieces else 0.0, audio.Clip(joined.rate, joined.samples[start:end])))
            run = [line]
        wav = target.with_name(target.stem + "-raw.wav")
        wav.write_bytes(audio.encode_wav(audio.join(pieces)))
        try:
            audio.master(wav, target, lufs=g.target_lufs)
        finally:
            wav.unlink(missing_ok=True)
        return str(target)

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

    def _judge(self, clip: audio.Clip, timeout: float = 3600, engine: str | None = None) -> str:
        """Return what an engine, by default the judge, hears in a clip, part by part."""
        engine = engine or self._judge_engine
        return " ".join(
            self._api.transcribe(audio.encode_wav(part), engine=engine, timeout=timeout).text.strip()
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
        parts: Sequence[dict[str, Any]] = (),
        takes: Sequence[Take] = (),
    ) -> Quality:
        """Measure `audio_file` like `evaluate`. `heard` is the judge transcript when the caller has it.

        `parts` are the judge parts of a render and `takes` its takes. WER then adds the errors of
        each part against its own lines, and a critical fact that the judge missed counts when the
        take check of the line that says it heard it.
        """
        episode = None
        if isinstance(script, str) and not script.lstrip().startswith("{") and not Path(script).expanduser().exists():
            reference, facts, lang = script, (), "en"
        else:
            episode = load_episode(script)
            reference = episode.text
            facts = tuple((f, episode.spoken(f)) for f in episode.critical_facts)
            lang = episode.language
        clip = audio.decode(audio_file, rate=16000)
        judged: list[JudgePart] = []
        if parts and episode is not None:
            for row in parts:
                text = "\n".join(line.text for line in episode.lines[row["first_line"] : row["last_line"] + 1])
                e = _errors(text, episode, row["heard"], lang)
                judged.append(JudgePart(**row, reference_words=e.reference_words, errors=e.errors, wer=round(e.wer, 4)))
            heard = " ".join(p.heard for p in judged)
            errors, words = sum(p.errors for p in judged), sum(p.reference_words for p in judged)
        else:
            if heard is None:
                heard = self._judge(clip, timeout)
            e = _errors(reference, episode, heard, lang) if episode else word_errors(reference, heard, language=lang)
            errors, words = e.errors, e.reference_words
        checks = [self._fact_check(f, said, heard, episode, judged, takes, lang) for f, said in facts]
        if any(not c.is_heard for c in checks) and not takes and self._asr_engine not in (None, self._judge_engine):
            # Without takes, the ASR engine hears the whole file once more as the second opinion.
            second = self._judge(clip, timeout, engine=self._asr_engine)
            names = _script_names(episode)
            checks = [
                c
                if c.is_heard or not (_has_fact(c.fact, second, lang, names) or _has_fact(said, second, lang, names))
                else replace(c, heard_by=(f"file:{self._asr_engine}",))
                for c, (_, said) in zip(checks, facts, strict=True)
            ]
        missing = tuple(c.fact for c in checks if not c.is_heard)
        loud = audio.measure(audio_file)
        pauses = audio.silences(clip, min_seconds=1.0)
        seconds = clip.seconds
        wpm = word_count(reference) / (seconds / 60) if seconds else 0.0
        longest = max((p[1] for p in pauses), default=0.0)
        clipped = audio.clipped_samples(audio.decode(audio_file))
        # The report and the failure message show the same rounded value.
        wer = round(errors / max(words, 1), 4)
        checks_ = [
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
            words,
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
            tuple(msg for ok, msg in checks_ if not ok),
            heard,
            tuple(checks),
            tuple(judged),
        )

    def _fact_check(
        self,
        fact: str,
        said: str,
        heard: str,
        episode: Episode | None,
        parts: Sequence[JudgePart],
        takes: Sequence[Take],
        lang: str,
    ) -> FactCheck:
        """Look for one fact in the judge transcript, then in the take check of each line that says it."""
        names = _script_names(episode)
        heard_by = (
            [f"mix:{self._judge_engine}"]
            if _has_fact(fact, heard, lang, names) or _has_fact(said, heard, lang, names)
            else []
        )
        if episode is None:
            return FactCheck(fact, tuple(heard_by))
        lines = [
            i
            for i, line in enumerate(episode.lines)
            if _has_fact(fact, line.text, lang, names) or _has_fact(said, episode.spoken(line.text), lang, names)
        ]
        if not lines:
            return FactCheck(fact, tuple(heard_by))
        line = lines[0]
        for i in lines:
            if i < len(takes) and (
                _has_fact(fact, takes[i].heard, lang, names) or _has_fact(said, takes[i].heard, lang, names)
            ):
                heard_by.append(f"take:{takes[i].heard_by or self._asr_engine}")
                line = i
                break
        take = takes[line] if line < len(takes) else None
        part = next((p for p in parts if p.first_line <= line <= p.last_line), None)
        return FactCheck(
            fact,
            tuple(heard_by),
            line,
            take.start if take else None,
            take.end if take else None,
            _excerpt(episode, part, line) if part else "",
            take.heard if take else "",
        )
