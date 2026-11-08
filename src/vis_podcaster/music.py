"""Podcast music: an original theme and sting from a seed, and the shapes that put music around speech.

`compose` makes the music itself, note by note, so the music is new for each seed and has no license terms.
`Music` also takes your own files. `series.assemble` mixes the music into a book: the theme opens the book
and stays quietly under the first words, a sting fills each gap between chapters, and the theme closes the book.
"""

from __future__ import annotations

import array
import functools
import hashlib
import itertools
import math
import operator
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from . import audio

RATE = audio.RATE
MOODS = ("warm", "bright", "calm")

_N = 4096  # samples in one cycle of a wave table
_MASK = _N - 1
_DUCK = 0.35  # seconds: the theme starts to go down this long before the first word
_DIP = 0.15  # seconds: the theme is at the bed level this long after the first word
_HOLD = 3.0  # seconds of bed under the first words
_FADE = 3.0  # seconds: then the bed fades out
_EDGE = 0.02  # seconds of fade at a cut edge, so that no click occurs
_STING_ROOM = 0.6  # seconds of silence before and after a sting
_OUTRO_PAUSE = 1.0  # seconds of silence between the last word and the closing theme
_OUTRO_FADE = 3.0  # seconds of fade when the closing theme is cut


@dataclass(frozen=True)
class Music:
    """The music of a book. `theme` and `sting` are audio files that FFmpeg reads; None leaves that place silent.

    The theme plays alone for `lead` seconds, stays `bed` dB lower under the first words, then fades out.
    After the last chapter it plays again, for at most `outro` seconds. The sting fills each gap between
    chapters. `level` is the loudness of the music alone, in dB relative to the speech.
    """

    theme: Path | None = None
    sting: Path | None = None
    lead: float = 5.0
    bed: float = -15.0
    outro: float = 30.0
    level: float = -3.0


@dataclass(frozen=True)
class _Style:
    bpm: float
    drums: bool
    arp: bool
    pad: bool
    sevenths: bool


_STYLES = {
    "warm": _Style(92, drums=True, arp=False, pad=False, sevenths=False),
    "bright": _Style(112, drums=True, arp=True, pad=False, sevenths=False),
    "calm": _Style(74, drums=False, arp=True, pad=True, sevenths=True),
}
# Chords as (root in semitones above the key, is minor), one bar each.
_PROGRESSIONS = (
    ((0, False), (9, True), (5, False), (7, False)),
    ((0, False), (7, False), (9, True), (5, False)),
    ((9, True), (5, False), (0, False), (7, False)),
    ((0, False), (5, False), (9, True), (7, False)),
)
_CADENCE = ((5, False), (7, False), (0, False))
_PENTATONIC = (0, 2, 4, 7, 9)
# Melody rhythms as (start beat, length in beats) in one bar.
_RHYTHMS = (
    ((0, 1), (1, 0.5), (1.5, 0.5), (2, 2)),
    ((0, 0.5), (0.5, 0.5), (1, 1), (2, 0.5), (2.5, 1.5)),
    ((0, 1.5), (1.5, 0.5), (2, 1), (3, 1)),
)
_KEYS = (1.0, 0.16, 0.05, 0.02)
_TINE = (0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.25)
_BASS = (1.0, 0.42, 0.14, 0.05)
_PAD = (1.0, 0.38, 0.21, 0.14, 0.1, 0.08, 0.06, 0.05, 0.04)
_BELL = ((1.0, 1.0, 1.5), (2.0, 0.42, 0.9), (2.76, 0.3, 0.55), (4.07, 0.16, 0.32), (5.4, 0.08, 0.22))


def compose(folder: str | Path, *, mood: str = "warm", seed: int | str = 0) -> Music:
    """Write an original `theme.wav` and `sting.wav` into `folder` and return their `Music`.

    The same mood and seed give the same files; another seed gives another key, other chords and another
    melody. The theme is eight bars that end on the home chord. Its first two bars are the `lead`, so the
    first word comes on the third bar. Raise ValueError for an unknown mood.
    """
    if mood not in _STYLES:
        raise ValueError(f"unknown mood {mood!r}; use one of {', '.join(MOODS)}")
    folder = Path(folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    song = _Song(_STYLES[mood], random.Random(_number(f"{mood}:{seed}")))
    theme, sting = folder / "theme.wav", folder / "sting.wav"
    _finish(song.theme(), theme)
    _finish(song.sting(), sting)
    return Music(theme=theme, sting=sting, lead=2 * song.bar)


def resolve(value: Any, folder: str | Path, *, seed: int | str = "") -> Music | None:
    """Return the `Music` that `value` asks for, or None for no music.

    `value` is None, a `Music`, a mood from MOODS (composed into `folder` with `seed`), the path of a theme
    file, or a mapping with the fields of `Music` and the optional keys `mood` and `seed`.
    Raise ValueError for a bad value and FileNotFoundError that names a missing file.
    """
    if value is None:
        return None
    names = [f.name for f in fields(Music)]
    if isinstance(value, Music):
        music = value
    elif isinstance(value, str | Path) and str(value) in MOODS:
        music = compose(folder, mood=str(value), seed=seed)
    elif isinstance(value, str | Path):
        music = Music(theme=Path(value))
    elif isinstance(value, Mapping):
        unknown = sorted(set(value) - {*names, "mood", "seed"})
        if unknown:
            raise ValueError(f"unknown music keys {unknown}; use mood, seed, {', '.join(names)}")
        base = compose(folder, mood=value["mood"], seed=value.get("seed", seed)) if "mood" in value else Music()
        music = replace(base, **{k: v for k, v in value.items() if k in names})
    else:
        raise ValueError(f"music must be a mood ({', '.join(MOODS)}), a file, a mapping or a Music, not {value!r}")
    files = {}
    for name in ("theme", "sting"):
        path = getattr(music, name)
        if path is not None:
            path = Path(path).expanduser()
            if not path.is_file():
                # A bare word here is most often a mood with a typing error.
                hint = f"; a mood is one of {', '.join(MOODS)}" if name == "theme" and isinstance(value, str) else ""
                raise FileNotFoundError(f"music {name} file {str(path)!r} does not exist{hint}")
        files[name] = path
    music = replace(music, **files)
    if music.lead < 0 or music.outro < 0:
        raise ValueError(f"lead and outro must be zero or more seconds, not {music.lead} and {music.outro}")
    if music.bed > 0:
        raise ValueError(f"bed must be zero or less dB, not {music.bed}")
    return music


def opening(theme: array.array, onset: int, bed: float) -> array.array:
    """Return the opening part of `theme`: full level until the first word at frame `onset`, then `bed` dB lower
    for a few seconds, then a fade to silence."""
    low = 10 ** (bed / 20)
    start, dip = onset - round(_DUCK * RATE), onset + round(_DIP * RATE)
    hold_end, fade = onset + round(_HOLD * RATE), round(_FADE * RATE)
    end = min(len(theme), hold_end + fade)
    out = array.array("h", theme[:end])
    for k in range(max(0, start), end):
        if k < dip:
            gain = 1 - (1 - low) * (0.5 - 0.5 * math.cos(math.pi * (k - start) / (dip - start)))
        elif k < hold_end:
            gain = low
        else:
            gain = low * (0.5 + 0.5 * math.cos(math.pi * (k - hold_end) / fade))
        out[k] = int(out[k] * gain)
    return _edge(out)


def closing(theme: array.array, seconds: float) -> array.array:
    """Return the closing part: a short pause, then `theme` from its start, cut to `seconds` with a fade when it is
    longer."""
    limit = round(seconds * RATE)
    part = array.array("h", theme[:limit])
    if len(theme) > limit:
        fade = min(len(part), round(_OUTRO_FADE * RATE))
        for k in range(fade):
            i = len(part) - fade + k
            part[i] = int(part[i] * (0.5 + 0.5 * math.cos(math.pi * k / fade)))
    return array.array("h", bytes(2 * round(_OUTRO_PAUSE * RATE))) + _edge(part)


def between(sting: array.array | None, frames: int) -> array.array:
    """Return the gap between two chapters: `frames` of silence, or longer with the sting in its middle."""
    if sting is None or not len(sting):
        return array.array("h", bytes(2 * frames))
    room = round(_STING_ROOM * RATE)
    out = array.array("h", bytes(2 * max(frames, 2 * room + len(sting))))
    out[room : room + len(sting)] = _edge(array.array("h", sting))
    return out


def _edge(clip: array.array) -> array.array:
    """Fade the first and last `_EDGE` seconds of `clip` in place, and return it."""
    n = min(len(clip) // 2, round(_EDGE * RATE))
    for k in range(n):
        clip[k] = int(clip[k] * k / n)
        clip[-1 - k] = int(clip[-1 - k] * k / n)
    return clip


def _number(seed: int | str) -> int:
    return int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:8], "big")


def _hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


@functools.cache
def _table(partials: tuple[float, ...]) -> tuple[float, ...]:
    """One cycle of a wave with these harmonic amplitudes, with a peak of 1."""
    wave = [sum(a * math.sin(2 * math.pi * (h + 1) * i / _N) for h, a in enumerate(partials) if a) for i in range(_N)]
    peak = max(map(abs, wave)) or 1.0
    return tuple(v / peak for v in wave)


def _osc(partials: tuple[float, ...], freq: float, n: int) -> list[float]:
    table, step = _table(partials), freq * _N / RATE
    return [table[int(k * step) & _MASK] for k in range(n)]


def _env(n: int, attack: float, decay: float, floor: float, release: float) -> list[float]:
    """Linear attack, exponential decay toward `floor`, and a linear release over the last `release` seconds."""
    a, r = min(n, max(1, round(attack * RATE))), max(1, min(n, round(release * RATE)))
    step = math.exp(-1.0 / (decay * RATE))
    curve = list(itertools.accumulate(itertools.repeat(step, n - 1), operator.mul, initial=1.0))
    if floor:
        curve = [floor + (1 - floor) * v for v in curve]
    for k in range(a):
        curve[k] *= k / a
    for k in range(r):
        curve[n - r + k] *= (r - k) / r
    return curve


def _times(gain: float, values: Any) -> map:
    return map(functools.partial(operator.mul, gain), values)


def _sum(*parts: Any) -> list[float]:
    out = list(parts[0])
    for part in parts[1:]:
        out = list(map(operator.add, out, part))
    return out


class _Song:
    """One piece of music: a key, eight bars of chords and a short melody, all from `rng`."""

    def __init__(self, style: _Style, rng: random.Random):
        self.style, self.rng = style, rng
        self.key = rng.choice((0, 2, 3, 5, 7, 8, 10))
        self.beat = 60.0 / style.bpm
        self.bar = 4 * self.beat
        progression = rng.choice(_PROGRESSIONS)
        self.chords = (*progression, progression[0], *_CADENCE)
        scale = sorted(m for m in range(72, 89) if (m - self.key) % 12 in _PENTATONIC)
        self.tonic = next(m for m in scale if (m - self.key) % 12 == 0)
        self.phrases = (self._phrase(scale, last=None), self._phrase(scale, last=(4, 7)))
        self.notes: dict[tuple, list[float]] = {}
        self.kick, self.snare, self.hat = (
            self._kick(),
            self._noise(0.25, 0.05, 0.3, 0.6),
            self._noise(0.08, 0.016, 1.0, 0.5),
        )

    def _phrase(self, scale: list[int], last: tuple[int, ...] | None) -> list[tuple[float, float, int]]:
        rng = self.rng
        homes = [i for i, m in enumerate(scale) if (m - self.key) % 12 in (0, 4, 7)]
        i = rng.choice(homes[:3])
        notes = []
        for start, length in rng.choice(_RHYTHMS):
            notes.append((start, length, scale[i]))
            i = max(0, min(len(scale) - 1, i + rng.choice((-2, -1, 1, 1, 2))))
        if last:
            end = min(
                (j for j, m in enumerate(scale) if (m - self.key) % 12 in last),
                key=lambda j: abs(scale[j] - notes[-1][2]),
            )
            notes[-1] = (*notes[-1][:2], scale[end])
        return notes

    def _kick(self) -> list[float]:
        phase, out = 0.0, []
        for k in range(round(0.4 * RATE)):
            t = k / RATE
            phase += 2 * math.pi * (46 + 95 * math.exp(-t / 0.03)) / RATE
            out.append(math.sin(phase) * math.exp(-t / 0.17))
        return out

    def _noise(self, seconds: float, decay: float, brightness: float, tone: float) -> list[float]:
        """A noise hit: `brightness` 1 keeps only the hiss; a lower value adds a body tone at 190 Hz."""
        low, out = 0.0, []
        for k in range(round(seconds * RATE)):
            t = k / RATE
            white = self.rng.uniform(-1, 1)
            low += 0.3 * (white - low)
            body = (1 - brightness) * math.sin(2 * math.pi * 190 * t) * math.exp(-t / 0.035)
            out.append(tone * (white - low) * math.exp(-t / decay) + body)
        return out

    def _note(self, kind: str, midi: int, seconds: float) -> list[float]:
        key = (kind, midi, round(seconds * RATE))
        if key not in self.notes:
            n, f = key[2], _hz(midi)
            if kind == "keys":
                body = map(operator.mul, _osc(_KEYS, f, n), _env(n, 0.004, 1.1, 0.0, 0.08))
                tine = map(operator.mul, _osc(_TINE, f, n), _env(n, 0.002, 0.09, 0.0, 0.05))
                self.notes[key] = _sum(body, _times(0.35, tine))
            elif kind == "bass":
                self.notes[key] = list(map(operator.mul, _osc(_BASS, f, n), _env(n, 0.006, 0.3, 0.55, 0.05)))
            elif kind == "pad":
                wide = map(operator.add, _osc(_PAD, f * 1.003, n), _osc(_PAD, f / 1.003, n))
                self.notes[key] = list(map(operator.mul, wide, _env(n, 0.6, 1e9, 1.0, 0.9)))
            elif kind == "bell":
                parts = (
                    map(operator.mul, _osc((1.0,), f * ratio, n), _times(amp, _env(n, 0.002, decay, 0.0, 0.03)))
                    for ratio, amp, decay in _BELL
                )
                self.notes[key] = _sum(*parts)
            else:
                self.notes[key] = self._pluck(f, n)
        return self.notes[key]

    def _pluck(self, freq: float, n: int) -> list[float]:
        """A plucked string (Karplus-Strong)."""
        period = max(2, round(RATE / freq))
        white = [self.rng.uniform(-1, 1) for _ in range(period + 1)]
        line = [(white[i] + white[i + 1]) / 2 for i in range(period)]
        out, i = [0.0] * n, 0
        for k in range(n):
            nxt = i + 1 if i + 1 < period else 0
            out[k] = line[i]
            line[i] = 0.4985 * (line[i] + line[nxt])
            i = nxt
        return _env_end(out)

    def _tones(self, chord: tuple[int, bool], low: int) -> list[int]:
        root, minor = chord
        steps = (0, 3, 7) if minor else (0, 4, 7)
        if self.style.sevenths:
            steps += (10,) if minor else (11,)
        return sorted(low + (self.key + root + s - low) % 12 for s in steps)

    def _place(self, buf: list[float], at: float, note: list[float], gain: float) -> None:
        start = max(0, round(at * RATE))
        end = min(len(buf), start + len(note))
        if start < end:
            buf[start:end] = map(operator.add, buf[start:end], _times(gain, note[: end - start]))

    def _human(self, at: float, gain: float) -> tuple[float, float]:
        return at + self.rng.uniform(-0.006, 0.006), gain * self.rng.uniform(0.85, 1.0)

    def theme(self) -> list[float]:
        """Eight bars: the melody and the chords, the groove from bar three, the melody again, the home chord."""
        style, beat, bar = self.style, self.beat, self.bar
        buf = [0.0] * round((8 * bar + 3.0) * RATE)
        for b, chord in enumerate(self.chords):
            t0, last = b * bar, b == 7
            hits = ((0, 4 * beat + 2.0),) if last else ((0, 2.5 * beat), (2.5 * beat, 1.5 * beat))
            for at, length in hits:
                for m in self._tones(chord, 55):
                    when, gain = self._human(t0 + at, 0.2)
                    self._place(buf, when, self._note("keys", m, length), gain)
            root = 40 + (self.key + chord[0] - 40) % 12
            line = (
                ((0, 4 * beat + 1.5, root),)
                if last
                else (
                    (0, 1.5 * beat, root),
                    (1.5 * beat, 0.5 * beat, root),
                    (2 * beat, 1.5 * beat, root),
                    (3.5 * beat, 0.5 * beat, root + 7),
                )
            )
            for at, length, m in line:
                self._place(buf, t0 + at, self._note("bass", m, length), 0.34)
            if style.pad:
                for m in self._tones(chord, 55):
                    self._place(buf, t0, self._note("pad", m, bar + (2.0 if last else 0.3)), 0.06)
            if style.arp and 2 <= b < 7:
                tones = self._tones(chord, 62)
                order = tones + tones[-2:0:-1]
                for k in range(8):
                    when, gain = self._human(t0 + k * beat / 2, 0.16)
                    self._place(buf, when, self._note("pluck", order[k % len(order)], 0.6), gain)
            if style.drums:
                self._drums(buf, t0, b)
        for b, phrase in ((0, self.phrases[0]), (1, self.phrases[1]), (6, self.phrases[1])):
            for start, length, m in phrase:
                when, gain = self._human(b * bar + start * beat, 0.18)
                self._place(buf, when, self._note("bell", m, max(1.2, length * beat + 0.6)), gain)
        self._place(buf, 7 * bar, self._note("bell", self.tonic, 2.6), 0.2)
        return buf

    def _drums(self, buf: list[float], t0: float, b: int) -> None:
        beat = self.beat
        if b == 7:
            self._place(buf, t0, self.kick, 0.45)
            return
        for k in range(8):
            when, gain = self._human(t0 + k * beat / 2, 0.07 if k % 2 else 0.1)
            self._place(buf, when, self.hat, gain if b >= 2 else gain * 0.6)
        if b < 2:
            return
        kicks = (0, 2, 2.5) if self.style.bpm > 100 else (0, 2)
        for k in kicks:
            self._place(buf, t0 + k * beat, self.kick, 0.45)
        for k in (1, 3):
            when, gain = self._human(t0 + k * beat, 0.2)
            self._place(buf, when, self.snare, gain)

    def sting(self) -> list[float]:
        """The first notes of the melody over the home chord, which then rings out."""
        beat = self.beat
        buf = [0.0] * round((2 * beat + 2.2) * RATE)
        for m in self._tones((0, False), 55):
            self._place(buf, 0, self._note("keys", m, 2 * beat + 1.8), 0.2)
        self._place(buf, 0, self._note("bass", 40 + (self.key - 40) % 12, 2 * beat + 1.2), 0.34)
        if self.style.drums:
            self._place(buf, 0, self.kick, 0.4)
        for start, _, m in (n for n in self.phrases[0] if n[0] < 2):
            self._place(buf, start * beat, self._note("bell", m, 1.4), 0.18)
        self._place(buf, 2 * beat, self._note("bell", self.tonic, 1.9), 0.2)
        return buf


def _env_end(out: list[float]) -> list[float]:
    n = min(len(out), round(0.04 * RATE))
    for k in range(n):
        out[len(out) - n + k] *= (n - k) / n
    return out


def _room(seconds: float = 1.6, rt60: float = 1.3) -> array.array:
    """The impulse response of a small warm room: noise that decays by 60 dB in `rt60` seconds."""
    rng = random.Random(7)
    k60, low, taps = math.log(1000) / (rt60 * RATE), 0.0, []
    for k in range(round(seconds * RATE)):
        low += 0.25 * (rng.uniform(-1, 1) - low)
        taps.append(low * math.exp(-k * k60) * min(1.0, k / (0.012 * RATE)))
    # A unit energy response keeps the room sound at the level that the mix weight gives.
    scale = 32767 / math.sqrt(sum(v * v for v in taps))
    return array.array("h", (round(v * scale) for v in taps))


def _finish(buf: list[float], out: Path) -> None:
    """Add room sound to `buf` and write it to `out` as 16-bit mono WAV, with a peak under full scale."""
    scale = 0.5 * 32767 / (max(map(abs, buf)) or 1.0)
    dry = array.array("h", (int(v * scale) for v in buf))
    part = out.with_name(f".{out.name}.part")
    with tempfile.TemporaryDirectory(prefix="vis-podcaster-music-") as tmp:
        work = Path(tmp)
        (work / "dry.wav").write_bytes(audio.encode_wav(audio.Clip(RATE, dry)))
        (work / "room.wav").write_bytes(audio.encode_wav(audio.Clip(RATE, _room())))
        graph = (
            "[0:a]asplit=2[d][w];[w][1:a]afir=gtype=none:irnorm=-1[r];"
            "[d][r]amix=inputs=2:weights=1 0.28:normalize=0,alimiter=limit=0.9:level=false"
        )
        audio._run(
            ["-y", "-i", str(work / "dry.wav"), "-i", str(work / "room.wav"), "-filter_complex", graph]
            + ["-ar", str(RATE), "-ac", "1", "-c:a", "pcm_s16le", "-flags", "+bitexact", "-fflags", "+bitexact"]
            + ["-f", "wav", str(part)]
        )
    os.replace(part, out)
