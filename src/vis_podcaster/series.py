"""Series tools: join chapter renders into one audiobook and check the scripts of a series together."""

from __future__ import annotations

import array
import json
import os
import re
import subprocess
import tempfile
import wave
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from . import audio
from .music import Music, between, closing, opening, resolve
from .script import Episode, load_episode
from .text import tokens

_ENCODE_TIMEOUT = 1800
_CHUNK_FRAMES = 1 << 16
_SAME_WORDS = 4
_FORMATS = {
    ".mp3": ("mp3", ["-c:a", "libmp3lame", "-b:a", "96k", "-id3v2_version", "3"]),
    ".m4b": ("mp4", ["-c:a", "aac", "-b:a", "80k", "-movflags", "+faststart"]),
    ".m4a": ("mp4", ["-c:a", "aac", "-b:a", "80k", "-movflags", "+faststart"]),
}


@dataclass(frozen=True)
class Chapter:
    """One chapter of a book. `start` and `end` are seconds on the book timeline; `source` is the render folder."""

    title: str
    start: float
    end: float
    source: Path


@dataclass(frozen=True)
class Book:
    """The result of `assemble`: the audio file, its WebVTT transcript, the chapters and the measured loudness."""

    audio: Path
    transcript: Path
    chapters: tuple[Chapter, ...]
    seconds: float
    loudness: audio.Loudness

    def __str__(self) -> str:
        return (
            f"{self.audio.name}: {len(self.chapters)} chapters, {_hms(self.seconds)}, "
            f"{self.loudness.lufs:.1f} LUFS, true peak {self.loudness.true_peak:.1f} dBTP"
        )


def _hms(seconds: float) -> str:
    h, rest = divmod(round(seconds), 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _vtt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    return f"{h:02d}:{m:02d}:{rest // 1000:02d}.{rest % 1000:03d}"


def _parse_vtt_time(text: str) -> float:
    parts = text.strip().split()[0].split(":")
    seconds = float(parts[-1]) + 60 * int(parts[-2])
    return seconds + (3600 * int(parts[-3]) if len(parts) > 2 else 0)


def _cues(folder: Path, manifest: dict) -> list[tuple[float, float, str]]:
    """Return (start, end, text) of each cue of a render, from its VTT or else from its manifest takes."""
    for name in ("episode.vtt", "transcript.vtt"):
        path = folder / name
        if path.is_file():
            cues = []
            for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
                rows = [row for row in block.strip().splitlines() if row.strip()]
                for i, row in enumerate(rows):
                    if "-->" in row:
                        start, end = (_parse_vtt_time(x) for x in row.split("-->"))
                        cues.append((start, end, "\n".join(rows[i + 1 :])))
                        break
            return cues
    return [
        (float(t["start"]), float(t["end"]), str(t["text"]))
        for t in manifest.get("takes", ())
        if {"start", "end", "text"} <= set(t)
    ]


def _render(item: Any, number: int) -> tuple[Path, dict]:
    """Return the render folder and its manifest. Raise FileNotFoundError that names the chapter.

    `item` is a render folder, its `manifest.json` or the `Render` that `Podcaster.render` returns.
    Raise ValueError that names the chapter for a render that did not end or whose judge heard nothing.
    """
    path = Path(getattr(item, "manifest", item)).expanduser()
    folder, manifest = (path.parent, path) if path.suffix == ".json" else (path, path / "manifest.json")
    if not manifest.is_file():
        raise FileNotFoundError(f"chapter {number} ({folder}): {manifest.name} is missing; render the chapter first")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    title = str(data.get("title") or f"Chapter {number}")
    data["title"] = title
    # A render writes its fingerprint when it ends and its judge heard each part; a version 0.1 manifest
    # has no fingerprint key.
    if "fingerprint" in data and not data["fingerprint"]:
        raise ValueError(
            f"chapter {number} {title!r} ({folder}): the render did not end, or its judge heard nothing; "
            "render the chapter again"
        )
    if not (folder / "episode-raw.wav").is_file():
        raise FileNotFoundError(
            f"chapter {number} {title!r} ({folder}): episode-raw.wav is missing; render the chapter again"
        )
    return folder, data


def _music_clip(path: Path, out: Path, lufs: float) -> array.array:
    """Return a music file as 16-bit mono samples at the book rate, with the loudness `lufs`."""
    audio.master(path, out, lufs=lufs, chain="")
    with wave.open(str(out)) as r:
        return array.array("h", r.readframes(r.getnframes()))


def _first_voice(r: wave.Wave_read, limit: float = 10.0) -> int:
    """Return the first frame of `r` above -45 dBFS, looking at most `limit` seconds in; then rewind `r`."""
    clip = audio.Clip(audio.RATE, array.array("h", r.readframes(round(limit * audio.RATE))))
    r.rewind()
    first = next((k for k, db in enumerate(audio.frame_levels(clip)) if db > -45.0), 0)
    return first * round(audio.FRAME * audio.RATE)


def _mix(frames: bytes, overlay: array.array) -> tuple[bytes, array.array]:
    """Add the start of `overlay` to the 16-bit `frames`; return the mixed frames and the rest of `overlay`."""
    pcm = array.array("h", frames)
    n = min(len(pcm), len(overlay))
    pcm[:n] = array.array("h", (max(-32768, min(32767, a + b)) for a, b in zip(pcm[:n], overlay[:n], strict=True)))
    return pcm.tobytes(), overlay[n:]


def _meta(value: str) -> str:
    """Escape a value for an FFMETADATA file."""
    return re.sub(r"([=;#\\\n])", r"\\\1", value)


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def assemble(
    renders: Sequence[Any],
    out_file: str | Path,
    *,
    gap: float = 1.6,
    lufs: float = -16.0,
    title: str | None = None,
    artist: str | None = None,
    titles: Sequence[str] | None = None,
    music: Music | Mapping[str, Any] | str | None = None,
    timeout: float = _ENCODE_TIMEOUT,
) -> Book:
    """Join chapter renders, in order, into one audiobook file with chapter marks.

    Each item of `renders` is a render folder, its `manifest.json` or a `Render`; the manifest `title` is the chapter
    title, and the `show` of the first manifest `script` is the default book `title`. Each chapter's
    `episode-raw.wav` is mastered to `lufs` alone, so every chapter has the same loudness. `gap` seconds
    of silence go between chapters; the gap belongs to the chapter before it.
    The suffix of `out_file` selects the format: `.mp3` (96 kbit/s, ID3v2.3 chapters) or `.m4b`/`.m4a`
    (AAC 80 kbit/s, chapters, faststart). Also writes `<name>.vtt`, with all cues on the book timeline and
    one `NOTE` per chapter, and `<name>.chapters.md`. `timeout` limits the final encoding in seconds.
    `titles` gives one chapter title for each render, in place of the manifest titles.
    `music` adds music: a mood from `music.MOODS` composes an original theme and sting with the book title as
    the seed, a file path is a theme, and a `music.Music` or a mapping of its fields gives your own files and
    levels. The theme opens the book and stays quietly under the first words; the sting plays in each gap, and
    the gap grows to fit it; the theme closes the book. The first chapter mark starts at zero and the last one
    includes the closing theme; the cues move with the music.
    The same input gives the same files. Raise ValueError for a bad argument or a render that is not complete,
    and FileNotFoundError that names the chapter for a missing manifest or raw mix.
    """
    out = Path(out_file).expanduser()
    fmt = _FORMATS.get(out.suffix.lower())
    if fmt is None:
        raise ValueError(f"unsupported book type {out.suffix!r}; use .mp3, .m4b or .m4a")
    if not renders:
        raise ValueError("renders is empty; give at least one render folder")
    if gap < 0:
        raise ValueError(f"gap must be zero or more seconds, not {gap}")
    items = [_render(item, n) for n, item in enumerate(renders, start=1)]
    if titles is not None:
        titles = [str(t) for t in titles]
        if len(titles) != len(items):
            raise ValueError(f"titles has {len(titles)} names for {len(items)} renders; give one for each render")
    if title is None:
        script = items[0][1].get("script")
        title = str(script.get("show") or "") if isinstance(script, dict) else ""
    out.parent.mkdir(parents=True, exist_ok=True)
    gap_frames = round(gap * audio.RATE)
    chapters: list[Chapter] = []
    vtt = ["WEBVTT", ""]
    cue_number = 1
    with tempfile.TemporaryDirectory(prefix="vis-podcaster-book-") as tmp:
        work = Path(tmp)
        score = resolve(music, work / "music", seed=title)
        theme = _music_clip(score.theme, work / "theme.wav", lufs + score.level) if score and score.theme else None
        sting = _music_clip(score.sting, work / "sting.wav", lufs + score.level) if score and score.sting else None
        joined = work / "book.wav"
        clock = 0
        with wave.open(str(joined), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(audio.RATE)
            # The opening theme is on the book timeline: it goes on into a gap when the first chapter is short.
            overlay = array.array("h")
            for i, (folder, manifest) in enumerate(items):
                mastered = work / f"{i:03d}.wav"
                audio.master(folder / "episode-raw.wav", mastered, lufs=lufs)
                with wave.open(str(mastered)) as r:
                    if (r.getnchannels(), r.getsampwidth(), r.getframerate()) != (1, 2, audio.RATE):
                        raise audio.AudioError(f"chapter {i + 1}: mastered audio is not 16-bit mono {audio.RATE} Hz")
                    if i:
                        fill, overlay = _mix(between(sting, gap_frames).tobytes(), overlay)
                        w.writeframes(fill)
                        clock += len(fill) // 2
                    elif theme is not None:
                        # The first word comes at the end of the lead; the theme goes on under it.
                        voice = _first_voice(r)
                        offset = max(0, min(len(theme), round(score.lead * audio.RATE)) - voice)
                        overlay = opening(theme, offset + voice, score.bed)
                        w.writeframes(overlay[:offset].tobytes())
                        overlay = overlay[offset:]
                        clock += offset
                    start = clock
                    while frames := r.readframes(_CHUNK_FRAMES):
                        if overlay:
                            frames, overlay = _mix(frames, overlay)
                        w.writeframes(frames)
                    clock += r.getnframes()
                begin, end = start / audio.RATE, clock / audio.RATE
                name = titles[i] if titles is not None else manifest["title"]
                vtt += [f"NOTE {name.replace('-->', '->')}", ""]
                for cue_start, cue_end, text in _cues(folder, manifest):
                    a, b = begin + min(cue_start, end - begin), begin + min(cue_end, end - begin)
                    vtt += [str(cue_number), f"{_vtt_time(a)} --> {_vtt_time(b)}", text, ""]
                    cue_number += 1
                chapters.append(Chapter(name, begin if i else 0.0, end, folder))
            if theme is not None:
                tail, overlay = _mix(closing(theme, score.outro).tobytes(), overlay)
                tail += overlay.tobytes()
                w.writeframes(tail)
                clock += len(tail) // 2
                chapters[-1] = replace(chapters[-1], end=clock / audio.RATE)
        # The gap after a chapter belongs to it, so the chapter marks leave no hole.
        chapters = [
            Chapter(c.title, c.start, chapters[k + 1].start if k + 1 < len(chapters) else c.end, c.source)
            for k, c in enumerate(chapters)
        ]
        meta = [";FFMETADATA1", "genre=Podcast", "comment=Produced with Vis podcaster"]
        if title:
            meta += [f"title={_meta(title)}", f"album={_meta(title)}"]
        if artist:
            meta += [f"artist={_meta(artist)}"]
        for c in chapters:
            ms = round(c.start * 1000), round(c.end * 1000)
            meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={ms[0]}", f"END={ms[1]}", f"title={_meta(c.title)}"]
        meta_file = work / "chapters.ffmeta"
        meta_file.write_text("\n".join(meta) + "\n", encoding="utf-8")
        muxer, codec = fmt
        part = out.with_name(f".{out.name}.part")
        args = ["-i", str(joined), "-i", str(meta_file), "-map", "0:a", "-map_metadata", "1", "-map_chapters", "1"]
        args += [*codec, "-flags", "+bitexact", "-fflags", "+bitexact", "-f", muxer, str(part)]
        cmd = [audio.ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args]
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
        if proc.returncode:
            part.unlink(missing_ok=True)
            tail = proc.stderr.decode(errors="replace").strip().splitlines()[-3:]
            raise audio.AudioError(f"ffmpeg failed: {' | '.join(tail)}")
        os.replace(part, out)
    seconds = chapters[-1].end
    transcript = out.with_suffix(".vtt")
    _write_text(transcript, "\n".join(vtt))
    rows = [f"# {title}", ""] if title else []
    rows += ["| Start | Chapter |", "| --- | --- |"]
    rows += [f"| {_hms(c.start)} | {c.title.replace('|', '/')} |" for c in chapters]
    _write_text(out.with_name(f"{out.stem}.chapters.md"), "\n".join(rows) + f"\n\nTotal: {_hms(seconds)}\n")
    return Book(out, transcript, tuple(chapters), seconds, audio.measure(out))


IssueKind = Literal["opener", "closer", "pronunciation_conflict", "pronunciation_missing"]


@dataclass(frozen=True)
class SeriesIssue:
    """One problem that spans episodes. `episodes` holds the titles of the episodes it touches."""

    kind: IssueKind
    episodes: tuple[str, ...]
    message: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.message}"


@dataclass(frozen=True)
class SeriesReport:
    """The result of `check_series`. The series passes when it has no issues."""

    issues: tuple[SeriesIssue, ...]

    @property
    def is_passed(self) -> bool:
        return not self.issues

    def __str__(self) -> str:
        if self.is_passed:
            return "Series check passed: no issues."
        return f"Series check found {len(self.issues)} issue(s):\n" + "\n".join(f"- {i}" for i in self.issues)


def _quoted(titles: Iterable[str]) -> str:
    names = [f'"{t}"' for t in titles]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _same_words(episodes: list[Episode], kind: Literal["opener", "closer"]) -> list[SeriesIssue]:
    """Find first lines that start alike, and last lines that start or end alike, such as one sign-off."""
    which = "first" if kind == "opener" else "last"
    issues: list[SeriesIssue] = []
    found: set[tuple[str, ...]] = set()
    for end in ("start", "end") if kind == "closer" else ("start",):
        groups: dict[tuple[str, ...], list[tuple[Episode, list[str]]]] = {}
        for ep in episodes:
            words = tokens(ep.lines[0 if kind == "opener" else -1].text, language=ep.language)
            if end == "end":
                words.reverse()
            if len(words) >= _SAME_WORDS:
                groups.setdefault(tuple(words[:_SAME_WORDS]), []).append((ep, words))
        for group in groups.values():
            titles = tuple(ep.title for ep, _ in group)
            # Two equal lines start and end alike: one issue tells it.
            if len(group) < 2 or titles in found:
                continue
            found.add(titles)
            shared = list(group[0][1])
            for _, words in group[1:]:
                n = next(
                    (k for k, (a, b) in enumerate(zip(shared, words, strict=False)) if a != b),
                    min(len(shared), len(words)),
                )
                shared = shared[:n]
            if end == "end":
                shared.reverse()
            issues.append(
                SeriesIssue(
                    kind,
                    titles,
                    f'{_quoted(titles)} {end} their {which} line with the same words "{" ".join(shared)}"; '
                    f"write a different {which} line for each episode.",
                )
            )
    return issues


def _pronunciations(episodes: list[Episode]) -> list[SeriesIssue]:
    said: dict[str, dict[str, list[str]]] = {}
    for ep in episodes:
        for written, spoken in ep.pronunciations.items():
            said.setdefault(written, {}).setdefault(spoken, []).append(ep.title)
    issues = []
    for written, forms in said.items():
        if len(forms) > 1:
            titles = tuple(dict.fromkeys(t for owners in forms.values() for t in owners))
            uses = "; ".join(f'"{s}" in {_quoted(owners)}' for s, owners in forms.items())
            issues.append(
                SeriesIssue(
                    "pronunciation_conflict",
                    titles,
                    f'"{written}" has different pronunciations: {uses}; use one spoken form in every episode.',
                )
            )
        word = re.compile(rf"(?<!\w){re.escape(written)}(?!\w)")
        lacking = tuple(ep.title for ep in episodes if written not in ep.pronunciations and word.search(ep.text))
        if lacking:
            spoken, owners = next(iter(forms.items()))
            issues.append(
                SeriesIssue(
                    "pronunciation_missing",
                    tuple(dict.fromkeys((*lacking, *owners))),
                    f'"{written}" occurs in {_quoted(lacking)} without a pronunciation, but {_quoted(owners)} '
                    f'says it "{spoken}"; add "{written}": "{spoken}" to the pronunciations of {_quoted(lacking)}.',
                )
            )
    return issues


def check_series(scripts: Sequence[Episode | str | Path | dict[str, Any]]) -> SeriesReport:
    """Check the scripts of one series together, in order.

    Each item is an Episode, a mapping, a script path or JSON text. Reports episodes that open or close with the same four or
    more normalized words, a written form that episodes pronounce differently, and a written form that
    one episode pronounces while another episode uses it without a pronunciation.
    """
    episodes = [s if isinstance(s, Episode) else load_episode(s) for s in scripts]
    issues = _same_words(episodes, "opener") + _same_words(episodes, "closer") + _pronunciations(episodes)
    return SeriesReport(tuple(issues))
