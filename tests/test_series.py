"""Series tools: one audiobook from chapter renders, and checks across the scripts of a series."""

import array
import copy
import hashlib
import itertools
import json
import math
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from vis_podcaster import audio
from vis_podcaster.script import load_episode
from vis_podcaster.series import Book, assemble, check_series

RATE = audio.RATE


def tone(seconds: float, freq: float, amp: int) -> audio.Clip:
    n = int(seconds * RATE)
    return audio.Clip(RATE, array.array("h", (int(amp * math.sin(2 * math.pi * freq * i / RATE)) for i in range(n))))


def make_render(folder: Path, title: str, seconds: float, *, freq: float = 220.0, amp: int = 4000) -> Path:
    """Write a small render folder: a tone raw mix, a manifest with two takes and a VTT."""
    folder.mkdir(parents=True)
    (folder / "episode-raw.wav").write_bytes(audio.encode_wav(tone(seconds, freq, amp)))
    half = round(seconds / 2, 3)
    takes = [
        {"line": 0, "speaker": "a", "text": "First line.", "start": 0.3, "end": half},
        {"line": 1, "speaker": "b", "text": "Second line.", "start": half + 0.2, "end": seconds - 0.5},
    ]
    manifest = {"title": title, "script": {"title": title, "show": "Long Story Short"}, "takes": takes, "quality": {}}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "episode.vtt").write_text(
        "WEBVTT\n\n"
        f"1\n00:00:00.300 --> 00:00:{half:06.3f}\n<v Ann>First line.\n\n"
        f"2\n00:00:{half + 0.2:06.3f} --> 00:00:{seconds - 0.5:06.3f}\n<v Ben>Second line.\n"
    )
    return folder


def ffprobe(path: Path) -> dict:
    exe = shutil.which("ffprobe") or str(Path(audio.ffmpeg()).with_name("ffprobe"))
    out = subprocess.run(
        [exe, "-v", "error", "-show_chapters", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        check=True,
    ).stdout
    return json.loads(out)


@pytest.fixture
def renders(tmp_path):
    return [
        make_render(tmp_path / "ch1", "One: Start", 3.0, amp=2500),
        make_render(tmp_path / "ch2", "Two: Middle", 4.0, freq=330.0, amp=12000),
        make_render(tmp_path / "ch3", "Three: End", 2.5, freq=440.0, amp=6000),
    ]


def check_chapter_marks(book: Book, path: Path, titles: list[str], gap: float) -> None:
    probe = ffprobe(path)
    marks = [(float(c["start_time"]), float(c["end_time"]), c["tags"]["title"]) for c in probe["chapters"]]
    assert [m[2] for m in marks] == titles
    assert [c.title for c in book.chapters] == titles
    for (start, end, _), chapter in zip(marks, book.chapters, strict=True):
        assert start == pytest.approx(chapter.start, abs=0.002)
        assert end == pytest.approx(chapter.end, abs=0.002)
    # The gap belongs to the chapter before it: no hole between marks.
    assert marks[1][0] == pytest.approx(3.0 + gap, abs=0.01)
    assert marks[2][0] == pytest.approx(3.0 + gap + 4.0 + gap, abs=0.01)
    assert all(a[1] == pytest.approx(b[0], abs=0.002) for a, b in itertools.pairwise(marks))
    assert book.seconds == pytest.approx(3.0 + 4.0 + 2.5 + 2 * gap, abs=0.01)
    assert float(probe["format"]["duration"]) == pytest.approx(book.seconds, abs=0.1)


def test_mp3_book_has_chapter_marks_transcript_and_list(renders, tmp_path):
    out = tmp_path / "book" / "series.mp3"
    book = assemble(renders, out, gap=1.6, title="My Series", artist="Ann and Ben")
    titles = ["One: Start", "Two: Middle", "Three: End"]
    check_chapter_marks(book, out, titles, 1.6)
    probe = ffprobe(out)
    assert probe["format"]["format_name"] == "mp3"
    tags = probe["format"]["tags"]
    assert tags["title"] == "My Series" and tags["artist"] == "Ann and Ben"
    assert out.read_bytes()[:5] == b"ID3\x03\x00"  # ID3v2.3
    assert book.loudness.lufs == pytest.approx(-16.0, abs=1.0)
    assert "3 chapters" in str(book) and "series.mp3" in str(book)

    vtt = (tmp_path / "book" / "series.vtt").read_text()
    assert book.transcript == tmp_path / "book" / "series.vtt"
    assert vtt.startswith("WEBVTT")
    notes = [row for row in vtt.splitlines() if row.startswith("NOTE")]
    assert notes == [f"NOTE {t}" for t in titles]
    # Cues of chapter two move by chapter one plus the gap; numbers run over the whole book.
    assert "3\n00:00:04.900 --> 00:00:06.600\n<v Ann>First line." in vtt
    assert "\n6\n00:00:" in vtt and "\n7\n" not in vtt

    listing = (tmp_path / "book" / "series.chapters.md").read_text()
    assert "| 0:00:00 | One: Start |" in listing
    assert "| 0:00:05 | Two: Middle |" in listing
    assert "| 0:00:10 | Three: End |" in listing


@pytest.mark.parametrize("suffix", [".m4b", ".m4a"])
def test_m4b_book_has_chapter_marks_and_faststart(renders, tmp_path, suffix):
    out = tmp_path / f"series{suffix}"
    book = assemble(renders, out, gap=1.0)
    check_chapter_marks(book, out, ["One: Start", "Two: Middle", "Three: End"], 1.0)
    probe = ffprobe(out)
    assert "mp4" in probe["format"]["format_name"]
    data = out.read_bytes()
    assert data.find(b"moov") < data.find(b"mdat")  # faststart: the index comes first
    assert (tmp_path / "series.vtt").is_file() and (tmp_path / "series.chapters.md").is_file()


def test_each_chapter_is_mastered_to_the_same_loudness(renders, tmp_path):
    out = tmp_path / "series.mp3"
    book = assemble(renders, out, lufs=-18.0)
    clip = audio.decode(out)
    for k, chapter in enumerate(book.chapters):
        part = clip.samples[int((chapter.start + 0.2) * RATE) : int((chapter.start + 2.0) * RATE)]
        wav = tmp_path / f"part{k}.wav"
        wav.write_bytes(audio.encode_wav(audio.Clip(RATE, array.array("h", part))))
        assert audio.measure(wav).lufs == pytest.approx(-18.0, abs=1.0), chapter.title


def test_manifest_paths_work_and_a_second_run_gives_the_same_book(renders, tmp_path):
    out = tmp_path / "series.mp3"
    first = assemble([r / "manifest.json" for r in renders], out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    vtt = first.transcript.read_text()
    second = assemble([str(r) for r in renders], out)
    assert hashlib.sha256(out.read_bytes()).hexdigest() == digest
    assert second.chapters == first.chapters
    assert second.transcript.read_text() == vtt
    assert first.chapters[0].source == renders[0]
    assert not list(tmp_path.glob(".*.part"))


def test_a_single_chapter_book_has_one_mark_and_the_show_as_title(renders, tmp_path):
    book = assemble(renders[:1], tmp_path / "one.mp3")
    assert ffprobe(tmp_path / "one.mp3")["format"]["tags"]["title"] == "Long Story Short"
    assert (tmp_path / "one.chapters.md").read_text().startswith("# Long Story Short\n")
    assert len(book.chapters) == 1 and book.chapters[0].start == 0.0
    assert book.chapters[0].end == pytest.approx(3.0, abs=0.01)
    assert len(ffprobe(tmp_path / "one.mp3")["chapters"]) == 1


def test_bad_arguments_raise_value_error(renders, tmp_path):
    with pytest.raises(ValueError, match=r"\.ogg"):
        assemble(renders, tmp_path / "series.ogg")
    with pytest.raises(ValueError, match="empty"):
        assemble([], tmp_path / "series.mp3")
    with pytest.raises(ValueError, match="gap"):
        assemble(renders, tmp_path / "series.mp3", gap=-1)
    assert not (tmp_path / "series.ogg").exists()


def test_a_missing_raw_mix_or_manifest_names_the_chapter(renders, tmp_path):
    (renders[1] / "episode-raw.wav").unlink()
    with pytest.raises(FileNotFoundError, match=r"chapter 2 'Two: Middle'.*episode-raw\.wav"):
        assemble(renders, tmp_path / "series.mp3")
    with pytest.raises(FileNotFoundError, match=r"chapter 2 .*nowhere.*manifest\.json"):
        assemble([renders[0], tmp_path / "nowhere"], tmp_path / "series.mp3")
    assert not (tmp_path / "series.mp3").exists()


def test_an_unfinished_render_is_not_a_chapter(renders, tmp_path):
    manifest = renders[1] / "manifest.json"
    manifest.write_text(json.dumps({**json.loads(manifest.read_text()), "fingerprint": None}))
    with pytest.raises(ValueError, match=r"chapter 2 'Two: Middle'.*did not end"):
        assemble(renders, tmp_path / "series.mp3")
    assert not (tmp_path / "series.mp3").exists()


def test_the_results_of_render_are_chapters_too(renders, tmp_path):
    results = [SimpleNamespace(manifest=str(folder / "manifest.json")) for folder in renders]
    book = assemble(results, tmp_path / "series.m4b")
    assert [c.title for c in book.chapters] == ["One: Start", "Two: Middle", "Three: End"]


SCRIPT = {
    "title": "Walls",
    "hosts": [{"id": "a", "name": "Ann", "voice": "emma"}, {"id": "b", "name": "Ben", "voice": "carter"}],
    "pronunciations": {"Qin": "Chin"},
    "critical_facts": ["Ming dynasty"],
    "sources": ["https://whc.unesco.org/en/list/438"],
    "lines": [
        {"speaker": "a", "text": "Who built the wall?"},
        {"speaker": "b", "text": "Qin Shi Huang joined it, and the Ming dynasty rebuilt it."},
        {"speaker": "a", "text": "So the wall is many walls."},
    ],
}


def episode(title: str, lines: list[str], pronunciations: dict[str, str] | None = None):
    data = copy.deepcopy(SCRIPT)
    data["title"] = title
    data["lines"] = [{"speaker": "ab"[i % 2], "text": t} for i, t in enumerate(lines)]
    data["pronunciations"] = pronunciations or {}
    data["critical_facts"] = []
    return load_episode(data)


def kinds(report):
    return [i.kind for i in report.issues]


def test_a_varied_series_passes():
    report = check_series(
        [
            episode("One", ["Who built the wall?", "The Ming dynasty rebuilt it.", "So it is many walls."]),
            episode("Two", ["Why is the canal so long?", "It joins two rivers.", "Next time, the roads."]),
        ]
    )
    assert report.is_passed and report.issues == ()
    assert "passed" in str(report)


def test_shared_openers_and_closers_are_found_after_normalization():
    report = check_series(
        [
            episode("One", ["Welcome back to the show, friends!", "Walls.", "Thanks for listening, see you soon."]),
            episode("Two", ["welcome back — to the SHOW today.", "Canals.", "Bye for now."]),
            episode("Three", ["Welcome back to a new week.", "Roads.", "Thanks for listening, see you next week."]),
        ]
    )
    assert kinds(report) == ["opener", "closer"]
    opener, closer = report.issues
    assert opener.episodes == ("One", "Two")
    assert '"welcome back to the show"' in opener.message and "first line" in opener.message
    assert closer.episodes == ("One", "Three")
    assert '"thanks for listening see you"' in closer.message
    assert "opener:" in str(report) and "closer:" in str(report)


def test_a_shared_sign_off_at_the_end_of_the_last_lines_is_found():
    report = check_series(
        [
            episode(
                "One", ["Who built the wall?", "Walls.", "That was the wall. Thanks for listening to Long Story Short."]
            ),
            episode(
                "Two",
                ["Why is the canal long?", "Canals.", "So much for canals. Thanks for listening to Long Story Short."],
            ),
        ]
    )
    assert kinds(report) == ["closer"] and report.issues[0].episodes == ("One", "Two")
    assert 'end their last line with the same words "thanks for listening to long story short"' in str(report)


def test_three_shared_words_are_not_an_opener():
    report = check_series(
        [
            episode("One", ["Welcome back friends to walls.", "Walls.", "Done one."]),
            episode("Two", ["Welcome back friends of canals.", "Canals.", "Done two."]),
        ]
    )
    assert report.is_passed


def test_pronunciation_conflict_and_missing_pronunciation(tmp_path):
    path = tmp_path / "three.json"
    data = copy.deepcopy(SCRIPT)
    data["title"] = "Three"
    data["pronunciations"] = {}
    data["lines"][2]["text"] = "Qin also built roads, and Qinghai is far away."
    path.write_text(json.dumps(data))
    report = check_series(
        [
            episode("One", ["Who made the first wall?", "Qin did.", "So it is many walls."], {"Qin": "Chin"}),
            episode("Two", ["Why the canal?", "Qin again.", "Next time, roads."], {"Qin": "Kin"}),
            str(path),
        ]
    )
    assert kinds(report) == ["pronunciation_conflict", "pronunciation_missing"]
    conflict, missing = report.issues
    assert conflict.episodes == ("One", "Two")
    assert '"Chin" in "One"' in conflict.message and '"Kin" in "Two"' in conflict.message
    assert missing.episodes[0] == "Three"
    assert 'add "Qin": "Chin" to the pronunciations of "Three"' in missing.message


def test_a_written_form_inside_a_longer_word_is_not_missing():
    report = check_series(
        [
            episode("One", ["Who built the wall?", "Qin did.", "Many walls."], {"Qin": "Chin"}),
            episode("Two", ["Why the canal?", "Qinghai is far.", "Next time, roads."]),
        ]
    )
    assert report.is_passed


def test_a_book_with_music_opens_and_closes_with_the_theme_and_has_stings(renders, tmp_path):
    from vis_podcaster.music import closing, compose

    score = compose(tmp_path / "music", mood="bright", seed=5)
    plain = assemble(renders, tmp_path / "plain.mp3", gap=1.6)
    book = assemble(renders, tmp_path / "music.mp3", gap=1.6, music=score)
    theme_frames = len(audio.decode(score.theme).samples)
    sting_seconds = audio.decode(score.sting).seconds
    # The first word comes at the end of the lead; the first mark still starts at zero.
    assert book.chapters[0].start == 0.0
    vtt = book.transcript.read_text()
    assert f"00:00:{score.lead + 0.3:06.3f} --> " in vtt
    # Each gap grows to hold the sting, and still belongs to the chapter before it.
    first_gap = book.chapters[1].start - (score.lead + 3.0)
    assert sting_seconds < first_gap < sting_seconds + 2.5
    assert book.chapters[0].end == pytest.approx(book.chapters[1].start, abs=0.002)
    # The closing theme belongs to the last chapter.
    speech_end = book.chapters[2].start + 2.5
    assert book.seconds == pytest.approx(
        speech_end + len(closing(array.array("h", bytes(2 * theme_frames)), 30.0)) / RATE, abs=0.01
    )
    assert book.chapters[2].end == pytest.approx(book.seconds, abs=0.002)
    marks = ffprobe(tmp_path / "music.mp3")["chapters"]
    assert float(marks[-1]["end_time"]) == pytest.approx(book.seconds, abs=0.01)

    mix, dry = audio.decode(tmp_path / "music.mp3").samples, audio.decode(tmp_path / "plain.mp3").samples

    def level(clip, a: float, b: float) -> float:
        part = clip[round(a * RATE) : round(b * RATE)]
        return 10 * math.log10(sum(v * v for v in part) / len(part) / 32768**2 + 1e-12)

    def under(a: float, b: float, shift: float) -> float:
        """The level of the music alone under speech: the mix minus the same speech without music."""
        n0, n1, s = round(a * RATE), round(b * RATE), round(shift * RATE)
        diff = [x - y for x, y in zip(mix[n0:n1], dry[n0 - s : n1 - s], strict=True)]
        return 10 * math.log10(sum(v * v for v in diff) / len(diff) / 32768**2 + 1e-12)

    alone = level(mix, 0.2, score.lead - 0.6)
    assert alone > -35
    # Under the first words the theme is about 15 dB lower than alone; later in the book there is no music.
    assert under(score.lead + 0.5, score.lead + 2.5, score.lead) == pytest.approx(alone - 15, abs=5)
    ch2 = book.chapters[1].start
    assert under(ch2 + 0.5, ch2 + 3.5, ch2 - plain.chapters[1].start) < alone - 35
    # The sting plays in the gap after chapter one, where the plain book is silent.
    assert level(mix, score.lead + 3.0 + 0.5, ch2 - 0.5) > -40
    # This first chapter ends under the theme, so the theme fades out in the gap; it does not stop at once.
    assert level(mix, score.lead + 3.0 + 0.05, score.lead + 3.0 + 0.5) > alone - 30
    assert level(dry, 3.0 + 0.2, 3.0 + 1.4) < -70


def test_titles_replace_the_manifest_titles(renders, tmp_path):
    book = assemble(renders, tmp_path / "named.mp3", titles=["Prologue", "Part One", "Epilogue"])
    assert [c.title for c in book.chapters] == ["Prologue", "Part One", "Epilogue"]
    assert [c["tags"]["title"] for c in ffprobe(tmp_path / "named.mp3")["chapters"]] == [
        "Prologue",
        "Part One",
        "Epilogue",
    ]
    with pytest.raises(ValueError, match="titles has 2 names for 3 renders"):
        assemble(renders, tmp_path / "bad.mp3", titles=["One", "Two"])
