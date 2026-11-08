"""Podcast music: original pieces from a seed, music settings, and the shapes that put music around speech."""

import array
import math
import wave
from pathlib import Path

import pytest

from vis_podcaster import audio
from vis_podcaster.music import MOODS, Music, between, closing, compose, opening, resolve

RATE = audio.RATE


def samples(path: Path) -> array.array:
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, RATE)
        return array.array("h", w.readframes(w.getnframes()))


def rms_db(clip) -> float:
    return 10 * math.log10(sum(v * v for v in clip) / max(1, len(clip)) / 32768**2 + 1e-12)


@pytest.fixture(scope="module")
def warm(tmp_path_factory):
    return compose(tmp_path_factory.mktemp("warm"), mood="warm", seed="Long Story Short")


def test_a_seed_gives_the_same_music_and_another_seed_other_music(warm, tmp_path):
    again = compose(tmp_path / "again", mood="warm", seed="Long Story Short")
    other = compose(tmp_path / "other", mood="warm", seed="Another Show")
    for name in ("theme", "sting"):
        assert getattr(again, name).read_bytes() == getattr(warm, name).read_bytes()
        assert getattr(other, name).read_bytes() != getattr(warm, name).read_bytes()


def test_the_theme_is_eight_bars_with_a_two_bar_lead_and_sound_in_each_bar(warm):
    theme, sting = samples(warm.theme), samples(warm.sting)
    bar = warm.lead / 2
    assert 8 * bar < len(theme) / RATE < 8 * bar + 4
    assert 2 < len(sting) / RATE < 6
    for b in range(8):
        assert rms_db(theme[round(b * bar * RATE) : round((b + 1) * bar * RATE)]) > -40, f"bar {b + 1} is silent"
    for clip in (theme, sting):
        assert max(map(abs, clip)) < 32000
        assert rms_db(clip) > -35


@pytest.mark.parametrize("mood", MOODS)
def test_each_mood_writes_a_theme_and_a_sting(mood, tmp_path):
    music = compose(tmp_path, mood=mood, seed=1)
    assert (music.theme, music.sting) == (tmp_path / "theme.wav", tmp_path / "sting.wav")
    assert rms_db(samples(music.theme)) > -35 and rms_db(samples(music.sting)) > -35
    assert not list(tmp_path.glob(".*.part"))


def test_a_faster_mood_has_a_shorter_lead(tmp_path):
    leads = {mood: compose(tmp_path / mood, mood=mood, seed=1).lead for mood in MOODS}
    assert leads["bright"] < leads["warm"] < leads["calm"]


def test_an_unknown_mood_names_the_moods(tmp_path):
    with pytest.raises(ValueError, match="warm, bright, calm"):
        compose(tmp_path, mood="sad")


def test_resolve_takes_a_mood_a_file_a_mapping_or_music(warm, tmp_path):
    assert resolve(None, tmp_path) is None
    assert resolve(warm, tmp_path) == warm
    assert resolve("warm", tmp_path / "m", seed="Long Story Short").theme.read_bytes() == warm.theme.read_bytes()
    assert resolve(str(warm.theme), tmp_path) == Music(theme=warm.theme)
    mapped = resolve({"theme": str(warm.theme), "sting": warm.sting, "bed": -20}, tmp_path)
    assert mapped == Music(theme=warm.theme, sting=warm.sting, bed=-20)
    composed = resolve({"mood": "calm", "seed": 2, "level": -6}, tmp_path / "c")
    assert composed.level == -6 and composed.theme == tmp_path / "c" / "theme.wav"


def test_resolve_refuses_missing_files_and_bad_values(warm, tmp_path):
    with pytest.raises(FileNotFoundError, match="nowhere.mp3"):
        resolve("nowhere.mp3", tmp_path)
    with pytest.raises(FileNotFoundError, match="'wram' does not exist; a mood is one of warm, bright, calm"):
        resolve("wram", tmp_path)
    with pytest.raises(FileNotFoundError, match="sting.*gone.wav"):
        resolve({"sting": tmp_path / "gone.wav"}, tmp_path)
    with pytest.raises(ValueError, match="unknown music keys"):
        resolve({"volume": 3}, tmp_path)
    with pytest.raises(ValueError, match="bed"):
        resolve({"theme": warm.theme, "bed": 3}, tmp_path)
    with pytest.raises(ValueError, match="lead"):
        resolve({"theme": warm.theme, "lead": -1}, tmp_path)
    with pytest.raises(ValueError, match="mood"):
        resolve(42, tmp_path)


def test_the_opening_goes_down_under_the_first_word_then_stops():
    theme = array.array("h", [10000] * (40 * RATE))
    onset = 5 * RATE
    out = opening(theme, onset, -15.0)
    assert out[onset - RATE] == 10000
    assert out[onset + RATE] == pytest.approx(10000 * 10 ** (-15 / 20), abs=2)
    assert onset + 2 * RATE < len(out) < onset + 10 * RATE
    assert abs(out[-1]) < 20


def test_the_closing_cuts_a_long_theme_with_a_fade_and_keeps_a_short_one():
    long = array.array("h", [8000] * (60 * RATE))
    out = closing(long, 30.0)
    first = next(i for i, v in enumerate(out) if v)
    assert 0.3 * RATE < first < 3 * RATE
    assert len(out) - first == pytest.approx(30 * RATE, abs=2)
    assert out[first + 10 * RATE] == 8000 and abs(out[-10]) < 100
    short = array.array("h", [8000] * (5 * RATE))
    assert len(closing(short, 30.0)) - first == pytest.approx(len(short), abs=2)


def test_a_gap_with_a_sting_grows_to_fit_it_with_room_around_it():
    sting = array.array("h", [9000] * (3 * RATE))
    gap = between(sting, round(1.6 * RATE))
    loud = [i for i, v in enumerate(gap) if abs(v) > 100]
    assert len(gap) > len(sting) and len(loud) == pytest.approx(len(sting), abs=0.05 * RATE)
    assert loud[0] > 0.3 * RATE and len(gap) - loud[-1] > 0.3 * RATE
    assert len(between(None, 1234)) == 1234 and not any(between(None, 1234))
