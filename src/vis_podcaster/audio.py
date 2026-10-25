"""Audio steps: decode, trim, join, master and measure. FFmpeg does the codec and loudness work."""

from __future__ import annotations

import array
import io
import json
import math
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

RATE = 24000
FRAME = 0.02
_FFMPEG_TIMEOUT = 600
# Speech mastering before loudness normalization: remove rumble, then even out the level gently.
VOICE_CHAIN = "highpass=f=70,acompressor=threshold=-21dB:ratio=2.5:attack=5:release=150:makeup=2"


class AudioError(RuntimeError):
    """FFmpeg is missing or an audio step failed."""


def ffmpeg() -> str:
    """Return the FFmpeg path or raise AudioError."""
    path = shutil.which("ffmpeg") or next(
        (p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg") if Path(p).exists()), None
    )
    if not path:
        raise AudioError("ffmpeg is not installed; install it, for example with `brew install ffmpeg`")
    return path


def _run(args: list[str], data: bytes | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [ffmpeg(), "-hide_banner", "-nostdin", *args],
        input=data,
        capture_output=True,
        timeout=_FFMPEG_TIMEOUT,
        check=False,
    )
    if proc.returncode:
        tail = proc.stderr.decode(errors="replace").strip().splitlines()[-3:]
        raise AudioError(f"ffmpeg failed: {' | '.join(tail)}")
    return proc


@dataclass(frozen=True)
class Clip:
    """Mono 16-bit PCM samples at `rate` Hz."""

    rate: int
    samples: array.array

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.rate


def decode(audio: bytes | str | Path, rate: int = RATE) -> Clip:
    """Decode any audio that FFmpeg reads to mono 16-bit PCM at `rate` Hz."""
    data = audio if isinstance(audio, bytes) else Path(audio).read_bytes()
    try:
        with wave.open(io.BytesIO(data)) as w:
            if w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == rate:
                samples = array.array("h", w.readframes(w.getnframes()))
                if sys.byteorder == "big":
                    samples.byteswap()
                return Clip(rate, samples)
    except (wave.Error, EOFError):
        pass
    raw = _run(["-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", str(rate), "pipe:1"], data).stdout
    samples = array.array("h", raw)
    if sys.byteorder == "big":
        samples.byteswap()
    return Clip(rate, samples)


def encode_wav(clip: Clip) -> bytes:
    """Return the clip as a WAV file."""
    buf = io.BytesIO()
    samples = clip.samples
    if sys.byteorder == "big":
        samples = array.array("h", samples)
        samples.byteswap()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(clip.rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def frame_levels(clip: Clip) -> list[float]:
    """Return the RMS level of each 20 ms frame in dBFS."""
    n = max(1, int(clip.rate * FRAME))
    levels = []
    s = clip.samples
    for i in range(0, len(s), n):
        chunk = s[i : i + n]
        power = sum(x * x for x in chunk) / max(len(chunk), 1)
        levels.append(10 * math.log10(power / 32768**2) if power else -120.0)
    return levels


def trim(clip: Clip, *, threshold_db: float = -45.0, margin: float = 0.06) -> Clip:
    """Remove leading and trailing silence, keeping `margin` seconds of room on each side."""
    levels = frame_levels(clip)
    voiced = [i for i, db in enumerate(levels) if db > threshold_db]
    if not voiced:
        return Clip(clip.rate, array.array("h"))
    n = int(clip.rate * FRAME)
    start = max(0, voiced[0] * n - int(margin * clip.rate))
    end = min(len(clip.samples), (voiced[-1] + 1) * n + int(margin * clip.rate))
    return Clip(clip.rate, clip.samples[start:end])


def split(clip: Clip, max_seconds: float = 60.0, *, search: float = 10.0) -> list[Clip]:
    """Cut a clip into parts of at most `max_seconds`, each cut in the quietest 300 ms before the limit."""
    levels, n, window = frame_levels(clip), int(clip.rate * FRAME), 15
    limit, reach = int(max_seconds / FRAME), int(search / FRAME)
    parts, start = [], 0
    while len(levels) - start > limit:
        lo, hi = start + max(limit - reach, window), start + limit
        quiet = min(range(lo, hi - window + 1), key=lambda i: sum(levels[i : i + window]))
        cut = quiet + window // 2
        parts.append(Clip(clip.rate, clip.samples[start * n : cut * n]))
        start = cut
    parts.append(Clip(clip.rate, clip.samples[start * n :]))
    return parts


def join(parts: list[tuple[float, Clip]], rate: int = RATE) -> Clip:
    """Join clips; each pair gives the silence in seconds before its clip."""
    out = array.array("h")
    for gap, clip in parts:
        if clip.rate != rate:
            raise AudioError(f"clip rate {clip.rate} differs from {rate}")
        out.extend(array.array("h", bytes(2 * int(gap * rate))))
        out.extend(clip.samples)
    return Clip(rate, out)


def silences(clip: Clip, *, threshold_db: float = -40.0, min_seconds: float = 1.0) -> list[tuple[float, float]]:
    """Return (start, length) in seconds of each inner silence at least `min_seconds` long."""
    levels = frame_levels(clip)
    found, run_start = [], None
    for i, db in enumerate(levels + [0.0]):
        if db <= threshold_db and run_start is None:
            run_start = i
        elif db > threshold_db and run_start is not None:
            length = (i - run_start) * FRAME
            if length >= min_seconds and run_start > 0 and i < len(levels):
                found.append((run_start * FRAME, length))
            run_start = None
    return found


def clipped_samples(clip: Clip) -> int:
    """Count samples at full scale, a sign of distortion."""
    return sum(1 for x in clip.samples if x >= 32767 or x <= -32768)


@dataclass(frozen=True)
class Loudness:
    """EBU R128 measurement: integrated loudness, true peak and loudness range."""

    lufs: float
    true_peak: float
    lra: float


def _loudnorm_json(stderr: bytes) -> dict:
    text = stderr.decode(errors="replace")
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", text, re.DOTALL)
    if not match:
        raise AudioError("ffmpeg loudnorm printed no measurement")
    return json.loads(match.group(0))


def measure(path: str | Path) -> Loudness:
    """Measure a file with FFmpeg loudnorm."""
    m = _loudnorm_json(_run(["-i", str(path), "-af", "loudnorm=print_format=json", "-f", "null", "-"]).stderr)
    return Loudness(float(m["input_i"]), float(m["input_tp"]), float(m["input_lra"]))


def master(
    wav: str | Path, out: str | Path, *, lufs: float = -16.0, true_peak: float = -1.5, bitrate: str = "96k"
) -> Loudness:
    """Apply VOICE_CHAIN, a gain to `lufs` and a peak limiter below `true_peak`; encode by suffix.

    The gain is measured, then corrected up to two times until the result is within 0.2 LU.
    `.mp3` uses LAME at `bitrate`; `.m4a` uses AAC; `.wav` stays PCM. Return the result loudness.
    """
    out = Path(out)
    codec = {
        ".mp3": ["-c:a", "libmp3lame", "-b:a", bitrate],
        ".m4a": ["-c:a", "aac", "-b:a", bitrate],
        ".wav": ["-c:a", "pcm_s16le"],
    }.get(out.suffix.lower())
    if codec is None:
        raise AudioError(f"unsupported output type {out.suffix!r}; use .mp3, .m4a or .wav")
    probe = _loudnorm_json(
        _run(["-i", str(wav), "-af", f"{VOICE_CHAIN},loudnorm=print_format=json", "-f", "null", "-"]).stderr
    )
    gain = lufs - float(probe["input_i"])
    # The limiter works on sample peaks; 1 dB of room keeps the true peak under the ceiling.
    limit = 10 ** ((true_peak - 1.0) / 20)
    result = None
    for _ in range(4):
        chain = f"{VOICE_CHAIN},volume={gain:.2f}dB,alimiter=limit={limit:.4f}:attack=2:release=60:level=false"
        _run(["-y", "-i", str(wav), "-af", chain, "-ar", str(RATE), "-ac", "1", *codec, str(out)])
        result = measure(out)
        is_loud = abs(result.lufs - lufs) <= 0.2
        if is_loud and result.true_peak <= true_peak:
            break
        gain += lufs - result.lufs
        # Lossy encoding can add overshoot above the limiter; lower the limit by that excess.
        if result.true_peak > true_peak:
            limit *= 10 ** ((true_peak - result.true_peak - 0.2) / 20)
    return result
