"""Qwen3-TTS on Apple silicon through MLX, next to the Vis speech engines.

The model runs in a separate Python environment that has `mlx-audio`. A pool of worker processes serves
the takes: a take uses a free worker, or starts a new one when all are busy. Each worker holds one model:
the CustomVoice model for the built-in speakers or the Base model for cloned voices. It drops one model
before it loads the other, and it runs under MLX cache and memory limits (VIS_PODCASTER_MLX_CACHE_GIB,
default 1, and VIS_PODCASTER_MLX_MEMORY_GIB, default 5). All other calls go to the wrapped Vis speech API.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vis_podcaster import audio

log = logging.getLogger(__name__)

ENGINE = "qwen3-tts-mlx"
MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
# The Base model clones a voice from a short clip and its transcript.
CLONE_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
# Each worker loads its own model copy of about 2 GB, and several workers on one GPU multiply that memory
# for little speed. So one worker serves the takes by default.
MAX_WORKERS = 1
# Set VIS_PODCASTER_QWEN_PYTHON to a Python that has mlx-audio installed.
PYTHON = Path(os.environ.get("VIS_PODCASTER_QWEN_PYTHON", "~/.vis/podcaster/mlx-venv/bin/python")).expanduser()
# Bundled preset voices: one clean clip, its transcript and its license for each voice.
PRESETS = Path(__file__).with_name("voices")
# Voices that you add. Set VIS_PODCASTER_VOICES to keep them in another folder.
LIBRARY = Path(os.environ.get("VIS_PODCASTER_VOICES", "~/.vis/podcaster/voices")).expanduser()
# The built-in CustomVoice speakers. Each one speaks every model language; the label names its origin.
VOICES = (
    ("serena", "Serena (warm female)"),
    ("vivian", "Vivian (bright female)"),
    ("ryan", "Ryan (dynamic English male)"),
    ("aiden", "Aiden (sunny American male)"),
    ("eric", "Eric (lively male)"),
    ("dylan", "Dylan (youthful male)"),
    ("uncle_fu", "Uncle Fu (mellow male)"),
    ("ono_anna", "Ono Anna (Japanese female)"),
    ("sohee", "Sohee (Korean female)"),
)
LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
}
MIN_CLIP_SECONDS = 3.0
MAX_CLIP_SECONDS = 30.0

WORKER = r"""
import gc, json, os, sys, wave
proto = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model
try:
    cache_gib = float(os.environ.get("VIS_PODCASTER_MLX_CACHE_GIB", "1"))
    memory_gib = float(os.environ.get("VIS_PODCASTER_MLX_MEMORY_GIB", "5"))
    if cache_gib <= 0 or memory_gib <= 0:
        raise ValueError(f"cache {cache_gib} GiB and memory {memory_gib} GiB must be above 0")
    mx.set_cache_limit(int(cache_gib * 2**30))
    mx.set_memory_limit(int(memory_gib * 2**30))
except Exception as e:
    # Without limits, MLX keeps freed buffers in its cache and the worker grows without bound.
    proto.write(json.dumps({"error": f"cannot set the MLX memory limits: {type(e).__name__}: {e}"}) + "\n")
    proto.flush()
    sys.exit(3)
name, model = None, None
proto.write(json.dumps({"ready": True}) + "\n"); proto.flush()
for line in sys.stdin:
    req = json.loads(line)
    try:
        if req["model"] != name:
            # Drop the old model before the next one loads: two 1.7B models double the memory.
            name, model = None, None
            gc.collect()
            mx.clear_cache()
            model = load_model(req["model"])
            name = req["model"]
        if req.get("seed") is not None:
            mx.random.seed(int(req["seed"]))
            np.random.seed(int(req["seed"]) % 2**32)
        if req.get("ref_audio"):
            kw = {"ref_audio": req["ref_audio"], "ref_text": req["ref_text"]}
        else:
            kw = {"voice": req["voice"]}
        rate, parts = 24000, []
        for r in model.generate(text=req["text"], lang_code=req["language"], split_pattern="", **kw):
            rate = r.sample_rate
            parts.append(np.asarray(r.audio, dtype=np.float32).reshape(-1))
        pcm = (np.clip(np.concatenate(parts), -1.0, 1.0) * 32767).astype("<i2")
        with wave.open(req["out"], "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())
        proto.write(json.dumps({"ok": True}) + "\n")
    except Exception as e:
        proto.write(json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
    finally:
        mx.clear_cache()
    proto.flush()
"""


@dataclass(frozen=True)
class Voice:
    """A Qwen3-TTS speaker, in the shape of a Vis voice."""

    id: str
    label: str
    language: str | None = None
    is_imported: bool = False
    gender: str | None = None
    style: str | None = None
    license: str | None = None


@dataclass(frozen=True)
class Speech:
    """Synthesized WAV audio, in the shape of a Vis speech result."""

    content: bytes
    engine: str
    voice: str


def _entries(folder: Path) -> list[dict]:
    index = folder / "voices.json"
    if not index.exists():
        return []
    return json.loads(index.read_text(encoding="utf-8"))


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise ValueError(f"the voice name {name!r} has no letters or digits")
    return slug


class QwenSpeech:
    """A speech API that adds the `qwen3-tts-mlx` engine to a Vis speech API.

    `base` is the Vis speech API or a callable that returns it. `python` must have `mlx-audio`.
    Cloned voices come from the bundled `presets` folder and the `library` folder of added voices.
    Threads can speak at the same time: each take uses a free worker or starts a new one. Each worker
    holds its own copy of the model, so at most `max_workers` run; a take beyond that waits for one.
    """

    def __init__(
        self,
        base: Any | Callable[[], Any],
        *,
        python: str | Path = PYTHON,
        model: str = MODEL,
        clone_model: str = CLONE_MODEL,
        presets: str | Path = PRESETS,
        library: str | Path = LIBRARY,
        max_workers: int = MAX_WORKERS,
    ):
        self._base_ref = base
        self._python = Path(python)
        self._model = model
        self._clone_model = clone_model
        self._presets = Path(presets)
        self._library = Path(library)
        self._idle: list[subprocess.Popen] = []
        self._starting = 0
        self._workers: set[subprocess.Popen] = set()
        self._pool_lock = threading.Condition()
        self._max_workers = max(1, max_workers)
        self._library_lock = threading.Lock()

    @property
    def _base(self) -> Any:
        b = self._base_ref
        return b() if callable(b) and not hasattr(b, "speak") else b

    @property
    def is_available(self) -> bool:
        return self._python.exists()

    def _clones(self) -> dict[str, tuple[dict, Path, bool]]:
        """Return each cloned voice by id: its entry, its clip and whether you added it."""
        out = {}
        for folder, is_imported in ((self._presets, False), (self._library, True)):
            for e in _entries(folder):
                out[e["id"]] = (e, folder / f"{e['id']}.wav", is_imported)
        return out

    def voices(self, engine: str | None = None) -> list[Any]:
        if engine != ENGINE:
            return self._base.voices(engine)
        if not self.is_available:
            raise RuntimeError(f"{ENGINE}: no Python with mlx-audio at {self._python}")
        clones = [
            Voice(e["id"], e["label"], e.get("language"), imported, e.get("gender"), e.get("style"), e.get("license"))
            for e, _, imported in self._clones().values()
        ]
        return [*clones, *(Voice(vid, label) for vid, label in VOICES)]

    def speak(
        self,
        text: str,
        *,
        voice: str | None = None,
        engine: str | None = None,
        seed: int | None = None,
        timeout: float = 600,
        language: str = "en",
    ) -> Any:
        if engine != ENGINE:
            return self._base.speak(text, voice=voice, engine=engine, seed=seed, timeout=timeout)
        voice = voice or VOICES[0][0]
        lang = LANGUAGES.get(language.lower()[:2], "auto")
        clone = self._clones().get(voice)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "take.wav"
            request = {"text": text, "seed": seed, "language": lang, "out": str(out)}
            if clone:
                entry, clip, _ = clone
                request |= {"model": self._clone_model, "ref_audio": str(clip), "ref_text": entry["transcript"]}
            else:
                request |= {"model": self._model, "voice": voice}
            reply = self._ask(request)
            if "error" in reply:
                raise RuntimeError(f"{ENGINE}: {reply['error']}")
            return Speech(out.read_bytes(), ENGINE, voice)

    def add_voice(
        self,
        recording: str | Path | bytes,
        name: str,
        *,
        language: str | None = None,
        transcript: str | None = None,
        engine: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Clone a voice from 3 to 30 seconds of one clean speaker into the voice library.

        Without `transcript`, the Vis ASR writes it. A voice with the same name is replaced.
        """
        if engine != ENGINE:
            return self._base.add_voice(
                recording, name, language=language, transcript=transcript, engine=engine, **kwargs
            )
        vid = _slug(name)
        if any(e["id"] == vid for e in _entries(self._presets)):
            raise ValueError(f"{vid!r} is a bundled voice; choose another name")
        source = recording if isinstance(recording, bytes) else Path(recording).expanduser()
        clip = audio.trim(audio.decode(source))
        if not MIN_CLIP_SECONDS <= clip.seconds <= MAX_CLIP_SECONDS:
            raise ValueError(f"the clip has {clip.seconds:.1f} s of speech; give 3 to 30 seconds")
        wav = audio.encode_wav(clip)
        text = (transcript or "").strip() or self._base.transcribe(wav).text.strip()
        if not text:
            raise ValueError("the clip has no words that ASR can hear; give its transcript")
        entry = {"id": vid, "label": name, "language": language, "transcript": text}
        with self._library_lock:
            self._library.mkdir(parents=True, exist_ok=True)
            (self._library / f"{vid}.wav").write_bytes(wav)
            self._write([*(e for e in _entries(self._library) if e["id"] != vid), entry])
        return Voice(vid, name, language, True)

    def update_voice(
        self,
        voice_id: str,
        *,
        engine: str | None = None,
        name: str | None = None,
        language: str | None = None,
        transcript: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Change the label, language or clip transcript of an added voice. None removes a value."""
        if engine != ENGINE:
            changes = {"name": name, "language": language, "transcript": transcript, **kwargs}
            return self._base.update_voice(voice_id, engine=engine, **changes)
        with self._library_lock:
            entries = _entries(self._library)
            entry = next((e for e in entries if e["id"] == voice_id), None)
            if entry is None:
                raise KeyError(f"no added voice {voice_id!r}")
            entry["label"] = name or entry["label"]
            entry["language"] = language
            if transcript:
                entry["transcript"] = transcript
            self._write(entries)
        return Voice(voice_id, entry["label"], entry["language"], True)

    def remove_voice(self, voice_id: str, *, engine: str | None = None, **kwargs: Any) -> bool:
        """Delete an added voice. Return False when it does not exist."""
        if engine != ENGINE:
            return self._base.remove_voice(voice_id, engine=engine, **kwargs)
        with self._library_lock:
            entries = _entries(self._library)
            kept = [e for e in entries if e["id"] != voice_id]
            if len(kept) == len(entries):
                return False
            self._write(kept)
            (self._library / f"{voice_id}.wav").unlink(missing_ok=True)
        return True

    def is_added(self, voice_id: str) -> bool:
        """Return True when `voice_id` is a voice in the library of added voices."""
        return any(e["id"] == voice_id for e in _entries(self._library))

    def transcribe(self, audio: bytes, *, engine: str | None = None, timeout: float = 600) -> Any:
        return self._base.transcribe(audio, engine=engine, timeout=timeout)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def close(self) -> None:
        """Stop every free worker now and each busy one after its take. The next take starts one again."""
        with self._pool_lock:
            procs, self._idle, self._workers = self._idle, [], set()
        for proc in procs:
            _stop(proc)

    def _write(self, entries: list[dict]) -> None:
        index = self._library / "voices.json"
        tmp = index.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(index)

    def _ask(self, request: dict) -> dict:
        proc = self._take()
        try:
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
        except (OSError, ValueError) as e:
            log.warning("%s: the worker failed (%s); stopping it", ENGINE, e)
            line = ""
        if not line:
            self._drop(proc)
            raise RuntimeError(f"{ENGINE}: the worker stopped")
        self._give(proc)
        return json.loads(line)

    def _take(self) -> subprocess.Popen:
        """Return a free worker, or start one when all are busy. Wait only at `max_workers`."""
        with self._pool_lock:
            while True:
                while self._idle:
                    proc = self._idle.pop()
                    if proc.poll() is None:
                        return proc
                    self._workers.discard(proc)
                if len(self._workers) + self._starting < self._max_workers:
                    self._starting += 1
                    break
                self._pool_lock.wait()
        try:
            proc = self._start()
        finally:
            with self._pool_lock:
                self._starting -= 1
                self._pool_lock.notify()
        with self._pool_lock:
            self._workers.add(proc)
        return proc

    def _give(self, proc: subprocess.Popen) -> None:
        """Return a worker after its take. A worker that `close` dropped stops here."""
        with self._pool_lock:
            if proc in self._workers and proc.poll() is None:
                self._idle.append(proc)
                self._pool_lock.notify()
                return
        _stop(proc)

    def _drop(self, proc: subprocess.Popen) -> None:
        with self._pool_lock:
            self._workers.discard(proc)
            self._pool_lock.notify()
        _stop(proc)

    def _start(self) -> subprocess.Popen:
        if not self.is_available:
            raise RuntimeError(f"{ENGINE}: no Python with mlx-audio at {self._python}")
        proc = subprocess.Popen(
            [str(self._python), "-c", WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        ready = proc.stdout.readline()
        if not ready.startswith('{"ready"'):
            proc.kill()
            detail = ready.strip() or "no reply"
            raise RuntimeError(f"{ENGINE}: the worker did not start with {self._python}: {detail}")
        return proc


def _stop(proc: subprocess.Popen) -> None:
    """Close the worker's input so it exits, and kill it after 10 s."""
    if proc.poll() is not None:
        return
    try:
        proc.stdin.close()
    except OSError as e:
        log.warning("%s: could not close the worker input: %s", ENGINE, e)
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        log.warning("%s: the worker did not stop in 10 s; killing it", ENGINE)
        proc.kill()
