"""Qwen3-TTS on Apple silicon through MLX, next to the Vis speech engines.

The model runs in a separate Python environment that has `mlx-audio`. One worker process loads the
model once and serves every line of a render. All other calls go to the wrapped Vis speech API.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENGINE = "qwen3-tts-mlx"
MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
# Set VIS_PODCASTER_QWEN_PYTHON to a Python that has mlx-audio installed.
PYTHON = Path(os.environ.get("VIS_PODCASTER_QWEN_PYTHON", "~/.vis/podcaster/mlx-venv/bin/python")).expanduser()
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
LANGUAGES = {"en": "english", "zh": "chinese", "ja": "japanese", "ko": "korean", "de": "german", "fr": "french"}

WORKER = r"""
import json, os, sys, wave
proto = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model
model = load_model(sys.argv[1])
proto.write(json.dumps({"ready": True}) + "\n"); proto.flush()
for line in sys.stdin:
    req = json.loads(line)
    try:
        if req.get("seed") is not None:
            mx.random.seed(int(req["seed"]))
            np.random.seed(int(req["seed"]) % 2**32)
        rate, parts = 24000, []
        for r in model.generate(text=req["text"], voice=req["voice"], lang_code=req["language"], split_pattern=""):
            rate = r.sample_rate
            parts.append(np.asarray(r.audio, dtype=np.float32).reshape(-1))
        pcm = (np.clip(np.concatenate(parts), -1.0, 1.0) * 32767).astype("<i2")
        with wave.open(req["out"], "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())
        proto.write(json.dumps({"ok": True}) + "\n")
    except Exception as e:
        proto.write(json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n")
    proto.flush()
"""


@dataclass(frozen=True)
class Voice:
    """A Qwen3-TTS speaker, in the shape of a Vis voice."""

    id: str
    label: str
    language: str | None = None
    is_imported: bool = False


@dataclass(frozen=True)
class Speech:
    """Synthesized WAV audio, in the shape of a Vis speech result."""

    content: bytes
    engine: str
    voice: str


class QwenSpeech:
    """A speech API that adds the `qwen3-tts-mlx` engine to a Vis speech API.

    `base` is the Vis speech API or a callable that returns it. `python` must have `mlx-audio`.
    """

    def __init__(self, base: Any | Callable[[], Any], *, python: str | Path = PYTHON, model: str = MODEL):
        self._base_ref = base
        self._python = Path(python)
        self._model = model
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @property
    def _base(self) -> Any:
        b = self._base_ref
        return b() if callable(b) and not hasattr(b, "speak") else b

    @property
    def is_available(self) -> bool:
        return self._python.exists()

    def voices(self, engine: str | None = None) -> list[Any]:
        if engine != ENGINE:
            return self._base.voices(engine)
        if not self.is_available:
            raise RuntimeError(f"{ENGINE}: no Python with mlx-audio at {self._python}")
        return [Voice(vid, label) for vid, label in VOICES]

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
        lang = LANGUAGES.get(language.lower()[:2], "auto")
        with self._lock, tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "take.wav"
            reply = self._ask(
                {"text": text, "voice": voice or VOICES[0][0], "seed": seed, "language": lang, "out": str(out)}
            )
            if "error" in reply:
                raise RuntimeError(f"{ENGINE}: {reply['error']}")
            return Speech(out.read_bytes(), ENGINE, voice or VOICES[0][0])

    def transcribe(self, audio: bytes, *, engine: str | None = None, timeout: float = 600) -> Any:
        return self._base.transcribe(audio, engine=engine, timeout=timeout)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def close(self) -> None:
        """Stop the worker process. The next take starts it again."""
        proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            proc.stdin.close()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _ask(self, request: dict) -> dict:
        proc = self._worker()
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            self.close()
            raise RuntimeError(f"{ENGINE}: the worker stopped")
        return json.loads(line)

    def _worker(self) -> subprocess.Popen:
        if self._proc and self._proc.poll() is None:
            return self._proc
        if not self.is_available:
            raise RuntimeError(f"{ENGINE}: no Python with mlx-audio at {self._python}")
        proc = subprocess.Popen(
            [str(self._python), "-c", WORKER, self._model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        ready = proc.stdout.readline()
        if not ready.startswith('{"ready"'):
            proc.kill()
            raise RuntimeError(f"{ENGINE}: the model {self._model} did not load")
        self._proc = proc
        return proc
