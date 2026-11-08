"""Vis entrypoint. The production logic lives in vis_podcaster; this file binds it to the host."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import blockether.vis.extension as vis

from vis_podcaster import (
    Book,
    Gates,
    Podcaster,
    Quality,
    QwenSpeech,
    Render,
    ScriptCheck,
    SeriesReport,
    VoiceChoice,
    music,
    series,
)


def _voices_activity(*, phase, result, **_):
    if phase != "success":
        return None
    rows = tuple((v.engine, v.id, v.label, "cloned" if v.is_imported else "built-in") for v in result)
    return vis.ActivityPresentation(
        "List podcast voices",
        f"{len(result)} voices",
        (vis.ActivityTable(("Engine", "Voice", "Label", "Kind"), rows[:200]),),
    )


def _voice_activity(label):
    def render(*, phase, result, args, kwargs, **_):
        if phase != "success":
            return None
        if isinstance(result, bool):
            target = kwargs.get("voice_id") or (args[0] if args else "voice")
            return vis.ActivityPresentation(label, f"{target}: {'removed' if result else 'not found'}")
        return vis.ActivityPresentation(label, f"{result.label} ({result.engine}:{result.id})")

    return render


def _check_activity(*, phase, result, **_):
    if phase != "success":
        return None
    errors = len(result.errors)
    warnings = len(result.issues) - errors
    summary = (
        f"{'Ready' if result.is_ready else 'Not ready'}: {result.words} words, {errors} errors, {warnings} warnings"
    )
    return vis.ActivityPresentation("Check podcast script", summary, (vis.ActivityText(str(result)[:4000]),))


def _evaluate_activity(*, phase, result, **_):
    if phase != "success":
        return None
    verdict = "passed" if result.is_passed else f"{len(result.failures)} gate(s) failed"
    return vis.ActivityPresentation(
        "Evaluate podcast", f"WER {result.wer:.1%}, {verdict}", (vis.ActivityText(str(result)),)
    )


def _render_activity(*, phase, result, **_):
    if phase != "success":
        return None
    q = result.quality
    verdict = "passed" if q.is_passed else f"{len(q.failures)} gate(s) failed"
    state = "unchanged, " if result.is_skipped else f"{len(result.changed)} changed, " if result.changed else ""
    audios = ((result.attachment_id, "Episode"), (result.review_attachment_id, "Changes"))
    blocks = tuple(vis.ActivityAudio(row, label) for row, label in audios if row)
    return vis.ActivityPresentation(
        "Render podcast",
        f"{state}{q.seconds / 60:.1f} min, {len(result.takes)} lines, WER {q.wer:.1%}, {verdict}",
        (*blocks, vis.ActivityText(str(result))),
    )


def _assemble_activity(*, phase, result, **_):
    if phase != "success":
        return None
    rows = tuple((c.title, f"{c.start / 60:.1f}", f"{(c.end - c.start) / 60:.1f}") for c in result.chapters)
    return vis.ActivityPresentation(
        "Assemble audiobook",
        str(result),
        (vis.ActivityTable(("Chapter", "Start (min)", "Length (min)"), rows[:200]),),
    )


def _series_activity(*, phase, result, **_):
    if phase != "success":
        return None
    verdict = "passed" if result.is_passed else f"{len(result.issues)} issue(s)"
    return vis.ActivityPresentation("Check podcast series", verdict, (vis.ActivityText(str(result)[:4000]),))


def _here(value):
    """Resolve a relative path against the session workspace; keep JSON text, mappings and bytes."""
    if isinstance(value, str) and not value.lstrip().startswith("{"):
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else vis.workspace_root() / path)
    return value


def _music(value):
    """Resolve the files of a music argument against the session workspace; keep moods and other values."""
    if isinstance(value, str) and value not in music.MOODS:
        return _here(value)
    if isinstance(value, dict):
        return {k: _here(v) if k in ("theme", "sting") and isinstance(v, str) else v for k, v in value.items()}
    return value


def _speech():
    speech = getattr(vis, "speech", None)
    if speech is None:
        raise RuntimeError("this Vis build has no speech API (vis.speech); update Vis")
    return speech


class VisPodcaster(Podcaster):
    """Two-host podcasts with Vis speech engines: voices, script checks, verified renders and quality gates."""

    @vis.method(activity=vis.Activity(label="List podcast voices", show_start=False, render=_voices_activity))
    def voices(self, engine: str | None = None, *, language: str | None = "en") -> list[VoiceChoice]:
        return super().voices(engine, language=language)

    @vis.method(activity=vis.Activity(label="Add podcast voice", render=_voice_activity("Add podcast voice")))
    def add_voice(
        self, recording: str | bytes, name: str, *, transcript: str | None = None, language: str = "en"
    ) -> VoiceChoice:
        return super().add_voice(_here(recording), name, transcript=transcript, language=language)

    @vis.method(
        activity=vis.Activity(
            label="Update podcast voice", show_start=False, render=_voice_activity("Update podcast voice")
        )
    )
    def update_voice(
        self, voice_id: str, *, name: str | None = None, language: str | None = None, transcript: str | None = None
    ) -> VoiceChoice:
        return super().update_voice(voice_id, name=name, language=language, transcript=transcript)

    @vis.method(
        activity=vis.Activity(
            label="Remove podcast voice", show_start=False, render=_voice_activity("Remove podcast voice")
        )
    )
    def remove_voice(self, voice_id: str) -> bool:
        return super().remove_voice(voice_id)

    @vis.method(activity=vis.Activity(label="Check podcast script", show_start=False, render=_check_activity))
    def check(self, script: str | dict) -> ScriptCheck:
        return super().check(_here(script))

    @vis.method(activity=vis.Activity(label="Render podcast", render=_render_activity))
    def render(
        self,
        script: str | dict,
        out_dir: str,
        *,
        max_takes: int = 3,
        max_wer: float = 0.08,
        gates: Gates | None = None,
        force: bool = False,
    ) -> Render:
        def progress(done: int, total: int, message: str) -> None:
            vis.publish_activity(
                vis.ActivityPresentation("Render podcast", message, (vis.ActivityProgress("Lines", done, total),))
            )

        result = super().render(
            _here(script),
            _here(out_dir),
            max_takes=max_takes,
            max_wer=max_wer,
            gates=gates,
            progress=progress,
            force=force,
        )
        try:
            row = vis.attach(result.audio, label="Episode", audience="user")
        except Exception:  # noqa: BLE001 - without a python_execution block there is no transcript to attach to
            return result
        review = vis.attach(result.review, label="Changes", audience="user")["id"] if result.review else None
        return dataclasses.replace(result, attachment_id=row["id"], review_attachment_id=review)

    @vis.method(activity=vis.Activity(label="Evaluate podcast", render=_evaluate_activity))
    def evaluate(
        self, audio_file: str, script: str | dict, *, gates: Gates | None = None, timeout: float = 3600
    ) -> Quality:
        return super().evaluate(_here(audio_file), _here(script), gates=gates, timeout=timeout)

    @vis.method(activity=vis.Activity(label="Check podcast series", show_start=False, render=_series_activity))
    def check_series(self, scripts: list[str | dict]) -> SeriesReport:
        return series.check_series([_here(s) for s in scripts])

    @vis.method(activity=vis.Activity(label="Assemble audiobook", render=_assemble_activity))
    def assemble(
        self,
        renders: list[str],
        out_file: str,
        *,
        gap: float = 1.6,
        lufs: float = -16.0,
        title: str | None = None,
        artist: str | None = None,
        titles: list[str] | None = None,
        music: str | dict | None = None,
        timeout: float = 1800,
    ) -> Book:
        return series.assemble(
            [_here(r) for r in renders],
            _here(out_file),
            gap=gap,
            lufs=lufs,
            title=title,
            artist=artist,
            titles=titles,
            music=_music(music),
            timeout=timeout,
        )


# The overrides only bind Activities; doc() shows the library contract.
for _name in ("voices", "add_voice", "update_voice", "remove_voice", "check", "render", "evaluate"):
    getattr(VisPodcaster, _name).__doc__ = getattr(Podcaster, _name).__doc__
for _name in ("check_series", "assemble"):
    getattr(VisPodcaster, _name).__doc__ = getattr(series, _name).__doc__


vis.register_extension(
    vis.Extension(
        name="vis-podcaster",
        description="Produce two-host podcasts with Vis speech engines, ASR-verified takes and quality gates.",
        alias="podcast",
        prompt=(
            "podcast: write a script, check it with podcast.check(script), then podcast.render(script, out_dir). "
            "For a series: podcast.check_series(scripts), then podcast.assemble(render_dirs, 'book.m4b'). "
            "podcast.assemble(..., music='warm') adds an original theme and stings; it also works on one render. "
            "Read doc('vis-podcaster/podcast') before the first episode."
        ),
        symbols=[vis.Symbol(VisPodcaster(QwenSpeech(_speech)), name="podcast")],
    )
)
