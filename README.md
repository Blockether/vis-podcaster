# vis-podcaster

Turn a written dialogue into a mastered two-host podcast episode with Vis speech engines.

## When to use

- You want a short spoken episode on a topic, with two hosts who talk to each other.
- You want each line checked by speech recognition, so that a mispronounced line is recorded again.
- You want to compare your episode with audio from another tool, against the same quality gates.

## What you get

Ask Vis for an episode in a chat, for example: "Make a five-minute podcast about the Broad Street
cholera outbreak." The `podcast` skill then does these steps:

1. It researches the topic, writes the script and checks the facts against sources.
2. It scores the script against ten review criteria, such as hook, roles, listening and ending.
3. It speaks each line, checks it with speech recognition and records it again when the check fails.
4. A second recognizer judges the finished file.

The result is `episode.mp3`, `episode.vtt` subtitles and a quality report. The report gives the word
error rate, the critical facts that the listener hears, loudness and pace.

## Requirements

- A Vis build with the speech API (`vis.speech`).
- For the fastest and most natural voices: Apple silicon and a Python with `mlx-audio` at
  `~/.vis/podcaster/mlx-venv/bin/python`, or in `VIS_PODCASTER_QWEN_PYTHON`. Without it, the same
  voices work on `qwen3-tts-local`, but slower.
- A five-minute episode takes about five minutes to render on `qwen3-tts-mlx`.

## Install

Install `vis-podcaster` from the Vis Extension Center, then select it for your sessions.

## Use it from Python

```python
check = await podcast.check("episode.json")
print(check)
r = await podcast.render("episode.json", "out/episode")
print(r)
```

`skills/podcast/SKILL.md` gives the script format, the voices and the repair steps.
`examples/great-wall.json` is a complete script.

## Voices

The named voices (`stuart`, `caro`, `fabien`, `estelle` and others) are cloned from bundled clips
with CC0-1.0 or CC-BY-4.0 licenses. `src/vis_podcaster/voices/voices.json` gives the source and the
credit of each clip. Clone only a voice whose owner agrees.

## Develop

```bash
uv sync
uv run pytest
uv run ruff check src tests
```

## License

Apache License 2.0. See `LICENSE`.
