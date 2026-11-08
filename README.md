# vis-podcaster

Turn a written dialogue into a mastered two-host podcast episode with Vis speech engines.

## When to use

- You want a short spoken episode on a topic, with two hosts who talk to each other.
- You want each line checked by speech recognition, so that a mispronounced line is recorded again.
- You want to compare your episode with audio from another tool, against the same quality gates.
- You want a series of episodes joined into one audiobook with chapters.

## What you get

Ask Vis for an episode in a chat, for example: "Make a five-minute podcast about the Broad Street
cholera outbreak." The `podcast` skill then does these steps:

1. It researches the topic, writes the script and checks the facts against sources.
2. It scores the script against ten review criteria, such as hook, roles, listening and ending.
3. It speaks each line, checks it with speech recognition and records it again when the check fails.
4. A second recognizer judges the finished file.

The result is `episode.mp3`, `episode.vtt` subtitles and a quality report. The report gives the word
error rate, the critical facts that the listener hears, loudness and pace. For each missing fact, it
gives the line, the time and what each recognizer heard.

When you change the script and render again, only the changed lines are spoken again, and the judge
hears again only the parts of the episode near them. `changes.mp3` then holds only those lines, so
you can review an edit without the full episode. A render of an unchanged script returns the earlier
result at once.

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

For a series, check the scripts together, render each episode into its own directory, then join
the renders into one audiobook with chapters:

```python
print(await podcast.check_series(["ep1.json", "ep2.json"]))
book = await podcast.assemble(["out/ep1", "out/ep2"], "out/book.m4b")
print(book)
```

To add music, give a mood: `warm`, `bright` or `calm`. The podcaster composes an original theme and
sting for the book, so the music has no license terms. The theme opens the book and plays quietly
under the first words. The sting plays between the chapters, and the theme closes the book. Music
also works on one render:

```python
book = await podcast.assemble(["out/episode"], "out/episode-music.mp3", music="warm")
```

To use your own music, give a file, such as `music="theme.mp3"`, or a mapping, such as
`{"theme": "theme.mp3", "sting": "sting.wav", "bed": -18}`.

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
