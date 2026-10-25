---
name: podcast
description: Produce a two-host spoken podcast episode with Vis speech engines. Use it when a person asks for a podcast, an audio episode, a dialogue recording or custom host voices.
---

# Two-host podcast

The `podcast` tools turn a written dialogue into a mastered episode. Each line is spoken,
checked by speech recognition and recorded again when the check fails. A second recognizer then
judges the finished file against measurable quality gates.

## Steps

1. List the voices with `print(await podcast.voices())`. Pick two voices that differ clearly.
   Prefer `serena` and `aiden` on `qwen3-tts-mlx`: they sound most natural and have the least noise.
   This engine needs a Python with `mlx-audio` at `~/.vis/podcaster/mlx-venv/bin/python` or in
   `VIS_PODCASTER_QWEN_PYTHON`. Without it, use `emma` and `carter` on `vibevoice-local`.
2. For a custom voice, ask for 3 to 30 seconds of one clean speaker and the words of the clip.
   Call `await podcast.add_voice("clip.wav", "Ada", transcript="...")`. In the host, write the
   returned `id` as `"voice": "pocket-tts-local:<id>"`. Use `podcast.update_voice` to rename a
   voice and `podcast.remove_voice` to delete it. Clone only a voice whose owner agrees.
3. Research the topic and write the script as JSON. Follow [Script rules](#script-rules).
4. Run `print(await podcast.check(script))`. Fix every error. Read every warning and fix it, or
   tell the person why it stays.
5. Run `r = await podcast.render(script, "out/episode")`, then `print(r)`. A five-minute episode
   takes about seven minutes, the final evaluation included.
6. Read the gates in `r.quality`. When a gate fails, follow [Repair](#repair) and render again.
   Takes are cached, so only changed lines are spoken again.
7. Give the person `out/episode/episode.mp3`, `episode.vtt` and the quality report. State any
   gate that still fails and any line in `r.unverified`.

## Script rules

```json
{
  "title": "The Great Wall of China",
  "show": "Long Story Short",
  "language": "en",
  "seed": 2024,
  "hosts": [
    {"id": "maya", "name": "Maya", "voice": "emma", "role": "host"},
    {"id": "daniel", "name": "Daniel", "voice": "carter", "role": "guest"}
  ],
  "pronunciations": {"Qin": "Chin"},
  "critical_facts": ["Ming dynasty"],
  "sources": ["https://whc.unesco.org/en/list/438"],
  "lines": [
    {"speaker": "maya", "text": "Welcome to Long Story Short."},
    {"speaker": "daniel", "text": "Thanks, Maya.", "pause": 0.5}
  ]
}
```

- Write for the ear: short sentences, contractions, questions and reactions between the hosts.
- Keep each turn under 70 words. Alternate the speakers. Give no host more than 75% of the words.
- Write numbers as words, such as "thirteen sixty-eight". Digits make the engine guess.
- Add a pronunciation for a name or an acronym that the engine can misread. The written form
  stays in the transcript; only the engine reads the spoken form.
- List in `critical_facts` the exact phrases that a listener must hear. The check refuses a fact
  that the lines do not contain.
- List the `sources` of the factual claims. Do not invent facts that you cannot source.
- `pause` adds seconds of silence before a turn, for example before a new section.
- Change `seed` for a completely new set of takes.

## Repair

| Failure | Action |
| --- | --- |
| A line in `r.unverified` | `print(r)` gives its WER and pace. Read `heard` in its take. Simplify the sentence or add a pronunciation. |
| WER above the gate | Find the lines with the highest `wer` in `r.takes` and fix them as above. |
| A missing critical fact | Say the fact more plainly, or move it to a short sentence of its own. |
| Pace outside the gate | Shorten long turns or add `pause` values. |
| Loudness or peak | Report it; mastering sets them, so this means a broken input file. |

## Compare another producer

`await podcast.evaluate("their.mp3", script)` applies the same gates to audio from any tool. Give
the same script, or the plain reference text, to compare two producers fairly. See
[references/quality.md](references/quality.md) for the metrics and their limits.
