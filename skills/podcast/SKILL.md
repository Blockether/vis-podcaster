---
name: podcast
description: Produce a two-host spoken podcast episode with Vis speech engines. Use it when a person asks for a podcast, an audio episode, a dialogue recording or custom host voices.
---

# Two-host podcast

The `podcast` tools turn a written dialogue into a mastered episode. Each line is spoken,
checked by speech recognition and recorded again when the check fails. A second recognizer then
judges the finished file against measurable quality gates.

## Steps

1. Pick two voices that differ clearly, from [Best voices](#best-voices). Run
   `print(await podcast.voices(language="fr"))` to list the voices of another language.
2. For a custom voice, ask for 3 to 30 seconds of one clean speaker and the words of the clip.
   Call `await podcast.add_voice("clip.wav", "Ada", transcript="...")` and use the returned `id`
   as the host `voice`. Without `transcript`, speech recognition writes it. Use
   `podcast.update_voice` to rename a voice and `podcast.remove_voice` to delete it. Clone only a
   voice whose owner agrees.
3. Research the topic and write the script as JSON. Follow [Write a good episode](#write-a-good-episode)
   and [Script rules](#script-rules).
4. Check the facts before you render. The quality gates measure only what the listener hears, not
   whether it is true. Follow [Check the facts](#check-the-facts).
   Then score the script with [the review criteria](references/craft.md#review-an-episode). Revise it
   until it scores 10 of 10. A reviewer who did not write the script gives the fairest score.
5. Run `print(await podcast.check(script))`. Fix every error. Read every warning and fix it, or
   tell the person why it stays.
6. Run `r = await podcast.render(script, "out/episode")`, then `print(r)`. On `qwen3-tts-mlx`, a
   five-minute episode takes about five minutes: three for the lines and two for the final
   judgement. The next line speaks while speech recognition checks the current one.
7. Read the gates in `r.quality`. When a gate fails, follow [Repair](#repair) and render again.
   Takes are cached, so only changed lines are spoken again.
8. Give the person `out/episode/episode.mp3`, `episode.vtt` and the quality report. State any
   gate that still fails and any line in `r.unverified`.

## Best voices

Use `qwen3-tts-mlx` on Apple silicon. It is the fastest and most natural engine: a line takes
about half its own length to speak. It needs a Python with `mlx-audio` at
`~/.vis/podcaster/mlx-venv/bin/python` or in `VIS_PODCASTER_QWEN_PYTHON`.

| Language | Pair | Other voices |
| --- | --- | --- |
| English | `stuart` (calm male narrator, British) and `caro` (warm female narrator) | `peter` (deep male narrator), `charles` (clear male, British), `una` (casual female host, American) |
| French | `fabien` (upbeat male presenter) and `estelle` (conversational female) | |
| Spanish | `enrique` (neutral male) and `andrea` (warm female) | |
| Any model language | `aiden` (sunny American male) and `serena` (warm female) | `ryan`, `eric`, `dylan`, `vivian`, `uncle_fu`, `ono_anna`, `sohee` |

- The named voices are cloned from bundled clips with CC0 or CC-BY-4.0 licenses. `voices.json`
  in the package gives the source and the credit of each clip.
- Set the script `language`. The model speaks English, Chinese, Japanese, Korean, German, French,
  Spanish, Italian, Portuguese and Russian. Any voice speaks any of these languages, with its own accent.
- `una` adds casual fillers such as "uh", drops some consonants and can turn a statement into a
  question. Use her for a relaxed host, not for a narrator or for lines full of facts.
- `estelle` speaks about a third faster than `fabien`. Give her the shorter turns, or add `pause`
  values before her long turns.
- The model has no Polish and no other language outside the list. For such a language, choose a
  voice of that language from `podcast.voices(language=...)` on a Vis engine.
- Without the MLX Python, the same built-in names work on `qwen3-tts-local`, but slower. The next
  choice is `emma` and `carter` on `vibevoice-local`.

## Write a good episode

Write the episode the way a radio producer does: for a listener who cannot scroll back.
[references/craft.md](references/craft.md) gives each rule with its source.

1. **One point.** Say what the episode is about in one sentence before you write. Write the
   last line, the takeaway, next.
2. **Hook first.** Open with a surprising fact, a scene or a question, in the first two or three
   lines. Keep the full answer for later in the episode. Then name the show and the hosts. Never
   open with "Welcome to".
3. **A conversation, not a shared narration.** Choose a [format](#choose-a-format). Give each fact
   of the story to one host. The other host never continues the story with facts of the same
   story; two voices that read one story in turns sound like a split audiobook.
4. **Segments with questions.** Give each segment one question, then answer it. Mark each change
   of topic: "That's the history. Now, the chemistry."
5. **Story, then meaning.** Tell a concrete event with a person, a place and a time, then say
   what it means. Use one vivid example instead of three abstract claims.
6. **Specific reactions.** React to the fact that was said: "Wait, rice? In mortar?" Do not
   start turns with "Great question", "Absolutely" or "Exactly".
7. **Earned ending.** End on the takeaway or one image, said once. Do not list everything again,
   and do not announce it: "Here's what I'm taking home" is a recap too.

Plan the length at about 170 words a minute, the speed of `qwen3-tts-mlx`: 750 words is about four
and a half minutes.

### Check the facts

Each episode in the first benchmark passed every gate and still had factual errors. Do these
checks on the finished script:

1. Give each factual claim a source in `sources`. Open each URL and confirm that it says the claim.
2. Check the order of events. Confirm that "before", "after", "earlier" and "later" agree with
   the dates.
3. Check each direction word: "ahead" or "behind", "north" or "south", "winter" or "summer".
4. Do the arithmetic between lines. Two numbers in different turns must agree with each other.
5. Check each date for anachronism: a technology, a name or a title that did not exist yet.
6. Remove popular myths, or tell them as myths with the correction.

### Choose a format

| Format | Who knows what | The other host brings |
| --- | --- | --- |
| Storyteller and newcomer | One host tells a story. The other hears it for the first time. | Wrong guesses, surprise, jokes, their own life, common-sense doubt, "so what does that mean for me?" |
| Interview | The guest is the expert and speaks from experience: "When I first saw the data…". | The questions, the follow-ups and the pushback. The host sums up nothing. |
| Two sides | Each host owns one position and its evidence. | A real counter-argument. One host can change their mind at the end, for a reason. |
| Two stories | Each host brings a different story on one theme. | Their own story, then the comparison. |

In each format, the newcomer or interviewer does not know the facts of the story. Give that host
reactions, not information: a guess that is wrong, a memory, an opinion, a joke, a doubt. Let them
interrupt and react to an image, not to the last sentence.

Avoid a lecture with heckles: a long block from one host, then a quip, many times in a row.

- Let the storyteller answer the newcomer: pick up their guess, their story or their question.
- Make only some newcomer turns jokes. Most are questions, doubts or short reactions.
- Add quick exchanges of two to six words in each turn now and then: "Three feet?" "Three feet."
- Do not start a turn by repeating the number that the other host just said.

### Phrases that sound generated

`podcast.check` warns about these in English and French scripts. Replace each with something
specific. For other languages, it checks only sentence length and the question share of each host.

| Avoid | Write instead |
| --- | --- |
| "Great question!", "That's fascinating!" | A reaction to the fact: "Wait, the whole thing?" |
| "Absolutely.", "Exactly.", "Totally." at the start of many turns | The next piece of content, or a doubt |
| "Let's dive in", "deep dive", "buckle up" | The first fact |
| "delve", "tapestry", "a testament to", "pivotal", "intricate", "game-changer" | A plain word: "look at", "mix", "proof", "key", "complex", "big change" |
| "It's not just X, it's Y", "It wasn't X. It was Y." | Say Y |
| "To sum up..." with a list of every point | One takeaway sentence |
| "Next time on...", "stay tuned" | Nothing, or one concrete reason to come back |

## Script rules

```json
{
  "title": "The Great Wall of China",
  "show": "Long Story Short",
  "language": "en",
  "seed": 2024,
  "hosts": [
    {"id": "maya", "name": "Maya", "voice": "caro", "role": "host"},
    {"id": "daniel", "name": "Daniel", "voice": "stuart", "role": "guest"}
  ],
  "pronunciations": {"Qin": "Chin"},
  "critical_facts": ["Ming dynasty"],
  "sources": ["https://whc.unesco.org/en/list/438"],
  "lines": [
    {"speaker": "daniel", "text": "The postcard Great Wall is younger than you think."},
    {"speaker": "maya", "text": "How much younger? My school book said two thousand years."},
    {"speaker": "daniel", "text": "Most of it is Ming dynasty work, from about six hundred years ago.", "pause": 0.5}
  ]
}
```

- Write for the ear: one idea in each sentence, at most about 20 words, and contractions. The
  check warns above 25 words. Repeat a key word instead of a synonym. Give the source before the quote: "NASA says...".
- Keep each turn under 70 words; most turns need 10 to 50. Vary their length. Alternate the
  speakers. Give no host more than 75% of the words.
- Use few numbers. Round them and give one comparison: "about thirteen thousand miles, half the
  way around the Earth".
- In English, write numbers as words, such as "thirteen sixty-eight". Digits make the engine guess.
- In another language, write a number as digits and give its words in `pronunciations`, such as
  `"1889": "dix-huit cent quatre-vingt-neuf"`. The engine reads the words, and the check accepts a
  transcript with either form.
- Add a pronunciation for a name or an acronym that the engine can misread. The written form
  stays in the transcript; only the engine reads the spoken form. Write the spoken form without
  hyphens: the engine pauses at a hyphen, and the listener hears two words.
- List in `critical_facts` the exact phrases that a listener must hear. Choose the facts that
  carry the point: a full name, a key number, a date. The check refuses a fact that the lines do
  not contain in the same form, so "Caesar's" does not match "Julius Caesar".
- List the `sources` of the factual claims. Do not invent facts that you cannot source.
- `pause` adds seconds of silence before a turn, for example before a new section.
- Change `seed` for a completely new set of takes.

## Repair

| Failure | Action |
| --- | --- |
| A line in `r.unverified` | `print(r)` gives its WER and pace. Read `heard` in its take. Simplify the sentence or add a pronunciation. |
| WER above the gate | Find the lines with the highest `wer` in `r.takes` and fix them as above. |
| WER above the gate, but every take is verified | Compare `r.quality.transcript` with the `heard` text of each take. Fix only a line whose own take is wrong. Else the judge misheard the file; this happens most in French. Report it. |
| A missing critical fact | Say the fact more plainly, or move it to a short sentence of its own. |
| Pace outside the gate | Shorten long turns or add `pause` values. |
| Loudness or peak | Report it; mastering sets them, so this means a broken input file. |

## Compare another producer

`await podcast.evaluate("their.mp3", script)` applies the same gates to audio from any tool. Give
the same script, or the plain reference text, to compare two producers fairly. See
[references/quality.md](references/quality.md) for the metrics and their limits.
