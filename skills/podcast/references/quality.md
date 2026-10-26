# Quality metrics

`podcast.evaluate` and `podcast.render` measure the finished file. A gate fails when its value is
outside the default range in `Gates`.

| Metric | Meaning | Default gate |
| --- | --- | --- |
| WER | Word error rate of the judge transcript against the script: (substitutions + deletions + insertions) / reference words | at most 5% |
| Fact recall | Share of `critical_facts` that the judge transcript contains, with at most 20% word errors and name spelling variants | 100% |
| Loudness | Integrated loudness, EBU R128 | -16 ± 1 LUFS |
| True peak | Highest inter-sample peak | at most -1.0 dBTP |
| Pace | Script words per minute of audio; an elision such as "l'exposition" is one word | 120 to 190 |
| Longest pause | Longest inner silence below -40 dBFS | at most 2.0 s |
| Clipped samples | Samples at full scale | 0 |

## Measurement rules

- The judge is `vibevoice-asr-local`. It transcribes the whole file in parts of at most one minute,
  cut in pauses, because its cost grows faster than the length.
- The per-line check during rendering uses `parakeet-local`. If its WER is above the gate, the judge
  hears the same take too, and the better result counts. A take is retaken only when both mishear it.
- Normalization makes the texts comparable: lowercase, no punctuation, no accents, no elision such
  as "l'", English digits as words, years in both spoken forms, and joined or split compounds such
  as "watchtowers".
- WER takes the better of the written script and the script with pronunciations applied.
- WER measures intelligibility, not naturalness. Listen to the episode before you publish it.

## Limits

- No gate checks facts. A wrong date, a reversed direction or a myth passes every gate. Check
  the facts in the script before you render.
- Recognizers misspell rare names. Such errors raise WER although the audio can be correct.
- `parakeet-local` can hear the first words of a line in another language as English, such as
  "Exactly" for "Exactement". The judge check of a failed take removes most of these errors.
- Digits become words in English and French. In another language, a transcript with "1889" matches
  only a script line that has the digits too. See the number rule in the skill.
- In French, endings that sound the same count as one word: "sauvée" and "sauver", "il reproche"
  and "ils reprochent". Other homophones, such as "cent" and "sans", still count as errors.
- `parakeet-local` can drop a sentence of audio longer than about ten seconds that does not start
  at a sentence. Use it for single lines only.
- The judge `vibevoice-asr-local` gets no language hint. On French it can measure 7 to 8% WER on
  audio that `parakeet-local` hears at 2%, line by line. Shorter judge parts do not help. Compare
  the take transcripts in `manifest.json` before you change a line.

## Criteria for a better producer

Compare two producers with the same script, the same judge and the same gates. A producer is
better when it is at least as good on every criterion and better on one.

| Criterion | Metric |
| --- | --- |
| Intelligibility | Judge WER of the whole file |
| Content | Fact recall of `critical_facts` |
| Delivery standard | Loudness, true peak, clipped samples |
| Dialogue | One distinct voice for each host |
| Self-repair | Lines verified and retaken without a person |
| Reproducibility | Same script and seed give the same takes |
| Iteration cost | Seconds to render again after a change of one line |
| Setup | Models and packages to install outside Vis |

## Result: Great Wall episode

Both producers spoke the same 31 lines of an earlier `examples/great-wall.json`. The judge and gates were the
same. The comparison producer was the single-narrator Qwen3-TTS skill (voice Aiden, -19 LUFS
target). vis-podcaster rendered the episode twice: with `vibevoice-local` (emma, carter) and with
`qwen3-tts-mlx` (serena, aiden).

| Criterion | vis-podcaster, Qwen3-TTS | vis-podcaster, VibeVoice | Qwen3-TTS skill |
| --- | --- | --- | --- |
| Judge WER | 1.8% | 2.0% | 1.4% |
| Fact recall | 7/7 | 7/7 | 7/7 |
| Loudness, true peak | -16.0 LUFS, -2.0 dBTP | -16.1 LUFS, -1.8 dBTP | -19.5 LUFS, -1.9 dBTP |
| Noise floor in lines (quietest 5%) | -70 dB | -57 dB (emma), -64 dB (carter) | not measured |
| Hosts | Two voices (serena, aiden) | Two voices (emma, carter) | One narrator for both hosts |
| Self-repair | Per-line check, 3 automatic retakes | Per-line check, 4 automatic retakes | Manual retakes, WER only at the end |
| Render time | 7.9 min with verification and judge | 6.7 min with verification and judge | 3.7 min speech only |
| Iteration | Cached takes; a policy change keeps them | Cached takes; a policy change keeps them | Manual retake of each chunk |
| Setup | MLX environment with `mlx-audio`, a 1.7B model | None beyond Vis | MLX, a 1.7B model download, Python 3.11 |

The emma preset has a constant room noise. Mastering adds about 12 dB of gain, so the noise is
audible. Use `qwen3-tts-mlx` voices when the MLX environment is available.

Most errors of both producers are Chinese place names, such as Shanhai Pass and Badaling. A
stricter per-line limit (3% instead of 8%) did not lower the judge WER, so the default stays 8%.
