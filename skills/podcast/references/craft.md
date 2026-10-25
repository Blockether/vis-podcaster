# Podcast craft

These rules come from radio and podcast training material. Each rule names its sources. `check`
marks a rule that `podcast.check` tests; the other rules need your judgement.

## Structure

| Rule | Sources | Check |
| --- | --- | --- |
| Give the episode one point that fits in one sentence. Write the intro and the last line first. | [S3], [S4] | |
| Open with a hook: tease a revelation or pose the central question. Reveal it step by step; do not put the answer first. | [S1], [S3] | `check`: greeting first |
| After the hook, name the show, the hosts and what the listener gets. Keep it short. | [S10] | |
| Make one moment say why the topic matters to the listener. | [S3] | |
| Mark every change of topic with a sentence. Listeners cannot scan back. | [S1], [S3] | |
| Drive each segment with a question. Stories need people who want or see different things. | [S3], [S7] | |
| Tell an anecdote, then reflect on what it means. Alternate the two. | [S7], [S15] | |
| Make a point once and move on. Repeat only the one or two key ideas. | [S3], [S12] | |
| End on the takeaway. Do not summarize the summary, announce the takeaway or tease "next time on". | [S4], [S11] | `check`: recap at the end |
| Plan about 150 words a minute for a human host; above about 160 a human sounds rushed. `qwen3-tts-mlx` speaks about 170 and still sounds clear, so the render gate accepts 120 to 190. | [S10] | render gate: pace |

## Two hosts

| Rule | Sources | Check |
| --- | --- | --- |
| Give the hosts different jobs. The host who does not know the story carries reactions, guesses, opinions and their own experience, not facts of the same story. A host who only asks "what?" and "why?" is not needed, and a host who narrates in turns is not a conversation. | [S4], [S7] | `check`: question share |
| The listener's host predicts what the listener does not understand. | [S4] | |
| React to what the other host actually said. Follow it up. | [S4] | `check`: bare agreement |
| Let a host say "I don't know", then add what they do know. | [S4] | |
| Give each host at least one objection with a reason. Constant agreement is dead air. | [S7] | |
| Vary the length of turns. Equal blocks sound robotic. | [S3] | |
| Choose two clearly different voices. | [S3] | `check`: shared voice |
| Use hedges and fillers rarely. A few make dialogue human; many sound fake in synthetic speech. | [S5], [S9] | |

## Writing for the ear

| Rule | Sources | Check |
| --- | --- | --- |
| Put one fact or idea in a sentence. Keep it to about 20 words. | [S1], [S5], [S6] | `check`: sentence above 25 words |
| If you would not say it that way, do not write it that way. Use contractions and plain words. | [S1], [S2], [S5] | |
| Give the source before the quote. | [S1] | |
| Use few numbers. Round them and compare them with something familiar. | [S1], [S13] | `check`: digits |
| Repeat a key word instead of a synonym. | [S2] | |
| Use concrete, visual examples. Show first, then explain. | [S3], [S4] | |
| Do not let one host say what the other is about to say. | [S3], [S12] | |
| Listen to a draft render before you finish. | [S3], [S5] | |

## Phrases that sound generated

Avoid "delve", "deep dive", "tapestry", "a testament to", "pivotal", "intricate", "interplay",
"meticulous" and "fostering" [S8]; "deep dive" is the template phrase of NotebookLM dialogues
[S9]. Avoid "it's not just X, it's Y", reflexive lists of three and claims about legacy [S8]. Do
not use the same structure for every topic [S14]. Turns that open with praise or agreement
("Great question", "Absolutely", "Exactly") are a common sign of generated dialogue; this is
an inference from [S4] and [S8], not a quoted rule. `podcast.check` looks for these phrases in
English and French scripts.

## Facts

| Rule | Sources |
| --- | --- |
| Treat every number, name and date as a claim. Check each against a primary source. | [S6b] |
| Read the script as if everything is wrong: check logic and context, not only figures. | [S6b] |
| Check a surprising or technical claim against a paper or an expert, not one secondary page. | [S6b] |
| Say the source in the episode, before the claim. Hedge a claim that is uncertain. | [S1] |
| Do not misrepresent the view of a named person. | [S6b] |
| Check the order of events, direction words and the arithmetic between lines. Quality gates do not find these errors. | [quality.md](quality.md#limits) |

## Sources

- [S1] NPR Training, print to radio: https://www.npr.org/sections/npr-training/2025/05/30/g-s1-65814/the-journey-from-print-to-radio-storytelling-a-guide-for-navigating-a-new-landscape
- [S2] NPR Training, "Would you say it that way?": https://www.npr.org/sections/npr-training/2025/05/31/g-s1-66074/would-you-say-it-that-way-tips-on-writing-for-your-voice
- [S3] NPR Training, campfire tales: https://www.npr.org/sections/npr-training/2025/05/31/g-s1-65875/campfire-tales-the-essentials-of-writing-for-radio
- [S4] NPR Training, reporter two-ways: https://www.npr.org/sections/npr-training/2025/05/28/g-s1-67598/reporter-two-ways-how-to-sound-natural-even-with-a-script
- [S5] NPR, seven tips for a podcast script: https://www.npr.org/2023/02/17/1156597697/how-to-write-script-speech-podcast-voice
- [S6] Hearing Voices, writing for the ear: https://hearingvoices.com/tow/narration-writing-for-the-ear/
- [S6b] Transom, fact-checking This American Life: https://transom.org/2017/fact-check-fact-wreck/
- [S7] Ira Glass, Transom review: https://transom.org/2004/ira-glass/
- [S8] Wikipedia, signs of AI writing: https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing
- [S9] Simon Willison on NotebookLM audio: https://simonwillison.net/2024/Sep/29/notebooklm-audio-overview/
- [S10] Descript, podcast scripts: https://www.descript.com/blog/article/podcast-script
- [S11] Transom HowSound, writing part 2: https://transom.org/2025/writing-part-2/
- [S12] Transom HowSound, "We do it for the ears": https://transom.org/2024/we-do-it-for-the-ears-right/
- [S13] BBC News style guide, numbers: https://www.bbc.co.uk/newsstyleguide/numbers
- [S14] NotebookLM template study: https://arxiv.org/abs/2511.08654
- [S15] Anecdote and reflection: https://www.natelistrom.com/2022/12/07/anecdote-reflection.html
