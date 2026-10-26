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

## Review an episode

Score the script before you render it, against these ten criteria. Give one point for each
criterion with no problem. An episode is ready at 10 of 10. For each missing point, name the
line and write the replacement.

| # | Criterion | A point needs |
| --- | --- | --- |
| 1 | Hook | The first two or three lines raise a question or a scene. The answer comes later. |
| 2 | Roles | Each fact of the story belongs to one host. The other host never continues the narration. |
| 3 | Listening | Each turn answers the turn before it: a guess, a joke, a story or a question. No question goes unanswered. |
| 4 | Rhythm | Turn lengths vary. Quick exchanges of two to six words occur. No run of more than four long-block-then-quip pairs. |
| 5 | Two people | Each host has their own attitude, words and running thread. You cannot swap their lines. |
| 6 | Spoken words | Every line sounds said, not written: short sentences, contractions, no lists of facts, no stock phrases, no forced jokes. |
| 7 | Tension | A real disagreement, or a wrong belief that evidence corrects. The episode says why the topic matters. |
| 8 | Facts | Every claim is sourced and correct, with no myth, anachronism or contradiction between lines. |
| 9 | Shape | One point. Each change of segment is marked. Nothing is said twice without a reason. |
| 10 | Ending | One image or line that closes the loop to the hook. No recap, no announced takeaway, no flat sign-off. |

To reach 10 of 10:

- Take a point only for a problem that a listener would notice, or for a broken rule. Do not take a point for taste.
- Change only the lines that the review names. A full rewrite brings new problems in each round.
- After each round, run `podcast.check` again. Then give the script to a new reviewer.

## Faults that reviews find

Independent reviews of three benchmark episodes took points for these faults. Look for each one
before you give the script to a reviewer.

| Criterion | Fault | Fix |
| --- | --- | --- |
| Roles | The newcomer knows the story: "The pump handle is the famous bit." | A reaction to what was just said: "Wait. So taking the handle off did nothing?" |
| Roles | The host names a person or a fact before the guest does. | Ask an open question: "Who threw away ten days, and why?" |
| Listening | The next turn ignores a guess, a joke or a question. | Answer it first, then go on. |
| Listening | One turn asks two questions, and the answer fits either. | Ask one question in a turn. |
| Listening | A comparison gives only one of its numbers. | Say both numbers. |
| Rhythm | Many long blocks in a row, each followed by a quip, with no quick exchange. | Add exchanges of two to six words, an interruption or a story from the other host. |
| Spoken words | One host starts many turns with "So" and repeats the other's line. | A reaction, a guess or a new question. |
| Spoken words | A host opens by repeating the other's last words: "Like politicians." | Answer the words, or go on. |
| Spoken words | Praise or a teacher's line: "Now you've got it." | The next fact. |
| Spoken words | A staged set-up: one host stops so that the other can finish the sentence. | Each host says their own lines. |
| Spoken words | A stock joke: "Roll the credits", "I'm telling my doctor". | A joke that only this story can give. |
| Spoken words | A colon definition: "The idea was miasma: bad air." | Two short sentences: "The big idea was miasma. Bad air." |
| Spoken words | A public notice: "The World Health Organization's estimate is…" | Say it as a person: "And cholera's still around. More than a million cases a year." |
| Spoken words | A list of noun fragments, or two numbers in one sentence. | Full sentences, with one number in each. |
| Spoken words | A pronoun that points to the wrong noun by ear. | Repeat the noun. |
| Spoken words | "Not X, it's Y": "On ne prolonge pas une antenne, on prolonge une caisse." | Say Y. |
| Facts | An absolute word that the source does not support: "Nobody knew about germs." | Narrow it: "Germs were only a theory then." |
| Facts | A bet or a score with the wrong verdict. | Change the bet so that the verdict agrees with the source. |
| Facts | "Then" or "for years" puts an event at the wrong time. | Give the date or the order that the source gives. |
| Facts | A counterfactual or "nobody can prove it" that no source makes. | Remove it, or say what the host cannot prove: "I can't prove it. Neither can you." |
| Shape | A segment change that does not name the new time or place. | "Now, fifteen eighty-two." |
| Shape | A jump to today and back without a marker. | Mark both moves, or put today in its own segment. |
| Ending | The conclusion comes two or three times, from both hosts. | One line or image, from one host. |
| Ending | The last line changes the words of the hook: "The pump came to her." | Repeat the hook exactly: "The outbreak came to her." |
| Ending | An image that does not explain itself. | Say why the image matters in the same line. |

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
