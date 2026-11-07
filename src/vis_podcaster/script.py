"""The episode script: hosts, lines, pronunciations and the facts that listeners must hear.

This module uses only the standard library. `check_script` reports every problem at once;
`parse_episode` raises `ScriptError` when a problem is an error.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from vis_podcaster.text import fact_heard, word_count

MAX_HOSTS = 4
MAX_LINE_WORDS = 70
WORDS_PER_MINUTE = 170
_ACRONYM = re.compile(r"\b[A-Z]{2,}s?\b")
_DIGITS = re.compile(r"\d")
MAX_SENTENCE_WORDS = 25
_SENTENCE = re.compile(r"[^.?!]+[.?!]*")
# Stock phrases that make a dialogue sound generated. Each pattern names the phrase it finds.
SLOP = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(great|good|excellent) question\b",
        r"\b(that's|that is|it's) (so |really |absolutely )?(fascinating|mind-blowing|incredible)\b",
        r"\bmind-blowing\b",
        r"\b(let's|let us) (dive|jump) (in|into|right in)\b",
        r"\bdeep dive\b",
        r"\bdelv(e|es|ing)\b",
        r"\bbuckle up\b",
        r"\bgame[- ]changer\b",
        r"\btapestry\b",
        r"\ba testament to\b",
        r"\bat the end of the day\b",
        r"\bin today's (fast-paced )?world\b",
        r"\b(pivotal|intricate|interplay|meticulous(ly)?|fostering)\b",
        r"\bnot (just|only) [^.?!,]+, (but|it's)\b",
        r"\b(isn't|is not|wasn't|was not|it's not) (just |really )?[^.?!,]+[.,] (it's|it is|it was)\b",
        r"\bnext time on\b",
        r"\bstay tuned\b",
    )
)
_SLOP_FR = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(bonne|excellente|super) question\b",
        r"\bc'est (vraiment |tellement |absolument )?(fascinant|incroyable|passionnant)\b",
        r"\bplong(eons|er) (dans|au c(oe|œ)ur)\b",
        r"\battachez vos ceintures\b",
        r"\bau final\b",
        r"\bpas (seulement|juste|simplement) [^.?!,]+, (mais|c'est)\b",
        r"\bce n'est pas [^.?!,]+[.,] c'est\b",
        r"\b(restez à l'écoute|à la prochaine)\b",
    )
)


@dataclass(frozen=True)
class _Style:
    slop: tuple[re.Pattern[str], ...]
    agreement: re.Pattern[str]
    greeting: re.Pattern[str]
    recap: re.Pattern[str]


# Craft rules for each language. A language without an entry gets only the checks that need no words.
_STYLES = {
    "en": _Style(
        slop=SLOP,
        # A turn that starts with one of these only agrees. A few are natural; many sound like an echo.
        agreement=re.compile(
            r"^(absolutely|exactly|totally|definitely|right|yeah|yes|oh,? wow|wow|true|precisely|indeed)"
            r"( that| so)?\b[,.!?]",
            re.IGNORECASE,
        ),
        greeting=re.compile(r"^(welcome|hello|hi|hey)\b", re.IGNORECASE),
        recap=re.compile(
            r"\b(to sum (it )?up|in summary|to recap|let's recap|the key takeaway|"
            r"(here's|so) what (i'm|i am) taking (home|away)|my takeaway)\b",
            re.IGNORECASE,
        ),
    ),
    "fr": _Style(
        slop=_SLOP_FR,
        agreement=re.compile(
            r"^(exactement|absolument|tout à fait|c'est vrai|c'est ça|oui|ouais|effectivement|en effet|"
            r"carrément|voilà|ah oui|waouh|wow)\b[,.!?]",
            re.IGNORECASE,
        ),
        greeting=re.compile(r"^(bienvenue|bonjour|salut|bonsoir)\b", re.IGNORECASE),
        recap=re.compile(
            r"\b(en résumé|pour résumer|en somme|pour récapituler|ce qu'il faut retenir|"
            r"(voilà )?ce que je retiens)\b",
            re.IGNORECASE,
        ),
    ),
}


class ScriptError(ValueError):
    """The script has errors. `issues` lists them all."""

    def __init__(self, issues: tuple[Issue, ...]):
        self.issues = issues
        super().__init__("; ".join(f"{i.where}: {i.message}" for i in issues if i.severity == "error"))


@dataclass(frozen=True)
class Issue:
    """One problem in a script. An error stops rendering; a warning does not."""

    severity: Literal["error", "warning"]
    where: str
    message: str


@dataclass(frozen=True)
class Host:
    """One voice of the show. `engine` None means the engine that owns `voice`."""

    id: str
    name: str
    voice: str
    engine: str | None = None
    role: str = ""


@dataclass(frozen=True)
class Line:
    """One speaker turn. `pause` is extra silence in seconds before the turn."""

    speaker: str
    text: str
    pause: float = 0.0


@dataclass(frozen=True)
class Episode:
    """A complete episode script."""

    title: str
    hosts: tuple[Host, ...]
    lines: tuple[Line, ...]
    show: str = ""
    description: str = ""
    language: str = "en"
    pronunciations: dict[str, str] = field(default_factory=dict)
    critical_facts: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    seed: int = 1000

    def host(self, host_id: str) -> Host:
        return next(h for h in self.hosts if h.id == host_id)

    def spoken(self, text: str) -> str:
        """Return `text` with the pronunciations applied, as the engine must read it."""
        for written, said in sorted(self.pronunciations.items(), key=lambda kv: -len(kv[0])):
            text = re.sub(rf"(?<!\w){re.escape(written)}(?!\w)", said, text)
        return text

    @property
    def text(self) -> str:
        """The written words of every line, in order."""
        return "\n".join(line.text for line in self.lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "show": self.show,
            "description": self.description,
            "language": self.language,
            "seed": self.seed,
            "hosts": [{k: v for k, v in vars(h).items() if v not in (None, "")} for h in self.hosts],
            "lines": [{k: v for k, v in vars(ln).items() if not (k == "pause" and not v)} for ln in self.lines],
            "pronunciations": dict(self.pronunciations),
            "critical_facts": list(self.critical_facts),
            "sources": list(self.sources),
        }


def _text(value: Any) -> str | None:
    return " ".join(value.split()) if isinstance(value, str) and value.strip() else None


def _strings(data: dict, key: str, issues: list[Issue]) -> tuple[str, ...]:
    raw = data.get(key, [])
    if not isinstance(raw, list) or not all(_text(v) for v in raw):
        issues.append(Issue("error", key, "must be a list of nonblank strings"))
        return ()
    return tuple(_text(v) for v in raw)


def _hosts(data: dict, issues: list[Issue]) -> list[Host]:
    raw_hosts = data.get("hosts")
    if not isinstance(raw_hosts, list) or not raw_hosts:
        issues.append(Issue("error", "hosts", "must be a nonempty list"))
        return []
    if len(raw_hosts) > MAX_HOSTS:
        issues.append(Issue("error", "hosts", f"has {len(raw_hosts)} entries; the limit is {MAX_HOSTS}"))
    hosts = []
    for i, raw in enumerate(raw_hosts):
        where = f"hosts[{i}]"
        if not isinstance(raw, dict):
            issues.append(Issue("error", where, "must be an object"))
            continue
        hid, voice = _text(raw.get("id")), _text(raw.get("voice"))
        if not hid or not voice:
            issues.append(Issue("error", where, "needs a nonblank id and voice"))
            continue
        hosts.append(
            Host(hid, _text(raw.get("name")) or hid, voice, _text(raw.get("engine")), _text(raw.get("role")) or "")
        )
    ids = [h.id for h in hosts]
    if len(set(ids)) != len(ids):
        issues.append(Issue("error", "hosts", "host ids must be unique"))
    if len({(h.engine, h.voice.lower()) for h in hosts}) != len(hosts):
        issues.append(Issue("warning", "hosts", "two hosts share one voice; listeners cannot tell them apart"))
    return hosts


def _lines(data: dict, ids: set[str], issues: list[Issue]) -> list[Line]:
    raw_lines = data.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        issues.append(Issue("error", "lines", "must be a nonempty list"))
        return []
    lines = []
    for i, raw in enumerate(raw_lines):
        where = f"lines[{i}]"
        if not isinstance(raw, dict):
            issues.append(Issue("error", where, "must be an object"))
            continue
        speaker, text = _text(raw.get("speaker")), _text(raw.get("text"))
        if not speaker or speaker not in ids:
            issues.append(Issue("error", where, f"speaker {speaker!r} is not a host id ({', '.join(sorted(ids))})"))
            continue
        if not text:
            issues.append(Issue("error", where, "text must be nonblank"))
            continue
        pause = raw.get("pause", 0.0)
        if not isinstance(pause, int | float) or isinstance(pause, bool) or not 0 <= pause <= 5:
            issues.append(Issue("error", where, "pause must be a number of seconds from 0 to 5"))
            pause = 0.0
        lines.append(Line(speaker, text, float(pause)))
    return lines


def _review(ep: Episode, issues: list[Issue]) -> None:
    """Add the warnings and errors that need the whole episode."""
    spoken_counts = {h.id: 0 for h in ep.hosts}
    known = {w.upper() for w in ep.pronunciations}
    for i, line in enumerate(ep.lines):
        where = f"lines[{i}]"
        words = word_count(ep.spoken(line.text))
        spoken_counts[line.speaker] += words
        if words > MAX_LINE_WORDS:
            issues.append(Issue("error", where, f"has {words} words; split turns above {MAX_LINE_WORDS} words"))
        if _DIGITS.search(ep.spoken(line.text)):
            issues.append(Issue("warning", where, "contains digits; write numbers as words or add a pronunciation"))
        for acronym in sorted(set(_ACRONYM.findall(line.text)) - known):
            issues.append(
                Issue("warning", where, f"{acronym!r} has no pronunciation; the engine may spell it or guess")
            )
        if i and ep.lines[i - 1].speaker == line.speaker and not line.pause:
            issues.append(Issue("warning", where, "continues the same speaker; merge the turns or add a pause"))
    total = sum(spoken_counts.values()) or 1
    for hid, count in spoken_counts.items():
        if count == 0:
            issues.append(Issue("warning", "hosts", f"host {hid!r} never speaks"))
        elif len(ep.hosts) > 1 and count / total > 0.75:
            issues.append(Issue("warning", "hosts", f"host {hid!r} speaks {count / total:.0%} of the words"))
    for i, fact in enumerate(ep.critical_facts):
        if not fact_heard(fact, ep.text, tolerance=0.0, language=ep.language):
            issues.append(Issue("error", f"critical_facts[{i}]", f"{fact!r} does not occur in the lines"))
    if not ep.sources:
        issues.append(Issue("warning", "sources", "no sources; factual claims are not traceable"))
    _review_craft(ep, issues)


def _review_craft(ep: Episode, issues: list[Issue]) -> None:
    """Add warnings for writing that sounds generated or is hard to follow by ear."""
    style = _STYLES.get(ep.language.lower().split("-")[0])
    for i, line in enumerate(ep.lines):
        where = f"lines[{i}]"
        for sentence in _SENTENCE.findall(line.text):
            if word_count(sentence) > MAX_SENTENCE_WORDS:
                issues.append(Issue("warning", where, f"has a sentence above {MAX_SENTENCE_WORDS} words; split it"))
                break
        text = line.text.replace("’", "'")
        for pattern in style.slop if style else ():
            if m := pattern.search(text):
                issues.append(Issue("warning", where, f"stock phrase {m.group(0)!r}; say something specific"))
    if style and ep.lines:
        texts = [line.text.replace("’", "'").strip() for line in ep.lines]
        if style.greeting.match(texts[0]):
            issues.append(Issue("warning", "lines[0]", "opens with a greeting; open with the hook, then name the show"))
        for i in range(max(0, len(ep.lines) - 3), len(ep.lines)):
            if style.recap.search(texts[i]):
                issues.append(Issue("warning", f"lines[{i}]", "ends with a recap; end on one takeaway or image"))
        agreeing = [i for i, text in enumerate(texts) if style.agreement.match(text)]
        if len(agreeing) > max(2, len(ep.lines) // 8):
            issues.append(
                Issue("warning", "lines", f"{len(agreeing)} turns start with bare agreement; react to the content")
            )
    if len(ep.hosts) > 1:
        for host in ep.hosts:
            turns = [line.text.strip() for line in ep.lines if line.speaker == host.id]
            asks = sum(t.endswith("?") for t in turns)
            if len(turns) >= 4 and asks / len(turns) > 0.6:
                issues.append(
                    Issue(
                        "warning",
                        "hosts",
                        f"host {host.id!r} ends {asks} of {len(turns)} turns with a question; "
                        "give them reactions, guesses, opinions and pushback too",
                    )
                )


def check_script(data: dict[str, Any] | Episode) -> tuple[Episode | None, tuple[Issue, ...]]:
    """Validate a script mapping. Return the episode, or None when errors stop parsing, and every issue."""
    if isinstance(data, Episode):
        data = data.to_dict()
    issues: list[Issue] = []
    if not isinstance(data, dict):
        return None, (Issue("error", "script", "must be a JSON object"),)
    title = _text(data.get("title"))
    if not title:
        issues.append(Issue("error", "title", "must be a nonblank string"))
    hosts = _hosts(data, issues)
    lines = _lines(data, {h.id for h in hosts}, issues)
    prons = data.get("pronunciations", {})
    if not isinstance(prons, dict) or not all(_text(k) and _text(v) for k, v in prons.items()):
        issues.append(Issue("error", "pronunciations", "must map nonblank written forms to nonblank spoken forms"))
        prons = {}
    seed = data.get("seed", 1000)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**31 - 1:
        issues.append(Issue("error", "seed", "must be an integer from 0 to 2147483647"))
        seed = 1000
    language = _text(data.get("language")) or "en"
    facts, sources = _strings(data, "critical_facts", issues), _strings(data, "sources", issues)
    if any(i.severity == "error" for i in issues):
        return None, tuple(issues)
    ep = Episode(
        title=title,
        hosts=tuple(hosts),
        lines=tuple(lines),
        show=_text(data.get("show")) or "",
        description=_text(data.get("description")) or "",
        language=language,
        pronunciations={_text(k): _text(v) for k, v in prons.items()},
        critical_facts=facts,
        sources=sources,
        seed=seed,
    )
    _review(ep, issues)
    if any(i.severity == "error" for i in issues):
        return None, tuple(issues)
    return ep, tuple(issues)


def parse_episode(data: dict[str, Any] | Episode) -> Episode:
    """Validate a script mapping and return the episode. Raise ScriptError that lists every error."""
    episode, issues = check_script(data)
    if episode is None:
        raise ScriptError(tuple(i for i in issues if i.severity == "error"))
    return episode


def load_episode(script: str | Path | dict[str, Any] | Episode) -> Episode:
    """Accept an Episode, a mapping, a JSON file path or JSON text, and return the validated episode."""
    if isinstance(script, Episode | dict):
        return parse_episode(script)
    text = str(script)
    if not text.lstrip().startswith("{"):
        text = Path(text).expanduser().read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ScriptError((Issue("error", "script", f"invalid JSON at line {e.lineno}: {e.msg}"),)) from e
    return parse_episode(data)


def estimated_minutes(episode: Episode) -> float:
    """Estimate the running time from the spoken words at a conversational pace."""
    return word_count(episode.spoken(episode.text)) / WORDS_PER_MINUTE
