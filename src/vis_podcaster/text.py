"""Text rules that the checks share: normalization, word error rate and fact recall.

The rules make a script and an ASR transcript comparable. They expand digits, ordinals
and symbols into words, so "1987" in a transcript can match "nineteen eighty-seven".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from itertools import pairwise

_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = ((10**9, "billion"), (10**6, "million"), (1000, "thousand"))
_ORDINAL_WORDS = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}
_SYMBOLS = {"%": " percent ", "&": " and ", "+": " plus ", "=": " equals ", "@": " at "}
_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:st|nd|rd|th)?")
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def cardinal(n: int) -> str:
    """Spell a nonnegative integer below one trillion in English words."""
    if n < 0 or n >= 10**12:
        raise ValueError(f"cannot spell {n}")
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + cardinal(rest))
    for size, name in _SCALES:
        if n >= size:
            rest = n % size
            return cardinal(n // size) + " " + name + ("" if rest == 0 else " " + cardinal(rest))
    raise AssertionError(n)


def year(n: int) -> str:
    """Spell a year the way people say it: 1987 is "nineteen eighty seven"."""
    if 1100 <= n <= 9999 and n % 1000 >= 100 and n % 100 != 0:
        return cardinal(n // 100) + " " + (("oh " + _ONES[n % 100]) if n % 100 < 10 else cardinal(n % 100))
    if 1100 <= n <= 9999 and n % 100 == 0 and n % 1000 != 0:
        return cardinal(n // 100) + " hundred"
    return cardinal(n)


def ordinal(n: int) -> str:
    """Spell an ordinal: 3 is "third" and 21 is "twenty first"."""
    words = cardinal(n).split()
    last = words[-1]
    if last in _ORDINAL_WORDS:
        words[-1] = _ORDINAL_WORDS[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def _number_forms(token: str) -> tuple[str, ...]:
    """Return the spoken forms of one numeric token, the most likely first."""
    if token[-2:] in ("st", "nd", "rd", "th"):
        return (ordinal(int(token[:-2])),)
    if "." in token:
        whole, frac = token.split(".")
        return (cardinal(int(whole)) + " point " + " ".join(_ONES[int(d)] for d in frac),)
    n = int(token)
    if n >= 10**12:
        return (" ".join(_ONES[int(d)] for d in token),)
    forms = [cardinal(n)]
    if 1100 <= n <= 2099 and year(n) != forms[0]:
        forms.insert(0, year(n))
    if 2010 <= n <= 2099:
        forms.insert(0, "twenty " + cardinal(n % 100))
    british = re.sub(r"hundred (?!thousand|million|billion)(?=\w)", "hundred and ", cardinal(n))
    if british != forms[-1]:
        forms.append(british)
    return tuple(forms)


def _clean(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    for symbol, word in _SYMBOLS.items():
        text = text.replace(symbol, word)
    return re.sub(r"[-–—/]", " ", text)


def tokens(text: str, *, forms: dict[str, str] | None = None) -> list[str]:
    """Normalize text to lowercase word tokens. Digits become words.

    `forms` picks the spoken form for a numeric token; else the first form applies.
    """
    out: list[str] = []
    for raw in _TOKEN.findall(_NUMBER.sub(lambda m: f" {m.group(0)} ", _clean(text))):
        if _NUMBER.fullmatch(raw):
            choice = (forms or {}).get(raw) or _number_forms(raw)[0]
            out.extend(choice.split())
        else:
            out.append(raw)
    return out


def _distance(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) of the best word alignment."""
    prev = [(j, 0, 0, j) for j in range(len(hyp) + 1)]  # (cost, subs, dels, ins)
    for i, r in enumerate(ref, 1):
        cur = [(i, 0, i, 0)]
        for j, h in enumerate(hyp, 1):
            diag = prev[j - 1]
            sub = (diag[0] + (r != h), diag[1] + (r != h), diag[2], diag[3])
            dele = (prev[j][0] + 1, prev[j][1], prev[j][2] + 1, prev[j][3])
            ins = (cur[j - 1][0] + 1, cur[j - 1][1], cur[j - 1][2], cur[j - 1][3] + 1)
            cur.append(min(sub, dele, ins))
        prev = cur
    _, s, d, n = prev[-1]
    return s, d, n


def _merge_compounds(ref: list[str], hyp: list[str]) -> list[str]:
    """Join or split hypothesis words where only spacing differs: "watch towers" and "watchtowers"."""
    vocab = set(ref)
    pairs = {a + b: (a, b) for a, b in pairwise(ref)}
    out, i = [], 0
    while i < len(hyp):
        word = hyp[i]
        if i + 1 < len(hyp) and word not in vocab and word + hyp[i + 1] in vocab:
            out.append(word + hyp[i + 1])
            i += 2
            continue
        if word not in vocab and word in pairs:
            out.extend(pairs[word])
        else:
            out.append(word)
        i += 1
    return out


@dataclass(frozen=True)
class WordErrors:
    """Word error rate of a hypothesis against a reference."""

    wer: float
    reference_words: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def proper_names(text: str) -> frozenset[str]:
    """Return the normalized words that start with a capital letter inside a sentence."""
    found = re.findall(r"(?<![.!?]\s)(?<!^)(?<=\s)([A-Z][a-z]{3,})", text)
    return frozenset(w.lower() for w in found)


def word_errors(reference: str, hypothesis: str, *, names: frozenset[str] = frozenset()) -> WordErrors:
    """Compare two texts after normalization. Numbers in the hypothesis take their best spoken form.

    WER is (substitutions + deletions + insertions) / reference words. A heard word that is a
    spelling variant of one of `names`, such as "Shanghai" for "Shanhai", counts as that name.
    """
    ref = tokens(reference)
    chosen: dict[str, str] = {}
    numbers = dict.fromkeys(m.group(0) for m in _NUMBER.finditer(_clean(hypothesis)))
    for token in numbers:
        options = _number_forms(token)
        if len(options) > 1:
            chosen[token] = min(
                options, key=lambda o: sum(_distance(ref, tokens(hypothesis, forms={**chosen, token: o})))
            )
    hyp = _merge_compounds(ref, tokens(hypothesis, forms=chosen))
    if names:
        hyp = [_respell(tuple(names), [w])[0] if w not in ref else w for w in hyp]
    s, d, n = _distance(ref, hyp)
    return WordErrors((s + d + n) / max(len(ref), 1), len(ref), s, d, n)


@lru_cache(maxsize=256)
def _fact_tokens(fact: str) -> tuple[str, ...]:
    return tuple(tokens(fact))


def _respell(want: tuple[str, ...], heard: list[str]) -> list[str]:
    """Replace heard words that are spelling variants of a wanted word, such as "shanghai" for "shanhai"."""
    out = []
    for word in heard:
        near = max(want, key=lambda w: SequenceMatcher(None, w, word).ratio())
        out.append(near if len(near) > 3 and SequenceMatcher(None, near, word).ratio() >= 0.8 else word)
    return out


def fact_heard(fact: str, transcript_tokens: list[str], *, tolerance: float = 0.2, fuzzy: bool = False) -> bool:
    """Tell whether a transcript contains a fact phrase, with at most `tolerance` word errors.

    `fuzzy` accepts spelling variants of names longer than three letters.
    """
    want = list(_fact_tokens(fact))
    if fuzzy and want:
        transcript_tokens = _respell(tuple(want), transcript_tokens)
    if not want:
        return True
    width = len(want)
    budget = int(width * tolerance)
    for size in range(max(1, width - budget), width + budget + 1):
        for start in range(max(1, len(transcript_tokens) - size + 1)):
            if sum(_distance(want, transcript_tokens[start : start + size])) <= budget:
                return True
    return False


def word_count(text: str) -> int:
    """Count spoken words after normalization."""
    return len(tokens(text))
