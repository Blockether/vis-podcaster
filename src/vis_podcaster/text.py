"""Text rules that the checks share: normalization, word error rate and fact recall.

The rules make a script and an ASR transcript comparable. They expand digits, ordinals
and symbols into words, so "1987" in a transcript can match "nineteen eighty-seven". French
text gets French number words and one form for endings that sound the same.
"""

from __future__ import annotations

import re
import unicodedata
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
# French and Italian elision: "l'exposition" is the two words "l" and "exposition".
_ELISION = re.compile(
    r"\b(l|d|j|n|s|c|m|t|qu|jusqu|lorsqu|puisqu|quoiqu|dell|all|nell|sull|dall|un|quest|quell)'(?=[a-z])"
)
# Spelling variants get one form. British spellings become American, so "travelled" matches "traveled".
# French plural number words lose their "s", so "trois-cent" in a transcript matches "trois cents".
_SPELLINGS = (
    (re.compile(r"^scept"), "skept"),
    (re.compile(r"(?<=[ae])ll(?=ed$|ing$|ers?$)"), "l"),
    (re.compile(r"(?<=\w{3})our(?=$|s$|ed$|ing$|ite$|ites$|able$)"), "or"),
    (re.compile(r"(?<=\w{3})is(?=e$|es$|ed$|ing$|ation$|ations$)"), "iz"),
    (re.compile(r"^(cent|met|kilomet|millimet|centimet|theat|fib|lit|spect|somb|lust|meag)re(?=s?$)"), r"\1er"),
    (re.compile(r"(?<=\w{3})ogue$"), "og"),
    (re.compile(r"^(cent|vingt)s$"), r"\1"),
)
# Hesitations that a voice can add. A heard one that the script does not have is not an error.
FILLERS = frozenset({"uh", "um", "uhm", "erm", "hmm", "mm", "euh", "ehm"})
_LIGATURES = {"œ": "oe", "æ": "ae", "ß": "ss", "ø": "o", "ł": "l", "đ": "d"}


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


_FR_ONES = ("zero", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "onze", "douze")
_FR_ONES += ("treize", "quatorze", "quinze", "seize")
_FR_TENS = ("", "dix", "vingt", "trente", "quarante", "cinquante", "soixante")
_FR_SCALES = ((10**9, "milliard"), (10**6, "million"), (1000, "mille"))


def cardinal_fr(n: int) -> str:
    """Spell a nonnegative integer below one trillion in French words, with spaces for hyphens."""
    if n < 0 or n >= 10**12:
        raise ValueError(f"cannot spell {n}")
    if n < 17:
        return _FR_ONES[n]
    if n < 20:
        return "dix " + _FR_ONES[n - 10]
    if n < 100:
        tens, unit = divmod(n, 10)
        if tens in (7, 9):
            # 70 is "soixante-dix" and 90 is "quatre-vingt-dix": the tens take 10 to 19.
            head, unit = ("soixante", unit + 10) if tens == 7 else ("quatre vingt", unit + 10)
        else:
            head = "quatre vingt" if tens == 8 else _FR_TENS[tens]
        if unit == 0:
            return head
        joiner = " et " if unit in (1, 11) and tens not in (8, 9) else " "
        return head + joiner + cardinal_fr(unit)
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = "cent" if hundreds == 1 else cardinal_fr(hundreds) + " cent"
        return head + ("" if rest == 0 else " " + cardinal_fr(rest))
    for size, name in _FR_SCALES:
        if n >= size:
            count, rest = divmod(n, size)
            head = name if (count == 1 and size == 1000) else cardinal_fr(count) + " " + name
            return head + ("" if rest == 0 else " " + cardinal_fr(rest))
    raise AssertionError(n)


def _fold_fr(word: str) -> str:
    """Give French words that sound the same one form.

    "sauvée", "sauvés" and "sauver" become "sauve"; "paie", "paye" and "paix" become "pai".
    """
    word = word.replace("ay", "ai")
    if word == "ils":
        return "il"
    if len(word) >= 4 and word[-1] in "sx" and not word.endswith("ss"):
        word = word[:-1]
    if len(word) >= 6 and word.endswith("ent"):
        word = word[:-2]
    if len(word) >= 5 and word[-2:] in ("ee", "er", "ez"):
        word = word[:-1]
    if len(word) >= 4 and word[-1] == "e" and word[-2] in "aiou":
        word = word[:-1]
    return word


def _plain(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'")
    for letter, plain in _LIGATURES.items():
        text = text.replace(letter, plain)
    # Accents go, so "début" is one token and "a" matches "à" in a transcript.
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _clean(text: str) -> str:
    text = _ELISION.sub(r"\1 ", _plain(text))
    # Thousands separators: "18,000" in English, "18 000" in French.
    text = re.sub(r"(?<![\d.,])\d{1,3}(?:[,\u00a0\u202f ]\d{3})+\b", lambda m: re.sub(r"\D", "", m.group(0)), text)
    for symbol, word in _SYMBOLS.items():
        text = text.replace(symbol, word)
    return re.sub(r"[-–—/]", " ", text)


def tokens(text: str, *, forms: dict[str, str] | None = None, language: str = "en") -> list[str]:
    """Normalize text to lowercase word tokens. Digits become words of `language`.

    `forms` picks the spoken form for a numeric token; else the first form applies. In French,
    whole numbers become French words and homophone endings get one form; see `_fold_fr`.
    """
    french = language.lower().startswith("fr")
    out: list[str] = []
    for raw in _TOKEN.findall(_NUMBER.sub(lambda m: f" {m.group(0)} ", _clean(text))):
        if french and raw.isdigit():
            out.extend(_fold_fr(w) for w in cardinal_fr(int(raw)).split())
        elif _NUMBER.fullmatch(raw):
            choice = (forms or {}).get(raw) or _number_forms(raw)[0]
            out.extend(choice.split())
        elif french:
            out.append(_fold_fr(raw))
        else:
            for pattern, plain in _SPELLINGS:
                raw = pattern.sub(plain, raw)
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


def word_errors(
    reference: str, hypothesis: str, *, names: frozenset[str] = frozenset(), language: str = "en"
) -> WordErrors:
    """Compare two texts after normalization. Numbers in the hypothesis take their best spoken form.

    WER is (substitutions + deletions + insertions) / reference words. A heard word that is a
    spelling variant of one of `names`, such as "Shanghai" for "Shanhai", counts as that name.
    """
    ref = tokens(reference, language=language)
    chosen: dict[str, str] = {}
    numbers = dict.fromkeys(m.group(0) for m in _NUMBER.finditer(_clean(hypothesis)))
    for token in numbers:
        options = _number_forms(token)
        if len(options) > 1:
            chosen[token] = min(
                options,
                key=lambda o: sum(_distance(ref, tokens(hypothesis, forms={**chosen, token: o}, language=language))),
            )
    # A heard filler that the script does not have is not an error.
    hyp = [
        w
        for w in _merge_compounds(ref, tokens(hypothesis, forms=chosen, language=language))
        if w not in FILLERS or w in ref
    ]
    if names:
        hyp = [_respell(tuple(names), [w])[0] if w not in ref else w for w in hyp]
    s, d, n = _distance(ref, hyp)
    return WordErrors((s + d + n) / max(len(ref), 1), len(ref), s, d, n)


@lru_cache(maxsize=256)
def _fact_tokens(fact: str, language: str) -> tuple[str, ...]:
    return tuple(tokens(fact, language=language))


def _respell(want: tuple[str, ...], heard: list[str]) -> list[str]:
    """Replace heard words that are spelling variants of a wanted word, such as "shanghai" for "shanhai"."""
    out = []
    for word in heard:
        near = max(want, key=lambda w: SequenceMatcher(None, w, word).ratio())
        out.append(near if len(near) > 3 and SequenceMatcher(None, near, word).ratio() >= 0.8 else word)
    return out


def fact_heard(
    fact: str, transcript_tokens: list[str], *, tolerance: float = 0.2, fuzzy: bool = False, language: str = "en"
) -> bool:
    """Tell whether a transcript contains a fact phrase, with at most `tolerance` word errors.

    `fuzzy` accepts spelling variants of names longer than three letters.
    """
    want = list(_fact_tokens(fact, language))
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
    """Count spoken words after normalization. An elision such as "l'exposition" counts as one word."""
    return len(tokens(text)) - len(_ELISION.findall(_plain(text)))
