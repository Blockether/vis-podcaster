"""Text rules that the checks share: normalization, word error rate and fact recall.

The rules make a script and an ASR transcript comparable. They expand digits, ordinals
and symbols into words, so "1987" in a transcript can match "nineteen eighty-seven". French
text gets French number words and one form for endings that sound the same.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from itertools import islice, pairwise, product

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
# A decimal is one token, so "2.5" keeps its "point".
_TOKEN = re.compile(r"\d+\.\d+|[a-z0-9]+(?:'[a-z]+)?")
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
    (re.compile(r"^practis(?=e$|es$|ed$|ing$)"), "practic"),
    (re.compile(r"(?<=\w{3})is(?=e$|es$|ed$|ing$|ation$|ations$)"), "iz"),
    (re.compile(r"(?<=\w{2})ys(?=e$|es$|ed$|ing$)"), "yz"),
    (re.compile(r"^(cent|met|kilomet|millimet|centimet|theat|fib|lit|spect|somb|lust|meag)re(?=s?$)"), r"\1er"),
    (re.compile(r"(?<=\w{3})ogue$"), "og"),
    (re.compile(r"(?<=dg)ement(?=s?$)"), "ment"),
    (re.compile(r"(?<=[bdfklmnprstvz])eable$"), "able"),
    (re.compile(r"^(defen|offen|preten|licen)ce(?=s?$)"), r"\1se"),
    (re.compile(r"^(to|for|after|back|up|down|on|in|out|home|sky|north|south|east|west)wards$"), r"\1ward"),
    (re.compile(r"^grey"), "gray"),
    (re.compile(r"^programme(?=s?$)"), "program"),
    (re.compile(r"^(cent|vingt)s$"), r"\1"),
)
_WORDS = {
    "okay": "ok",
    "ageing": "aging",
    "learnt": "learned",
    "burnt": "burned",
    "dreamt": "dreamed",
    "spelt": "spelled",
    "spoilt": "spoiled",
    "leapt": "leaped",
    "mr": "mister",
    "vs": "versus",
    "etc": "etcetera",
}
# English word pairs that sound like one word get that word: "all right" is "alright".
_PHRASES = {("all", "right"): "alright", ("per", "cent"): "percent", ("et", "cetera"): "etcetera", ("o", "k"): "ok"}
# English words that sound the same get one form. An ASR engine picks one of them from context,
# so a transcript with "write" for the script's "right" is not an error.
_HOMOPHONES = {
    word: group[0]
    for group in (
        ("its", "it's"),
        ("your", "you're"),
        ("their", "there", "they're"),
        ("whose", "who's"),
        ("to", "too", "two"),
        ("for", "four", "fore"),
        ("one", "won"),
        ("eight", "ate"),
        ("by", "buy", "bye"),
        ("no", "know"),
        ("new", "knew"),
        ("here", "hear"),
        ("right", "write", "rite"),
        ("where", "wear"),
        ("peace", "piece"),
        ("whole", "hole"),
        ("weather", "whether"),
        ("site", "sight", "cite"),
        ("break", "brake"),
        ("role", "roll"),
        ("meet", "meat"),
        ("week", "weak"),
        ("wait", "weight"),
        ("way", "weigh"),
        ("plain", "plane"),
        ("past", "passed"),
        ("allowed", "aloud"),
        ("principal", "principle"),
        ("peak", "peek", "pique"),
        ("complement", "compliment"),
        ("discreet", "discrete"),
        ("patience", "patients"),
        ("presence", "presents"),
        ("through", "threw"),
        ("would", "wood"),
        ("which", "witch"),
        ("some", "sum"),
        ("son", "sun"),
        ("sole", "soul"),
        ("steel", "steal"),
        ("tale", "tail"),
        ("male", "mail"),
        ("made", "maid"),
        ("sale", "sail"),
        ("waste", "waist"),
        ("pair", "pear"),
        ("course", "coarse"),
        ("council", "counsel"),
        ("board", "bored"),
        ("bear", "bare"),
        ("idle", "idol"),
        ("minor", "miner"),
        ("real", "reel"),
        ("rain", "reign", "rein"),
        ("scene", "seen"),
        ("sell", "cell"),
        ("stationary", "stationery"),
    )
    for word in group
}
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


def _clean(text: str, french: bool = False) -> str:
    text = _ELISION.sub(r"\1 ", _plain(text))
    # Thousands separators: "18,000" in English, "18 000" in French. In English, a plain space separates
    # two numbers, so "In 476 300 Goths" has two.
    sep = r"[\u00a0\u202f ]" if french else r"[,\u00a0\u202f]"
    text = re.sub(rf"(?<![\d.,])\d{{1,3}}(?:{sep}\d{{3}})+\b", lambda m: re.sub(r"\D", "", m.group(0)), text)
    for symbol, word in _SYMBOLS.items():
        text = text.replace(symbol, word)
    return re.sub(r"[-–—/]", " ", text)


def _raw(text: str, french: bool) -> list[str]:
    """Split text into raw tokens. A number with decimals is one token; in French, "2,5" is a decimal."""
    text = _clean(text, french)
    if french:
        text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    return _TOKEN.findall(_NUMBER.sub(lambda m: f" {m.group(0)} ", text))


def tokens(text: str, *, forms: dict[str, str] | None = None, language: str = "en") -> list[str]:
    """Normalize text to lowercase word tokens. Digits become words of `language`.

    `forms` picks the spoken form for a numeric token; else the first form applies. In French,
    numbers become French words, "2,5" is "deux virgule cinq", and homophone endings get one
    form; see `_fold_fr`.
    """
    french = language.lower().startswith("fr")
    return [word for raw in _raw(text, french) for word in _say(raw, forms, french)]


def _say(raw: str, forms: dict[str, str] | None, french: bool) -> list[str]:
    """Return the normalized words of one raw token; see `tokens`."""
    if french and raw.replace(".", "").isdigit():
        whole, _, decimals = raw.partition(".")
        said = cardinal_fr(int(whole))
        if decimals:
            # "2,05" is "deux virgule zéro cinq".
            rest = decimals.lstrip("0")
            said += " virgule" + " zero" * (len(decimals) - len(rest)) + (f" {cardinal_fr(int(rest))}" if rest else "")
        return [_fold_fr(w) for w in said.split()]
    if _NUMBER.fullmatch(raw):
        return ((forms or {}).get(raw) or _number_forms(raw)[0]).split()
    if french:
        return [_fold_fr(raw)]
    for pattern, plain in _SPELLINGS:
        raw = pattern.sub(plain, raw)
    return [_WORDS.get(raw, raw)]


# Between two words of a transcript, `_BREAK` tells that one spoken number cannot go on there.
_BREAK = "|"
# A line end, a dash or a punctuation mark ends a number. A point or a comma between two digits does not.
_PAUSE = re.compile(r"[\n;:!?()–—]|(?<!\d)[.,]|[.,](?!\d)")


@lru_cache(maxsize=1024)
def _heard_words(text: str, language: str) -> tuple[str, ...]:
    """Normalize text like `tokens`, with `_BREAK` where a spoken number cannot go on.

    A break goes at punctuation and around each number in digits, but not between a number and its
    scale word. So "In 2008, 40 percent" and "In 2008 40 percent" have two numbers, and "20 million" has one.
    """
    french = language.lower().startswith("fr")
    out: list[str] = []
    for segment in _PAUSE.split(text):
        raws = _raw(segment, french)
        for i, raw in enumerate(raws):
            words = _say(raw, None, french)
            if raw[0].isdigit():
                after = _say(raws[i + 1], None, french)[0] if i + 1 < len(raws) else ""
                # "1 million 200,000" is one number, like "20 million".
                goes_on = bool(out) and out[-1] in _SCALE_RANK and _goes_on(out[-1], words)
                words = ([] if goes_on else [_BREAK]) + words + ([] if after in _SCALE_WORDS else [_BREAK])
            out.extend(words)
        out.append(_BREAK)
    return tuple(w for i, w in enumerate(out) if w != _BREAK or (i and out[i - 1] != _BREAK))


# Number words of the normalized text. A fact must have its numbers exactly: "1.5 million" is not "2.5 million".
_EN_NUMBER_WORDS = frozenset(
    [*_ONES, *_TENS[2:], *(name for _, name in _SCALES), "hundred"]
    + [w for n in (*range(1, 20), *range(20, 100, 10), 100, 1000, 10**6, 10**9) for w in ordinal(n).split()]
)
_FR_NUMBER_WORDS = frozenset(
    _fold_fr(w) for w in (*_FR_ONES[2:], "zero", *_FR_TENS[1:], "cent", "mille", "million", "milliard")
)
_SCALE_WORDS = frozenset({"hundred", "thousand", "million", "billion", "cent", "mille", "milliard"})
# An ordinal ends a number, so "the first two weeks" has the two numbers "first" and "two".
_ORDINALS = frozenset(ordinal(n).split()[-1] for n in (*range(1, 20), *range(20, 100, 10), 100, 1000, 10**6, 10**9))


def _number_parts(words: list[str], language: str) -> list[str | None]:
    """Give each normalized word its number form: None outside a number, "" for the "et" of "vingt et un".

    "point", "virgule" and "oh" count only between number words. In English, "a" counts as "one"
    before a scale word: "a hundred". In French, "un" or "une" counts as "one" next to a number:
    before a scale word or "virgule", and after "et" or a number word, as in "quatre-vingt-un" and
    "deux virgule un". `_fr_stop` ends a number before the article of "trente un homme". "and" is
    not a number word: `_canon` removes the "and" of "two hundred and five".
    """
    french = language.lower().startswith("fr")
    vocab = _FR_NUMBER_WORDS if french else _EN_NUMBER_WORDS
    numeric = vocab | {"un", "une"} if french else vocab
    parts: list[str | None] = []
    for i, word in enumerate(words):
        before = words[i - 1] if i else ""
        after = words[i + 1] if i + 1 < len(words) else ""
        in_number = bool(parts) and parts[-1] is not None
        if word in vocab:
            parts.append(word)
        elif word in ("point", "virgule", "oh") and before in numeric and after in numeric:
            parts.append("zero" if word == "oh" else "point")
        elif (word == "a" and not french and after in _SCALE_WORDS) or (
            word in ("un", "une")
            and (after in _SCALE_WORDS or before == "et" or (french and (after == "virgule" or in_number)))
        ):
            parts.append("one")
        elif french and word == "et" and before in vocab and after in ("un", "une", "onze"):
            parts.append("")
        else:
            parts.append(None)
    return parts


def _numbers(words: list[str], language: str) -> list[str]:
    """Return the number words of normalized `words` in order; see `_number_parts`."""
    return [part for part in _number_parts(words, language) if part]


_SCALE_RANK = {"hundred": 1, "thousand": 2, "million": 3, "billion": 4}
_EN_DIGITS = frozenset(_ONES[1:10])
_EN_TEENS = frozenset(_ONES[10:])
_EN_SMALL = frozenset([*_ONES[1:], *_TENS[2:]])


_EN_TENS = frozenset(_TENS[2:])
_FR_UNITS = frozenset(_fold_fr(w) for w in _FR_ONES[1:10])
_FR_TEENS = frozenset(_fold_fr(w) for w in _FR_ONES[11:])
_FR_TENS_WORDS = frozenset(_fold_fr(w) for w in _FR_TENS[2:])
_FR_BEFORE_ONE = frozenset(_fold_fr(w) for w in ("et", "vingt", "cent", "mille", "million", "milliard"))


def _small_stop(before: str, word: str, starts: bool = False) -> bool:
    """Tell whether an English number stops between two small number words.

    "eight forty", "twelve seven" and "fifty twelve" are two numbers each. A year that `starts` with
    "twenty" is one number: "twenty twelve", "twenty twenty".
    """
    if before in _EN_DIGITS:
        return word in _EN_SMALL
    if before in _EN_TEENS:
        return word in _EN_DIGITS
    return before in _EN_TENS and (word in _EN_TEENS or word in _EN_TENS) and not (before == "twenty" and starts)


def _fr_stop(before: str, word: str) -> bool:
    """Tell whether a French number stops between two number words.

    "deux trois", "douze sept" and "trente quarante" are two numbers each. "dix sept", "quatre vingt",
    "vingt deux" and "soixante douze" are one. "un" goes on only after "et", "vingt" or a scale word:
    "cent un" is one number, but "trente un homme" is 30 and the article "un".
    """
    if word in ("un", "une"):
        return before not in _FR_BEFORE_ONE
    if before in _FR_UNITS:
        tens = word in _FR_TENS_WORDS and not (before == "quatre" and word == "vingt")
        return word in _FR_UNITS or word in _FR_TEENS or word == "dix" or tens
    if before in _FR_TEENS:
        return word in _FR_UNITS
    return before in _FR_TENS_WORDS and word in _FR_TENS_WORDS


def _goes_on(scale: str, run: list[str]) -> bool:
    """Tell whether the English number words `run` go on a number that ends with the scale word `scale`.

    "two hundred" goes on with "twenty one thousand" but not with "two hundred". "one thousand" goes on
    with "two hundred" but not with "two thousand".
    """
    if scale == "hundred":
        return next((w for w in run if w in _SCALE_RANK), "") != "hundred"
    bigger = next((w for w in run if w in _SCALE_RANK and w != "hundred"), "")
    return not bigger or _SCALE_RANK[bigger] < _SCALE_RANK[scale]


def _cuts(words: list[str], parts: list[str | None], breaks: set[int], language: str) -> list[bool]:
    """Tell for each place from 0 to `len(words)` whether a spoken number cannot go on across it.

    A number cannot go on at a break, after an ordinal or next to a word that is not a number word. It
    also stops where the grammar of numbers does not let it go on. In English, that is between two
    small number words ("eight forty" is 8 and 40; see `_small_stop`) and at a scale word before a
    number with a scale that is not lower ("one hundred two hundred"; see `_goes_on`). For French, see
    `_fr_stop`. The digits after "point" go on: "three point one four".
    """
    english = language.lower().startswith("en")
    french = language.lower().startswith("fr")
    cuts = [True] * (len(words) + 1)
    decimal = False
    for i in range(1, len(words)):
        before, word = words[i - 1], words[i]
        goes_on = parts[i - 1] is not None and parts[i] is not None and i not in breaks and before not in _ORDINALS
        if goes_on and french and not decimal:
            goes_on = not _fr_stop(before, word)
        elif goes_on and english and not decimal:
            if _small_stop(before, word, cuts[i - 1]):
                goes_on = False
            elif before in _SCALE_RANK and word in _EN_SMALL:
                run = []
                for j in range(i, len(words)):
                    if parts[j] is None or (j > i and (j in breaks or _small_stop(words[j - 1], words[j]))):
                        break
                    run.append(words[j])
                goes_on = _goes_on(before, run)
        cuts[i] = not goes_on
        decimal = goes_on and (decimal or parts[i] == "point")
    return cuts


def _number_words(words: list[str], language: str) -> list[str]:
    """Give English number words the form of a number in digits.

    "a hundred" is "one hundred", as a transcript can write "100". "two hundred and five" is "205"
    too, so "and" after a scale word and before a number word goes. A transcript can write the "oh"
    of "nineteen oh five" as "o". Other languages keep their words.
    """
    if not language.lower().startswith("en"):
        return words
    words = [
        "one" if w == "a" and words[i + 1 : i + 2] and words[i + 1] in _SCALE_WORDS else w for i, w in enumerate(words)
    ]
    between = [0 < i < len(words) - 1 and words[i + 1] in _EN_NUMBER_WORDS for i in range(len(words))]
    return [
        "oh" if w == "o" and between[i] and words[i - 1] in _EN_NUMBER_WORDS else w
        for i, w in enumerate(words)
        if not (w == "and" and between[i] and words[i - 1] in _SCALE_WORDS)
    ]


def _canon(words: list[str], language: str, *, numbers: bool = False) -> list[str]:
    """Give English words and word pairs that sound the same one form; see `_PHRASES` and `_HOMOPHONES`.

    Comparisons use this form, not `tokens`. A possessive and a plural sound the same, so
    "founder's" and "founders'" are both "founders". With `numbers`, number words and their
    homophones keep their form, so "two" is not "to" and "won" is not "one", and the "and" of
    "two hundred and five" goes, so it matches "205".
    """
    if not language.lower().startswith("en"):
        if numbers and language.lower().startswith("fr"):
            # A French number has the same words in both genders: "vingt et une" is "vingt et un".
            return ["un" if w == "une" else w for w in words]
        return words
    out: list[str] = []
    for word in words:
        if out and (out[-1], word) in _PHRASES:
            out[-1] = _PHRASES[out[-1], word]
            continue
        if word.endswith("'s") and word not in _HOMOPHONES:
            word = word[:-2] + "s"
        out.append(word)
    if numbers:
        out = _number_words(out, language)
        keep = _EN_NUMBER_WORDS
        return [w if w in keep or _HOMOPHONES.get(w) in keep else _HOMOPHONES.get(w, w) for w in out]
    return [_HOMOPHONES.get(w, w) for w in out]


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


def _letter_runs(words: list[str]) -> dict[str, tuple[str, ...]]:
    """Map each run of at least two single letters to its joined form: "s e o" gives "seo"."""
    runs: dict[str, tuple[str, ...]] = {}
    start = 0
    for end in range(len(words) + 1):
        if end < len(words) and len(words[end]) == 1:
            continue
        for a in range(start, end - 1):
            for b in range(a + 2, end + 1):
                runs.setdefault("".join(words[a:b]), tuple(words[a:b]))
        start = end + 1
    return runs


def _merge_compounds(ref: list[str], hyp: list[str]) -> list[str]:
    """Join or split hypothesis words where only spacing differs: "watch towers" and "watchtowers".

    Spelled letters join too, so "s e o" matches "seo", and "seo" matches "s e o".
    """
    vocab = set(ref)
    pairs = {a + b: (a, b) for a, b in pairwise(ref)}
    ref_runs = _letter_runs(ref)
    pairs.update({joined: run for joined, run in ref_runs.items() if len(run) > 2})
    hyp_runs = _letter_runs(hyp)
    out, i = [], 0
    while i < len(hyp):
        word = hyp[i]
        if len(word) == 1 and i + 1 < len(hyp) and len(hyp[i + 1]) == 1:
            stop = i + 2
            while stop < len(hyp) and len(hyp[stop]) == 1:
                stop += 1
            joined = next(
                (
                    "".join(hyp[i:end])
                    for end in range(stop, i + 1, -1)
                    if "".join(hyp[i:end]) in vocab and "".join(hyp[i:end]) not in ref_runs
                ),
                None,
            )
            if joined and joined in hyp_runs:
                out.append(joined)
                i += len(joined)
                continue
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


def _nearest(ref: list[str], want: list[str]) -> int:
    """Return the fewest word edits that turn `want` into a part of `ref`."""
    prev = [0] * (len(ref) + 1)
    for i, w in enumerate(want, 1):
        cur = [i]
        for j, r in enumerate(ref, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (w != r)))
        prev = cur
    return min(prev)


def word_errors(
    reference: str,
    hypothesis: str,
    *,
    names: frozenset[str] = frozenset(),
    language: str = "en",
    pronunciations: Mapping[str, str] | None = None,
) -> WordErrors:
    """Compare two texts after normalization. Numbers in the hypothesis take their best spoken form.

    WER is (substitutions + deletions + insertions) / reference words. A heard word that is a
    spelling variant of one of `names`, such as "Shanghai" for "Shanhai", counts as that name.
    `pronunciations` maps a written form to its spoken form: a heard spoken form counts as the
    written form, so a transcript that writes "fack" in one line and "FAQ" in the next has no error.
    """

    def heard(forms: dict[str, str]) -> list[str]:
        words = _canon(_number_words(tokens(hypothesis, forms=forms, language=language), language), language)
        return _unsay(words, pronunciations, language) if pronunciations else words

    ref = _canon(_number_words(tokens(reference, language=language), language), language)
    chosen: dict[str, str] = {}
    # French numbers have one spoken form.
    french = language.lower().startswith("fr")
    numbers = () if french else dict.fromkeys(m.group(0) for m in _NUMBER.finditer(_clean(hypothesis)))
    for token in numbers:
        options = _number_forms(token)
        if len(options) > 1:
            # Each form counts by the nearest part of the reference. A full alignment for each form of each
            # number takes minutes on a long transcript.
            chosen[token] = min(
                options, key=lambda o: _nearest(ref, _canon(_number_words(o.split(), language), language))
            )
    # A heard filler that the script does not have is not an error.
    hyp = [w for w in _merge_compounds(ref, heard(chosen)) if w not in FILLERS or w in ref]
    if names:
        hyp = _respell_names(tuple(sorted(set(_canon(sorted(names), language)))), hyp, set(ref))
    s, d, n = _distance(ref, hyp)
    return WordErrors((s + d + n) / max(len(ref), 1), len(ref), s, d, n)


@lru_cache(maxsize=256)
def _fact_forms(fact: str, language: str) -> tuple[tuple[str, ...], ...]:
    """Return the compared tokens of a fact, once for each mix of the spoken forms of its numbers.

    "1487" is "fourteen eighty seven" or "one thousand four hundred eighty seven". French numbers
    have one form.
    """
    french = language.lower().startswith("fr")
    numbers = list(dict.fromkeys(raw for raw in _raw(fact, french) if not french and _NUMBER.fullmatch(raw)))
    forms: list[tuple[str, ...]] = []
    for mix in islice(product(*(_number_forms(raw) for raw in numbers)), 32):
        form = tuple(_canon(tokens(fact, forms=dict(zip(numbers, mix)), language=language), language, numbers=True))
        if form not in forms:
            forms.append(form)
    return tuple(forms)


@lru_cache(maxsize=256)
def _fact_names(fact: str, language: str, known: frozenset[str] = frozenset()) -> tuple[str, ...]:
    """Return the tokens of the names of a fact: its capitalized words, which can have spelling variants.

    A sentence starts with a capital letter too, so the first word is a name only when `known` has it,
    the fact is one word, or the next word is capitalized: "Sticky rice" has no name, "Shanhai Pass" has two.
    """
    words = re.findall(r"[\w'-]+", fact)
    capitalized = re.findall(r"(?<![\w'])[A-Z][\w'-]*", fact)
    first_is_plain = len(words) > 1 and not words[1][:1].isupper() and _plain(words[0]).split("'")[0] not in known
    if capitalized and capitalized[0] == words[0] and first_is_plain:
        capitalized = capitalized[1:]
    return tuple(_canon(tokens(" ".join(capitalized), language=language), language))


def _respell(want: tuple[str, ...], heard: list[str]) -> list[str]:
    """Replace heard words that are spelling variants of a wanted word, such as "shanghai" for "shanhai"."""
    out = []
    for word in heard:
        near = max(want, key=lambda w: SequenceMatcher(None, w, word).ratio())
        out.append(near if len(near) > 3 and SequenceMatcher(None, near, word).ratio() >= 0.8 else word)
    return out


def _unsay(words: list[str], pronunciations: Mapping[str, str], language: str) -> list[str]:
    """Put each spoken form of `pronunciations` (written form to spoken form) back as its written form."""
    pairs = []
    for written, said in pronunciations.items():
        say = tuple(_canon(tokens(said, language=language), language))
        write = _canon(tokens(written, language=language), language)
        if say and list(say) != write:
            pairs.append((say, write))
    pairs.sort(key=lambda pair: -len(pair[0]))
    out, i = [], 0
    while i < len(words):
        for say, write in pairs:
            if tuple(words[i : i + len(say)]) == say:
                out.extend(write)
                i += len(say)
                break
        else:
            out.append(words[i])
            i += 1
    return out


def _respell_names(names: tuple[str, ...], hyp: list[str], vocab: set[str]) -> list[str]:
    """Respell heard words that the reference does not have as the nearest name.

    Two such words in a row that join to a name variant become that name: "shang hai" is "shanhai".
    """
    out, i = [], 0
    while i < len(hyp):
        word = hyp[i]
        if word in vocab:
            out.append(word)
            i += 1
            continue
        if i + 1 < len(hyp) and hyp[i + 1] not in vocab:
            joined = _respell(names, [word + hyp[i + 1]])[0]
            if joined in names:
                out.append(joined)
                i += 2
                continue
        out.append(_respell(names, [word])[0])
        i += 1
    return out


def fact_heard(
    fact: str,
    transcript: str | list[str],
    *,
    tolerance: float = 0.2,
    fuzzy: bool = False,
    language: str = "en",
    names: frozenset[str] = frozenset(),
) -> bool:
    """Tell whether a transcript contains a fact phrase, with at most `tolerance` word errors.

    The numbers of the fact must be there exactly, so "1.5 million" does not pass for "2.5 million",
    and "2 million" does not pass for "22 million". A number in digits passes in each of its spoken
    forms, so "1487" passes for "fourteen eighty seven" and "one thousand four hundred eighty seven".
    `fuzzy` accepts spelling variants of the fact's names: its capitalized words longer than three
    letters, so "Shanghai" passes for "Shanhai" but "ice" does not pass for "rice". The first word is
    a name only when `names`, such as `proper_names(script)`, has it, the fact is one word, or the
    next word is capitalized. Spacing differences, such as "per cent" for "percent", and fillers that
    the fact does not have are not errors.

    Give `transcript` as text, not as `tokens`. Its punctuation, its line ends and its numbers in
    digits show where a number ends, so "40 percent" passes in "In 2008, 40 percent".
    """
    forms = _fact_forms(fact, language)
    if not forms[0]:
        return True
    words = list(_heard_words(transcript, language)) if isinstance(transcript, str) else transcript
    heard = _canon(words, language, numbers=True)
    known = _fact_names(fact, language, names) if fuzzy else ()
    return any(_has_phrase(list(want), heard, known, tolerance, language) for want in forms)


def _has_phrase(want: list[str], heard: list[str], known: tuple[str, ...], tolerance: float, language: str) -> bool:
    """Tell whether `heard` has `want` with at most `tolerance` word errors and exactly its numbers."""
    heard = [w for w in _merge_compounds(want, heard) if w not in FILLERS or w in want]
    if known:
        heard = _respell(known, heard)
    words: list[str] = []
    breaks: set[int] = set()
    for word in heard:
        if word == _BREAK:
            breaks.add(len(words))
        else:
            words.append(word)
    numbers = _numbers(want, language)
    # A window must not cut a number: "two million" is not in "twenty two million".
    cuts = _cuts(words, _number_parts(words, language), breaks, language)
    width = len(want)
    budget = int(width * tolerance)
    for size in range(max(1, width - budget), width + budget + 1):
        for start in range(max(1, len(words) - size + 1)):
            end = start + size
            if not (cuts[start] and cuts[min(end, len(words))]):
                continue
            window = words[start:end]
            if _numbers(window, language) == numbers and sum(_distance(want, window)) <= budget:
                return True
    return False


def word_count(text: str) -> int:
    """Count spoken words after normalization. An elision such as "l'exposition" counts as one word."""
    return len(tokens(text)) - len(_ELISION.findall(_plain(text)))
