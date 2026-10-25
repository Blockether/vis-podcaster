"""Normalization, WER and fact rules that the quality gates rely on."""

from vis_podcaster.text import cardinal, fact_heard, ordinal, proper_names, tokens, word_count, word_errors, year


def test_numbers_are_spelled():
    assert cardinal(21196) == "twenty one thousand one hundred ninety six"
    assert year(1368) == "thirteen sixty eight"
    assert year(2005) == "two thousand five"
    assert ordinal(21) == "twenty first"
    assert ordinal(3) == "third"


def test_digits_in_a_transcript_match_words_in_a_script():
    script = (
        "It was finished in fourteen ninety-two and runs twenty-one thousand, one hundred and ninety-six kilometers."
    )
    heard = "It was finished in 1492 and runs 21,196 kilometers."
    assert word_errors(script, heard).wer == 0.0


def test_spaced_thousands_are_one_number_but_separate_numbers_stay_apart():
    # The tour-eiffel episode heard "18 000" for the script's "18000".
    assert tokens("environ 18 000 pièces") == tokens("environ 18000 pièces")
    assert tokens("in 1889 300 workers") == tokens("in 1889") + tokens("300 workers")


def test_modern_years_have_both_spoken_forms():
    assert word_errors("In twenty twelve, a survey", "In 2012, a survey").wer == 0.0
    assert word_errors("In two thousand twelve, a survey", "In 2012, a survey").wer == 0.0


def test_compounds_that_differ_only_in_spacing_are_equal():
    assert word_errors("brick walls and watchtowers", "brick walls and watch towers").wer == 0.0


def test_wer_counts_each_error_kind():
    errors = word_errors("one two three four", "one three four five")
    assert (errors.substitutions, errors.deletions, errors.insertions) == (0, 1, 1)
    assert errors.wer == 0.5


def test_name_variants_count_only_when_names_are_given():
    script = "It runs from Shanhai Pass to the sea."
    heard = "It runs from Shanghai Pass to the sea."
    assert word_errors(script, heard).errors == 1
    assert word_errors(script, heard, names=proper_names(script)).errors == 0


def test_facts_allow_small_errors_but_not_other_words():
    heard = tokens("the ming wall was rebuilt in brick")
    assert fact_heard("the Ming wall was rebuilt", heard)
    assert not fact_heard("sticky rice", heard)
    assert fact_heard("Shanhai Pass", tokens("through shanghai pass"), fuzzy=True)
    assert not fact_heard("Shanhai Pass", tokens("the shanghai tower"), fuzzy=True)


def test_accents_and_elision_make_french_words_comparable():
    assert tokens("L'Exposition universelle, au début. Où ? Œuvre") == [
        "l",
        "exposition",
        "universelle",
        "au",
        "debut",
        "ou",
        "oeuvre",
    ]
    assert fact_heard("Exposition universelle", tokens("C'est l'année de l'Exposition universelle"), tolerance=0.0)
    assert word_errors("à la tour", "a la tour").wer == 0


def test_an_elision_counts_as_one_spoken_word():
    assert word_count("C'est l'année de l'Exposition") == 4
    assert word_count("It's the tower") == 3


def test_french_plural_number_words_match_their_singular():
    assert word_errors("trois cents mètres, quatre-vingts ans", "trois-cent mètres, quatre-vingt ans").wer == 0


def test_british_spellings_and_added_fillers_are_not_errors():
    # Parakeet wrote "travelled", "sceptic" and "uh" for correct takes of the broad-street episode.
    script = "The water traveled. The skeptic was at the center of the theater, in color."
    heard = "The water travelled. The sceptic was, uh, at the centre of the theatre, in colour."
    assert word_errors(script, heard).wer == 0.0
    assert word_errors("Uh, wait.", "Wait.").deletions == 1
    assert tokens("notre votre filled") == ["notre", "votre", "filled"]


def test_english_contractions_stay_one_word():
    assert tokens("I'm sure it's fine, don't worry, let's go") == [
        "i'm",
        "sure",
        "it's",
        "fine",
        "don't",
        "worry",
        "let's",
        "go",
    ]
