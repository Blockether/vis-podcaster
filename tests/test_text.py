"""Normalization, WER and fact rules that the quality gates rely on."""

from vis_podcaster.text import cardinal, fact_heard, ordinal, proper_names, tokens, word_errors, year


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
