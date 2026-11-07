"""Normalization, WER and fact rules that the quality gates rely on."""

from vis_podcaster.text import (
    cardinal,
    cardinal_fr,
    fact_heard,
    ordinal,
    proper_names,
    tokens,
    word_count,
    word_errors,
    year,
)


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
    # The French tour-eiffel episode heard "18 000" for the script's "18000". In English, a space
    # separates two numbers.
    assert tokens("environ 18 000 pièces", language="fr") == tokens("environ 18000 pièces", language="fr")
    assert tokens("in 1889 300 workers") == tokens("in 1889") + tokens("300 workers")


def test_modern_years_have_both_spoken_forms():
    assert word_errors("In twenty twelve, a survey", "In 2012, a survey").wer == 0.0
    assert word_errors("In two thousand twelve, a survey", "In 2012, a survey").wer == 0.0


def test_the_spoken_words_of_a_number_are_not_errors():
    # A transcript can say the "and" or the "a" of a number, and write the "oh" of a year as "o".
    assert word_errors("In 2008, a survey", "In two thousand and eight, a survey").wer == 0.0
    assert word_errors("About 100 towers.", "About a hundred towers.").wer == 0.0
    assert word_errors("Built in 1905.", "Built in nineteen o five.").wer == 0.0


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


def test_french_numbers_are_french_words():
    assert cardinal_fr(71) == "soixante et onze" and cardinal_fr(91) == "quatre vingt onze"
    assert cardinal_fr(1889) == "mille huit cent quatre vingt neuf" and cardinal_fr(2_000_000) == "deux million"
    # The tour-eiffel judge wrote words for some numbers and digits for others.
    assert (
        word_errors(
            "En 1889, 150 000 visiteurs.",
            "En mille huit cent quatre-vingt-neuf, cent cinquante mille visiteurs.",
            language="fr",
        ).wer
        == 0
    )
    assert word_errors("En 1889.", "En mille huit cent quatre-vingt-neuf.").wer > 0


def test_french_homophone_endings_are_not_errors():
    heard = "Prolonger pour qui ? Ils payent. Accusé. Il reproche."
    assert word_errors("Prolongée pour qui ? Ils paient. Accusée. Ils reprochent.", heard, language="fr").wer == 0


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


def test_decimals_keep_their_point():
    # Zero-to-demo-day said "two point five percent" for "2.5%" and the check lost the "point".
    assert tokens("2.5%") == ["two", "point", "five", "percent"]
    assert word_errors("It grew 2.5% in 2024.", "It grew two point five percent in twenty twenty four.").wer == 0
    # A French decimal keeps its "virgule", also when the text writes it with a point.
    said = tokens("En deux virgule cinq", language="fr")
    assert tokens("En 2,5", language="fr") == tokens("En 2.5", language="fr") == said


def test_measured_spelling_variants_are_not_errors():
    # The zero-to-demo-day judge wrote these forms for correct takes.
    script = "Alright, that judgement is likeable. Our defence moved toward the grey programme. Okay, I learnt it."
    heard = "All right, that judgment is likable. Our defense moved towards the gray program. OK, I learned it."
    assert word_errors(script, heard).wer == 0
    assert word_errors("We were all right in the end.", "We were alright in the end.").wer == 0


def test_spelled_letters_match_an_acronym():
    assert word_errors("Good SEO matters for an AI app.", "Good S E O matters for an A I app.").wer == 0
    assert word_errors("Good S.E.O. matters.", "Good SEO matters.").wer == 0


def test_homophones_and_possessives_are_not_errors():
    assert word_errors("Right. Its price went up too.", "Write. It's price went up two.").wer == 0
    assert word_errors("The founders' shares.", "The founder's shares.").wer == 0


def test_a_name_heard_as_two_words_counts_as_the_name():
    script = "It runs from Shanhai Pass to the sea."
    heard = "It runs from Shang hai Pass to the sea."
    assert word_errors(script, heard, names=proper_names(script)).errors == 0


def test_spoken_forms_in_a_transcript_count_as_written():
    # Zero-to-demo-day: the judge wrote the spoken form in one line and the written form in the next.
    script = "Apply to YC. YC funds teams. Ask the FAQ bot about FAQ pages."
    heard = "Apply to Y C. YC funds teams. Ask the fack bot about FAQ pages."
    assert word_errors(script, heard).errors == 1
    assert word_errors(script, heard, pronunciations={"YC": "Y C", "FAQ": "fack"}).wer == 0


def test_facts_tolerate_spacing_and_fillers():
    assert fact_heard("2.5 percent", tokens("prices rose two point five per cent last year"))
    assert fact_heard("2.5 percent", tokens("prices rose two point five, um, percent last year"))
    assert not fact_heard("2.5 percent", tokens("prices rose two point six percent last year"))
    # Only names get spelling variants: "ice" is not "rice".
    assert not fact_heard("sticky rice", tokens("builders mixed sticky ice into it"), fuzzy=True)


def test_a_fact_needs_its_exact_numbers():
    heard = tokens("about two point five million people live there")
    assert fact_heard("2.5 million people", heard)
    assert not fact_heard("1.5 million people", heard)
    assert not fact_heard("2.5% growth", tokens("two point six percent growth"))
    assert not fact_heard("costs 9.99 dollars", tokens("it costs nine point nine eight dollars"))
    assert not fact_heard("built in 1487", tokens("it was built in 1489"))
    assert not fact_heard("21 196 kilomètres", tokens("vingt mille cent quatre-vingt-seize kilomètres"), language="fr")
    # Other spoken forms of the same number pass.
    assert fact_heard("21,196 kilometres", tokens("twenty one thousand one hundred and ninety six kilometres"))
    assert fact_heard("a hundred towers", tokens("almost 100 towers"))
    assert fact_heard(
        "21 196 kilomètres", tokens("vingt et un mille cent quatre-vingt-seize kilomètres"), language="fr"
    )


def test_a_fact_in_digits_matches_a_number_said_with_and():
    # The script check needs the fact exactly, and a script line says "two hundred and twenty-one".
    line = tokens("The big moment comes in two hundred and twenty-one BC.")
    assert fact_heard("221 BC", line, tolerance=0.0)
    assert fact_heard("two hundred and twenty-one BC", tokens("in 221 BC"), tolerance=0.0)
    assert fact_heard("1001 nights", tokens("one thousand and one nights"), tolerance=0.0)
    assert not fact_heard("222 BC", line, tolerance=0.0)


def test_a_fact_number_is_not_part_of_a_longer_number():
    assert not fact_heard("2 million people", tokens("about twenty two million people"))
    assert not fact_heard("500 workers", tokens("it took 2,500 workers"))
    assert not fact_heard("40 metres high", tokens("it is 140 metres high"))
    assert not fact_heard("1 million", tokens("one point one million"))
    assert not fact_heard("2 millions de briques", tokens("22 millions de briques", language="fr"), language="fr")
    # An ordinal ends a number.
    assert fact_heard("two weeks", tokens("in the first two weeks"), tolerance=0.0)


def test_two_numbers_next_to_each_other_stay_two_numbers():
    # In a text, punctuation, a line end and a number in digits end a number.
    assert fact_heard("40 percent of farms", "In 2008, 40 percent of farms failed.", tolerance=0.0)
    assert fact_heard("in 2008", "In 2008, 40 percent of farms failed.", tolerance=0.0)
    assert fact_heard("7 towers", "Of those 12, 7 towers survive.", tolerance=0.0)
    assert fact_heard("3 million", "2 million, 3 million", tolerance=0.0)
    assert fact_heard("3 million", "2 million 3 million", tolerance=0.0)
    assert fact_heard("twenty people", "In nineteen ninety, twenty people died.", tolerance=0.0)
    assert fact_heard("twenty people", "It ended in nineteen ninety\ntwenty people died.", tolerance=0.0)
    assert fact_heard("20 people", "In 1990 20 people died.", tolerance=0.0)
    # Without punctuation, the grammar of English numbers shows where a number ends.
    assert fact_heard("4 soldiers", tokens("no one four soldiers"), tolerance=0.0)
    assert fact_heard("40 percent", tokens("in two thousand eight forty percent"), tolerance=0.0)
    assert fact_heard("300,000 people", tokens("in nineteen thirty seven three hundred thousand people"), tolerance=0.0)
    assert fact_heard("200 metres", tokens("between one hundred and two hundred metres"), tolerance=0.0)
    assert fact_heard("12 towers", tokens("in fourteen fifty twelve towers stood"), tolerance=0.0)
    # One number stays one number.
    assert fact_heard("221,000 men", tokens("two hundred and twenty one thousand men"), tolerance=0.0)
    assert not fact_heard("21,000 men", tokens("two hundred and twenty one thousand men"))
    assert not fact_heard("20 million", "In 1990, 120 million people")
    assert not fact_heard("4 metres", tokens("three point one four metres"))
    assert not fact_heard("5", tokens("in one thousand five six thousand people died"))
    assert not fact_heard("200,000 people", "1 million 200,000 people")
    assert not fact_heard("200 people", "1 thousand 200 people")


def test_a_space_between_two_english_numbers_keeps_them_two_numbers():
    # In English, a space is not a thousands separator: "In 476 300 Goths" has two numbers.
    script = "In 476 300 Goths crossed."
    assert word_errors(script, "In four hundred seventy six three hundred Goths crossed.").wer == 0
    assert word_errors(script, "In 476,300 Goths crossed.").wer > 0
    assert fact_heard("300 Goths", "In 476 300 Goths crossed the river.", tolerance=0.0)
    assert fact_heard("500 men", "By 1,500 500 men had left.", tolerance=0.0)


def test_a_fact_in_digits_passes_in_each_spoken_form():
    assert fact_heard(
        "paid 1,487 workers", tokens("paid one thousand four hundred eighty seven workers"), tolerance=0.0
    )
    assert fact_heard("in 2015 the wall", tokens("in two thousand fifteen the wall"), tolerance=0.0)
    assert fact_heard("built in 1905", tokens("built in nineteen o five"), tolerance=0.0)
    heard = tokens("deux virgule cinq millions de visiteurs par an", language="fr")
    assert fact_heard("2,5 millions de visiteurs par an", heard, tolerance=0.0, language="fr")
    assert not fact_heard("2,6 millions de visiteurs par an", heard, language="fr")


def test_a_french_number_has_both_genders_and_ends_where_its_grammar_ends():
    # "une" is the feminine "un": "vingt et une personnes" are 21 people.
    assert fact_heard("21 personnes", "Il y avait vingt et une personnes.", tolerance=0.0, language="fr")
    assert fact_heard("1 heure", "Il est une heure.", tolerance=0.0, language="fr")
    assert fact_heard("101 personnes", "Il y avait cent une personnes.", tolerance=0.0, language="fr")
    # Without punctuation, the grammar of French numbers shows where a number ends.
    assert fact_heard("7 tours", tokens("sur ces douze sept tours", language="fr"), tolerance=0.0, language="fr")
    heard = tokens("en mille neuf cent trente-sept trois cent mille personnes", language="fr")
    assert fact_heard("300 000 personnes", heard, tolerance=0.0, language="fr")
    heard = tokens("en mille neuf cent quarante un homme vint", language="fr")
    assert fact_heard("1940", heard, tolerance=0.0, language="fr")
    # One number stays one number, also with its "un".
    assert not fact_heard("7 tours", tokens("dix-sept tours", language="fr"), language="fr")
    assert not fact_heard("12 tours", tokens("soixante-douze tours", language="fr"), language="fr")
    assert not fact_heard("80 tours de garde", "Il y avait quatre-vingt-un tours de garde.", language="fr")
    assert not fact_heard("1 victoire", "Ils ont eu quatre-vingt-une victoires.", tolerance=0.0, language="fr")
    assert not fact_heard("En 2000 il est parti", "en deux mille un il est parti", language="fr")
    fact = "Il y avait 1,5 million d'habitants en France à cette époque"
    assert not fact_heard(fact, "Il y avait cinq millions d'habitants en France à cette époque", language="fr")
    assert not fact_heard(
        "2,1 tonnes de fer chaque jour dans cette usine",
        "deux tonnes de fer chaque jour dans cette usine",
        language="fr",
    )
    # In French, "a" is a verb, not the number "one".
    assert fact_heard("Il a 100 ans aujourd'hui", "Il avait cent ans aujourd'hui", language="fr")


def test_only_real_names_in_a_fact_get_spelling_variants():
    # A sentence starts with a capital letter: "Sticky" is not a name, so "tricky" is not "sticky".
    assert not fact_heard("Sticky rice", tokens("builders mixed tricky rice into it"), fuzzy=True)
    assert fact_heard("Shanhai Pass", tokens("through shanghai pass"), fuzzy=True)
    heard = tokens("beijin is the capital")
    assert fact_heard("Beijing is the capital", heard, fuzzy=True, names=proper_names("They went to Beijing."))
    assert not fact_heard("Beijing is the capital", heard, fuzzy=True)
