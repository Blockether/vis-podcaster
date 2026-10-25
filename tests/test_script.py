"""Script validation reports every problem with its location."""

import copy

import pytest

from vis_podcaster.script import ScriptError, check_script, load_episode

SCRIPT = {
    "title": "Walls",
    "hosts": [{"id": "a", "name": "Ann", "voice": "emma"}, {"id": "b", "name": "Ben", "voice": "carter"}],
    "pronunciations": {"Qin": "Chin"},
    "critical_facts": ["Ming dynasty"],
    "sources": ["https://whc.unesco.org/en/list/438"],
    "lines": [
        {"speaker": "a", "text": "Who built the wall?"},
        {"speaker": "b", "text": "Qin Shi Huang joined it, and the Ming dynasty rebuilt it."},
    ],
}


def where(issues, severity):
    return {i.where for i in issues if i.severity == severity}


def test_a_good_script_has_no_issues():
    episode, issues = check_script(SCRIPT)
    assert episode is not None and issues == ()
    assert episode.spoken(episode.lines[1].text).startswith("Chin Shi Huang")


def test_errors_are_all_reported_at_once():
    bad = copy.deepcopy(SCRIPT)
    bad["lines"].append({"speaker": "c", "text": "Hi"})
    bad["lines"].append({"speaker": "a", "text": " "})
    bad["seed"] = -1
    episode, issues = check_script(bad)
    assert episode is None
    assert {"lines[2]", "lines[3]", "seed"} <= where(issues, "error")


def test_review_rules():
    s = copy.deepcopy(SCRIPT)
    s["critical_facts"] = ["Han dynasty"]
    s["lines"][0]["text"] = "Is it 2000 years old, says NASA?"
    s["lines"].append({"speaker": "b", "text": " ".join(["word"] * 71)})
    _, issues = check_script(s)
    assert {"critical_facts[0]", "lines[2]"} <= where(issues, "error")
    warnings = [i.message for i in issues if i.severity == "warning"]
    assert any("digits" in m for m in warnings) and any("NASA" in m for m in warnings)
    assert any("same speaker" in m for m in warnings)


def test_craft_rules_flag_generated_sounding_dialogue():
    s = copy.deepcopy(SCRIPT)
    s["language"] = "en"
    s["lines"] = [
        {"speaker": "a", "text": "Welcome to the show. Let's dive in. Who built the wall?"},
        {"speaker": "b", "text": "Great question! It is not just a wall, but a network."},
        {"speaker": "a", "text": "Absolutely. So who started it?"},
        {"speaker": "b", "text": "Exactly. The Qin did, and the Ming dynasty rebuilt it."},
        {"speaker": "a", "text": "Right. Why brick?"},
        {"speaker": "b", "text": "Totally. Brick lasts."},
        {"speaker": "a", "text": "Did it work?"},
        {"speaker": "b", "text": "Yes. Sometimes."},
        {"speaker": "a", "text": "To sum up, many walls. Is that fair?"},
    ]
    _, issues = check_script(s)
    warnings = {(i.where, i.message.split(";")[0]) for i in issues if i.severity == "warning"}
    assert ("lines[0]", "opens with a greeting") in warnings
    assert ("lines[0]", 'stock phrase "Let\'s dive in"') in warnings
    assert ("lines[1]", "stock phrase 'Great question'") in warnings
    assert ("lines[1]", "stock phrase 'not just a wall, but'") in warnings
    assert ("lines[8]", "ends with a recap") in warnings
    assert any(w == "lines" and "bare agreement" in m for w, m in warnings)
    assert any(w == "hosts" and "'a' ends 5 of 5 turns" in m for w, m in warnings)


def test_craft_rules_catch_a_contrast_across_two_sentences_and_an_announced_takeaway():
    s = copy.deepcopy(SCRIPT)
    s["lines"][0]["text"] = "The map wasn't the clue. It was the pump."
    s["lines"][1]["text"] = "Here's what I'm taking home: water carries it."
    warnings = {(i.where, i.message.split(";")[0]) for i in check_script(s)[1] if i.severity == "warning"}
    assert ("lines[0]", 'stock phrase "wasn\'t the clue. It was"') in warnings
    assert ("lines[1]", "ends with a recap") in warnings


def test_french_craft_rules_use_french_phrases():
    s = copy.deepcopy(SCRIPT)
    s["language"] = "fr"
    s["lines"][0]["text"] = "Bonjour à tous. Bonne question, qui a construit le mur ?"
    s["lines"][1]["text"] = "Voilà ce que je retiens : ce n’est pas un mur, c’est un réseau."
    warnings = {(i.where, i.message.split(";")[0]) for i in check_script(s)[1] if i.severity == "warning"}
    assert ("lines[0]", "opens with a greeting") in warnings
    assert ("lines[0]", "stock phrase 'Bonne question'") in warnings
    assert ("lines[1]", "stock phrase \"ce n'est pas un mur, c'est\"") in warnings
    assert ("lines[1]", "ends with a recap") in warnings


def test_craft_rules_leave_unknown_languages_to_length_checks():
    s = copy.deepcopy(SCRIPT)
    s["language"] = "pl"
    s["lines"][0]["text"] = "Great question. Who built the wall?"
    assert check_script(s)[1] == ()
    long = " ".join(["word"] * 26) + "."
    s["lines"][1]["text"] = long
    assert any("sentence above" in i.message for i in check_script(s)[1])


def test_load_accepts_text_and_raises_with_issues(tmp_path):
    path = tmp_path / "s.json"
    path.write_text('{"title": "x"}')
    with pytest.raises(ScriptError) as caught:
        load_episode(path)
    assert {"hosts", "lines"} <= where(caught.value.issues, "error")
    assert load_episode(SCRIPT).title == "Walls"
