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


def test_load_accepts_text_and_raises_with_issues(tmp_path):
    path = tmp_path / "s.json"
    path.write_text('{"title": "x"}')
    with pytest.raises(ScriptError) as caught:
        load_episode(path)
    assert {"hosts", "lines"} <= where(caught.value.issues, "error")
    assert load_episode(SCRIPT).title == "Walls"
