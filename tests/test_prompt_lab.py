from prompt_lab.evaluate import score
from prompt_lab.llm import MockClient
from prompt_lab.parsing import parse_label
from prompt_lab.templates import available_techniques, render


def test_all_four_techniques_exist():
    assert available_techniques() == [
        "chain_of_thought", "few_shot", "role_based", "zero_shot",
    ]


def test_render_inserts_review_text():
    for technique in available_techniques():
        prompt = render(technique, "  The lid leaks.  ")
        assert "The lid leaks." in prompt
        assert "$text" not in prompt


def test_parse_single_word():
    assert parse_label("Positive") == "positive"
    assert parse_label("negative.") == "negative"


def test_parse_prefers_answer_line():
    output = "Some positive words appear, but the complaint dominates.\nAnswer: negative"
    assert parse_label(output) == "negative"


def test_parse_falls_back_to_last_label():
    assert parse_label("Not neutral; overall this is positive") == "positive"


def test_parse_returns_none_when_no_label():
    assert parse_label("I'm not sure.") is None
    assert parse_label("") is None


def test_mock_client_baseline():
    client = MockClient()
    assert client.complete("", "I love it, best ever") == "positive"
    assert client.complete("", "It broke, total waste") == "negative"
    assert client.complete("", "It is blue") == "neutral"


def test_score_counts_accuracy_and_unparsed():
    results = [
        {"expected": "positive", "predicted": "positive", "correct": True},
        {"expected": "negative", "predicted": "", "correct": False},
    ]
    s = score(results)
    assert s["accuracy"] == 0.5
    assert s["unparsed"] == 1
    assert s["per_label"]["neutral"] is None
