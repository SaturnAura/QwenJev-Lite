import pytest

from qwenjev.prompt import render_branch, render_state
from qwenjev.schema import build_question
from qwenjev.testing import StubTokenizer
from qwenjev.tokenize import tokenize_branch

QUESTION = {
    "type": "choice",
    "instructions": "Which team should handle this ticket?",
    "criteria": {
        "payments": "Payout failures and payment processing",
        "account": "Login and account access",
        "other": "Something else",
    },
}


def test_branch_layout():
    branch = render_branch(build_question("queue", QUESTION))
    assert branch.text.startswith("Question: Which team should handle this ticket?\nOptions:\n")
    assert "A. Payout failures and payment processing\n" in branch.text
    assert branch.text.endswith("Answer: (")
    assert branch.labels == ["A", "B", "C"]


def test_option_character_spans_point_at_the_descriptions():
    branch = render_branch(build_question("queue", QUESTION))
    for (start, end), option in zip(branch.option_char_spans, QUESTION["criteria"].values()):
        assert branch.text[start:end] == option


def test_token_spans_and_decision_index():
    tokenizer = StubTokenizer()
    rendered = render_branch(build_question("queue", QUESTION))
    branch = tokenize_branch(tokenizer, rendered)
    assert len(branch) == len(rendered.text)
    assert branch.decision_index == len(branch.ids) - 1
    for (start, end), option in zip(branch.option_spans, QUESTION["criteria"].values()):
        assert "".join(chr(i) for i in branch.ids[start : end + 1]) == option


def test_state_wrapper():
    assert render_state("  hello  ") == "<state>\nhello\n</state>\n"


def test_score_branch_uses_levels_header():
    rendered = render_branch(
        build_question(
            "sev",
            {"type": "score", "instructions": "Severity?", "criteria": {"low": "Low", "high": "High"}},
        )
    )
    assert "Levels:" in rendered.text
    assert "naming the best level" in rendered.text


def test_bool_branch_lists_its_two_criteria():
    rendered = render_branch(build_question("esc", {"type": "bool", "instructions": "Urgent?"}))
    assert "A. Yes" in rendered.text
    assert "B. No" in rendered.text
