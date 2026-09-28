import pytest

from qwenjev.schema import (
    BoolQuestion,
    ChoiceQuestion,
    ScoreQuestion,
    build_question,
)


def test_choice_keeps_declaration_order():
    question = build_question(
        "queue",
        {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {"payments": "Payouts", "account": "Login", "other": "Else"},
        },
    )
    assert isinstance(question, ChoiceQuestion)
    assert [o.key for o in question.options] == ["payments", "account", "other"]
    assert question.labels() == ["A", "B", "C"]


def test_bool_defaults_to_yes_no():
    question = build_question("escalate", {"type": "bool", "instructions": "Urgent?"})
    assert isinstance(question, BoolQuestion)
    assert [o.key for o in question.options] == ["yes", "no"]


def test_documented_alias_noul_is_a_bool():
    question = build_question(
        "escalate", {"type": "noul", "instructions": "Does this need a human?"}
    )
    assert isinstance(question, BoolQuestion)


def test_score_gets_values_and_expectation():
    question = build_question(
        "severity",
        {
            "type": "score",
            "instructions": "How severe is this?",
            "criteria": {"low": "Low", "medium": "Medium", "high": "High"},
        },
    )
    assert isinstance(question, ScoreQuestion)
    assert [o.value for o in question.options] == [0.0, 1.0, 2.0]
    value = question.expected_value({"low": 0.1, "medium": 0.3, "high": 0.6})
    assert value == pytest.approx(1.5)


def test_explicit_values_are_respected():
    question = build_question(
        "carbon",
        {
            "type": "score",
            "instructions": "Carbon price?",
            "criteria": {
                "low": {"description": "Low", "value": 10},
                "high": {"description": "High", "value": 90},
            },
        },
    )
    assert [o.value for o in question.options] == [10.0, 90.0]
    assert question.expected_value({"low": 0.25, "high": 0.75}) == pytest.approx(70.0)


def test_missing_instructions_rejected():
    with pytest.raises(ValueError):
        build_question("q", {"type": "choice", "criteria": {"a": "A"}})


def test_unknown_type_rejected():
    with pytest.raises(ValueError):
        build_question("q", {"type": "regression", "instructions": "?"})


def test_bool_needs_two_criteria():
    with pytest.raises(ValueError):
        build_question(
            "q",
            {"type": "bool", "instructions": "?", "criteria": {"a": "A", "b": "B", "c": "C"}},
        )
