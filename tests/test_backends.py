from collections import OrderedDict

import pytest

from qwenjev.backends import LayaBackend, _bool_polarity
from qwenjev.config import JevLimits


class StubAgent:
    """Records what the backend asked for and returns canned answers."""

    def __init__(self, noul=0.8, choice=None):
        self.calls = []
        self.noul = noul
        self.choice = choice or {}

    def predict(self, state, questions):
        self.calls.append((state, questions))
        answers = {}
        for qid, spec in questions.items():
            if spec["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": self.noul, "confidence": 0.9}
            elif spec["type"] == "score":
                answers[qid] = {
                    "type": "score",
                    "score": 1.0,
                    "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2},
                    "confidence": 0.5,
                }
            else:
                keys = list(spec["criteria"])
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[0],
                    "probabilities": self.choice or {keys[0]: 0.7, keys[-1]: 0.3},
                    "confidence": 0.5,
                }
        return {"answers": answers, "usage": {"input_tokens": 42}}


def backend(noul=0.8) -> LayaBackend:
    instance = LayaBackend.__new__(LayaBackend)
    instance.model_path = "stub"
    instance.agent = StubAgent(noul=noul)
    instance.limits = JevLimits()
    return instance


def test_bool_polarity_follows_names_not_positions():
    assert _bool_polarity(["yes", "no"]) == ("yes", "no")
    assert _bool_polarity(["no", "yes"]) == ("yes", "no")
    assert _bool_polarity(["true", "false"]) == ("true", "false")
    # unknown names fall back to last = true, first = false (Laya's own convention)
    assert _bool_polarity(["denied", "granted"]) == ("granted", "denied")


def test_noul_probability_is_not_inverted():
    """Regression: p(true) was once mapped onto the 'no' option."""

    instance = backend(noul=0.8)
    response = instance.decide(
        "thanks for the help",
        {"toxic": {"type": "bool", "instructions": "Toxic?", "claim": "This is toxic."}},
    )
    probabilities = response.decisions["toxic"].probabilities
    assert probabilities["yes"] == pytest.approx(0.8)
    assert probabilities["no"] == pytest.approx(0.2)
    assert response.decisions["toxic"].answer == "yes"


def test_claim_is_placed_at_the_head_of_the_state():
    instance = backend()
    instance.decide(
        "the comment text",
        {"toxic": {"type": "bool", "instructions": "Toxic?", "claim": "This comment is toxic."}},
    )
    state, questions = instance.agent.calls[0]
    assert state["message"].startswith("This comment is toxic.")
    assert state["message"].endswith("the comment text")
    assert questions["toxic"]["type"] == "noul"
    assert questions["toxic"]["criteria"]["true"] == "Yes"


def test_choice_questions_keep_their_options_and_get_layas_instruction():
    instance = backend()
    instance.decide(
        "the text",
        {
            "answer": {
                "type": "choice",
                "instructions": "Which option is correct?",
                "criteria": {"A": "first", "B": "second"},
            }
        },
    )
    _, questions = instance.agent.calls[0]
    assert questions["answer"]["criteria"] == {"A": "first", "B": "second"}
    assert "best applies to `message`" in questions["answer"]["instructions"]
    assert "Which option is correct?" in questions["answer"]["instructions"]


def test_score_questions_become_level_lists():
    instance = backend()
    response = instance.decide(
        "the text",
        {
            "severity": {
                "type": "score",
                "instructions": "How bad?",
                "criteria": {"low": "Low", "mid": "Mid", "high": "High"},
            }
        },
    )
    _, questions = instance.agent.calls[0]
    assert questions["severity"]["criteria"] == ["Low", "Mid", "High"]
    assert response.decisions["severity"].probabilities["mid"] == pytest.approx(0.6)


def test_usage_and_timing_are_reported():
    instance = backend()
    response = instance.decide(
        "text",
        {
            "a": {"type": "bool", "instructions": "?", "claim": "c"},
            "b": {"type": "bool", "instructions": "?", "claim": "c"},
        },
    )
    assert response.usage["input_tokens"] == 84
    assert response.usage["requests"] == 2
    assert response.timing["total_ms"] >= 0
