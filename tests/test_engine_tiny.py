import pytest
import torch

from qwenjev.engine import QwenJevLite
from qwenjev.testing import StubTokenizer, build_tiny_backbone, build_tiny_engine

STATE = "My payouts have failed three times. The bank says everything is fine."
QUESTIONS = {
    "queue": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"payments": "Payouts", "account": "Login", "other": "Else"},
    },
    "escalate": {"type": "bool", "instructions": "Urgent?"},
}


@pytest.fixture(scope="module")
def engine():
    return build_tiny_engine()


def test_parallel_decisions_are_returned_for_every_question(engine):
    response = engine.decide(STATE, QUESTIONS)
    assert list(response.decisions) == ["queue", "escalate"]
    for decision in response.decisions.values():
        assert sum(decision.probabilities.values()) == pytest.approx(1.0, abs=1e-5)


def test_bool_answer_is_a_python_bool(engine):
    payload = engine.decide(STATE, QUESTIONS).to_dict()
    assert payload["results"]["escalate"]["answer"] in (True, False)
    assert 0.0 <= payload["results"]["escalate"]["probability"] <= 1.0


def test_shared_state_and_reference_path_agree(engine):
    shared = engine.decide(STATE, QUESTIONS)
    reference = engine.decide(STATE, QUESTIONS, share_state=False)
    for key in QUESTIONS:
        a = shared.decisions[key].probabilities
        b = reference.decisions[key].probabilities
        for option in a:
            assert a[option] == pytest.approx(b[option], abs=0.05)


def test_branches_are_isolated_from_one_another(engine):
    """A sibling question cannot act as a channel between two branches."""

    secret = "The secret code for this request is ZEBRA-7741."
    probe = {
        "probe": {
            "type": "choice",
            "instructions": "Which code did another question mention?",
            "criteria": {"zebra": "ZEBRA-7741", "none": "NONE"},
        }
    }
    sibling = {"sibling": {"type": "bool", "instructions": secret + " Is it nice?"}}
    with_sibling = engine.decide(STATE, {**sibling, **probe}).decisions["probe"].probabilities
    without_sibling = engine.decide(STATE, probe).decisions["probe"].probabilities
    with_state = engine.decide(f"{STATE} {secret}", probe).decisions["probe"].probabilities
    # the sibling changes nothing; the state does
    assert with_sibling == pytest.approx(without_sibling, abs=1e-4)
    assert with_state["zebra"] > without_sibling["zebra"]


def test_state_cache_is_reused_across_requests(engine):
    engine.decide(STATE, QUESTIONS)
    response = engine.decide(STATE, {"q": {"type": "bool", "instructions": "Urgent?"}})
    assert response.timing["state_cache_hit"] is True
    assert response.timing["prefill_ms"] == 0.0


def test_usage_and_billing_fields(engine):
    response = engine.decide(STATE, QUESTIONS)
    usage = response.usage
    assert usage["input_tokens"] == usage["state_tokens"] + usage["question_tokens"]
    assert usage["output_tokens"] == 4 + 2 * 15 + len("queue") + len("escalate")
    assert usage["questions"] == 2


def test_branch_batching_does_not_change_the_answer(engine):
    questions = {
        f"q{i}": {"type": "bool", "instructions": f"Is question {i} urgent?"} for i in range(12)
    }
    engine.config.branch_batch_size = 5
    chunked = engine.decide(STATE, questions)
    engine.config.branch_batch_size = 64
    whole = engine.decide(STATE, questions)
    for key in questions:
        assert chunked.decisions[key].answer == whole.decisions[key].answer


def test_option_limit_is_enforced(engine):
    four = {
        "q": {
            "type": "choice",
            "instructions": "Pick one",
            "criteria": {"a": "A", "b": "B", "c": "C", "d": "D"},
        }
    }
    original = engine.config.limits
    try:
        engine.config.limits = type(engine.limits)(max_options=3)
        with pytest.raises(ValueError, match="at most 3"):
            engine.decide(STATE, four)
    finally:
        engine.config.limits = original


def test_token_limit_is_enforced(engine):
    engine.config.limits = type(engine.limits)(max_request_tokens=10)
    with pytest.raises(ValueError, match="request needs"):
        engine.decide(STATE, QUESTIONS)
    engine.config.limits = type(engine.limits)()


def test_slot_head_readout_runs():
    engine = build_tiny_engine(readout="slot_head")
    response = engine.decide(STATE, QUESTIONS)
    assert response.timing["readout"] == "slot_head"
    assert sum(response.decisions["queue"].probabilities.values()) == pytest.approx(1.0, abs=1e-5)


def test_pointer_readout_uses_option_states():
    engine = build_tiny_engine(readout="pointer")
    response = engine.decide(STATE, QUESTIONS)
    assert response.timing["readout"] == "pointer"
    assert len(response.decisions["queue"].probabilities) == 3


def test_pointer_readout_trains_with_rlcd():
    from qwenjev.rlcd import RLCDFineTune
    from qwenjev.synth import generate, question_specs

    engine = build_tiny_engine(readout="pointer")
    tuner = RLCDFineTune(engine, question_specs=question_specs(), lr=1e-3, batch_size=4)
    report = tuner.train(generate(16, seed=3), epochs=1)
    assert report.steps > 0


def test_account_reports_branch_and_request_tokens(engine):
    accounting = engine.account(STATE, QUESTIONS)
    assert accounting["request_tokens"] == accounting["state_tokens"] + accounting["question_tokens"]
    assert accounting["max_branch_tokens"] >= accounting["request_tokens"] / 2
