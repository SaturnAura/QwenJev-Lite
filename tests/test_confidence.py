import pytest

from qwenjev.confidence import (
    billing_output_tokens,
    bool_confidence,
    choice_confidence,
    normalised_entropy,
    score_confidence,
)
from qwenjev.testing import StubTokenizer


def test_documented_choice_example():
    """'For three options with a maximum probability of 0.8, this gives 0.7.'"""

    assert choice_confidence([0.8, 0.1, 0.1]) == pytest.approx(0.7)


def test_single_option_returns_one():
    assert choice_confidence([1.0]) == 1.0


def test_confidence_is_zero_for_uniform():
    assert choice_confidence([0.25] * 4) == pytest.approx(0.0)


def test_bool_confidence_is_the_two_option_case():
    assert bool_confidence(0.9) == pytest.approx(0.8)


def test_score_confidence_peaks_on_a_modal_level():
    assert score_confidence([0.0, 1.0, 0.0]) == pytest.approx(1.0)
    assert score_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert score_confidence([0.25, 0.25, 0.25, 0.25]) == pytest.approx(1 - 1.0 / 3.0 * 1.0)


def test_entropy_extremes():
    assert normalised_entropy([1.0]) == 0.0
    assert normalised_entropy([0.5, 0.5]) == pytest.approx(1.0)


def test_billing_figure_matches_the_documented_recipe():
    tokenizer = StubTokenizer()
    # 4 shared tokens + 15 per answer + the token length of the identifier
    assert billing_output_tokens(["a"], tokenizer) == 4 + 15 + 1
    assert billing_output_tokens(["aa", "b"], tokenizer) == 4 + 30 + 3
    assert billing_output_tokens(["queue"], tokenizer) == 4 + 15 + 5
