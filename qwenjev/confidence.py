"""Arithmetic summary fields, kept separate from the trained distribution.

The essay (section 5) reports the official adapter's formula for ``Choice``::

    c = (pmax - 1/K) / (1 - 1/K)

"For three options with a maximum probability of 0.8, this gives 0.7. The adapter
handles the one-option case separately, returning 1." The ``Score`` type "uses a
different formula reflecting distance from the modal level"; the essay does not
publish it, so :func:`score_confidence` is our reconstruction and is labelled as
such.
"""

from __future__ import annotations

from typing import Mapping, Sequence


def choice_confidence(probabilities: Sequence[float]) -> float:
    """Adapter confidence for a finite choice: how far the leader is above uniform."""

    k = len(probabilities)
    if k == 0:
        raise ValueError("no options")
    if k == 1:
        return 1.0
    pmax = float(max(probabilities))
    return (pmax - 1.0 / k) / (1.0 - 1.0 / k)


def bool_confidence(p_yes: float) -> float:
    """A yes/no decision is a two-option choice."""

    return choice_confidence([float(p_yes), 1.0 - float(p_yes)])


def score_confidence(probabilities: Sequence[float]) -> float:
    """Reconstruction: 1 minus the normalised distance from the modal level.

    The modal level is the highest-probability level; tied modes are averaged so the
    field stays symmetric on a flat distribution.
    """

    k = len(probabilities)
    if k == 0:
        raise ValueError("no levels")
    if k == 1:
        return 1.0
    probs = [float(p) for p in probabilities]
    top = max(probs)
    tied = [i for i, p in enumerate(probs) if p == top]
    modal = sum(tied) / len(tied)
    expected_distance = sum(p * abs(i - modal) for i, p in enumerate(probs))
    return 1.0 - expected_distance / (k - 1)


def normalised_entropy(probabilities: Sequence[float]) -> float:
    """Entropy of the distribution divided by ``log K`` (1.0 = uniform)."""

    import math

    probs = [float(p) for p in probabilities if p > 0]
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy / math.log(len(probabilities))


def billing_output_tokens(question_ids: Sequence[str], tokenizer) -> int:
    """The ``output_tokens`` billing figure.

    "For yes/no questions, the count fits exactly: 4 shared tokens, plus 15 per
    answer, plus the token length of each question's identifier." The identifier is
    never sent to the model.
    """

    total = 4
    for qid in question_ids:
        total += 15 + len(tokenizer.encode(str(qid), add_special_tokens=False))
    return total


def summarise(kind: str, probabilities: Mapping[str, float]) -> float:
    values = list(probabilities.values())
    if kind == "score":
        return score_confidence(values)
    if kind == "bool":
        return bool_confidence(values[0] if len(values) == 2 else values[0])
    return choice_confidence(values)
