"""Reliability analysis, following the essay's methods section.

"The MMLU calibration figure uses ten equal-width bins: [0, 0.1), [0.1, 0.2), and
so on, with 1.0 included in the last bin. Expected calibration error is the
sample-weighted absolute difference between accuracy and average top probability in
each bin."  (:mod:`qwenjev.probes` and the CLI reuse these helpers.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Bin:
    lower: float
    upper: float
    count: int
    accuracy: float
    mean_probability: float

    @property
    def gap(self) -> float:
        return abs(self.accuracy - self.mean_probability)


def reliability_bins(probabilities, correct, n_bins: int = 10) -> list[Bin]:
    """Equal-width bins over ``[0, 1]``, with 1.0 folded into the last bin."""

    probs = np.asarray(probabilities, dtype=float)
    hits = np.asarray(correct, dtype=bool)
    if probs.shape != hits.shape:
        raise ValueError("probabilities and correct must have the same shape")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[Bin] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            sel = (probs >= lo) & (probs <= hi)
        else:
            sel = (probs >= lo) & (probs < hi)
        count = int(sel.sum())
        bins.append(
            Bin(
                lower=float(lo),
                upper=float(hi),
                count=count,
                accuracy=float(hits[sel].mean()) if count else 0.0,
                mean_probability=float(probs[sel].mean()) if count else 0.0,
            )
        )
    return bins


def expected_calibration_error(probabilities, correct, n_bins: int = 10) -> float:
    bins = reliability_bins(probabilities, correct, n_bins=n_bins)
    total = sum(b.count for b in bins)
    if total == 0:
        raise ValueError("no observations")
    return sum(b.count * b.gap for b in bins) / total


def mean_top_probability(probabilities) -> float:
    probs = np.asarray(probabilities, dtype=float)
    return float(probs.mean()) if probs.size else float("nan")


def accuracy(probabilities, correct) -> float:
    hits = np.asarray(correct, dtype=bool)
    return float(hits.mean()) if hits.size else float("nan")


def brier_score(probabilities, correct) -> float:
    """Binary Brier score for a stated probability of being correct."""

    probs = np.asarray(probabilities, dtype=float)
    hits = np.asarray(correct, dtype=float)
    return float(np.mean((probs - hits) ** 2))


def multiclass_brier(probability_rows, targets) -> float:
    """Proper multiclass Brier score: squared distance to the one-hot outcome.

    The essay's section 5: "Brier loss, the squared distance between the predicted
    distribution and the observed one-hot outcome."
    """

    total = 0.0
    for row, target in zip(probability_rows, targets):
        row = np.asarray(row, dtype=float)
        one_hot = np.zeros_like(row)
        one_hot[int(target)] = 1.0
        total += float(((row - one_hot) ** 2).sum())
    return total / max(len(probability_rows), 1)


def negative_log_likelihood(probabilities, correct=None) -> float:
    """``-log p`` of the observed outcome.

    ``probabilities`` is the probability assigned to the outcome that happened; for
    binary outcomes pass the probability of the positive class together with
    ``correct``.
    """

    probs = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    if correct is None:
        return float(-np.mean(np.log(probs)))
    hits = np.asarray(correct, dtype=bool)
    observed = np.where(hits, probs, 1.0 - probs)
    return float(-np.mean(np.log(np.clip(observed, 1e-12, 1.0))))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval - the error bars used in the essay's figures."""

    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


class TemperatureScaler:
    """Post-hoc calibration (Guo et al. 2017) applied to the readout logits."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def probabilities(self, logits) -> np.ndarray:
        z = np.asarray(logits, dtype=float) / self.temperature
        z = z - z.max(axis=-1, keepdims=True)
        exp = np.exp(z)
        return exp / exp.sum(axis=-1, keepdims=True)

    @staticmethod
    def _nll(logit_rows, targets, temperature: float) -> float:
        """Mean NLL; every row may have its own number of options."""

        total = 0.0
        for row, target in zip(logit_rows, targets):
            z = np.asarray(row, dtype=float) / max(temperature, 1e-6)
            z = z - z.max()
            exp = np.exp(z)
            total += -math.log(max(float(exp[target] / exp.sum()), 1e-12))
        return total / max(len(logit_rows), 1)

    def fit(self, logits, targets, iterations: int = 200) -> "TemperatureScaler":
        """Fit ``T`` by minimising negative log likelihood (golden-section search).

        ``logits`` may be a rectangular ``(N, K)`` array, or a list of per-row lists
        when the questions offer different numbers of options.
        """

        logit_rows = [list(np.asarray(row, dtype=float)) for row in logits]
        targets = [int(t) for t in targets]

        def nll(t: float) -> float:
            return self._nll(logit_rows, targets, t)

        lo, hi = 0.01, 100.0
        phi = (math.sqrt(5) - 1) / 2
        a, b = lo, hi
        c, d = b - phi * (b - a), a + phi * (b - a)
        fc, fd = nll(c), nll(d)
        for _ in range(iterations):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = nll(c)
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = nll(d)
            if b - a < 1e-4:
                break
        self.temperature = float((a + b) / 2)
        return self


def summarise_reliability(probabilities, correct, n_bins: int = 10) -> dict:
    """Reliability of the *chosen* answer, following the essay's figure.

    ``probabilities`` is the probability the model gave the answer it returned;
    ``correct`` says whether that answer was right. Extra fields (NLL, multiclass
    Brier, the probability of the correct answer) are added by
    :meth:`qwenjev.rlcd.Evaluation.reliability`, which has the full distributions.
    """

    bins = reliability_bins(probabilities, correct, n_bins=n_bins)
    successes = int(np.asarray(correct, dtype=bool).sum())
    total = len(correct)
    lo, hi = wilson_interval(successes, total)
    return {
        "n": total,
        "accuracy": accuracy(probabilities, correct),
        "mean_top_probability": mean_top_probability(probabilities),
        "ece": expected_calibration_error(probabilities, correct, n_bins=n_bins),
        "wilson95": [lo, hi],
        "bins": [
            {
                "range": [b.lower, b.upper],
                "count": b.count,
                "accuracy": b.accuracy,
                "mean_probability": b.mean_probability,
            }
            for b in bins
        ],
    }
