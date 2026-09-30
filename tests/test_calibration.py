import numpy as np
import pytest

from qwenjev.calibration import (
    TemperatureScaler,
    expected_calibration_error,
    multiclass_brier,
    negative_log_likelihood,
    reliability_bins,
    summarise_reliability,
    wilson_interval,
)


def test_bins_cover_the_unit_interval_including_one():
    probs = [0.0, 0.05, 0.95, 1.0]
    correct = [False, False, True, True]
    bins = reliability_bins(probs, correct, n_bins=10)
    assert bins[0].count == 2
    assert bins[-1].count == 2
    assert sum(b.count for b in bins) == 4


def test_perfectly_calibrated_data_has_low_ece():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.05, 0.95, size=20000)
    correct = rng.random(20000) < probs
    assert expected_calibration_error(probs, correct, n_bins=10) < 0.02


def test_overconfidence_shows_up_as_ece():
    probs = np.full(1000, 0.9)
    correct = np.zeros(1000, dtype=bool)
    assert expected_calibration_error(probs, correct) == pytest.approx(0.9, abs=1e-6)


def test_temperature_scaling_improves_calibration():
    logits = np.array([[4.0, 0.0], [3.0, 0.0], [5.0, 0.0], [2.0, 0.0]])
    targets = np.array([1, 0, 0, 1])  # deliberately adversarial
    before = TemperatureScaler()
    probabilities = before.probabilities(logits)[np.arange(len(targets)), targets]
    after = TemperatureScaler().fit(logits, targets)
    calibrated = after.probabilities(logits)[np.arange(len(targets)), targets]
    assert after.temperature > 1.0
    assert np.mean(-np.log(calibrated)) <= np.mean(-np.log(probabilities)) + 1e-9


def test_wilson_interval_brackets_the_estimate():
    lo, hi = wilson_interval(8, 10)
    assert lo < 0.8 < hi
    assert wilson_interval(0, 0)[0] != wilson_interval(0, 0)[1]


def test_summary_reports_the_expected_fields():
    summary = summarise_reliability([0.9, 0.2, 0.8], [True, False, True])
    assert set(summary) >= {"n", "accuracy", "mean_top_probability", "ece", "wilson95", "bins"}


def test_ece_is_measured_against_the_chosen_answer():
    """Confidently wrong answers must show up, not cancel out."""

    # three options, the model always answers "A" with 0.9 and is right half the time
    probabilities = [0.9, 0.9]
    correct = [True, False]
    assert expected_calibration_error(probabilities, correct) == pytest.approx(0.4)


def test_multiclass_brier_penalises_confident_errors():
    rows = [[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]]
    right = multiclass_brier(rows, [0, 0])
    wrong = multiclass_brier(rows, [1, 1])
    assert right < 0.1
    assert wrong > 1.5


def test_nll_of_the_true_answer():
    assert negative_log_likelihood([1.0, 0.5]) == pytest.approx(0.3466, abs=1e-4)
