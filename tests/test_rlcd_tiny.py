import numpy as np
import pytest

from qwenjev.rlcd import RLCDFineTune, brier_loss, fit_temperature, log_loss
from qwenjev.synth import generate, question_specs
from qwenjev.testing import build_tiny_engine


def test_proper_scoring_rules_are_minimised_by_the_true_distribution():
    import torch

    targets = torch.tensor([0, 0, 1])
    # p(true) = 0.8 for these labels
    good = torch.log(torch.tensor([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8]]))
    bad = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]))
    assert log_loss(good, targets) < log_loss(bad, targets)
    assert brier_loss(good.exp(), targets) < brier_loss(bad.exp(), targets)


def test_training_reduces_the_loss_on_synthetic_tickets():
    engine = build_tiny_engine(readout="slot_head")
    specs = question_specs()
    train = generate(32, seed=11, question="queue")
    # One batch per step on a randomly initialised backbone: a noisy mini-batch order
    # makes the start/end comparison meaningless at this size.
    tuner = RLCDFineTune(engine, question_specs=specs, lr=1e-2, batch_size=32)
    before = -np.log(tuner.evaluate(train).probabilities).mean()
    report = tuner.train(train, epochs=20)
    after = -np.log(tuner.evaluate(train).probabilities).mean()
    assert report.steps == 20
    assert after < before
    first = float(np.mean(report.history[:4]))
    last = float(np.mean(report.history[-4:]))
    assert last < first
    assert report.loss < before


def test_evaluation_reports_reliability_fields():
    engine = build_tiny_engine(readout="slot_head")
    specs = question_specs()
    tuner = RLCDFineTune(engine, question_specs=specs, batch_size=8)
    evaluation = tuner.evaluate(generate(16, seed=5, question="queue"))
    summary = evaluation.reliability()
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert 0.0 <= summary["ece"] <= 1.0
    assert summary["brier"] >= 0.0
    assert summary["nll"] > 0.0
    assert 0.0 <= summary["mean_probability_of_correct_answer"] <= 1.0
    assert len(evaluation.confusion()) >= 1


def test_temperature_fitting_returns_a_positive_scale():
    engine = build_tiny_engine(readout="slot_head")
    specs = question_specs()
    tuner = RLCDFineTune(engine, question_specs=specs, batch_size=8)
    evaluation = tuner.evaluate(generate(16, seed=7, question="escalate"))
    temperature = fit_temperature(engine, evaluation)
    assert temperature > 0
    assert np.isfinite(temperature)


def test_reserved_label_readout_cannot_be_trained():
    engine = build_tiny_engine(readout="reserved_label")
    with pytest.raises(ValueError, match="no trainable parameters"):
        RLCDFineTune(engine, question_specs=question_specs())
