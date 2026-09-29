import numpy as np
import pytest

from qwenjev.rlcd import RLCDFineTune, brier_loss, fit_temperature, log_loss
from qwenjev.synth import generate, question_specs
from qwenjev.testing import build_tiny_engine


def test_each_answer_gets_its_own_row_and_unknown_answers_keep_the_reserved_rows():
    """The fix for "training made things worse": rows are addressed per answer.

    Two questions about the same state with different label spaces must train and read
    different rows, and an answer the table never saw must still be read from the
    pretrained label rows rather than from whatever a task happened to train.
    """

    import torch

    from qwenjev.rlcd import build_slot_table
    from qwenjev.schema import (
        SLOT_RESERVE,
        SLOT_UNKNOWN_ROW,
        build_question,
        option_ids,
        question_option_ids,
    )

    engine = build_tiny_engine(readout="slot_head")
    specs = question_specs()  # queue: payments/account/other, escalate: yes/no
    samples = generate(8, seed=3, question="queue") + generate(8, seed=4, question="escalate")
    table, counts = build_slot_table(samples, question_specs=specs, max_slots=engine.readout.max_slots)

    queue_ids = question_option_ids(build_question("queue", specs["queue"]))
    escalate_ids = question_option_ids(build_question("escalate", specs["escalate"]))
    rows = [table[answer] for answer in queue_ids + escalate_ids]
    assert len(set(rows)) == len(rows), "answer rows must not overlap"
    assert min(rows) >= SLOT_RESERVE, "trained rows must stay off the reserved letters"
    assert all(counts[answer] == 8 for answer in queue_ids)
    assert all(counts[answer] == 8 for answer in escalate_ids)

    engine.readout.set_slot_table(table)
    # An unseen answer falls back to its position's reserved row; a seen one does not.
    assert engine.readout.row_for(None, 0) == 0
    assert engine.readout.row_for(None, 3) == 3
    assert engine.readout.row_for("\x1fnot-a-real-answer", 0) == 0
    assert engine.readout.row_for(None, 40) == SLOT_UNKNOWN_ROW
    assert engine.readout.row_for(queue_ids[0], 2) == table[queue_ids[0]]

    # The same label space in a different option order reads the same rows.
    reordered = option_ids("choice", ["other", "payments", "account"], question_id="queue")
    assert {answer.split("\x1f")[-1]: table[answer] for answer in reordered} == {
        answer.split("\x1f")[-1]: table[answer] for answer in queue_ids
    }
    # A question that offers the same answers is still a different set of rows.
    same_answers_other_question = option_ids(
        "bool", ["yes", "no"], question_id="escalate"
    )
    same_answers_queue = option_ids("bool", ["yes", "no"], question_id="queue")
    assert all("\x1fescalate\x1f" in answer for answer in same_answers_other_question)
    assert not set(same_answers_other_question) & set(same_answers_queue)


def test_training_one_label_space_leaves_another_label_space_alone():
    """Train on one question type, then check the other type's rows never moved."""

    import copy

    import torch

    from qwenjev.rlcd import build_slot_table
    from qwenjev.schema import build_question, question_option_ids

    engine = build_tiny_engine(readout="slot_head")
    specs = question_specs()
    samples = generate(32, seed=9, question="queue") + generate(32, seed=10, question="escalate")
    table, _counts = build_slot_table(samples, question_specs=specs, max_slots=engine.readout.max_slots)
    engine.readout.set_slot_table(table)

    escalate_rows = [table[answer] for answer in question_option_ids(build_question("escalate", specs["escalate"]))]
    before = copy.deepcopy(engine.readout.weight[escalate_rows])

    queue_only = [sample for sample in samples if sample.question_id == "queue"]
    tuner = RLCDFineTune(engine, question_specs=specs, lr=1e-1, batch_size=16, anchor=0.0)
    tuner.train(queue_only, epochs=5)

    after = engine.readout.weight[escalate_rows]
    assert torch.equal(before, after), "training 'queue' moved rows that 'escalate' reads"
    queue_rows = [table[answer] for answer in question_option_ids(build_question("queue", specs["queue"]))]
    assert not torch.equal(before[:1], engine.readout.weight[queue_rows[:1]])


def test_prototype_rows_are_the_mean_state_of_each_answer():
    """The closed-form initialiser reads one row per answer off a single forward pass."""

    import torch

    from qwenjev.rlcd import build_slot_table
    from qwenjev.schema import build_question, question_option_ids

    engine = build_tiny_engine(readout="slot_head")
    specs = question_specs()
    samples = generate(24, seed=13, question="escalate")
    table, _counts = build_slot_table(samples, question_specs=specs, max_slots=engine.readout.max_slots)
    engine.readout.set_slot_table(table)
    tuner = RLCDFineTune(engine, question_specs=specs, batch_size=8, optimizer="sgd")

    rows = tuner.prototypes(samples)
    ids = question_option_ids(build_question("escalate", specs["escalate"]))
    assert set(rows) == set(ids), "every answer that was chosen needs a row"
    for answer in ids:
        assert rows[answer].shape == (engine.readout.proj.weight.shape[1],)
    # Centring means the two rows of a label space point against each other.
    cosine = float(
        torch.nn.functional.cosine_similarity(rows[ids[0]], rows[ids[1]], dim=0)
    )
    assert cosine < 0

    written = engine.readout.load_rows(rows, cap=0.5)
    assert written == len(ids)
    norms = torch.stack([engine.readout.weight[table[answer]].norm() for answer in ids])
    assert torch.allclose(norms.mean(), torch.tensor(0.5), atol=1e-5)


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
