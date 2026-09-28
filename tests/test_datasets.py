from pathlib import Path

import pytest

from qwenjev.datasets import (
    DATASETS,
    DatasetUnavailable,
    default_root,
    items_to_samples,
    load_fever,
    load_hwu64,
    load_items,
    load_jigsaw,
    load_mmlu,
    status,
)
from qwenjev.evaluation import evaluate_items
from qwenjev.testing import build_tiny_engine

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ------------------------------------------------------------------------ FEVER
def test_fever_uses_the_evidence_file_and_falls_back_to_the_claim():
    items = load_fever(FIXTURES / "fever", split="validation")
    assert len(items) == 3
    by_id = {item.item_id: item for item in items}
    assert by_id["1"].meta["has_evidence"] is True
    assert "Champ de Mars" in by_id["1"].state
    assert by_id["2"].targets["evidence"] == "refutes"
    assert by_id["3"].meta["has_evidence"] is False
    assert by_id["3"].state == by_id["3"].meta["claim"]
    spec = by_id["1"].questions["evidence"]
    assert list(spec["criteria"]) == ["supports", "refutes", "not_enough_info"]


def test_fever_treats_bare_annotations_as_no_evidence(tmp_path):
    """Raw FEVER rows point at page#sentence, which is not readable evidence."""

    root = tmp_path / "fever"
    root.mkdir()
    (root / "validation.jsonl").write_text(
        '{"id": 9, "label": "SUPPORTS", "claim": "A claim.", '
        '"evidence": [[["Some_Page", 0]]]}\n',
        encoding="utf-8",
    )
    items = load_fever(root, split="validation")
    assert items[0].meta["has_evidence"] is False
    assert items[0].state == "A claim."


# ------------------------------------------------------------------ MMLU / Pro
def test_mmlu_accepts_index_and_letter_answers():
    items = load_mmlu(FIXTURES / "mmlu", split="test")
    targets = [item.targets["answer"] for item in items]
    assert targets == ["B", "B", "C"]
    first = items[0]
    assert first.questions["answer"]["criteria"]["B"] == "4"
    assert first.state.startswith("Subject: elementary_mathematics")
    assert first.questions["answer"]["instructions"] == "Which option is correct?"


def test_mmlu_pro_style_options_and_mode():
    items = load_mmlu(FIXTURES / "mmlu_pro", split="pro_sample", state_mode="context")
    assert len(items) == 1
    assert len(items[0].questions["answer"]["criteria"]) == 10
    assert items[0].state == "Subject: math"
    assert items[0].targets["answer"] == "C"


# --------------------------------------------------------------- CLINC150 / HWU64
def test_clinc150_reads_the_official_split_shape():
    items = load_items("clinc150", root=FIXTURES / "clinc150", split="test")
    assert len(items) == 4
    criteria = items[0].questions["intent"]["criteria"]
    assert set(criteria) == {"weather", "play_music", "restaurant_reservation"}
    assert criteria["play_music"] == "play music"
    assert items[0].targets["intent"] == "weather"


def test_clinc150_oos_split_is_available():
    items = load_items("clinc150", root=FIXTURES / "clinc150", split="oos_test")
    assert [item.targets["intent"] for item in items] == ["oos", "oos"]


def test_clinc150_can_limit_the_label_space():
    items = load_items(
        "clinc150", root=FIXTURES / "clinc150", split="test", max_intents=2
    )
    assert len(items) == 3  # the test split minus the rows of the dropped intent
    criteria = set(items[0].questions["intent"]["criteria"])
    assert len(criteria) == 2
    for item in items:
        assert set(item.targets) == {"intent"}
        assert item.targets["intent"] in criteria


def test_every_target_is_one_of_the_offered_criteria():
    for key, root, split in (
        ("fever", FIXTURES / "fever", "validation"),
        ("mmlu", FIXTURES / "mmlu", "test"),
        ("clinc150", FIXTURES / "clinc150", "test"),
        ("hwu64", FIXTURES / "hwu64", "test"),
        ("jigsaw", FIXTURES / "jigsaw", "test"),
    ):
        items = load_items(key, root=root, split=split)
        assert items, key
        for item in items:
            for question_id, spec in item.questions.items():
                target = item.targets[question_id]
                if spec["type"] == "bool":
                    assert target in {"yes", "no"}, (key, item.item_id, question_id, target)
                else:
                    assert target in spec["criteria"], (key, item.item_id, question_id, target)


def test_hwu64_asks_intent_and_domain():
    items = load_hwu64(FIXTURES / "hwu64", split="test", include_domain=True)
    assert len(items) == 4
    first = items[0]
    assert first.questions["intent"]["criteria"]["alarm_set"] == "alarm set"
    assert first.targets == {"intent": "alarm_set", "domain": "alarm"}
    assert len(first.questions["domain"]["criteria"]) == 3


# ---------------------------------------------------------------------- Jigsaw
def test_jigsaw_joins_the_label_file_and_asks_six_questions():
    items = load_jigsaw(FIXTURES / "jigsaw", split="test")
    assert len(items) == 3
    assert len(items[0].questions) == 6
    assert items[0].targets["toxic"] == "no"
    assert items[1].targets["toxic"] == "yes"
    assert items[1].targets["identity_hate"] == "no"
    assert items[2].targets["threat"] == "yes"
    assert items[2].questions["threat"]["instructions"].startswith("Does this comment contain threat")


# --------------------------------------------------------------- wiring and eval
def test_items_flatten_into_supervised_samples():
    items = load_jigsaw(FIXTURES / "jigsaw", split="test")
    samples = items_to_samples(items)
    assert len(samples) == 18
    assert samples[0].spec["type"] == "bool"
    assert samples[0].state == items[0].state


def test_evaluate_items_reports_reliability_per_question():
    engine = build_tiny_engine()
    items = load_fever(FIXTURES / "fever", split="validation")
    result = evaluate_items(engine, items, dataset="fever", split="validation", example_count=1)
    summary = result.summary()
    assert summary["n_items"] == 3
    assert summary["n_decisions"] == 3
    assert 0.0 <= summary["overall"]["accuracy"] <= 1.0
    assert summary["overall"]["ece"] >= 0.0
    assert summary["usage"]["decisions_per_request"] == 1.0
    assert set(summary["by_question"]) == {"evidence"}
    assert len(summary["examples"]) == 1


def test_evaluate_items_handles_several_questions_per_state():
    engine = build_tiny_engine()
    items = load_jigsaw(FIXTURES / "jigsaw", split="test")
    result = evaluate_items(engine, items, dataset="jigsaw", example_count=0)
    summary = result.summary()
    assert summary["n_items"] == 3
    assert summary["n_decisions"] == 18
    assert summary["usage"]["decisions_per_request"] == 6.0
    assert set(summary["by_question"]) == set(items[0].questions)


def test_training_on_dataset_samples_runs_end_to_end():
    from qwenjev.rlcd import RLCDFineTune

    engine = build_tiny_engine(readout="slot_head")
    items = load_jigsaw(FIXTURES / "jigsaw", split="test")
    tuner = RLCDFineTune(engine, batch_size=6, lr=1e-3)
    report = tuner.train(items_to_samples(items), epochs=1)
    assert report.steps == 3
    evaluation = tuner.evaluate(items_to_samples(items))
    assert evaluation.reliability()["n"] == 18


# ------------------------------------------------------------------ diagnostics
def test_missing_dataset_explains_how_to_get_it(tmp_path):
    with pytest.raises(DatasetUnavailable) as excinfo:
        load_items("mmlu", root=tmp_path / "mmlu")
    message = str(excinfo.value)
    assert "not available locally" in message
    assert "cais/mmlu" in message


def test_status_lists_every_dataset_with_a_hint(tmp_path):
    report = status(tmp_path)
    assert {entry["dataset"] for entry in report} == set(DATASETS)
    assert {"fever", "mmlu", "mmlu_pro", "clinc150", "hwu64", "jigsaw"} == set(DATASETS)
    for entry in report:
        assert entry["available"] is False
        assert entry["root"].startswith(str(tmp_path))
        assert entry["hint"]


def test_default_root_honours_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("QWENJEV_DATA", str(tmp_path))
    assert default_root("fever") == tmp_path / "fever"
