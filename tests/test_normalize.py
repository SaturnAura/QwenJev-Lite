import json
from pathlib import Path

import pytest

from qwenjev.datasets import load_items
from qwenjev.normalize import (
    build_task_plans,
    catalog,
    load_manifest_notes,
    load_normalized,
    normalize_all,
)
from qwenjev.schema import slot_label


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    """A miniature version of the real ``datasets/`` tree."""

    root = tmp_path / "datasets"
    _write(
        root / "MNLISNLI" / "Stanford-nli" / "snli_1.0_train.csv",
        "pairID,sentence1,sentence2,label\n"
        "a,A man is playing guitar.,A man is making music.,entailment\n"
        "b,A man is playing guitar.,A woman is sleeping.,contradiction\n"
        "c,A man is playing guitar.,Someone is outdoors.,neutral\n",
    )
    _write(
        root / "Jigsaw-Toxic-Comment-Classification" / "train.csv",
        "id,comment_text,toxic,severe_toxic,obscene,threat,insult,identity_hate\n"
        "1,thanks for the help,0,0,0,0,0,0\n"
        "2,you are an idiot,1,0,1,0,1,0\n"
        "3,i will find you,1,1,0,1,0,0\n",
    )
    _write(
        root / "GoEmotions" / "data" / "emotions.txt",
        "\n".join(["admiration", "amusement", "anger", "neutral"]) + "\n",
    )
    _write(
        root / "GoEmotions" / "data" / "train.tsv",
        "what a lovely day\t0,3\tabc123\n"
        "i am so angry\t2\tdef456\n",
    )
    _write(
        root / "TruthfulQA" / "TruthfulQA.csv",
        "Type,Category,Question,Best Answer,Correct Answers,Incorrect Answers,Source\n"
        'Adversarial,Misconceptions,Where do fortune cookies originate?,'
        'The precise origin is unclear,The precise origin is unclear;San Francisco,'
        "China;Japan;Los Angeles;Kyoto,http://x\n"
        "Adversarial,Misconceptions,What happens if you eat watermelon seeds?,"
        "They pass through your body,They pass through your body;Nothing happens,"
        "You grow watermelons;You get sick;You die;You have bad dreams,http://y\n"
        "Adversarial,Misconceptions,Where did the fortune cookie come from?,"
        "San Francisco,San Francisco;California,"
        "China;Japan;Los Angeles;Kyoto,http://z\n",
    )
    _write(
        root / "CLINC150" / "PLACE_DATASETS_HERE.txt",
        "the parquet files go here\n",
    )
    _write(
        root / "IntentGrasp" / "train.jsonl",
        json.dumps(
            {
                "id": "atis--train--0",
                "metadata": {"original_task": "atis"},
                "context": "i want to fly from boston",
                "question": "What is the intent of the user?",
                "options": ["book a flight", "ask about meals", "ask about weather"],
                "answer_intent": ["book a flight"],
                "answer_index": [0],
            }
        )
        + "\n",
    )
    return root


def test_slot_labels_extend_past_z():
    assert slot_label(0) == "A"
    assert slot_label(25) == "Z"
    assert slot_label(26) == "AA"
    assert slot_label(51) == "AZ"
    assert slot_label(52) == "BA"
    assert len({slot_label(i) for i in range(200)}) == 200


def test_normalize_all_writes_the_canonical_format(raw_root: Path, tmp_path: Path):
    out = tmp_path / "normalized"
    manifest = normalize_all(raw_root, out, train_limit=50, test_limit=50)

    assert manifest["tasks"]["snli"]["status"] == "ok"
    assert manifest["tasks"]["jigsaw"]["status"] == "ok"
    assert manifest["tasks"]["goemotions"]["status"] == "ok"
    assert manifest["tasks"]["truthfulqa"]["status"] == "ok"
    assert manifest["tasks"]["intentgrasp"]["status"] == "ok"

    # every written line is a complete record
    for path in out.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            assert set(record) == {"id", "dataset", "split", "state", "questions", "targets", "meta"}
            for qid, spec in record["questions"].items():
                assert spec["type"] in {"choice", "bool", "score"}
                assert qid in record["targets"]


def test_snli_groups_hypotheses_under_one_premise(raw_root: Path, tmp_path: Path):
    normalize_all(raw_root, tmp_path / "n", tasks=["snli"], train_limit=0, test_limit=0)
    items = load_normalized(tmp_path / "n" / "snli_train.jsonl")
    assert len(items) == 1
    assert len(items[0].questions) == 3
    assert items[0].state == "A man is playing guitar."
    assert set(items[0].targets.values()) == {"entailment", "contradiction", "neutral"}


def test_jigsaw_reads_its_own_rows_not_another_task(raw_root: Path, tmp_path: Path):
    """Regression: the closed-over split rows were once bound to TruthfulQA's."""

    normalize_all(raw_root, tmp_path / "n", tasks=["jigsaw", "truthfulqa"], train_limit=0, test_limit=0)
    items = load_normalized(tmp_path / "n" / "jigsaw_train.jsonl")
    assert len(items) == 3
    assert items[0].targets["toxic"] == "no"
    assert items[1].targets["toxic"] == "yes"
    for entry in items:
        assert entry.state and "comment_text" not in entry.state


def test_goemotions_uses_the_emotion_names(raw_root: Path, tmp_path: Path):
    normalize_all(raw_root, tmp_path / "n", tasks=["goemotions"], train_limit=0, test_limit=0)
    items = load_normalized(tmp_path / "n" / "goemotions_train.jsonl")
    assert len(items[0].questions) == 4
    assert items[0].targets["admiration"] == "yes"
    assert items[0].targets["anger"] == "no"
    assert items[1].targets["anger"] == "yes"


def test_truthfulqa_builds_a_choice_with_the_best_answer(raw_root: Path, tmp_path: Path):
    normalize_all(raw_root, tmp_path / "n", tasks=["truthfulqa"], train_limit=0, test_limit=0)
    items = load_normalized(tmp_path / "n" / "truthfulqa_train.jsonl")
    question = items[0].questions["truthful"]
    assert question["type"] == "choice"
    target = items[0].targets["truthful"]
    assert target in question["criteria"]
    # the correct option is the item's own best answer, not one of the false ones
    assert question["criteria"][target] == items[0].meta["best_answer"]
    assert len(question["criteria"]) == 5


def test_intentgrasp_keeps_the_option_list(raw_root: Path, tmp_path: Path):
    normalize_all(raw_root, tmp_path / "n", tasks=["intentgrasp"], train_limit=0, test_limit=0)
    items = load_normalized(tmp_path / "n" / "intentgrasp_train.jsonl")
    question = items[0].questions["intent"]
    assert len(question["criteria"]) == 3
    assert items[0].targets["intent"] == "A"
    assert question["criteria"]["A"] == "book a flight"


def test_missing_optional_sidecar_is_reported(raw_root: Path, tmp_path: Path):
    plans = {plan.key: plan for plan in build_task_plans(raw_root)}
    assert "clinc150" in plans
    assert "intents.txt" in plans["clinc150"].warning


def test_catalog_and_manifest_notes(raw_root: Path, tmp_path: Path):
    out = tmp_path / "n"
    normalize_all(raw_root, out, tasks=["jigsaw"], train_limit=20, test_limit=20)
    assert "jigsaw" in catalog(out)
    notes = load_manifest_notes(out)
    assert "jigsaw" in notes


def test_normalized_files_are_picked_up_by_load_items(raw_root: Path, tmp_path: Path):
    out = tmp_path / "n"
    normalize_all(raw_root, out, tasks=["jigsaw"], train_limit=2, test_limit=2)
    items = load_items("jigsaw", split="train", normalized=out, limit=2)
    assert len(items) == 2

    # An explicit raw root bypasses the normalised lookup: the fixture adapter sees its
    # own three rows, not the two-item normalised file written above.
    fixture = Path(__file__).resolve().parent / "fixtures" / "jigsaw"
    raw_items = load_items("jigsaw", root=fixture, split="test", normalized=out)
    assert len(raw_items) == 3
