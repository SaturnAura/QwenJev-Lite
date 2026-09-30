"""Normalise every dataset in ``datasets/`` into one decision-record format.

The canonical record is the engine's own request shape, so normalised files feed the
trainer and the evaluator without any further adaptation:

    {"id": "jigsaw/train/000123",
     "dataset": "jigsaw", "split": "train", "task": "jigsaw",
     "state": "the shared text",
     "questions": {"toxic": {"type": "bool",
                             "instructions": "Does this comment contain toxic content?",
                             "claim": "This comment contains toxic content.",
                             "criteria": {"yes": "Yes", "no": "No"}}},
     "targets": {"toxic": "yes"},
     "meta": {"source_file": "...", "row": 123}}

Two phrase forms are carried for boolean questions: ``instructions`` (a question, what
the Qwen branch asks) and ``claim`` (a statement, what Laya's ``noul`` template wants to
see at the head of the state). Each backend uses the one it was measured on.

Run it with::

    python -m qwenjev.cli normalize                     # data/raw -> data/ready/
    python -m qwenjev.cli normalize --tasks jigsaw snli --test-limit 100
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .datasets import DecisionItem, _clean

#: Default caps so a normalised run stays inside a laptop's patience (0 = no cap).
DEFAULT_TRAIN_LIMIT = 2000
DEFAULT_TEST_LIMIT = 400


# ------------------------------------------------------------------ readers
def _read_csv(path: Path, **kwargs) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, **kwargs)]


def _read_tsv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.reader(handle, delimiter="\t") if row]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_parquet(path: Path) -> list[dict]:
    import pandas as pd

    return pd.read_parquet(path).to_dict(orient="records")


def _read_any(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    if path.suffix == ".parquet":
        return _read_parquet(path)
    if path.suffix == ".tsv":
        rows = _read_tsv_rows(path)
        return rows
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else [payload]
    return _read_csv(path)


def _stable_index(text: str, modulo: int) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) % modulo


def _as_list(value: Any) -> list:
    """``list`` for python/numpy sequences, ``[]`` for None, safe on scalars."""

    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _int_or_none(value: Any) -> int | None:
    """``int`` for a real number, ``None`` for None / NaN / anything else."""

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return int(number)


def _iter_limited(rows: Sequence[Any], limit: int | None):
    """Indices to visit so a cap samples the whole file instead of its head.

    CLINC150 and HWU64 ship label-sorted parquet files and IntentGrasp is grouped by
    source corpus, so ``rows[:limit]`` can be a single class or a single corpus.
    The indices are spread evenly over the whole file, which matters just as much when
    ``limit`` is more than half of ``rows``: a stride of one used to keep only the
    front of the file, so a 750-row label-sorted intent file was scored on its first
    40 intents and never on the other 110.
    """

    total = len(rows)
    if not limit or limit >= total:
        return range(total)
    stride = max(1, total // limit)
    if stride > 1:
        return range(0, total, stride)
    # More than half of the file fits in the cap: stride 1 would take a prefix, so
    # walk the file evenly instead.
    return [round(index * total / limit) for index in range(limit)]


def _split_rows(rows: Sequence[dict], key: Callable[[dict], str], test_fraction: float = 0.2):
    """Deterministic hash split, so a rerun puts the same rows in the same bucket."""

    cutoff = int(round(test_fraction * 10000))
    train, test = [], []
    for row in rows:
        (test if _stable_index(key(row), 10000) < cutoff else train).append(row)
    return train, test


# ------------------------------------------------------------------ builders
def bool_question(question: str, claim: str, context: str | None = None) -> dict:
    spec = {
        "type": "bool",
        "instructions": question,
        "claim": claim,
        "criteria": {"yes": "Yes", "no": "No"},
    }
    if context:
        spec["context"] = context
    return spec


def score_question(instructions: str, levels: dict[str, str], context: str | None = None) -> dict:
    """Ordered levels, lowest first: ``{"low": "text", "high": "text"}``."""

    spec = {"type": "score", "instructions": instructions, "criteria": dict(levels)}
    if context:
        spec["context"] = context
    return spec


def choice_question(instructions: str, criteria: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def item(
    *,
    dataset: str,
    split: str,
    index: Any,
    state: str,
    questions: dict[str, dict],
    targets: dict[str, str],
    source: Path | None = None,
    meta: dict | None = None,
) -> DecisionItem:
    payload = {"row": index}
    if source is not None:
        payload["source_file"] = source.name
    if meta:
        payload.update(meta)
    return DecisionItem(
        item_id=f"{dataset}/{split}/{index}",
        state=state,
        questions=questions,
        targets=targets,
        meta=payload,
    )


# --------------------------------------------------------------- converters
LABEL = {
    "entailment": "entailment",
    "neutral": "neutral",
    "contradiction": "contradiction",
    "e": "entailment",
    "n": "neutral",
    "c": "contradiction",
}
NLI_CRITERIA = {
    "entailment": "The sentence must be true whenever the premise is true",
    "neutral": "The sentence may or may not be true; the premise does not settle it",
    "contradiction": "The sentence must be false whenever the premise is true",
}


def convert_nli(path: Path, split: str, *, dataset: str, limit: int | None) -> list[DecisionItem]:
    """Group hypotheses by premise: one shared state, one isolated branch each."""

    rows = _read_any(path)
    groups: dict[str, list[dict]] = {}
    # A stride here would split the hypotheses of one premise across the sample and
    # turn ~3-branch requests into single-branch requests, so group over every row and
    # cap the *premises* afterwards.
    for row in rows:
        premise = _clean(row.get("sentence1") or row.get("premise") or "")
        hypothesis = _clean(row.get("sentence2") or row.get("hypothesis") or "")
        label = str(row.get("label") or row.get("gold_label") or "").strip().lower()
        if not premise or not hypothesis or label not in LABEL:
            continue
        groups.setdefault(premise, []).append({"hypothesis": hypothesis, "label": LABEL[label]})

    items: list[DecisionItem] = []
    for premise, group in groups.items():
        questions = {}
        targets = {}
        for i, entry in enumerate(group):
            qid = f"h{i}"
            questions[qid] = choice_question(
                f'Does the premise entail this sentence: "{entry["hypothesis"]}"?',
                dict(NLI_CRITERIA),
            )
            targets[qid] = entry["label"]
        items.append(
            item(
                dataset=dataset,
                split=split,
                index=len(items),
                state=premise,
                questions=questions,
                targets=targets,
                source=path,
                meta={"hypotheses": len(group)},
            )
        )
        if limit and len(items) >= limit:
            break
    return items


FEVER_CRITERIA = {
    "supports": "The claim is true",
    "refutes": "The claim is false",
    "not_enough_info": "There is not enough information to tell whether the claim is true",
}


def convert_fever(path: Path, split: str, *, dataset: str, limit: int | None) -> list[DecisionItem]:
    rows = _read_any(path)
    items: list[DecisionItem] = []
    for index in _iter_limited(rows, limit):
        row = rows[index]
        label = str(row.get("label", "")).strip().upper().replace(" ", "_")
        key = {
            "SUPPORTS": "supports",
            "REFUTES": "refutes",
            "NOT_ENOUGH_INFO": "not_enough_info",
            "NOTENOUGHINFO": "not_enough_info",
        }.get(label)
        claim = _clean(row.get("claim", ""))
        if key is None or not claim:
            continue
        items.append(
            item(
                dataset=dataset,
                split=split,
                index=row.get("id", index),
                state=claim,
                questions={
                    "verdict": choice_question(
                        "Claim: " + claim + "\nIs this claim true, false, or not verifiable?",
                        dict(FEVER_CRITERIA),
                    )
                },
                targets={"verdict": key},
                source=path,
                meta={
                    "claim": claim,
                    "verifiable": str(row.get("verifiable", "")),
                    "has_evidence_text": False,
                },
            )
        )
        if limit and len(items) >= limit:
            break
    return items


JIGSAW_LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")


def load_emotion_names(data_dir: Path) -> list[str]:
    path = data_dir / "emotions.txt"
    if not path.is_file():
        raise FileNotFoundError(f"GoEmotions needs emotions.txt next to the tsv files: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def convert_goemotions(
    path: Path, split: str, *, dataset: str, limit: int | None, emotions: Sequence[str]
) -> list[DecisionItem]:
    rows = _read_tsv_rows(path)
    items: list[DecisionItem] = []
    for index in _iter_limited(rows, limit):
        row = rows[index]
        if len(row) < 2:
            continue
        text = _clean(row[0])
        active = {int(part) for part in row[1].split(",") if part.strip().isdigit()}
        if not text or not active:
            continue
        questions = {
            name: bool_question(
                f"Does this text express {name.replace('_', ' ')}?",
                f"This text expresses {name.replace('_', ' ')}.",
            )
            for name in emotions
        }
        targets = {name: ("yes" if i in active else "no") for i, name in enumerate(emotions)}
        items.append(
            item(
                dataset=dataset,
                split=split,
                index=row[2] if len(row) > 2 else index,
                state=text,
                questions=questions,
                targets=targets,
                source=path,
                meta={"emotions": sorted(active)},
            )
        )
        if limit and len(items) >= limit:
            break
    return items


SENTIMENT_LEVELS = {"negative": "Negative", "ambiguous": "Neutral or mixed", "positive": "Positive"}


def load_sentiment_map(data_dir: Path) -> dict:
    path = data_dir / "sentiment_mapping.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def convert_goemotions_sentiment(
    path: Path,
    split: str,
    *,
    dataset: str,
    limit: int | None,
    emotions: Sequence[str],
    sentiment_map: dict,
) -> list[DecisionItem]:
    positive = set(sentiment_map.get("positive", []))
    negative = set(sentiment_map.get("negative", []))
    rows = _read_tsv_rows(path)
    items: list[DecisionItem] = []
    for index in _iter_limited(rows, limit):
        row = rows[index]
        if len(row) < 2:
            continue
        text = _clean(row[0])
        if not text:
            continue
        active: set[str] = set()
        for part in row[1].split(","):
            if part.strip().isdigit():
                emotion_index = int(part)
                if 0 <= emotion_index < len(emotions):
                    active.add(emotions[emotion_index])
        if not active:
            continue
        if active & positive:
            level = "positive"
        elif active & negative:
            level = "negative"
        else:
            level = "ambiguous"
        items.append(
            item(
                dataset=dataset,
                split=split,
                index=row[2] if len(row) > 2 else index,
                state=text,
                questions={
                    "sentiment": score_question(
                        "What is the sentiment of this text?", dict(SENTIMENT_LEVELS)
                    )
                },
                targets={"sentiment": level},
                source=path,
            )
        )
        if limit and len(items) >= limit:
            break
    return items


SEVERITY_LEVELS = {"not_toxic": "Not toxic", "toxic": "Toxic", "severe": "Severely toxic"}


def convert_jigsaw_severity_rows(
    rows: Sequence[dict], split: str, limit: int | None
) -> list[DecisionItem]:
    items: list[DecisionItem] = []
    for index in _iter_limited(rows, limit):
        row = rows[index]
        text = _clean(row.get("comment_text") or row.get("text") or "")
        if not text:
            continue
        if int(float(row.get("severe_toxic", 0) or 0)) == 1:
            level = "severe"
        elif int(float(row.get("toxic", 0) or 0)) == 1:
            level = "toxic"
        else:
            level = "not_toxic"
        items.append(
            item(
                dataset="jigsaw_severity",
                split=split,
                index=row.get("id", index),
                state=text,
                questions={
                    "severity": score_question("How toxic is this comment?", dict(SEVERITY_LEVELS))
                },
                targets={"severity": level},
                meta={"synthetic_split": True},
            )
        )
        if limit and len(items) >= limit:
            break
    return items


SUPPORT_LEVELS = {"false": "False", "cannot_verify": "Cannot be verified", "true": "True"}


def convert_fever_support(
    path: Path, split: str, *, dataset: str, limit: int | None
) -> list[DecisionItem]:
    rows = _read_any(path)
    items: list[DecisionItem] = []
    for index in _iter_limited(rows, limit):
        row = rows[index]
        label = str(row.get("label", "")).strip().upper().replace(" ", "_")
        key = {
            "SUPPORTS": "true",
            "REFUTES": "false",
            "NOT_ENOUGH_INFO": "cannot_verify",
            "NOTENOUGHINFO": "cannot_verify",
        }.get(label)
        claim = _clean(row.get("claim", ""))
        if key is None or not claim:
            continue
        items.append(
            item(
                dataset=dataset,
                split=split,
                index=row.get("id", index),
                state=claim,
                questions={
                    "truth": score_question(
                        "How likely is this claim to be true?", dict(SUPPORT_LEVELS)
                    )
                },
                targets={"truth": key},
                source=path,
            )
        )
        if limit and len(items) >= limit:
            break
    return items


def convert_mmlu_pro(path: Path, split: str, *, dataset: str, limit: int | None) -> list[DecisionItem]:
    rows = _read_any(path)
    items: list[DecisionItem] = []
    for index in _iter_limited(rows, limit):
        row = rows[index]
        question = _clean(row.get("question", ""))
        options = _as_list(row.get("options"))
        if not options:
            options = _as_list(row.get("choices"))
        if not question or not options:
            continue
        options = [_clean(o) for o in options]
        answer = row.get("answer")
        if isinstance(answer, (int, float)) and not isinstance(answer, bool):
            index_answer = int(answer)
        elif row.get("answer_index") is not None:
            index_answer = int(row["answer_index"])
        elif isinstance(answer, str) and len(answer.strip()) == 1:
            index_answer = ord(answer.strip().upper()) - ord("A")
        else:
            continue
        if not 0 <= index_answer < len(options):
            continue
        letters = [chr(ord("A") + i) for i in range(len(options))]
        category = _clean(row.get("category", ""))
        items.append(
            item(
                dataset=dataset,
                split=split,
                index=row.get("question_id", index),
                state=(f"Category: {category}\n" if category else "") + question,
                questions={
                    "answer": choice_question(
                        "Which option is correct?",
                        {letter: option for letter, option in zip(letters, options)},
                    )
                },
                targets={"answer": letters[index_answer]},
                source=path,
                meta={"category": category, "n_options": len(options)},
            )
        )
        if limit and len(items) >= limit:
            break
    return items


def convert_intentgrasp(path: Path, split: str, *, dataset: str, limit: int | None) -> list[DecisionItem]:
    rows = _read_any(path)
    items: list[DecisionItem] = []
    for index in _iter_limited(rows, limit):
        row = rows[index]
        context = _clean(row.get("context", ""))
        options = [_clean(o) for o in _as_list(row.get("options"))]
        answers = _as_list(row.get("answer_index"))
        if not context or not options or not answers:
            continue
        answer = int(answers[0])
        if not 0 <= answer < len(options):
            continue
        letters = [chr(ord("A") + i) for i in range(len(options))]
        metadata = row.get("metadata") or {}
        items.append(
            item(
                dataset=dataset,
                split=split,
                index=row.get("id", index),
                state=context,
                questions={
                    "intent": choice_question(
                        _clean(row.get("question") or "What is the intent of the user?"),
                        {letter: option for letter, option in zip(letters, options)},
                    )
                },
                targets={"intent": letters[answer]},
                source=path,
                meta={
                    "corpus": metadata.get("original_task", ""),
                    "n_options": len(options),
                    "answer_intent": (_as_list(row.get("answer_intent")) or [""])[0],
                },
            )
        )
        if limit and len(items) >= limit:
            break
    return items


def convert_intent_tsv(
    path: Path,
    split: str,
    *,
    dataset: str,
    limit: int | None,
    max_labels: int | None = None,
) -> list[DecisionItem]:
    """Header-less ``utterance<TAB>intent`` file; the labels come from the file itself.

    ``max_labels`` keeps only the N most frequent intents, which gives a short branch
    that every readout (reserved-label, slot head, Laya) can answer.
    """

    rows = _read_tsv_rows(path)
    pairs: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    for row in rows:
        if len(row) < 2:
            continue
        text = _clean(row[0])
        label = _clean(row[1]).lower().replace(" ", "_")
        if not text or not label:
            continue
        pairs.append((text, label))
        counts[label] = counts.get(label, 0) + 1
    if not pairs:
        return []
    if max_labels and len(counts) > max_labels:
        keep = {
            name
            for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_labels]
        }
        pairs = [(text, label) for text, label in pairs if label in keep]
    criteria = {name: name.replace("_", " ") for name in sorted({label for _, label in pairs})}

    items: list[DecisionItem] = []
    for index in _iter_limited(pairs, limit):
        text, label = pairs[index]
        items.append(
            item(
                dataset=dataset,
                split=split,
                index=index,
                state=text,
                questions={
                    "intent": choice_question(
                        "Which intent does this utterance express?", dict(criteria)
                    )
                },
                targets={"intent": label},
                source=path,
                meta={"n_labels": len(criteria)},
            )
        )
        if limit and len(items) >= limit:
            break
    return items


def convert_intent_parquet(
    path: Path, split: str, *, dataset: str, limit: int | None, label_names: Sequence[str] | None
) -> list[DecisionItem]:
    """CLINC150 / HWU64 style ``utterance`` + integer ``label`` tables."""

    rows = _read_any(path)
    labels = {
        label
        for label in (_int_or_none(row.get("label")) for row in rows)
        if label is not None
    }
    names = list(label_names) if label_names else [f"label_{i:03d}" for i in range(max(labels) + 1)]
    resolved = label_names is not None
    items: list[DecisionItem] = []
    for index in _iter_limited(rows, limit):
        row = rows[index]
        text = _clean(row.get("utterance") or row.get("text") or "")
        key = _int_or_none(row.get("label"))
        if not text or key is None or key >= len(names):
            continue
        items.append(
            item(
                dataset=dataset,
                split=split,
                index=index,
                state=text,
                questions={
                    "intent": choice_question(
                        "Which intent does this utterance express?",
                        {name: name.replace("_", " ") for name in names},
                    )
                },
                targets={"intent": names[key]},
                source=path,
                meta={"n_labels": len(names), "label_names_resolved": resolved, "label_index": key},
            )
        )
        if limit and len(items) >= limit:
            break
    return items


def read_label_names(directory: Path) -> list[str] | None:
    """Optional sidecar: ``intents.txt`` / ``labels.txt`` / ``classes.json``."""

    for name in ("intents.txt", "labels.txt", "label_names.txt", "classes.txt"):
        path = directory / name
        if path.is_file():
            return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    path = directory / "classes.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [str(x) for x in payload]
    return None


# --------------------------------------------------------------- task plans
@dataclass
class SplitPlan:
    name: str
    build: Callable[[int | None], list[DecisionItem]]
    sources: list[Path] = field(default_factory=list)
    note: str = ""


@dataclass
class TaskPlan:
    key: str
    title: str
    state: str
    decision: str
    splits: list[SplitPlan]
    note: str = ""
    ready: bool = True
    warning: str = ""


def find_dir(src: Path, *names: str) -> Path | None:
    if not src.is_dir():
        return None
    lowered = {p.name.lower(): p for p in src.iterdir() if p.is_dir()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def find_file(directory: Path, *patterns: str) -> Path | None:
    if not directory or not directory.is_dir():
        return None
    files = [p for p in directory.rglob("*") if p.is_file()]
    for pattern in patterns:
        needle = pattern.lower()
        for path in files:
            if needle in path.name.lower():
                return path
    return None


def build_task_plans(src: Path, *, seed: int = 0) -> list[TaskPlan]:
    plans: list[TaskPlan] = []

    def add(plan: TaskPlan | None) -> None:
        if plan is not None:
            plans.append(plan)

    # --- MultiNLI / SNLI: one premise, every hypothesis as its own branch -------
    mnli_dir = find_dir(src, "MNLISNLI")
    if mnli_dir:
        multi = find_dir(mnli_dir, "Multi-genre-nli")
        snli = find_dir(mnli_dir, "Stanford-nli")
        if multi:
            train = find_file(multi, "train.csv", "train")
            matched = find_file(multi, "dev_matched", "matched")
            mismatched = find_file(multi, "dev_mismatched", "mismatched")
            splits = []
            if train:
                splits.append(SplitPlan("train", lambda limit, p=train: convert_nli(p, "train", dataset="mnli", limit=limit), [train]))
            if matched:
                splits.append(SplitPlan("test", lambda limit, p=matched: convert_nli(p, "test", dataset="mnli", limit=limit), [matched]))
            if mismatched:
                splits.append(SplitPlan("test_ood", lambda limit, p=mismatched: convert_nli(p, "test_ood", dataset="mnli", limit=limit), [mismatched]))
            add(TaskPlan("mnli", "MultiNLI - entailment", "premise", "3-way entailment per hypothesis", splits,
                         note="test = dev_matched, test_ood = dev_mismatched (out of domain)"))
        if snli:
            splits = []
            for name, patterns in (("train", ("train.csv",)), ("test", ("test.csv",)), ("valid", ("dev.csv",))):
                path = find_file(snli, *patterns)
                if path:
                    splits.append(SplitPlan(name, lambda limit, p=path, n=name: convert_nli(p, n, dataset="snli", limit=limit), [path]))
            add(TaskPlan("snli", "SNLI - entailment", "premise", "3-way entailment per hypothesis", splits,
                         note="5 hypotheses share one premise, so this is the shared-state showcase"))

    # --- FEVER ----------------------------------------------------------------
    fever = find_dir(src, "FEVER")
    if fever:
        splits = []
        for name, patterns in (
            ("train", ("train.jsonl", "train")),
            ("valid", ("shared_task_dev", "paper_dev", "dev")),
            ("test", ("paper_test", "test")),
        ):
            path = find_file(fever, *patterns)
            if path:
                splits.append(SplitPlan(name, lambda limit, p=path, n=name: convert_fever(p, n, dataset="fever", limit=limit), [path]))
        add(TaskPlan("fever", "FEVER - fact verification", "claim", "supports / refutes / not enough info", splits,
                     note="the provided files carry no wiki sentences, so this is claim-only (closed-book) verification",
                     warning="no evidence text: state = claim"))
        support_splits = []
        for name, patterns in (
            ("train", ("train.jsonl", "train")),
            ("valid", ("shared_task_dev", "paper_dev", "dev")),
            ("test", ("paper_test", "test")),
        ):
            path = find_file(fever, *patterns)
            if path:
                support_splits.append(
                    SplitPlan(
                        name,
                        lambda limit, p=path, n=name: convert_fever_support(
                            p, n, dataset="fever_support", limit=limit
                        ),
                        [path],
                    )
                )
        add(
            TaskPlan(
                "fever_support",
                "FEVER - claim truth level",
                "claim",
                "score: false / cannot verify / true",
                support_splits,
            )
        )

    # --- Jigsaw ---------------------------------------------------------------
    jigsaw = find_dir(src, "Jigsaw-Toxic-Comment-Classification", "Jigsaw", "Jigsaw-Toxic")
    if jigsaw:
        train_file = find_file(jigsaw, "train.csv", "train")
        if train_file:
            rows = _read_any(train_file)
            train_rows, test_rows = _split_rows(rows, lambda row: str(row.get("id") or row.get("comment_text", "")))
            jigsaw_splits = [
                SplitPlan(
                    "train",
                    lambda limit, rows=train_rows: convert_jigsaw_rows(rows, "train", limit),
                    [train_file],
                ),
                SplitPlan(
                    "test",
                    lambda limit, rows=test_rows: convert_jigsaw_rows(rows, "test", limit),
                    [train_file],
                    note="held out from train.csv: the shipped test.csv has no labels",
                ),
            ]
            add(
                TaskPlan(
                    "jigsaw",
                    "Jigsaw toxic comment classification",
                    "comment text",
                    "six independent yes/no decisions",
                    jigsaw_splits,
                    note="test split is a deterministic 20% hold-out of train.csv",
                )
            )
            add(
                TaskPlan(
                    "jigsaw_severity",
                    "Jigsaw - comment severity",
                    "comment text",
                    "score: not toxic / toxic / severely toxic",
                    [
                        SplitPlan(
                            "train",
                            lambda limit, rows=train_rows: convert_jigsaw_severity_rows(
                                rows, "train", limit
                            ),
                            [train_file],
                        ),
                        SplitPlan(
                            "test",
                            lambda limit, rows=test_rows: convert_jigsaw_severity_rows(
                                rows, "test", limit
                            ),
                            [train_file],
                        ),
                    ],
                    note="ordered severity derived from the toxic / severe_toxic columns",
                )
            )

    # --- GoEmotions -----------------------------------------------------------
    goemotions = find_dir(src, "GoEmotions")
    if goemotions:
        data_dir = find_dir(goemotions, "data") or goemotions
        try:
            emotions = load_emotion_names(data_dir)
        except FileNotFoundError as exc:
            emotions = []
            add(TaskPlan("goemotions", "GoEmotions", "text", "28 yes/no decisions", [], warning=str(exc), ready=False))
        else:
            splits = []
            for name, patterns in (("train", ("train.tsv",)), ("valid", ("dev.tsv", "dev")), ("test", ("test.tsv",))):
                path = find_file(data_dir, *patterns)
                if path:
                    splits.append(SplitPlan(name, lambda limit, p=path, n=name: convert_goemotions(p, n, dataset="goemotions", limit=limit, emotions=emotions), [path]))
            add(TaskPlan("goemotions", "GoEmotions - emotion classification", "text", "28 yes/no decisions (Ekman-mappable)", splits))
            sentiment_map = load_sentiment_map(data_dir)
            if sentiment_map:
                sentiment_splits = []
                for name, patterns in (
                    ("train", ("train.tsv",)),
                    ("valid", ("dev.tsv", "dev")),
                    ("test", ("test.tsv",)),
                ):
                    path = find_file(data_dir, *patterns)
                    if path:
                        sentiment_splits.append(
                            SplitPlan(
                                name,
                                lambda limit, p=path, n=name: convert_goemotions_sentiment(
                                    p,
                                    n,
                                    dataset="goemotions_sentiment",
                                    limit=limit,
                                    emotions=emotions,
                                    sentiment_map=sentiment_map,
                                ),
                                [path],
                            )
                        )
                add(
                    TaskPlan(
                        "goemotions_sentiment",
                        "GoEmotions - sentiment",
                        "text",
                        "score: negative / neutral-or-mixed / positive",
                        sentiment_splits,
                    )
                )

    # --- TruthfulQA -----------------------------------------------------------
    truthful = find_dir(src, "TruthfulQA")
    if truthful:
        path = find_file(truthful, "truthfulqa.csv", "truthful_qa")
        if path:
            rows = _read_any(path)
            train_rows, test_rows = _split_rows(rows, lambda row: str(row.get("Question", "")), test_fraction=0.3)
            truthful_splits = [
                SplitPlan(
                    "train",
                    lambda limit, rows=train_rows: convert_truthfulqa_rows(rows, "train", limit, seed),
                    [path],
                ),
                SplitPlan(
                    "test",
                    lambda limit, rows=test_rows: convert_truthfulqa_rows(rows, "test", limit, seed),
                    [path],
                    note="deterministic 30% hold-out: the released CSV has no official split",
                ),
            ]
            add(
                TaskPlan(
                    "truthfulqa",
                    "TruthfulQA - truthful answer selection",
                    "question",
                    "5-way choice: the best answer against four false ones",
                    truthful_splits,
                    note="MC reading of the set, not the official MC1/MC2 metrics",
                )
            )

    # --- MMLU-Pro -------------------------------------------------------------
    mmlu = find_dir(src, "MMLU-Pro", "MMLU_Pro", "MMLUPro")
    if mmlu:
        test_file = find_file(mmlu, "test-00000", "test")
        valid_file = find_file(mmlu, "validation")
        splits = []
        if valid_file:
            splits.append(SplitPlan("train", lambda limit, p=valid_file: convert_mmlu_pro(p, "train", dataset="mmlu_pro", limit=limit), [valid_file],
                                    note="only the 70-row validation split is labelled for training"))
        if test_file:
            splits.append(SplitPlan("test", lambda limit, p=test_file: convert_mmlu_pro(p, "test", dataset="mmlu_pro", limit=limit), [test_file]))
        add(TaskPlan("mmlu_pro", "MMLU-Pro - multiple choice", "category + question", "10-way choice", splits))

    # --- IntentGrasp ----------------------------------------------------------
    intent = find_dir(src, "IntentGrasp", "Intent-Grasp")
    if intent:
        splits = []
        for name, patterns in (("train", ("train.jsonl",)), ("test", ("test.jsonl",))):
            path = find_file(intent, *patterns)
            if path:
                splits.append(SplitPlan(name, lambda limit, p=path, n=name: convert_intentgrasp(p, n, dataset="intentgrasp", limit=limit), [path]))
        add(TaskPlan("intentgrasp", "IntentGrasp - intent detection", "utterance/context", "choice over the item's own option list", splits,
                     note="options ship with the data, so no label-name lookup is needed"))

    # --- intent test sets that ship with real label names ---------------------
    for key, dir_names, title in (
        ("banking77", ("BANKING77",), "BANKING77 - intent detection"),
        ("clinc150", ("CLINC150", "CLINC"), "CLINC150 - intent detection"),
        ("hwu64", ("HWU64", "HWU"), "HWU64 - intent detection"),
    ):
        directory = find_dir(src, *dir_names)
        if not directory:
            continue
        tsv = find_file(directory, "testset.tsv", "tfidf")
        if not tsv:
            continue
        splits = [
            SplitPlan(
                "test",
                lambda limit, p=tsv, k=key: convert_intent_tsv(
                    p, "test", dataset=k, limit=limit
                ),
                [tsv],
                note="the provided labelled examples, held out from training",
            )
        ]
        # When the integer labels have been named (scripts/map_labels.py writes
        # intents.txt), the dataset's own train split becomes training data for the
        # same label space - which is what lets the readout learn it.
        recovered = read_label_names(directory)
        train_file = find_file(directory, "train-00000", "train")
        if recovered and train_file:
            splits.append(
                SplitPlan(
                    "train",
                    lambda limit, p=train_file, k=key, nm=recovered: convert_intent_parquet(
                        p, "train", dataset=k, limit=limit, label_names=nm
                    ),
                    [train_file],
                    note="labels recovered from the labelled examples",
                )
            )
        add(
            TaskPlan(
                key,
                title,
                "utterance",
                "choice over every intent in the file",
                splits,
                note="label names come from the file, so the options are readable",
            )
        )
        add(
            TaskPlan(
                f"{key}_top15",
                f"{title} (15 most frequent intents)",
                "utterance",
                "choice over 15 intents",
                [
                    SplitPlan(
                        "test",
                        lambda limit, p=tsv, k=key: convert_intent_tsv(
                            p, "test", dataset=f"{k}_top15", limit=limit, max_labels=15
                        ),
                        [tsv],
                        note="closed 15-label subset, so every readout can answer",
                    )
                ],
                note="subsets the label space to 15 intents to keep the branch short",
            )
        )

    # --- CLINC150 / HWU64 without a named file: label ids only ---------------
    for key, dir_names, split_files in (
        ("clinc150", ("CLINC150", "CLINC"), (("train", ("train-00000", "train")), ("test", ("validation", "test")))),
        ("hwu64", ("HWU64", "HWU"), (("train", ("train-00000", "train")), ("test", ("test-00000", "test")))),
    ):
        directory = find_dir(src, *dir_names)
        if not directory:
            continue
        if find_file(directory, "testset.tsv", "tfidf"):
            continue  # the named test file above already covers this collection
        names = read_label_names(directory)
        splits = []
        for split_name, patterns in split_files:
            path = find_file(directory, *patterns)
            if path:
                splits.append(
                    SplitPlan(
                        split_name,
                        lambda limit, p=path, n=split_name, nm=names, k=key: convert_intent_parquet(
                            p, n, dataset=k, limit=limit, label_names=nm
                        ),
                        [path],
                    )
                )
        warning = ""
        if names is None:
            warning = (
                f"label ids only: drop the intent names into {directory.name}/intents.txt "
                "(one per line, in label-id order) to make the options readable"
            )
        add(
            TaskPlan(
                key,
                f"{key.upper()} - intent detection",
                "utterance",
                f"choice over {len(names) if names else '?'} intents",
                splits,
                note="a many-label closed-set task; the readout needs the slot head",
                warning=warning,
            )
        )
    return plans


def convert_jigsaw_rows(rows: Sequence[dict], split: str, limit: int | None) -> list[DecisionItem]:
    items: list[DecisionItem] = []
    for index in _iter_limited(rows, limit):
        row = rows[index]
        text = _clean(row.get("comment_text") or row.get("text") or "")
        if not text:
            continue
        questions = {
            label: bool_question(
                f"Does this comment contain {label.replace('_', ' ')} content?",
                f"This comment contains {label.replace('_', ' ')} content.",
            )
            for label in JIGSAW_LABELS
        }
        targets = {
            label: ("yes" if int(float(row.get(label, 0) or 0)) == 1 else "no")
            for label in JIGSAW_LABELS
        }
        items.append(
            item(
                dataset="jigsaw",
                split=split,
                index=row.get("id", index),
                state=text,
                questions=questions,
                targets=targets,
                meta={"synthetic_split": True},
            )
        )
        if limit and len(items) >= limit:
            break
    return items


def convert_truthfulqa_rows(rows: Sequence[dict], split: str, limit: int | None, seed: int) -> list[DecisionItem]:
    rng = random.Random(seed)
    items: list[DecisionItem] = []
    for index in _iter_limited(rows, limit):
        row = rows[index]
        question = _clean(row.get("Question", ""))
        best = _clean(row.get("Best Answer", ""))
        wrong = [x.strip() for x in str(row.get("Incorrect Answers", "")).split(";") if x.strip()]
        if not question or not best or not wrong:
            continue
        options = [best] + rng.sample(wrong, min(4, len(wrong)))
        order = list(range(len(options)))
        rng.shuffle(order)
        letters = [chr(ord("A") + i) for i in range(len(options))]
        criteria = {letters[slot]: options[i] for slot, i in enumerate(order)}
        items.append(
            item(
                dataset="truthfulqa",
                split=split,
                index=index,
                state=question,
                questions={"truthful": choice_question("Which answer to the question is truthful?", criteria)},
                targets={"truthful": letters[order.index(0)]},
                meta={"category": row.get("Category", ""), "best_answer": best, "synthetic_split": True},
            )
        )
        if limit and len(items) >= limit:
            break
    return items


# ------------------------------------------------------------------ writing
def write_items(path: Path, items: Sequence[DecisionItem], *, dataset: str, split: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    decisions = 0
    questions: set[str] = set()
    with path.open("w", encoding="utf-8") as handle:
        for entry in items:
            decisions += len(entry.questions)
            questions.update(entry.questions)
            record = {
                "id": entry.item_id,
                "dataset": dataset,
                "split": split,
                "state": entry.state,
                "questions": entry.questions,
                "targets": entry.targets,
                "meta": entry.meta,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "file": str(path),
        "items": len(items),
        "decisions": decisions,
        "question_ids": sorted(questions),
    }


def normalize_all(
    src: Path | str,
    out_dir: Path | str,
    *,
    tasks: Sequence[str] | None = None,
    train_limit: int = DEFAULT_TRAIN_LIMIT,
    test_limit: int = DEFAULT_TEST_LIMIT,
    seed: int = 0,
) -> dict:
    """Convert every available dataset and write ``<out_dir>/<task>_<split>.jsonl``."""

    src = Path(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plans = build_task_plans(src, seed=seed)
    selected = set(tasks) if tasks else None
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": str(src),
        "out_dir": str(out_dir),
        "train_limit": train_limit,
        "test_limit": test_limit,
        "tasks": {},
    }

    for plan in plans:
        if selected is not None and plan.key not in selected:
            continue
        entry: dict[str, Any] = {
            "title": plan.title,
            "state": plan.state,
            "decision": plan.decision,
            "note": plan.note,
            "warning": plan.warning,
            "splits": {},
        }
        if not plan.ready:
            entry["status"] = "skipped"
            manifest["tasks"][plan.key] = entry
            continue
        if not plan.splits:
            entry["status"] = "missing"
            manifest["tasks"][plan.key] = entry
            continue
        for split_plan in plan.splits:
            limit = train_limit if split_plan.name == "train" else test_limit
            limit = None if not limit else limit
            items = split_plan.build(limit)
            if not items:
                continue
            path = out_dir / f"{plan.key}_{split_plan.name}.jsonl"
            entry["splits"][split_plan.name] = {
                **write_items(path, items, dataset=plan.key, split=split_plan.name),
                "sources": [p.name for p in split_plan.sources],
                "note": split_plan.note,
                "example": _example(items[0]),
            }
        entry["status"] = "ok" if entry["splits"] else "empty"
        manifest["tasks"][plan.key] = entry

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def _example(entry: DecisionItem) -> dict:
    first_question = next(iter(entry.questions))
    return {
        "id": entry.item_id,
        "state": entry.state[:160],
        "question": entry.questions[first_question],
        "target": entry.targets[first_question],
        "n_questions": len(entry.questions),
    }


def load_normalized(path: Path | str) -> list[DecisionItem]:
    """Read a normalised JSONL file back into :class:`DecisionItem` objects."""

    path = Path(path)
    items: list[DecisionItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: line {len(items) + 1} is not a complete JSON object ({exc.msg}). "
                    "A formatted file is JSONL: exactly one record per line, with no "
                    "pretty-printing (see README, 'Paths and data formats')."
                ) from exc
            meta = dict(record.get("meta") or {})
            meta.setdefault("dataset", record.get("dataset"))
            meta.setdefault("split", record.get("split"))
            items.append(
                DecisionItem(
                    item_id=record.get("id", ""),
                    state=record["state"],
                    questions=record["questions"],
                    targets=record["targets"],
                    meta=meta,
                )
            )
    return items


def catalog(out_dir: Path | str) -> dict[str, list[str]]:
    """``{dataset: [split, ...]}`` for the files present in a normalised directory."""

    out_dir = Path(out_dir)
    return _discover(out_dir, _known_task_keys())


#: Task keys a normalised directory may contain. Split names may contain underscores,
#: so this list is what makes ``mnli_test_ood.jsonl`` parse as (mnli, test_ood).
TASK_KEYS: tuple[str, ...] = (
    "mnli",
    "snli",
    "fever",
    "fever_support",
    "jigsaw",
    "jigsaw_severity",
    "goemotions",
    "goemotions_sentiment",
    "truthfulqa",
    "mmlu_pro",
    "intentgrasp",
    "clinc150",
    "hwu64",
    "synthetic",
)


def _known_task_keys() -> list[str]:
    return list(TASK_KEYS)


def _discover(out_dir: Path, known: Sequence[str]) -> dict[str, list[str]]:
    """Match filenames against the known task keys, longest key first.

    Split names contain underscores (``mnli_test_ood``) and so do task keys
    (``mmlu_pro``), so the parse is a longest-prefix match rather than a split on "_".
    """

    found: dict[str, list[str]] = {}
    if not out_dir.is_dir():
        return found
    ordered = sorted(known, key=len, reverse=True)
    for path in sorted(out_dir.glob("*.jsonl")):
        stem = path.stem
        for key in ordered:
            if stem.startswith(key + "_"):
                found.setdefault(key, []).append(stem[len(key) + 1 :])
                break
        else:
            dataset, _, split = stem.rpartition("_")
            if dataset:
                found.setdefault(dataset, []).append(split)
    return {key: sorted(set(splits)) for key, splits in found.items()}




def load_manifest_notes(out_dir: Path | str) -> dict[str, str]:
    """``{task: note}`` from a manifest, for the benchmark report."""

    path = Path(out_dir) / "manifest.json"
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    notes = {}
    for key, entry in (manifest.get("tasks") or {}).items():
        note = entry.get("note") or ""
        if entry.get("warning"):
            note = (note + " " if note else "") + entry["warning"]
        if note:
            notes[key] = note
    return notes


def refresh_manifest(out_dir: Path | str, src: Path | str | None = None, *, seed: int = 0) -> dict:
    """Rebuild ``manifest.json`` from the JSONL files that are actually on disk.

    ``normalize_all`` writes a manifest as it goes, so running it for a subset of tasks
    used to drop the other entries. This reconstructs the full picture from the files,
    which is also what the benchmark report reads for its task notes.
    """

    out_dir = Path(out_dir)
    plans = {plan.key: plan for plan in build_task_plans(Path(src) if src else Path("datasets"), seed=seed)}
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": str(src) if src else None,
        "out_dir": str(out_dir),
        "tasks": {},
        "rebuilt_from_files": True,
    }
    for task, splits in sorted(catalog(out_dir).items()):
        plan = plans.get(task)
        entry: dict[str, Any] = {
            "title": plan.title if plan else task,
            "state": plan.state if plan else "",
            "decision": plan.decision if plan else "",
            "note": plan.note if plan else "",
            "warning": plan.warning if plan else "",
            "splits": {},
        }
        for split in splits:
            path = out_dir / f"{task}_{split}.jsonl"
            items = load_normalized(path)
            entry["splits"][split] = {
                "file": str(path),
                "items": len(items),
                "decisions": sum(len(entry_.questions) for entry_ in items),
                "question_ids": sorted({qid for entry_ in items for qid in entry_.questions}),
                "example": _example(items[0]) if items else None,
            }
        entry["status"] = "ok" if entry["splits"] else "empty"
        manifest["tasks"][task] = entry
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
