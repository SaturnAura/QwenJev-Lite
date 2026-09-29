"""Build the training splits the choice tasks were missing.

The formatted mix in ``data/ready`` leaves several choice label spaces with *no*
training data at all, or with almost none:

* MMLU-Pro ships 70 labelled validation rows for 10-way questions;
* CLINC150 / HWU64 only became trainable once their integer labels were named, and
  the 15-intent subsets that the test set scores are a *different* label space from
  the full 150- / 64-intent one;
* BANKING77 ships no training split at all.

Every split written here is disjoint from the test rows it will be scored on: the
evaluation indices are computed with the same :func:`qwenjev.normalize._iter_limited`
helper the normaliser used, and are excluded from the training rows.

    python scripts/build_extra_splits.py
    python scripts/build_extra_splits.py --mmlu-limit 6000 --top15-limit 2000

The label spaces are checked against the test files before anything is written: a
training split whose criteria disagree with the test set would train rows that the
test never reads, which is the exact failure this project is fixing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwenjev.normalize import (  # noqa: E402
    _as_list,
    _clean,
    _iter_limited,
    _int_or_none,
    _read_any,
    _read_tsv_rows,
    choice_question,
    convert_intent_parquet,
    item,
    read_label_names,
    write_items,
)
from qwenjev.schema import build_question, question_option_ids  # noqa: E402

#: Items per task in a formatted test split (``normalize``'s ``test_limit``).
TEST_LIMIT = 400


def _answers_of(record: dict) -> set[str]:
    """The rows a normalised record will read."""

    return {
        answer
        for qid, spec in record["questions"].items()
        for answer in question_option_ids(build_question(qid, spec))
    }


def _answers_in(path: Path) -> set[str]:
    """Every readout row a whole normalised file reads."""

    answers: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                answers |= _answers_of(json.loads(line))
    return answers


def _subset(items, keep: set[str], *, dataset: str, split: str, source: Path):
    """Restrict already-built intent items to ``keep`` (the test set's label space)."""

    criteria = {name: name.replace("_", " ") for name in sorted(keep)}
    out = []
    for index, entry in enumerate(items):
        target = entry.targets.get("intent")
        if target not in criteria:
            continue
        out.append(
            item(
                dataset=dataset,
                split=split,
                index=index,
                state=entry.state,
                questions={"intent": choice_question("Which intent does this utterance express?", dict(criteria))},
                targets={"intent": target},
                source=source,
            )
        )
    return out


def _test_criteria_keys(test_path: Path) -> set[str]:
    """The label names a formatted test split declares (its label space)."""

    record = json.loads(test_path.open(encoding="utf-8").readline())
    return set(record["questions"]["intent"]["criteria"])


def build_mmlu_pro(raw_dir: Path, out_dir: Path, limit: int) -> dict:
    """Training rows for MMLU-Pro, taken from the test file's unused rows."""

    source = next(raw_dir.joinpath("MMLU-Pro").glob("test-*.parquet"))
    rows = _read_any(source)
    held_out = set(_iter_limited(rows, TEST_LIMIT))
    train_rows = [row for i, row in enumerate(rows) if i not in held_out]
    # Spread over the whole complement rather than the first *limit* rows.
    picked = [train_rows[i] for i in _iter_limited(train_rows, limit)]

    letters = lambda n: [chr(ord("A") + i) for i in range(n)]  # noqa: E731
    items = []
    for index, row in enumerate(picked):
        options = [str(option) for option in _as_list(row.get("options"))]
        answer = _int_or_none(row.get("answer_index"))
        if answer is None:
            raw = row.get("answer")
            if isinstance(raw, str) and len(raw.strip()) == 1:
                answer = ord(raw.strip().upper()) - ord("A")
            else:
                answer = _int_or_none(raw)
        if not options or answer is None or not 0 <= answer < len(options):
            continue
        category = _clean(row.get("category", ""))
        items.append(
            item(
                dataset="mmlu_pro",
                split="train",
                index=row.get("question_id", index),
                state=(f"Category: {category}\n" if category else "") + _clean(row.get("question", "")),
                questions={
                    "answer": choice_question(
                        "Which option is correct?",
                        {letter: _clean(opt) for letter, opt in zip(letters(len(options)), options)},
                    )
                },
                targets={"answer": letters(len(options))[answer]},
                source=source,
                meta={"category": category, "n_options": len(options), "from_test_file": True},
            )
        )
    return {"items": items, "source": source, "held_out": len(held_out)}


def intent_pairs_from_tsv(path: Path) -> list[tuple[str, str]]:
    """``(utterance, intent)`` from a header-less ``utterance<TAB>intent`` file."""

    pairs = []
    for row in _read_tsv_rows(path):
        if len(row) < 2:
            continue
        text, label = _clean(row[0]), _clean(row[1]).lower().replace(" ", "_")
        if text and label:
            pairs.append((text, label))
    return pairs


def intent_items(pairs, *, dataset: str, split: str, source: Path, label_space=None):
    """Items for ``(utterance, intent)`` pairs, worded exactly like the normaliser."""

    names = sorted({label for _, label in pairs} if label_space is None else set(label_space))
    criteria = {name: name.replace("_", " ") for name in names}
    return [
        item(
            dataset=dataset,
            split=split,
            index=index,
            state=text,
            questions={"intent": choice_question("Which intent does this utterance express?", dict(criteria))},
            targets={"intent": label},
            source=source,
            meta={"n_labels": len(criteria)},
        )
        for index, (text, label) in enumerate(pairs)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--out", default=str(ROOT / "data" / "ready"))
    parser.add_argument("--mmlu-limit", type=int, default=6000, help="MMLU-Pro training rows (0 = all)")
    parser.add_argument("--full-limit", type=int, default=4000, help="CLINC150/HWU64 full-space rows")
    parser.add_argument("--top15-limit", type=int, default=2000, help="rows per 15-intent subset")
    args = parser.parse_args()

    raw_dir, out_dir = Path(args.raw), Path(args.out)
    written: dict[str, dict] = {}

    def emit(name: str, items, *, dataset: str, split: str, source: Path, expect: Path) -> None:
        """Write ``items``, after proving every answer the test reads has a row here."""

        path = out_dir / f"{name}_{split}.jsonl"
        summary = write_items(path, items, dataset=dataset, split=split)
        mine, theirs = _answers_in(path), _answers_in(expect)
        missing = theirs - mine
        if missing:
            path.unlink()
            raise SystemExit(
                f"{name}: {expect.name} reads {len(missing)} answers this split never trains "
                f"(e.g. {sorted(missing)[:2]}); refusing to write a split that leaves them "
                "on the pretrained rows"
            )
        written[f"{name}_{split}"] = {**summary, "source": source.name, "matches": expect.name}
        print(f"  {name}_{split:<20} {summary['items']:>6} items  {summary['decisions']:>6} decisions"
              f"  (answers cover {expect.name}: {len(mine)} rows)")

    print("MMLU-Pro")
    built = build_mmlu_pro(raw_dir, out_dir, args.mmlu_limit)
    emit("mmlu_pro", built["items"], dataset="mmlu_pro", split="train",
         source=built["source"], expect=out_dir / "mmlu_pro_test.jsonl")

    for key, patterns in (
        ("clinc150", ("train-*.parquet", "train")),
        ("hwu64", ("train-*.parquet", "train")),
    ):
        directory = next(
            (d for d in raw_dir.iterdir() if d.is_dir() and d.name.lower().startswith(key[:4])), None
        )
        if directory is None:
            print(f"{key}: no raw directory, skipped")
            continue
        names = read_label_names(directory)
        train_file = next((p for pattern in patterns for p in sorted(directory.glob(pattern))), None)
        if names is None or train_file is None:
            print(f"{key}: needs intents.txt and a train parquet, skipped")
            continue
        print(f"{key} (full label space)")
        full = convert_intent_parquet(train_file, "train", dataset=key, limit=args.full_limit, label_names=names)
        emit(key, full, dataset=key, split="train", source=train_file, expect=out_dir / f"{key}_test.jsonl")

        top15 = out_dir / f"{key}_top15_test.jsonl"
        if top15.is_file():
            print(f"{key} (15-intent subset)")
            everything = convert_intent_parquet(train_file, "train", dataset=f"{key}_top15", limit=None, label_names=names)
            subset = _subset(everything, _test_criteria_keys(top15), dataset=f"{key}_top15", split="train", source=train_file)
            subset = subset[: args.top15_limit] if args.top15_limit else subset
            emit(f"{key}_top15", subset, dataset=f"{key}_top15", split="train", source=train_file, expect=top15)

    # BANKING77 has no training file at all: the labelled TSV is used, minus the
    # rows the test split was built from.
    bank_dir = raw_dir / "BANKING77"
    tsv = next(iter(sorted(bank_dir.glob("*tfidf*.tsv"))), None)
    if tsv is not None:
        pairs = intent_pairs_from_tsv(tsv)
        held_out = set(_iter_limited(pairs, TEST_LIMIT))
        leftovers = [pair for i, pair in enumerate(pairs) if i not in held_out]
        print(f"BANKING77 ({len(leftovers)} rows left after removing the {len(held_out)} test rows)")
        # The label space is the one the test file declares, so both sides agree even
        # if the left-over rows happen to miss a class.
        space = _test_criteria_keys(out_dir / "banking77_test.jsonl") if (out_dir / "banking77_test.jsonl").is_file() else None
        full = intent_items(
            leftovers,
            dataset="banking77",
            split="train",
            source=tsv,
            label_space=space or {label for _, label in pairs},
        )
        emit("banking77", full, dataset="banking77", split="train", source=tsv,
             expect=out_dir / "banking77_test.jsonl")
        top15 = out_dir / "banking77_top15_test.jsonl"
        if top15.is_file():
            subset = _subset(full, _test_criteria_keys(top15), dataset="banking77_top15", split="train", source=tsv)
            emit("banking77_top15", subset, dataset="banking77_top15", split="train", source=tsv, expect=top15)

    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "extra_splits.json").write_text(
        json.dumps(written, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {len(written)} splits to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
