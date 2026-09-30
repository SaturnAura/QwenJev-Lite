"""Build the training splits that the sources do not ship.

The formatted mix in ``data/ready`` is missing training data for a few *choice* label
spaces:

* CLINC150 / HWU64 only became trainable once their integer labels were named, and
  the 15-intent subsets the test set scores are a *different* label space from the
  full 150- / 64-intent one;
* both of those sources ship an official training parquet, so the subsets can be built
  from it rather than re-split.

Every split written here is disjoint from the rows it will be scored on, and the label
spaces are checked against the test file before anything is written: a training split
whose criteria disagree with the test set would train rows that the test never reads.

    python scripts/build_extra_splits.py
    python scripts/build_extra_splits.py --top15-limit 2000
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

    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "extra_splits.json").write_text(
        json.dumps(written, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {len(written)} splits to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
