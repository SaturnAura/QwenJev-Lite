"""Convert Berkeley's MMLU CSV release into the JSONL the loader reads.

    curl -L -o mmlu.tar https://people.eecs.berkeley.edu/~hendrycks/data.tar
    tar xf mmlu.tar
    python scripts/mmlu_to_jsonl.py --src data/mmlu_raw --out data/mmlu

The release lays out ``data/test/*_test.csv`` (and ``val``/``dev``), with the question
in the first column, four choices next, and the answer letter last.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

SPLIT_DIRS = {"test": "test", "val": "val", "dev": "dev", "auxiliary_train": "auxiliary_train"}


def convert_split(src: Path, split: str, out_dir: Path) -> int:
    directory = src / SPLIT_DIRS.get(split, split)
    if not directory.is_dir():
        raise SystemExit(f"no such split directory: {directory}")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
        for path in sorted(glob.glob(str(directory / "*.csv"))):
            subject = os.path.basename(path).rsplit("_", 1)[0]
            with open(path, encoding="utf-8") as source:
                for row in csv.reader(source):
                    if len(row) < 6 or not row[0].strip():
                        continue
                    choices = [cell.strip() for cell in row[1:5]]
                    answer = row[5].strip().upper()
                    if answer not in "ABCD" or not all(choices):
                        continue
                    handle.write(
                        json.dumps(
                            {
                                "question": row[0].strip(),
                                "choices": choices,
                                "answer": "ABCD".index(answer),
                                "subject": subject,
                            }
                        )
                        + "\n"
                    )
                    written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="directory that contains test/ val/ dev/")
    parser.add_argument("--out", required=True, help="destination, e.g. data/mmlu")
    parser.add_argument("--splits", nargs="+", default=["test", "val"], choices=sorted(SPLIT_DIRS))
    args = parser.parse_args()
    for split in args.splits:
        written = convert_split(Path(args.src), split, Path(args.out))
        print(f"{split}: {written} items -> {Path(args.out) / (split + '.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
