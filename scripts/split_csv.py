"""Give a single-file dataset a deterministic train/valid/test split column.

Some sources (HWU64's canonical CSV, a Jigsaw export) ship one table and no split.
This adds a ``split`` column by hashing the utterance, so the split is stable across
runs and the loader can select it:

    python scripts/split_csv.py --src NLU-Evaluation-Data-Canonical-Form.csv \\
        --out data/hwu64 --text-column answer --ratios 0.8 0.1 0.1

The loader looks for ``train.csv`` / ``valid.csv`` / ``test.csv`` (or any ``*.csv``
holding a ``split`` column).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def bucket(text: str, ratios: list[float]) -> str:
    digest = hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    cumulative = 0.0
    names = ["train", "valid", "test"][: len(ratios)]
    for name, ratio in zip(names, ratios):
        cumulative += ratio
        if value < cumulative:
            return name
    return names[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True, help="destination directory")
    parser.add_argument("--text-column", default=None, help="column used for hashing")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    args = parser.parse_args()

    with open(args.src, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise SystemExit(f"{args.src} has no rows")

    text_column = args.text_column or fieldnames[0]
    for row in rows:
        row["split"] = bucket(str(row.get(text_column, "")), list(args.ratios))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    written: dict[str, list[dict]] = {}
    for row in rows:
        written.setdefault(row["split"], []).append(row)
    for name, subset in written.items():
        path = out_dir / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames + ["split"])
            writer.writeheader()
            writer.writerows(subset)
        counts[name] = len(subset)
        print(f"{name}: {len(subset)} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
