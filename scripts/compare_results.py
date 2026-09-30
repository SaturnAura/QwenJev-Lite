"""Print the per-task comparison table from one or more recorded result files.

    python scripts/compare_results.py artifacts/results.json

Each argument is a JSON payload written by ``test.py`` or ``qwenjev benchmark``; the
first one supplies the task list and the baselines, the rest are appended as extra
columns. Family means (bool / choice / score) are printed underneath, which is how the
reports quote a single number per question type.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FAMILIES = ("bool", "choice", "score")


def accuracy(entry: dict, variant: str):
    payload = entry.get(variant)
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("overall"), dict):
        payload = payload["overall"]
    value = payload.get("accuracy")
    return float(value) if value is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="result files, first one first")
    parser.add_argument("--tasks", nargs="*", default=None, help="only these tasks")
    parser.add_argument("--family", default=None, choices=list(FAMILIES), help="only this family")
    parser.add_argument("--ece", action="store_true", help="also print ECE")
    args = parser.parse_args()

    payloads = []
    for name in args.results:
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        payloads.append((path.stem, json.loads(path.read_text(encoding="utf-8"))))
    base_name, base = payloads[0]
    families = base["families"]
    columns = [(base_name, variant) for variant in base["variants"]]
    for stem, payload in payloads[1:]:
        columns += [(stem, variant) for variant in payload["variants"]]

    by_name = dict(payloads)
    tasks = sorted(families)
    if args.family:
        tasks = [task for task in tasks if families[task] == args.family]
    if args.tasks:
        wanted = set(args.tasks)
        tasks = [task for task in tasks if task in wanted]

    header = f"{'task':<24}{'family':<8}"
    for name, variant in columns:
        header += f"{name[:9] + ':' + variant[:8]:>19}"
    print(header)
    table: dict[str, list[float | None]] = {}
    for task in tasks:
        values = []
        for name, variant in columns:
            entry = base["results"].get(task, {}) if name == base_name else by_name[name]["results"].get(task, {})
            values.append(accuracy(entry, variant))
        table[task] = values
        row = f"{task:<24}{families[task]:<8}"
        row += "".join("%19s" % ("-" if value is None else "%.3f" % value) for value in values)
        print(row)

    print()
    for family in FAMILIES:
        subset = [task for task in tasks if families[task] == family]
        if not subset:
            continue
        for index, (name, variant) in enumerate(columns):
            values = [table[task][index] for task in subset if table[task][index] is not None]
            if values:
                print(
                    "  %-9s %-14s n=%2d mean=%.4f"
                    % (family, f"{name}:{variant}", len(values), sum(values) / len(values))
                )
        values = [value for task in subset for value in [table[task][0]] if value is not None]
        means = []
        for index in range(len(columns)):
            column = [table[task][index] for task in subset if table[task][index] is not None]
            means.append((len(column), sum(column) / len(column) if column else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
