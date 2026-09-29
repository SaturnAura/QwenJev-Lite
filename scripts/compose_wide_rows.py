"""Merge two readouts: the shared reserved rows, plus private rows for wide label spaces.

The trained slot head answers a question with the row for each option's *position*,
which means it can only serve label spaces the reserved letter rows cover (``K <= 26``).
A 150-way intent task needs more rows than exist: under the positional scheme its last
124 options compete for one row, which is why CLINC150 sat at 0.077.

This script keeps the trained head exactly as it is for every label space the reserved
rows cover, and appends the rows another run estimated for label spaces wider than that
budget (``scripts/…`` uses :meth:`qwenjev.rlcd.RLCDFineTune.prototypes`, the mean state
of each answer, which needs a single forward pass and no gradient steps).

    python scripts/compose_wide_rows.py \\
        --base models/qwenjev-multitask-v2/readout.pt \\
        --wide models/_p3/readout.pt \\
        --out models/qwenjev-multitask-v2/readout.pt

The result is a strict extension: answers that are in the table read their own row, and
every other answer keeps falling back to the positional row it used before.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwenjev.schema import FAMILY_SEP, SLOT_LETTERS  # noqa: E402


def label_space_size(answer: str) -> int:
    """Number of options in the label space an answer id belongs to."""

    # type | question id | every option key ... | this answer's key
    return max(1, len(answer.split(FAMILY_SEP)) - 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="checkpoint that keeps the shared rows")
    parser.add_argument("--wide", required=True, help="checkpoint with rows for wide label spaces")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-options", type=int, default=len(SLOT_LETTERS) + 1,
                        help="copy answers from label spaces larger than this")
    parser.add_argument("--source", default=None, help="card.json to record the merge in")
    args = parser.parse_args()

    base = torch.load(args.base, map_location="cpu")
    wide = torch.load(args.wide, map_location="cpu")
    base_weight = base["readout"]["proj.weight"].float()
    wide_weight = wide["readout"]["proj.weight"].float()
    wide_table = wide.get("slot_table") or {}

    copied = {}
    cursor = base_weight.shape[0]
    new_rows = []
    for answer, row in sorted(wide_table.items(), key=lambda kv: (label_space_size(kv[0]), kv[0])):
        if label_space_size(answer) <= args.min_options:
            continue
        copied[answer] = cursor + len(new_rows)
        new_rows.append(wide_weight[row])
    if not new_rows:
        raise SystemExit("no wide label space found in the second checkpoint")
    weight = torch.cat([base_weight, torch.stack(new_rows)], dim=0)
    print(
        f"base rows {base_weight.shape[0]} + {len(new_rows)} private rows "
        f"for {len({a.rsplit(FAMILY_SEP, 1)[0] for a in copied})} label spaces "
        f"-> {weight.shape[0]} rows"
    )

    table = dict(base.get("slot_table") or {})
    table.update(copied)
    state = dict(base["readout"])
    state["proj.weight"] = weight
    payload = {
        "readout": state,
        "readout_name": base.get("readout_name", "slot_head"),
        "temperature": float(base.get("temperature", 1.0)),
        "slot_table": table,
        "composed_from": {
            "base": str(args.base),
            "wide": str(args.wide),
            "wide_answers": len(copied),
            "min_options": args.min_options,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"wrote {out}")

    if args.source:
        card_path = Path(args.source)
        card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.is_file() else {}
        card["composed_from"] = payload["composed_from"]
        card["rows"] = int(weight.shape[0])
        card["trained_answers"] = len(table)
        card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"updated {card_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
