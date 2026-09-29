"""Compose the shipped readout: private rows per label space, on the reserved rows.

The trained slot head answers a question with the row for each option's *position*,
which means it can only serve label spaces the reserved letter rows cover (``K <= 26``).
A 150-way intent task needs more rows than exist: under the positional scheme its last
124 options compete for one row, which is why CLINC150 sat at 0.077.

The composed head has three parts:

* rows ``0 .. SLOT_RESERVE-1`` stay the *pretrained* label rows, so an answer the head
  has never seen reads what the backbone itself would say - the behaviour that keeps
  ``python demo.py`` honest on a brand-new question;
* every label space the training mix covered, of any width, gets its own block: rows
  copied from the trained head by position where the reserved budget covered it, and
  prototype rows (the mean state of each answer, one forward pass, no gradient steps)
  where it did not;
* an answer in none of those blocks still falls back to its reserved row.

    python scripts/compose_wide_rows.py \\
        --base models/_v7/readout.pt \\
        --wide models/_p3/readout.pt \\
        --out models/qwenjev-multitask-v2/readout.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwenjev.schema import FAMILY_SEP, SLOT_LETTERS, SLOT_RESERVE  # noqa: E402


def label_space_size(answer: str) -> int:
    """Number of options in the label space an answer id belongs to."""

    # type | question id | every option key ... | this answer's key
    return max(1, len(answer.split(FAMILY_SEP)) - 3)


def option_positions(data_dir: Path) -> dict[str, int]:
    """Where each answer sits inside the row it was trained on.

    The trained head reads a row by the option's position in its rendered list, so the
    position has to come from the order the normaliser actually wrote the criteria in -
    which is not always sorted (FEVER's is ``supports / refutes / not_enough_info``).
    """

    positions: dict[str, int] = {}
    for path in sorted(data_dir.glob("*_train.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                for question_id, spec in record["questions"].items():
                    criteria = spec.get("criteria") or {}
                    if not isinstance(criteria, dict):
                        continue
                    space = FAMILY_SEP.join(sorted(str(key) for key in criteria))
                    prefix = f"{spec.get('type', 'choice')}{FAMILY_SEP}{question_id}{FAMILY_SEP}{space}"
                    for index, key in enumerate(criteria):
                        positions.setdefault(f"{prefix}{FAMILY_SEP}{key}", index)
    return positions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="checkpoint holding the trained positional rows")
    parser.add_argument("--wide", required=True, help="checkpoint holding the reserved + prototype rows")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "ready"),
                        help="formatted train files, read for the option order of each label space")
    parser.add_argument("--source", default=None, help="card.json to record the merge in")
    args = parser.parse_args()

    base = torch.load(args.base, map_location="cpu")
    wide = torch.load(args.wide, map_location="cpu")
    base_weight = base["readout"]["proj.weight"].float()
    wide_weight = wide["readout"]["proj.weight"].float()
    wide_table = wide.get("slot_table") or {}

    positions = option_positions(Path(args.data_dir))
    head = wide_weight[:SLOT_RESERVE].clone()  # pretrained label rows + neutral row
    new_rows = []
    table: dict[str, int] = {}
    unplaced = 0
    for answer, row in sorted(wide_table.items(), key=lambda kv: (label_space_size(kv[0]), kv[0])):
        if label_space_size(answer) > len(SLOT_LETTERS):
            new_rows.append(wide_weight[row])
        else:
            index = positions.get(answer)
            if index is None or index >= base_weight.shape[0]:
                unplaced += 1
                continue
            new_rows.append(base_weight[index])
        table[answer] = SLOT_RESERVE + len(new_rows) - 1
    if not new_rows:
        raise SystemExit("no label space found in the prototype checkpoint")

    weight = torch.cat([head, torch.stack(new_rows)], dim=0)
    print(
        f"{SLOT_RESERVE} reserved rows + {len(new_rows)} private rows for "
        f"{len({a.rsplit(FAMILY_SEP, 1)[0] for a in table})} label spaces -> {weight.shape[0]} rows"
    )
    if unplaced:
        print(f"  {unplaced} answers had no position in {args.data_dir}; they keep their reserved row")

    state = dict(wide["readout"])
    state["proj.weight"] = weight
    payload = {
        "readout": state,
        "readout_name": wide.get("readout_name", "slot_head"),
        "temperature": float(wide.get("temperature", 1.0)),
        "slot_table": table,
        "composed_from": {
            "base": str(args.base),
            "wide": str(args.wide),
            "answers": len(table),
            "reserved": SLOT_RESERVE,
            "unplaced": unplaced,
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
