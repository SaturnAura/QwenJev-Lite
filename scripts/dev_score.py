"""Score a readout checkpoint on a small dev slice, next to the pretrained readout.

Used to pick hyper-parameters without paying for the full test every time:

    python scripts/dev_score.py --checkpoint models/qwenjev-multitask-v2/readout.pt

The slice covers all three question types plus the tasks that are known to be hard
(the relevance yes/no judgements).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwenjev.backends import QwenBackend  # noqa: E402
from qwenjev.config import QwenJevConfig, default_model_path  # noqa: E402
from qwenjev.engine import QwenJevLite  # noqa: E402
from qwenjev.evaluation import evaluate_items  # noqa: E402
from qwenjev.normalize import load_normalized  # noqa: E402

DEV_TASKS = (
    "scifact_rel_bool",
    "nfcorpus_rel_bool",
    "arguana_rel_bool",
    "jigsaw",
    "goemotions",
    "mnli",
    "snli",
    "scifact_rel_score",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/ready")
    parser.add_argument("--model", default=None, help="backbone path (default $QWENJEV_MODEL)")
    parser.add_argument("--tasks", nargs="*", default=list(DEV_TASKS))
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.model = args.model or default_model_path()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )
    model.eval()
    zero = QwenBackend(QwenJevLite(model, tokenizer, QwenJevConfig(readout="reserved_label")))
    trained = QwenBackend(
        QwenJevLite(
            model,
            tokenizer,
            QwenJevConfig(readout="slot_head", readout_checkpoint=args.checkpoint),
        )
    )

    rows = []
    for task in tqdm(args.tasks, desc="dev slice", unit="task", disable=args.quiet):
        items = load_normalized(Path(args.data_dir) / f"{task}_test.jsonl")
        if args.limit and len(items) > args.limit:
            stride = max(1, len(items) // args.limit)
            items = items[::stride][: args.limit]
        line = {"task": task}
        for name, backend in (("zero", zero), ("trained", trained)):
            try:
                summary = evaluate_items(
                    backend, items, dataset=task, batch_size=args.batch, progress=False
                ).summary()
            except Exception as exc:  # a readout that cannot answer this label space
                line[name] = float("nan")
                line[f"{name}_ece"] = float("nan")
                line[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
                continue
            line[name] = summary["overall"]["accuracy"]
            line[f"{name}_ece"] = summary["overall"]["ece"]
        rows.append(line)

    print(f"\n{'task':<22} {'zero':>7} {'trained':>8} {'delta':>7}")
    def fmt(value: float) -> str:
        return "n/a" if value != value else f"{value:.3f}"

    for row in rows:
        delta = row["trained"] - row["zero"]
        delta_text = "n/a" if delta != delta else f"{delta:+.3f}"
        print(
            f"{row['task']:<22} {fmt(row['zero']):>7} {fmt(row['trained']):>8} "
            f"{delta_text:>7}"
        )
    paired = [r for r in rows if r["zero"] == r["zero"] and r["trained"] == r["trained"]]
    zero_mean = sum(r["zero"] for r in paired) / len(paired)
    trained_mean = sum(r["trained"] for r in paired) / len(paired)
    print(f"{'MEAN':<22} {zero_mean:>7.3f} {trained_mean:>8.3f} {trained_mean-zero_mean:>+7.3f}")
    mean = sum(r["trained"] for r in rows if r["trained"] == r["trained"]) / max(
        1, len([r for r in rows if r["trained"] == r["trained"]])
    )
    print(f"{'MEAN (trained, all tasks)':<22} {'':>7} {mean:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
