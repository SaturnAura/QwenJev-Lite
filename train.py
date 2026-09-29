"""Train one QwenJev-lite readout on every formatted train split.

    python train.py                        # default settings, writes models/qwenjev-multitask-v2/
    python train.py --epochs 2 --max-samples 12000
    python train.py --tasks snli jigsaw    # only these tasks

The training data is the class-balanced mix of every ``data/ready/<task>_train.jsonl``
(the formatted sets produced from ``data/raw`` and the query/document collections).
The output folder gets ``readout.pt`` and ``card.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from tqdm.auto import tqdm

from qwenjev.config import QwenJevConfig
from qwenjev.datasets import items_to_samples
from qwenjev.engine import QwenJevLite
from qwenjev.normalize import load_normalized
from qwenjev.rlcd import RLCDFineTune, _balance_samples

#: Long-option tasks whose label names are missing in the provided files; they cost the
#: most time per step and teach the readout little, so they are skipped by default.
DEFAULT_EXCLUDE = ("clinc150", "hwu64")


def discover_tasks(data_dir: Path, only: list[str] | None = None, exclude: list[str] | None = None) -> list[str]:
    tasks = sorted(path.stem[: -len("_train")] for path in data_dir.glob("*_train.jsonl"))
    if only:
        tasks = [task for task in tasks if task in set(only)]
    if exclude:
        tasks = [task for task in tasks if task not in set(exclude)]
    return tasks


def assemble(
    data_dir: Path,
    tasks: list[str],
    *,
    decisions_per_task: int,
    items_per_task: int,
    items_per_label: int,
    seed: int,
    progress: bool,
):
    rng = random.Random(seed)
    samples: list = []
    per_task: dict[str, dict] = {}
    for task in tqdm(tasks, desc="loading train splits", unit="task", disable=not progress):
        path = data_dir / f"{task}_train.jsonl"
        if not path.is_file():
            tqdm.write(f"  skip {task}: no {path.name}")
            continue
        items = load_normalized(path)
        rng.shuffle(items)
        # A many-label task needs roughly a fixed number of examples *per class*;
        # otherwise a 150-way classification gets the same 150 rows as a 3-way one.
        labels = 0
        for spec in items[0].questions.values():
            criteria = spec.get("criteria")
            labels = max(labels, len(criteria) if isinstance(criteria, dict) else 2)
        budget = max(items_per_task, items_per_label * labels)
        taken: list = []
        decisions = 0
        for entry in items:
            taken.append(entry)
            decisions += len(entry.questions)
            if budget and len(taken) >= budget:
                break
            if decisions_per_task and decisions >= decisions_per_task:
                break
        samples.extend(items_to_samples(taken))
        per_task[task] = {"items": len(taken), "decisions": decisions}
        tqdm.write(f"  {task:<24} {len(taken):>5} items  {decisions:>6} decisions")
    return samples, per_task


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/ready", help="folder with <task>_train.jsonl")
    parser.add_argument("--model-dir", default="models/qwenjev-multitask-v2")
    parser.add_argument("--model", default=r"C:\qwen3.5-4B", help="backbone checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", nargs="*", default=None, help="only these tasks")
    parser.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE),
                        help="tasks to leave out (default: the two long-option tasks)")
    parser.add_argument("--decisions-per-task", type=int, default=250)
    parser.add_argument("--items-per-task", type=int, default=0,
                        help="cap by distinct states, not decisions (0 = off)")
    parser.add_argument("--items-per-label", type=int, default=0,
                        help="also allow this many states per option slot (0 = off)")
    parser.add_argument("--max-samples", type=int, default=6000, help="cap after balancing (0 = no cap)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--objective", default="log_loss", choices=["log_loss", "brier"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-balance", action="store_true", help="keep the natural class mix")
    parser.add_argument("--anchor", type=float, default=0.0,
                        help="L2 pull toward the pretrained readout (0 = free training)")
    parser.add_argument("--report", default="artifacts/train_multitask_v2.json")
    parser.add_argument("--quiet", action="store_true", help="hide the progress bars")
    args = parser.parse_args()
    progress = not args.quiet

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"no such data folder: {data_dir}", file=sys.stderr)
        return 2
    tasks = discover_tasks(data_dir, args.tasks, args.exclude)
    if not tasks:
        print(f"no *_train.jsonl found in {data_dir}", file=sys.stderr)
        return 2
    print(f"training tasks ({len(tasks)}): {', '.join(tasks)}")

    samples, per_task = assemble(
        data_dir,
        tasks,
        decisions_per_task=args.decisions_per_task,
        items_per_task=args.items_per_task,
        items_per_label=args.items_per_label,
        seed=args.seed,
        progress=progress,
    )
    raw_total = len(samples)
    if not args.no_balance:
        samples = _balance_samples(samples, seed=args.seed)
        random.Random(args.seed).shuffle(samples)
    if args.max_samples and len(samples) > args.max_samples:
        samples = samples[: args.max_samples]
    print(
        f"{raw_total} training decisions -> {len(samples)} after "
        f"{'class balancing' if not args.no_balance else 'no balancing'}"
    )

    config = QwenJevConfig(
        model_path=args.model,
        device=args.device,
        readout="slot_head",
        readout_checkpoint=None,
    )
    print(f"loading backbone {args.model} ...")
    engine = QwenJevLite.from_pretrained(config=config)
    tuner = RLCDFineTune(
        engine,
        objective=args.objective,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        seed=args.seed,
        balance="none",  # already balanced above
        anchor=args.anchor,
    )
    report = tuner.train(samples, epochs=args.epochs, progress=progress)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "readout.pt"
    tuner.save(str(checkpoint))
    card = {
        "model_dir": str(model_dir),
        "backbone": args.model,
        "readout": "slot_head",
        "objective": args.objective,
        "balance": "none" if args.no_balance else "class",
        "tasks": per_task,
        "decisions_raw": raw_total,
        "decisions_used": len(samples),
        "epochs": report.epochs,
        "steps": report.steps,
        "mean_loss": report.loss,
        "seconds": round(report.seconds, 1),
        "lr": args.lr,
        "anchor": args.anchor,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    (model_dir / "card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(card, indent=2), encoding="utf-8")

    print()
    print(f"steps          : {report.steps} over {report.epochs} epoch(s) "
          f"({report.seconds/60:.1f} min)")
    print(f"mean train loss: {report.loss:.4f}")
    print(f"saved          : {checkpoint}")
    print(f"card           : {model_dir / 'card.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
