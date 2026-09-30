"""Train one QwenJev-lite decision head from a folder of formatted decision records.

Four inputs are all you need - the backbone, the data, where to save the head, and the
hyper-parameters - and every one of them has a default:

    # the whole thing, with the defaults (./qwen3.5-4B, data/ready, models/qwenjev-run-<stamp>)
    python train.py --model /path/to/qwen3.5-4B

    # an explicit run
    python train.py \
      --model /path/to/qwen3.5-4B \
      --data-dir data/ready \
      --checkpoint-dir models/my-run \
      --epochs 1 --lr 1e-3 --batch-size 8 \
      --items-per-label 12 --min-items-per-task 200 --prototype-init

The data directory holds ``<task>_train.jsonl`` files (one JSON object per line, the
engine's request shape plus ``targets``; see README, "Paths and data formats"). The output
folder is created if needed and gets ``readout.pt`` + ``card.json``. ``--checkpoint-dir``
may be left empty, in which case a timestamped folder under ``models/`` is created.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path

from tqdm.auto import tqdm

from qwenjev.config import QwenJevConfig, default_model_path
from qwenjev.datasets import items_to_samples
from qwenjev.engine import QwenJevLite
from qwenjev.normalize import load_normalized
from qwenjev.rlcd import RLCDFineTune, _balance_samples, build_slot_table
from qwenjev.schema import SLOT_LETTERS, SLOT_RESERVE

#: No task is skipped by default now that every label space owns its own rows: the
#: many-label intent tasks used to overwrite the rows the 15-label tasks read.
DEFAULT_EXCLUDE: tuple[str, ...] = ()


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
    min_items_per_task: int,
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
            # A task is done when it has seen enough *decisions* (so every task gets a
            # comparable share of the gradient), with a floor on distinct states so a
            # 28-questions-per-item task is not trained on nine texts.
            enough_decisions = not decisions_per_task or decisions >= decisions_per_task
            enough_items = len(taken) >= max(budget, min_items_per_task)
            if enough_decisions and enough_items:
                break
            if decisions_per_task and decisions >= 6 * decisions_per_task:
                break
        samples.extend(items_to_samples(taken))
        per_task[task] = {"items": len(taken), "decisions": decisions}
        tqdm.write(f"  {task:<24} {len(taken):>5} items  {decisions:>6} decisions")
    return samples, per_task


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/ready", help="folder with <task>_train.jsonl")
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", default="",
                        help="where to write readout.pt + card.json; empty = models/qwenjev-run-<timestamp>")
    parser.add_argument("--model-dir", dest="checkpoint_dir", help=argparse.SUPPRESS)  # old name
    parser.add_argument("--model", default=None,
                        help="backbone checkpoint or Hub repo id "
                             "(default: $QWENJEV_MODEL, else ./qwen3.5-4B)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", nargs="*", default=None, help="only these tasks")
    parser.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE),
                        help="tasks to leave out (default: the two long-option tasks)")
    parser.add_argument("--decisions-per-task", type=int, default=250)
    parser.add_argument("--items-per-task", type=int, default=0,
                        help="cap by distinct states, not decisions (0 = off)")
    parser.add_argument("--items-per-label", type=int, default=0,
                        help="also allow this many states per option slot (0 = off)")
    parser.add_argument("--min-items-per-task", type=int, default=0,
                        help="always take at least this many distinct states per task")
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
    parser.add_argument("--row-norm-cap", default="auto",
                        help="hold every trained row at this norm ('auto' = the norm of the "
                             "reserved label rows, 0 = unconstrained)")
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"],
                        help="sgd fits a linear readout by accumulating class means")
    parser.add_argument("--max-grad-norm", type=float, default=0.0,
                        help="clip the readout gradient (0 = off; the raw norm is ~1e4, so "
                             "clipping to 1 leaves SGD with a uselessly small step)")
    parser.add_argument("--momentum", type=float, default=0.0, help="SGD momentum")
    parser.add_argument("--prototype-init", action="store_true",
                        help="start each row at the mean state of the answer (one forward pass)")
    parser.add_argument("--init-from", default=None,
                        help="start from an existing readout.pt (keeps its rows and table)")
    parser.add_argument("--report", default="", help="optional extra copy of the card")
    parser.add_argument("--quiet", action="store_true", help="hide the progress bars")
    args = parser.parse_args()
    args.model = args.model or default_model_path()
    if not args.checkpoint_dir:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.checkpoint_dir = str(Path("models") / f"qwenjev-run-{stamp}")
        print(f"no --checkpoint-dir given: writing the decision head to {args.checkpoint_dir}")
    progress = not args.quiet

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"no such data folder: {data_dir}", file=sys.stderr)
        return 2
    if not any(data_dir.glob("*_train.jsonl")):
        print(
            f"{data_dir} has no <task>_train.jsonl; point --data-dir at a folder of formatted\n"
            f"decision records (one JSON object per line) - see README, 'Paths and data formats'.",
            file=sys.stderr,
        )
        return 2
    tasks = discover_tasks(data_dir, args.tasks, args.exclude)
    if not tasks:
        print(f"no *_train.jsonl found in {data_dir}", file=sys.stderr)
        return 2
    print(f"training tasks ({len(tasks)}): {', '.join(tasks)}")
    print(f"backbone       : {args.model}")
    print(f"data           : {data_dir}")
    print(f"checkpoint dir : {args.checkpoint_dir}")
    print(f"hyper-params   : epochs={args.epochs} lr={args.lr} batch={args.batch_size} "
          f"objective={args.objective} optimizer={args.optimizer} anchor={args.anchor}")

    samples, per_task = assemble(
        data_dir,
        tasks,
        decisions_per_task=args.decisions_per_task,
        items_per_task=args.items_per_task,
        items_per_label=args.items_per_label,
        min_items_per_task=args.min_items_per_task,
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
        readout_checkpoint=args.init_from,
    )
    print(f"loading backbone {args.model} ...")
    engine = QwenJevLite.from_pretrained(config=config)

    max_slots = engine.limits.max_slots
    table, answer_decisions = build_slot_table(samples, max_slots=max_slots)
    if not hasattr(engine.readout, "set_slot_table"):
        raise SystemExit("the slot_head readout is required to train this mix")
    # Continuing from a checkpoint on a *subset* of the tasks keeps every row the
    # earlier run trained and only appends rows for answers it never saw.
    inherited = dict(engine.readout.slot_table)
    if inherited:
        cursor = max([SLOT_RESERVE - 1, *inherited.values()]) + 1
        for answer in table:
            if answer not in inherited:
                inherited[answer] = cursor
                cursor += 1
        if cursor > max_slots:
            raise SystemExit(f"resuming needs {cursor} rows but the head has {max_slots}")
        table = inherited
        answer_decisions = {a: d for a, d in answer_decisions.items() if a in table}
    engine.readout.set_slot_table(table)
    rows_used = max([SLOT_RESERVE, *(row + 1 for row in table.values())])
    print(
        f"{len(table)} trained answers -> {rows_used}/{max_slots} rows "
        f"({SLOT_RESERVE} reserved + {rows_used - SLOT_RESERVE} trained)"
    )
    for answer, decisions in sorted(answer_decisions.items(), key=lambda kv: -kv[1])[:12]:
        qtype, question, rest = answer.split(chr(31), 2)
        space, key = rest.rsplit(chr(31), 1)
        print(
            f"  {qtype:<7} {question[:16]:<16} {len(space.split(chr(31))):>4} options  "
            f"{key[:20]:<20} {decisions:>6} decisions"
        )
    print(f"  ... {max(0, len(answer_decisions) - 12)} more answers")

    reserved_norm = float(engine.readout.weight[: len(SLOT_LETTERS)].float().norm(dim=1).mean())
    try:
        row_norm_cap = reserved_norm if args.row_norm_cap == "auto" else float(args.row_norm_cap)
    except ValueError:
        raise SystemExit("--row-norm-cap takes 'auto' or a number")
    row_norm_cap = row_norm_cap or None
    if row_norm_cap:
        print(f"row norm cap   : {row_norm_cap:.3f} (reserved label rows: {reserved_norm:.3f})")

    tuner = RLCDFineTune(
        engine,
        objective=args.objective,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        seed=args.seed,
        balance="none",  # already balanced above
        anchor=args.anchor,
        row_norm_cap=row_norm_cap,
        optimizer=args.optimizer,
        max_grad_norm=args.max_grad_norm,
        momentum=args.momentum,
    )
    if args.prototype_init:
        rows = tuner.prototypes(samples, progress=progress)
        written = engine.readout.load_rows(rows, cap=row_norm_cap)
        print(f"prototype rows : {written} answers initialised from their mean state")
    report = tuner.train(samples, epochs=args.epochs, progress=progress)

    model_dir = Path(args.checkpoint_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "readout.pt"
    tuner.save(str(checkpoint))
    card = {
        "checkpoint_dir": str(model_dir),
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
        "row_norm_cap": row_norm_cap,
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "label_spaces": answer_decisions,
        "slot_rows": rows_used,
        "max_slots": max_slots,
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
