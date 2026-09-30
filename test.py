"""Score a decision head over a folder of formatted test splits.

Four inputs are all you need - the backbone, the trained head, the data, and how much of
it to score - and each has a default:

    # the shipped head over the whole benchmark
    python test.py --model /path/to/qwen3.5-4B --limit 0

    # your own run
    python test.py \
      --model /path/to/qwen3.5-4B \
      --data-dir data/ready \
      --checkpoint-dir models/my-run \
      --variants qwen_zeroshot qwen_trained \
      --limit 0 --batch 16 \
      --out artifacts/my-results.json

``qwen_zeroshot`` is the untrained decision head (the pretrained label rows, no training);
``qwen_trained`` loads ``<checkpoint-dir>/readout.pt`` produced by ``train.py``. Leaving
``--checkpoint-dir`` empty uses the shipped head, or scores the untrained one only if that head
is not on disk. ``--variants laya`` is available if an external Laya checkpoint is
installed, and is not needed otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

from qwenjev.benchmark import task_baseline
from qwenjev.config import default_laya_path, default_model_path
from qwenjev.evaluation import evaluate_items
from qwenjev.normalize import load_normalized


def discover_tasks(data_dir: Path, only: list[str] | None = None) -> list[str]:
    tasks = sorted(path.stem[: -len("_test")] for path in data_dir.glob("*_test.jsonl"))
    if (data_dir / "mnli_test_ood.jsonl").is_file():
        tasks.append("mnli_ood")
    if only:
        wanted = set(only)
        tasks = [task for task in tasks if task in wanted]
    return tasks


def task_path(data_dir: Path, task: str) -> Path:
    if task == "mnli_ood":
        return data_dir / "mnli_test_ood.jsonl"
    return data_dir / f"{task}_test.jsonl"


def task_family(data_dir: Path, task: str) -> str:
    """choice / bool / score, read from the task's own question definition."""

    path = task_path(data_dir, task)
    with path.open("r", encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    types = {spec.get("type") for spec in first["questions"].values()}
    for name in ("choice", "bool", "score"):
        if name in types:
            return name
    return "?"


def build_variants(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from qwenjev.backends import LayaBackend, QwenBackend
    from qwenjev.config import QwenJevConfig
    from qwenjev.engine import QwenJevLite

    variants = []
    wanted = args.variants
    if "laya" in wanted:
        print(f"loading Laya from {args.laya_path} ...")
        variants.append(("laya", LayaBackend(model_path=args.laya_path), "Laya, as shipped"))

    if "qwen_zeroshot" in wanted or "qwen_trained" in wanted:
        # One copy of the backbone, shared by every QwenJev variant: two 4B models would
        # not fit next to each other on a 24 GB card.
        print(f"loading backbone {args.model} ...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map={"": args.device}
        )
        model.eval()
        if "qwen_zeroshot" in wanted:
            config = QwenJevConfig(model_path=args.model, device=args.device, readout="reserved_label")
            engine = QwenJevLite(model, tokenizer, config)
            variants.append(("qwen_zeroshot", QwenBackend(engine), "pretrained readout, no training"))
        if "qwen_trained" in wanted:
            checkpoint = Path(args.checkpoint_dir) / "readout.pt"
            if not checkpoint.is_file():
                print(f"  no trained readout at {checkpoint} - run train.py first", file=sys.stderr)
            else:
                config = QwenJevConfig(
                    model_path=args.model,
                    device=args.device,
                    readout="slot_head",
                    readout_checkpoint=str(checkpoint),
                )
                trained = QwenJevLite(model, tokenizer, config)
                variants.append(("qwen_trained", QwenBackend(trained), f"RLCD readout {checkpoint}"))
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/ready")
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", default="",
                        help="folder with the trained readout.pt; empty = the shipped head, "
                             "or the untrained head only if that is missing")
    parser.add_argument("--model-dir", dest="checkpoint_dir", help=argparse.SUPPRESS)  # old name
    parser.add_argument("--model", default=None, help="backbone path or Hub repo id (default $QWENJEV_MODEL)")
    parser.add_argument("--laya-path", default=None, help="Laya checkpoint (default $QWENJEV_LAYA, else ./laya)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variants", nargs="+", default=["qwen_zeroshot", "qwen_trained"],
                        choices=["laya", "qwen_zeroshot", "qwen_trained"])
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=40, help="items per task (0 = all)")
    parser.add_argument("--batch", type=int, default=16,
                        help="requests per forward pass for the QwenJev variants (1 = one at a time)")
    parser.add_argument("--out", default="artifacts/results.json")
    parser.add_argument("--markdown", default="artifacts/RESULTS.md")
    parser.add_argument("--quiet", action="store_true", help="hide the progress bars")
    args = parser.parse_args()
    args.model = args.model or default_model_path()
    args.laya_path = args.laya_path or default_laya_path()
    if not args.checkpoint_dir:
        shipped = Path("models/qwenjev-multitask-v2/readout.pt")
        if shipped.is_file():
            args.checkpoint_dir = str(shipped.parent)
            print(f"no --checkpoint-dir given: using the shipped head at {args.checkpoint_dir}")
        else:
            args.checkpoint_dir = ""
            if "qwen_trained" in args.variants:
                args.variants = [v for v in args.variants if v != "qwen_trained"]
                print("no --checkpoint-dir given and no shipped head on disk: scoring the untrained head only")
    progress = not args.quiet

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"no such data folder: {data_dir}", file=sys.stderr)
        return 2
    if not any(data_dir.glob("*_test.jsonl")):
        print(
            f"{data_dir} has no <task>_test.jsonl; point --data-dir at a folder of formatted\n"
            f"decision records (one JSON object per line) - see README, 'Paths and data formats'.",
            file=sys.stderr,
        )
        return 2
    tasks = discover_tasks(data_dir, args.tasks)
    if not tasks:
        print(f"no *_test.jsonl found in {data_dir}", file=sys.stderr)
        return 2
    print(f"backbone       : {args.model}")
    print(f"data           : {data_dir}  ({len(tasks)} test splits)")
    print(f"checkpoint dir : {args.checkpoint_dir or '(none - untrained head only)'}")
    print(f"variants       : {', '.join(args.variants)}   items/task: {args.limit or 'all'}   batch: {args.batch}")
    variants = build_variants(args)
    if not variants:
        print("no variant to run", file=sys.stderr)
        return 2
    print(f"{len(tasks)} test tasks x {len(variants)} variants, {args.limit or 'all'} items each\n")

    results: dict[str, dict] = {}
    baselines: dict[str, dict] = {}
    families: dict[str, str] = {}
    outer = tqdm(tasks, desc="tasks", unit="task", disable=not progress)
    for task in outer:
        path = task_path(data_dir, task)
        items = load_normalized(path)
        stride = max(1, len(items) // args.limit) if args.limit else 1
        if args.limit and len(items) > args.limit:
            items = items[::stride][: args.limit]
        families[task] = task_family(data_dir, task)
        baselines[task] = task_baseline(items)
        results[task] = {}
        for name, backend, _note in variants:
            try:
                evaluation = evaluate_items(
                    backend,
                    items,
                    dataset=task,
                    split="test",
                    example_count=0,
                    progress=False,
                    batch_size=args.batch,
                )
                summary = evaluation.summary()
                results[task][name] = summary
                if progress:
                    tqdm.write(
                        f"  {task:<24} {name:<14} acc={summary['overall']['accuracy']:.3f} "
                        f"ece={summary['overall']['ece']:.3f} n={summary['overall']['n']}"
                    )
            except Exception as exc:  # a readout that cannot answer this question
                results[task][name] = {"error": f"{type(exc).__name__}: {exc}"}
                if progress:
                    tqdm.write(f"  {task:<24} {name:<14} FAILED {exc}")
            outer.set_postfix(task=task, variant=name, refresh=False)
    payload = {
        "data_dir": str(data_dir),
        "limit": args.limit,
        "variants": [name for name, _b, _n in variants],
        "families": families,
        "baselines": baselines,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = render(payload)
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text(markdown, encoding="utf-8")
    print()
    print(markdown)
    print(f"wrote {args.out} and {args.markdown}")
    return 0


def render(payload: dict) -> str:
    variants = payload["variants"]
    families = payload["families"]
    baseline = payload["baselines"]
    results = payload["results"]

    lines = ["# Dataset results", "",
             f"- items per task: {payload['limit'] or 'all'}",
             f"- variants: {', '.join('`%s`' % v for v in variants)}", ""]

    header = ["task", "type", "n", "majority"] + [f"{v} acc" for v in variants] + [f"{v} ECE" for v in variants]
    lines += ["## Accuracy and calibration", "",
              "| " + " | ".join(header) + " |",
              "|" + "|".join(["---"] * len(header)) + "|"]
    for task, per_variant in results.items():
        first = next((per_variant[v] for v in variants if "overall" in per_variant.get(v, {})), None)
        row = [task, families.get(task, "?"), str(int(first["overall"]["n"])) if first else "-",
               f"{baseline[task]['majority_accuracy']:.3f}"]
        for v in variants:
            entry = per_variant.get(v, {})
            row.append(f"{entry['overall']['accuracy']:.3f}" if "overall" in entry
                       else ("err" if "error" in entry else "-"))
        for v in variants:
            entry = per_variant.get(v, {})
            row.append(f"{entry['overall']['ece']:.3f}" if "overall" in entry
                       else ("err" if "error" in entry else "-"))
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Speed", "",
              "| task | variant | requests | ms/request | decisions/request |",
              "|---|---|---|---|---|"]
    for task, per_variant in results.items():
        for name in variants:
            summary = per_variant.get(name)
            if not summary or "overall" not in summary:
                continue
            usage = summary["usage"]
            requests = usage.get("requests") or summary["n_items"]
            lines.append(
                f"| {task} | {name} | {requests} | {usage['total_ms']/max(requests,1):.1f} | "
                f"{summary['usage'].get('decisions_per_request', 0):.2f} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
