"""Sweep how far the trained readout may move away from the pretrained one.

The slot head starts as the reserved vocabulary rows of the backbone's own head, so the
trained matrix is a *correction* of the pretrained readout::

    W(alpha) = W0 + alpha * (W_trained - W0)

``alpha = 0`` keeps the pretrained behaviour, ``alpha = 1`` is the trained head, and
anything in between trades the tasks where training helps against the tasks where the
pretrained rows were already right (the label spaces the head never saw, such as the
15-intent tasks). The same effect can be had by a stronger L2 anchor during training;
this sweep gets it for free.

    python scripts/interpolate.py --checkpoint models/qwenjev-multitask-v2/readout.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwenjev.backends import QwenBackend  # noqa: E402
from qwenjev.config import QwenJevConfig  # noqa: E402
from qwenjev.engine import QwenJevLite  # noqa: E402
from qwenjev.evaluation import evaluate_items  # noqa: E402
from qwenjev.normalize import load_normalized  # noqa: E402
from qwenjev.schema import SLOT_LETTERS  # noqa: E402

DEV_TASKS = (
    "scifact_rel_bool",
    "arguana_rel_bool",
    "jigsaw",
    "goemotions",
    "mnli",
    "snli",
    "banking77_top15",
    "clinc150_top15",
    "mmlu_pro",
    "scifact_rel_score",
    "jigsaw_severity",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/ready")
    parser.add_argument("--model", default=r"C:\qwen3.5-4B")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--tasks", nargs="*", default=list(DEV_TASKS))
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", default=None, help="write the best alpha's head here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    backbone = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )
    backbone.eval()

    engine = QwenJevLite(
        backbone, tokenizer, QwenJevConfig(readout="slot_head")
    )
    trained = torch.load(args.checkpoint, map_location=engine.device)["readout"]
    weight_key = "proj.weight"
    W_trained = trained[weight_key].float()
    if W_trained.shape[0] != engine.readout.proj.weight.shape[0]:
        # Older checkpoints were trained with a narrower head; rebuild the readout at
        # the checkpoint's own width so the blend is exact (a 256-row head is the
        # pretrained letter rows plus zeros, which is what it was trained from).
        from qwenjev.readout import build_readout

        engine.readout = build_readout(
            "slot_head",
            tokenizer,
            backbone.get_output_embeddings(),
            int(backbone.config.text_config.hidden_size),
            max_slots=int(W_trained.shape[0]),
        ).to(engine.device)
    W0 = engine.readout.proj.weight.detach().float().clone()  # reserved letter rows

    items_by_task = {}
    for task in args.tasks:
        items = load_normalized(Path(args.data_dir) / f"{task}_test.jsonl")
        if args.limit and len(items) > args.limit:
            stride = max(1, len(items) // args.limit)
            items = items[::stride][: args.limit]
        items_by_task[task] = items

    backend = QwenBackend(engine)
    results: dict[float, dict] = {}
    for alpha in tqdm(args.alphas, desc="alpha sweep", unit="alpha", disable=args.quiet):
        with torch.no_grad():
            engine.readout.proj.weight.copy_((W0 + alpha * (W_trained - W0)).to(engine.readout.proj.weight.dtype))
        per_task = {}
        for task, items in items_by_task.items():
            summary = evaluate_items(
                backend, items, dataset=task, batch_size=args.batch, progress=False
            ).summary()
            per_task[task] = summary["overall"]["accuracy"]
        results[alpha] = per_task
        if not args.quiet:
            print(f"alpha={alpha:<5} mean={sum(per_task.values())/len(per_task):.4f}")

    header = f"{'task':<22}" + "".join(f"{a:>9}" for a in args.alphas)
    print("\n" + header)
    for task in args.tasks:
        print(f"{task:<22}" + "".join(f"{results[a][task]:>9.3f}" for a in args.alphas))
    means = {a: sum(results[a].values()) / len(results[a]) for a in args.alphas}
    print(f"{'MEAN':<22}" + "".join(f"{means[a]:>9.3f}" for a in args.alphas))
    best = max(means, key=means.get)
    print(f"\nbest alpha = {best} (mean {means[best]:.4f})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            engine.readout.proj.weight.copy_(
                (W0 + best * (W_trained - W0)).to(engine.readout.proj.weight.dtype)
            )
        state = dict(trained)
        state[weight_key] = engine.readout.proj.weight.detach().cpu()
        torch.save(
            {
                "readout": state,
                "readout_name": "slot_head",
                "temperature": 1.0,
                "alpha": best,
                "source": str(args.checkpoint),
            },
            args.out,
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
