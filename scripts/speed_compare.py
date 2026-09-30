"""Time QwenJev-lite and Laya on the same requests, with the units spelled out.

Two different units get quoted for these models and they do not mean the same thing:

* Laya answers **one question per call**, so its cost per call *is* its cost per
  decision, and it re-encodes the state for every one of them;
* our engine answers **every question about one state in one batched call** (a shared
  prefill, then the branches), so a call holds several decisions and its per-call cost
  grows only with the number of branches.

The crossover therefore depends on how long the state is and how many questions share
it, which is exactly what this script sweeps.

    python scripts/speed_compare.py                    # short and long states, 1/8/64 questions
    python scripts/speed_compare.py --repeat 3 --state-chars 4000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwenjev.config import QwenJevConfig, default_laya_path, default_model_path  # noqa: E402

SENTENCE = (
    "The customer reports that the payout to their bank was rejected by the processor "
    "three times in a row, that the bank says the account is in good standing, and that "
    "the beneficiary details match what the platform recorded. "
)


def state_of(characters: int) -> str:
    text = "Ticket history. "
    while len(text) < characters:
        text += SENTENCE
    return text[:characters]


def questions(count: int, options: int) -> dict:
    criteria = {f"label_{i}": f"Candidate answer number {i}" for i in range(options)}
    return {
        f"q{i}": {"type": "choice", "instructions": "Which option is correct?", "criteria": dict(criteria)}
        for i in range(count)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="backbone path (default $QWENJEV_MODEL)")
    parser.add_argument("--laya-path", default=None, help="Laya checkpoint (default $QWENJEV_LAYA)")
    parser.add_argument("--checkpoint", default=str(ROOT / "models" / "qwenjev-multitask-v2"))
    parser.add_argument("--state-chars", nargs="*", type=int, default=[600, 6000])
    parser.add_argument("--questions", nargs="*", type=int, default=[1, 8, 64])
    parser.add_argument("--options", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--skip-laya", action="store_true")
    args = parser.parse_args()
    args.model = args.model or default_model_path()
    args.laya_path = args.laya_path or default_laya_path()

    import torch

    from qwenjev.backends import LayaBackend, QwenBackend
    from qwenjev.engine import QwenJevLite

    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "readout.pt"
    engine = QwenJevLite.from_pretrained(
        config=QwenJevConfig(
            model_path=args.model, readout="slot_head", readout_checkpoint=str(checkpoint)
        )
    )
    ours = QwenBackend(engine)
    laya = None if args.skip_laya else LayaBackend(model_path=args.laya_path)

    print(f"questions per state: {args.questions}, options each: {args.options}, "
          f"repeats: {args.repeat}, checkpoint: {checkpoint.name}")
    print(f"{'state':>7} {'questions':>9} | {'ours ms/call':>12} {'ours ms/dec':>11} "
          f"| {'laya ms/call':>12} {'laya ms/dec':>11} | winner")
    for characters in args.state_chars:
        state = state_of(characters)
        for count in args.questions:
            requests = [(state, questions(count, args.options))]
            ours.decide_batch(requests)  # warmup
            t0 = time.perf_counter()
            for _ in range(args.repeat):
                ours.decide_batch(requests)
            ours_ms = (time.perf_counter() - t0) * 1000 / args.repeat

            laya_ms = float("nan")
            if laya is not None:
                specs = requests[0][1]
                laya.decide(state, {"q0": specs["q0"]})  # warmup
                t0 = time.perf_counter()
                for _ in range(args.repeat):
                    for qid, spec in specs.items():
                        laya.decide(state, {qid: spec})
                laya_ms = (time.perf_counter() - t0) * 1000 / args.repeat

            ours_dec = ours_ms / count
            laya_dec = laya_ms / count if laya_ms == laya_ms else float("nan")
            if laya_ms != laya_ms:
                verdict = "-"
            elif ours_ms < laya_ms:
                verdict = f"ours {laya_ms / ours_ms:.1f}x"
            else:
                verdict = f"Laya {ours_ms / laya_ms:.1f}x"
            print(
                f"{characters:>7} {count:>9} | {ours_ms:>12.0f} {ours_dec:>11.1f} "
                f"| {laya_ms:>12.0f} {laya_dec:>11.1f} | {verdict}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
