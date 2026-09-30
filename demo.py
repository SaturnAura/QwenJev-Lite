"""User-facing demo: typed decisions over any text, in one forward pass.

    python demo.py                                   # the reference example
    python demo.py --state "My card was declined twice." \
        --claim "This needs urgent attention" \
        --choice "Which team?"=payments=Payouts,account=Login,other=Else \
        --score "How urgent?"=no time pressure,needs attention,blocking issue

    python demo.py --questions questions.json --state-file ticket.txt

Every question is typed (``bool`` / ``choice`` / ``score``), all of them are answered
in a single forward pass, and the output is a distribution per question - no text is
generated. ``--checkpoint-dir`` picks the trained readout (``readout.pt``); without it the
pretrained readout is used.
"""

from __future__ import annotations

import argparse
import json
import sys

from qwenjev.config import QwenJevConfig, default_model_path
from pathlib import Path

EXAMPLE_STATE = (
    "My payouts have failed three times. The bank says everything is fine. "
    "The customer has been waiting nine days and is threatening a chargeback."
)
EXAMPLE_QUESTIONS = {
    "queue": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "payments": "Payout failures and payment processing",
            "account": "Login and account access",
            "other": "Something else",
        },
    },
    "escalate": {
        "type": "bool",
        "instructions": "Does this message require urgent human attention?",
        "claim": "This message requires urgent human attention.",
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": {
            "low": "No time pressure",
            "medium": "Needs attention this week",
            "high": "Blocking issue or hard deadline",
        },
    },
}


def parse_choice(text: str) -> dict:
    """``"Which team?"=payments=Payouts,account=Login`` -> a choice question."""

    instruction, _, body = text.partition("=")
    criteria = {}
    for item in body.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, description = item.partition("=")
        criteria[key.strip()] = description.strip() or key.strip()
    if len(criteria) < 2:
        raise ValueError("--choice needs at least two options (key=description,key=description)")
    return {"type": "choice", "instructions": instruction.strip(), "criteria": criteria}


def parse_score(text: str) -> dict:
    """``"How urgent?"=low,medium,high`` -> an ordered score question."""

    instruction, _, body = text.partition("=")
    levels = [level.strip() for level in body.split(",") if level.strip()]
    if len(levels) < 2:
        raise ValueError("--score needs at least two ordered levels")
    return {
        "type": "score",
        "instructions": instruction.strip(),
        "criteria": {f"L{i}": level for i, level in enumerate(levels)},
    }


def parse_bool(text: str) -> dict:
    return {
        "type": "bool",
        "instructions": text.strip(),
        "claim": text.strip() if text.strip().endswith(".") else text.strip() + ".",
    }


def bar(probability: float, width: int = 28) -> str:
    return "█" * int(round(probability * width))


def show(state: str, questions: dict, response) -> None:
    print("state")
    for line in state.splitlines():
        print(f"  {line}")
    print()
    for qid, decision in response.decisions.items():
        question = questions[qid]
        print(f"{qid}  [{decision.type}]  {question.get('instructions', '')}")
        for key, probability in decision.probabilities.items():
            mark = "->" if key == decision.answer else "  "
            label = key
            if isinstance(question.get("criteria"), dict):
                label = f"{key}={question['criteria'].get(key, key)}"
            print(f"  {mark} {probability:6.3f} {bar(probability):<28} {label}")
        extra = f"  score={decision.score_value:.2f}" if decision.score_value is not None else ""
        print(f"     answer={decision.answer}  confidence={decision.confidence:.3f}{extra}")
        print()
    usage, timing = response.usage, response.timing
    print(
        f"[{usage['questions']} questions, {usage['input_tokens']} input tokens, "
        f"{usage['output_tokens']} billed output tokens, {timing['total_ms']:.0f} ms, "
        f"readout={timing['readout']}, shared_state={timing['shared_state']}]"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", default=None, help="the text to decide about")
    parser.add_argument("--state-file", default=None)
    parser.add_argument("--questions", default=None, help="JSON string or a .json file")
    parser.add_argument("--bool", nargs="*", default=[], help="yes/no questions")
    parser.add_argument("--choice", nargs="*", default=[], help='"instruction"=key=desc,key=desc')
    parser.add_argument("--score", nargs="*", default=[], help='"instruction"=level,level,level')
    parser.add_argument("--model", default=None, help="backbone path or Hub repo id (default $QWENJEV_MODEL)")
    parser.add_argument("--checkpoint-dir", "--model-dir", dest="checkpoint_dir",
                        default="models/qwenjev-multitask-v2",
                        help="folder with the trained readout.pt (omit to use the pretrained readout)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json", action="store_true", help="print the raw payload instead")
    args = parser.parse_args()
    args.model = args.model or default_model_path()

    questions: dict = {}
    if args.bool:
        questions.update({f"bool{i}": parse_bool(text) for i, text in enumerate(args.bool)})
    if args.choice:
        questions.update({f"choice{i}": parse_choice(text) for i, text in enumerate(args.choice)})
    if args.score:
        questions.update({f"score{i}": parse_score(text) for i, text in enumerate(args.score)})
    if args.questions:
        raw = Path(args.questions).read_text(encoding="utf-8") if Path(args.questions).is_file() else args.questions
        questions.update(json.loads(raw))
    if not questions:
        questions = dict(EXAMPLE_QUESTIONS)

    state = args.state
    if args.state_file:
        state = Path(args.state_file).read_text(encoding="utf-8")
    if not state:
        state = EXAMPLE_STATE

    from qwenjev.engine import QwenJevLite

    checkpoint = Path(args.checkpoint_dir) / "readout.pt" if args.checkpoint_dir else None
    if checkpoint and not checkpoint.is_file():
        print(f"no trained readout at {checkpoint}; run `python train.py` or drop --checkpoint-dir",
              file=sys.stderr)
        checkpoint = None
    config = QwenJevConfig(
        model_path=args.model,
        device=args.device,
        readout="slot_head" if checkpoint else "reserved_label",
        readout_checkpoint=str(checkpoint) if checkpoint else None,
    )
    print(f"loading {args.model} with the {'trained' if checkpoint else 'pretrained'} readout ...")
    engine = QwenJevLite.from_pretrained(config=config)
    response = engine.decide(state, questions)
    if args.json:
        print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))
    else:
        show(state, questions, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
