"""Typed decision questions, mirroring the documented Jev request shape.

The essay's example payload::

    {
        "state": "My payouts have failed three times. ...",
        "questions": {
            "queue": {
                "type": "choice",
                "instructions": "Which team should handle this ticket?",
                "criteria": {"payments": "...", "account": "...", "other": "..."}
            },
            "escalate": {
                "type": "bool",
                "instructions": "Does this message require urgent human attention?"
            }
        }
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class QuestionType(str, Enum):
    CHOICE = "choice"
    BOOL = "bool"
    SCORE = "score"


#: Label tokens reserved for option slots. The essay notes the base model's tokenizer
#: splits digits one at a time, so letters are the only stable single-token labels.
SLOT_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def slot_label(index: int) -> str:
    """``A..Z, AA, AB, ...`` for arbitrarily many options.

    The reserved-label readout still only understands the 26 single-token letters; a
    larger label space is answered by the trained slot head, which indexes slots rather
    than tokens.
    """

    if index < 0:
        raise ValueError("slot index must be >= 0")
    name = ""
    position = index + 1
    while position:
        position, remainder = divmod(position - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


@dataclass
class DecisionSample:
    """One supervised decision: (state, question, observed outcome).

    Used by the RLCD trainer and by the dataset adapters. ``spec`` carries the
    question definition when it varies per item (MMLU choices, Jigsaw label sets);
    otherwise the caller passes a shared ``question_specs`` mapping.
    """

    state: str
    question_id: str
    target: str
    spec: "dict | None" = None


@dataclass
class Question:
    """Base class for a single decision."""

    id: str
    instructions: str
    options: "list[Option]" = field(default_factory=list)
    type: QuestionType = QuestionType.CHOICE
    #: Per-branch text that belongs to this question rather than to the shared state
    #: (a candidate passage in a retrieval task). Our engine renders it into the branch;
    #: Laya's ``noul`` template wants it directly after the claim.
    context: str = ""

    def labels(self) -> list[str]:
        return [slot_label(i) for i in range(len(self.options))]


@dataclass
class Option:
    """One allowed answer: the caller's key plus the text shown to the model."""

    key: str
    description: str
    #: For ``score`` questions: the numeric value of this level.
    value: float | None = None


@dataclass
class ChoiceQuestion(Question):
    type: QuestionType = QuestionType.CHOICE


@dataclass
class BoolQuestion(Question):
    type: QuestionType = QuestionType.BOOL

    def __post_init__(self) -> None:
        if not self.options:
            self.options = [Option("yes", "Yes"), Option("no", "No")]


@dataclass
class ScoreQuestion(Question):
    """Ordered levels; the response carries a distribution and its expected value."""

    type: QuestionType = QuestionType.SCORE

    def expected_value(self, probabilities: "Mapping[str, float]") -> float:
        total = 0.0
        for i, opt in enumerate(self.options):
            value = opt.value if opt.value is not None else float(i)
            total += probabilities.get(opt.key, 0.0) * value
        return total


def _coerce_options(raw: Any, *, require_values: bool) -> list[Option]:
    if isinstance(raw, Mapping):
        items: Sequence[tuple[str, Any]] = list(raw.items())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        items = []
        for i, entry in enumerate(raw):
            if isinstance(entry, Mapping):
                key = str(entry.get("key", entry.get("id", i)))
                desc = entry.get("description", entry.get("text", ""))
                value = entry.get("value")
            else:
                key, desc, value = str(entry), str(entry), None
            items.append((key, {"description": desc, "value": value}))
    else:
        raise TypeError("options/criteria must be a mapping or a sequence")

    options: list[Option] = []
    for key, payload in items:
        if isinstance(payload, Mapping):
            desc = str(payload.get("description", payload.get("text", "")))
            value = payload.get("value")
        else:
            desc, value = str(payload), None
        options.append(
            Option(key=str(key), description=desc, value=float(value) if value is not None else None)
        )
    if require_values and any(o.value is None for o in options):
        # Fall back to declaration order when the caller does not supply values.
        for i, opt in enumerate(options):
            opt.value = float(i)
    return options


def build_question(question_id: str, spec: Mapping[str, Any] | Question) -> Question:
    """Build a typed question from the wire format (or pass a Question through)."""

    if isinstance(spec, Question):
        if not spec.id:
            spec.id = question_id
        return spec

    qtype = str(spec.get("type", "choice")).lower()
    instructions = str(spec.get("instructions", "")).strip()
    context = str(spec.get("context", "") or "")
    if not instructions:
        raise ValueError(f"question {question_id!r} is missing 'instructions'")

    if qtype in {"bool", "boolean", "yesno", "yes_no", "noul"}:
        raw = spec.get("criteria")
        options = _coerce_options(raw, require_values=False) if raw else [
            Option("yes", "Yes"),
            Option("no", "No"),
        ]
        if len(options) != 2:
            raise ValueError("bool questions take exactly two criteria (yes/no)")
        return BoolQuestion(
            id=question_id, instructions=instructions, options=options, context=context
        )

    if qtype in {"score", "ordered", "ordinal"}:
        raw = spec.get("criteria") or spec.get("levels")
        if not raw:
            raise ValueError(f"score question {question_id!r} needs ordered 'criteria'")
        options = _coerce_options(raw, require_values=True)
        return ScoreQuestion(
            id=question_id, instructions=instructions, options=options, context=context
        )

    if qtype in {"choice", "classify", "classification"}:
        raw = spec.get("criteria") or spec.get("options")
        if not raw:
            raise ValueError(f"choice question {question_id!r} needs 'criteria'")
        options = _coerce_options(raw, require_values=False)
        return ChoiceQuestion(
            id=question_id, instructions=instructions, options=options, context=context
        )

    raise ValueError(f"unknown question type {qtype!r} for question {question_id!r}")
