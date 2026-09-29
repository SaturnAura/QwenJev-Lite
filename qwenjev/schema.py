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

#: Rows ``0 .. 25`` of a trained slot head stay the pretrained label rows, and row
#: ``SLOT_UNKNOWN_ROW`` is a neutral row for an option past the 26 reserved ones whose
#: label space the head never saw. Every trained label space gets its own block
#: *after* them, so training one label space can never overwrite the rows another
#: label space falls back on.
SLOT_UNKNOWN_ROW = len(SLOT_LETTERS)
SLOT_RESERVE = SLOT_UNKNOWN_ROW + 1

#: Separator used when an answer is turned into a dictionary key.
FAMILY_SEP = "\x1f"


def label_space(option_keys: "Sequence[str]") -> str:
    """Canonical name of a label space: its answer keys, sorted.

    Sorting is what makes the name independent of the order a dataset happens to list
    its options in - CLINC150's train file and its test file ship the same 150 intents
    in different orders, and they must still be the same label space.
    """

    return FAMILY_SEP.join(sorted(str(key) for key in option_keys))


def option_ids(
    question_type: "QuestionType | str",
    option_keys: "Sequence[str]",
    *,
    question_id: str = "",
) -> list[str]:
    """One stable id per allowed answer: its question, its label space, then its key.

    The essay's readout is ``z = W h + b`` over the *K allowed answers* of one question.
    Keying the rows by the question and the answer, rather than by their position in the
    row, is what lets one head serve every question without overwriting itself: every
    question owns a private block of rows, and *within* a block the row is chosen by the
    answer's key rather than by its position, so two files that list the same options in
    a different order still read the same row. The question id is part of the identity
    because two different questions can offer the same answers - six Jigsaw labels, 28
    GoEmotions emotions and five retrieval collections all ask ``yes``/``no`` - and
    pooling them into one pair of rows measured *worse than chance* on the collection
    with the fewest samples (scifact relevance 0.882 -> 0.212).
    """

    qtype = question_type.value if isinstance(question_type, QuestionType) else str(question_type)
    space = label_space(option_keys)
    return [
        f"{qtype}{FAMILY_SEP}{question_id}{FAMILY_SEP}{space}{FAMILY_SEP}{key}"
        for key in option_keys
    ]


def question_option_ids(question: "Question") -> list[str]:
    """Answer ids of a built :class:`Question`, in the order the options are shown."""

    return option_ids(
        question.type, [option.key for option in question.options], question_id=question.id
    )


def family_signature(question_type: "QuestionType | str", option_keys: "Sequence[str]") -> str:
    """Name of a whole label space (used for reporting and for dataset checks)."""

    qtype = question_type.value if isinstance(question_type, QuestionType) else str(question_type)
    return f"{qtype}{FAMILY_SEP}{label_space(option_keys)}"

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
