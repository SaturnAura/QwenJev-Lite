"""Serialisation of the shared state and the isolated question branches."""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Question, QuestionType

STATE_TEMPLATE = "<state>\n{state}\n</state>\n"

CHOICE_INSTRUCTION = "Reply with a single letter naming the best option."
SCORE_INSTRUCTION = "Reply with a single letter naming the best level."
DECISION_CUE = "Answer: ("


@dataclass
class RenderedBranch:
    """A rendered question branch plus the token index of every option it offers."""

    question_id: str
    text: str
    labels: list[str]
    #: character span of each option description inside ``text``
    option_char_spans: list[tuple[int, int]] = field(default_factory=list)
    #: filled in by :func:`tokenize_branch`
    option_token_spans: list[tuple[int, int]] = field(default_factory=list)
    decision_index: int = -1

    @property
    def num_options(self) -> int:
        return len(self.labels)


def render_branch(question: Question) -> RenderedBranch:
    """Render one question as a self-contained branch using the documented templates."""

    header = "Options" if question.type is QuestionType.CHOICE else "Levels"
    if question.type is QuestionType.SCORE:
        header = "Levels"
        instruction = SCORE_INSTRUCTION
    else:
        instruction = CHOICE_INSTRUCTION

    parts = [f"Question: {question.instructions}\n"]
    if question.context:
        parts.append(f"Passage:\n{question.context}\n\n")
    parts.append(f"{header}:\n")
    for label, option in zip(question.labels(), question.options):
        parts.append(f"{label}. {option.description}\n")
    parts.append(f"{instruction}\n")
    parts.append(DECISION_CUE)
    text = "".join(parts)

    spans: list[tuple[int, int]] = []
    cursor = (
        len(f"Question: {question.instructions}\n")
        + (len(f"Passage:\n{question.context}\n\n") if question.context else 0)
        + len(f"{header}:\n")
    )
    for label, option in zip(question.labels(), question.options):
        start = cursor + len(f"{label}. ")
        end = start + len(option.description)
        spans.append((start, end))
        cursor = end + 1  # newline

    return RenderedBranch(
        question_id=question.id,
        text=text,
        labels=question.labels(),
        option_char_spans=spans,
    )


def render_state(state: str) -> str:
    return STATE_TEMPLATE.format(state=state.strip())
