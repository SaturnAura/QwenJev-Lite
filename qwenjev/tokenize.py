"""Tokenisation of the branch templates, including option spans."""

from __future__ import annotations

from dataclasses import dataclass

from .prompt import RenderedBranch


@dataclass
class TokenizedBranch:
    question_id: str
    ids: list[int]
    labels: list[str]
    #: token index of the last token of each option description
    option_end_indices: list[int]
    option_spans: list[tuple[int, int]]
    decision_index: int

    def __len__(self) -> int:
        return len(self.ids)


def tokenize_branch(tokenizer, branch: RenderedBranch) -> TokenizedBranch:
    encoded = tokenizer(
        branch.text, add_special_tokens=False, return_offsets_mapping=True
    )
    ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])

    option_spans: list[tuple[int, int]] = []
    option_end_indices: list[int] = []
    for start, end in branch.option_char_spans:
        first = last = -1
        for i, (ts, te) in enumerate(offsets):
            if te <= start or ts >= end:
                continue
            if first < 0:
                first = i
            last = i
        if first < 0:
            raise ValueError(f"could not locate option span {start}:{end} in {branch.question_id!r}")
        option_spans.append((first, last))
        option_end_indices.append(last)

    return TokenizedBranch(
        question_id=branch.question_id,
        ids=ids,
        labels=list(branch.labels),
        option_end_indices=option_end_indices,
        option_spans=option_spans,
        decision_index=len(ids) - 1,
    )
