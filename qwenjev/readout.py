"""Probability readouts: the part of the graph that replaces text generation.

The essay describes the readout as ``z = W h + b`` followed by a softmax over the
``K`` allowed answers, and notes that the rows of ``W`` may be reserved rows of the
language-model head rather than a separately trained classifier. Both arrangements
are implemented here:

* :class:`ReservedLabelReadout` reads the mass the pretrained LM head puts on the
  reserved option-label tokens (works out of the box, ``K <= 26``);
* :class:`SlotHeadReadout` is the dedicated ``K``-slot head, initialised from those
  same reserved rows and then trained against outcomes by :mod:`qwenjev.rlcd`;
* :class:`PointerReadout` scores each option's own final hidden state against the
  decision position ("pointer-style scorer", essay section 4).

Every readout returns a ``(B, K_max)`` tensor of log-probabilities; rows with fewer
options are padded with ``-inf``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import SLOT_LETTERS


@dataclass
class ReadoutLogits:
    """Log-probabilities over option slots plus the pre-softmax scores."""

    log_probs: torch.Tensor  # (B, K_max), -inf padding for short rows
    raw_logits: torch.Tensor  # (B, K_max)

    @property
    def probabilities(self) -> torch.Tensor:
        return self.log_probs.exp()


class Readout(nn.Module):
    """Interface: map hidden states to log-probabilities over the option slots."""

    name = "readout"
    #: highest number of option slots the readout supports
    max_options: int = len(SLOT_LETTERS)
    #: whether scoring needs the per-option hidden states (listwise pointer)
    uses_option_states: bool = False

    def score(
        self,
        decision_hidden: torch.Tensor,
        *,
        labels: list[list[str]],
        option_hidden: list[torch.Tensor] | None = None,
    ) -> ReadoutLogits:  # pragma: no cover - interface
        raise NotImplementedError

    @staticmethod
    def _pad_rows(rows: list[torch.Tensor], device) -> torch.Tensor:
        k_max = max(r.shape[0] for r in rows)
        out = torch.full((len(rows), k_max), float("-inf"), dtype=torch.float32, device=device)
        for i, row in enumerate(rows):
            out[i, : row.shape[0]] = row
        return out

    @staticmethod
    def _renormalise(rows: list[torch.Tensor], device) -> torch.Tensor:
        """Per-row softmax over exactly that row's slots, padded with -inf."""
        return F.log_softmax(Readout._pad_rows(rows, device), dim=-1)


class ReservedLabelReadout(Readout):
    """Mass on reserved label tokens, renormalised over the allowed option slots."""

    name = "reserved_label"
    max_options = len(SLOT_LETTERS)

    def __init__(self, tokenizer, lm_head: nn.Linear):
        super().__init__()
        self._lm_head = [lm_head]  # plain reference; the backbone owns the weights
        self._forms: dict[str, list[int]] = {}
        for letter in SLOT_LETTERS:
            forms: list[int] = []
            for text in (letter, " " + letter):
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) == 1 and ids[0] not in forms:
                    forms.append(ids[0])
            if not forms:  # pragma: no cover - tokenizer specific
                raise RuntimeError(f"tokenizer has no single-token form for label {letter!r}")
            self._forms[letter] = forms

    @property
    def weight(self) -> torch.Tensor:
        return self._lm_head[0].weight

    def score(
        self,
        decision_hidden: torch.Tensor,
        *,
        labels: list[list[str]],
        option_hidden: list[torch.Tensor] | None = None,
    ) -> ReadoutLogits:
        device = decision_hidden.device
        rows: list[torch.Tensor] = []
        for i, row_labels in enumerate(labels):
            ids: list[int] = []
            slot_cols: list[list[int]] = []
            for letter in row_labels:
                cols = []
                for token_id in self._forms[letter]:
                    cols.append(len(ids))
                    ids.append(token_id)
                slot_cols.append(cols)
            table = self.weight[torch.tensor(ids, device=device, dtype=torch.long)].float()
            log_probs = F.log_softmax(decision_hidden[i].float() @ table.T, dim=-1)
            rows.append(
                torch.stack([torch.logsumexp(log_probs[cols], dim=-1) for cols in slot_cols])
            )
        return ReadoutLogits(log_probs=self._renormalise(rows, device), raw_logits=self._pad_rows(rows, device))


class SlotHeadReadout(Readout):
    """Dedicated ``K``-slot head: ``z_k = w_k . h + b_k`` (essay section 1)."""

    name = "slot_head"

    def __init__(self, hidden_size: int, max_slots: int = 256, init_weight: torch.Tensor | None = None):
        super().__init__()
        self.max_options = max_slots - 1  # the API caps requests at 255 options
        self.proj = nn.Linear(hidden_size, max_slots, bias=False)
        self.max_slots = max_slots
        with torch.no_grad():
            self.proj.weight.zero_()
            if init_weight is not None:
                n = min(init_weight.shape[0], max_slots)
                self.proj.weight[:n] = init_weight[:n].to(self.proj.weight.dtype)

    @property
    def weight(self) -> torch.Tensor:
        return self.proj.weight.detach()

    def score(
        self,
        decision_hidden: torch.Tensor,
        *,
        labels: list[list[str]],
        option_hidden: list[torch.Tensor] | None = None,
    ) -> ReadoutLogits:
        all_logits = decision_hidden.float() @ self.proj.weight.float().T
        rows = [all_logits[i, : len(row_labels)] for i, row_labels in enumerate(labels)]
        return ReadoutLogits(
            log_probs=self._renormalise(rows, decision_hidden.device),
            raw_logits=self._pad_rows(rows, decision_hidden.device),
        )


class PointerReadout(Readout):
    """Pointer scorer: compare the decision state with each option's own state."""

    name = "pointer"
    uses_option_states = True

    def __init__(self, hidden_size: int, max_slots: int = 256, init_weight: torch.Tensor | None = None):
        super().__init__()
        self.max_options = max_slots - 1
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = hidden_size**-0.5
        with torch.no_grad():
            nn.init.normal_(self.query.weight, std=0.02)
            nn.init.normal_(self.key.weight, std=0.02)

    def score(
        self,
        decision_hidden: torch.Tensor,
        *,
        labels: list[list[str]],
        option_hidden: list[torch.Tensor] | None = None,
    ) -> ReadoutLogits:
        assert option_hidden is not None, "pointer readout needs per-option hidden states"
        q = self.query(decision_hidden.float())
        rows: list[torch.Tensor] = []
        for i, opt in enumerate(option_hidden):
            k = self.key(opt.float())
            rows.append((q[i] * k).sum(-1) * self.scale)
        return ReadoutLogits(
            log_probs=self._renormalise(rows, decision_hidden.device),
            raw_logits=self._pad_rows(rows, decision_hidden.device),
        )


def build_readout(
    name: str, tokenizer, lm_head: nn.Linear, hidden_size: int, max_slots: int = 256
) -> Readout:
    if name == "reserved_label":
        return ReservedLabelReadout(tokenizer, lm_head)
    init = None
    try:
        ids = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in SLOT_LETTERS]
        init = lm_head.weight[torch.tensor(ids)].detach().clone()
    except Exception:  # pragma: no cover - tokenizer specific
        init = None
    if name == "slot_head":
        return SlotHeadReadout(hidden_size, max_slots=max_slots, init_weight=init)
    if name == "pointer":
        return PointerReadout(hidden_size, max_slots=max_slots, init_weight=init)
    raise ValueError(f"unknown readout {name!r}")
