"""Limits and defaults observed in the essay.

The numbers below are the ones the essay recovered from the Jev API surface. They are
part of the reproduction: validation, token accounting and cache sizing all use them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Default local backbone (Qwen3.5-4B, hybrid linear/full attention, 32 layers).
DEFAULT_MODEL_PATH = r"C:\qwen3.5-4B"


@dataclass(frozen=True)
class JevLimits:
    """Hard limits the API enforces (essay section 2)."""

    #: "The API accepts at most 255 options (2**8 - 1)".
    max_options: int = 255
    #: "Each branch (the state plus one question) is capped at roughly 32,768 tokens".
    max_branch_tokens: int = 32_768
    #: "and the whole request at roughly 65,536" -- the state is counted once.
    max_request_tokens: int = 65_536
    #: Billing figure: "4 shared tokens, plus 15 per answer, plus the token length of
    #: each question's identifier".
    billing_base_tokens: int = 4
    billing_tokens_per_answer: int = 15
    #: Number of option slots the dedicated readout head is sized for. Every trained
    #: label space gets a private block, so this has to hold the 26 reserved rows plus
    #: one row per option of every label space in the training mix.
    max_slots: int = 1024


@dataclass
class QwenJevConfig:
    """Runtime configuration for :class:`qwenjev.engine.QwenJevLite`."""

    model_path: str = DEFAULT_MODEL_PATH
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    limits: JevLimits = field(default_factory=JevLimits)

    #: Readout used to turn hidden states into probabilities.
    #: ``reserved_label`` = probability mass the LM head puts on reserved label tokens
    #: (works out of the box, K <= 26). ``slot_head`` = a dedicated K-slot linear head
    #: (the essay's "final-position head", trained by RLCD). ``pointer`` = listwise
    #: pointer scorer over per-option hidden states.
    readout: str = "reserved_label"
    #: Optional checkpoint produced by :mod:`qwenjev.rlcd`.
    readout_checkpoint: str | None = None

    #: Share one state encoding across all branches of a request (essay section 2).
    share_state: bool = True
    #: Keep encoded states in an LRU so identical states are not re-encoded.
    state_cache_size: int = 4
    #: Branches are evaluated in batches of this size (essay section 7).
    branch_batch_size: int = 64
    #: A batch also stops growing once ``branches x longest_branch`` exceeds this.
    max_batch_tokens: int = 8192

    def __post_init__(self) -> None:
        if self.readout not in {"reserved_label", "slot_head", "pointer"}:
            raise ValueError(f"unknown readout {self.readout!r}")
        if self.branch_batch_size < 1:
            raise ValueError("branch_batch_size must be >= 1")
        if self.max_batch_tokens < 1:
            raise ValueError("max_batch_tokens must be >= 1")
        if self.state_cache_size < 0:
            raise ValueError("state_cache_size must be >= 0")
