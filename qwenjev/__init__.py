"""QwenJev-lite: a Jev-style decision model built on a Qwen3.5-4B backbone.

The design reproduces the architecture reconstructed in *Jev's Architecture Unmasked*
(archerhume, 2026-09-17):

1. end inference with a readout instead of an autoregressive decode loop;
2. encode the shared state once and isolate the question branches;
3. keep a causal transformer backbone;
4. read the option list jointly (listwise) before choosing;
5. train the predictive distribution (RLCD), then compute confidence arithmetically;
6. schedule the branches as a batch, not as a conversation.

See README.md for the mapping between the essay's sections and this package.
"""

from .config import JevLimits, QwenJevConfig
from .schema import BoolQuestion, ChoiceQuestion, QuestionType, ScoreQuestion, build_question
from .engine import QwenJevLite

__all__ = [
    "JevLimits",
    "QwenJevConfig",
    "BoolQuestion",
    "ChoiceQuestion",
    "ScoreQuestion",
    "QuestionType",
    "build_question",
    "QwenJevLite",
]

__version__ = "0.1.0"
