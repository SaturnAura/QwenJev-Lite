"""QwenJev-lite: a typed decision model on a transformer backbone.

The design has six parts:

1. end inference with a readout instead of an autoregressive decode loop;
2. encode the shared state once and isolate the question branches;
3. keep a causal transformer backbone (any backbone works: the engine only reads
   hidden states, so a multimodal encoder is usable the same way);
4. read the option list jointly (listwise) before choosing;
5. train the predictive distribution (RLCD), then compute confidence arithmetically;
6. schedule the branches as a batch, not as a conversation.

See README.md for how the pieces fit together.
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
