"""A tiny random Qwen3.5 backbone and a stub tokenizer, for fast CPU tests.

The full model is a 4.5B-parameter checkpoint; the pipeline around it (state
sharing, branch isolation, readout, accounting, calibration) is best verified on a
small model that exercises the same code paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StubTokenizer:
    """Byte-level tokenizer: every ASCII character is one token."""

    vocab_size: int = 512
    pad_token_id: int = 0
    eos_token_id: int = 1

    def _ids(self, text: str) -> list[int]:
        return [ord(ch) % self.vocab_size for ch in text]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._ids(text)

    def decode(self, ids) -> str:
        return "".join(chr(i) for i in ids)

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        ids = self._ids(text)
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


def build_tiny_backbone(seed: int = 0, vocab_size: int = 512):
    """Randomly initialised Qwen3.5 model with the same architecture as the 4B one."""

    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    torch.manual_seed(seed)
    config = Qwen3_5Config(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": vocab_size,
            "full_attention_interval": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 4096,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 64,
            "spatial_merge_size": 2,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "in_channels": 3,
        },
    )
    model = Qwen3_5ForConditionalGeneration(config)
    model.eval()
    return model


def build_tiny_engine(**kwargs):
    """A :class:`qwenjev.engine.QwenJevLite` wired to the tiny backbone."""

    from .config import QwenJevConfig
    from .engine import QwenJevLite

    config = QwenJevConfig(device="cpu", dtype="float32", **kwargs)
    tokenizer = StubTokenizer()
    model = build_tiny_backbone(vocab_size=tokenizer.vocab_size)
    model = model.to(dtype=torch.float32)
    return QwenJevLite(model, tokenizer, config)
