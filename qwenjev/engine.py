"""The QwenJev-lite engine: shared state, isolated branches, parallel readouts."""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch

from .config import JevLimits, QwenJevConfig
from .confidence import billing_output_tokens, summarise
from .prompt import render_branch, render_state
from .readout import Readout, build_readout
from .schema import Question, QuestionType, build_question, question_option_ids
from .tokenize import TokenizedBranch, tokenize_branch


@dataclass
class QuestionDecision:
    """The parallel numerical answer to one question."""

    question_id: str
    type: str
    probabilities: "OrderedDict[str, float]"
    answer: str
    confidence: float
    score_value: float | None = None
    raw_logits: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "probabilities": {k: round(v, 6) for k, v in self.probabilities.items()},
            "answer": self.answer,
        }
        if self.type == "bool":
            payload["probability"] = round(self.probabilities[self.answer], 6)
            payload["answer"] = self.answer == "yes"
        if self.score_value is not None:
            payload["score"] = round(self.score_value, 6)
        payload["confidence"] = round(self.confidence, 6)
        return payload


@dataclass
class JevResponse:
    decisions: "OrderedDict[str, QuestionDecision]"
    usage: dict[str, int]
    timing: dict[str, float | bool]
    hidden: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": {qid: d.to_dict() for qid, d in self.decisions.items()},
            "usage": self.usage,
            "timing": self.timing,
        }


class _LRUCache:
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._store: "OrderedDict[Any, Any]" = OrderedDict()

    def get(self, key: Any):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: Any, value: Any) -> None:
        if self.maxsize <= 0:
            return
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)


class QwenJevLite:
    """Encode the state once, answer every question in parallel, read out numbers."""

    def __init__(self, model, tokenizer, config: QwenJevConfig | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or QwenJevConfig()
        self.device = next(model.parameters()).device

        lm_head = model.get_output_embeddings()
        if lm_head is None:  # pragma: no cover - defensive
            raise ValueError("backbone has no output embedding / lm_head")
        hidden_size = int(model.config.text_config.hidden_size)
        # A checkpoint fixes the width of its own head (older runs used 256 rows), so
        # read it before building the module rather than after.
        state = None
        if self.config.readout_checkpoint:
            state = torch.load(self.config.readout_checkpoint, map_location=self.device)
        max_slots = self.limits.max_slots
        if state is not None and isinstance(state.get("readout"), dict):
            rows = state["readout"].get("proj.weight")
            if rows is not None:
                # The checkpoint fixes the width of its own head, whatever the config
                # default is: a head trained with 256 rows must still load.
                max_slots = int(rows.shape[0])
        self.readout: Readout = build_readout(
            self.config.readout, tokenizer, lm_head, hidden_size, max_slots=max_slots
        ).to(self.device)
        self.readout.eval()
        if state is not None:
            trained_with = state.get("readout_name")
            if trained_with and trained_with != self.readout.name:
                raise ValueError(
                    f"checkpoint {self.config.readout_checkpoint!r} was trained with the "
                    f"{trained_with!r} readout but the engine is using {self.readout.name!r}; "
                    f"pass readout={trained_with!r} as well"
                )
            self.readout.load_state_dict(state["readout"])
            if hasattr(self.readout, "set_slot_table"):
                self.readout.set_slot_table(state.get("slot_table"))
            self.readout_temperature = float(state.get("temperature", 1.0))
        else:
            self.readout_temperature = 1.0
        self._state_cache = _LRUCache(self.config.state_cache_size)

    @property
    def limits(self) -> JevLimits:
        return self.config.limits

    def clear_state_cache(self) -> None:
        """Drop every encoded state."""

        self._state_cache._store.clear()

    def set_state_cache_size(self, size: int) -> int:
        """Resize the state cache; returns the previous size."""

        previous = self._state_cache.maxsize
        self._state_cache.maxsize = size
        self.config.state_cache_size = size
        if size <= 0:
            self._state_cache._store.clear()
        return previous

    # -- construction ----------------------------------------------------------
    @classmethod
    def from_pretrained(cls, model_path: str | None = None, config: QwenJevConfig | None = None, **kwargs):
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        config = config or QwenJevConfig(**kwargs)
        if model_path is not None:
            config.model_path = model_path
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        model = AutoModelForImageTextToText.from_pretrained(
            config.model_path,
            dtype=getattr(torch, config.dtype),
            device_map={"": config.device},
        )
        model.eval()
        return cls(model, tokenizer, config)

    # -- state encoding --------------------------------------------------------
    @torch.no_grad()
    def encode_state(self, state: str):
        """Encode the shared state once; returns ``(cache, token_count, seconds)``."""

        key = state.strip()
        cached = self._state_cache.get(key)
        if cached is not None:
            cache, n_tokens = cached
            return cache, n_tokens, 0.0

        text = render_state(state)
        ids = torch.tensor(
            [self.tokenizer(text, add_special_tokens=False)["input_ids"]], device=self.device
        )
        t0 = time.perf_counter()
        out = self.model.model(input_ids=ids, use_cache=True, return_dict=True)
        cache = out.past_key_values
        elapsed = (time.perf_counter() - t0) * 1000
        self._state_cache.put(key, (cache, ids.shape[1]))
        return cache, ids.shape[1], elapsed

    # -- branch execution ------------------------------------------------------
    def _chunk_branches(self, branches: Sequence[TokenizedBranch]):
        """Pack the independent branches into batches (essay section 7).

        Two limits: the configured batch size, and a token budget so that long
        branches cannot blow up the padding of a large batch.
        """

        chunks: list[list[TokenizedBranch]] = []
        current: list[TokenizedBranch] = []
        longest = 0
        for branch in branches:
            longest = max(longest, len(branch))
            too_many = len(current) + 1 > self.config.branch_batch_size
            too_long = bool(current) and (len(current) + 1) * longest > self.config.max_batch_tokens
            if current and (too_many or too_long):
                chunks.append(current)
                current, longest = [], len(branch)
            current.append(branch)
        if current:
            chunks.append(current)
        return chunks

    def _expand_prefix(self, cache, n: int):
        """Give every branch its own view of the shared prefix ("prefix KV cache
        with separate causal suffixes"), without disturbing the cached state."""

        branch_cache = copy.deepcopy(cache)
        branch_cache.reorder_cache(torch.zeros(n, dtype=torch.long, device=self.device))
        return branch_cache

    @torch.no_grad()
    def _run_branches(
        self,
        prefix_cache,
        state_ids: Sequence[int],
        branches: Sequence[TokenizedBranch],
    ):
        n = len(branches)
        state_len = len(state_ids)
        full = prefix_cache is None
        length = (state_len if full else 0) + max(len(b) for b in branches)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = torch.full((n, length), pad_id, dtype=torch.long, device=self.device)
        for i, branch in enumerate(branches):
            if full:
                input_ids[i, :state_len] = torch.tensor(state_ids, device=self.device)
            start = state_len if full else 0
            input_ids[i, start : start + len(branch)] = torch.tensor(branch.ids, device=self.device)

        if full:  # reference path: every branch carries its own copy of the state
            # ``length`` already contains the state, so the mask must be that wide too.
            mask = torch.zeros((n, length), dtype=torch.long, device=self.device)
            mask[:, :state_len] = 1
            for i, branch in enumerate(branches):
                mask[i, state_len : state_len + len(branch)] = 1
            past = None
        else:
            mask = torch.ones((n, state_len + length), dtype=torch.long, device=self.device)
            for i, branch in enumerate(branches):
                mask[i, state_len + len(branch) :] = 0
            past = self._expand_prefix(prefix_cache, n)

        out = self.model.model(
            input_ids=input_ids,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        hidden = out.last_hidden_state
        offset = state_len if full else 0
        decision = torch.stack([hidden[i, offset + b.decision_index] for i, b in enumerate(branches)])
        option_hidden = [
            hidden[i, [offset + j for j in b.option_end_indices], :] for i, b in enumerate(branches)
        ]
        return decision, option_hidden

    def _score(self, decision, option_hidden, branches: Sequence[TokenizedBranch], questions):
        labels = [list(b.labels) for b in branches]
        return self.readout.score(
            decision,
            labels=labels,
            option_hidden=option_hidden if self.readout.uses_option_states else None,
            option_ids=[question_option_ids(q) for q in questions],
        )

    def _probabilities(self, logits):
        """Softmax over the row's slots, after the fitted readout temperature."""

        return torch.softmax(logits.raw_logits / self.readout_temperature, dim=-1)

    # -- public API ------------------------------------------------------------
    @torch.no_grad()
    def decide_batch(
        self,
        requests: Sequence[tuple[str, Mapping[str, Any] | Iterable]],
        *,
        return_hidden: bool = False,
    ) -> list[JevResponse]:
        """Answer many independent requests in shared forward passes.

        Each request keeps its own state and its own isolated branches; the rows of one
        forward pass simply belong to different requests. This is the evaluation path
        (one pass answers several single-question tasks) and it produces the same
        numbers as calling :meth:`decide` once per request.
        """

        t_start = time.perf_counter()
        prepared = []
        for state, questions in requests:
            question_list = [build_question(qid, spec) for qid, spec in questions.items()]
            self._validate(question_list)
            state_ids = list(
                self.tokenizer(render_state(state), add_special_tokens=False)["input_ids"]
            )
            branches = [tokenize_branch(self.tokenizer, render_branch(q)) for q in question_list]
            self.check_limits(len(state_ids), branches)
            prepared.append((state, state_ids, branches, question_list))
        if not prepared:
            return []

        rows: list[tuple[int, int]] = [
            (request_index, branch_index)
            for request_index, (_state, _ids, branches, _questions) in enumerate(prepared)
            for branch_index in range(len(branches))
        ]
        # Length bucketing keeps the padding inside a chunk small.
        rows.sort(
            key=lambda pair: len(prepared[pair[0]][1]) + len(prepared[pair[0]][2][pair[1]])
        )

        scored: dict[tuple[int, int], Any] = {}
        hidden_dump: dict[str, Any] = {}
        for chunk in self._chunk_rows(rows, prepared):
            length = max(
                len(prepared[ri][1]) + len(prepared[ri][2][bi]) for ri, bi in chunk
            )
            pad_id = self.tokenizer.pad_token_id or 0
            input_ids = torch.full((len(chunk), length), pad_id, dtype=torch.long, device=self.device)
            mask = torch.zeros((len(chunk), length), dtype=torch.long, device=self.device)
            for row_index, (ri, bi) in enumerate(chunk):
                state_ids = prepared[ri][1]
                branch = prepared[ri][2][bi]
                input_ids[row_index, : len(state_ids)] = torch.tensor(state_ids, device=self.device)
                input_ids[row_index, len(state_ids) : len(state_ids) + len(branch)] = torch.tensor(
                    branch.ids, device=self.device
                )
                mask[row_index, : len(state_ids) + len(branch)] = 1
            out = self.model.model(
                input_ids=input_ids, attention_mask=mask, return_dict=True, use_cache=False
            )
            hidden = out.last_hidden_state
            decision = torch.stack(
                [
                    hidden[i, len(prepared[ri][1]) + prepared[ri][2][bi].decision_index]
                    for i, (ri, bi) in enumerate(chunk)
                ]
            )
            option_hidden = [
                hidden[
                    i,
                    [
                        len(prepared[ri][1]) + position
                        for position in prepared[ri][2][bi].option_end_indices
                    ],
                    :,
                ]
                for i, (ri, bi) in enumerate(chunk)
            ]
            labels = [list(prepared[ri][2][bi].labels) for ri, bi in chunk]
            option_ids = [
                question_option_ids(prepared[ri][3][bi]) for ri, bi in chunk
            ]
            logits = self.readout.score(
                decision,
                labels=labels,
                option_hidden=option_hidden if self.readout.uses_option_states else None,
                option_ids=option_ids,
            )
            probabilities = torch.softmax(logits.raw_logits / self.readout_temperature, dim=-1)
            for i, (ri, bi) in enumerate(chunk):
                scored[(ri, bi)] = (
                    probabilities[i],
                    logits.raw_logits[i],
                    decision[i],
                    option_hidden[i],
                )

        branch_ms = (time.perf_counter() - t_start) * 1000
        responses: list[JevResponse] = []
        for ri, (state, state_ids, branches, question_list) in enumerate(prepared):
            decisions: "OrderedDict[str, QuestionDecision]" = OrderedDict()
            for bi, question in enumerate(question_list):
                row_probabilities, raw_logits, _decision, _options = scored[(ri, bi)]
                k = len(branches[bi].labels)
                row = row_probabilities[:k]
                probs = OrderedDict(
                    (option.key, float(p)) for option, p in zip(question.options, row.tolist())
                )
                answer_key = max(probs, key=probs.get)
                decisions[question.id] = QuestionDecision(
                    question_id=question.id,
                    type=question.type.value,
                    probabilities=probs,
                    answer=answer_key,
                    confidence=summarise(question.type.value, probs),
                    score_value=(
                        question.expected_value(probs)
                        if question.type is QuestionType.SCORE
                        else None
                    ),
                    raw_logits=[float(x) for x in raw_logits[:k].tolist()],
                )
                if return_hidden:
                    hidden_dump[question.id] = {
                        "decision": _decision.float().cpu(),
                        "options": _options.float().cpu(),
                        "labels": list(branches[bi].labels),
                        "option_keys": [o.key for o in question.options],
                    }
            question_tokens = sum(len(branch) for branch in branches)
            usage = {
                "state_tokens": len(state_ids),
                "question_tokens": question_tokens,
                "input_tokens": len(state_ids) + question_tokens,
                "output_tokens": billing_output_tokens(
                    [question.id for question in question_list], self.tokenizer
                ),
                "questions": len(question_list),
                "branches": len(question_list),
                "requests": 1,
            }
            timing = {
                "prefill_ms": 0.0,
                "branch_ms": round(branch_ms / max(len(prepared), 1), 2),
                "total_ms": round(branch_ms / max(len(prepared), 1), 2),
                "state_cache_hit": False,
                "shared_state": False,
                "readout": self.readout.name,
                "batched": True,
            }
            responses.append(
                JevResponse(
                    decisions=decisions,
                    usage=usage,
                    timing=timing,
                    hidden=hidden_dump if return_hidden else None,
                )
            )
        return responses

    def _chunk_rows(
        self,
        rows: Sequence[tuple[int, int]],
        prepared: Sequence[tuple[str, list[int], list[TokenizedBranch], list[Question]]],
    ) -> list[list[tuple[int, int]]]:
        """Group rows into forward passes under a token budget."""

        chunks: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        longest = 0
        for pair in rows:
            ri, bi = pair
            length = len(prepared[ri][1]) + len(prepared[ri][2][bi])
            longest = max(longest, length)
            too_many = len(current) + 1 > self.config.branch_batch_size
            too_long = bool(current) and (len(current) + 1) * longest > self.config.max_batch_tokens
            if current and (too_many or too_long):
                chunks.append(current)
                current, longest = [], length
            current.append(pair)
        if current:
            chunks.append(current)
        return chunks

    def decide(
        self,
        state: str,
        questions: Mapping[str, Mapping[str, Any] | Question] | Iterable[Question],
        *,
        share_state: bool | None = None,
        return_hidden: bool = False,
    ) -> JevResponse:
        t_start = time.perf_counter()
        if isinstance(questions, Mapping):
            question_list = [build_question(qid, spec) for qid, spec in questions.items()]
        else:
            question_list = [build_question(q.id, q) for q in questions]

        self._validate(question_list)

        rendered = [render_branch(q) for q in question_list]
        branches = [tokenize_branch(self.tokenizer, r) for r in rendered]
        state_ids = list(self.tokenizer(render_state(state), add_special_tokens=False)["input_ids"])
        state_len = len(state_ids)
        self.check_limits(state_len, branches)

        share_state = self.config.share_state if share_state is None else share_state
        if share_state:
            prefix_cache, _, prefill_ms = self.encode_state(state)
        else:
            prefix_cache = None
            prefill_ms = 0.0
        cache_hit = share_state and prefill_ms == 0.0

        t_branch = time.perf_counter()
        decisions: "OrderedDict[str, QuestionDecision]" = OrderedDict()
        hidden_dump: dict[str, Any] = {}
        index = 0
        requests_made = 0
        for chunk in self._chunk_branches(branches):
            requests_made += 1
            decision, option_hidden = self._run_branches(prefix_cache, state_ids, chunk)
            logits = self._score(
                decision,
                option_hidden,
                chunk,
                question_list[index : index + len(chunk)],
            )
            probabilities = self._probabilities(logits)
            for i, (question, branch) in enumerate(
                zip(question_list[index : index + len(chunk)], chunk)
            ):
                row = probabilities[i, : len(branch.labels)]
                probs = OrderedDict(
                    (option.key, float(p)) for option, p in zip(question.options, row.tolist())
                )
                answer_key = max(probs, key=probs.get)
                confidence = summarise(question.type.value, probs)
                score_value = (
                    question.expected_value(probs) if question.type is QuestionType.SCORE else None
                )
                decisions[question.id] = QuestionDecision(
                    question_id=question.id,
                    type=question.type.value,
                    probabilities=probs,
                    answer=answer_key,
                    confidence=confidence,
                    score_value=score_value,
                    raw_logits=[float(x) for x in logits.raw_logits[i, : len(branch.labels)].tolist()],
                )
                if return_hidden:
                    hidden_dump[question.id] = {
                        "decision": decision[i].float().cpu(),
                        "options": option_hidden[i].float().cpu(),
                        "labels": list(branch.labels),
                        "option_keys": [o.key for o in question.options],
                    }
            index += len(chunk)
        branch_ms = (time.perf_counter() - t_branch) * 1000

        question_tokens = sum(len(b) for b in branches)
        usage = {
            "state_tokens": state_len,
            "question_tokens": question_tokens,
            "input_tokens": state_len + question_tokens,
            "output_tokens": billing_output_tokens([q.id for q in question_list], self.tokenizer),
            "questions": len(question_list),
            "branches": len(question_list),
            "requests": requests_made,
        }
        timing = {
            "prefill_ms": round(prefill_ms, 2),
            "branch_ms": round(branch_ms, 2),
            "total_ms": round((time.perf_counter() - t_start) * 1000, 2),
            "state_cache_hit": bool(cache_hit),
            "shared_state": bool(share_state),
            "readout": self.readout.name,
        }
        return JevResponse(
            decisions=decisions, usage=usage, timing=timing, hidden=hidden_dump or None
        )

    # -- validation ------------------------------------------------------------
    def _validate(self, questions: Sequence[Question]) -> None:
        limits = self.limits
        if not questions:
            raise ValueError("a request needs at least one question")
        for question in questions:
            if len(question.options) > limits.max_options:
                raise ValueError(
                    f"question {question.id!r} has {len(question.options)} options; "
                    f"the API accepts at most {limits.max_options}"
                )
            if len(question.options) > self.readout.max_options:
                raise ValueError(
                    f"question {question.id!r} has {len(question.options)} options but the "
                    f"{self.readout.name!r} readout supports at most {self.readout.max_options}"
                )
            if len(question.options) < 1:
                raise ValueError(f"question {question.id!r} has no options")

    def check_limits(self, state_len: int, branches: Sequence[TokenizedBranch]) -> None:
        branch_lens = [len(b) for b in branches]
        if state_len + max(branch_lens) > self.limits.max_branch_tokens:
            raise ValueError(
                f"branch needs {state_len + max(branch_lens)} tokens > "
                f"{self.limits.max_branch_tokens}"
            )
        total = state_len + sum(branch_lens)
        if total > self.limits.max_request_tokens:
            raise ValueError(
                f"request needs {total} tokens > {self.limits.max_request_tokens}"
            )

    def account(self, state: str, questions: Mapping[str, Any]) -> dict[str, int]:
        """Token accounting before running anything (the request limits of section 2)."""

        question_list = [build_question(qid, spec) for qid, spec in questions.items()]
        state_len = len(self.tokenizer(render_state(state), add_special_tokens=False)["input_ids"])
        branch_lens = [
            len(tokenize_branch(self.tokenizer, render_branch(q)).ids) for q in question_list
        ]
        per_branch = [state_len + x for x in branch_lens]
        return {
            "state_tokens": state_len,
            "question_tokens": sum(branch_lens),
            "request_tokens": state_len + sum(branch_lens),
            "max_branch_tokens": max(per_branch) if per_branch else state_len,
            "output_tokens": billing_output_tokens([q.id for q in question_list], self.tokenizer),
        }

    @staticmethod
    def assert_within_limits(accounting: Mapping[str, int], limits: JevLimits | None = None) -> None:
        """Check a pre-computed accounting dict against the API limits."""

        limits = limits or JevLimits()
        if accounting["max_branch_tokens"] > limits.max_branch_tokens:
            raise ValueError(
                f"branch needs {accounting['max_branch_tokens']} tokens > {limits.max_branch_tokens}"
            )
        if accounting["request_tokens"] > limits.max_request_tokens:
            raise ValueError(
                f"request needs {accounting['request_tokens']} tokens > {limits.max_request_tokens}"
            )
