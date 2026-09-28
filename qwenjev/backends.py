"""Two interchangeable decision backends behind one interface.

* :class:`QwenBackend` wraps :class:`qwenjev.engine.QwenJevLite` (Qwen3.5-4B plus our
  shared-state prefill, isolated branches and readout).
* :class:`LayaBackend` wraps `Laya <https://github.com/NandhaKishorM/laya>`_, a small
  non-autoregressive decision model, so the same datasets can be scored against a
  purpose-built jev-like model.

Both return a :class:`qwenjev.engine.JevResponse`, so evaluation, accounting and
reporting are written once. Each backend renders the *same* normalised question in the
shape it was measured on: our engine keeps the question in the branch, Laya's ``noul``
template wants the claim at the head of the state (its README measures that layout as
roughly twice as accurate as the alternative).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Mapping

from .config import JevLimits
from .confidence import summarise
from .engine import JevResponse, QuestionDecision
from .schema import QuestionType

TRUE_KEYS = {"true", "yes", "y", "1", "true_label", "positive", "entailment"}
FALSE_KEYS = {"false", "no", "n", "0", "false_label", "negative"}


def _bool_polarity(keys: "list[str]") -> tuple[str, str]:
    """Which key means "the claim holds" and which means "it does not".

    Our schema writes boolean criteria as ``{"yes": ..., "no": ...}``; Laya's ``noul``
    template is phrased as a claim plus ``true``/``false``. Getting this backwards
    inverts every probability, so it is resolved by name rather than by position.
    """

    true_key = next((k for k in keys if str(k).lower() in TRUE_KEYS), None)
    false_key = next((k for k in keys if str(k).lower() in FALSE_KEYS), None)
    if true_key is None:
        true_key = keys[-1]
    if false_key is None:
        false_key = next(k for k in keys if k != true_key)
    return true_key, false_key


class LayaBackend:
    """Adapter around ``laya.load(...).predict(...)``."""

    name = "laya"
    kind = "laya"

    #: Laya writes instructions that refer to ``message``; this is the template its
    #: README measures as best for choice questions.
    CHOICE_INSTRUCTION = "Which of the options best applies to `message`?"
    #: "The claim at the start of `message` is true of the text that follows it."
    NOUL_INSTRUCTION = (
        "The claim at the start of `message` is true of the text that follows it."
    )

    def __init__(self, model_path: str = r"C:\laya", device: str | None = None):
        import laya

        self.model_path = model_path
        self.agent = laya.load(model_path, device=device)
        self.limits = JevLimits()

    # -- rendering -------------------------------------------------------------
    @staticmethod
    def _criteria_keys(criteria: Mapping[str, Any] | list | None) -> list[str]:
        if criteria is None:
            return ["no", "yes"]
        if isinstance(criteria, Mapping):
            return list(criteria)
        return [str(item) for item in criteria]

    def _laya_question(self, question: dict) -> dict:
        qtype = str(question.get("type", "choice")).lower()
        if qtype in {"bool", "noul", "boolean", "yesno"}:
            criteria = question.get("criteria") or {}
            keys = list(criteria) if criteria else ["yes", "no"]
            true_key, false_key = _bool_polarity(keys)
            definition: dict[str, Any] = {
                "type": "noul",
                "instructions": self.NOUL_INSTRUCTION,
                "criteria": {
                    "false": (criteria or {}).get(false_key) or "No",
                    "true": (criteria or {}).get(true_key) or "Yes",
                },
            }
            return definition
        if qtype == "score":
            criteria = question.get("criteria") or []
            levels = list(criteria.values()) if isinstance(criteria, Mapping) else list(criteria)
            return {
                "type": "score",
                "instructions": question.get("instructions", ""),
                "criteria": [str(level) for level in levels],
            }
        return {
            "type": "choice",
            "instructions": self.CHOICE_INSTRUCTION
            + (("\n" + question["instructions"]) if question.get("instructions") else ""),
            "criteria": dict(question.get("criteria") or {}),
        }

    def _laya_state(self, state: str, question: dict) -> dict:
        qtype = str(question.get("type", "choice")).lower()
        context = str(question.get("context") or "")
        if qtype in {"bool", "noul", "boolean", "yesno"}:
            claim = question.get("claim") or question.get("instructions") or "The claim is true."
            # The noul template reads claim-then-evidence, so a branch-level passage
            # goes right after the claim instead of the shared state.
            return {"message": f"{claim}\n{context or state}"}
        if context:
            return {"message": f"{context}\n\n{state}".strip()}
        return {"message": state}

    # -- interface -------------------------------------------------------------
    def decide(
        self,
        state: str,
        questions: Mapping[str, Any],
        *,
        share_state: bool | None = None,
        return_hidden: bool = False,
    ) -> JevResponse:
        from .schema import build_question

        prepared: "OrderedDict[str, Any]" = OrderedDict()
        laya_questions: dict[str, dict] = {}
        for qid, spec in questions.items():
            question = build_question(qid, spec)
            prepared[qid] = question
            laya_questions[qid] = self._laya_question(spec if isinstance(spec, dict) else {})
            if not isinstance(spec, dict):
                raise TypeError("Laya backend needs the question as a dict")

        # One state per question: Laya is a single-state model, so a request is
        # naturally one question. Group by rendered state to keep the call count honest.
        decisions: "OrderedDict[str, QuestionDecision]" = OrderedDict()
        input_tokens = 0
        started = time.perf_counter()
        for qid, question in prepared.items():
            spec = questions[qid]
            laya_state = self._laya_state(state, spec)
            result = self.agent.predict(laya_state, {qid: laya_questions[qid]})
            answer = result["answers"][qid]
            input_tokens += int(result.get("usage", {}).get("input_tokens", 0))
            decisions[qid] = self._to_decision(qid, question, answer)
        elapsed = (time.perf_counter() - started) * 1000

        usage = {
            "input_tokens": input_tokens,
            "state_tokens": 0,
            "question_tokens": 0,
            "output_tokens": 0,
            "questions": len(decisions),
            "branches": len(decisions),
            "requests": len(decisions),
        }
        timing = {
            "prefill_ms": round(elapsed, 2),
            "branch_ms": 0.0,
            "total_ms": round(elapsed, 2),
            "state_cache_hit": False,
            "shared_state": False,
            "readout": "laya",
        }
        return JevResponse(decisions=decisions, usage=usage, timing=timing)

    def _to_decision(self, qid: str, question, answer: dict) -> QuestionDecision:
        keys = [option.key for option in question.options]
        kind = answer.get("type")
        if kind == "noul":
            p_true = float(answer.get("noul", 0.0))
            true_key, false_key = _bool_polarity(keys)
            probabilities = OrderedDict(
                (key, p_true if key == true_key else 1.0 - p_true) for key in keys
            )
            confidence = float(answer.get("confidence", 0.0))
        elif kind == "score":
            raw = answer.get("probabilities", {})
            probabilities = OrderedDict(
                (key, float(raw.get(str(i), 0.0))) for i, key in enumerate(keys)
            )
            confidence = float(answer.get("confidence", 0.0))
        else:
            raw = answer.get("probabilities", {})
            probabilities = OrderedDict((key, float(raw.get(key, 0.0))) for key in keys)
            confidence = float(answer.get("confidence", 0.0))
        total = sum(probabilities.values()) or 1.0
        probabilities = OrderedDict((k, v / total) for k, v in probabilities.items())
        answer_key = max(probabilities, key=probabilities.get)
        if question.type is not QuestionType.SCORE:
            confidence = summarise(question.type.value, probabilities)
        return QuestionDecision(
            question_id=qid,
            type=question.type.value,
            probabilities=probabilities,
            answer=answer_key,
            confidence=confidence,
            score_value=(question.expected_value(probabilities)
                         if question.type is QuestionType.SCORE else None),
            raw_logits=[float("nan")] * len(probabilities),
        )

    def account(self, state: str, questions: Mapping[str, Any]) -> dict[str, int]:
        tokens = 0
        for spec in questions.values():
            payload = self._laya_state(state, spec)
            tokens += len(self.agent.tok.encode(payload["message"])) if hasattr(self.agent, "tok") else 0
        return {
            "state_tokens": tokens,
            "question_tokens": 0,
            "request_tokens": tokens,
            "max_branch_tokens": tokens,
            "output_tokens": 0,
        }

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # Agent is a context manager that releases the weights.
        with self.agent:
            pass
        return False


class QwenBackend:
    """Thin wrapper that presents :class:`QwenJevLite` as a backend."""

    name = "qwenjev"
    kind = "qwen"

    def __init__(self, engine):
        self.engine = engine
        self.limits = engine.limits

    def decide(self, state, questions, *, share_state=None, return_hidden=False) -> JevResponse:
        return self.engine.decide(
            state, questions, share_state=share_state, return_hidden=return_hidden
        )

    def decide_batch(self, requests, *, return_hidden=False):
        """Batched evaluation path: several requests per forward pass."""

        return self.engine.decide_batch(requests, return_hidden=return_hidden)

    def account(self, state, questions):
        return self.engine.account(state, questions)

    def __getattr__(self, item):
        return getattr(self.engine, item)


def build_backend(kind: str, **kwargs):
    if kind in {"qwen", "qwenjev", "qwenjev-lite"}:
        from .config import QwenJevConfig
        from .engine import QwenJevLite

        config = QwenJevConfig(**kwargs) if kwargs else None
        return QwenBackend(QwenJevLite.from_pretrained(config=config))
    if kind == "laya":
        return LayaBackend(**kwargs)
    raise ValueError(f"unknown backend {kind!r}; use 'qwenjev' or 'laya'")
