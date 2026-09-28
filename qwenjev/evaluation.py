"""Evaluate a decision model over a whole dataset of Jev-shaped requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .datasets import DecisionItem
from .rlcd import Evaluation


@dataclass
class DatasetEvaluation:
    """Per-decision predictions plus the aggregate reliability numbers."""

    dataset: str
    split: str
    overall: Evaluation
    by_question: dict[str, Evaluation]
    usage: dict[str, Any] = field(default_factory=dict)
    examples: list[dict] = field(default_factory=list)
    n_items: int = 0

    def summary(self, n_bins: int = 10) -> dict:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "n_items": self.n_items,
            "n_decisions": int(self.overall.correct.size),
            "overall": {k: v for k, v in self.overall.reliability(n_bins).items() if k != "bins"},
            "by_question": {
                qid: {k: v for k, v in evaluation.reliability(n_bins).items() if k != "bins"}
                for qid, evaluation in self.by_question.items()
            },
            "usage": self.usage,
            "examples": self.examples,
        }


def _evaluation_from_rows(rows: Sequence[dict]) -> Evaluation:
    def array(key: str, dtype):
        return np.asarray([row[key] for row in rows], dtype=dtype)

    return Evaluation(
        top_probabilities=array("top_probability", float),
        true_probabilities=array("true_probability", float),
        correct=array("correct", bool),
        targets=[row["target"] for row in rows],
        answers=[row["answer"] for row in rows],
        logits=[row["logits"] for row in rows],
        probability_rows=[row["probabilities"] for row in rows],
        target_indices=array("target_index", int),
    )


def evaluate_items(
    engine,
    items: Sequence[DecisionItem],
    *,
    dataset: str = "dataset",
    split: str = "",
    share_state: bool | None = None,
    max_items: int | None = None,
    example_count: int = 3,
    progress: bool = False,
    batch_size: int | None = None,
) -> DatasetEvaluation:
    """One request per item: every question about a state is answered together."""

    rows: list[dict] = []
    by_question: dict[str, list[dict]] = {}
    examples: list[dict] = []
    usage_totals: dict[str, float] = {}
    selected = list(items)[:max_items] if max_items else list(items)

    iterator = selected
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(selected, desc=f"{dataset}:{split or 'test'}", unit="item")

    def handle(item, response) -> None:
        for name, value in response.usage.items():
            usage_totals[name] = usage_totals.get(name, 0) + value
        for name in ("prefill_ms", "branch_ms", "total_ms"):
            usage_totals[name] = usage_totals.get(name, 0.0) + float(response.timing[name])
        usage_totals["state_cache_hits"] = usage_totals.get("state_cache_hits", 0) + int(
            bool(response.timing.get("state_cache_hit"))
        )

        for question_id, decision in response.decisions.items():
            keys = list(decision.probabilities)
            target = item.targets[question_id]
            if target not in decision.probabilities:
                raise ValueError(
                    f"{dataset}: target {target!r} is not among the criteria of "
                    f"question {question_id!r} ({keys})"
                )
            probabilities = [float(decision.probabilities[key]) for key in keys]
            target_index = keys.index(target)
            predicted = int(np.argmax(probabilities))
            row = {
                "item_id": item.item_id,
                "question_id": question_id,
                "target": target,
                "answer": keys[predicted],
                "correct": predicted == target_index,
                "probabilities": probabilities,
                "keys": keys,
                "target_index": target_index,
                "top_probability": probabilities[predicted],
                "true_probability": probabilities[target_index],
                "logits": list(decision.raw_logits or probabilities),
            }
            rows.append(row)
            by_question.setdefault(question_id, []).append(row)
            if len(examples) < example_count * max(len(item.questions), 1):
                examples.append(
                    {
                        "item_id": item.item_id,
                        "question_id": question_id,
                        "state": item.state[:400],
                        "target": target,
                        "answer": keys[predicted],
                        "correct": row["correct"],
                        "probability": round(row["true_probability"], 4),
                        "distribution": {k: round(p, 4) for k, p in zip(keys, probabilities)},
                    }
                )

    def tick() -> None:
        if progress and rows:
            correct = sum(1 for row in rows if row["correct"])
            iterator.set_postfix(acc=f"{correct/len(rows):.3f}", refresh=False)

    batched = (
        bool(batch_size)
        and batch_size > 1
        and share_state is None
        and hasattr(engine, "decide_batch")
    )
    if batched:
        # Several requests share one forward pass; the numbers are identical.
        for start in range(0, len(selected), batch_size):
            group = selected[start : start + batch_size]
            responses = engine.decide_batch([(item.state, item.questions) for item in group])
            for item, response in zip(group, responses):
                handle(item, response)
            if progress:
                iterator.update(len(group))
            tick()
    else:
        for item in iterator:
            handle(item, engine.decide(item.state, item.questions, share_state=share_state))
            tick()

    if not rows:
        raise ValueError(f"{dataset}: no decidable items")

    usage = {key: value for key, value in usage_totals.items() if key not in {"prefill_ms", "branch_ms", "total_ms"}}
    usage["prefill_ms"] = round(usage_totals.get("prefill_ms", 0.0), 1)
    usage["branch_ms"] = round(usage_totals.get("branch_ms", 0.0), 1)
    usage["total_ms"] = round(usage_totals.get("total_ms", 0.0), 1)
    usage["decisions_per_request"] = round(len(rows) / max(len(selected), 1), 2)

    return DatasetEvaluation(
        dataset=dataset,
        split=split,
        overall=_evaluation_from_rows(rows),
        by_question={qid: _evaluation_from_rows(part) for qid, part in by_question.items()},
        usage=usage,
        examples=examples[:example_count],
        n_items=len(selected),
    )
