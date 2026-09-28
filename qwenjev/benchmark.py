"""Run every backend over every normalised task and record accuracy and speed."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .datasets import DecisionItem, normalized_items
from .evaluation import DatasetEvaluation, evaluate_items
from .normalize import catalog, load_normalized

#: The tasks the benchmark knows about, in report order.
TASKS: tuple[str, ...] = (
    "mnli",
    "mnli_ood",
    "snli",
    "fever",
    "jigsaw",
    "goemotions",
    "truthfulqa",
    "mmlu_pro",
    "intentgrasp",
    "clinc150",
    "hwu64",
)

#: Which split to score for each task.
SPLITS: dict[str, str] = {
    "mnli": "test",
    "mnli_ood": "test_ood",
    "snli": "test",
    "fever": "test",
    "jigsaw": "test",
    "goemotions": "test",
    "truthfulqa": "test",
    "mmlu_pro": "test",
    "intentgrasp": "test",
    "clinc150": "test",
    "hwu64": "test",
}

#: Tasks whose normalised files exist even though the source name differs.
SOURCE_KEY: dict[str, str] = {"mnli_ood": "mnli"}


@dataclass
class Variant:
    """One (backend, configuration) pair to score."""

    name: str
    backend: Any
    note: str = ""

    def decide(self, *args, **kwargs):
        return self.backend.decide(*args, **kwargs)


@dataclass
class BenchmarkReport:
    generated_at: str
    test_limit: int | None
    results: dict[str, dict[str, dict]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    variants: list[str] = field(default_factory=list)
    baselines: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "test_limit": self.test_limit,
            "variants": self.variants,
            "results": self.results,
            "baselines": self.baselines,
            "errors": self.errors,
        }


def task_baseline(items: Sequence[DecisionItem]) -> dict[str, float]:
    """Majority-class accuracy and uniform chance, per decision."""

    from collections import Counter

    hits = total = 0
    chance: list[float] = []
    per_question: dict[str, Counter] = {}
    majority_class: dict[str, str] = {}
    for entry in items:
        for qid, target in entry.targets.items():
            per_question.setdefault(qid, Counter())[target] += 1
    # One majority class per question. Without picking a single winner, a sample with
    # every class tied at one item would score 1.0.
    for qid, counter in per_question.items():
        majority_class[qid] = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    for entry in items:
        for qid, target in entry.targets.items():
            counter = per_question[qid]
            hits += target == majority_class[qid]
            total += 1
            chance.append(1.0 / max(len(counter), 1))
    return {
        "n": float(total),
        "majority_accuracy": hits / total if total else float("nan"),
        # A majority predictor recalls the majority class and nothing else, so its
        # balanced accuracy is the same as picking uniformly at random.
        "majority_balanced": sum(chance) / len(chance) if chance else float("nan"),
        "uniform_chance": sum(chance) / len(chance) if chance else float("nan"),
    }


def load_task_items(
    task: str,
    *,
    data_root: Path | str,
    split: str | None = None,
    limit: int | None = None,
) -> list[DecisionItem]:
    key = SOURCE_KEY.get(task, task)
    split = split or SPLITS.get(task, "test")
    path = Path(data_root) / f"{key}_{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no normalised file for {task}: {path}")
    items = load_normalized(path)
    if not limit or limit >= len(items):
        return items
    # Even stride, not a head slice: several sources ship label- or corpus-sorted
    # files, so the first N rows can be a single class.
    stride = len(items) // limit
    return items[::stride][:limit]


def run_benchmark(
    variants: Sequence[Variant],
    *,
    data_root: Path | str = "data/ready",
    tasks: Iterable[str] | None = None,
    test_limit: int | None = 150,
    example_count: int = 0,
    progress: bool = True,
    out_path: Path | str | None = None,
) -> BenchmarkReport:
    report = BenchmarkReport(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        test_limit=test_limit,
        variants=[variant.name for variant in variants],
    )
    selected = list(tasks) if tasks else list(TASKS)
    for task in selected:
        report.results[task] = {}
        for variant in variants:
            started = time.perf_counter()
            try:
                items = load_task_items(task, data_root=data_root, limit=test_limit)
                report.baselines[task] = task_baseline(items)
                evaluation = evaluate_items(
                    variant,
                    items,
                    dataset=task,
                    split=SPLITS.get(task, "test"),
                    example_count=example_count,
                )
                summary = evaluation.summary()
                summary["wall_seconds"] = round(time.perf_counter() - started, 1)
                report.results[task][variant.name] = summary
                if progress:
                    overall = summary["overall"]
                    print(
                        f"{task:<14} {variant.name:<22} acc={overall['accuracy']:.3f} "
                        f"ece={overall['ece']:.3f} n={overall['n']:<5} "
                        f"{summary['usage']['total_ms']/1000:.1f}s",
                        flush=True,
                    )
                if out_path:
                    # Long runs should survive an interruption: keep the partial file fresh.
                    Path(out_path).write_text(
                        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
                    )
            except Exception as exc:  # keep the run going, report what failed
                report.errors[f"{task}/{variant.name}"] = f"{type(exc).__name__}: {exc}"
                if progress:
                    print(f"{task:<14} {variant.name:<22} FAILED {type(exc).__name__}: {exc}", flush=True)
    return report


# ------------------------------------------------------------------ reporting
ROW_ORDER = (
    ("overall", "accuracy"),
    ("overall", "mean_top_probability"),
    ("overall", "ece"),
    ("overall", "brier"),
    ("overall", "nll"),
)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: BenchmarkReport, *, notes: dict[str, str] | None = None) -> str:
    notes = notes or {}
    lines = ["# Dataset results", ""]
    lines.append(f"- generated: {report.generated_at}")
    lines.append(f"- items per task: {report.test_limit}")
    lines.append(f"- variants: {', '.join(f'`{v}`' for v in report.variants)}")
    lines.append("")

    lines += ["## Accuracy", "", _table(report, "accuracy"), ""]
    lines += [
        "## Balanced accuracy (mean per-class recall; use this when one class dominates)",
        "",
        _table(report, "balanced_accuracy"),
        "",
    ]
    lines += ["## Calibration (ECE over the answer the model chose)", "", _table(report, "ece"), ""]
    lines += ["## Speed", "", _speed_table(report), ""]

    if notes:
        lines += ["## Task notes", ""]
        for task, note in notes.items():
            lines.append(f"- **{task}**: {note}")
        lines.append("")
    if report.errors:
        lines += ["## Failures", ""]
        for key, message in report.errors.items():
            lines.append(f"- `{key}`: {message}")
        lines.append("")
    return "\n".join(lines)


def _table(report: BenchmarkReport, metric: str) -> str:
    header = ["task"] + [f"{v}" for v in report.variants]
    extra = []
    if metric == "accuracy":
        header += ["majority", "chance"]
        extra = ["majority_accuracy", "uniform_chance"]
    elif metric == "balanced_accuracy":
        header += ["majority"]
        extra = ["majority_balanced"]
    rows = []
    for task, per_variant in report.results.items():
        row = [task]
        for variant in report.variants:
            summary = per_variant.get(variant)
            row.append(_fmt(summary["overall"][metric] if summary else None))
        for key in extra:
            row.append(_fmt((report.baselines.get(task) or {}).get(key)))
        rows.append(row)
    return _render(header, rows)


def _speed_table(report: BenchmarkReport) -> str:
    header = ["task", "variant", "requests", "ms/request", "tok/req", "n"]
    rows = []
    for task, per_variant in report.results.items():
        for variant in report.variants:
            summary = per_variant.get(variant)
            if not summary:
                continue
            usage = summary["usage"]
            requests = usage.get("requests") or usage.get("questions") or summary["n_items"]
            total_ms = usage.get("total_ms", 0.0)
            rows.append(
                [
                    task,
                    variant,
                    requests,
                    _fmt(total_ms / requests if requests else None, 1),
                    _fmt(usage.get("input_tokens", 0) / requests if requests else None, 1),
                    summary["overall"]["n"],
                ]
            )
    return _render(header, rows)


def _render(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)
