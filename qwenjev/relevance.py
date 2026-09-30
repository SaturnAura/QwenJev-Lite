"""Build relevance-judgement datasets from query/document collections.

Given a collection of documents and judged query-document pairs, this module turns
every query into a small *relevance judgement* suite:

``<collection>_rel_bool``
    state = the query; every candidate passage is one yes/no branch
    ("does this passage satisfy the query?"). Positive passages come from the judged
    answers, negative passages from the rest of the corpus.

``<collection>_rel_choice``
    state = the query; one judged passage plus several distractors are offered as a
    single choice question ("which passage best satisfies the query?").

``<collection>_rel_score``
    state = the query; every candidate passage is one ordered-level question
    ("how well does this passage satisfy the query?") over
    not relevant / relevant / highly relevant.

    python -m qwenjev.cli relevance --src <collection root> \\
        --collections scifact nfcorpus trec-covid --queries 60
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .datasets import DecisionItem, _clean
from .normalize import bool_question, choice_question, item, score_question, write_items

DEFAULT_DOC_CHARS = 500
DEFAULT_OPTION_CHARS = 160
LETTERS = "ABCDEFGHIJ"

#: Ordered relevance levels used by the ``score`` judgement.
REL_LEVELS = {
    "not_relevant": "Not relevant",
    "relevant": "Relevant",
    "highly_relevant": "Highly relevant",
}


@dataclass
class JudgedCollection:
    """Documents, queries and the judged query-document answers."""

    name: str
    directory: Path
    corpus: dict[str, str] = field(default_factory=dict)
    query_text: dict[str, str] = field(default_factory=dict)
    answers: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)

    def splits(self) -> list[str]:
        return sorted(self.answers)

    def positives(self, split: str, qid: str) -> list[str]:
        return [
            doc_id
            for doc_id, score in self.answers.get(split, {}).get(qid, {}).items()
            if score > 0
        ]

    def judged(self, split: str, qid: str) -> set[str]:
        return set(self.answers.get(split, {}).get(qid, {}))


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_collection(directory: Path | str) -> JudgedCollection:
    """Read ``corpus.jsonl``, ``queries.jsonl`` and every ``qrels/*.tsv``."""

    directory = Path(directory)
    corpus_path = directory / "corpus.jsonl"
    queries_path = directory / "queries.jsonl"
    if not corpus_path.is_file() or not queries_path.is_file():
        raise FileNotFoundError(f"not a query/document collection: {directory}")

    collection = JudgedCollection(name=directory.name, directory=directory)
    for record in _iter_jsonl(corpus_path):
        title = _clean(record.get("title") or "")
        text = _clean(record.get("text") or "")
        collection.corpus[str(record["_id"])] = (title + ". " + text).strip(". ") if title else text
    for record in _iter_jsonl(queries_path):
        collection.query_text[str(record["_id"])] = _clean(record.get("text") or "")

    qrels_dir = directory / "qrels"
    if qrels_dir.is_dir():
        for path in sorted(qrels_dir.glob("*.tsv")):
            split = path.stem
            table: dict[str, dict[str, int]] = {}
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    qid = str(row.get("query-id") or row.get("query_id") or "").strip()
                    doc_id = str(row.get("corpus-id") or row.get("doc_id") or "").strip()
                    if not qid or not doc_id:
                        continue
                    try:
                        score = int(float(row.get("score", 1) or 1))
                    except (TypeError, ValueError):
                        score = 1
                    table.setdefault(qid, {})[doc_id] = score
            if table:
                collection.answers[split] = table
    if not collection.answers:
        raise FileNotFoundError(f"no qrels/*.tsv in {directory}")
    return collection


def _relevance_level(score: int) -> str:
    if score >= 2:
        return "highly_relevant"
    if score == 1:
        return "relevant"
    return "not_relevant"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _sample_negatives(
    collection: JudgedCollection,
    split: str,
    qid: str,
    count: int,
    rng: random.Random,
) -> list[str]:
    """Uniformly sample passages that are *not* in the judged answers for this query."""

    judged = collection.judged(split, qid)
    pool = [doc_id for doc_id in collection.corpus if doc_id not in judged]
    if not pool:
        return []
    picks: list[str] = []
    seen: set[str] = set()
    for _ in range(50 * count):
        if len(picks) >= count:
            break
        doc_id = rng.choice(pool)
        if doc_id not in seen:
            picks.append(doc_id)
            seen.add(doc_id)
    return picks


def build_items(
    collection: JudgedCollection,
    *,
    split: str = "test",
    mode: str = "bool",
    query_ids: Sequence[str] | None = None,
    positives: int = 2,
    negatives: int = 4,
    doc_chars: int = DEFAULT_DOC_CHARS,
    option_chars: int = DEFAULT_OPTION_CHARS,
    seed: int = 0,
) -> list[DecisionItem]:
    if mode not in {"bool", "choice", "score"}:
        raise ValueError("mode must be 'bool', 'choice' or 'score'")
    rng = random.Random(seed)
    table = collection.answers.get(split) or {}
    if query_ids is None:
        query_ids = [qid for qid in table if collection.query_text.get(qid)]
        query_ids = list(query_ids)
        rng.shuffle(query_ids)

    items: list[DecisionItem] = []
    for qid in query_ids:
        query = collection.query_text.get(qid)
        if not query:
            continue
        judged = collection.judged(split, qid)
        positive_pool = [doc_id for doc_id in collection.positives(split, qid) if doc_id in collection.corpus]
        if not positive_pool:
            continue
        positive_docs = positive_pool[:positives]
        if mode == "choice":
            n_pos, n_neg = 1, 3
        else:
            n_pos, n_neg = len(positive_docs), negatives
        negative_docs = _sample_negatives(collection, split, qid, n_neg, rng)
        if not negative_docs:
            continue
        candidates = positive_docs[:n_pos] + negative_docs[:n_neg]
        rng.shuffle(candidates)
        if len(candidates) < 2:
            continue

        task = f"{collection.name}_rel_{mode}"
        if mode == "bool":
            questions, targets = {}, {}
            for i, doc_id in enumerate(candidates):
                key = f"d{i}"
                questions[key] = bool_question(
                    "Does this passage satisfy the query?",
                    "This passage satisfies the query.",
                    context=_truncate(collection.corpus[doc_id], doc_chars),
                )
                targets[key] = "yes" if doc_id in judged else "no"
            state, meta = query, {"query_id": qid, "doc_ids": candidates, "n_candidates": len(candidates)}
        elif mode == "score":
            questions, targets = {}, {}
            for i, doc_id in enumerate(candidates):
                key = f"d{i}"
                score = collection.answers[split][qid].get(doc_id, 0)
                questions[key] = score_question(
                    "How well does this passage satisfy the query?",
                    dict(REL_LEVELS),
                    context=_truncate(collection.corpus[doc_id], doc_chars),
                )
                targets[key] = _relevance_level(score)
            state, meta = query, {"query_id": qid, "doc_ids": candidates, "n_candidates": len(candidates)}
        else:
            letters = list(LETTERS[: len(candidates)])
            label_of = dict(zip(letters, candidates))
            letter_of = {doc_id: letter for letter, doc_id in label_of.items()}
            positive = positive_docs[0]
            criteria = {
                letter: _truncate(collection.corpus[doc_id], option_chars)
                for letter, doc_id in label_of.items()
            }
            questions = {"passage": choice_question("Which passage best satisfies the query?", criteria)}
            targets = {"passage": letter_of[positive]}
            state, meta = query, {"query_id": qid, "doc_ids": candidates, "n_candidates": len(candidates)}

        items.append(
            item(
                dataset=task,
                split=split,
                index=qid,
                state=state,
                questions=questions,
                targets=targets,
                meta=meta,
            )
        )
    return items


COLLECTION_NAMES: tuple[str, ...] = (
    "arguana",
    "scifact",
    "nfcorpus",
    "vihealthqa",
    "scidocs",
    "trec-covid",
    "trec-covid-v2",
    "webis-touche2020",
    "climate-fever",
    "hotpotqa",
    "nq",
    "msmarco",
    "dbpedia-entity",
)


def normalize_relevance(
    src_root: Path | str,
    out_dir: Path | str,
    *,
    collections: Sequence[str] | None = None,
    modes: Sequence[str] = ("bool", "choice", "score"),
    queries: int = 60,
    train_queries: int = 150,
    positives: int = 2,
    negatives: int = 4,
    doc_chars: int = DEFAULT_DOC_CHARS,
    option_chars: int = DEFAULT_OPTION_CHARS,
    seed: int = 0,
    progress: bool = True,
) -> dict:
    src_root = Path(src_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = list(collections) if collections else [
        name for name in COLLECTION_NAMES if (src_root / name / "corpus.jsonl").is_file()
    ]
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": str(src_root),
        "out_dir": str(out_dir),
        "queries_per_task": queries,
        "tasks": {},
    }
    for name in wanted:
        directory = src_root / name
        if not (directory / "corpus.jsonl").is_file():
            manifest["tasks"][name] = {"status": "missing", "path": str(directory)}
            continue
        collection = load_collection(directory)
        eval_split = "test" if "test" in collection.answers else collection.splits()[0]
        train_split = "train" if "train" in collection.answers else eval_split
        rng = random.Random(seed)
        pool = [qid for qid in collection.answers.get(eval_split, {}) if collection.query_text.get(qid)]
        rng.shuffle(pool)
        held_out = pool[:queries]
        train_ids = (
            [qid for qid in collection.answers.get(train_split, {}) if collection.query_text.get(qid)][:train_queries]
            if train_split != eval_split
            else pool[queries : queries + train_queries]
        )
        entry: dict[str, Any] = {
            "title": f"relevance judgement: {name}",
            "source": str(directory),
            "eval_split": eval_split,
            "train_split": train_split,
            "splits": {},
        }
        for mode in modes:
            for split_name, ids, source_split in (
                ("test", held_out, eval_split),
                ("train", train_ids, train_split),
            ):
                if not ids:
                    continue
                items = build_items(
                    collection,
                    split=source_split,
                    mode=mode,
                    query_ids=ids,
                    positives=positives,
                    negatives=negatives,
                    doc_chars=doc_chars,
                    option_chars=option_chars,
                    seed=seed,
                )
                if not items:
                    continue
                task = f"{name}_rel_{mode}"
                path = out_dir / f"{task}_{split_name}.jsonl"
                info = write_items(path, items, dataset=task, split=split_name)
                entry["splits"][f"{mode}_{split_name}"] = info
        entry["status"] = "ok" if entry["splits"] else "empty"
        manifest["tasks"][name] = entry
        if progress:
            detail = ", ".join(
                f"{key}:{value['items']}i/{value['decisions']}d"
                for key, value in entry["splits"].items()
            )
            print(f"{name:<18} {detail or entry['status']}", flush=True)
    (out_dir / "manifest_relevance.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
