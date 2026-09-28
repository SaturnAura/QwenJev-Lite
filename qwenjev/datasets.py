"""Adapters that turn public NLU / fact-checking datasets into Jev-shaped requests.

Every dataset is mapped onto the architecture's shape: **one shared state, a set of
isolated typed questions, one observed outcome per question**.

====================  ==========================  ================================
dataset               shared state                decision
====================  ==========================  ================================
FEVER                 evidence for a claim        supports / refutes / not enough
MMLU, MMLU-Pro        subject + question stem     which option is correct
CLINC150              utterance                   which intent (150 + out of scope)
HWU64                 utterance                   which intent (64, 3 domains)
Jigsaw                comment                     six independent toxicity bools
====================  ==========================  ================================

Loading is deliberately format-agnostic: a directory holding an HF ``save_to_disk``
dump, JSONL, JSON, CSV, TSV or Parquet all work, and column names are matched against
several spellings so different mirrors can be dropped in unchanged. When nothing is
found the loader raises :class:`DatasetUnavailable` with the exact commands to fetch
the data; the same text is printed by ``qwenjev datasets``.
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .schema import SLOT_LETTERS, DecisionSample

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
DEFAULT_RAW_DIR = DEFAULT_DATA_ROOT / "raw"
DEFAULT_NORMALIZED_DIR = DEFAULT_DATA_ROOT / "ready"


class DatasetUnavailable(RuntimeError):
    """Raised when the local files for a dataset cannot be found."""

    def __init__(self, dataset: str, roots: Sequence[Path], hint: str):
        self.dataset = dataset
        self.roots = [str(r) for r in roots]
        super().__init__(
            f"dataset {dataset!r} is not available locally.\n"
            f"looked in: {', '.join(str(r) for r in roots)}\n\n{hint}"
        )


@dataclass
class DecisionItem:
    """One request: a shared state plus the questions asked about it."""

    item_id: str
    state: str
    questions: dict[str, dict]
    targets: dict[str, str]
    meta: dict[str, Any] = field(default_factory=dict)

    def samples(self, question_ids: Iterable[str] | None = None) -> list[DecisionSample]:
        ids = list(question_ids) if question_ids is not None else list(self.questions)
        return [
            DecisionSample(
                state=self.state,
                question_id=qid,
                target=self.targets[qid],
                spec=self.questions[qid],
            )
            for qid in ids
        ]


def items_to_samples(items: Sequence[DecisionItem], question_ids: Iterable[str] | None = None):
    """Flatten items into supervised samples for :mod:`qwenjev.rlcd`."""

    samples: list[DecisionSample] = []
    for item in items:
        samples.extend(item.samples(question_ids))
    return samples


# --------------------------------------------------------------------------- io
_SUFFIXES = (".jsonl", ".json", ".csv", ".tsv", ".parquet")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for value in payload.values():  # e.g. {"data": [...]}
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
        return [payload]
    raise ValueError(f"unsupported json payload in {path}")


def _read_delimited(path: Path, delimiter: str) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]


def _read_parquet(path: Path) -> list[dict]:
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("reading parquet needs pandas") from exc
    return pd.read_parquet(path).to_dict(orient="records")


def _read_file(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    if path.suffix == ".json":
        return _read_json(path)
    if path.suffix == ".csv":
        return _read_delimited(path, ",")
    if path.suffix == ".tsv":
        return _read_delimited(path, "\t")
    if path.suffix == ".parquet":
        return _read_parquet(path)
    raise ValueError(f"unsupported file type {path}")


def _find_files(root: Path, split: str, patterns: Sequence[str] | None = None) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        return []
    globs = list(patterns or ())
    for suffix in _SUFFIXES:
        globs.extend([f"{split}{suffix}", f"{split}-*{suffix}", f"*{split}*{suffix}"])
    seen: dict[str, Path] = {}
    for pattern in globs:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                seen.setdefault(str(path).lower(), path)
    return sorted(seen.values(), key=lambda p: (0 if p.name.lower().startswith(split) else 1, p.name))


def _load_hf_saved_dir(root: Path, split: str) -> list[dict] | None:
    """Read a ``datasets.save_to_disk`` directory when one is present."""

    if not root.is_dir():
        return None
    if not (root / "dataset_info.json").is_file() and not (root / "state.json").is_file():
        return None
    try:
        from datasets import load_from_disk
    except Exception:  # pragma: no cover - optional dependency
        return None
    loaded = load_from_disk(str(root))
    if hasattr(loaded, "keys"):
        if split in loaded:
            dataset = loaded[split]
        elif len(loaded) == 1:
            dataset = next(iter(loaded.values()))
        else:
            raise DatasetUnavailable(root.name, [root], f"split {split!r} not in {list(loaded)}")
    else:
        dataset = loaded
    return [dict(row) for row in dataset]


def _load_from_hf_cache(hf_ids: Sequence[str], split: str, config: str | None) -> list[dict] | None:
    """Last resort: the local HuggingFace cache. Never touches the network."""

    try:
        from datasets import load_dataset
    except Exception:  # pragma: no cover - optional dependency
        return None
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    for dataset_id in hf_ids:
        try:
            loaded = load_dataset(dataset_id, config, split=split)
        except Exception:
            continue
        return [dict(row) for row in loaded]
    return None


def read_records(
    root: Path,
    split: str,
    *,
    patterns: Sequence[str] | None = None,
    hf_ids: Sequence[str] = (),
    hf_config: str | None = None,
    hint: str = "",
) -> list[dict]:
    """Read ``split``, trying saved HF dumps, plain files, then the HF cache."""

    rows = _load_hf_saved_dir(root, split)
    if rows is not None:
        return rows
    files = _find_files(root, split, patterns)
    if files:
        rows = []
        for path in files:
            rows.extend(_read_file(path))
        return rows
    if hf_ids:
        rows = _load_from_hf_cache(hf_ids, split, hf_config)
        if rows is not None:
            return rows
    raise DatasetUnavailable(root.name, [root], hint)


def _field(row: dict, names: Sequence[str], default: Any = None) -> Any:
    lowered = {str(k).lower(): k for k in row}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None and row[key] is not None:
            return row[key]
    return default


def _clean(text: Any) -> str:
    return " ".join(str(text).split())


def render_hint(template: str, root: Path | str) -> str:
    """Fill ``{root}`` without touching the braces inside the hint text."""

    base = str(root).rstrip("\\/")
    return template.replace("{root}/", base + os.sep).replace("{root}", base)


CHOICE_TEMPLATE = "Which option is correct?"


def letter_criteria(labels: Sequence[str], descriptions: Sequence[str]) -> dict[str, str]:
    return {label: _clean(desc) for label, desc in zip(labels, descriptions)}


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if text in {"yes", "true", "1"}:
        return 1
    if text in {"no", "false", "0"}:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


# ------------------------------------------------------------------------ FEVER
FEVER_HINT = """FEVER - download the claim files into {root}

  paper_dev.jsonl, paper_test.jsonl, train.jsonl
    curl -L -o "{root}/train.jsonl"      https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl
    curl -L -o "{root}/paper_dev.jsonl"  https://s3-eu-west-1.amazonaws.com/fever.public/paper_dev.jsonl
    curl -L -o "{root}/paper_test.jsonl" https://s3-eu-west-1.amazonaws.com/fever.public/paper_test.jsonl

  or straight from the HuggingFace cache:
    python -c "from datasets import load_dataset; load_dataset('fever','v1.0').save_to_disk(r'{root}')"

  The claim files carry the label and the claim but no wiki sentences. Add an
  evidence file to get the decision the architecture is actually built for:
    evidence.jsonl -> {"id": <claim id>, "evidence": "<sentences>"}
  Build it from the wiki dump with the bundled script:
    python scripts/fever_evidence.py --claims "{root}/paper_dev.jsonl" --wiki-pages "{root}/wiki-pages" --out "{root}/evidence.jsonl"
  Without it the task degenerates to closed-book fact checking (state = claim).
"""

FEVER_LABELS = {
    "supports": "Supports",
    "refutes": "Refutes",
    "not_enough_info": "Not enough information",
}

_FEVER_KEYS = {
    "SUPPORTS": "supports",
    "REFUTES": "refutes",
    "NOT_ENOUGH_INFO": "not_enough_info",
    "NOTENOUGHINFO": "not_enough_info",
    "NOT_ENOUGH_INFORMATION": "not_enough_info",
}


def load_fever(
    root: Path,
    split: str = "validation",
    *,
    limit: int | None = None,
    evidence: Path | None = None,
) -> list[DecisionItem]:
    hint = render_hint(FEVER_HINT, root)
    rows = read_records(
        root,
        split,
        patterns=("*.jsonl", "*.json"),
        hf_ids=("fever", "fever_gold", "pietrolesci/fever"),
        hf_config="v1.0",
        hint=hint,
    )
    evidence_map = _read_evidence_map(evidence or _default_evidence_path(root))
    items: list[DecisionItem] = []
    for index, row in enumerate(rows):
        raw_label = _clean(_field(row, ("label", "gold_label"), "")).upper().replace(" ", "_")
        key = _FEVER_KEYS.get(raw_label)
        if key is None:
            continue
        claim = _clean(_field(row, ("claim", "text", "sentence"), ""))
        if not claim:
            continue
        item_id = str(_field(row, ("id", "uid", "claim_id"), index))
        context = _evidence_text(row, evidence_map.get(item_id))
        state = context if context else claim
        items.append(
            DecisionItem(
                item_id=item_id,
                state=state,
                questions={
                    "evidence": {
                        "type": "choice",
                        "instructions": f"Claim: {claim}\nDoes the state support this claim?",
                        "criteria": dict(FEVER_LABELS),
                    }
                },
                targets={"evidence": key},
                meta={"claim": claim, "has_evidence": bool(context), "label": key},
            )
        )
        if limit and len(items) >= limit:
            break
    if not items:
        raise DatasetUnavailable("fever", [root], hint)
    return items


def _default_evidence_path(root: Path) -> Path:
    if root.is_dir():
        for name in ("evidence.jsonl", "evidence.json", "wiki_pages.jsonl"):
            candidate = root / name
            if candidate.is_file():
                return candidate
    return DEFAULT_RAW_DIR / "FEVER" / "_no_evidence_.jsonl"


def _read_evidence_map(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        rows = _read_file(path)
    except Exception:
        return {}
    mapping: dict[str, str] = {}
    for row in rows:
        key = _field(row, ("id", "claim_id", "page"))
        value = _field(row, ("evidence", "text", "sentences", "lines"))
        if key is None or value is None:
            continue
        text = _evidence_value_to_text(value)
        if text:
            mapping[str(key)] = text
    return mapping


def _evidence_value_to_text(value: Any) -> str:
    """Text for an evidence field, whichever shape the source uses."""

    if isinstance(value, str):
        return _clean(value)
    if isinstance(value, dict):
        direct = _field(value, ("text", "sentence", "line"))
        if direct is not None:
            return _clean(direct)
        value = _field(value, ("evidence", "sentences", "lines"), [])
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for entry in value:
            if isinstance(entry, str):
                parts.append(_clean(entry))
            elif isinstance(entry, dict):
                direct = _field(entry, ("text", "sentence", "line"))
                if direct is not None:
                    parts.append(_clean(direct))
            elif isinstance(entry, (list, tuple)):
                parts.append(_evidence_value_to_text(entry))
        joined = " ".join(part for part in parts if part)
        if joined:
            return _clean(joined)
        # Raw FEVER annotations carry page#sentence *references*, not text. Those are
        # not evidence, so the caller falls back to the claim; use
        # scripts/fever_evidence.py with the wiki dump for real evidence.
        return ""
    return ""


def iter_evidence_pairs(node: Any):
    """Yield ``(page, sentence_id)`` from either FEVER evidence shape.

    The raw claim files nest lists (``evidence[annotator][group][pair]``); the
    HuggingFace conversion replaces that with dicts carrying ``page`` and
    ``sentence_id``. Both are walked here.
    """

    if isinstance(node, dict):
        page = node.get("page") or node.get("page_id") or node.get("title")
        sentence = node.get("sentence_id", node.get("sentence"))
        if page is not None and sentence is not None:
            yield (str(page), sentence)
        elif "evidence" in node:
            yield from iter_evidence_pairs(node["evidence"])
        return
    if isinstance(node, (list, tuple)):
        if len(node) == 2 and isinstance(node[0], str) and not isinstance(node[1], (list, tuple, dict)):
            yield (node[0], node[1])
            return
        for child in node:
            yield from iter_evidence_pairs(child)


def _evidence_text(row: dict, extra: str | None) -> str:
    if extra:
        return extra
    raw = _field(row, ("evidence", "context", "evidence_text"))
    if raw is None:
        return ""
    if isinstance(raw, str):
        return _clean(raw)
    if isinstance(raw, dict) and "sentences" in raw:
        raw = raw["sentences"]
    lines: list[str] = []
    for entry in raw if isinstance(raw, (list, tuple)) else [raw]:
        if isinstance(entry, dict):
            text = _field(entry, ("text", "sentence", "line"))
            if text is not None:
                lines.append(_clean(text))
        elif isinstance(entry, str):
            lines.append(_clean(entry))
    return " ".join(line for line in lines if line)


# ------------------------------------------------------------------ MMLU / Pro
MMLU_HINT = """MMLU / MMLU-Pro - put the test split in {root}

  test.jsonl, one record per item:
    {"question": ..., "choices": ["...", ...], "answer": 0, "subject": "..."}
  (`answer` may be an index or a letter; `choices` may have 4 or 10 entries.)

  from Berkeley's tarball:
    curl -L -o mmlu.tar https://people.eecs.berkeley.edu/~hendrycks/data.tar
    tar xf mmlu.tar            # then convert data/test/*_test.csv with the snippet in the README

  or from the HuggingFace cache:
    python -c "from datasets import load_dataset; load_dataset('cais/mmlu','all').save_to_disk(r'{root}')"
    python -c "from datasets import load_dataset; load_dataset('TIGER-Lab/MMLU-Pro').save_to_disk(r'{root}')"
"""

MMLU_PRO_HINT = """MMLU-Pro - put the test split in {root}

  test.jsonl, one record per item with ten options:
    {"question": ..., "options": ["...", x10], "answer": "C", "category": "math"}

  from the HuggingFace cache (no auth):
    python -c "from datasets import load_dataset; load_dataset('TIGER-Lab/MMLU-Pro').save_to_disk(r'{root}')"

  or the parquet from the repo (the loader reads parquet directly):
    git clone https://github.com/TIGER-AI-Lab/MMLU-Pro
    copy MMLU-Pro/data/test-00000-of-00001.parquet -> {root}
"""


def load_mmlu(
    root: Path,
    split: str = "test",
    *,
    limit: int | None = None,
    state_mode: str = "question",
    max_options: int | None = 26,
) -> list[DecisionItem]:
    hint = render_hint(MMLU_HINT, root)
    rows = read_records(
        root,
        split,
        patterns=("*test*.jsonl", "*test*.json", "*test*.parquet", "*test*.csv", "*.jsonl", "*.csv"),
        hf_ids=("cais/mmlu", "TIGER-Lab/MMLU-Pro"),
        hint=hint,
    )
    items: list[DecisionItem] = []
    for index, row in enumerate(rows):
        question = _clean(_field(row, ("question", "question_stem", "stem", "input"), ""))
        options = _field(row, ("choices", "options", "candidates", "answers"), None)
        if isinstance(options, str):
            options = [part.strip() for part in options.split("\n") if part.strip()]
        if not question or not options:
            continue
        options = [_clean(option) for option in options]
        if max_options and len(options) > max_options:
            continue
        target = _answer_key(_field(row, ("answer", "label", "gold", "correct", "answer_key")), len(options))
        if target is None:
            continue
        subject = _clean(_field(row, ("subject", "category", "domain"), ""))
        labels = list(SLOT_LETTERS[: len(options)])
        if state_mode == "question":
            state = f"Subject: {subject}\nQuestion: {question}" if subject else f"Question: {question}"
            instructions = CHOICE_TEMPLATE
        else:
            state = f"Subject: {subject}" if subject else ""
            instructions = question
        items.append(
            DecisionItem(
                item_id=str(_field(row, ("id", "question_id"), index)),
                state=state,
                questions={
                    "answer": {
                        "type": "choice",
                        "instructions": instructions,
                        "criteria": letter_criteria(labels, options),
                    }
                },
                targets={"answer": target},
                meta={"subject": subject, "question": question, "n_options": len(options)},
            )
        )
        if limit and len(items) >= limit:
            break
    if not items:
        raise DatasetUnavailable("mmlu", [root], hint)
    return items


def _answer_key(answer: Any, n_options: int) -> str | None:
    if answer is None or isinstance(answer, bool):
        return None
    limit = min(n_options, len(SLOT_LETTERS))
    if isinstance(answer, (int, float)):
        index = int(answer)
        return SLOT_LETTERS[index] if 0 <= index < limit else None
    text = str(answer).strip()
    if len(text) == 1 and text.upper() in SLOT_LETTERS:
        index = SLOT_LETTERS.index(text.upper())
        return SLOT_LETTERS[index] if index < limit else None
    if text.isdigit():
        index = int(text)
        return SLOT_LETTERS[index] if 0 <= index < limit else None
    return None


# --------------------------------------------------------------- CLINC150 / HWU64
CLINC_HINT = """CLINC150 - put the official JSON in {root}

  The official files are dictionaries of splits:
    {"train": [[text, intent], ...], "test": [...], "oos_test": [...], ...}

  git clone https://github.com/clinc/oos-eval
  copy oos-eval/data/data_full.json   -> {root}/data_full.json
  copy oos-eval/data/data_small.json  -> {root}/data_small.json

  or from the HuggingFace cache (configs: small, plus, imbalanced):
    python -c "from datasets import load_dataset; load_dataset('clinc_oos','plus').save_to_disk(r'{root}')"
"""

HWU64_HINT = """HWU64 (NLU Evaluation Data) - put the CSV in {root}

  train.csv, test.csv with columns: utterance, intent, scenario

  git clone https://github.com/xliuhw/NLU-Evaluation-Data
  copy NLU-Evaluation-Data/dataset/NLU-Evaluation-Data-Canonical-Form.csv -> {root}/test.csv

  The canonical CSV names its columns answer / scenario / intent / utterance;
  both spellings are accepted.
"""

INTENT_INSTRUCTIONS = "Which intent does this utterance express?"
DOMAIN_INSTRUCTIONS = "Which domain does this utterance belong to?"


def load_clinc150(
    root: Path,
    split: str = "test",
    *,
    limit: int | None = None,
    max_intents: int | None = None,
    include_domain: bool = False,
) -> list[DecisionItem]:
    hint = render_hint(CLINC_HINT, root)
    rows = _read_clinc_rows(root, split, hint)
    items = _intent_items(
        rows,
        dataset="clinc150",
        limit=limit,
        max_intents=max_intents,
        include_domain=include_domain,
    )
    if not items:
        raise DatasetUnavailable("clinc150", [root], hint)
    return items


def _read_clinc_rows(root: Path, split: str, hint: str) -> list[dict]:
    """CLINC ships ``{"train": [[text, intent], ...]}`` rather than records."""

    if root.is_dir():
        names = [f"{split}.json", "data_full.json", "full.json", "data_small.json", "small.json"]
        for name in names:
            path = root / name
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                break
            values = list(payload.values())
            if not values or not isinstance(values[0], list):
                break
            key = split if split in payload else _first_matching_key(payload, split)
            if key is None:
                continue
            return [
                {"text": entry[0], "intent": entry[1]}
                for entry in payload[key]
                if isinstance(entry, (list, tuple)) and len(entry) >= 2
            ]
    return read_records(root, split, hf_ids=("clinc_oos",), hint=hint)


def _first_matching_key(payload: dict, split: str) -> str | None:
    for key in payload:
        if split in key or key in split:
            return key
    return None


def load_hwu64(
    root: Path,
    split: str = "test",
    *,
    limit: int | None = None,
    max_intents: int | None = None,
    include_domain: bool = True,
) -> list[DecisionItem]:
    hint = render_hint(HWU64_HINT, root)
    files = _find_files(root, split, patterns=("*.csv", "*.jsonl", "*.tsv"))
    if files:
        rows: list[dict] = []
        for path in files:
            rows.extend(_read_file(path))
    else:
        rows = read_records(root, split, hf_ids=("hwu64",), hint=hint)
    items = _intent_items(
        rows,
        dataset="hwu64",
        limit=limit,
        max_intents=max_intents,
        include_domain=include_domain,
    )
    if not items:
        raise DatasetUnavailable("hwu64", [root], hint)
    return items


def _intent_items(
    rows: Sequence[dict],
    *,
    dataset: str,
    limit: int | None,
    max_intents: int | None = None,
    include_domain: bool = False,
) -> list[DecisionItem]:
    prepared: list[tuple[str, str, str, str]] = []
    for index, row in enumerate(rows):
        text = _clean(_field(row, ("text", "utterance", "answer", "query", "sentence"), ""))
        intent = _field(row, ("intent", "label", "category", "intent_name"))
        if isinstance(intent, (list, tuple)):
            intent = intent[0] if intent else None
        if isinstance(intent, str) and "," in intent:
            # HWU64's annotated file can carry several intents per utterance.
            intent = intent.split(",")[0]
        if isinstance(intent, (int, float)) and not isinstance(intent, bool):
            names = _field(row, ("intent_names", "label_names"), None)
            if names is None:
                continue
            intent = names[int(intent)]
        if not text or intent is None:
            continue
        domain = _clean(_field(row, ("scenario", "domain", "intent_domain", "service"), ""))
        prepared.append((str(index), text, _intent_name(intent), domain))
    if not prepared:
        return []

    counts: dict[str, int] = {}
    for _, _, intent, _ in prepared:
        counts[intent] = counts.get(intent, 0) + 1
    if max_intents:
        keep = {name for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_intents]}
    else:
        keep = set(counts)

    domains = sorted({domain for _, _, _, domain in prepared if domain})
    items: list[DecisionItem] = []
    for index, text, intent, domain in prepared:
        if intent not in keep:
            continue
        questions = {
            "intent": {
                "type": "choice",
                "instructions": INTENT_INSTRUCTIONS,
                "criteria": {name: name.replace("_", " ") for name in sorted(keep)},
            }
        }
        targets = {"intent": intent}
        if include_domain and domains:
            questions["domain"] = {
                "type": "choice",
                "instructions": DOMAIN_INSTRUCTIONS,
                "criteria": {name: name.replace("_", " ") for name in domains},
            }
            targets["domain"] = domain
        items.append(
            DecisionItem(
                item_id=f"{dataset}-{index}",
                state=text,
                questions=questions,
                targets=targets,
                meta={"intent": intent, "domain": domain},
            )
        )
        if limit and len(items) >= limit:
            break
    return items


def _intent_name(name: Any) -> str:
    return str(name).strip().lower().replace(" ", "_")


# ---------------------------------------------------------------------- Jigsaw
JIGSAW_HINT = """Jigsaw toxic comment classification - put the CSVs in {root}

  train.csv, test.csv, test_labels.csv with columns:
    id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

  Kaggle needs an API token (~/.kaggle/kaggle.json):
    kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -d "{root}"
    (then unpack the zip; the official test split ships without labels, so
     test.csv must be joined with test_labels.csv - the loader does that for you)

  or from the HuggingFace cache (no auth):
    python -c "from datasets import load_dataset; load_dataset('google/jigsaw_toxicity_pred', data_dir='jigsaw-toxic-comment-classification-challenge').save_to_disk(r'{root}')"
  (if that id is gated, the mirror 'thesofakillers/jigsaw-toxic-comment-classification-challenge' works too)
"""

JIGSAW_LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")


def load_jigsaw(
    root: Path,
    split: str = "test",
    *,
    limit: int | None = None,
    labels: Sequence[str] = JIGSAW_LABELS,
) -> list[DecisionItem]:
    hint = render_hint(JIGSAW_HINT, root)
    rows = _read_jigsaw_rows(root, split, hint)
    items: list[DecisionItem] = []
    for index, row in enumerate(rows):
        text = _clean(_field(row, ("comment_text", "text", "comment", "body"), ""))
        if not text:
            continue
        questions = {
            label: {
                "type": "bool",
                "instructions": f"Does this comment contain {label.replace('_', ' ')} content?",
            }
            for label in labels
        }
        targets: dict[str, str] = {}
        for label in labels:
            value = _field(row, (label, f"{label}_label"))
            if value is None:
                targets = {}
                break
            targets[label] = "yes" if _to_int(value) == 1 else "no"
        if not targets:
            continue
        items.append(
            DecisionItem(
                item_id=str(_field(row, ("id", "comment_id"), index)),
                state=text,
                questions=questions,
                targets=targets,
                meta={"labels": list(labels)},
            )
        )
        if limit and len(items) >= limit:
            break
    if not items:
        raise DatasetUnavailable("jigsaw", [root], hint)
    return items


def _read_jigsaw_rows(root: Path, split: str, hint: str) -> list[dict]:
    """Try the plain files first (joining the separate label file), then HF."""

    merged = _merge_jigsaw_csvs(root, split)
    if merged:
        return merged
    return read_records(
        root,
        split,
        hf_ids=(
            "google/jigsaw_toxicity_pred",
            "thesofakillers/jigsaw-toxic-comment-classification-challenge",
        ),
        hint=hint,
    )


def _merge_jigsaw_csvs(root: Path, split: str) -> list[dict]:
    if not root.is_dir():
        return []
    comments: dict[str, dict] = {}
    for path in _find_files(root, split, patterns=("*.csv", "*.jsonl", "*.tsv")):
        if split in ("test",) and "_labels" in path.name:
            continue
        for row in _read_file(path):
            item_id = str(_field(row, ("id", "comment_id"), ""))
            text = _field(row, ("comment_text", "text", "comment"))
            if item_id and text is not None:
                comments.setdefault(item_id, {"id": item_id, "comment_text": text})
    if not comments:
        return []
    for path in _find_files(root, f"{split}_labels", patterns=("*.csv",)):
        for row in _read_file(path):
            item_id = str(_field(row, ("id", "comment_id"), ""))
            if item_id in comments:
                comments[item_id].update(row)
    rows = list(comments.values())
    labelled = [row for row in rows if _field(row, ("toxic", "toxic_label")) is not None]
    return labelled or rows


# -------------------------------------------------------------------- registry
@dataclass
class DatasetSpec:
    key: str
    title: str
    task: str
    state: str
    decision: str
    default_split: str
    readout: str
    loader: Callable[..., list[DecisionItem]]
    hint: str
    note: str = ""
    hf_ids: tuple[str, ...] = ()

    def load(self, root: Path | None = None, **kwargs) -> list[DecisionItem]:
        return self.loader(root or default_root(self.key), **kwargs)


DATASETS: dict[str, DatasetSpec] = {
    "fever": DatasetSpec(
        key="fever",
        title="FEVER - fact verification",
        task="choice: supports / refutes / not enough information",
        state="evidence sentences (falls back to the claim)",
        decision="is the claim supported by the state?",
        default_split="validation",
        readout="reserved_label (3 options)",
        loader=load_fever,
        hint=FEVER_HINT,
        note="Add evidence.jsonl for the real +evidence task; claim-only is closed-book fact checking.",
        hf_ids=("fever",),
    ),
    "mmlu": DatasetSpec(
        key="mmlu",
        title="MMLU / MMLU-Pro - multiple choice",
        task="choice: A..J",
        state="subject + question stem",
        decision="which option is correct?",
        default_split="test",
        readout="reserved_label (4-10 options)",
        loader=load_mmlu,
        hint=MMLU_HINT,
        note="Same reliability analysis as the essay's Figure 5.",
        hf_ids=("cais/mmlu", "TIGER-Lab/MMLU-Pro"),
    ),
    "clinc150": DatasetSpec(
        key="clinc150",
        title="CLINC150 - intent detection",
        task="choice: 150 intents + out of scope",
        state="utterance",
        decision="which intent does this utterance express?",
        default_split="test",
        readout="slot_head (150 labels exceed the 26 reserved letters)",
        loader=load_clinc150,
        hint=CLINC_HINT,
        note="Use --max-intents 20 for a quick zero-shot run with the reserved-label readout.",
        hf_ids=("clinc_oos",),
    ),
    "mmlu_pro": DatasetSpec(
        key="mmlu_pro",
        title="MMLU-Pro - multiple choice, ten options",
        task="choice: A..J",
        state="category + question stem",
        decision="which option is correct?",
        default_split="test",
        readout="reserved_label (10 options)",
        loader=load_mmlu,
        hint=MMLU_PRO_HINT,
        note="Same loader as MMLU; the Pro split is usually one parquet or jsonl file.",
        hf_ids=("TIGER-Lab/MMLU-Pro",),
    ),
    "hwu64": DatasetSpec(
        key="hwu64",
        title="HWU64 - intent detection (NLU Evaluation Data)",
        task="choice: 64 intents, plus an optional 3-way domain question",
        state="utterance",
        decision="which intent does this utterance express?",
        default_split="test",
        readout="slot_head (64 labels)",
        loader=load_hwu64,
        hint=HWU64_HINT,
        note="Two decisions per item when include_domain is on: intent and domain.",
        hf_ids=("hwu64",),
    ),
    "jigsaw": DatasetSpec(
        key="jigsaw",
        title="Jigsaw toxic comment classification",
        task="six independent yes/no decisions",
        state="comment text",
        decision="does the comment contain <label> content?",
        default_split="test",
        readout="reserved_label (2 options per question)",
        loader=load_jigsaw,
        hint=JIGSAW_HINT,
        note="Six isolated questions share one comment encoding: the architecture's ideal shape.",
        hf_ids=(
            "google/jigsaw_toxicity_pred",
            "thesofakillers/jigsaw-toxic-comment-classification-challenge",
        ),
    ),
}


def default_root(key: str) -> Path:
    """``$QWENJEV_DATA/<key>`` when the variable is set, else ``<repo>/data/raw/<key>``."""

    override = os.environ.get("QWENJEV_DATA")
    base = Path(override) if override else DEFAULT_RAW_DIR
    direct = base / key
    if direct.is_dir() or not base.is_dir():
        return direct
    # The raw folders keep their original capitalisation (FEVER, HWU64, ...).
    lowered = {path.name.lower(): path for path in base.iterdir() if path.is_dir()}
    return lowered.get(key.lower(), direct)


def default_normalized_dir() -> Path:
    """Where the formatted train/test files live: ``$QWENJEV_NORMALIZED`` or ``data/ready``."""

    override = os.environ.get("QWENJEV_NORMALIZED")
    return Path(override) if override else DEFAULT_NORMALIZED_DIR


def normalized_items(key: str, split: str, *, limit: int | None = None, directory: Path | None = None):
    """Read ``<directory>/<key>_<split>.jsonl``, or return ``None`` when absent."""

    from .normalize import load_normalized  # local import: normalize imports this module

    directory = Path(directory) if directory else default_normalized_dir()
    path = directory / f"{key}_{split}.jsonl"
    if not path.is_file():
        return None
    items = load_normalized(path)
    return items[:limit] if limit else items


def load_items(
    key: str,
    *,
    root: Path | str | None = None,
    split: str | None = None,
    limit: int | None = None,
    normalized: Path | str | None = None,
    **kwargs,
) -> list[DecisionItem]:
    """Load a dataset by key.

    Order: a normalised file (``qwenjev normalize`` output), then the raw adapters
    from this module, then the ``synthetic`` generator.
    """

    # An explicit ``root`` means "read this raw directory"; the normalised files are
    # only consulted for the default lookup path.
    if key != "synthetic" and root is None:
        wanted = split or (DATASETS[key].default_split if key in DATASETS else "test")
        cached = normalized_items(key, wanted, limit=limit, directory=normalized)
        if cached is not None:
            return cached

    if key == "synthetic":
        from .synth import question_specs, sample_state

        rng = random.Random(kwargs.pop("seed", 0))
        shift = bool(kwargs.pop("shift", False))
        specs = question_specs()
        wanted = kwargs.pop("question", None)
        items = []
        for index in range(limit or 64):
            state, queue, urgent = sample_state(rng, shift=shift)
            questions = {k: v for k, v in specs.items() if wanted in (None, k)}
            truth = {"queue": queue, "escalate": "yes" if urgent else "no"}
            items.append(
                DecisionItem(
                    item_id=f"synthetic-{index}",
                    state=state,
                    questions=questions,
                    targets={k: truth[k] for k in questions},
                )
            )
        return items
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {key!r}; known: {sorted(DATASETS) + ['synthetic']}")
    spec = DATASETS[key]
    return spec.loader(Path(root) if root else default_root(key), split=split or spec.default_split, limit=limit, **kwargs)


def status(root: Path | str | None = None) -> list[dict]:
    """Which datasets are present on disk (used by ``qwenjev datasets``)."""

    report = []
    for key, spec in DATASETS.items():
        directory = Path(root) if root else default_root(key)
        if directory.is_dir():
            files = sorted(p.name for p in directory.iterdir() if p.is_file())
        else:
            files = []
        report.append(
            {
                "dataset": key,
                "title": spec.title,
                "root": str(directory),
                "available": bool(files) or directory.is_file(),
                "files": files[:12],
                "hf_ids": list(spec.hf_ids),
                "readout": spec.readout,
                "task": spec.task,
                "note": spec.note,
                "hint": render_hint(spec.hint, directory),
            }
        )
    return report
