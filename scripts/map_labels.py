"""Recover a human-readable label name for every integer label of a dataset.

Some releases (the CLINC150 / HWU64 parquet files here) store the intent as an integer
and ship the names in a separate file with a *different* sample. When the two share a
label space, the names can be recovered by matching the two label clusters in the
backbone's own embedding space:

    python scripts/map_labels.py \
        --examples data/raw/CLINC150/clinc150-tfidf-testset.tsv \
        --parquet "data/raw/CLINC150/train-00000-of-00001 (1).parquet" \
        --out data/raw/CLINC150/intents.txt

The output is one name per line in label-id order, which is exactly what
``qwenjev.normalize`` looks for (``intents.txt``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwenjev.config import default_model_path  # noqa: E402


@torch.no_grad()
def embed(model, tokenizer, texts, *, batch_size: int = 16, max_length: int = 128, device: str = "cuda:0"):
    """Mean-pooled last hidden state, L2-normalised."""

    vectors = []
    for start in tqdm(range(0, len(texts), batch_size), desc="embedding", unit="batch"):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)
        out = model.model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        hidden = out.last_hidden_state.float()
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        vectors.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
    return torch.cat(vectors).numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", required=True, help="TSV/CSV with text and label name")
    parser.add_argument("--parquet", required=True, help="dataset whose integer labels to name")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=None, help="backbone path (default $QWENJEV_MODEL)")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--max-examples", type=int, default=4000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.model = args.model or default_model_path()

    import pandas as pd

    # labelled examples -> name -> texts
    delimiter = "\t" if args.examples.endswith(".tsv") else ","
    with open(args.examples, encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.reader(handle, delimiter=delimiter) if len(r) >= 2]
    by_name: dict[str, list[str]] = defaultdict(list)
    for text, name in rows:
        by_name[name.strip()].append(text.strip())
    names = sorted(by_name)
    print(f"{len(names)} names from {args.examples}")

    # parquet -> integer label -> texts
    frame = pd.read_parquet(args.parquet)
    text_column = args.text_column or ("utterance" if "utterance" in frame else frame.columns[0])
    label_column = args.label_column or ("label" if "label" in frame else frame.columns[1])
    by_label: dict[int, list[str]] = defaultdict(list)
    per_label = 10
    for text, label in zip(frame[text_column], frame[label_column]):
        if pd.isna(label) or pd.isna(text):
            continue
        bucket = by_label[int(label)]
        if len(bucket) < per_label:  # balanced across labels: the file is label-sorted
            bucket.append(str(text).strip())
    labels = sorted(by_label)
    print(f"{len(labels)} integer labels from {Path(args.parquet).name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": args.device}
    )
    model.eval()

    name_texts = [" ; ".join(by_name[n][:5]) for n in names]
    label_texts = [" ; ".join(by_label[l][:8]) for l in labels]
    all_vectors = embed(model, tokenizer, name_texts + label_texts, device=args.device)
    name_vectors = all_vectors[: len(names)]
    label_vectors = all_vectors[len(names) :]

    similarity = cosine_similarity(label_vectors, name_vectors)
    row, col = linear_sum_assignment(-similarity)
    mapping = {labels[r]: names[c] for r, c in zip(row, col)}
    scores = [similarity[r, c] for r, c in zip(row, col)]
    print(
        f"assignment: mean similarity {np.mean(scores):.3f}, "
        f"median {np.median(scores):.3f}, min {np.min(scores):.3f}"
    )
    for index in (0, 1, 2, len(labels) - 1):
        if index in mapping:
            print(f"  label {index:>3} -> {mapping[index]}")

    ordered = [mapping.get(label, f"label_{label}") for label in range(max(labels) + 1)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(ordered) + "\n", encoding="utf-8")
    Path(args.out).with_suffix(".mapping.json").write_text(
        json.dumps({str(k): v for k, v in mapping.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {args.out} ({len(ordered)} names)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
