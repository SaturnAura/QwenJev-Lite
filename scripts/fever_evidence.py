"""Attach wiki sentences to FEVER claims so the task becomes evidence-grounded.

The official claim files (``paper_dev.jsonl`` and friends) give a label, a claim and
the *location* of the supporting sentences, but not the sentence text. This script
joins them with the wiki-pages dump:

    curl -L -o data/fever/wiki-pages.zip https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
    python -c "import zipfile; zipfile.ZipFile('data/fever/wiki-pages.zip').extractall('data/fever/wiki-pages')"
    python scripts/fever_evidence.py --claims data/fever/paper_dev.jsonl \\
        --wiki-pages data/fever/wiki-pages --out data/fever/evidence.jsonl

Output: one JSON object per claim, ``{"id": ..., "evidence": "sentence one. sentence two."}``,
which is exactly the file ``qwenjev.datasets.load_fever`` picks up automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwenjev.datasets import iter_evidence_pairs  # noqa: E402


def collect_requests(claims_path: Path) -> dict[str, set[int]]:
    """Map page id -> the sentence ids the annotations point at."""

    wanted: dict[str, set[int]] = {}
    with claims_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for page, sentence in iter_evidence_pairs(record.get("evidence") or []):
                if sentence is None:
                    continue
                wanted.setdefault(page, set()).add(int(sentence))
    return wanted


def iter_wiki_pages(directory: Path):
    for path in sorted(directory.rglob("wiki-pages*")) + sorted(directory.glob("*.jsonl")):
        if path.is_dir():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", required=True)
    parser.add_argument("--wiki-pages", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    claims_path = Path(args.claims)
    wanted = collect_requests(claims_path)
    print(f"{len(wanted)} pages referenced by {claims_path.name}")

    # page id -> {sentence id: text}
    texts: dict[str, dict[int, str]] = {}
    for page in iter_wiki_pages(Path(args.wiki_pages)):
        page_id = str(page.get("id"))
        if page_id not in wanted:
            continue
        lines = {}
        for entry in str(page.get("text", "")).split("\n"):
            if "\t" in entry:
                index, sentence = entry.split("\t", 1)
            else:
                index, sentence = len(lines), entry
            try:
                lines[int(index)] = sentence.strip()
            except ValueError:
                continue
        texts[page_id] = lines
        if len(texts) == len(wanted):
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with claims_path.open("r", encoding="utf-8") as handle, out_path.open(
        "w", encoding="utf-8"
    ) as out:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sentences: list[str] = []
            for page, index in iter_evidence_pairs(record.get("evidence") or []):
                if index is None:
                    continue
                text = texts.get(str(page), {}).get(int(index))
                if text and text not in sentences:
                    sentences.append(text)
            if sentences:
                out.write(json.dumps({"id": record.get("id"), "evidence": " ".join(sentences)}) + "\n")
                written += 1
    print(f"{written} claims with evidence -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
