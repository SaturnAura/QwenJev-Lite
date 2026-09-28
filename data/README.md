# `data/` - raw sources and the formatted train/test sets

```
data/
  raw/     the original downloaded sources, untouched
  ready/   the one-format train/test sets every script reads
```

## `data/raw/`

The nine provided datasets in whatever layout they shipped in. (The query/document
collections stay outside the repo, in `D:\AutoBM25\dataset_en`.)

| folder | layout | becomes |
|---|---|---|
| `MNLISNLI/` | CSV (`pairID, sentence1, sentence2, label`) | `mnli`, `mnli_ood`, `snli` |
| `FEVER/` | JSONL | `fever`, `fever_support` |
| `Jigsaw-Toxic-Comment-Classification/` | CSV with six label columns | `jigsaw`, `jigsaw_severity` |
| `GoEmotions/` | TSV + `emotions.txt` + `sentiment_mapping.json` | `goemotions`, `goemotions_sentiment` |
| `TruthfulQA/` | CSV | `truthfulqa` |
| `MMLU-Pro/` | parquet | `mmlu_pro` |
| `IntentGrasp/` | JSONL | `intentgrasp` |
| `CLINC150/`, `HWU64/` | parquet (`utterance`, integer `label`) | `clinc150`, `hwu64` |

## `data/ready/`

One JSON object per line, in the engine's own request shape; `manifest.json` records
every split, its source file and its caveats.

```json
{"id": "jigsaw/test/0001b41b…", "dataset": "jigsaw", "split": "test",
 "state": "the shared text",
 "questions": {"toxic": {"type": "bool",
                         "instructions": "Does this comment contain toxic content?",
                         "claim": "This comment contains toxic content.",
                         "criteria": {"yes": "Yes", "no": "No"}}},
 "targets": {"toxic": "no"}, "meta": {"source_file": "train.csv"}}
```

Three question types appear here, matching the three the essay describes: `choice`
(pick one of K options), `bool` (yes/no) and `score` (ordered levels).

## Rebuilding

```bash
python -m qwenjev.cli normalize            # data/raw -> data/ready
python -m qwenjev.cli relevance --src "D:\AutoBM25\dataset_en" ^
       --collections arguana scifact nfcorpus vihealthqa scidocs trec-covid ^
       --queries 60 --train-queries 150     # query/document collections -> data/ready
python -m qwenjev.cli datasets             # what is present, what a task still needs
```

Then run `python train.py` followed by `python test.py`.
