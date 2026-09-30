<div align="center">

# QwenJev-lite

**A JEV-like model that uses RLCD on Qwen**

![python](https://img.shields.io/badge/python-3.11-3776ab?logo=python&logoColor=white)
![torch](https://img.shields.io/badge/torch-2.6.0%2Bcu126-ee4c2c?logo=pytorch&logoColor=white)
![transformers](https://img.shields.io/badge/transformers-5.14-ffd21e)
![tests](https://img.shields.io/badge/tests-92%20passing-2ea44f)
![license](https://img.shields.io/badge/license-MIT-blue)

[Results](#results) · [Quickstart](#quickstart) · [The shipped model](#the-shipped-model) · [Train and test](#train-and-test) · [Data](#data) · [Layout](#layout) · [How it works](#how-it-works) · [RLCD in brief](#rlcd-in-brief) · [vs. BERT](#compared-with-a-bert-base-classifier) · [Adaptations & data](#what-we-adapted-in-the-model-and-the-data) · [References](#references) · [TODO](#todo)

</div>

> Chinese version: [`README_CN.md`](README_CN.md).

QwenJev-lite formulates a transformer as a **typed decision model**: the shared state is encoded once, each question becomes an isolated branch, and inference returns a probability distribution over the allowed answers rather than generated text. Questions are typed — a finite `choice`, a `bool`, or an ordered `score` — and one request may cover several types on the same state.

The backbone is **Qwen3.5-4B** (32 layers: 24 linear-attention + 8 full-attention, 4.54B parameters — 4.21B language + 0.33B vision, so the backbone is itself multimodal), and training **updates only a 512-row decision head at the output end** (≈5 MB), so a full experiment fits on a single 24 GB GPU. Across 26 test splits it reaches **0.772 overall**; the same reserved label rows used directly as the **Qwen baseline** with **no training at all** (`qwen_zeroshot`) reach **0.722**.

```bash
pip install -r requirements.txt
export QWENJEV_MODEL=/path/to/qwen3.5-4B      # Windows: $env:QWENJEV_MODEL="D:\qwen3.5-4B"
python demo.py --checkpoint-dir models/qwenjev-multitask-v2
```

## Results

The full evaluation command is `python test.py --limit 0`, run over the 26 test splits (bool 7 + choice 12 + score 7) that both variants can answer. The values below are accuracy; raw data is in [`artifacts/results.json`](artifacts/results.json) and per-task detail in [`artifacts/RESULTS_CN.md`](artifacts/RESULTS_CN.md).

| Question type / accuracy     | Laya (reference) | Qwen baseline (`qwen_zeroshot`) | **our trained model** |
| ---------------------------- | ---------------- | --------------------------------- | --------------------------- |
| judgement`bool` (7 tasks)  | 0.520            | 0.717                             | **0.793**             |
| choice`choice` (12 tasks)  | 0.511            | 0.749                             | **0.760**             |
| score`score` (7 tasks)     | 0.258            | 0.681                             | **0.771**             |
| **overall (26 tasks)** | **0.446**  | 0.722                             | **0.772**             |

The **Laya** column is a comparable third-party open baseline, measured on the same 26 test splits and listed here only for reference. This repository neither installs nor requires it — `test.py` scores the project's own two variants by default, while `--variants laya` remains available to anyone who already holds that checkpoint.

## Quickstart

```bash
pip install -r requirements.txt          # torch 2.6 + transformers 5.14; tests need no GPU, model or data

export QWENJEV_MODEL=/path/to/qwen3.5-4B # Windows: $env:QWENJEV_MODEL="D:\qwen3.5-4B"

python demo.py --checkpoint-dir models/qwenjev-multitask-v2   # one state, three question types, one pass
python serve.py                                              # the same thing as a web page on :8300
pytest -q                                                    # 92 tests on a tiny backbone, CPU only
```

| Goal                                                              | Command                                                             |
| ----------------------------------------------------------------- | ------------------------------------------------------------------- |
| Run the full benchmark (Qwen baseline + our model, progress bars) | `python test.py --limit 0`                                        |
| Train a decision head (four inputs, all defaulted)                | `python train.py --model /path/to/qwen3.5-4B`                     |
| Rebuild the shipped decision head from its three sources          | [`models/README.md`](models/README.md#rebuild-the-shipped-head)    |
| Convert raw sources into the canonical format                     | `python -m qwenjev.cli normalize --src data/raw --out data/ready` |

## The shipped model

`models/qwenjev-multitask-v2/readout.pt` is the shipped decision head: **512 rows = 27 reserved + 485 private**, covering the 90 label spaces in `data/ready`.

## Train and test

A run needs **four inputs**, and each of them has a default:

| input                         | flag                                                                                                                                                  | default                                         | notes                                                                                                                                                                                           |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| base model                    | `--model`                                                                                                                                           | `$QWENJEV_MODEL`, else `./qwen3.5-4B`       | a local path or a Hub repo id; the engine only reads hidden states, so another causal backbone — or a multimodal one, with the image/audio tokens placed inside`state` — works the same way |
| where to save / load the head | `--checkpoint-dir`                                                                                                                                  | *empty* → `models/qwenjev-run-<timestamp>` | created if missing;`train.py` writes `readout.pt` + `card.json` there, `test.py` reads `readout.pt` from there                                                                        |
| data                          | `--data-dir`                                                                                                                                        | `data/ready`                                  | a folder of`<task>_train.jsonl` / `<task>_test.jsonl` (any `<task>_<split>.jsonl` works)                                                                                                  |
| hyper-parameters              | `--epochs`, `--lr`, `--batch-size`, `--objective`, `--items-per-label`, `--min-items-per-task`, `--prototype-init`, `--optimizer`, … | see`python train.py --help`                   | `--limit`, `--batch` and `--variants` for `test.py`                                                                                                                                     |

For `test.py`, an empty `--checkpoint-dir` means **use the shipped head**; if that head is not on disk, the run falls back to scoring the Qwen baseline only rather than failing. Results are written to `--out` / `--markdown` (parent folders are created), defaulting to `artifacts/results.json` and `artifacts/RESULTS.md`.

**The data format.** One folder, and inside it two files per task. Each file is **JSONL: exactly one record per line** — one JSON object, **no pretty-printing** (a multi-line record is rejected with a message pointing here). A record is the engine's request shape plus the observed answer, and it covers all three question types. The three lines below are deliberately kept as single lines, matching their form in the file.

A `choice` question — settling the next day's arrangement in a group chat:

```json
{"id": "weekend/train/0001", "dataset": "weekend", "split": "train", "state": "Friday night, three of us are pinning down tomorrow in the group chat: Zhe wants to hike, Yu wants a gallery, and I said either is fine.", "questions": {"plan": {"type": "choice", "instructions": "What should we do tomorrow?", "criteria": {"hike": "Hike", "museum": "Gallery", "home": "Stay in"}}}, "targets": {"plan": "museum"}}
```

A `bool` question — the takeout has arrived, judging whether this meal is heavy for her right now:

```json
{"id": "takeout/train/0002", "dataset": "takeout", "split": "train", "state": "Dinner just arrived: a spicy hotpot on one side, a plain noodle soup on the other. Yu's stomach has been off all week, but her chopsticks went for the hotpot first.", "questions": {"spicy": {"type": "bool", "instructions": "Is this bowl heavy for her right now?"}}, "targets": {"spicy": "yes"}}
```

A `score` question — rating the film on the way out (ordered levels, lowest first):

```json
{"id": "movie/train/0003", "dataset": "movie", "split": "train", "state": "The credits roll: Zhe calls it the best film he has seen all year, Yu is yawning next to him, and I am stuck in the middle.", "questions": {"rating": {"type": "score", "instructions": "How good was the film?", "criteria": {"bad": "Bad", "ok": "Fine", "good": "Great"}}}, "targets": {"rating": "ok"}}
```

Rules to observe: each `targets` value must be one of that question's `criteria` keys; `choice` and `score` require `criteria` (ordered — for `score` the declaration order is the scale, so levels must run from lowest to highest), while `bool` may omit it and defaults to yes/no; the question id (`plan`, `rating`, …) is only an identifier, and what decides which row the decision head reads is `(question id, label space, answer key)`.

**Train** — the four inputs above, with the save path omitted so that it is created automatically:

```bash
python train.py \
  --model /path/to/qwen3.5-4B \
  --data-dir data/ready \
  --epochs 1 --lr 1e-3 --batch-size 8 \
  --objective log_loss \
  --items-per-label 12 --min-items-per-task 200 --prototype-init
# or name this run yourself:
#   --checkpoint-dir models/my-run       (created if it does not exist)
#   --epochs 0                           (0 = closed-form prototype rows only, no gradient steps)
```

The script prints the resolved base model, data folder, destination and hyper-parameters before it starts, and writes `readout.pt` + `card.json` into the destination when it finishes.

**Evaluate** — the same three paths, plus the output files (parent folders are created):

```bash
python test.py \
  --model /path/to/qwen3.5-4B \
  --data-dir data/ready \
  --checkpoint-dir models/my-run \
  --variants qwen_zeroshot qwen_trained \
  --limit 0 --batch 16 \
  --out artifacts/my-results.json
```

**A single request**:

```python
from qwenjev.config import QwenJevConfig
from qwenjev.engine import QwenJevLite

engine = QwenJevLite.from_pretrained(config=QwenJevConfig(
    model_path="qwen3.5-4B",                       # any local path or Hub repo id
    readout="slot_head",
    readout_checkpoint="models/qwenjev-multitask-v2/readout.pt",
))
print(engine.decide("My payouts have failed three times.", {
    "queue": {"type": "choice", "instructions": "Which team should handle this ticket?",
              "criteria": {"payments": "Payouts", "account": "Login", "other": "Else"}},
    "escalate": {"type": "bool", "instructions": "Does this require urgent attention?"},
}).to_dict())
```

## Data

`data/ready/` ships with the repository (≈97 MB of JSONL): the formatted train/test sets together with `manifest.json`, which records the source file, row counts and caveats of every split. The sources used are:

| source                                                     | which tasks                                                                                                                                                                  |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MultiNLI / SNLI                                            | `mnli`, `mnli_ood`, `snli` (one premise with several hypotheses)                                                                                                       |
| Jigsaw Toxic Comment                                       | `jigsaw` (six independent yes/no labels), `jigsaw_severity` (ordered)                                                                                                    |
| GoEmotions                                                 | `goemotions` (28 yes/no emotions), `goemotions_sentiment` (ordered)                                                                                                      |
| TruthfulQA                                                 | `truthfulqa` (5-way)                                                                                                                                                       |
| IntentGrasp                                                | `intentgrasp` (the options ship with each item)                                                                                                                            |
| CLINC150 / HWU64                                           | `clinc150`, `clinc150_top15`, `hwu64`, `hwu64_top15`                                                                                                                 |
| BEIR (arguana / nfcorpus / scidocs / scifact / trec-covid) | 15 relevance-judgement tasks built from each collection's queries and qrels: the labelled document becomes the positive option, non-qrels the negatives; splits are by query |

`artifacts/extra_splits.json` records the training splits this repository builds itself (source, row counts, and the check that every answer the test file reads is trained), and `data/ready/manifest.json` records the provenance of every other split. The authors and maintainers of these datasets, annotations and tools made this work possible, and are gratefully acknowledged; full sources are listed under [References](#references).

## Layout

```
qwenjev/           the library: schema / prompt / tokenize / readout / engine / rlcd /
                   calibration / datasets / normalize / relevance / backends / benchmark /
                   probes / api / cli
train.py           train one decision head over the train splits of a data directory (progress bar)
test.py            score the Qwen baseline and our model over the test splits (progress bar)
demo.py serve.py   user-facing CLI demo and web demo
scripts/           build_extra_splits / compose_wide_rows / interpolate / dev_score /
                   map_labels / compare_results / speed_compare / split_csv (and a few
                   converters for rebuilding the canonical files from other mirrors)
models/            the shipped decision head + the three heads it is composed from
data/ready/        formatted train/test JSONL + manifest.json (in git, ≈97 MB)
data/raw/          the source datasets as provided (git-ignored)
artifacts/         RESULTS_CN.md (final numbers), results.json (raw evaluation),
                   extra_splits.json (provenance of the built splits)
tests/             92 tests + the mini fixtures the dataset adapters are tested against
```

## How it works

**Inference performs no decoding.** Given the hidden state `h` at a branch's decision position, the decision head computes `z = W h + b` over the K allowed answers and applies a softmax, returning a probability distribution directly ([`qwenjev/readout.py`](qwenjev/readout.py) — `ReservedLabelReadout` / `SlotHeadReadout` / `PointerReadout`).

- **Reserved label rows give a working model before any training.** `ReservedLabelReadout` renormalises the pretrained LM head's mass over the option-label tokens, so **K ≤ 26 answers work with zero training**; those same rows also serve as the initialisation of the trained head and as its fallback for unseen answers.
- **Questions are typed**: finite choices, yes/no, ordered scores ([`qwenjev/schema.py`](qwenjev/schema.py)).
- **Options are read listwise.** All options are placed into the branch as one ordered list, and the decision head reads the position *after* the list ([`qwenjev/prompt.py`](qwenjev/prompt.py), [`qwenjev/tokenize.py`](qwenjev/tokenize.py)) — so the answer can depend on the whole candidate set.
- **The shared state is encoded once.** It goes into a KV/recurrent cache that is expanded per branch, and each branch attends to that state plus only its own suffix; **isolation is structural**, not a prompting convention ([`qwenjev/engine.py`](qwenjev/engine.py)).
- **Branches are scheduled as a batch** under a batch size and a token budget (`_chunk_branches`); `share_state=False` is the reference path that re-encodes the state per branch, kept as a differential test. Limits are **32,768 tokens per branch** and **65,536 per request**, with the state counted once (`JevLimits`, `check_limits`, `account`).
- **Training is RLCD**: the backbone stays frozen and the distribution is fitted from outcomes with a proper scoring rule ([`qwenjev/rlcd.py`](qwenjev/rlcd.py)) — see [RLCD in brief](#rlcd-in-brief).
- **Calibration is measured, not assumed**: reliability bins, ECE, Brier and Wilson intervals, with `confidence` kept arithmetic ([`qwenjev/calibration.py`](qwenjev/calibration.py), [`qwenjev/confidence.py`](qwenjev/confidence.py)). One evaluation code path drives both backends behind a single interface ([`qwenjev/backends.py`](qwenjev/backends.py)).

## RLCD in brief

**RLCD** (Reinforcement Learning for Calibrated Decisions) is built on one premise: **train from outcomes rather than from a fixed label set**. Its idea can be stated in one sentence: *a decision is a distribution over the answers the question allows, so it should be trained with an objective that is minimised only when the distribution is correct.*

**The decision head.** Given the hidden state `h ∈ R^d` at the decision position of a branch and the `K` answers the question allows, the model assigns each answer a score and normalises:

```
z_k = w_k · h + b_k                     (k = 1 … K)
p_k = softmax(z / τ)_k = exp(z_k/τ) / Σ_j exp(z_j/τ)
```

`w_k` is a row of the decision head — **one row per allowed answer, not per position** — and `τ` is a temperature (1.0 by default, changed only when fitted on an independent calibration split).

**The objective.** Let `y` be the answer that actually occurred. RLCD minimises a **proper scoring rule** of the predicted distribution:

```
log loss :  L = −log p_y = −z_y/τ + log Σ_j exp(z_j/τ)
Brier    :  L = Σ_k (p_k − 1{y = k})²
```

Both are **strictly proper**: their minimum in expectation lies **exactly** at `p = P(y | state, question)`, so what is learned is the probability itself rather than merely a higher score. The gradient is equally direct:

```
∂L/∂z_k = p_k − 1{y = k}      (log loss, τ = 1)
```

That is, each sample pushes the row of the answer that occurred upward and pushes the remaining rows downward in proportion to their current probability.

**Why calibration matters.** A classifier judged only on argmax correctness may be arbitrarily over-confident; here **over-confidence is penalised directly by the loss** and examined with ECE — on the 26-task benchmark the trained model reaches a mean ECE of **0.125** against **0.158** for the Qwen baseline, while also achieving higher accuracy.

**Why it carries a reinforcement component.** The supervision is a **decision plus its outcome** (`state → question → answer → occurred`) and the fitted parameters are the rows corresponding to answers, while **label spaces never trained remain usable** (they read back the reserved rows). The same objective can therefore extend a deployed model to new answer sets without rebuilding a head for each label set. This repository implements the **supervised form** of that objective: the backbone is frozen and the decision head is fitted with log loss or Brier (`train.py --objective log_loss|brier`); the same loss can carry backbone gradients when memory permits.

**Two structural details obtained from experiments** (both part of the training recipe):

- **A private row block per `(question, label space, answer)`.** When rows are shared by position, training a 2–4-option task overwrites rows read by a 10–15-option task (0.733 with the shared rows on the same task, versus 0.760 once the rows are private); answers that never occurred read back the pretrained rows, so an untrained label space **preserves zero-shot behaviour position by position**.
- **Rows are projected back to the pretrained logit scale after every step** (`‖w‖ ≈ 0.74`). The decision states have norm ≈ 157, so without that projection a single step at lr 1e-3 moves a fresh row roughly a fifth of its useful length and the logits explode (measured training loss 33, against 5.01 for a uniform distribution); the shipped shared rows are additionally blended back by α=0.5 (`W(0.5) = W₀ + 0.5·(W_trained − W₀)`), and label spaces wider than the reserved rows use the closed-form prototype rows described above, because 150-way cross-entropy over roughly 20 samples per class does not converge within one epoch of gradient steps.

## Compared with a BERT-base classifier

The usual way to use a transformer as a classifier is "BERT-base + one linear head"; the differences that matter are the following:

|                             | BERT-base + one linear head                                             | **QwenJev-lite**                                                                                                    |
| --------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| backbone                    | encoder, ~110M parameters, 12 layers                                    | causal transformer, 4.54B, 32 layers (24 linear-attention + 8 full-attention)                                             |
| questions per forward pass  | one — the pooled`[CLS]` state feeds a single head                    | many — the state is encoded once and each question is an isolated branch, completed in one batch                         |
| where the answer comes from | a head whose**class count is fixed when it is built**             | the state at the branch's decision position plus**K rows, one per allowed answer, with K free to vary per request** |
| adding an answer set        | a new head, a retrain, and one checkpoint per task                      | the**same head covers any label space**: reserved rows give zero-shot ability, private rows cover trained ones      |
| mixing question types       | one classifier per type, one pass per question                          | one request may pose`choice` + `bool` + `score` about the same state                                                |
| uncertainty                 | a softmax that is typically over-confident and**never evaluated** | the distribution**itself is the trained object**, evaluated with ECE / Brier / NLL                                  |
| cost profile                | low cost per pass, but**one question per pass**                   | one prefill per state (4.54B), then ≈20 ms per extra question on that state                                              |
| generation                  | none (classification only)                                              | none, by design —**the decision head replaces the decode loop**                                                    |

In short: a BERT classifier answers "which of my N classes does this belong to", with a head that must exist before training; QwenJev-lite answers "which of these K answers is it", where the answer set is part of the request, multiple questions on one state stay isolated, and the output is a **calibrated distribution** rather than a single argmax.

## What we adapted in the model and the data

**In the model.** We replaced the decode loop and the fixed classifier with a decision head — computing `z = W h + b` over the allowed answers and applying a softmax, **generating no tokens**, so latency does not grow with output length. **Reserved label rows take priority**: the **Qwen baseline** (`qwen_zeroshot`) renormalises the pretrained head's mass over the option-label tokens, which works without training (K ≤ 26) and supplies the trained model with both its initialisation and its fallback for unseen answers. **Shared prefill and branch isolation**: the state is encoded once into a KV/recurrent cache, the cache is expanded per branch, and each branch attends only to the state and its own suffix, so isolation is structural rather than a prompting convention. **Branch scheduling in batches**: branches are packed into a single forward pass under a batch size and a token budget, so one request completes every question on a state in a few forward passes. **Listwise option rendering**: all options are placed into the branch as one ordered list and the decision head reads the position *after* the list, so an answer may depend on the whole candidate set. **One wire format for typed questions**: the trainer, the evaluator, the CLI, the HTTP API and both backends read the same record. **One row per answer**: the rows are determined by `(question id, label space, answer key)`, so the same options in a different order still read the same row and rows never mix across questions. **The training recipe** is a frozen backbone plus RLCD (log loss or Brier), rows projected back to the pretrained logit scale after every step, an α=0.5 blend for the shared rows, closed-form prototype rows for label spaces wider than the reserved budget, and an optional temperature fitted on a held-out split.

**In the data.** **Each dataset corresponds to one canonical record**: every source is normalised into the engine's own request shape (`{"state": …, "questions": {…}, "targets": {…}}`, one JSON per line), so the same file serves training, evaluation, the CLI and the HTTP API with no per-dataset code path. **Everything is expressed as a typed question**: a task is either a `choice` with `criteria`, a `bool` whose `criteria` is optional, or an ordered `score`; boolean questions supply both an `instructions` (question) phrasing and a `claim` (statement) phrasing, because the two backends were measured on different phrasings. **The label space belongs to the data rather than the code**: the `criteria` keys are the answer identities and the decision head maps `(question id, label space, answer key)` to a row, which is why option order does not affect the result and why several tasks can share one decision head without interfering. **Split discipline**: every training row is disjoint from the rows it is scored on, truncation samples uniformly across the whole file rather than taking a prefix, and the training splits this repository builds itself (`artifacts/extra_splits.json`) are validated before being written — if the test set reads an answer that was never trained, the script refuses to write. **Budgets are allocated per label space**: `--items-per-label` keeps a 150-way task from being crowded out by a two-answer one. **Precision**: the backbone runs in bf16 while the decision head stays in float32 (a training step is ~1e-3, below bf16's resolution at that magnitude), and the logits are computed in float32 as well.

## References

The model, data and methods of this project build on prior work; the resources used and their sources are as follows.

**Model**

- **Qwen3.5-4B** (backbone): [huggingface.co/Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).

**Datasets**

- BEIR (retrieval evaluation benchmark): [github.com/beir-cellar/beir](https://github.com/beir-cellar/beir).
- CLINC150 / BANKING / HWU64 (intent detection): [github.com/SaturnAura/BenchmarkingIntentDetection](https://github.com/SaturnAura/BenchmarkingIntentDetection).
- MMLU / MMLU-Pro: [huggingface.co/datasets/TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/tree/main).
- TruthfulQA: [huggingface.co/datasets/domenicrosati/TruthfulQA](https://huggingface.co/datasets/domenicrosati/TruthfulQA/viewer).
- MNLI / SNLI: [github.com/Tikquuss/nli_dataset](https://github.com/Tikquuss/nli_dataset).
- FEVER: [fever.ai](https://fever.ai/).
- Jigsaw Toxic Comment Classification: [github.com/praj2408/Jigsaw-Toxic-Comment-classification](https://github.com/praj2408/Jigsaw-Toxic-Comment-classification).
- GoEmotions: [kaggle.com/datasets/debarshichanda/goemotions](https://www.kaggle.com/datasets/debarshichanda/goemotions).
- IntentGrasp: [huggingface.co/datasets/yuweiyin/IntentGrasp](https://huggingface.co/datasets/yuweiyin/IntentGrasp).

**Prior work and inspiration**

The design of this project was informed by the following articles and repositories: [archerhume.com/posts/jevs-architecture-unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/?v=3), [zhuanlan.zhihu.com/p/2087218714244658951](https://zhuanlan.zhihu.com/p/2087218714244658951), [huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev), [medium.com/data-science-in-your-pocket/openjev-qwen-3-5-as-jev-ai](https://medium.com/data-science-in-your-pocket/openjev-qwen-3-5-as-jev-ai-85f584d2922b), [aiprofitboardroom.com/blog/openjev](https://aiprofitboardroom.com/blog/openjev/).

The authors and maintainers of the models, datasets and tools listed above made this work possible, and are gratefully acknowledged.

## TODO

The following work remains outstanding in this version:

- TODO: **Support swapping the backbone, including multimodal backbones.** The engine requests exactly one item from the backbone — the hidden state at each decision position — so any model that produces hidden states can serve as the backbone. Swapping in another causal LM, or a multimodal encoder (image / audio tokens placed alongside the text inside `state`, or inside a branch's `context`), requires no change to the decision head, the trainer, the evaluator or the HTTP surface, and is performed solely through `--model` (a local path or a Hub repo id); see [Train and test](#train-and-test).
- TODO: **Accelerate inference with flash attention and related methods.** On the pure-torch fallback path, a single decision takes about 37 ms, and the advantage of the per-call design appears only once a single state carries more than about 64 questions. We are introducing fused kernels such as `flash-linear-attention` and `causal-conv1d` to enable the fast path for linear attention, with the goal of reducing per-call latency for long sequences with many questions; `scripts/speed_compare.py` measures that path.
