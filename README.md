<div align="center">

# QwenJev-Lite

**One shared state. Isolated question branches. Parallel probability readouts — no text is generated.**

![python](https://img.shields.io/badge/python-3.11-3776ab?logo=python&logoColor=white)
![torch](https://img.shields.io/badge/torch-2.6.0%2Bcu126-ee4c2c?logo=pytorch&logoColor=white)
![transformers](https://img.shields.io/badge/transformers-5.14-ffd21e)
![tests](https://img.shields.io/badge/tests-92%20passing-2ea44f)
![license](https://img.shields.io/badge/license-MIT-blue)

[Quickstart](#quickstart) · [Results](#results) · [Model](#the-model) · [How it works](#how-it-works) · [RLCD](#rlcd-in-one-page) · [vs. a BERT classifier](#compared-with-a-bert-base-classifier) · [Paths & formats](#paths-and-data-formats) · [Data](#data) · [Speed](#speed) · [References](#references)

</div>

> Chinese version: [`README_CN.md`](README_CN.md).

QwenJev-lite turns a transformer into a **typed decision model**: the shared state is
encoded once, every question becomes an isolated branch, and inference ends in a
probability distribution over the allowed answers instead of generated text. Questions
are typed — a finite `choice`, a `bool`, or an ordered `score` — and one request can mix
them on the same state.

The backbone is **Qwen3.5-4B** (32 layers: 24 linear-attention + 8 full-attention, 4.54B
parameters) and training only touches a **512-row readout** (≈5 MB), which is why a whole
experiment fits on one 24 GB card. On 26 test splits it scores **0.772 overall** against
**0.722** for the untrained readout and **0.446** for the Laya baseline.

```bash
pip install -r requirements.txt
export QWENJEV_MODEL=/path/to/qwen3.5-4B      # Windows: $env:QWENJEV_MODEL="D:\qwen3.5-4B"
python demo.py --model-dir models/qwenjev-multitask-v2
```

## Scope

* **The backbone is swappable, multimodal included.** The engine asks the backbone for
  exactly one thing — the hidden state at each decision position. Anything that produces
  hidden states works: another causal LM, or a multimodal encoder, in which case the image
  / audio tokens simply travel with the text inside `state` (or inside a branch's
  `context`). The readout, the trainer, the evaluator and the HTTP surface do not change;
  swapping is one `--model` (a local path or a Hub repo id) — see
  [Paths & data formats](#paths-and-data-formats).
* **Training touches the readout only.** The backbone stays frozen, so a full benchmark
  run takes 14 minutes and the shipped artefact is 5 MB.
* **Everything reported here is reproducible from this repository**: the model files, the
  formatted data (`data/ready/`), the raw evaluation output (`artifacts/results.json`) and
  the scripts that build both.

## Results

One run of `python test.py --variants laya qwen_zeroshot qwen_trained --limit 0` over the
26 test splits every variant can answer (bool 7 + choice 12 + score 7). Raw data in
[`artifacts/results.json`](artifacts/results.json), full per-task detail in
[`artifacts/RESULTS_CN.md`](artifacts/RESULTS_CN.md).

| Question type | Laya (baseline) | pretrained readout | **trained readout** |
|---|---|---|---|
| judgement `bool` (7 tasks) | 0.520 | 0.717 | **0.794** |
| choice `choice` (12 tasks) | 0.511 | 0.749 | **0.760** |
| score `score` (7 tasks) | 0.258 | 0.681 | **0.771** |
| **overall (26 tasks)** | **0.446** | 0.722 | **0.772** |

Mean ECE (lower is better): Laya 0.289 / pretrained 0.158 / **trained 0.125**.

**Label spaces wider than the 26 reserved rows** are listed separately, because the
pretrained readout has no rows for them and cannot answer at all:

| label space | trained readout | Laya | pretrained readout |
|---|---|---|---|
| CLINC150 (150 intents) | 0.527 | 0.545 | cannot answer |
| HWU64 (64 intents) | 0.540 | 0.345 | cannot answer |

Where the gain comes from, and where it does not: the trained readout leads by a wide
margin on relevance judgements (`scifact_rel_choice` 1.000, `trec-covid_rel_choice`
0.900, `arguana_rel_bool` 0.803) and turns "cannot answer" into 0.53–0.54 on the two wide
intent spaces; it stays below the pretrained readout on the two 15-intent subsets
(0.733 / 0.467) and on some closed-book style judgements — the direct cost of keeping the
reserved rows for every label space they cover.

## Quickstart

```bash
pip install -r requirements.txt          # torch 2.6 + transformers 5.14; tests need no GPU, model or data

export QWENJEV_MODEL=/path/to/qwen3.5-4B # Windows: $env:QWENJEV_MODEL="D:\qwen3.5-4B"

python demo.py --model-dir models/qwenjev-multitask-v2   # one state, three question types, one pass
python serve.py                                          # the same thing as a web page on :8300
pytest -q                                                # 92 tests on a tiny backbone, CPU only
```

| I want to… | command |
|---|---|
| run the full benchmark (Laya + both readouts, progress bars) | `python test.py --variants laya qwen_zeroshot qwen_trained --limit 0` |
| train my own readout | `python train.py --prototype-init --items-per-label 12 --min-items-per-task 200` |
| rebuild the shipped head from its three sources | [`models/README.md`](models/README.md#rebuild-the-shipped-head) |
| turn raw sources into the canonical format | `python -m qwenjev.cli normalize --src data/raw --out data/ready` |
| serve `POST /v1/decide` | `python -m qwenjev.cli serve --port 8300` |
| run the architecture probes (leakage, option interaction, latency…) | `python -m qwenjev.cli probe all` |
| poke at the whole pipeline without a GPU | add `--fake` to any entry point (tiny random backbone, seconds) |

## The model

`models/qwenjev-multitask-v2/readout.pt` is the shipped readout: **512 rows = 27 reserved
+ 485 private**, covering the 90 label spaces in `data/ready`.

* rows 0–26 stay the **pretrained label rows**, so a question the head has never seen
  still reads the backbone's own signal;
* every label space seen during training gets its own private block — rows copied by
  position for label spaces the reserved rows cover, and **closed-form prototype rows**
  (the mean state of each answer, one forward pass, no gradient steps) for label spaces
  wider than 26 options, where positional rows simply run out;
* anything that misses both falls back to its reserved row.

Four directories (`models/README.md` has the chain and load snippet; the three source
files reproduce the shipped file **bit for bit**):

| directory | rows | role |
|---|---|---|
| `qwenjev-multitask-v2/` | 512 | the shipped readout (`card.json` carries the benchmark) |
| `shared-rows/` | 256 | the shared trained rows (α=0.5 blend) |
| `raw-trained-rows/` | 256 | the α=1 trained rows, for re-sweeping α |
| `prototype-rows/` | 1024 | closed-form rows for the wide label spaces |

## How it works

| piece | where |
|---|---|
| readout at the end of inference: `z = W h + b`, softmax over the K allowed answers, no decode loop | [`qwenjev/readout.py`](qwenjev/readout.py) — `ReservedLabelReadout` / `SlotHeadReadout` / `PointerReadout` |
| probability from the reserved label tokens of the pretrained head (K ≤ 26, zero training) | `ReservedLabelReadout` renormalises the LM head's mass over the option tokens |
| typed questions: finite choices, yes/no, ordered scores | [`qwenjev/schema.py`](qwenjev/schema.py) |
| encode the shared state once, isolate the branches | [`qwenjev/engine.py`](qwenjev/engine.py): one prefill into a KV/recurrent cache, expanded per branch, each branch seeing the state plus only its own suffix |
| branch scheduling as a batch (size + token budget), one forward per batch | `_chunk_branches`; `share_state=False` is the reference path that re-encodes the state per branch, kept as a differential test |
| request limits and billing shape (32,768 per branch, 65,536 per request, state counted once) | `JevLimits`, `check_limits`, `account` |
| options are read as one ordered list inside the branch; the readout reads the decision position after it | [`qwenjev/prompt.py`](qwenjev/prompt.py), [`qwenjev/tokenize.py`](qwenjev/tokenize.py) |
| RLCD: train the distribution from outcomes with a proper scoring rule (backbone frozen) | [`qwenjev/rlcd.py`](qwenjev/rlcd.py) |
| calibration and confidence: reliability bins, ECE, Brier, Wilson intervals; `confidence` is arithmetic | [`qwenjev/calibration.py`](qwenjev/calibration.py), [`qwenjev/confidence.py`](qwenjev/confidence.py) |
| two backends behind one interface (this engine / Laya), one evaluation code path | [`qwenjev/backends.py`](qwenjev/backends.py) |

## RLCD in one page

**RLCD** (Reinforcement Learning for Calibrated Decisions) trains the model **from
outcomes instead of from a fixed label set**. The idea is one sentence: *a decision is a
distribution over the answers the question allows, so train that distribution with a rule
that is only minimised when the distribution is the truth.*

**The readout.** For a state `h ∈ R^d` (the hidden state at the decision position of a
branch) and the `K` answers the question allows, the model produces one score per answer
and normalises:

```
z_k = w_k · h + b_k                     (k = 1 … K)
p_k = softmax(z / τ)_k = exp(z_k/τ) / Σ_j exp(z_j/τ)
```

`w_k` is a row of the readout — one row per *allowed answer*, not per position — and `τ`
is a temperature (1.0 unless a calibration split fits something else).

**The objective.** Let `y` be the answer that actually happened. RLCD minimises a *proper
scoring rule* of the predicted distribution:

```
log loss :  L = −log p_y = −z_y/τ + log Σ_j exp(z_j/τ)
Brier    :  L = Σ_k (p_k − 1{y = k})²
```

Both are strictly proper: their expected value is minimised **exactly** when
`p = P(y | state, question)`, so the trained numbers are probabilities and not just
scores. The gradient is the intuitive one,

```
∂L/∂z_k = p_k − 1{y = k}      (log loss, τ = 1)
```

i.e. every observation pushes the answer that happened up and the others down in
proportion to how much probability they claimed.

**Why "calibrated" is the point.** A classifier that only has to be argmax-correct can be
arbitrarily over-confident; here over-confidence is paid for directly by the loss, and the
result is checked with ECE. On the benchmark the trained readout reaches **0.125** mean ECE
against 0.158 (untrained readout) and 0.289 (Laya), while *also* improving accuracy.

**What makes it "reinforcement"-flavoured rather than plain supervised learning** is what
it consumes and what it can learn: the supervision is a *decision plus its outcome*
(`state → question → answer → happened`), the parameters being fitted are the answers'
rows, and a label space that was never trained still has usable rows (the reserved ones),
so the same objective extends a deployed model to new answer sets instead of requiring a
new head. Our implementation is the **supervised form** of that objective — the backbone
is frozen and the readout is fitted with log loss or Brier (`train.py --objective
log_loss|brier`); the same loss works with backbone gradients if the memory is there.

**Two structural details** that the measurements forced on us, both of which are part of
the training recipe:

* **one private row block per `(question, label space, answer)`.** With rows shared by
  position, training a 2–4-option task overwrote the rows a 10–15-option task reads
  (`clinc150_top15` 0.893 → 0.733 with the shared head, against 0.760 on the same task
  now that the choices are separate). Answers the head never saw keep their pretrained
  row, so an unseen label space keeps the zero-shot behaviour exactly.
* **rows are held at the pretrained logit scale** (`‖w‖ ≈ 0.74`) after every step. The
  decision states have norm ≈ 157, so without that projection a single step at lr 1e-3
  moves a fresh row a fifth of its useful length and the logits explode (measured training
  loss 33 against 5.01 for a uniform head). The shipped shared rows are additionally
  blended halfway back to their initialisation (`W(0.5) = W₀ + 0.5·(W_trained − W₀)`) and
  the label spaces wider than the reserved rows use the closed-form prototype rows
  described above, because 150-way cross-entropy over ~20 samples per class does not move
  in one epoch of gradient steps.

## Compared with a BERT-base classifier

The usual way to turn a transformer into a classifier is "BERT-base + one linear head".
This project makes different choices on purpose:

| | BERT-base + linear head | **QwenJev-lite** |
|---|---|---|
| backbone | encoder, ~110M parameters, 12 layers | causal transformer, 4.54B, 32 layers (24 linear-attention + 8 full-attention) |
| questions per forward pass | one — the `[CLS]` vector encodes exactly one question | many — the state is encoded once, each question is an isolated branch, all branches run as one batch |
| where the answer comes from | the pooled `[CLS]` state → a head whose class count is fixed when it is built | the state at the decision position of the branch → `K` rows, one per allowed answer, and `K` can differ per request |
| adding an answer set | new head, retrain, new checkpoint per task | the same head answers any label space: reserved rows zero-shot, private rows once trained |
| mixing question types | one classifier per type, one pass per question | one request carries `choice` + `bool` + `score` questions about the same state |
| uncertainty | a softmax that is typically over-confident and never scored | the distribution *is* the trained object, scored by ECE / Brier / NLL |
| cost profile | cheap per pass, one question per pass | one prefill per state (a 4.54B model), then ≈20 ms per extra question on the same state |
| generation | none (classification only) | none, by design — the readout replaces the decode loop |

Short version: a BERT classifier answers *"which of my N classes?"* with a head that must
exist before training; QwenJev-lite answers *"which of these K answers?"* for a question
whose answers are part of the request, keeps many such questions isolated on one shared
state, and reports a calibrated distribution instead of an argmax.

## What we adapted in the structure

1. **Readout instead of a decode loop or a fixed head.** `z = W h + b` over the allowed
   answers, softmax, done — no tokens are generated, so latency does not grow with output
   length.
2. **Reserved label rows first.** The zero-shot readout renormalises the pretrained head's
   mass on the option-label tokens, which gives a working model with no training at all
   (K ≤ 26) and, more importantly, gives the trained head its initialisation and its
   fallback for answers it has never seen.
3. **Shared prefill + isolated branches.** The state is encoded once into a KV/recurrent
   cache; the cache is expanded per branch; every branch attends to the state and its own
   suffix only. Branch isolation is structural here, not a prompting convention.
4. **Branch scheduling as a batch.** Branches are packed into forward passes under a batch
   size and a token budget, so one request answers every question on a state in a handful
   of passes.
5. **Options are rendered listwise.** All options are printed into the branch as one
   ordered list and the readout reads the position *after* the list, so the answer can
   depend on the whole set of candidates rather than on each candidate in isolation.
6. **Typed questions, one record shape.** `choice` / `bool` / `score` share one wire
   format, and the trainer, the evaluator, the CLI, the HTTP API and both backends all
   consume that same record.
7. **One row block per answer, plus a documented fallback.** The readout's rows are keyed
   by `(question id, label space, answer key)`, so two files that list the same options in
   a different order still read the same row, and rows never mix across questions.
8. **Training recipe.** Backbone frozen; RLCD with log loss or Brier; rows projected back
   to the pretrained logit scale; the correction blended halfway for the shared rows;
   closed-form prototype rows for label spaces wider than the reserved budget; an optional
   temperature fitted on a held-out split.

## How the data is organised

* **One canonical record per dataset.** Every source is normalised into the engine's own
  request shape — `{"state": …, "questions": {…}, "targets": {…}}`, one JSON per line —
  so the same file feeds training, evaluation, the CLI and the HTTP API with no
  per-dataset code path.
* **Everything is a typed question.** A task becomes a `choice` (with `criteria`), a
  `bool` (yes/no, `criteria` optional) or an ordered `score`; a boolean question carries
  both an `instructions` phrasing and a `claim` phrasing because the two backends were
  measured on different templates.
* **The label space is data, not code.** The `criteria` keys are the answer identities;
  the readout maps `(question id, label space, answer key)` to a row, which is why option
  order does not matter and why several tasks can share one head without interfering.
* **Split discipline.** Every training row is disjoint from the rows it is scored on; the
  caps are spread evenly across a file rather than taken as a prefix; and the splits this
  repository builds itself (`artifacts/extra_splits.json`) are checked before they are
  written — if the test file reads an answer the training file never trains, the script
  refuses to write it.
* **Budgets per label space.** `--items-per-label` keeps a many-answer task (150 intents)
  from being starved by a two-answer one, which is what flattened the choice family
  before.
* **Precision.** The backbone runs bf16; the readout stays float32 (a training step is
  ~1e-3, below bf16's resolution at that magnitude), and the logits are computed in
  float32.

## Paths and data formats

Every path is configurable, and no absolute path is hard-coded anywhere:

| path | flag | environment | default |
|---|---|---|---|
| backbone | `--model` | `QWENJEV_MODEL` | `./qwen3.5-4B` (or a Hub repo id) |
| data directory | `--data-dir` | — | `data/ready` |
| readout output | `--model-dir` / `--out` | — | `models/qwenjev-multitask-v2` |
| Laya baseline | `--laya-path` | `QWENJEV_LAYA` | `./laya` |

A data directory holds two files per task, `<task>_train.jsonl` and `<task>_test.jsonl`
(any `<task>_<split>.jsonl` works). Each line is one decision record — the engine's request
shape plus the observed outcome:

```json
{"id": "mytask/test/0001", "dataset": "mytask", "split": "test",
 "state": "shared text: encoded once, then every question below reads it",
 "questions": {
   "queue":  {"type": "choice", "instructions": "Which team should handle this ticket?",
              "criteria": {"payments": "Payout failures", "account": "Login", "other": "Else"}},
   "urgent": {"type": "bool", "instructions": "Does this need urgent attention?"},
   "grade":  {"type": "score", "instructions": "How severe is it?",
              "criteria": {"low": "Low", "mid": "Medium", "high": "High"}}},
 "targets": {"queue": "payments", "urgent": "no", "grade": "high"}}
```

Rules that matter: each `targets` value must be one of that question's `criteria` keys;
`choice` and `score` need `criteria` (ordered), `bool` may omit it (defaults to
yes/no); the question id (`queue`, …) is just an identifier — what decides which rows the
readout uses is `(question id, label space, answer key)`.

**Train a readout** (another backbone, your own data, your own output directory):

```bash
python train.py \
  --model "$QWENJEV_MODEL" \
  --data-dir data/ready \
  --model-dir models/my-run \
  --objective log_loss \
  --prototype-init --items-per-label 12 --min-items-per-task 200 \
  --epochs 0                     # 0 = closed-form prototype rows only; use 1 + --lr for gradient steps
```

**Evaluate** (drop `--variants laya` if Laya is not installed):

```bash
python test.py \
  --model "$QWENJEV_MODEL" \
  --data-dir data/ready \
  --model-dir models/my-run \
  --variants qwen_zeroshot qwen_trained \
  --limit 0 --batch 16 \
  --out artifacts/my-results.json
```

**One request, in Python**:

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

`data/ready/` ships with the repository (≈92 MB of JSONL): the formatted train/test sets
plus `manifest.json`, which records the source file, the row counts and the caveats of
every split. The sources behind them:

| source | which tasks |
|---|---|
| MultiNLI / SNLI | `mnli`, `mnli_ood`, `snli` (one premise, several hypotheses per state) |
| Jigsaw Toxic Comment | `jigsaw` (six independent yes/no labels), `jigsaw_severity` (ordered) |
| GoEmotions | `goemotions` (28 yes/no emotions), `goemotions_sentiment` (ordered) |
| TruthfulQA | `truthfulqa` (5-way) |
| IntentGrasp | `intentgrasp` (the options ship with each item) |
| CLINC150 / HWU64 | `clinc150`, `clinc150_top15`, `hwu64`, `hwu64_top15` |
| BEIR (arguana / nfcorpus / scidocs / scifact / trec-covid) | 15 relevance-judgement tasks built from each collection's queries and qrels: the labelled document becomes the positive option, non-qrels the negatives; splits are by query |

`artifacts/extra_splits.json` records the training splits this repository builds itself
(source, row counts, and the check that every answer the test file reads is trained), and
`data/ready/manifest.json` records the provenance of every other split. Thanks are due to
the authors and maintainers of these datasets, annotations and tooling — the citation
list is in [References](#references).

## Speed

**Every number below is the pure-torch fallback.** This machine has no Triton and no CUDA
toolkit, so `flash-linear-attention` and `causal-conv1d` are **not installed**: the 24
linear-attention layers run the torch implementation (`torch 2.6.0+cu126`,
`transformers 5.14`, RTX 3090 / SM86; the 8 full-attention layers already run on SDPA).
**No flash-attention number is claimed anywhere in this repository** - the kernels could
not be installed here, and we would rather report the slow path explicitly than quote a
number we did not measure.

Units matter, so both are spelled out: Laya answers **one question per call** and
re-encodes the state for each one; this engine answers **every question about one state in
one call** (a shared prefill, then the branches), so its per-call cost barely moves with
the number of questions.

| work | ours (torch fallback) | Laya |
|---|---|---|
| the 26-task benchmark, 16 items per call | 2.5 s/call (≈68 decisions) → **36.7 ms/decision** | 23 ms/call → **23.4 ms/decision** |
| 1 question, 15 options, short state | 94 ms | ~25 ms |
| one state, 1 / 8 / 64 questions | 297 / 283 / 1511 ms **per call** | 23 / 193 / 1496 ms per call |
| the same with a 6,000-character state | 461 / 297 / 1655 ms per call | 26 / 191 / 1621 ms per call |

Read the last two rows as cost *per call*: ours is flat (the state is prefilled once and
each extra question costs ≈20 ms), Laya's is linear, so the wall clock meets at roughly 64
questions on one state and stays with us after that; per *decision* we are 1.0–1.6×
slower on this fallback path.

The fused kernels are what the fused linear-attention path needs, and the
"1,500 questions in a few hundred ms" regime belongs to them - so with those kernels the
numbers above would improve, and we do not quote an estimate for it. Why they are missing
here: `causal-conv1d` has no Windows wheel and building it needs `nvcc`, and
`flash-linear-attention` needs Triton (which `torch.compile` needs as well, so that path
is closed too). On Linux, `pip install flash-linear-attention causal-conv1d` enables the
fast path; `scripts/speed_compare.py` reproduces the table above for any state length and
question count, so the same script measures the fast path once it exists.

## Layout

```
qwenjev/           the library: schema / prompt / tokenize / readout / engine / rlcd /
                   calibration / datasets / normalize / relevance / backends / benchmark /
                   probes / api / cli
train.py           train one readout over the train splits of a data directory (progress bar)
test.py            score Laya + both readouts over the test splits (progress bar)
demo.py serve.py   user-facing CLI demo and web demo
scripts/           build_extra_splits / compose_wide_rows / interpolate / dev_score /
                   map_labels / compare_results / speed_compare / split_csv (and a few
                   converters for rebuilding the canonical files from other mirrors)
models/            the shipped readout + the three heads it is composed from
data/ready/        formatted train/test JSONL + manifest.json (in git, ≈92 MB)
data/raw/          the source datasets as provided (git-ignored)
artifacts/         RESULTS_CN.md (final numbers), results.json (raw evaluation),
                   extra_splits.json (provenance of the built splits)
tests/             92 tests + the mini fixtures the dataset adapters are tested against
```

## References

<!-- To be filled in by the author: the works, models, datasets and tools this project
     builds on and thanks. -->
