<div align="center">

# QwenJev-Lite

**A Jev-style decision model on a local Qwen3.5-4B: one shared state, isolated question branches, parallel probability readouts — no text is ever generated.**

![python](https://img.shields.io/badge/python-3.11-3776ab?logo=python&logoColor=white)
![torch](https://img.shields.io/badge/torch-2.6.0%2Bcu126-ee4c2c?logo=pytorch&logoColor=white)
![transformers](https://img.shields.io/badge/transformers-5.14-ffd21e)
![tests](https://img.shields.io/badge/tests-92%20passing-2ea44f)
![license](https://img.shields.io/badge/license-MIT-blue)

[Quickstart](#quickstart) · [Results](#results) · [The shipped readout](#the-shipped-readout) · [How it works](#how-it-works) · [Datasets](#datasets) · [Speed](#speed) · [Layout](#layout)

</div>

> **中文速览**：把 archerhume《Jev's Architecture Unmasked》里的三件事——**共享状态只编码一次**、
> **问题分支互相隔离**、**末端直接输出概率分布（不生成文本）**——在本地 Qwen3.5-4B 上复现，读出用 RLCD
> 按结果训练。交付模型在 28 个测试集上按题型平均 **bool 0.794 / choice 0.728 / score 0.771，总体 0.755**
> （Laya 0.446、零样本读出 0.722），平均 ECE 0.138。三个读出文件就能复现交付模型（**逐位一致**），
> 见 [`models/README.md`](models/README.md)；一条命令看演示：
> `python demo.py --model-dir models/qwenjev-multitask-v2`。

## Results

One full run of `test.py --limit 0` over the 28 formatted test splits
([raw](artifacts/dataset_results_v9.json) · [tables](artifacts/DATASETS_V9.md) · [notes](artifacts/REPORT_V9_CN.md)):

| question type | Laya | pretrained readout | **trained readout** |
|---|---|---|---|
| judgement (`bool`, 7 tasks) | 0.520 | 0.717 | **0.794** |
| choice (14 tasks) | 0.502 | 0.749 (12 tasks) | **0.728** (14 tasks) |
| choice, the 12 tasks both readouts can answer | 0.511 | 0.749 | **0.760** |
| score (7 tasks) | 0.258 | 0.681 | **0.771** |
| **overall (28 tasks)** | **0.446** | 0.722 (26 tasks) | **0.755** |

*Calibration*: mean ECE 0.291 (Laya) / 0.158 (pretrained) / **0.138** (trained).
*The two label spaces wider than the 26 reserved rows* are where training is the only
option — the pretrained readout cannot answer them at all:

| label space | **trained** | Laya |
|---|---|---|
| CLINC150 (150-way intent) | **0.527** | 0.545 |
| HWU64 (64-way intent) | **0.540** | 0.345 |

Where the profit comes from, and where it does not: the trained head wins on relevance
judgements (`scifact_rel_choice` 1.000, `trec-covid_rel_choice` 0.900, `arguana_rel_bool`
0.803) and on the wide intent spaces, and it stays below the pretrained readout on the
two 15-intent subsets (0.733 / 0.467 against 0.893 / 0.613) and on closed-book FEVER-like
tasks. The Chinese reports list every experiment that was tried and rejected.

## Quickstart

```bash
pip install -r requirements.txt                     # torch 2.6 + transformers 5.14, CPU is enough for tests

python demo.py --model-dir models/qwenjev-multitask-v2   # the essay's opening example, all three question types
python serve.py                                          # the same thing as a web page on :8300
pytest -q                                                # 92 tests on a tiny backbone: no GPU, no checkpoint, no data
```

| what you want | command |
|---|---|
| the full benchmark (Laya + both readouts, progress bars) | `python test.py --variants laya qwen_zeroshot qwen_trained --limit 0` |
| retrain the readout from `data/ready` | `python train.py --prototype-init --items-per-label 12 --min-items-per-task 200` |
| rebuild the shipped head from the other three | [`models/README.md`](models/README.md#rebuild-the-shipped-head) |
| raw datasets → one canonical format | `python -m qwenjev.cli normalize --src data/raw --out data/ready` |
| one HTTP request, several questions on one state | `python -m qwenjev.cli serve --port 8300` then `POST /v1/decide` |
| the essay's probes (visibility, option interaction, latency…) | `python -m qwenjev.cli probe all --out artifacts/probes.json` |

The backbone path defaults to `C:\qwen3.5-4B`; every entry point takes `--model` or
`QWENJEV_MODEL` style overrides, and `--fake` swaps in a tiny random Qwen3.5 so the whole
pipeline runs on CPU in seconds.

## The shipped readout

`models/qwenjev-multitask-v2/readout.pt` is the delivered head: **512 rows = 27 reserved
+ 485 private**, covering 90 label spaces in `data/ready`.

* rows 0–26 stay the **pretrained label rows**, so a question the head has never seen
  reads what the backbone itself would say (that is what keeps `demo.py` sensible on a
  brand-new question);
* every label space the training mix covers gets its own block — rows copied from the
  trained head by position for label spaces the reserved rows cover, and closed-form
  **prototype rows** (the mean state of each answer) for label spaces wider than 26
  options, where a positional row simply runs out;
* an answer in neither case still falls back to its reserved row.

The three source checkpoints ([`models/shared-rows`](models/README.md),
[`models/raw-trained-rows`](models/README.md),
[`models/prototype-rows`](models/README.md)) are in the repo too, and composing them
reproduces the shipped file **bit for bit** (`torch.equal`, checked). That is the whole
reason the checkpoints are committed: ~16 MB buys a self-contained, verifiable artifact.

## How it works

| essay | code |
|---|---|
| §1 end inference with a readout: `z = W h + b`, softmax over the K allowed answers, no decode loop | [`qwenjev/readout.py`](qwenjev/readout.py) — `ReservedLabelReadout`, `SlotHeadReadout`, `PointerReadout` |
| §1 the same computation from reserved vocabulary rows | `ReservedLabelReadout` reads the mass the LM head puts on the option-label tokens |
| §1 typed questions: finite choices, yes/no, ordered scores | [`qwenjev/schema.py`](qwenjev/schema.py) |
| §2 share the state, isolate the questions | [`qwenjev/engine.py`](qwenjev/engine.py) — one prefill, the KV/recurrent cache expanded per branch, each branch attending only to the state plus its own suffix |
| §2 prefix cache with separate causal suffixes | `_expand_prefix` + a right-padded branch batch; `share_state=False` is the reference path that recomputes the state per branch (used as a differential test) |
| §2 32,768 tokens per branch, 65,536 per request, state counted once | `JevLimits`, `check_limits`, `account` |
| §4 let the options interact before choosing | options are rendered as one list; the readout reads the decision position *after* the whole list |
| §5 RLCD: train the distribution with a proper scoring rule | [`qwenjev/rlcd.py`](qwenjev/rlcd.py) — `log_loss` / `brier_loss`, readout trained, backbone frozen |
| §5 calibration: reliability bins, ECE, Wilson intervals; `confidence` is arithmetic | [`qwenjev/calibration.py`](qwenjev/calibration.py), [`qwenjev/confidence.py`](qwenjev/confidence.py) |
| §7 schedule branches as a batch, not a conversation | `_chunk_branches` (batch size + token budget), one forward per batch |

<details>
<summary><b>Faithfulness notes and known deviations</b> (click)</summary>

* **Backbone frozen during RLCD.** The essay suggests the backbone is adapted; we train
  the readout only (the cheap faithful version, and it keeps a full run under half an
  hour). `RLCDFineTune` uses ordinary autograd, so unfreezing is a one-line change.
* **`score_confidence` is a reconstruction.** The essay publishes the exact `Choice`
  formula and says `Score` "uses a different formula reflecting distance from the modal
  level" without printing it.
* **Digit labels.** Jev's tokenizer splits digits one at a time, Qwen's does too
  (`"10"` → `["1", "0"]`), which is why the reserved-label readout uses letters.
* **Determinism.** The local engine returns bit-identical distributions for identical
  payloads, so the essay's request-noise observations have no counterpart here; the
  option-interaction probe prints a duplicate-request control to make that explicit.
* **Reference card.** The essay's card-in-the-option-list result does not reproduce
  (our readout ignores a card inside the option list). The essay itself flags that
  experiment as order sensitivity rather than a recovered attention mask, so we report
  the difference instead of tuning the prompt until it agrees.

The essays' experiments, with both our numbers and the essay's, are in
[`artifacts/REPORT.md`](artifacts/REPORT.md) and `artifacts/probes.json`.
</details>

## Datasets

One canonical record for every source — the engine's own request shape — so the trainer,
the evaluator and both backends work everywhere without special cases:

```json
{"id": "jigsaw/test/0001b41b…", "dataset": "jigsaw", "split": "test",
 "state": "the shared text",
 "questions": {"toxic": {"type": "bool",
                         "instructions": "Does this comment contain toxic content?",
                         "claim": "This comment contains toxic content.",
                         "criteria": {"yes": "Yes", "no": "No"}}},
 "targets": {"toxic": "no"}, "meta": {"source_file": "train.csv", "row": 5}}
```

Boolean questions carry both phrasings because the two backends were measured on
different templates: our engine asks a question in the branch, Laya's `noul` head wants a
claim at the head of the state.

| source | tasks (test splits) | note |
|---|---|---|
| MNLI / SNLI | `mnli`, `mnli_ood`, `snli` | one premise, ~3 hypotheses per state: the architecture's showcase |
| Jigsaw | `jigsaw`, `jigsaw_severity` | 6 independent yes/no labels per comment; test = 20% hold-out of `train.csv` |
| GoEmotions | `goemotions`, `goemotions_sentiment` | 28 yes/no emotions per comment |
| TruthfulQA | `truthfulqa` | 5-way MC reading; 30% deterministic hold-out (the release has no split) |
| IntentGrasp | `intentgrasp` | options ship with each item |
| CLINC150 / HWU64 | `clinc150`, `clinc150_top15`, `hwu64`, `hwu64_top15` | 150/64 intents plus closed 15-intent subsets; `intents.txt` recovered by clustering |
| BEIR (arguana, nfcorpus, scidocs, scifact, trec-covid) | 15 tasks | built from query + qrels: the labelled document is the positive option, non-qrels are the negatives; splits are by query |

<details>
<summary><b>What was dropped, and why</b> (click)</summary>

Three sources cannot support the decision this architecture is for, so they were removed
on 30 September 2026 rather than reported with a footnote. `data/raw_backup/` keeps a
byte-for-byte copy of `data/raw/` from before the removal (0.70 GB, git-ignored).

| dropped | why | restore with |
|---|---|---|
| `fever`, `fever_support` | the supplied claim files carry no wiki sentences, so the shared state is the bare claim: that measures closed-book fact checking (0.44), not evidence-based verification | the claim files **plus** an `evidence.jsonl` built from the wiki dump (`scripts/fever_evidence.py`) |
| `mmlu_pro` | 70 labelled validation rows; the 10 rows of the head could only be trained from the test file's own unused rows | the Berkeley MMLU tarball (`scripts/mmlu_to_jsonl.py`) |
| `banking77`, `banking77_top15` | a 770-row TF-IDF *testset* and no training split at all | the official `train.csv` from PolyAI/BANKING77 |

Two sampling details worth knowing, because both bit us:

* caps are **spread evenly over the whole file**, not taken as a prefix: CLINC150/HWU64's
  test sets are label-sorted, so a prefix of them covered 81 / 41 of 150 / 64 classes;
* `clinc150` and `hwu64` ship integer label ids with no names; `intents.txt` is recovered
  by clustering the labelled examples (`scripts/map_labels.py`).

</details>

## Speed

Units matter, so both are spelled out. Laya answers **one question per call** and
re-encodes the state for each one; our engine answers **every question about one state in
one call** (a shared prefill, then the branches), so its per-call cost barely moves with
the number of questions. Measured here (RTX 3090, torch 2.6+cu126, no fused kernels):

| work | ours | Laya |
|---|---|---|
| the 28-task benchmark, 23,145 decisions, 16 items per call | 14.1 min -> **36.7 ms/decision**, 2.5 s/call (about 68 decisions) | 9.0 min -> **23.4 ms/decision** (1 per call) |
| 1 question, 15 options, short state | 94 ms | ~25 ms |
| one state, 1 / 8 / 64 questions | 297 / 283 / 1511 ms **per call** | 23 / 193 / 1496 ms per call |
| the same with a 6,000-character state | 461 / 297 / 1655 ms per call | 26 / 191 / 1621 ms per call |

Read the last two rows as cost *per call*: ours is flat - the state is prefilled once and
each extra question costs about 20 ms - while Laya's is linear. The wall clock meets at
roughly 64 questions on one state and stays with us after that; per *decision* we are
1.0-1.6x slower here. The fused kernels are what the essay's "1,500 questions in a few
hundred ms" needs, and they are **not installable on this machine**: `causal-conv1d` has
no Windows wheel and needs `nvcc`, `flash-linear-attention` needs Triton (which
`torch.compile` wants too), and the Hub-kernel route is pinned to SM121. All four attempts
are logged in [`artifacts/REPORT_V9_CN.md`](artifacts/REPORT_V9_CN.md); on Linux,
`pip install flash-linear-attention causal-conv1d` lights up the fast path.
`scripts/speed_compare.py` reproduces the table for any state length and question count.

One fix landed while measuring this: `decide_batch` used to carry the state into every
branch, so a long state with many questions cost 12x more than it had to (2.6 s -> 0.3 s
with 8 questions, 20.4 s -> 1.7 s with 64, on a 6,000-character state). A request whose
`state tokens x branches` exceeds `QwenJevConfig.state_reuse_tokens` (800) now takes the
cached per-state path; short states keep the row-batched one, which is 5-27x faster there
because it pays the per-call overhead once for 16 items.

## Layout

```
qwenjev/            the library: schema, prompt, tokenize, readout, engine, rlcd,
                    calibration, datasets, normalize, backends, benchmark, probes, api, cli
train.py            train one readout over every split in data/ready        (progress bar)
test.py             score Laya + both readouts on every split               (progress bar)
demo.py serve.py    user-facing CLI demo and web demo
scripts/            build_extra_splits, compose_wide_rows, interpolate, dev_score,
                    map_labels, mmlu_to_jsonl, fever_evidence, split_csv, compare_results
models/             the shipped readout + the three heads it is composed from
data/ready/         formatted train/test JSONL + manifest.json   (in git, 92 MB)
data/raw/           the seven source datasets as provided        (git-ignored; backup in data/raw_backup)
artifacts/          every result file and the four Chinese reports (REPORT_V5…V9_CN.md)
tests/              92 tests + fixtures for the dataset adapters
```

## Reference

Archer Hume, *Jev's Architecture Unmasked*, 17 September 2026. Section numbers above
refer to that essay. Experiments ran on a desktop RTX 3090 in September 2026; every
number quoted here comes from a recorded run in `artifacts/`. MIT licensed.
