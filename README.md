# QwenJev-Lite

QwenJev-lite 是对 archerhume《Jev's Architecture Unmasked》(2026-09-17) 中重建的 Jev 架构的一次复现，
底座模型是 `C:\qwen3.5-4B`（Qwen3.5-4B，32 层混合 linear/full attention，4.54B 参数）。

复现的六个部件：**共享状态编码一次**、**问题分支互相隔离**、**并行概率读出（不生成文本）**、
**listwise 选项交互**、**按结果训练分布（RLCD）**、**分支批调度**。

关键结果（真实 4B 模型，RTX 3090，见 [`artifacts/REPORT.md`](artifacts/REPORT.md)）：

| 论文中的观察 | 本实现 |
|---|---|
| 兄弟问题的秘密对探针问题概率为 0.00，放进 state 后升到 0.90–0.92 | 0.203 → 0.203（**完全无泄漏**），放进 state 后 0.740 |
| 参考卡放进共享 state 时 48/48 全对，均值 p≈0.88 | 12/12 全对，均值 p=0.891 |
| 加一个无关选项把 log-odds 降低 0.28，10/10 block 下降 | 降低 0.309，10/10 block 下降，重复请求对照组为 0.000 |
| `output_tokens` = 4 + 15×回答数 + 问题标识符长度 | 逐项吻合（`"q"`→20，`"escalate"` 等）|
| 1500 个问题仍能在几百毫秒内返回 | 16 个问题 303 ms，而拆成 16 次独立请求要 4290 ms（**14.2×**）|
| RLCD：用 proper scoring rule 训练分布 | 准确率 0.870→0.938，ECE 0.094→0.027，Brier 0.230→0.093 |

数据集分两层：`data/raw/` 是原始数据，`data/ready/` 是统一格式的训练/测试集
（9 个数据集的 13 个任务 + 从 query/文档集合构建的 18 个相关性判别任务，
三种题型 choice / bool / score 都有）。训练脚本 `python train.py`、测试脚本
`python test.py`（都带进度条），跑完把 Laya 和 QwenJev-lite 并排对比。
上一轮的结果 + 速度见 [Results](#the-nine-datasets-laya-vs-qwenjev-lite)。

---

## What this is

The essay reverse-engineered a commercial decision model from its API and published a
reconstruction: a causal transformer that encodes a shared state once, evaluates
isolated question branches in parallel, and ends inference with a **readout** that maps
representations directly to probabilities instead of generating text.

QwenJev-lite implements that reconstruction on top of a local Qwen3.5-4B checkpoint.
It is a *lite* reproduction in the sense that matters for honesty:

* the **interface, the computation graph and the training objective** follow the essay;
* the **serving performance** does not — the essay's numbers come from a production
  stack, ours come from a desktop RTX 3090 without the fused linear-attention kernels
  (see [Performance](#performance));
* the **weights** are Qwen3.5-4B plus a trained readout, not TypeSafe's model, so the
  behavioural numbers (accuracy, calibration, token counts) differ in absolute terms.
  Every probe prints both what we measured and what the essay reported.

## Architecture to code

| Essay | Implementation |
|---|---|
| §1 "End inference with a readout" — `z = W h + b`, softmax over `K`, no decode loop | [`qwenjev/readout.py`](qwenjev/readout.py): `ReservedLabelReadout`, `SlotHeadReadout`, `PointerReadout` |
| §1 the same computation from reserved vocabulary rows | `ReservedLabelReadout` reads the mass the LM head puts on the option-label tokens and renormalises over the allowed slots |
| §1 typed questions: finite choices, yes/no, ordered scores | [`qwenjev/schema.py`](qwenjev/schema.py): `ChoiceQuestion`, `BoolQuestion`, `ScoreQuestion` (+ the documented `noul` alias) |
| §2 "Share the state, isolate the questions" | [`qwenjev/engine.py`](qwenjev/engine.py): the state is prefilled once into a KV/recurrent cache, the cache is expanded per branch, and each branch attends only to the state and its own suffix |
| §2 prefix KV cache with separate causal suffixes (Hydragen/DeFT) | `QwenJevLite._expand_prefix` + a right-padded branch batch; `share_state=False` runs the reference path where each row carries its own copy |
| §2 per-branch 32,768 / per-request 65,536 token limits, state counted once | `JevLimits`, `QwenJevLite.check_limits`, `QwenJevLite.account` |
| §3 causal backbone with broad knowledge | frozen Qwen3.5-4B (`Qwen3_5ForConditionalGeneration`), hybrid linear/full attention |
| §4 "Let the options interact before choosing" | options are rendered as an ordered list in one branch; the readout sees the decision position *after* the whole list; `PointerReadout` scores each option's own hidden state |
| §5 RLCD — train the distribution with a proper scoring rule | [`qwenjev/rlcd.py`](qwenjev/rlcd.py): `log_loss`, `brier_loss`, `RLCDFineTune`, `fit_temperature` |
| §5 calibration: reliability bins, ECE, Wilson intervals | [`qwenjev/calibration.py`](qwenjev/calibration.py) |
| §5 the `confidence` field is arithmetic, not learned: `c = (pmax − 1/K)/(1 − 1/K)` | [`qwenjev/confidence.py`](qwenjev/confidence.py) |
| §7 "Schedule branches as a batch, not a conversation" | `QwenJevLite._chunk_branches` (batch size + token budget), one forward per batch |
| Methods: `output_tokens` = 4 + 15×answers + identifier length | `confidence.billing_output_tokens` |
| The essay's experiments | [`qwenjev/probes.py`](qwenjev/probes.py) — visibility, reference card, option interaction, option order, fake options, accounting, latency |
| Training and testing on public benchmarks | [`qwenjev/datasets.py`](qwenjev/datasets.py) + [`qwenjev/evaluation.py`](qwenjev/evaluation.py) — see [Datasets](#datasets) |
| One canonical record for every dataset | [`qwenjev/normalize.py`](qwenjev/normalize.py) — `qwenjev normalize` |
| Laya and our engine behind one interface | [`qwenjev/backends.py`](qwenjev/backends.py) |
| Accuracy, calibration and speed per task | [`qwenjev/benchmark.py`](qwenjev/benchmark.py) — `qwenjev benchmark` |

## Quick start

```bash
# 0) 面向用户的 demo（论文开场例子：三种题型一次前向答完）
python demo.py                     # 命令行
python serve.py                    # 网页：http://127.0.0.1:8300

# 1) train one readout on every formatted train split (progress bar)
python train.py --no-balance --epochs 1 --max-samples 6000 --lr 5e-4 --anchor 1e-2

# 2) score Laya, the pretrained readout and the trained readout on every test split
python test.py                     # -> artifacts/DATASETS_V2.md

# the essay's opening example, end to end
python -m qwenjev.cli demo

# a single shared state, several questions, one batched forward
python -m qwenjev.cli --readout slot_head demo --state "My payouts have failed three times." \
  --questions '{"queue": {"type": "choice", "instructions": "Which team should handle this ticket?", "criteria": {"payments": "Payouts", "account": "Login", "other": "Else"}}}'

# the essay's experiments (add --quick for a fast subset)
python -m qwenjev.cli probe all --out artifacts/probes.json

# RLCD: train the readout against outcomes, then measure calibration
python -m qwenjev.cli train --out artifacts/readout.pt --report artifacts/train.json

# everything, plus artifacts/REPORT.md
python -m qwenjev.cli reproduce

# the HTTP surface
python -m qwenjev.cli serve --port 8300
curl -s localhost:8300/v1/decide -H 'content-type: application/json' -d '{
  "state": "My payouts have failed three times. The bank says everything is fine.",
  "questions": {
    "queue": {"type": "choice", "instructions": "Which team should handle this ticket?",
              "criteria": {"payments": "Payout failures", "account": "Login", "other": "Else"}},
    "escalate": {"type": "bool", "instructions": "Does this require urgent human attention?"}}}'
```

`--fake` swaps in a tiny randomly initialised Qwen3.5 backbone (~0.3M parameters) so
the whole pipeline runs on CPU in seconds; that is how the test suite exercises it.

## Datasets

The nine datasets in `data/raw/` arrive in five different shapes (parquet, JSONL, CSV,
TSV, no header, three different label conventions). `qwenjev normalize`
turns all of them into one canonical record — the engine's own request shape — so the
trainer, the evaluator and both backends work on every task without special cases:

```json
{"id": "jigsaw/test/0001b41b…", "dataset": "jigsaw", "split": "test",
 "state": "the shared text",
 "questions": {"toxic": {"type": "bool",
                         "instructions": "Does this comment contain toxic content?",   // Qwen branch
                         "claim": "This comment contains toxic content.",              // Laya noul head
                         "criteria": {"yes": "Yes", "no": "No"}}},
 "targets": {"toxic": "no"}, "meta": {"source_file": "train.csv", "row": 5}}
```

Boolean questions carry both phrasings because the two backends were measured on
different templates: our engine asks a question in the branch, Laya's `noul` head wants
a claim at the head of the state.

| task | shared state | decision | n (train/test) |
|---|---|---|---|
| `mnli`, `mnli_ood`, `snli` | premise | 3-way entailment per hypothesis (≈3 questions per state) | 2000 / 400 |
| `fever` | claim | supports / refutes / not enough information | 2000 / 400 |
| `jigsaw` | comment | six independent yes/no labels | 2000 / 400 |
| `goemotions` | comment | 28 independent yes/no emotions | 2000 / 400 |
| `truthfulqa` | question | 5-way: the best answer against four false ones | 577 / 240 |
| `mmlu_pro` | category + question stem | 10-way choice | 70 / 400 |
| `intentgrasp` | user utterance | choice over that item's own option list | 2000 / 400 |
| `clinc150`, `hwu64` | utterance | 150-way / 64-way intent | 2000 / 400 |

The formatted files live in `data/ready/`; `data/ready/manifest.json` records the source
file, counts and caveats for every split.

```bash
python -m qwenjev.cli normalize                       # data/raw -> data/ready/*.jsonl
python -m qwenjev.cli normalize --tasks jigsaw snli --test-limit 100
```

Two details worth knowing, because they bit us:

* caps are taken with an even stride across the whole file, not as a head slice —
  CLINC150 and HWU64 are label-sorted and IntentGrasp is grouped by corpus, so `[:400]`
  is a single class or a single corpus;
* `clinc150` and `hwu64` ship integer label ids with no names. Both are normalised so
  the pipeline runs, but the options read `label_000…label_149`, which no zero-shot
  model can answer. Drop `data/raw/CLINC150/intents.txt` (one name per line, in label-id
  order) and re-run `normalize` to fix it — until then their numbers are not meaningful.

### Backends and the benchmark

Both models implement the same interface, so they are scored by the same code:

```bash
# one shared multi-task readout trained on every task's train split
python -m qwenjev.cli train --tasks mnli snli fever jigsaw goemotions truthfulqa \
       mmlu_pro intentgrasp clinc150 hwu64 --decisions-per-task 500 --epochs 2 \
       --out artifacts/readout_multitask.pt --report artifacts/train_multitask.json

# Laya (C:\laya) vs our zero-shot readout vs our trained readout, all tasks
python -m qwenjev.cli benchmark \
  --variants laya:laya qwen_zeroshot:qwenjev@reserved_label \
             qwen_rlcd:qwenjev@slot_head:artifacts/readout_multitask.pt \
  --test-limit 100 --out artifacts/dataset_results.json --markdown artifacts/DATASETS.md
```

`benchmark` writes accuracy, balanced accuracy, ECE, Brier, NLL, a majority-class and
uniform-chance baseline, and the speed split (requests, ms/request, tokens/request).
See [`artifacts/DATASETS.md`](artifacts/DATASETS.md) for the full tables.

Check what is on disk and what is still missing — the status command prints the exact
download line for every gap:

```bash
python -m qwenjev.cli datasets
python -m qwenjev.cli datasets --json
```

### Getting the data

**This machine cannot reach the network** (`huggingface.co`, `raw.githubusercontent.com`
and Kaggle all fail from the sandbox), so nothing is bundled. Pick whichever column of
commands you have access to and drop the result into `data/<dataset>/`.

| dataset | HuggingFace (no auth) | original source | files the loader wants |
|---|---|---|---|
| FEVER | `python -c "from datasets import load_dataset; load_dataset('fever','v1.0').save_to_disk('data/fever')"` | `curl -L -o data/fever/paper_dev.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/paper_dev.jsonl` (same for `train.jsonl`, `paper_test.jsonl`) | `paper_dev.jsonl` etc. + optional `evidence.jsonl` |
| MMLU | `load_dataset('cais/mmlu','all').save_to_disk('data/mmlu')` | `curl -L https://people.eecs.berkeley.edu/~hendrycks/data.tar \| tar xf -` then `python scripts/mmlu_to_jsonl.py --src mmlu_raw --out data/mmlu` | `test.jsonl` or an HF dump |
| MMLU-Pro | `load_dataset('TIGER-Lab/MMLU-Pro').save_to_disk('data/mmlu_pro')` | `git clone https://github.com/TIGER-AI-Lab/MMLU-Pro` (10 options per item) | `test.jsonl` / `*.parquet` |
| CLINC150 | `load_dataset('clinc_oos','plus').save_to_disk('data/clinc150')` (configs: `small`, `plus`, `imbalanced`) | `git clone https://github.com/clinc/oos-eval` → `data/data_full.json`, `data/data_small.json` | `data_full.json` / `data_small.json` |
| HWU64 | no canonical Hub id — use GitHub | `git clone https://github.com/xliuhw/NLU-Evaluation-Data` → `dataset/NLU-Evaluation-Data-Canonical-Form.csv` | `train.csv`, `test.csv` |
| Jigsaw | `load_dataset('google/jigsaw_toxicity_pred', data_dir='jigsaw-toxic-comment-classification-challenge').save_to_disk('data/jigsaw')` (or the ungated mirror `thesofakillers/jigsaw-toxic-comment-classification-challenge`) | Kaggle: `kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -d data/jigsaw` (needs `~/.kaggle/kaggle.json`) | `train.csv`, `test.csv`, `test_labels.csv` |

Every loader also accepts JSONL / JSON / CSV / TSV / Parquet with several column-name
spellings, an HF `save_to_disk` directory, or `$QWENJEV_DATA` pointing elsewhere, so a
mirror can be dropped in unchanged. The full expected layout is in
[`data/README.md`](data/README.md).

Three conversions have no ready-made file, so they ship as scripts:

```bash
# Berkeley MMLU tarball -> data/mmlu/test.jsonl + val.jsonl
python scripts/mmlu_to_jsonl.py --src mmlu_raw --out data/mmlu

# FEVER: attach wiki sentences so the state is evidence, not the bare claim
python scripts/fever_evidence.py --claims data/fever/paper_dev.jsonl \
       --wiki-pages data/fever/wiki-pages --out data/fever/evidence.jsonl

# a single CSV with no split column (HWU64's canonical form, a Jigsaw export)
python scripts/split_csv.py --src NLU-Evaluation-Data-Canonical-Form.csv \
       --out data/hwu64 --text-column answer --ratios 0.8 0.1 0.1
```

### Single-task evaluation

```bash
# zero-shot, no training, with the pretrained backbone
python -m qwenjev.cli eval --dataset snli --limit 200
python -m qwenjev.cli eval --dataset mmlu --limit 200
python -m qwenjev.cli eval --dataset jigsaw --limit 200 --backend laya
python -m qwenjev.cli --readout slot_head eval --dataset goemotions \
       --checkpoint artifacts/readout_multitask.pt --limit 200 --no-share-state
```

`eval` writes per-question reliability — accuracy, balanced accuracy, mean top
probability, ECE, Brier, NLL, 95% Wilson interval, the reliability bins and a few
examples — so a dataset can be read exactly like the essay's Figure 5.
`--no-share-state` reruns the same data with one state encoding per branch, and
`--progress` prints progress for long runs.

Cost note: the pure-torch linear-attention fallback costs ~0.3-0.7 s per request, so
`--limit` is your friend.

### Caveats worth knowing before you trust a number

* **FEVER** without `evidence.jsonl` is closed-book fact checking (state = claim), which
  is a much harder and less meaningful task than the +evidence one the architecture was
  designed for. `meta["has_evidence"]` tells you which mode an item used.
* **CLINC150 / HWU64** have more labels than the 26 reserved letters, so the zero-shot
  readout cannot answer them (the engine raises instead of truncating the label space),
  and their label names are missing, so nothing can answer them well yet.
* **Jigsaw / GoEmotions** are heavily imbalanced (majority class 96% / 96%). Accuracy
  and ECE look excellent for a model that always answers "no"; read the balanced
  accuracy column.
* **MMLU-Pro** here is the released test split with its answers; the essay's 84.6%
  comes from a different, undisclosed base model.

## Results

### The trained readout beats both the pretrained readout and Laya

Full test set (36 tasks / 6,865 items / 24,487 decisions), one row per question family,
averaged over the 33 tasks all three variants can answer:

| family | Laya | pretrained readout | **trained readout** |
|---|---|---|---|
| judgement (bool, 8 tasks) | 0.503 | 0.702 | **0.803** |
| choice (16 tasks) | 0.457 | 0.663 | **0.669** |
| score (9 tasks) | 0.270 | 0.661 | **0.734** |
| **overall (33 tasks)** | **0.417** | 0.672 | **0.719** |

Speed over the whole benchmark: Laya 23.5 ms/request (1 decision per request, 10.1 min),
the trained readout 138.4 ms/request (3.53 decisions per request, 16.8 min).

How it was reached, in one line each:

* class-balanced sampling destroyed the prior (`P(yes)=0.78` on a task whose true rate is
  `0.043`) — training without it recovered 0.503 → 0.630;
* the remaining gap was structural: one linear `W h + b` readout has to serve every task,
  and training on some label spaces overwrote the reserved rows that the *unseen* label
  spaces rely on (the 15-intent tasks fell 0.89 → 0.33);
* so the head is trained with an L2 anchor to its initialisation
  (`train.py --anchor 1e-2`) and the final matrix keeps half of the trained correction,
  `W(0.5) = W₀ + 0.5·(W_trained − W₀)` (`scripts/interpolate.py`), which gave 0.824 on the
  dev slice against 0.728 for the pretrained readout and 0.739 for the fully trained one.

Credit where it is due: Laya still wins the *full-label* intent tasks (CLINC150 150-class
0.512 vs 0.085; BANKING77 77-class 0.247 vs 0.182) — those have no training split here, so
the slot head cannot learn their label spaces.

中文版全程记录：[`artifacts/REPORT_V5_CN.md`](artifacts/REPORT_V5_CN.md)。

Full output: [`artifacts/REPORT.md`](artifacts/REPORT.md) and `artifacts/report.json`.
The dataset numbers are in [`artifacts/DATASETS.md`](artifacts/DATASETS.md) and
`artifacts/dataset_results.json`.

### The nine datasets: Laya vs QwenJev-lite

100 test items per task (600 Jigsaw decisions, 2800 GoEmotions decisions), one shared
multi-task readout trained on 4579 decisions from the train splits (2 epochs, 34 min).
`acc` is exact-match accuracy, `bal` is balanced accuracy (mean per-class recall).

| task | n | majority | chance | Laya acc | Laya bal | Qwen zero-shot acc | Qwen RLCD acc | Qwen RLCD bal | Qwen RLCD ECE |
|---|---|---|---|---|---|---|---|---|---|
| mnli (matched) | 297 | 0.364 | 0.333 | 0.549 | 0.548 | 0.764 | 0.754 | 0.763 | 0.136 |
| mnli (mismatched) | 297 | 0.394 | 0.333 | 0.542 | 0.540 | 0.828 | 0.815 | 0.817 | 0.077 |
| snli | 296 | 0.385 | 0.333 | 0.639 | 0.637 | 0.889 | **0.919** | 0.919 | 0.049 |
| fever | 100 | 0.360 | 0.333 | 0.280 | 0.333 | 0.330 | 0.410 | 0.373 | 0.391 |
| jigsaw | 600 | 0.962 | 0.583 | 0.907 | 0.783 | 0.863 | **0.970** | 0.550 | 0.023 |
| goemotions | 2800 | 0.957 | 0.607 | 0.896 | 0.691 | 0.718 | 0.956 | 0.500 | 0.042 |
| truthfulqa | 100 | 0.310 | 0.200 | 0.210 | 0.208 | 0.570 | **0.620** | 0.632 | 0.236 |
| mmlu_pro | 100 | 0.140 | 0.100 | 0.030 | 0.032 | 0.240 | 0.270 | 0.231 | 0.363 |
| intentgrasp | 100 | 0.190 | 0.100 | 0.280 | 0.225 | 0.370 | 0.350 | 0.414 | 0.519 |
| clinc150 | 100 | 0.010 | 0.007 | 0.020 | 0.020 | n/a | 0.020 | 0.020 | 0.687 |
| hwu64 | 100 | 0.030 | 0.016 | 0.030 | 0.021 | n/a | 0.050 | 0.056 | 0.572 |

Mean over the nine tasks where both models are applicable (CLINC150/HWU64 excluded:
no label names): **Laya 0.481, QwenJev-lite zero-shot 0.619, QwenJev-lite RLCD 0.674**
(balanced accuracy 0.444 / 0.626 / 0.578).

What the table actually says:

* **Where the base model has the knowledge, it dominates.** On entailment — the shape
  the architecture is built for, one premise with ~3 hypotheses — QwenJev-lite is
  0.919 vs Laya's 0.639 on SNLI and 0.815 vs 0.542 on the mismatched (out-of-domain)
  MultiNLI. Laya is a 640 MB specialist; Qwen3.5-4B is a 9 GB generalist.
* **Where the task is a simple surface judgement, Laya wins on accuracy and speed.**
  Jigsaw (0.907) and GoEmotions (0.896) are its home turf.
* **RLCD training does what it promises on most tasks** — SNLI +0.030, Jigsaw +0.107,
  GoEmotions +0.238, TruthfulQA +0.050, FEVER +0.080, MMLU-Pro +0.030 — and it collapses
  the calibration error with it (SNLI ECE 0.098 → 0.049, Jigsaw 0.142 → 0.023).
* **On the two imbalanced binary tasks it buys accuracy by predicting the prior**: the
  trained head answers "no" almost everywhere, so Jigsaw accuracy rises to 0.970 while
  balanced accuracy *falls* to 0.550, and GoEmotions lands exactly on the 0.956 majority
  baseline. Log loss on a 1:10 class ratio does that; class-balanced sampling or a
  weighted loss is the fix. This is the honest failure mode of "train the distribution",
  and it is why both columns are in the table.
* **CLINC150/HWU64 are unusable until the label names are added** — 0.02 and 0.05
  against a chance rate of 0.007 and 0.016.

### Speed

Wall clock for the whole benchmark (4890 decisions per variant):

| variant | requests | decisions | decisions/request | ms/request | total |
|---|---|---|---|---|---|
| Laya | 4890 | 4890 | 1.00 | **27** | **130 s** |
| QwenJev-lite (zero-shot) | 900 | 4690 | 5.21 | 583 | 525 s |
| QwenJev-lite (RLCD) | 1100 | 4890 | 4.45 | 587 | 646 s |

Laya is **22× faster per request**, and it is a single-question model: every decision
costs one call. Our engine puts 1-28 decisions in one call (GoEmotions answers 28
questions about a comment in one batch; Jigsaw 6; MultiNLI ~3), which is why the wall
clock gap is 5× rather than 22×. On the pure-torch linear-attention fallback that is
the honest picture — install `flash-linear-attention` and `causal-conv1d` and the
per-request cost drops roughly ten-fold, at which point the shared-state batching wins
outright.

Per decision the gap is 4.9× (27 ms vs 132 ms) and it shrinks as a state carries more
questions, because our cost is roughly *fixed per call + ~12 ms per branch* while Laya
charges 27 ms per decision. Measured directly on one state:

| questions on one state | Laya | QwenJev-lite | winner |
|---|---|---|---|
| 1 | 26 ms (1 request) | 335 ms (1 request) | Laya, 12.8× |
| 10 | 239 ms (10 requests) | 319 ms (1 request) | Laya, 1.3× |
| 100 | 2708 ms (100 requests) | **1496 ms (2 requests)** | **QwenJev-lite, 1.8×** |

The crossover sits between 10 and 20 questions; GoEmotions (28 questions per comment)
is already a win for us on wall clock (67 s vs 77 s for 2800 decisions).

中文版：[`artifacts/REPORT_CN.md`](artifacts/REPORT_CN.md).

An honest nuance from the ablation: sharing the state does **not** help when the state
is short. On 10 GoEmotions items (280 decisions, comment ≈ 15 tokens), `--no-share-state`
took 5.5 s against 6.7 s shared — re-encoding a 15-token state is free and the cache
plumbing is not. Sharing pays when the state is long relative to the branches, which is
exactly the `QS → S` claim in the essay and what `probe latency` measures
(4,097-token state: 1.44 s to encode once).

### The opening example

```
state:   "My payouts have failed three times. The bank says everything is fine.
          The customer has been waiting nine days and is threatening a chargeback."

queue    → payments 0.951 | account 0.005 | other 0.044     (confidence 0.927)
escalate → yes 0.931                                        (confidence 0.862)

usage:  state 37 tokens + questions 84 tokens = 121 input, 37 billed output
timing: 566 ms state encode + 305 ms for both branches in one forward
```

Both distributions came from one forward pass. No text was generated.

### Probes

| Probe | QwenJev-lite | The essay |
|---|---|---|
| visibility: secret in a sibling question | p(secret) = 0.203 | 0.00 |
| visibility: secret absent | p(secret) = 0.203 | (same as sibling) |
| visibility: secret in the shared state | p(secret) = 0.740 | 0.90–0.92 |
| reference card in the shared state | 12/12 correct, mean p = 0.891 | 48/48, mean p ≈ 0.88 |
| reference card inside the option list | 2/12 correct, mean p = 0.28 | 12/16 with the card last, ~0.5 elsewhere |
| irrelevant extra option | log-odds −3.034 → −3.343, change −0.309, 10/10 blocks down | +0.38 → +0.11, change −0.28, 10/10 blocks down |
| different extra option ("wild birds") | change −0.060 | interval included zero |
| option order reversed | p(payments) 0.913 → 0.890 | 0.84–0.89 → 0.93–0.96 |
| injected fake options | 3 slots, no displacement | no displacement |
| token accounting | exactly additive; mixed request = sum of parts | additive in controlled examples |

The isolation result is the cleanest reproduction: the probability is *identical* to
four decimal places whether the sibling question carries the secret or not, and moves
only when the declaration crosses into the shared state. Branch isolation is
structural here — linear-attention layers keep a per-row recurrent state and the
full-attention layers never look across the batch — not a prompt-level convention.

The reference-card result does **not** match the essay: our readout uses the shared
state but ignores a reference card that sits inside the option list. The essay itself
flags this experiment as order sensitivity rather than a recovered attention mask, so
we report the difference instead of tuning the prompt until it agrees.

### Latency

```
state encode (cache off)      261 tok:  376 ms | 1031 tok:  458 ms | 4097 tok: 1437 ms
questions, one cached state     1:  398 ms |  8:  265 ms |  64:  725 ms | 256: 2899 ms
shared vs separate requests     4:  272 vs 1108 ms (4.1x) | 16: 303 vs 4290 ms (14.2x)
```

The last line is the essay's central claim: sending `Q` questions against one shared
state costs roughly one request, while `Q` independent requests re-encode the state
every time.

### RLCD and calibration

The synthetic domain in [`qwenjev/synth.py`](qwenjev/synth.py) generates support tickets
with a deterministic outcome rule, disjoint train/shift templates, and a "primary
issue" marker so some cases are genuinely ambiguous. 1024 tickets, 2 epochs, log loss,
backbone frozen, readout trained (`2560 × 256` head, 78 s on the 3090):

| metric | before RLCD | after RLCD | after temperature | shifted data |
|---|---|---|---|---|
| accuracy | 0.870 | **0.938** | 0.938 | 0.766 |
| mean top probability | 0.786 | 0.962 | 0.897 | 0.921 |
| ECE | 0.094 | **0.027** | 0.047 | 0.161 |
| Brier | 0.230 | **0.093** | 0.101 | 0.391 |
| NLL | 0.406 | **0.224** | 0.206 | 1.011 |

Training the distribution on outcomes does two things at once: it raises accuracy and
it makes the probabilities mean something (ECE 0.094 → 0.027). Under distribution
shift — unseen phrasings for the same decision — accuracy falls to 0.766 and ECE rises
to 0.161, which is exactly the essay's warning that held-out workflow data under shift
matters more than another aggregate benchmark score.

Two honest notes on the temperature column: it is fitted on a **separate** calibration
split by minimising NLL, and NLL-optimal is not ECE-optimal — here it improves NLL
(0.224 → 0.206) while making ECE slightly worse (0.027 → 0.047). Post-hoc scaling is a
different objective from training, and the essay does not claim otherwise.

## Design decisions and deviations

* **Readout default.** Two paths exist. `reserved_label` reads the pretrained LM head's
  mass on the reserved label tokens (`K ≤ 26`), so the service works with no training
  at all. `slot_head` is the essay's dedicated `K`-slot head (`K ≤ 255`), initialised
  from those same reserved rows and trained by RLCD. The engine refuses a request the
  active readout cannot answer rather than silently changing the mechanism.
* **`mode=ref` (share_state=False).** The reference path recomputes the state for every
  branch — the thing the essay says Jev avoids. It exists so the shared path can be
  differentially tested against it (`tests/test_engine_tiny.py` asserts they agree).
* **Backbone frozen during RLCD.** The essay suggests the backbone is adapted. We train
  the readout only: it is the cheap faithful version and it keeps a full run under two
  minutes. `RLCDFineTune` uses ordinary autograd, so unfreezing is a one-line change if
  memory allows.
* **`score_confidence` is a reconstruction.** The essay publishes the exact `Choice`
  formula and says `Score` "uses a different formula reflecting distance from the modal
  level" without printing it.
* **Digit labels.** The essay notes Jev's tokenizer splits digits one at a time; Qwen's
  does too (`"10"` → `["1", "0"]`), which is why the reserved-label readout uses letter
  labels rather than numbers.
* **Determinism.** The local engine returns bit-identical distributions for identical
  payloads, so the essay's request-noise observations have no counterpart here; the
  option-interaction probe prints a duplicate-request control to make that explicit.

## Performance

The shipping model's fused linear-attention kernels (`flash-linear-attention`,
`causal-conv1d`) are **not installed** on this machine, so `transformers` falls back to
a pure-torch chunked delta rule. Consequences:

* a correct but slow path: a single 32-token branch costs ~250–300 ms, and 4,097-token
  states take ~1.4 s to encode;
* the essay's 30k tokens in 160 ms and 1,500 questions in a few hundred ms would need
  those kernels; our 16-question request is 303 ms and 256 questions is 2.9 s;
* everything architectural — one state encoding, isolated branches, parallel readouts,
  additive accounting — is measurable regardless. Install `flash-linear-attention` and
  `causal-conv1d` to close the gap.

## Tests

```bash
python -m pytest            # 89 tests, ~24 s on CPU: no GPU, checkpoint or downloaded data
```

The suite runs the whole pipeline against a tiny random Qwen3.5 backbone
(`qwenjev/testing.py`): schema and prompt layout, token accounting, confidence
formulas, reliability metrics and temperature fitting, branch isolation and state
caching, the HTTP surface, RLCD training, the probes, and every dataset adapter
(against the small fixtures in `tests/fixtures/`), plus the normaliser (including the
label-sorted-file and late-binding regressions), the Laya translation layer and the
benchmark baselines.

## Layout

```
qwenjev/
  schema.py        typed questions and validation
  prompt.py        state/branch templates
  tokenize.py      branch tokens + option spans (for the pointer readout)
  readout.py       reserved-label / slot-head / pointer readouts
  engine.py        shared-state encoding, branch batching, decide()
  confidence.py    adapter confidence formulas, billing figure
  calibration.py   bins, ECE, Wilson intervals, temperature scaling
  rlcd.py          proper scoring rules and the readout trainer
  synth.py         synthetic ticket domain with a train/shift split
  datasets.py      FEVER / MMLU / MMLU-Pro / CLINC150 / HWU64 / Jigsaw adapters
  evaluation.py    dataset evaluation: per-question reliability and usage
  normalize.py     one canonical record for every dataset in data/raw
  backends.py      QwenBackend + LayaBackend behind a single decide() interface
  benchmark.py     per-task accuracy / calibration / speed, with baselines
  probes.py        the essay's experiments
  api.py           FastAPI surface (/v1/decide, /v1/models, /healthz)
  cli.py           demo, probe, datasets, normalize, eval, benchmark, train, account, serve, reproduce
  report.py        markdown report renderer
  testing.py       tiny backbone + stub tokenizer
scripts/           mmlu_to_jsonl.py, fever_evidence.py, split_csv.py
train.py           train one readout on all of data/ready (progress bar)
test.py            score Laya + both readouts on all of data/ready (progress bar)
demo.py            user-facing CLI demo: the essay's example, all three question types
serve.py           user-facing web demo (judgement / choice / score tabs)
scripts/interpolate.py   sweep how far the trained readout may move from the pretrained one
scripts/dev_score.py     quick dev-slice score for a checkpoint
data/raw/          the nine source datasets as provided (git-ignored)
data/ready/        the formatted train/test JSONL + manifest.json
models/            trained readouts (readout.pt + card.json per run)
artifacts/         report.json, REPORT.md, probes.json, readout.pt,
                   dataset_results.json, DATASETS.md, REPORT_CN.md, train_multitask.json
tests/             89 tests, plus fixtures for the dataset adapters
```

## Reference

Archer Hume, *Jev's Architecture Unmasked*, 17 September 2026
(`C:\Users\tzy\Desktop\Jev's Architecture Unmasked — archerhume.pdf`). Section numbers in
this README refer to that essay. Experiments ran on 28 September 2026.
