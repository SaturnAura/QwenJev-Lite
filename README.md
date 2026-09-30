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

本轮（V9）做了两件事：**融合算子**和**数据集清理**。融合算子（`flash-linear-attention` / `causal-conv1d`）在这台机器上装不上——`causal-conv1d` 没有 Windows 轮子、本机没有 nvcc、Triton 下载卡死、`torch.compile` 也依赖 Triton，四条路都试过，逐条记录在 [`artifacts/REPORT_V9_CN.md`](artifacts/REPORT_V9_CN.md)。数据集方面：删前先把 `data/raw` 整体备份到 `data/raw_backup/`（0.70 GB，已 gitignore），然后删掉三个"按设计用不了"的来源（FEVER 缺 wiki 证据句、MMLU-Pro 只有 70 条带标签验证样本且训练集只能从它自己的测试文件里挖、BANKING77 完全没有训练集），共 5 个任务；理由与恢复办法写进 README 的 Dropped datasets 一节。

清理后重跑全部 **28 个测试集**：按题型平均 —— Laya 0.446 / 零样本读出 0.722（只能答 26 项）/ 训练后读出 **0.755**；其中 bool **0.794**、choice **0.728**（两边都能答的 12 项上 0.760 对零样本 0.749）、score **0.771**，平均 ECE 0.138（Laya 0.291 / 零样本 0.158）。删数据**不会让模型变强**（行是按"问题+标签空间+答案"私有的，其它任务读到的行一个字都没变，dev 切片逐项一致），变的只是统计口径，报告里写明了。

数据集分两层：`data/raw/`（7 个保留来源；删掉的 3 个见上面 Dropped datasets）和 `data/ready/` 统一格式的训练/测试集（13 个来源任务 + 从 query/文档集合构建的 15 个相关性判别任务，共 28 个测试集，三种题型 choice / bool / score 都有）。训练脚本 `python train.py`、测试脚本 `python test.py`（都带进度条），跑完把 Laya 和 QwenJev-lite 并排对比。最新一轮的结果与速度见 [Results](#results)。

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

The seven datasets in `data/raw/` arrive in five different shapes (parquet, JSONL, CSV,
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
| `jigsaw` | comment | six independent yes/no labels | 2000 / 400 |
| `goemotions` | comment | 28 independent yes/no emotions | 2000 / 400 |
| `truthfulqa` | question | 5-way: the best answer against four false ones | 577 / 240 |
| `intentgrasp` | user utterance | choice over that item's own option list | 2000 / 400 |
| `clinc150`, `hwu64` | utterance | 150-way / 64-way intent | 4000 / 400 |
| `clinc150_top15`, `hwu64_top15` | utterance | closed 15-intent subset | 1500 / 2000, each / 75-150 |

`scripts/build_extra_splits.py` adds the splits the sources do not ship (CLINC150/HWU64
full-space and 15-intent training sets, from the datasets' own train parquets). Every
generated split is checked against the test file it will be scored on, and refused if a
single answer the test reads was never trained.

The formatted files live in `data/ready/`; `data/ready/manifest.json` records the source
file, counts and caveats for every split.

```bash
python -m qwenjev.cli normalize                       # data/raw -> data/ready/*.jsonl
python -m qwenjev.cli normalize --tasks jigsaw snli --test-limit 100
```

Two details worth knowing, because they bit us:

* caps are **spread evenly over the whole file**, not taken as a head slice, and they are
  spread even when the cap is more than half the file: CLINC150/HWU64's test sets are
  label-sorted, so a prefix of them covered 81 / 41 of 150 / 64 classes instead of all of
  them;
* `clinc150` and `hwu64` ship integer label ids with no names; `intents.txt` (one name per
  line, in label-id order) is recovered by clustering the labelled examples
  (`scripts/map_labels.py`), so the options read as intent names and the zero-shot readout
  can answer them.

### Backends and the benchmark

Both models implement the same interface, so they are scored by the same code:

```bash
# one shared multi-task readout trained on every task's train split
python -m qwenjev.cli train --tasks mnli snli jigsaw goemotions truthfulqa \
       intentgrasp clinc150 hwu64 --decisions-per-task 500 --epochs 2 \
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

* **FEVER and MMLU-Pro are no longer in the mix** - the data this machine has cannot
  support them as designed. See [Dropped datasets](#dropped-datasets) for the reasons,
  what is kept in the backup and how to restore them.
* **Label spaces wider than 26 options** (CLINC150, HWU64, BANKING77) cannot be answered
  by the pretrained readout at all - its rows are the reserved single-token letters and the
  engine raises rather than silently truncating the label space. Those three need the
  trained head, which is why its choice average is reported over 18 tasks and the
  pretrained readout's over 15.
* **The three intent test sets were rebuilt for this round.** Their source TSVs are sorted
  by label, and the old cap took the first `limit` rows when the stride collapsed to one,
  so 400 test items covered only the first 81 / 40 / 41 classes. The cap now spreads over
  the whole file, so 400 items cover all 150 / 77 / 64 classes - every variant's numbers on
  those three tasks changed, Laya's included.
* **Jigsaw / GoEmotions** are heavily imbalanced (majority class 96% / 96%). Accuracy
  and ECE look excellent for a model that always answers "no"; read the balanced
  accuracy column.
* **TruthfulQA, Jigsaw and the relevance collections have no official train split**
  either, so their training halves are deterministic hold-outs (`30%` of the released
  CSV, `20%` of `train.csv`, and 500 of the 560 queries per collection). The rows are
  disjoint from the rows they are scored on; that is the whole of the claim.

## Dropped datasets

Three sources cannot support the decision the architecture is for, so they were removed
on 30 September 2026 rather than reported with a footnote. The originals are kept:

* `data/raw_backup/` holds a byte-for-byte copy of `data/raw/` from before the removal
  (0.70 GB, git-ignored - it never enters a commit),
* `git log` has the commit that removed them, so `data/ready/*.jsonl` is recoverable too.

| dropped | why it cannot be used as designed | what would restore it |
|---|---|---|
| `fever`, `fever_support` | the supplied claim files (`train.jsonl`, `shared_task_dev.jsonl`, `paper_test.jsonl`) carry the label and the claim but **no wiki sentences**, so the shared state is the bare claim: the task measured closed-book fact checking (0.44), not the evidence-based verification this architecture exists for. The loaders already label the state `has_evidence: false`. | drop the claim files in `data/raw/FEVER/` **plus** `evidence.jsonl` built from the wiki dump with `scripts/fever_evidence.py` |
| `mmlu_pro` | ships 70 labelled validation rows for 10-way questions, and the only way to train the 10 rows was to mine the *test* file's unused rows. The rows are disjoint from the evaluation rows, but a train split carved out of the test file is not an honest train/test pair, and the essay's 84.6% comes from a different, undisclosed base model. | the Berkeley MMLU tarball (`scripts/mmlu_to_jsonl.py`) plus MMLU-Pro's own validation set at a usable size |
| `banking77`, `banking77_top15` | ships a 770-row TF-IDF *testset* and no training split at all; the head was being trained on that file's leftovers. | the official `train.csv` (3083 rows) from PolyAI/BANKING77 |

Everything else is kept, including the parts that are homework rather than releases:
the 18 relevance-judgement tasks are built from the BEIR collections' queries and qrels
(the positive document plus negatives, as you asked), the `score` variants
(`jigsaw_severity`, `goemotions_sentiment`) are derived from the shipped label columns,
and `mnli_ood` is MultiNLI's mismatched dev split.

## Results

### The trained readout beats both the pretrained readout and Laya

Full test set: one run of every variant over the 28 formatted test splits that are left
after the dataset cleanup (`test.py --limit 0`; raw results in
`artifacts/dataset_results_v9.json`, rendered tables in `artifacts/DATASETS_V9.md`,
Chinese write-up in [`artifacts/REPORT_V9_CN.md`](artifacts/REPORT_V9_CN.md); the
architecture work is in [`artifacts/REPORT_V8_CN.md`](artifacts/REPORT_V8_CN.md)):

| family | Laya | pretrained readout | **trained readout** |
|---|---|---|---|
| judgement (bool, 7 tasks) | 0.520 | 0.717 | **0.794** |
| choice (14 tasks) | 0.502 | 0.749 (12 tasks) | **0.728** (14 tasks) |
| choice (the 12 tasks both readouts can answer) | 0.511 | 0.749 | **0.760** |
| score (7 tasks) | 0.258 | 0.681 | **0.771** |
| **overall (28 tasks)** | **0.446** | 0.722 (26 tasks) | **0.755** |

Calibration: mean ECE 0.291 (Laya) / 0.158 (pretrained) / **0.138** (trained; bool 0.109,
score 0.124, choice 0.160). Speed over the whole benchmark: Laya 23 ms/request (1 decision
per request, 9.0 min), the trained readout 159 ms/request (3.5 decisions per request,
14.1 min).

The two label spaces wider than the reserved 26 rows are where the trained head is
decisive, because the pretrained readout *cannot answer them at all*:

| label space | before this work | **trained head now** | Laya |
|---|---|---|---|
| CLINC150 (150-way) | 0.077 | **0.527** | 0.545 |
| HWU64 (64-way) | 0.225 | **0.540** | 0.345 |

(The 0.077 / 0.225 were measured on those tasks' old test files - see the sampling caveat
under [Datasets](#datasets) - so read the left column as a direction, not a like-for-like
pair. BANKING77 was dropped, see [Dropped datasets](#dropped-datasets).)

How it was reached, in one line each:

* **the choice family barely moved for a structural reason.** The readout picked a row by
  the option's *position*, so training a 2- to 4-option task rewrote the very rows a
  10- to 15-option task reads (clinc150_top15 0.893 -> 0.733, hwu64_top15 0.613 -> 0.467,
  intentgrasp 0.350 -> 0.225, while the tasks that *were* in the training mix improved).
  Rows are now private per *(question, label space, answer)*, and an answer the head never
  saw still falls back to its pretrained letter row, so an unseen label space keeps the
  zero-shot behaviour exactly (tested in `tests/test_rlcd_tiny.py`);
* **the gradient budget was lopsided.** goemotions alone contributed 3360 decisions while a
  choice task got ~120, so choice held 14.7% of the gradient. Budgeting per label space
  (`--items-per-label`) raised that to 36%;
* **three label spaces had no way to be learned.** CLINC150/HWU64 only became trainable once
  their integer labels were named, the 15-intent subsets are a *different* label space from
  the full 150-/64-way one, MMLU-Pro ships 70 labelled rows and BANKING77 ships none.
  `scripts/build_extra_splits.py` builds all of them and refuses to write a split whose
  answers the test set would not read;
* **numerically**, decision states have norm ~157 while the pretrained label rows have norm
  0.74, so one step at lr 1e-3 moves a fresh row a fifth of its useful length (measured mean
  training loss 33.0 against 5.01 for a uniform head). Rows are now projected back to the
  reserved rows' norm after every step (`--row-norm-cap auto`), SGD is the default (Adam's
  per-coordinate scaling collapses the rows of a wide label space onto one direction), and
  gradient clipping is off by default: `clip_grad_norm_(1.0)` scales a gradient of norm ~60
  down 60x, which Adam shrugs off and which leaves SGD with a uselessly small step;
* **a 150-way label space still cannot be reached by gradient steps** (loss 4.90 after 124
  steps, i.e. nothing, because the loss is averaged over the options and only ~1% of each
  step separates the answers). The delivered head therefore keeps the shared trained rows
  where the reserved budget covers the label space and uses *closed-form prototype rows*
  (the mean state of the samples that chose each answer) where it does not
  (`train.py --prototype-init` + `scripts/compose_wide_rows.py`);
* earlier rounds: class-balanced sampling destroyed the prior (`P(yes)=0.78` on a task whose true rate is
  `0.043`) — training without it recovered 0.503 → 0.630;
* the remaining gap was structural: one linear `W h + b` readout has to serve every task,
  and training on some label spaces overwrote the reserved rows that the *unseen* label
  spaces rely on (the 15-intent tasks fell 0.89 → 0.33);
* so the head is trained with an L2 anchor to its initialisation
  (`train.py --anchor 1e-2`) and the final matrix keeps half of the trained correction,
  `W(0.5) = W₀ + 0.5·(W_trained − W₀)` (`scripts/interpolate.py`), which gave 0.824 on the
  dev slice against 0.728 for the pretrained readout and 0.739 for the fully trained one;
* more training data (per-item budgets, 5.6k → 11.4k decisions, 500 queries per relevance
  collection) moved jigsaw/goemotions up but left the average flat; training sample
  budgets of 400 decisions per task, 3 epochs, class balancing, training on the
  recovered CLINC150/HWU64 label spaces and averaging two training runs were all tried
  and rejected — the numbers are in
  [`artifacts/REPORT_V6_CN.md`](artifacts/REPORT_V6_CN.md).

Laya is beaten on CLINC150 and HWU64 outright (0.527 vs 0.545 is a tie at 400 items;
0.540 vs 0.345) and still wins the two 15-intent subsets (0.733 / 0.467 against
0.893 / 0.613 for the pretrained readout) - the honest cost of keeping the shared rows
wherever the reserved budget covers the label space.

中文版全程记录：[`artifacts/REPORT_V5_CN.md`](artifacts/REPORT_V5_CN.md)。

Full output: [`artifacts/REPORT.md`](artifacts/REPORT.md) and `artifacts/report.json`.
The dataset numbers are in [`artifacts/DATASETS.md`](artifacts/DATASETS.md) and
`artifacts/dataset_results.json`.

### First round: the nine datasets (historical record)

This subsection is the first round's write-up, from before the readout work in
[`artifacts/REPORT_V8_CN.md`](artifacts/REPORT_V8_CN.md) and the dataset cleanup; the
current numbers are in [Results](#results) above. 100 test items per task (600 Jigsaw decisions, 2800 GoEmotions decisions), one shared
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

﻿## Performance

The shipping model's fused linear-attention kernels are **not installable on this
machine**, and that is now a measured statement rather than a guess (30 September 2026):

| attempt | result |
|---|---|
| `pip install flash-linear-attention` | the wheel exists (`flash_linear_attention-0.5.2-py3-none-any.whl`, pure Python) but is useless on its own: `fla.ops.gated_delta_rule` needs Triton, which is not installed |
| `pip install causal-conv1d` | **no Windows wheel exists** (`pip download --only-binary=:all:` reports "from versions: none"), and building it from source needs `nvcc` - the CUDA toolkit is not on this box |
| `pip install triton-windows` | the index answers but the wheel download stalls (no route to the file host from this sandbox); Triton is also what `torch.compile` needs - `torch.compile(model)` fails with "Cannot find a working triton installation" |
| `USE_HUB_KERNELS` / `kernels-community` | transformers 5.x can pull prebuilt kernels from the Hub, but the only `Qwen3_5GatedDeltaNet` entry is pinned to **SM121** (GB10) and `_HUB_KERNEL_MAPPING` has no `flash-linear-attention`; Qwen3.5 imports `causal_conv1d` and `fla` directly, so the Hub route does not switch the fast path on for an RTX 3090 |

So the model runs the pure-torch chunked delta rule for its 24 linear-attention layers
(the other 8 layers already use SDPA - `_attn_implementation: sdpa`). What that costs,
measured on this box with the shipped head:

| work | time |
|---|---|
| 1 question, 15 options, short state, one request | 94 ms |
| 16 questions x 15 options on one state | 1.17 s (73 ms per decision) |
| 64 questions x 15 options on one state | 4.70 s (73 ms per decision) |
| the whole 28-task benchmark (14409 decisions, 4 batches of requests) | 137 ms/request, 3.5 decisions/request |

Two consequences worth stating: a correct but slow path (a 4,097-token state takes
~1.4 s to encode), and the essay's 30k tokens in 160 ms / 1,500 questions in a few
hundred ms would need those kernels. Everything architectural - one state encoding,
isolated branches, parallel readouts, additive accounting - is measurable regardless.
On a Linux box, `pip install flash-linear-attention causal-conv1d` lights up the fast
path in `transformers`; on Windows it needs the CUDA toolkit plus MSVC to build
`causal-conv1d`, and `triton-windows` for the Triton half.

## Tests

```bash
python -m pytest            # 92 tests, ~30 s on CPU: no GPU, checkpoint or downloaded data
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
scripts/map_labels.py    recover a label space's names by matching label clusters
scripts/build_extra_splits.py  MMLU-Pro / CLINC150 / HWU64 / BANKING77 training splits,
                               each checked against the test file it will be scored on
scripts/compose_wide_rows.py   merge shared rows + private rows for wide label spaces
scripts/compare_results.py     per-task and per-family comparison across result files
data/raw/          the nine source datasets as provided (git-ignored)
data/ready/        the formatted train/test JSONL + manifest.json
models/            trained readouts (readout.pt + card.json per run)
artifacts/         report.json, REPORT.md, probes.json, readout.pt,
                   dataset_results.json, DATASETS.md, REPORT_CN.md, train_multitask.json
tests/             92 tests, plus fixtures for the dataset adapters
```

## Reference

Archer Hume, *Jev's Architecture Unmasked*, 17 September 2026
(`C:\Users\tzy\Desktop\Jev's Architecture Unmasked — archerhume.pdf`). Section numbers in
this README refer to that essay. Experiments ran on 28 September 2026.
