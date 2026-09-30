# QwenJev-lite 数据集清理 + 速度排查报告（V9）

本轮按你的要求做了两件事：**（1）** 试着装 `flash-linear-attention` / `causal-conv1d`，看能不能把速度提上去；**（2）** 把"本身有问题、按设计用不了"的数据集备份后删掉，再跑一次全量看效果。架构与训练方法的诊断在 [`REPORT_V8_CN.md`](REPORT_V8_CN.md)，本轮只记这两件事。

---

## 1. 融合算子（速度）：这台机器装不上，原因逐条可查

机器事实：`torch 2.6.0+cu126`、RTX 3090（SM86）、Python 3.11.5、**没有 triton**、**没有 nvcc**。逐项尝试结果：

| 尝试 | 结果 |
|---|---|
| `pip install flash-linear-attention` | 包有（`flash_linear_attention-0.5.2-py3-none-any.whl`，纯 Python），但 `fla.ops.gated_delta_rule` 要 Triton，本机没有 → 单独装它没有任何作用 |
| `pip install causal-conv1d` | **Windows 没有预编译轮子**（`pip download --only-binary=:all:` 直接回 *from versions: none*），源码编译要 `nvcc`，本机没有 CUDA toolkit |
| `pip install triton-windows` | 索引能连上，但下载 Wheel 时卡死（沙箱连不到文件服务器）；`torch.compile` 也依赖它，实测报 *Cannot find a working triton installation* |
| `kernels` / `USE_HUB_KERNELS` | transformers 5.x 支持从 Hub 拉预编译 kernel，但 `Qwen3_5GatedDeltaNet` 那条映射写死了 **SM121（GB10）**，`_HUB_KERNEL_MAPPING` 里也没有 `flash-linear-attention`；Qwen3.5 的建模代码是直接 `from causal_conv1d import ...`、`from fla.ops...`，所以 Hub 路线在 3090 上打不开快路径 |

结论：**在当前 Windows + 无 Triton + 无 nvcc 的环境里融合算子装不上**（四条路都被堵死，不是没试）。模型继续跑纯 torch 的分块 delta rule；注意力部分模型本身已经用 SDPA（`_attn_implementation: sdpa`，8 个 full-attention 层），没有可捡的便宜。

实测速度（交付模型，`decide_batch`，含 padding）：

| 场景 | 耗时 |
|---|---|
| 1 个问题（15 个选项，短 state） | 94 ms |
| 同一 state 上 16 个问题×15 选项 | 1.17 s（73 ms/决策） |
| 同一 state 上 64 个问题×15 选项 | 4.70 s（73 ms/决策） |
| 全量 28 个任务（14409 条决策） | 137 ms/请求，3.5 决策/请求 |

想要 10 倍级别的提升，只能到 Linux 上 `pip install flash-linear-attention causal-conv1d`（或在 Windows 装 CUDA toolkit + MSVC 自行编译 `causal-conv1d`，再配 `triton-windows`）。装完 transformers 会自动走快路径，这份数字可以直接重跑对比。

---

## 2. 数据集清理：备份 → 删除 → 说明

**备份**：删除前把整个 `data/raw` 逐字节复制到 `data/raw_backup/`（0.70 GB，已加入 `.gitignore`，不进仓库）；`data/ready/*.jsonl` 的旧版本在 git 历史里可找回。

### 删除的三个来源（5 个任务）

| 删除 | 为什么按设计用不了 | 恢复需要什么 |
|---|---|---|
| `fever`、`fever_support` | 拿到的 claim 文件（`train.jsonl` / `shared_task_dev.jsonl` / `paper_test.jsonl`）只有标签和 claim，**没有 wiki 句子**，共享 state 只能是 claim 本身——量到的是"闭卷事实核查"（0.44），不是这个架构要做的"带证据核验"；读出器本来就把 state 标成 `has_evidence: false` | `data/raw/FEVER/` 放回 claim 文件 **加上** wiki dump 生成的 `evidence.jsonl`（`scripts/fever_evidence.py`） |
| `mmlu_pro` | 官方只有 70 条带标签验证样本；为了训 10 路题的行，训练集是从**它自己的 test 文件**里挖的（虽与评测行不重叠，但"从测试文件挖训练集"不能算诚实的 train/test 对）；论文的 84.6% 也来自另一个不公开的底座模型 | Berkeley MMLU tarball（`scripts/mmlu_to_jsonl.py`）+ 体量够用的 MMLU-Pro 验证集 |
| `banking77`、`banking77_top15` | 只给了 770 行 TF-IDF **testset**，完全没有训练集；之前是拿这个文件的剩余行当训练集 | PolyAI/BANKING77 官方 `train.csv`（3083 行） |

### 保留但必须写清口径的

* `truthfulqa`（817 行，官方无 train split，用 30% 确定性留出）、`jigsaw`（官方 test.csv 无标签，用 train.csv 的 20% 留出）、18 个相关性判别任务（按你的要求由 BEIR 的 query+qrels 构建：正样本是标注答案、负样本不是 qrel；按 query 切分训练/测试）。这些的训练行与评测行**互不重叠**，能说的只有这一点。
* `trec-covid` 三个任务只有测试集（该 collection 只有 50 个 query，不够切）；它们的答案 id 与另外四个 collection 的 rel_* 训练集**完全相同**（同一个 `(问题, 标签空间, 答案)`），所以照样读到训练行——这是"同问题跨集合迁移"，不是泄漏。

### 删除会不会让模型变强？

**不会。** 交付模型的行按 `(问题, 标签空间, 答案)` 私有，删掉几个任务的数据不可能改变其它任务读到的行。实测也如此：删完重新生成原型行（`models/_p4`）再重新拼装后，dev 切片上保留下来的任务逐项与 V8 一致（只有 hwu64 因全局缩放系数变化 0.425 → 0.450）。**变化的只是统计口径**：不再把"按设计用不了"的任务算进平均分。

---

## 3. 清理后的全量结果（28 个测试集）

`artifacts/dataset_results_v9.json`，表格 `artifacts/DATASETS_V9.md`；模型 `models/qwenjev-multitask-v2/readout.pt`（512 行 = 27 保留 + 485 私有，90 个标签空间）。

按题型平均（28 个测试集，一次跑完；零样本读出因为只有 26 个字母行，答不了"选项数 >26"的两个宽空间）：

| 题型 | Laya | 零样本读出 | **训练后读出** |
|---|---|---|---|
| bool（7 项） | 0.520 | 0.717 | **0.794** |
| choice（14 项） | 0.502 | 0.749（12 项） | **0.728**（14 项） |
| choice（两边都能答的 12 项） | 0.511 | 0.749 | **0.760** |
| score（7 项） | 0.258 | 0.681 | **0.771** |
| **总体（28 项）** | **0.446** | 0.722（26 项） | **0.755** |

平均 ECE：Laya 0.291 / 零样本 0.158 / 训练后 **0.138**（bool 0.109、score 0.124、choice 0.160）。

速度（同一轮全量）：Laya 23 ms/请求（23145 请求、1.00 决策/请求、9.0 min）；
训练后读出 159 ms/请求（5335 请求、3.52 决策/请求、14.1 min）。按决策摊，
Laya 23 ms、我们 39 ms；按"一个 state 问多个问题"我们更省。

逐任务（完整表见 `artifacts/DATASETS_V9.md`）：

| task | 题型 | Laya | 零样本 | 训练后 |
|---|---|---|---|---|
| arguana_rel_bool | bool | 0.450 | 0.330 | **0.803** |
| goemotions | bool | 0.907 | 0.722 | **0.958** |
| jigsaw | bool | 0.897 | 0.869 | **0.965** |
| nfcorpus_rel_bool | bool | 0.349 | 0.701 | 0.685 |
| scidocs_rel_bool | bool | 0.361 | 0.800 | 0.669 |
| scifact_rel_bool | bool | 0.258 | **0.889** | 0.804 |
| trec-covid_rel_bool | bool | 0.420 | **0.707** | 0.670 |
| arguana_rel_choice | choice | 0.650 | 0.900 | **0.983** |
| clinc150（150 类） | choice | 0.545 | – | **0.527** |
| clinc150_top15 | choice | 0.893 | **0.893** | 0.733 |
| hwu64（64 类） | choice | 0.345 | – | **0.540** |
| hwu64_top15 | choice | 0.600 | **0.613** | 0.467 |
| intentgrasp | choice | **0.315** | 0.323 | 0.302 |
| mnli | choice | 0.582 | **0.792** | 0.782 |
| mnli_ood | choice | 0.596 | **0.824** | 0.808 |
| nfcorpus_rel_choice | choice | 0.333 | 0.617 | **0.733** |
| scidocs_rel_choice | choice | 0.467 | 0.950 | 0.950 |
| scifact_rel_choice | choice | 0.417 | 0.967 | **1.000** |
| snli | choice | 0.617 | 0.861 | **0.868** |
| trec-covid_rel_choice | choice | 0.420 | 0.640 | **0.900** |
| truthfulqa | choice | 0.242 | **0.604** | 0.592 |
| arguana_rel_score | score | 0.190 | 0.647 | **0.840** |
| goemotions_sentiment | score | 0.425 | **0.605** | 0.578 |
| jigsaw_severity | score | 0.385 | 0.860 | **0.915** |
| nfcorpus_rel_score | score | 0.262 | 0.555 | **0.583** |
| scidocs_rel_score | score | 0.217 | 0.622 | **0.831** |
| scifact_rel_score | score | 0.144 | 0.781 | **0.961** |
| trec-covid_rel_score | score | 0.187 | **0.697** | 0.690 |

对比 V8（33 个任务、含本轮删掉的 5 个）：题型内部的逐任务数字完全一致，
变化的是平均口径——V8 的 33 项总平均 0.713、choice 0.671；本轮 28 项总平均 0.755、
choice 0.728。宽标签空间的 clinc150 0.542 → 0.527、hwu64 0.542 → 0.540 是因为
原型行在删数据后重新生成，全局缩放系数变了（见 §2 末）。

## 4. 复现命令

```bash
# 0) 删除前先备份（本轮动作）：copy data/raw -> data/raw_backup（0.70 GB，已 gitignore）

# 1) 只对保留下来的来源重新生成统一格式（FEVER / MMLU-Pro / BANKING77 已删）
python -m qwenjev.cli normalize --src data/raw --out data/ready

# 2) 原型行（约 20 分钟，宽标签空间用）
python train.py --no-balance --epochs 0 --items-per-label 12 --min-items-per-task 200 \
  --prototype-init --row-norm-cap auto --optimizer sgd --max-grad-norm 0 \
  --model-dir models/_p4 --report artifacts/_p4.json

# 3) 拼装交付模型（0-26 行保持预训练字母行，其余每个标签空间一块行）
python scripts/compose_wide_rows.py \
  --base models/_hybrid/readout.pt --wide models/_p4/readout.pt --data-dir data/ready \
  --out models/qwenjev-multitask-v2/readout.pt

# 4) 全量对比（28 个测试集 × Laya / 零样本 / 训练后）
python test.py --variants laya qwen_zeroshot qwen_trained --limit 0 --batch 16 \
  --out artifacts/dataset_results_v9.json --markdown artifacts/DATASETS_V9.md

# 5) 融合算子（本机做不到，Linux 上可直接跑）
# pip install flash-linear-attention causal-conv1d
```
