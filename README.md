<div align="center">

# QwenJev-Lite

**One shared state. Isolated question branches. Parallel probability readouts — no text is generated.**

![python](https://img.shields.io/badge/python-3.11-3776ab?logo=python&logoColor=white)
![torch](https://img.shields.io/badge/torch-2.6.0%2Bcu126-ee4c2c?logo=pytorch&logoColor=white)
![transformers](https://img.shields.io/badge/transformers-5.14-ffd21e)
![tests](https://img.shields.io/badge/tests-92%20passing-2ea44f)
![license](https://img.shields.io/badge/license-MIT-blue)

[Quickstart](#quickstart) · [Results](#results) · [Model](#the-model) · [How it works](#how-it-works) · [Data](#data) · [Paths & formats](#paths-and-data-formats) · [Speed](#speed) · [References](#引用--references)

</div>

> **中文速览**：一个"类型化决策"模型 —— 共享状态只编码一次，每个问题一条互相隔离的分支，末端直接把隐藏状态映射成答案的概率分布（不生成任何文本）。交付模型在 26 个测试集上按题型平均 **bool 0.794 / choice 0.760 / score 0.771，总体 0.772**（对比 Laya 0.446、零样本读出 0.722），平均 ECE 0.125。模型文件在 `models/`（512 行读出，可逐位复现），数据在 `data/ready/`，结果在 `artifacts/`。跑起来只要三步：

```bash
pip install -r requirements.txt
export QWENJEV_MODEL=/path/to/qwen3.5-4B      # Windows: $env:QWENJEV_MODEL="D:\qwen3.5-4B"
python demo.py --model-dir models/qwenjev-multitask-v2
```

## 声明 / Scope

* **骨架是 Qwen3.5-4B**（32 层混合注意力：24 层线性注意力 + 8 层全注意力，4.54B 参数），
  但**架构不绑定它**。引擎向骨架只要一件事——每个决策位置的隐藏状态——所以任何 causal
  transformer 都能当骨架，**多模态骨架同理**：把图像/音频 token 与文本拼进 `state`（或分支
  context）即可，读出、训练、评测、HTTP 接口都不用改。换骨架只要 `--model`（本地路径或
  Hub repo id），见 [Paths & data formats](#paths-and-data-formats)。
* 训练**只动读出**（512 行，约 5 MB 的 `readout.pt`），backbone 全程冻结；这样一张 24 GB
  显卡就能把一个完整实验跑完（全量评测 14 分钟）。
* 本仓库不含任何第三方代码，只依赖公开的 PyPI 包。数据来源与致谢见 [Data](#data)。

## Results

一次完整评测：`python test.py --variants laya qwen_zeroshot qwen_trained --limit 0`，
在 `data/ready` 里 **26 个三个变体都能作答的测试集**上（bool 7 + choice 12 + score 7）。
原始数据 [`artifacts/results.json`](artifacts/results.json)，完整明细 [`artifacts/RESULTS_CN.md`](artifacts/RESULTS_CN.md)。

| 题型 | Laya（对比基线） | 零样本读出 | **训练后读出** |
|---|---|---|---|
| 判断 `bool`（7 项） | 0.520 | 0.717 | **0.794** |
| 单选 `choice`（12 项） | 0.511 | 0.749 | **0.760** |
| 打分 `score`（7 项） | 0.258 | 0.681 | **0.771** |
| **总体（26 项）** | **0.446** | 0.722 | **0.772** |

平均 ECE（越低越好）：Laya 0.289 / 零样本读出 0.158 / **训练后读出 0.125**。

**超出 26 个保留行的标签空间**（不计入上表，因为零样本读出只有 26 个单字字母行、无法作答）：

| 标签空间 | 训练后读出 | Laya | 零样本读出 |
|---|---|---|---|
| CLINC150（150 类意图） | 0.527 | 0.545 | 无法作答 |
| HWU64（64 类意图） | 0.540 | 0.345 | 无法作答 |

收益来自哪、不来自哪，一并说清楚：训练后读出在相关性判别上大幅领先（`scifact_rel_choice`
1.000、`trec-covid_rel_choice` 0.900、`arguana_rel_bool` 0.803）并在两个宽标签空间上把
"无法作答"变成可用；但在两个 15 类意图子集（0.733 / 0.467）和部分闭卷型判断上仍低于
零样本读出 —— 这是"保留行只覆盖前 26 个答案"的直接代价。

## Quickstart

```bash
pip install -r requirements.txt          # torch 2.6 + transformers 5.14；跑测试不需要 GPU/模型/数据

export QWENJEV_MODEL=/path/to/qwen3.5-4B # Windows: $env:QWENJEV_MODEL="D:\qwen3.5-4B"

python demo.py --model-dir models/qwenjev-multitask-v2   # 命令行演示：一个 state，三种题型一次答完
python serve.py                                          # 同一件事的网页版（:8300）
pytest -q                                                # 92 个测试，用微型骨架在 CPU 上跑
```

| 想做什么 | 命令 |
|---|---|
| 全量评测（Laya + 两种读出，带进度条） | `python test.py --variants laya qwen_zeroshot qwen_trained --limit 0` |
| 自己重新训一个读出 | `python train.py --prototype-init --items-per-label 12 --min-items-per-task 200` |
| 从三个源文件重建交付读出 | [`models/README.md`](models/README.md#rebuild-the-shipped-head) |
| 原始数据 → 统一格式 | `python -m qwenjev.cli normalize --src data/raw --out data/ready` |
| 起 HTTP 服务（`POST /v1/decide`） | `python -m qwenjev.cli serve --port 8300` |
| 架构自检探针（泄漏、选项交互、延迟…） | `python -m qwenjev.cli probe all` |
| 没有 GPU、只想跑通全流程 | 任意入口加 `--fake`：换成微型随机骨架，几秒钟出结果 |

## The model

`models/qwenjev-multitask-v2/readout.pt` 是交付读出：**512 行 = 27 行保留 + 485 行私有**，
覆盖 `data/ready` 里的 90 个标签空间。

* 第 0–26 行保持**预训练标签行**，因此没见过的答案读到的就是骨架自己的判断；
* 训练覆盖过的每个标签空间各占一块私有行：选项数 ≤ 26 的按位置从训练行复制，选项数 > 26
  的用**闭式原型行**（该答案对应状态的平均向量，一次前向、无梯度步骤）；
* 两者都不命中的答案，回落到保留行。

四个目录（`models/README.md` 有链条图和 load 代码，三个源文件拼起来可**逐位复现**交付文件）：

| 目录 | 行数 | 作用 |
|---|---|---|
| `qwenjev-multitask-v2/` | 512 | 交付读出（`card.json` 带本次评测成绩） |
| `shared-rows/` | 256 | 共享训练行（α=0.5 混合后的结果） |
| `raw-trained-rows/` | 256 | α=1 的训练行，用于重扫 α |
| `prototype-rows/` | 1024 | 宽标签空间的闭式原型行 |

## How it works

| 部件 | 实现在哪 |
|---|---|
| 末端读出：`z = W h + b`，在 K 个允许答案上做 softmax，没有解码循环 | [`qwenjev/readout.py`](qwenjev/readout.py)：`ReservedLabelReadout` / `SlotHeadReadout` / `PointerReadout` |
| 直接用预训练词表里的保留标签行读概率（K ≤ 26，无需训练） | `ReservedLabelReadout`：取 LM head 在选项标签 token 上的质量并重新归一化 |
| 三种问题类型：有限选择、是/否、有序打分 | [`qwenjev/schema.py`](qwenjev/schema.py) |
| 共享状态编码一次，问题分支互相隔离 | [`qwenjev/engine.py`](qwenjev/engine.py)：state 预填一次进 KV/递推缓存，按分支展开，每条分支只看 state + 自己的后缀 |
| 分支批调度（批大小 + token 预算），一次前向一批 | `_chunk_branches`；`share_state=False` 是"每条分支重算 state"的参考实现，用于差分测试 |
| 请求上限与计费口径（每分支 32,768 / 每请求 65,536 token，state 只计一次） | `JevLimits`、`check_limits`、`account` |
| 选项在分支里作为一个有序列表整体读入，读出看的是列表之后的决策位 | [`qwenjev/prompt.py`](qwenjev/prompt.py)、[`qwenjev/tokenize.py`](qwenjev/tokenize.py) |
| RLCD：用 proper scoring rule 按结果训练分布（backbone 冻结） | [`qwenjev/rlcd.py`](qwenjev/rlcd.py)（log loss / Brier） |
| 校准与置信度：可靠性分箱、ECE、Brier、Wilson 区间；`confidence` 是算术量 | [`qwenjev/calibration.py`](qwenjev/calibration.py)、[`qwenjev/confidence.py`](qwenjev/confidence.py) |
| 两个后端同一接口（本引擎 / Laya），同一套评测代码 | [`qwenjev/backends.py`](qwenjev/backends.py) |

## Paths and data formats

三个路径都可以用参数或环境变量指定，代码里没有任何绝对路径：

| 路径 | 参数 | 环境变量 | 默认 |
|---|---|---|---|
| 骨架模型 | `--model` | `QWENJEV_MODEL` | `./qwen3.5-4B`（相对当前工作目录，也可以是 Hub repo id） |
| 数据目录 | `--data-dir` | — | `data/ready` |
| 读出输出 | `--model-dir` / `--out` | — | `models/qwenjev-multitask-v2` |
| Laya 基线 | `--laya-path` | `QWENJEV_LAYA` | `./laya` |

数据目录里每个任务两个文件：`<task>_train.jsonl` 和 `<task>_test.jsonl`（可选 `<task>_<split>.jsonl`）。
每行是一条**决策记录**，就是引擎的请求格式加一个 `targets`：

```json
{"id": "mytask/test/0001", "dataset": "mytask", "split": "test",
 "state": "共享文本：这段话只编码一次，可以喂问题分支里的所有问题",
 "questions": {
   "queue":  {"type": "choice", "instructions": "Which team should handle this ticket?",
              "criteria": {"payments": "Payout failures", "account": "Login", "other": "Else"}},
   "urgent": {"type": "bool", "instructions": "Does this need urgent attention?"},
   "grade":  {"type": "score", "instructions": "How severe is it?",
              "criteria": {"low": "Low", "mid": "Medium", "high": "High"}}},
 "targets": {"queue": "payments", "urgent": "no", "grade": "high"}}
```

要点：`targets` 的取值必须是同一条 `criteria` 的键；`choice` / `score` 用 `criteria`（有序），
`bool` 可以省略（默认 `{"yes": ..., "no": ...}`）；问题 id（`queue`…）只是标识符，真正决定读出行的是
`(问题 id, 标签空间, 答案键)`。

**训练一个自己的读出**（换个骨架、换自己的数据、输出到自己的目录）：

```bash
python train.py \
  --model "$QWENJEV_MODEL" \
  --data-dir data/ready \
  --model-dir models/my-run \
  --objective log_loss \
  --prototype-init --items-per-label 12 --min-items-per-task 200 \
  --epochs 0                     # epochs 0 = 只用闭式原型行；要梯度训练就写 1 并配合 --lr
```

**评测**（只想跑自己的读出、不想装 Laya 就省略 `--variants laya`）：

```bash
python test.py \
  --model "$QWENJEV_MODEL" \
  --data-dir data/ready \
  --model-dir models/my-run \
  --variants qwen_zeroshot qwen_trained \
  --limit 0 --batch 16 \
  --out artifacts/my-results.json
```

**单条推理**：

```python
from qwenjev.config import QwenJevConfig
from qwenjev.engine import QwenJevLite

engine = QwenJevLite.from_pretrained(config=QwenJevConfig(
    model_path="qwen3.5-4B",                       # 或任何本地路径 / Hub repo id
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

`data/ready/` 已经在仓库里（约 92 MB 的 JSONL）：formatted train/test 集 + `manifest.json`
（每个 split 的来源文件、条数、注意事项）。用到的来源：

| 来源 | 提供哪些任务 |
|---|---|
| MultiNLI / SNLI | `mnli`、`mnli_ood`、`snli`（一个前提配多条假设） |
| Jigsaw Toxic Comment | `jigsaw`（6 个独立是/否）、`jigsaw_severity`（有序打分） |
| GoEmotions | `goemotions`（28 个情感是/否）、`goemotions_sentiment`（有序打分） |
| TruthfulQA | `truthfulqa`（5 选 1） |
| IntentGrasp | `intentgrasp`（选项随样本给出） |
| CLINC150 / HWU64 | `clinc150`、`clinc150_top15`、`hwu64`、`hwu64_top15` |
| BEIR（arguana / nfcorpus / scidocs / scifact / trec-covid） | 15 个相关性判别任务：用 query 与标注答案构造正样本选项、非 qrel 文档构造负样本选项，按 query 切分训练/测试 |

`artifacts/extra_splits.json` 记录了本仓库**自行补齐**的训练集（来源、条数、校验结果）；
`data/ready/manifest.json` 记录了所有 split 的出处。上面这些数据集、标注以及相关工具的作者与
维护者让这项工作成为可能，在此一并致谢 —— 引用信息见文末 References。

<details>
<summary><b>未纳入的数据集及原因</b>（点击展开）</summary>

删前已把整个 `data/raw` 备份到 `data/raw_backup/`（0.70 GB，已 gitignore）。三个来源无法按设计使用：

| 未纳入 | 原因 |
|---|---|
| `fever`、`fever_support` | 拿到的 claim 文件没有 wiki 证据句，state 只能是 claim 本身 —— 量到的是闭卷事实核查，而不是这个架构要做的"带证据核验" |
| `mmlu_pro` | 官方只有 70 条带标签验证样本，训练集只能从它自己的测试文件里挖，不构成诚实的 train/test 对 |
| `banking77` | 只给了 770 行 TF-IDF testset，完全没有训练集 |

想要恢复：把对应来源放回 `data/raw/<名字>/` 后重跑
`python -m qwenjev.cli normalize --src data/raw --out data/ready`
（FEVER 还需要 wiki dump 生成的 `evidence.jsonl`）。这些 `*_train.jsonl` 的构造规则见
[`scripts/build_extra_splits.py`](scripts/build_extra_splits.py)。
</details>

## Speed

单位说清楚：Laya **一次调用答一个问题**并为每个问题重新编码 state；本引擎**一次调用答完同一个
state 上的所有问题**（先共享预填，再并行分支）。实测（RTX 3090，torch 2.6+cu126，未装融合算子）：

| 场景 | 本引擎 | Laya |
|---|---|---|
| 26 项全量评测（每次调用 16 个 item） | 2.5 s/次调用（约 68 决策）→ **36.7 ms/决策** | 23 ms/次调用 → **23.4 ms/决策** |
| 1 个问题（15 选项、短 state） | 94 ms | ~25 ms |
| 同一 state：1 / 8 / 64 个问题 | 297 / 283 / 1511 ms **每次调用** | 23 / 193 / 1496 ms |
| 换成 6000 字符的长 state | 461 / 297 / 1655 ms 每次调用 | 26 / 191 / 1621 ms |

读法：本引擎的**每次调用成本几乎是平的**（state 只预填一次，每多一个问题约 +20 ms），Laya 是线性的；
墙钟在"一个 state 约 64 个问题"时打平，之后本引擎更省。按决策摊我们慢 1.0–1.6 倍。融合算子
（`flash-linear-attention` + `causal-conv1d`）能显著改变这个比例，但在本机装不了：`causal-conv1d`
没有 Windows 轮子且缺 `nvcc`、`flash-linear-attention` 依赖 Triton（`torch.compile` 同样依赖它）。
Linux 上 `pip install flash-linear-attention causal-conv1d` 即可打开快路径；
`scripts/speed_compare.py` 可复现上表。

## Layout

```
qwenjev/           库：schema / prompt / tokenize / readout / engine / rlcd / calibration /
                   datasets / normalize / relevance / backends / benchmark / probes / api / cli
train.py           用 data/ready 里的 train split 训练一个读出（带进度条）
test.py            在 data/ready 的 test split 上评测 Laya + 两种读出（带进度条）
demo.py serve.py   面向用户的命令行演示 / 网页演示
scripts/           build_extra_splits / compose_wide_rows / interpolate / dev_score /
                   map_labels / mmlu_to_jsonl / fever_evidence / split_csv /
                   compare_results / speed_compare
models/            交付读出 + 组成它的三个源文件（models/README.md）
data/ready/        统一格式的 train/test JSONL + manifest.json（在仓库里，约 92 MB）
data/raw/          原始数据（gitignore；删除前的备份在 data/raw_backup/）
artifacts/         RESULTS_CN.md（最终结果）、results.json（原始评测数据）、extra_splits.json
tests/             92 个测试 + 数据集适配器用的 mini fixtures
```

## 引用 / References

<!-- 作者填写：本工作引用与致谢的对象（论文、模型、数据集、工具）。 -->
