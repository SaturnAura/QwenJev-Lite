<div align="center">

# QwenJev-Lite

**共享状态只编码一次 · 问题分支互相隔离 · 末端直接输出概率分布（不生成文本）**

![python](https://img.shields.io/badge/python-3.11-3776ab?logo=python&logoColor=white)
![torch](https://img.shields.io/badge/torch-2.6.0%2Bcu126-ee4c2c?logo=pytorch&logoColor=white)
![transformers](https://img.shields.io/badge/transformers-5.14-ffd21e)
![tests](https://img.shields.io/badge/tests-92%20passing-2ea44f)
![license](https://img.shields.io/badge/license-MIT-blue)

[快速开始](#快速开始) · [结果](#结果) · [交付模型](#交付模型) · [工作原理](#工作原理) · [RLCD](#rlcd-一页说清) · [对比-bert](#与-bert-base-分类器的区别) · [路径与格式](#路径与数据格式) · [数据](#数据) · [速度](#速度) · [引用](#引用--references)

</div>

QwenJev-lite 把 Transformer 变成一个**类型化决策模型**：共享状态只编码一次，每个问题是一条
互相隔离的分支，推理末端给出的不是生成的文本，而是**允许答案上的概率分布**。问题是有类型的
——有限选择 `choice`、是/否 `bool`、有序打分 `score`——同一个 state 上可以一次问完多种类型。

骨架是 **Qwen3.5-4B**（32 层：24 层线性注意力 + 8 层全注意力，4.54B 参数），训练**只动一个
512 行的读出**（约 5 MB），所以整轮实验在一张 24 GB 显卡上就能跑完。在 26 个测试集上总体
**0.772**，未训练读出 **0.722**，对比基线 Laya **0.446**。

```bash
pip install -r requirements.txt
export QWENJEV_MODEL=/path/to/qwen3.5-4B      # Windows: $env:QWENJEV_MODEL="D:\qwen3.5-4B"
python demo.py --model-dir models/qwenjev-multitask-v2
```

> 英文版见 [`README.md`](README.md)。

## 声明 / Scope

* **骨架可换，多模态也行。** 引擎只向骨架要一样东西：每个决策位置的隐藏状态。任何能产出隐藏
  状态的模型都能当骨架 —— 换一个 causal LM，或者换多模态编码器（图像/音频 token 直接和文本
  一起放进 `state`，或放进某条分支的 `context`），读出、训练、评测、HTTP 接口一行都不用改；
  换骨架只是 `--model`（本地路径或 Hub repo id），见[路径与数据格式](#路径与数据格式)。
* **只训读出。** backbone 全程冻结，整轮基准评测 14 分钟，交付产物 5 MB。
* **本文所有数字都能从这个仓库复现**：模型文件、统一格式的数据（`data/ready/`）、原始评测输出
  （`artifacts/results.json`）以及生成二者的脚本。

## 结果

一次完整评测：`python test.py --variants laya qwen_zeroshot qwen_trained --limit 0`，跑在 26 个
三个变体都能作答的测试集上（bool 7 + choice 12 + score 7）。原始数据
[`artifacts/results.json`](artifacts/results.json)，逐任务明细
[`artifacts/RESULTS_CN.md`](artifacts/RESULTS_CN.md)。

| 题型 | Laya（对比基线） | 零样本读出 | **训练后读出** |
|---|---|---|---|
| 判断 `bool`（7 项） | 0.520 | 0.717 | **0.794** |
| 单选 `choice`（12 项） | 0.511 | 0.749 | **0.760** |
| 打分 `score`（7 项） | 0.258 | 0.681 | **0.771** |
| **总体（26 项）** | **0.446** | 0.722 | **0.772** |

平均 ECE（越低越好）：Laya 0.289 / 零样本读出 0.158 / **训练后读出 0.125**。

**超出 26 个保留行的标签空间**单列，因为零样本读出没有可用的行、完全无法作答：

| 标签空间 | 训练后读出 | Laya | 零样本读出 |
|---|---|---|---|
| CLINC150（150 类意图） | 0.527 | 0.545 | 无法作答 |
| HWU64（64 类意图） | 0.540 | 0.345 | 无法作答 |

收益来自哪、不来自哪，一并说清：训练后读出在相关性判别上大幅领先
（`scifact_rel_choice` 1.000、`trec-covid_rel_choice` 0.900、`arguana_rel_bool` 0.803），并把两个
宽标签空间从"无法作答"变成 0.53–0.54；但在两个 15 类意图子集（0.733 / 0.467）和部分闭卷型
判断上仍低于零样本读出 —— 这是"保留行只覆盖前 26 个答案"的直接代价。

## 快速开始

```bash
pip install -r requirements.txt          # torch 2.6 + transformers 5.14；跑测试不需要 GPU/模型/数据

export QWENJEV_MODEL=/path/to/qwen3.5-4B # Windows: $env:QWENJEV_MODEL="D:\qwen3.5-4B"

python demo.py --model-dir models/qwenjev-multitask-v2   # 一个 state，三种题型，一次前向答完
python serve.py                                          # 同一件事的网页版（:8300）
pytest -q                                                # 92 个测试，微型骨架，纯 CPU
```

| 我想…… | 命令 |
|---|---|
| 跑全量基准（Laya + 两种读出，带进度条） | `python test.py --variants laya qwen_zeroshot qwen_trained --limit 0` |
| 自己训一个读出 | `python train.py --prototype-init --items-per-label 12 --min-items-per-task 200` |
| 用三个源文件重建交付读出 | [`models/README.md`](models/README.md#rebuild-the-shipped-head) |
| 原始数据 → 统一格式 | `python -m qwenjev.cli normalize --src data/raw --out data/ready` |
| 起 HTTP 服务（`POST /v1/decide`） | `python -m qwenjev.cli serve --port 8300` |
| 跑架构自检探针（泄漏、选项交互、延迟…） | `python -m qwenjev.cli probe all` |
| 没有 GPU 也想跑通全流程 | 任意入口加 `--fake`：换成微型随机骨架，几秒钟出结果 |

## 交付模型

`models/qwenjev-multitask-v2/readout.pt` 是交付读出：**512 行 = 27 行保留 + 485 行私有**，
覆盖 `data/ready` 里的 90 个标签空间。

* 第 0–26 行保持**预训练标签行**，因此没见过的答案读到的就是骨架自己的信号；
* 训练覆盖过的每个标签空间各占一块私有行：选项数 ≤ 26 的按位置从训练行复制；选项数 > 26 的用
  **闭式原型行**（该答案对应状态的平均向量，一次前向、无梯度步骤），因为位置行在这里用完了；
* 两者都不命中的答案，回落到保留行。

四个目录（`models/README.md` 有链条图和加载代码；三个源文件拼起来可**逐位复现**交付文件）：

| 目录 | 行数 | 作用 |
|---|---|---|
| `qwenjev-multitask-v2/` | 512 | 交付读出（`card.json` 带本次评测成绩） |
| `shared-rows/` | 256 | 共享训练行（α=0.5 混合后的结果） |
| `raw-trained-rows/` | 256 | α=1 的训练行，用于重扫 α |
| `prototype-rows/` | 1024 | 宽标签空间的闭式原型行 |

## 工作原理

| 部件 | 实现在哪 |
|---|---|
| 推理末端读出：`z = W h + b`，在 K 个允许答案上做 softmax，没有解码循环 | [`qwenjev/readout.py`](qwenjev/readout.py)：`ReservedLabelReadout` / `SlotHeadReadout` / `PointerReadout` |
| 用预训练词表里的保留标签行直接读概率（K ≤ 26，零训练） | `ReservedLabelReadout`：把 LM head 在选项标签 token 上的概率质量重新归一化 |
| 三种问题类型：有限选择、是/否、有序打分 | [`qwenjev/schema.py`](qwenjev/schema.py) |
| 共享状态编码一次、分支互相隔离 | [`qwenjev/engine.py`](qwenjev/engine.py)：state 预填一次进 KV/递推缓存，按分支展开，每条分支只看 state + 自己的后缀 |
| 分支批调度（批大小 + token 预算），一次前向一批 | `_chunk_branches`；`share_state=False` 是"每条分支重算 state"的参考实现，用作差分测试 |
| 请求上限与计费口径（每分支 32,768 / 每请求 65,536 token，state 只计一次） | `JevLimits`、`check_limits`、`account` |
| 选项作为一个有序列表整体读入，读出看列表之后的决策位 | [`qwenjev/prompt.py`](qwenjev/prompt.py)、[`qwenjev/tokenize.py`](qwenjev/tokenize.py) |
| RLCD：用 proper scoring rule 按结果训练分布（backbone 冻结） | [`qwenjev/rlcd.py`](qwenjev/rlcd.py) |
| 校准与置信度：可靠性分箱、ECE、Brier、Wilson 区间；`confidence` 是算术量 | [`qwenjev/calibration.py`](qwenjev/calibration.py)、[`qwenjev/confidence.py`](qwenjev/confidence.py) |
| 两个后端同一接口（本引擎 / Laya），同一套评测代码 | [`qwenjev/backends.py`](qwenjev/backends.py) |

## RLCD 一页说清

**RLCD**（Reinforcement Learning for Calibrated Decisions）的核心是：**从结果而不是从一个固定的
标签集训练**。一句话——*一个决策就是"该问题允许的答案"上的一个分布，那就用一个"只有分布正确时
才最小"的目标去训它*。

**读出。** 给定 state 的隐藏状态 `h ∈ R^d`（分支决策位上的状态）和该问题允许的 `K` 个答案，
模型给每个答案一个分数并归一化：

```
z_k = w_k · h + b_k                     (k = 1 … K)
p_k = softmax(z / τ)_k = exp(z_k/τ) / Σ_j exp(z_j/τ)
```

`w_k` 是读出的一行 —— **一行对应一个"允许的答案"，而不是一个位置**；`τ` 是温度（默认 1.0，
只有在独立校准集上拟合时才会变）。

**目标函数。** 设 `y` 是实际发生的答案，RLCD 最小化预测分布的 **proper scoring rule**：

```
log loss :  L = −log p_y = −z_y/τ + log Σ_j exp(z_j/τ)
Brier    :  L = Σ_k (p_k − 1{y = k})²
```

两者都是**严格 proper** 的：期望意义上的极小值点**恰好**是 `p = P(y | state, question)`，
所以训出来的是概率本身，而不只是"分数更高"。梯度也很直观：

```
∂L/∂z_k = p_k − 1{y = k}      （log loss，τ = 1）
```

即：每一条样本把"发生的事情"那一行推高，把其它行按它们当前认领的概率比例推低。

**为什么"calibrated（校准）"是关键。** 只求 argmax 正确的分类器可以无限过信；这里过信会被损失
直接惩罚，而校准用 ECE 检查。在 26 项基准上，训练后读出的平均 ECE 是 **0.125**，未训练读出
0.158、Laya 0.289，同时准确率还更高。

**它为什么带一点"reinforcement"的味道。** 因为监督信号是"决策 + 结果"
（`state → 问题 → 答案 → 实际发生`），被拟合的参数是"答案对应的行"，而且**没训过的标签空间
仍然可用**（读到保留行），所以同一个目标可以把一个已部署的模型扩展到新的答案集合，而不是每换
一个标签集就要新做一个 head。本仓库实现的是这个目标的**有监督形式**：backbone 冻结，用 log loss
或 Brier 拟合读出（`train.py --objective log_loss|brier`）；同样的损失在有显存的情况下也能带上
backbone 的梯度。

**两个被实验逼出来的结构细节**（都属于训练配方的一部分）：

* **每个 `(问题, 标签空间, 答案)` 一块私有行。** 行按位置共享时，训练 2–4 选项的任务会改写
  10–15 选项任务要读的行（同一任务上共享行 0.733，换成私有行后 0.760）；而没见过的答案读的是
  预训练行，所以未训练的标签空间**逐位保持零样本行为**。
* **每步之后把行投影回预训练的对数尺度**（`‖w‖ ≈ 0.74`）。决策态范数约 157，不投影时 lr=1e-3
  一步就能把新行推走 1/5 的"合适长度"、对数爆炸（实测训练损失 33，而均匀分布只有 5.01）。
  交付的共享行另外做了 α=0.5 回拉（`W(0.5) = W₀ + 0.5·(W_trained − W₀)`）；宽于保留行的标签
  空间改用上面的闭式原型行，因为 150 类、每类约 20 条样本的交叉熵在一个 epoch 的梯度步里推不动。

## 与 BERT-base 分类器的区别

把 Transformer 变成分类器的常见做法是 "BERT-base + 一个线性头"。本项目的选择不同：

| | BERT-base + 线性头 | **QwenJev-lite** |
|---|---|---|
| 骨架 | 编码器，约 110M 参数，12 层 | causal Transformer，4.54B，32 层（24 线性注意力 + 8 全注意力） |
| 一次前向能答几个问题 | 一个 —— `[CLS]` 向量只编码一个问题 | 多个 —— state 编码一次，每个问题一条隔离分支，所有分支一次批处理 |
| 答案从哪来 | 池化后的 `[CLS]` 状态 → 类别数在建模型时就固定的 head | 该分支决策位上的状态 → `K` 行，一行一个允许答案，`K` 可以每个请求都不同 |
| 增加一组答案 | 新 head、重训、每任务一个 checkpoint | 同一个读出能答任意标签空间：保留行零样本，训练过的读私有行 |
| 混用题型 | 每种题型一个分类器，一次前向问一个 | 一个请求里 `choice` + `bool` + `score` 可以问同一个 state |
| 不确定性 | 通常过信的 softmax，且从不评估 | 分布**本身就是**被训练的对象，用 ECE / Brier / NLL 评估 |
| 成本形态 | 单次便宜，一次只答一个问题 | 每个 state 一次预填（4.54B 模型），之后同一 state 每多一个问题约 +20 ms |
| 生成 | 无（纯分类） | 设计上就没有 —— 读出取代了解码循环 |

一句话：BERT 分类器回答"这属于我 N 个类里的哪一个"，头必须在训练前就存在；QwenJev-lite 回答
"这些 K 个答案里是哪一个"，答案集合是请求的一部分，同一个 state 上的多个问题互相隔离，并且给出
的是**校准过的分布**而不是一个 argmax。

## 我们对模型结构做了哪些改编

1. **用读出取代解码循环/固定分类头。** 对允许答案做 `z = W h + b` + softmax，不生成 token，
   延迟不随输出长度增长。
2. **保留标签行优先。** 零样本读出把预训练头在选项标签 token 上的质量重新归一化 —— 不训练就能
   用（K ≤ 26），更重要的是给训练后的读出提供**初始化**和**没见过的答案的回落**。
3. **共享预填 + 分支隔离。** state 只编码一次进 KV/递推缓存，缓存按分支展开，每条分支只注意
   state 和自己的后缀；隔离是结构性的，不是提示词的约定。
4. **分支按批调度。** 在批大小和 token 预算下把分支打包进一次前向，因此一个请求几次前向就能答完
   一个 state 上的所有问题。
5. **选项按列表整体渲染。** 所有选项作为一个有序列表放进分支，读出读的是列表**之后**的决策位，
   因此答案可以取决于整个候选集合，而不是孤立看每个候选。
6. **类型化问题、统一记录。** `choice`/`bool`/`score` 共用一套线格式，训练器、评测器、CLI、
   HTTP 接口和两个后端读的是同一条记录。
7. **一行对应一个答案 + 明确的回落。** 读出的行由 `(问题 id, 标签空间, 答案键)` 决定，所以同一
   组选项换个顺序仍然读同一行，行也不会跨问题串味。
8. **训练配方。** backbone 冻结；RLCD（log loss 或 Brier）；每步把行投影回预训练对数尺度；共享行
   做 α=0.5 回拉；宽于保留预算的标签空间用闭式原型行；可选地在留出集上拟合温度。

## 我们如何组织与优化数据

* **每个数据集一条统一记录。** 所有来源都归一化成引擎自己的请求形状
  （`{"state": …, "questions": {…}, "targets": {…}}`，一行一个 JSON），因此同一份文件同时喂给
  训练、评测、CLI 和 HTTP 接口，没有任何按数据集分的代码路径。
* **一切都表达成类型化问题。** 一个任务要么是 `choice`（带 `criteria`）、要么是 `bool`
  （是/否，`criteria` 可省）、要么是有序 `score`；是/否题同时带 `instructions`（问句）和
  `claim`（陈述句）两种措辞，因为两个后端各自在一种措辞上被测量过。
* **标签空间是数据，不是代码。** `criteria` 的键就是答案身份；读出把
  `(问题 id, 标签空间, 答案键)` 映射到一行 —— 这就是"选项顺序不影响结果"和"多任务共用一个读出
  却不互相干扰"的原因。
* **切分纪律。** 每一条训练行都与它被评测的行不重叠；截断取样是在整个文件上均匀取，而不是取前缀；
  本仓库自己构建的训练集（`artifacts/extra_splits.json`）写盘前会做校验 —— 只要测试集要读的
  某个答案没被训练到，脚本就拒绝写。
* **按标签空间分配预算。** `--items-per-label` 保证 150 类这种多答案任务不会被 2 答案任务挤掉，
  这正是之前选择能力被压平的原因。
* **精度。** backbone 跑 bf16；读出保持 float32（训练步长约 1e-3，低于 bf16 在该量级上的分辨率），
  对数也在 float32 里算。

## 路径与数据格式

所有路径都可配置，代码里没有任何硬编码的绝对路径：

| 路径 | 参数 | 环境变量 | 默认 |
|---|---|---|---|
| 骨架模型 | `--model` | `QWENJEV_MODEL` | `./qwen3.5-4B`（也可以是 Hub repo id） |
| 数据目录 | `--data-dir` | — | `data/ready` |
| 读出输出 | `--model-dir` / `--out` | — | `models/qwenjev-multitask-v2` |
| Laya 基线 | `--laya-path` | `QWENJEV_LAYA` | `./laya` |

一个数据目录里，每个任务两个文件：`<task>_train.jsonl` 和 `<task>_test.jsonl`
（`<task>_<split>.jsonl` 都行）。每行是一条**决策记录** —— 引擎的请求形状加上实际结果：

```json
{"id": "mytask/test/0001", "dataset": "mytask", "split": "test",
 "state": "共享文本：只编码一次，下面所有问题都读它",
 "questions": {
   "queue":  {"type": "choice", "instructions": "Which team should handle this ticket?",
              "criteria": {"payments": "Payout failures", "account": "Login", "other": "Else"}},
   "urgent": {"type": "bool", "instructions": "Does this need urgent attention?"},
   "grade":  {"type": "score", "instructions": "How severe is it?",
              "criteria": {"low": "Low", "mid": "Medium", "high": "High"}}},
 "targets": {"queue": "payments", "urgent": "no", "grade": "high"}}
```

要注意的规则：`targets` 的取值必须是同一条 `criteria` 的键；`choice` / `score` 必须有
`criteria`（有序），`bool` 可以省（默认是/否）；问题 id（`queue`…）只是标识符 ——
真正决定读出用哪一行的是 `(问题 id, 标签空间, 答案键)`。

**训练自己的读出**（换骨架、换自己的数据、输出到自己的目录）：

```bash
python train.py \
  --model "$QWENJEV_MODEL" \
  --data-dir data/ready \
  --model-dir models/my-run \
  --objective log_loss \
  --prototype-init --items-per-label 12 --min-items-per-task 200 \
  --epochs 0                     # 0 = 只用闭式原型行；要梯度训练就写 1 并配合 --lr
```

**评测**（没装 Laya 就省略 `--variants laya`）：

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
    model_path="qwen3.5-4B",                       # 任何本地路径或 Hub repo id
    readout="slot_head",
    readout_checkpoint="models/qwenjev-multitask-v2/readout.pt",
))
print(engine.decide("My payouts have failed three times.", {
    "queue": {"type": "choice", "instructions": "Which team should handle this ticket?",
              "criteria": {"payments": "Payouts", "account": "Login", "other": "Else"}},
    "escalate": {"type": "bool", "instructions": "Does this require urgent attention?"},
}).to_dict())
```

## 数据

`data/ready/` 随仓库提供（约 92 MB 的 JSONL）：统一格式的 train/test 集，外加 `manifest.json`
（记录每个 split 的来源文件、条数与注意事项）。用到的来源：

| 来源 | 提供哪些任务 |
|---|---|
| MultiNLI / SNLI | `mnli`、`mnli_ood`、`snli`（一个前提配多条假设） |
| Jigsaw Toxic Comment | `jigsaw`（6 个独立是/否）、`jigsaw_severity`（有序打分） |
| GoEmotions | `goemotions`（28 个情感是/否）、`goemotions_sentiment`（有序打分） |
| TruthfulQA | `truthfulqa`（5 选 1） |
| IntentGrasp | `intentgrasp`（选项随样本给出） |
| CLINC150 / HWU64 | `clinc150`、`clinc150_top15`、`hwu64`、`hwu64_top15` |
| BEIR（arguana / nfcorpus / scidocs / scifact / trec-covid） | 15 个相关性判别任务：用 query 与标注答案构造正样本选项、非 qrel 文档构造负样本选项，按 query 切分训练/测试 |

`artifacts/extra_splits.json` 记录本仓库**自行补齐**的训练集（来源、条数，以及"测试集要读的每个
答案都被训练到"的校验），`data/ready/manifest.json` 记录其余 split 的出处。这些数据集、标注与
相关工具的作者和维护者让这项工作成为可能，在此一并致谢 —— 引用信息见文末。

## 速度

**下表所有数字都是「纯 torch 回退路径」的结果。** 本机没有 Triton、也没有 CUDA toolkit，
因此 `flash-linear-attention` 与 `causal-conv1d` **都没有安装**：24 层线性注意力跑的是 torch
实现（环境：`torch 2.6.0+cu126`、`transformers 5.14`、RTX 3090 / SM86；另外 8 层全注意力本身
已经走 SDPA）。**本仓库没有任何"带 flash attention"的数字** —— 这里的融合算子装不上，我们宁可
明确标注"这是慢路径"，也不引用一个没实测过的数字。

单位说清楚：Laya **一次调用答一个问题**、并为每个问题重新编码 state；本引擎**一次调用答完同一个
state 上的全部问题**（先共享预填，再并行分支），所以每次调用的成本几乎不随问题数增长。

| 场景 | 本引擎（纯 torch 回退） | Laya |
|---|---|---|
| 26 项全量评测（每次调用 16 个 item） | 2.5 s/次调用（约 68 决策）→ **36.7 ms/决策** | 23 ms/次调用 → **23.4 ms/决策** |
| 1 个问题（15 选项、短 state） | 94 ms | ~25 ms |
| 同一 state：1 / 8 / 64 个问题 | 297 / 283 / 1511 ms **每次调用** | 23 / 193 / 1496 ms |
| 换成 6000 字符的长 state | 461 / 297 / 1655 ms 每次调用 | 26 / 191 / 1621 ms |

读法：本引擎的**每次调用成本几乎是平的**（state 只预填一次，每多一个问题约 +20 ms），Laya 是线性
的；墙钟在"一个 state 约 64 个问题"处打平，之后本引擎更省；在这条回退路径上，按**决策**摊我们慢
1.0–1.6 倍。

融合算子正是那条"线性注意力融合路径"需要的，所谓"1500 个问题几百毫秒"属于那个量级 ——
**装上之后上表会变快，但我们不给没实测过的估计值**。本机装不上的原因：`causal-conv1d` 没有
Windows 轮子、自己编译需要 `nvcc`；`flash-linear-attention` 依赖 Triton（`torch.compile` 同样
依赖它，所以那条路也堵着）。Linux 上 `pip install flash-linear-attention causal-conv1d` 即可打开
快路径；`scripts/speed_compare.py` 能复现上表（任何 state 长度与问题数），等快路径可用时用同一个
脚本直接测即可。

## 目录结构

```
qwenjev/           库：schema / prompt / tokenize / readout / engine / rlcd / calibration /
                   datasets / normalize / relevance / backends / benchmark / probes / api / cli
train.py           用数据目录里的 train split 训练一个读出（带进度条）
test.py            在 test split 上评测 Laya + 两种读出（带进度条）
demo.py serve.py   面向用户的命令行演示 / 网页演示
scripts/           build_extra_splits / compose_wide_rows / interpolate / dev_score /
                   map_labels / compare_results / speed_compare / split_csv
                   ??????????????????????
models/            交付读出 + 组成它的三个源文件（见 models/README.md）
data/ready/        统一格式的 train/test JSONL + manifest.json（在仓库里，约 92 MB）
data/raw/          原始数据（gitignore）
artifacts/         RESULTS_CN.md（最终结果）、results.json（原始评测数据）、
                   extra_splits.json（补齐训练集的出处）
tests/             92 个测试 + 数据集适配器用的 mini fixtures
```

## 引用 / References

<!-- 作者填写：本工作引用与致谢的对象（论文、模型、数据集、工具）。 -->
