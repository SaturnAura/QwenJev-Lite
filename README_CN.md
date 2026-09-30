<div align="center">

# QwenJev-lite

**一种在 Qwen 上使用 RLCD 的 JEV-like 模型**

![python](https://img.shields.io/badge/python-3.11-3776ab?logo=python&logoColor=white)
![torch](https://img.shields.io/badge/torch-2.6.0%2Bcu126-ee4c2c?logo=pytorch&logoColor=white)
![transformers](https://img.shields.io/badge/transformers-5.14-ffd21e)
![tests](https://img.shields.io/badge/tests-92%20passing-2ea44f)
![license](https://img.shields.io/badge/license-MIT-blue)

[快速开始](#快速开始) · [结果](#结果) · [交付模型](#交付模型) · [工作原理](#工作原理) · [RLCD 简析](#rlcd-简析) · [对比 BERT](#与-bert-base-分类器的区别) · [改编与数据](#我们对模型与数据做了哪些改编) · [路径与格式](#路径与数据格式) · [数据](#数据) · [目录结构](#目录结构) · [引用](#引用--references) · [TODO](#todo)

</div>

> 英文版见 [`README.md`](README.md)。

QwenJev-lite 将 Transformer 形式化为一个**类型化决策模型**：共享状态仅编码一次，每个问题构成一条相互隔离的分支，推理末端输出的并非生成文本，而是**允许答案上的概率分布**。问题具有类型 —— 有限选择 `choice`、是/否 `bool`、有序打分 `score` —— 同一 state 上可一次性完成多种类型的提问。

骨架为 **Qwen3.5-4B**（32 层：24 层线性注意力 + 8 层全注意力，4.54B 参数），训练**仅更新末端一个 512 行的决策头**（约 5 MB），因此整轮实验可在单张 24 GB 显卡上完成。在 26 个测试集上总体 **0.772**，Qwen 基线 **0.722**，对比基线 Laya **0.446**。

```bash
pip install -r requirements.txt
export QWENJEV_MODEL=/path/to/qwen3.5-4B      # Windows: $env:QWENJEV_MODEL="D:\qwen3.5-4B"
python demo.py --model-dir models/qwenjev-multitask-v2
```

## 结果

完整评测命令为 `python test.py --variants laya qwen_zeroshot qwen_trained --limit 0`，在 26 个三个变体均可作答的测试集上进行（bool 7 + choice 12 + score 7）。表中数值为准确率；原始数据见 [`artifacts/results.json`](artifacts/results.json)，逐任务明细见 [`artifacts/RESULTS_CN.md`](artifacts/RESULTS_CN.md)。

| 题型/准确率 | Laya（对比基线） | Qwen 基线 | **我们的训练模型** |
| --- | --- | --- | --- |
| 判断 `bool`（7 项） | 0.520 | 0.717 | **0.794** |
| 单选 `choice`（12 项） | 0.511 | 0.749 | **0.760** |
| 打分 `score`（7 项） | 0.258 | 0.681 | **0.771** |
| **总体（26 项）** | **0.446** | 0.722 | **0.772** |

## 快速开始

```bash
pip install -r requirements.txt          # torch 2.6 + transformers 5.14；跑测试不需要 GPU/模型/数据

export QWENJEV_MODEL=/path/to/qwen3.5-4B # Windows: $env:QWENJEV_MODEL="D:\qwen3.5-4B"

python demo.py --model-dir models/qwenjev-multitask-v2   # 一个 state，三种题型，一次前向答完
python serve.py                                          # 同一件事的网页版（:8300）
pytest -q                                                # 92 个测试，微型骨架，纯 CPU
```

| 目标 | 命令 |
| --- | --- |
| 跑全量基准（Laya + Qwen 基线 + 我们的模型，带进度条） | `python test.py --variants laya qwen_zeroshot qwen_trained --limit 0` |
| 训练自定义决策头 | `python train.py --prototype-init --items-per-label 12 --min-items-per-task 200` |
| 由三个源文件重建交付的决策头 | [`models/README.md`](models/README.md#rebuild-the-shipped-head) |
| 将原始数据转为统一格式 | `python -m qwenjev.cli normalize --src data/raw --out data/ready` |

## 交付模型

`models/qwenjev-multitask-v2/readout.pt` 为交付的决策头：**512 行 = 27 行保留 + 485 行私有**，覆盖 `data/ready` 中的 90 个标签空间。

## 工作原理

推理末端不进行解码：给定分支决策位上的隐藏状态 `h`，决策头以 `z = W h + b` 在 K 个允许答案上计算得分并施加 softmax，直接输出概率分布（[`qwenjev/readout.py`](qwenjev/readout.py) 中的 `ReservedLabelReadout` / `SlotHeadReadout` / `PointerReadout`）；其中 `ReservedLabelReadout` 将预训练 LM 头在选项标签 token 上的概率质量重新归一化，因此在 K ≤ 26 时无需任何训练即可作答。问题分为三种类型 —— 有限选择、是/否、有序打分（[`qwenjev/schema.py`](qwenjev/schema.py)）；所有选项作为有序列表整体置于分支中，决策头读取列表**之后**的决策位（[`qwenjev/prompt.py`](qwenjev/prompt.py)、[`qwenjev/tokenize.py`](qwenjev/tokenize.py)）。共享 state 仅编码一次并写入 KV/递推缓存，随后按分支展开，每条分支仅可见该 state 与自身的后缀（[`qwenjev/engine.py`](qwenjev/engine.py)）；分支在批大小与 token 预算约束下打包进单次前向（`_chunk_branches`，其中 `share_state=False` 为"每条分支重算 state"的参考实现，用作差分测试），并受"每分支 32,768 / 每请求 65,536 token、state 仅计一次"的上限约束（`JevLimits`、`check_limits`、`account`）。训练采用 RLCD：backbone 全程冻结，以 proper scoring rule 从结果中训练分布（[`qwenjev/rlcd.py`](qwenjev/rlcd.py)）；校准与置信度基于可靠性分箱、ECE、Brier 与 Wilson 区间，`confidence` 为算术量（[`qwenjev/calibration.py`](qwenjev/calibration.py)、[`qwenjev/confidence.py`](qwenjev/confidence.py)）。同一套评测代码同时驱动本引擎与 Laya 两个后端（[`qwenjev/backends.py`](qwenjev/backends.py)）。

## RLCD 简析

**RLCD**（Reinforcement Learning for Calibrated Decisions）的核心在于：**从结果而非固定标签集进行训练**。其思想可概括为一句话：*一个决策即"该问题所允许答案"上的一个分布，因此应以一个仅当分布正确时才取得最小值的目标函数来训练它*。

**决策头。** 给定 state 的隐藏状态 `h ∈ R^d`（位于分支决策位）与该问题所允许的 `K` 个答案，模型为每个答案给出一个分数并归一化：

```
z_k = w_k · h + b_k                     (k = 1 … K)
p_k = softmax(z / τ)_k = exp(z_k/τ) / Σ_j exp(z_j/τ)
```

`w_k` 为决策头的一行 —— **一行对应一个"允许的答案"，而非一个位置**；`τ` 为温度（默认为 1.0，仅在独立校准集上拟合时改变）。

**目标函数。** 设 `y` 为实际发生的答案，RLCD 最小化预测分布的 **proper scoring rule**：

```
log loss :  L = −log p_y = −z_y/τ + log Σ_j exp(z_j/τ)
Brier    :  L = Σ_k (p_k − 1{y = k})²
```

二者均为**严格 proper**：其期望意义上的极小值点**恰好**位于 `p = P(y | state, question)`，因此训练得到的是概率本身，而非仅"分数更高"。梯度同样直观：

```
∂L/∂z_k = p_k − 1{y = k}      （log loss，τ = 1）
```

即：每条样本将"实际发生"的那一行推高，并按当前概率占比将其余各行推低。

**校准为何关键。** 仅追求 argmax 正确的分类器可以任意过信；此处过信会被损失直接惩罚，并以 ECE 检验 —— 在 26 项基准上，我们的训练模型平均 ECE 为 **0.125**，Qwen 基线 0.158、Laya 0.289，同时准确率更高。

**为何带有 reinforcement 的成分。** 监督信号为"决策 + 结果"（`state → 问题 → 答案 → 实际发生`），被拟合的参数为"答案对应的行"，且**未训练过的标签空间仍然可用**（读回保留行），因此同一目标可将已部署模型扩展至新的答案集合，而无需为每个标签集重新构建 head。本仓库实现的是该目标的**有监督形式**：backbone 冻结，以 log loss 或 Brier 拟合决策头（`train.py --objective log_loss|brier`）；在显存允许时，同一损失亦可带上 backbone 的梯度。

**由实验得出的两个结构细节**（均属于训练配方的一部分）：

- **每个 `(问题, 标签空间, 答案)` 独占一块私有行。** 若行按位置共享，训练 2–4 选项的任务会改写 10–15 选项任务所读取的行（同一任务上共享行为 0.733，改为私有行后为 0.760）；未出现过的答案读回预训练行，因此未训练的标签空间**逐位保持零样本行为**。
- **每步之后将行投影回预训练的对数尺度**（`‖w‖ ≈ 0.74`）。决策态范数约为 157，若不投影，lr=1e-3 下单步即可将新行推离约 1/5 的"合适长度"并导致对数爆炸（实测训练损失 33，而均匀分布仅为 5.01）；交付的共享行另作 α=0.5 回拉（`W(0.5) = W₀ + 0.5·(W_trained − W₀)`），宽于保留行的标签空间则改用上述闭式原型行，因为 150 类、每类约 20 条样本的交叉熵无法在一个 epoch 的梯度步内收敛。

## 与 BERT-base 分类器的区别

将 Transformer 用作分类器的常见做法是 "BERT-base + 一个线性头"：编码器约 110M 参数、12 层，仅将池化后的 `[CLS]` 状态送入一个类别数在建模时即固定的 head，因此一次前向仅回答一个问题；每新增一组答案都需新 head、重新训练并为每个任务保存一个 checkpoint，其 softmax 通常过信且从不评估，唯一优势是单次开销较低。QwenJev-lite 采用 causal Transformer（4.54B、32 层：24 层线性注意力 + 8 层全注意力）：state 仅编码一次，每个问题构成一条相互隔离的分支并在一批内完成，答案来自该分支决策位上的状态与 `K` 行（一行对应一个允许答案，`K` 可随请求变化），因此同一决策头可应对任意标签空间（保留行提供零样本能力，训练后的私有行覆盖已训练标签空间），一个请求中可对同一 state 同时提出 `choice` + `bool` + `score`。代价形态亦不同：每个 state 需一次预填（4.54B 模型），此后同一 state 每增加一个问题约增加 20 ms；且分布**本身**即为训练对象，以 ECE / Brier / NLL 评估，全程不进行生成，决策头直接取代了解码循环。概括而言：BERT 分类器回答"这属于我 N 个类中的哪一个"，其 head 必须在训练前存在；QwenJev-lite 回答"这些 K 个答案中是哪一个"，答案集合是请求的一部分，同一 state 上的多个问题相互隔离，并给出**校准后的分布**而非单一 argmax。

## 我们对模型与数据做了哪些改编

**模型上。** 我们以决策头取代解码循环与固定分类头 —— 对允许答案计算 `z = W h + b` 并施加 softmax，不生成 token，延迟不随输出长度增长。保留标签行优先：Qwen 基线将预训练头在选项标签 token 上的质量重新归一化，无需训练即可使用（K ≤ 26），同时为我们的训练模型提供初始化与未见过答案的回落。共享预填与分支隔离：state 仅编码一次并写入 KV/递推缓存，缓存按分支展开，每条分支仅注意 state 与自身后缀，隔离具有结构性而非依赖提示词约定。分支按批调度：在批大小与 token 预算下打包进单次前向，因此一个请求通过数次前向即可完成一个 state 上的全部问题。选项按列表整体渲染：所有选项作为有序列表置于分支中，决策头读取列表**之后**的决策位，因此答案可取决于整个候选集合，而非孤立评估每个候选。类型化问题共用同一线格式：`choice` / `bool` / `score` 的训练器、评测器、CLI、HTTP 接口与两个后端读取同一条记录。一行对应一个答案：决策头的行由 `(问题 id, 标签空间, 答案键)` 决定，因此同一组选项改变顺序仍读取同一行，行亦不会跨问题混用。训练配方为 backbone 冻结 + RLCD（log loss 或 Brier）、每步将行投影回预训练对数尺度、共享行作 α=0.5 回拉、宽于保留预算的标签空间使用闭式原型行、可选地在留出集上拟合温度。

**数据上。** 每个数据集对应一条统一记录：所有来源均归一化为引擎自身的请求结构（`{"state": …, "questions": {…}, "targets": {…}}`，一行一个 JSON），因此同一份文件可同时用于训练、评测、CLI 与 HTTP 接口，不存在任何按数据集划分的代码路径。一切均表达为类型化问题：一个任务或为带 `criteria` 的 `choice`，或为 `criteria` 可省的 `bool`，或为有序 `score`；是/否题同时提供 `instructions`（问句）与 `claim`（陈述句）两种措辞，因为两个后端分别在不同的措辞上被测量。标签空间属于数据而非代码：`criteria` 的键即答案身份，决策头将 `(问题 id, 标签空间, 答案键)` 映射到一行，这正是"选项顺序不影响结果"与"多任务共用一个决策头却互不干扰"的原因。切分纪律：每条训练行与其被评测的行不重叠，截断取样在整个文件上均匀抽取而非取前缀，本仓库自行构建的训练集（`artifacts/extra_splits.json`）在写盘前会做校验 —— 若测试集需要读取的某个答案未被训练到，脚本将拒绝写入。预算按标签空间分配：`--items-per-label` 保证 150 类这类多答案任务不会被 2 答案任务挤占。精度：backbone 以 bf16 运行，决策头保持 float32（训练步长约 1e-3，低于 bf16 在该量级上的分辨率），对数亦在 float32 中计算。

## 路径与数据格式

所有路径均可配置，代码中不含任何硬编码的绝对路径：

| 路径 | 参数 | 环境变量 | 默认 |
| --- | --- | --- | --- |
| 骨架模型 | `--model` | `QWENJEV_MODEL` | `./qwen3.5-4B`（也可以是 Hub repo id） |
| 数据目录 | `--data-dir` | — | `data/ready` |
| 决策头输出 | `--model-dir` / `--out` | — | `models/qwenjev-multitask-v2` |
| Laya 基线 | `--laya-path` | `QWENJEV_LAYA` | `./laya` |

一个数据目录中，每个任务包含两个文件：`<task>_train.jsonl` 与 `<task>_test.jsonl`（`<task>_<split>.jsonl` 亦可），每行是一条**决策记录** —— 引擎的请求结构加上实际结果。以下按三种题型各给出一例。

单选（`choice`）：周末群聊中确定次日行程。

```json
{"id": "weekend/train/0001", "dataset": "weekend", "split": "train",
 "state": "周五晚上，三个人在群里敲定明天去哪：阿哲想去爬山，小雨想去看展，我说都行。",
 "questions": {
   "plan": {"type": "choice", "instructions": "明天去做什么？",
            "criteria": {"hike": "爬山", "museum": "看展", "home": "在家休息"}}},
 "targets": {"plan": "museum"}}
```

是/否（`bool`）：外卖送到，判断这一餐是否重口。

```json
{"id": "takeout/train/0002", "dataset": "takeout", "split": "train",
 "state": "外卖到了，一边是麻辣香锅，一边是清汤面。小雨这两天胃不舒服，手却先伸向了香锅。",
 "questions": {
   "spicy": {"type": "bool", "instructions": "这一顿对现在的她来说重口吗？"}},
 "targets": {"spicy": "yes"}}
```

打分（`score`）：散场后为影片评级。

```json
{"id": "movie/train/0003", "dataset": "movie", "split": "train",
 "state": "电影散场：阿哲说这是他今年看过最好的片子，小雨在旁边打哈欠，我夹在中间。",
 "questions": {
   "rating": {"type": "score", "instructions": "这部电影好看吗？",
              "criteria": {"bad": "难看", "ok": "还行", "good": "好看"}}},
 "targets": {"rating": "ok"}}
```

需要遵守的规则：`targets` 的取值必须是同一 `criteria` 的键；`choice` / `score` 必须提供 `criteria`（有序），`bool` 可省略（默认为是/否）；问题 id（`plan`…）仅为标识符 —— 真正决定决策头使用哪一行的，是 `(问题 id, 标签空间, 答案键)`。

**训练自定义决策头**（可更换骨架、更换数据、指定输出目录）：

```bash
python train.py \
  --model "$QWENJEV_MODEL" \
  --data-dir data/ready \
  --model-dir models/my-run \
  --objective log_loss \
  --prototype-init --items-per-label 12 --min-items-per-task 200 \
  --epochs 0                     # 0 = 只用闭式原型行；要梯度训练就写 1 并配合 --lr
```

**评测**（未安装 Laya 时可省略 `--variants laya`）：

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

`data/ready/` 随仓库提供（约 92 MB 的 JSONL）：统一格式的 train/test 集，以及 `manifest.json`（记录每个 split 的来源文件、条数与注意事项）。所用来源如下：

| 来源 | 提供哪些任务 |
| --- | --- |
| MultiNLI / SNLI | `mnli`、`mnli_ood`、`snli`（一个前提配多条假设） |
| Jigsaw Toxic Comment | `jigsaw`（6 个独立是/否）、`jigsaw_severity`（有序打分） |
| GoEmotions | `goemotions`（28 个情感是/否）、`goemotions_sentiment`（有序打分） |
| TruthfulQA | `truthfulqa`（5 选 1） |
| IntentGrasp | `intentgrasp`（选项随样本给出） |
| CLINC150 / HWU64 | `clinc150`、`clinc150_top15`、`hwu64`、`hwu64_top15` |
| BEIR（arguana / nfcorpus / scidocs / scifact / trec-covid） | 15 个相关性判别任务：用 query 与标注答案构造正样本选项、非 qrel 文档构造负样本选项，按 query 切分训练/测试 |

`artifacts/extra_splits.json` 记录本仓库**自行补齐**的训练集（来源、条数，以及"测试集要读的每个答案都被训练到"的校验），`data/ready/manifest.json` 记录其余 split 的出处。这些数据集、标注与相关工具的作者与维护者使本工作成为可能，谨此致谢 —— 引用信息见文末。

## 目录结构

```
qwenjev/           库：schema / prompt / tokenize / readout / engine / rlcd / calibration /
                   datasets / normalize / relevance / backends / benchmark / probes / api / cli
train.py           用数据目录里的 train split 训练一个决策头（带进度条）
test.py            在 test split 上评测 Laya + Qwen 基线与我们的模型（带进度条）
demo.py serve.py   面向用户的命令行演示 / 网页演示
scripts/           build_extra_splits / compose_wide_rows / interpolate / dev_score /
                   map_labels / compare_results / speed_compare / split_csv
                   （另外还有几个把其他镜像重建回统一格式的转换脚本）
models/            交付的决策头 + 组成它的三个源文件（见 models/README.md）
data/ready/        统一格式的 train/test JSONL + manifest.json（在仓库里，约 92 MB）
data/raw/          原始数据（gitignore）
artifacts/         RESULTS_CN.md（最终结果）、results.json（原始评测数据）、
                   extra_splits.json（补齐训练集的出处）
tests/             92 个测试 + 数据集适配器用的 mini fixtures
```

## 引用 / References

<!-- 作者填写：本工作引用与致谢的对象（论文、模型、数据集、工具）。 -->

## TODO

以下为后续工作方向，本版本尚未实现：

- TODO: **支持更换骨架，包括多模态骨架。** 引擎仅向骨架索取一项信息 —— 每个决策位置的隐藏状态；任何能够输出隐藏状态的模型均可作为骨架。更换为其他 causal LM，或更换为多模态编码器（图像/音频 token 与文本一同置于 `state`，或置于某条分支的 `context`）时，决策头、训练、评测与 HTTP 接口均无需改动，仅需通过 `--model` 指定（本地路径或 Hub repo id），详见[路径与数据格式](#路径与数据格式)。
- TODO: **采用 flash attention 等方式加速推理。** 当前在纯 torch 回退路径上，推理速度尚未超过 Laya 基线：按单次决策计算，本引擎耗时约为 Laya 的 1.0–1.6 倍，仅在单个 state 上的问题数超过约 64 条后墙钟耗时才开始领先。我们正在引入 `flash-linear-attention` 与 `causal-conv1d` 等融合算子以开启线性注意力的快路径，目标是在长序列、多问题场景下将每次调用的耗时降至 Laya 以下；该路径的对比脚本为 `scripts/speed_compare.py`。
