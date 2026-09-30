<div align="center">

# QwenJev-lite

**一种在 Qwen 上使用 RLCD 的 JEV-like 模型**

![python](https://img.shields.io/badge/python-3.11-3776ab?logo=python&logoColor=white)
![torch](https://img.shields.io/badge/torch-2.6.0%2Bcu126-ee4c2c?logo=pytorch&logoColor=white)
![transformers](https://img.shields.io/badge/transformers-5.14-ffd21e)
![tests](https://img.shields.io/badge/tests-92%20passing-2ea44f)
![license](https://img.shields.io/badge/license-MIT-blue)

[结果](#结果) · [快速开始](#快速开始) · [交付模型](#交付模型) · [训练与测试](#训练与测试) · [数据](#数据) · [速度](#速度) · [目录结构](#目录结构) · [工作原理](#工作原理) · [RLCD 简析](#rlcd-简析) · [对比 BERT](#与-bert-base-分类器的区别) · [改编与数据](#我们对模型与数据做了哪些改编) · [引用](#引用--references) · [TODO](#todo)

</div>

> 英文版见 [`README.md`](README.md)。

QwenJev-lite 将 Transformer 形式化为一个**类型化决策模型**：共享状态仅编码一次，每个问题构成一条相互隔离的分支，推理末端输出的并非生成文本，而是**允许答案上的概率分布**。问题具有类型 —— 有限选择 `choice`、是/否 `bool`、有序打分 `score` —— 同一 state 上可一次性完成多种类型的提问。

骨架为 **Qwen3.5-4B**（32 层：24 层线性注意力 + 8 层全注意力，4.54B 参数 —— 4.21B 语言 + 0.33B 视觉，骨架本身就是多模态的），训练**仅更新末端一个 512 行的决策头**（约 5 MB），因此整轮实验可在单张 24 GB 显卡上完成。在 26 个测试集上总体 **0.772**，同一批保留标签行**完全不训练**直接当读出时为 **0.722**（`qwen_zeroshot`）。

```bash
pip install -r requirements.txt
export QWENJEV_MODEL=/path/to/qwen3.5-4B      # Windows: $env:QWENJEV_MODEL="D:\qwen3.5-4B"
python demo.py --checkpoint-dir models/qwenjev-multitask-v2
```

## 结果

完整评测命令是 `python test.py --limit 0`，在 26 个两个变体都能作答的测试集上进行（bool 7 + choice 12 + score 7）。表中数值为准确率；原始数据见 [`artifacts/results.json`](artifacts/results.json)，逐任务明细见 [`artifacts/RESULTS_CN.md`](artifacts/RESULTS_CN.md)。

| 题型/准确率 | Laya（参照） | 预训练读出（`qwen_zeroshot`） | **我们训练后的模型** |
| --- | --- | --- | --- |
| 判断 `bool`（7 项） | 0.520 | 0.717 | **0.793** |
| 单选 `choice`（12 项） | 0.511 | 0.749 | **0.760** |
| 打分 `score`（7 项） | 0.258 | 0.681 | **0.771** |
| **总体（26 项）** | **0.446** | 0.722 | **0.772** |

**Laya** 那一列是另一个同类的开放基线，测于同样这 26 个测试集，留在这里只是方便你对照结果。本仓库不安装、也不要求你运行它 —— `test.py` 默认只评测我们自己的两个变体；如果你已经有那份 checkpoint，`--variants laya` 仍然可用。

## 快速开始

```bash
pip install -r requirements.txt          # torch 2.6 + transformers 5.14；跑测试不需要 GPU/模型/数据

export QWENJEV_MODEL=/path/to/qwen3.5-4B # Windows: $env:QWENJEV_MODEL="D:\qwen3.5-4B"

python demo.py --checkpoint-dir models/qwenjev-multitask-v2   # 一个 state，三种题型，一次前向答完
python serve.py                                          # 同一件事的网页版（:8300）
pytest -q                                                # 92 个测试，微型骨架，纯 CPU
```

| 目标 | 命令 |
| --- | --- |
| 跑全量基准（预训练读出 + 我们的模型，带进度条） | `python test.py --limit 0` |
| 训练决策头（四个输入，全部有默认值） | `python train.py --model /path/to/qwen3.5-4B` |
| 由三个源文件重建交付的决策头 | [`models/README.md`](models/README.md#rebuild-the-shipped-head) |
| 将原始数据转为统一格式 | `python -m qwenjev.cli normalize --src data/raw --out data/ready` |

## 交付模型

`models/qwenjev-multitask-v2/readout.pt` 为交付的决策头：**512 行 = 27 行保留 + 485 行私有**，覆盖 `data/ready` 中的 90 个标签空间。

## 训练与测试

一次实验只需要**四个输入**，而且每一个都有默认值：

| 输入 | 参数 | 默认 | 说明 |
| --- | --- | --- | --- |
| base 模型 | `--model` | `$QWENJEV_MODEL`，否则 `./qwen3.5-4B` | 本地路径或 Hub repo id；引擎只读取隐藏状态，所以换其它 causal 骨架、或换成多模态骨架（图像/音频 token 放进 `state`）都可以 |
| 决策头保存/读取地址 | `--checkpoint-dir` | *留空* → `models/qwenjev-run-<时间戳>` | 不存在会自动创建；`train.py` 往里写 `readout.pt` + `card.json`，`test.py` 从里面读 `readout.pt` |
| 数据地址 | `--data-dir` | `data/ready` | 一个目录，里面每个任务一对 `<task>_train.jsonl` / `<task>_test.jsonl`（`<task>_<split>.jsonl` 都行） |
| 超参数 | `--epochs`、`--lr`、`--batch-size`、`--objective`、`--items-per-label`、`--min-items-per-task`、`--prototype-init`、`--optimizer` 等 | 见 `python train.py --help` | `test.py` 另有 `--limit`、`--batch`、`--variants` |

对 `test.py` 来说，`--checkpoint-dir` 留空表示**使用交付的决策头**；如果磁盘上没有这个头，它会自动只评测预训练读出，而不是报错。结果写到 `--out` / `--markdown`（父目录会自动创建），默认 `artifacts/results.json` 与 `artifacts/RESULTS.md`。

**数据格式。** 一个目录、每个任务两个文件。文件是 **JSONL：一行一条记录** —— 一行一个 JSON 对象，**不要美化换行**（多行记录会被拒绝，并提示来看这一节）。一条记录就是引擎的请求形状加上实际结果，三种题型都覆盖。下面三行故意很长，这就是文件里一行的样子。

`choice` 题 —— 群里敲定明天的安排：

```json
{"id": "weekend/train/0001", "dataset": "weekend", "split": "train", "state": "Friday night, three of us are pinning down tomorrow in the group chat: Zhe wants to hike, Yu wants a gallery, and I said either is fine.", "questions": {"plan": {"type": "choice", "instructions": "What should we do tomorrow?", "criteria": {"hike": "Hike", "museum": "Gallery", "home": "Stay in"}}}, "targets": {"plan": "museum"}}
```

`bool` 题 —— 外卖到了，这碗对她现在来说重不重：

```json
{"id": "takeout/train/0002", "dataset": "takeout", "split": "train", "state": "Dinner just arrived: a spicy hotpot on one side, a plain noodle soup on the other. Yu's stomach has been off all week, but her chopsticks went for the hotpot first.", "questions": {"spicy": {"type": "bool", "instructions": "Is this bowl heavy for her right now?"}}, "targets": {"spicy": "yes"}}
```

`score` 题 —— 散场时给这部电影打分（有序等级，低到高）：

```json
{"id": "movie/train/0003", "dataset": "movie", "split": "train", "state": "The credits roll: Zhe calls it the best film he has seen all year, Yu is yawning next to him, and I am stuck in the middle.", "questions": {"rating": {"type": "score", "instructions": "How good was the film?", "criteria": {"bad": "Bad", "ok": "Fine", "good": "Great"}}}, "targets": {"rating": "ok"}}
```

需要遵守的规则：`targets` 的取值必须是该问题 `criteria` 的键之一；`choice` 与 `score` 必须给 `criteria`（有序 —— `score` 的声明顺序就是刻度，请从最低到最高排列），`bool` 可以省略、默认是 yes/no；问题 id（`plan`、`rating` …）只是标识符，真正决定决策头读哪一行的是 `(问题 id, 标签空间, 答案键)`。

**训练** —— 上面四个输入，保存地址留空即自动创建：

```bash
python train.py \
  --model /path/to/qwen3.5-4B \
  --data-dir data/ready \
  --epochs 1 --lr 1e-3 --batch-size 8 \
  --objective log_loss \
  --items-per-label 12 --min-items-per-task 200 --prototype-init
# 也可以自己指定这次运行：
#   --checkpoint-dir models/my-run            （不存在会自动创建）
#   --epochs 0                           （0 = 只用闭式原型行，不做梯度步）
```

开始前脚本会打印解析后的 base 模型、数据目录、保存位置和超参数，结束后把 `readout.pt` + `card.json` 写进保存位置。

**评测** —— 同样三个路径，加上输出文件（父目录会自动创建）：

```bash
python test.py \
  --model /path/to/qwen3.5-4B \
  --data-dir data/ready \
  --checkpoint-dir models/my-run \
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

`data/ready/` 随仓库提供（约 97 MB 的 JSONL）：统一格式的 train/test 集，外加 `manifest.json`，记录每个 split 的来源文件、条数与注意事项。用到的来源：

| 来源 | 提供哪些任务 |
| --- | --- |
| MultiNLI / SNLI | `mnli`、`mnli_ood`、`snli`（一个前提配多条假设） |
| Jigsaw Toxic Comment | `jigsaw`（6 个独立是/否）、`jigsaw_severity`（有序打分） |
| GoEmotions | `goemotions`（28 个情感是/否）、`goemotions_sentiment`（有序打分） |
| TruthfulQA | `truthfulqa`（5 选 1） |
| IntentGrasp | `intentgrasp`（选项随样本给出） |
| CLINC150 / HWU64 | `clinc150`、`clinc150_top15`、`hwu64`、`hwu64_top15` |
| BEIR（arguana / nfcorpus / scidocs / scifact / trec-covid） | 15 个相关性判别任务：用 query 与标注答案构造正样本选项、非 qrel 文档构造负样本选项，按 query 切分训练/测试 |

`artifacts/extra_splits.json` 记录本仓库**自行补齐**的训练集（来源、条数，以及"测试集要读的每个答案都被训练到"的校验），`data/ready/manifest.json` 记录其余 split 的出处。这些数据集、标注与相关工具的作者和维护者让这项工作成为可能，在此一并致谢 —— 引用信息见文末。

## 速度

**下表所有数字都是「纯 torch 回退路径」的结果。** 本机没有 Triton、也没有 CUDA toolkit，因此 `flash-linear-attention` 与 `causal-conv1d` **都没有安装**：24 层线性注意力跑的是 torch 实现（环境：`torch 2.6.0+cu126`、`transformers 5.14`、RTX 3090 / SM86；另外 8 层全注意力本身已经走 SDPA）。**本仓库没有任何"带 flash attention"的数字** —— 这里的融合算子装不上，我们宁可明确标注"这是慢路径"，也不引用一个没实测过的数字。

**单位说清楚：** 本引擎**一次调用答完同一个 state 上的全部问题**（先共享预填，再并行分支），所以每次调用的成本几乎不随问题数增长：

| 场景 | 本引擎（纯 torch 回退） |
| --- | --- |
| 26 项全量评测（每次调用 16 个 item） | 2.5 s/次调用（约 68 决策）→ **36.7 ms/决策** |
| 1 个问题（15 选项、短 state） | 94 ms |
| 同一 state：1 / 8 / 64 个问题 | 297 / 283 / 1511 ms **每次调用** |
| 换成 6000 字符的长 state | 461 / 297 / 1655 ms 每次调用 |

最后两行的单位是**每次调用**：它几乎是平的 —— state 只预填一次，每多一个问题约 +20 ms。融合算子正是那条"线性注意力融合路径"需要的，所谓"1500 个问题几百毫秒"属于那个量级 —— **装上之后上表会变快，但我们不给没实测过的估计值**。本机装不上的原因：`causal-conv1d` 没有 Windows 轮子、自己编译需要 `nvcc`；`flash-linear-attention` 依赖 Triton（`torch.compile` 同样依赖它，所以那条路也堵着）。Linux 上 `pip install flash-linear-attention causal-conv1d` 即可打开快路径；`scripts/speed_compare.py` 能复现上表（任何 state 长度与问题数），等快路径可用时用同一个脚本直接测即可。

## 目录结构

```
qwenjev/           库：schema / prompt / tokenize / readout / engine / rlcd / calibration /
                   datasets / normalize / relevance / backends / benchmark / probes / api / cli
train.py           用数据目录里的 train split 训练一个决策头（带进度条）
test.py            在 test split 上评测预训练读出 + 我们的模型（带进度条）
demo.py serve.py   面向用户的命令行演示 / 网页演示
scripts/           build_extra_splits / compose_wide_rows / interpolate / dev_score /
                   map_labels / compare_results / speed_compare / split_csv
                   （另有一些把别的镜像转成统一格式的转换脚本）
models/            交付的决策头 + 组成它的三个源文件
data/ready/        统一格式的 train/test JSONL + manifest.json（在仓库里，约 97 MB）
data/raw/          原始数据（gitignore）
artifacts/         RESULTS_CN.md（最终结果）、results.json（原始评测数据）、
                   extra_splits.json（补齐训练集的出处）
tests/             92 个测试 + 数据集适配器用的 mini fixtures
```

## 工作原理

**推理末端不进行解码。** 给定分支决策位上的隐藏状态 `h`，决策头以 `z = W h + b` 在 K 个允许答案上计算得分并施加 softmax，直接输出概率分布（[`qwenjev/readout.py`](qwenjev/readout.py) 中的 `ReservedLabelReadout` / `SlotHeadReadout` / `PointerReadout`）。

- **保留标签行让模型在训练之前就能工作。** `ReservedLabelReadout` 把预训练 LM head 在选项标签 token 上的概率质量重新归一化，因此 **K ≤ 26 个答案时零训练即可作答**；同一批行也是训练后决策头的初始化与"没见过的答案"的回落。
- **问题是有类型的**：有限选择、是/否、有序打分（[`qwenjev/schema.py`](qwenjev/schema.py)）。
- **选项按列表整体读入。** 所有选项作为一个有序列表放进分支，决策头读的是列表**之后**的位置（[`qwenjev/prompt.py`](qwenjev/prompt.py)、[`qwenjev/tokenize.py`](qwenjev/tokenize.py)）—— 因此答案可以取决于整个候选集合。
- **共享状态只编码一次。** state 进入 KV/递推缓存，缓存按分支展开，每条分支只注意 state 和自己的后缀；**隔离是结构性的**，不是提示词的约定（[`qwenjev/engine.py`](qwenjev/engine.py)）。
- **分支按批调度**：在批大小和 token 预算下打包进一次前向（`_chunk_branches`）；`share_state=False` 是"每条分支重算 state"的参考实现，用作差分测试。上限为**每分支 32,768 token**、**每请求 65,536 token**，state 只计一次（`JevLimits`、`check_limits`、`account`）。
- **训练用 RLCD**：backbone 冻结，用 proper scoring rule 按结果拟合分布（[`qwenjev/rlcd.py`](qwenjev/rlcd.py)）—— 见 [RLCD 简析](#rlcd-简析)。
- **校准是被测量的，不是被假设的**：可靠性分箱、ECE、Brier、Wilson 区间，`confidence` 保持为算术量（[`qwenjev/calibration.py`](qwenjev/calibration.py)、[`qwenjev/confidence.py`](qwenjev/confidence.py)）。两个后端共用同一套评测代码（[`qwenjev/backends.py`](qwenjev/backends.py)）。

## RLCD 简析

**RLCD**（Reinforcement Learning for Calibrated Decisions）的核心在于：**从结果而非固定标签集进行训练**。其思想可概括为一句话：*一个决策即"该问题所允许答案"上的一个分布，因此应以一个仅当分布正确时才取得最小值的目标函数来训练它。*

**决策头。** 给定分支决策位上的隐藏状态 `h ∈ R^d` 与该问题允许的 `K` 个答案，模型给每个答案一个得分并归一化：

```
z_k = w_k · h + b_k                     (k = 1 … K)
p_k = softmax(z / τ)_k = exp(z_k/τ) / Σ_j exp(z_j/τ)
```

`w_k` 是决策头的一行 —— **一行对应一个"允许的答案"，而不是一个位置**；`τ` 是温度（默认 1.0，只有在独立校准集上拟合时才会变）。

**目标函数。** 设 `y` 为实际发生的答案，RLCD 最小化预测分布的 **proper scoring rule**：

```
log loss :  L = −log p_y = −z_y/τ + log Σ_j exp(z_j/τ)
Brier    :  L = Σ_k (p_k − 1{y = k})²
```

两者都是**严格 proper** 的：期望意义上的极小值点**恰好**是 `p = P(y | state, question)`，所以学到的是概率本身，而不只是"分数更高"。梯度同样直接：

```
∂L/∂z_k = p_k − 1{y = k}      （log loss，τ = 1）
```

即：每条样本把"实际发生"的那一行推高，其余行按它们当前认领的概率比例推低。

**为什么校准是关键。** 只以 argmax 正确与否来评判的分类器可以任意过信；而这里**过信会被损失直接惩罚**，并用 ECE 检查 —— 在 26 项基准上，我们训练后的模型平均 ECE 为 **0.125**，预训练读出为 **0.158**，同时准确率还更高。

**为什么它带有"强化"的成分。** 监督信号是**决策加结果**（`state → 问题 → 答案 → 实际发生`），被拟合的参数是"答案对应的行"，而**没训过的标签空间依然可用**（读回保留行）。因此同一个目标可以把已部署的模型扩展到新的答案集合，而不必为每个标签集重建一个头。本仓库实现的是该目标的**有监督形式**：backbone 冻结，用 log loss 或 Brier 拟合决策头（`train.py --objective log_loss|brier`）；在显存允许时，同样的损失也可以带上 backbone 的梯度。

**两个由实验逼出来的结构细节**（都属于训练配方的一部分）：

- **每个 `(问题, 标签空间, 答案)` 各占一块私有行。** 当行按位置共享时，训练 2–4 选项的任务会覆盖 10–15 选项任务要读的行（同一任务上共享行 0.733，改成私有行后 0.760）；没出现过的答案读回预训练行，因此未训练的标签空间**逐位保持零样本行为**。
- **每步之后把行投影回预训练的对数尺度**（`‖w‖ ≈ 0.74`）。决策态范数约 157，不做投影时 lr=1e-3 一步就能把新行推走约 1/5 的"合适长度"、对数爆炸（实测训练损失 33，而均匀分布只有 5.01）；交付的共享行另外做了 α=0.5 回拉（`W(0.5) = W₀ + 0.5·(W_trained − W₀)`），宽于保留行的标签空间改用上面的闭式原型行，因为 150 类、每类约 20 条样本的交叉熵在一个 epoch 的梯度步里推不动。

## 与 BERT-base 分类器的区别

把 Transformer 当分类器用的常见做法是「BERT-base + 一个线性头」。真正的差别在这几处：

| | BERT-base + 一个线性头 | **QwenJev-lite** |
| --- | --- | --- |
| 骨架 | 编码器，约 110M 参数，12 层 | causal Transformer，4.54B，32 层（24 线性注意力 + 8 全注意力） |
| 一次前向答几个问题 | 一个 —— 池化后的 `[CLS]` 状态只喂给一个头 | 多个 —— state 编码一次，每个问题一条隔离分支，一次批处理答完 |
| 答案从哪来 | 类别数**在建模型时就固定**的头 | 分支决策位上的状态 + **K 行，一行一个允许答案，K 可随请求变化** |
| 增加一组答案 | 新头、重训、每任务一个 checkpoint | **同一个头覆盖任意标签空间**：保留行给零样本能力，私有行覆盖训练过的空间 |
| 混用题型 | 每种题型一个分类器，一次前向问一个 | 一个请求里 `choice` + `bool` + `score` 可以问同一个 state |
| 不确定性 | 通常过信、且**从不评估**的 softmax | **被训练的对象就是这个分布**，用 ECE / Brier / NLL 评估 |
| 成本形态 | 单次便宜，但**一次只答一个问题** | 每个 state 一次预填（4.54B），之后同一 state 每多一个问题约 +20 ms |
| 生成 | 无（纯分类） | 设计上就没有 —— **决策头取代了解码循环** |

一句话：BERT 分类器回答"这属于我 N 个类里的哪一个"，头必须在训练前就存在；QwenJev-lite 回答"这些 K 个答案里是哪一个"，答案集合是请求的一部分，同一个 state 上的多个问题互相隔离，输出的是**校准过的分布**而不是一个 argmax。

## 我们对模型与数据做了哪些改编

**模型方面。** 我们用决策头取代了解码循环与固定分类器 —— 在允许答案上算 `z = W h + b` 再做 softmax，**不生成任何 token**，因此延迟不随输出长度增长。**保留标签行优先**：预训练读出（`qwen_zeroshot`）把预训练头在选项标签 token 上的质量重新归一化，K ≤ 26 时无需训练即可工作，同时为训练后的模型提供初始化与"没见过的答案"的回落。**共享预填 + 分支隔离**：state 只编码一次进 KV/递推缓存，缓存按分支展开，每条分支只注意 state 和自己的后缀，隔离是结构性的而不是提示词的约定。**分支按批调度**：在批大小与 token 预算下打包进一次前向，一个请求几次前向就能答完一个 state 上的全部问题。**选项按列表整体渲染**：所有选项作为一个有序列表放进分支，决策头读列表**之后**的位置，答案可以取决于整个候选集合。**类型化问题共用一套线格式**：训练器、评测器、CLI、HTTP 接口和两个后端读的是同一条记录。**一行对应一个答案**：行由 `(问题 id, 标签空间, 答案键)` 决定，所以同一组选项换个顺序仍读同一行，行也不会跨问题串味。**训练配方**：backbone 冻结 + RLCD（log loss 或 Brier）、每步把行投影回预训练对数尺度、共享行做 α=0.5 回拉、宽于保留预算的标签空间用闭式原型行、可选地在留出集上拟合温度。

**数据方面。** **每个数据集一条统一记录**：所有来源都归一化成引擎自己的请求形状（`{"state": …, "questions": {…}, "targets": {…}}`，一行一个 JSON），因此同一份文件同时服务训练、评测、CLI 和 HTTP 接口，没有按数据集分的代码路径。**一切都表达成类型化问题**：一个任务要么是带 `criteria` 的 `choice`、要么是 `criteria` 可省的 `bool`、要么是有序 `score`；是/否题同时给出 `instructions`（问句）与 `claim`（陈述句）两种措辞，因为两个后端各自在一种措辞上被测量过。**标签空间属于数据而不是代码**：`criteria` 的键就是答案身份，决策头把 `(问题 id, 标签空间, 答案键)` 映射到一行 —— 这就是"选项顺序不影响结果"和"多任务共用一个决策头却不互相干扰"的原因。**切分纪律**：每条训练行都与它被评测的行不重叠，截断取样在整个文件上均匀取而不是取前缀，本仓库自己构建的训练集（`artifacts/extra_splits.json`）写盘前会校验 —— 只要测试集要读的某个答案没被训练到，脚本就拒绝写。**按标签空间分配预算**：`--items-per-label` 保证 150 类任务不会被 2 答案任务挤掉。**精度**：backbone 跑 bf16，决策头保持 float32（训练步长约 1e-3，低于 bf16 在该量级的分辨率），对数也在 float32 里算。

## 引用 / References

<!-- 作者填写：本工作引用与致谢的对象（论文、模型、数据集、工具）。 -->

## TODO

本版本仍有以下工作待完成：

- TODO：**支持更换骨架，包括多模态骨架。** 引擎只向骨架索取一样东西 —— 每个决策位置的隐藏状态 —— 因此任何能产出隐藏状态的模型都可以当骨架。换成另一个 causal LM，或换成多模态编码器（把图像 / 音频 token 与文本一起放进 `state`，或放进某条分支的 `context`），决策头、训练器、评测器和 HTTP 接口都不需要改动，仅通过 `--model`（本地路径或 Hub repo id）选择；见[训练与测试](#训练与测试)。
- TODO：**用 flash attention 及相关方法加速推理。** 在纯 torch 回退路径上，单条决策约 37 ms，并且只有在同一个 state 上问超过约 64 个问题时每次调用的优势才体现出来。我们正在引入 `flash-linear-attention`、`causal-conv1d` 之类的融合算子来打开线性注意力的快路径，目标是在长序列、多问题的场景下把每次调用的延迟降下来；`scripts/speed_compare.py` 用于测量这条路径。
