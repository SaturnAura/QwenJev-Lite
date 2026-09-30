> 英文版见 [`README.md`](README.md)。

QwenJev-lite 将 Transformer 形式化为一个**类型化决策模型**：共享状态仅编码一次，每个问题构成一条相互隔离的分支，推理末端输出的并非生成文本，而是**允许答案上的概率分布**。问题具有类型 —— 有限选择 `choice`、是/否 `bool`、有序打分 `score` —— 同一 state 上可一次性完成多种类型的提问。

骨架为 **Qwen3.5-4B**（32 层：24 层线性注意力 + 8 层全注意力，4.54B 参数 —— 4.21B 语言 + 0.33B 视觉，骨架本身即为多模态），训练**仅更新末端一个 512 行的决策头**（约 5 MB），因此整轮实验可在单张 24 GB 显卡上完成。在 26 个测试集上总体 **0.772**；同一批保留标签行**完全不训练**、直接作为 Qwen 基线使用时（`qwen_zeroshot`）为 **0.722**。

```bash
pip install -r requirements.txt
export QWENJEV_MODEL=/path/to/qwen3.5-4B      # Windows: $env:QWENJEV_MODEL="D:\qwen3.5-4B"
python demo.py --checkpoint-dir models/qwenjev-multitask-v2
```

## 结果

完整评测命令为 `python test.py --limit 0`，在 26 个两个变体均可作答的测试集上进行（bool 7 + choice 12 + score 7）。表中数值为准确率；原始数据见 [`artifacts/results.json`](artifacts/results.json)，逐任务明细见 [`artifacts/RESULTS_CN.md`](artifacts/RESULTS_CN.md)。

| 题型/准确率             | Laya（参照）    | Qwen 基线（`qwen_zeroshot`） | **我们训练后的模型** |
| ----------------------- | --------------- | ------------------------------ | -------------------------- |
| 判断`bool`（7 项）    | 0.520           | 0.717                          | **0.793**            |
| 单选`choice`（12 项） | 0.511           | 0.749                          | **0.760**            |
| 打分`score`（7 项）   | 0.258           | 0.681                          | **0.771**            |
| **总体（26 项）** | **0.446** | 0.722                          | **0.772**            |

**Laya** 一列为同类的第三方开放基线，在相同的 26 个测试集上测得，此处列出仅为提供参照。本仓库不安装、也不要求运行该基线 —— `test.py` 默认仅评测本项目自身的两个变体；如已持有对应 checkpoint，`--variants laya` 仍然可用。

## 快速开始

```bash
pip install -r requirements.txt          # torch 2.6 + transformers 5.14；运行测试无需 GPU、模型或数据

export QWENJEV_MODEL=/path/to/qwen3.5-4B # Windows: $env:QWENJEV_MODEL="D:\qwen3.5-4B"

python demo.py --checkpoint-dir models/qwenjev-multitask-v2   # 一个 state、三种题型、一次前向完成
python serve.py                                              # 同一功能的网页版（:8300）
pytest -q                                                    # 92 个测试，微型骨架，纯 CPU
```

| 目标                                             | 命令                                                                |
| ------------------------------------------------ | ------------------------------------------------------------------- |
| 运行全量基准（Qwen 基线 + 我们的模型，带进度条） | `python test.py --limit 0`                                        |
| 训练决策头（四类输入，均有默认值）               | `python train.py --model /path/to/qwen3.5-4B`                     |
| 由三个源文件重建交付的决策头                     | [`models/README.md`](models/README.md#rebuild-the-shipped-head)    |
| 将原始数据转换为统一格式                         | `python -m qwenjev.cli normalize --src data/raw --out data/ready` |

## 交付模型

`models/qwenjev-multitask-v2/readout.pt` 为交付的决策头：**512 行 = 27 行保留 + 485 行私有**，覆盖 `data/ready` 中的 90 个标签空间。

## 训练与测试

一次实验仅需**四类输入**，且每类均提供默认值：

| 输入                | 参数                                                                                                                                                 | 默认                                        | 说明                                                                                                                        |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| base 模型           | `--model`                                                                                                                                          | `$QWENJEV_MODEL`，否则 `./qwen3.5-4B`   | 本地路径或 Hub repo id；引擎仅读取隐藏状态，因此换用其它 causal 骨架、或换用多模态骨架（图像/音频 token 置于`state`）均可 |
| 决策头保存/读取地址 | `--checkpoint-dir`                                                                                                                                 | *留空* → `models/qwenjev-run-<时间戳>` | 不存在时自动创建；`train.py` 写入 `readout.pt` + `card.json`，`test.py` 从中读取 `readout.pt`                     |
| 数据地址            | `--data-dir`                                                                                                                                       | `data/ready`                              | 一个目录，其中每个任务对应一对`<task>_train.jsonl` / `<task>_test.jsonl`（`<task>_<split>.jsonl` 亦可）               |
| 超参数              | `--epochs`、`--lr`、`--batch-size`、`--objective`、`--items-per-label`、`--min-items-per-task`、`--prototype-init`、`--optimizer` 等 | 见`python train.py --help`                | `test.py` 另有 `--limit`、`--batch`、`--variants`                                                                   |

对 `test.py` 而言，`--checkpoint-dir` 留空表示**使用交付的决策头**；若磁盘上不存在该头，则自动改为仅评测 Qwen 基线，而不会报错。结果写入 `--out` / `--markdown`（父目录自动创建），默认为 `artifacts/results.json` 与 `artifacts/RESULTS.md`。

**数据格式。** 一个目录，每个任务两个文件。文件为 **JSONL：每行一条记录** —— 一行一个 JSON 对象，**不要美化换行**（多行记录会被拒绝，并提示查阅本节）。一条记录即引擎的请求结构加上实际结果，覆盖三种题型。以下三行刻意保持为单行，与文件中的实际形态一致。

`choice` 题 —— 群聊中确定次日安排：

```json
{"id": "weekend/train/0001", "dataset": "weekend", "split": "train", "state": "Friday night, three of us are pinning down tomorrow in the group chat: Zhe wants to hike, Yu wants a gallery, and I said either is fine.", "questions": {"plan": {"type": "choice", "instructions": "What should we do tomorrow?", "criteria": {"hike": "Hike", "museum": "Gallery", "home": "Stay in"}}}, "targets": {"plan": "museum"}}
```

`bool` 题 —— 外卖送到，判断该餐对当前的她而言是否重口：

```json
{"id": "takeout/train/0002", "dataset": "takeout", "split": "train", "state": "Dinner just arrived: a spicy hotpot on one side, a plain noodle soup on the other. Yu's stomach has been off all week, but her chopsticks went for the hotpot first.", "questions": {"spicy": {"type": "bool", "instructions": "Is this bowl heavy for her right now?"}}, "targets": {"spicy": "yes"}}
```

`score` 题 —— 散场时为影片评级（有序等级，由低到高）：

```json
{"id": "movie/train/0003", "dataset": "movie", "split": "train", "state": "The credits roll: Zhe calls it the best film he has seen all year, Yu is yawning next to him, and I am stuck in the middle.", "questions": {"rating": {"type": "score", "instructions": "How good was the film?", "criteria": {"bad": "Bad", "ok": "Fine", "good": "Great"}}}, "targets": {"rating": "ok"}}
```

需要遵守的规则：`targets` 的取值必须是该问题 `criteria` 的键之一；`choice` 与 `score` 必须提供 `criteria`（有序 —— `score` 的声明顺序即为刻度，须由最低到最高排列），`bool` 可省略，默认为 yes/no；问题 id（`plan`、`rating` …）仅为标识符，真正决定决策头读取哪一行的是 `(问题 id, 标签空间, 答案键)`。

**训练** —— 使用上述四类输入，保存路径留空即自动创建：

```bash
python train.py \
  --model /path/to/qwen3.5-4B \
  --data-dir data/ready \
  --epochs 1 --lr 1e-3 --batch-size 8 \
  --objective log_loss \
  --items-per-label 12 --min-items-per-task 200 --prototype-init
# 亦可自行指定本次运行：
#   --checkpoint-dir models/my-run       （不存在时自动创建）
#   --epochs 0                           （0 = 仅使用闭式原型行，不执行梯度步）
```

脚本在开始前会打印解析后的 base 模型、数据目录、保存位置与超参数，结束后将 `readout.pt` 与 `card.json` 写入保存位置。

**评测** —— 同样三个路径，另加输出文件（父目录自动创建）：

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

`data/ready/` 随仓库提供（约 97 MB 的 JSONL）：统一格式的 train/test 集，以及 `manifest.json`（记录每个 split 的来源文件、条数与注意事项）。所用来源如下：

| 来源                                                        | 提供哪些任务                                                                                               |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| MultiNLI / SNLI                                             | `mnli`、`mnli_ood`、`snli`（一个前提配多条假设）                                                     |
| Jigsaw Toxic Comment                                        | `jigsaw`（6 个独立是/否）、`jigsaw_severity`（有序打分）                                               |
| GoEmotions                                                  | `goemotions`（28 个情感是/否）、`goemotions_sentiment`（有序打分）                                     |
| TruthfulQA                                                  | `truthfulqa`（5 选 1）                                                                                   |
| IntentGrasp                                                 | `intentgrasp`（选项随样本给出）                                                                          |
| CLINC150 / HWU64                                            | `clinc150`、`clinc150_top15`、`hwu64`、`hwu64_top15`                                               |
| BEIR（arguana / nfcorpus / scidocs / scifact / trec-covid） | 15 个相关性判别任务：用 query 与标注答案构造正样本选项、非 qrel 文档构造负样本选项，按 query 切分训练/测试 |

`artifacts/extra_splits.json` 记录本仓库**自行补齐**的训练集（来源、条数，以及"测试集要读的每个答案都被训练到"的校验），`data/ready/manifest.json` 记录其余 split 的出处。相关数据集、标注与工具的作者与维护者使本工作成为可能，谨此致谢，完整出处见[引用](#引用)。

## 目录结构

```
qwenjev/           库：schema / prompt / tokenize / readout / engine / rlcd / calibration /
                   datasets / normalize / relevance / backends / benchmark / probes / api / cli
train.py           用数据目录里的 train split 训练一个决策头（带进度条）
test.py            在 test split 上评测 Qwen 基线 + 我们的模型（带进度条）
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

- **保留标签行使模型在训练之前即可工作。** `ReservedLabelReadout` 将预训练 LM head 在选项标签 token 上的概率质量重新归一化，因此 **K ≤ 26 个答案时零训练即可作答**；同一批行同时充当训练后决策头的初始化与"未见过答案"的回落。
- **问题具有类型**：有限选择、是/否、有序打分（[`qwenjev/schema.py`](qwenjev/schema.py)）。
- **选项按列表整体读入。** 所有选项作为有序列表置于分支中，决策头读取列表**之后**的位置（[`qwenjev/prompt.py`](qwenjev/prompt.py)、[`qwenjev/tokenize.py`](qwenjev/tokenize.py)）—— 因此答案可取决于整个候选集合。
- **共享状态仅编码一次。** state 进入 KV/递推缓存，缓存按分支展开，每条分支仅注意该 state 与自身的后缀；**隔离是结构性的**，而非提示词的约定（[`qwenjev/engine.py`](qwenjev/engine.py)）。
- **分支按批调度**：在批大小与 token 预算下打包进单次前向（`_chunk_branches`）；`share_state=False` 为"每条分支重算 state"的参考实现，用作差分测试。上限为**每分支 32,768 token**、**每请求 65,536 token**，state 仅计一次（`JevLimits`、`check_limits`、`account`）。
- **训练采用 RLCD**：backbone 冻结，以 proper scoring rule 按结果拟合分布（[`qwenjev/rlcd.py`](qwenjev/rlcd.py)）—— 见 [RLCD 简析](#rlcd-简析)。
- **校准是被测量的，而非被假设的**：可靠性分箱、ECE、Brier、Wilson 区间，`confidence` 保持为算术量（[`qwenjev/calibration.py`](qwenjev/calibration.py)、[`qwenjev/confidence.py`](qwenjev/confidence.py)）。两个后端共用同一套评测代码（[`qwenjev/backends.py`](qwenjev/backends.py)）。

## RLCD 简析

**RLCD**（Reinforcement Learning for Calibrated Decisions）的核心在于：**从结果而非固定标签集进行训练**。其思想可概括为一句话：*一个决策即"该问题所允许答案"上的一个分布，因此应以一个仅当分布正确时才取得最小值的目标函数来训练它。*

**决策头。** 给定分支决策位上的隐藏状态 `h ∈ R^d` 与该问题允许的 `K` 个答案，模型为每个答案给出一个得分并归一化：

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

二者均为**严格 proper**：其期望意义上的极小值点**恰好**位于 `p = P(y | state, question)`，因此学到的是概率本身，而非仅"分数更高"。梯度同样直接：

```
∂L/∂z_k = p_k − 1{y = k}      （log loss，τ = 1）
```

即：每条样本将"实际发生"的那一行推高，其余各行按其当前概率占比被推低。

**校准为何关键。** 仅以 argmax 正确与否评判的分类器可以任意过信；而此处**过信会被损失直接惩罚**，并以 ECE 检验 —— 在 26 项基准上，训练后的模型平均 ECE 为 **0.125**，Qwen 基线为 **0.158**，同时准确率更高。

**为何带有"强化"成分。** 监督信号为**决策加结果**（`state → 问题 → 答案 → 实际发生`），被拟合的参数为"答案对应的行"，而**未经训练的标签空间依然可用**（读回保留行）。因此同一目标可将已部署模型扩展至新的答案集合，而无需为每个标签集重建一个头。本仓库实现的是该目标的**有监督形式**：backbone 冻结，以 log loss 或 Brier 拟合决策头（`train.py --objective log_loss|brier`）；在显存允许时，同一损失亦可带上 backbone 的梯度。

**由实验得出的两个结构细节**（均属于训练配方的一部分）：

- **每个 `(问题, 标签空间, 答案)` 各占一块私有行。** 当行按位置共享时，训练 2–4 选项的任务会覆盖 10–15 选项任务所需读取的行（同一任务上共享行为 0.733，改为私有行后为 0.760）；未出现过的答案读回预训练行，因此未经训练的标签空间**逐位保持零样本行为**。
- **每步之后将行投影回预训练的对数尺度**（`‖w‖ ≈ 0.74`）。决策态范数约为 157，若不做投影，lr=1e-3 下单步即可将新行推离约 1/5 的"合适长度"并导致对数爆炸（实测训练损失 33，而均匀分布仅为 5.01）；交付的共享行另作 α=0.5 回拉（`W(0.5) = W₀ + 0.5·(W_trained − W₀)`），宽于保留行的标签空间则改用上述闭式原型行，因为 150 类、每类约 20 条样本的交叉熵无法在一个 epoch 的梯度步内收敛。

## 与 BERT-base 分类器的区别

将 Transformer 用作分类器的常见做法是"BERT-base + 一个线性头"，差异主要体现在以下几处：

|                    | BERT-base + 一个线性头                       | **QwenJev-lite**                                                           |
| ------------------ | -------------------------------------------- | -------------------------------------------------------------------------------- |
| 骨架               | 编码器，约 110M 参数，12 层                  | causal Transformer，4.54B，32 层（24 线性注意力 + 8 全注意力）                   |
| 一次前向答几个问题 | 一个 —— 池化后的`[CLS]` 状态只喂给一个头 | 多个 —— state 编码一次，每个问题一条隔离分支，一次批处理答完                   |
| 答案从哪来         | 类别数**在建模型时即固定**的头         | 分支决策位上的状态 +**K 行，一行一个允许答案，K 可随请求变化**             |
| 增加一组答案       | 新头、重训、每任务一个 checkpoint            | **同一个头覆盖任意标签空间**：保留行提供零样本能力，私有行覆盖训练过的空间 |
| 混用题型           | 每种题型一个分类器，一次前向问一个           | 一个请求里`choice` + `bool` + `score` 可问同一个 state                     |
| 不确定性           | 通常过信、且**从不评估**的 softmax     | **被训练的对象即该分布**，以 ECE / Brier / NLL 评估                        |
| 成本形态           | 单次开销低，但**一次仅答一个问题**     | 每个 state 一次预填（4.54B），此后同一 state 每增加一个问题约 +20 ms             |
| 生成               | 无（纯分类）                                 | 设计上即无 ——**决策头取代了解码循环**                                    |

概括而言：BERT 分类器回答"这属于我 N 个类中的哪一个"，其头必须在训练前存在；QwenJev-lite 回答"这些 K 个答案中是哪一个"，答案集合是请求的一部分，同一 state 上的多个问题相互隔离，输出的是**校准后的分布**而非单一 argmax。

## 我们对模型与数据做了哪些改编

**模型方面。** 我们以决策头取代解码循环与固定分类器 —— 在允许答案上计算 `z = W h + b` 并施加 softmax，**不生成任何 token**，因此延迟不随输出长度增长。**保留标签行优先**：Qwen 基线（`qwen_zeroshot`）将预训练头在选项标签 token 上的质量重新归一化，K ≤ 26 时无需训练即可工作，同时为训练后的模型提供初始化与"未见过答案"的回落。**共享预填与分支隔离**：state 仅编码一次并写入 KV/递推缓存，缓存按分支展开，每条分支仅注意 state 与自身的后缀，隔离具有结构性而非依赖提示词约定。**分支按批调度**：在批大小与 token 预算下打包进单次前向，一个请求通过数次前向即可完成一个 state 上的全部问题。**选项按列表整体渲染**：所有选项作为有序列表置于分支中，决策头读取列表**之后**的位置，答案可取决于整个候选集合。**类型化问题共用同一线格式**：训练器、评测器、CLI、HTTP 接口与两个后端读取同一条记录。**一行对应一个答案**：行由 `(问题 id, 标签空间, 答案键)` 决定，因此同一组选项改变顺序仍读取同一行，行也不会跨问题混用。**训练配方**：backbone 冻结 + RLCD（log loss 或 Brier）、每步将行投影回预训练对数尺度、共享行作 α=0.5 回拉、宽于保留预算的标签空间使用闭式原型行、可选地在留出集上拟合温度。

**数据方面。** **每个数据集对应一条统一记录**：所有来源均归一化为引擎自身的请求结构（`{"state": …, "questions": {…}, "targets": {…}}`，一行一个 JSON），因此同一份文件可同时服务训练、评测、CLI 与 HTTP 接口，不存在任何按数据集划分的代码路径。**一切均表达为类型化问题**：一个任务或为带 `criteria` 的 `choice`，或为 `criteria` 可省的 `bool`，或为有序 `score`；是/否题同时给出 `instructions`（问句）与 `claim`（陈述句）两种措辞，因为两个后端分别在不同的措辞上被测量。**标签空间属于数据而非代码**：`criteria` 的键即答案身份，决策头将 `(问题 id, 标签空间, 答案键)` 映射到一行 —— 这正是"选项顺序不影响结果"与"多任务共用一个决策头却互不干扰"的原因。**切分纪律**：每条训练行与其被评测的行不重叠，截断取样在整个文件上均匀抽取而非取前缀，本仓库自行构建的训练集（`artifacts/extra_splits.json`）在写盘前会做校验 —— 若测试集需要读取的某个答案未被训练到，脚本将拒绝写入。**按标签空间分配预算**：`--items-per-label` 保证 150 类任务不会被 2 答案任务挤占。**精度**：backbone 以 bf16 运行，决策头保持 float32（训练步长约 1e-3，低于 bf16 在该量级上的分辨率），对数亦在 float32 中计算。

## 引用

本项目的模型、数据与方法均建立于既有工作之上，所用资源及其出处如下。

**模型**

- **Qwen3.5-4B**（骨架模型）：[huggingface.co/Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)。

**数据集**

- BEIR（检索评测基准）：[github.com/beir-cellar/beir](https://github.com/beir-cellar/beir)。
- CLINC150 / BANKING / HWU64（意图识别）：[github.com/SaturnAura/BenchmarkingIntentDetection](https://github.com/SaturnAura/BenchmarkingIntentDetection)。
- MMLU / MMLU-Pro：[huggingface.co/datasets/TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/tree/main)。
- TruthfulQA：[huggingface.co/datasets/domenicrosati/TruthfulQA](https://huggingface.co/datasets/domenicrosati/TruthfulQA/viewer)。
- MNLI / SNLI：[github.com/Tikquuss/nli_dataset](https://github.com/Tikquuss/nli_dataset)。
- FEVER：[fever.ai](https://fever.ai/)。
- Jigsaw Toxic Comment Classification：[github.com/praj2408/Jigsaw-Toxic-Comment-classification](https://github.com/praj2408/Jigsaw-Toxic-Comment-classification)。
- GoEmotions：[kaggle.com/datasets/debarshichanda/goemotions](https://www.kaggle.com/datasets/debarshichanda/goemotions)。
- IntentGrasp: [huggingface.co/datasets/yuweiyin/IntentGrasp](https://huggingface.co/datasets/yuweiyin/IntentGrasp)。

**启发与参考**

本项目的设计思路受以下文章与仓库启发：[archerhume.com/posts/jevs-architecture-unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/?v=3)、[zhuanlan.zhihu.com/p/2087218714244658951](https://zhuanlan.zhihu.com/p/2087218714244658951)、[huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)、[github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)、[medium.com/data-science-in-your-pocket/openjev-qwen-3-5-as-jev-ai](https://medium.com/data-science-in-your-pocket/openjev-qwen-3-5-as-jev-ai-85f584d2922b)、[aiprofitboardroom.com/blog/openjev](https://aiprofitboardroom.com/blog/openjev/)。

上述模型、数据集与工具的作者与维护者使本工作成为可能，谨此致谢。

## TODO

本版本仍有以下工作待完成：

- TODO：**支持更换骨架，包括多模态骨架。** 引擎仅向骨架索取一项信息 —— 每个决策位置的隐藏状态 —— 因此任何能够输出隐藏状态的模型均可作为骨架。更换为另一个 causal LM，或更换为多模态编码器（将图像/音频 token 与文本一同置于 `state`，或置于某条分支的 `context`），决策头、训练器、评测器与 HTTP 接口均无需改动，仅通过 `--model`（本地路径或 Hub repo id）选择；见[训练与测试](#训练与测试)。
- TODO：**以 flash attention 及相关方法加速推理。** 在纯 torch 回退路径上，单条决策约需 37 ms，且仅在单个 state 上的问题数超过约 64 条时，每次调用的优势才得以体现。我们正在引入 `flash-linear-attention`、`causal-conv1d` 等融合算子以开启线性注意力的快路径，目标是在长序列、多问题场景下降低每次调用的延迟；`scripts/speed_compare.py` 用于测量该路径。
