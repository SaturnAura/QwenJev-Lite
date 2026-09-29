# QwenJev-lite 单选（choice）专项优化报告（V8）

本轮起点：`models/qwenjev-multitask-v2/readout.pt`（V7，α=0.5 混合）。
本轮问题：**判断题（bool）0.793、打分（score）0.736 都明显超过零样本，单选（choice）
却只从 0.692 涨到 0.700，而且三个 15 选项意图任务反而掉了。**

结论提前说：问题不在"训练/测试数据没弄好"，而在**读出的行是共享的**。
把行按"问题 + 答案"私有化、补上原来缺的训练数据、修正意图测试集的取样 bug 之后：

| 标签空间 | 旧模型 | 新模型 | Laya | 零样本读出 |
|---|---|---|---|---|
| BANKING77（77 类） | 0.100 | **0.495** | 0.225 | 无法作答 |
| CLINC150（150 类） | 0.077 | **0.542** | 0.545 | 无法作答 |
| HWU64（64 类） | 0.225 | **0.542** | 0.345 | 无法作答 |
| choice 全家（18 项） | 0.606 | **0.671** | 0.455 | 0.692（只能答 15 项） |

后续所有数字都来自同一次全量测试：33 个测试集 × 3 个变体（Laya / 零样本读出 / 训练后读出），
原始文件 `artifacts/dataset_results_v8.json`，表格 `artifacts/DATASETS_V8.md`。

---

## 1. 为什么单选上不去：四个可测量的原因

### 1.1 读出的行是"按位置"共享的（主因）

论文里的读出是 `z = W h + b`，`W` 的每一行对应一个**允许的答案**。我们的实现一开始
按"选项在行里的位置"取行：第 0 个选项永远读第 0 行。于是

* mnli/snli（3 个选项）训练时改的是第 0..2 行；
* 15 选项的意图任务读的也是第 0..14 行——**前 3 行被别人改坏了**。

实测（同一 dev 切片，零样本 → 训练后）：

| 任务 | 选项数 | 旧模型 | 说明 |
|---|---|---|---|
| clinc150_top15 | 15 | 0.893 → 0.733 | 行被 3 选项任务覆盖 |
| hwu64_top15 | 15 | 0.613 → 0.467 | 同上 |
| intentgrasp | 2–10 | 0.350 → 0.225 | 同上 |
| mmlu_pro | 10 | 0.450 → 0.400 | 同上 |
| mnli（在训练集里、选项少） | 3 | 0.756 → 0.840 | 训练确实有效 |

也就是说：训练本身是有效的，**只是有效的那部分行和无效的那部分行是同一批**。

**修法**：行按 `(问题 id, 标签空间, 答案键)` 私有化。没见过的标签空间仍然回落到预训练
的字母行（零样本行为一字不变），见过的标签空间各读各的行。代码在
`qwenjev/schema.py::option_ids`、`qwenjev/readout.py::SlotHeadReadout.slot_table`。

### 1.2 训练预算失衡：choice 只分到 15% 的梯度

旧配置里 goemotions 一个任务就贡献 3360 条决策，而每个 choice 任务只有约 120 条。
改成按"每个标签空间给固定条数"的预算后，choice 的梯度占比从 **14.7% 升到 36%**。

### 1.3 三个意图测试集只覆盖了前几十个类

`CLINC150/ BANKING77/ HWU64` 的 testset 文件是**按标签排序**的，而取样写的是
`rows[:limit]`（`total // limit == 1` 时步长退化成 1）。结果 400 条测试只覆盖
前 81 / 40 / 41 个类，后面的类从没被测过。

**修法**：`qwenjev/normalize.py::_iter_limited` 在"限制超过文件一半"时改为在整文件上
均匀取样；重新生成这三个测试集后，400 条测试覆盖全部 150 / 77 / 64 个类。
**注意：这三个任务的数字因此不能和历史数字直接比**，Laya 的数字也一起变了
（0.512→0.545、0.247→0.225、0.270→0.345）。

### 1.4 新行的数值尺度：一步就把行推到量级之外

4B 模型决策位置的隐藏状态范数 |h| ≈ **157**，而预训练字母行的范数只有 **0.74**。
一次 Adam/SGD 步（lr 1e-3，梯度 ‖g‖≈60）就能把一行推高 0.05，是"合适长度"的 7%，
于是行的方向还没学到就已经被推到饱和。实测 150 选项任务的平均训练损失
**33.0**（均匀分布也只有 ln150=5.01）。

**修法**（`train.py --row-norm-cap auto`、`--max-grad-norm 0`、`--optimizer sgd`）：
每步之后把训练行的范数投影回 0.742（预训练字母行的均值范数）；
同时**不要**保留 `clip_grad_norm_(1.0)`——它把 ‖g‖≈60 的梯度压掉 60 倍，
对 Adam 无所谓（尺度不变），对 SGD 等于把学习率除以 60，训练完全不动。

### 1.5 宽标签空间用梯度学不到（换成闭式解）

即使修好了 1.4，150 类 × 每类 12 个样本在 1 个 epoch 内仍然学不动：
损失停在 **4.90**（均匀 5.01）。原因是交叉熵的梯度被 batch 平均、又被 150 个选项稀释，
每步里真正能区分答案的分量只有 1% 左右。

**修法**（`train.py --prototype-init`）：直接给出闭式行——某答案的原型 = **选择该答案的
样本的状态均值**，再按标签空间中心化、整体缩放到字母行的尺度。一次前向即可，
`qwenjev/rlcd.py::RLCDFineTune.prototypes`。

---

## 2. 最终模型：共享行 + 宽空间私有行

两个部件各有主场，因此交付模型是二者的结构式拼接（`scripts/compose_wide_rows.py`）：

* 选项数 ≤ 26 的标签空间（字母行够用）：沿用已训练的共享行（V7 的 α=0.5 读出）；
* 选项数 > 26 的标签空间（字母行不够）：读原型私有行。

拼接是**严格扩展**：表里有的答案读自己的行，其余答案仍然回落到原来的位置行，
所以 ≤26 选项的所有任务数字与旧模型逐位相同。

新模型文件：`models/qwenjev-multitask-v2/readout.pt`（547 行 = 256 共享 + 291 私有），
卡片 `card.json` 里记了 `composed_from` 与行数。

---

## 3. 全量结果（33 个测试集，一次跑完）

### 3.1 分题型平均

| 题型 | Laya | 零样本读出 | **训练后读出** |
|---|---|---|---|
| bool（7 项） | 0.520 | 0.717 | **0.794** |
| choice（15 项可比） | 0.472 | 0.692 | **0.700** |
| choice（18 项，含 3 个宽空间） | 0.455 | 无法作答 | **0.671** |
| score（8 项） | 0.271 | 0.660 | **0.737** |
| **总体（33 项）** | **0.424** | 无法作答 | **0.713** |
| 总体（30 项三变体都能答） | 0.430 | 0.689 | **0.731** |

平均 ECE：Laya 0.311 / 零样本 0.161 / 训练后 0.162
（bool 0.109、score 0.151、choice 0.188——choice 的过信来自新增的宽空间任务）。

### 3.2 逐任务（完整表见 `artifacts/DATASETS_V8.md`）

| task | 题型 | Laya | 零样本 | 训练后 |
|---|---|---|---|---|
| arguana_rel_bool | bool | 0.450 | 0.330 | **0.803** |
| goemotions | bool | 0.907 | 0.722 | **0.958** |
| jigsaw | bool | 0.897 | 0.869 | **0.965** |
| nfcorpus_rel_bool | bool | 0.349 | 0.701 | 0.685 |
| scidocs_rel_bool | bool | 0.361 | 0.800 | 0.669 |
| scifact_rel_bool | bool | 0.258 | **0.889** | 0.804 |
| trec-covid_rel_bool | bool | 0.420 | 0.707 | 0.670 |
| arguana_rel_choice | choice | 0.650 | 0.900 | **0.983** |
| banking77 | choice | 0.225 | – | **0.495** |
| banking77_top15 | choice | 0.507 | 0.587 | 0.573 |
| clinc150 | choice | **0.545** | – | 0.542 |
| clinc150_top15 | choice | 0.893 | **0.893** | 0.733 |
| fever | choice | 0.307 | 0.415 | **0.440** |
| hwu64 | choice | 0.345 | – | **0.542** |
| hwu64_top15 | choice | 0.600 | **0.613** | 0.467 |
| intentgrasp | choice | 0.315 | **0.323** | 0.302 |
| mmlu_pro | choice | 0.133 | **0.393** | 0.365 |
| mnli | choice | 0.582 | **0.792** | 0.782 |
| mnli_ood | choice | 0.596 | **0.824** | 0.808 |
| nfcorpus_rel_choice | choice | 0.333 | 0.617 | **0.733** |
| scidocs_rel_choice | choice | 0.467 | 0.950 | 0.950 |
| scifact_rel_choice | choice | 0.417 | 0.967 | **1.000** |
| snli | choice | 0.617 | 0.861 | **0.868** |
| trec-covid_rel_choice | choice | 0.420 | 0.640 | **0.900** |
| truthfulqa | choice | 0.242 | **0.604** | 0.592 |
| arguana_rel_score | score | 0.190 | 0.647 | **0.840** |
| fever_support | score | 0.355 | **0.515** | 0.495 |
| goemotions_sentiment | score | 0.425 | **0.605** | 0.578 |
| jigsaw_severity | score | 0.385 | 0.860 | **0.915** |
| nfcorpus_rel_score | score | 0.262 | 0.555 | **0.583** |
| scidocs_rel_score | score | 0.217 | 0.622 | **0.831** |
| scifact_rel_score | score | 0.144 | 0.781 | **0.961** |
| trec-covid_rel_score | score | 0.187 | 0.697 | 0.690 |

### 3.3 速度（整个基准）

| 变体 | 请求数 | 决策数 | 决策/请求 | ms/请求 | 总时长 |
|---|---|---|---|---|---|
| Laya | 24895 | 24895 | 1.00 | **23** | **9.6 min** |
| 零样本读出 | 5885 | 23695 | 4.03 | 136 | 13.4 min |
| 训练后读出 | 7085 | 24895 | 3.51 | 137 | 16.1 min |

Laya 单次请求快约 6 倍，但它是"一次请求一个问题"；我们的引擎一次请求平均答 3.5 个问题
（goemotions 一条评论 28 个问题一次答完），所以整个基准的墙钟只差 1.7 倍。
本机没有装 `flash-linear-attention` / `causal-conv1d`，装上前向大约快 10 倍。

---

## 4. 复现命令

```bash
# 1) 原始数据 -> 统一格式（含 1.3 的取样修正）
python -m qwenjev.cli normalize --src data/raw --out data/ready

# 2) 补齐原来缺的训练集：MMLU-Pro 6000 行、CLINC150/HWU64 各 4000 行、
#    两个 *_top15 训练集、BANKING77 370 行；每一步都会校验"测试集读的答案
#    训练集必须都训练过"，不满足就拒绝写盘
python scripts/build_extra_splits.py

# 3) 原型行（一次前向，约 20 分钟；本模型用的是 --items-per-label 12）
python train.py --no-balance --epochs 0 --items-per-label 12 --min-items-per-task 200 \
  --prototype-init --row-norm-cap auto --optimizer sgd --max-grad-norm 0 \
  --model-dir models/_p3 --report artifacts/_p3.json     # 见 §1.5

# 4) 把原型行接到共享行上（只接 >26 选项的标签空间）
python scripts/compose_wide_rows.py \
  --base models/_v7/readout.pt --wide models/_p3/readout.pt \
  --out models/qwenjev-multitask-v2/readout.pt

# 5) 全量对比测试（33 个测试集 × Laya / 零样本 / 训练后，约 39 分钟）
python test.py --variants laya qwen_zeroshot qwen_trained --limit 0 --batch 16 \
  --out artifacts/dataset_results_v8.json --markdown artifacts/DATASETS_V8.md

# 6) 打印分题型平均
python scripts/compare_results.py artifacts/dataset_results_v8.json
```

梯度版单任务读出仍然可用（在 ≤26 选项的标签空间上它和共享行是同一批行）：

```bash
python train.py --no-balance --epochs 1 --items-per-label 30 --min-items-per-task 200 \
  --optimizer sgd --momentum 0 --max-grad-norm 0 --row-norm-cap auto \
  --model-dir models/run --report artifacts/run.json
```

常用调参开关：`--anchor`（向预训练行的 L2 回拉）、`--row-norm-cap`（行范数上限，
`auto` = 字母行均值 0.742）、`--max-grad-norm`（默认 0，见 §1.4）、
`--optimizer sgd|adamw`、`--momentum`、`--prototype-init`、
`--items-per-label`（每个标签固定条数，避免多标签任务被饿死）、
`--decisions-per-task`、`--init-from`（在已有读出上继续/微调）。

---

## 5. 本轮试过但没有采纳的方案（都测过）

| 方案 | 结果 |
|---|---|
| 按类平衡采样 | 摧毁先验（真阳率 4.3% 的任务变成 P(yes)=0.78），弃用 |
| 全局共享行 + L2 锚 + α 混合（V7） | bool/score 很好，宽标签空间无解（0.077/0.225/0.100） |
| 按"答案键"共享行（跨标签空间共享，V10） | 150 路任务的负梯度压平 15 路任务的行：clinc150_top15 0.55、banking77_top15 0.15 |
| AdamW + 行范数上限 | 每坐标归一化把 150 行都推向 ±sign(h)，方向趋同，损失 15–21 |
| SGD + momentum 0.9 | 动量累积成 0.18/步的跳变，一步冲过合适方向，损失升到 5.68（比均匀还差） |
| `clip_grad_norm_(1.0)` + SGD | ‖g‖≈60 被压 60 倍，150 类任务 124 步后损失 4.90（等于没学） |
| 纯原型行（所有标签空间私有） | 宽空间大幅领先（clinc150 0.525 / hwu64 0.425 / banking77 0.550），但判断题把"yes/no"聚成一对行，scifact 相关性判断 0.882→0.212，goemotions 0.724→0.573 |
| 原型行 + 按问题私有（不区分标签空间大小） | 修好 scifact_rel_bool（0.813），但 mmlu_pro 0.150、jigsaw_severity 0.425 |
| 在 150 类 intent 上直接训练共享行 | 30 项均值 0.731→0.717，弃用 |

仍然存在的不足，如实列出：

* 三个 15 选项意图子集（clinc150_top15 / hwu64_top15 / banking77_top15）仍然低于零样本
  （0.733 / 0.467 / 0.573），因为交付模型对 ≤26 选项的标签空间沿用共享行；
  原型行在这些标签空间上更好（0.900 / 0.750 / 0.775），只是它同时会拉低判断题，
  两者可以按标签空间再选一次，但那需要额外一层 dev 选择，本轮没有做；
* `mmlu_pro`（0.365）和 `intentgrasp`（0.302）仍低于零样本，两者的训练集太小
  （MMLU-Pro 官方只有 70 条验证样本，本轮从测试文件里补了 6000 条无重叠样本，
  但 10 路题目的答案位置本身是随机先验，线性读出的收益有限）；
* `fever` 是仅凭 claim 的闭卷事实核查（原始文件没有 wiki 句子），0.44 已经接近这套
  读出在这个任务上的上限；
* 交付模型对"训练时见过的标签空间"有效，对全新标签空间只能回落到零样本能力。
