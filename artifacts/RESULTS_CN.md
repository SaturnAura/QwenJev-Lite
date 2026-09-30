# 最终结果（QwenJev-Lite）

一次完整运行：`python test.py --variants laya qwen_zeroshot qwen_trained --limit 0`，
在 data/ready 里 **26 个三个变体都能作答的测试集** 上（bool 7 项 + choice 12 项 + score 7 项）。
原始数据 `artifacts/results.json`，模型 `models/qwenjev-multitask-v2/`（card.json 内含同一份成绩单）。

## 1. 按题型平均

| 题型 | Laya | 零样本读出 | **训练后读出** |
|---|---|---|---|
| 判断 bool（7 项） | 0.520 | 0.717 | **0.793** |
| 单选 choice（12 项） | 0.511 | 0.749 | **0.760** |
| 打分 score（7 项） | 0.258 | 0.681 | **0.771** |
| **总体（26 项）** | **0.446** | 0.722 | **0.772** |

平均 ECE：Laya 0.289 / 零样本读出 0.158 / **训练后读出 0.125**。

## 2. 逐任务

| 任务 | 题型 | Laya | 零样本 | 训练后 | 训练后 ECE |
|---|---|---|---|---|---|
| arguana_rel_bool | bool | 0.450 | 0.330 | **0.803** | 0.025 |
| goemotions | bool | 0.907 | 0.722 | **0.958** | 0.023 |
| jigsaw | bool | 0.897 | 0.869 | **0.965** | 0.007 |
| nfcorpus_rel_bool | bool | 0.349 | **0.701** | 0.685 | 0.174 |
| scidocs_rel_bool | bool | 0.361 | **0.800** | 0.669 | 0.200 |
| scifact_rel_bool | bool | 0.258 | **0.889** | 0.804 | 0.112 |
| trec-covid_rel_bool | bool | 0.420 | **0.707** | 0.670 | 0.225 |
| arguana_rel_choice | choice | 0.650 | 0.900 | **0.983** | 0.055 |
| clinc150_top15 | choice | 0.893 | **0.893** | 0.733 | 0.072 |
| hwu64_top15 | choice | 0.600 | **0.613** | 0.467 | 0.192 |
| intentgrasp | choice | 0.315 | **0.323** | 0.302 | 0.448 |
| mnli | choice | 0.582 | **0.792** | 0.782 | 0.091 |
| mnli_ood | choice | 0.596 | **0.824** | 0.808 | 0.073 |
| nfcorpus_rel_choice | choice | 0.333 | 0.617 | **0.733** | 0.159 |
| scidocs_rel_choice | choice | 0.467 | 0.950 | 0.950 | 0.076 |
| scifact_rel_choice | choice | 0.417 | 0.967 | **1.000** | 0.043 |
| snli | choice | 0.617 | 0.861 | **0.868** | 0.057 |
| trec-covid_rel_choice | choice | 0.420 | 0.640 | **0.900** | 0.097 |
| truthfulqa | choice | 0.242 | **0.604** | 0.592 | 0.250 |
| arguana_rel_score | score | 0.190 | 0.647 | **0.840** | 0.125 |
| goemotions_sentiment | score | 0.425 | **0.605** | 0.578 | 0.225 |
| jigsaw_severity | score | 0.385 | 0.860 | **0.915** | 0.031 |
| nfcorpus_rel_score | score | 0.262 | 0.555 | **0.583** | 0.195 |
| scidocs_rel_score | score | 0.217 | 0.622 | **0.831** | 0.069 |
| scifact_rel_score | score | 0.144 | 0.781 | **0.961** | 0.085 |
| trec-covid_rel_score | score | 0.187 | 0.697 | 0.690 | 0.139 |

## 3. 超出 26 个保留行的标签空间（单独列出，不计入上面的平均）

这两个任务的选项数超过 26，预训练读出（只有 26 个单字字母行）**无法作答**，因此不参与第 1 节的对比：

| 标签空间 | 训练后读出 | Laya | 零样本读出 |
|---|---|---|---|
| CLINC150（150 类意图） | 0.527 | 0.545 | 无法作答 |
| HWU64（64 类意图） | 0.540 | 0.345 | 无法作答 |

## 4. 速度

同一次运行（23145 条决策）：训练后读出 14.1 分钟；Laya 9.0 分钟。
单位说明：Laya 一次调用答一个问题；本引擎一次调用答完同一个 state 上的全部问题。
按决策摊：本引擎 **36.7 ms/决策**，Laya **23.4 ms/决策**。
`scripts/speed_compare.py` 可复现，并给出不同 state 长度/问题数下的逐项对比。

## 5. 交付物清单

| 文件 | 内容 |
|---|---|
| `models/qwenjev-multitask-v2/readout.pt` | 交付读出：512 行 = 27 行保留 + 485 行私有，覆盖 90 个标签空间 |
| `models/shared-rows/` `models/raw-trained-rows/` `models/prototype-rows/` | 拼装交付读出的三个源文件 |
| `data/ready/` | 统一格式的训练/测试集（`manifest.json` 记录每个 split 的来源与条数） |
| `artifacts/results.json` | 第 1、2 节的原始评测数据（26 个测试集 × 3 个变体） |
| `artifacts/extra_splits.json` | 由本仓库补齐的训练集清单（来源与条数） |
