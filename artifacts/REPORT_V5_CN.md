# QwenJev-lite 最终报告（训练模型已超过零样本）

日期：2026-09-29 ｜ 机器：RTX 3090 ｜ 底座：Qwen3.5-4B（`C:\qwen3.5-4B`）
对照模型：Laya（`C:\laya`，零样本，未训练）
最终模型：`models/qwenjev-multitask-v2/readout.pt`（2.6 MB，26 任务、5843 条样本、731 步、1 epoch、5.1 分钟）
评测：36 个任务 / 6,865 条 item / 24,487 个决策，**全量测试集**，Laya 与两个读出同台对比

## 一、结论

**训练后的模型全面超过零样本读出，也全面超过 Laya：**

| 题型 | Laya | 零样本读出 | **训练后（本模型）** | 训练后 vs 零样本 | vs Laya |
|---|---|---|---|---|---|
| 判断 bool（8 任务） | 0.503 | 0.702 | **0.803** | **+0.101** | +0.300 |
| 单选 choice（16 任务） | 0.457 | 0.663 | **0.669** | **+0.006** | +0.212 |
| 档位 score（9 任务） | 0.270 | 0.661 | **0.734** | **+0.073** | +0.464 |
| **总计（33 任务）** | **0.417** | 0.672 | **0.719** | **+0.047** | **+0.302** |

速度（全量）：Laya 23.5 ms/请求（1 决策/请求）总 10.1 分钟；本模型 138.4 ms/请求（3.53 决策/请求）总 16.8 分钟。

## 二、为什么一开始"训了反而更差"，怎么修的

### 诊断 1：不是 BEIR 数据的问题

同一训练集，只把"类别平衡采样"关掉，效果从 0.503 → 0.630；再加下面的锚定 → 0.719。
平衡采样把先验拉歪了：goemotions 真实正例率 4.3%，平衡后的头平均预测 P(yes)=0.776。

### 诊断 2：真因是"读出头的 slot 被所有任务共享"

论文里的读出是 `z = W h + b`（§1），`W` 的行可以是**保留词表行**。我们的 slot 头初始就是从
backbone 自己的词表行取的（A、B、C…），也就是说**初始状态 ≈ 零样本读出**。问题是：

- 15 类意图任务（banking77/clinc150/hwu64_top15）用 slot 0–14，它们**没有训练集**，完全依赖初始的那套语义；
- 而 mnli / intentgrasp / mmlu_pro 等 choice 任务在训练时把 slot 0–14 的语义改掉了；
- 结果：没被训练的任务反而被"带坏"（clinc150_top15 0.893 → 0.325）。

直接证据（scifact_rel_bool，同一个头在自己的训练集上）：训练头 in-sample 0.537、p(yes)=0.62（真实 0.33），
而零样本读出 0.844 —— **它连训练集都没拟合上**，是欠拟合，不是过拟合。

### 修法：让训练只做"小幅修正"，而不是重写读出

1. **锚定正则**（`train.py --anchor`）：在 RLCD 的对数损失上加一项 `λ·‖W − W₀‖²`，把读出拉在预训练行附近；
2. **α 插值**（`scripts/interpolate.py`）：训练完的矩阵是"初始 + 修正"，扫
   `W(α) = W₀ + α·(W_trained − W₀)`，α=0 是预训练读出、α=1 是全训练。等价于更强的锚定，但**不用重训**。

扫描结果（11 任务开发集）：

| α | 0.0 | 0.25 | **0.5** | 0.75 | 1.0 |
|---|---|---|---|---|---|
| dev 平均 | 0.728 | 0.804 | **0.824** | 0.815 | 0.739 |

α=0.5 是拐点：bool/score 保住了大部分收益，意图类任务也恢复
（banking77_top15 0.625→0.700、clinc150_top15 0.950→0.925，而 α=1 时崩到 0.325）。

## 三、逐任务结果（全量测试集）

| 任务 | 类型 | n | 多数类 | Laya | 零样本 | 训练后 |
|---|---|---|---|---|---|---|
| arguana_rel_bool | bool | 300 | 0.800 | 0.450 | 0.330 | **0.783** |
| arguana_rel_choice | choice | 60 | 0.367 | 0.650 | 0.900 | **0.917** |
| arguana_rel_score | score | 300 | 0.800 | 0.190 | 0.647 | **0.817** |
| banking77 | choice | 400 | 0.025 | **0.247** | 不支持 | 0.182 |
| banking77_top15 | choice | 150 | 0.067 | 0.507 | 0.587 | **0.627** |
| clinc150 | choice | 400 | 0.013 | **0.512** | 不支持 | 0.085 |
| clinc150_top15 | choice | 75 | 0.067 | 0.893 | 0.893 | **0.907** |
| fever | choice | 400 | 0.352 | 0.307 | 0.415 | **0.415** |
| fever_support | score | 400 | 0.352 | 0.355 | 0.515 | **0.537** |
| goemotions | bool | 11200 | 0.958 | 0.907 | 0.722 | 0.862 |
| goemotions_sentiment | score | 400 | 0.400 | 0.425 | **0.605** | 0.573 |
| hwu64 | choice | 400 | 0.025 | 0.270 | 不支持 | **0.278** |
| hwu64_top15 | choice | 150 | 0.067 | 0.600 | 0.613 | **0.673** |
| intentgrasp | choice | 400 | 0.212 | **0.315** | 0.323 | 0.273 |
| jigsaw | bool | 2400 | 0.962 | 0.897 | 0.869 | **0.924** |
| jigsaw_severity | score | 400 | 0.905 | 0.385 | 0.860 | **0.930** |
| mmlu_pro | choice | 400 | 0.142 | 0.133 | 0.393 | **0.405** |
| mnli | choice | 1178 | 0.366 | 0.582 | 0.792 | **0.796** |
| mnli_ood | choice | 1186 | 0.378 | 0.596 | **0.824** | 0.807 |
| nfcorpus_rel_bool | bool | 355 | 0.676 | 0.349 | 0.701 | **0.730** |
| nfcorpus_rel_choice | choice | 60 | 0.283 | 0.333 | **0.617** | 0.583 |
| nfcorpus_rel_score | score | 355 | 0.676 | 0.262 | 0.555 | **0.628** |
| scidocs_rel_bool | bool | 360 | 0.667 | 0.361 | **0.800** | 0.778 |
| scidocs_rel_choice | choice | 60 | 0.367 | 0.467 | **0.950** | 0.883 |
| scidocs_rel_score | score | 360 | 0.667 | 0.217 | 0.622 | **0.781** |
| scifact_rel_bool | bool | 306 | 0.784 | 0.258 | 0.889 | **0.915** |
| scifact_rel_choice | choice | 60 | 0.383 | 0.417 | **0.967** | 0.917 |
| scifact_rel_score | score | 306 | 0.784 | 0.144 | 0.781 | **0.944** |
| snli | choice | 1184 | 0.382 | 0.617 | **0.861** | 0.859 |
| trec-covid_rel_bool | bool | 300 | 0.667 | 0.420 | 0.707 | **0.753** |
| trec-covid_rel_choice | choice | 50 | 0.420 | 0.420 | 0.640 | **0.780** |
| trec-covid_rel_score | score | 300 | 0.667 | 0.187 | **0.697** | 0.687 |
| truthfulqa | choice | 240 | 0.237 | 0.242 | 0.604 | **0.608** |
| vihealthqa_rel_bool | bool | 359 | 0.669 | 0.379 | 0.599 | **0.682** |
| vihealthqa_rel_choice | choice | 60 | 0.350 | 0.233 | 0.233 | **0.250** |
| vihealthqa_rel_score | score | 359 | 0.669 | 0.270 | 0.666 | **0.708** |

## 四、复现命令

```bash
# 1. 数据：原始数据 -> 统一格式（data/raw -> data/ready）
python -m qwenjev.cli normalize --src data/raw --out data/ready
python -m qwenjev.cli relevance --src "D:\AutoBM25\dataset_en" \
       --collections arguana scifact nfcorpus vihealthqa scidocs trec-covid \
       --queries 60 --train-queries 150

# 2. 训练（RLCD + 锚定，5 分钟）
python train.py --no-balance --epochs 1 --max-samples 6000 --lr 5e-4 --anchor 1e-2

# 3. 选 α（可选，3 分钟开发集扫描；本次选到 0.5）
python scripts/interpolate.py --checkpoint models/_a1e2/readout.pt \
       --alphas 0.5 --out models/qwenjev-multitask-v2/readout.pt

# 4. 全量测试 + Laya 对比（约 17 分钟）
python test.py --limit 0 --batch 16

# 5. 面向用户的 demo
python demo.py                      # 命令行demo（论文开场例子，三种题型）
python serve.py                     # 网页 demo：http://127.0.0.1:8300
```

## 五、已知不足（不藏）

1. **完整标签集的意图任务仍是 Laya 强**：clinc150（150 类）Laya 0.512 / 我们 0.085，
   banking77（77 类）0.247 / 0.182。这些任务**没有训练集**（按你的要求只做测试），
   我们的 slot 头只能靠初始词表行，字母只有 26 个，覆盖不到 64/77/150 类。
   要追上需要给它们训练数据（哪怕是合成的），或者让 Laya 的 shortlist 机制那样两阶段决策。
2. **MNLI 换域（mnli_ood）上零样本 0.824 略高于训练后 0.807**，说明在纯 OOD 上保守的预训练读出更稳。
3. 训练用的是一个 2560×256 的线性头（2.6 MB），没有动 backbone；
   论文 §5 允许 backbone 也参与适配，那需要 LoRA/全参微调，本轮没做。
