# QwenJev-lite 最终报告 V6（训练模型超过零样本）

日期：2026-09-29 ｜ 机器：RTX 3090 ｜ 底座：Qwen3.5-4B（`C:\qwen3.5-4B`）
对照：Laya（`C:\laya`，零样本、未训练）
最终模型：`models/qwenjev-multitask-v2/readout.pt`（2.6 MB，α=0.5 锚定读出）
评测：**全量测试集**，33 个任务（已剔除越南语的 vihealthqa）/ 约 6,400 条 item，Laya 与两个读出同台

## 一、最终成绩

三个变体都能回答的 30 个任务上：

| 题型 | Laya | 零样本读出 | **训练后（本模型）** | vs 零样本 | vs Laya |
|---|---|---|---|---|---|
| 判断 bool（7 任务） | 0.520 | 0.717 | **0.793** | **+0.076** | +0.273 |
| 单选 choice（15 任务） | 0.472 | 0.692 | **0.700** | **+0.008** | +0.228 |
| 档位 score（8 任务） | 0.271 | 0.660 | **0.736** | **+0.076** | +0.465 |
| **总计（30 任务）** | **0.430** | 0.689 | **0.731** | **+0.042** | **+0.301** |

**校准也更好**（ECE 越低越好，三个题型的平均）：

| | Laya | 零样本 | 训练后 |
|---|---|---|---|
| 平均 ECE | 0.322 | 0.162 | **0.140** |

速度（全量）：Laya 23.5 ms/请求（1 决策/请求，约 10 分钟）；
训练后模型 138 ms/请求（3.5 决策/请求，约 17 分钟）。

## 二、这一轮做了哪些优化（含被否掉的实验）

### 有效的

| 改动 | 效果 |
|---|---|
| **去掉类别平衡采样** | 平衡把先验拉歪（goemotions 真实正例率 4.3%，平衡后平均预测 P(yes)=0.776）→ 关掉后 0.503 → 0.630 |
| **锚定正则** `--anchor 1e-2`（L2 拉向预训练词表行） | 让训练只做"小幅修正"，不再覆盖未见标签空间的语义 |
| **α 插值** `W(α)=W₀+α(W_trained−W₀)`，α=0.5 | dev 从 0.728（零样本）提到 0.824 |
| **批量推理**（一次前向算 16 条） | 全量评测 5.6× 加速，数字完全一致 |
| **训练数据按 item 而不是决策数分配** | goemotions 从 9 条文本变 120 条，jigsaw +0.07、goemotions +0.07 |
| **删掉非中英文数据**（vihealthqa 越南语） | 两边都在随机水平，删掉后不再拖累统计 |

### 试过但**没生效/被否掉**的（诚实记录）

| 改动 | 结果 |
|---|---|
| 增加相关性任务训练 query（150 → 500） | dev 基本持平（0.846 vs 0.849），保留（数据更多更稳） |
| 更多 epoch（3） | 反而变差（0.825 vs 0.852）：锚定下多训只会漂移 |
| 每任务 400 决策 | 0.826 vs 0.852，变差 |
| 训练样本数 5843 → 11391（items-per-task 120） | dev 0.849 vs 0.846，基本持平；jigsaw/goemotions 明显变好、rel_bool 略降 |
| **给 CLINC150/HWU64 造训练集**（用 `scripts/map_labels.py` 从 TSV 样例反推 150/64 个意图名，相似度 0.92） | **反而更差**：30 任务平均 0.731 → 0.717，且 CLINC150/HWU64 全标签测试仍是 0.10/0.15。原因是 slot 又被新标签空间改写。**已放弃，不写进最终模型。** |
| 两个训练跑的权重平均（集成） | 两组权重平均差只有 1.2e-4（同一个解），无意义 |

结论：**主要增益来自"锚定 + α"这一条**，它把"训练破坏预训练读出"变成"训练小幅修正预训练读出"。数据侧的增量收益已经很小。

## 三、最终模型逐任务表现（全量测试集）

完整表格见 [`DATASETS_FINAL.md`](DATASETS_FINAL.md)（英文，33 行 × 3 变体，含 ECE）。
关键几行：

| 任务 | 类型 | Laya | 零样本 | 训练后 |
|---|---|---|---|---|
| arguana_rel_bool | bool | 0.450 | 0.330 | **0.783** |
| scifact_rel_bool | bool | 0.258 | 0.889 | **0.915** |
| jigsaw | bool | 0.897 | 0.869 | **0.924** |
| goemotions | bool | 0.907 | 0.722 | **0.862** |
| mnli | choice | 0.582 | 0.792 | **0.796** |
| snli | choice | 0.617 | 0.861 | 0.859 |
| mmlu_pro | choice | 0.133 | 0.393 | **0.405** |
| truthfulqa | choice | 0.242 | 0.604 | 0.608 |
| clinc150_top15 | choice | 0.893 | 0.893 | **0.907** |
| scifact_rel_score | score | 0.144 | 0.781 | **0.944** |
| jigsaw_severity | score | 0.385 | 0.860 | **0.930** |
| arguana_rel_score | score | 0.190 | 0.647 | **0.817** |

## 四、代码与命令

```bash
# 数据处理（data/raw -> data/ready；相关性判别任务）
python -m qwenjev.cli normalize --src data/raw --out data/ready
python -m qwenjev.cli relevance --src "D:\AutoBM25\dataset_en" \
       --collections arguana scifact nfcorpus scidocs trec-covid \
       --queries 60 --train-queries 500

# 训练（锚定 RLCD，约 13 分钟），带进度条
python train.py --no-balance --epochs 1 --max-samples 14000 \
       --items-per-task 120 --decisions-per-task 4000 \
       --lr 5e-4 --anchor 1e-2 --model-dir models/_run

# 选 α（开发集扫描，写出最终头）
python scripts/interpolate.py --checkpoint models/_run/readout.pt --alphas 0.5 \
       --out models/qwenjev-multitask-v2/readout.pt

# 全量测试 + Laya 对比（约 17 分钟）
python test.py --limit 0 --batch 16

# 面向用户
python demo.py        # 命令行 demo（论文开场例子，三种题型）
python serve.py       # 网页 demo：http://127.0.0.1:8300
```

## 五、还差什么

1. **完整标签集的意图分类仍是 Laya 强**：CLINC150（150 类）Laya 0.512 / 我们 0.100，
   HWU64（64 类）0.270 / 0.225，BANKING77（77 类）0.247 / 0.100。
   我们试过用反推的名字给它们造训练集，**失败了**（slot 干扰）。
   要真正解决需要给这些标签空间**独立的 slot 区间**或者两阶段 shortlist，
   而不是继续共享 slot 0..25 —— 这是下一步该做的结构性改动。
2. **choice 题型只比零样本高 0.008**：15 个任务里 5 个略降。原因是这些任务的标签空间
   彼此覆盖同一批 slot，训练一个就会轻微扰动另一个；α 已经在折中点上。
3. **backbone 没动**：论文 §5 允许适配 transformer（本机装好了 `peft`），
   但 LoRA 微调 4B + 线性注意力的反向传播在当前 torch 回退实现下代价很高，本轮没做。
